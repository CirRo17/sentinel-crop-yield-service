"""Step3：地块级长势定级。

消费 Step2 输出的三张栅格：NDVI_MVC、ZScore、GrowthLevel。

地块定级规则：
    1. 地块平均 Z-Score 先生成基础等级。
    2. 差等级像元占比 poor_pct >= 30% 时，直接定为差。
    3. poor_pct >= 15% 时，基础等级降一级。
    4. poor_pct >= 30% 或 NDVI 变异系数 ndvi_cv >= 0.30 时，标记异常。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask

from configs.paths import ProjectPaths


DEFAULT_INPUT_DIR = Path("data/output/growth_monitoring")
DEFAULT_OUTPUT_CSV = DEFAULT_INPUT_DIR / "parcel_growth.csv"
DEFAULT_OUTPUT_JSON = DEFAULT_INPUT_DIR / "parcel_growth_summary.json"
DEFAULT_OUTPUT_SHP = DEFAULT_INPUT_DIR / "parcel_growth.shp"

CROP_NAMES = {0: "Others", 1: "Rice", 2: "Wheat", 3: "Maize", 4: "Rapeseed"}
LEVEL_LABELS = {0: "Unclassified", 1: "Excellent", 2: "Good", 3: "Normal", 4: "Poor"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step3：地块级长势定级。")
    parser.add_argument("--ndvi", type=Path, default=None, help="Step2 输出 NDVI_MVC_YYYY_MM.tif。")
    parser.add_argument("--zscore", type=Path, default=None, help="Step2 输出 ZScore_YYYY_MM.tif。")
    parser.add_argument("--growth-level", type=Path, default=None, help="Step2 输出 GrowthLevel_YYYY_MM.tif。")
    parser.add_argument(
        "--step2-stats",
        "--step1-stats",
        dest="step2_stats",
        type=Path,
        default=None,
        help="Step2 stats JSON，可自动读取三个栅格路径。",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--parcels",
        type=Path,
        default=None,
        help="地块 Shapefile。不指定则从配置文件推导（默认承接作物分类第六步输出）。",
    )
    parser.add_argument("--parcel-id-field", default="FID")
    parser.add_argument("--crop-type-field", default="crop_type")
    parser.add_argument("--target-year", type=int, default=None)
    parser.add_argument("--target-month", type=int, default=None)
    parser.add_argument("--baseline-years", default=None)
    parser.add_argument("--z-good", type=float, default=0.5)
    parser.add_argument("--z-normal", type=float, default=-0.5)
    parser.add_argument("--z-below", type=float, default=-1.5)
    parser.add_argument("--poor-pct-critical", type=float, default=30.0)
    parser.add_argument("--poor-pct-downgrade", type=float, default=15.0)
    parser.add_argument("--cv-anomaly", type=float, default=0.30)
    parser.add_argument("--all-touched", action="store_true")
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-shp", type=Path, default=DEFAULT_OUTPUT_SHP)
    return parser.parse_args()


def _safe_float(value: Any) -> float | None:
    """把 NaN/Inf/非法数值统一转成 None，便于 CSV/JSON 输出。"""
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(result) or np.isinf(result):
        return None
    return result


def _resolve_inputs(args: argparse.Namespace) -> tuple[Path, Path, Path, dict[str, Any]]:
    """解析 Step2 栅格输入；优先使用命令行显式路径，其次读取 stats JSON。"""
    stats: dict[str, Any] = {}
    if args.step2_stats:
        if not args.step2_stats.exists():
            raise FileNotFoundError(f"Step2 stats 不存在：{args.step2_stats}")
        with open(args.step2_stats, encoding="utf-8") as f:
            stats = json.load(f)
        outputs = stats.get("outputs") or {}
        ndvi = args.ndvi or Path(outputs.get("ndvi_mvc", ""))
        zscore = args.zscore or Path(outputs.get("zscore", ""))
        growth_level = args.growth_level or Path(outputs.get("growth_level", ""))
    else:
        ndvi = args.ndvi or DEFAULT_INPUT_DIR / "NDVI_MVC.tif"
        zscore = args.zscore or DEFAULT_INPUT_DIR / "ZScore.tif"
        growth_level = args.growth_level or DEFAULT_INPUT_DIR / "GrowthLevel.tif"

    for path, label in [(ndvi, "NDVI MVC"), (zscore, "Z-Score"), (growth_level, "GrowthLevel")]:
        if not path.exists():
            raise FileNotFoundError(f"{label} 栅格不存在：{path}")
    return ndvi, zscore, growth_level, stats


def classify_parcel(
    z_mean: float | None,
    poor_pct: float | None,
    has_crop: bool,
    z_good: float,
    z_normal: float,
    z_below: float,
    poor_pct_critical: float,
    poor_pct_downgrade: float,
) -> int:
    """根据地块平均 Z-Score 和 poor_pct 惩罚规则定级。"""
    if not has_crop or z_mean is None:
        return 0
    if z_mean > z_good:
        base = 1
    elif z_mean >= z_normal:
        base = 2
    elif z_mean >= z_below:
        base = 3
    else:
        base = 4

    poor = poor_pct if poor_pct is not None else 0.0
    if poor >= poor_pct_critical:
        return 4
    if poor >= poor_pct_downgrade:
        return min(base + 1, 4)
    return base


def is_anomaly(poor_pct: float | None, cv: float | None, poor_pct_critical: float, cv_anomaly: float) -> bool:
    """根据差区占比和 NDVI 变异系数判断是否异常。"""
    poor = poor_pct if poor_pct is not None else 0.0
    ndvi_cv = cv if cv is not None else 0.0
    return poor >= poor_pct_critical or ndvi_cv >= cv_anomaly


def _masked_values(src: rasterio.DatasetReader, geom: Any, all_touched: bool) -> np.ndarray:
    """提取单个地块范围内的有效连续型栅格值。"""
    data, _ = mask(src, [geom], crop=True, filled=True, nodata=src.nodata, all_touched=all_touched)
    values = data[0].astype("float32")
    valid = np.isfinite(values)
    if src.nodata is not None:
        valid &= values != float(src.nodata)
    return values[valid]


def _masked_level_counts(src: rasterio.DatasetReader, geom: Any, all_touched: bool) -> dict[int, int]:
    """统计单个地块内 1-4 级长势像元数量。"""
    data, _ = mask(src, [geom], crop=True, filled=True, nodata=0, all_touched=all_touched)
    levels = data[0].astype("int16")
    return {level: int(np.count_nonzero(levels == level)) for level in (1, 2, 3, 4)}


def _pct(count: int, total: int) -> float | None:
    """计算百分比；无有效像元时返回 None。"""
    if total <= 0:
        return None
    return round(count / total * 100.0, 2)


def _row_for_parcel(
    parcel_id: Any,
    crop_type: int,
    geom: Any,
    ndvi_src: rasterio.DatasetReader,
    z_src: rasterio.DatasetReader,
    level_src: rasterio.DatasetReader,
    args: argparse.Namespace,
    target_year: int | None,
    target_month: int | None,
    baseline_years: str | None,
) -> dict[str, Any]:
    """生成单个地块的长势统计和最终定级结果。"""
    has_crop = 1 <= crop_type <= 4
    z_values = _masked_values(z_src, geom, args.all_touched)
    ndvi_values = _masked_values(ndvi_src, geom, args.all_touched)
    level_counts = _masked_level_counts(level_src, geom, args.all_touched)

    z_mean = _safe_float(np.mean(z_values)) if z_values.size else None
    ndvi_mean = _safe_float(np.mean(ndvi_values)) if ndvi_values.size else None
    ndvi_std = _safe_float(np.std(ndvi_values)) if ndvi_values.size else None
    ndvi_cv = round(ndvi_std / ndvi_mean, 4) if ndvi_mean and ndvi_std is not None and ndvi_mean > 0 else None

    total_level_pixels = sum(level_counts.values())
    poor_pct = _pct(level_counts[4], total_level_pixels)
    pixel_area_ha = abs(z_src.transform.a * z_src.transform.e) / 10000.0
    parcel_level = classify_parcel(
        z_mean,
        poor_pct,
        has_crop,
        args.z_good,
        args.z_normal,
        args.z_below,
        args.poor_pct_critical,
        args.poor_pct_downgrade,
    )

    return {
        "parcel_id": parcel_id,
        "crop_type_code": crop_type,
        "crop_type_name": CROP_NAMES.get(crop_type, str(crop_type)),
        "classified": 1 if has_crop else 0,
        "target_year": target_year,
        "target_month": target_month,
        "baseline_years": baseline_years,
        "ndvi_mean": ndvi_mean,
        "ndvi_std": ndvi_std,
        "ndvi_cv": ndvi_cv,
        "z_score_mean": z_mean,
        "pixel_count_L1": level_counts[1],
        "pixel_count_L2": level_counts[2],
        "pixel_count_L3": level_counts[3],
        "pixel_count_L4": level_counts[4],
        "excellent_pct": _pct(level_counts[1], total_level_pixels),
        "good_pct": _pct(level_counts[2], total_level_pixels),
        "normal_pct": _pct(level_counts[3], total_level_pixels),
        "poor_pct": poor_pct,
        "valid_area_ha": round(total_level_pixels * pixel_area_ha, 4),
        "parcel_growth_level": parcel_level,
        "parcel_growth_label": LEVEL_LABELS[parcel_level],
        "anomaly": "Anomaly" if has_crop and is_anomaly(poor_pct, ndvi_cv, args.poor_pct_critical, args.cv_anomaly) else "Normal",
    }


def _write_parcel_shapefile(parcels: gpd.GeoDataFrame, rows: list[dict[str, Any]], output_shp: Path) -> None:
    """Write parcel geometries with selected growth fields."""

    result = parcels.iloc[: len(rows)].copy()
    result["grw_lv"] = [row["parcel_growth_level"] for row in rows]
    result["grw_lbl"] = [row["parcel_growth_label"] for row in rows]
    result["z_mean"] = [row["z_score_mean"] for row in rows]
    result["anom"] = [1 if row["anomaly"] == "Anomaly" else 0 for row in rows]

    output_shp.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output_shp, driver="ESRI Shapefile", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(args.config)
    parcels_path = args.parcels or paths.parcel_majority_shp
    ndvi_path, zscore_path, growth_level_path, step2_stats = _resolve_inputs(args)

    if not parcels_path.exists():
        raise FileNotFoundError(f"地块文件不存在：{parcels_path}")
    parcels = gpd.read_file(parcels_path)
    if args.crop_type_field not in parcels.columns:
        raise ValueError(f"地块矢量中不存在字段：{args.crop_type_field}")

    id_field = args.parcel_id_field
    if id_field not in parcels.columns:
        parcels["_pid"] = range(len(parcels))
        id_field = "_pid"

    target_year = args.target_year if args.target_year is not None else step2_stats.get("target_year")
    target_month = args.target_month if args.target_month is not None else step2_stats.get("target_month")
    baseline_years = args.baseline_years if args.baseline_years is not None else step2_stats.get("baseline_years")

    rows: list[dict[str, Any]] = []
    with rasterio.open(zscore_path) as z_src, rasterio.open(ndvi_path) as ndvi_src, rasterio.open(growth_level_path) as level_src:
        if z_src.crs and parcels.crs and str(parcels.crs) != str(z_src.crs):
            parcels = parcels.to_crs(z_src.crs)

        for _, parcel in parcels.iterrows():
            geom = parcel.geometry
            if geom is None or geom.is_empty:
                continue
            crop_type = int(parcel[args.crop_type_field]) if parcel[args.crop_type_field] is not None else 0
            rows.append(
                _row_for_parcel(
                    parcel[id_field],
                    crop_type,
                    geom.__geo_interface__,
                    ndvi_src,
                    z_src,
                    level_src,
                    args,
                    target_year,
                    target_month,
                    baseline_years,
                )
            )

    fieldnames = [
        "parcel_id", "crop_type_code", "crop_type_name", "classified",
        "target_year", "target_month", "baseline_years",
        "ndvi_mean", "ndvi_std", "ndvi_cv", "z_score_mean",
        "pixel_count_L1", "pixel_count_L2", "pixel_count_L3", "pixel_count_L4",
        "excellent_pct", "good_pct", "normal_pct", "poor_pct", "valid_area_ha",
        "parcel_growth_level", "parcel_growth_label", "anomaly",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    _write_parcel_shapefile(parcels, rows, args.output_shp)

    crop_rows = [row for row in rows if row["classified"] == 1]
    valid_growth_rows = [row for row in crop_rows if row["parcel_growth_level"] in (1, 2, 3, 4)]
    summary = {
        "inputs": {
            "ndvi": str(ndvi_path),
            "zscore": str(zscore_path),
            "growth_level": str(growth_level_path),
            "parcels": str(parcels_path),
        },
        "target_year": target_year,
        "target_month": target_month,
        "baseline_years": baseline_years,
        "parameters": {
            "z_good": args.z_good,
            "z_normal": args.z_normal,
            "z_below": args.z_below,
            "poor_pct_critical": args.poor_pct_critical,
            "poor_pct_downgrade": args.poor_pct_downgrade,
            "cv_anomaly": args.cv_anomaly,
        },
        "parcel_count": len(rows),
        "classified_parcel_count": len(crop_rows),
        "valid_growth_parcel_count": len(valid_growth_rows),
        "level_distribution": {
            str(level): int(sum(1 for row in crop_rows if row["parcel_growth_level"] == level))
            for level in (0, 1, 2, 3, 4)
        },
        "anomaly_count": int(sum(1 for row in crop_rows if row["anomaly"] == "Anomaly")),
        "output_csv": str(args.output_csv),
        "output_shp": str(args.output_shp),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=== Step3 地块级长势定级完成 ===")
    print(f"CSV: {args.output_csv}")
    print(f"JSON: {args.output_json}")


if __name__ == "__main__":
    main()
