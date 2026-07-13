"""Step3：地块级病虫害胁迫定级。

消费 Step2 输出的像元级异常评分栅格，按地块统计 mean/max/pixel_count，
并生成地块级病虫害胁迫等级矢量。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from configs.paths import ProjectPaths
DEFAULT_STEP2_STATS = Path("data/output/pest_detect/pixel/pest_step2_stats.json")
DEFAULT_OUTPUT = Path("data/output/pest_detect/parcel/parcel_pest_stress_grade.shp")

GRADE_NAMES = {
    0: "无有效像素",
    1: "正常/低病虫害胁迫",
    2: "轻度病虫害胁迫",
    3: "中度病虫害胁迫",
    4: "重度病虫害胁迫",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step3：地块级病虫害胁迫定级。")
    parser.add_argument("--raster", type=Path, default=None, help="Step2 输出的异常评分栅格。")
    parser.add_argument("--step2-stats", type=Path, default=DEFAULT_STEP2_STATS, help="Step2 stats JSON，可自动读取异常评分栅格路径。")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--parcels", type=Path, default=None, help="地块矢量。不指定则从配置文件推导（默认承接作物分类第六步输出）。")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="输出地块级病虫害胁迫等级矢量。")
    parser.add_argument("--mean-heavy", type=float, default=70.0, help="均值达到该阈值定为重度。")
    parser.add_argument("--mean-medium", type=float, default=50.0, help="均值达到该阈值定为中度。")
    parser.add_argument("--mean-light", type=float, default=20.0, help="均值超过该阈值定为轻度。")
    parser.add_argument("--max-hotspot", type=float, default=80.0, help="均值超过轻度阈值且最大值达到该阈值时升为中度。")
    return parser.parse_args()


def classify_pest_grade(
    mean_score: float | None,
    max_score: float | None,
    mean_heavy: float,
    mean_medium: float,
    mean_light: float,
    max_hotspot: float,
) -> int:
    """根据地块内像素均值和最大值返回病虫害胁迫等级。"""
    if mean_score is None or max_score is None or np.isnan(mean_score) or np.isnan(max_score):
        return 0
    if mean_score >= mean_heavy:
        return 4
    if mean_score >= mean_medium:
        return 3
    if mean_score > mean_light and max_score >= max_hotspot:
        return 3
    if mean_score > mean_light:
        return 2
    return 1


def _resolve_raster(args: argparse.Namespace) -> Path:
    if args.raster:
        return args.raster
    if not args.step2_stats.exists():
        raise FileNotFoundError(f"Step2 stats 不存在：{args.step2_stats}")
    with open(args.step2_stats, encoding="utf-8") as f:
        stats = json.load(f)
    path = Path((stats.get("outputs") or {}).get("anomaly_score", ""))
    if not path.exists():
        raise FileNotFoundError(f"异常评分栅格不存在：{path}")
    return path


def zonal_grade(
    score_raster: Path,
    parcel_vector: Path,
    output_vector: Path,
    *,
    mean_heavy: float,
    mean_medium: float,
    mean_light: float,
    max_hotspot: float,
) -> None:
    """按地块统计像元得分，并写出地块级等级矢量。"""
    if not parcel_vector.exists():
        raise FileNotFoundError(f"地块矢量不存在：{parcel_vector}")
    parcels = gpd.read_file(parcel_vector)
    if parcels.empty:
        raise ValueError(f"地块矢量为空：{parcel_vector}")

    with rasterio.open(score_raster) as src:
        if parcels.crs is None:
            raise ValueError("地块矢量没有 CRS，请先为矢量数据定义坐标系。")
        if src.crs is None:
            raise ValueError("得分栅格没有 CRS，请先为栅格数据定义坐标系。")
        if parcels.crs != src.crs:
            parcels = parcels.to_crs(src.crs)

        zone_shapes = (
            (geom, zone_id)
            for zone_id, geom in enumerate(parcels.geometry, start=1)
            if geom is not None and not geom.is_empty
        )
        zones = rasterize(
            zone_shapes,
            out_shape=src.shape,
            transform=src.transform,
            fill=0,
            dtype="int32",
        )

        score_band = src.read(1, masked=True)
        score_values = score_band.astype("float64").filled(np.nan)
        valid_mask = zones > 0
        valid_mask &= ~np.ma.getmaskarray(score_band)
        valid_mask &= np.isfinite(score_values)
        if src.nodata is not None:
            valid_mask &= score_values != src.nodata

        valid_zones = zones[valid_mask]
        valid_scores = score_values[valid_mask]
        zone_count = len(parcels) + 1

        counts = np.bincount(valid_zones, minlength=zone_count)
        sums = np.bincount(valid_zones, weights=valid_scores, minlength=zone_count)

        mean_array = np.full(zone_count, np.nan, dtype="float64")
        np.divide(sums, counts, out=mean_array, where=counts > 0)

        max_array = np.full(zone_count, -np.inf, dtype="float64")
        if valid_scores.size:
            np.maximum.at(max_array, valid_zones, valid_scores)

        mean_scores = [float(mean_array[zone_id]) if counts[zone_id] else None for zone_id in range(1, zone_count)]
        max_scores = [float(max_array[zone_id]) if counts[zone_id] else None for zone_id in range(1, zone_count)]
        pixel_counts = [int(counts[zone_id]) for zone_id in range(1, zone_count)]
        grades = [
            classify_pest_grade(mean_score, max_score, mean_heavy, mean_medium, mean_light, max_hotspot)
            for mean_score, max_score in zip(mean_scores, max_scores)
        ]

    result = parcels.copy()
    result["mean_score"] = mean_scores
    result["max_score"] = max_scores
    result["pixel_count"] = pixel_counts
    result["pest_grade"] = grades
    result["grade_name"] = [GRADE_NAMES[grade] for grade in grades]

    output_vector.parent.mkdir(parents=True, exist_ok=True)
    result.to_file(output_vector, driver="ESRI Shapefile", encoding="utf-8")


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(args.config)
    parcels = args.parcels or paths.parcels
    raster = _resolve_raster(args)
    zonal_grade(
        raster,
        parcels,
        args.output,
        mean_heavy=args.mean_heavy,
        mean_medium=args.mean_medium,
        mean_light=args.mean_light,
        max_hotspot=args.max_hotspot,
    )
    print("=== Step3 地块级病虫害胁迫定级完成 ===")
    print(f"输入评分栅格：{raster}")
    print(f"输出等级矢量：{args.output}")


if __name__ == "__main__":
    main()
