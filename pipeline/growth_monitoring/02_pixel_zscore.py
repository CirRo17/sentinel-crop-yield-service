"""Step2：像元级 Z-Score 距平长势分级。

像元分级流程：
目标月 NDVI 最大值合成 -> 多年同期 NDVI 均值/标准差 -> NDVI 距平和 Z-Score ->
像元级长势等级。

长势等级编码：
    0 = 未分级 / 非作物 / 无有效数据
    1 = 优，Z > 0.5
    2 = 良，-0.5 <= Z <= 0.5
    3 = 中，-1.5 <= Z < -0.5
    4 = 差，Z < -1.5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from configs.paths import ProjectPaths


DEFAULT_OUTPUT_DIR = Path("data/output/growth_monitoring")
DEFAULT_STATS = DEFAULT_OUTPUT_DIR / "growth_step2_stats.json"
DEFAULT_FEATURE_STACK = Path("data/exported/feature_stack/feature_stack_multiband.tif")
DEFAULT_METADATA = Path("data/exported/feature_stack/feature_stack_multiband_metadata.json")
DEFAULT_BASELINE_MANIFEST = Path("data/output/growth_monitoring/baseline/baseline_manifest.json")

LEVEL_LABELS = {0: "Unclassified", 1: "Excellent", 2: "Good", 3: "Normal", 4: "Poor"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step2：像元级 Z-Score 距平长势分级。")
    current = parser.add_mutually_exclusive_group()
    current.add_argument("--current-ndvi", type=Path, help="目标月 NDVI MVC 单波段 GeoTIFF。")
    current.add_argument(
        "--feature-stack",
        "--current-feature-stack",
        dest="feature_stack",
        type=Path,
        default=DEFAULT_FEATURE_STACK,
        help="当前任务标准特征栈，默认承接作物分类流程输入。",
    )

    baseline = parser.add_mutually_exclusive_group()
    baseline.add_argument("--baseline-ndvi", type=Path, action="append", help="多年同期 NDVI MVC，可重复传入。")
    baseline.add_argument("--baseline-feature-stack", type=Path, action="append", help="多年同期特征栈，可重复传入。")
    baseline.add_argument(
        "--baseline-manifest",
        type=Path,
        default=DEFAULT_BASELINE_MANIFEST,
        help="Step1 自动准备的多年同期基准清单。",
    )

    parser.add_argument(
        "--metadata",
        "--current-metadata",
        dest="metadata",
        type=Path,
        default=DEFAULT_METADATA,
        help="当前任务特征栈 metadata JSON，默认承接作物分类流程输入。",
    )
    parser.add_argument("--baseline-metadata", type=Path, action="append", default=None, help="基准特征栈 metadata JSON。")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--parcels",
        type=Path,
        default=None,
        help="地块 Shapefile。不指定则从配置文件推导（默认承接作物分类第六步输出）。",
    )
    parser.add_argument("--crop-type-field", default="crop_type")
    parser.add_argument("--target-year", type=int, required=True)
    parser.add_argument("--target-month", type=int, required=True)
    parser.add_argument("--baseline-start-year", type=int, default=None)
    parser.add_argument("--baseline-end-year", type=int, default=None)
    parser.add_argument("--min-valid-ndvi", type=float, default=0.0)
    parser.add_argument("--max-valid-ndvi", type=float, default=1.0)
    parser.add_argument("--min-baseline-std", type=float, default=0.001)
    parser.add_argument("--z-good", type=float, default=0.5)
    parser.add_argument("--z-normal", type=float, default=-0.5)
    parser.add_argument("--z-below", type=float, default=-1.5)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    return parser.parse_args()


def _load_band_names(raster_path: Path, metadata_path: Path | None) -> list[str]:
    """从 metadata.band_names 或 GeoTIFF band description 读取波段名。"""
    if metadata_path and metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            meta = json.load(f)
        names = meta.get("band_names") or []
        if names:
            return [str(name) for name in names]
    with rasterio.open(raster_path) as src:
        return [src.descriptions[index - 1] or f"band_{index}" for index in range(1, src.count + 1)]


def _ndvi_band_indexes(names: list[str]) -> list[int]:
    """查找所有 NDVI 波段索引，用于多时相/多景最大值合成。"""
    indexes = [
        index
        for index, name in enumerate(names, start=1)
        if name.lower() == "ndvi" or name.lower().endswith("_ndvi")
    ]
    if not indexes:
        raise ValueError("特征栈中未找到 NDVI 波段。")
    return indexes


def _read_single_band(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    """读取单波段栅格，并把 nodata 转成 NaN 参与后续计算。"""
    if not path.exists():
        raise FileNotFoundError(f"输入栅格不存在：{path}")
    with rasterio.open(path) as src:
        data = src.read(1, masked=False).astype("float32")
        profile = src.profile.copy()
        nodata = src.nodata
    if nodata is not None:
        data = np.where(data == nodata, np.nan, data)
    return data, profile


def _read_ndvi_mvc_from_feature_stack(path: Path, metadata_path: Path | None) -> tuple[np.ndarray, dict[str, Any]]:
    """从标准特征栈中提取 NDVI 波段并做最大值合成。"""
    if not path.exists():
        raise FileNotFoundError(f"特征栈不存在：{path}")
    names = _load_band_names(path, metadata_path)
    band_indexes = _ndvi_band_indexes(names)
    with rasterio.open(path) as src:
        arrays = [src.read(index, masked=False).astype("float32") for index in band_indexes]
        profile = src.profile.copy()
    return np.nanmax(np.stack(arrays, axis=0), axis=0).astype("float32"), profile


def _valid_ndvi(data: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    """按 GEE 配置过滤有效 NDVI 范围。"""
    return np.where(np.isfinite(data) & (data >= min_value) & (data <= max_value), data, np.nan).astype("float32")


def _assert_same_grid(reference: dict[str, Any], candidate: dict[str, Any], label: str) -> None:
    """确保目标月与基准年栅格已经对齐到同一网格。"""
    keys = ("width", "height", "crs", "transform")
    mismatched = [key for key in keys if reference.get(key) != candidate.get(key)]
    if mismatched:
        raise ValueError(f"{label} 与目标 NDVI 栅格网格不一致：{', '.join(mismatched)}")


def _read_baseline_arrays(args: argparse.Namespace, current_profile: dict[str, Any]) -> list[np.ndarray]:
    """读取多年同期 NDVI 基准输入。"""
    arrays: list[np.ndarray] = []
    if args.baseline_ndvi:
        for path in args.baseline_ndvi:
            data, profile = _read_single_band(path)
            _assert_same_grid(current_profile, profile, str(path))
            arrays.append(_valid_ndvi(data, args.min_valid_ndvi, args.max_valid_ndvi))
        return arrays

    if args.baseline_manifest and args.baseline_manifest.exists():
        with open(args.baseline_manifest, encoding="utf-8") as f:
            baseline_doc = json.load(f)
        for item in baseline_doc.get("items", []):
            path = Path(str(item["feature_stack"]))
            metadata = Path(str(item["metadata"])) if item.get("metadata") else None
            data, profile = _read_ndvi_mvc_from_feature_stack(path, metadata)
            _assert_same_grid(current_profile, profile, str(path))
            arrays.append(_valid_ndvi(data, args.min_valid_ndvi, args.max_valid_ndvi))
        return arrays

    metadata_paths = args.baseline_metadata or []
    if not args.baseline_feature_stack:
        raise FileNotFoundError(
            "未找到多年同期基准数据。请先运行 "
            "python -m pipeline.growth_monitoring.01_prepare_baseline，"
            "或显式传入 --baseline-manifest / --baseline-feature-stack / --baseline-ndvi。"
        )
    if metadata_paths and len(metadata_paths) != len(args.baseline_feature_stack):
        raise ValueError("--baseline-metadata 数量必须与 --baseline-feature-stack 一致。")
    for index, path in enumerate(args.baseline_feature_stack):
        metadata = metadata_paths[index] if metadata_paths else None
        data, profile = _read_ndvi_mvc_from_feature_stack(path, metadata)
        _assert_same_grid(current_profile, profile, str(path))
        arrays.append(_valid_ndvi(data, args.min_valid_ndvi, args.max_valid_ndvi))
    return arrays


def _rasterize_crop_type(parcels_path: Path, crop_type_field: str, profile: dict[str, Any]) -> np.ndarray:
    """将地块 crop_type 字段栅格化到目标 NDVI 网格。"""
    if not parcels_path.exists():
        raise FileNotFoundError(f"地块文件不存在：{parcels_path}")
    parcels = gpd.read_file(parcels_path)
    if crop_type_field not in parcels.columns:
        raise ValueError(f"地块矢量中不存在字段：{crop_type_field}")

    raster_crs = profile.get("crs")
    if raster_crs and parcels.crs and str(parcels.crs) != str(raster_crs):
        parcels = parcels.to_crs(raster_crs)

    shapes: Iterable[tuple[Any, int]] = (
        (geom, int(value))
        for geom, value in zip(parcels.geometry, parcels[crop_type_field])
        if geom is not None and not geom.is_empty and value is not None
    )
    return rasterize(
        shapes,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        dtype="int16",
    )


def _classify_growth(z_score: np.ndarray, crop_type: np.ndarray, z_good: float, z_normal: float, z_below: float) -> np.ndarray:
    """按 Z-Score 阈值生成像元级长势等级。"""
    class_mask = (crop_type >= 1) & (crop_type <= 4) & np.isfinite(z_score)
    levels = np.zeros(z_score.shape, dtype="int8")
    levels[class_mask & (z_score > z_good)] = 1
    levels[class_mask & (z_score >= z_normal) & (z_score <= z_good)] = 2
    levels[class_mask & (z_score >= z_below) & (z_score < z_normal)] = 3
    levels[class_mask & (z_score < z_below)] = 4
    return levels


def _write(path: Path, data: np.ndarray, profile: dict[str, Any], dtype: str, nodata: float | int, description: str) -> None:
    """写出单波段 GeoTIFF，并设置 band description。"""
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    if dtype == "float32":
        out_profile["predictor"] = 3
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(data.astype(dtype), 1)
        dst.set_band_description(1, description)


def _level_counts(levels: np.ndarray) -> dict[str, int]:
    """统计 0-4 各长势等级的像元数量。"""
    return {str(level): int(np.count_nonzero(levels == level)) for level in range(0, 5)}


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(args.config)
    parcels = args.parcels or paths.parcel_majority_shp

    if args.current_ndvi:
        current_ndvi, profile = _read_single_band(args.current_ndvi)
    else:
        current_ndvi, profile = _read_ndvi_mvc_from_feature_stack(args.feature_stack, args.metadata)
    current_ndvi = _valid_ndvi(current_ndvi, args.min_valid_ndvi, args.max_valid_ndvi)

    baseline_arrays = _read_baseline_arrays(args, profile)
    if not baseline_arrays:
        raise ValueError("至少需要一个多年同期基准 NDVI 输入。")

    baseline_stack = np.stack(baseline_arrays, axis=0)
    baseline_mean = np.nanmean(baseline_stack, axis=0).astype("float32")
    baseline_std = np.nanstd(baseline_stack, axis=0).astype("float32")
    baseline_std = np.where(np.isfinite(baseline_std), np.maximum(baseline_std, args.min_baseline_std), np.nan)

    anomaly = (current_ndvi - baseline_mean).astype("float32")
    z_score = (anomaly / baseline_std).astype("float32")
    crop_type = _rasterize_crop_type(parcels, args.crop_type_field, profile)
    growth_level = _classify_growth(z_score, crop_type, args.z_good, args.z_normal, args.z_below)

    month_tag = f"{args.target_year}_{args.target_month:02d}"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "ndvi_mvc": args.output_dir / f"NDVI_MVC_{month_tag}.tif",
        "zscore": args.output_dir / f"ZScore_{month_tag}.tif",
        "growth_level": args.output_dir / f"GrowthLevel_{month_tag}.tif",
    }

    _write(paths["ndvi_mvc"], np.where(np.isfinite(current_ndvi), current_ndvi, -9999.0), profile, "float32", -9999.0, "NDVI")
    _write(paths["zscore"], np.where(np.isfinite(z_score), z_score, -9999.0), profile, "float32", -9999.0, "z_score")
    _write(paths["growth_level"], growth_level, profile, "int8", 0, "growth_level")

    crop_mask = (crop_type >= 1) & (crop_type <= 4) & np.isfinite(z_score)
    stats = {
        "target_year": args.target_year,
        "target_month": args.target_month,
        "feature_stack": str(args.feature_stack) if not args.current_ndvi else None,
        "metadata": str(args.metadata) if not args.current_ndvi and args.metadata else None,
        "current_ndvi": str(args.current_ndvi) if args.current_ndvi else None,
        "baseline_years": (
            f"{args.baseline_start_year}-{args.baseline_end_year}"
            if args.baseline_start_year is not None and args.baseline_end_year is not None
            else None
        ),
        "parameters": {
            "min_valid_ndvi": args.min_valid_ndvi,
            "max_valid_ndvi": args.max_valid_ndvi,
            "min_baseline_std": args.min_baseline_std,
            "z_good": args.z_good,
            "z_normal": args.z_normal,
            "z_below": args.z_below,
        },
        "level_labels": LEVEL_LABELS,
        "valid_crop_pixel_count": int(np.count_nonzero(crop_mask)),
        "growth_level_counts": _level_counts(growth_level),
        "z_score_mean": float(np.nanmean(z_score[crop_mask])) if np.any(crop_mask) else None,
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("=== Step2 像元级长势分级完成 ===")
    print(f"NDVI MVC: {paths['ndvi_mvc']}")
    print(f"Z-Score: {paths['zscore']}")
    print(f"GrowthLevel: {paths['growth_level']}")
    print(f"Stats: {args.stats}")


if __name__ == "__main__":
    main()
