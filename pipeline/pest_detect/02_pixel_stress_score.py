"""Step2：像元级病虫害/胁迫异常评分。

本脚本对应 GEE 版 step1_病虫害胁迫：
当前状态相对历史同期偏低 + 相对上一期下降 + 多指数一致异常 +
空间连片性 + 新增扩张，输出像元级异常评分和异常等级。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom

from image_core.spectral import nbr, ndre, ndvi, safe_ratio


DEFAULT_INPUTS_MANIFEST = Path("data/output/pest_detect/inputs/pest_detect_inputs_manifest.json")
DEFAULT_OUTPUT_DIR = Path("data/output/pest_detect/pixel")
DEFAULT_STATS = DEFAULT_OUTPUT_DIR / "pest_step2_stats.json"

INDEX_NAMES = ["ndvi", "ndre", "gndvi", "ndmi", "nbr"]
SCORE_BANDS = [
    "current_anomaly_score",
    "current_anomaly_class",
    "preliminary_anomaly_score",
    "previous_anomaly_score",
    "z_score_anomaly",
    "recent_drop_anomaly",
    "multi_index_agreement",
    "neighborhood_anomaly_density",
    "expansion_density",
    "current_core_anomaly",
    "previous_core_anomaly",
    "anomaly_expansion",
    "persistent_anomaly",
    "relieved_anomaly",
    "ndvi",
    "ndre",
    "gndvi",
    "ndmi",
    "nbr",
]


def configure_gdal_proj() -> None:
    try:
        proj_dir = Path(rasterio.__file__).resolve().parent / "proj_data"
        if (proj_dir / "proj.db").exists():
            os.environ.setdefault("PROJ_LIB", str(proj_dir))
            os.environ.setdefault("PROJ_DATA", str(proj_dir))
        os.environ.setdefault("PROJ_IGNORE_BUILD_INFO", "YES")
    except Exception:
        pass


configure_gdal_proj()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step2：像元级病虫害/胁迫异常评分。")
    parser.add_argument("--feature-stack", type=Path, default=None, help="当前 feature stack。不传则从 inputs manifest 的 current 字段推导。")
    parser.add_argument("--metadata", type=Path, default=None, help="当前 feature stack metadata。不传则从 inputs manifest 推导。")
    parser.add_argument("--inputs-manifest", type=Path, default=DEFAULT_INPUTS_MANIFEST, help="Step1 输出的输入清单。")
    parser.add_argument("--previous-feature-stack", type=Path, default=None, help="上一期 feature stack；未传入时从 inputs manifest 读取。")
    parser.add_argument("--previous-metadata", type=Path, default=None, help="上一期 metadata。")
    parser.add_argument("--pre-previous-feature-stack", type=Path, default=None, help="上上一期 feature stack；未传入时从 inputs manifest 读取。")
    parser.add_argument("--pre-previous-metadata", type=Path, default=None, help="上上一期 metadata。")
    parser.add_argument("--baseline-current-feature-stack", type=Path, action="append", default=None, help="当前窗口历史同期 feature stack，可重复传入。")
    parser.add_argument("--baseline-current-metadata", type=Path, action="append", default=None, help="当前窗口历史同期 metadata。")
    parser.add_argument("--baseline-previous-feature-stack", type=Path, action="append", default=None, help="上一期窗口历史同期 feature stack，可重复传入。")
    parser.add_argument("--baseline-previous-metadata", type=Path, action="append", default=None, help="上一期窗口历史同期 metadata。")
    parser.add_argument("--min-baseline-std", type=float, default=0.03, help="历史标准差下限，避免除以过小标准差。")
    parser.add_argument("--core-threshold", type=float, default=60.0, help="核心异常斑块阈值。")
    parser.add_argument("--neighborhood-radius-m", type=float, default=30.0, help="邻域密度半径，单位米。")
    parser.add_argument("--weight-zscore", type=float, default=0.45)
    parser.add_argument("--weight-recent-drop", type=float, default=0.20)
    parser.add_argument("--weight-agreement", type=float, default=0.15)
    parser.add_argument("--weight-neighborhood", type=float, default=0.10)
    parser.add_argument("--weight-expansion", type=float, default=0.10)
    parser.add_argument("--target-start", default=None, help="当前监测窗口开始日期，仅用于输出命名。")
    parser.add_argument("--target-end", default=None, help="当前监测窗口结束日期，仅用于输出命名。")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS)
    return parser.parse_args()


def _load_band_names(path: Path, metadata_path: Path | None) -> list[str]:
    if metadata_path and metadata_path.exists():
        with open(metadata_path, encoding="utf-8-sig") as f:
            metadata = json.load(f)
        names = metadata.get("band_names") or []
        if names:
            return [str(name) for name in names]
    with rasterio.open(path) as src:
        return [src.descriptions[index - 1] or f"band_{index}" for index in range(1, src.count + 1)]


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _load_region_geometry(path: Path, default_crs: str) -> tuple[dict[str, Any], str]:
    if path.suffix.lower() in {".json", ".geojson"}:
        with open(path, encoding="utf-8-sig") as f:
            geojson = json.load(f)
        crs = default_crs
        if isinstance(geojson.get("crs"), dict):
            crs = str(geojson["crs"].get("properties", {}).get("name") or default_crs)
        if geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
            if not features:
                raise ValueError(f"AOI contains no features: {path}")
            if len(features) == 1:
                return features[0]["geometry"], crs
            return {"type": "GeometryCollection", "geometries": [item["geometry"] for item in features]}, crs
        if geojson.get("type") == "Feature":
            return geojson["geometry"], crs
        return geojson, crs

    import geopandas as gpd

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"AOI contains no features: {path}")
    geometry = gdf.geometry.unary_union.__geo_interface__
    crs = str(gdf.crs) if gdf.crs is not None else default_crs
    return geometry, crs


def _resolve_aoi_path(metadata_path: Path | None, manifest: dict[str, Any]) -> Path | None:
    metadata = _load_json(metadata_path)
    for candidate in [
        metadata.get("aoi"),
        (metadata.get("aoi_mask") or {}).get("path"),
        manifest.get("geometry") if manifest else None,
    ]:
        if candidate:
            path = Path(str(candidate))
            if path.exists():
                return path
    return None


def _build_aoi_mask(aoi_path: Path | None, profile: dict[str, Any]) -> np.ndarray:
    shape = (int(profile["height"]), int(profile["width"]))
    if aoi_path is None:
        return np.ones(shape, dtype=bool)
    default_crs = "EPSG:4326"
    geometry, src_crs = _load_region_geometry(aoi_path, default_crs)
    dst_crs = profile.get("crs")
    if dst_crs is not None and src_crs:
        geometry = transform_geom(src_crs, dst_crs, geometry)
    mask = geometry_mask([geometry], out_shape=shape, transform=profile["transform"], invert=True)
    if not np.any(mask):
        raise ValueError(f"AOI has no overlap with current feature stack: {aoi_path}")
    return mask


def _band_index(names: list[str], candidates: list[str]) -> int | None:
    lower_names = [name.lower() for name in names]
    for candidate in candidates:
        candidate = candidate.lower()
        for index, name in enumerate(lower_names, start=1):
            if name == candidate or name.endswith(f"_{candidate}"):
                return index
    return None


def _read_feature_indices(path: Path, metadata_path: Path | None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"feature stack 不存在：{path}")
    names = _load_band_names(path, metadata_path)
    with rasterio.open(path) as src:
        profile = src.profile.copy()

        def read_any(candidates: list[str]) -> np.ndarray | None:
            index = _band_index(names, candidates)
            if index is None:
                return None
            return src.read(index, masked=False).astype("float32")

        red = read_any(["red"])
        green = read_any(["green"])
        nir = read_any(["nir"])
        rededge = read_any(["rededge", "rededge1"])
        swir = read_any(["swir", "swir16"])

        result: dict[str, np.ndarray] = {}
        result["ndvi"] = read_any(["ndvi"]) if read_any(["ndvi"]) is not None else ndvi(nir, red)
        result["ndre"] = read_any(["ndre"]) if read_any(["ndre"]) is not None else ndre(nir, rededge)
        result["gndvi"] = safe_ratio(nir, green)
        result["ndmi"] = safe_ratio(nir, swir)
        result["nbr"] = read_any(["nbr"]) if read_any(["nbr"]) is not None else nbr(nir, swir)

    for key, value in result.items():
        result[key] = np.where(np.isfinite(value), value, np.nan).astype("float32")
    return result, profile


def _assert_same_grid(reference: dict[str, Any], candidate: dict[str, Any], label: str) -> None:
    mismatched = [key for key in ("width", "height", "crs", "transform") if reference.get(key) != candidate.get(key)]
    if mismatched:
        raise ValueError(f"{label} 与当前 feature stack 网格不一致：{', '.join(mismatched)}")


def _load_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.inputs_manifest and args.inputs_manifest.exists():
        with open(args.inputs_manifest, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _resolve_inputs(args: argparse.Namespace) -> dict[str, Any]:
    manifest = _load_manifest(args)
    current = manifest.get("current") or {}
    previous = manifest.get("previous") or {}
    pre_previous = manifest.get("pre_previous") or {}
    baseline = manifest.get("baseline") or {}

    current_feature_stack = args.feature_stack or Path(current.get("feature_stack", ""))
    current_metadata = args.metadata or Path(current.get("metadata", ""))

    current_baseline_paths = args.baseline_current_feature_stack
    current_baseline_meta = args.baseline_current_metadata
    previous_baseline_paths = args.baseline_previous_feature_stack
    previous_baseline_meta = args.baseline_previous_metadata

    if current_baseline_paths is None:
        current_items = baseline.get("current_window") or []
        current_baseline_paths = [Path(item["feature_stack"]) for item in current_items]
        current_baseline_meta = [Path(item["metadata"]) for item in current_items]
    if previous_baseline_paths is None:
        previous_items = baseline.get("previous_window") or []
        previous_baseline_paths = [Path(item["feature_stack"]) for item in previous_items]
        previous_baseline_meta = [Path(item["metadata"]) for item in previous_items]

    return {
        "current_feature_stack": current_feature_stack,
        "current_metadata": current_metadata,
        "previous_feature_stack": args.previous_feature_stack or Path(previous.get("feature_stack", "")),
        "previous_metadata": args.previous_metadata or Path(previous.get("metadata", "")),
        "pre_previous_feature_stack": args.pre_previous_feature_stack or Path(pre_previous.get("feature_stack", "")),
        "pre_previous_metadata": args.pre_previous_metadata or Path(pre_previous.get("metadata", "")),
        "current_baseline_paths": current_baseline_paths or [],
        "current_baseline_meta": current_baseline_meta or [],
        "previous_baseline_paths": previous_baseline_paths or [],
        "previous_baseline_meta": previous_baseline_meta or [],
        "manifest": manifest,
    }


def _baseline_stats(
    paths: list[Path],
    metadata_paths: list[Path],
    reference_profile: dict[str, Any],
    label: str,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if not paths:
        raise ValueError(f"缺少 {label} 历史同期基线 feature stack。")
    if metadata_paths and len(metadata_paths) != len(paths):
        raise ValueError(f"{label} metadata 数量必须与 feature stack 数量一致。")

    by_index: dict[str, list[np.ndarray]] = {name: [] for name in INDEX_NAMES}
    for index, path in enumerate(paths):
        metadata = metadata_paths[index] if metadata_paths else None
        arrays, profile = _read_feature_indices(path, metadata)
        _assert_same_grid(reference_profile, profile, str(path))
        for name in INDEX_NAMES:
            by_index[name].append(arrays[name])

    means = {name: np.nanmean(np.stack(values, axis=0), axis=0).astype("float32") for name, values in by_index.items()}
    stds = {name: np.nanstd(np.stack(values, axis=0), axis=0).astype("float32") for name, values in by_index.items()}
    return means, stds


def _clamp01(data: np.ndarray) -> np.ndarray:
    return np.clip(data, 0.0, 1.0).astype("float32")


def _anomaly_from_z(source: dict[str, np.ndarray], mean: dict[str, np.ndarray], std: dict[str, np.ndarray], name: str, min_std: float) -> np.ndarray:
    denominator = np.maximum(std[name], min_std)
    z_drop = (mean[name] - source[name]) / denominator
    return _clamp01((z_drop - 0.5) / 2.0)


def _recent_drop(source: dict[str, np.ndarray], reference: dict[str, np.ndarray], name: str, expected_drop: float) -> np.ndarray:
    return _clamp01((reference[name] - source[name]) / expected_drop)


def _score_without_space(
    source: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    baseline_mean: dict[str, np.ndarray],
    baseline_std: dict[str, np.ndarray],
    min_std: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z_parts = {name: _anomaly_from_z(source, baseline_mean, baseline_std, name, min_std) for name in INDEX_NAMES}
    z_score_anomaly = np.nanmean(np.stack([z_parts[name] for name in INDEX_NAMES], axis=0), axis=0).astype("float32")
    recent_drop_anomaly = np.nanmean(
        np.stack(
            [
                _recent_drop(source, reference, "ndvi", 0.12),
                _recent_drop(source, reference, "ndre", 0.08),
                _recent_drop(source, reference, "ndmi", 0.10),
            ],
            axis=0,
        ),
        axis=0,
    ).astype("float32")
    agreement = np.nanmean(
        np.stack(
            [
                z_parts["ndvi"] > 0.45,
                z_parts["ndre"] > 0.45,
                z_parts["ndmi"] > 0.45,
                z_parts["nbr"] > 0.45,
            ],
            axis=0,
        ).astype("float32"),
        axis=0,
    ).astype("float32")
    preliminary = (z_score_anomaly * 0.55 + recent_drop_anomaly * 0.25 + agreement * 0.20).astype("float32")
    return preliminary, z_score_anomaly, recent_drop_anomaly, agreement


def _density(mask: np.ndarray, radius_pixels: int) -> np.ndarray:
    if radius_pixels <= 0:
        return mask.astype("float32")
    source = mask.astype("float32")
    total = np.zeros(source.shape, dtype="float32")
    kernel_count = 0
    for dy in range(-radius_pixels, radius_pixels + 1):
        for dx in range(-radius_pixels, radius_pixels + 1):
            if dx * dx + dy * dy > radius_pixels * radius_pixels:
                continue
            kernel_count += 1
            src_y0 = max(0, -dy)
            src_y1 = source.shape[0] - max(0, dy)
            src_x0 = max(0, -dx)
            src_x1 = source.shape[1] - max(0, dx)
            dst_y0 = max(0, dy)
            dst_y1 = source.shape[0] - max(0, -dy)
            dst_x0 = max(0, dx)
            dst_x1 = source.shape[1] - max(0, -dx)
            total[dst_y0:dst_y1, dst_x0:dst_x1] += source[src_y0:src_y1, src_x0:src_x1]
    return total / float(kernel_count)


def _write_multiband(path: Path, arrays: list[np.ndarray], names: list[str], profile: dict[str, Any]) -> None:
    out_profile = profile.copy()
    out_profile.update(count=len(arrays), dtype="float32", nodata=-9999.0, compress="deflate", predictor=3)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        for index, (array, name) in enumerate(zip(arrays, names), start=1):
            dst.write(np.where(np.isfinite(array), array, -9999.0).astype("float32"), index)
            dst.set_band_description(index, name)


def _write_single(path: Path, array: np.ndarray, profile: dict[str, Any], dtype: str, nodata: float | int, name: str) -> None:
    out_profile = profile.copy()
    out_profile.update(count=1, dtype=dtype, nodata=nodata, compress="deflate")
    if dtype == "float32":
        out_profile["predictor"] = 3
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype), 1)
        dst.set_band_description(1, name)


def main() -> None:
    args = parse_args()
    resolved = _resolve_inputs(args)

    current, profile = _read_feature_indices(resolved["current_feature_stack"], resolved["current_metadata"])
    aoi_path = _resolve_aoi_path(resolved["current_metadata"], resolved["manifest"])
    aoi_mask = _build_aoi_mask(aoi_path, profile)
    previous, previous_profile = _read_feature_indices(resolved["previous_feature_stack"], resolved["previous_metadata"])
    pre_previous, pre_previous_profile = _read_feature_indices(resolved["pre_previous_feature_stack"], resolved["pre_previous_metadata"])
    _assert_same_grid(profile, previous_profile, "上一期 feature stack")
    _assert_same_grid(profile, pre_previous_profile, "上上一期 feature stack")

    current_mean, current_std = _baseline_stats(
        resolved["current_baseline_paths"],
        resolved["current_baseline_meta"],
        profile,
        "当前窗口",
    )
    previous_mean, previous_std = _baseline_stats(
        resolved["previous_baseline_paths"],
        resolved["previous_baseline_meta"],
        profile,
        "上一期窗口",
    )

    preliminary, z_score_anomaly, recent_drop_anomaly, agreement = _score_without_space(
        current,
        previous,
        current_mean,
        current_std,
        args.min_baseline_std,
    )
    previous_preliminary, _, _, _ = _score_without_space(
        previous,
        pre_previous,
        previous_mean,
        previous_std,
        args.min_baseline_std,
    )

    preliminary_score = preliminary * 100.0
    previous_score = previous_preliminary * 100.0
    preliminary_score = np.where(aoi_mask, preliminary_score, np.nan).astype("float32")
    previous_score = np.where(aoi_mask, previous_score, np.nan).astype("float32")
    current_core = (preliminary_score >= args.core_threshold).astype("float32")
    previous_core = (previous_score >= args.core_threshold).astype("float32")

    pixel_size = max(abs(float(profile["transform"].a)), abs(float(profile["transform"].e)))
    radius_pixels = max(1, int(round(args.neighborhood_radius_m / pixel_size)))
    neighborhood_density = _density(current_core, radius_pixels).astype("float32")
    anomaly_expansion = ((current_core == 1) & (previous_core == 0)).astype("float32")
    persistent_anomaly = ((current_core == 1) & (previous_core == 1)).astype("float32")
    relieved_anomaly = ((current_core == 0) & (previous_core == 1)).astype("float32")
    expansion_density = _density(anomaly_expansion, radius_pixels).astype("float32")

    score01 = (
        z_score_anomaly * args.weight_zscore
        + recent_drop_anomaly * args.weight_recent_drop
        + agreement * args.weight_agreement
        + neighborhood_density * args.weight_neighborhood
        + expansion_density * args.weight_expansion
    ).astype("float32")
    score100 = np.clip(score01 * 100.0, 0.0, 100.0).astype("float32")
    score100 = np.where(aoi_mask, score100, np.nan).astype("float32")

    anomaly_class = np.ones(score100.shape, dtype="int16")
    anomaly_class[score100 >= 30.0] = 2
    anomaly_class[score100 >= 60.0] = 3
    anomaly_class[score100 >= 80.0] = 4
    valid = np.isfinite(score100)
    anomaly_class[~valid] = 0

    target_tag = "current"
    manifest_current = (resolved["manifest"].get("current") or {}) if resolved["manifest"] else {}
    target_start = args.target_start or manifest_current.get("start_date")
    target_end = args.target_end or manifest_current.get("end_date")
    if target_start and target_end:
        target_tag = f"{str(target_start).replace('-', '')}_{str(target_end).replace('-', '')}"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    score_path = args.output_dir / f"pest_anomaly_score_{target_tag}.tif"
    class_path = args.output_dir / f"pest_anomaly_class_{target_tag}.tif"
    stack_path = args.output_dir / f"pest_anomaly_stack_{target_tag}.tif"

    _write_single(score_path, np.where(valid, score100, -9999.0), profile, "float32", -9999.0, "current_anomaly_score")
    _write_single(class_path, anomaly_class, profile, "int16", 0, "current_anomaly_class")
    stack_arrays = [
        score100,
        anomaly_class.astype("float32"),
        preliminary_score,
        previous_score,
        z_score_anomaly,
        recent_drop_anomaly,
        agreement,
        neighborhood_density,
        expansion_density,
        current_core,
        previous_core,
        anomaly_expansion,
        persistent_anomaly,
        relieved_anomaly,
        current["ndvi"],
        current["ndre"],
        current["gndvi"],
        current["ndmi"],
        current["nbr"],
    ]
    stack_arrays = [np.where(aoi_mask, array, np.nan).astype("float32") for array in stack_arrays]
    _write_multiband(stack_path, stack_arrays, SCORE_BANDS, profile)

    stats = {
        "inputs": {
            "feature_stack": str(resolved["current_feature_stack"]),
            "metadata": str(resolved["current_metadata"]),
            "aoi": str(aoi_path) if aoi_path else None,
            "inputs_manifest": str(args.inputs_manifest),
            "previous_feature_stack": str(resolved["previous_feature_stack"]),
            "pre_previous_feature_stack": str(resolved["pre_previous_feature_stack"]),
        },
        "target_start": target_start,
        "target_end": target_end,
        "parameters": {
            "min_baseline_std": args.min_baseline_std,
            "core_threshold": args.core_threshold,
            "neighborhood_radius_m": args.neighborhood_radius_m,
            "weights": {
                "z_score": args.weight_zscore,
                "recent_drop": args.weight_recent_drop,
                "multi_index_agreement": args.weight_agreement,
                "neighborhood_density": args.weight_neighborhood,
                "expansion": args.weight_expansion,
            },
        },
        "score_mean": float(np.nanmean(score100[valid])) if np.any(valid) else None,
        "score_p50": float(np.nanpercentile(score100[valid], 50)) if np.any(valid) else None,
        "score_p75": float(np.nanpercentile(score100[valid], 75)) if np.any(valid) else None,
        "score_p90": float(np.nanpercentile(score100[valid], 90)) if np.any(valid) else None,
        "class_pixel_counts": {str(level): int(np.count_nonzero((anomaly_class == level) & aoi_mask)) for level in range(0, 5)},
        "outputs": {
            "anomaly_score": str(score_path),
            "anomaly_class": str(class_path),
            "anomaly_stack": str(stack_path),
        },
    }
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    with open(args.stats, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("=== Step2 像元级病虫害胁迫评分完成 ===")
    print(f"异常评分：{score_path}")
    print(f"异常等级：{class_path}")
    print(f"统计信息：{args.stats}")


if __name__ == "__main__":
    main()
