"""从 Sentinel-1/2 数据构建标准特征栈。

读取 Sentinel-2 和可选的 Sentinel-1 场景清单，按配置时相生成合成影像，
计算光谱指数和雷达特征，并写出后续分类、训练、估产共用的多波段
GeoTIFF 与 metadata。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask

from data_sources.aws_element84.config import CLOUD_SCL_VALUES
from image_core.spectral import evi, nbr, ndmi, ndre, ndvi, ndwi
from data_sources.aws_element84 import cache_source_rasters as cache
from configs.paths import ProjectPaths

INDEX_BANDS = ["ndvi", "ndwi", "evi", "ndre", "ndmi", "nbr"]
S1_BANDS = ["vv", "vh", "vh_vv"]


# ---------------------------------------------------------------------------
# 单景影像数组读取
# ---------------------------------------------------------------------------

def scene_to_s2_arrays(
    scene: dict[str, Any],
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
    aoi_mask: np.ndarray,
    cache_dir: Path | None,
) -> dict[str, np.ndarray] | None:
    assets = scene.get("assets", {})
    if any(cache.SEMANTIC_ASSETS[name] not in assets for name in cache.BASE_BANDS):
        return None

    arrays: dict[str, np.ndarray] = {}
    for semantic_name in cache.BASE_BANDS:
        arrays[semantic_name] = cache.read_asset_on_grid(
            assets[cache.SEMANTIC_ASSETS[semantic_name]],
            dst_crs,
            transform,
            width,
            height,
            Resampling.bilinear,
            cache_dir=cache_dir,
        )

    valid = aoi_mask.copy()
    valid &= arrays["red"] > 0
    valid &= arrays["nir"] > 0

    scl_href = assets.get(cache.SEMANTIC_ASSETS["scl"])
    if scl_href:
        scl = cache.read_asset_on_grid(
            scl_href,
            dst_crs,
            transform,
            width,
            height,
            Resampling.nearest,
            scale=1.0,
            cache_dir=cache_dir,
        ).astype("uint8")
        valid &= ~np.isin(scl, list(CLOUD_SCL_VALUES))

    return {name: np.where(valid, data, np.nan).astype("float32") for name, data in arrays.items()}


def scene_to_s1_arrays(
    scene: dict[str, Any],
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
    aoi_mask: np.ndarray,
    cache_dir: Path | None,
) -> dict[str, np.ndarray] | None:
    assets = scene.get("assets", {})
    if "vv" not in assets or "vh" not in assets:
        return None
    vv = cache.read_asset_on_grid(assets["vv"], dst_crs, transform, width, height, Resampling.bilinear, scale=1.0, cache_dir=cache_dir)
    vh = cache.read_asset_on_grid(assets["vh"], dst_crs, transform, width, height, Resampling.bilinear, scale=1.0, cache_dir=cache_dir)
    valid = aoi_mask.copy()
    valid &= np.isfinite(vv)
    valid &= np.isfinite(vh)
    valid &= vv != 0
    valid &= vh != 0
    vv = np.where(valid, vv, np.nan).astype("float32")
    vh = np.where(valid, vh, np.nan).astype("float32")
    return {"vv": vv, "vh": vh, "vh_vv": (vh / (vv + 1e-6)).astype("float32")}


# ---------------------------------------------------------------------------
# 合成与指数计算工具

def nanmedian_stack(arrays: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(arrays, axis=0)
    return np.nanmedian(stack, axis=0).astype("float32")


def add_s2_indices(composite: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    red = composite["red"]
    green = composite["green"]
    blue = composite["blue"]
    nir = composite["nir"]
    rededge = composite["rededge"]
    swir = composite["swir"]
    composite["ndvi"] = ndvi(nir, red)
    composite["ndwi"] = ndwi(green, nir)
    composite["evi"] = evi(nir, red, blue)
    composite["ndre"] = ndre(nir, rededge)
    composite["ndmi"] = ndmi(nir, swir)
    composite["nbr"] = nbr(nir, swir)
    return composite


def build_s2_timepoint_composite(
    timepoint: dict[str, Any],
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
    aoi_mask: np.ndarray,
    max_scenes: int | None,
    cache_dir: Path | None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    scene_arrays: dict[str, list[np.ndarray]] = {name: [] for name in cache.BASE_BANDS}
    used_scene_ids: list[str] = []
    scenes = timepoint.get("scenes", [])[:max_scenes] if max_scenes is not None else timepoint.get("scenes", [])
    for scene in scenes:
        print(f"正在读取 Sentinel-2 场景 {scene.get('id', '<unknown>')}，时相 {timepoint.get('label')}", flush=True)
        try:
            arrays = scene_to_s2_arrays(scene, dst_crs, transform, width, height, aoi_mask, cache_dir)
        except Exception as exc:
            print(f"跳过 Sentinel-2 场景 {scene.get('id', '<unknown>')}：{str(exc)[:120]}", flush=True)
            continue
        if arrays is None:
            continue
        used_scene_ids.append(scene.get("id", ""))
        for name in cache.BASE_BANDS:
            scene_arrays[name].append(arrays[name])
    if not used_scene_ids:
        raise ValueError(f"时相 {timepoint.get('label')} 没有可用的 Sentinel-2 场景")
    return add_s2_indices({name: nanmedian_stack(values) for name, values in scene_arrays.items()}), used_scene_ids


def build_s1_timepoint_composite(
    timepoint: dict[str, Any],
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
    aoi_mask: np.ndarray,
    max_scenes: int | None,
    cache_dir: Path | None,
) -> tuple[dict[str, np.ndarray], list[str]]:
    scene_arrays: dict[str, list[np.ndarray]] = {name: [] for name in S1_BANDS}
    used_scene_ids: list[str] = []
    scenes = timepoint.get("scenes", [])[:max_scenes] if max_scenes is not None else timepoint.get("scenes", [])
    for scene in scenes:
        print(f"正在读取 Sentinel-1 场景 {scene.get('id', '<unknown>')}，时相 {timepoint.get('label')}", flush=True)
        try:
            arrays = scene_to_s1_arrays(scene, dst_crs, transform, width, height, aoi_mask, cache_dir)
        except Exception as exc:
            print(f"跳过 Sentinel-1 场景 {scene.get('id', '<unknown>')}：{str(exc)[:120]}", flush=True)
            continue
        if arrays is None:
            continue
        used_scene_ids.append(scene.get("id", ""))
        for name in S1_BANDS:
            scene_arrays[name].append(arrays[name])
    if not used_scene_ids:
        raise ValueError(f"时相 {timepoint.get('label')} 没有可用的 Sentinel-1 场景")
    return {name: nanmedian_stack(values) for name, values in scene_arrays.items()}, used_scene_ids


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def write_feature_stack(
    output_path: Path,
    band_arrays: list[np.ndarray],
    band_names: list[str],
    dst_crs: Any,
    transform: Any,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = band_arrays[0].shape
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(band_arrays),
        "dtype": "float32",
        "crs": dst_crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        for index, (name, data) in enumerate(zip(band_names, band_arrays), start=1):
            dst.write(data.astype("float32"), index)
            dst.set_band_description(index, name)


# ---------------------------------------------------------------------------
# 命令行入口

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建月度 Sentinel-1/2 标准特征栈。")
    parser.add_argument("--s2-manifest", type=Path, default=cache.DEFAULT_S2_MANIFEST)
    parser.add_argument("--s1-manifest", type=Path, default=cache.DEFAULT_S1_MANIFEST)
    parser.add_argument("--include-s1", action="store_true", help="包含 Sentinel-1 VV/VH/VH_VV 特征")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--no-cache", action="store_true", help="禁用 AOI 对齐数组缓存")
    parser.add_argument("--resolution", type=float, default=cache.DEFAULT_RESOLUTION_M)
    parser.add_argument("--max-scenes-per-timepoint", type=int, default=None)
    parser.add_argument("--timepoints", nargs="*", default=None, help="可选时间点标签，例如 2025-04 2025-07")
    parser.add_argument(
        "--timepoint-name-mode",
        choices=("slot", "label"),
        default="slot",
        help="默认使用稳定特征前缀 t1/t2/...；选择 label 时保留 2025_07 这类来源标签。",
    )
    return parser.parse_args()


def feature_prefix(timepoint: dict[str, Any], index: int, mode: str) -> str:
    if mode == "label":
        return str(timepoint["label"]).replace("-", "_")
    return f"t{index}"


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(args.config)
    cache_dir = None if args.no_cache else (args.cache_dir or paths.cache_dir)
    output = args.output or paths.feature_stack
    metadata_path = args.metadata or paths.feature_stack_metadata
    s2_manifest = cache.load_manifest(args.s2_manifest)
    s2_timepoints = cache.select_timepoints(s2_manifest, args.timepoints)
    s1_manifest = cache.load_manifest(args.s1_manifest) if args.include_s1 and args.s1_manifest.exists() else None
    s1_by_label = {item.get("label"): item for item in cache.select_timepoints(s1_manifest, args.timepoints)}

    geometry_path = Path(s2_manifest["geometry"])
    aoi_geometry = cache.load_aoi_geometry(geometry_path)
    sample_scene = cache.first_scene_with_red(s2_timepoints)
    dst_crs, transform, width, height, aoi_in_dst_crs = cache.build_reference_grid(
        sample_scene["assets"]["red"], aoi_geometry, args.resolution
    )
    aoi_mask = geometry_mask([aoi_in_dst_crs], out_shape=(height, width), transform=transform, invert=True)

    band_arrays: list[np.ndarray] = []
    band_names: list[str] = []
    s2_metadata = []
    s1_metadata = []

    for timepoint_index, timepoint in enumerate(s2_timepoints, start=1):
        label = feature_prefix(timepoint, timepoint_index, args.timepoint_name_mode)
        s2_composite, s2_used = build_s2_timepoint_composite(
            timepoint, dst_crs, transform, width, height, aoi_mask, args.max_scenes_per_timepoint, cache_dir
        )
        s2_names = cache.BASE_BANDS + INDEX_BANDS
        for name in s2_names:
            band_names.append(f"{label}_{name}")
            band_arrays.append(s2_composite[name])
        s2_metadata.append(
            {
                "label": timepoint["label"],
                "feature_prefix": label,
                "used_scene_count": len(s2_used),
                "used_scene_ids": s2_used,
                "bands": [f"{label}_{name}" for name in s2_names],
            }
        )
        print(f"Built {timepoint['label']} Sentinel-2 composite from {len(s2_used)} scenes", flush=True)

        if args.include_s1:
            s1_timepoint = s1_by_label.get(timepoint["label"])
            if not s1_timepoint or not s1_timepoint.get("scenes"):
                s1_metadata.append({"label": timepoint["label"], "used_scene_count": 0, "error": "该时相没有 Sentinel-1 场景"})
                continue
            try:
                s1_composite, s1_used = build_s1_timepoint_composite(
                    s1_timepoint, dst_crs, transform, width, height, aoi_mask, args.max_scenes_per_timepoint, cache_dir
                )
                for name in S1_BANDS:
                    band_names.append(f"{label}_{name}")
                    band_arrays.append(s1_composite[name])
                s1_metadata.append(
                    {
                        "label": timepoint["label"],
                        "feature_prefix": label,
                        "used_scene_count": len(s1_used),
                        "used_scene_ids": s1_used,
                        "bands": [f"{label}_{name}" for name in S1_BANDS],
                    }
                )
                print(f"Built {timepoint['label']} Sentinel-1 composite from {len(s1_used)} scenes", flush=True)
            except Exception as exc:
                s1_metadata.append({"label": timepoint["label"], "used_scene_count": 0, "error": str(exc)[:200]})
                print(f"跳过时相 {timepoint['label']} 的 Sentinel-1 特征：{str(exc)[:120]}", flush=True)

    write_feature_stack(output, band_arrays, band_names, dst_crs, transform)
    metadata = {
        "source_manifests": {"sentinel2": str(args.s2_manifest), "sentinel1": str(args.s1_manifest) if args.include_s1 else None},
        "aoi": str(geometry_path),
        "output": str(output),
        "cache_dir": str(cache_dir) if cache_dir else None,
        "crs": str(dst_crs),
        "resolution": args.resolution,
        "width": width,
        "height": height,
        "band_count": len(band_names),
        "band_names": band_names,
        "timepoint_name_mode": args.timepoint_name_mode,
        "sentinel2_timepoints": s2_metadata,
        "sentinel1_timepoints": s1_metadata,
        "notes": "Unified Sentinel-1/2 feature stack. S1 uses VV, VH, and VH/VV; S2 uses semantic optical bands and indices.",
    }
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"已保存特征栈：{output}", flush=True)
    print(f"已保存元数据：{metadata_path}", flush=True)


if __name__ == "__main__":
    main()




