"""缓存并标准化 Sentinel-1/2 源影像。

读取 Sentinel-2 和可选的 Sentinel-1 场景清单，把选定源影像按 AOI
对齐到统一网格，并把对齐后的数据保存到本地缓存目录。
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import bounds as geometry_bounds, geometry_mask
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, transform_geom

from data_sources.aws_element84.config import S2_SCALE


# ---------------------------------------------------------------------------
# GDAL / PROJ 配置

def configure_gdal_proj() -> None:
    try:
        import rasterio as _rasterio

        proj_dir = Path(_rasterio.__file__).resolve().parent / "proj_data"
        if (proj_dir / "proj.db").exists():
            os.environ.setdefault("PROJ_LIB", str(proj_dir))
            os.environ.setdefault("PROJ_DATA", str(proj_dir))
        os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
        os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
        os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.tiff")
        os.environ.setdefault("GDAL_HTTP_MULTIRANGE", "YES")
        os.environ.setdefault("VSI_CACHE", "TRUE")
        os.environ.setdefault("VSI_CACHE_SIZE", "50000000")
        os.environ.setdefault("PROJ_IGNORE_BUILD_INFO", "YES")
    except Exception:
        pass


configure_gdal_proj()

# ---------------------------------------------------------------------------
# 通用常量
# ---------------------------------------------------------------------------

DEFAULT_S2_MANIFEST = Path("data/exported/shared/manifests/sentinel2_scenes.json")
DEFAULT_S1_MANIFEST = Path("data/exported/shared/manifests/sentinel1_scenes.json")
DEFAULT_CACHE_DIR = Path("data/exported/cache")
DEFAULT_LOCAL_SOURCE_DIR = Path("data/source/runtime/aws_local")
DEFAULT_LOCAL_S2_MANIFEST = Path("data/exported/shared/manifests/sentinel2_scenes_local.json")
DEFAULT_LOCAL_S1_MANIFEST = Path("data/exported/shared/manifests/sentinel1_scenes_local.json")
DEFAULT_RESOLUTION_M = 10.0

SEMANTIC_ASSETS = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "rededge": "rededge1",
    "nir": "nir",
    "swir": "swir16",
    "scl": "scl",
}
BASE_BANDS = ["blue", "green", "red", "rededge", "nir", "swir"]


# ---------------------------------------------------------------------------
# 清单与几何辅助函数

def load_manifest(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def load_aoi_geometry(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        geojson = json.load(f)
    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
        if not features:
            raise ValueError(f"{path} 中没有找到任何要素")
        if len(features) == 1:
            return features[0]["geometry"]
        return {"type": "GeometryCollection", "geometries": [item["geometry"] for item in features]}
    if geojson.get("type") == "Feature":
        return geojson["geometry"]
    return geojson


def select_timepoints(manifest: dict[str, Any] | None, labels: list[str] | None) -> list[dict[str, Any]]:
    if not manifest:
        return []
    timepoints = manifest.get("timepoints", [])
    if not labels:
        return timepoints
    selected = set(labels)
    result = [item for item in timepoints if item.get("label") in selected]
    if not result:
        raise ValueError(f"没有找到请求的时间点：{sorted(selected)}")
    return result


def first_scene_with_red(timepoints: list[dict[str, Any]]) -> dict[str, Any]:
    for timepoint in timepoints:
        for scene in timepoint.get("scenes", []):
            if scene.get("assets", {}).get("red"):
                return scene
    raise ValueError("没有找到包含 red 资产的 Sentinel-2 场景")


# ---------------------------------------------------------------------------
# 网格与缓存基础设施

def normalize_href(href: str) -> str:
    if href.startswith("s3://"):
        bucket_key = href[len("s3://"):]
        bucket, key = bucket_key.split("/", 1)
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return href


def build_reference_grid(
    sample_href: str, aoi_geometry: dict[str, Any], resolution: float
) -> tuple[Any, Any, int, int, dict[str, Any]]:
    with rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        GDAL_HTTP_MULTIRANGE="YES",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="50000000",
    ):
        with rasterio.open(normalize_href(sample_href)) as src:
            dst_crs = src.crs
            bounds = transform_bounds("EPSG:4326", dst_crs, *geometry_bounds(aoi_geometry), densify_pts=21)
            left, bottom, right, top = bounds
            width = max(1, int(math.ceil((right - left) / resolution)))
            height = max(1, int(math.ceil((top - bottom) / resolution)))
            transform = from_origin(left, top, resolution, resolution)
    aoi_in_dst_crs = transform_geom("EPSG:4326", dst_crs, aoi_geometry)
    return dst_crs, transform, width, height, aoi_in_dst_crs


def cache_path_for(
    cache_dir: Path,
    href: str,
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
    resampling: Resampling,
    scale: float,
) -> Path:
    key = json.dumps(
        {
            "href": href,
            "crs": str(dst_crs),
            "transform": tuple(round(v, 8) for v in transform),
            "width": width,
            "height": height,
            "resampling": resampling.name,
            "scale": scale,
        },
        sort_keys=True,
    )
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.npy"


def read_asset_on_grid(
    href: str,
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
    resampling: Resampling,
    scale: float = S2_SCALE,
    cache_dir: Path | None = None,
) -> np.ndarray:
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_path_for(cache_dir, href, dst_crs, transform, width, height, resampling, scale)
        if cache_path.exists():
            return np.load(cache_path).astype("float32")

    data = read_asset_raw_on_grid(href, dst_crs, transform, width, height, resampling)
    array = data.astype("float32") / scale
    if cache_path is not None:
        np.save(cache_path, array)
    return array


def read_asset_raw_on_grid(
    href: str,
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
    resampling: Resampling,
) -> np.ndarray:
    """读取源影像资产并对齐到项目网格，不进行反射率缩放。"""
    with rasterio.Env(
        AWS_NO_SIGN_REQUEST="YES",
        GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
        CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.tiff",
        GDAL_HTTP_MULTIRANGE="YES",
        VSI_CACHE="TRUE",
        VSI_CACHE_SIZE="50000000",
    ):
        with rasterio.open(normalize_href(href)) as src:
            # Sentinel-1 GRD 资产可能只有 GCP 而没有内嵌 CRS，WarpedVRT 无法直接处理。
            if src.crs is None and src.gcps:
                raise RuntimeError(
                    f"暂不支持仅包含 GCP 的 Sentinel-1 数据源；请使用仅 S2 模式。数据源：{href[:80]}..."
                )
            with WarpedVRT(
                src,
                crs=dst_crs,
                transform=transform,
                width=width,
                height=height,
                resampling=resampling,
                src_nodata=src.nodata,
                dst_nodata=0,
            ) as vrt:
                return vrt.read(1, out_dtype="float32", masked=False)


# ---------------------------------------------------------------------------
# 场景级缓存辅助函数

def cache_s2_scene(
    scene: dict[str, Any],
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
    cache_dir: Path,
) -> None:
    assets = scene.get("assets", {})
    for semantic_name in BASE_BANDS:
        asset_name = SEMANTIC_ASSETS[semantic_name]
        if asset_name not in assets:
            continue
        print(f"正在缓存 S2 {scene.get('id')} {semantic_name}", flush=True)
        read_asset_on_grid(
            assets[asset_name],
            dst_crs,
            transform,
            width,
            height,
            Resampling.bilinear,
            cache_dir=cache_dir,
        )

    scl_name = SEMANTIC_ASSETS["scl"]
    if scl_name in assets:
        print(f"正在缓存 S2 {scene.get('id')} scl", flush=True)
        read_asset_on_grid(
            assets[scl_name],
            dst_crs,
            transform,
            width,
            height,
            Resampling.nearest,
            scale=1.0,
            cache_dir=cache_dir,
        )


def cache_s1_scene(
    scene: dict[str, Any],
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
    cache_dir: Path,
) -> None:
    assets = scene.get("assets", {})
    for band_name in ["vv", "vh"]:
        if band_name not in assets:
            continue
        print(f"正在缓存 S1 {scene.get('id')} {band_name}", flush=True)
        read_asset_on_grid(
            assets[band_name],
            dst_crs,
            transform,
            width,
            height,
            Resampling.bilinear,
            scale=1.0,
            cache_dir=cache_dir,
        )


# ---------------------------------------------------------------------------
# GeoTIFF 落库辅助函数，用于生产本地资产模式
# ---------------------------------------------------------------------------

def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in value)


def write_aligned_geotiff(
    output_path: Path,
    data: np.ndarray,
    dst_crs: Any,
    transform: Any,
    band_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = data.shape
    blockxsize = 256 if width >= 256 else 16
    blockysize = 256 if height >= 256 else 16
    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "dtype": "float32",
        "crs": dst_crs,
        "transform": transform,
        "nodata": 0,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": blockxsize,
        "blockysize": blockysize,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data.astype("float32"), 1)
        dst.set_band_description(1, band_name)


def materialize_asset(
    *,
    href: str,
    output_path: Path,
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
    resampling: Resampling,
    band_name: str,
    overwrite: bool,
) -> Path:
    if output_path.exists() and not overwrite:
        return output_path
    data = read_asset_raw_on_grid(href, dst_crs, transform, width, height, resampling)
    write_aligned_geotiff(output_path, data, dst_crs, transform, band_name)
    return output_path


def materialize_scene_assets(
    scene: dict[str, Any],
    *,
    sensor: str,
    asset_names: list[str],
    nearest_assets: set[str],
    local_root: Path,
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
    overwrite: bool,
) -> dict[str, str]:
    assets = scene.get("assets", {})
    scene_dir = local_root / sensor / safe_name(scene.get("id") or "unknown_scene")
    local_assets: dict[str, str] = {}
    for asset_name in asset_names:
        href = assets.get(asset_name)
        if not href:
            continue
        output_path = scene_dir / f"{safe_name(asset_name)}.tif"
        resampling = Resampling.nearest if asset_name in nearest_assets else Resampling.bilinear
        print(f"正在落库 {sensor} {scene.get('id')} {asset_name} -> {output_path}", flush=True)
        materialize_asset(
            href=href,
            output_path=output_path,
            dst_crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            resampling=resampling,
            band_name=asset_name,
            overwrite=overwrite,
        )
        local_assets[asset_name] = str(output_path)
    return local_assets


def write_local_manifest(
    *,
    source_manifest: dict[str, Any],
    output_path: Path,
    sensor: str,
    asset_names: list[str],
    nearest_assets: set[str],
    local_root: Path,
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
    overwrite: bool,
) -> None:
    local_manifest = copy.deepcopy(source_manifest)
    materialized_by_scene: dict[str, dict[str, str]] = {}

    def update_scene(scene: dict[str, Any]) -> None:
        scene_id = scene.get("id", "")
        if scene_id not in materialized_by_scene:
            materialized_by_scene[scene_id] = materialize_scene_assets(
                scene,
                sensor=sensor,
                asset_names=asset_names,
                nearest_assets=nearest_assets,
                local_root=local_root,
                dst_crs=dst_crs,
                transform=transform,
                width=width,
                height=height,
                overwrite=overwrite,
            )
        scene.setdefault("assets", {}).update(materialized_by_scene[scene_id])

    for scene in local_manifest.get("scenes", []):
        update_scene(scene)
    for timepoint in local_manifest.get("timepoints", []):
        for scene in timepoint.get("scenes", []):
            update_scene(scene)

    local_manifest["provider"] = f"{source_manifest.get('provider', 'AWS Open Data')}（已落库为本地 GeoTIFF 资产）"
    local_manifest["local_asset_mode"] = "aoi_aligned_geotiff"
    local_manifest["local_source_dir"] = str(local_root)
    local_manifest["local_grid"] = {
        "crs": str(dst_crs),
        "transform": tuple(transform),
        "width": width,
        "height": height,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(local_manifest, f, indent=2, ensure_ascii=False)


def limit_timepoint_scenes(timepoints: list[dict[str, Any]], max_scenes: int | None) -> list[dict[str, Any]]:
    if max_scenes is None:
        return timepoints
    limited = copy.deepcopy(timepoints)
    for timepoint in limited:
        timepoint["scenes"] = timepoint.get("scenes", [])[:max_scenes]
        timepoint["scene_count"] = len(timepoint["scenes"])
    return limited


# ---------------------------------------------------------------------------
# 命令行入口

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="缓存或落库与 AOI 对齐的 Sentinel-1/2 源栅格。")
    parser.add_argument("--s2-manifest", type=Path, default=DEFAULT_S2_MANIFEST)
    parser.add_argument("--s1-manifest", type=Path, default=DEFAULT_S1_MANIFEST)
    parser.add_argument("--include-s1", action="store_true", help="同时处理 Sentinel-1 VV/VH 资产")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument(
        "--materialize-local-assets",
        action="store_true",
        help="写出与 AOI 对齐的 GeoTIFF 资产和本地清单，供后续步骤全本地处理。",
    )
    parser.add_argument("--local-source-dir", type=Path, default=DEFAULT_LOCAL_SOURCE_DIR)
    parser.add_argument("--local-s2-manifest", type=Path, default=DEFAULT_LOCAL_S2_MANIFEST)
    parser.add_argument("--local-s1-manifest", type=Path, default=DEFAULT_LOCAL_S1_MANIFEST)
    parser.add_argument("--overwrite-local-assets", action="store_true")
    parser.add_argument("--resolution", type=float, default=DEFAULT_RESOLUTION_M)
    parser.add_argument("--max-scenes-per-timepoint", type=int, default=None)
    parser.add_argument("--timepoints", nargs="*", default=None, help="可选时间点标签，例如 2025-04 2025-07")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    s2_manifest = load_manifest(args.s2_manifest)
    s2_timepoints = select_timepoints(s2_manifest, args.timepoints)
    geometry_path = Path(s2_manifest["geometry"])
    aoi_geometry = load_aoi_geometry(geometry_path)
    sample_scene = first_scene_with_red(s2_timepoints)

    dst_crs, transform, width, height, aoi_in_dst_crs = build_reference_grid(
        sample_scene["assets"]["red"],
        aoi_geometry,
        args.resolution,
    )
    geometry_mask([aoi_in_dst_crs], out_shape=(height, width), transform=transform, invert=True)

    if args.materialize_local_assets:
        s2_timepoints = limit_timepoint_scenes(s2_timepoints, args.max_scenes_per_timepoint)
        s2_local_manifest = copy.deepcopy(s2_manifest)
        s2_local_manifest["timepoints"] = s2_timepoints
        s2_scene_ids = {scene.get("id") for tp in s2_timepoints for scene in tp.get("scenes", [])}
        s2_local_manifest["scenes"] = [
            scene for scene in s2_manifest.get("scenes", []) if scene.get("id") in s2_scene_ids
        ]
        s2_local_manifest["scene_count"] = len(s2_local_manifest["scenes"])
        s2_local_manifest["timepoint_count"] = len(s2_timepoints)
        write_local_manifest(
            source_manifest=s2_local_manifest,
            output_path=args.local_s2_manifest,
            sensor="sentinel2",
            asset_names=["blue", "green", "red", "rededge1", "nir", "swir16", "scl"],
            nearest_assets={"scl"},
            local_root=args.local_source_dir,
            dst_crs=dst_crs,
            transform=transform,
            width=width,
            height=height,
            overwrite=args.overwrite_local_assets,
        )
        print(f"已保存本地 Sentinel-2 清单：{args.local_s2_manifest}", flush=True)

        if args.include_s1 and args.s1_manifest.exists():
            s1_manifest = load_manifest(args.s1_manifest)
            s1_timepoints = limit_timepoint_scenes(
                select_timepoints(s1_manifest, args.timepoints),
                args.max_scenes_per_timepoint,
            )
            s1_local_manifest = copy.deepcopy(s1_manifest)
            s1_local_manifest["timepoints"] = s1_timepoints
            s1_scene_ids = {scene.get("id") for tp in s1_timepoints for scene in tp.get("scenes", [])}
            s1_local_manifest["scenes"] = [
                scene for scene in s1_manifest.get("scenes", []) if scene.get("id") in s1_scene_ids
            ]
            s1_local_manifest["scene_count"] = len(s1_local_manifest["scenes"])
            s1_local_manifest["timepoint_count"] = len(s1_timepoints)
            write_local_manifest(
                source_manifest=s1_local_manifest,
                output_path=args.local_s1_manifest,
                sensor="sentinel1",
                asset_names=["vv", "vh"],
                nearest_assets=set(),
                local_root=args.local_source_dir,
                dst_crs=dst_crs,
                transform=transform,
                width=width,
                height=height,
                overwrite=args.overwrite_local_assets,
            )
            print(f"已保存本地 Sentinel-1 清单：{args.local_s1_manifest}", flush=True)

        print("本地落库完成。请在 Sentinel 特征构建中使用 *_local.json 清单。", flush=True)
        return

    cached_s2 = 0
    for timepoint in s2_timepoints:
        scenes = timepoint.get("scenes", [])
        if args.max_scenes_per_timepoint is not None:
            scenes = scenes[: args.max_scenes_per_timepoint]
        for scene in scenes:
            cache_s2_scene(scene, dst_crs, transform, width, height, args.cache_dir)
            cached_s2 += 1

    cached_s1 = 0
    if args.include_s1 and args.s1_manifest.exists():
        s1_manifest = load_manifest(args.s1_manifest)
        s1_timepoints = select_timepoints(s1_manifest, args.timepoints)
        for timepoint in s1_timepoints:
            scenes = timepoint.get("scenes", [])
            if args.max_scenes_per_timepoint is not None:
                scenes = scenes[: args.max_scenes_per_timepoint]
            for scene in scenes:
                cache_s1_scene(scene, dst_crs, transform, width, height, args.cache_dir)
                cached_s1 += 1

    print(f"已缓存 Sentinel-2 场景数：{cached_s2}", flush=True)
    print(f"已缓存 Sentinel-1 场景数：{cached_s1}", flush=True)
    print(f"缓存目录：{args.cache_dir}", flush=True)


if __name__ == "__main__":
    main()


