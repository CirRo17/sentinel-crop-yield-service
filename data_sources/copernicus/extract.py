"""从 Copernicus SAFE 目录提取 Sentinel-2 波段为单波段 GeoTIFF。

读取 download.py 下载的 .SAFE 目录，找到各波段的 .jp2 文件，
将 20m 波段重采样到 10m 网格，输出与 aws_element84/cache_source_rasters.py
兼容的单波段 GeoTIFF，供 build_features.py 直接消费。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

from data_sources.copernicus.config import SAFE_BAND_MAP

# 10m 波段 — 直接读，不重采样
BANDS_10M = {"blue", "green", "red", "nir"}
# 20m 波段 — 需要重采样到 10m
BANDS_20M = {"rededge", "swir", "swir22", "scl"}


def configure_gdal_proj() -> None:
    """配置 PROJ 库路径，避免 rasterio 找不到投影数据。"""
    try:
        proj_dir = (
            Path(rasterio.__file__).resolve().parent / "proj_data"
        )
        if (proj_dir / "proj.db").exists():
            os.environ.setdefault("PROJ_LIB", str(proj_dir))
            os.environ.setdefault("PROJ_DATA", str(proj_dir))
        os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
        os.environ.setdefault("PROJ_IGNORE_BUILD_INFO", "YES")
    except Exception:
        pass


configure_gdal_proj()


def _find_band_file(safe_dir: Path, suffix: str) -> Path | None:
    """在 SAFE 目录中递归查找匹配后缀的 .jp2 文件。"""
    matches = sorted(
        p
        for p in safe_dir.rglob("*.jp2")
        if p.name.lower().endswith(suffix)
    )
    return matches[0] if matches else None


def _find_all_bands(safe_dir: Path) -> dict[str, Path]:
    """查找 SAFE 中所有需要的波段文件。

    Returns:
        语义波段名 → .jp2 文件路径的映射。
    """
    found: dict[str, Path] = {}
    for suffix, semantic_name in SAFE_BAND_MAP.items():
        path = _find_band_file(safe_dir, suffix)
        if path:
            found[semantic_name] = path

    required = {"blue", "green", "red", "rededge", "nir"}
    missing = required - set(found)
    if missing:
        raise FileNotFoundError(
            f"SAFE 目录 {safe_dir.name} 缺少必需波段：{', '.join(sorted(missing))}"
        )

    return found


def _reference_grid(band_files: dict[str, Path]) -> tuple[Any, Any, int, int]:
    """以 red (B04) 的 10m 网格为参考。"""
    with rasterio.open(band_files["red"]) as src:
        return src.crs, src.transform, src.width, src.height


def _read_10m(path: Path, ref_crs: Any, ref_transform: Any, ref_width: int, ref_height: int) -> np.ndarray:
    """读取 10m 波段，必要时重投影到参考网格。"""
    with rasterio.open(path) as src:
        if src.crs == ref_crs and src.transform == ref_transform and src.width == ref_width and src.height == ref_height:
            return src.read(1).astype("float32")
        dst = np.full((ref_height, ref_width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
        return dst


def _read_20m(path: Path, ref_crs: Any, ref_transform: Any, ref_width: int, ref_height: int) -> np.ndarray:
    """读取 20m 波段，重采样到 10m 参考网格。"""
    with rasterio.open(path) as src:
        dst = np.full((ref_height, ref_width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
        return dst


def _write_band(
    output_path: Path,
    data: np.ndarray,
    crs: Any,
    transform: Any,
    description: str,
    scale: float = 1.0,
) -> None:
    """写出单波段 GeoTIFF。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": np.nan,
        "compress": "deflate",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        scaled = (data / scale).astype("float32")
        dst.write(scaled, 1)
        dst.set_band_description(1, description)


def extract_scene(
    safe_dir: Path,
    output_dir: Path,
    reflectance_scale: float = 10000.0,
    overwrite: bool = False,
) -> dict[str, str]:
    """从单个 SAFE 目录提取所有波段为 TIFF。

    Args:
        safe_dir: .SAFE 目录路径。
        output_dir: 输出的单波段 TIFF 目录（每个场景一个子目录）。
        reflectance_scale: 反射率缩放系数。Sentinel-2 L2A 默认 10000。
        overwrite: 是否覆盖已有输出。

    Returns:
        语义波段名 → 输出 TIFF 路径的映射。
    """
    scene_name = safe_dir.name.replace(".SAFE", "")
    scene_out = output_dir / scene_name

    # 检查是否已提取
    if not overwrite and scene_out.exists():
        existing = list(scene_out.glob("*.tif"))
        if existing:
            return {p.stem: str(p) for p in existing}

    band_files = _find_all_bands(safe_dir)
    ref_crs, ref_transform, ref_width, ref_height = _reference_grid(band_files)

    assets: dict[str, str] = {}

    for semantic_name, jp2_path in band_files.items():
        if semantic_name in BANDS_10M:
            data = _read_10m(jp2_path, ref_crs, ref_transform, ref_width, ref_height)
        elif semantic_name in BANDS_20M:
            data = _read_20m(jp2_path, ref_crs, ref_transform, ref_width, ref_height)
        else:
            # swir22 这种可选波段也是 20m
            data = _read_20m(jp2_path, ref_crs, ref_transform, ref_width, ref_height)

        out_path = scene_out / f"{semantic_name}.tif"
        _write_band(out_path, data, ref_crs, ref_transform, semantic_name, reflectance_scale)
        assets[semantic_name] = str(out_path)

    return assets


def extract_from_manifest(
    manifest_path: Path,
    source_base_dir: Path,
    output_base_dir: Path,
    reflectance_scale: float = 10000.0,
    overwrite: bool = False,
    timepoints: list[str] | None = None,
) -> dict[str, Any]:
    """根据 manifest 提取所有已下载场景的波段。

    manifest 中的每个场景需要有 `_local_safe_path` 字段（由 download.py 写入）。

    Returns:
        更新后的 manifest，assets 指向提取后的本地 TIFF 路径。
    """
    with open(manifest_path, encoding="utf-8-sig") as f:
        manifest = json.load(f)

    tp_labels = set(timepoints) if timepoints else None
    total_scenes = 0
    success = 0

    for tp in manifest.get("timepoints", []):
        if tp_labels and tp["label"] not in tp_labels:
            continue

        for scene in tp.get("scenes", []):
            total_scenes += 1

            safe_path = scene.get("_local_safe_path")
            if not safe_path:
                print(
                    f"  警告：场景 {scene.get('id', '?')} 缺少 _local_safe_path，"
                    f"请先运行 download.py",
                    flush=True,
                )
                continue

            safe_dir = Path(safe_path)
            if not safe_dir.exists():
                print(f"  警告：SAFE 目录不存在：{safe_dir}", flush=True)
                continue

            try:
                assets = extract_scene(
                    safe_dir, output_base_dir, reflectance_scale, overwrite
                )
                # 更新 assets 为本地路径（兼容 build_features.py 的 SEMANTIC_ASSETS 映射）
                scene["assets"] = {
                    name: assets.get(name, scene["assets"].get(name, ""))
                    for name in [
                        "blue", "green", "red", "rededge",
                        "nir", "swir", "swir22", "scl",
                    ]
                }
                scene["_extracted"] = True
                success += 1
            except Exception as exc:
                print(
                    f"  错误：{safe_dir.name} — {exc}",
                    flush=True,
                )
                scene["_extract_error"] = str(exc)[:500]
                scene["_extracted"] = False

    print(f"\n提取完成：{success}/{total_scenes} 个场景成功", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Copernicus SAFE 目录提取波段为单波段 GeoTIFF。"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="download.py 更新后的 manifest JSON（含 _local_safe_path）。",
    )
    parser.add_argument(
        "--safe-dir",
        type=Path,
        default=None,
        help="SAFE 产品所在目录（manifest 中已有 _local_safe_path 时可选）。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/source/runtime/copernicus_extracted"),
        help="提取后的单波段 TIFF 输出目录。",
    )
    parser.add_argument(
        "--reflectance-scale",
        type=float,
        default=10000.0,
        help="反射率缩放系数，Sentinel-2 L2A 默认 10000。",
    )
    parser.add_argument(
        "--timepoints",
        nargs="*",
        default=None,
        help="可选，只提取指定时相标签对应的场景。",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重新提取已存在的输出。",
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=None,
        help="输出更新后的 manifest（含本地 TIFF 路径）。默认覆盖原文件。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest 不存在：{args.manifest}")

    updated = extract_from_manifest(
        manifest_path=args.manifest,
        source_base_dir=args.safe_dir or Path("."),
        output_base_dir=args.output_dir,
        reflectance_scale=args.reflectance_scale,
        overwrite=args.force,
        timepoints=args.timepoints,
    )

    output_path = args.output_manifest or args.manifest
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(updated, f, indent=2, ensure_ascii=False)

    print(f"更新后的 manifest 已保存到 {output_path}")


if __name__ == "__main__":
    main()
