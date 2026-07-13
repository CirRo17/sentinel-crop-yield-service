"""从 Copernicus SAFE 目录直接构建标准特征栈。

读取 download.py 输出的 manifest，直接从 .SAFE 目录中的 .jp2 波段文件
构建多时相特征栈。一步完成：波段读取 → 20m 重采样 → 时相中值合成 → 光谱指数计算。

输出格式与 image_core/feature_schema.py 兼容，可直接供下游 pipeline 消费。
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import bounds as geometry_bounds
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from rasterio.warp import reproject, transform_geom

from configs.paths import ProjectPaths
from data_sources.copernicus.config import SAFE_BAND_MAP
from image_core.spectral import evi, nbr, ndmi, ndre, ndvi, ndwi

# ---------------------------------------------------------------------------
# 语义波段与输出顺序
# ---------------------------------------------------------------------------
REQUIRED_BANDS = ("blue", "green", "red", "rededge", "nir", "swir")
INDEX_BANDS = ("ndvi", "ndwi", "evi", "ndre", "ndmi", "nbr")

BANDS_10M = {"blue", "green", "red", "nir"}
BANDS_20M = {"rededge", "swir", "swir22", "scl"}

# AOI 覆盖率阈值（低于此值警告）
DEFAULT_COVERAGE_THRESHOLD = 0.90

# ---------------------------------------------------------------------------
# GDAL / PROJ 配置
# ---------------------------------------------------------------------------


def configure_gdal_proj() -> None:
    try:
        proj_dir = Path(rasterio.__file__).resolve().parent / "proj_data"
        if (proj_dir / "proj.db").exists():
            os.environ.setdefault("PROJ_LIB", str(proj_dir))
            os.environ.setdefault("PROJ_DATA", str(proj_dir))
        os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
        os.environ.setdefault("PROJ_IGNORE_BUILD_INFO", "YES")
    except Exception:
        pass


configure_gdal_proj()

# ---------------------------------------------------------------------------
# SAFE .jp2 查找
# ---------------------------------------------------------------------------


def _find_band_jp2(safe_dir: Path, suffix: str) -> Path | None:
    """在 SAFE 目录中递归查找匹配后缀的 .jp2 文件。"""
    matches = sorted(
        p for p in safe_dir.rglob("*.jp2") if p.name.lower().endswith(suffix)
    )
    return matches[0] if matches else None


def _find_all_bands(safe_dir: Path) -> dict[str, Path]:
    """查找 SAFE 中所有需要的波段 .jp2 文件。"""
    found: dict[str, Path] = {}
    for suffix, semantic_name in SAFE_BAND_MAP.items():
        path = _find_band_jp2(safe_dir, suffix)
        if path:
            found[semantic_name] = path

    missing = {"blue", "green", "red", "rededge", "nir"} - set(found)
    if missing:
        raise FileNotFoundError(
            f"SAFE 目录 {safe_dir.name} 缺少必需波段：{', '.join(sorted(missing))}"
        )
    return found


# ---------------------------------------------------------------------------
# 波段读取与重采样
# ---------------------------------------------------------------------------


def _reference_grid(band_files: dict[str, Path]) -> tuple[Any, Any, int, int]:
    """以 red (B04) 的 10m 网格为参考。"""
    with rasterio.open(band_files["red"]) as src:
        return src.crs, src.transform, src.width, src.height


def _reference_grid_from_aoi(
    geometry: dict[str, Any],
    src_crs: str,
    sample_band: Path,
    resolution: float = 10.0,
) -> tuple[Any, Any, int, int, dict[str, Any]]:
    """Build a 10 m grid from the AOI bounds in the Sentinel-2 CRS."""
    with rasterio.open(sample_band) as src:
        dst_crs = src.crs

    geometry_in_dst = transform_geom(src_crs, dst_crs, geometry)
    left, bottom, right, top = geometry_bounds(geometry_in_dst)
    left = math.floor(left / resolution) * resolution
    bottom = math.floor(bottom / resolution) * resolution
    right = math.ceil(right / resolution) * resolution
    top = math.ceil(top / resolution) * resolution
    width = max(1, int(round((right - left) / resolution)))
    height = max(1, int(round((top - bottom) / resolution)))
    return dst_crs, from_origin(left, top, resolution, resolution), width, height, geometry_in_dst


def _read_band_to_grid(
    jp2_path: Path,
    ref_crs: Any,
    ref_transform: Any,
    ref_width: int,
    ref_height: int,
    is_20m: bool,
) -> np.ndarray:
    """Read one .jp2 band onto the target AOI grid."""
    dst = np.full((ref_height, ref_width), np.nan, dtype="float32")
    with rasterio.open(jp2_path) as src:
        resampling = Resampling.bilinear if is_20m else Resampling.nearest
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            src_nodata=0,
            dst_nodata=np.nan,
            resampling=resampling,
        )
    return dst


# ---------------------------------------------------------------------------
# 时相中值合成
# ---------------------------------------------------------------------------


def _composite_scenes(
    safe_dirs: list[Path],
    band_name: str,
    suffix: str,
    ref_crs: Any,
    ref_transform: Any,
    ref_width: int,
    ref_height: int,
    reflectance_scale: float,
) -> np.ndarray:
    """对同一时相的多个场景逐块做中值合成（避免整景加载导致 OOM）。"""
    # 找到所有有效的 jp2 路径
    jp2_paths: list[Path] = []
    for safe_dir in safe_dirs:
        jp2_path = _find_band_jp2(safe_dir, suffix)
        if jp2_path is not None:
            jp2_paths.append(jp2_path)

    if not jp2_paths:
        raise FileNotFoundError(f"没有找到 {band_name} 的有效数据")

    is_20m = band_name in BANDS_20M
    stack = []
    for jp2_path in jp2_paths:
        data = _read_band_to_grid(
            jp2_path, ref_crs, ref_transform, ref_width, ref_height, is_20m
        )
        data[~np.isfinite(data) | (data == 0)] = np.nan
        stack.append(data)
    return (np.nanmedian(np.stack(stack, axis=0), axis=0) / reflectance_scale).astype("float32")


def _read_band_block(
    jp2_path: Path,
    ref_crs: Any,
    ref_transform: Any,
    ref_width: int,
    ref_height: int,
    is_20m: bool,
    row_start: int,
    row_end: int,
) -> np.ndarray:
    """读取单个 .jp2 波段的指定行范围，必要时重采样到参考网格。"""
    block_h = row_end - row_start
    with rasterio.open(jp2_path) as src:
        # 同网格同分辨率 → 直接窗口读取，最快
        if not is_20m and src.crs == ref_crs and src.transform == ref_transform:
            window = rasterio.windows.Window(0, row_start, ref_width, block_h)
            return src.read(1, window=window).astype("float32")

        # 需要重采样：只对目标块对应的源区域做 reproject
        # 计算源影像中对应的行范围
        # ref 是 10m，src 是 20m → scale = 10/20 = 0.5
        scale = ref_transform.a / src.res[0]
        src_row_start = max(0, int(row_start * scale) - 2)
        src_row_end = min(src.height, int(row_end * scale) + 2)
        if src_row_start >= src_row_end:
            return np.full((block_h, ref_width), np.nan, dtype="float32")

        # 读源块
        src_window = rasterio.windows.Window(
            0, src_row_start, src.width, src_row_end - src_row_start
        )
        src_data = src.read(1, window=src_window).astype("float32")

        # 只 reproject 到目标块
        dst_block = np.full((block_h, ref_width), np.nan, dtype="float32")
        src_transform_block = src.window_transform(src_window)

        reproject(
            source=src_data,
            destination=dst_block,
            src_transform=src_transform_block,
            src_crs=src.crs,
            dst_transform=rasterio.windows.transform(
                rasterio.windows.Window(0, row_start, ref_width, block_h),
                ref_transform,
            ),
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
        return dst_block


# ---------------------------------------------------------------------------
# AOI 覆盖检查
# ---------------------------------------------------------------------------


def _load_aoi_geometry(path: Path) -> dict[str, Any]:
    """加载 AOI 几何（支持 GeoJSON 和 Shapefile）。"""
    if not path.exists():
        raise FileNotFoundError(f"AOI 文件不存在：{path}")

    suffix = path.suffix.lower()
    if suffix in {".json", ".geojson"}:
        with open(path, encoding="utf-8-sig") as f:
            geojson = json.load(f)
        if geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
            if not features:
                raise ValueError(f"{path} 中没有要素。")
            return features[0]["geometry"]
        if geojson.get("type") == "Feature":
            return geojson["geometry"]
        return geojson

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ImportError("读取 Shapefile AOI 需要 geopandas。") from exc
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"{path} 中没有要素。")
    if gdf.crs and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf.geometry.unary_union.__geo_interface__


def _build_aoi_mask(
    geometry: dict[str, Any],
    src_crs: str,
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
) -> np.ndarray:
    """构建 AOI 掩膜（True = 在研究区内）。"""
    if dst_crs is not None and src_crs:
        geometry = transform_geom(src_crs, dst_crs, geometry)
    return geometry_mask(
        [geometry], out_shape=(height, width), transform=transform, invert=True
    )


def _check_coverage(
    composite: np.ndarray,
    aoi_mask: np.ndarray,
    label: str,
    threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    """检查合成影像在 AOI 内的有效像素覆盖率。

    Args:
        composite: 合成后的单波段数组（已含 NaN 表示无数据）。
        aoi_mask: AOI 掩膜（True = 在研究区内）。
        label: 时相标签（仅用于输出）。
        threshold: 覆盖率阈值，低于此值发出警告。

    Returns:
        覆盖率统计 dict。
    """
    aoi_pixels = int(np.sum(aoi_mask))
    if aoi_pixels == 0:
        return {"label": label, "aoi_pixels": 0, "valid_pixels": 0, "coverage": 0.0, "warning": "AOI 为空"}

    valid = np.isfinite(composite) & aoi_mask
    valid_pixels = int(np.sum(valid))
    coverage = valid_pixels / aoi_pixels

    result: dict[str, Any] = {
        "label": label,
        "aoi_pixels": aoi_pixels,
        "valid_pixels": valid_pixels,
        "coverage": round(coverage, 4),
    }

    if coverage < threshold:
        pct = coverage * 100
        result["warning"] = (
            f"  *** 覆盖率不足: {label} AOI 内仅 {pct:.1f}% 像素有效 "
            f"({valid_pixels}/{aoi_pixels})，阈值 {threshold*100:.0f}%"
        )
        print(result["warning"], flush=True)
    else:
        print(f"  覆盖率: {coverage*100:.1f}% OK", flush=True)

    return result


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def build_features_from_manifest(
    manifest_path: Path,
    output_path: Path,
    metadata_path: Path | None = None,
    reflectance_scale: float = 10000.0,
    timepoints: list[str] | None = None,
    aoi_geometry_path: Path | None = None,
    coverage_threshold: float = DEFAULT_COVERAGE_THRESHOLD,
) -> dict[str, Any]:
    """从 manifest 构建多时相特征栈。

    Args:
        manifest_path: download.py 输出的 manifest JSON。
        output_path: 输出的多波段特征栈 GeoTIFF。
        metadata_path: 元数据 JSON 输出路径。默认与 output_path 同名 .json。
        reflectance_scale: S2 L2A 反射率缩放系数，默认 10000。
        timepoints: 可选，只处理指定时相标签。
        aoi_geometry_path: 可选，AOI 矢量文件用于覆盖检查。
        coverage_threshold: AOI 覆盖率阈值，默认 90%。

    Returns:
        特征栈元数据 dict。
    """
    with open(manifest_path, encoding="utf-8-sig") as f:
        manifest = json.load(f)

    # 加载 AOI（用于覆盖检查）
    aoi_geometry = None
    aoi_src_crs = "EPSG:4326"
    if aoi_geometry_path and aoi_geometry_path.exists():
        aoi_geometry = _load_aoi_geometry(aoi_geometry_path)

    tp_labels = set(timepoints) if timepoints else None
    all_band_arrays: list[np.ndarray] = []
    all_band_names: list[str] = []
    timepoint_meta: list[dict[str, Any]] = []
    ref_crs: Any = None
    ref_transform: Any = None
    ref_width: int = 0
    ref_height: int = 0

    for tp in manifest.get("timepoints", []):
        label = tp["label"]
        if tp_labels and label not in tp_labels:
            continue

        # 收集该时相下所有已下载场景的 SAFE 目录
        safe_dirs: list[Path] = []
        for scene in tp.get("scenes", []):
            safe_path = scene.get("_local_safe_path")
            if safe_path:
                safe_dir = Path(safe_path)
                if safe_dir.exists():
                    safe_dirs.append(safe_dir)

        if not safe_dirs:
            print(f"  警告：时相 {label} 没有已下载的场景，跳过。", flush=True)
            continue

        print(f"\n[{label}] {len(safe_dirs)} 个场景，合成中...", flush=True)

        # 用 AOI 外接范围确定参考网格，避免跨 tile AOI 被第一个 SAFE 的 tile 裁掉。
        first_bands = _find_all_bands(safe_dirs[0])
        if ref_crs is None:
            if aoi_geometry is not None:
                ref_crs, ref_transform, ref_width, ref_height, _ = _reference_grid_from_aoi(
                    aoi_geometry,
                    aoi_src_crs,
                    first_bands["red"],
                )
            else:
                ref_crs, ref_transform, ref_width, ref_height = _reference_grid(first_bands)

        # 对每个语义波段做中值合成
        semantic_arrays: dict[str, np.ndarray] = {}
        for band_name in REQUIRED_BANDS:
            # 找到对应的 jp2 后缀
            suffix = None
            for sfx, name in SAFE_BAND_MAP.items():
                if name == band_name:
                    suffix = sfx
                    break
            if suffix is None:
                raise ValueError(f"未找到 {band_name} 的 SAFE 波段映射")

            print(f"  {band_name}...", flush=True, end=" ")
            semantic_arrays[band_name] = _composite_scenes(
                safe_dirs, band_name, suffix,
                ref_crs, ref_transform, ref_width, ref_height,
                reflectance_scale,
            )
            print("OK", flush=True)

        # 覆盖检查（用 nir 波段代表，它是所有波段中数据最完整的）
        coverage_info = None
        if aoi_geometry is not None:
            aoi_mask = _build_aoi_mask(aoi_geometry, aoi_src_crs, ref_crs, ref_transform, ref_width, ref_height)
            coverage_info = _check_coverage(semantic_arrays["nir"], aoi_mask, label, coverage_threshold)

        # 计算光谱指数
        semantic_arrays["ndvi"] = ndvi(semantic_arrays["nir"], semantic_arrays["red"])
        semantic_arrays["ndwi"] = ndwi(semantic_arrays["green"], semantic_arrays["nir"])
        semantic_arrays["evi"] = evi(semantic_arrays["nir"], semantic_arrays["red"], semantic_arrays["blue"])
        semantic_arrays["ndre"] = ndre(semantic_arrays["nir"], semantic_arrays["rededge"])
        semantic_arrays["ndmi"] = ndmi(semantic_arrays["nir"], semantic_arrays["swir"])
        semantic_arrays["nbr"] = nbr(semantic_arrays["nir"], semantic_arrays["swir"])

        # 按固定顺序输出，用 t1/t2 前缀匹配模型期望
        slot_prefix = f"t{len(timepoint_meta) + 1}"  # t1, t2, ...
        output_order = [*REQUIRED_BANDS, *INDEX_BANDS]
        for name in output_order:
            all_band_names.append(f"{slot_prefix}_{name}")
            all_band_arrays.append(semantic_arrays[name])

        tp_entry: dict[str, Any] = {
            "label": label,
            "slot": slot_prefix,
            "scene_count": len(safe_dirs),
            "composite_method": "median",
            "output_bands": [f"{slot_prefix}_{name}" for name in output_order],
        }
        if coverage_info:
            tp_entry["coverage"] = coverage_info
        timepoint_meta.append(tp_entry)

    if not all_band_arrays:
        raise RuntimeError("没有生成任何波段数据，请先运行 download.py。")

    # 写出特征栈 GeoTIFF
    output_path.parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "height": ref_height,
        "width": ref_width,
        "count": len(all_band_arrays),
        "dtype": "float32",
        "crs": ref_crs,
        "transform": ref_transform,
        "nodata": np.nan,
        "BIGTIFF": "YES",
    }

    with rasterio.open(output_path, "w", **profile) as dst:
        for idx, (name, data) in enumerate(zip(all_band_names, all_band_arrays), start=1):
            dst.write(data, idx)
            dst.set_band_description(idx, name)

    print(f"\n已保存特征栈：{output_path}  ({len(all_band_arrays)} 个波段)", flush=True)

    # 写出元数据
    metadata = {
        "source": "copernicus_safe",
        "manifest": str(manifest_path),
        "aoi": str(aoi_geometry_path) if aoi_geometry_path else None,
        "timepoints": timepoint_meta,
        "output": str(output_path),
        "band_count": len(all_band_names),
        "band_names": all_band_names,
        "reflectance_scale": reflectance_scale,
    }
    if metadata_path is None:
        metadata_path = output_path.with_suffix(".json")
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"已保存元数据：{metadata_path}", flush=True)

    return metadata


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Copernicus SAFE 目录直接构建标准特征栈。"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="download.py 输出的 manifest JSON（含 _local_safe_path）。",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/default.yaml"),
        help="项目配置文件，用于推导输出路径。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="输出特征栈路径。未指定时使用 ProjectPaths 推导。",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="输出元数据 JSON 路径。默认与输出 tif 同名 .json。",
    )
    parser.add_argument(
        "--reflectance-scale",
        type=float,
        default=10000.0,
        help="Sentinel-2 L2A 反射率缩放系数，默认 10000。",
    )
    parser.add_argument(
        "--timepoints",
        nargs="*",
        default=None,
        help="可选，只处理指定时相标签（如 2026-05）。",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=None,
        help="AOI 矢量文件路径。用于检查覆盖完整性。未指定时从配置 project.geometry 读取。",
    )
    parser.add_argument(
        "--coverage-threshold",
        type=float,
        default=DEFAULT_COVERAGE_THRESHOLD,
        help=f"AOI 覆盖率阈值，默认 {DEFAULT_COVERAGE_THRESHOLD*100:.0f}%%。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.manifest.exists():
        raise FileNotFoundError(f"Manifest 不存在：{args.manifest}")

    paths = ProjectPaths(args.config)
    output = args.output or paths.feature_stack
    metadata_path = args.metadata or paths.feature_stack_metadata

    # 解析 AOI 路径
    geometry_path = args.geometry
    if geometry_path is None:
        geo_str = paths.config.get("project", {}).get("geometry", "")
        if geo_str:
            geo_candidate = Path(geo_str)
            if geo_candidate.exists():
                geometry_path = geo_candidate
    if geometry_path and not geometry_path.exists():
        print(f"  警告：AOI 文件不存在，跳过覆盖检查：{geometry_path}", flush=True)
        geometry_path = None

    build_features_from_manifest(
        manifest_path=args.manifest,
        output_path=output,
        metadata_path=metadata_path,
        reflectance_scale=args.reflectance_scale,
        timepoints=args.timepoints,
        aoi_geometry_path=geometry_path,
        coverage_threshold=args.coverage_threshold,
    )


if __name__ == "__main__":
    main()
