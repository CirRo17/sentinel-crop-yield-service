"""步骤 04：用水稻/玉米/小麦/油菜专项图构建训练样本。

输入可以是单个 GeoTIFF，也可以是包含多张 GeoTIFF 的目录。脚本会把标签图
按最近邻重采样到特征栈网格，然后抽取像素级训练样本，输出给 05_train_rf.py。

类别 0（Others）来自所有专项图之外的区域。
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
from rasterio.features import geometry_mask
from rasterio.transform import array_bounds
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, transform_geom

from crop_classifier_core.config import TARGET_LABELS
from crop_classifier_core.feature_schema import duplicate_names


DEFAULT_FEATURE_STACK = Path("data/exported/feature_stack_2025_07_s2_test.tif")
DEFAULT_FEATURE_METADATA = Path("data/exported/feature_stack_2025_07_s2_test_metadata.json")
DEFAULT_LABEL_ROOT = Path("data/input/lables")
DEFAULT_OUTPUT = Path("data/exported/pixel_training_data.npz")
DEFAULT_REPORT = Path("data/exported/pixel_training_data_report.json")


def configure_gdal_proj() -> None:
    try:
        import rasterio as _rasterio

        proj_dir = Path(_rasterio.__file__).resolve().parent / "proj_data"
        if (proj_dir / "proj.db").exists():
            os.environ.setdefault("PROJ_LIB", str(proj_dir))
            os.environ.setdefault("PROJ_DATA", str(proj_dir))
        os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
        os.environ.setdefault("PROJ_IGNORE_BUILD_INFO", "YES")
    except Exception:
        pass


configure_gdal_proj()


def parse_codes(value: str | None) -> set[int]:
    if not value:
        return set()
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从四类作物专项图构建像素级训练样本。")
    parser.add_argument("--feature-stack", type=Path, default=DEFAULT_FEATURE_STACK, help="03b 输出的特征栈。")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_FEATURE_METADATA, help="特征栈元数据 JSON。")
    parser.add_argument("--rice-map", type=Path, default=DEFAULT_LABEL_ROOT / "rice", help="水稻专项图文件或目录。")
    parser.add_argument("--maize-map", type=Path, default=DEFAULT_LABEL_ROOT / "maize", help="玉米专项图文件或目录。")
    parser.add_argument("--wheat-map", type=Path, default=DEFAULT_LABEL_ROOT / "wheat", help="小麦专项图文件或目录。")
    parser.add_argument("--rapeseed-map", type=Path, default=DEFAULT_LABEL_ROOT / "rapeseed", help="油菜专项图文件或目录。")
    parser.add_argument("--sample-region", type=Path, default=None, help="样本采集区域 GeoJSON/矢量文件。默认使用元数据中的 aoi。")
    parser.add_argument("--sample-region-crs", default="EPSG:4326", help="样本采集区域缺少 CRS 时使用的坐标系。")
    parser.add_argument("--positive-codes", default=None, help="专项图有效编码。默认所有大于 0 且非 NoData 的像元有效。")
    parser.add_argument("--max-per-class", type=int, default=5000, help="每类最多抽样像元数。")
    parser.add_argument("--min-per-class", type=int, default=50, help="每类最低样本数。")
    parser.add_argument("--erode-pixels", type=int, default=1, help="腐蚀掩膜像元数，用于剔除边界混合像元。")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def load_region_geometry(path: Path, default_crs: str) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(f"缺少样本采集区域文件：{path}")

    if path.suffix.lower() in {".json", ".geojson"}:
        with open(path, encoding="utf-8-sig") as f:
            geojson = json.load(f)
        crs = default_crs
        if isinstance(geojson.get("crs"), dict):
            crs = str(geojson["crs"].get("properties", {}).get("name") or default_crs)
        if geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
            if not features:
                raise ValueError(f"{path} 中没有要素。")
            if len(features) == 1:
                return features[0]["geometry"], crs
            return {"type": "GeometryCollection", "geometries": [item["geometry"] for item in features]}, crs
        if geojson.get("type") == "Feature":
            return geojson["geometry"], crs
        return geojson, crs

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ValueError("读取非 GeoJSON 样本采集区域需要安装 geopandas。") from exc

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"{path} 中没有要素。")
    geometry = gdf.geometry.unary_union.__geo_interface__
    crs = str(gdf.crs) if gdf.crs is not None else default_crs
    return geometry, crs


def build_region_mask(
    region_path: Path,
    default_region_crs: str,
    ref: rasterio.DatasetReader,
) -> tuple[np.ndarray, dict[str, Any]]:
    geometry, src_crs = load_region_geometry(region_path, default_region_crs)
    geometry_for_mask = geometry
    if ref.crs is not None and src_crs:
        geometry_for_mask = transform_geom(src_crs, ref.crs, geometry)
    mask = geometry_mask([geometry_for_mask], out_shape=(ref.height, ref.width), transform=ref.transform, invert=True)
    count = int(np.count_nonzero(mask))
    if count == 0:
        raise ValueError(f"样本采集区域 {region_path} 与特征栈范围没有重叠像元。")
    return mask, {
        "path": str(region_path),
        "source_crs": src_crs,
        "target_crs": str(ref.crs) if ref.crs else None,
        "valid_pixel_count": count,
    }


def feature_names(src: rasterio.DatasetReader, metadata: dict[str, Any] | None) -> list[str]:
    if metadata and metadata.get("band_names"):
        names = [str(name) for name in metadata["band_names"]]
    else:
        names = [src.descriptions[index - 1] or f"band_{index}" for index in range(1, src.count + 1)]
    if len(names) != src.count:
        raise ValueError(f"特征名数量 {len(names)} 与特征栈波段数 {src.count} 不一致。")
    duplicates = duplicate_names(names)
    if duplicates:
        raise ValueError(f"特征栈 band 名重复：{', '.join(duplicates)}")
    return names


def label_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        files = sorted([*path.rglob("*.tif"), *path.rglob("*.tiff")])
        if files:
            return files
    raise FileNotFoundError(f"没有找到标签栅格：{path}")


def intersects_reference(label_path: Path, ref: rasterio.DatasetReader) -> bool:
    ref_bounds = array_bounds(ref.height, ref.width, ref.transform)
    with rasterio.open(label_path) as src:
        if src.crs is None or ref.crs is None:
            return True
        ref_in_src = transform_bounds(ref.crs, src.crs, *ref_bounds, densify_pts=21)
        left = max(ref_in_src[0], src.bounds.left)
        bottom = max(ref_in_src[1], src.bounds.bottom)
        right = min(ref_in_src[2], src.bounds.right)
        top = min(ref_in_src[3], src.bounds.top)
        return left < right and bottom < top


def read_label_mosaic(path: Path, ref: rasterio.DatasetReader) -> tuple[np.ndarray, list[str]]:
    output = np.zeros((ref.height, ref.width), dtype="int64")
    used_files: list[str] = []
    for item in label_files(path):
        if not intersects_reference(item, ref):
            continue
        with rasterio.open(item) as src:
            with WarpedVRT(
                src,
                crs=ref.crs,
                transform=ref.transform,
                width=ref.width,
                height=ref.height,
                resampling=Resampling.nearest,
            ) as vrt:
                data = vrt.read(1, masked=False)
                nodata = src.nodata
        valid = np.isfinite(data)
        if nodata is not None:
            valid &= data != nodata
        output[valid] = data[valid].astype("int64")
        used_files.append(str(item))
    if not used_files:
        raise ValueError(f"{path} 中没有与特征栈范围相交的标签栅格。")
    return output, used_files


def positive_mask(data: np.ndarray, positive_codes: set[int]) -> np.ndarray:
    if positive_codes:
        return np.isin(data.astype("int64"), list(positive_codes))
    return np.isfinite(data) & (data > 0)


def erode_mask(mask: np.ndarray, pixels: int) -> np.ndarray:
    result = mask.astype(bool)
    for _ in range(max(0, pixels)):
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        neighbors = [
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]
        result = np.logical_and.reduce(neighbors)
    return result


def build_label_masks(
    args: argparse.Namespace,
    ref: rasterio.DatasetReader,
    region_mask: np.ndarray,
) -> tuple[dict[int, np.ndarray], dict[str, Any]]:
    rice_map, rice_files = read_label_mosaic(args.rice_map, ref)
    maize_map, maize_files = read_label_mosaic(args.maize_map, ref)
    wheat_map, wheat_files = read_label_mosaic(args.wheat_map, ref)
    rapeseed_map, rapeseed_files = read_label_mosaic(args.rapeseed_map, ref)

    special_positive = parse_codes(args.positive_codes)

    rice = positive_mask(rice_map, special_positive) & region_mask
    wheat = positive_mask(wheat_map, special_positive) & region_mask
    maize = positive_mask(maize_map, special_positive) & region_mask
    rapeseed = positive_mask(rapeseed_map, special_positive) & region_mask
    main_crop = rice | wheat | maize | rapeseed
    noncrop = region_mask & ~main_crop

    masks = {
        0: erode_mask(noncrop, args.erode_pixels),
        1: erode_mask(rice, args.erode_pixels),
        2: erode_mask(wheat, args.erode_pixels),
        3: erode_mask(maize, args.erode_pixels),
        4: erode_mask(rapeseed, args.erode_pixels),
    }

    # ---- 跨类互斥清洗 -------------------------------------------------------
    # 同一个像素可能被多个作物类（1-4）的掩膜同时命中（例如轮作地块的专项图
    # 同时覆盖水稻和小麦），将这些歧义像素从所有作物掩膜中移除，使其自然地
    # 归入类别 0（Others / 不确定）。
    crop_overlap = np.zeros_like(masks[0], dtype="int8")
    for code in [1, 2, 3, 4]:
        crop_overlap += masks[code].astype("int8")
    ambiguous = crop_overlap > 1
    if np.any(ambiguous):
        for code in [1, 2, 3, 4]:
            masks[code] = masks[code] & ~ambiguous
        print(
            f"警告：{int(np.count_nonzero(ambiguous))} 个像素被多个作物类同时命中，"
            f"已从所有作物掩膜中移除，归入类别 0（Others）。"
        )

    sources = {
        "rice_files": rice_files,
        "maize_files": maize_files,
        "wheat_files": wheat_files,
        "rapeseed_files": rapeseed_files,
    }
    return masks, sources


def sample_class(
    flat_features: np.ndarray,
    valid_pixels: np.ndarray,
    class_mask: np.ndarray,
    label: int,
    max_per_class: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.flatnonzero(valid_pixels & class_mask.reshape(-1))
    if candidates.size == 0:
        return np.empty((0, flat_features.shape[1]), dtype="float32"), np.empty((0,), dtype="uint8")
    if candidates.size > max_per_class:
        candidates = rng.choice(candidates, size=max_per_class, replace=False)
    X = flat_features[candidates].astype("float32")
    y = np.full(candidates.size, label, dtype="uint8")
    return X, y


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.random_state)
    metadata = load_json(args.metadata)
    sample_region = args.sample_region
    if sample_region is None and metadata and metadata.get("aoi"):
        sample_region = Path(str(metadata["aoi"]))
    if sample_region is None:
        raise ValueError("必须通过 --sample-region 或特征栈元数据 aoi 指定样本采集区域。")

    if not args.feature_stack.exists():
        raise FileNotFoundError(f"缺少特征栈：{args.feature_stack}")

    with rasterio.open(args.feature_stack) as src:
        names = feature_names(src, metadata)
        stack = src.read(masked=False).astype("float32")
        flat = np.moveaxis(stack, 0, -1).reshape(-1, src.count)
        valid_pixels = np.all(np.isfinite(flat), axis=1)
        region_mask, region_info = build_region_mask(sample_region, args.sample_region_crs, src)
        masks, sources = build_label_masks(args, src, region_mask)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    class_rows: list[dict[str, Any]] = []
    for code in [0, 1, 2, 3, 4]:
        X_cls, y_cls = sample_class(flat, valid_pixels, masks[code], code, args.max_per_class, rng)
        if len(y_cls) < args.min_per_class:
            raise ValueError(
                f"类别 {code}（{TARGET_LABELS.get(code, code)}）样本数 {len(y_cls)} "
                f"低于最低要求 {args.min_per_class}。请检查标签编码或降低 --min-per-class。"
            )
        X_parts.append(X_cls)
        y_parts.append(y_cls)
        class_rows.append(
            {
                "class_code": code,
                "label": TARGET_LABELS.get(code, str(code)),
                "sample_count": int(len(y_cls)),
                "candidate_pixel_count_after_erode": int(np.count_nonzero(masks[code])),
            }
        )
        print(f"类别 {code}（{TARGET_LABELS.get(code, code)}）：{len(y_cls)} 个样本")

    X = np.concatenate(X_parts).astype("float32")
    y = np.concatenate(y_parts).astype("uint8")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        X=X,
        y=y,
        feature_names=np.array(names),
        source=np.array("rice/maize/wheat/rapeseed specialty maps"),
    )

    report = {
        "source": "rice specialty map + maize specialty map + wheat specialty map + rapeseed specialty map",
        "feature_stack": str(args.feature_stack),
        "metadata": str(args.metadata) if args.metadata.exists() else None,
        "output": str(args.output),
        "feature_names": names,
        "sample_count": int(len(y)),
        "sample_region": region_info,
        "class_counts": class_rows,
        "label_sources": sources,
        "parameters": {
            "rice_map": str(args.rice_map),
            "maize_map": str(args.maize_map),
            "wheat_map": str(args.wheat_map),
            "rapeseed_map": str(args.rapeseed_map),
            "sample_region": str(sample_region),
            "sample_region_crs": args.sample_region_crs,
            "positive_codes": args.positive_codes,
            "max_per_class": args.max_per_class,
            "min_per_class": args.min_per_class,
            "erode_pixels": args.erode_pixels,
            "random_state": args.random_state,
        },
    }
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"已保存训练样本：{args.output}")
    print(f"已保存样本报告：{args.report}")


if __name__ == "__main__":
    main()
