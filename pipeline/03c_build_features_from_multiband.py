"""步骤 03d：从本地多波段 GeoTIFF 构建标准特征栈。

这是 03b_build_features.py 的本地输入适配入口。脚本接收一张或多张
已经对齐到同一网格的多波段影像，把波段序号映射到 blue/green/red
等语义波段，计算光谱指数，并写出训练、离线预测和 API 共用的
t1_*/t2_* 标准特征栈。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom

from crop_classifier_core.spectral import evi, nbr, ndre, ndvi, ndwi


DEFAULT_OUTPUT = Path("data/exported/feature_stack_multiband.tif")
DEFAULT_METADATA = Path("data/exported/feature_stack_multiband_metadata.json")

REQUIRED_BANDS = ("blue", "green", "red", "rededge", "nir")
OPTIONAL_BANDS = ("swir",)
INDEX_BANDS = ("ndvi", "ndwi", "evi", "ndre", "nbr")


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


def parse_band_map(value: str) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise argparse.ArgumentTypeError(f"无效的波段映射项：{item!r}。格式应为 name=index。")
        name, index = item.split("=", 1)
        name = name.strip().lower()
        if name not in {*REQUIRED_BANDS, *OPTIONAL_BANDS}:
            valid = ", ".join([*REQUIRED_BANDS, *OPTIONAL_BANDS])
            raise argparse.ArgumentTypeError(f"未知语义波段 {name!r}。可用名称：{valid}。")
        try:
            band_index = int(index)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"波段序号必须是整数：{item!r}") from exc
        if band_index < 1:
            raise argparse.ArgumentTypeError(f"波段序号从 1 开始，且必须 >= 1：{item!r}")
        mapping[name] = band_index
    missing = [name for name in REQUIRED_BANDS if name not in mapping]
    if missing:
        raise argparse.ArgumentTypeError(f"波段映射缺少必需波段：{', '.join(missing)}")
    return mapping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从本地多波段 GeoTIFF 构建 t1/t2 标准特征栈。")
    parser.add_argument(
        "--input",
        dest="inputs",
        type=Path,
        action="append",
        required=True,
        help="本地多波段 GeoTIFF。多时相输入时可重复传入。",
    )
    parser.add_argument(
        "--band-map",
        type=parse_band_map,
        required=True,
        help="从 1 开始的语义波段映射，例如 blue=1,green=2,red=3,rededge=4,nir=5,swir=6。",
    )
    parser.add_argument(
        "--time-slots",
        nargs="*",
        default=None,
        help="可选的稳定时相槽前缀。默认按输入顺序使用 t1 t2 ...。",
    )
    parser.add_argument("--reflectance-scale", type=float, default=1.0, help="读取波段后除以该比例系数。")
    parser.add_argument("--zero-is-nodata", action="store_true", help="将 0 值视为缺失数据。")
    parser.add_argument(
        "--geometry",
        type=Path,
        default=None,
        help="可选 AOI 矢量文件，支持 GeoJSON/JSON；若安装了 geopandas，也支持 Shapefile 等格式。",
    )
    parser.add_argument(
        "--geometry-crs",
        default="EPSG:4326",
        help="当 AOI 文件没有 CRS 信息时使用的 CRS，默认 EPSG:4326。",
    )
    parser.add_argument(
        "--allow-missing-swir",
        action="store_true",
        help="允许输入缺少 swir。输出会省略 swir 和 nbr，因此模型也必须按该 9 特征 schema 训练。",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> list[str]:
    if args.reflectance_scale <= 0:
        raise ValueError("--reflectance-scale 必须 > 0。")
    if "swir" not in args.band_map and not args.allow_missing_swir:
        raise ValueError(
            "--band-map 未包含 swir。请添加 swir=<band_index>，或传入 --allow-missing-swir，"
            "并使用输出的 9 特征 schema 进行训练/预测。"
        )
    slots = args.time_slots or [f"t{index}" for index in range(1, len(args.inputs) + 1)]
    if len(slots) != len(args.inputs):
        raise ValueError("--time-slots 的数量必须与 --input 影像数量一致。")
    if len(set(slots)) != len(slots):
        raise ValueError("--time-slots 的值不能重复。")
    return [str(slot) for slot in slots]


def load_aoi_geometry(path: Path, default_crs: str) -> tuple[dict[str, Any], str]:
    if not path.exists():
        raise FileNotFoundError(f"缺少 AOI 矢量文件：{path}")

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
        raise ValueError("读取非 GeoJSON AOI 需要安装 geopandas。") from exc

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"{path} 中没有要素。")
    geometry = gdf.geometry.unary_union.__geo_interface__
    crs = str(gdf.crs) if gdf.crs is not None else default_crs
    return geometry, crs


def build_aoi_mask(
    geometry_path: Path,
    default_geometry_crs: str,
    dst_crs: Any,
    transform: Any,
    width: int,
    height: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    geometry, src_crs = load_aoi_geometry(geometry_path, default_geometry_crs)
    geometry_for_mask = geometry
    if dst_crs is not None and src_crs:
        geometry_for_mask = transform_geom(src_crs, dst_crs, geometry)
    mask = geometry_mask([geometry_for_mask], out_shape=(height, width), transform=transform, invert=True)
    return mask, {
        "path": str(geometry_path),
        "source_crs": src_crs,
        "target_crs": str(dst_crs) if dst_crs else None,
        "valid_pixel_count": int(np.count_nonzero(mask)),
    }


def read_semantic_bands(
    path: Path,
    band_map: dict[str, int],
    reflectance_scale: float,
    zero_is_nodata: bool,
    aoi_mask: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"缺少本地多波段输入：{path}")

    arrays: dict[str, np.ndarray] = {}
    with rasterio.open(path) as src:
        for name, band_index in band_map.items():
            if band_index > src.count:
                raise ValueError(f"{path} 只有 {src.count} 个波段，但 {name} 映射到了第 {band_index} 波段。")
            data = src.read(band_index, masked=False).astype("float32") / float(reflectance_scale)
            nodata = src.nodata
            invalid = ~np.isfinite(data)
            if nodata is not None:
                invalid |= data == float(nodata) / float(reflectance_scale)
            if zero_is_nodata:
                invalid |= data == 0
            if aoi_mask is not None:
                invalid |= ~aoi_mask
            arrays[name] = np.where(invalid, np.nan, data).astype("float32")

        metadata = {
            "path": str(path),
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "crs": str(src.crs) if src.crs else None,
            "transform": list(src.transform),
            "nodata": src.nodata,
            "descriptions": [desc for desc in src.descriptions],
        }
    return arrays, metadata


def add_indices(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    arrays["ndvi"] = ndvi(arrays["nir"], arrays["red"])
    arrays["ndwi"] = ndwi(arrays["green"], arrays["nir"])
    arrays["evi"] = evi(arrays["nir"], arrays["red"], arrays["blue"])
    arrays["ndre"] = ndre(arrays["nir"], arrays["rededge"])
    if "swir" in arrays:
        arrays["nbr"] = nbr(arrays["nir"], arrays["swir"])
    return arrays


def validate_same_grid(reference: dict[str, Any], candidate: dict[str, Any]) -> None:
    keys = ("width", "height", "crs", "transform")
    mismatched = [key for key in keys if candidate.get(key) != reference.get(key)]
    if mismatched:
        raise ValueError(
            "所有多波段输入必须预先对齐到同一网格。"
            f"{candidate['path']} 存在差异：{', '.join(mismatched)}"
        )


def output_profile(first_input: Path, band_count: int) -> dict[str, Any]:
    with rasterio.open(first_input) as src:
        profile = src.profile.copy()
    profile.update(
        count=band_count,
        dtype="float32",
        nodata=np.nan,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=min(256, profile["width"]),
        blockysize=min(256, profile["height"]),
    )
    return profile


def write_feature_stack(path: Path, arrays: list[np.ndarray], names: list[str], profile: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        for index, (name, data) in enumerate(zip(names, arrays), start=1):
            dst.write(data.astype("float32"), index)
            dst.set_band_description(index, name)


def main() -> None:
    args = parse_args()
    slots = validate_args(args)

    band_arrays: list[np.ndarray] = []
    band_names: list[str] = []
    input_metadata: list[dict[str, Any]] = []
    reference_grid: dict[str, Any] | None = None
    aoi_mask: np.ndarray | None = None
    aoi_metadata: dict[str, Any] | None = None

    if args.geometry is not None:
        with rasterio.open(args.inputs[0]) as src:
            aoi_mask, aoi_metadata = build_aoi_mask(
                args.geometry,
                args.geometry_crs,
                src.crs,
                src.transform,
                src.width,
                src.height,
            )

    for slot, input_path in zip(slots, args.inputs):
        semantic_arrays, metadata = read_semantic_bands(
            input_path,
            args.band_map,
            args.reflectance_scale,
            args.zero_is_nodata,
            aoi_mask,
        )
        if reference_grid is None:
            reference_grid = metadata
        else:
            validate_same_grid(reference_grid, metadata)

        feature_arrays = add_indices(semantic_arrays)
        output_order = [*REQUIRED_BANDS]
        if "swir" in feature_arrays:
            output_order.append("swir")
        output_order.extend(name for name in INDEX_BANDS if name in feature_arrays)

        for name in output_order:
            band_names.append(f"{slot}_{name}")
            band_arrays.append(feature_arrays[name])

        input_metadata.append(
            {
                **metadata,
                "feature_prefix": slot,
                "band_map": args.band_map,
                "output_bands": [f"{slot}_{name}" for name in output_order],
            }
        )

    profile = output_profile(args.inputs[0], len(band_arrays))
    write_feature_stack(args.output, band_arrays, band_names, profile)

    metadata = {
        "source": "local_multiband_geotiff",
        "inputs": input_metadata,
        "output": str(args.output),
        "band_count": len(band_names),
        "band_names": band_names,
        "timepoint_name_mode": "slot",
        "reflectance_scale": args.reflectance_scale,
        "zero_is_nodata": args.zero_is_nodata,
        "allow_missing_swir": args.allow_missing_swir,
        "aoi_mask": aoi_metadata,
        "notes": (
            "从本地多波段影像构建的标准特征栈。"
            "AWS 单波段 assets 和本地多波段输入都应先转换到该 schema，再交给下游流程。"
        ),
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metadata, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"已保存特征栈：{args.output}")
    print(f"已保存元数据：{args.metadata}")
    print(f"特征波段：{', '.join(band_names)}")


if __name__ == "__main__":
    main()
