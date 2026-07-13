"""
Calculate Sentinel-2 L2A NDVI, NDRE, NDMI and valid-pixel ratios
only inside AOI target-crop parcels, e.g. wheat parcels.

Recommended usage:
    python scripts\data_process\calc_s2_indices_csv.py ^
      --config D:\Projects\crop_harvest_window\configs\harvest_window.yaml ^
      --s2-dir D:\Projects\crop_harvest_window\data\S2_T49RFQ ^
      --out D:\Projects\crop_harvest_window\data\output\s2_t49rfq_wheat_aoi_indices.csv

If you do not want to use config:
    python scripts\data_process\calc_s2_indices_csv.py ^
      --s2-dir D:\Projects\crop_harvest_window\data\S2_T49RFQ ^
      --parcels D:\Projects\crop_harvest_window\data\your_parcels.shp ^
      --aoi D:\Projects\crop_harvest_window\data\your_aoi.shp ^
      --crop-type-code 1 ^
      --out D:\Projects\crop_harvest_window\data\output\s2_t49rfq_wheat_aoi_indices.csv

What it does:
    1. Read parcel boundary.
    2. Filter target crop parcels, default from config crop.crop_type_code.
    3. Intersect target parcels with AOI.
    4. Mask Sentinel-2 bands by those target parcels.
    5. Remove cloud / cloud shadow / invalid pixels with SCL and MSK_CLDPRB.
    6. Export scene-level CSV:
        NDVI = (B08 - B04) / (B08 + B04)
        NDRE = (B8A - B05) / (B8A + B05)
        NDMI = (B08 - B11) / (B08 + B11)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Optional

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.mask import mask
from rasterio.warp import reproject


INVALID_SCL_CLASSES = {
    0,   # No data
    1,   # Saturated / defective
    2,   # Dark area pixels
    3,   # Cloud shadows
    8,   # Cloud medium probability
    9,   # Cloud high probability
    10,  # Thin cirrus
    11,  # Snow / ice
}

CLOUD_SCL_CLASSES = {3, 8, 9, 10}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate S2 indices only inside AOI target-crop parcels."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/harvest_window.yaml"),
        help="Optional harvest_window.yaml. Used to read paths.parcels, paths.aoi and crop.crop_type_code.",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"D:\Projects\crop_harvest_window"),
        help="Project root. Relative paths in config are resolved against this directory.",
    )
    parser.add_argument(
        "--s2-dir",
        type=Path,
        default=Path(r"D:\Projects\crop_harvest_window\data\S2_T49RFQ"),
        help="Directory containing Sentinel-2 .SAFE folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(r"D:\Projects\crop_harvest_window\data\output\s2_t49rfq_wheat_aoi_indices.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--parcels",
        type=Path,
        default=None,
        help="Optional parcel boundary file. Overrides config paths.parcels.",
    )
    parser.add_argument(
        "--aoi",
        type=Path,
        default=None,
        help="Optional AOI boundary file. Overrides config paths.aoi.",
    )
    parser.add_argument(
        "--crop-type-code",
        type=int,
        default=None,
        help="Target crop_type code. Overrides config crop.crop_type_code.",
    )
    parser.add_argument(
        "--crop-column",
        type=str,
        default="crop_type",
        help="Crop type column name in parcel file.",
    )
    parser.add_argument(
        "--cloud-prob-threshold",
        type=float,
        default=50.0,
        help="Pixels with MSK_CLDPRB >= this threshold are treated as cloudy.",
    )
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=0.7,
        help="Minimum target-crop valid-pixel ratio in 0-1. Lower values cannot be tagged keep.",
    )
    parser.add_argument(
        "--keep-cloud-threshold",
        type=float,
        default=None,
        help="Maximum target-crop cloud percentage for keep. Defaults to config s2.keep_threshold or 10.",
    )
    parser.add_argument(
        "--review-cloud-threshold",
        type=float,
        default=None,
        help="Maximum target-crop cloud percentage for review. Defaults to config s2.review_threshold or 30.",
    )
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Reading --config requires pyyaml. Please install pyyaml or pass --parcels/--aoi manually.") from exc
    with path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f)
    return payload or {}


def resolve_path(project_root: Path, value: str | Path | None) -> Optional[Path]:
    if value is None or str(value).strip() == "":
        return None
    p = Path(value)
    return p if p.is_absolute() else project_root / p


def config_value(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = config
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def find_one(safe_dir: Path, patterns: list[str]) -> Optional[Path]:
    for pattern in patterns:
        matches = sorted(safe_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def find_s2_paths(safe_dir: Path) -> dict[str, Optional[Path]]:
    return {
        "B04": find_one(safe_dir, ["**/IMG_DATA/R10m/*_B04_10m.jp2"]),
        "B08": find_one(safe_dir, ["**/IMG_DATA/R10m/*_B08_10m.jp2"]),
        "B05": find_one(safe_dir, ["**/IMG_DATA/R20m/*_B05_20m.jp2"]),
        "B8A": find_one(safe_dir, ["**/IMG_DATA/R20m/*_B8A_20m.jp2"]),
        "B11": find_one(safe_dir, ["**/IMG_DATA/R20m/*_B11_20m.jp2"]),
        "SCL": find_one(safe_dir, ["**/IMG_DATA/R20m/*_SCL_20m.jp2"]),
        "CLDPRB": find_one(safe_dir, ["**/*MSK_CLDPRB_20m.jp2", "**/*CLDPRB*20m.jp2"]),
    }


def parse_date_from_name(name: str) -> str:
    match = re.search(r"_(20\d{6})T", name)
    if match:
        raw = match.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    return ""


def parse_tile_from_name(name: str) -> str:
    match = re.search(r"_(T\d{2}[A-Z]{3})_", name)
    return match.group(1) if match else ""


def load_target_geometries(
    parcels_path: Path,
    aoi_path: Optional[Path],
    crop_column: str,
    crop_type_code: int,
) -> tuple[Any, dict[str, Any]]:
    try:
        import geopandas as gpd
    except ImportError as exc:
        raise RuntimeError("This script requires geopandas to read parcels/AOI.") from exc

    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcel file not found: {parcels_path}")

    parcels = gpd.read_file(parcels_path)
    parcels = parcels[~parcels.geometry.is_empty & parcels.geometry.notna()].copy()
    total_parcels = int(len(parcels))

    if crop_column not in parcels.columns:
        raise ValueError(
            f"Column '{crop_column}' not found in parcel file. "
            f"Available columns: {list(parcels.columns)}"
        )

    target = parcels[parcels[crop_column].astype(str) == str(crop_type_code)].copy()
    target_before_aoi = int(len(target))

    aoi_features = None
    if aoi_path is not None:
        if not aoi_path.exists():
            raise FileNotFoundError(f"AOI file not found: {aoi_path}")

        aoi = gpd.read_file(aoi_path)
        aoi = aoi[~aoi.geometry.is_empty & aoi.geometry.notna()].copy()
        aoi_features = int(len(aoi))

        if target.crs and aoi.crs and target.crs != aoi.crs:
            aoi = aoi.to_crs(target.crs)

        aoi_union = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union
        target = target[target.intersects(aoi_union)].copy()
        # Clip to AOI, otherwise parcels that cross AOI boundary will include outside-AOI area.
        if len(target):
            target["geometry"] = target.geometry.intersection(aoi_union)
            target = target[~target.geometry.is_empty & target.geometry.notna()].copy()

    target_after_aoi = int(len(target))
    if target_after_aoi == 0:
        raise ValueError(
            f"No target crop parcels found. crop_column={crop_column}, crop_type_code={crop_type_code}, "
            f"parcels={parcels_path}, aoi={aoi_path}"
        )

    area_mu = None
    if "mu" in target.columns:
        try:
            area_mu = float(target["mu"].fillna(0).sum())
        except Exception:
            area_mu = None

    meta = {
        "parcels_path": str(parcels_path),
        "aoi_path": str(aoi_path) if aoi_path else "",
        "crop_column": crop_column,
        "crop_type_code": crop_type_code,
        "total_parcels": total_parcels,
        "target_parcels_before_aoi": target_before_aoi,
        "target_parcels_after_aoi": target_after_aoi,
        "aoi_features": aoi_features,
        "target_area_mu_if_available": area_mu,
        "crs": str(target.crs) if target.crs else "",
    }
    return target[["geometry"]].copy(), meta


def target_shapes_for_raster(target_gdf, dst_crs) -> list[dict]:
    gdf = target_gdf
    if gdf.crs and dst_crs and gdf.crs != dst_crs:
        gdf = gdf.to_crs(dst_crs)
    return [geom.__geo_interface__ for geom in gdf.geometry if geom is not None and not geom.is_empty]


def read_raster_masked(
    path: Path,
    target_gdf,
) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray]:
    with rasterio.open(path) as src:
        shapes = target_shapes_for_raster(target_gdf, src.crs)
        if not shapes:
            raise ValueError("No target geometries available after CRS conversion.")
        # 用 nodata=0 填充裁剪区域，避免 uint16 不能存 NaN 的问题
        data, transform = mask(src, shapes, crop=True, filled=True, nodata=0)
        band = data[0]
        arr = band.astype(np.float32)
        target_mask = geometry_mask(
            shapes,
            out_shape=arr.shape,
            transform=transform,
            invert=True,
            all_touched=False,
        )
        # 将填充值 0 和原始 nodata 都设为 NaN
        arr[arr == 0] = np.nan
        if src.nodata is not None:
            arr[arr == src.nodata] = np.nan

        valid_mask = target_mask & np.isfinite(arr)

        profile = src.profile.copy()
        profile.update(
            height=arr.shape[0],
            width=arr.shape[1],
            transform=transform,
            crs=src.crs,
            dtype="float32",
        )
        return arr, profile, valid_mask, target_mask


def read_resampled_to_match(
    src_path: Path,
    ref_profile: dict,
    resampling: Resampling,
) -> np.ndarray:
    with rasterio.open(src_path) as src:
        dst = np.full(
            (int(ref_profile["height"]), int(ref_profile["width"])),
            np.nan,
            dtype=np.float32,
        )
        reproject(
            source=rasterio.band(src, 1),
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=resampling,
            src_nodata=src.nodata,
            dst_nodata=np.nan,
        )
        return dst


def calc_index(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    denominator = a + b
    out = np.full(a.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(denominator) > 1e-6)
    out[valid] = (a[valid] - b[valid]) / denominator[valid]
    out[(out < -1.5) | (out > 1.5)] = np.nan
    return out


def cloud_valid_mask(
    ref_profile: dict,
    target_mask: np.ndarray,
    scl_path: Optional[Path],
    cldprb_path: Optional[Path],
    cloud_prob_threshold: float,
) -> tuple[np.ndarray, float, float]:
    height = int(ref_profile["height"])
    width = int(ref_profile["width"])
    if target_mask.shape != (height, width):
        raise ValueError(
            f"Target mask shape {target_mask.shape} does not match raster shape {(height, width)}."
        )

    valid = target_mask.copy()
    cloud = np.zeros((height, width), dtype=bool)

    if scl_path is not None:
        scl = read_resampled_to_match(scl_path, ref_profile, Resampling.nearest)
        scl_int = np.rint(scl).astype(np.int16)
        scl_finite = np.isfinite(scl)
        invalid_scl = np.isin(scl_int, list(INVALID_SCL_CLASSES))
        cloud |= target_mask & scl_finite & np.isin(scl_int, list(CLOUD_SCL_CLASSES))
        valid &= scl_finite & (~invalid_scl)

    if cldprb_path is not None:
        cld = read_resampled_to_match(cldprb_path, ref_profile, Resampling.nearest)
        cld_finite = np.isfinite(cld)
        cloud_prob = cld_finite & (cld >= cloud_prob_threshold)
        cloud |= target_mask & cloud_prob
        valid &= cld_finite & (~cloud_prob)

    total = int(np.count_nonzero(target_mask))
    cloud_ratio_pct = float(np.count_nonzero(cloud)) / total * 100.0 if total else np.nan
    clear_ratio_pct = float(np.count_nonzero(valid)) / total * 100.0 if total else np.nan
    return valid, cloud_ratio_pct, clear_ratio_pct


def summarize_index(
    index: np.ndarray,
    valid_mask: np.ndarray,
    target_mask: np.ndarray,
) -> dict[str, float | int]:
    total = int(np.count_nonzero(target_mask))
    finite_mask = target_mask & valid_mask & np.isfinite(index)
    values = index[finite_mask]
    valid_count = int(values.size)

    return {
        "mean": float(np.nanmean(values)) if valid_count else np.nan,
        "median": float(np.nanmedian(values)) if valid_count else np.nan,
        "std": float(np.nanstd(values)) if valid_count else np.nan,
        "p10": float(np.nanpercentile(values, 10)) if valid_count else np.nan,
        "p90": float(np.nanpercentile(values, 90)) if valid_count else np.nan,
        "min": float(np.nanmin(values)) if valid_count else np.nan,
        "max": float(np.nanmax(values)) if valid_count else np.nan,
        "valid_pixel_count": valid_count,
        "total_pixel_count": total,
        "valid_pixel_ratio": float(valid_count) / total if total else np.nan,
        "valid_pixel_ratio_pct": float(valid_count) / total * 100.0 if total else np.nan,
    }


def fmt(value: Any, ndigits: int = 6) -> Any:
    try:
        if value is None:
            return ""
        value_float = float(value)
        if not np.isfinite(value_float):
            return ""
        return round(value_float, ndigits)
    except Exception:
        return value


def quality_tag(
    cloud_ratio_pct: float,
    valid_ratio_0_1: float,
    min_valid_ratio: float,
    keep_cloud_threshold: float,
    review_cloud_threshold: float,
) -> str:
    if not np.isfinite(cloud_ratio_pct) or not np.isfinite(valid_ratio_0_1):
        return "drop_or_fill"

    if cloud_ratio_pct <= keep_cloud_threshold:
        cloud_tag = "keep"
    elif cloud_ratio_pct <= review_cloud_threshold:
        cloud_tag = "review"
    else:
        cloud_tag = "drop_or_fill"

    if valid_ratio_0_1 < 0.4:
        return "drop_or_fill"
    if valid_ratio_0_1 < min_valid_ratio and cloud_tag == "keep":
        return "review"
    return cloud_tag


def process_scene(
    safe_dir: Path,
    target_gdf,
    cloud_prob_threshold: float,
    min_valid_ratio: float,
    keep_cloud_threshold: float,
    review_cloud_threshold: float,
) -> dict[str, Any]:
    paths = find_s2_paths(safe_dir)
    required = ["B04", "B08", "B05", "B8A", "B11"]
    missing = [key for key in required if paths[key] is None]

    row: dict[str, Any] = {
        "date": parse_date_from_name(safe_dir.name),
        "tile": parse_tile_from_name(safe_dir.name),
        "safe_name": safe_dir.name,
        "status": "ok",
        "missing_required_bands": ";".join(missing),
        "has_scl": paths["SCL"] is not None,
        "has_cldprb": paths["CLDPRB"] is not None,
    }

    if missing:
        row["status"] = "missing_required_bands"
        return row

    # NDVI at 10 m: B08 and B04.
    b04, p10, valid_b04, target_10m = read_raster_masked(paths["B04"], target_gdf)
    b08 = read_resampled_to_match(paths["B08"], p10, Resampling.bilinear)
    valid_10m, cloud_ratio_10m, clear_ratio_10m = cloud_valid_mask(
        p10, target_10m, paths["SCL"], paths["CLDPRB"], cloud_prob_threshold
    )
    ndvi = calc_index(b08, b04)
    ndvi_stats = summarize_index(ndvi, valid_10m & valid_b04, target_10m)

    # NDRE at 20 m: B8A and B05.
    b05, p20, valid_b05, target_20m = read_raster_masked(paths["B05"], target_gdf)
    b8a = read_resampled_to_match(paths["B8A"], p20, Resampling.bilinear)
    valid_20m, cloud_ratio_20m, clear_ratio_20m = cloud_valid_mask(
        p20, target_20m, paths["SCL"], paths["CLDPRB"], cloud_prob_threshold
    )
    ndre = calc_index(b8a, b05)
    ndre_stats = summarize_index(ndre, valid_20m & valid_b05, target_20m)

    # NDMI at 20 m: B08 resampled to B11 grid and B11.
    b11, p20_b11, valid_b11, target_20m_b11 = read_raster_masked(
        paths["B11"], target_gdf
    )
    b08_to_20m = read_resampled_to_match(paths["B08"], p20_b11, Resampling.bilinear)
    valid_20m_b11, _, _ = cloud_valid_mask(
        p20_b11,
        target_20m_b11,
        paths["SCL"],
        paths["CLDPRB"],
        cloud_prob_threshold,
    )
    ndmi = calc_index(b08_to_20m, b11)
    ndmi_stats = summarize_index(
        ndmi, valid_20m_b11 & valid_b11, target_20m_b11
    )

    effective_ratio = min(
        float(ndvi_stats["valid_pixel_ratio"]),
        float(ndre_stats["valid_pixel_ratio"]),
        float(ndmi_stats["valid_pixel_ratio"]),
    )
    classification_cloud_ratio = max(cloud_ratio_10m, cloud_ratio_20m)

    row.update(
        {
            "cloud_prob_threshold": cloud_prob_threshold,
            "cloud_pixel_ratio_10m_pct": fmt(cloud_ratio_10m),
            "cloud_pixel_ratio_20m_pct": fmt(cloud_ratio_20m),
            "clear_pixel_ratio_10m_pct": fmt(clear_ratio_10m),
            "clear_pixel_ratio_20m_pct": fmt(clear_ratio_20m),
            "classification_cloud_ratio_pct": fmt(classification_cloud_ratio),
            "effective_pixel_ratio": fmt(effective_ratio),
            "effective_pixel_ratio_pct": fmt(effective_ratio * 100.0),
            "quality_tag": quality_tag(
                classification_cloud_ratio,
                effective_ratio,
                min_valid_ratio,
                keep_cloud_threshold,
                review_cloud_threshold,
            ),

            "ndvi_mean": fmt(ndvi_stats["mean"]),
            "ndvi_median": fmt(ndvi_stats["median"]),
            "ndvi_std": fmt(ndvi_stats["std"]),
            "ndvi_p10": fmt(ndvi_stats["p10"]),
            "ndvi_p90": fmt(ndvi_stats["p90"]),
            "ndvi_min": fmt(ndvi_stats["min"]),
            "ndvi_max": fmt(ndvi_stats["max"]),
            "ndvi_valid_pixel_count": ndvi_stats["valid_pixel_count"],
            "ndvi_total_pixel_count": ndvi_stats["total_pixel_count"],
            "ndvi_valid_pixel_ratio": fmt(ndvi_stats["valid_pixel_ratio"]),
            "ndvi_valid_pixel_ratio_pct": fmt(ndvi_stats["valid_pixel_ratio_pct"]),

            "ndre_mean": fmt(ndre_stats["mean"]),
            "ndre_median": fmt(ndre_stats["median"]),
            "ndre_std": fmt(ndre_stats["std"]),
            "ndre_p10": fmt(ndre_stats["p10"]),
            "ndre_p90": fmt(ndre_stats["p90"]),
            "ndre_min": fmt(ndre_stats["min"]),
            "ndre_max": fmt(ndre_stats["max"]),
            "ndre_valid_pixel_count": ndre_stats["valid_pixel_count"],
            "ndre_total_pixel_count": ndre_stats["total_pixel_count"],
            "ndre_valid_pixel_ratio": fmt(ndre_stats["valid_pixel_ratio"]),
            "ndre_valid_pixel_ratio_pct": fmt(ndre_stats["valid_pixel_ratio_pct"]),

            "ndmi_mean": fmt(ndmi_stats["mean"]),
            "ndmi_median": fmt(ndmi_stats["median"]),
            "ndmi_std": fmt(ndmi_stats["std"]),
            "ndmi_p10": fmt(ndmi_stats["p10"]),
            "ndmi_p90": fmt(ndmi_stats["p90"]),
            "ndmi_min": fmt(ndmi_stats["min"]),
            "ndmi_max": fmt(ndmi_stats["max"]),
            "ndmi_valid_pixel_count": ndmi_stats["valid_pixel_count"],
            "ndmi_total_pixel_count": ndmi_stats["total_pixel_count"],
            "ndmi_valid_pixel_ratio": fmt(ndmi_stats["valid_pixel_ratio"]),
            "ndmi_valid_pixel_ratio_pct": fmt(ndmi_stats["valid_pixel_ratio_pct"]),
        }
    )
    return row


def main() -> int:
    args = parse_args()
    config = load_yaml_config(args.config)

    parcels_from_config = resolve_path(args.project_root, config_value(config, "paths.parcels"))
    aoi_from_config = resolve_path(args.project_root, config_value(config, "paths.aoi"))
    crop_from_config = config_value(config, "crop.crop_type_code")
    keep_cloud_threshold = (
        args.keep_cloud_threshold
        if args.keep_cloud_threshold is not None
        else float(config_value(config, "s2.keep_threshold", 10.0))
    )
    review_cloud_threshold = (
        args.review_cloud_threshold
        if args.review_cloud_threshold is not None
        else float(config_value(config, "s2.review_threshold", 30.0))
    )

    parcels_path = args.parcels or parcels_from_config
    aoi_path = args.aoi or aoi_from_config
    crop_type_code = args.crop_type_code if args.crop_type_code is not None else crop_from_config

    if parcels_path is None:
        raise ValueError("Parcel path is missing. Pass --parcels or set paths.parcels in config.")
    if crop_type_code is None:
        raise ValueError("crop_type_code is missing. Pass --crop-type-code or set crop.crop_type_code in config.")
    if not 0.0 <= args.cloud_prob_threshold <= 100.0:
        raise ValueError("--cloud-prob-threshold must be between 0 and 100.")
    if not 0.0 <= args.min_valid_ratio <= 1.0:
        raise ValueError("--min-valid-ratio must be between 0 and 1.")
    if not 0.0 <= keep_cloud_threshold <= review_cloud_threshold <= 100.0:
        raise ValueError(
            "Cloud thresholds must satisfy 0 <= keep <= review <= 100."
        )

    crop_type_code = int(crop_type_code)

    print("Target settings:")
    print(f"  s2_dir: {args.s2_dir}")
    print(f"  parcels: {parcels_path}")
    print(f"  aoi: {aoi_path}")
    print(f"  crop_column: {args.crop_column}")
    print(f"  crop_type_code: {crop_type_code}")
    print(f"  keep_cloud_threshold: {keep_cloud_threshold}%")
    print(f"  review_cloud_threshold: {review_cloud_threshold}%")
    print(f"  out: {args.out}")

    target_gdf, target_meta = load_target_geometries(
        parcels_path=parcels_path,
        aoi_path=aoi_path,
        crop_column=args.crop_column,
        crop_type_code=crop_type_code,
    )

    print("Target parcel summary:")
    print(json.dumps(target_meta, ensure_ascii=False, indent=2))

    safe_dirs = sorted(
        args.s2_dir.glob("*.SAFE"),
        key=lambda path: (parse_date_from_name(path.name), path.name),
    )
    if not safe_dirs:
        raise FileNotFoundError(f"No .SAFE folders found in {args.s2_dir}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for idx, safe_dir in enumerate(safe_dirs, start=1):
        print(f"[{idx}/{len(safe_dirs)}] Processing {safe_dir.name}", flush=True)
        try:
            rows.append(
                process_scene(
                    safe_dir=safe_dir,
                    target_gdf=target_gdf,
                    cloud_prob_threshold=args.cloud_prob_threshold,
                    min_valid_ratio=args.min_valid_ratio,
                    keep_cloud_threshold=keep_cloud_threshold,
                    review_cloud_threshold=review_cloud_threshold,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "date": parse_date_from_name(safe_dir.name),
                    "tile": parse_tile_from_name(safe_dir.name),
                    "safe_name": safe_dir.name,
                    "status": "error",
                    "error": str(exc)[:800],
                }
            )

    # Add target metadata columns to every row for traceability.
    for row in rows:
        row["target_crop_type_code"] = crop_type_code
        row["target_parcels_after_aoi"] = target_meta["target_parcels_after_aoi"]
        row["target_area_mu_if_available"] = target_meta["target_area_mu_if_available"]
        row["parcels_path"] = target_meta["parcels_path"]
        row["aoi_path"] = target_meta["aoi_path"]

    fieldnames = [
        "date",
        "tile",
        "safe_name",
        "status",
        "quality_tag",
        "target_crop_type_code",
        "target_parcels_after_aoi",
        "target_area_mu_if_available",
        "missing_required_bands",
        "has_scl",
        "has_cldprb",
        "cloud_prob_threshold",
        "cloud_pixel_ratio_10m_pct",
        "cloud_pixel_ratio_20m_pct",
        "clear_pixel_ratio_10m_pct",
        "clear_pixel_ratio_20m_pct",
        "classification_cloud_ratio_pct",
        "effective_pixel_ratio",
        "effective_pixel_ratio_pct",

        "ndvi_mean",
        "ndvi_median",
        "ndvi_std",
        "ndvi_p10",
        "ndvi_p90",
        "ndvi_min",
        "ndvi_max",
        "ndvi_valid_pixel_count",
        "ndvi_total_pixel_count",
        "ndvi_valid_pixel_ratio",
        "ndvi_valid_pixel_ratio_pct",

        "ndre_mean",
        "ndre_median",
        "ndre_std",
        "ndre_p10",
        "ndre_p90",
        "ndre_min",
        "ndre_max",
        "ndre_valid_pixel_count",
        "ndre_total_pixel_count",
        "ndre_valid_pixel_ratio",
        "ndre_valid_pixel_ratio_pct",

        "ndmi_mean",
        "ndmi_median",
        "ndmi_std",
        "ndmi_p10",
        "ndmi_p90",
        "ndmi_min",
        "ndmi_max",
        "ndmi_valid_pixel_count",
        "ndmi_total_pixel_count",
        "ndmi_valid_pixel_ratio",
        "ndmi_valid_pixel_ratio_pct",

        "parcels_path",
        "aoi_path",
        "error",
    ]

    with args.out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Done. Output CSV: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
