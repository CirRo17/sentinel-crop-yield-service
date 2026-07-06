from __future__ import annotations

import math
import re
import uuid
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import geopandas as gpd
import rasterio
import requests
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.mask import mask
from rasterio.transform import array_bounds, from_origin, xy
from rasterio.warp import reproject
from rasterio.vrt import WarpedVRT

from .config import EARTH_SEARCH_URL, OUTPUT_DIR, S2_DIR
from .crop_distribution import (
    download_external_crop_raster,
    download_external_crop_vector,
    downloaded_crop_raster,
    local_crop_raster,
)
from .geometry import bounds as fc_bounds
from .geometry import shapes_for_crs
from .yield_model import estimate_yield, lai_from_ci, uncertainty


SCENE_RE = re.compile(r"MSIL2A_(\d{8}T\d{6}).*_(T\d{2}[A-Z]{3})_")


@dataclass
class Scene:
    scene_id: str
    acquired: date
    source: str
    red: str
    nir: str
    red_edge: Optional[str] = None
    scl: Optional[str] = None
    safe_path: Optional[str] = None
    cloud_cover: Optional[float] = None
    red_index: int = 1
    nir_index: int = 1
    red_edge_index: int = 1


def _parse_scene_date(name: str) -> date:
    match = SCENE_RE.search(name)
    if not match:
        return date.fromtimestamp(0)
    return datetime.strptime(match.group(1), "%Y%m%dT%H%M%S").date()


def scan_local_scenes() -> List[Scene]:
    scenes: List[Scene] = []
    for safe in sorted(S2_DIR.glob("*.SAFE")):
        red = next(safe.rglob("*_B04_10m.jp2"), None)
        nir = next(safe.rglob("*_B08_10m.jp2"), None)
        red_edge = next(safe.rglob("*_B05_20m.jp2"), None)
        scl = next(safe.rglob("*_SCL_20m.jp2"), None)
        if red and nir:
            scenes.append(
                Scene(
                    scene_id=safe.name,
                    acquired=_parse_scene_date(safe.name),
                    source="local",
                    red=str(red),
                    nir=str(nir),
                    red_edge=str(red_edge) if red_edge else None,
                    scl=str(scl) if scl else None,
                    safe_path=str(safe),
                )
            )
    return scenes


DATE_RE = re.compile(r"(20\d{2})[-_]?([01]\d)[-_]?([0-3]\d)")


def _date_from_name(path: str | Path) -> date:
    match = DATE_RE.search(Path(path).name)
    if not match:
        return date.fromtimestamp(0)
    y, m, d = match.groups()
    try:
        return date(int(y), int(m), int(d))
    except ValueError:
        return date.fromtimestamp(0)


def _band_kind(path: str | Path) -> Optional[str]:
    name = Path(path).stem.lower()
    tokens = re.split(r"[^a-z0-9]+", name)
    joined = " ".join(tokens)
    if any(t in tokens for t in ("scl", "qa", "mask")):
        return "scl"
    if "rededge" in joined or "red edge" in joined or any(t in tokens for t in ("re", "b05", "b5")):
        return "red_edge"
    if any(t in tokens for t in ("nir", "b08", "b8", "ir")):
        return "nir"
    if any(t in tokens for t in ("red", "b04", "b4", "r")):
        return "red"
    return None


def _band_indices(path: str | Path) -> Dict[str, int]:
    with rasterio.open(path) as src:
        indexes: Dict[str, int] = {}
        descriptions = [d.lower() if d else "" for d in src.descriptions]
        for idx, desc in enumerate(descriptions, start=1):
            if "nir" in desc or "near infrared" in desc or "b08" in desc or desc == "b8":
                indexes.setdefault("nir", idx)
            elif "red edge" in desc or "rededge" in desc or "b05" in desc or desc == "b5":
                indexes.setdefault("red_edge", idx)
            elif "red" in desc or "b04" in desc or desc == "b4":
                indexes.setdefault("red", idx)
        if src.count >= 5:
            indexes.setdefault("red", 3)
            indexes.setdefault("red_edge", 4)
            indexes.setdefault("nir", 5)
        elif src.count == 4:
            indexes.setdefault("red", 3)
            indexes.setdefault("nir", 4)
        elif src.count == 2:
            indexes.setdefault("red", 1)
            indexes.setdefault("nir", 2)
        elif src.count == 1:
            kind = _band_kind(path)
            if kind:
                indexes[kind] = 1
        return indexes


def _scene_group_id(path: str | Path) -> str:
    name = Path(path).stem.lower()
    name = re.sub(r"(rededge|red_edge|red|nir|scl|b0?[458]|mask|qa)", "", name)
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_") or Path(path).stem


def _collect_tifs(input_dir: Optional[str], file_paths: Optional[List[str]]) -> List[Path]:
    paths: List[Path] = []
    if input_dir:
        folder = Path(input_dir)
        if folder.is_dir():
            for suffix in ("*.tif", "*.tiff", "*.TIF", "*.TIFF"):
                paths.extend(folder.rglob(suffix))
    if file_paths:
        paths.extend(Path(p) for p in file_paths if p)
    return sorted({p for p in paths if p.exists()})


def scan_tif_scenes(
    input_dir: Optional[str] = None,
    file_paths: Optional[List[str]] = None,
    source: str = "local_tif",
) -> List[Scene]:
    tif_paths = _collect_tifs(input_dir, file_paths)
    scenes: List[Scene] = []
    grouped: Dict[str, Dict[str, Path]] = {}

    for path in tif_paths:
        indexes = _band_indices(path)
        with rasterio.open(path) as src:
            if src.count > 1 and "red" in indexes and "nir" in indexes:
                scenes.append(
                    Scene(
                        scene_id=path.name,
                        acquired=_date_from_name(path),
                        source=source,
                        red=str(path),
                        nir=str(path),
                        red_edge=str(path) if "red_edge" in indexes else None,
                        safe_path=str(path),
                        red_index=indexes["red"],
                        nir_index=indexes["nir"],
                        red_edge_index=indexes.get("red_edge", 1),
                    )
                )
                continue
        kind = _band_kind(path)
        if kind:
            grouped.setdefault(_scene_group_id(path), {})[kind] = path

    for scene_id, bands in grouped.items():
        if "red" in bands and "nir" in bands:
            scenes.append(
                Scene(
                    scene_id=scene_id,
                    acquired=_date_from_name(bands["red"]),
                    source=source,
                    red=str(bands["red"]),
                    nir=str(bands["nir"]),
                    red_edge=str(bands["red_edge"]) if "red_edge" in bands else None,
                    scl=str(bands["scl"]) if "scl" in bands else None,
                    safe_path=str(Path(bands["red"]).parent),
                )
            )
    return sorted(scenes, key=lambda scene: (scene.acquired, scene.scene_id))


def scene_metadata(scene: Scene) -> Dict[str, Any]:
    with rasterio.open(scene.red) as src:
        transformer = Transformer.from_crs(src.crs, "EPSG:4326", always_xy=True)
        left, bottom, right, top = src.bounds
        xs, ys = transformer.transform([left, right], [bottom, top])
    return {
        "scene_id": scene.scene_id,
        "date": scene.acquired.isoformat(),
        "source": scene.source,
        "safe_path": scene.safe_path,
        "bounds": [min(xs), min(ys), max(xs), max(ys)],
        "cloud_cover": scene.cloud_cover,
    }


def _parse_date(value: Optional[str]) -> Optional[date]:
    if not value:
        return None
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


def _date_filter(scenes: Iterable[Scene], start: Optional[str], end: Optional[str]) -> List[Scene]:
    start_date = _parse_date(start)
    end_date = _parse_date(end)
    out = []
    for scene in scenes:
        if start_date and scene.acquired < start_date:
            continue
        if end_date and scene.acquired > end_date:
            continue
        out.append(scene)
    return out


def _select_target_scenes(scenes: List[Scene], target: Optional[str], start: Optional[str], end: Optional[str]) -> List[Scene]:
    filtered = _date_filter(scenes, start, end)
    if not filtered:
        filtered = scenes
    if not filtered:
        return []
    if target:
        target_date = _parse_date(target)
        best_delta = min(abs((scene.acquired - target_date).days) for scene in filtered)
        return [scene for scene in filtered if abs((scene.acquired - target_date).days) == best_delta]
    latest = max(scene.acquired for scene in filtered)
    return [scene for scene in filtered if scene.acquired == latest]


def query_aws_scenes(
    fc: Dict[str, Any],
    start: Optional[str],
    end: Optional[str],
    limit: int = 5,
    max_cloud_cover: Optional[float] = 70.0,
) -> List[Scene]:
    minx, miny, maxx, maxy = fc_bounds(fc)
    datetime_range = f"{start or '2020-01-01'}T00:00:00Z/{end or datetime.utcnow().date().isoformat()}T23:59:59Z"
    payload = {
        "collections": ["sentinel-2-l2a"],
        "bbox": [minx, miny, maxx, maxy],
        "datetime": datetime_range,
        "limit": limit,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    if max_cloud_cover is not None:
        payload["query"] = {"eo:cloud_cover": {"lt": float(max_cloud_cover)}}
    response = requests.post(EARTH_SEARCH_URL, json=payload, timeout=30)
    response.raise_for_status()
    scenes: List[Scene] = []
    for item in response.json().get("features", []):
        assets = item.get("assets", {})
        red = assets.get("red") or assets.get("B04")
        nir = assets.get("nir") or assets.get("B08")
        red_edge = assets.get("rededge1") or assets.get("B05")
        scl = assets.get("scl") or assets.get("SCL")
        if not red or not nir:
            continue
        acquired = datetime.fromisoformat(item["properties"]["datetime"].replace("Z", "+00:00")).date()
        scenes.append(
            Scene(
                scene_id=item["id"],
                acquired=acquired,
                source="aws",
                red=red["href"],
                nir=nir["href"],
                red_edge=red_edge["href"] if red_edge else None,
                scl=scl["href"] if scl else None,
                cloud_cover=item.get("properties", {}).get("eo:cloud_cover"),
            )
        )
    return scenes


def _read_masked(path: str, fc: Dict[str, Any], nodata=0, band_index: int = 1):
    with rasterio.open(path) as src:
        shapes = shapes_for_crs(fc, src.crs)
        arr, transform = mask(src, shapes, crop=True, filled=True, nodata=nodata, indexes=band_index)
        return (arr[0] if arr.ndim == 3 else arr), transform, src.crs


def _resample_array_to_resolution(arr, transform, crs, resolution_m: float, resampling=Resampling.bilinear):
    current_x = abs(float(transform.a))
    current_y = abs(float(transform.e))
    target_resolution = float(resolution_m)
    if getattr(crs, "is_geographic", False):
        target_resolution = target_resolution / 111320.0
    if math.isclose(current_x, target_resolution, rel_tol=0.01) and math.isclose(current_y, target_resolution, rel_tol=0.01):
        return arr, transform

    west, south, east, north = array_bounds(arr.shape[0], arr.shape[1], transform)
    width = max(1, int(math.ceil((east - west) / target_resolution)))
    height = max(1, int(math.ceil((north - south) / target_resolution)))
    dst_transform = from_origin(west, north, target_resolution, target_resolution)
    dst = np.zeros((height, width), dtype="float32")
    reproject(
        source=arr.astype("float32"),
        destination=dst,
        src_transform=transform,
        src_crs=crs,
        src_nodata=0,
        dst_transform=dst_transform,
        dst_crs=crs,
        dst_nodata=0,
        resampling=resampling,
    )
    return dst, dst_transform


def _read_warped(path: str, dst_crs, dst_transform, dst_shape, resampling=Resampling.nearest, band_index: int = 1):
    with rasterio.open(path) as src:
        src_nodata = src.nodata if src.nodata is not None else 0
        with WarpedVRT(
            src,
            crs=dst_crs,
            transform=dst_transform,
            width=dst_shape[1],
            height=dst_shape[0],
            resampling=resampling,
            src_nodata=src_nodata,
            nodata=0,
        ) as vrt:
            return vrt.read(band_index)


def _pixel_area_ha(transform, crs=None) -> float:
    area = abs(transform.a * transform.e)
    if crs is not None and getattr(crs, "is_geographic", False):
        area *= 111320.0 * 111320.0
    return area / 10000.0


def _write_output_tif(array, transform, crs, prefix: str, suffix: str) -> Dict[str, str]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"{prefix}_{suffix}.tif"
    data = np.where(np.isfinite(array), array, -9999).astype("float32")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="float32",
        crs=crs,
        transform=transform,
        nodata=-9999.0,
        compress="deflate",
        predictor=2,
    ) as dst:
        dst.write(data, 1)
    return {"url": f"/outputs/{path.name}", "path": str(path)}


def _valid_scl_mask(scene: Scene, dst_crs, dst_transform, dst_shape):
    if not scene.scl:
        return None
    try:
        scl = _read_warped(scene.scl, dst_crs, dst_transform, dst_shape, Resampling.nearest)
    except Exception:
        return None
    return np.isin(scl, [4, 5])


def _resolve_vector_path(path_or_dir: str | Path) -> Path:
    path = Path(path_or_dir)
    if path.is_dir():
        candidates = []
        for pattern in ("*.shp", "*.geojson", "*.json", "*.gpkg", "*.zip"):
            candidates.extend(path.glob(pattern))
        if candidates:
            return sorted(candidates)[0]
    return path


def _read_crop_vector(path: str | Path):
    path = _resolve_vector_path(path)
    if path.suffix.lower() == ".zip":
        gdf = gpd.read_file(f"zip://{path}")
    else:
        gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Crop distribution vector is empty.")
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg=4326)
    return gdf


def _crop_vector_mask(
    crop: str,
    vector_path: str | Path,
    field: str,
    dst_crs,
    dst_transform,
    dst_shape,
):
    gdf = _read_crop_vector(vector_path)
    if field not in gdf.columns:
        raise ValueError(f"Crop distribution vector has no '{field}' field.")
    code = { "rice": 1, "wheat": 2, "maize": 3 }[crop]
    subset = gdf[gdf[field].astype("float64") == float(code)]
    if subset.empty:
        return np.zeros(dst_shape, dtype=bool)
    subset = subset.to_crs(dst_crs)
    shapes = [(geom, 1) for geom in subset.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros(dst_shape, dtype=bool)
    return rasterize(
        shapes,
        out_shape=dst_shape,
        transform=dst_transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    ).astype(bool)


def _crop_mask(
    crop: str,
    source: str,
    tif_url: Optional[str],
    dst_crs,
    dst_transform,
    dst_shape,
    vector_path: Optional[str] = None,
    vector_url: Optional[str] = None,
    crop_type_field: str = "crop_type",
):
    if source == "none":
        return None, "none", None
    if source in {"local_vector", "api_vector"} or vector_path or vector_url:
        path = Path(vector_path) if vector_path else None
        used = "local_vector"
        if source == "api_vector" or vector_url:
            path = download_external_crop_vector(crop, vector_url)
            used = "api_vector" if path else "api_vector_unavailable"
        if path is None:
            return None, used, None
        try:
            return _crop_vector_mask(crop, path, crop_type_field, dst_crs, dst_transform, dst_shape), used, str(path)
        except Exception:
            return None, f"{used}_read_failed", str(path)
    path: Optional[Path] = None
    used = source
    if source in {"downloaded", "auto"}:
        path = downloaded_crop_raster(crop)
        if path is not None:
            used = "downloaded"
        elif source == "downloaded":
            return None, "downloaded_unavailable", None
    if path is None and source in {"api", "auto"}:
        path = download_external_crop_raster(crop, tif_url)
        if path is not None:
            used = "api"
        if path is None and source == "api":
            return None, "api_unavailable", None
    if path is None and source in {"local", "auto", "api"}:
        path = local_crop_raster(crop)
        used = "local" if path else "none"
    if path is None:
        return None, used, None
    try:
        arr = _read_warped(str(path), dst_crs, dst_transform, dst_shape, Resampling.nearest)
        if used in {"api", "downloaded"}:
            crop_code = { "rice": 1, "wheat": 2, "maize": 3 }[crop]
            mask_arr = arr == crop_code
            if used == "api" and not np.any(mask_arr):
                mask_arr = arr > 0
        else:
            mask_arr = arr > 0
        return mask_arr, used, str(path)
    except Exception:
        return None, f"{used}_read_failed", str(path)


def _grid_features(values, yields, lai, valid, transform, crs, limit=1400):
    rows, cols = np.where(valid)
    total = len(rows)
    if total == 0:
        return {"type": "FeatureCollection", "features": []}
    step = max(1, math.ceil(total / limit))
    transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    features = []
    for row, col in zip(rows[::step], cols[::step]):
        x1, y1 = xy(transform, int(row), int(col), offset="ul")
        x2, y2 = xy(transform, int(row), int(col), offset="lr")
        xs, ys = transformer.transform([x1, x2], [y1, y2])
        minx, maxx = min(xs), max(xs)
        miny, maxy = min(ys), max(ys)
        props = {
            "ndvi": float(values[row, col]),
            "yield_kg_ha": float(yields[row, col]),
        }
        if lai is not None:
            props["lai"] = float(lai[row, col])
        features.append(
            {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [minx, miny],
                        [maxx, miny],
                        [maxx, maxy],
                        [minx, maxy],
                        [minx, miny],
                    ]],
                },
            }
        )
    return {"type": "FeatureCollection", "features": features}


def compute_scene(
    scene: Scene,
    fc: Dict[str, Any],
    crop: str,
    index: str,
    crop_mask_source: str,
    crop_distribution_tif_url: Optional[str],
    crop_distribution_vector_path: Optional[str],
    crop_distribution_vector_url: Optional[str],
    crop_type_field: str,
    target_resolution_m: int,
    use_cloud_mask: bool,
    include_grid: bool,
    export_tif: bool = False,
    output_prefix: Optional[str] = None,
    model_coefficients=None,
    yield_function: str = "default",
    lai_k=None,
    lai_m=None,
) -> Dict[str, Any]:
    red, transform, crs = _read_masked(scene.red, fc, nodata=0, band_index=scene.red_index)
    red, transform = _resample_array_to_resolution(red, transform, crs, target_resolution_m, Resampling.bilinear)
    nir = _read_warped(scene.nir, crs, transform, red.shape, Resampling.bilinear, band_index=scene.nir_index)
    red = red.astype("float32")
    nir = nir.astype("float32")
    denominator = nir + red
    ndvi = np.divide(nir - red, denominator, out=np.full_like(red, np.nan), where=denominator != 0)
    valid = np.isfinite(ndvi) & (red > 0) & (nir > 0) & (ndvi >= -1) & (ndvi <= 1)

    if use_cloud_mask:
        scl_valid = _valid_scl_mask(scene, crs, transform, red.shape)
        if scl_valid is not None and np.any(valid & scl_valid):
            valid &= scl_valid

    crop_mask, used_crop_source, crop_path = _crop_mask(
        crop,
        crop_mask_source,
        crop_distribution_tif_url,
        crs,
        transform,
        red.shape,
        crop_distribution_vector_path,
        crop_distribution_vector_url,
        crop_type_field,
    )
    if crop_mask is not None:
        if np.any(valid & crop_mask):
            valid &= crop_mask
        elif crop_mask_source == "auto" and used_crop_source == "api":
            fallback_mask, fallback_source, fallback_path = _crop_mask(
                crop, "local", None, crs, transform, red.shape
            )
            if fallback_mask is not None and np.any(valid & fallback_mask):
                valid &= fallback_mask
                used_crop_source = f"api_no_overlap_fallback_{fallback_source}"
                crop_path = fallback_path
            else:
                valid &= crop_mask
        else:
            valid &= crop_mask

    lai = None
    if scene.red_edge:
        try:
            red_edge = _read_warped(
                scene.red_edge,
                crs,
                transform,
                red.shape,
                Resampling.bilinear,
                band_index=scene.red_edge_index,
            ).astype("float32")
            ci = np.divide(nir, red_edge, out=np.full_like(nir, np.nan), where=red_edge > 0) - 1
            lai = lai_from_ci(ci, crop, lai_k, lai_m)
        except Exception:
            lai = None

    predictor = lai if index == "lai" and lai is not None else ndvi
    valid &= np.isfinite(predictor)
    yields = estimate_yield(predictor, crop, model_coefficients, yield_function)
    valid &= np.isfinite(yields) & (yields > 0)

    if not np.any(valid):
        raise ValueError(f"No valid pixels found for scene {scene.scene_id}.")

    pixel_area = _pixel_area_ha(transform, crs)
    values = predictor[valid]
    yield_values = yields[valid]
    area = float(np.count_nonzero(valid) * pixel_area)
    total_yield = float(np.sum(yield_values * pixel_area))
    hist_counts, hist_edges = np.histogram(yield_values, bins=6)
    histogram = []
    for i, count in enumerate(hist_counts):
        histogram.append(
            {
                "min": float(hist_edges[i]),
                "max": float(hist_edges[i + 1]),
                "area_ha": float(count * pixel_area),
                "percent": float(count / np.count_nonzero(valid) * 100.0),
            }
        )

    grid = _grid_features(ndvi, yields, lai, valid, transform, crs) if include_grid else None
    mean_yield = float(np.mean(yield_values))
    output_files = {}
    if export_tif:
        token = uuid.uuid4().hex[:8]
        safe_prefix = re.sub(r"[^A-Za-z0-9._-]+", "_", output_prefix or f"{crop}_{scene.acquired}_{token}").strip("_")
        ndvi_out = np.where(valid, ndvi, np.nan)
        output_files["ndvi"] = _write_output_tif(ndvi_out, transform, crs, safe_prefix, "ndvi")
        if lai is not None:
            output_files["lai"] = _write_output_tif(np.where(valid, lai, np.nan), transform, crs, safe_prefix, "lai")
        output_files["yield"] = _write_output_tif(np.where(valid, yields, np.nan), transform, crs, safe_prefix, "yield_kg_ha")
    return {
        "scene": scene_metadata(scene),
        "target_resolution_m": target_resolution_m,
        "area_ha": area,
        "valid_pixel_count": int(np.count_nonzero(valid)),
        "ndvi_mean": float(np.nanmean(ndvi[valid])),
        "ndvi_median": float(np.nanmedian(ndvi[valid])),
        "lai_mean": float(np.nanmean(lai[valid])) if lai is not None else None,
        "index_used": index if index == "ndvi" or lai is not None else "ndvi",
        "mean_yield_kg_ha": mean_yield,
        "median_yield_kg_ha": float(np.median(yield_values)),
        "std_yield_kg_ha": float(np.std(yield_values)),
        "total_yield_kg": total_yield,
        "histogram": histogram,
        "uncertainty": uncertainty(crop, mean_yield),
        "crop_mask_source": used_crop_source,
        "crop_mask_path": crop_path,
        "grid": grid,
        "output_files": output_files,
    }


def estimate(
    fc: Dict[str, Any],
    crop: str,
    target_date: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    satellite_source: str,
    local_image_dir: Optional[str],
    local_image_files: Optional[List[str]],
    uav_image_dir: Optional[str],
    uav_image_files: Optional[List[str]],
    aws_cloud_cover_max: Optional[float],
    index: str,
    crop_mask_source: str,
    crop_distribution_tif_url: Optional[str],
    crop_distribution_vector_path: Optional[str],
    crop_distribution_vector_url: Optional[str],
    crop_type_field: str,
    target_resolution_m: float,
    use_cloud_mask: bool,
    include_grid: bool,
    export_tif: bool = False,
    output_prefix: Optional[str] = None,
    model_coefficients=None,
    yield_function: str = "default",
    lai_k=None,
    lai_m=None,
) -> Dict[str, Any]:
    if satellite_source == "uav":
        local_scenes = scan_tif_scenes(uav_image_dir, uav_image_files, "uav")
        source_used = "uav"
    elif satellite_source == "local_files":
        local_scenes = scan_tif_scenes(local_image_dir, local_image_files, "local_files")
        source_used = "local_files"
    else:
        local_scenes = scan_local_scenes() if satellite_source in {"auto", "local"} else []
        if satellite_source == "local" and (local_image_dir or local_image_files):
            local_scenes.extend(scan_tif_scenes(local_image_dir, local_image_files, "local_files"))
        source_used = "local"
    scenes = _select_target_scenes(local_scenes, target_date, start_date, end_date)
    if not scenes and satellite_source in {"auto", "aws"}:
        aws = query_aws_scenes(fc, start_date, end_date, max_cloud_cover=aws_cloud_cover_max)
        scenes = _select_target_scenes(aws, target_date, start_date, end_date)
        source_used = "aws"
    if not scenes:
        raise ValueError("No Sentinel-2 L2A scenes found for the requested date range.")

    scene_results = []
    errors = []
    for scene in scenes:
        try:
            scene_results.append(
                compute_scene(
                    scene,
                    fc,
                    crop,
                    index,
                    crop_mask_source,
                    crop_distribution_tif_url,
                    crop_distribution_vector_path,
                    crop_distribution_vector_url,
                    crop_type_field,
                    target_resolution_m,
                    use_cloud_mask,
                    include_grid,
                    export_tif,
                    output_prefix,
                    model_coefficients,
                    yield_function,
                    lai_k,
                    lai_m,
                )
            )
        except Exception as exc:
            errors.append({"scene_id": scene.scene_id, "error": str(exc)})
    if not scene_results:
        raise ValueError(f"All candidate scenes failed: {errors}")

    total_area = sum(item["area_ha"] for item in scene_results)
    total_yield = sum(item["total_yield_kg"] for item in scene_results)
    mean_yield = total_yield / total_area if total_area else 0.0
    grids = [item["grid"] for item in scene_results if item.get("grid")]
    grid_features = []
    for grid in grids:
        grid_features.extend(grid["features"])
    histogram = _merge_histograms(scene_results)
    output_files = {}
    for item in scene_results:
        for key, value in item.get("output_files", {}).items():
            output_files.setdefault(key, []).append(value)
    return {
        "crop": crop,
        "source_used": source_used,
        "target_resolution_m": target_resolution_m,
        "aws_cloud_cover_max": aws_cloud_cover_max,
        "yield_function": yield_function,
        "scene_count": len(scene_results),
        "scenes": [item["scene"] for item in scene_results],
        "summary": {
            "total_cropland_area_ha": total_area,
            "estimated_area_ha": total_area,
            "total_yield_kg": total_yield,
            "average_yield_kg_ha": mean_yield,
            "median_yield_kg_ha": float(np.mean([item["median_yield_kg_ha"] for item in scene_results])),
            "std_yield_kg_ha": float(np.mean([item["std_yield_kg_ha"] for item in scene_results])),
            "ndvi_mean": float(np.mean([item["ndvi_mean"] for item in scene_results])),
            "lai_mean": _mean_optional([item["lai_mean"] for item in scene_results]),
            "uncertainty": uncertainty(crop, mean_yield),
        },
        "histogram": histogram,
        "grid": {"type": "FeatureCollection", "features": grid_features},
        "output_files": output_files,
        "details": scene_results,
        "errors": errors,
    }


def _mean_optional(values):
    clean = [v for v in values if v is not None]
    return float(np.mean(clean)) if clean else None


def _merge_histograms(results: List[Dict[str, Any]]):
    all_bins = []
    for result in results:
        all_bins.extend(result["histogram"])
    if not all_bins:
        return []
    min_v = min(item["min"] for item in all_bins)
    max_v = max(item["max"] for item in all_bins)
    if min_v == max_v:
        max_v += 1
    edges = np.linspace(min_v, max_v, 7)
    areas = np.zeros(6)
    for item in all_bins:
        mid = (item["min"] + item["max"]) / 2.0
        idx = min(5, max(0, int(np.searchsorted(edges, mid, side="right") - 1)))
        areas[idx] += item["area_ha"]
    total = float(np.sum(areas))
    return [
        {
            "min": float(edges[i]),
            "max": float(edges[i + 1]),
            "area_ha": float(areas[i]),
            "percent": float(areas[i] / total * 100.0) if total else 0.0,
        }
        for i in range(6)
    ]


def time_series(
    fc: Dict[str, Any],
    crop: str,
    start_date: Optional[str],
    end_date: Optional[str],
    satellite_source: str = "local",
    local_image_dir: Optional[str] = None,
    local_image_files: Optional[List[str]] = None,
    uav_image_dir: Optional[str] = None,
    uav_image_files: Optional[List[str]] = None,
    crop_mask_source: str = "local",
    crop_distribution_tif_url: Optional[str] = None,
    crop_distribution_vector_path: Optional[str] = None,
    crop_distribution_vector_url: Optional[str] = None,
    crop_type_field: str = "crop_type",
    target_resolution_m: float = 10,
    use_cloud_mask: bool = True,
    lai_k=None,
    lai_m=None,
) -> List[Dict[str, Any]]:
    if satellite_source == "uav":
        candidate_scenes = scan_tif_scenes(uav_image_dir, uav_image_files, "uav")
    elif satellite_source == "local_files":
        candidate_scenes = scan_tif_scenes(local_image_dir, local_image_files, "local_files")
    else:
        candidate_scenes = scan_local_scenes()
        if satellite_source == "local" and (local_image_dir or local_image_files):
            candidate_scenes.extend(scan_tif_scenes(local_image_dir, local_image_files, "local_files"))
    series = []
    for scene in _date_filter(candidate_scenes, start_date, end_date):
        try:
            result = compute_scene(
                scene,
                fc,
                crop,
                "ndvi",
                crop_mask_source,
                crop_distribution_tif_url,
                crop_distribution_vector_path,
                crop_distribution_vector_url,
                crop_type_field,
                target_resolution_m,
                use_cloud_mask,
                False,
                False,
                None,
                None,
                "default",
                lai_k,
                lai_m,
            )
            series.append(
                {
                    "date": scene.acquired.isoformat(),
                    "scene_id": scene.scene_id,
                    "ndvi": result["ndvi_mean"],
                    "ndvi_mean": result["ndvi_mean"],
                    "lai": result["lai_mean"],
                    "lai_mean": result["lai_mean"],
                    "yield_kg_ha": result["mean_yield_kg_ha"],
                }
            )
        except Exception:
            continue
    return sorted(series, key=lambda item: item["date"])
