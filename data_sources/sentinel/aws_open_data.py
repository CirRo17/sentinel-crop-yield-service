"""访问 Sentinel AWS Open Data / Earth Search STAC 的辅助工具。"""

from __future__ import annotations

import json
from calendar import monthrange
from pathlib import Path
from typing import Any

import requests
from shapely.geometry import box, mapping, shape

from data_sources.common.config import EARTH_SEARCH_URL


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_GEOMETRY_PATH = Path("data/input/aoi_caobuhu.geojson")
DEFAULT_EXPORTED_DIR = Path("data/exported")

DEFAULT_CONFIG = {
    "project": {
        "name": "CropClassifier",
        "target_resolution_m": 10,
        "target_crs": "EPSG:4326",
        "scene_limit": 30,
    },
    "season": {
        "year": 2025,
        "start_date": "2025-04-01",
        "end_date": "2025-09-30",
        "composite": "monthly_median",
        "target_months": [4, 7, 9],
        "recommended_timepoints": "3-5",
    },
    "sentinel2": {
        "enabled": True,
        "collection": "sentinel-2-l2a",
        "max_cloud": 30,
    },
    "sentinel1": {
        "enabled": True,
        "collection": "sentinel-1-grd",
        "required_polarizations": ["VV", "VH"],
        "instrument_mode": "IW",
    },
}


def load_yaml_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return DEFAULT_CONFIG

    with open(path, encoding="utf-8") as f:
        loaded = yaml.safe_load(f) or {}

    config = DEFAULT_CONFIG.copy()
    for key, value in loaded.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    return config


def load_geojson_geometry(path: Path = DEFAULT_GEOMETRY_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        geojson = json.load(f)

    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features", [])
        if not features:
            raise ValueError(f"{path} 中没有找到任何要素")
        return features[0]["geometry"]

    if geojson.get("type") == "Feature":
        return geojson["geometry"]

    return geojson



def simplify_geometry_for_stac(geometry: dict[str, Any], tolerance: float = 0.0002, max_chars: int = 50000) -> dict[str, Any]:
    """Reduce AOI payload size for STAC intersects queries.

    The full AOI is still used as the project boundary. This simplified geometry
    is only for scene discovery, where a very detailed polygon can exceed STAC
    request size limits.
    """
    if len(json.dumps(geometry)) <= max_chars:
        return geometry

    geom = shape(geometry)
    simplified = geom.simplify(tolerance, preserve_topology=True)
    simplified_geojson = mapping(simplified)
    if len(json.dumps(simplified_geojson)) <= max_chars:
        return simplified_geojson

    minx, miny, maxx, maxy = geom.bounds
    return mapping(box(minx, miny, maxx, maxy))

def bbox_from_geometry(geometry: dict[str, Any]) -> list[float]:
    bounds = shape(geometry).bounds
    return [bounds[0], bounds[1], bounds[2], bounds[3]]


def monthly_windows(season: dict[str, Any]) -> list[dict[str, Any]]:
    year = int(season.get("year", str(season["start_date"])[:4]))
    months = season.get("target_months") or []
    if not months:
        return [
            {
                "label": "season",
                "year": year,
                "month": None,
                "start_date": season["start_date"],
                "end_date": season["end_date"],
            }
        ]

    windows = []
    for month in months:
        month = int(month)
        last_day = monthrange(year, month)[1]
        windows.append(
            {
                "label": f"{year}-{month:02d}",
                "year": year,
                "month": month,
                "start_date": f"{year}-{month:02d}-01",
                "end_date": f"{year}-{month:02d}-{last_day:02d}",
            }
        )
    return windows


def search_earth(
    *,
    collection: str,
    geometry: dict[str, Any],
    start_date: str,
    end_date: str,
    limit: int,
    query: dict[str, Any] | None = None,
    sortby: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    payload: dict[str, Any] = {
        "collections": [collection],
        "intersects": simplify_geometry_for_stac(geometry),
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": limit,
    }
    if query:
        payload["query"] = query
    if sortby:
        payload["sortby"] = sortby

    response = requests.post(EARTH_SEARCH_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json().get("features", [])


def summarize_feature(feature: dict[str, Any], asset_names: list[str]) -> dict[str, Any]:
    assets = {}
    for name in asset_names:
        href = feature.get("assets", {}).get(name, {}).get("href")
        if href:
            assets[name] = href

    properties = feature.get("properties", {})
    return {
        "id": feature.get("id", ""),
        "collection": feature.get("collection", ""),
        "datetime": properties.get("datetime"),
        "properties": {
            "platform": properties.get("platform"),
            "cloud_cover": properties.get("eo:cloud_cover"),
            "orbit_state": properties.get("sat:orbit_state"),
            "instrument_mode": properties.get("sar:instrument_mode"),
            "polarizations": properties.get("sar:polarizations"),
        },
        "assets": assets,
    }


def write_manifest(
    *,
    output_path: Path,
    source: str,
    collection: str,
    geometry_path: Path,
    start_date: str,
    end_date: str,
    scenes: list[dict[str, Any]],
    timepoints: list[dict[str, Any]] | None = None,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": source,
        "provider": "AWS Open Data via Element84 Earth Search STAC",
        "collection": collection,
        "geometry": str(geometry_path),
        "spatial_filter": "STAC intersects AOI geometry",
        "start_date": start_date,
        "end_date": end_date,
        "scene_count": len(scenes),
        "scenes": scenes,
    }
    if timepoints is not None:
        manifest["timepoint_count"] = len(timepoints)
        manifest["timepoints"] = timepoints
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)




