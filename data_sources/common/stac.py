"""通过 Element84 Earth Search STAC API 检索 Sentinel-2 场景。

本模块把输入 GeoJSON geometry 转换为边界框，发送 STAC 查询请求，
并提取供特征计算和分类使用的场景摘要。
"""

from __future__ import annotations

from typing import Any

import requests
from shapely.geometry import shape

from data_sources.common.config import EARTH_SEARCH_URL
from data_sources.sentinel.config import BAND_ASSETS, SENTINEL_COLLECTION
from data_sources.common.schemas import SceneRequest, SceneSummary


def _bbox_from_geojson(geometry: dict[str, Any]) -> list[float]:
    bounds = shape(geometry).bounds
    return [bounds[0], bounds[1], bounds[2], bounds[3]]


def search_sentinel_scenes(request: SceneRequest) -> list[dict[str, Any]]:
    payload = {
        "collections": [SENTINEL_COLLECTION],
        "bbox": _bbox_from_geojson(request.geometry),
        "datetime": f"{request.start_date.isoformat()}T00:00:00Z/{request.end_date.isoformat()}T23:59:59Z",
        "limit": request.limit,
        "query": {
            "eo:cloud_cover": {"lt": request.max_cloud},
        },
        "sortby": [
            {"field": "properties.eo:cloud_cover", "direction": "asc"},
            {"field": "properties.datetime", "direction": "asc"},
        ],
    }

    response = requests.post(EARTH_SEARCH_URL, json=payload, timeout=60)
    response.raise_for_status()
    return response.json().get("features", [])


def summarize_scene(feature: dict[str, Any]) -> SceneSummary:
    assets = {}
    for name, asset_name in BAND_ASSETS.items():
        href = feature.get("assets", {}).get(asset_name, {}).get("href")
        if href:
            assets[name] = href

    return SceneSummary(
        id=feature.get("id", ""),
        datetime=feature.get("properties", {}).get("datetime"),
        cloud_cover=feature.get("properties", {}).get("eo:cloud_cover"),
        assets=assets,
    )
