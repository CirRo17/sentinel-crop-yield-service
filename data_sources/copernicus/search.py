"""通过 Copernicus Data Space STAC API 检索 Sentinel-2 场景。

搜索逻辑与 aws_element84 模块一致，使用 Copernicus STAC 端点。
输出格式兼容的 manifest JSON，供 download/extract 步骤和
build_features.py 消费。
"""

from __future__ import annotations

import argparse
import json
from calendar import monthrange
from pathlib import Path
from typing import Any

import requests
from shapely.geometry import shape

from data_sources.copernicus.config import (
    COPERNICUS_STAC_URL,
    DEFAULT_LIMIT,
    DEFAULT_MAX_CLOUD,
    SENTINEL_COLLECTION,
)
from data_sources.copernicus.auth import get_access_token


# 搜索需要的波段资产名（同 aws_element84 的 BAND_ASSETS）
BAND_ASSETS = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "rededge1": "rededge1",
    "nir": "nir",
    "swir16": "swir16",
    "swir22": "swir22",
    "scl": "scl",
}
S2_ASSETS = list(BAND_ASSETS.values())


def _bbox_from_geometry(geometry: dict[str, Any]) -> list[float]:
    bounds = shape(geometry).bounds
    return [bounds[0], bounds[1], bounds[2], bounds[3]]


def search_scenes(
    geometry: dict[str, Any],
    start_date: str,
    end_date: str,
    collection: str = SENTINEL_COLLECTION,
    max_cloud: float = DEFAULT_MAX_CLOUD,
    limit: int = DEFAULT_LIMIT,
    access_token: str | None = None,
) -> list[dict[str, Any]]:
    """搜索 Copernicus STAC，返回 features 列表。"""
    payload: dict[str, Any] = {
        "collections": [collection],
        "bbox": _bbox_from_geometry(geometry),
        "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
        "limit": limit,
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [
            {"field": "properties.eo:cloud_cover", "direction": "asc"},
            {"field": "properties.datetime", "direction": "asc"},
        ],
    }

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    response = requests.post(
        COPERNICUS_STAC_URL + "/search",
        json=payload,
        headers=headers,
        timeout=60,
    )
    response.raise_for_status()
    return response.json().get("features", [])


def summarize_feature(feature: dict[str, Any]) -> dict[str, Any]:
    """从 STAC feature 提取场景摘要（兼容 aws_element84 manifest 格式）。"""
    assets = {}
    for name, asset_name in BAND_ASSETS.items():
        stac_assets = feature.get("assets", {})
        href = stac_assets.get(asset_name, {}).get("href")
        if href:
            assets[name] = href

    props = feature.get("properties", {})
    return {
        "id": feature.get("id", ""),
        "datetime": props.get("datetime"),
        "cloud_cover": props.get("eo:cloud_cover"),
        "assets": assets,
    }


def build_manifest(
    geometry: dict[str, Any],
    geometry_path: Path,
    start_date: str,
    end_date: str,
    target_months: list[int] | None = None,
    collection: str = SENTINEL_COLLECTION,
    max_cloud: float = DEFAULT_MAX_CLOUD,
    limit: int = DEFAULT_LIMIT,
    access_token: str | None = None,
) -> dict[str, Any]:
    """构建与 aws_element84 完全兼容的 manifest。

    按 target_months 分组，每个月作为一个时相（timepoint）。
    如果未指定 target_months，则将整个日期范围作为一个时相。
    """
    if target_months is None:
        # 不做月度划分，整个范围作为一个时相
        features = search_scenes(
            geometry, start_date, end_date, collection, max_cloud, limit, access_token
        )
        scenes = [summarize_feature(f) for f in features]
        timepoints = [
            {
                "label": f"{start_date[:7]}_{end_date[:7]}",
                "start_date": start_date,
                "end_date": end_date,
                "composite": "median",
                "scene_count": len(scenes),
                "scenes": scenes,
            }
        ]
    else:
        all_scenes: list[dict[str, Any]] = []
        timepoints = []
        for month in target_months:
            year = int(start_date[:4])
            last_day = monthrange(year, month)[1]
            window_start = f"{year}-{month:02d}-01"
            window_end = f"{year}-{month:02d}-{last_day:02d}"

            features = search_scenes(
                geometry, window_start, window_end, collection, max_cloud, limit, access_token
            )
            window_scenes = [summarize_feature(f) for f in features]
            all_scenes.extend(window_scenes)
            timepoints.append(
                {
                    "label": f"{year}-{month:02d}",
                    "start_date": window_start,
                    "end_date": window_end,
                    "composite": "monthly_median",
                    "scene_count": len(window_scenes),
                    "scenes": window_scenes,
                }
            )

    return {
        "source": "copernicus",
        "collection": collection,
        "geometry": str(geometry_path),
        "start_date": start_date,
        "end_date": end_date,
        "timepoints": timepoints,
    }


def write_manifest(manifest: dict[str, Any], output_path: Path) -> None:
    """写出 manifest JSON 文件。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="从 Copernicus Data Space 检索 Sentinel-2 场景清单。"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML 配置文件，可从中读取 geometry 和时间参数。",
    )
    parser.add_argument(
        "--geometry",
        type=Path,
        default=None,
        help="AOI 文件路径。未指定时从配置 project.geometry 读取。",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help="起始日期 YYYY-MM-DD。未指定时从配置 season.start_date 读取。",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="结束日期 YYYY-MM-DD。未指定时从配置 season.end_date 读取。",
    )
    parser.add_argument(
        "--target-months",
        type=int,
        nargs="*",
        default=None,
        help="按月分组的月份列表。未指定时从配置 season.target_months 读取。",
    )
    parser.add_argument(
        "--max-cloud",
        type=float,
        default=None,
        help=f"最大云量百分比，默认读取配置或 {DEFAULT_MAX_CLOUD}。",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"每月最大场景数，默认读取配置或 {DEFAULT_LIMIT}。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/exported/shared/manifests/copernicus_s2_scenes.json"),
        help="输出 manifest JSON 路径。",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_params(args) -> tuple[Path, str, str, list[int] | None, float, int]:
    """从 CLI 参数和可选的 YAML 配置中解析最终参数。"""
    config = {}
    if args.config and args.config.exists():
        config = _load_yaml(args.config)

    project = config.get("project", {})
    season = config.get("season", {})
    sentinel2 = config.get("sentinel2", {})

    geometry = args.geometry or Path(str(project.get("geometry", "")))
    start_date = args.start_date or str(season.get("start_date", ""))
    end_date = args.end_date or str(season.get("end_date", ""))
    target_months = args.target_months or season.get("target_months")
    max_cloud = args.max_cloud if args.max_cloud is not None else float(
        sentinel2.get("max_cloud", DEFAULT_MAX_CLOUD)
    )
    limit = args.limit if args.limit is not None else int(
        config.get("project", {}).get("scene_limit", DEFAULT_LIMIT)
    )

    if not geometry or not geometry.exists():
        raise FileNotFoundError(
            f"AOI 文件不存在：{geometry}。请通过 --geometry 或配置 project.geometry 指定。"
        )
    if not start_date or not end_date:
        raise ValueError(
            "缺少时间范围。请通过 --start-date/--end-date 或配置 season 指定。"
        )

    return geometry, start_date, end_date, target_months, max_cloud, limit


def main() -> None:
    args = parse_args()
    geometry_path, start_date, end_date, target_months, max_cloud, limit = _resolve_params(args)

    # 加载 AOI geometry（支持 GeoJSON 和 Shapefile）
    suffix = geometry_path.suffix.lower()
    if suffix in {".json", ".geojson"}:
        with open(geometry_path, encoding="utf-8-sig") as f:
            geojson = json.load(f)
        if geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
            if not features:
                raise ValueError("AOI 中没有要素。")
            geometry = features[0]["geometry"]
        elif geojson.get("type") == "Feature":
            geometry = geojson["geometry"]
        else:
            geometry = geojson
    else:
        import geopandas as gpd
        gdf = gpd.read_file(geometry_path)
        if gdf.empty:
            raise ValueError(f"AOI 中没有要素：{geometry_path}")
        if gdf.crs and str(gdf.crs).upper() != "EPSG:4326":
            gdf = gdf.to_crs("EPSG:4326")
        geometry = gdf.geometry.unary_union.__geo_interface__

    # 搜索不需要 token，但如果有会用来加速
    try:
        token = get_access_token()
    except Exception:
        token = None
        print("警告：未配置 Copernicus 凭证，使用无认证搜索（可能受限）。")

    manifest = build_manifest(
        geometry=geometry,
        geometry_path=geometry_path,
        start_date=start_date,
        end_date=end_date,
        target_months=target_months,
        max_cloud=max_cloud,
        limit=limit,
        access_token=token,
    )

    write_manifest(manifest, args.output)
    timepoint_count = len(manifest["timepoints"])
    scene_count = sum(tp["scene_count"] for tp in manifest["timepoints"])
    print(
        f"已保存 {timepoint_count} 个时相、{scene_count} 个场景到 {args.output}"
    )


if __name__ == "__main__":
    main()
