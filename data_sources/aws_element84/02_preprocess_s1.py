"""检索 Sentinel-1 雷达影像场景清单。`r`n`r`n通过 Element84 Earth Search 查询公开 Sentinel-1 GRD 场景，保留包含 VV/VH`r`n极化的 IW 模式场景，并生成 manifest，供后续本地缓存和特征构建使用。`r`n"""

from __future__ import annotations

import argparse
from pathlib import Path

from data_sources.aws_element84.aws_open_data import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_EXPORTED_DIR,
    DEFAULT_GEOMETRY_PATH,
    load_geojson_geometry,
    load_yaml_config,
    monthly_windows,
    search_earth,
    summarize_feature,
    write_manifest,
)


S1_ASSETS = ["vv", "vh", "safe-manifest", "thumbnail"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 AWS Open Data 检索 Sentinel-1 场景清单。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPORTED_DIR / "sentinel1_scenes.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def has_vv_vh_iw(feature: dict) -> bool:
    properties = feature.get("properties", {})
    polarizations = set(properties.get("sar:polarizations", []))
    return properties.get("sar:instrument_mode") == "IW" and {"VV", "VH"}.issubset(polarizations)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    geometry = load_geojson_geometry(args.geometry)

    season = config["season"]
    sentinel1 = config.get("sentinel1", {})
    collection = sentinel1.get("collection", "sentinel-1-grd")
    limit = args.limit or int(config.get("project", {}).get("scene_limit", 30))

    timepoints = []
    scenes = []
    for window in monthly_windows(season):
        features = search_earth(
            collection=collection,
            geometry=geometry,
            start_date=window["start_date"],
            end_date=window["end_date"],
            limit=limit,
            sortby=[{"field": "properties.datetime", "direction": "asc"}],
        )
        features = [feature for feature in features if has_vv_vh_iw(feature)]
        window_scenes = [summarize_feature(feature, S1_ASSETS) for feature in features]
        scenes.extend(window_scenes)
        timepoints.append(
            {
                **window,
                "composite": season.get("composite", "monthly_median"),
                "scene_count": len(window_scenes),
                "scenes": window_scenes,
            }
        )

    write_manifest(
        output_path=args.output,
        source="sentinel1",
        collection=collection,
        geometry_path=args.geometry,
        start_date=season["start_date"],
        end_date=season["end_date"],
        scenes=scenes,
        timepoints=timepoints,
    )
    print(f"已保存 {len(timepoints)} 个 Sentinel-1 时相、{len(scenes)} 个场景到 {args.output}")


if __name__ == "__main__":
    main()

