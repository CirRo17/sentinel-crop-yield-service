"""检索 Sentinel-2 光学影像场景清单。`r`n`r`n通过 Element84 Earth Search 查询公开 Sentinel-2 L2A COG 场景，按配置的`r`n季节窗口生成场景 manifest，供后续本地缓存、特征构建和业务 pipeline 使用。`r`n"""

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


S2_ASSETS = ["blue", "green", "red", "rededge1", "nir", "nir08", "swir16", "swir22", "scl", "visual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 AWS Open Data 检索 Sentinel-2 场景清单。")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--geometry", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPORTED_DIR / "sentinel2_scenes.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _resolve_geometry(config: dict, args: argparse.Namespace) -> Path:
    """优先 CLI --geometry，其次配置 project.geometry，最后默认值。"""
    if args.geometry:
        return args.geometry
    project = config.get("project", {})
    if project.get("geometry"):
        return Path(str(project["geometry"]))
    return DEFAULT_GEOMETRY_PATH


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    geometry_path = _resolve_geometry(config, args)
    geometry = load_geojson_geometry(geometry_path)

    season = config["season"]
    sentinel2 = config["sentinel2"]
    collection = sentinel2.get("collection", "sentinel-2-l2a")
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
            query={"eo:cloud_cover": {"lt": sentinel2.get("max_cloud", 30)}},
            sortby=[
                {"field": "properties.eo:cloud_cover", "direction": "asc"},
                {"field": "properties.datetime", "direction": "asc"},
            ],
        )
        window_scenes = [summarize_feature(feature, S2_ASSETS) for feature in features]
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
        source="sentinel2",
        collection=collection,
        geometry_path=geometry_path,
        start_date=season["start_date"],
        end_date=season["end_date"],
        scenes=scenes,
        timepoints=timepoints,
    )
    print(f"已保存 {len(timepoints)} 个 Sentinel-2 时相、{len(scenes)} 个场景到 {args.output}")


if __name__ == "__main__":
    main()

