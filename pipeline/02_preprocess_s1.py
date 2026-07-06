"""Step 02: fetch Sentinel-1 radar scene inputs from AWS Open Data.

This step searches public Sentinel-1 GRD scenes through Element84 Earth Search,
keeps IW scenes that contain VV and VH polarizations, and writes a scene
manifest under ``data/exported/``. Later this can grow into terrain correction,
speckle filtering, and temporal compositing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.aws_open_data import (
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
    parser = argparse.ArgumentParser(description="Fetch Sentinel-1 scene manifest from AWS Open Data.")
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
    print(f"Saved {len(timepoints)} Sentinel-1 timepoints and {len(scenes)} scenes to {args.output}")


if __name__ == "__main__":
    main()
