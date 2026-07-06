"""Step 01: fetch Sentinel-2 optical scene inputs from AWS Open Data.

This step is intentionally runnable today. It searches public Sentinel-2 L2A
COG scenes through Element84 Earth Search and writes a scene manifest under
``data/exported/``. Later this step can grow from "manifest creation" into
cloud masking and temporal compositing without changing downstream pipeline
steps.
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


S2_ASSETS = ["blue", "green", "red", "rededge1", "nir", "nir08", "swir16", "swir22", "scl", "visual"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch Sentinel-2 scene manifest from AWS Open Data.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_EXPORTED_DIR / "sentinel2_scenes.json")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    geometry = load_geojson_geometry(args.geometry)

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
        geometry_path=args.geometry,
        start_date=season["start_date"],
        end_date=season["end_date"],
        scenes=scenes,
        timepoints=timepoints,
    )
    print(f"Saved {len(timepoints)} Sentinel-2 timepoints and {len(scenes)} scenes to {args.output}")


if __name__ == "__main__":
    main()
