"""Step 1: prepare multi-year same-month growth baseline from Copernicus.

This script searches Copernicus Data Space for Sentinel-2 L2A scenes, downloads
the SAFE products through OData, and builds one baseline feature stack per
historical year/month. The output baseline manifest is consumed by
``02_pixel_zscore.py``.
"""

from __future__ import annotations

import argparse
import json
from calendar import monthrange
from pathlib import Path
from typing import Any

import geopandas as gpd

from data_sources.copernicus.auth import get_access_token
from data_sources.copernicus.build_features import build_features_from_manifest
from data_sources.copernicus.config import DEFAULT_LIMIT, DEFAULT_MAX_CLOUD, SENTINEL_COLLECTION
from data_sources.copernicus.download import download_from_manifest
from data_sources.copernicus.search import build_manifest, write_manifest


DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_GEOMETRY_PATH = Path("data/input/aoi/tuanlinpu_aoi.shp")
DEFAULT_CURRENT_METADATA = Path("data/exported/feature_stack/feature_stack_multiband_metadata.json")
DEFAULT_OUTPUT_DIR = Path("data/output/growth_monitoring/baseline")
DEFAULT_BASELINE_MANIFEST = DEFAULT_OUTPUT_DIR / "growth_monitoring_baseline_manifest.json"
DEFAULT_SOURCE_DIR = Path("data/source/copernicus")
FUNCTION_NAME = "growth_monitoring"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare multi-year same-month Sentinel-2 growth baseline using Copernicus."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--geometry",
        type=Path,
        default=None,
        help="AOI vector file. If omitted, uses current metadata aoi, then project.geometry, then default AOI.",
    )
    parser.add_argument(
        "--current-metadata",
        type=Path,
        default=DEFAULT_CURRENT_METADATA,
        help="Current feature stack metadata used to resolve AOI automatically.",
    )
    parser.add_argument("--target-month", type=int, required=True)
    parser.add_argument("--baseline-start-year", type=int, required=True)
    parser.add_argument("--baseline-end-year", type=int, required=True)
    parser.add_argument("--max-cloud", type=float, default=None)
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument(
        "--max-scenes-per-year",
        type=int,
        default=None,
        help="Limit scenes used per year after search. Default uses all searched scenes.",
    )
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--baseline-manifest", type=Path, default=DEFAULT_BASELINE_MANIFEST)
    parser.add_argument("--coverage-threshold", type=float, default=0.90)
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _metadata_aoi_path(metadata_path: Path) -> Path | None:
    if not metadata_path.exists():
        return None
    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    candidates = [
        metadata.get("aoi"),
        (metadata.get("aoi_mask") or {}).get("path"),
    ]
    for candidate in candidates:
        if candidate:
            path = Path(str(candidate))
            if path.exists():
                return path
    return None


def _resolve_geometry_path(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.geometry:
        return args.geometry
    metadata_aoi = _metadata_aoi_path(args.current_metadata)
    if metadata_aoi:
        return metadata_aoi
    configured = config.get("project", {}).get("geometry")
    if configured:
        return Path(str(configured))
    return DEFAULT_GEOMETRY_PATH


def _load_geometry(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"AOI file does not exist: {path}")

    if path.suffix.lower() in {".json", ".geojson"}:
        with open(path, encoding="utf-8-sig") as f:
            geojson = json.load(f)
        if geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
            if not features:
                raise ValueError(f"AOI contains no features: {path}")
            if len(features) == 1:
                return features[0]["geometry"]
            return {"type": "GeometryCollection", "geometries": [item["geometry"] for item in features]}
        if geojson.get("type") == "Feature":
            return geojson["geometry"]
        return geojson

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"AOI contains no features: {path}")
    if gdf.crs and str(gdf.crs).upper() != "EPSG:4326":
        gdf = gdf.to_crs("EPSG:4326")
    return gdf.geometry.unary_union.__geo_interface__


def _month_window(year: int, month: int) -> tuple[str, str, str]:
    last_day = monthrange(year, month)[1]
    label = f"{year}-{month:02d}"
    return label, f"{label}-01", f"{label}-{last_day:02d}"


def _limit_scenes(manifest: dict[str, Any], max_scenes: int | None) -> None:
    if max_scenes is None:
        return
    for timepoint in manifest.get("timepoints", []):
        timepoint["scenes"] = timepoint.get("scenes", [])[:max_scenes]
        timepoint["scene_count"] = len(timepoint["scenes"])


def main() -> None:
    args = parse_args()
    if args.baseline_start_year > args.baseline_end_year:
        raise ValueError("--baseline-start-year cannot be greater than --baseline-end-year.")
    if not 1 <= args.target_month <= 12:
        raise ValueError("--target-month must be in 1..12.")

    config = _load_yaml(args.config)
    sentinel2 = config.get("sentinel2", {})
    project = config.get("project", {})
    collection = str(sentinel2.get("collection") or SENTINEL_COLLECTION)
    max_cloud = float(args.max_cloud if args.max_cloud is not None else sentinel2.get("max_cloud", DEFAULT_MAX_CLOUD))
    scene_limit = int(args.scene_limit if args.scene_limit is not None else project.get("scene_limit", DEFAULT_LIMIT))

    geometry_path = _resolve_geometry_path(args, config)
    geometry = _load_geometry(geometry_path)
    token = get_access_token()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_items: list[dict[str, Any]] = []

    for year in range(args.baseline_start_year, args.baseline_end_year + 1):
        label, start_date, end_date = _month_window(year, args.target_month)
        try:
            print(f"Searching Copernicus Sentinel-2 scenes for {label}...", flush=True)

            manifest = build_manifest(
                geometry=geometry,
                geometry_path=geometry_path,
                start_date=start_date,
                end_date=end_date,
                target_months=[args.target_month],
                collection=collection,
                max_cloud=max_cloud,
                limit=scene_limit,
                access_token=token,
            )
            _limit_scenes(manifest, args.max_scenes_per_year)

            timepoint = manifest["timepoints"][0]
            if not timepoint.get("scenes"):
                print(f"  Warning: no usable scenes found for {label}; skipped.", flush=True)
                continue

            year_dir = args.output_dir / f"{year}_{args.target_month:02d}"
            manifest_path = year_dir / f"{FUNCTION_NAME}_baseline_copernicus_s2_scenes_{year}_{args.target_month:02d}.json"
            write_manifest(manifest, manifest_path)

            print(f"Downloading Copernicus SAFE products for {label}...", flush=True)
            updated_manifest = download_from_manifest(
                manifest_path=manifest_path,
                output_dir=args.source_dir,
                token=token,
                skip_existing=True,
                timepoints=[label],
            )
            write_manifest(updated_manifest, manifest_path)

            downloaded_scene_count = sum(
                1
                for tp in updated_manifest.get("timepoints", [])
                if tp.get("label") == label
                for scene in tp.get("scenes", [])
                if scene.get("_downloaded") and scene.get("_local_safe_path")
            )
            if downloaded_scene_count == 0:
                print(f"  Warning: no scenes downloaded for {label}; skipped.", flush=True)
                continue

            feature_stack = year_dir / f"{FUNCTION_NAME}_baseline_feature_stack_{year}_{args.target_month:02d}.tif"
            metadata = year_dir / f"{FUNCTION_NAME}_baseline_feature_stack_{year}_{args.target_month:02d}_metadata.json"
            build_features_from_manifest(
                manifest_path=manifest_path,
                output_path=feature_stack,
                metadata_path=metadata,
                timepoints=[label],
                aoi_geometry_path=geometry_path,
                coverage_threshold=args.coverage_threshold,
            )

            baseline_items.append(
                {
                    "year": year,
                    "month": args.target_month,
                    "label": label,
                    "scene_count": downloaded_scene_count,
                    "feature_stack": str(feature_stack),
                    "metadata": str(metadata),
                    "manifest": str(manifest_path),
                }
            )
        except Exception as exc:
            print(f"  Warning: failed to build baseline for {label}; skipped. {str(exc)[:300]}", flush=True)
            continue

    if not baseline_items:
        raise ValueError("No historical baseline feature stacks were built.")

    baseline_doc = {
        "source": f"{FUNCTION_NAME}_copernicus_auto_baseline",
        "function": FUNCTION_NAME,
        "geometry": str(geometry_path),
        "target_month": args.target_month,
        "baseline_start_year": args.baseline_start_year,
        "baseline_end_year": args.baseline_end_year,
        "collection": collection,
        "max_cloud": max_cloud,
        "scene_limit": scene_limit,
        "items": baseline_items,
    }
    args.baseline_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.baseline_manifest, "w", encoding="utf-8") as f:
        json.dump(baseline_doc, f, indent=2, ensure_ascii=False)

    print("=== Growth baseline preparation completed ===", flush=True)
    print(f"Baseline years built: {len(baseline_items)}", flush=True)
    print(f"Baseline manifest: {args.baseline_manifest}", flush=True)


if __name__ == "__main__":
    main()
