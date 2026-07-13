"""Step 1: prepare pest detection inputs from Copernicus Sentinel-2.

The script uses the current feature stack produced by the crop-classification
data flow, then prepares:
- previous period feature stack;
- pre-previous period feature stack;
- historical baseline stacks for the current period;
- historical baseline stacks for the previous period.

The output manifest is consumed by ``02_pixel_stress_score.py``.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import geopandas as gpd

from data_sources.copernicus.auth import get_access_token
from data_sources.copernicus.build_features import build_features_from_manifest
from data_sources.copernicus.config import DEFAULT_LIMIT, DEFAULT_MAX_CLOUD, SENTINEL_COLLECTION
from data_sources.copernicus.download import download_from_manifest
from data_sources.copernicus.search import search_scenes, summarize_feature, write_manifest


FUNCTION_NAME = "pest_detect"
DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DEFAULT_GEOMETRY_PATH = Path("data/input/aoi/tuanlinpu_aoi.shp")
DEFAULT_CURRENT_METADATA = Path("data/exported/feature_stack/feature_stack_multiband_metadata.json")
DEFAULT_CURRENT_FEATURE_STACK = Path("data/exported/feature_stack/feature_stack_multiband.tif")
DEFAULT_OUTPUT_DIR = Path("data/output/pest_detect/inputs")
DEFAULT_INPUTS_MANIFEST = DEFAULT_OUTPUT_DIR / "pest_detect_inputs_manifest.json"
DEFAULT_SOURCE_DIR = Path("data/source/copernicus")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare pest detection input feature stacks using Copernicus.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--geometry", type=Path, default=None)
    parser.add_argument("--current-feature-stack", type=Path, default=DEFAULT_CURRENT_FEATURE_STACK)
    parser.add_argument("--current-metadata", type=Path, default=DEFAULT_CURRENT_METADATA)
    parser.add_argument("--current-start", required=True, help="Current monitoring window start date, YYYY-MM-DD.")
    parser.add_argument("--current-end", required=True, help="Current monitoring window end date, YYYY-MM-DD.")
    parser.add_argument("--baseline-start-year", type=int, required=True)
    parser.add_argument("--baseline-end-year", type=int, required=True)
    parser.add_argument("--baseline-day-padding", type=int, default=5)
    parser.add_argument("--max-cloud", type=float, default=None)
    parser.add_argument("--scene-limit", type=int, default=None)
    parser.add_argument("--max-scenes-per-window", type=int, default=None)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--inputs-manifest", type=Path, default=DEFAULT_INPUTS_MANIFEST)
    parser.add_argument("--coverage-threshold", type=float, default=0.90)
    parser.add_argument(
        "--only-period",
        default=None,
        help="Prepare only one computed period label, e.g. baseline_current_2025_20250527_20250705.",
    )
    return parser.parse_args()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _date_text(value: date) -> str:
    return value.strftime("%Y-%m-%d")


def _safe_label(prefix: str, start: date, end: date) -> str:
    return f"{prefix}_{start:%Y%m%d}_{end:%Y%m%d}"


def _metadata_aoi_path(metadata_path: Path) -> Path | None:
    if not metadata_path.exists():
        return None
    with open(metadata_path, encoding="utf-8-sig") as f:
        metadata = json.load(f)
    for candidate in [metadata.get("aoi"), (metadata.get("aoi_mask") or {}).get("path")]:
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


def _period(start: date, end: date, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "start_date": _date_text(start),
        "end_date": _date_text(end),
        "composite": "median",
    }


def _manifest_for_period(
    *,
    period: dict[str, Any],
    geometry: dict[str, Any],
    geometry_path: Path,
    collection: str,
    max_cloud: float,
    scene_limit: int,
    token: str,
    max_scenes: int | None,
) -> dict[str, Any]:
    features = search_scenes(
        geometry=geometry,
        start_date=period["start_date"],
        end_date=period["end_date"],
        collection=collection,
        max_cloud=max_cloud,
        limit=scene_limit,
        access_token=token,
    )
    scenes = [summarize_feature(feature) for feature in features]
    if max_scenes is not None:
        scenes = scenes[:max_scenes]
    return {
        "source": "copernicus",
        "collection": collection,
        "geometry": str(geometry_path),
        "start_date": period["start_date"],
        "end_date": period["end_date"],
        "timepoints": [
            {
                **period,
                "scene_count": len(scenes),
                "scenes": scenes,
            }
        ],
    }


def _prepare_period_stack(
    *,
    period: dict[str, Any],
    collection: str,
    geometry_path: Path,
    geometry: dict[str, Any],
    max_cloud: float,
    scene_limit: int,
    output_dir: Path,
    source_dir: Path,
    token: str,
    max_scenes: int | None,
    coverage_threshold: float,
) -> dict[str, Any]:
    period_dir = output_dir / period["label"]
    manifest_path = period_dir / f"{FUNCTION_NAME}_copernicus_s2_scenes_{period['label']}.json"
    feature_stack = period_dir / f"{FUNCTION_NAME}_feature_stack_{period['label']}.tif"
    metadata = period_dir / f"{FUNCTION_NAME}_feature_stack_{period['label']}_metadata.json"

    if feature_stack.exists() and metadata.exists():
        scene_count = 0
        if manifest_path.exists():
            try:
                with open(manifest_path, encoding="utf-8-sig") as f:
                    existing_manifest = json.load(f)
                scene_count = sum(
                    1
                    for tp in existing_manifest.get("timepoints", [])
                    for scene in tp.get("scenes", [])
                    if scene.get("_downloaded") and scene.get("_local_safe_path")
                )
            except Exception:
                scene_count = 0
        print(f"  Reusing existing pest input stack: {feature_stack}", flush=True)
        return {
            **period,
            "scene_count": scene_count,
            "feature_stack": str(feature_stack),
            "metadata": str(metadata),
            "manifest": str(manifest_path),
        }

    manifest = _manifest_for_period(
        period=period,
        geometry=geometry,
        geometry_path=geometry_path,
        collection=collection,
        max_cloud=max_cloud,
        scene_limit=scene_limit,
        token=token,
        max_scenes=max_scenes,
    )
    timepoint = manifest["timepoints"][0]
    if not timepoint.get("scenes"):
        raise ValueError(f"No usable Copernicus Sentinel-2 scenes found for {period['label']}.")
    write_manifest(manifest, manifest_path)

    updated_manifest = download_from_manifest(
        manifest_path=manifest_path,
        output_dir=source_dir,
        token=token,
        skip_existing=True,
        timepoints=[period["label"]],
    )
    write_manifest(updated_manifest, manifest_path)

    downloaded_scene_count = sum(
        1
        for tp in updated_manifest.get("timepoints", [])
        if tp.get("label") == period["label"]
        for scene in tp.get("scenes", [])
        if scene.get("_downloaded") and scene.get("_local_safe_path")
    )
    if downloaded_scene_count == 0:
        raise RuntimeError(f"No scenes downloaded for {period['label']}.")

    build_features_from_manifest(
        manifest_path=manifest_path,
        output_path=feature_stack,
        metadata_path=metadata,
        timepoints=[period["label"]],
        aoi_geometry_path=geometry_path,
        coverage_threshold=coverage_threshold,
    )
    return {
        **period,
        "scene_count": downloaded_scene_count,
        "feature_stack": str(feature_stack),
        "metadata": str(metadata),
        "manifest": str(manifest_path),
    }


def _baseline_periods(
    source_start: date,
    source_end: date,
    start_year: int,
    end_year: int,
    padding: int,
    prefix: str,
) -> list[dict[str, Any]]:
    periods = []
    for year in range(start_year, end_year + 1):
        start = date(year, source_start.month, source_start.day) - timedelta(days=padding)
        end = date(year, source_end.month, source_end.day) + timedelta(days=padding)
        periods.append(_period(start, end, _safe_label(f"{prefix}_{year}", start, end)))
    return periods


def _required_item(label: str, func) -> dict[str, Any]:
    try:
        return func()
    except Exception as exc:
        raise RuntimeError(f"Failed to prepare required pest input {label}: {exc}") from exc


def _optional_items(periods: list[dict[str, Any]], func) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for period in periods:
        try:
            items.append(func(period))
        except Exception as exc:
            print(f"  Warning: failed to prepare {period['label']}; skipped. {str(exc)[:300]}", flush=True)
    return items


def main() -> None:
    args = parse_args()
    if args.baseline_start_year > args.baseline_end_year:
        raise ValueError("--baseline-start-year cannot be greater than --baseline-end-year.")
    if not args.current_feature_stack.exists():
        raise FileNotFoundError(f"Current feature stack does not exist: {args.current_feature_stack}")

    current_start = _parse_date(args.current_start)
    current_end = _parse_date(args.current_end)
    if current_start >= current_end:
        raise ValueError("--current-start must be earlier than --current-end.")

    window_days = (current_end - current_start).days
    previous_end = current_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=window_days)
    pre_previous_end = previous_start - timedelta(days=1)
    pre_previous_start = pre_previous_end - timedelta(days=window_days)

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

    def prepare(period: dict[str, Any]) -> dict[str, Any]:
        return _prepare_period_stack(
            period=period,
            collection=collection,
            geometry_path=geometry_path,
            geometry=geometry,
            max_cloud=max_cloud,
            scene_limit=scene_limit,
            output_dir=args.output_dir,
            source_dir=args.source_dir,
            token=token,
            max_scenes=args.max_scenes_per_window,
            coverage_threshold=args.coverage_threshold,
        )

    previous_period = _period(previous_start, previous_end, _safe_label("previous", previous_start, previous_end))
    pre_previous_period = _period(pre_previous_start, pre_previous_end, _safe_label("pre_previous", pre_previous_start, pre_previous_end))

    print("Preparing previous period Sentinel-2 stack...", flush=True)
    previous_item = _required_item("previous", lambda: prepare(previous_period))

    print("Preparing pre-previous period Sentinel-2 stack...", flush=True)
    pre_previous_item = _required_item("pre_previous", lambda: prepare(pre_previous_period))

    current_baseline_periods = _baseline_periods(
        current_start,
        current_end,
        args.baseline_start_year,
        args.baseline_end_year,
        args.baseline_day_padding,
        "baseline_current",
    )
    previous_baseline_periods = _baseline_periods(
        previous_start,
        previous_end,
        args.baseline_start_year,
        args.baseline_end_year,
        args.baseline_day_padding,
        "baseline_previous",
    )

    all_periods = {
        previous_period["label"]: previous_period,
        pre_previous_period["label"]: pre_previous_period,
        **{period["label"]: period for period in current_baseline_periods},
        **{period["label"]: period for period in previous_baseline_periods},
    }
    if args.only_period:
        period = all_periods.get(args.only_period)
        if period is None:
            raise ValueError(
                f"Unknown --only-period {args.only_period}. Available: {', '.join(sorted(all_periods))}"
            )
        item = prepare(period)
        print("=== Pest single-period input preparation completed ===", flush=True)
        print(json.dumps(item, indent=2, ensure_ascii=False), flush=True)
        return

    print("Preparing current-window historical baselines...", flush=True)
    current_baseline_items = _optional_items(current_baseline_periods, prepare)

    print("Preparing previous-window historical baselines...", flush=True)
    previous_baseline_items = _optional_items(previous_baseline_periods, prepare)

    if not current_baseline_items:
        raise RuntimeError("No current-window historical baseline stacks were prepared.")
    if not previous_baseline_items:
        raise RuntimeError("No previous-window historical baseline stacks were prepared.")

    manifest = {
        "source": f"{FUNCTION_NAME}_copernicus_inputs",
        "function": FUNCTION_NAME,
        "geometry": str(geometry_path),
        "current": {
            "start_date": _date_text(current_start),
            "end_date": _date_text(current_end),
            "feature_stack": str(args.current_feature_stack),
            "metadata": str(args.current_metadata),
        },
        "previous": previous_item,
        "pre_previous": pre_previous_item,
        "baseline": {
            "start_year": args.baseline_start_year,
            "end_year": args.baseline_end_year,
            "day_padding": args.baseline_day_padding,
            "current_window": current_baseline_items,
            "previous_window": previous_baseline_items,
        },
        "parameters": {
            "collection": collection,
            "max_cloud": max_cloud,
            "scene_limit": scene_limit,
            "max_scenes_per_window": args.max_scenes_per_window,
        },
    }
    args.inputs_manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.inputs_manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("=== Pest input preparation completed ===", flush=True)
    print(f"Inputs manifest: {args.inputs_manifest}", flush=True)


if __name__ == "__main__":
    main()
