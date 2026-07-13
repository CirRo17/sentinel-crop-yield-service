"""Step 15: rule-based harvest-window recommendation.

This v1 pipeline combines Sentinel-2 quality/timing metadata, Sentinel-1
coverage metadata, historical weather aggregates, and either forecast records
or a forecast-like historical fallback to recommend a regional harvest window.

Usage:
    python pipeline/15_harvest_window.py
    python pipeline/15_harvest_window.py --config configs/harvest_window.yaml
    python pipeline/15_harvest_window.py --forecast-csv data/weather/forecast.csv

Forecast CSV columns:
    date,t2m_c,precip_mm,swvl1,wind_speed_m_s
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import h5py
import geopandas as gpd
import numpy as np
import yaml


DEFAULT_CONFIG = Path("configs/harvest_window.yaml")


@dataclass(frozen=True)
class S2Summary:
    scene_count: int
    usable_count: int
    latest_usable_date: date | None
    first_date: date | None
    last_date: date | None
    usable_tags: tuple[str, ...]
    cloud_screening: str
    cloud_screening_note: str
    scenes: list[dict[str, Any]]
    indices_csv: str | None
    indices_status: str
    index_scene_count: int
    latest_index_date: date | None
    latest_ndvi: float | None
    latest_ndre: float | None
    latest_ndmi: float | None
    index_maturity_score: float
    index_confidence: float
    availability_confidence: float
    confidence: float


@dataclass(frozen=True)
class S1Summary:
    scene_count: int
    preferred_orbit: int
    preferred_count: int
    latest_preferred_date: date | None
    confidence: float


@dataclass(frozen=True)
class WeatherDaily:
    date: date
    t2m_c: float
    precip_mm: float
    swvl1: float
    wind_speed_m_s: float
    source: str


@dataclass(frozen=True)
class WeatherSummary:
    daily: list[WeatherDaily]
    months_present: list[str]
    months_missing: list[str]
    first_date: date | None
    last_date: date | None
    cumulative_gdd: float
    confidence: float


@dataclass(frozen=True)
class MaturitySummary:
    reference_date: date
    calendar_score: float
    s2_score: float
    s1_score: float
    gdd_score: float
    maturity_score: float
    confidence: float


@dataclass(frozen=True)
class ForecastDayScore:
    date: date
    t2m_c: float
    precip_mm: float
    swvl1: float
    wind_speed_m_s: float
    dry_streak_days: int
    weather_score: float
    harvest_score: float
    risk_flags: list[str]


@dataclass(frozen=True)
class ParcelSummary:
    parcels_path: str
    aoi_path: str | None
    crs: str
    aoi_crs: str | None
    total_features: int
    aoi_features: int
    target_crop_type: int
    target_crop_label: str
    target_parcels: int
    target_area_mu: float
    bounds: tuple[float, float, float, float] | None
    aoi_bounds: tuple[float, float, float, float] | None
    sample_output: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recommend a regional harvest window.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--forecast-csv", type=Path, help="Optional forecast CSV file.")
    parser.add_argument("--forecast-json", type=Path, help="Optional forecast JSON file.")
    parser.add_argument("--output-dir", type=Path, help="Override output directory.")
    parser.add_argument(
        "--skip-s2-index-refresh",
        action="store_true",
        help="Use the existing S2 indices CSV without running calc_s2_indices_csv.py.",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def clamp(value: float, min_value: float = 0.0, max_value: float = 1.0) -> float:
    return max(min_value, min(max_value, value))


def parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def xml_text(root: ET.Element, tag_name: str, default: str = "") -> str:
    for element in root.iter():
        if local_name(element.tag) == tag_name:
            return (element.text or "").strip()
    return default


def classify_s2_cloud_score(config: dict[str, Any], score: float) -> str:
    s2_cfg = config.get("s2", {})
    keep_threshold = float(s2_cfg.get("keep_threshold", 10.0))
    review_threshold = float(s2_cfg.get("review_threshold", 30.0))
    if score <= keep_threshold:
        return "keep"
    if score <= review_threshold:
        return "review"
    return "drop_or_fill"


def read_s2_cloud_geometries(root: Path, config: dict[str, Any]) -> gpd.GeoDataFrame | None:
    paths = config["paths"]
    parcels_value = paths.get("parcels")
    aoi_value = paths.get("aoi")
    crop_cfg = config["crop"]
    target_code = int(crop_cfg["crop_type_code"])

    parcels_path = resolve_path(root, parcels_value) if parcels_value else None
    aoi_path = resolve_path(root, aoi_value) if aoi_value else None

    if parcels_path and parcels_path.exists():
        parcels = gpd.read_file(parcels_path, columns=["crop_type", "geometry"])
        if "crop_type" in parcels.columns:
            target = parcels[parcels["crop_type"] == target_code].copy()
        else:
            target = parcels.copy()

        if aoi_path and aoi_path.exists() and len(target):
            aoi = gpd.read_file(aoi_path)
            if target.crs and aoi.crs and target.crs != aoi.crs:
                aoi = aoi.to_crs(target.crs)
            target = target[target.intersects(aoi.union_all())].copy()

        target = target[~target.geometry.is_empty & target.geometry.notna()]
        if len(target):
            return target[["geometry"]].copy()

    if aoi_path and aoi_path.exists():
        aoi = gpd.read_file(aoi_path)
        aoi = aoi[~aoi.geometry.is_empty & aoi.geometry.notna()]
        if len(aoi):
            return aoi[["geometry"]].copy()

    return None


def s2_cloud_probability_path(safe_dir: Path) -> Path | None:
    for pattern in ("**/MSK_CLDPRB_20m.jp2", "**/MSK_CLDPRB_60m.jp2"):
        matches = sorted(safe_dir.glob(pattern))
        if matches:
            return matches[0]
    return None


def apply_aoi_s2_cloud_screening(
    root: Path,
    config: dict[str, Any],
    scenes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str, str]:
    s2_cfg = config.get("s2", {})
    if not bool(s2_cfg.get("use_aoi_cloud_screening", True)):
        return scenes, "inventory_first_pass", "AOI cloud screening is disabled in config."

    try:
        import rasterio
        from rasterio.mask import mask
    except ImportError:
        return (
            scenes,
            "inventory_first_pass",
            "rasterio is not installed; using scene-level S2 quality inventory tags.",
        )

    cloud_geometries = read_s2_cloud_geometries(root, config)
    if cloud_geometries is None or not len(cloud_geometries):
        return (
            scenes,
            "inventory_first_pass",
            "No AOI or target-crop parcel geometry was available for AOI cloud screening.",
        )

    s2_dir = resolve_path(root, config["paths"]["s2_dir"])
    probability_threshold = float(s2_cfg.get("aoi_cloud_probability_threshold", 50.0))
    screened: list[dict[str, Any]] = []
    computed_count = 0
    skipped_count = 0

    for scene in scenes:
        scene_copy = dict(scene)
        safe_name = str(scene.get("safe_name", ""))
        safe_dir = s2_dir / safe_name if safe_name else None
        cloud_path = s2_cloud_probability_path(safe_dir) if safe_dir and safe_dir.exists() else None
        if cloud_path is None:
            skipped_count += 1
            screened.append(scene_copy)
            continue

        try:
            with rasterio.open(cloud_path) as dataset:
                geometries = cloud_geometries
                if geometries.crs and dataset.crs and geometries.crs != dataset.crs:
                    geometries = geometries.to_crs(dataset.crs)
                shapes = [geometry.__geo_interface__ for geometry in geometries.geometry]
                masked, _ = mask(dataset, shapes, crop=True, filled=False)
                data = masked[0]
                valid_mask = ~data.mask if np.ma.isMaskedArray(data) else np.isfinite(data)
                if not np.any(valid_mask):
                    skipped_count += 1
                    screened.append(scene_copy)
                    continue
                valid_values = np.asarray(data)[valid_mask]
                cloud_pct = float(np.count_nonzero(valid_values >= probability_threshold)) / float(
                    valid_values.size
                ) * 100.0
        except Exception as exc:
            skipped_count += 1
            scene_copy["aoi_cloud_error"] = str(exc)[:200]
            screened.append(scene_copy)
            continue

        scene_copy["aoi_cloud_pct"] = round(cloud_pct, 6)
        scene_copy["aoi_cloud_probability_threshold"] = probability_threshold
        scene_copy["first_pass"] = classify_s2_cloud_score(config, cloud_pct)
        scene_copy["quality_score"] = round(cloud_pct, 6)
        screened.append(scene_copy)
        computed_count += 1

    if computed_count == 0:
        return (
            scenes,
            "inventory_first_pass",
            "AOI cloud screening could not read any cloud-probability rasters; using inventory tags.",
        )

    note = (
        f"Used target-crop parcels within AOI against MSK_CLDPRB rasters; "
        f"cloudy pixels are cloud probability >= {probability_threshold:g}. "
        f"Computed {computed_count} scenes"
    )
    if skipped_count:
        note += f"; fell back to inventory tags for {skipped_count} scenes."
    else:
        note += "."
    return screened, "aoi_cloud_probability", note


def s2_scene_details(scenes: list[dict[str, Any]], usable_tags: tuple[str, ...]) -> list[dict[str, Any]]:
    fields = [
        "date",
        "safe_name",
        "first_pass",
        "quality_score",
        "aoi_cloud_pct",
        "aoi_cloud_probability_threshold",
        "cloud_pct",
        "cloud_shadow_pct",
        "land_cloud_pct",
        "nodata_pct",
        "has_cldprb",
        "has_scl",
        "aoi_cloud_error",
        "notes",
    ]
    details: list[dict[str, Any]] = []
    for scene in sorted(scenes, key=lambda item: str(item.get("date", ""))):
        row = {field: scene.get(field, "") for field in fields}
        row["usable"] = scene.get("first_pass") in usable_tags
        details.append(row)
    return details


def refresh_s2_indices_csv(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    skip_refresh: bool,
) -> tuple[Path, str]:
    paths = config["paths"]
    csv_path = resolve_path(
        root,
        paths.get(
            "s2_indices_csv",
            "data/output/s2_t49rfq_wheat_aoi_indices.csv",
        ),
    )
    if skip_refresh:
        return csv_path, "existing_csv" if csv_path.exists() else "missing_csv"

    script_path = resolve_path(
        root,
        paths.get(
            "s2_indices_script",
            "scripts/data_process/calc_s2_indices_csv.py",
        ),
    )
    if not script_path.exists():
        if csv_path.exists():
            return csv_path, f"script_missing_using_existing:{script_path}"
        raise FileNotFoundError(f"S2 indices script not found: {script_path}")

    command = [
        sys.executable,
        str(script_path),
        "--config",
        str(config_path),
        "--project-root",
        str(root),
        "--s2-dir",
        str(resolve_path(root, paths["s2_dir"])),
        "--out",
        str(csv_path),
    ]
    result = subprocess.run(command, cwd=root, check=False)
    if result.returncode == 0 and csv_path.exists():
        return csv_path, "refreshed"
    if csv_path.exists():
        return csv_path, f"refresh_failed_using_existing:exit_{result.returncode}"
    raise RuntimeError(
        f"S2 indices calculation failed with exit code {result.returncode}, "
        f"and no CSV is available at {csv_path}."
    )


def descending_index_score(value: float, green: float, mature: float) -> float:
    if green <= mature:
        raise ValueError("S2 green threshold must be greater than mature threshold.")
    return clamp((green - value) / (green - mature))


def read_s2_index_metrics(
    csv_path: Path,
    config: dict[str, Any],
    usable_tags: tuple[str, ...],
) -> dict[str, Any]:
    defaults = {
        "scene_count": 0,
        "latest_date": None,
        "ndvi": None,
        "ndre": None,
        "ndmi": None,
        "maturity_score": 0.0,
        "confidence": 0.0,
    }
    if not csv_path.exists():
        return defaults

    index_cfg = config.get("s2", {}).get("maturity_indices", {})
    statistic = str(index_cfg.get("statistic", "median")).lower()
    if statistic not in {"mean", "median"}:
        raise ValueError("s2.maturity_indices.statistic must be mean or median.")
    minimum_effective_ratio = float(
        index_cfg.get("minimum_effective_pixel_ratio", 0.4)
    )
    crop = config["crop"]
    season_start = parse_date(crop["season_start"])
    harvest_deadline = parse_date(crop["harvest_deadline"])

    candidates: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                observation_date = parse_date(str(row["date"]))
                effective_ratio = float(row["effective_pixel_ratio"])
                values = {
                    name: float(row[f"{name}_{statistic}"])
                    for name in ("ndvi", "ndre", "ndmi")
                }
            except (KeyError, TypeError, ValueError):
                continue
            if row.get("status") != "ok":
                continue
            if row.get("quality_tag") not in usable_tags:
                continue
            if not season_start <= observation_date <= harvest_deadline:
                continue
            if effective_ratio < minimum_effective_ratio:
                continue
            if not all(np.isfinite(value) for value in values.values()):
                continue
            candidates.append(
                {
                    "date": observation_date,
                    "effective_ratio": clamp(effective_ratio),
                    **values,
                }
            )

    if not candidates:
        return defaults

    latest = max(candidates, key=lambda row: row["date"])
    thresholds = {
        "ndvi": (
            float(index_cfg.get("ndvi_green", 0.45)),
            float(index_cfg.get("ndvi_mature", 0.25)),
        ),
        "ndre": (
            float(index_cfg.get("ndre_green", 0.28)),
            float(index_cfg.get("ndre_mature", 0.15)),
        ),
        "ndmi": (
            float(index_cfg.get("ndmi_green", 0.16)),
            float(index_cfg.get("ndmi_mature", 0.06)),
        ),
    }
    weights = index_cfg.get(
        "weights",
        {"ndvi": 0.50, "ndre": 0.35, "ndmi": 0.15},
    )
    index_scores = {
        name: descending_index_score(latest[name], *thresholds[name])
        for name in ("ndvi", "ndre", "ndmi")
    }
    weight_sum = sum(float(weights.get(name, 0.0)) for name in index_scores)
    maturity_score = (
        sum(
            index_scores[name] * float(weights.get(name, 0.0))
            for name in index_scores
        )
        / weight_sum
        if weight_sum
        else 0.0
    )
    return {
        "scene_count": len(candidates),
        "latest_date": latest["date"],
        "ndvi": latest["ndvi"],
        "ndre": latest["ndre"],
        "ndmi": latest["ndmi"],
        "maturity_score": round(clamp(maturity_score), 4),
        "confidence": round(latest["effective_ratio"], 4),
    }


def read_s2_summary(
    root: Path,
    config: dict[str, Any],
    parcels: ParcelSummary | None = None,
    indices_csv: Path | None = None,
    indices_status: str = "not_loaded",
) -> S2Summary:
    paths = config["paths"]
    usable_tags = tuple(config["s2"]["usable_tags"])
    inventory_path = resolve_path(root, paths["s2_quality_inventory"])

    scenes: list[dict[str, Any]] = []
    if inventory_path.exists():
        payload = json.loads(inventory_path.read_text(encoding="utf-8"))
        scenes = payload.get("scenes", [])
    else:
        s2_dir = resolve_path(root, paths["s2_dir"])
        for safe_dir in sorted(s2_dir.glob("*.SAFE")):
            metadata = safe_dir / "MTD_MSIL2A.xml"
            if not metadata.exists():
                continue
            xml_root = ET.parse(metadata).getroot()
            product_start = xml_text(xml_root, "PRODUCT_START_TIME")
            scenes.append(
                {
                    "safe_name": safe_dir.name,
                    "date": product_start[:10],
                    "first_pass": "unknown",
                }
            )

    if parcels is not None:
        scenes, cloud_screening, cloud_screening_note = apply_aoi_s2_cloud_screening(
            root, config, scenes
        )
    else:
        cloud_screening = "inventory_first_pass"
        cloud_screening_note = "No parcel/AOI summary was loaded before S2 screening."

    dates = [parse_date(str(scene["date"])) for scene in scenes if scene.get("date")]
    usable_dates = [
        parse_date(str(scene["date"]))
        for scene in scenes
        if scene.get("date") and scene.get("first_pass") in usable_tags
    ]
    minimum_usable = float(config["s2"]["minimum_usable_scenes"])
    availability_confidence = clamp(len(usable_dates) / minimum_usable)
    index_metrics = (
        read_s2_index_metrics(indices_csv, config, usable_tags)
        if indices_csv is not None
        else {
            "scene_count": 0,
            "latest_date": None,
            "ndvi": None,
            "ndre": None,
            "ndmi": None,
            "maturity_score": 0.0,
            "confidence": 0.0,
        }
    )
    confidence = (
        min(availability_confidence, float(index_metrics["confidence"]))
        if index_metrics["latest_date"] is not None
        else availability_confidence
    )

    return S2Summary(
        scene_count=len(scenes),
        usable_count=len(usable_dates),
        latest_usable_date=max(usable_dates) if usable_dates else None,
        first_date=min(dates) if dates else None,
        last_date=max(dates) if dates else None,
        usable_tags=usable_tags,
        cloud_screening=cloud_screening,
        cloud_screening_note=cloud_screening_note,
        scenes=s2_scene_details(scenes, usable_tags),
        indices_csv=str(indices_csv) if indices_csv else None,
        indices_status=indices_status,
        index_scene_count=int(index_metrics["scene_count"]),
        latest_index_date=index_metrics["latest_date"],
        latest_ndvi=(
            round(float(index_metrics["ndvi"]), 6)
            if index_metrics["ndvi"] is not None
            else None
        ),
        latest_ndre=(
            round(float(index_metrics["ndre"]), 6)
            if index_metrics["ndre"] is not None
            else None
        ),
        latest_ndmi=(
            round(float(index_metrics["ndmi"]), 6)
            if index_metrics["ndmi"] is not None
            else None
        ),
        index_maturity_score=float(index_metrics["maturity_score"]),
        index_confidence=float(index_metrics["confidence"]),
        availability_confidence=round(availability_confidence, 4),
        confidence=round(confidence, 4),
    )


def read_s1_summary(root: Path, config: dict[str, Any]) -> S1Summary:
    s1_dir = resolve_path(root, config["paths"]["s1_dir"])
    preferred_orbit = int(config["s1"]["preferred_relative_orbit"])
    preferred_dates: list[date] = []
    scene_count = 0

    for manifest in sorted(s1_dir.glob("**/*.SAFE/manifest.safe")):
        scene_count += 1
        xml_root = ET.parse(manifest).getroot()
        rel_orbit = xml_text(xml_root, "relativeOrbitNumber")
        start_time = xml_text(xml_root, "startTime")
        if rel_orbit and int(rel_orbit) == preferred_orbit and start_time:
            preferred_dates.append(parse_date(start_time[:10]))

    minimum_preferred = float(config["s1"]["minimum_preferred_scenes"])
    confidence = clamp(len(preferred_dates) / minimum_preferred)
    return S1Summary(
        scene_count=scene_count,
        preferred_orbit=preferred_orbit,
        preferred_count=len(preferred_dates),
        latest_preferred_date=max(preferred_dates) if preferred_dates else None,
        confidence=round(confidence, 4),
    )


def normalize_crop_mapping(mapping: dict[Any, Any]) -> dict[int, str]:
    return {int(key): str(value) for key, value in mapping.items()}


def read_parcel_summary(root: Path, config: dict[str, Any], output_dir: Path) -> ParcelSummary | None:
    paths = config["paths"]
    parcels_value = paths.get("parcels")
    if not parcels_value:
        return None

    parcels_path = resolve_path(root, parcels_value)
    if not parcels_path.exists():
        return None

    crop_cfg = config["crop"]
    target_code = int(crop_cfg["crop_type_code"])
    crop_mapping = normalize_crop_mapping(crop_cfg.get("crop_type_mapping", {}))
    target_label = crop_mapping.get(target_code, str(target_code))

    columns = ["OBJECTID", "crop_type", "mu", "mj"]
    parcels = gpd.read_file(parcels_path, columns=columns)
    total_features = int(len(parcels))
    parcels_crs = str(parcels.crs) if parcels.crs else ""
    parcels_bounds = tuple(float(v) for v in parcels.total_bounds) if len(parcels) else None

    aoi_path_value = paths.get("aoi")
    aoi_path = resolve_path(root, aoi_path_value) if aoi_path_value else None
    aoi_crs = None
    aoi_bounds = None
    if aoi_path and aoi_path.exists():
        aoi = gpd.read_file(aoi_path)
        aoi_crs = str(aoi.crs) if aoi.crs else ""
        if parcels.crs and aoi.crs and parcels.crs != aoi.crs:
            aoi = aoi.to_crs(parcels.crs)
        aoi_bounds = tuple(float(v) for v in aoi.total_bounds) if len(aoi) else None
        parcels = parcels[parcels.intersects(aoi.union_all())].copy()

    aoi_features = int(len(parcels))
    target = parcels[parcels["crop_type"] == target_code].copy()
    target_parcels = int(len(target))
    target_area_mu = float(target["mu"].fillna(0).sum()) if "mu" in target.columns else 0.0

    sample_output: str | None = None
    if target_parcels:
        output_dir.mkdir(parents=True, exist_ok=True)
        sample = target.sort_values("mu", ascending=False).head(200)
        sample_path = output_dir / "wheat_parcels_sample.geojson"
        sample.to_file(sample_path, driver="GeoJSON")
        sample_output = str(sample_path)

    return ParcelSummary(
        parcels_path=str(parcels_path),
        aoi_path=str(aoi_path) if aoi_path else None,
        crs=parcels_crs,
        aoi_crs=aoi_crs,
        total_features=total_features,
        aoi_features=aoi_features,
        target_crop_type=target_code,
        target_crop_label=target_label,
        target_parcels=target_parcels,
        target_area_mu=round(target_area_mu, 4),
        bounds=parcels_bounds,
        aoi_bounds=aoi_bounds,
        sample_output=sample_output,
    )


def month_range(start: str, end: str) -> list[str]:
    current = datetime.strptime(start, "%Y%m").date().replace(day=1)
    final = datetime.strptime(end, "%Y%m").date().replace(day=1)
    months: list[str] = []
    while current <= final:
        months.append(current.strftime("%Y%m"))
        year = current.year + (1 if current.month == 12 else 0)
        month = 1 if current.month == 12 else current.month + 1
        current = current.replace(year=year, month=month)
    return months


def read_weather_history(root: Path, config: dict[str, Any]) -> WeatherSummary:
    weather_dir = resolve_path(root, config["paths"]["weather_history_dir"])
    by_date: dict[date, list[WeatherDaily]] = {}

    for nc_file in sorted(weather_dir.glob("*/data_*.nc")):
        with h5py.File(nc_file, "r") as dataset:
            times = dataset["valid_time"][()]
            t2m_c = np.nanmean(dataset["t2m"][()], axis=(1, 2)) - 273.15
            precip_mm = np.nanmean(dataset["tp"][()], axis=(1, 2)) * 1000.0
            swvl1 = np.nanmean(dataset["swvl1"][()], axis=(1, 2))
            u10 = np.nanmean(dataset["u10"][()], axis=(1, 2))
            v10 = np.nanmean(dataset["v10"][()], axis=(1, 2))
            wind_speed = np.sqrt(np.square(u10) + np.square(v10))

            hourly: dict[date, dict[str, list[float]]] = {}
            for idx, timestamp in enumerate(times):
                dt = datetime.fromtimestamp(int(timestamp), UTC)
                day = dt.date()
                hourly.setdefault(
                    day,
                    {
                        "t2m_c": [],
                        "precip_mm": [],
                        "swvl1": [],
                        "wind_speed_m_s": [],
                    },
                )
                hourly[day]["t2m_c"].append(float(t2m_c[idx]))
                hourly[day]["precip_mm"].append(float(precip_mm[idx]))
                hourly[day]["swvl1"].append(float(swvl1[idx]))
                hourly[day]["wind_speed_m_s"].append(float(wind_speed[idx]))

            precip_aggregation = str(
                config["weather"].get("precipitation_daily_aggregation", "max")
            ).lower()
            for day, values in hourly.items():
                if precip_aggregation == "sum":
                    daily_precip = float(np.nansum(values["precip_mm"]))
                elif precip_aggregation == "last":
                    daily_precip = float(values["precip_mm"][-1])
                else:
                    daily_precip = float(np.nanmax(values["precip_mm"]))
                by_date.setdefault(day, []).append(
                    WeatherDaily(
                        date=day,
                        t2m_c=float(np.nanmean(values["t2m_c"])),
                        precip_mm=daily_precip,
                        swvl1=float(np.nanmean(values["swvl1"])),
                        wind_speed_m_s=float(np.nanmean(values["wind_speed_m_s"])),
                        source="history",
                    )
                )

    daily: list[WeatherDaily] = []
    for day in sorted(by_date):
        records = by_date[day]
        daily.append(
            WeatherDaily(
                date=day,
                t2m_c=float(np.nanmean([r.t2m_c for r in records])),
                precip_mm=float(np.nanmean([r.precip_mm for r in records])),
                swvl1=float(np.nanmean([r.swvl1 for r in records])),
                wind_speed_m_s=float(np.nanmean([r.wind_speed_m_s for r in records])),
                source="history",
            )
        )

    return summarize_weather_daily(config, daily)


def summarize_weather_daily(config: dict[str, Any], daily: list[WeatherDaily]) -> WeatherSummary:
    crop = config["crop"]
    base_temp = float(crop["gdd_base_c"])
    season_start = parse_date(crop["season_start"])
    daily = sorted(daily, key=lambda record: record.date)
    first_date = daily[0].date if daily else None
    last_date = daily[-1].date if daily else None
    months_present = sorted({record.date.strftime("%Y%m") for record in daily})
    if months_present:
        expected_months = month_range(min(months_present), max(months_present))
        months_missing = [month for month in expected_months if month not in months_present]
    else:
        months_missing = []

    cumulative_gdd = 0.0
    for record in daily:
        if record.date >= season_start:
            cumulative_gdd += max(0.0, record.t2m_c - base_temp)

    expected_days = 0
    if first_date and last_date:
        expected_days = (last_date - first_date).days + 1
    completeness = len(daily) / expected_days if expected_days else 0.0
    confidence = clamp(completeness * (0.85 if months_missing else 1.0))

    return WeatherSummary(
        daily=daily,
        months_present=sorted(set(months_present)),
        months_missing=months_missing,
        first_date=first_date,
        last_date=last_date,
        cumulative_gdd=round(cumulative_gdd, 2),
        confidence=round(confidence, 4),
    )


def forecast_as_of_date(config: dict[str, Any], weather: WeatherSummary) -> date | None:
    forecast_cfg = config.get("forecast", {})
    configured = forecast_cfg.get("as_of_date") or forecast_cfg.get("data_cutoff_date")
    if configured:
        return parse_date(str(configured))
    return weather.last_date


def weather_until(config: dict[str, Any], weather: WeatherSummary, cutoff: date | None) -> WeatherSummary:
    if cutoff is None:
        return weather
    return summarize_weather_daily(
        config,
        [record for record in weather.daily if record.date <= cutoff],
    )


def score_linear_date(reference: date, start: date, full: date) -> float:
    if reference <= start:
        return 0.0
    if reference >= full:
        return 1.0
    return clamp((reference - start).days / max(1, (full - start).days))


def calculate_maturity(
    config: dict[str, Any],
    s2: S2Summary,
    s1: S1Summary,
    weather: WeatherSummary,
) -> MaturitySummary:
    crop = config["crop"]
    maturity_start = parse_date(crop["maturity_start"])
    maturity_full = parse_date(crop["maturity_full"])
    reference_candidates = [
        d
        for d in [
            weather.last_date,
            s2.latest_index_date or s2.latest_usable_date,
            s1.latest_preferred_date,
        ]
        if d
    ]
    reference_date = max(reference_candidates) if reference_candidates else date.today()

    calendar_score = score_linear_date(reference_date, maturity_start, maturity_full)
    if s2.latest_index_date is not None:
        s2_score = s2.index_maturity_score * s2.confidence
    elif s2.latest_usable_date:
        s2_score = (
            score_linear_date(s2.latest_usable_date, maturity_start, maturity_full)
            * s2.confidence
        )
    else:
        s2_score = 0.0
    s1_score = (
        score_linear_date(s1.latest_preferred_date, maturity_start, maturity_full)
        * s1.confidence
        if s1.latest_preferred_date
        else 0.0
    )
    gdd_score = clamp(weather.cumulative_gdd / float(crop["gdd_target_c_day"]))

    weights = config["scoring"]["maturity_weights"]
    weighted = (
        calendar_score * float(weights["calendar"])
        + s2_score * float(weights["s2"])
        + s1_score * float(weights["s1"])
        + gdd_score * float(weights["gdd"])
    )
    weight_sum = sum(float(value) for value in weights.values())
    maturity_score = weighted / weight_sum if weight_sum else 0.0
    confidence = float(np.nanmean([s2.confidence, s1.confidence, weather.confidence]))

    return MaturitySummary(
        reference_date=reference_date,
        calendar_score=round(calendar_score, 4),
        s2_score=round(s2_score, 4),
        s1_score=round(s1_score, 4),
        gdd_score=round(gdd_score, 4),
        maturity_score=round(clamp(maturity_score), 4),
        confidence=round(clamp(confidence), 4),
    )


def read_forecast_csv(path: Path) -> list[WeatherDaily]:
    records: list[WeatherDaily] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            records.append(
                WeatherDaily(
                    date=parse_date(row["date"]),
                    t2m_c=float(row["t2m_c"]),
                    precip_mm=float(row["precip_mm"]),
                    swvl1=float(row["swvl1"]),
                    wind_speed_m_s=float(row["wind_speed_m_s"]),
                    source="forecast_file",
                )
            )
    return records


def read_forecast_json(path: Path) -> list[WeatherDaily]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("daily", payload if isinstance(payload, list) else [])
    return [
        WeatherDaily(
            date=parse_date(row["date"]),
            t2m_c=float(row["t2m_c"]),
            precip_mm=float(row["precip_mm"]),
            swvl1=float(row["swvl1"]),
            wind_speed_m_s=float(row["wind_speed_m_s"]),
            source="forecast_file",
        )
        for row in rows
    ]


def build_fallback_forecast(
    config: dict[str, Any],
    weather: WeatherSummary,
) -> list[WeatherDaily]:
    horizon = int(config["forecast"]["horizon_days"])
    tail_days = int(config["forecast"]["historical_tail_days"])
    if not weather.daily:
        return []
    source_tail = weather.daily[-tail_days:]
    start_date = weather.daily[-1].date + timedelta(days=1)
    records: list[WeatherDaily] = []
    for offset in range(horizon):
        source = source_tail[offset % len(source_tail)]
        records.append(
            WeatherDaily(
                date=start_date + timedelta(days=offset),
                t2m_c=source.t2m_c,
                precip_mm=source.precip_mm,
                swvl1=source.swvl1,
                wind_speed_m_s=source.wind_speed_m_s,
                source="forecast_fallback_historical_tail",
            )
        )
    return records


def read_historical_after_as_of_forecast(
    config: dict[str, Any],
    available_weather: WeatherSummary,
    as_of_date: date | None,
) -> list[WeatherDaily]:
    if as_of_date is None:
        return []
    horizon = int(config["forecast"]["horizon_days"])
    records = [
        record
        for record in available_weather.daily
        if record.date > as_of_date
    ][:horizon]
    return [
        WeatherDaily(
            date=record.date,
            t2m_c=record.t2m_c,
            precip_mm=record.precip_mm,
            swvl1=record.swvl1,
            wind_speed_m_s=record.wind_speed_m_s,
            source="historical_after_as_of",
        )
        for record in records
    ]


def forecast_coordinates(
    config: dict[str, Any],
    parcels: ParcelSummary | None,
) -> tuple[float, float]:
    forecast_cfg = config["forecast"]
    if forecast_cfg.get("latitude") is not None and forecast_cfg.get("longitude") is not None:
        return float(forecast_cfg["latitude"]), float(forecast_cfg["longitude"])

    bounds = None
    if parcels and parcels.aoi_bounds:
        bounds = parcels.aoi_bounds
    elif parcels and parcels.bounds:
        bounds = parcels.bounds
    if bounds:
        west, south, east, north = bounds
        return (south + north) / 2.0, (west + east) / 2.0

    raise ValueError("Forecast latitude/longitude is not configured and no AOI bounds are available.")


def safe_number(value: Any, fallback: float | None = None) -> float | None:
    if value is None:
        return fallback
    try:
        result = float(value)
    except (TypeError, ValueError):
        return fallback
    if math.isnan(result):
        return fallback
    return result


def fetch_open_meteo_forecast(
    config: dict[str, Any],
    parcels: ParcelSummary | None,
    weather: WeatherSummary,
) -> list[WeatherDaily]:
    forecast_cfg = config["forecast"]
    latitude, longitude = forecast_coordinates(config, parcels)
    forecast_days = min(16, max(1, int(forecast_cfg["horizon_days"])))
    latest_swvl1 = weather.daily[-1].swvl1 if weather.daily else 0.35
    params = {
        "latitude": f"{latitude:.6f}",
        "longitude": f"{longitude:.6f}",
        "hourly": ",".join(
            [
                "temperature_2m",
                "precipitation",
                "soil_moisture_0_to_1cm",
                "wind_speed_10m",
            ]
        ),
        "forecast_days": str(forecast_days),
        "timezone": forecast_cfg.get("timezone", "Asia/Shanghai"),
        "temperature_unit": "celsius",
        "wind_speed_unit": "ms",
        "precipitation_unit": "mm",
    }
    url = f"{forecast_cfg['api_url']}?{urllib.parse.urlencode(params)}"
    request = urllib.request.Request(url, headers={"User-Agent": "crop-harvest-window/0.1"})
    with urllib.request.urlopen(request, timeout=float(forecast_cfg.get("timeout_seconds", 30))) as response:
        payload = json.loads(response.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    temps = hourly.get("temperature_2m", [])
    precip = hourly.get("precipitation", [])
    soil = hourly.get("soil_moisture_0_to_1cm", [])
    wind = hourly.get("wind_speed_10m", [])
    if not times:
        raise ValueError("Open-Meteo response does not contain hourly time records.")

    by_date: dict[date, dict[str, list[float]]] = {}
    for idx, time_text in enumerate(times):
        day = datetime.fromisoformat(str(time_text)).date()
        by_date.setdefault(day, {"t2m_c": [], "precip_mm": [], "swvl1": [], "wind_speed_m_s": []})
        by_date[day]["t2m_c"].append(safe_number(temps[idx] if idx < len(temps) else None, 0.0) or 0.0)
        by_date[day]["precip_mm"].append(
            safe_number(precip[idx] if idx < len(precip) else None, 0.0) or 0.0
        )
        by_date[day]["swvl1"].append(
            safe_number(soil[idx] if idx < len(soil) else None, latest_swvl1) or latest_swvl1
        )
        by_date[day]["wind_speed_m_s"].append(
            safe_number(wind[idx] if idx < len(wind) else None, 0.0) or 0.0
        )

    return [
        WeatherDaily(
            date=day,
            t2m_c=float(np.nanmean(values["t2m_c"])),
            precip_mm=float(np.nansum(values["precip_mm"])),
            swvl1=float(np.nanmean(values["swvl1"])),
            wind_speed_m_s=float(np.nanmean(values["wind_speed_m_s"])),
            source="open_meteo",
        )
        for day, values in sorted(by_date.items())
    ]


def realtime_forecast_alignment(
    config: dict[str, Any],
    weather: WeatherSummary,
) -> tuple[bool, dict[str, Any]]:
    """Return whether a realtime forecast API is temporally aligned with the data cutoff."""
    forecast_cfg = config["forecast"]
    runtime_date = datetime.now().date()
    cutoff_date = weather.last_date
    metadata: dict[str, Any] = {
        "runtime_date": runtime_date.isoformat(),
        "data_cutoff_date": cutoff_date.isoformat() if cutoff_date else None,
    }
    if bool(forecast_cfg.get("allow_realtime_api_for_historical", False)):
        metadata["alignment_policy"] = "allow_realtime_api_for_historical"
        return True, metadata
    if cutoff_date is None:
        metadata["alignment_policy"] = "no_weather_cutoff_available"
        return True, metadata
    if cutoff_date < runtime_date - timedelta(days=1):
        metadata["alignment_policy"] = "data_cutoff_relative"
        metadata["reason"] = "historical_data_cutoff_precedes_realtime_forecast_window"
        return False, metadata
    metadata["alignment_policy"] = "realtime_cutoff_aligned"
    return True, metadata


def load_forecast(
    root: Path,
    config: dict[str, Any],
    weather: WeatherSummary,
    available_weather: WeatherSummary,
    parcels: ParcelSummary | None,
    forecast_csv: Path | None,
    forecast_json: Path | None,
) -> tuple[list[WeatherDaily], str, float, dict[str, Any]]:
    if forecast_csv:
        return read_forecast_csv(resolve_path(root, forecast_csv)), "forecast_csv", 0.9, {}
    if forecast_json:
        return read_forecast_json(resolve_path(root, forecast_json)), "forecast_json", 0.9, {}

    as_of_date = forecast_as_of_date(config, weather)
    historical_future = read_historical_after_as_of_forecast(
        config,
        available_weather,
        as_of_date,
    )
    historical_future_metadata = {
        "as_of_date": as_of_date.isoformat() if as_of_date else None,
        "available_weather_first_date": (
            available_weather.first_date.isoformat() if available_weather.first_date else None
        ),
        "available_weather_last_date": (
            available_weather.last_date.isoformat() if available_weather.last_date else None
        ),
    }

    provider = str(config["forecast"].get("provider", "historical_tail")).lower()
    if provider in {"historical_after_as_of", "historical_after_cutoff"}:
        if historical_future:
            return historical_future, "historical_after_as_of", 0.95, historical_future_metadata
        return build_fallback_forecast(config, weather), "historical_tail_fallback", 0.45, {
            **historical_future_metadata,
            "reason": "no_weather_records_after_as_of_date",
        }

    if provider == "open_meteo":
        latitude, longitude = forecast_coordinates(config, parcels)
        use_realtime, alignment_metadata = realtime_forecast_alignment(config, weather)
        if not use_realtime:
            if historical_future:
                return historical_future, "historical_after_as_of", 0.95, {
                    "latitude": round(latitude, 6),
                    "longitude": round(longitude, 6),
                    "api_url": config["forecast"]["api_url"],
                    **historical_future_metadata,
                    **alignment_metadata,
                }
            return build_fallback_forecast(config, weather), "historical_cutoff_fallback", 0.45, {
                "latitude": round(latitude, 6),
                "longitude": round(longitude, 6),
                "api_url": config["forecast"]["api_url"],
                **historical_future_metadata,
                **alignment_metadata,
            }
        try:
            records = fetch_open_meteo_forecast(config, parcels, weather)
            return records, "open_meteo", 0.9, {
                "latitude": round(latitude, 6),
                "longitude": round(longitude, 6),
                "api_url": config["forecast"]["api_url"],
                **alignment_metadata,
            }
        except Exception as exc:
            return build_fallback_forecast(config, weather), "open_meteo_failed_fallback", 0.45, {
                "latitude": round(latitude, 6),
                "longitude": round(longitude, 6),
                "api_url": config["forecast"]["api_url"],
                **alignment_metadata,
                "error": str(exc)[:500],
            }

    if historical_future:
        return historical_future, "historical_after_as_of", 0.95, historical_future_metadata
    return build_fallback_forecast(config, weather), "historical_tail_fallback", 0.45, {
        **historical_future_metadata,
        "reason": "no_weather_records_after_as_of_date",
    }


def score_piecewise_low_good(value: float, good: float, poor: float) -> float:
    if value <= good:
        return 1.0
    if value >= poor:
        return 0.0
    return clamp(1.0 - (value - good) / (poor - good))


def score_range(value: float, good_min: float, good_max: float, poor_min: float, poor_max: float) -> float:
    if good_min <= value <= good_max:
        return 1.0
    if value < good_min:
        if value <= poor_min:
            return 0.0
        return clamp((value - poor_min) / (good_min - poor_min))
    if value >= poor_max:
        return 0.0
    return clamp(1.0 - (value - good_max) / (poor_max - good_max))


def score_forecast_days(
    config: dict[str, Any],
    forecast: list[WeatherDaily],
    maturity: MaturitySummary,
    historical_daily: list[WeatherDaily],
) -> list[ForecastDayScore]:
    weather_cfg = config["weather"]
    harvest_weights = config["scoring"]["harvest_score_weights"]
    dry_threshold = float(weather_cfg["dry_day_precip_mm"])

    dry_streak = 0
    for record in reversed(historical_daily[-10:]):
        if record.precip_mm <= dry_threshold:
            dry_streak += 1
        else:
            break

    scored: list[ForecastDayScore] = []
    for record in forecast:
        dry_streak = dry_streak + 1 if record.precip_mm <= dry_threshold else 0
        precip_score = score_piecewise_low_good(
            record.precip_mm,
            float(weather_cfg["dry_day_precip_mm"]),
            float(weather_cfg["wet_day_precip_mm"]),
        )
        soil_score = score_piecewise_low_good(
            record.swvl1,
            float(weather_cfg["soil_moisture_good"]),
            float(weather_cfg["soil_moisture_poor"]),
        )
        wind_score = score_range(
            record.wind_speed_m_s,
            float(weather_cfg["wind_good_min_m_s"]),
            float(weather_cfg["wind_good_max_m_s"]),
            0.0,
            float(weather_cfg["wind_poor_m_s"]),
        )
        temp_score = score_range(
            record.t2m_c,
            float(weather_cfg["temp_good_min_c"]),
            float(weather_cfg["temp_good_max_c"]),
            float(weather_cfg["temp_poor_min_c"]),
            float(weather_cfg["temp_poor_max_c"]),
        )
        dry_bonus = clamp(dry_streak / 3.0)
        weather_score = (
            precip_score * 0.35
            + soil_score * 0.25
            + wind_score * 0.15
            + temp_score * 0.15
            + dry_bonus * 0.10
        )
        harvest_score = (
            maturity.maturity_score * float(harvest_weights["maturity"])
            + weather_score * float(harvest_weights["weather"])
        )
        risks = []
        if record.precip_mm > float(weather_cfg["wet_day_precip_mm"]):
            risks.append("heavy_rain")
        elif record.precip_mm > float(weather_cfg["dry_day_precip_mm"]):
            risks.append("rain")
        if record.swvl1 > float(weather_cfg["soil_moisture_poor"]):
            risks.append("wet_soil")
        if record.wind_speed_m_s > float(weather_cfg["wind_poor_m_s"]):
            risks.append("strong_wind")
        if record.t2m_c < float(weather_cfg["temp_poor_min_c"]) or record.t2m_c > float(
            weather_cfg["temp_poor_max_c"]
        ):
            risks.append("temperature_extreme")

        scored.append(
            ForecastDayScore(
                date=record.date,
                t2m_c=round(record.t2m_c, 3),
                precip_mm=round(record.precip_mm, 3),
                swvl1=round(record.swvl1, 4),
                wind_speed_m_s=round(record.wind_speed_m_s, 3),
                dry_streak_days=dry_streak,
                weather_score=round(clamp(weather_score), 4),
                harvest_score=round(clamp(harvest_score), 4),
                risk_flags=risks,
            )
        )
    return scored


def choose_best_window(
    scored_days: list[ForecastDayScore],
    window_days: int,
    confidence: float,
    risk_penalties: dict[str, float],
) -> dict[str, Any]:
    if not scored_days:
        return {
            "start_date": None,
            "end_date": None,
            "average_harvest_score": 0.0,
            "level": "unavailable",
            "risk_summary": ["no_forecast_records"],
        }

    best_slice = scored_days[:window_days]
    best_score = -math.inf
    best_raw_score = 0.0
    best_adjusted_score = 0.0
    for start in range(0, max(1, len(scored_days) - window_days + 1)):
        candidate = scored_days[start : start + window_days]
        raw_average = float(np.nanmean([day.harvest_score for day in candidate]))
        risk_penalty = 0.0
        for day in candidate:
            for risk in day.risk_flags:
                risk_penalty += float(risk_penalties.get(risk, 0.0))
        adjusted_average = raw_average - risk_penalty / max(1, len(candidate))
        if adjusted_average > best_score:
            best_score = adjusted_average
            best_raw_score = raw_average
            best_adjusted_score = adjusted_average
            best_slice = candidate

    risks = sorted({risk for day in best_slice for risk in day.risk_flags})
    average_score = round(best_raw_score, 4)
    adjusted_score = round(best_adjusted_score, 4)
    if adjusted_score >= 0.75 and confidence >= 0.75 and not {"heavy_rain", "wet_soil"} & set(risks):
        level = "high"
    elif adjusted_score >= 0.55:
        level = "medium"
    else:
        level = "low"

    return {
        "start_date": best_slice[0].date.isoformat(),
        "end_date": best_slice[-1].date.isoformat(),
        "average_harvest_score": average_score,
        "risk_adjusted_score": adjusted_score,
        "level": level,
        "risk_summary": risks or ["low_weather_risk"],
    }


def as_jsonable(value: Any) -> Any:
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, list):
        return [as_jsonable(item) for item in value]
    if hasattr(value, "__dataclass_fields__"):
        return {key: as_jsonable(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {key: as_jsonable(item) for key, item in value.items()}
    return value


def zh_level(level: str) -> str:
    return {
        "high": "高",
        "medium": "中",
        "low": "低",
        "unavailable": "不可用",
    }.get(level, level)


def zh_risk(risk: str) -> str:
    return {
        "low_weather_risk": "天气风险较低",
        "rain": "有降雨风险",
        "heavy_rain": "强降雨风险",
        "wet_soil": "土壤偏湿",
        "strong_wind": "大风风险",
        "temperature_extreme": "极端温度风险",
        "no_forecast_records": "缺少天气预报记录",
    }.get(risk, risk)


def zh_crop(label: str, code: int | str) -> str:
    return {
        "wheat": "小麦",
        "rice": "水稻",
        "maize": "玉米",
        "non_crop": "非作物",
        "unknown": "未知",
    }.get(str(label), str(label)) + f"（crop_type={code}）"


def zh_forecast_source(source: str) -> str:
    return {
        "open_meteo": "Open-Meteo 实时天气预报 API",
        "forecast_csv": "本地 CSV 天气预报文件",
        "forecast_json": "本地 JSON 天气预报文件",
        "historical_after_as_of": "预测基准日之后的已下载历史天气",
        "historical_cutoff_fallback": "历史天气尾段模拟预报",
        "historical_tail_fallback": "历史天气尾段模拟预报",
        "open_meteo_failed_fallback": "Open-Meteo 请求失败后使用历史天气尾段模拟预报",
    }.get(source, source)


def write_outputs(
    output_dir: Path,
    payload: dict[str, Any],
    scored_days: list[ForecastDayScore],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "harvest_window_report.json").write_text(
        json.dumps(as_jsonable(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (output_dir / "harvest_window_daily_scores.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fieldnames = [
            "date",
            "t2m_c",
            "precip_mm",
            "swvl1",
            "wind_speed_m_s",
            "dry_streak_days",
            "weather_score",
            "harvest_score",
            "risk_flags",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for day in scored_days:
            row = asdict(day)
            row["date"] = day.date.isoformat()
            row["risk_flags"] = ";".join(day.risk_flags)
            writer.writerow(row)

    s2 = as_jsonable(payload.get("s2"))
    s2_scenes = s2.get("scenes", []) if isinstance(s2, dict) else []
    with (output_dir / "harvest_window_s2_scenes.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        fieldnames = [
            "date",
            "safe_name",
            "first_pass",
            "usable",
            "quality_score",
            "aoi_cloud_pct",
            "aoi_cloud_probability_threshold",
            "cloud_pct",
            "cloud_shadow_pct",
            "land_cloud_pct",
            "nodata_pct",
            "has_cldprb",
            "has_scl",
            "aoi_cloud_error",
            "notes",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for scene in s2_scenes:
            writer.writerow(scene)

    window = payload["recommendation"]
    maturity = as_jsonable(payload["maturity"])
    parcels = as_jsonable(payload.get("parcels"))
    forecast = payload.get("forecast", {})
    risk_text = "、".join(zh_risk(risk) for risk in window["risk_summary"])
    lines = [
        "# 最佳收获窗口推荐结果",
        "",
        f"- 推荐收获窗口：{window['start_date']} 至 {window['end_date']}",
        f"- 推荐等级：{zh_level(window['level'])}",
        f"- 天气预报来源：{zh_forecast_source(str(forecast.get('source', '')))}",
        f"- 平均收获分数：{window['average_harvest_score']}",
        f"- 风险调整后分数：{window['risk_adjusted_score']}",
        f"- 成熟度分数：{maturity['maturity_score']}",
        f"- 主要风险：{risk_text}",
    ]
    if s2:
        lines.append(f"- S2 云量筛选：{s2['cloud_screening']}（{s2['cloud_screening_note']}）")
    if s2 and s2.get("latest_index_date"):
        lines.extend(
            [
                f"- S2 index date: {s2['latest_index_date']}",
                (
                    "- S2 indices: "
                    f"NDVI={s2['latest_ndvi']}, "
                    f"NDRE={s2['latest_ndre']}, "
                    f"NDMI={s2['latest_ndmi']}"
                ),
                (
                    "- S2 index maturity: "
                    f"score={s2['index_maturity_score']}, "
                    f"confidence={s2['index_confidence']}"
                ),
            ]
        )
    if parcels:
        lines.extend(
            [
                f"- AOI 内地块数：{parcels['aoi_features']}",
                f"- 目标作物：{zh_crop(parcels['target_crop_label'], parcels['target_crop_type'])}",
                f"- 目标作物地块数：{parcels['target_parcels']}",
                f"- 目标作物面积：{parcels['target_area_mu']} 亩",
            ]
        )
    try:
        window_start = parse_date(str(window["start_date"]))
        window_end = parse_date(str(window["end_date"]))
        window_days = [
            day
            for day in scored_days
            if window_start <= day.date <= window_end
        ]
    except (TypeError, ValueError):
        window_days = []
    if window_days:
        lines.extend(
            [
                "",
                "## 推荐窗口天气",
                "",
                "| 日期 | 平均气温(°C) | 降水(mm) | 土壤湿度 | 风速(m/s) | 连续干燥天数 | 天气分 | 收获分 | 风险 |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for day in window_days:
            day_risk_text = "、".join(zh_risk(risk) for risk in day.risk_flags) or "无"
            lines.append(
                "| "
                f"{day.date.isoformat()} | "
                f"{day.t2m_c:.3f} | "
                f"{day.precip_mm:.3f} | "
                f"{day.swvl1:.4f} | "
                f"{day.wind_speed_m_s:.3f} | "
                f"{day.dry_streak_days} | "
                f"{day.weather_score:.4f} | "
                f"{day.harvest_score:.4f} | "
                f"{day_risk_text} |"
            )
    note = [
        "",
        "说明：当前结果是 v1 规则模型生成的区域级推荐。已使用 AOI 和小麦地块筛选。",
        "目前还没有接入真实收获日期标签；后续加入地块级遥感时序统计后，",
        "可以进一步输出每个小麦地块的独立收获窗口。",
    ]
    if forecast.get("source") == "open_meteo":
        note.append("当前天气部分已使用真实天气预报 API，预报日期为运行当天之后的未来日期。")
    elif forecast.get("source") == "historical_after_as_of":
        note.append("当前天气部分使用预测基准日之后已下载的历史天气，作为回放模式下的预测期天气。")
    elif "fallback" in str(forecast.get("source", "")):
        note.append("当前天气部分未使用真实预报，使用历史天气尾段作为模拟预报。")
    note.append("")
    lines.extend(note)
    (output_dir / "harvest_window_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_warnings(
    s2: S2Summary,
    s1: S1Summary,
    weather: WeatherSummary,
    forecast_source: str,
    parcels: ParcelSummary | None,
) -> list[str]:
    warnings = [
        "No true harvest-date labels were provided; v1 uses explainable rules, not supervised learning.",
    ]
    if parcels is None:
        warnings.append("No parcel/AOI boundary was loaded; output is regional, not field-level.")
    elif parcels.target_parcels == 0:
        warnings.append("No target crop parcels were found inside the AOI.")
    if weather.months_missing:
        warnings.append("Missing weather months: " + ", ".join(weather.months_missing))
    if forecast_source == "historical_after_as_of":
        warnings.append(
            "Using downloaded historical weather after the decision date as forecast-period weather."
        )
    if forecast_source == "historical_tail_fallback":
        warnings.append("No real forecast file/API was provided; using historical-tail fallback.")
    if forecast_source == "historical_cutoff_fallback":
        warnings.append("No post-cutoff weather records were available; using historical-tail fallback.")
    if forecast_source == "open_meteo_failed_fallback":
        warnings.append("Open-Meteo forecast API failed; using historical-tail fallback.")
    if s2.confidence < 0.8:
        warnings.append("S2 usable-scene confidence is below 0.8.")
    if s2.cloud_screening != "aoi_cloud_probability":
        warnings.append("AOI-specific S2 cloud screening was not applied: " + s2.cloud_screening_note)
    if s2.latest_index_date is None:
        warnings.append(
            "No usable S2 NDVI/NDRE/NDMI record was available; S2 maturity used the date fallback."
        )
    if s2.indices_status.startswith("refresh_failed"):
        warnings.append("S2 indices refresh failed; maturity used the existing CSV.")
    if s1.confidence < 0.8:
        warnings.append("Preferred S1 orbit coverage confidence is below 0.8.")
    return warnings


def log_progress(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def main() -> int:
    args = parse_args()
    project_root = Path.cwd()
    config_path = resolve_path(project_root, args.config)
    log_progress(f"Step 1/9 Load config: {config_path}")
    config = load_config(config_path)

    output_dir = (
        resolve_path(project_root, args.output_dir)
        if args.output_dir
        else resolve_path(project_root, config["paths"]["output_dir"])
    )

    log_progress("Step 2/9 Load parcels and AOI. S2 cloud screening will use this area first.")
    parcels = read_parcel_summary(project_root, config, output_dir)
    if parcels:
        log_progress(
            "Parcels/AOI loaded: "
            f"aoi_features={parcels.aoi_features}, "
            f"target_parcels={parcels.target_parcels}, "
            f"target_area_mu={parcels.target_area_mu}."
        )
    else:
        log_progress("No parcels/AOI loaded; S2 will use scene-level quality inventory only.")

    log_progress(
        "Step 3/9 Calculate AOI target-crop S2 indices, then load S2 cloud and index summaries."
    )
    s2_indices_csv, s2_indices_status = refresh_s2_indices_csv(
        project_root,
        config_path,
        config,
        args.skip_s2_index_refresh,
    )
    log_progress(
        f"S2 indices CSV: status={s2_indices_status}, path={s2_indices_csv}."
    )
    s2 = read_s2_summary(
        project_root,
        config,
        parcels,
        indices_csv=s2_indices_csv,
        indices_status=s2_indices_status,
    )
    log_progress(
        "S2 loaded: "
        f"cloud_screening={s2.cloud_screening}; "
        f"note={s2.cloud_screening_note}; "
        f"scene_count={s2.scene_count}, usable_count={s2.usable_count}, "
        f"latest_usable_date={s2.latest_usable_date}; "
        f"latest_index_date={s2.latest_index_date}, "
        f"NDVI={s2.latest_ndvi}, NDRE={s2.latest_ndre}, NDMI={s2.latest_ndmi}, "
        f"index_maturity_score={s2.index_maturity_score}, "
        f"index_confidence={s2.index_confidence}."
    )

    log_progress("Step 4/9 Load Sentinel-1 coverage.")
    s1 = read_s1_summary(project_root, config)
    log_progress(
        "S1 loaded: "
        f"scene_count={s1.scene_count}, preferred_orbit={s1.preferred_orbit}, "
        f"preferred_count={s1.preferred_count}, latest_preferred_date={s1.latest_preferred_date}."
    )

    log_progress("Step 5/9 Load historical weather and calculate GDD.")
    available_weather = read_weather_history(project_root, config)
    as_of_date = forecast_as_of_date(config, available_weather)
    weather = weather_until(config, available_weather, as_of_date)
    log_progress(
        "Historical weather loaded: "
        f"available_range={available_weather.first_date} to {available_weather.last_date}, "
        f"decision_as_of={as_of_date}, "
        f"history_range={weather.first_date} to {weather.last_date}, "
        f"daily_records={len(weather.daily)}, cumulative_gdd={weather.cumulative_gdd}, "
        f"missing_months={weather.months_missing or 'none'}."
    )

    log_progress("Step 6/9 Calculate maturity from calendar, S2, S1, and GDD.")
    maturity = calculate_maturity(config, s2, s1, weather)
    log_progress(
        "Maturity calculated: "
        f"reference_date={maturity.reference_date}, "
        f"maturity_score={maturity.maturity_score}, confidence={maturity.confidence}."
    )

    log_progress("Step 7/9 Load forecast: input file first, otherwise configured forecast API.")
    forecast, forecast_source, forecast_confidence, forecast_metadata = load_forecast(
        project_root,
        config,
        weather,
        available_weather,
        parcels,
        args.forecast_csv,
        args.forecast_json,
    )
    log_progress(
        "Forecast loaded: "
        f"source={forecast_source}, records={len(forecast)}, "
        f"confidence={forecast_confidence}, metadata={forecast_metadata}."
    )

    log_progress("Step 8/9 Score forecast days and select the best continuous harvest window.")
    scored_days = score_forecast_days(config, forecast, maturity, weather.daily)
    overall_confidence = round(
        float(np.nanmean([maturity.confidence, weather.confidence, forecast_confidence])), 4
    )
    recommendation = choose_best_window(
        scored_days,
        int(config["forecast"]["window_days"]),
        overall_confidence,
        config["scoring"].get("window_risk_penalties", {}),
    )
    log_progress(
        "Window selected: "
        f"{recommendation['start_date']} to {recommendation['end_date']}, "
        f"level={recommendation['level']}, "
        f"risk_adjusted_score={recommendation.get('risk_adjusted_score')}."
    )

    payload = {
        "run_time_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "mode": "regional_v1_rule_based",
        "crop": config["crop"],
        "s2": s2,
        "s1": s1,
        "parcels": parcels,
        "weather_history": {
            "months_present": weather.months_present,
            "months_missing": weather.months_missing,
            "first_date": weather.first_date,
            "last_date": weather.last_date,
            "available_first_date": available_weather.first_date,
            "available_last_date": available_weather.last_date,
            "as_of_date": as_of_date,
            "daily_records": len(weather.daily),
            "available_daily_records": len(available_weather.daily),
            "cumulative_gdd": weather.cumulative_gdd,
            "confidence": weather.confidence,
        },
        "forecast": {
            "source": forecast_source,
            "records": len(forecast),
            "confidence": forecast_confidence,
            "metadata": forecast_metadata,
        },
        "maturity": maturity,
        "recommendation": recommendation,
        "daily_scores": scored_days,
        "overall_confidence": overall_confidence,
        "warnings": build_warnings(s2, s1, weather, forecast_source, parcels),
    }

    log_progress(f"Step 9/9 Write output files: {output_dir.resolve()}")
    write_outputs(output_dir, payload, scored_days)
    log_progress("Harvest-window recommendation complete.")
    print(f"Recommended window: {recommendation['start_date']} to {recommendation['end_date']}")
    print(f"Level: {recommendation['level']}")
    print(f"Output directory: {output_dir.resolve()}")
    print(f"S2 scene screening: {output_dir.resolve() / 'harvest_window_s2_scenes.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
