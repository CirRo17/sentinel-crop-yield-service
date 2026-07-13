"""
Preprocess Sentinel-1 GRD scenes and calculate VV/VH features inside
AOI target-crop parcels.

Raw SAFE products are processed with ESA SNAP GPT:
  Apply-Orbit-File -> ThermalNoiseRemoval -> Remove-GRD-Border-Noise
  -> Calibration (Sigma0 VV/VH) -> AOI Subset -> Refined Lee
  -> Range-Doppler Terrain Correction -> LinearToFromdB -> GeoTIFF

The processed two-band GeoTIFFs are cached and then summarized only over
target-crop pixels. Existing valid cache files are reused.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from xml.etree import ElementTree as ET

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.mask import mask


SNAP_GRAPH = """\
<graph id="S1_GRD_AOI_Preprocessing">
  <version>1.0</version>
  <node id="Read">
    <operator>Read</operator>
    <sources/>
    <parameters>
      <file>${source}</file>
      <formatName>SENTINEL-1</formatName>
    </parameters>
  </node>
  <node id="Apply-Orbit-File">
    <operator>Apply-Orbit-File</operator>
    <sources><sourceProduct refid="Read"/></sources>
    <parameters>
      <orbitType>Sentinel Precise (Auto Download)</orbitType>
      <polyDegree>3</polyDegree>
      <continueOnFail>true</continueOnFail>
    </parameters>
  </node>
  <node id="ThermalNoiseRemoval">
    <operator>ThermalNoiseRemoval</operator>
    <sources><sourceProduct refid="Apply-Orbit-File"/></sources>
    <parameters>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <removeThermalNoise>true</removeThermalNoise>
      <outputNoise>false</outputNoise>
      <reIntroduceThermalNoise>false</reIntroduceThermalNoise>
    </parameters>
  </node>
  <node id="Remove-GRD-Border-Noise">
    <operator>Remove-GRD-Border-Noise</operator>
    <sources><sourceProduct refid="ThermalNoiseRemoval"/></sources>
    <parameters>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <borderLimit>500</borderLimit>
      <trimThreshold>0.5</trimThreshold>
    </parameters>
  </node>
  <node id="Calibration">
    <operator>Calibration</operator>
    <sources><sourceProduct refid="Remove-GRD-Border-Noise"/></sources>
    <parameters>
      <sourceBands/>
      <auxFile>Latest Auxiliary File</auxFile>
      <outputImageInComplex>false</outputImageInComplex>
      <outputImageScaleInDb>false</outputImageScaleInDb>
      <createGammaBand>false</createGammaBand>
      <createBetaBand>false</createBetaBand>
      <selectedPolarisations>VV,VH</selectedPolarisations>
      <outputSigmaBand>true</outputSigmaBand>
    </parameters>
  </node>
  <node id="Subset">
    <operator>Subset</operator>
    <sources><sourceProduct refid="Calibration"/></sources>
    <parameters>
      <sourceBands>Sigma0_VV,Sigma0_VH</sourceBands>
      <geoRegion>${geoRegion}</geoRegion>
      <subSamplingX>1</subSamplingX>
      <subSamplingY>1</subSamplingY>
      <fullSwath>false</fullSwath>
      <tiePointGridNames/>
      <copyMetadata>true</copyMetadata>
    </parameters>
  </node>
  <node id="Speckle-Filter">
    <operator>Speckle-Filter</operator>
    <sources><sourceProduct refid="Subset"/></sources>
    <parameters>
      <sourceBands>Sigma0_VV,Sigma0_VH</sourceBands>
      <filter>Refined Lee</filter>
      <filterSizeX>7</filterSizeX>
      <filterSizeY>7</filterSizeY>
      <dampingFactor>2</dampingFactor>
      <estimateENL>true</estimateENL>
      <enl>1.0</enl>
      <numLooksStr>1</numLooksStr>
      <windowSize>7x7</windowSize>
      <targetWindowSizeStr>3x3</targetWindowSizeStr>
      <sigmaStr>0.9</sigmaStr>
      <anSize>50</anSize>
    </parameters>
  </node>
  <node id="Terrain-Correction">
    <operator>Terrain-Correction</operator>
    <sources><sourceProduct refid="Speckle-Filter"/></sources>
    <parameters>
      <sourceBands>Sigma0_VV,Sigma0_VH</sourceBands>
      <demName>Copernicus 30m Global DEM</demName>
      <externalDEMNoDataValue>0.0</externalDEMNoDataValue>
      <externalDEMApplyEGM>true</externalDEMApplyEGM>
      <demResamplingMethod>BILINEAR_INTERPOLATION</demResamplingMethod>
      <imgResamplingMethod>BILINEAR_INTERPOLATION</imgResamplingMethod>
      <pixelSpacingInMeter>${pixelSpacing}</pixelSpacingInMeter>
      <mapProjection>${mapProjection}</mapProjection>
      <nodataValueAtSea>true</nodataValueAtSea>
      <saveSelectedSourceBand>true</saveSelectedSourceBand>
      <saveDEM>false</saveDEM>
      <saveLatLon>false</saveLatLon>
      <saveIncidenceAngleFromEllipsoid>false</saveIncidenceAngleFromEllipsoid>
      <saveLocalIncidenceAngle>false</saveLocalIncidenceAngle>
      <saveProjectedLocalIncidenceAngle>false</saveProjectedLocalIncidenceAngle>
      <saveLayoverShadowMask>false</saveLayoverShadowMask>
      <applyRadiometricNormalization>false</applyRadiometricNormalization>
    </parameters>
  </node>
  <node id="LinearToFromdB">
    <operator>LinearToFromdB</operator>
    <sources><sourceProduct refid="Terrain-Correction"/></sources>
    <parameters>
      <sourceBands>Sigma0_VV,Sigma0_VH</sourceBands>
    </parameters>
  </node>
  <node id="Write">
    <operator>Write</operator>
    <sources><sourceProduct refid="LinearToFromdB"/></sources>
    <parameters>
      <file>${output}</file>
      <formatName>GeoTIFF-BigTIFF</formatName>
      <deleteOutputOnFailure>true</deleteOutputOnFailure>
      <writeEntireTileRows>true</writeEntireTileRows>
      <clearCacheAfterRowWrite>false</clearCacheAfterRowWrite>
    </parameters>
  </node>
</graph>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Sentinel-1 GRD scenes and calculate VV/VH features "
            "inside AOI target-crop parcels."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(r"D:\Projects\crop_harvest_window\configs\harvest_window.yaml"),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(r"D:\Projects\crop_harvest_window"),
    )
    parser.add_argument(
        "--s1-dir",
        type=Path,
        default=Path(r"D:\Projects\crop_harvest_window\data\S1\relative orbit 113"),
        help="Directory containing Sentinel-1 GRD .SAFE folders.",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=Path(r"D:\Projects\crop_harvest_window\data\S1\processed_orbit113"),
        help="Cache directory for processed two-band Sigma0 dB GeoTIFFs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(
            r"D:\Projects\crop_harvest_window\data\output"
            r"\s1_orbit113_wheat_aoi_features.csv"
        ),
    )
    parser.add_argument("--parcels", type=Path)
    parser.add_argument("--aoi", type=Path)
    parser.add_argument("--crop-type-code", type=int)
    parser.add_argument("--crop-column", default="crop_type")
    parser.add_argument(
        "--relative-orbit",
        type=int,
        default=None,
        help="Only process this relative orbit. Defaults to config.",
    )
    parser.add_argument(
        "--snap-gpt",
        type=Path,
        default=None,
        help="Path to SNAP gpt.exe. Auto-detected when omitted.",
    )
    parser.add_argument(
        "--preprocessed-only",
        action="store_true",
        help="Do not invoke SNAP; only use valid files already in --processed-dir.",
    )
    parser.add_argument(
        "--overwrite-preprocessed",
        action="store_true",
        help="Re-run SNAP even when a valid cached GeoTIFF exists.",
    )
    parser.add_argument("--pixel-spacing", type=float, default=10.0)
    parser.add_argument("--target-epsg", type=int, default=32649)
    parser.add_argument(
        "--aoi-buffer-degrees",
        type=float,
        default=0.02,
        help="Buffer around target parcels for SNAP Subset, in degrees.",
    )
    parser.add_argument("--min-valid-ratio", type=float, default=0.7)
    return parser.parse_args()


def load_yaml_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Reading --config requires pyyaml.") from exc
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def config_value(config: dict[str, Any], dotted: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current


def resolve_path(project_root: Path, value: str | Path | None) -> Optional[Path]:
    if value is None or not str(value).strip():
        return None
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def parse_date(text: str) -> str:
    match = re.search(r"(20\d{6})T", text)
    if match:
        raw = match.group(1)
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    match = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", text)
    return "-".join(match.groups()) if match else ""


def parse_relative_orbit(safe_dir: Path) -> str:
    manifest = safe_dir / "manifest.safe"
    if not manifest.exists():
        return ""
    try:
        root = ET.parse(manifest).getroot()
        for element in root.iter():
            if element.tag.rsplit("}", 1)[-1] == "relativeOrbitNumber":
                return (element.text or "").strip()
    except Exception:
        return ""
    return ""


def fmt(value: Any, digits: int = 6) -> Any:
    try:
        number = float(value)
        return round(number, digits) if np.isfinite(number) else ""
    except (TypeError, ValueError):
        return ""


def load_target_geometries(
    parcels_path: Path,
    aoi_path: Optional[Path],
    crop_column: str,
    crop_type_code: int,
):
    try:
        import geopandas as gpd
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("This script requires geopandas and pandas.") from exc

    if not parcels_path.exists():
        raise FileNotFoundError(f"Parcel file not found: {parcels_path}")

    parcels = gpd.read_file(parcels_path)
    parcels = parcels[parcels.geometry.notna() & ~parcels.geometry.is_empty].copy()
    if parcels.crs is None:
        raise ValueError(f"Parcel CRS is missing: {parcels_path}")
    if crop_column not in parcels.columns:
        raise ValueError(
            f"Column '{crop_column}' is missing. Available: {list(parcels.columns)}"
        )

    crop_values = parcels[crop_column]
    numeric_values = np.asarray(pd.to_numeric(crop_values, errors="coerce"))
    target = parcels[numeric_values == float(crop_type_code)].copy()
    before_aoi = len(target)

    if aoi_path is not None:
        if not aoi_path.exists():
            raise FileNotFoundError(f"AOI file not found: {aoi_path}")
        aoi = gpd.read_file(aoi_path)
        aoi = aoi[aoi.geometry.notna() & ~aoi.geometry.is_empty].copy()
        if aoi.crs is None:
            raise ValueError(f"AOI CRS is missing: {aoi_path}")
        if aoi.crs != target.crs:
            aoi = aoi.to_crs(target.crs)
        aoi_union = aoi.union_all() if hasattr(aoi, "union_all") else aoi.unary_union
        target = target[target.intersects(aoi_union)].copy()
        if len(target):
            target["geometry"] = target.geometry.intersection(aoi_union)
            target = target[
                target.geometry.notna() & ~target.geometry.is_empty
            ].copy()

    if target.empty:
        raise ValueError(
            f"No target parcels remain for {crop_column}={crop_type_code}."
        )
    if not bool(target.geometry.is_valid.all()):
        target["geometry"] = target.geometry.make_valid()

    metadata = {
        "parcels_path": str(parcels_path),
        "aoi_path": str(aoi_path) if aoi_path else "",
        "crop_column": crop_column,
        "crop_type_code": crop_type_code,
        "total_parcels": int(len(parcels)),
        "target_parcels_before_aoi": int(before_aoi),
        "target_parcels_after_aoi": int(len(target)),
        "crs": str(target.crs),
    }
    return target[["geometry"]].copy(), metadata


def target_bbox_wkt(target_gdf, buffer_degrees: float) -> str:
    target_wgs84 = target_gdf.to_crs("EPSG:4326")
    west, south, east, north = target_wgs84.total_bounds
    west -= buffer_degrees
    south -= buffer_degrees
    east += buffer_degrees
    north += buffer_degrees
    return (
        f"POLYGON (({west:.8f} {south:.8f}, {east:.8f} {south:.8f}, "
        f"{east:.8f} {north:.8f}, {west:.8f} {north:.8f}, "
        f"{west:.8f} {south:.8f}))"
    )


def find_snap_gpt(explicit_path: Optional[Path]) -> Optional[Path]:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(explicit_path)
    if os.environ.get("SNAP_GPT"):
        candidates.append(Path(os.environ["SNAP_GPT"]))
    command = shutil.which("gpt.exe") or shutil.which("gpt")
    if command:
        candidates.append(Path(command))
    candidates.extend(
        [
            Path(r"C:\Program Files\SNAP\bin\gpt.exe"),
            Path(r"C:\Program Files\snap\bin\gpt.exe"),
            Path(r"C:\Program Files\esa-snap\bin\gpt.exe"),
        ]
    )
    return next((path.resolve() for path in candidates if path.is_file()), None)


def find_scenes(s1_dir: Path, relative_orbit: Optional[int]) -> list[dict[str, Any]]:
    scenes: list[dict[str, Any]] = []
    for safe_dir in sorted(s1_dir.glob("**/*.SAFE")):
        # Windows path matching is case-insensitive, so manifest.safe can also
        # match the *.SAFE pattern. Only product directories are scenes.
        if not safe_dir.is_dir():
            continue
        orbit = parse_relative_orbit(safe_dir)
        if relative_orbit is not None and orbit and int(orbit) != relative_orbit:
            continue
        scenes.append(
            {
                "date": parse_date(safe_dir.name),
                "scene_name": safe_dir.name,
                "safe_dir": safe_dir,
                "relative_orbit": orbit,
            }
        )
    return sorted(scenes, key=lambda row: (row["date"], row["scene_name"]))


def processed_path(processed_dir: Path, scene_name: str) -> Path:
    stem = scene_name[:-5] if scene_name.endswith(".SAFE") else scene_name
    return processed_dir / f"{stem}_Sigma0_dB.tif"


def identify_vv_vh_bands(dataset) -> tuple[int, int]:
    descriptions = [str(value or "").lower() for value in dataset.descriptions]
    vv = next((index + 1 for index, name in enumerate(descriptions) if "vv" in name), None)
    vh = next((index + 1 for index, name in enumerate(descriptions) if "vh" in name), None)
    if vv is not None and vh is not None:
        return vv, vh
    if dataset.count >= 2:
        return 1, 2
    raise ValueError("Processed raster must contain both VV and VH bands.")


def validate_processed(path: Path) -> tuple[bool, str]:
    if not path.exists() or path.stat().st_size < 4096:
        return False, "file is missing or too small"
    try:
        with rasterio.open(path) as dataset:
            identify_vv_vh_bands(dataset)
            if dataset.crs is None:
                return False, "CRS is missing"
            if dataset.width <= 0 or dataset.height <= 0:
                return False, "raster dimensions are invalid"
            sample = dataset.read(
                1,
                out_shape=(1, min(64, dataset.height), min(64, dataset.width)),
                masked=True,
            )
            if sample.count() == 0:
                return False, "raster contains no valid sample pixels"
    except Exception as exc:
        return False, str(exc)
    return True, ""


def run_snap(
    gpt_path: Path,
    graph_path: Path,
    scene: dict[str, Any],
    output_path: Path,
    geo_region: str,
    target_epsg: int,
    pixel_spacing: float,
) -> tuple[bool, str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    log_path = output_path.with_suffix(".snap.log")
    source = Path(scene["safe_dir"]) / "manifest.safe"
    command = [
        str(gpt_path),
        str(graph_path),
        f"-Psource={source}",
        f"-Poutput={output_path}",
        f"-PgeoRegion={geo_region}",
        f"-PmapProjection=EPSG:{target_epsg}",
        f"-PpixelSpacing={pixel_spacing}",
        "-c",
        "2048M",
        "-q",
        "4",
    ]
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    log_text = (
        f"COMMAND: {' '.join(command)}\n\nSTDOUT:\n{result.stdout}"
        f"\n\nSTDERR:\n{result.stderr}"
    )
    log_path.write_text(log_text, encoding="utf-8")
    if result.returncode != 0:
        return False, f"SNAP GPT failed with exit code {result.returncode}; see {log_path}"

    valid, reason = validate_processed(output_path)
    if not valid:
        return False, f"SNAP output validation failed: {reason}; see {log_path}"
    return True, ""


def target_shapes(target_gdf, crs) -> list[dict[str, Any]]:
    gdf = target_gdf.to_crs(crs) if target_gdf.crs != crs else target_gdf
    return [
        geometry.__geo_interface__
        for geometry in gdf.geometry
        if geometry is not None and not geometry.is_empty
    ]


def summarize(values: np.ndarray, valid: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    selected = values[target & valid & np.isfinite(values)]
    count = int(selected.size)
    total = int(np.count_nonzero(target))
    return {
        "mean": float(np.mean(selected)) if count else np.nan,
        "median": float(np.median(selected)) if count else np.nan,
        "std": float(np.std(selected)) if count else np.nan,
        "p10": float(np.percentile(selected, 10)) if count else np.nan,
        "p90": float(np.percentile(selected, 90)) if count else np.nan,
        "min": float(np.min(selected)) if count else np.nan,
        "max": float(np.max(selected)) if count else np.nan,
        "valid_pixel_count": count,
        "total_pixel_count": total,
        "valid_pixel_ratio": count / total if total else np.nan,
        "valid_pixel_ratio_pct": count / total * 100.0 if total else np.nan,
    }


def quality_tag(valid_ratio: float, min_valid_ratio: float) -> str:
    if not np.isfinite(valid_ratio) or valid_ratio < 0.4:
        return "drop_or_fill"
    if valid_ratio < min_valid_ratio:
        return "review"
    return "keep"


def process_features(
    scene: dict[str, Any],
    raster_path: Path,
    target_gdf,
    min_valid_ratio: float,
) -> dict[str, Any]:
    with rasterio.open(raster_path) as dataset:
        vv_band, vh_band = identify_vv_vh_bands(dataset)
        shapes = target_shapes(target_gdf, dataset.crs)
        if not shapes:
            raise ValueError("No target geometries are available in raster CRS.")
        data, transform = mask(
            dataset,
            shapes,
            indexes=[vv_band, vh_band],
            crop=True,
            filled=True,
            nodata=np.nan,
        )
        target = geometry_mask(
            shapes,
            out_shape=data.shape[1:],
            transform=transform,
            invert=True,
            all_touched=False,
        )
        pixel_size_x = abs(float(dataset.transform.a))
        pixel_size_y = abs(float(dataset.transform.e))
        raster_crs = str(dataset.crs)

    vv_db = data[0].astype(np.float32)
    vh_db = data[1].astype(np.float32)
    valid = (
        target
        & np.isfinite(vv_db)
        & np.isfinite(vh_db)
        & (vv_db >= -60.0)
        & (vv_db <= 40.0)
        & (vh_db >= -60.0)
        & (vh_db <= 40.0)
    )
    difference_db = vv_db - vh_db
    with np.errstate(over="ignore", invalid="ignore"):
        ratio_linear = np.power(10.0, difference_db / 10.0)

    vv_stats = summarize(vv_db, valid, target)
    vh_stats = summarize(vh_db, valid, target)
    difference_stats = summarize(difference_db, valid, target)
    ratio_stats = summarize(ratio_linear, valid, target)
    effective_ratio = min(
        float(vv_stats["valid_pixel_ratio"]),
        float(vh_stats["valid_pixel_ratio"]),
        float(difference_stats["valid_pixel_ratio"]),
        float(ratio_stats["valid_pixel_ratio"]),
    )

    row = {
        "date": scene["date"],
        "scene_name": scene["scene_name"],
        "relative_orbit": scene["relative_orbit"],
        "status": "ok",
        "quality_tag": quality_tag(effective_ratio, min_valid_ratio),
        "processed_path": str(raster_path),
        "raster_crs": raster_crs,
        "pixel_size_x": fmt(pixel_size_x),
        "pixel_size_y": fmt(pixel_size_y),
        "effective_pixel_ratio": fmt(effective_ratio),
        "effective_pixel_ratio_pct": fmt(effective_ratio * 100.0),
    }
    for prefix, stats in (
        ("vv_db", vv_stats),
        ("vh_db", vh_stats),
        ("vv_minus_vh_db", difference_stats),
        ("vv_div_vh_linear", ratio_stats),
    ):
        for statistic in ("mean", "median", "std", "p10", "p90", "min", "max"):
            row[f"{prefix}_{statistic}"] = fmt(stats[statistic])
        row[f"{prefix}_valid_pixel_count"] = stats["valid_pixel_count"]
        row[f"{prefix}_total_pixel_count"] = stats["total_pixel_count"]
        row[f"{prefix}_valid_pixel_ratio"] = fmt(stats["valid_pixel_ratio"])
        row[f"{prefix}_valid_pixel_ratio_pct"] = fmt(
            stats["valid_pixel_ratio_pct"]
        )
    return row


def add_temporal_features(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows.sort(key=lambda row: (row.get("date", ""), row.get("scene_name", "")))
    previous_by_orbit: dict[str, dict[str, Any]] = {}
    change_fields = (
        "vv_db_mean_change",
        "vh_db_mean_change",
        "vv_minus_vh_db_mean_change",
        "vv_db_mean_change_rate_per_day",
        "vh_db_mean_change_rate_per_day",
        "vv_minus_vh_db_mean_change_rate_per_day",
    )
    for row in rows:
        row["days_from_prev"] = ""
        for field in change_fields:
            row[field] = ""
        row["wetness_signal"] = ""
        row["structure_change_signal"] = ""
        if row.get("status") != "ok" or not row.get("date"):
            continue

        orbit = str(row.get("relative_orbit", "unknown"))
        previous = previous_by_orbit.get(orbit)
        if previous is not None:
            current_date = datetime.strptime(row["date"], "%Y-%m-%d").date()
            previous_date = datetime.strptime(previous["date"], "%Y-%m-%d").date()
            days = (current_date - previous_date).days
            if days > 0:
                vv_change = float(row["vv_db_mean"]) - float(previous["vv_db_mean"])
                vh_change = float(row["vh_db_mean"]) - float(previous["vh_db_mean"])
                diff_change = float(row["vv_minus_vh_db_mean"]) - float(
                    previous["vv_minus_vh_db_mean"]
                )
                row.update(
                    {
                        "days_from_prev": days,
                        "vv_db_mean_change": fmt(vv_change),
                        "vh_db_mean_change": fmt(vh_change),
                        "vv_minus_vh_db_mean_change": fmt(diff_change),
                        "vv_db_mean_change_rate_per_day": fmt(vv_change / days),
                        "vh_db_mean_change_rate_per_day": fmt(vh_change / days),
                        "vv_minus_vh_db_mean_change_rate_per_day": fmt(
                            diff_change / days
                        ),
                        "wetness_signal": (
                            "possible_wetter_surface"
                            if vv_change >= 1.0 and diff_change >= 0.5
                            else "possible_drier_surface"
                            if vv_change <= -1.0 and diff_change <= -0.5
                            else "stable_or_uncertain"
                        ),
                        "structure_change_signal": (
                            "possible_canopy_structure_change"
                            if abs(vh_change) >= 1.0
                            else "stable_or_uncertain"
                        ),
                    }
                )
        previous_by_orbit[orbit] = row
    return rows


def output_fieldnames() -> list[str]:
    fields = [
        "date",
        "scene_name",
        "relative_orbit",
        "status",
        "quality_tag",
        "target_crop_type_code",
        "target_parcels_after_aoi",
        "processed_path",
        "raster_crs",
        "pixel_size_x",
        "pixel_size_y",
        "effective_pixel_ratio",
        "effective_pixel_ratio_pct",
    ]
    for prefix in ("vv_db", "vh_db", "vv_minus_vh_db", "vv_div_vh_linear"):
        fields.extend(
            [
                f"{prefix}_mean",
                f"{prefix}_median",
                f"{prefix}_std",
                f"{prefix}_p10",
                f"{prefix}_p90",
                f"{prefix}_min",
                f"{prefix}_max",
                f"{prefix}_valid_pixel_count",
                f"{prefix}_total_pixel_count",
                f"{prefix}_valid_pixel_ratio",
                f"{prefix}_valid_pixel_ratio_pct",
            ]
        )
    fields.extend(
        [
            "days_from_prev",
            "vv_db_mean_change",
            "vh_db_mean_change",
            "vv_minus_vh_db_mean_change",
            "vv_db_mean_change_rate_per_day",
            "vh_db_mean_change_rate_per_day",
            "vv_minus_vh_db_mean_change_rate_per_day",
            "wetness_signal",
            "structure_change_signal",
            "parcels_path",
            "aoi_path",
            "error",
        ]
    )
    return fields


def main() -> int:
    args = parse_args()
    if args.pixel_spacing <= 0:
        raise ValueError("--pixel-spacing must be greater than zero.")
    if args.aoi_buffer_degrees < 0:
        raise ValueError("--aoi-buffer-degrees cannot be negative.")
    if not 0.0 <= args.min_valid_ratio <= 1.0:
        raise ValueError("--min-valid-ratio must be between 0 and 1.")

    config = load_yaml_config(args.config)
    parcels_path = args.parcels or resolve_path(
        args.project_root, config_value(config, "paths.parcels")
    )
    aoi_path = args.aoi or resolve_path(
        args.project_root, config_value(config, "paths.aoi")
    )
    crop_type_code = (
        args.crop_type_code
        if args.crop_type_code is not None
        else config_value(config, "crop.crop_type_code")
    )
    relative_orbit = (
        args.relative_orbit
        if args.relative_orbit is not None
        else config_value(config, "s1.preferred_relative_orbit")
    )
    if parcels_path is None or crop_type_code is None:
        raise ValueError("Parcel path and crop type code are required.")
    crop_type_code = int(crop_type_code)
    relative_orbit = int(relative_orbit) if relative_orbit is not None else None

    print("Target settings:")
    print(f"  s1_dir: {args.s1_dir}")
    print(f"  processed_dir: {args.processed_dir}")
    print(f"  parcels: {parcels_path}")
    print(f"  aoi: {aoi_path}")
    print(f"  crop_type_code: {crop_type_code}")
    print(f"  relative_orbit: {relative_orbit}")
    print(f"  output: {args.out}")

    target_gdf, target_meta = load_target_geometries(
        parcels_path, aoi_path, args.crop_column, crop_type_code
    )
    print("Target parcel summary:")
    print(json.dumps(target_meta, ensure_ascii=False, indent=2))

    scenes = find_scenes(args.s1_dir, relative_orbit)
    if not scenes:
        raise FileNotFoundError(f"No matching .SAFE scenes found in {args.s1_dir}")
    print(f"Found {len(scenes)} matching S1 scenes.")

    args.processed_dir.mkdir(parents=True, exist_ok=True)
    graph_path = args.processed_dir / "s1_grd_aoi_preprocess_graph.xml"
    graph_path.write_text(SNAP_GRAPH, encoding="utf-8")
    geo_region = target_bbox_wkt(target_gdf, args.aoi_buffer_degrees)
    gpt_path = None if args.preprocessed_only else find_snap_gpt(args.snap_gpt)

    cache_states = {
        scene["scene_name"]: validate_processed(
            processed_path(args.processed_dir, scene["scene_name"])
        )[0]
        for scene in scenes
    }
    needs_snap = args.overwrite_preprocessed or not all(cache_states.values())
    if needs_snap and not args.preprocessed_only and gpt_path is None:
        raise RuntimeError(
            "SNAP gpt.exe was not found. Install ESA SNAP with Sentinel-1 Toolbox, "
            "then pass --snap-gpt, set SNAP_GPT, or add SNAP bin to PATH. "
            "Use --preprocessed-only only when valid cached GeoTIFFs already exist."
        )
    if gpt_path:
        print(f"SNAP GPT: {gpt_path}")

    rows: list[dict[str, Any]] = []
    for index, scene in enumerate(scenes, start=1):
        print(
            f"[{index}/{len(scenes)}] {scene['date']} {scene['scene_name']}",
            flush=True,
        )
        output_path = processed_path(args.processed_dir, scene["scene_name"])
        valid_cache, cache_reason = validate_processed(output_path)
        error = ""

        if args.overwrite_preprocessed or not valid_cache:
            if args.preprocessed_only:
                error = f"Valid preprocessed cache is unavailable: {cache_reason}"
            else:
                print("  Running SNAP preprocessing...", flush=True)
                success, error = run_snap(
                    gpt_path,
                    graph_path,
                    scene,
                    output_path,
                    geo_region,
                    args.target_epsg,
                    args.pixel_spacing,
                )
                valid_cache = success
        else:
            print("  Using cached preprocessed raster.", flush=True)

        if valid_cache:
            try:
                rows.append(
                    process_features(
                        scene, output_path, target_gdf, args.min_valid_ratio
                    )
                )
                continue
            except Exception as exc:
                error = f"Feature calculation failed: {exc}"

        rows.append(
            {
                "date": scene["date"],
                "scene_name": scene["scene_name"],
                "relative_orbit": scene["relative_orbit"],
                "status": "error",
                "processed_path": str(output_path),
                "error": error[:1200],
            }
        )

    rows = add_temporal_features(rows)
    for row in rows:
        row["target_crop_type_code"] = crop_type_code
        row["target_parcels_after_aoi"] = target_meta[
            "target_parcels_after_aoi"
        ]
        row["parcels_path"] = target_meta["parcels_path"]
        row["aoi_path"] = target_meta["aoi_path"]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=output_fieldnames(), extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

    successful = sum(row.get("status") == "ok" for row in rows)
    print(f"Done. Successful scenes: {successful}/{len(rows)}")
    print(f"Output CSV: {args.out}")
    return 0 if successful else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        raise SystemExit(1)
