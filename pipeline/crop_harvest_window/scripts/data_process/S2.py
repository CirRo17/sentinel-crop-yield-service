"""Sentinel-2 L2A scene-level quality screening.

This script scans SAFE products under data/S2, reads each MTD_MSIL2A.xml,
checks key spectral/quality bands, and writes a quality inventory for the
first-stage harvest-window feature selection workflow.

Usage:
    python 11.py
    python 11.py --s2-dir data/S2 --output-dir data/output/s2_quality
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET


DEFAULT_S2_DIR = Path("data/source/sentinel2")
DEFAULT_OUTPUT_DIR = Path("data/exported/harvest_window")

KEEP_THRESHOLD = 10.0
REVIEW_THRESHOLD = 30.0

KEY_BANDS = {
    "has_b02": r"_B02_10m\.jp2$",
    "has_b03": r"_B03_10m\.jp2$",
    "has_b04": r"_B04_10m\.jp2$",
    "has_b08": r"_B08_10m\.jp2$",
    "has_b11": r"_B11_20m\.jp2$",
    "has_b12": r"_B12_20m\.jp2$",
    "has_scl": r"_SCL_20m\.jp2$",
    "has_cldprb": r"MSK_CLDPRB_20m\.jp2$",
}


@dataclass(frozen=True)
class SceneQuality:
    scene_id: str
    safe_name: str
    date: str
    satellite: str
    tile: str
    processing_baseline: str
    cloud_pct: float
    land_cloud_pct: float
    cloud_shadow_pct: float
    nodata_pct: float
    saturated_defective_pct: float
    degraded_msi_pct: float
    jp2_count: int
    r10m_jp2_count: int
    r20m_jp2_count: int
    has_b02: bool
    has_b03: bool
    has_b04: bool
    has_b08: bool
    has_b11: bool
    has_b12: bool
    has_scl: bool
    has_cldprb: bool
    quality_score: float
    first_pass: str
    notes: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a scene-level quality inventory for Sentinel-2 L2A SAFE products."
    )
    parser.add_argument(
        "--s2-dir",
        type=Path,
        default=DEFAULT_S2_DIR,
        help=f"Directory containing *.SAFE products. Default: {DEFAULT_S2_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for CSV/JSON/Markdown outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--keep-threshold",
        type=float,
        default=KEEP_THRESHOLD,
        help="Scenes with cloud + shadow percentage <= this value are tagged keep.",
    )
    parser.add_argument(
        "--review-threshold",
        type=float,
        default=REVIEW_THRESHOLD,
        help="Scenes above keep threshold and <= this value are tagged review.",
    )
    return parser.parse_args()


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def xml_text(root: ET.Element, tag_name: str, default: str = "") -> str:
    for element in root.iter():
        if local_name(element.tag) == tag_name:
            return (element.text or "").strip()
    return default


def xml_float(root: ET.Element, tag_name: str, default: float = 0.0) -> float:
    value = xml_text(root, tag_name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def find_safe_dirs(s2_dir: Path) -> list[Path]:
    if not s2_dir.exists():
        raise FileNotFoundError(f"S2 directory does not exist: {s2_dir}")
    return sorted(path for path in s2_dir.iterdir() if path.is_dir() and path.suffix == ".SAFE")


def parse_safe_name(safe_dir: Path) -> tuple[str, str, str]:
    parts = safe_dir.name.split("_")
    satellite = parts[1] if len(parts) > 1 else ""
    date_match = re.search(r"_(\d{8})T", safe_dir.name)
    tile_match = re.search(r"_(T\d{2}[A-Z]{3})_", safe_dir.name)
    date = (
        datetime.strptime(date_match.group(1), "%Y%m%d").date().isoformat()
        if date_match
        else ""
    )
    tile = tile_match.group(1) if tile_match else ""
    return satellite, date, tile


def band_presence(jp2_files: Iterable[Path]) -> dict[str, bool]:
    names = [path.name for path in jp2_files]
    return {
        key: any(re.search(pattern, name, flags=re.IGNORECASE) for name in names)
        for key, pattern in KEY_BANDS.items()
    }


def classify_scene(score: float, keep_threshold: float, review_threshold: float) -> str:
    if score <= keep_threshold:
        return "keep"
    if score <= review_threshold:
        return "review"
    return "drop_or_fill"


def build_notes(scene: SceneQuality, expected_jp2_count: int = 68) -> str:
    notes: list[str] = []
    if scene.jp2_count != expected_jp2_count:
        notes.append(f"JP2 count is {scene.jp2_count}, expected {expected_jp2_count}")
    missing = [
        field.replace("has_", "").upper()
        for field in KEY_BANDS
        if not getattr(scene, field)
    ]
    if missing:
        notes.append("missing key layers: " + ", ".join(missing))
    if scene.nodata_pct >= 20.0:
        notes.append("high full-tile nodata; verify AOI coverage later")
    return "; ".join(notes)


def scan_scene(
    safe_dir: Path,
    keep_threshold: float,
    review_threshold: float,
) -> SceneQuality:
    metadata_path = safe_dir / "MTD_MSIL2A.xml"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata XML: {metadata_path}")

    root = ET.parse(metadata_path).getroot()
    satellite, fallback_date, tile = parse_safe_name(safe_dir)
    product_start = xml_text(root, "PRODUCT_START_TIME")
    date = product_start[:10] if product_start else fallback_date
    jp2_files = sorted(safe_dir.rglob("*.jp2"))
    presence = band_presence(jp2_files)

    cloud_pct = xml_float(root, "Cloud_Coverage_Assessment")
    cloud_shadow_pct = xml_float(root, "CLOUD_SHADOW_PERCENTAGE")
    quality_score = cloud_pct + cloud_shadow_pct
    first_pass = classify_scene(quality_score, keep_threshold, review_threshold)

    scene = SceneQuality(
        scene_id=safe_dir.name.split("_", 1)[0],
        safe_name=safe_dir.name,
        date=date,
        satellite=satellite,
        tile=tile,
        processing_baseline=xml_text(root, "PROCESSING_BASELINE"),
        cloud_pct=round(cloud_pct, 6),
        land_cloud_pct=round(xml_float(root, "CLOUDY_PIXEL_OVER_LAND_PERCENTAGE"), 6),
        cloud_shadow_pct=round(cloud_shadow_pct, 6),
        nodata_pct=round(xml_float(root, "NODATA_PIXEL_PERCENTAGE"), 6),
        saturated_defective_pct=round(
            xml_float(root, "SATURATED_DEFECTIVE_PIXEL_PERCENTAGE"), 6
        ),
        degraded_msi_pct=round(xml_float(root, "DEGRADED_MSI_DATA_PERCENTAGE"), 6),
        jp2_count=len(jp2_files),
        r10m_jp2_count=sum("\\R10m\\" in str(path) for path in jp2_files),
        r20m_jp2_count=sum("\\R20m\\" in str(path) for path in jp2_files),
        quality_score=round(quality_score, 6),
        first_pass=first_pass,
        notes="",
        **presence,
    )
    return SceneQuality(**{**asdict(scene), "notes": build_notes(scene)})


def write_csv(rows: list[SceneQuality], output_path: Path) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(SceneQuality.__dataclass_fields__)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_json(
    rows: list[SceneQuality],
    output_path: Path,
    keep_threshold: float,
    review_threshold: float,
) -> None:
    payload = {
        "summary": summarize(rows, keep_threshold, review_threshold),
        "scenes": [asdict(row) for row in rows],
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def summarize(
    rows: list[SceneQuality],
    keep_threshold: float,
    review_threshold: float,
) -> dict[str, object]:
    counts = {"keep": 0, "review": 0, "drop_or_fill": 0}
    for row in rows:
        counts[row.first_pass] = counts.get(row.first_pass, 0) + 1
    return {
        "scene_count": len(rows),
        "first_pass_counts": counts,
        "date_start": rows[0].date if rows else "",
        "date_end": rows[-1].date if rows else "",
        "tiles": sorted({row.tile for row in rows if row.tile}),
        "thresholds": {
            "keep": f"cloud_pct + cloud_shadow_pct <= {keep_threshold}",
            "review": (
                f"{keep_threshold} < cloud_pct + cloud_shadow_pct <= "
                f"{review_threshold}"
            ),
            "drop_or_fill": f"cloud_pct + cloud_shadow_pct > {review_threshold}",
        },
    }


def write_markdown(
    rows: list[SceneQuality],
    output_path: Path,
    keep_threshold: float,
    review_threshold: float,
) -> None:
    summary = summarize(rows, keep_threshold, review_threshold)
    lines = [
        "# Sentinel-2 Quality Screening Summary",
        "",
        f"- Scene count: {summary['scene_count']}",
        f"- Date range: {summary['date_start']} to {summary['date_end']}",
        f"- Tiles: {', '.join(summary['tiles'])}",
        f"- First-pass counts: {summary['first_pass_counts']}",
        "",
        "| Date | Scene | Cloud % | Shadow % | Score | First pass | Notes |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            "| "
            f"{row.date} | {row.scene_id} | {row.cloud_pct:.2f} | "
            f"{row.cloud_shadow_pct:.2f} | {row.quality_score:.2f} | "
            f"{row.first_pass} | {row.notes} |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_console_summary(
    rows: list[SceneQuality],
    keep_threshold: float,
    review_threshold: float,
) -> None:
    print("S2 scene-level quality screening")
    print("=" * 36)
    for row in rows:
        print(
            f"{row.date}  scene={row.scene_id}  "
            f"score={row.quality_score:5.2f}  tag={row.first_pass}"
        )
    counts = summarize(rows, keep_threshold, review_threshold)["first_pass_counts"]
    print("-" * 36)
    print(
        f"scenes={len(rows)}  keep={counts['keep']}  "
        f"review={counts['review']}  drop_or_fill={counts['drop_or_fill']}"
    )


def main() -> int:
    args = parse_args()
    if args.keep_threshold > args.review_threshold:
        print("ERROR: --keep-threshold must be <= --review-threshold", file=sys.stderr)
        return 2

    safe_dirs = find_safe_dirs(args.s2_dir)
    if not safe_dirs:
        print(f"ERROR: no .SAFE directories found under {args.s2_dir}", file=sys.stderr)
        return 1

    rows = [
        scan_scene(
            safe_dir=safe_dir,
            keep_threshold=args.keep_threshold,
            review_threshold=args.review_threshold,
        )
        for safe_dir in safe_dirs
    ]
    rows.sort(key=lambda row: row.date)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output_dir / "s2_T49RFQ_quality_inventory.csv")
    write_json(
        rows,
        args.output_dir / "s2_T49RFQ_quality_inventory.json",
        args.keep_threshold,
        args.review_threshold,
    )
    write_markdown(
        rows,
        args.output_dir / "s2_T49RFQ_quality_summary.md",
        args.keep_threshold,
        args.review_threshold,
    )

    print_console_summary(rows, args.keep_threshold, args.review_threshold)
    print(f"\nOutputs written to: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
