"""Download ERA5-Land historical weather data from the CDS API.

This helper writes monthly NetCDF files in the layout expected by
pipeline/15_harvest_window.py:

    data/weather/history/YYYYMM/data_0.nc

Example:
    python scripts/data_process/download_era5_land.py \
        --north 30.94 --west 112.03 --south 30.74 --east 112.28 \
        --start-date 2025-01-01 --end-date 2025-06-30

Before downloading, install and configure the CDS API client:
    pip install cdsapi
    https://cds.climate.copernicus.eu/how-to-api
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any


DATASET = "reanalysis-era5-land"
DEFAULT_OUTPUT_DIR = Path("data/input/weather/history")
TARGET_FILENAME = "data_0.nc"

VARIABLES = [
    "2m_temperature",
    "volumetric_soil_water_layer_1",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]

HOURS = [f"{hour:02d}:00" for hour in range(24)]
REQUIRED_NETCDF_KEYS = [
    "valid_time",
    "latitude",
    "longitude",
    "t2m",
    "swvl1",
    "u10",
    "v10",
    "tp",
]


@dataclass(frozen=True)
class MonthlyRequest:
    year: int
    month: int
    start_date: date
    end_date: date
    target_path: Path
    request: dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download ERA5-Land monthly NetCDF files for harvest-window weather history."
    )
    parser.add_argument("--north", type=float, default=30.94, help="CDS area north latitude.")
    parser.add_argument("--west", type=float, default=112.03, help="CDS area west longitude.")
    parser.add_argument("--south", type=float, default=30.74, help="CDS area south latitude.")
    parser.add_argument("--east", type=float, default=112.28, help="CDS area east longitude.")
    parser.add_argument("--start-date", default=2025-2-1, help="Start date, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=2025-2-28, help="End date, YYYY-MM-DD.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download months even when data_0.nc already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print monthly CDS requests without downloading.",
    )
    return parser.parse_args()


def parse_iso_date(value: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date {value!r}; expected YYYY-MM-DD.") from exc


def validate_args(args: argparse.Namespace) -> tuple[date, date]:
    if args.north <= args.south:
        raise ValueError("--north must be greater than --south.")
    if args.east <= args.west:
        raise ValueError("--east must be greater than --west.")

    start_date = parse_iso_date(args.start_date)
    end_date = parse_iso_date(args.end_date)
    if end_date < start_date:
        raise ValueError("--end-date must be on or after --start-date.")
    return start_date, end_date


def month_start(value: date) -> date:
    return value.replace(day=1)


def next_month(value: date) -> date:
    if value.month == 12:
        return value.replace(year=value.year + 1, month=1, day=1)
    return value.replace(month=value.month + 1, day=1)


def month_end(value: date) -> date:
    return next_month(month_start(value)) - timedelta(days=1)


def iter_month_ranges(start_date: date, end_date: date) -> list[tuple[date, date]]:
    ranges: list[tuple[date, date]] = []
    current = month_start(start_date)
    while current <= end_date:
        current_start = max(start_date, current)
        current_end = min(end_date, month_end(current))
        ranges.append((current_start, current_end))
        current = next_month(current)
    return ranges


def day_values(start_date: date, end_date: date) -> list[str]:
    values: list[str] = []
    current = start_date
    while current <= end_date:
        values.append(f"{current.day:02d}")
        current += timedelta(days=1)
    return values


def build_request(
    args: argparse.Namespace,
    start_date: date,
    end_date: date,
) -> dict[str, Any]:
    return {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [f"{start_date.year:04d}"],
        "month": [f"{start_date.month:02d}"],
        "day": day_values(start_date, end_date),
        "time": HOURS,
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [args.north, args.west, args.south, args.east],
    }


def build_monthly_requests(
    args: argparse.Namespace,
    start_date: date,
    end_date: date,
) -> list[MonthlyRequest]:
    requests: list[MonthlyRequest] = []
    for current_start, current_end in iter_month_ranges(start_date, end_date):
        month_name = f"{current_start.year:04d}{current_start.month:02d}"
        target_path = args.output_dir / month_name / TARGET_FILENAME
        requests.append(
            MonthlyRequest(
                year=current_start.year,
                month=current_start.month,
                start_date=current_start,
                end_date=current_end,
                target_path=target_path,
                request=build_request(args, current_start, current_end),
            )
        )
    return requests


def validate_netcdf(path: Path) -> tuple[bool, list[str]]:
    try:
        import h5py
    except ImportError as exc:
        raise RuntimeError("h5py is required to validate downloaded NetCDF files.") from exc

    with h5py.File(path, "r") as dataset:
        keys = set(dataset.keys())
    missing = [key for key in REQUIRED_NETCDF_KEYS if key not in keys]
    return not missing, missing


def format_size(path: Path) -> str:
    size = path.stat().st_size
    if size >= 1024 * 1024:
        return f"{size / (1024 * 1024):.2f} MB"
    if size >= 1024:
        return f"{size / 1024:.2f} KB"
    return f"{size} B"


def print_month_summary(monthly: MonthlyRequest, status: str, validate: bool = True) -> None:
    print(
        f"[{status}] {monthly.start_date} to {monthly.end_date} -> {monthly.target_path}",
        flush=True,
    )
    if monthly.target_path.exists():
        print(f"  size: {format_size(monthly.target_path)}", flush=True)
        if validate:
            ok, missing = validate_netcdf(monthly.target_path)
            if ok:
                print("  validation: ok", flush=True)
            else:
                print(f"  validation: missing keys {missing}", flush=True)


def print_dry_run(monthly_requests: list[MonthlyRequest]) -> None:
    print(f"Dataset: {DATASET}")
    for monthly in monthly_requests:
        print(
            f"\nTarget: {monthly.target_path}\n"
            f"Date range: {monthly.start_date} to {monthly.end_date}\n"
            f"Request:\n{json.dumps(monthly.request, indent=2)}"
        )


def download_months(monthly_requests: list[MonthlyRequest], overwrite: bool) -> None:
    pending: list[MonthlyRequest] = []
    for monthly in monthly_requests:
        if monthly.target_path.exists() and not overwrite:
            print_month_summary(monthly, "skip-existing")
        else:
            pending.append(monthly)

    if not pending:
        return

    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError(
            "cdsapi is not installed. Install it in this environment with: pip install cdsapi"
        ) from exc

    client = cdsapi.Client()
    for monthly in pending:
        monthly.target_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[download] {monthly.start_date} to {monthly.end_date} -> {monthly.target_path}",
            flush=True,
        )
        client.retrieve(DATASET, monthly.request, str(monthly.target_path))
        print_month_summary(monthly, "downloaded")


def main() -> int:
    args = parse_args()
    try:
        start_date, end_date = validate_args(args)
        monthly_requests = build_monthly_requests(args, start_date, end_date)
        if args.dry_run:
            print_dry_run(monthly_requests)
        else:
            download_months(monthly_requests, args.overwrite)
    except Exception as exc:
        print(f"ERROR: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
