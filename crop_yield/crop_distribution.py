from __future__ import annotations

from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from .config import CROP_DISTRIBUTION_DIR, CROP_MODELS, EXTERNAL_CROP_DISTRIBUTION_API, OUTPUT_DIR


def local_crop_raster(crop: str) -> Optional[Path]:
    pattern = CROP_MODELS[crop]["local_glob"]
    files = sorted(CROP_DISTRIBUTION_DIR.glob(pattern))
    return files[0] if files else None


def downloaded_crop_raster(crop: str) -> Optional[Path]:
    crop_path = OUTPUT_DIR / f"external_crop_distribution_{crop}.tif"
    if crop_path.exists():
        return crop_path
    patterns = [
        "external_crop_distribution*.tif",
        "*crop*classification*.tif",
        "*crop*distribution*.tif",
        "*classification*.tif",
    ]
    for pattern in patterns:
        files = sorted(OUTPUT_DIR.glob(pattern))
        if files:
            return files[0]
    return None


def _looks_like_tiff(response: requests.Response) -> bool:
    ctype = response.headers.get("content-type", "").lower()
    if "tiff" in ctype or "geotiff" in ctype:
        return True
    return response.content[:4] in (b"II*\x00", b"MM\x00*")


def _vector_suffix(response: requests.Response, url: str) -> Optional[str]:
    ctype = response.headers.get("content-type", "").lower()
    lower_url = url.lower()
    if "zip" in ctype or lower_url.endswith(".zip"):
        return ".zip"
    if "json" in ctype or lower_url.endswith(".geojson") or lower_url.endswith(".json"):
        return ".geojson"
    if "octet-stream" in ctype and response.content[:2] == b"PK":
        return ".zip"
    return None


def download_external_crop_raster(crop: str, explicit_url: Optional[str] = None) -> Optional[Path]:
    code = CROP_MODELS[crop]["code"]
    cache_path = OUTPUT_DIR / f"external_crop_distribution_{crop}.tif"
    if explicit_url is None and cache_path.exists():
        return cache_path
    candidates = []
    if explicit_url:
        candidates.append(explicit_url)
    base = EXTERNAL_CROP_DISTRIBUTION_API
    candidates.extend(
        [
            urljoin(base, "artifacts/data/output/crop_classification.tif/download"),
            urljoin(base, "artifacts/crop_classification.tif/download"),
            urljoin(base, "artifacts/test_schema_check_classification.tif/download"),
            urljoin(base, "api-predictions/latest/classification"),
            urljoin(base, f"api/tif?crop={crop}"),
            urljoin(base, f"api/tif?crop_code={code}"),
            urljoin(base, f"api/crop-distribution/tif?crop={crop}"),
            urljoin(base, f"api/crop-distribution/tif?crop_code={code}"),
            urljoin(base, f"api/export/tif?crop={crop}"),
            urljoin(base, f"download?crop_code={code}"),
        ]
    )

    for url in candidates:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            if not _looks_like_tiff(response):
                continue
            try:
                cache_path.write_bytes(response.content)
                return cache_path
            except PermissionError:
                if cache_path.exists():
                    return cache_path
                raise
        except requests.RequestException:
            continue
    return None


def download_external_crop_vector(crop: str, explicit_url: Optional[str] = None) -> Optional[Path]:
    code = CROP_MODELS[crop]["code"]
    if explicit_url:
        candidates = [explicit_url]
    else:
        base = EXTERNAL_CROP_DISTRIBUTION_API
        candidates = [
            urljoin(base, f"api/crop-distribution/shp?crop={crop}"),
            urljoin(base, f"api/crop-distribution/shp?crop_code={code}"),
            urljoin(base, f"api/crop-distribution/vector?crop={crop}"),
            urljoin(base, f"api/crop-distribution/vector?crop_code={code}"),
            urljoin(base, f"api/shp?crop={crop}"),
            urljoin(base, f"api/shp?crop_code={code}"),
            urljoin(base, f"api/geojson?crop={crop}"),
            urljoin(base, f"api/geojson?crop_code={code}"),
        ]
    for url in candidates:
        try:
            response = requests.get(url, timeout=20)
            response.raise_for_status()
            suffix = _vector_suffix(response, url)
            if not suffix:
                continue
            path = OUTPUT_DIR / f"external_crop_distribution_{crop}{suffix}"
            path.write_bytes(response.content)
            return path
        except requests.RequestException:
            continue
    return None


def crop_distribution_status() -> dict:
    status = {
        "base_url": EXTERNAL_CROP_DISTRIBUTION_API,
        "reachable": False,
        "local": {},
    }
    try:
        response = requests.get(EXTERNAL_CROP_DISTRIBUTION_API, timeout=5)
        status["reachable"] = response.ok
        status["status_code"] = response.status_code
    except requests.RequestException as exc:
        status["error"] = str(exc)
    for crop in CROP_MODELS:
        path = local_crop_raster(crop)
        status["local"][crop] = str(path) if path else None
    status["downloaded"] = {
        crop: str(path) if (path := downloaded_crop_raster(crop)) else None
        for crop in CROP_MODELS
    }
    return status
