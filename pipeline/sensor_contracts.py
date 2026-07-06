"""Sensor contracts used by the numbered pipeline.

The pipeline should depend on semantic bands, not on a specific platform. Today
the available source is Sentinel-2; later UAV multispectral imagery can provide
the same semantic bands through a different adapter.
"""

from __future__ import annotations

from dataclasses import dataclass


SEMANTIC_OPTICAL_BANDS = [
    "blue",
    "green",
    "red",
    "rededge",
    "nir",
    "swir",
]

MINIMUM_UAV_MULTISPECTRAL_BANDS = ["red", "rededge", "nir"]
RECOMMENDED_UAV_MULTISPECTRAL_BANDS = ["blue", "green", "red", "rededge", "nir"]

TYPICAL_UAV_CENTER_WAVELENGTH_NM = {
    "blue": 450,
    "green": 560,
    "red": 650,
    "rededge": 730,
    "nir": 840,
}

UAV_PREPROCESSING_REQUIREMENTS = [
    "radiometric_correction",
    "band_coregistration",
    "orthomosaic",
    "spatial_coregistration",
]

REQUIRED_INDEX_BANDS = {
    "ndvi": ["nir", "red"],
    "ndwi": ["green", "nir"],
    "evi": ["nir", "red", "blue"],
    "ndre": ["nir", "rededge"],
}

OPTIONAL_INDEX_BANDS = {
    "nbr": ["nir", "swir"],
    "lswi": ["nir", "swir"],
}


@dataclass(frozen=True)
class SensorContract:
    name: str
    source_type: str
    native_resolution_m: float | None
    supported_semantic_bands: list[str]
    minimum_semantic_bands: list[str]
    recommended_semantic_bands: list[str]
    center_wavelength_nm: dict[str, int]
    preprocessing_requirements: list[str]
    notes: str


SENSOR_CONTRACTS = {
    "sentinel2": SensorContract(
        name="sentinel2",
        source_type="satellite",
        native_resolution_m=10.0,
        supported_semantic_bands=["blue", "green", "red", "rededge", "nir", "swir"],
        minimum_semantic_bands=["red", "rededge", "nir"],
        recommended_semantic_bands=["blue", "green", "red", "rededge", "nir", "swir"],
        center_wavelength_nm={},
        preprocessing_requirements=["cloud_masking", "spatial_coregistration", "temporal_compositing"],
        notes="Current fallback data source. Uses public Sentinel-2 L2A COG assets.",
    ),
    "uav_multispectral": SensorContract(
        name="uav_multispectral",
        source_type="uav",
        native_resolution_m=None,
        supported_semantic_bands=["blue", "green", "red", "rededge", "nir"],
        minimum_semantic_bands=MINIMUM_UAV_MULTISPECTRAL_BANDS,
        recommended_semantic_bands=RECOMMENDED_UAV_MULTISPECTRAL_BANDS,
        center_wavelength_nm=TYPICAL_UAV_CENTER_WAVELENGTH_NM,
        preprocessing_requirements=UAV_PREPROCESSING_REQUIREMENTS,
        notes=(
            "Final target data source. SWIR is usually unavailable on UAV multispectral sensors, "
            "so SWIR-based features must be optional or replaced."
        ),
    ),
}


def validate_required_indices(sensor_name: str) -> list[str]:
    contract = SENSOR_CONTRACTS[sensor_name]
    available = set(contract.supported_semantic_bands)
    missing_indices = []

    for index_name, required_bands in REQUIRED_INDEX_BANDS.items():
        if not set(required_bands).issubset(available):
            missing_indices.append(index_name)

    return missing_indices
