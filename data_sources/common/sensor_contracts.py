"""数据源传感器能力约定。

功能型 pipeline 应该依赖 blue、red、nir 这类语义波段，而不是直接绑定
Sentinel-2、无人机或其他具体平台。不同数据源适配器负责把原始波段映射到
这里定义的统一语义波段。
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

# 无人机多光谱通常至少需要这些波段，才能支撑基础植被指数。
MINIMUM_UAV_MULTISPECTRAL_BANDS = ["red", "rededge", "nir"]
RECOMMENDED_UAV_MULTISPECTRAL_BANDS = ["blue", "green", "red", "rededge", "nir"]

# 常见无人机多光谱相机的典型中心波长，仅用于能力描述和数据校验参考。
TYPICAL_UAV_CENTER_WAVELENGTH_NM = {
    "blue": 450,
    "green": 560,
    "red": 650,
    "rededge": 730,
    "nir": 840,
}

# 无人机影像进入功能 pipeline 前，通常需要先完成这些预处理。
UAV_PREPROCESSING_REQUIREMENTS = [
    "radiometric_correction",
    "band_coregistration",
    "orthomosaic",
    "spatial_coregistration",
]

# 当前项目中基础光谱指数所需的语义波段。
REQUIRED_INDEX_BANDS = {
    "ndvi": ["nir", "red"],
    "ndwi": ["green", "nir"],
    "evi": ["nir", "red", "blue"],
    "ndre": ["nir", "rededge"],
}

# 依赖 SWIR 等并非所有数据源都具备的可选指数。
OPTIONAL_INDEX_BANDS = {
    "nbr": ["nir", "swir"],
    "lswi": ["nir", "swir"],
}


@dataclass(frozen=True)
class SensorContract:
    """描述一个数据源或传感器可提供的标准化影像能力。"""

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
        notes="当前可用的卫星数据源，使用公开 Sentinel-2 L2A COG 资产。",
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
            "后续重点支持的数据源。无人机多光谱通常不包含 SWIR，"
            "因此依赖 SWIR 的特征应作为可选项或用其他特征替代。"
        ),
    ),
}


def validate_required_indices(sensor_name: str) -> list[str]:
    """检查指定传感器是否缺少计算基础光谱指数所需的波段。"""

    contract = SENSOR_CONTRACTS[sensor_name]
    available = set(contract.supported_semantic_bands)
    missing_indices = []

    for index_name, required_bands in REQUIRED_INDEX_BANDS.items():
        if not set(required_bands).issubset(available):
            missing_indices.append(index_name)

    return missing_indices
