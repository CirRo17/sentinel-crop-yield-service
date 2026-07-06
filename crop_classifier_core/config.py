from __future__ import annotations

from pathlib import Path


EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"
SENTINEL_COLLECTION = "sentinel-2-l2a"

DEFAULT_MAX_CLOUD = 30.0
DEFAULT_LIMIT = 8
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "crop_classifier.joblib"
DEFAULT_MODEL_INFO_PATH = DEFAULT_MODEL_PATH.with_name("model_info.json")

BAND_ASSETS = {
    "blue": "blue",
    "green": "green",
    "red": "red",
    "rededge1": "rededge1",
    "rededge2": "rededge2",
    "rededge3": "rededge3",
    "nir": "nir",
    "nir08": "nir08",
    "swir16": "swir16",
    "swir22": "swir22",
    "scl": "scl",
}

S2_SCALE = 10000.0

# Sentinel-2 Scene Classification Layer values to ignore.
CLOUD_SCL_VALUES = {0, 1, 3, 8, 9, 10, 11}

MODEL_FEATURE_NAMES = [
    "evi_mean_max",
    "evi_mean_mean",
    "evi_mean_min",
    "evi_mean_std",
    "nbr_mean_max",
    "nbr_mean_mean",
    "nbr_mean_min",
    "nbr_mean_std",
    "ndre_mean_max",
    "ndre_mean_mean",
    "ndre_mean_min",
    "ndre_mean_std",
    "ndvi_amplitude",
    "ndvi_mean_max",
    "ndvi_mean_mean",
    "ndvi_mean_min",
    "ndvi_mean_std",
    "ndwi_mean_max",
    "ndwi_mean_mean",
    "ndwi_mean_min",
    "ndwi_mean_std",
    "scene_count",
    "valid_pixel_ratio_max",
    "valid_pixel_ratio_mean",
    "valid_pixel_ratio_min",
    "valid_pixel_ratio_std",
]

# 像素级标准特征名：6 个基础波段 + 5 个光谱指数。
# 该顺序与 03b 的 Sentinel 单波段输入适配，以及 03d 的本地多波段输入适配保持一致。
PIXEL_FEATURE_NAMES = [
    "blue",
    "green",
    "red",
    "rededge",
    "nir",
    "swir",
    "ndvi",
    "ndwi",
    "evi",
    "ndre",
    "nbr",
]

AGRIFIELDNET_LABELS = {
    0: "Background",
    1: "Wheat",
    2: "Mustard",
    3: "Lentil",
    4: "Green pea",
    5: "Sugarcane",
    6: "Garlic",
    7: "Maize",
    8: "Gram",
    9: "Coriander",
    10: "Potato",
    11: "Bersem",
    12: "Rice",
}

TARGET_LABELS = {
    0: "Others",
    1: "Rice",
    2: "Wheat",
    3: "Maize",
    4: "Rapeseed",
}

VALID_OUTPUT_CLASSES = frozenset(TARGET_LABELS)


def normalize_output_classes(classes):
    """把历史/技术标签归一化到对外公开的 0-4 类别集合。"""
    import numpy as np

    normalized = np.asarray(classes).copy()
    normalized[~np.isin(normalized, list(VALID_OUTPUT_CLASSES))] = 0
    return normalized.astype("uint8")

AGRIFIELDNET_TO_TARGET = {
    0: 0,
    1: 2,
    2: 4,
    3: 0,
    4: 0,
    5: 0,
    6: 0,
    7: 3,
    8: 0,
    9: 0,
    10: 0,
    11: 0,
    12: 1,
}

