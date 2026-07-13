"""历史模型和默认特征名配置。"""

from pathlib import Path


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "crop_classification_classifier.joblib"
DEFAULT_MODEL_INFO_PATH = DEFAULT_MODEL_PATH.with_name("crop_classification_model_info.json")

# 早期统计型模型使用的特征名，保留用于兼容旧模型和历史产物。
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
# 该顺序与 Sentinel 特征构建和本地多波段特征构建保持一致。
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
