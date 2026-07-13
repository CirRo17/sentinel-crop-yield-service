"""Sentinel 数据源配置。"""

SENTINEL_COLLECTION = "sentinel-2-l2a"

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

# Sentinel-2 场景分类层中需要剔除的无效、云、云影和雪等类别。
CLOUD_SCL_VALUES = {0, 1, 3, 8, 9, 10, 11}
