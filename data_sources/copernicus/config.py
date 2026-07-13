"""Copernicus Data Space 数据源配置。"""

# STAC 搜索端点
COPERNICUS_STAC_URL = "https://catalogue.dataspace.copernicus.eu/stac"

# OData endpoint for product metadata and authenticated product downloads.
COPERNICUS_ODATA_URL = "https://catalogue.dataspace.copernicus.eu/odata/v1"
COPERNICUS_ODATA_DOWNLOAD_URL = "https://download.dataspace.copernicus.eu/odata/v1"

# OAuth2 token 端点
COPERNICUS_TOKEN_URL = (
    "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
    "protocol/openid-connect/token"
)

# Sentinel-2 L2A collection ID
SENTINEL_COLLECTION = "sentinel-2-l2a"

# SAFE 目录中 .jp2 文件 → 语义波段映射
# 用文件名后缀匹配，不区分大小写
SAFE_BAND_MAP = {
    "_b02_10m.jp2": "blue",
    "_b03_10m.jp2": "green",
    "_b04_10m.jp2": "red",
    "_b05_20m.jp2": "rededge",
    "_b08_10m.jp2": "nir",
    "_b11_20m.jp2": "swir",
    "_b12_20m.jp2": "swir22",
    "_scl_20m.jp2": "scl",
}

# 需要排除的 SCL 类别（同 aws_element84/config.py）
CLOUD_SCL_VALUES = {0, 1, 3, 8, 9, 10, 11}

# Sentinel-2 地表反射率缩放系数
S2_SCALE = 10000.0

# 默认云量上限
DEFAULT_MAX_CLOUD = 30.0

# 默认每时相最大场景数
DEFAULT_LIMIT = 8

# token 缓存文件（避免每次运行都重新认证）
DEFAULT_TOKEN_CACHE = None  # 由 auth.py 内部决定路径

# 下载重试次数
MAX_RETRIES = 3
