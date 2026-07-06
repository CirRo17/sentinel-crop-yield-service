from pathlib import Path
import os


BASE_DIR = Path(__file__).resolve().parents[1]
S2_DIR = BASE_DIR / "Data" / "S2"
CROP_DISTRIBUTION_DIR = BASE_DIR / "Crop_distribution"
OUTPUT_DIR = BASE_DIR / "outputs"
UPLOAD_DIR = BASE_DIR / "uploads"
GEOB_CACHE_DIR = OUTPUT_DIR / "geoboundaries"

for path in (OUTPUT_DIR, UPLOAD_DIR, GEOB_CACHE_DIR):
    path.mkdir(parents=True, exist_ok=True)


EXTERNAL_CROP_DISTRIBUTION_API = os.getenv(
    "CROP_DISTRIBUTION_API_URL",
    "http://192.168.110.53:8000/",
).rstrip("/") + "/"

EARTH_SEARCH_URL = "https://earth-search.aws.element84.com/v1/search"

CROP_MODELS = {
    "maize": {
        "label": "玉米",
        "code": 3,
        "formula": [77763.098231, -80886.711657, 24626.403283],
        "rmse_kg_ha": 450.0,
        "local_glob": "Maize30m/*.tif",
    },
    "wheat": {
        "label": "小麦",
        "code": 2,
        "formula": [92997.916893, -114110.350163, 38604.301220],
        "rmse_kg_ha": 520.0,
        "local_glob": "Wheat30m/*.tif",
    },
    "rice": {
        "label": "水稻（中稻）",
        "code": 1,
        "formula": [-10128.830316, 13475.878556, 4032.546616],
        "rmse_kg_ha": 380.0,
        "local_glob": "Rice10m/*.tif",
    },
}

INDEX_OPTIONS = {
    "ndvi": "NDVI: 适合成熟期长势和产量回归估算。",
    "lai": "LAI: 基于红边叶绿素指数 CI=B8/B5-1 的半经验幂函数，适合冠层结构分析。",
}

LAI_DEFAULTS = {
    "maize": {"k": 0.44, "m": 0.9},
    "wheat": {"k": 0.40, "m": 0.9},
    "rice": {"k": 0.42, "m": 0.9},
}
