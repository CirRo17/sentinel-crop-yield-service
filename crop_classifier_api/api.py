from __future__ import annotations

import asyncio
import importlib
import json
import math
import shutil
import threading
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import rasterio
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field, ValidationError
import yaml

from crop_classifier_core.config import TARGET_LABELS, normalize_output_classes
from crop_classifier_core.feature_schema import (
    BASE_FEATURE_NAMES,
    base_feature_name,
    feature_prefix,
    require_feature_stack_schema,
)
from crop_classifier_core.spectral import evi, nbr, ndre, ndvi, ndwi

_yield_est = importlib.import_module("pipeline.10_yield_estimation")
CROP_MODELS = _yield_est.CROP_MODELS
CROP_CODE_TO_NAME = _yield_est.CROP_CODE_TO_NAME
estimate_yield = _yield_est.estimate_yield
lai_from_ci = _yield_est.lai_from_ci
uncertainty = _yield_est.uncertainty
attach_raster_majority_to_parcels = importlib.import_module(
    "pipeline.09_parcel_majority"
).attach_raster_majority_to_parcels


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DATA_UPLOADS = ROOT / "data" / "uploads"
API_PREDICTIONS = ROOT / "data" / "output" / "api_predictions"
DATA_OUTPUT = ROOT / "data" / "output"
DATA_EXPORTED = ROOT / "data" / "exported"
MODEL_FILE = ROOT / "models" / "crop_classifier.joblib"
MODEL_INFO_FILE = ROOT / "models" / "model_info.json"
HOME_TEMPLATE = Path(__file__).with_name("home.html")
CLASS_MAPPING_FILE = ROOT / "configs" / "class_mapping.yaml"

task_store: dict[str, dict[str, Any]] = {}
task_lock = threading.Lock()
yield_task_store: dict[str, dict[str, Any]] = {}
yield_task_lock = threading.Lock()


class UploadResponse(BaseModel):
    file_id: str = Field(..., description="Unique uploaded file id.")
    filename: str = Field(..., description="Original filename.")
    size_bytes: int = Field(..., description="Uploaded file size.")


class ParcelUploadResponse(BaseModel):
    parcel_file_id: str = Field(..., description="Unique uploaded parcel dataset id.")
    filename: str = Field(..., description="Original uploaded zip filename.")
    shapefile: str = Field(..., description="Detected .shp path inside extracted upload.")
    size_bytes: int = Field(..., description="Uploaded zip size.")


class InferRequest(BaseModel):
    file_ids: list[str] = Field(..., description="上传的影像 file_id 列表。服务端按文件名中的日期自动分组。")
    parcel_file_id: Optional[str] = Field(None, description="Optional parcel shapefile zip id.")
    red_band: int = Field(3, description="1-based Red band index.")
    nir_band: int = Field(5, description="1-based NIR band index.")
    blue_band: int = Field(1, description="1-based Blue band index. Use 0 if absent.")
    green_band: int = Field(2, description="1-based Green band index. Use 0 if absent.")
    rededge_band: int = Field(4, description="1-based Red Edge band index. Use 0 if absent.")
    swir_band: int = Field(0, description="1-based SWIR band index. Use 0 if absent.")
    reflectance_scale: float = Field(1.0, gt=0, description="Reflectance scale.")
    top_k: int = Field(1, ge=1, le=5, description="Return Top-1 to Top-5 class ranking.")


class InferStartResponse(BaseModel):
    task_id: str = Field(..., description="Inference task id.")
    status: str = Field(default="queued", description="Initial task state.")


class InferTopPrediction(BaseModel):
    class_code: int = Field(..., description="Predicted class code.")
    label: str = Field(..., description="Predicted class label.")
    confidence: float = Field(..., description="Confidence score.")


class InferStatusResponse(BaseModel):
    task_id: str = Field(..., description="Inference task id.")
    status: str = Field(..., description="queued, running, completed, or failed.")
    progress: float = Field(..., description="Estimated task progress from 0 to 100.")
    message: Optional[str] = Field(None, description="Error or status message.")
    valid_pixel_count: Optional[int] = Field(None, description="Valid pixels used in inference.")
    model_features: Optional[list[str]] = Field(None, description="Model features used in this run.")
    top_predictions: Optional[list[InferTopPrediction]] = Field(None, description="Top-k overall class ranking.")
    downloads: Optional[dict[str, str]] = Field(None, description="Download URLs when task is complete.")


class PredictUploadResponse(BaseModel):
    file_id: str = Field(..., description="Uploaded file id.")
    task_id: str = Field(..., description="Inference task id.")
    status: str = Field(..., description="Initial task state.")


# ---------------------------------------------------------------------------
# 估产 (yield) 相关模型
# ---------------------------------------------------------------------------

class YieldEstimateRequest(BaseModel):
    task_id: str = Field(..., description="已完成分类推理的 task_id。")
    index: str = Field("ndvi", description="产量估算所用的植被指数：ndvi 或 lai。")
    yield_function: str = Field("default", description="产量函数类型：default, linear, exponential, power, logarithmic, polynomial。")
    model_coefficients: Optional[list[float]] = Field(None, description="自定义模型系数，覆盖默认值。")
    lai_k: Optional[float] = Field(None, description="LAI 模型参数 k。")
    lai_m: Optional[float] = Field(None, description="LAI 模型参数 m。")


class YieldEstimateResponse(BaseModel):
    yield_task_id: str = Field(..., description="估产任务 ID。")
    status: str = Field(default="queued", description="初始任务状态。")


class YieldCropResult(BaseModel):
    crop_code: int
    crop_name: str
    label: str
    area_ha: float
    pixel_count: int
    mean_yield_kg_ha: Optional[float] = None
    total_yield_kg: float = 0.0
    warning: Optional[str] = None


class YieldStatusResponse(BaseModel):
    yield_task_id: str = Field(..., description="估产任务 ID。")
    status: str = Field(..., description="queued, running, completed, 或 failed。")
    progress: float = Field(..., description="进度 0-100。")
    message: Optional[str] = Field(None, description="状态或错误消息。")
    summary: Optional[dict[str, Any]] = Field(None, description="估产汇总，完成时返回。")
    crops: Optional[list[YieldCropResult]] = Field(None, description="分作物结果。")
    downloads: Optional[dict[str, str]] = Field(None, description="下载链接。")


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def _read_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _file_info(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "name": path.name,
        "path": _relative(path),
        "size_bytes": stat.st_size,
        "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
    }


def _existing_file(*candidates: str) -> Path:
    for candidate in candidates:
        path = ROOT / candidate
        if path.exists():
            return path
    raise HTTPException(status_code=404, detail=f"Artifact not found. Checked: {', '.join(candidates)}")


def _artifact_registry() -> dict[str, Path]:
    registry: dict[str, Path] = {}
    roots = [
        ROOT / "models",
        ROOT / "configs",
        DATA_EXPORTED,
        DATA_OUTPUT,
    ]
    for base in roots:
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if path.is_file():
                key = _relative(path).replace("\\", "/")
                registry[key] = path
    return registry


def _api_prediction_file(job_id: str, suffix: str) -> Path:
    path = API_PREDICTIONS / f"{job_id}_{suffix}"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Prediction artifact not found for job_id: {job_id}")
    return path


def _safe_extract_zip(zip_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    target_root = target_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            member_path = target_root / member.filename
            if target_root not in member_path.resolve().parents:
                raise HTTPException(status_code=400, detail="Invalid zip path traversal entry.")
        archive.extractall(target_root)


def _parcel_shp_from_file_id(parcel_file_id: str) -> Path:
    parcel_dir = DATA_UPLOADS / f"{parcel_file_id}_parcels"
    if not parcel_dir.exists():
        raise HTTPException(status_code=404, detail=f"Parcel upload does not exist: {parcel_file_id}")
    shapefiles = sorted(parcel_dir.rglob("*.shp"))
    if not shapefiles:
        raise HTTPException(status_code=400, detail=f"Parcel upload has no .shp file: {parcel_file_id}")
    if len(shapefiles) > 1:
        raise HTTPException(status_code=400, detail=f"Parcel upload contains multiple .shp files: {parcel_file_id}")
    return shapefiles[0]


def _report_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Report not found: {_relative(path)}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _map_summary(path: Path) -> dict[str, Any]:
    with rasterio.open(path) as src:
        data = src.read(1, masked=False)
        nodata = src.nodata
        normalized = normalize_output_classes(data)
        values, counts = np.unique(normalized, return_counts=True)
        pixel_area = None
        if src.transform:
            pixel_area = abs(src.transform.a * src.transform.e)

        rows = []
        for value, count in zip(values.tolist(), counts.tolist()):
            rows.append(
                {
                    "class_code": int(value),
                    "label": TARGET_LABELS.get(int(value), str(value)),
                    "pixel_count": int(count),
                    "area_square_meters": float(count * pixel_area) if pixel_area else None,
                }
            )

        return {
            "raster": _relative(path),
            "width": src.width,
            "height": src.height,
            "crs": str(src.crs) if src.crs else None,
            "nodata": nodata,
            "counts": rows,
        }


async def _parse_infer_request(request: Request) -> InferRequest:
    content_type = (request.headers.get("content-type") or "").lower()

    try:
        if "application/json" in content_type:
            payload = await request.json()
        elif "application/x-www-form-urlencoded" in content_type or "multipart/form-data" in content_type:
            form = await request.form()
            payload = dict(form)
        else:
            try:
                payload = await request.json()
            except Exception:
                form = await request.form()
                payload = dict(form)

        return InferRequest.model_validate(payload)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON body: {exc.msg}") from exc


def _load_model() -> tuple[Any, list[str], dict[str, Any]]:
    if not MODEL_FILE.exists() or not MODEL_INFO_FILE.exists():
        raise HTTPException(status_code=503, detail="Model artifacts are missing. Run pipeline step 05 first.")

    model = joblib.load(MODEL_FILE)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    if hasattr(model, "named_steps"):
        rf_step = model.named_steps.get("rf")
        if rf_step is not None and hasattr(rf_step, "n_jobs"):
            rf_step.n_jobs = 1

    with open(MODEL_INFO_FILE, encoding="utf-8") as f:
        model_info = json.load(f)

    feature_names = [str(name) for name in model_info.get("feature_names", [])]
    if not feature_names:
        raise HTTPException(status_code=503, detail="model_info.json does not contain feature_names.")

    return model, feature_names, model_info


def _read_band(src: rasterio.DatasetReader, band_index: int, reflectance_scale: float) -> Optional[np.ndarray]:
    if band_index <= 0:
        return None
    if band_index > src.count:
        raise ValueError(f"Band index {band_index} exceeds uploaded raster band count {src.count}.")

    data = src.read(band_index, masked=False).astype("float32")
    return data / float(reflectance_scale)


def _base_feature_name(name: str) -> str:
    return base_feature_name(name)


def _feature_prefix(name: str) -> str | None:
    return feature_prefix(name)


def _read_uploaded_feature_stack(src: rasterio.DatasetReader, feature_names: list[str]) -> np.ndarray | None:
    descriptions = [desc or "" for desc in src.descriptions]
    if not all(name in descriptions for name in feature_names):
        return None
    schema_check = require_feature_stack_schema(feature_names, descriptions)
    return src.read(schema_check.selected_band_indexes, masked=False).astype("float32")


def _build_feature_arrays(src: rasterio.DatasetReader, params: InferRequest, feature_names: list[str]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    base_features = [_base_feature_name(name) for name in feature_names]
    prefixes = {prefix for prefix in (_feature_prefix(name) for name in feature_names) if prefix}
    if len(prefixes) > 1:
        raise ValueError(
            "The loaded model requires multiple timepoint slots "
            f"({', '.join(sorted(prefixes))}). Upload a prebuilt multi-timepoint feature stack with matching band names."
        )

    band_map = {
        "blue": params.blue_band,
        "green": params.green_band,
        "red": params.red_band,
        "rededge": params.rededge_band,
        "nir": params.nir_band,
        "swir": params.swir_band,
    }

    required_bands = set()
    if any(name in base_features for name in ("blue", "evi")):
        required_bands.add("blue")
    if any(name in base_features for name in ("green", "ndwi")):
        required_bands.add("green")
    if any(name in base_features for name in ("red", "ndvi", "evi")):
        required_bands.add("red")
    if any(name in base_features for name in ("rededge", "ndre")):
        required_bands.add("rededge")
    if any(name in base_features for name in ("nir", "ndvi", "ndwi", "evi", "ndre", "nbr")):
        required_bands.add("nir")
    if any(name in base_features for name in ("swir", "nbr")):
        required_bands.add("swir")

    for name in sorted(required_bands):
        band = _read_band(src, band_map[name], params.reflectance_scale)
        if band is None:
            raise ValueError(f"Feature '{name}' requires a valid band index.")
        arrays[name] = band

    if "ndvi" in base_features:
        arrays["ndvi"] = ndvi(arrays["nir"], arrays["red"])
    if "ndwi" in base_features:
        arrays["ndwi"] = ndwi(arrays["green"], arrays["nir"])
    if "evi" in base_features:
        arrays["evi"] = evi(arrays["nir"], arrays["red"], arrays["blue"])
    if "ndre" in base_features:
        arrays["ndre"] = ndre(arrays["nir"], arrays["rededge"])
    if "nbr" in base_features:
        arrays["nbr"] = nbr(arrays["nir"], arrays["swir"])

    missing = [name for name in base_features if name not in arrays]
    if missing:
        raise ValueError(f"Missing model features: {missing}")

    return {model_name: arrays[_base_feature_name(model_name)] for model_name in feature_names}


def _top_predictions(classes: np.ndarray, confidence: np.ndarray, model_info: dict[str, Any], top_k: int) -> list[dict[str, Any]]:
    valid = np.isfinite(confidence) & (confidence != -9999.0)
    if not np.any(valid):
        return []

    votes: dict[int, list[float]] = {}
    for cls, conf in zip(classes[valid].tolist(), confidence[valid].tolist()):
        votes.setdefault(int(cls), []).append(float(conf))

    ranked = sorted(votes.items(), key=lambda item: (len(item[1]), float(np.mean(item[1]))), reverse=True)[:top_k]
    return [
        {
            "class_code": code,
            "label": TARGET_LABELS.get(code, str(code)),
            "confidence": float(np.mean(scores)),
        }
        for code, scores in ranked
    ]


def _build_downloads(task_id: str, include_shp: bool) -> dict[str, str]:
    downloads = {
        "classification": f"/api/infer/download/{task_id}?format=classification",
        "confidence": f"/api/infer/download/{task_id}?format=confidence",
        "metadata": f"/api/infer/download/{task_id}?format=metadata",
    }
    if include_shp:
        downloads["shp"] = f"/api/infer/download/{task_id}?format=shp"
    return downloads


def _auto_detect_bands(src: rasterio.DatasetReader) -> dict[str, int]:
    """从 GeoTIFF 波段描述自动识别波段索引。

    返回 dict，键为 blue/green/red/rededge/nir/swir，值为 1-based 波段索引。
    无法识别时返回空 dict。
    """
    descriptions = [d.strip().lower() if d else "" for d in src.descriptions]

    # 如果波段描述为空，尝试根据波段数猜测 Sentinel-2 标准顺序
    if not any(descriptions):
        if src.count == 13:
            # Sentinel-2 L2A 典型 13 波段
            return {"blue": 2, "green": 3, "red": 4, "rededge": 5, "nir": 8, "swir": 11}
        if src.count >= 5:
            # 常见 5+ 波段多光谱
            return {"blue": 1, "green": 2, "red": 3, "rededge": 4, "nir": 5}
        if src.count == 4:
            return {"blue": 1, "green": 2, "red": 3, "nir": 4}
        return {}

    # Sentinel-2 资产名: B1/B2/B3/B4/B5/B8/B11
    s2_map = {
        "b1": ("blue",), "b01": ("blue",), "b2": ("blue",), "b02": ("blue",),
        "b3": ("green",), "b03": ("green",),
        "b4": ("red",), "b04": ("red",),
        "b5": ("rededge",), "b05": ("rededge",),
        "b6": ("rededge",), "b06": ("rededge",),
        "b7": ("rededge",), "b07": ("rededge",),
        "b8": ("nir",), "b08": ("nir",),
        "b8a": ("nir",), "b08a": ("nir",),
        "b11": ("swir",), "b12": ("swir",),
    }
    # 通用词汇
    word_map = {
        "blue": "blue", "b": "blue",
        "green": "green", "g": "green",
        "red": "red", "r": "red",
        "rededge": "rededge", "red edge": "rededge", "re": "rededge",
        "nir": "nir", "near infrared": "nir", "ir": "nir",
        "swir": "swir", "shortwave": "swir",
    }

    detected: dict[str, int] = {}
    for idx, desc in enumerate(descriptions, start=1):
        # 精确匹配 S2 命名
        if desc in s2_map:
            for band in s2_map[desc]:
                detected.setdefault(band, idx)
            continue
        # 单词匹配
        for word, band in word_map.items():
            if word in desc:
                detected.setdefault(band, idx)
                break

    return detected


def _build_features_for_file(
    file_path: Path, params: InferRequest, feature_names: list[str]
) -> tuple[dict[str, np.ndarray], Any, Any, int, int]:
    """对单个文件读取波段、自动识别、构建特征数组。"""
    with rasterio.open(file_path) as src:
        detected = _auto_detect_bands(src)
        if detected:
            if detected.get("blue") and params.blue_band == InferRequest.model_fields["blue_band"].default:
                params.blue_band = detected["blue"]
            if detected.get("green") and params.green_band == InferRequest.model_fields["green_band"].default:
                params.green_band = detected["green"]
            if detected.get("red") and params.red_band == InferRequest.model_fields["red_band"].default:
                params.red_band = detected["red"]
            if detected.get("rededge") and params.rededge_band == InferRequest.model_fields["rededge_band"].default:
                params.rededge_band = detected["rededge"]
            if detected.get("nir") and params.nir_band == InferRequest.model_fields["nir_band"].default:
                params.nir_band = detected["nir"]
            if detected.get("swir") and params.swir_band == InferRequest.model_fields["swir_band"].default:
                params.swir_band = detected["swir"]

        arrays = _build_feature_arrays(src, params, feature_names)
        return arrays, src.profile, src.crs, src.height, src.width


def _extract_date_from_filename(file_id: str) -> Optional[str]:
    """从上传文件的原始文件名中提取日期，返回 'YYYY-MM' 格式。

    支持命名约定:
      - 202504_xxx.tif, 2025-04_xxx.tif  →  2025-04
      - xxx_20250415_xxx.tif              →  2025-04
    """
    import re

    meta_path = DATA_UPLOADS / f"{file_id}.meta.json"
    filename = ""
    if meta_path.exists():
        try:
            with open(meta_path, encoding="utf-8") as f:
                meta = json.load(f)
            filename = meta.get("original_filename", "")
        except Exception:
            pass

    if not filename:
        return None

    # 前缀模式: 202504_xxx 或 2025-04_xxx
    m = re.match(r"(20\d{2})([01]\d)", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}"
    m = re.match(r"(20\d{2})-([01]\d)", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    # 中间模式: xxx_20250415_xxx
    m = re.search(r"(20\d{2})([01]\d)[0-3]\d", filename)
    if m:
        return f"{m.group(1)}-{m.group(2)}"

    return None


def _group_files_by_date(file_ids: list[str]) -> tuple[list[list[Path]], list[str]]:
    """按文件名日期自动将 file_id 分组为时相。

    返回: (timepoint_scenes_paths, time_labels)
    例如 file_ids 包含:
      - abc123 (原始名 202504_S2_east.tif)  →  2025-04
      - def456 (原始名 202504_S2_west.tif)  →  2025-04
      - ghi789 (原始名 202507_S2.tif)       →  2025-07
    结果: ([[abc123.tif, def456.tif], [ghi789.tif]], ["2025-04", "2025-07"])
    """
    groups: dict[str, list[Path]] = {}
    undated: list[Path] = []

    for fid in file_ids:
        p = DATA_UPLOADS / f"{fid}.tif"
        if not p.exists():
            continue
        date_key = _extract_date_from_filename(fid)
        if date_key:
            groups.setdefault(date_key, []).append(p)
        else:
            undated.append(p)

    # 未识别日期的文件各自成组
    for p in undated:
        groups.setdefault(f"_unknown_{len(groups) + 1}", []).append(p)

    sorted_keys = sorted(groups.keys())
    # 按时间顺序映射为 t1, t2, ... 以匹配模型特征命名约定
    time_labels = [f"t{i}" for i in range(1, len(sorted_keys) + 1)]
    return [groups[key] for key in sorted_keys], time_labels


def _composite_scenes_to_stack(
    timepoint_scenes: list[list[Path]],
    time_labels: list[str],
    params: InferRequest,
    feature_names: list[str],
) -> tuple[np.ndarray, dict, int, int]:
    """多景原始影像按时间分组 → 对齐合成 → 多时相特征栈。

    每个 timepoint 可能包含多景（空间不重叠），服务端：
    1. 计算所有景的并集范围作为参考网格
    2. 逐景 warp 到统一网格
    3. 按时相分组做中值合成
    4. 逐时相计算光谱指数
    5. 按模型特征名拼接成栈
    """
    from rasterio.warp import reproject, transform_bounds
    from rasterio.transform import array_bounds, from_origin

    all_paths = [p for group in timepoint_scenes for p in group]
    if not all_paths:
        raise ValueError("timepoint_scenes is empty.")

    # 1. 用第一个景确定参考 CRS 和分辨率
    with rasterio.open(all_paths[0]) as ref_src:
        ref_crs = ref_src.crs
        ref_res = abs(ref_src.transform.a)

    # 2. 计算所有景的并集范围
    union_bounds = [float('inf'), float('inf'), float('-inf'), float('-inf')]
    for path in all_paths:
        with rasterio.open(path) as src:
            if src.crs != ref_crs:
                b = transform_bounds(src.crs, ref_crs, *src.bounds)
            else:
                b = src.bounds
        union_bounds[0] = min(union_bounds[0], b[0])
        union_bounds[1] = min(union_bounds[1], b[1])
        union_bounds[2] = max(union_bounds[2], b[2])
        union_bounds[3] = max(union_bounds[3], b[3])

    west, south, east, north = union_bounds
    width = max(1, int(math.ceil((east - west) / ref_res)))
    height = max(1, int(math.ceil((north - south) / ref_res)))
    ref_transform = from_origin(west, north, ref_res, ref_res)
    ref_profile = {"driver": "GTiff", "width": width, "height": height,
                   "count": 1, "dtype": "float32", "crs": ref_crs,
                   "transform": ref_transform}

    # 3. 逐时相合成 + 构建特征
    all_timepoint_arrays: dict[str, np.ndarray] = {}

    base_bands = ["blue", "green", "red", "rededge", "nir", "swir"]
    for group_idx, group_paths in enumerate(timepoint_scenes):
        label = time_labels[group_idx] if group_idx < len(time_labels) else f"t{group_idx + 1}"

        # 逐景 warp 到参考网格
        band_accum: dict[str, list[np.ndarray]] = {b: [] for b in base_bands}
        for path in group_paths:
            with rasterio.open(path) as src:
                detected = _auto_detect_bands(src)
                if detected:
                    for b_name, b_idx in detected.items():
                        if b_name in base_bands:
                            if b_name == "blue" and params.blue_band == InferRequest.model_fields["blue_band"].default:
                                params.blue_band = b_idx
                            elif b_name == "green" and params.green_band == InferRequest.model_fields["green_band"].default:
                                params.green_band = b_idx
                            elif b_name == "red" and params.red_band == InferRequest.model_fields["red_band"].default:
                                params.red_band = b_idx
                            elif b_name == "rededge" and params.rededge_band == InferRequest.model_fields["rededge_band"].default:
                                params.rededge_band = b_idx
                            elif b_name == "nir" and params.nir_band == InferRequest.model_fields["nir_band"].default:
                                params.nir_band = b_idx
                            elif b_name == "swir" and params.swir_band == InferRequest.model_fields["swir_band"].default:
                                params.swir_band = b_idx

            band_map = {
                "blue": params.blue_band, "green": params.green_band,
                "red": params.red_band, "rededge": params.rededge_band,
                "nir": params.nir_band, "swir": params.swir_band,
            }

            for b_name, b_idx in band_map.items():
                if b_idx <= 0:
                    continue
                try:
                    with rasterio.open(path) as src:
                        if b_idx > src.count:
                            continue
                        dst = np.zeros((height, width), dtype="float32")
                        src_data = src.read(b_idx, masked=False).astype("float32")
                        src_data = src_data / params.reflectance_scale
                        reproject(
                            source=src_data,
                            destination=dst,
                            src_transform=src.transform,
                            src_crs=src.crs,
                            dst_transform=ref_transform,
                            dst_crs=ref_crs,
                            resampling=rasterio.enums.Resampling.bilinear,
                        )
                        band_accum[b_name].append(dst)
                except Exception:
                    continue

        # 中值合成
        composite: dict[str, np.ndarray] = {}
        for b_name, arrays in band_accum.items():
            if not arrays:
                continue
            if len(arrays) == 1:
                composite[b_name] = arrays[0]
            else:
                stack = np.stack(arrays, axis=0)
                composite[b_name] = np.nanmedian(stack, axis=0).astype("float32")

        if "red" not in composite or "nir" not in composite:
            raise ValueError(f"Timepoint {label} missing required red/NIR bands after compositing.")

        # 计算光谱指数
        red = composite["red"]; nir = composite["nir"]
        denom = nir + red
        composite["ndvi"] = np.divide(nir - red, denom, out=np.full_like(red, np.nan), where=denom != 0)
        if "green" in composite:
            composite["ndwi"] = np.divide(composite["green"] - nir, composite["green"] + nir,
                                          out=np.full_like(red, np.nan), where=(composite["green"] + nir) != 0)
        if "blue" in composite:
            composite["evi"] = 2.5 * np.divide(nir - red, nir + 6.0 * red - 7.5 * composite["blue"] + 1.0,
                                               out=np.full_like(red, np.nan), where=(nir + 6.0 * red - 7.5 * composite["blue"] + 1.0) != 0)
        if "rededge" in composite:
            composite["ndre"] = np.divide(nir - composite["rededge"], nir + composite["rededge"],
                                          out=np.full_like(red, np.nan), where=(nir + composite["rededge"]) != 0)
        if "swir" in composite:
            composite["nbr"] = np.divide(nir - composite["swir"], nir + composite["swir"],
                                         out=np.full_like(red, np.nan), where=(nir + composite["swir"]) != 0)

        for name, arr in composite.items():
            all_timepoint_arrays[f"{label}_{name}"] = arr

    # 4. 按模型特征顺序组栈
    stack_arrays: list[np.ndarray] = []
    for model_name in feature_names:
        found = False
        for tp_name, arr in all_timepoint_arrays.items():
            if tp_name == model_name or tp_name.endswith(f"_{model_name}"):
                stack_arrays.append(arr)
                found = True
                break
        # 回退：末尾匹配
        if not found:
            for tp_name, arr in all_timepoint_arrays.items():
                if tp_name.endswith(f"_{model_name}"):
                    stack_arrays.append(arr)
                    found = True
                    break

    if not stack_arrays:
        raise ValueError(f"No features matched model features {feature_names}. Available: {list(all_timepoint_arrays.keys())}")

    stack = np.stack(stack_arrays, axis=0)
    return stack, ref_profile, height, width


def _run_inference(
    task_id: str,
    upload_paths: list[Path],
    params: InferRequest,
    timepoint_scenes: list[list[Path]],
    time_labels: list[str],
) -> None:
    try:
        with task_lock:
            task_store[task_id]["status"] = "running"
            task_store[task_id]["progress"] = 10.0
            task_store[task_id]["message"] = "Loading model."

        model, feature_names, model_info = _load_model()

        with task_lock:
            task_store[task_id]["progress"] = 20.0
            task_store[task_id]["message"] = "Preprocessing bands and indices."

        # 单文件：尝试预建特征栈
        if len(upload_paths) == 1 and len(timepoint_scenes) == 1 and len(timepoint_scenes[0]) == 1:
            with rasterio.open(upload_paths[0]) as src:
                stack = _read_uploaded_feature_stack(src, feature_names)
                ref_profile = src.profile.copy()
                ref_height, ref_width = src.height, src.width

            if stack is None:
                arrays, ref_profile, _, ref_height, ref_width = _build_features_for_file(
                    upload_paths[0], params, feature_names
                )
                stack = np.stack([arrays[name] for name in feature_names], axis=0)
        else:
            # 多景/多时相 → 对齐、合成、构建特征栈
            stack, ref_profile, ref_height, ref_width = _composite_scenes_to_stack(
                timepoint_scenes, time_labels, params, feature_names
            )

        feature_count, height, width = stack.shape
        flat = np.moveaxis(stack, 0, -1).reshape(-1, feature_count).astype("float32")
        valid = np.all(np.isfinite(flat), axis=1)

        with task_lock:
            task_store[task_id]["progress"] = 65.0
            task_store[task_id]["message"] = "Running model inference."

        class_values = np.zeros(flat.shape[0], dtype="uint8")
        confidence_values = np.full(flat.shape[0], -9999.0, dtype="float32")
        valid_count = int(np.count_nonzero(valid))

        if valid_count:
            batch = flat[valid]
            predictions = model.predict(batch).astype("uint8")
            class_values[valid] = normalize_output_classes(predictions)
            if hasattr(model, "predict_proba"):
                confidence_values[valid] = model.predict_proba(batch).max(axis=1).astype("float32")
            else:
                confidence_values[valid] = 1.0

        class_2d = class_values.reshape(height, width)
        confidence_2d = confidence_values.reshape(height, width)

        API_PREDICTIONS.mkdir(parents=True, exist_ok=True)
        class_path = API_PREDICTIONS / f"{task_id}_classification.tif"
        conf_path = API_PREDICTIONS / f"{task_id}_confidence.tif"
        parcel_shp_path = API_PREDICTIONS / f"{task_id}_parcels" / f"{task_id}_parcels.shp"
        parcel_zip_path = API_PREDICTIONS / f"{task_id}_parcels.zip"
        meta_path = API_PREDICTIONS / f"{task_id}_metadata.json"

        class_profile = ref_profile.copy()
        class_profile.update(count=1, dtype="uint8", nodata=None, compress="deflate")
        conf_profile = ref_profile.copy()
        conf_profile.update(count=1, dtype="float32", nodata=-9999.0, compress="deflate", predictor=3)

        with rasterio.open(class_path, "w", **class_profile) as dst:
            dst.write(class_2d.astype("uint8"), 1)
            dst.set_band_description(1, "crop_class")

        with rasterio.open(conf_path, "w", **conf_profile) as dst:
            dst.write(confidence_2d.astype("float32"), 1)
            dst.set_band_description(1, "confidence")

        parcel_info = None
        if params.parcel_file_id:
            with task_lock:
                task_store[task_id]["progress"] = 85.0
                task_store[task_id]["message"] = "Aggregating raster classes to parcel shapefile."

            parcel_info = attach_raster_majority_to_parcels(
                class_path,
                _parcel_shp_from_file_id(params.parcel_file_id),
                parcel_shp_path,
                parcel_zip_path,
                field="crop_type",
                raster_band=1,
                include_all=True,
            )
        top_predictions = _top_predictions(class_values, confidence_values, model_info, params.top_k)
        downloads = _build_downloads(task_id, include_shp=parcel_info is not None)
        input_files = [str(p.relative_to(ROOT)) for p in upload_paths]
        metadata = {
            "task_id": task_id,
            "status": "completed",
            "input_files": input_files,
            "time_labels": time_labels if len(upload_paths) > 1 else None,
            "model_features": feature_names,
            "valid_pixel_count": valid_count,
            "top_predictions": top_predictions,
            "raster": {
                "width": width,
                "height": height,
                "band_count": ref_profile.get("count", len(upload_paths)),
                "crs": str(ref_profile.get("crs", "")),
            },
            "preprocess": {
                "reflectance_scale": params.reflectance_scale,
                "band_mapping": {
                    "blue": params.blue_band,
                    "green": params.green_band,
                    "red": params.red_band,
                    "rededge": params.rededge_band,
                    "nir": params.nir_band,
                    "swir": params.swir_band,
                },
            },
            "outputs": downloads,
        }
        if parcel_info is not None:
            metadata["parcel_shapefile"] = parcel_info
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        with task_lock:
            task_store[task_id]["status"] = "completed"
            task_store[task_id]["progress"] = 100.0
            task_store[task_id]["message"] = "Completed."
            task_store[task_id]["valid_pixel_count"] = valid_count
            task_store[task_id]["model_features"] = feature_names
            task_store[task_id]["top_predictions"] = top_predictions
            task_store[task_id]["downloads"] = downloads
    except Exception as exc:
        with task_lock:
            task_store[task_id]["status"] = "failed"
            task_store[task_id]["progress"] = 100.0
            task_store[task_id]["message"] = str(exc)[:500]


# ---------------------------------------------------------------------------
# 估产后台任务
# ---------------------------------------------------------------------------

def _run_yield_estimation(yield_task_id: str, params: YieldEstimateRequest) -> None:
    try:
        with yield_task_lock:
            yield_task_store[yield_task_id]["status"] = "running"
            yield_task_store[yield_task_id]["progress"] = 5.0
            yield_task_store[yield_task_id]["message"] = "Loading inference metadata."

        # 1. 读取分类推理的元数据
        infer_meta_path = API_PREDICTIONS / f"{params.task_id}_metadata.json"
        if not infer_meta_path.exists():
            raise ValueError(f"Inference metadata not found for task_id: {params.task_id}")
        with open(infer_meta_path, encoding="utf-8") as f:
            infer_meta = json.load(f)

        # 2. 读取所有上传影像，计算逐像素 NDVI 均值
        input_files = infer_meta.get("input_files", [infer_meta.get("input_file", "")])
        if not input_files:
            raise ValueError("No input files found in inference metadata.")
        band_mapping = infer_meta["preprocess"]["band_mapping"]
        reflectance_scale = float(infer_meta["preprocess"]["reflectance_scale"])
        red_band = band_mapping.get("red", 3)
        nir_band = band_mapping.get("nir", 5)
        rededge_band = band_mapping.get("rededge", 4)

        with yield_task_lock:
            yield_task_store[yield_task_id]["progress"] = 15.0
            yield_task_store[yield_task_id]["message"] = "Reading bands and computing NDVI mean across timepoints."

        ndvi_arrays = []
        ci_arrays = []
        class_profile = None
        for rel_path in input_files:
            fpath = ROOT / rel_path
            if not fpath.exists():
                continue
            with rasterio.open(fpath) as src:
                if class_profile is None:
                    class_profile = src.profile.copy()
                if red_band <= 0 or red_band > src.count:
                    continue
                if nir_band <= 0 or nir_band > src.count:
                    continue
                red = src.read(red_band, masked=False).astype("float32") / reflectance_scale
                nir = src.read(nir_band, masked=False).astype("float32") / reflectance_scale
            denom = nir + red
            ndvi = np.divide(nir - red, denom, out=np.full_like(red, np.nan), where=denom != 0)
            ndvi_arrays.append(ndvi)

            if params.index == "lai" and rededge_band > 0:
                with rasterio.open(fpath) as src:
                    if rededge_band <= src.count:
                        rededge = src.read(rededge_band, masked=False).astype("float32") / reflectance_scale
                ci = np.divide(nir, rededge, out=np.full_like(nir, np.nan), where=rededge > 0) - 1.0
                ci_arrays.append(ci)

        if not ndvi_arrays:
            raise ValueError("No valid input files for NDVI computation.")
        if class_profile is None:
            raise ValueError("Failed to read input raster profile.")

        # NDVI 逐像素均值
        ndvi_arr = np.nanmean(np.stack(ndvi_arrays, axis=0), axis=0).astype("float32")

        # LAI 逐像素均值
        ci_arr = None
        if params.index == "lai" and ci_arrays:
            ci_arr = np.nanmean(np.stack(ci_arrays, axis=0), axis=0).astype("float32")

        # 3. 读取分类栅格
        class_path = API_PREDICTIONS / f"{params.task_id}_classification.tif"
        if not class_path.exists():
            raise ValueError(f"Classification raster not found: {class_path}")

        with rasterio.open(class_path) as src:
            classification = src.read(1, masked=False).astype("int16")

        with yield_task_lock:
            yield_task_store[yield_task_id]["progress"] = 35.0
            yield_task_store[yield_task_id]["message"] = "Running yield estimation per crop."

        # 4. 逐作物估产
        pixel_area = abs(class_profile["transform"].a * class_profile["transform"].e) / 10000.0
        crop_results: list[dict[str, Any]] = []

        for code in sorted(CROP_CODE_TO_NAME):
            crop_name = CROP_CODE_TO_NAME[code]
            label = TARGET_LABELS.get(code, str(code))

            mask = classification == code
            pixel_count = int(np.count_nonzero(mask))
            if pixel_count == 0:
                crop_results.append({
                    "crop_code": code, "crop_name": crop_name, "label": label,
                    "area_ha": 0.0, "pixel_count": 0,
                    "mean_yield_kg_ha": None, "total_yield_kg": 0.0,
                    "warning": "no pixels found",
                })
                continue

            # 确定预测因子
            if params.index == "lai" and ci_arr is not None:
                crop_predictor = lai_from_ci(ci_arr, crop_name, params.lai_k, params.lai_m)
            else:
                crop_predictor = ndvi_arr

            masked_predictor = np.where(mask, crop_predictor, np.nan)

            try:
                pixel_yield = estimate_yield(
                    masked_predictor, crop_name,
                    override=params.model_coefficients,
                    function_type=params.yield_function,
                )
            except ValueError as exc:
                crop_results.append({
                    "crop_code": code, "crop_name": crop_name, "label": label,
                    "area_ha": 0.0, "pixel_count": pixel_count,
                    "error": str(exc),
                })
                continue

            valid = np.isfinite(pixel_yield) & (pixel_yield > 0)
            if not np.any(valid):
                crop_results.append({
                    "crop_code": code, "crop_name": crop_name, "label": label,
                    "area_ha": float(pixel_count * pixel_area), "pixel_count": pixel_count,
                    "mean_yield_kg_ha": None, "total_yield_kg": 0.0,
                    "warning": "no valid yield pixels",
                })
                continue

            values = pixel_yield[valid]
            area_ha = float(np.count_nonzero(valid) * pixel_area)
            crop_results.append({
                "crop_code": code,
                "crop_name": crop_name,
                "label": label,
                "area_ha": area_ha,
                "pixel_count": int(np.count_nonzero(valid)),
                "mean_yield_kg_ha": float(np.mean(values)),
                "median_yield_kg_ha": float(np.median(values)),
                "std_yield_kg_ha": float(np.std(values)),
                "total_yield_kg": float(np.sum(values * pixel_area)),
                "uncertainty": uncertainty(crop_name, float(np.mean(values))),
            })

        # 5. 汇总
        total_area = sum(r["area_ha"] for r in crop_results)
        total_yield = sum(r["total_yield_kg"] for r in crop_results)
        summary = {
            "total_cropland_area_ha": total_area,
            "total_yield_kg": total_yield,
            "average_yield_kg_ha": total_yield / total_area if total_area > 0 else 0.0,
            "index_used": params.index,
            "yield_function": params.yield_function,
        }

        # 6. 保存估产统计
        yield_meta_path = API_PREDICTIONS / f"{yield_task_id}_yield_metadata.json"
        yield_meta = {
            "yield_task_id": yield_task_id,
            "inference_task_id": params.task_id,
            "index": params.index,
            "yield_function": params.yield_function,
            "summary": summary,
            "crops": crop_results,
        }
        with open(yield_meta_path, "w", encoding="utf-8") as f:
            json.dump(yield_meta, f, indent=2, ensure_ascii=False)

        downloads = {
            "metadata": f"/api/yield/download/{yield_task_id}?format=metadata",
        }

        with yield_task_lock:
            yield_task_store[yield_task_id]["status"] = "completed"
            yield_task_store[yield_task_id]["progress"] = 100.0
            yield_task_store[yield_task_id]["message"] = "Completed."
            yield_task_store[yield_task_id]["summary"] = summary
            yield_task_store[yield_task_id]["crops"] = crop_results
            yield_task_store[yield_task_id]["downloads"] = downloads
    except Exception as exc:
        with yield_task_lock:
            yield_task_store[yield_task_id]["status"] = "failed"
            yield_task_store[yield_task_id]["progress"] = 100.0
            yield_task_store[yield_task_id]["message"] = str(exc)[:500]


def _portal_html(request: Request) -> str:
    root_path = request.scope.get("root_path", "")
    host_base = str(request.base_url).rstrip("/")
    app_base = host_base + root_path
    ws_scheme = "wss" if request.url.scheme == "https" else "ws"
    ws_base = f"{ws_scheme}://{request.url.netloc}{root_path}"

    template = HOME_TEMPLATE.read_text(encoding="utf-8")
    return template.replace("__APP_BASE__", app_base).replace("__API_BASE__", f"{app_base}/api").replace("__WS_BASE__", ws_base)


app = FastAPI(
    title="作物分类与估产服务",
    version="0.6.0",
    description="作物分类与估产 Web API：上传影像 → 分类推理 → 产量估算，端到端遥感作物监测。",
)


@app.get("/api/health", tags=["System"], summary="Health check")
def api_health() -> dict[str, Any]:
    return {"status": "ok", "service": "CropClassifier API", "version": app.version}


@app.get("/classes", tags=["Reference"], summary="Get class mapping")
def classes() -> dict[str, Any]:
    return _read_yaml(CLASS_MAPPING_FILE)


@app.get("/artifacts", tags=["Artifacts"], summary="List available artifacts")
def artifacts() -> dict[str, Any]:
    registry = _artifact_registry()
    items = [_file_info(path) for _, path in sorted(registry.items())]
    return {"count": len(items), "items": items}


@app.get("/artifacts/{name:path}/download", tags=["Artifacts"], summary="Download artifact")
def artifact_download(name: str) -> FileResponse:
    registry = _artifact_registry()
    key = name.replace("\\", "/")
    path = registry.get(key)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {name}")
    return FileResponse(path, filename=path.name)


@app.get("/artifacts/{name:path}", tags=["Artifacts"], summary="Get artifact info")
def artifact_info(name: str) -> dict[str, Any]:
    registry = _artifact_registry()
    key = name.replace("\\", "/")
    path = registry.get(key)
    if path is None:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {name}")
    return _file_info(path)


@app.post("/api/data/upload", response_model=UploadResponse, tags=["Data"], summary="Upload image file")
def upload_data(
    file: UploadFile = File(..., description="Multispectral GeoTIFF file."),
) -> UploadResponse:
    if not file.filename or not file.filename.lower().endswith((".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="Only GeoTIFF files are supported.")

    DATA_UPLOADS.mkdir(parents=True, exist_ok=True)
    file_id = uuid.uuid4().hex
    upload_path = DATA_UPLOADS / f"{file_id}.tif"
    with open(upload_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # 保存原始文件名，用于后续按日期自动分组
    meta = {"original_filename": file.filename or ""}
    meta_path = DATA_UPLOADS / f"{file_id}.meta.json"
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf)

    return UploadResponse(file_id=file_id, filename=file.filename, size_bytes=upload_path.stat().st_size)


@app.post("/api/data/upload-parcels", response_model=ParcelUploadResponse, tags=["Data"], summary="Upload parcel shapefile zip")
def upload_parcels(
    file: UploadFile = File(..., description="ZIP containing one Shapefile dataset (.shp/.shx/.dbf/.prj...)."),
) -> ParcelUploadResponse:
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Only .zip parcel Shapefile uploads are supported.")

    DATA_UPLOADS.mkdir(parents=True, exist_ok=True)
    parcel_file_id = uuid.uuid4().hex
    zip_path = DATA_UPLOADS / f"{parcel_file_id}_parcels.zip"
    extract_dir = DATA_UPLOADS / f"{parcel_file_id}_parcels"

    with open(zip_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        _safe_extract_zip(zip_path, extract_dir)
    except zipfile.BadZipFile as exc:
        raise HTTPException(status_code=400, detail="Invalid zip file.") from exc

    shapefiles = sorted(extract_dir.rglob("*.shp"))
    if not shapefiles:
        raise HTTPException(status_code=400, detail="Uploaded zip must contain one .shp file.")
    if len(shapefiles) > 1:
        raise HTTPException(status_code=400, detail="Uploaded zip must contain exactly one .shp file.")

    shp = shapefiles[0]
    required = [shp.with_suffix(ext) for ext in (".shx", ".dbf")]
    missing = [path.name for path in required if not path.exists()]
    if missing:
        raise HTTPException(status_code=400, detail=f"Uploaded Shapefile is missing required files: {', '.join(missing)}")

    return ParcelUploadResponse(
        parcel_file_id=parcel_file_id,
        filename=file.filename,
        shapefile=str(shp.relative_to(ROOT)),
        size_bytes=zip_path.stat().st_size,
    )


@app.post("/api/infer/start", response_model=InferStartResponse, tags=["Inference"], summary="Start inference task")
async def infer_start(
    request: Request,
) -> InferStartResponse:
    body = await _parse_infer_request(request)

    # 验证文件存在
    upload_paths: list[Path] = []
    for fid in body.file_ids:
        p = DATA_UPLOADS / f"{fid}.tif"
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"Uploaded file does not exist: {fid}")
        upload_paths.append(p)
    if not upload_paths:
        raise HTTPException(status_code=400, detail="file_ids must not be empty.")

    # 按文件名日期自动分组
    timepoint_scenes, time_labels = _group_files_by_date(body.file_ids)
    print(f"Auto-grouped {len(upload_paths)} files into {len(timepoint_scenes)} timepoints: {time_labels}")

    task_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    with task_lock:
        task_store[task_id] = {"status": "queued", "progress": 0.0, "message": "Queued."}

    threading.Thread(
        target=_run_inference,
        args=(task_id, upload_paths, body, timepoint_scenes, time_labels),
        daemon=True,
    ).start()
    return InferStartResponse(task_id=task_id, status="queued")


@app.get("/api/infer/status/{task_id}", response_model=InferStatusResponse, tags=["Inference"], summary="Check inference status")
def infer_status(
    task_id: str,
) -> InferStatusResponse:
    with task_lock:
        task = dict(task_store.get(task_id) or {})
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")

    top_predictions = None
    if task.get("top_predictions"):
        top_predictions = [InferTopPrediction(**item) for item in task["top_predictions"]]

    return InferStatusResponse(
        task_id=task_id,
        status=task.get("status", "unknown"),
        progress=float(task.get("progress", 0.0)),
        message=task.get("message"),
        valid_pixel_count=task.get("valid_pixel_count"),
        model_features=task.get("model_features"),
        top_predictions=top_predictions,
        downloads=task.get("downloads"),
    )


@app.get("/api/infer/tasks", tags=["Inference"], summary="List inference tasks")
def infer_tasks() -> dict[str, Any]:
    with task_lock:
        tasks = [
            {
                "task_id": task_id,
                "status": task.get("status"),
                "progress": task.get("progress"),
                "message": task.get("message"),
            }
            for task_id, task in sorted(task_store.items(), reverse=True)
        ]
    return {"tasks": tasks}


@app.get("/api/infer/download/{task_id}", tags=["Inference"], summary="Download inference output")
def infer_download(
    task_id: str,
    format: str = "classification",
) -> FileResponse:
    with task_lock:
        task = task_store.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Task is not completed yet.")

    suffix_map = {"classification": ".tif", "confidence": ".tif", "metadata": ".json", "shp": ".zip"}
    if format not in suffix_map:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    if format == "shp" and "shp" not in (task.get("downloads") or {}):
        raise HTTPException(status_code=404, detail="Parcel Shapefile was not generated for this task. Upload parcels and pass parcel_file_id.")

    artifact = "parcels" if format == "shp" else format
    path = API_PREDICTIONS / f"{task_id}_{artifact}{suffix_map[format]}"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Output file not found.")

    media_type = "application/zip" if format == "shp" else None
    return FileResponse(path, filename=path.name, media_type=media_type)


@app.get("/api-predictions/{job_id}/classification", tags=["Prediction"], summary="Download prediction classification")
def api_prediction_classification(job_id: str) -> FileResponse:
    path = _api_prediction_file(job_id, "classification.tif")
    return FileResponse(path, filename=path.name)


@app.get("/api-predictions/{job_id}/confidence", tags=["Prediction"], summary="Download prediction confidence")
def api_prediction_confidence(job_id: str) -> FileResponse:
    path = _api_prediction_file(job_id, "confidence.tif")
    return FileResponse(path, filename=path.name)


@app.get("/api-predictions/{job_id}/shp", tags=["Prediction"], summary="Download parcel-level Shapefile ZIP")
def api_prediction_shp(job_id: str) -> FileResponse:
    path = _api_prediction_file(job_id, "parcels.zip")
    return FileResponse(path, filename=path.name, media_type="application/zip")


@app.get("/api-predictions/{job_id}/metadata", tags=["Prediction"], summary="Get prediction metadata")
def api_prediction_metadata(job_id: str) -> dict[str, Any]:
    path = _api_prediction_file(job_id, "metadata.json")
    return _report_json(path)


@app.get("/reports/prediction", tags=["Reports"], summary="Get prediction report")
def report_prediction() -> dict[str, Any]:
    path = _existing_file(
        "data/output/prediction_info.json",
        "data/output/prediction_info_2025_07_test.json",
    )
    return _report_json(path)


@app.get("/reports/postprocess", tags=["Reports"], summary="Get postprocess report")
def report_postprocess() -> dict[str, Any]:
    path = _existing_file("data/output/parcel_postprocess/caobuhu_parcel_majority_summary.json")
    return _report_json(path)


@app.get("/reports/accuracy", tags=["Reports"], summary="Get accuracy report")
def report_accuracy() -> dict[str, Any]:
    path = _existing_file("data/output/accuracy_report_2025_07_selfcheck.json")
    return _report_json(path)


@app.get("/maps/summary", tags=["Maps"], summary="Get classification area summary")
def maps_summary() -> dict[str, Any]:
    path = _existing_file(
        "data/output/crop_classification_2025_07_test_clean.tif",
        "data/output/crop_classification.tif",
        "data/output/crop_classification_2025_07_test.tif",
    )
    return _map_summary(path)


# ---------------------------------------------------------------------------
# 估产 (yield) 端点
# ---------------------------------------------------------------------------

@app.post("/api/yield/estimate", response_model=YieldEstimateResponse, tags=["Yield"], summary="Start yield estimation")
def yield_estimate(body: YieldEstimateRequest) -> YieldEstimateResponse:
    # 验证推理任务已完成
    with task_lock:
        infer_task = task_store.get(body.task_id)
    if not infer_task:
        raise HTTPException(status_code=404, detail=f"Inference task not found: {body.task_id}")
    if infer_task.get("status") != "completed":
        raise HTTPException(status_code=400, detail=f"Inference task not completed. Current status: {infer_task.get('status')}")

    # 检查分类栅格是否存在
    class_path = API_PREDICTIONS / f"{body.task_id}_classification.tif"
    if not class_path.exists():
        raise HTTPException(status_code=404, detail=f"Classification raster not found for task: {body.task_id}")

    yield_task_id = f"yield_{body.task_id}"
    with yield_task_lock:
        if yield_task_id in yield_task_store and yield_task_store[yield_task_id].get("status") in ("queued", "running"):
            raise HTTPException(status_code=409, detail=f"Yield estimation already in progress for task: {body.task_id}")
        yield_task_store[yield_task_id] = {"status": "queued", "progress": 0.0, "message": "Queued."}

    threading.Thread(target=_run_yield_estimation, args=(yield_task_id, body), daemon=True).start()
    return YieldEstimateResponse(yield_task_id=yield_task_id, status="queued")


@app.get("/api/yield/status/{yield_task_id}", response_model=YieldStatusResponse, tags=["Yield"], summary="Check yield estimation status")
def yield_status(yield_task_id: str) -> YieldStatusResponse:
    with yield_task_lock:
        task = dict(yield_task_store.get(yield_task_id) or {})
    if not task:
        raise HTTPException(status_code=404, detail="Yield task not found.")

    crops = None
    if task.get("crops"):
        crops = [YieldCropResult(**item) for item in task["crops"]]

    return YieldStatusResponse(
        yield_task_id=yield_task_id,
        status=task.get("status", "unknown"),
        progress=float(task.get("progress", 0.0)),
        message=task.get("message"),
        summary=task.get("summary"),
        crops=crops,
        downloads=task.get("downloads"),
    )


@app.get("/api/yield/tasks", tags=["Yield"], summary="List yield estimation tasks")
def yield_tasks() -> dict[str, Any]:
    with yield_task_lock:
        tasks = [
            {
                "yield_task_id": tid,
                "status": t.get("status"),
                "progress": t.get("progress"),
                "message": t.get("message"),
            }
            for tid, t in sorted(yield_task_store.items(), reverse=True)
        ]
    return {"tasks": tasks}


@app.get("/api/yield/download/{yield_task_id}", tags=["Yield"], summary="Download yield metadata")
def yield_download(yield_task_id: str, format: str = "metadata") -> FileResponse:
    with yield_task_lock:
        task = yield_task_store.get(yield_task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Yield task not found.")
    if task.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Yield task is not completed yet.")

    if format == "metadata":
        path = API_PREDICTIONS / f"{yield_task_id}_yield_metadata.json"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {format}")
    if not path.exists():
        raise HTTPException(status_code=404, detail="Yield metadata file not found.")
    return FileResponse(path, filename=path.name, media_type="application/json")


@app.websocket("/ws/infer/{task_id}")
async def ws_infer_status(websocket: WebSocket, task_id: str) -> None:
    await websocket.accept()
    try:
        while True:
            with task_lock:
                task = dict(task_store.get(task_id) or {})

            if not task:
                await websocket.send_json({"task_id": task_id, "status": "missing", "message": "Task not found."})
                break

            await websocket.send_json(
                {
                    "task_id": task_id,
                    "status": task.get("status"),
                    "progress": task.get("progress", 0.0),
                    "message": task.get("message"),
                    "valid_pixel_count": task.get("valid_pixel_count"),
                    "top_predictions": task.get("top_predictions"),
                }
            )

            if task.get("status") in {"completed", "failed"}:
                break
            await asyncio.sleep(1)
    except WebSocketDisconnect:
        return


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home(request: Request) -> str:
    return _portal_html(request)
