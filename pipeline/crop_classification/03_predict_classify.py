"""执行整区作物分类预测。

读取标准特征栈和已训练模型，对整个 AOI 分块预测作物类别，并输出像素级
分类 GeoTIFF、置信度 GeoTIFF 和预测信息 JSON。
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window
from rasterio.warp import transform_geom

from crop_domain.labels import normalize_output_classes
from image_core.feature_schema import band_names_from_dataset, require_feature_stack_schema
from configs.paths import ProjectPaths



NODATA_CLASS = 255
NODATA_CONFIDENCE = -9999.0


def configure_gdal_proj() -> None:
    try:
        proj_dir = Path(rasterio.__file__).resolve().parent / "proj_data"
        if (proj_dir / "proj.db").exists():
            os.environ.setdefault("PROJ_LIB", str(proj_dir))
            os.environ.setdefault("PROJ_DATA", str(proj_dir))
        os.environ.setdefault("PROJ_IGNORE_BUILD_INFO", "YES")
    except Exception:
        pass


configure_gdal_proj()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict crop classes for a whole-AOI feature stack.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--feature-stack", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--model-info", type=Path, default=None)
    parser.add_argument("--classification", type=Path, default=None)
    parser.add_argument("--confidence", type=Path, default=None)
    parser.add_argument("--prediction-info", type=Path, default=None)
    parser.add_argument(
        "--timepoint",
        default=None,
        help="Optional feature prefix such as t1. Required only when matching an old single-slot model by suffix.",
    )
    parser.add_argument(
        "--allow-ambiguous-suffix",
        action="store_true",
        help="Use the first suffix match when several timepoints contain the same feature name.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_region_geometry(path: Path, default_crs: str) -> tuple[dict[str, Any], str]:
    if path.suffix.lower() in {".json", ".geojson"}:
        with open(path, encoding="utf-8-sig") as f:
            geojson = json.load(f)
        crs = default_crs
        if isinstance(geojson.get("crs"), dict):
            crs = str(geojson["crs"].get("properties", {}).get("name") or default_crs)
        if geojson.get("type") == "FeatureCollection":
            features = geojson.get("features", [])
            if not features:
                raise ValueError(f"{path} contains no features.")
            if len(features) == 1:
                return features[0]["geometry"], crs
            return {"type": "GeometryCollection", "geometries": [item["geometry"] for item in features]}, crs
        if geojson.get("type") == "Feature":
            return geojson["geometry"], crs
        return geojson, crs

    try:
        import geopandas as gpd
    except ImportError as exc:
        raise ValueError("Reading non-GeoJSON AOI masks requires geopandas.") from exc

    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError(f"{path} contains no features.")
    geometry = gdf.geometry.unary_union.__geo_interface__
    crs = str(gdf.crs) if gdf.crs is not None else default_crs
    return geometry, crs


def build_aoi_mask(
    aoi_path: Path | None,
    ref: rasterio.DatasetReader,
    default_crs: str = "EPSG:4326",
) -> np.ndarray | None:
    if aoi_path is None:
        return None
    if not aoi_path.exists():
        raise FileNotFoundError(f"AOI mask file does not exist: {aoi_path}")
    geometry, src_crs = load_region_geometry(aoi_path, default_crs)
    if ref.crs is not None and src_crs:
        geometry = transform_geom(src_crs, ref.crs, geometry)
    mask = geometry_mask([geometry], out_shape=(ref.height, ref.width), transform=ref.transform, invert=True)
    if not np.any(mask):
        raise ValueError(f"AOI mask has no overlap with feature stack: {aoi_path}")
    return mask


def output_profiles(src: rasterio.DatasetReader) -> tuple[dict[str, Any], dict[str, Any]]:
    class_profile = src.profile.copy()
    class_profile.update(
        count=1,
        dtype="uint8",
        nodata=NODATA_CLASS,
        compress="deflate",
        tiled=True,
        blockxsize=min(256, src.width),
        blockysize=min(256, src.height),
    )

    confidence_profile = src.profile.copy()
    confidence_profile.update(
        count=1,
        dtype="float32",
        nodata=NODATA_CONFIDENCE,
        compress="deflate",
        predictor=3,
        tiled=True,
        blockxsize=min(256, src.width),
        blockysize=min(256, src.height),
    )
    return class_profile, confidence_profile


def iter_windows(src: rasterio.DatasetReader) -> list[Window]:
    windows = [window for _, window in src.block_windows(1)]
    if windows:
        return windows
    return [Window(0, 0, src.width, src.height)]


def predict_window(model: Any, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict one raster window."""
    feature_count, rows, cols = data.shape
    flat = np.moveaxis(data, 0, -1).reshape(-1, feature_count).astype("float32")
    valid = np.all(np.isfinite(flat), axis=1)

    classes = np.full(flat.shape[0], NODATA_CLASS, dtype="uint8")
    confidence = np.full(flat.shape[0], NODATA_CONFIDENCE, dtype="float32")

    if np.any(valid):
        X = flat[valid]
        predictions = model.predict(X).astype("uint8")
        classes[valid] = normalize_output_classes(predictions)

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)
            confidence[valid] = probabilities.max(axis=1).astype("float32")
        else:
            confidence[valid] = 1.0

    return classes.reshape(rows, cols), confidence.reshape(rows, cols)


def main() -> None:
    args = parse_args()
    paths = ProjectPaths(args.config)

    feature_stack = args.feature_stack or paths.feature_stack
    metadata_path = args.metadata or paths.feature_stack_metadata
    model_path = args.model or paths.model_file
    model_info_path = args.model_info or paths.model_info
    classification = args.classification or paths.classification
    confidence = args.confidence or paths.classification_confidence
    prediction_info_path = args.prediction_info or paths.classification_info

    if not feature_stack.exists():
        raise FileNotFoundError(
            f"Missing feature stack: {feature_stack}. "
            "Run python -m data_sources.aws_element84.build_features or "
            "python -m image_core.build_features_from_multiband first."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing model: {model_path}. Run python -m pipeline.crop_classification.02_train_rf first."
        )

    metadata = load_json(metadata_path) if metadata_path.exists() else None
    model_info = load_json(model_info_path)
    model_features = [str(name) for name in model_info.get("feature_names", [])]
    if not model_features:
        raise ValueError(f"{model_info_path} does not contain feature_names.")

    model = joblib.load(model_path)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    if hasattr(model, "named_steps"):
        rf_step = model.named_steps.get("rf")
        if rf_step is not None and hasattr(rf_step, "n_jobs"):
            rf_step.n_jobs = 1

    classification.parent.mkdir(parents=True, exist_ok=True)
    confidence.parent.mkdir(parents=True, exist_ok=True)
    prediction_info_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(feature_stack) as src:
        band_names = band_names_from_dataset(src, metadata)
        schema_check = require_feature_stack_schema(
            model_features,
            band_names,
            allow_suffix_match=bool(args.timepoint or args.allow_ambiguous_suffix),
            suffix_prefix=args.timepoint,
        )
        band_indexes = schema_check.selected_band_indexes
        class_profile, confidence_profile = output_profiles(src)
        aoi_path = Path(str(metadata["aoi"])) if metadata and metadata.get("aoi") else None
        aoi_mask = build_aoi_mask(aoi_path, src)

        with rasterio.open(classification, "w", **class_profile) as class_dst, rasterio.open(
            confidence, "w", **confidence_profile
        ) as confidence_dst:
            total_valid = 0
            for window in iter_windows(src):
                data = src.read(band_indexes, window=window, masked=False)
                class_block, confidence_block = predict_window(model, data)
                if aoi_mask is not None:
                    window_mask = aoi_mask[
                        int(window.row_off): int(window.row_off + window.height),
                        int(window.col_off): int(window.col_off + window.width),
                    ]
                    class_block[~window_mask] = NODATA_CLASS
                    confidence_block[~window_mask] = NODATA_CONFIDENCE
                total_valid += int(np.count_nonzero(confidence_block != NODATA_CONFIDENCE))
                class_dst.write(class_block, 1, window=window)
                confidence_dst.write(confidence_block.astype("float32"), 1, window=window)

    selected_bands = [
        {"model_feature": feature, "stack_band_index": index, "stack_band_name": band_names[index - 1]}
        for feature, index in zip(model_features, band_indexes)
    ]
    prediction_info = {
        "feature_stack": str(feature_stack),
        "feature_metadata": str(metadata_path) if metadata_path.exists() else None,
        "model": str(model_path),
        "model_info": str(model_info_path),
        "classification": str(classification),
        "confidence": str(confidence),
        "aoi_mask": str(aoi_path) if aoi_path else None,
        "nodata_class": NODATA_CLASS,
        "nodata_confidence": NODATA_CONFIDENCE,
        "output_classes": {
            "0": "Others",
            "1": "Rice",
            "2": "Wheat",
            "3": "Maize",
            "4": "Rapeseed",
        },
        "class_normalization": "Invalid, outside-AOI, unknown, and missing-feature pixels are written as nodata class 255.",
        "feature_schema_hash": schema_check.schema_hash,
        "feature_schema_matched_by_suffix": schema_check.matched_by_suffix,
        "selected_bands": selected_bands,
        "valid_pixel_count": total_valid,
        "model_classes": model_info.get("classes"),
        "warning": (
            "Prediction raster is a technical output. Its accuracy depends on the training samples; "
            "a single-class or tiny training set is not a valid production classifier."
        ),
    }
    with open(prediction_info_path, "w", encoding="utf-8") as f:
        json.dump(prediction_info, f, indent=2, ensure_ascii=False)

    print(f"Saved classification: {classification}")
    print(f"Saved confidence: {confidence}")
    print(f"Saved prediction info: {prediction_info_path}")
    if model_info.get("classes") and len(model_info["classes"]) < 2:
        print("WARNING: model contains fewer than 2 classes; output is not a meaningful crop classification map.")


if __name__ == "__main__":
    main()
