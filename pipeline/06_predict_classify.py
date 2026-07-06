"""Step 06: run whole-AOI crop classification.

This step applies the trained Random Forest model to a step-03b feature stack
and writes pixel-level classification and confidence GeoTIFFs.

Important: this script only makes the raster prediction workflow real. The
current model is only as good as the samples produced by step 04.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import rasterio
from rasterio.windows import Window

from crop_classifier_core.config import normalize_output_classes
from crop_classifier_core.feature_schema import band_names_from_dataset, require_feature_stack_schema


DEFAULT_FEATURE_STACK = Path("data/exported/feature_stack.tif")
DEFAULT_METADATA = Path("data/exported/feature_stack_metadata.json")
DEFAULT_MODEL = Path("models/crop_classifier.joblib")
DEFAULT_MODEL_INFO = Path("models/model_info.json")
DEFAULT_CLASSIFICATION = Path("data/output/crop_classification.tif")
DEFAULT_CONFIDENCE = Path("data/output/crop_confidence.tif")
DEFAULT_PREDICTION_INFO = Path("data/output/prediction_info.json")

NODATA_CONFIDENCE = -9999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict crop classes for a whole-AOI feature stack.")
    parser.add_argument("--feature-stack", type=Path, default=DEFAULT_FEATURE_STACK)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--model-info", type=Path, default=DEFAULT_MODEL_INFO)
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--confidence", type=Path, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--prediction-info", type=Path, default=DEFAULT_PREDICTION_INFO)
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


def output_profiles(src: rasterio.DatasetReader) -> tuple[dict[str, Any], dict[str, Any]]:
    class_profile = src.profile.copy()
    class_profile.update(
        count=1,
        dtype="uint8",
        nodata=None,
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

    classes = np.zeros(flat.shape[0], dtype="uint8")
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
    if not args.feature_stack.exists():
        raise FileNotFoundError(f"Missing feature stack: {args.feature_stack}. Run step 03b first.")
    if not args.model.exists():
        raise FileNotFoundError(f"Missing model: {args.model}. Run step 05 first.")

    metadata = load_json(args.metadata) if args.metadata.exists() else None
    model_info = load_json(args.model_info)
    model_features = [str(name) for name in model_info.get("feature_names", [])]
    if not model_features:
        raise ValueError(f"{args.model_info} does not contain feature_names.")

    model = joblib.load(args.model)
    if hasattr(model, "n_jobs"):
        model.n_jobs = 1
    if hasattr(model, "named_steps"):
        rf_step = model.named_steps.get("rf")
        if rf_step is not None and hasattr(rf_step, "n_jobs"):
            rf_step.n_jobs = 1

    args.classification.parent.mkdir(parents=True, exist_ok=True)
    args.confidence.parent.mkdir(parents=True, exist_ok=True)
    args.prediction_info.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.feature_stack) as src:
        band_names = band_names_from_dataset(src, metadata)
        schema_check = require_feature_stack_schema(
            model_features,
            band_names,
            allow_suffix_match=bool(args.timepoint or args.allow_ambiguous_suffix),
            suffix_prefix=args.timepoint,
        )
        band_indexes = schema_check.selected_band_indexes
        class_profile, confidence_profile = output_profiles(src)

        with rasterio.open(args.classification, "w", **class_profile) as class_dst, rasterio.open(
            args.confidence, "w", **confidence_profile
        ) as confidence_dst:
            total_valid = 0
            for window in iter_windows(src):
                data = src.read(band_indexes, window=window, masked=False)
                class_block, confidence_block = predict_window(model, data)
                total_valid += int(np.count_nonzero(confidence_block != NODATA_CONFIDENCE))
                class_dst.write(class_block, 1, window=window)
                confidence_dst.write(confidence_block.astype("float32"), 1, window=window)

    selected_bands = [
        {"model_feature": feature, "stack_band_index": index, "stack_band_name": band_names[index - 1]}
        for feature, index in zip(model_features, band_indexes)
    ]
    prediction_info = {
        "feature_stack": str(args.feature_stack),
        "feature_metadata": str(args.metadata) if args.metadata.exists() else None,
        "model": str(args.model),
        "model_info": str(args.model_info),
        "classification": str(args.classification),
        "confidence": str(args.confidence),
        "nodata_class": None,
        "nodata_confidence": NODATA_CONFIDENCE,
        "output_classes": {
            "0": "Others",
            "1": "Rice",
            "2": "Wheat",
            "3": "Maize",
            "4": "Rapeseed",
        },
        "class_normalization": "Invalid, unknown, and missing-feature pixels are written as public class 0.",
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
    with open(args.prediction_info, "w", encoding="utf-8") as f:
        json.dump(prediction_info, f, indent=2, ensure_ascii=False)

    print(f"Saved classification: {args.classification}")
    print(f"Saved confidence: {args.confidence}")
    print(f"Saved prediction info: {args.prediction_info}")
    if model_info.get("classes") and len(model_info["classes"]) < 2:
        print("WARNING: model contains fewer than 2 classes; output is not a meaningful crop classification map.")


if __name__ == "__main__":
    main()
