"""后处理作物分类结果。

读取原始分类图和置信度图，将低置信度、无效值和未知类别像元归为类别 0，
并可选执行小斑块筛除，输出清理后的分类 GeoTIFF 和处理记录 JSON。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from rasterio.features import sieve

from crop_domain.labels import TARGET_LABELS, normalize_output_classes


DEFAULT_CLASSIFICATION = Path("data/output/crop_classification.tif")
DEFAULT_CONFIDENCE = Path("data/output/crop_confidence.tif")
DEFAULT_OUTPUT = Path("data/output/crop_classification_clean.tif")
DEFAULT_INFO = Path("data/output/postprocess_info.json")

NODATA_CONFIDENCE = -9999.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Postprocess crop classification GeoTIFFs.")
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    parser.add_argument("--confidence", type=Path, default=DEFAULT_CONFIDENCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--info", type=Path, default=DEFAULT_INFO)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.50,
        help="Pixels below this confidence are set to class 0.",
    )
    parser.add_argument(
        "--min-patch-pixels",
        type=int,
        default=0,
        help="Remove connected patches smaller than this many pixels. 0 disables sieving.",
    )
    parser.add_argument(
        "--connectivity",
        type=int,
        choices=(4, 8),
        default=8,
        help="Connectivity used by raster sieving.",
    )
    return parser.parse_args()


def validate_inputs(class_src: rasterio.DatasetReader, conf_src: rasterio.DatasetReader) -> None:
    if class_src.width != conf_src.width or class_src.height != conf_src.height:
        raise ValueError("Classification and confidence rasters have different dimensions.")
    if class_src.transform != conf_src.transform:
        raise ValueError("Classification and confidence rasters have different transforms.")
    if class_src.crs != conf_src.crs:
        raise ValueError("Classification and confidence rasters have different CRS.")


def clean_classes(
    classes: np.ndarray,
    confidence: np.ndarray,
    class_nodata: int,
    confidence_nodata: float,
    min_confidence: float,
    min_patch_pixels: int,
    connectivity: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw_valid = np.ones(classes.shape, dtype=bool) if class_nodata is None else classes != class_nodata
    confidence_valid = np.isfinite(confidence) & (confidence != confidence_nodata)
    keep = raw_valid & confidence_valid & (confidence >= min_confidence)

    cleaned = np.where(keep, normalize_output_classes(classes), 0).astype("uint8")
    low_confidence_count = int(np.count_nonzero(raw_valid & confidence_valid & (confidence < min_confidence)))
    missing_confidence_count = int(np.count_nonzero(raw_valid & ~confidence_valid))

    if min_patch_pixels > 1:
        sieved = sieve(
            cleaned,
            size=min_patch_pixels,
            connectivity=connectivity,
            mask=keep,
        ).astype("uint8")
        cleaned = np.where(keep, sieved, 0).astype("uint8")

    stats = class_counts(cleaned)
    info = {
        "raw_valid_pixel_count": int(np.count_nonzero(raw_valid)),
        "kept_pixel_count": int(np.count_nonzero(keep)),
        "low_confidence_pixel_count": low_confidence_count,
        "missing_confidence_pixel_count": missing_confidence_count,
        "class_counts": stats,
    }
    return cleaned, info


def class_counts(classes: np.ndarray) -> dict[str, dict[str, Any]]:
    values, counts = np.unique(classes, return_counts=True)
    result: dict[str, dict[str, Any]] = {}
    for value, count in zip(values.tolist(), counts.tolist()):
        code = int(value)
        result[str(code)] = {
            "label": TARGET_LABELS.get(code, str(code)),
            "pixel_count": int(count),
        }
    return result


def output_profile(src: rasterio.DatasetReader) -> dict[str, Any]:
    profile = src.profile.copy()
    profile.update(
        count=1,
        dtype="uint8",
        nodata=None,
        compress="deflate",
        tiled=True,
        blockxsize=min(256, src.width),
        blockysize=min(256, src.height),
    )
    return profile


def main() -> None:
    args = parse_args()
    if not args.classification.exists():
        raise FileNotFoundError(
            f"Missing classification raster: {args.classification}. "
            "Run python -m pipeline.crop_classification.03_predict_classify first."
        )
    if not args.confidence.exists():
        raise FileNotFoundError(
            f"Missing confidence raster: {args.confidence}. "
            "Run python -m pipeline.crop_classification.03_predict_classify first."
        )
    if not 0.0 <= args.min_confidence <= 1.0:
        raise ValueError("--min-confidence must be between 0 and 1.")
    if args.min_patch_pixels < 0:
        raise ValueError("--min-patch-pixels must be >= 0.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.info.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(args.classification) as class_src, rasterio.open(args.confidence) as conf_src:
        validate_inputs(class_src, conf_src)
        class_nodata = int(class_src.nodata) if class_src.nodata is not None else None
        confidence_nodata = float(conf_src.nodata) if conf_src.nodata is not None else NODATA_CONFIDENCE

        classes = class_src.read(1, masked=False)
        confidence = conf_src.read(1, masked=False)
        cleaned, stats = clean_classes(
            classes,
            confidence,
            class_nodata,
            confidence_nodata,
            args.min_confidence,
            args.min_patch_pixels,
            args.connectivity,
        )

        with rasterio.open(args.output, "w", **output_profile(class_src)) as dst:
            dst.write(cleaned, 1)
            dst.set_band_description(1, "cleaned_crop_class")

    info = {
        "classification": str(args.classification),
        "confidence": str(args.confidence),
        "output": str(args.output),
        "nodata_class": None,
        "parameters": {
            "min_confidence": args.min_confidence,
            "min_patch_pixels": args.min_patch_pixels,
            "connectivity": args.connectivity,
        },
        "stats": stats,
        "notes": (
            "Conservative postprocessing: low-confidence, invalid, and unknown pixels are set to class 0. "
            "Small-patch sieving is optional and controlled by min_patch_pixels."
        ),
    }
    with open(args.info, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"Saved cleaned classification: {args.output}")
    print(f"Saved postprocess info: {args.info}")


if __name__ == "__main__":
    main()

