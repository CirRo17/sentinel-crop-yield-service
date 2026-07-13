"""评估作物分类精度。

将后处理后的分类图与独立验证数据进行对比，输出混淆矩阵、总体精度、
Kappa 系数、生产者精度和用户精度。验证数据可以是标签栅格，也可以是
包含坐标和标签字段的 CSV 点样本。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from crop_domain.labels import TARGET_LABELS


DEFAULT_CLASSIFICATION = Path("data/output/crop_classification/crop_classification_clean.tif")
DEFAULT_REPORT = Path("data/output/accuracy_eval/accuracy_report.json")
DEFAULT_CONFUSION = Path("data/output/accuracy_eval/confusion_matrix.csv")
DEFAULT_CLASS_ACCURACY = Path("data/output/accuracy_eval/class_accuracy.csv")
DEFAULT_IGNORE_LABELS = [255]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate crop classification accuracy.")
    parser.add_argument("--classification", type=Path, default=DEFAULT_CLASSIFICATION)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--reference-raster", type=Path, help="Validation label raster.")
    source.add_argument("--reference-csv", type=Path, help="Validation point CSV with x,y,label columns.")
    parser.add_argument("--x-column", default="x")
    parser.add_argument("--y-column", default="y")
    parser.add_argument("--label-column", default="label")
    parser.add_argument(
        "--csv-crs",
        default=None,
        help="CRS of CSV x/y columns, for example EPSG:4326. Defaults to classification raster CRS.",
    )
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--confusion-matrix", type=Path, default=DEFAULT_CONFUSION)
    parser.add_argument("--class-accuracy", type=Path, default=DEFAULT_CLASS_ACCURACY)
    parser.add_argument(
        "--ignore-labels",
        type=int,
        nargs="*",
        default=DEFAULT_IGNORE_LABELS,
        help="Reference or predicted labels excluded from evaluation.",
    )
    parser.add_argument(
        "--classes",
        type=int,
        nargs="*",
        default=None,
        help="Explicit class codes to report. Defaults to observed classes plus target classes.",
    )
    return parser.parse_args()


def read_reference_raster(
    classification_path: Path,
    reference_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    with rasterio.open(classification_path) as pred_src, rasterio.open(reference_path) as ref_src:
        prediction = pred_src.read(1, masked=False)
        if (
            ref_src.crs == pred_src.crs
            and ref_src.transform == pred_src.transform
            and ref_src.width == pred_src.width
            and ref_src.height == pred_src.height
        ):
            reference = ref_src.read(1, masked=False)
            aligned = "native"
        else:
            with WarpedVRT(
                ref_src,
                crs=pred_src.crs,
                transform=pred_src.transform,
                width=pred_src.width,
                height=pred_src.height,
                resampling=Resampling.nearest,
            ) as vrt:
                reference = vrt.read(1, masked=False)
            aligned = "warped_to_classification_grid"

        metadata = {
            "source_type": "raster",
            "classification_crs": str(pred_src.crs),
            "reference_crs": str(ref_src.crs),
            "alignment": aligned,
            "width": pred_src.width,
            "height": pred_src.height,
        }
    return reference, prediction, metadata


def read_reference_csv(
    classification_path: Path,
    reference_path: Path,
    x_column: str,
    y_column: str,
    label_column: str,
    csv_crs: str | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    xs: list[float] = []
    ys: list[float] = []
    labels: list[int] = []

    with open(reference_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        required = {x_column, y_column, label_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Reference CSV is missing columns: {sorted(missing)}")
        for row in reader:
            try:
                xs.append(float(row[x_column]))
                ys.append(float(row[y_column]))
                labels.append(int(float(row[label_column])))
            except ValueError as exc:
                raise ValueError(f"Invalid CSV sample row: {row}") from exc

    with rasterio.open(classification_path) as pred_src:
        sample_xs = xs
        sample_ys = ys
        if csv_crs and pred_src.crs and str(pred_src.crs) != csv_crs:
            transformer = Transformer.from_crs(csv_crs, pred_src.crs, always_xy=True)
            sample_xs, sample_ys = transformer.transform(xs, ys)
        coords = list(zip(sample_xs, sample_ys))
        predictions = np.array([value[0] for value in pred_src.sample(coords)], dtype="int32")
        metadata = {
            "source_type": "csv",
            "classification_crs": str(pred_src.crs),
            "csv_crs": csv_crs or str(pred_src.crs),
            "sample_count": len(labels),
            "x_column": x_column,
            "y_column": y_column,
            "label_column": label_column,
        }

    return np.array(labels, dtype="int32"), predictions, metadata


def clean_pairs(
    reference: np.ndarray,
    prediction: np.ndarray,
    ignore_labels: list[int],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    ref = reference.reshape(-1).astype("int32")
    pred = prediction.reshape(-1).astype("int32")
    finite = np.isfinite(ref) & np.isfinite(pred)
    ignore = np.isin(ref, ignore_labels) | np.isin(pred, ignore_labels)
    keep = finite & ~ignore
    stats = {
        "raw_pair_count": int(ref.size),
        "evaluated_pair_count": int(np.count_nonzero(keep)),
        "ignored_pair_count": int(ref.size - np.count_nonzero(keep)),
    }
    return ref[keep], pred[keep], stats


def choose_classes(reference: np.ndarray, prediction: np.ndarray, explicit_classes: list[int] | None) -> list[int]:
    if explicit_classes:
        return sorted(set(int(value) for value in explicit_classes))
    observed = set(int(value) for value in np.unique(reference).tolist())
    observed.update(int(value) for value in np.unique(prediction).tolist())
    observed.update(code for code in TARGET_LABELS if code != 255)
    return sorted(observed)


def confusion_matrix(reference: np.ndarray, prediction: np.ndarray, classes: list[int]) -> np.ndarray:
    index = {code: idx for idx, code in enumerate(classes)}
    matrix = np.zeros((len(classes), len(classes)), dtype="int64")
    for ref, pred in zip(reference.tolist(), prediction.tolist()):
        if ref in index and pred in index:
            matrix[index[ref], index[pred]] += 1
    return matrix


def accuracy_metrics(matrix: np.ndarray, classes: list[int]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    total = int(matrix.sum())
    correct = int(np.trace(matrix))
    overall_accuracy = float(correct / total) if total else None

    row_totals = matrix.sum(axis=1)
    col_totals = matrix.sum(axis=0)
    expected = float((row_totals * col_totals).sum() / (total * total)) if total else None
    if total and expected is not None and expected < 1:
        kappa = float((overall_accuracy - expected) / (1 - expected))
    else:
        kappa = None

    per_class: list[dict[str, Any]] = []
    for idx, code in enumerate(classes):
        diagonal = int(matrix[idx, idx])
        reference_count = int(row_totals[idx])
        predicted_count = int(col_totals[idx])
        producer_accuracy = float(diagonal / reference_count) if reference_count else None
        user_accuracy = float(diagonal / predicted_count) if predicted_count else None
        per_class.append(
            {
                "class_code": int(code),
                "label": TARGET_LABELS.get(int(code), str(code)),
                "reference_count": reference_count,
                "predicted_count": predicted_count,
                "correct_count": diagonal,
                "producer_accuracy": producer_accuracy,
                "user_accuracy": user_accuracy,
            }
        )

    summary = {
        "sample_count": total,
        "correct_count": correct,
        "overall_accuracy": overall_accuracy,
        "kappa": kappa,
    }
    return summary, per_class


def write_confusion_csv(path: Path, matrix: np.ndarray, classes: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["reference\\prediction", *classes, "row_total"])
        row_totals = matrix.sum(axis=1)
        for code, row, total in zip(classes, matrix.tolist(), row_totals.tolist()):
            writer.writerow([code, *row, int(total)])
        writer.writerow(["col_total", *matrix.sum(axis=0).astype(int).tolist(), int(matrix.sum())])


def write_class_accuracy_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "class_code",
        "label",
        "reference_count",
        "predicted_count",
        "correct_count",
        "producer_accuracy",
        "user_accuracy",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if not args.classification.exists():
        raise FileNotFoundError(
            f"Missing classification raster: {args.classification}. "
            "Run python -m pipeline.crop_classification.04_postprocess first."
        )

    if args.reference_raster:
        if not args.reference_raster.exists():
            raise FileNotFoundError(f"Missing reference raster: {args.reference_raster}")
        reference, prediction, source_metadata = read_reference_raster(args.classification, args.reference_raster)
        reference_source = str(args.reference_raster)
    else:
        if not args.reference_csv.exists():
            raise FileNotFoundError(f"Missing reference CSV: {args.reference_csv}")
        reference, prediction, source_metadata = read_reference_csv(
            args.classification,
            args.reference_csv,
            args.x_column,
            args.y_column,
            args.label_column,
            args.csv_crs,
        )
        reference_source = str(args.reference_csv)

    reference_eval, prediction_eval, pair_stats = clean_pairs(reference, prediction, args.ignore_labels)
    if reference_eval.size == 0:
        raise ValueError("No valid validation pairs remain after applying ignore labels.")

    classes = choose_classes(reference_eval, prediction_eval, args.classes)
    matrix = confusion_matrix(reference_eval, prediction_eval, classes)
    summary, per_class = accuracy_metrics(matrix, classes)

    write_confusion_csv(args.confusion_matrix, matrix, classes)
    write_class_accuracy_csv(args.class_accuracy, per_class)

    report = {
        "classification": str(args.classification),
        "reference": reference_source,
        "source": source_metadata,
        "ignore_labels": args.ignore_labels,
        "classes": [
            {"class_code": int(code), "label": TARGET_LABELS.get(int(code), str(code))}
            for code in classes
        ],
        "pair_stats": pair_stats,
        "summary": summary,
        "per_class": per_class,
        "outputs": {
            "confusion_matrix": str(args.confusion_matrix),
            "class_accuracy": str(args.class_accuracy),
            "report": str(args.report),
        },
        "notes": (
            "Accuracy is meaningful only when reference samples are independent, correctly labeled, "
            "and representative of the target AOI and season."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Saved accuracy report: {args.report}")
    print(f"Saved confusion matrix: {args.confusion_matrix}")
    print(f"Saved class accuracy: {args.class_accuracy}")
    print(f"Overall accuracy: {summary['overall_accuracy']}")
    print(f"Kappa: {summary['kappa']}")


if __name__ == "__main__":
    main()

