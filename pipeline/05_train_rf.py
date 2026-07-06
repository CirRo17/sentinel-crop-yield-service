"""步骤 05：从已准备好的训练特征训练分类模型。

Supports pixel-level training data prepared by step 04.
training data.  Feature names are read from the .npz file — no hard-coded
schema dependency, so fast-mode (3 features) and full-mode (11 features)
both work without code changes.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from crop_classifier_core.config import TARGET_LABELS
from crop_classifier_core.feature_schema import duplicate_names, schema_hash


DATA_DIR = Path("data/exported")
MODELS_DIR = Path("models")
TRAINING_DATA_FILE = DATA_DIR / "pixel_training_data.npz"
MODEL_FILE = MODELS_DIR / "crop_classifier.joblib"
MODEL_INFO_FILE = MODELS_DIR / "model_info.json"


def _load_training_data() -> tuple[np.ndarray, np.ndarray, list[str], str]:
    if not TRAINING_DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {TRAINING_DATA_FILE}. Run python -m pipeline.04_prepare_samples first."
        )

    data = np.load(TRAINING_DATA_FILE, allow_pickle=True)
    X = data["X"].astype("float32")
    y = data["y"]

    if "feature_names" in data:
        feature_names = [str(name) for name in data["feature_names"].tolist()]
    else:
        raise ValueError(
            "training_data.npz is missing 'feature_names'. "
            "Re-run step 04 to regenerate."
        )

    if X.ndim != 2 or X.shape[1] != len(feature_names):
        raise ValueError(
            f"X shape {X.shape} does not match {len(feature_names)} features."
        )
    duplicates = duplicate_names(feature_names)
    if duplicates:
        raise ValueError(f"训练特征名存在重复：{', '.join(duplicates)}")

    source = "Unknown prepared pixel-level samples"
    if "source" in data:
        source_value = data["source"]
        source = str(source_value.tolist() if hasattr(source_value, "tolist") else source_value)

    return X, y, feature_names, source


def train_model() -> None:
    MODELS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("Train crop classifier (pixel-level)")
    print("=" * 60)

    X, y, feature_names, training_source = _load_training_data()
    print(f"Training data: {TRAINING_DATA_FILE}")
    print(f"Training source: {training_source}")
    print(f"Samples: {len(X):,}")
    print(f"Features: {X.shape[1]}  ({', '.join(feature_names)})")
    print(f"Classes: {sorted(np.unique(y).tolist())}")
    for cls in sorted(np.unique(y)):
        cnt = int((y == cls).sum())
        name = TARGET_LABELS.get(int(cls), str(cls))
        print(f"  class {cls} ({name}): {cnt:,} samples")

    # stratify requires ≥2 samples per class; skip when sample counts are low.
    unique, counts = np.unique(y, return_counts=True)
    can_stratify = bool(np.all(counts >= 2))
    split_kwargs: dict = dict(test_size=0.2, random_state=42)
    if can_stratify:
        split_kwargs["stratify"] = y
    X_train, X_test, y_train, y_test = train_test_split(X, y, **split_kwargs)

    # Pipeline: StandardScaler -> RandomForest
    # StandardScaler handles mixed-source reflectance / DN ranges.
    n_samples = len(X_train)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=100 if n_samples > 5000 else 300,
            max_depth=16 if n_samples > 5000 else 24,
            min_samples_split=20 if n_samples > 5000 else 10,
            min_samples_leaf=10 if n_samples > 5000 else 4,
            class_weight="balanced",
            random_state=42,
            n_jobs=1,
        )),
    ])
    model.fit(X_train, y_train)

    train_score = float(model.score(X_train, y_train))
    test_score = float(model.score(X_test, y_test))
    print(f"Train accuracy: {train_score:.4f}")
    print(f"Test accuracy: {test_score:.4f}")

    labels_in_test = sorted(np.unique(y_test).tolist())
    target_names = [TARGET_LABELS.get(int(label), str(label)) for label in labels_in_test]
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            model.predict(X_test),
            labels=labels_in_test,
            target_names=target_names,
            digits=3,
            zero_division=0,
        )
    )

    joblib.dump(model, MODEL_FILE)

    model_info = {
        "model_type": "RandomForestClassifier",
        "feature_names": feature_names,
        "feature_schema_hash": schema_hash(feature_names),
        "label_mapping": {str(key): value for key, value in TARGET_LABELS.items()},
        "training_data_file": str(TRAINING_DATA_FILE),
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "classes": [int(label) for label in sorted(np.unique(y).tolist())],
        "train_accuracy": train_score,
        "test_accuracy": test_score,
        "training_source": training_source,
        "note": f"Pixel-level model trained from {training_source}.",
    }
    with open(MODEL_INFO_FILE, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=True)

    print(f"\nSaved model: {MODEL_FILE}")
    print(f"Saved model info: {MODEL_INFO_FILE}")


if __name__ == "__main__":
    train_model()
