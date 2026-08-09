"""
Train the ArSL isolated-sign classifier.

Pipeline: raw landmark CSV -> our own tested feature pipeline (NOT the
dataset's undocumented pre-computed columns, see ml/preprocessing/features.py
for why) -> standardize -> small MLP -> export raw weights as JSON so the
browser can run inference with a from-scratch TypeScript forward pass
(no TensorFlow.js dependency, fully auditable).

Run: python3 ml/training/train.py
"""
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.preprocessing.features import Point2D, extract_features, PREPROCESSING_VERSION  # noqa: E402

RAW_PATH = "/mnt/user-data/uploads/ArSL_dataset.csv"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "public", "models", "arsl-31")
RANDOM_SEED = 42
MODEL_VERSION = "1.0.0"


def row_to_landmarks(row: pd.Series) -> list:
    return [Point2D(float(row[f"x{i}"]), float(row[f"y{i}"])) for i in range(21)]


def main() -> None:
    print("Loading raw dataset...")
    df = pd.read_csv(RAW_PATH, sep=";")
    print(f"  {df.shape[0]} rows, {df.shape[1]} columns")

    print("Computing 89-dim feature vectors via our tested pipeline...")
    X = np.array([extract_features(row_to_landmarks(row)) for _, row in df.iterrows()])
    classes = sorted(df["Sign"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = df["Sign"].map(class_to_idx).to_numpy()
    print(f"  X shape: {X.shape}, classes: {len(classes)}")

    print("\nSplitting 70/15/15 (stratified by class; NOTE: no signer-ID column "
          "exists in this dataset, so we cannot verify signer independence — "
          "documented as a known limitation).")
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
    )
    print(f"  train={len(X_train)} val={len(X_val)} test={len(X_test)}")

    print("\nStandardizing features (fit on train split only)...")
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < 1e-8] = 1e-8  # avoid div-by-zero on any constant feature

    def standardize(X_):
        return (X_ - mean) / std

    Xtr, Xva, Xte = standardize(X_train), standardize(X_val), standardize(X_test)

    print("Training MLP (89 -> 64 -> 31)...")
    clf = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        solver="adam",
        alpha=1e-3,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_SEED,
    )
    clf.fit(Xtr, y_train)
    print(f"  Converged after {clf.n_iter_} iterations.")

    val_pred = clf.predict(Xva)
    test_pred = clf.predict(Xte)

    val_acc = accuracy_score(y_val, val_pred)
    test_acc = accuracy_score(y_test, test_pred)
    test_f1_macro = f1_score(y_test, test_pred, average="macro")

    print(f"\nValidation accuracy: {val_acc:.4f}")
    print(f"Test accuracy:       {test_acc:.4f}")
    print(f"Test macro F1:       {test_f1_macro:.4f}")

    report = classification_report(
        y_test, test_pred, target_names=classes, digits=3, zero_division=0
    )
    print("\nPer-class test report:\n" + report)

    cm = confusion_matrix(y_test, test_pred)

    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Export raw weights for from-scratch TS forward pass ---
    # sklearn MLPClassifier with one hidden layer: coefs_[0] is (89,64),
    # coefs_[1] is (64,31); intercepts_[0] is (64,), intercepts_[1] is (31,)
    weights = {
        "W1": clf.coefs_[0].tolist(),
        "b1": clf.intercepts_[0].tolist(),
        "W2": clf.coefs_[1].tolist(),
        "b2": clf.intercepts_[1].tolist(),
        "featureMean": mean.tolist(),
        "featureStd": std.tolist(),
        "classes": classes,
    }
    with open(os.path.join(OUT_DIR, "weights.json"), "w") as f:
        json.dump(weights, f)
    print(f"\nWrote {os.path.join(OUT_DIR, 'weights.json')}")

    metadata = {
        "modelVersion": MODEL_VERSION,
        "language": "ArSL",
        "languageDisplayName": "Arabic Sign Language (alphabet)",
        "dataset": {
            "name": "ArSL-31 Alphabet Landmark Dataset",
            "source": "Zenodo (uploaded by user), CC BY 4.0",
            "license": "CC BY 4.0",
            "sampleCount": int(df.shape[0]),
            "classCount": len(classes),
        },
        "classes": classes,
        "featureFormat": {"count": 89, "order": "see types/features.ts FEATURE_NAMES"},
        "preprocessingVersion": PREPROCESSING_VERSION,
        "confidenceThreshold": 0.75,
        "temporalWindow": {"minStableFrames": 8, "maxGapFrames": 6},
        "modelArchitecture": "MLP 89-64-31, ReLU hidden, softmax output (sklearn MLPClassifier, exported as raw weights)",
        "evaluation": {
            "testAccuracy": round(float(test_acc), 4),
            "macroF1": round(float(test_f1_macro), 4),
            "signerIndependent": False,
            "notes": "Dataset has no signer-ID column; split is stratified-random, "
                     "not signer-independent. Real risk of inflated accuracy if the "
                     "same signer's near-duplicate frames span splits.",
        },
        "trainingDate": datetime.now(timezone.utc).isoformat(),
    }
    with open(os.path.join(OUT_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote {os.path.join(OUT_DIR, 'metadata.json')}")

    eval_report = {
        "valAccuracy": round(float(val_acc), 4),
        "testAccuracy": round(float(test_acc), 4),
        "testMacroF1": round(float(test_f1_macro), 4),
        "classificationReport": classification_report(
            y_test, test_pred, target_names=classes, digits=3,
            zero_division=0, output_dict=True
        ),
        "confusionMatrix": cm.tolist(),
        "classOrder": classes,
        "trainSize": len(X_train),
        "valSize": len(X_val),
        "testSize": len(X_test),
    }
    eval_dir = os.path.join(os.path.dirname(__file__), "..", "evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    with open(os.path.join(eval_dir, "eval_report.json"), "w") as f:
        json.dump(eval_report, f, indent=2)
    print(f"Wrote {os.path.join(eval_dir, 'eval_report.json')}")


if __name__ == "__main__":
    main()
