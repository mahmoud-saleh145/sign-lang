"""
Validates the raw ArSL dataset before it's allowed anywhere near training.
Run: python3 ml/datasets/validate_dataset.py
"""
import sys
import pandas as pd
import numpy as np

RAW_PATH = "/mnt/user-data/uploads/ArSL_dataset.csv"

EXPECTED_FEATURE_COLS = 89  # 42 raw + 18 mean + 14 angle + 15 distance
EXPECTED_CLASSES = 31


def main() -> int:
    errors = []

    df = pd.read_csv(RAW_PATH, sep=";")
    print(f"Loaded shape: {df.shape}")

    if df.columns[0] != "Sign":
        errors.append(f"First column expected 'Sign', got {df.columns[0]!r}")

    feature_cols = [c for c in df.columns if c != "Sign"]
    print(f"Feature columns: {len(feature_cols)} (expected {EXPECTED_FEATURE_COLS})")
    if len(feature_cols) != EXPECTED_FEATURE_COLS:
        errors.append(
            f"Expected {EXPECTED_FEATURE_COLS} feature columns, got {len(feature_cols)}"
        )

    raw_coord_cols = [c for c in feature_cols if len(c) <= 3 and c[0] in "xy"]
    print(f"Raw coordinate columns: {len(raw_coord_cols)} (expected 42)")

    n_classes = df["Sign"].nunique()
    print(f"Distinct classes: {n_classes} (expected {EXPECTED_CLASSES})")
    if n_classes != EXPECTED_CLASSES:
        errors.append(f"Expected {EXPECTED_CLASSES} classes, got {n_classes}")

    print("\nClass distribution:")
    counts = df["Sign"].value_counts().sort_index()
    for label, count in counts.items():
        print(f"  {label:12s} {count:4d}")
    print(f"  min={counts.min()} max={counts.max()} mean={counts.mean():.1f} std={counts.std():.1f}")

    n_missing = df[feature_cols].isna().sum().sum()
    print(f"\nMissing numeric values: {n_missing}")
    if n_missing > 0:
        errors.append(f"{n_missing} missing values found in feature columns")

    n_missing_labels = df["Sign"].isna().sum()
    if n_missing_labels > 0:
        errors.append(f"{n_missing_labels} rows with missing Sign label")

    non_numeric_report = {}
    for c in feature_cols:
        coerced = pd.to_numeric(df[c], errors="coerce")
        bad = coerced.isna().sum() - df[c].isna().sum()
        if bad > 0:
            non_numeric_report[c] = int(bad)
    if non_numeric_report:
        errors.append(f"Non-numeric values found in columns: {non_numeric_report}")
    else:
        print("All feature columns are fully numeric.")

    n_dupes = df.duplicated().sum()
    print(f"Fully duplicated rows: {n_dupes}")

    x_cols = [c for c in raw_coord_cols if c.startswith("x")]
    y_cols = [c for c in raw_coord_cols if c.startswith("y")]
    x_min, x_max = df[x_cols].min().min(), df[x_cols].max().max()
    y_min, y_max = df[y_cols].min().min(), df[y_cols].max().max()
    print(f"\nRaw x range: [{x_min:.4f}, {x_max:.4f}]")
    print(f"Raw y range: [{y_min:.4f}, {y_max:.4f}]")
    if not (0 <= x_min and x_max <= 1.5):
        errors.append(f"x coordinates outside expected MediaPipe-normalized range: [{x_min}, {x_max}]")

    print(f"\nUnique class labels: {sorted(df['Sign'].unique())}")

    print("\n" + "=" * 50)
    if errors:
        print(f"VALIDATION FAILED with {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    else:
        print("VALIDATION PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
