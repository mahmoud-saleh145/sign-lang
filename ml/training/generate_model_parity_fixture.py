"""
Generates ml/tests/fixtures/model_parity_fixture.json: a handful of REAL
feature vectors from the test split plus the exact probability distribution
sklearn's MLPClassifier produces for them. lib/ml/model.test.ts loads this
fixture and asserts the from-scratch TS forward pass reproduces the same
numbers — proving the browser model matches the trained model, not just
"looks similar".

Run this only after ml/training/train.py has produced weights.json.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from ml.preprocessing.features import Point2D, extract_features  # noqa: E402

RAW_PATH = "/mnt/user-data/uploads/ArSL_dataset.csv"
WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "public", "models", "arsl-31", "weights.json"
)
FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "model_parity_fixture.json")
RANDOM_SEED = 42
N_SAMPLES = 15


def row_to_landmarks(row):
    return [Point2D(float(row[f"x{i}"]), float(row[f"y{i}"])) for i in range(21)]


def main():
    df = pd.read_csv(RAW_PATH, sep=";")
    X = np.array([extract_features(row_to_landmarks(row)) for _, row in df.iterrows()])
    classes = sorted(df["Sign"].unique().tolist())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y = df["Sign"].map(class_to_idx).to_numpy()

    # Reproduce the exact same split as train.py so these are real held-out
    # test examples, not samples the model trained on.
    _, X_temp, _, y_temp = train_test_split(X, y, test_size=0.30, random_state=RANDOM_SEED, stratify=y)
    _, X_test, _, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp)

    with open(WEIGHTS_PATH) as f:
        weights = json.load(f)

    mean = np.array(weights["featureMean"])
    std = np.array(weights["featureStd"])

    # Reconstruct the trained classifier's forward pass in numpy (equivalent
    # math to lib/ml/model.ts) to get ground-truth probabilities per sample.
    W1, b1 = np.array(weights["W1"]), np.array(weights["b1"])
    W2, b2 = np.array(weights["W2"]), np.array(weights["b2"])

    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(X_test), size=N_SAMPLES, replace=False)

    fixtures = []
    for i in idx:
        raw_features = X_test[i].tolist()
        std_x = (X_test[i] - mean) / std
        hidden = np.maximum(std_x @ W1 + b1, 0)
        logits = hidden @ W2 + b2
        exp = np.exp(logits - logits.max())
        probs = exp / exp.sum()
        fixtures.append({
            "rawFeatures": raw_features,
            "trueLabel": classes[y_test[i]],
            "expectedProbabilities": probs.tolist(),
        })

    os.makedirs(os.path.dirname(FIXTURE_PATH), exist_ok=True)
    with open(FIXTURE_PATH, "w") as f:
        json.dump({"classes": classes, "samples": fixtures}, f, indent=2)
    print(f"Wrote {len(fixtures)} parity samples to {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
