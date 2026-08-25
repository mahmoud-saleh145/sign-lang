"""
Exports ONE real landmark sample per class (the per-class mean sample,
closest real row to that class's centroid) from the actual training CSV, for
use by the letter-sign visualizer (components/avatar/HandSignerAvatar.tsx).

These are genuine dataset rows -- not synthesized, not hand-authored. We
pick the real sample closest to each class's mean landmark position so the
visualizer shows a representative, low-noise example of each real sign
rather than an arbitrary first row.
"""
import json
import os
import numpy as np
import pandas as pd

RAW_PATH = "/mnt/user-data/uploads/ArSL_dataset.csv"
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "public", "models", "arsl-31", "letterLandmarks.json"
)


def main():
    df = pd.read_csv(RAW_PATH, sep=";")
    coord_cols = [f"{axis}{i}" for i in range(21) for axis in ("x", "y")]

    result = {}
    for label, group in df.groupby("Sign"):
        coords = group[coord_cols].to_numpy()
        centroid = coords.mean(axis=0)
        dists = np.linalg.norm(coords - centroid, axis=1)
        closest_idx = dists.argmin()
        row = group.iloc[closest_idx]
        landmarks = [{"x": float(row[f"x{i}"]), "y": float(row[f"y{i}"]), "z": 0.0} for i in range(21)]
        result[label] = landmarks

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(result, f)
    print(f"Wrote {len(result)} representative real landmark samples to {OUT_PATH}")


if __name__ == "__main__":
    main()