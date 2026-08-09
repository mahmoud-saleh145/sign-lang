"""
Exports real raw landmark rows for two distinct classes, so the TS
integration test can feed genuine hand landmark sequences (not synthetic
data) through the real trained model + real segmenter and verify continuous
recognition behavior end-to-end.
"""
import json
import os
import pandas as pd

RAW_PATH = "/mnt/user-data/uploads/ArSL_dataset.csv"
OUT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "tests", "fixtures", "continuous_stream_fixture.json"
)


def rows_for_class(df: pd.DataFrame, label: str, n: int):
    subset = df[df["Sign"] == label].head(n)
    landmarks_per_row = []
    for _, row in subset.iterrows():
        landmarks_per_row.append([
            {"x": float(row[f"x{i}"]), "y": float(row[f"y{i}"]), "z": 0.0} for i in range(21)
        ])
    return landmarks_per_row


def main():
    df = pd.read_csv(RAW_PATH, sep=";")
    fixture = {
        "Alef": rows_for_class(df, "Alef", 20),
        "Ba2": rows_for_class(df, "Ba2", 20),
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(fixture, f)
    print(f"Wrote {sum(len(v) for v in fixture.values())} real landmark frames to {OUT_PATH}")


if __name__ == "__main__":
    main()
