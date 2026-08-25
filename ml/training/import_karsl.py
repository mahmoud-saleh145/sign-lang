"""
Imports processed KArSL landmark data into the app's real dedicated-sign
registry format.

============================================================================
SIGN ID MATCHING: zero-padded folders vs. plain-integer labels
============================================================================
Real KArSL video folders are zero-padded (e.g. "0001", "0042", "0502"), but
the real labels spreadsheet (KARSL-502_Labels.xlsx) uses plain integer
SignID values ("1", "42", "502") with no padding at all. These MUST be
matched against each other correctly, or every sign silently fails to find
its label and gets skipped.

canonicalize_sign_id() below strips leading zeros from any purely-numeric
ID string ("0001" -> "1"), so both representations compare equal. This is
applied consistently everywhere a sign ID from a folder name is compared
against a sign ID from the labels file. Non-numeric IDs (for a differently-
structured dataset) are left unchanged rather than assumed to be numeric.

The ORIGINAL, on-disk folder name (e.g. "0001") is still preserved in each
output entry's `sourceSignId` field for traceability back to the source
video -- canonicalization is used only for matching, never for storage.

============================================================================
TWO SUPPORTED INPUT FORMATS
============================================================================
1. PREFERRED, REAL: output of ml/training/extract_karsl_skeletons.py --
   per-sign directories containing per-video JSON files shaped like
   {"signId": ..., "sourceVideo": ..., "frames": [{"frameIndex":...,
   "handDetected":..., "landmarks": [...]|null}, ...]}.

2. FALLBACK, UNVERIFIED: a directory containing one skeleton/joint file per
   FRAME (not per video), for a differently-structured dataset.

============================================================================
HONESTY NOTE
============================================================================
This script is deliberately strict: if the input doesn't match either
format, it fails loudly with a clear message about what it found instead
of silently producing wrong or fabricated data.

USAGE:
    python3 ml/training/import_karsl.py \
        --input processed-karsl \
        --labels /path/to/KARSL-502_Labels.xlsx \
        --output public/models/sign-vocabulary/dedicated-signs.json

WHAT IT DOES NOT DO:
    - It does not download or extract anything from video.
    - It does not fabricate any sign it can't find real data for.
    - It does not touch public/models/arsl-31/ (the alphabet model) at all.
    - If a sign has multiple videos, only the first (alphabetically) is
      used -- other repetitions are reported as available but unused.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

try:
    import pandas as pd
except ImportError:
    pd = None


DATASET_NAME = "KArSL-502"
DATASET_SOURCE = "https://hamzah-luqman.github.io/KArSL/"
DATASET_LICENSE = "Not explicitly stated by the authors beyond a citation request -- VERIFY before redistributing any derived data. See docs/SIGN_LANGUAGE_DATASETS.md."


@dataclass
class ImportStats:
    signs_found: int = 0
    signs_imported: int = 0
    signs_skipped: list[str] = field(default_factory=list)
    frames_total: int = 0


def canonicalize_sign_id(raw_id: str) -> str:
    """Canonical form for matching a KArSL sign ID across representations.

    "0001" and "1" both canonicalize to "1"; "0" stays "0"; "0502" -> "502".
    Purely numeric IDs have leading zeros stripped. Non-numeric IDs (a
    differently-structured dataset might use them) are returned unchanged --
    we don't assume every dataset's IDs are numeric.
    """
    stripped = raw_id.strip()
    if stripped.isdigit():
        return str(int(stripped))
    return stripped


def find_processed_video_jsons(sign_dir: str) -> list[str]:
    if not os.path.isdir(sign_dir):
        return []
    return sorted(
        os.path.join(sign_dir, f) for f in os.listdir(sign_dir) if f.lower().endswith(".json")
    )


def load_processed_video_json(path: str) -> Optional[list[list[dict]]]:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict) or "frames" not in data or not isinstance(data["frames"], list):
        return None

    frames: list[list[dict]] = []
    dropped = 0
    for frame in data["frames"]:
        if not isinstance(frame, dict):
            return None
        landmarks = frame.get("landmarks")
        if landmarks is None:
            dropped += 1
            continue
        try:
            frames.append(
                [{"x": float(p["x"]), "y": float(p["y"]), "z": float(p.get("z", 0.0))} for p in landmarks]
            )
        except (KeyError, TypeError, ValueError):
            return None

    if dropped:
        print(f"    (dropped {dropped}/{len(data['frames'])} frame(s) with no detected hand)")

    return frames if frames else None


def find_skeleton_files(sign_dir: str) -> list[str]:
    candidates = []
    for root, _dirs, files in os.walk(sign_dir):
        for fname in files:
            if fname.lower().endswith((".txt", ".csv", ".json")):
                candidates.append(os.path.join(root, fname))
    return sorted(candidates)


def parse_skeleton_file(path: str) -> Optional[list[dict]]:
    try:
        if path.lower().endswith(".json"):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list) and all(isinstance(p, dict) for p in data):
                return [{"x": float(p["x"]), "y": float(p["y"]), "z": float(p.get("z", 0.0))} for p in data]
            if isinstance(data, list) and all(isinstance(p, list) for p in data):
                return [{"x": float(p[0]), "y": float(p[1]), "z": float(p[2]) if len(p) > 2 else 0.0} for p in data]
            return None

        with open(path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        points = []
        for line in lines:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                return None
            x, y = float(parts[0]), float(parts[1])
            z = float(parts[2]) if len(parts) > 2 else 0.0
            points.append({"x": x, "y": y, "z": z})
        return points if points else None
    except (ValueError, KeyError, json.JSONDecodeError, OSError):
        return None


def load_labels(labels_path: Optional[str]) -> dict[str, str]:
    """Maps sign ID (exactly as written in the labels file, e.g. "1") ->
    Arabic word/meaning. Keys are NOT canonicalized here -- canonicalization
    happens at lookup time in main(), via a canonical index built from this
    dict (see build_canonical_label_index)."""
    if not labels_path or not os.path.exists(labels_path):
        return {}
    if pd is None:
        print("WARNING: pandas not installed; cannot read labels file. Install with:")
        print("  pip install pandas openpyxl --break-system-packages")
        return {}
    try:
        if labels_path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(labels_path, dtype=str)
        else:
            df = pd.read_csv(labels_path, dtype=str)
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: could not read labels file {labels_path}: {e}")
        return {}

    id_col = next((c for c in df.columns if "id" in c.lower() or "sign" in c.lower()), None)
    label_col = next(
        (c for c in df.columns if "arabic" in c.lower() or "meaning" in c.lower() or "label" in c.lower()),
        None,
    )
    if id_col is None or label_col is None:
        print(f"WARNING: could not identify ID/label columns in {labels_path}. Found columns: {list(df.columns)}")
        return {}

    return {str(row[id_col]): str(row[label_col]) for _, row in df.iterrows()}


def build_canonical_label_index(labels: dict[str, str]) -> dict[str, str]:
    """Builds a {canonical_id: label} index from the raw labels dict,
    detecting (and warning about, never silently resolving) any collision
    where two different raw IDs canonicalize to the same value -- that
    would be a genuine data ambiguity, not something to guess through."""
    index: dict[str, str] = {}
    seen_raw_id_for_canonical: dict[str, str] = {}
    for raw_id, label in labels.items():
        canonical = canonicalize_sign_id(raw_id)
        if canonical in index and seen_raw_id_for_canonical[canonical] != raw_id:
            print(
                f"WARNING: sign IDs {seen_raw_id_for_canonical[canonical]!r} and {raw_id!r} "
                f"both canonicalize to {canonical!r} -- keeping the first, skipping the collision "
                f"rather than silently overwriting."
            )
            continue
        index[canonical] = label
        seen_raw_id_for_canonical[canonical] = raw_id
    return index


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the KArSL dataset root directory")
    parser.add_argument("--labels", default=None, help="Path to the labels file (xlsx/csv mapping sign ID -> Arabic word)")
    parser.add_argument(
        "--output",
        default="public/models/sign-vocabulary/dedicated-signs.json",
        help="Where to write the resulting DedicatedSignEntry[] JSON",
    )
    parser.add_argument("--version", default="1.0.0", help="Version string to record in provenance metadata")
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: --input path does not exist or is not a directory: {args.input}")
        return 1

    labels = load_labels(args.labels)
    canonical_labels = build_canonical_label_index(labels)
    if not labels:
        print(
            "WARNING: no labels loaded. Every imported sign needs a real Arabic "
            "word/meaning -- without labels, nothing can be imported (we will "
            "not fabricate placeholder labels like 'sign_001')."
        )

    stats = ImportStats()
    entries = []

    sign_dirs = sorted(
        d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))
    )
    stats.signs_found = len(sign_dirs)

    if stats.signs_found == 0:
        print(f"ERROR: no subdirectories found under {args.input}. Expected one directory per sign.")
        print("If the real structure differs, this script needs adjusting -- please share the actual layout.")
        return 1

    for sign_id in sign_dirs:
        # sign_id here is the ORIGINAL, on-disk folder name (e.g. "0001").
        # Canonicalize only for the label lookup -- never for storage.
        label = canonical_labels.get(canonicalize_sign_id(sign_id))
        if not label:
            stats.signs_skipped.append(f"{sign_id} (no label found for canonical ID {canonicalize_sign_id(sign_id)!r})")
            continue

        sign_dir = os.path.join(args.input, sign_id)

        video_jsons = find_processed_video_jsons(sign_dir)
        frames: list[list[dict]] = []
        used_format = None

        if video_jsons:
            for video_json_path in video_jsons:
                parsed = load_processed_video_json(video_json_path)
                if parsed is not None:
                    frames = parsed
                    used_format = "processed-karsl"
                    if len(video_jsons) > 1:
                        others = [os.path.basename(p) for p in video_jsons if p != video_json_path]
                        print(
                            f"    Using {os.path.basename(video_json_path)}; "
                            f"{len(others)} other repetition(s) available but not used: {others}"
                        )
                    break

        if not frames:
            skeleton_files = find_skeleton_files(sign_dir)
            if not skeleton_files:
                stats.signs_skipped.append(f"{sign_id} (no processed-karsl JSON or .txt/.csv/.json skeleton files found)")
                continue

            for f in skeleton_files:
                parsed = parse_skeleton_file(f)
                if parsed is None:
                    stats.signs_skipped.append(f"{sign_id} (unparseable file: {f})")
                    frames = []
                    break
                frames.append(parsed)
            if frames:
                used_format = "per-frame-file (fallback)"

        if not frames:
            continue

        entries.append(
            {
                "text": label,
                "landmarkSequence": frames,
                "provenance": {
                    "datasetName": DATASET_NAME,
                    "source": DATASET_SOURCE,
                    "license": DATASET_LICENSE,
                    "version": args.version,
                },
                "sourceSignId": sign_id,  # original, zero-padded folder name preserved
            }
        )
        stats.signs_imported += 1
        stats.frames_total += len(frames)
        print(f"    Imported {sign_id} ({label}): {len(frames)} frame(s) via {used_format}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)

    print(f"\nSigns found (directories):  {stats.signs_found}")
    print(f"Signs imported:             {stats.signs_imported}")
    print(f"Signs skipped:              {len(stats.signs_skipped)}")
    if stats.signs_skipped:
        print("Skip reasons (first 20):")
        for reason in stats.signs_skipped[:20]:
            print(f"  - {reason}")
    print(f"Total frames imported:      {stats.frames_total}")
    print(f"\nWrote {args.output}")

    if stats.signs_imported == 0:
        print(
            "\nNo signs were imported. This almost certainly means the real "
            "file structure differs from what this script assumes -- please "
            "share the actual directory/file layout so the parser can be "
            "corrected, rather than guessing further."
        )
        return 1

    print(
        f"\nIMPORTANT: license field says '{DATASET_LICENSE}' -- verify this "
        "is accurate for your copy of the dataset before shipping these "
        "entries to real users."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())