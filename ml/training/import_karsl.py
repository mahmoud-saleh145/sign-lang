"""
Imports processed KArSL landmark data into the app's real dedicated-sign
registry format.

============================================================================
TWO SUPPORTED INPUT FORMATS
============================================================================
1. PREFERRED, REAL: output of ml/training/extract_karsl_skeletons.py --
   per-sign directories containing per-video JSON files shaped like
   {"signId": ..., "sourceVideo": ..., "frames": [{"frameIndex":...,
   "handDetected":..., "landmarks": [...]|null}, ...]}. This is our own
   controlled pipeline's real, known output -- run that script against real
   KArSL video first (see its own docstring), then run this one against its
   output directory.

2. FALLBACK, UNVERIFIED: a directory containing one skeleton/joint file per
   FRAME (not per video). This was this importer's original assumption,
   written from KArSL's published documentation before any real file
   format was known. It is kept only as a fallback for a differently-
   structured dataset; prefer format 1 whenever you have real KArSL videos,
   since extract_karsl_skeletons.py's output format is not a guess.

============================================================================
HONESTY NOTE
============================================================================
This script is deliberately strict: if the input doesn't match either
format, it fails loudly with a clear message about what it found instead
of silently producing wrong or fabricated data.

USAGE:
    # From real extracted video landmarks (preferred):
    python3 ml/training/import_karsl.py \
        --input processed-karsl \
        --labels /path/to/karsl/labels.xlsx  \
        --output public/models/sign-vocabulary/dedicated-signs.json

    # From a differently-structured per-frame-file dataset (fallback):
    python3 ml/training/import_karsl.py \
        --input /path/to/other/dataset \
        --labels /path/to/labels.xlsx \
        --output public/models/sign-vocabulary/dedicated-signs.json

WHAT IT DOES NOT DO:
    - It does not download or extract anything from video -- that's
      extract_karsl_skeletons.py's job.
    - It does not fabricate any sign it can't find real data for -- signs
      with unparseable or missing data are skipped and reported, not guessed.
    - It does not touch public/models/arsl-31/ (the alphabet model) at all.
    - If a sign has multiple videos, only the first (alphabetically) is
      used -- other repetitions are reported as available but unused, never
      silently merged or averaged together.
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
    pd = None  # only needed if --labels is an .xlsx/.csv file


DATASET_NAME = "KArSL-502"
DATASET_SOURCE = "https://hamzah-luqman.github.io/KArSL/"
DATASET_LICENSE = "Not explicitly stated by the authors beyond a citation request -- VERIFY before redistributing any derived data. See docs/SIGN_LANGUAGE_DATASETS.md."


@dataclass
class ImportStats:
    signs_found: int = 0
    signs_imported: int = 0
    signs_skipped: list[str] = field(default_factory=list)
    frames_total: int = 0


def find_processed_video_jsons(sign_dir: str) -> list[str]:
    """Finds per-video JSON files directly under a sign's directory (not
    recursive -- extract_karsl_skeletons.py writes exactly one JSON per
    video at this level)."""
    if not os.path.isdir(sign_dir):
        return []
    return sorted(
        os.path.join(sign_dir, f) for f in os.listdir(sign_dir) if f.lower().endswith(".json")
    )


def load_processed_video_json(path: str) -> Optional[list[list[dict]]]:
    """Loads one real processed-karsl per-video JSON (see
    extract_karsl_skeletons.py's docstring for the exact shape) and returns
    its frame landmark sequence, with frames that had no detected hand
    DROPPED (never fabricated) -- the number dropped is printed so it's
    never silently lost. Returns None if `path` isn't actually in this
    format at all (so the caller can fall back to the old per-frame-file
    assumption instead)."""
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
    """Looks for per-frame skeleton/joint files under a sign's directory.
    KArSL's documentation describes per-frame joint-position files; the
    exact extension/naming has not been verified against real files, so
    this checks a few plausible patterns and reports what it actually finds."""
    candidates = []
    for root, _dirs, files in os.walk(sign_dir):
        for fname in files:
            if fname.lower().endswith((".txt", ".csv", ".json")):
                candidates.append(os.path.join(root, fname))
    return sorted(candidates)


def parse_skeleton_file(path: str) -> Optional[list[dict]]:
    """Parses one skeleton/joint file into a list of {x, y, z} points.
    Expects either:
      - a JSON array of [x, y, z] triples or {x,y,z} objects, OR
      - a whitespace/comma-delimited text file with one point per line
        ("x y z" or "x,y,z").
    Returns None (not a fabricated guess) if the format doesn't match either
    of these -- caller is responsible for reporting the skip.
    """
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
    """Maps sign ID -> Arabic word/meaning. Returns {} if no labels file is
    given or it can't be read -- signs without a label are skipped later,
    never given a fabricated placeholder label."""
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
    except Exception as e:  # noqa: BLE001 -- report and continue with no labels
        print(f"WARNING: could not read labels file {labels_path}: {e}")
        return {}

    # KArSL's public label sheet format hasn't been verified; try the most
    # plausible column names and report what was actually found otherwise.
    id_col = next((c for c in df.columns if "id" in c.lower() or "sign" in c.lower()), None)
    label_col = next((c for c in df.columns if "arabic" in c.lower() or "meaning" in c.lower() or "label" in c.lower()), None)
    if id_col is None or label_col is None:
        print(f"WARNING: could not identify ID/label columns in {labels_path}. Found columns: {list(df.columns)}")
        return {}

    return {str(row[id_col]): str(row[label_col]) for _, row in df.iterrows()}


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
        label = labels.get(sign_id)
        if not label:
            stats.signs_skipped.append(f"{sign_id} (no label found)")
            continue

        sign_dir = os.path.join(args.input, sign_id)

        # 1. Preferred: real processed-karsl output (one JSON per video).
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

        # 2. Fallback: old per-frame-file assumption, for a differently-structured dataset.
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