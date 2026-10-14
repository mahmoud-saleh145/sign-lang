"""
INVENTORY of the raw KArSL video tree, before anything is extracted.

This answers "exactly how much real data do we have, and is any of it
broken?" It is READ-ONLY: it never writes into, moves, renames or deletes a
single raw video. Its only output is a JSON report plus a printed summary.

    python ml/training/inventory_karsl.py --input KArSL \
        --labels KARSL-502_Labels.xlsx --out karsl-inventory.json

WHAT IT REPORTS, and why each item is here
============================================================================
FOLDERS AND COUNTS. Sign folders, videos per sign (min / median / max),
total videos and total bytes. Folder names are kept as strings, zero-padding
intact, because "0071" is the join key to the processed tree and to
`sourceSignId` in the registry -- int() would destroy it.

SIGNER AND SESSION, recovered ONLY from the filename. Real KArSL files are
named like

    03_01_0071_(01_12_16_15_52_41)_c.mp4
    ^^ ^^ ^^^^  ^^^^^^^^ ^^^^^^^^  ^
    |  |  |     date     time      camera/stream suffix
    |  |  signId
    |  second numeric field
    first numeric field

The two leading numeric fields are reported VERBATIM as `field1` and
`field2`. This script does NOT claim which of them is the signer, because
that mapping is not stated anywhere in this repository and guessing it would
put invented metadata into the pipeline. What IS safe, and what the split
logic should key on, is the RECORDING DATE: it is unambiguous in the
filename, and takes sharing a date were recorded in one sitting. Files that
do not match the pattern are reported as unparsed rather than force-fitted.

MALFORMED VIDEOS. With --probe (default on), every file is opened with
OpenCV and its metadata read -- frame count, fps, resolution. Files that
cannot be opened, report zero frames, or report an absurd fps are listed
individually. No frames are decoded, so this costs milliseconds per file.

DUPLICATES. Exact-name collisions cannot occur inside one folder, so the
check that matters is content: files are grouped by (size, md5 of the first
128 KB). That catches a genuinely duplicated recording copied under two
names, cheaply, without hashing tens of gigabytes. It is a strong signal,
not a proof -- a reported pair should be confirmed before anything is
removed, and this script removes nothing.

LABEL COVERAGE. Every folder is matched against KARSL-502_Labels.xlsx using
the same canonicalization the importer uses ("0071" -> "71"), and the report
names the labels with no folder and the folders with no label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from typing import Optional

from console_safe import make_stdio_unicode_safe

try:
    import pandas as pd
except ImportError:
    pd = None

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

#: 03_01_0071_(01_12_16_15_52_41)_c.mp4
#: Deliberately strict: a name that does not match is reported unparsed
#: rather than partially guessed.
KARSL_NAME = re.compile(
    r"^(?P<field1>\d+)_(?P<field2>\d+)_(?P<sign>\d+)_"
    r"\((?P<date>\d{2}_\d{2}_\d{2})_(?P<time>\d{2}_\d{2}_\d{2})\)"
    r"_(?P<suffix>[A-Za-z0-9]+)$"
)

HASH_PREFIX_BYTES = 128 * 1024


def canonicalize_sign_id(raw_id: str) -> str:
    """Same rule as ml/training/import_karsl.py, so the two agree on matching."""
    stripped = raw_id.strip()
    return str(int(stripped)) if stripped.isdigit() else stripped


def prefix_hash(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read(HASH_PREFIX_BYTES))
    return h.hexdigest()


def probe_video(path: str) -> dict:
    """Opens the container and reads metadata only -- no frames are decoded."""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return {"openable": False, "frames": 0, "fps": 0.0, "width": 0, "height": 0}
    try:
        return {
            "openable": True,
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0.0),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
    finally:
        cap.release()


def load_labels(labels_path: Optional[str]) -> dict[str, dict]:
    if not labels_path:
        return {}
    if pd is None:
        print("WARNING: pandas is not installed, so label coverage is not checked.")
        return {}
    frame = pd.read_excel(labels_path)
    out: dict[str, dict] = {}
    for _, row in frame.iterrows():
        out[canonicalize_sign_id(str(row["SignID"]))] = {
            "signId": str(row["SignID"]),
            "arabic": str(row["Sign-Arabic"]),
            "english": str(row["Sign-English"]),
        }
    return out


def main() -> int:
    make_stdio_unicode_safe()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", required=True, help="KArSL root (one folder per sign ID)")
    parser.add_argument("--labels", default=None, help="KARSL-502_Labels.xlsx")
    parser.add_argument("--out", default="karsl-inventory.json")
    parser.add_argument(
        "--no-probe",
        action="store_true",
        help="Skip opening each video. Faster, but malformed files go undetected.",
    )
    parser.add_argument(
        "--no-hash",
        action="store_true",
        help="Skip the duplicate-content check (reads 128 KB per file).",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: --input is not a directory: {args.input}")
        return 1
    if not args.no_probe:
        try:
            import cv2  # noqa: F401
        except ImportError:
            print("ERROR: opencv-python is needed for --probe. Install it, or pass --no-probe.")
            return 1

    labels = load_labels(args.labels)

    entries = sorted(os.listdir(args.input))
    sign_dirs = [d for d in entries if os.path.isdir(os.path.join(args.input, d))]
    stray_files = [d for d in entries if not os.path.isdir(os.path.join(args.input, d))]
    unexpected_names = [d for d in sign_dirs if not re.fullmatch(r"\d{4}", d)]

    signs: dict[str, dict] = {}
    unparsed: list[str] = []
    non_video_files: list[str] = []
    nested_dirs: list[str] = []
    broken: list[dict] = []
    by_content: dict[tuple, list[str]] = defaultdict(list)
    field1_values: dict[str, int] = defaultdict(int)
    field2_values: dict[str, int] = defaultdict(int)
    suffix_values: dict[str, int] = defaultdict(int)
    total_bytes = 0
    total_videos = 0

    for i, sign_id in enumerate(sign_dirs, 1):
        sign_dir = os.path.join(args.input, sign_id)
        videos: list[dict] = []
        dates: dict[str, int] = defaultdict(int)
        signers: dict[str, int] = defaultdict(int)

        for name in sorted(os.listdir(sign_dir)):
            full = os.path.join(sign_dir, name)
            if os.path.isdir(full):
                nested_dirs.append(os.path.join(sign_id, name))
                continue
            if not name.lower().endswith(VIDEO_EXTENSIONS):
                non_video_files.append(os.path.join(sign_id, name))
                continue

            stem = os.path.splitext(name)[0]
            match = KARSL_NAME.match(stem)
            size = os.path.getsize(full)
            record: dict = {"file": name, "bytes": size}

            if match:
                record["field1"] = match.group("field1")
                record["field2"] = match.group("field2")
                record["signIdInName"] = match.group("sign")
                record["sessionDate"] = match.group("date")
                record["recordedAt"] = f"{match.group('date')}_{match.group('time')}"
                record["suffix"] = match.group("suffix")
                record["signIdMatchesFolder"] = (
                    canonicalize_sign_id(match.group("sign")) == canonicalize_sign_id(sign_id)
                )
                dates[match.group("date")] += 1
                signers[f"{match.group('field1')}_{match.group('field2')}"] += 1
                field1_values[match.group("field1")] += 1
                field2_values[match.group("field2")] += 1
                suffix_values[match.group("suffix")] += 1
            else:
                unparsed.append(os.path.join(sign_id, name))

            if not args.no_probe:
                info = probe_video(full)
                record.update(info)
                if not info["openable"] or info["frames"] <= 0 or not (1.0 <= info["fps"] <= 240.0):
                    broken.append({"path": os.path.join(sign_id, name), **info})

            if not args.no_hash:
                try:
                    by_content[(size, prefix_hash(full))].append(os.path.join(sign_id, name))
                except OSError as e:  # unreadable file is itself a finding
                    broken.append({"path": os.path.join(sign_id, name), "error": str(e)})

            videos.append(record)
            total_bytes += size
            total_videos += 1

        signs[sign_id] = {
            "videos": len(videos),
            "sessionDates": dict(sorted(dates.items())),
            "namePrefixes": dict(sorted(signers.items())),
            "files": videos,
        }
        if i % 25 == 0 or i == len(sign_dirs):
            print(f"  ...{i}/{len(sign_dirs)} sign folders, {total_videos} videos", flush=True)

    counts = [s["videos"] for s in signs.values()]
    duplicates = [paths for paths in by_content.values() if len(paths) > 1]

    covered = {canonicalize_sign_id(s) for s in signs}
    labels_without_folder = sorted(
        (k for k in labels if k not in covered), key=lambda k: int(k) if k.isdigit() else 0
    )
    folders_without_label = sorted(s for s in signs if canonicalize_sign_id(s) not in labels) if labels else []

    sessions_per_sign = [len(s["sessionDates"]) for s in signs.values() if s["sessionDates"]]
    multi_prefix_signs = sorted(s for s, v in signs.items() if len(v["namePrefixes"]) > 1)

    report = {
        "input": os.path.abspath(args.input),
        "signFolders": len(sign_dirs),
        "totalVideos": total_videos,
        "totalBytes": total_bytes,
        "videosPerSign": {
            "min": min(counts) if counts else 0,
            "median": statistics.median(counts) if counts else 0,
            "max": max(counts) if counts else 0,
            "mean": round(sum(counts) / len(counts), 2) if counts else 0,
        },
        "signsWithNoVideos": sorted(s for s, v in signs.items() if v["videos"] == 0),
        "nameFields": {
            "field1": dict(sorted(field1_values.items())),
            "field2": dict(sorted(field2_values.items())),
            "suffix": dict(sorted(suffix_values.items())),
        },
        "sessionsPerSign": {
            "min": min(sessions_per_sign) if sessions_per_sign else 0,
            "median": statistics.median(sessions_per_sign) if sessions_per_sign else 0,
            "max": max(sessions_per_sign) if sessions_per_sign else 0,
        },
        "signsWithMoreThanOneNamePrefix": multi_prefix_signs,
        "unparsedFilenames": unparsed,
        "unexpectedFolderNames": unexpected_names,
        "strayFilesAtRoot": stray_files,
        "nestedDirectories": nested_dirs,
        "nonVideoFiles": non_video_files,
        "signIdMismatches": [
            os.path.join(sign_id, f["file"])
            for sign_id, v in signs.items()
            for f in v["files"]
            if f.get("signIdMatchesFolder") is False
        ],
        "malformedVideos": broken,
        "duplicateContentGroups": duplicates,
        "labelCoverage": {
            "labelsLoaded": len(labels),
            "labelsWithFolder": len(labels) - len(labels_without_folder) if labels else 0,
            "labelsWithoutFolder": labels_without_folder,
            "foldersWithoutLabel": folders_without_label,
        },
        "signs": signs,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    bar = "=" * 72
    print(f"\n{bar}\nKArSL RAW INVENTORY\n{bar}")
    print(f"   root                     {report['input']}")
    print(f"   sign folders             {report['signFolders']}")
    print(f"   total videos             {report['totalVideos']}")
    print(f"   total size               {total_bytes / 1024 / 1024 / 1024:.2f} GB")
    v = report["videosPerSign"]
    print(f"   videos per sign          min {v['min']}  median {v['median']}  max {v['max']}  mean {v['mean']}")
    s = report["sessionsPerSign"]
    print(f"   session dates per sign   min {s['min']}  median {s['median']}  max {s['max']}")
    print(f"   filename field1 values   {report['nameFields']['field1']}")
    print(f"   filename field2 values   {report['nameFields']['field2']}")
    print(f"   filename suffix values   {report['nameFields']['suffix']}")
    print(f"   signs w/ >1 name prefix  {len(multi_prefix_signs)}")
    print(f"   unparsed filenames       {len(unparsed)}")
    print(f"   sign-id/folder mismatch  {len(report['signIdMismatches'])}")
    print(f"   malformed videos         {len(broken)}")
    print(f"   duplicate content groups {len(duplicates)}")
    print(f"   unexpected folder names  {unexpected_names or 'none'}")
    print(f"   stray files at root      {stray_files or 'none'}")
    print(f"   nested directories       {len(nested_dirs)}")
    print(f"   non-video files          {len(non_video_files)}")
    lc = report["labelCoverage"]
    print(f"   labels loaded            {lc['labelsLoaded']}")
    print(f"   labels with a folder     {lc['labelsWithFolder']}")
    print(f"   labels with NO folder    {len(lc['labelsWithoutFolder'])}")
    print(f"   folders with NO label    {len(lc['foldersWithoutLabel'])}")
    print(f"\n   full report -> {args.out}")
    print(bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
