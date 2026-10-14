"""
Verifies the raw KArSL video tree BEFORE extraction is started.

Extraction over the full dataset takes hours, so it is worth failing fast on a
structural problem rather than discovering it 300 signs in. This script reads
nothing but directory listings and file names -- it does not decode video --
so it finishes in seconds.

USAGE (from the project root):
    python ml/datasets/verify_karsl_source.py --input KArSL

    # if you expect a specific ID range:
    python ml/datasets/verify_karsl_source.py --input KArSL --expect-min 1 --expect-max 502

Exit code 0 = structure looks usable, 1 = do not start extraction.

WHAT "USABLE" MEANS HERE: every sign directory is named as a zero-padded
integer, contains at least one video with a supported extension, and no
directory is empty. Repetition COUNTS are reported but never treated as an
error -- real KArSL folders legitimately hold different numbers of takes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")


def scan(input_dir: str) -> dict:
    entries = sorted(os.listdir(input_dir))
    sign_dirs = [d for d in entries if os.path.isdir(os.path.join(input_dir, d))]
    stray_files = [f for f in entries if not os.path.isdir(os.path.join(input_dir, f))]

    numeric, non_numeric = [], []
    for d in sign_dirs:
        (numeric if d.isdigit() else non_numeric).append(d)

    per_sign: dict[str, list[str]] = {}
    empty: list[str] = []
    non_video: list[str] = []
    total_bytes = 0

    for d in sign_dirs:
        path = os.path.join(input_dir, d)
        videos: list[str] = []
        for root, _dirs, files in os.walk(path):
            for f in files:
                if f.lower().endswith(SUPPORTED_EXTENSIONS):
                    videos.append(os.path.join(root, f))
                    try:
                        total_bytes += os.path.getsize(os.path.join(root, f))
                    except OSError:
                        pass
                else:
                    non_video.append(os.path.relpath(os.path.join(root, f), input_dir))
        per_sign[d] = sorted(videos)
        if not videos:
            empty.append(d)

    return {
        "sign_dirs": sign_dirs,
        "numeric": numeric,
        "non_numeric": non_numeric,
        "stray_files": stray_files,
        "per_sign": per_sign,
        "empty": empty,
        "non_video": non_video,
        "total_bytes": total_bytes,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="KArSL root (one subdirectory per sign ID)")
    ap.add_argument("--expect-min", type=int, default=None, help="Lowest sign ID expected, e.g. 1")
    ap.add_argument("--expect-max", type=int, default=None, help="Highest sign ID expected, e.g. 502")
    ap.add_argument("--json-out", default=None, help="Optional path to write the full report as JSON")
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: --input is not a directory: {args.input}")
        return 1

    r = scan(args.input)
    counts = {s: len(v) for s, v in r["per_sign"].items()}
    total_videos = sum(counts.values())
    dist = Counter(counts.values())

    print("=" * 62)
    print("KArSL SOURCE VERIFICATION")
    print("=" * 62)
    print(f"Root:                       {os.path.abspath(args.input)}")
    print(f"Sign directories found:     {len(r['sign_dirs'])}")
    print(f"  numeric names:            {len(r['numeric'])}")
    print(f"  NON-numeric names:        {len(r['non_numeric'])} {r['non_numeric'][:10]}")
    print(f"Stray files at root:        {len(r['stray_files'])} {r['stray_files'][:10]}")
    print(f"Total videos:               {total_videos}")
    print(f"Total size:                 {r['total_bytes'] / 1e9:.2f} GB")
    print(f"Empty directories:          {len(r['empty'])} {r['empty'][:20]}")
    print(f"Non-video files inside:     {len(r['non_video'])} {r['non_video'][:5]}")

    if counts:
        vals = sorted(counts.values())
        print(f"Videos per sign:            min {vals[0]}, median {vals[len(vals) // 2]}, max {vals[-1]}")
        print(f"  distribution:             {dict(sorted(dist.items()))}")

    missing: list[int] = []
    if args.expect_min is not None and args.expect_max is not None:
        present = {int(d) for d in r["numeric"]}
        expected = set(range(args.expect_min, args.expect_max + 1))
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        print(f"Expected IDs {args.expect_min}-{args.expect_max}:")
        print(f"  missing:                  {len(missing)} {missing[:20]}")
        print(f"  unexpected:               {len(extra)} {extra[:20]}")

    problems: list[str] = []
    if not r["sign_dirs"]:
        problems.append("no sign directories at all")
    if r["non_numeric"]:
        problems.append(f"{len(r['non_numeric'])} non-numeric directory name(s)")
    if r["empty"]:
        problems.append(f"{len(r['empty'])} empty directory/ies")
    if total_videos == 0:
        problems.append("no video files found")
    if missing:
        problems.append(f"{len(missing)} expected sign ID(s) missing")

    print("-" * 62)
    if problems:
        print("DO NOT START EXTRACTION. Problems:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Structure matches what extract_karsl_skeletons.py expects.")
        print("Safe to start extraction.")
    print("=" * 62)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "signDirs": len(r["sign_dirs"]),
                    "totalVideos": total_videos,
                    "totalBytes": r["total_bytes"],
                    "videosPerSign": counts,
                    "empty": r["empty"],
                    "nonNumeric": r["non_numeric"],
                    "strayFiles": r["stray_files"],
                    "nonVideoFiles": r["non_video"],
                    "missingIds": missing,
                    "problems": problems,
                },
                f,
                indent=1,
            )
        print(f"Wrote {args.json_out}")

    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
