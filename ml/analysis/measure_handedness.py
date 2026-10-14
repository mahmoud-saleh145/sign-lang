"""
MEASUREMENT: how should the canonical left/right side of a KArSL hand track be
determined?

Three candidate signals, measured against each other on the real extracted
data:

  (a) MediaPipe's per-frame `rawLabel`, inverted (the "camera" convention: the
      model assumes a MIRRORED selfie image, KArSL video is not mirrored).
  (b) The same label, but taken as a per-track MAJORITY VOTE rather than from
      the single frame that seeded the slot.
  (c) GEOMETRY: a signer faces the camera, so in an un-mirrored recording the
      signer's RIGHT hand appears on the LEFT of the image (smaller x). On any
      frame where both hands are detected this gives an independent answer that
      uses no classifier at all.

(c) is the only one that is not a model output, so it is the reference the
other two are scored against.

USAGE (from the project root):
    python ml/analysis/measure_handedness.py --input processed-karsl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys


def track_frames(frames, side):
    return [(fr.get("hands") or {}).get(side) for fr in frames]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    args = ap.parse_args()

    files = []
    for sd in sorted(os.listdir(args.input)):
        d = os.path.join(args.input, sd)
        if os.path.isdir(d):
            files += [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".json")]

    n_files = 0
    # label stability within a slot's track
    stable = collections.Counter()
    scores = []
    # geometry vs stored side name
    geo_agree = geo_disagree = 0
    geo_files_agree = geo_files_disagree = 0
    # seed-vs-majority disagreement
    seed_vs_majority = 0
    both_frames_total = 0

    for p in files:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        ht = data.get("handTracking") or {}
        if not ht.get("primaryHand"):
            continue
        n_files += 1
        frames = data["frames"]

        for side in ("left", "right"):
            labels = []
            for h in track_frames(frames, side):
                if h:
                    labels.append(h.get("rawLabel"))
                    scores.append(h.get("score", 0.0))
            if not labels:
                continue
            c = collections.Counter(labels)
            top, top_n = c.most_common(1)[0]
            stable["frames"] += len(labels)
            stable["majority"] += top_n
            if len(c) > 1:
                stable["tracks_with_mixed_labels"] += 1
            stable["tracks"] += 1
            # The extractor seeds the slot name from the FIRST detection.
            if labels[0] != top:
                seed_vs_majority += 1

        # Geometry check, only on frames where BOTH sides are present.
        fa = fd = 0
        for fr in frames:
            hs = fr.get("hands") or {}
            l, r = hs.get("left"), hs.get("right")
            if not l or not r:
                continue
            both_frames_total += 1
            # signer's right hand should be at the SMALLER image x.
            if r["landmarks"][0]["x"] < l["landmarks"][0]["x"]:
                geo_agree += 1
                fa += 1
            else:
                geo_disagree += 1
                fd += 1
        if fa or fd:
            if fa >= fd:
                geo_files_agree += 1
            else:
                geo_files_disagree += 1

    scores.sort()

    def pct(a, b):
        return f"{a}/{b} = {a / b:.4f}" if b else "n/a"

    print("=" * 70)
    print("HANDEDNESS SIGNAL MEASUREMENT")
    print("=" * 70)
    print(f"Files with two-hand data:                {n_files}")
    print(f"Frames with BOTH hands detected:         {both_frames_total}")
    print()
    print("A) MediaPipe rawLabel stability WITHIN one tracked hand")
    print(f"   tracks:                               {stable['tracks']}")
    print(f"   detected frames:                      {stable['frames']}")
    print(f"   frames agreeing with track majority:  {pct(stable['majority'], stable['frames'])}")
    print(f"   tracks whose label CHANGES mid-track: {pct(stable['tracks_with_mixed_labels'], stable['tracks'])}")
    print(f"   tracks where the SEED frame's label")
    print(f"     disagrees with the majority:        {pct(seed_vs_majority, stable['tracks'])}")
    if scores:
        print(f"   handedness score: min {scores[0]:.3f}  p10 {scores[len(scores)//10]:.3f}  "
              f"median {scores[len(scores)//2]:.3f}  max {scores[-1]:.3f}")
    print()
    print("B) GEOMETRY (signer faces camera => signer's right hand has smaller image x)")
    print(f"   vs the stored side names, per frame:  {pct(geo_agree, geo_agree + geo_disagree)}")
    print(f"   per file (majority within the file):  {pct(geo_files_agree, geo_files_agree + geo_files_disagree)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
