"""
TEMPORAL HAND IDENTITY, measured on the specific real KArSL signs that the
single-hand extractor was observed to splice.

For each take: how many frames each side was seen in, how many had both, the
worst inter-frame wrist jump per side (a jump larger than a hand can physically
move in 1/30 s is evidence of an identity switch), how many such jumps there
are, how many detection reorderings the tracker absorbed, and how the sides
were named.

USAGE:
    python ml/analysis/measure_tracker.py --input processed-karsl \\
        --signs 0171 0227 0241 0255 0283 0487
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
from handedness import assign_sides  # noqa: E402

IMPOSSIBLE_JUMP = 0.15


def jumps(frames, key):
    out, prev = [], None
    for fr in frames:
        h = (fr.get("hands") or {}).get(key)
        if not h or not h.get("landmarks"):
            prev = None
            continue
        w = (h["landmarks"][0]["x"], h["landmarks"][0]["y"])
        if prev is not None:
            out.append(math.hypot(w[0] - prev[0], w[1] - prev[1]))
        prev = w
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--signs", nargs="+", required=True)
    args = ap.parse_args()

    print("=" * 96)
    print("HAND IDENTITY / TEMPORAL STABILITY")
    print("=" * 96)
    print(f"{'sign':5} {'take':38} {'L':>4} {'R':>4} {'both':>5} {'worstL':>7} {'worstR':>7} "
          f"{'bad':>4} {'swaps':>6} {'sides from':>18} {'conf':>5}")
    totals = {"files": 0, "bad": 0, "swaps": 0, "worst": 0.0, "geometry": 0, "label": 0, "unknown": 0}
    for sign in args.signs:
        d = os.path.join(args.input, sign)
        if not os.path.isdir(d):
            print(f"{sign:5} (not present)")
            continue
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            data = json.load(open(os.path.join(d, name), encoding="utf-8"))
            frames = data["frames"]
            slots = [[], []]
            labels = [[], []]
            for fr in frames:
                hs = fr.get("hands") or {}
                for i, key in enumerate(("left", "right")):
                    h = hs.get(key)
                    slots[i].append(h["landmarks"] if h else None)
                    labels[i].append(h.get("rawLabel") if h else None)
            a = assign_sides(slots[0], slots[1], labels[0], labels[1])

            jl, jr = jumps(frames, "left"), jumps(frames, "right")
            wl = max(jl) if jl else 0.0
            wr = max(jr) if jr else 0.0
            bad = sum(1 for j in jl + jr if j > IMPOSSIBLE_JUMP)
            nl = sum(1 for f in slots[0] if f)
            nr = sum(1 for f in slots[1] if f)
            both = sum(1 for i in range(len(slots[0])) if slots[0][i] and slots[1][i])
            swaps = (data.get("handTracking") or {}).get("slotSwapCount", 0)
            print(f"{sign:5} {name[:38]:38} {nl:4} {nr:4} {both:5} {wl:7.4f} {wr:7.4f} "
                  f"{bad:4} {swaps:6} {a.source:>18} {a.confidence:5.2f}")
            totals["files"] += 1
            totals["bad"] += bad
            totals["swaps"] += swaps
            totals["worst"] = max(totals["worst"], wl, wr)
            totals[{"geometry": "geometry", "handedness-label": "label", "unknown": "unknown"}[a.source]] += 1

    print("-" * 96)
    print(f"takes measured:                 {totals['files']}")
    print(f"impossible wrist jumps (>{IMPOSSIBLE_JUMP}):  {totals['bad']}")
    print(f"worst wrist jump on any side:   {totals['worst']:.4f}")
    print(f"detection reorderings absorbed: {totals['swaps']}")
    print(f"sides named by geometry:        {totals['geometry']}")
    print(f"sides named by label:           {totals['label']}")
    print(f"sides UNKNOWN:                  {totals['unknown']}")
    print("=" * 96)
    return 0


if __name__ == "__main__":
    sys.exit(main())
