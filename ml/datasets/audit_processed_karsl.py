"""
Quality audit of a processed-karsl/ tree, to be run BETWEEN extraction and
import_karsl.py.

The single most important check here is HAND IDENTITY CONTINUITY. The old
num_hands=1 extractor did not pin its one returned hand to one physical hand,
so a stored "temporal sequence" could be the left hand for the first half of a
sign spliced onto the right hand for the second half. Measured on real KArSL
video, that produced inter-frame wrist jumps of up to 0.237 in image units --
a quarter of the frame width in 1/30 s, which no hand can do. The two-hand
extractor tracks each hand as a persistent slot by nearest-wrist continuity;
this script verifies that the property actually holds in the output rather
than assuming it.

USAGE (from the project root):
    python ml/datasets/audit_processed_karsl.py --input processed-karsl

    # stricter minimum sequence length, and a JSON report:
    python ml/datasets/audit_processed_karsl.py --input processed-karsl \\
        --min-frames 8 --json-out karsl-audit.json

Exit code 0 = no blocking issues, 1 = issues that should be looked at before
importing.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
from collections import Counter

# Any single-frame wrist displacement above this is treated as physically
# impossible for a real hand and therefore evidence of an identity switch.
# Justification from real data: with nearest-wrist tracking the worst observed
# jump across sampled videos was 0.073; without it, swaps produced 0.21-0.24.
# 0.15 sits clearly between the two populations.
IMPOSSIBLE_JUMP = 0.15

DEFAULT_MIN_FRAMES = 8


def wrist(points):
    return points[0]["x"], points[0]["y"]


def dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def track_jumps(frames, side):
    """Inter-frame wrist displacements for one hand's track. A gap (hand not
    detected for a while) resets the chain -- a re-acquisition after an absence
    is not a jump."""
    out = []
    prev = None
    for fr in frames:
        h = (fr.get("hands") or {}).get(side)
        if not h or not h.get("landmarks"):
            prev = None
            continue
        w = wrist(h["landmarks"])
        if prev is not None:
            out.append(dist(w, prev))
        prev = w
    return out


def audit_file(path, sign_id, min_frames):
    issues = []
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        return {"corrupt": True, "issues": [f"corrupt JSON: {e}"]}

    frames = data.get("frames")
    if not isinstance(frames, list):
        return {"corrupt": True, "issues": ["no frames array"]}

    if data.get("signId") != sign_id:
        issues.append(f"signId mismatch: file says {data.get('signId')!r}, directory is {sign_id!r}")

    has_two_hand = any(isinstance(fr.get("hands"), dict) for fr in frames)
    left = sum(1 for fr in frames if (fr.get("hands") or {}).get("left"))
    right = sum(1 for fr in frames if (fr.get("hands") or {}).get("right"))
    both = sum(
        1 for fr in frames if (fr.get("hands") or {}).get("left") and (fr.get("hands") or {}).get("right")
    )
    any_hand = sum(1 for fr in frames if fr.get("handDetected"))

    if len(frames) == 0:
        issues.append("empty sequence (0 frames)")
    if any_hand == 0:
        issues.append("no hand detected in any frame")
    elif any_hand < min_frames:
        issues.append(f"only {any_hand} frame(s) with a hand (< {min_frames})")
    if not has_two_hand:
        issues.append("no `hands` field -- this file came from the OLD single-hand extractor")

    worst = {"left": 0.0, "right": 0.0}
    jump_counts = {"left": 0, "right": 0}
    for side in ("left", "right"):
        js = track_jumps(frames, side)
        if js:
            worst[side] = max(js)
            jump_counts[side] = sum(1 for j in js if j > IMPOSSIBLE_JUMP)
        if jump_counts[side]:
            issues.append(
                f"{side} track has {jump_counts[side]} impossible wrist jump(s), worst {worst[side]:.3f}"
                f" -- possible hand identity switch"
            )

    # Landmark sanity: 21 points, finite, in a plausible range.
    for fr in frames[:]:
        for side in ("left", "right"):
            h = (fr.get("hands") or {}).get(side)
            if not h:
                continue
            lms = h.get("landmarks") or []
            if len(lms) != 21:
                issues.append(f"{side} frame {fr.get('frameIndex')} has {len(lms)} landmarks, expected 21")
                break
            if any(not all(math.isfinite(p.get(k, 0.0)) for k in ("x", "y", "z")) for p in lms):
                issues.append(f"{side} frame {fr.get('frameIndex')} has non-finite landmark values")
                break

    # Fingerprint of the primary-hand sequence, to spot duplicate files.
    #
    # A take with no detected hand contributes nothing to the hash, so every
    # such take used to fingerprint as sha1("") and the whole set was then
    # reported as one enormous duplicate group. That is an artifact of the
    # hash, not a duplicate: the underlying videos are different recordings
    # that happen to share an empty landmark track. Those takes are already
    # reported by the no-hand check, so they get no fingerprint at all and
    # are excluded from the duplicate comparison.
    contributing = 0
    h = hashlib.sha1()
    for fr in frames:
        lms = fr.get("landmarks")
        if lms:
            h.update(("%.4f,%.4f" % (lms[0]["x"], lms[0]["y"])).encode())
            contributing += 1
    fingerprint = h.hexdigest() if contributing else None

    tracking = data.get("handTracking") or {}
    return {
        "corrupt": False,
        "issues": issues,
        "frames": len(frames),
        "anyHand": any_hand,
        "left": left,
        "right": right,
        "both": both,
        "twoHandData": has_two_hand,
        "worstJump": worst,
        "impossibleJumps": jump_counts,
        "slotSwapCount": tracking.get("slotSwapCount", 0),
        "primaryHand": tracking.get("primaryHand", "unknown"),
        "fingerprint": fingerprint,
        # Provenance, as written by the extractor. Absent on a tree produced
        # before provenance was recorded, and null on a file whose name did
        # not follow the dataset's convention -- both are reported, neither
        # is filled in with a guess.
        "sourceVideoStem": data.get("sourceVideoStem"),
        "sessionDate": (data.get("provenance") or {}).get("sessionDate"),
        "hasProvenance": bool(data.get("provenance")),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="processed-karsl/ root")
    ap.add_argument("--min-frames", type=int, default=DEFAULT_MIN_FRAMES)
    ap.add_argument("--json-out", default=None)
    ap.add_argument(
        "--exclusions-out",
        default=None,
        help="Write the machine-readable exclusion list the evaluation split consumes. "
        "This is the ONLY place these decisions are recorded -- downstream code reads "
        "the file rather than hard-coding sign IDs.",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.input):
        print(f"ERROR: --input is not a directory: {args.input}")
        return 1

    sign_dirs = sorted(
        d for d in os.listdir(args.input) if os.path.isdir(os.path.join(args.input, d))
    )

    results = []
    for sign_id in sign_dirs:
        d = os.path.join(args.input, sign_id)
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json"):
                continue
            r = audit_file(os.path.join(d, name), sign_id, args.min_frames)
            r["signId"] = sign_id
            r["file"] = name
            results.append(r)

    ok = [r for r in results if not r["corrupt"]]
    corrupt = [r for r in results if r["corrupt"]]
    with_issues = [r for r in ok if r["issues"]]

    two_hand_files = [r for r in ok if r["twoHandData"]]
    legacy_files = [r for r in ok if not r["twoHandData"]]
    both_majority = [r for r in two_hand_files if r["frames"] and r["both"] / max(r["frames"], 1) >= 0.5]
    one_hand_only = [r for r in two_hand_files if r["both"] == 0]
    intermittent = [
        r for r in two_hand_files if 0 < r["both"] < r["frames"] and r["both"] / max(r["frames"], 1) < 0.5
    ]

    seq_lengths = [r["anyHand"] for r in ok if r["anyHand"]]
    dupes = [
        k for k, v in Counter(r["fingerprint"] for r in ok if r["fingerprint"]).items() if v > 1
    ]

    all_jumps = [max(r["worstJump"].values()) for r in ok if r["worstJump"]]
    impossible_total = sum(sum(r["impossibleJumps"].values()) for r in ok)

    print("=" * 66)
    print("PROCESSED-KARSL QUALITY AUDIT")
    print("=" * 66)
    print(f"Sign directories:              {len(sign_dirs)}")
    print(f"Files audited:                 {len(results)}")
    print(f"  corrupt / unreadable:        {len(corrupt)}")
    print(f"  with two-hand data:          {len(two_hand_files)}")
    print(f"  OLD single-hand format:      {len(legacy_files)}")
    print()
    print(f"Two-handed (both >= 50% frames): {len(both_majority)}")
    print(f"One-handed (never both hands):   {len(one_hand_only)}")
    print(f"Intermittent second hand:        {len(intermittent)}")
    print()
    if seq_lengths:
        s = sorted(seq_lengths)
        print(f"Frames with a hand: min {s[0]}, mean {sum(s) / len(s):.1f}, median {s[len(s) // 2]}, max {s[-1]}")
    print(f"Empty / no-hand sequences:     {sum(1 for r in ok if r['anyHand'] == 0)}")
    print(f"Below --min-frames={args.min_frames}:          {sum(1 for r in ok if 0 < r['anyHand'] < args.min_frames)}")
    print(f"Duplicate sequences:           {len(dupes)}")
    print()
    print("HAND IDENTITY CONTINUITY")
    if all_jumps:
        a = sorted(all_jumps)
        print(f"  worst wrist jump per file:   min {a[0]:.4f}, median {a[len(a) // 2]:.4f}, max {a[-1]:.4f}")
    print(f"  impossible jumps (> {IMPOSSIBLE_JUMP}):   {impossible_total}")
    print(f"  slot swaps absorbed by tracker: {sum(r['slotSwapCount'] for r in ok)}")
    print()

    # ------------------------------------------------------------------
    # PROVENANCE AND SPLIT FEASIBILITY
    # ------------------------------------------------------------------
    # A leakage-safe evaluation on this dataset has to hold out a whole
    # RECORDING SESSION, not random takes: takes from one sitting are far
    # more alike than takes from another day, so a random split measures
    # something easier than deployment. That is only possible for a sign
    # with at least two sessions that each still have a USABLE take -- a
    # session whose every take lost its hands to detection failure is not a
    # session you can evaluate on. This block reports exactly which signs
    # can support such a split, so the split logic is written against the
    # real data rather than an assumption about it.
    with_prov = [r for r in ok if r["hasProvenance"]]
    with_stem = [r for r in ok if r["sourceVideoStem"]]

    takes_per_sign = Counter(r["signId"] for r in ok)
    sessions_of = {}
    usable_sessions_of = {}
    for r in ok:
        if not r["sessionDate"]:
            continue
        sessions_of.setdefault(r["signId"], Counter())[r["sessionDate"]] += 1
        if r["anyHand"] > 0:
            usable_sessions_of.setdefault(r["signId"], Counter())[r["sessionDate"]] += 1

    splittable = [s for s, c in usable_sessions_of.items() if len(c) >= 2]
    unsplittable = sorted(set(takes_per_sign) - set(splittable))
    smaller_usable = sorted(
        min(usable_sessions_of[s].values()) for s in splittable
    )

    tp = sorted(takes_per_sign.values())
    print("PROVENANCE")
    print(f"  files with sourceVideoStem:  {len(with_stem)} / {len(ok)}")
    print(f"  files with parsed provenance:{len(with_prov)} / {len(ok)}")
    print(f"  signs:                       {len(takes_per_sign)}")
    if tp:
        print(f"  takes per sign:              min {tp[0]}, median {tp[len(tp) // 2]}, max {tp[-1]}")
    dist = Counter(len(c) for c in sessions_of.values())
    print(f"  session dates per sign:      {dict(sorted(dist.items()))}")
    print()
    print("SESSION-HELD-OUT SPLIT FEASIBILITY")
    print(f"  signs with >=2 usable sessions: {len(splittable)}")
    print(f"  signs that CANNOT be split:     {len(unsplittable)}"
          + (f"  {unsplittable[:12]}" if unsplittable else ""))
    if smaller_usable:
        print(f"  smaller session's usable takes: min {smaller_usable[0]}, "
              f"median {smaller_usable[len(smaller_usable) // 2]}, max {smaller_usable[-1]}")
    print()

    if with_issues:
        print(f"FILES WITH ISSUES ({len(with_issues)}), first 20:")
        for r in with_issues[:20]:
            print(f"  {r['signId']}/{r['file']}")
            for i in r["issues"]:
                print(f"      - {i}")
    else:
        print("No per-file issues found.")

    blocking = len(corrupt) + impossible_total + len(legacy_files) + sum(
        1 for r in ok if r["anyHand"] == 0
    )
    print("-" * 66)
    print("BLOCKING ISSUES:" if blocking else "No blocking issues.")
    if blocking:
        print(f"  corrupt files:             {len(corrupt)}")
        print(f"  impossible wrist jumps:    {impossible_total}")
        print(f"  old single-hand files:     {len(legacy_files)}")
        print(f"  sequences with no hand:    {sum(1 for r in ok if r['anyHand'] == 0)}")
    print("=" * 66)

    if args.exclusions_out:
        # WHAT THIS FILE IS FOR, and what it deliberately is not.
        #
        # Two kinds of take cannot honestly take part in a leakage-safe
        # evaluation, and both are identified here by name so that no
        # downstream file has to hard-code a sign ID:
        #
        #   * a take whose wrist track jumps further in one frame than a hand
        #     can move -- its stored trajectory is very likely one hand's path
        #     spliced onto the other's, which is corrupt input to DTW however
        #     rare it is;
        #   * a take in which no hand was ever detected -- there is nothing to
        #     match against.
        #
        # And a sign with fewer than two USABLE sessions cannot be split by
        # session at all, so it is named too.
        #
        # This file EXCLUDES FROM MEASUREMENT ONLY. It is not a licence to
        # delete anything: the raw videos and the processed takes stay exactly
        # where they are, and the registry still carries every sign.
        excl = {
            "version": 1,
            "generatedAt": datetime.datetime.now().astimezone().isoformat(),
            "input": os.path.abspath(args.input),
            "impossibleJumpThreshold": IMPOSSIBLE_JUMP,
            "signsWithoutSessionSplit": unsplittable,
            "excludedTakes": {},
            "counts": {},
        }
        by_sign: dict[str, dict[str, str]] = {}
        for r in ok:
            reason = None
            if r["anyHand"] == 0:
                reason = "no-hand"
            elif sum(r["impossibleJumps"].values()) > 0:
                reason = "identity-switch"
            if reason:
                by_sign.setdefault(r["signId"], {})[os.path.splitext(r["file"])[0]] = reason
        excl["excludedTakes"] = {k: by_sign[k] for k in sorted(by_sign)}
        excl["counts"] = {
            "signsWithoutSessionSplit": len(unsplittable),
            "takesExcluded": sum(len(v) for v in by_sign.values()),
            "noHand": sum(1 for v in by_sign.values() for x in v.values() if x == "no-hand"),
            "identitySwitch": sum(
                1 for v in by_sign.values() for x in v.values() if x == "identity-switch"
            ),
            "signsAffected": len(by_sign),
        }
        with open(args.exclusions_out, "w", encoding="utf-8") as f:
            json.dump(excl, f, indent=1)
        c = excl["counts"]
        print(f"Wrote {args.exclusions_out}: {c['takesExcluded']} take(s) excluded "
              f"({c['noHand']} no-hand, {c['identitySwitch']} identity-switch) across "
              f"{c['signsAffected']} sign(s); {c['signsWithoutSessionSplit']} sign(s) "
              f"cannot be split by session.")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"Wrote {args.json_out}")

    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
