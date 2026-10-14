"""
Audits the built registry (public/models/sign-vocabulary/dedicated-signs.json)
after import_karsl.py has run.

Answers, from the file itself rather than from what the importer claimed:
how many signs, which IDs, do the labels still match the spreadsheet, are
multi-word phrases intact, which entries carry real two-hand data, and is any
entry left over from the old single-hand extraction.

USAGE (from the project root):
    python ml/datasets/audit_registry.py \\
        --registry public/models/sign-vocabulary/dedicated-signs.json \\
        --labels KARSL-502_Labels.xlsx \\
        --expect-min 1 --expect-max 502

Exit code 0 = registry looks sound, 1 = problems found.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training"))
from console_safe import make_stdio_unicode_safe  # noqa: E402

try:
    import pandas as pd
except ImportError:
    pd = None


def canonical(raw: str) -> str:
    s = str(raw).strip()
    return str(int(s)) if s.isdigit() else s


def load_labels(path):
    if not path or not os.path.exists(path) or pd is None:
        return {}
    df = pd.read_excel(path, dtype=str) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path, dtype=str)
    id_col = next((c for c in df.columns if "id" in c.lower() or "sign" in c.lower()), None)
    label_col = next(
        (c for c in df.columns if "arabic" in c.lower() or "meaning" in c.lower() or "label" in c.lower()), None
    )
    if id_col is None or label_col is None:
        return {}
    return {canonical(r[id_col]): str(r[label_col]) for _, r in df.iterrows()}


def main() -> int:
    # This prints real Arabic labels (label mismatches, invalid entries). On a
    # legacy Windows code page that would raise UnicodeEncodeError and abort
    # the audit part-way through. See ml/training/console_safe.py.
    make_stdio_unicode_safe()

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default="public/models/sign-vocabulary/dedicated-signs.json")
    ap.add_argument("--labels", default="KARSL-502_Labels.xlsx")
    ap.add_argument("--expect-min", type=int, default=None)
    ap.add_argument("--expect-max", type=int, default=None)
    ap.add_argument("--min-frames", type=int, default=8)
    args = ap.parse_args()

    if not os.path.exists(args.registry):
        print(f"ERROR: registry not found: {args.registry}")
        return 1

    with open(args.registry, encoding="utf-8") as f:
        entries = json.load(f)

    labels = load_labels(args.labels)
    problems = []

    ids = [e.get("sourceSignId") for e in entries]
    dupe_ids = [i for i, c in Counter(ids).items() if c > 1]
    numeric = sorted(int(i) for i in ids if i and str(i).isdigit())

    two_hand = [e for e in entries if e.get("handFrames")]
    legacy_only = [e for e in entries if not e.get("handFrames")]
    genuinely_two_handed = [e for e in two_hand if (e.get("handedness") or {}).get("twoHanded")]
    one_handed = [
        e
        for e in two_hand
        if not (e.get("handedness") or {}).get("twoHanded")
        and ((e.get("handedness") or {}).get("framesWithBoth", 0) == 0)
    ]
    intermittent = [
        e
        for e in two_hand
        if not (e.get("handedness") or {}).get("twoHanded")
        and ((e.get("handedness") or {}).get("framesWithBoth", 0) > 0)
    ]
    multiword = [e for e in entries if " " in (e.get("text") or "").strip()]

    def effective_sequence(entry):
        """The frames a CONSUMER sees.

        `landmarkSequence` is deliberately omitted from the shipped registry --
        it duplicates the primary hand already in `handFrames` and is rebuilt on
        load by lib/sign-language/signRegistry.ts. Checking the raw field alone
        reported every entry as empty, which is a bug in the audit, not the
        data.
        """
        seq = entry.get("landmarkSequence") or []
        if seq:
            return seq
        hand_frames = entry.get("handFrames") or []
        if not hand_frames:
            return []
        primary = (entry.get("handedness") or {}).get("primaryHand")
        primary = primary if primary in ("left", "right") else "right"
        frames = [f[primary] for f in hand_frames if f.get(primary)]
        if not frames:
            other = "left" if primary == "right" else "right"
            frames = [f[other] for f in hand_frames if f.get(other)]
        return frames

    invalid = []
    for e in entries:
        sid = e.get("sourceSignId")
        seq = effective_sequence(e)
        if len(seq) < args.min_frames:
            invalid.append(f"{sid}: only {len(seq)} legacy frame(s)")
            continue
        if any(len(fr) != 21 for fr in seq[:5]):
            invalid.append(f"{sid}: landmark frame is not 21 points")
        hf = e.get("handFrames")
        if hf is not None:
            if not hf:
                invalid.append(f"{sid}: handFrames present but empty")
            elif all(f.get("left") is None and f.get("right") is None for f in hf):
                invalid.append(f"{sid}: handFrames contains no hands at all")
        if not (e.get("text") or "").strip():
            invalid.append(f"{sid}: empty label")

    label_mismatches = []
    if labels:
        for e in entries:
            sid = e.get("sourceSignId")
            expected = labels.get(canonical(sid or ""))
            if expected is not None and expected != e.get("text"):
                label_mismatches.append(f"{sid}: registry {e.get('text')!r} vs labels {expected!r}")

    missing = []
    if args.expect_min is not None and args.expect_max is not None:
        want = set(range(args.expect_min, args.expect_max + 1))
        missing = sorted(want - set(numeric))

    print("=" * 66)
    print("REGISTRY AUDIT")
    print("=" * 66)
    print(f"File:                          {args.registry}")
    print(f"Total entries:                 {len(entries)}")
    if numeric:
        print(f"ID range:                      {min(numeric):04d} .. {max(numeric):04d}")
    print(f"Duplicate IDs:                 {len(dupe_ids)} {dupe_ids[:10]}")
    if args.expect_min is not None:
        print(f"Missing IDs in {args.expect_min}-{args.expect_max}:      {len(missing)} {missing[:20]}")
    print()
    print(f"Entries WITH two-hand data:    {len(two_hand)}")
    print(f"  genuinely two-handed:        {len(genuinely_two_handed)}")
    print(f"  one-handed (never both):     {len(one_handed)}")
    print(f"  intermittent second hand:    {len(intermittent)}")
    print(f"Entries WITHOUT (old format):  {len(legacy_only)}")
    print(f"Multi-word labels:             {len(multiword)}")
    print(f"Entries with left-hand data:   {sum(1 for e in entries if (e.get('handedness') or {}).get('framesWithLeft', 0) > 0)}")
    print(f"Entries with right-hand data:  {sum(1 for e in entries if (e.get('handedness') or {}).get('framesWithRight', 0) > 0)}")
    print(f"Entries with BOTH hands:       {sum(1 for e in entries if (e.get('handedness') or {}).get('framesWithBoth', 0) > 0)}")
    print(f"Entries with sourceTake:       {sum(1 for e in entries if e.get('sourceTake'))}")
    side_sources = Counter((e.get("handedness") or {}).get("sideSource", "absent") for e in entries)
    print(f"Side naming source:            {dict(side_sources)}")
    unknown_sides = [e for e in entries if (e.get("handedness") or {}).get("sideSource") == "unknown"]
    print(f"Entries with UNKNOWN sides:    {len(unknown_sides)}")
    print()
    print(f"Label mismatches vs xlsx:      {len(label_mismatches)}")
    for m in label_mismatches[:10]:
        print(f"    {m}")
    print(f"Invalid entries:               {len(invalid)}")
    for m in invalid[:10]:
        print(f"    {m}")

    if legacy_only:
        problems.append(f"{len(legacy_only)} entry/ies have no two-hand data (old single-hand extraction)")
    if dupe_ids:
        problems.append(f"{len(dupe_ids)} duplicate sign ID(s)")
    if missing:
        problems.append(f"{len(missing)} expected sign ID(s) missing")
    if invalid:
        problems.append(f"{len(invalid)} invalid entry/ies")
    if label_mismatches:
        problems.append(f"{len(label_mismatches)} label mismatch(es)")
    if unknown_sides:
        problems.append(
            f"{len(unknown_sides)} entry/ies whose left/right could not be determined "
            f"(they are matched as single unnamed hands, not guessed)"
        )
    # GEOMETRIC SELF-CHECK: the signer's right hand should sit at the smaller
    # image x. Verifying it here means the registry is checked against a
    # measurement rather than against whatever named it.
    agree = disagree = 0
    for e in entries:
        for f in e.get("handFrames") or []:
            left, right = f.get("left"), f.get("right")
            if not left or not right:
                continue
            if right[0]["x"] < left[0]["x"]:
                agree += 1
            else:
                disagree += 1
    if agree + disagree:
        rate = agree / (agree + disagree)
        print(f"Geometric side agreement:      {agree}/{agree + disagree} = {rate:.4f}")
        if rate < 0.8:
            problems.append(
                f"side naming disagrees with geometry on {1 - rate:.1%} of two-hand frames -- "
                f"the two hands are probably swapped registry-wide"
            )

    print("-" * 66)
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("Registry is internally consistent and fully two-hand.")
    print("=" * 66)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
