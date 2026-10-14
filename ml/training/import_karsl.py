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
    - If a sign has multiple videos, exactly ONE real take is imported --
      chosen by select_best_take() on content, never merged across takes.
      Which one is recorded in the entry's `sourceTake`.

CANONICAL SIDES: the two hand tracks in each processed file are re-named here
from the data itself (ml/training/handedness.assign_sides), NOT copied from
whatever the extractor called them. That makes this importer able to correct a
processed-karsl tree produced by an extractor whose naming convention was
wrong, without re-running MediaPipe over the whole dataset.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from console_safe import make_stdio_unicode_safe
from handedness import assign_sides

try:
    import pandas as pd
except ImportError:
    pd = None


#: Smallest share of a take's frames the second hand must appear in before it
#: counts as CAPTURED rather than a detection flicker.
#:
#: Measured on the real 502-sign registry, the per-sign both-hand fraction is
#: strongly bimodal: 477 signs sit at >= 0.5 (median 1.0), and the low tail is
#: 1-3 frames out of 24-68 (0.015-0.075). The next value above that tail is
#: 0.1395, so 0.10 falls inside a real gap in the data rather than being a
#: round number picked by hand.
MIN_SECOND_HAND_FRACTION = 0.10

DATASET_NAME = "KArSL-502"
DATASET_SOURCE = "https://hamzah-luqman.github.io/KArSL/"
DATASET_LICENSE = "Not explicitly stated by the authors beyond a citation request -- VERIFY before redistributing any derived data. See docs/SIGN_LANGUAGE_DATASETS.md."


@dataclass
class ImportStats:
    signs_found: int = 0
    signs_imported: int = 0
    signs_skipped: list[str] = field(default_factory=list)
    frames_total: int = 0
    two_handed_signs: int = 0


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


def _source_take(
    video_json_path: Optional[str],
    sign_id: str,
    take_count: int,
    reason: Optional[str],
) -> Optional[dict]:
    """Where this template's motion actually came from.

    The registry previously recorded only `sourceSignId`, so there was no way
    to tell WHICH of a sign's eight takes a template was built from -- and
    since select_best_take() picks by content rather than by position, that is
    not recoverable after the fact. Everything here is read off the processed
    file that was selected; nothing is guessed.
    """
    if not video_json_path:
        return None
    processed = os.path.basename(video_json_path)
    return {
        # The extractor writes "<video stem>.json"; the original recording is
        # that stem with the video extension. The stem is reported as-is
        # rather than an invented extension being appended.
        "processedFile": processed,
        "sourceVideoStem": os.path.splitext(processed)[0],
        "sourceFolder": sign_id,
        "takesAvailable": take_count,
        "selectionReason": reason or "unknown",
    }


def _canonical_primary(hand_frames: list[dict]) -> str:
    """Which side the legacy flat `landmarkSequence` should carry.

    Defined as THE SIDE WITH THE MOST REAL DETECTED FRAMES, read straight off
    the canonical two-hand timeline. That is deliberately not a translation of
    the extractor's own `primaryHand`, which was the previous approach and was
    wrong: the extractor reports its primary as a NAME indexed by SLOT ORDER,
    while `assign_sides` here returns sides indexed by STORED KEY order
    ("left", "right"). Those two orderings coincide only when the extractor
    happened to name slot 0 "left"; when it named slot 0 "right" the mapping
    inverted, and the legacy field was pointed at the wrong hand.

    Measured consequence on the real 502-sign registry: signs 0076 and 0089
    came out with `primaryHand: "left"` while the left hand appears in exactly
    ONE frame and the right in 50 and 51 respectively, so every legacy consumer
    saw a one-frame sequence for a fifty-frame sign.

    Deriving it from the frames cannot disagree with the frames.
    """
    left = sum(1 for f in hand_frames if f.get("left"))
    right = sum(1 for f in hand_frames if f.get("right"))
    if left == 0 and right == 0:
        return "unknown"
    return "left" if left > right else "right"


def load_processed_video_json(path: str):
    """Parses one extract_karsl_skeletons.py output file.

    Returns (legacy_frames, hand_frames, hand_stats) or None if unparseable.

      legacy_frames : list[list[point]]  -- the PRIMARY hand only, null frames
                                            dropped. Exactly what this importer
                                            produced before two-hand support, so
                                            every existing consumer is unaffected.
      hand_frames   : list[{"left": pts|None, "right": pts|None}] on a single
                      shared timeline (frames where AT LEAST ONE hand was seen),
                      or None when the input predates two-hand extraction.
      hand_stats    : dict describing what was actually found.

    NOTHING IS INVENTED. A hand that was not detected in a frame is None, never
    interpolated or copied from a neighbour. If a sign is genuinely one-handed,
    the other side stays None for the whole sequence and the avatar keeps that
    arm at rest.
    """
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None

    if not isinstance(data, dict) or "frames" not in data or not isinstance(data["frames"], list):
        return None

    def points(raw):
        return [{"x": float(p["x"]), "y": float(p["y"]), "z": float(p.get("z", 0.0))} for p in raw]

    legacy_frames: list[list[dict]] = []
    dropped = 0
    has_two_hand_data = False

    # The two stored keys are treated as opaque TRACKER SLOTS here, not as
    # left/right. HandSlotTracker guarantees each key holds one physical hand
    # for the whole video; what that hand IS gets decided below, from the data.
    # Re-deriving it here (rather than trusting the names the extractor wrote)
    # is what lets a processed-karsl tree produced by an older extractor be
    # corrected without re-running MediaPipe over 4,016 videos.
    slot_keys = ("left", "right")
    slot_frames: list[list[Optional[list[dict]]]] = [[], []]
    slot_labels: list[list[Optional[str]]] = [[], []]

    for frame in data["frames"]:
        if not isinstance(frame, dict):
            return None

        landmarks = frame.get("landmarks")
        if landmarks is None:
            dropped += 1
        else:
            try:
                legacy_frames.append(points(landmarks))
            except (KeyError, TypeError, ValueError):
                return None

        hands = frame.get("hands")
        if isinstance(hands, dict):
            has_two_hand_data = True
            for i, key in enumerate(slot_keys):
                h = hands.get(key)
                if isinstance(h, dict) and h.get("landmarks"):
                    try:
                        slot_frames[i].append(points(h["landmarks"]))
                    except (KeyError, TypeError, ValueError):
                        return None
                    slot_labels[i].append(h.get("rawLabel"))
                else:
                    slot_frames[i].append(None)
                    slot_labels[i].append(None)

    if dropped:
        print(f"    (dropped {dropped}/{len(data['frames'])} frame(s) with no detected hand)")

    assignment = assign_sides(slot_frames[0], slot_frames[1], slot_labels[0], slot_labels[1])

    hand_frames: list[dict] = []
    left_count = right_count = both_count = 0
    if has_two_hand_data:
        for i in range(len(slot_frames[0])):
            entry: dict = {"left": None, "right": None}
            for slot in (0, 1):
                side = assignment.slot_sides[slot]
                if side in ("left", "right") and slot_frames[slot][i] is not None:
                    entry[side] = slot_frames[slot][i]
            if entry["left"] is not None:
                left_count += 1
            if entry["right"] is not None:
                right_count += 1
            if entry["left"] is not None and entry["right"] is not None:
                both_count += 1
            if entry["left"] is not None or entry["right"] is not None:
                hand_frames.append(entry)

    if not legacy_frames and not hand_frames:
        return None

    tracking = data.get("handTracking") if isinstance(data.get("handTracking"), dict) else {}
    total = max(len(hand_frames), 1)
    hand_stats = {
        "twoHandData": has_two_hand_data,
        "framesWithLeft": left_count,
        "framesWithRight": right_count,
        "framesWithBoth": both_count,
        # A sign counts as two-handed when both hands are present together in a
        # clear majority of its frames. Signs where the second hand flickers in
        # for a couple of frames are treated as one-handed rather than animating
        # a barely-observed second arm.
        "twoHanded": both_count / total >= 0.5 if has_two_hand_data else False,
        # LEGACY CONVENIENCE ONLY. Says which side the legacy flat
        # `landmarkSequence` carries; it is NOT the identity of the sign and
        # nothing side-aware may branch on it. Translated through the canonical
        # assignment, so it names the same physical hand the extractor put in
        # the legacy field even when this import renamed the slots.
        "primaryHand": _canonical_primary(hand_frames),
        "slotSwapCount": tracking.get("slotSwapCount", 0),
        "sourceSlotNames": list(tracking.get("slotNames") or []),
        **assignment.as_dict(),
    }
    # slotSides describes the source file's slot order, which is an internal
    # detail of the extractor; the registry stores hands by name.
    hand_stats.pop("slotSides", None)

    return legacy_frames, (hand_frames if has_two_hand_data else None), hand_stats


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


def select_best_take(video_jsons: list[str]):
    """Chooses WHICH repetition of a sign to import.

    This used to be "the first file alphabetically", which is wrong in a way
    that quietly corrupts handedness for the whole dataset. Real example from
    sign 0001, whose eight takes extract as:

        01_02_0001_(15_11_16_16_41_12)   13 frames,  0 with both hands  <- first
        01_02_0001_(15_11_16_16_41_48)   14 frames, 14 with both hands
        01_02_0001_(15_11_16_16_41_56)   15 frames, 15 with both hands
        ... five more takes, all with both hands

    Alphabetical order picked the ONE take in which the second hand was never
    detected, so the sign would have been imported as one-handed and the avatar
    would rest an arm that should be moving. Two of the three signs in a
    20-video sample were affected this way.

    Selection is therefore by how completely a take captures the sign:
    most frames with BOTH hands first, then most frames with any hand. A
    genuinely one-handed sign has zero both-hand frames in every take, so it
    falls through to the second criterion and is unaffected. This picks among
    REAL recorded takes; it never merges or synthesises across them.

    Returns (path, frames, hand_frames, hand_stats, reason) or None.
    """
    scored = []
    for path in video_jsons:
        parsed = load_processed_video_json(path)
        if parsed is None:
            continue
        frames, hand_frames, hand_stats = parsed
        both = hand_stats.get("framesWithBoth", 0)
        any_hand = len(hand_frames) if hand_frames else len(frames)
        # A second hand that appears in one frame out of fifty is a detection
        # flicker, not a captured hand, and must not outrank a take that saw
        # more of the sign. Below the floor the both-hand count is treated as
        # zero and the take is ranked on how much of the sign it captured.
        effective_both = both if any_hand and both / any_hand >= MIN_SECOND_HAND_FRACTION else 0
        scored.append((effective_both, any_hand, path, frames, hand_frames, hand_stats, both))

    if not scored:
        return None

    scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
    effective_both, any_hand, path, frames, hand_frames, hand_stats, both = scored[0]

    if len(scored) == 1:
        reason = "only usable take"
    elif effective_both > 0:
        runner_up = scored[1][0]
        reason = f"most two-hand frames ({both}; next best {runner_up})"
    else:
        # Say so when a RIVAL take had both-hand frames that were discounted as
        # flicker -- otherwise "no take shows both hands" reads as a
        # contradiction of a rival's raw count and the choice looks arbitrary.
        discounted = max((t[6] for t in scored[1:]), default=0)
        if both > 0 or discounted > 0:
            reason = (
                f"second hand only flickers (best rival {max(both, discounted)} frame(s), below "
                f"{MIN_SECOND_HAND_FRACTION:.0%} of the take); ranked by frames with a hand ({any_hand})"
            )
        else:
            reason = f"no take shows both hands; most frames with a hand ({any_hand})"

    return path, frames, hand_frames, hand_stats, reason


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


def round_points(points, nd):
    return [{"x": round(p["x"], nd), "y": round(p["y"], nd), "z": round(p["z"], nd)} for p in points]


def round_entry(entry: dict, nd: int) -> dict:
    """Rounds every landmark coordinate. Lossy by design, and by an amount far
    below MediaPipe's own precision -- see --precision."""
    out = dict(entry)
    if out.get("landmarkSequence"):
        out["landmarkSequence"] = [round_points(fr, nd) for fr in out["landmarkSequence"]]
    if out.get("handFrames"):
        out["handFrames"] = [
            {
                "left": round_points(f["left"], nd) if f.get("left") else None,
                "right": round_points(f["right"], nd) if f.get("right") else None,
            }
            for f in out["handFrames"]
        ]
    return out


def main() -> int:
    # Real KArSL labels are Arabic and this script prints them as it goes. On a
    # console with a legacy Windows code page that raises UnicodeEncodeError,
    # which would kill the process before the output JSON is written -- the
    # registry would simply not appear, and the failure would surface as a
    # missing file rather than as an encoding problem. See console_safe.py.
    make_stdio_unicode_safe()

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Path to the KArSL dataset root directory")
    parser.add_argument("--labels", default=None, help="Path to the labels file (xlsx/csv mapping sign ID -> Arabic word)")
    parser.add_argument(
        "--output",
        default="public/models/sign-vocabulary/dedicated-signs.json",
        help="Where to write the resulting DedicatedSignEntry[] JSON",
    )
    parser.add_argument("--version", default="1.0.0", help="Version string to record in provenance metadata")
    parser.add_argument(
        "--precision",
        type=int,
        default=4,
        help=(
            "Decimal places to keep on landmark coordinates. MediaPipe coordinates are "
            "image-normalised, so 4 dp is 1e-4 of image width -- about 0.06 px on a 640-wide "
            "frame, far below landmark noise. This matters because the browser fetches the whole "
            "registry: at full float precision a 502-sign two-hand registry measures ~149 MB, "
            "and at 4 dp ~39 MB. Pass 0 to disable rounding."
        ),
    )
    parser.add_argument(
        "--keep-legacy-sequence",
        action="store_true",
        help=(
            "Also write the legacy per-entry landmarkSequence. It is an exact DUPLICATE of the "
            "primary hand already present in handFrames, so it is omitted by default and "
            "reconstructed on load by lib/sign-language/signRegistry.ts. Keeping it costs about "
            "14 MB across 502 signs and changes nothing functionally."
        ),
    )
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
        hand_frames = None
        hand_stats: dict = {}
        used_format = None
        video_json_path: Optional[str] = None
        take_count = len(video_jsons)
        take_reason: Optional[str] = None

        if video_jsons:
            best = select_best_take(video_jsons)
            if best is not None:
                video_json_path, frames, hand_frames, hand_stats, reason = best
                used_format = "processed-karsl"
                take_reason = reason
                if len(video_jsons) > 1:
                    print(
                        f"    Using {os.path.basename(video_json_path)} of {len(video_jsons)} take(s) -- {reason}"
                    )

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

        entry = {
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

        # TRACEABILITY. Which real recording this template came from, so a
        # recognition error can be traced back to a specific video rather than
        # only to a sign ID. Derived from the processed file that was actually
        # selected -- never invented, and omitted entirely when the fallback
        # per-frame-file path was used and there is no single source video.
        source_take = _source_take(video_json_path, sign_id, take_count, take_reason)
        if source_take:
            entry["sourceTake"] = source_take

        # Two-hand payload, only when the extractor actually produced it.
        # Entries imported from older single-hand output simply lack these
        # fields, and consumers treat them as one-handed -- no fabrication.
        if hand_frames:
            entry["handFrames"] = hand_frames
            entry["handedness"] = {
                "twoHanded": hand_stats.get("twoHanded", False),
                "framesWithLeft": hand_stats.get("framesWithLeft", 0),
                "framesWithRight": hand_stats.get("framesWithRight", 0),
                "framesWithBoth": hand_stats.get("framesWithBoth", 0),
                "primaryHand": hand_stats.get("primaryHand", "unknown"),
                # How left/right was decided for this take, and how strongly.
                # "geometry" is a measurement; "handedness-label" is a model
                # output; "unknown" means neither was usable and the sides
                # must not be trusted. See ml/training/handedness.py.
                "sideSource": hand_stats.get("sideSource", "unknown"),
                "sideConfidence": hand_stats.get("sideConfidence", 0.0),
                "geometryFrames": hand_stats.get("geometryFrames", 0),
            }
            if hand_stats.get("twoHanded"):
                stats.two_handed_signs += 1

        entries.append(entry)
        stats.signs_imported += 1
        stats.frames_total += len(frames)
        two = " [TWO-HANDED]" if hand_stats.get("twoHanded") else ""
        extra = f", {len(hand_frames)} two-hand frame(s)" if hand_frames else ""
        print(f"    Imported {sign_id} ({label}): {len(frames)} frame(s) via {used_format}{extra}{two}")

    if args.precision > 0:
        entries = [round_entry(e, args.precision) for e in entries]
    if not args.keep_legacy_sequence:
        for e in entries:
            if e.get("handFrames"):
                e.pop("landmarkSequence", None)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        # Compact separators, not indent=2: this file is fetched by the browser
        # on every cold load, and pretty-printing it costs tens of megabytes.
        json.dump(entries, f, ensure_ascii=False, separators=(",", ":"))

    print(f"\nSigns found (directories):  {stats.signs_found}")
    print(f"Signs imported:             {stats.signs_imported}")
    print(f"Signs skipped:              {len(stats.signs_skipped)}")
    if stats.signs_skipped:
        print("Skip reasons (first 20):")
        for reason in stats.signs_skipped[:20]:
            print(f"  - {reason}")
    print(f"Total frames imported:      {stats.frames_total}")
    print(f"Signs with two-hand data:   {stats.two_handed_signs}")
    size_mb = os.path.getsize(args.output) / 1e6
    print(f"\nWrote {args.output}  ({size_mb:.1f} MB)")
    if size_mb > 60:
        print(
            "WARNING: the browser fetches this entire file on first load. Consider a lower "
            "--precision, or splitting the registry, before shipping."
        )

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