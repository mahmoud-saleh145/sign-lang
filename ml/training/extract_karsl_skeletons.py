"""
Extracts frame-by-frame hand landmark sequences from real KArSL video files.

    KArSL/0001/video1.mp4  ->  processed-karsl/0001/video1.json

This script's ONLY job is VIDEO -> LANDMARKS. It does not recognize signs,
does not assign Arabic labels, and does not touch
public/models/sign-vocabulary/dedicated-signs.json directly -- that's
ml/training/import_karsl.py's job, which has been updated to consume this
script's real output format (see the "PROCESSED-KARSL FORMAT" section in
that file's docstring).

USAGE (run locally, e.g. on Windows, from the project root):
    python ml/training/extract_karsl_skeletons.py --input KArSL --output processed-karsl

Expected input layout:
    KArSL/
      0001/
        video1.mp4
        video2.mp4
      0002/
        video1.mp4
      ...

Produces:
    processed-karsl/
      0001/
        video1.json
        video2.json
      0002/
        video1.json
      manifest.json

Each per-video JSON:
    {
      "signId": "0001",              <- exact folder name, ALWAYS a string,
                                          zero-padding preserved (never int())
      "sourceVideo": "video1.mp4",
      "fps": 30.0,
      "frameCount": 120,             <- frames actually read
      "handTracking": {              <- NEW: how the two hands were tracked
        "numHands": 2,
        "slotNames": ["right", "left"],
        "sideSource": "geometry",     <- how left/right was decided
        "sideConfidence": 1.0,
        "slotDetectionCounts": [65, 41],
        "primaryHand": "right",       <- LEGACY convenience, not an identity
        "framesWithTwoHands": 41,
        "framesWithAnyHand": 65,
        "slotSwapCount": 1
      },
      "frames": [
        {
          "frameIndex": 0,
          "timestampMs": 0.0,
          "handDetected": true,          <- LEGACY: any hand seen this frame
          "landmarks": [...21 points] | null,   <- LEGACY: the PRIMARY hand
          "hands": {                     <- NEW: both hands, tracked
            "left":  {"landmarks": [...21], "score": 0.97, "rawLabel": "Right"} | null,
            "right": {"landmarks": [...21], "score": 0.99, "rawLabel": "Left"}  | null
          }
        },
        ...
      ]
    }

The legacy "landmarks" field is still written so existing consumers keep
working unchanged. It now carries the PRIMARY hand consistently, which also
fixes a real defect: with num_hands=1 the single returned hand was not pinned
to one physical hand, so six of fourteen sampled videos produced sequences
that silently swapped hands part-way through.

MISSING LANDMARKS: if MediaPipe doesn't detect a hand in a frame,
"handDetected" is false and "landmarks" is null for that frame. Nothing is
invented -- the frame is preserved in the sequence (so frame timing/count
stays accurate) but carries no fabricated coordinates. Downstream consumers
(see import_karsl.py) drop null-landmark frames when building an animation
sequence and report exactly how many were dropped.

DEPENDENCIES (only needed to run this script, not the rest of the app):
    pip install opencv-python mediapipe --break-system-packages
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import handedness as handedness_module
from handedness import assign_sides

SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

# ============================================================================
# PROVENANCE RECOVERED FROM THE FILENAME
# ============================================================================
# Real KArSL files are named
#
#     03_01_0071_(01_12_16_15_52_41)_c.mp4
#     ^^ ^^ ^^^^  ^^^^^^^^ ^^^^^^^^  ^
#     |  |  |     date     time      camera/stream suffix
#     |  |  signId
#     |  second numeric field
#     first numeric field
#
# The two leading numeric fields are recorded VERBATIM as field1/field2. This
# pipeline does NOT label either of them "signer": which one that is, is not
# documented anywhere in this repository, and writing a guess into every
# processed file would be inventing metadata. What is unambiguous, and what
# the evaluation split can safely key on, is `sessionDate` -- takes sharing a
# date were recorded in one sitting, and holding a whole session out is a far
# stronger test than holding out random takes.
#
# A filename that does not match leaves `provenance` null. Nothing is
# force-fitted and nothing is fabricated.
KARSL_NAME = re.compile(
    r"^(?P<field1>\d+)_(?P<field2>\d+)_(?P<sign>\d+)_"
    r"\((?P<date>\d{2}_\d{2}_\d{2})_(?P<time>\d{2}_\d{2}_\d{2})\)"
    r"_(?P<suffix>[A-Za-z0-9]+)$"
)


def parse_provenance(stem: str) -> Optional[dict]:
    """Structured provenance from a KArSL filename stem, or None if the name
    does not follow the dataset's convention."""
    match = KARSL_NAME.match(stem)
    if not match:
        return None
    return {
        "field1": match.group("field1"),
        "field2": match.group("field2"),
        "signIdInName": match.group("sign"),
        "sessionDate": match.group("date"),
        "recordedAt": f"{match.group('date')}_{match.group('time')}",
        "suffix": match.group("suffix"),
    }

LandmarkExtractor = Callable[["object"], Optional[list[dict]]]  # frame (np.ndarray) -> 21 points | None

# Same model family/asset the browser side already uses (lib/mediapipe/handLandmarker.ts),
# for consistency between live camera recognition and this offline extraction pipeline.
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL_CACHE_PATH = os.path.join(
    os.path.dirname(__file__), ".cache", "hand_landmarker.task"
)


def resolve_model_path(model_path: Optional[str] = None) -> str:
    """Returns a local path to the HandLandmarker .task model file,
    downloading it to a local cache on first use if not already present.
    Raises a clear RuntimeError (not a cryptic one) if it can't be obtained
    -- this requires real internet access on whatever machine runs this
    script; it will NOT work in a network-restricted sandbox.
    """
    if model_path:
        if not os.path.exists(model_path):
            raise RuntimeError(f"--model-path was given but does not exist: {model_path}")
        return model_path

    if os.path.exists(DEFAULT_MODEL_CACHE_PATH):
        return DEFAULT_MODEL_CACHE_PATH

    os.makedirs(os.path.dirname(DEFAULT_MODEL_CACHE_PATH), exist_ok=True)
    print(f"Downloading hand landmark model (one-time, ~10MB) from:\n  {DEFAULT_MODEL_URL}")
    try:
        import urllib.request

        urllib.request.urlretrieve(DEFAULT_MODEL_URL, DEFAULT_MODEL_CACHE_PATH)
    except Exception as e:  # noqa: BLE001 -- surfaced as a clear, actionable error
        raise RuntimeError(
            f"Could not download the hand landmark model automatically ({e}). "
            f"If you're behind a restricted network, download it manually from "
            f"{DEFAULT_MODEL_URL} and pass --model-path <file>."
        ) from e

    print(f"Saved model to {DEFAULT_MODEL_CACHE_PATH}")
    return DEFAULT_MODEL_CACHE_PATH


# ============================================================================
# TWO-HAND EXTRACTION
# ============================================================================
# WHY THIS CHANGED FROM num_hands=1
#
# Measured on 14 real KArSL videos sampled across the dataset, running
# MediaPipe with num_hands=2: ELEVEN of the fourteen show two hands in >=30%
# of frames, several in 93-100% of frames. Arabic Sign Language is largely
# two-handed and the single-hand extractor was discarding half of most signs.
#
# Worse, num_hands=1 does NOT pin the returned hand to one physical hand.
# Tracking which hand result.hand_landmarks[0] actually was, frame by frame,
# on those same videos:
#
#   0171  LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL RRRRRRRRRRRRRRRRRRRRRRRRRR
#   0241  RRRRRRRRLRRRRRRRRLLRRLLLLLLLLLLLLLLL
#   0255  LLLLLLLLLLLLLLLLLLLLLLLLL RRRRRRRRRRRRRR
#
# Six of the fourteen swapped hands mid-video, so the stored "temporal
# sequence" for those signs was a splice of one hand's trajectory followed by
# the other's -- not a valid trajectory for DTW matching or for animation.
#
# HOW SLOTS ARE ASSIGNED (this is what fixes the splice):
# Hands are tracked as two persistent SLOTS across the video, assigned by
# nearest-wrist continuity rather than by trusting MediaPipe's per-frame
# handedness label:
#   * 2 hands detected -> whichever of the two possible slot assignments
#     minimises total wrist displacement from the previous frame wins.
#   * 1 hand detected  -> goes to the slot whose last known wrist is closer,
#     provided it is within GATE; otherwise it seeds/reseeds a slot.
#   * 0 hands detected -> both slots are null for that frame.
# Slots are NAMED once per video, AFTER the whole video has been read, by
# ml/training/handedness.assign_sides() -- geometry first (a signer facing an
# un-mirrored camera has their right hand at the smaller image x), MediaPipe's
# per-track label majority as the fallback, and "unknown" when neither signal
# is usable. See that module for the measurements behind the rule.
#
# This REPLACES two things that were not hand identities at all: naming the
# slots from the handedness label on a single seed frame (measured to
# disagree with the track majority in 8.1% of tracks), and treating "the slot
# seen in more frames" as the primary/canonical hand (a coin flip whenever the
# two counts are equal, which for a symmetric two-handed sign they routinely
# are). The raw per-frame label and score are still preserved so any of this
# can be re-derived without re-extracting.

# Max normalised wrist movement between consecutive frames for a single
# detection to be considered the same hand continuing. 0.25 of image width is
# far larger than real inter-frame hand motion (measured p90 per-frame
# landmark displacement is ~0.009) but small enough to reject a jump to the
# other hand.
SLOT_CONTINUITY_GATE = 0.25


def _wrist(points: list[dict]) -> tuple[float, float]:
    return points[0]["x"], points[0]["y"]


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


class HandSlotTracker:
    """Assigns each frame's detections to two persistent slots, 0 and 1.

    Returns (slot0, slot1) where each is a dict {"landmarks", "score",
    "rawLabel"} or None. Slot NAMES (left/right) are decided separately, after
    the whole video is read, by handedness.assign_sides().
    """

    def __init__(self, gate: float = SLOT_CONTINUITY_GATE):
        self.gate = gate
        self.last: list[Optional[tuple[float, float]]] = [None, None]
        self.seed_labels: list[Optional[str]] = [None, None]
        self.switch_count = 0

    def step(self, detections: list[dict]):
        slots: list[Optional[dict]] = [None, None]

        if len(detections) >= 2:
            a, b = detections[0], detections[1]
            wa, wb = _wrist(a["landmarks"]), _wrist(b["landmarks"])
            if self.last[0] is not None and self.last[1] is not None:
                keep = _dist(wa, self.last[0]) + _dist(wb, self.last[1])
                swap = _dist(wb, self.last[0]) + _dist(wa, self.last[1])
                if swap < keep:
                    a, b = b, a
                    wa, wb = wb, wa
                    self.switch_count += 1
            slots[0], slots[1] = a, b
            self.last[0], self.last[1] = wa, wb
            for i, d in enumerate((a, b)):
                if self.seed_labels[i] is None:
                    self.seed_labels[i] = d["rawLabel"]

        elif len(detections) == 1:
            d = detections[0]
            w = _wrist(d["landmarks"])
            candidates = [
                (i, _dist(w, self.last[i])) for i in (0, 1) if self.last[i] is not None
            ]
            target = None
            if candidates:
                i, dist = min(candidates, key=lambda t: t[1])
                if dist <= self.gate:
                    target = i
            if target is None:
                # No usable history: fall back to the label, else the free slot.
                for i in (0, 1):
                    if self.seed_labels[i] == d["rawLabel"]:
                        target = i
                        break
                if target is None:
                    target = 0 if self.last[0] is None else 1
            slots[target] = d
            self.last[target] = w
            if self.seed_labels[target] is None:
                self.seed_labels[target] = d["rawLabel"]

        return slots


def build_real_mediapipe_extractor(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_path: Optional[str] = None,
    num_hands: int = 2,
):
    """Returns a real MediaPipe Tasks-API-backed extractor callable, reused
    across all frames of one video (HandLandmarker tracks between frames
    when reused this way in VIDEO running mode, rather than re-detecting
    cold every frame). Caller MUST call the returned extractor's `.close()`
    when done with a video.

    The extractor returns a LIST of detections for the frame, each
    {"landmarks": [...21 pts], "score": float, "rawLabel": "Left"|"Right"},
    ordered as MediaPipe returned them. Slot assignment happens downstream in
    HandSlotTracker -- deliberately NOT here, because it needs cross-frame
    state that belongs to the video, not to the model.

    Uses mp.tasks.vision.HandLandmarker -- the same MediaPipe Tasks family
    already used in the browser (lib/mediapipe/handLandmarker.ts).
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    resolved_model_path = resolve_model_path(model_path)

    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=resolved_model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)
    frame_counter = {"n": 0}

    def extractor(frame_bgr):
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # detect_for_video requires monotonically increasing integer ms timestamps.
        timestamp_ms = frame_counter["n"]
        frame_counter["n"] += 1
        result = landmarker.detect_for_video(mp_image, timestamp_ms)
        out = []
        for i, hand in enumerate(result.hand_landmarks):
            info = result.handedness[i][0] if i < len(result.handedness) else None
            out.append(
                {
                    "landmarks": [
                        {"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand
                    ],
                    "score": float(info.score) if info is not None else 0.0,
                    "rawLabel": info.category_name if info is not None else "Right",
                }
            )
        return out

    extractor.close = landmarker.close  # type: ignore[attr-defined]
    return extractor


@dataclass
class VideoResult:
    sign_id: str
    source_video: str
    output_path: str
    frame_count: int
    frames_with_hand: int
    status: str  # "success" | "failed"
    error: Optional[str] = None


def process_video(
    video_path: str,
    sign_id: str,
    extractor: LandmarkExtractor,
    max_frames: Optional[int] = None,
    progress_every: int = 30,
) -> dict:
    """Reads one video frame-by-frame (streaming -- never loads the whole
    video into memory) and returns the per-video JSON-ready dict described in
    the module docstring. Raises on an unreadable video; caller catches and
    records the failure so one bad video doesn't abort the whole run.

    Both hands are tracked as persistent slots (see HandSlotTracker) so a
    hand's trajectory is never spliced with the other hand's.

    BACKWARD COMPATIBILITY: the legacy per-frame "landmarks" field is still
    written, carrying the PRIMARY hand (the slot detected in more frames).
    Existing consumers keep working unchanged -- and, unlike before, that
    legacy field is now temporally coherent instead of silently swapping
    hands mid-sign.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    tracker = HandSlotTracker()
    raw_frames: list[dict] = []
    frame_index = 0
    slot_hits = [0, 0]
    two_hand_frames = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            detections = extractor(frame)
            slots = tracker.step(detections)
            for i in (0, 1):
                if slots[i] is not None:
                    slot_hits[i] += 1
            if slots[0] is not None and slots[1] is not None:
                two_hand_frames += 1

            timestamp_ms = (frame_index / fps * 1000) if fps > 0 else float(frame_index)
            raw_frames.append(
                {
                    "frameIndex": frame_index,
                    "timestampMs": round(timestamp_ms, 2),
                    "slots": slots,
                }
            )
            frame_index += 1

            if progress_every and frame_index % progress_every == 0:
                print(f"    ...{frame_index} frames read", flush=True)
    finally:
        cap.release()

    assignment = assign_sides(
        [rf["slots"][0]["landmarks"] if rf["slots"][0] else None for rf in raw_frames],
        [rf["slots"][1]["landmarks"] if rf["slots"][1] else None for rf in raw_frames],
        [rf["slots"][0]["rawLabel"] if rf["slots"][0] else None for rf in raw_frames],
        [rf["slots"][1]["rawLabel"] if rf["slots"][1] else None for rf in raw_frames],
    )
    names = list(assignment.slot_sides)

    # PRIMARY HAND IS A LEGACY CONVENIENCE, NOT AN IDENTITY. The legacy
    # per-frame "landmarks" field has to carry exactly one hand, so one is
    # chosen -- but nothing downstream may treat it as "the" hand of the sign.
    # Side-aware matching uses `hands`, which is named canonically above.
    primary_slot = 0 if slot_hits[0] >= slot_hits[1] else 1

    frames: list[dict] = []
    for rf in raw_frames:
        slots = rf["slots"]
        hands = {"left": None, "right": None}
        for i in (0, 1):
            if slots[i] is not None and names[i] in ("left", "right"):
                hands[names[i]] = slots[i]
        primary = slots[primary_slot]
        frames.append(
            {
                "frameIndex": rf["frameIndex"],
                "timestampMs": rf["timestampMs"],
                # Legacy fields, preserved so existing consumers keep working.
                "handDetected": primary is not None,
                "landmarks": primary["landmarks"] if primary else None,
                # New: both hands, temporally tracked.
                "hands": hands,
            }
        )

    stem = os.path.splitext(os.path.basename(video_path))[0]
    return {
        "signId": sign_id,
        "sourceVideo": os.path.basename(video_path),
        # The stem is the join key between a raw video, its processed JSON and
        # the registry's sourceTake, so it is stored rather than re-derived by
        # each consumer stripping an extension its own way.
        "sourceVideoStem": stem,
        "provenance": parse_provenance(stem),
        "fps": fps,
        "frameCount": len(frames),
        "handTracking": {
            "numHands": 2,
            "slotNames": names,
            # How the slots got their left/right names, and how sure that is.
            # Carried forward so a consumer can tell a geometrically-decided
            # side from a label-guessed one, and can refuse an "unknown".
            **assignment.as_dict(),
            "slotDetectionCounts": slot_hits,
            "primaryHand": names[primary_slot] if names[primary_slot] else "unknown",
            "framesWithTwoHands": two_hand_frames,
            "framesWithAnyHand": sum(1 for f in frames if f["handDetected"]),
            # How often the nearest-wrist assignment had to swap the two
            # detections relative to MediaPipe's ordering. High values mean
            # MediaPipe's ordering was unstable -- which is exactly the
            # failure the tracker exists to absorb.
            "slotSwapCount": tracker.switch_count,
        },
        "frames": frames,
    }


def find_sign_directories(input_dir: str) -> list[str]:
    """Returns sign ID folder names EXACTLY as they appear on disk -- always
    strings, zero-padding preserved (e.g. "0001" stays "0001", never becomes 1)."""
    return sorted(
        d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))
    )


def find_videos(sign_dir: str) -> list[str]:
    videos = []
    for root, _dirs, files in os.walk(sign_dir):
        for fname in files:
            if fname.lower().endswith(SUPPORTED_EXTENSIONS):
                videos.append(os.path.join(root, fname))
    return sorted(videos)


def run_extraction(
    input_dir: str,
    output_dir: str,
    extractor_factory: Callable[[], LandmarkExtractor],
    max_frames: Optional[int] = None,
    limit: Optional[int] = None,
    resume: bool = False,
    progress_every: int = 0,
) -> list[VideoResult]:
    """Core orchestration, factored out from main() so tests can call it
    directly with a fake extractor_factory instead of requiring real
    MediaPipe + real video files."""
    sign_dirs = find_sign_directories(input_dir)
    if not sign_dirs:
        print(f"ERROR: no subdirectories found under {input_dir}. Expected one directory per sign ID.")
        return []

    all_videos: list[tuple[str, str]] = []  # (sign_id, video_path)
    for sign_id in sign_dirs:
        for video_path in find_videos(os.path.join(input_dir, sign_id)):
            all_videos.append((sign_id, video_path))

    if limit is not None:
        all_videos = all_videos[:limit]

    print(f"Found {len(sign_dirs)} sign folders, {len(all_videos)} video file(s) to process.\n")

    results: list[VideoResult] = []
    skipped = 0
    start = time.time()

    for i, (sign_id, video_path) in enumerate(all_videos, 1):
        rel = os.path.relpath(video_path, input_dir)

        out_sign_dir = os.path.join(output_dir, sign_id)
        os.makedirs(out_sign_dir, exist_ok=True)
        out_name = os.path.splitext(os.path.basename(video_path))[0] + ".json"
        out_path = os.path.join(out_sign_dir, out_name)

        # RESUME. A full KArSL pass is tens of thousands of videos and hours of
        # wall clock; without this, one interruption throws all of it away. A
        # previous output is only trusted if it parses and carries frames --
        # a truncated file written when the process died is redone, not kept.
        if resume and os.path.exists(out_path):
            reusable = False
            try:
                with open(out_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                reusable = isinstance(existing.get("frames"), list) and len(existing["frames"]) > 0
            except (OSError, ValueError, AttributeError):
                reusable = False
            if reusable:
                skipped += 1
                results.append(
                    VideoResult(
                        sign_id=sign_id,
                        source_video=os.path.basename(video_path),
                        output_path=out_path,
                        frame_count=len(existing["frames"]),
                        frames_with_hand=sum(1 for fr in existing["frames"] if fr.get("handDetected")),
                        status="skipped",
                    )
                )
                if skipped % 250 == 0:
                    print(f"[{i}/{len(all_videos)}] ...{skipped} already extracted, skipping", flush=True)
                continue

        print(f"[{i}/{len(all_videos)}] {rel}")
        extractor = extractor_factory()
        try:
            data = process_video(
                video_path,
                sign_id,
                extractor,
                max_frames=max_frames,
                progress_every=progress_every,
            )
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            frames_with_hand = sum(1 for fr in data["frames"] if fr["handDetected"])
            ht = data.get("handTracking", {})
            print(
                f"    OK -- {data['frameCount']} frames, "
                f"{frames_with_hand}/{data['frameCount']} with a detected hand, "
                f"{ht.get('framesWithTwoHands', 0)} with BOTH hands"
                + (f", {ht['slotSwapCount']} slot swap(s) absorbed" if ht.get("slotSwapCount") else "")
            )
            results.append(
                VideoResult(
                    sign_id=sign_id,
                    source_video=os.path.basename(video_path),
                    output_path=out_path,
                    frame_count=data["frameCount"],
                    frames_with_hand=frames_with_hand,
                    status="success",
                )
            )
        except Exception as e:  # noqa: BLE001 -- one bad video must not abort the run
            print(f"    FAILED -- {e}")
            results.append(
                VideoResult(
                    sign_id=sign_id,
                    source_video=os.path.basename(video_path),
                    output_path=out_path,
                    frame_count=0,
                    frames_with_hand=0,
                    status="failed",
                    error=str(e),
                )
            )
        finally:
            close = getattr(extractor, "close", None)
            if close:
                close()

    elapsed = time.time() - start
    succeeded = [r for r in results if r.status == "success"]
    reused = [r for r in results if r.status == "skipped"]
    failed = [r for r in results if r.status == "failed"]

    manifest = {
        "inputDir": input_dir,
        "videosFound": len(all_videos),
        "videosProcessed": len(succeeded),
        "videosReused": len(reused),
        "videosFailed": len(failed),
        "elapsedSeconds": round(elapsed, 1),
        "videos": [
            {
                "signId": r.sign_id,
                "sourceVideo": r.source_video,
                "outputPath": os.path.relpath(r.output_path, output_dir),
                "frameCount": r.frame_count,
                "framesWithHand": r.frames_with_hand,
                "status": r.status,
                "error": r.error,
            }
            for r in results
        ],
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nDone in {elapsed:.1f}s.")
    print(f"  Videos found:     {len(all_videos)}")
    print(f"  Processed OK:     {len(succeeded)}")
    print(f"  Reused (resume):  {len(reused)}")
    print(f"  Failed:           {len(failed)}")
    if failed:
        print("  Failures:")
        for r in failed:
            print(f"    - {r.sign_id}/{r.source_video}: {r.error}")
    print(f"\nWrote {os.path.join(output_dir, 'manifest.json')}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="KArSL root directory (contains one subfolder per sign ID)")
    parser.add_argument("--output", default="processed-karsl", help="Where to write processed-karsl/")
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--model-path",
        default=None,
        help="Path to a local hand_landmarker.task file. If omitted, it's downloaded "
        "automatically (one-time, ~10MB) to ml/training/.cache/hand_landmarker.task",
    )
    parser.add_argument(
        "--mirrored-source",
        action="store_true",
        help=(
            "Set only when the SOURCE VIDEO is mirrored (a selfie-style recording). "
            "MediaPipe's handedness label is documented as assuming a mirrored image, "
            "but measured against geometry on real KArSL video the label maps to the "
            "signer's anatomy DIRECTLY (95.9%% agreement over 9,852 two-hand frames), "
            "so no inversion is applied by default. See ml/training/handedness.py. "
            "Side naming is geometry-first regardless; this flag only affects the "
            "label fallback used when a take never shows both hands at once."
        ),
    )
    parser.add_argument(
        "--num-hands",
        type=int,
        default=2,
        help="Hands MediaPipe may detect per frame. Default 2; measurements showed 11 of 14 sampled real KArSL videos are two-handed.",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames read per video (for quick tests)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on total videos processed (for quick tests)")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip videos whose processed JSON already exists and parses with frames. "
        "A full KArSL pass is hours long; without this an interruption restarts it from zero.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print a line every N frames within a video. Default 0 (silent) -- at dataset "
        "scale the per-frame chatter is megabytes of console output.",
    )
    args = parser.parse_args()

    handedness_module.INVERT_LABEL = args.mirrored_source

    if not os.path.isdir(args.input):
        print(f"ERROR: --input path does not exist or is not a directory: {args.input}")
        return 1

    try:
        import cv2  # noqa: F401
        import mediapipe  # noqa: F401
    except ImportError as e:
        print(f"ERROR: missing dependency ({e}).")
        print("Install with: pip install opencv-python mediapipe --break-system-packages")
        return 1

    def factory():
        return build_real_mediapipe_extractor(
            args.min_detection_confidence,
            args.min_tracking_confidence,
            args.model_path,
            num_hands=args.num_hands,
        )

    results = run_extraction(
        args.input,
        args.output,
        factory,
        max_frames=args.max_frames,
        limit=args.limit,
        resume=args.resume,
        progress_every=args.progress_every,
    )
    # A resumed run that had nothing left to do is a success, not a failure.
    return 0 if any(r.status in ("success", "skipped") for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())