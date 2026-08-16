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
      "frames": [
        {
          "frameIndex": 0,
          "timestampMs": 0.0,
          "handDetected": true,
          "landmarks": [{"x":.., "y":.., "z":..}, ... 21 points] | null
        },
        ...
      ]
    }

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
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".wmv")

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


def build_real_mediapipe_extractor(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
    model_path: Optional[str] = None,
):
    """Returns a real MediaPipe Tasks-API-backed extractor callable, reused
    across all frames of one video (HandLandmarker tracks between frames
    when reused this way in VIDEO running mode, rather than re-detecting
    cold every frame). Caller MUST call the returned extractor's `.close()`
    when done with a video.

    Uses mp.tasks.vision.HandLandmarker -- verified against the actually
    installed mediapipe package via introspection (mediapipe>=0.10 removed
    the older mp.solutions.hands API entirely). This is the same MediaPipe
    Tasks family already used in the browser (lib/mediapipe/handLandmarker.ts).
    """
    import mediapipe as mp
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.core.base_options import BaseOptions

    resolved_model_path = resolve_model_path(model_path)

    options = mp_vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=resolved_model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
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
        if not result.hand_landmarks:
            return None
        hand = result.hand_landmarks[0]
        return [{"x": float(lm.x), "y": float(lm.y), "z": float(lm.z)} for lm in hand]

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
    video into memory) and returns the per-video JSON-ready dict described
    in the module docstring. Raises on an unreadable video; caller is
    responsible for catching and recording the failure so one bad video
    doesn't abort the whole run.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames: list[dict] = []
    frame_index = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_index >= max_frames:
                break

            landmarks = extractor(frame)
            timestamp_ms = (frame_index / fps * 1000) if fps > 0 else float(frame_index)

            frames.append(
                {
                    "frameIndex": frame_index,
                    "timestampMs": round(timestamp_ms, 2),
                    "handDetected": landmarks is not None,
                    "landmarks": landmarks,
                }
            )
            frame_index += 1

            if progress_every and frame_index % progress_every == 0:
                print(f"    ...{frame_index} frames read", flush=True)
    finally:
        cap.release()

    return {
        "signId": sign_id,
        "sourceVideo": os.path.basename(video_path),
        "fps": fps,
        "frameCount": len(frames),
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
    start = time.time()

    for i, (sign_id, video_path) in enumerate(all_videos, 1):
        rel = os.path.relpath(video_path, input_dir)
        print(f"[{i}/{len(all_videos)}] {rel}")

        out_sign_dir = os.path.join(output_dir, sign_id)
        os.makedirs(out_sign_dir, exist_ok=True)
        out_name = os.path.splitext(os.path.basename(video_path))[0] + ".json"
        out_path = os.path.join(out_sign_dir, out_name)

        extractor = extractor_factory()
        try:
            data = process_video(video_path, sign_id, extractor, max_frames=max_frames)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            frames_with_hand = sum(1 for fr in data["frames"] if fr["handDetected"])
            print(
                f"    OK -- {data['frameCount']} frames, "
                f"{frames_with_hand}/{data['frameCount']} with a detected hand"
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
    failed = [r for r in results if r.status == "failed"]

    manifest = {
        "inputDir": input_dir,
        "videosFound": len(all_videos),
        "videosProcessed": len(succeeded),
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
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap on frames read per video (for quick tests)")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on total videos processed (for quick tests)")
    args = parser.parse_args()

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
            args.min_detection_confidence, args.min_tracking_confidence, args.model_path
        )

    results = run_extraction(args.input, args.output, factory, max_frames=args.max_frames, limit=args.limit)
    return 0 if any(r.status == "success" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())