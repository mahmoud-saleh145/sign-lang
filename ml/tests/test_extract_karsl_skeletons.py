"""
Tests for ml/training/extract_karsl_skeletons.py using SYNTHETIC fixtures
only: tiny generated video files (solid-color frames, no real hand) and a
fake, deterministic landmark extractor. No real KArSL data is used or
required to run these tests.

One test (test_real_mediapipe_extractor_runs_without_crashing) does invoke
the REAL MediaPipe extractor against a synthetic blank-frame video -- this
proves the real integration doesn't crash and correctly reports "no hand
detected" (which is accurate for a blank frame), without needing a real
video of an actual hand.
"""
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
import extract_karsl_skeletons as ek  # noqa: E402

import cv2
import numpy as np


def make_synthetic_video(path: str, num_frames: int = 10, fps: float = 30.0, size=(64, 48)):
    """Writes a tiny synthetic video: solid color frames, no real hand
    content. Purely for testing I/O and pipeline mechanics."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    for i in range(num_frames):
        frame = np.full((size[1], size[0], 3), fill_value=(i * 10) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def fake_extractor_always_hand(frame):
    """Deterministic fake extractor: always 'detects' a fixed synthetic 21-point pose."""
    return [{"x": 0.1 * i, "y": 0.2 * i, "z": 0.0} for i in range(21)]


def fake_extractor_never_hand(frame):
    return None


def fake_extractor_alternating(state={"n": 0}):  # noqa: B006 -- deliberate mutable default for a closure counter
    def extractor(frame):
        state["n"] += 1
        return fake_extractor_always_hand(frame) if state["n"] % 2 == 0 else None
    return extractor


class TestProcessVideo:
    def test_reads_correct_frame_count_and_fps_from_a_real_video_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "video1.mp4")
            make_synthetic_video(video_path, num_frames=8, fps=25.0)

            result = ek.process_video(video_path, "0001", fake_extractor_always_hand)

            assert result["signId"] == "0001"
            assert result["sourceVideo"] == "video1.mp4"
            assert result["frameCount"] == 8
            assert len(result["frames"]) == 8
            assert result["fps"] > 0

    def test_frame_indices_and_timestamps_are_sequential_and_correct(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "v.mp4")
            make_synthetic_video(video_path, num_frames=5, fps=10.0)
            result = ek.process_video(video_path, "0002", fake_extractor_always_hand)

            indices = [f["frameIndex"] for f in result["frames"]]
            assert indices == list(range(5))
            # frame 1 at 10fps should be ~100ms in
            assert abs(result["frames"][1]["timestampMs"] - 100.0) < 5.0

    def test_frames_with_no_detected_hand_have_null_landmarks_not_fabricated_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "v.mp4")
            make_synthetic_video(video_path, num_frames=4)
            result = ek.process_video(video_path, "0003", fake_extractor_never_hand)

            for frame in result["frames"]:
                assert frame["handDetected"] is False
                assert frame["landmarks"] is None

    def test_frames_with_detected_hand_carry_21_real_points(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "v.mp4")
            make_synthetic_video(video_path, num_frames=3)
            result = ek.process_video(video_path, "0004", fake_extractor_always_hand)

            for frame in result["frames"]:
                assert frame["handDetected"] is True
                assert len(frame["landmarks"]) == 21
                assert all({"x", "y", "z"} <= set(p.keys()) for p in frame["landmarks"])

    def test_max_frames_caps_how_many_frames_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "v.mp4")
            make_synthetic_video(video_path, num_frames=20)
            result = ek.process_video(video_path, "0005", fake_extractor_always_hand, max_frames=5)
            assert result["frameCount"] == 5

    def test_unreadable_video_raises_rather_than_silently_producing_empty_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            fake_video = os.path.join(tmp, "not_a_video.mp4")
            with open(fake_video, "w") as f:
                f.write("this is not a video file")
            try:
                ek.process_video(fake_video, "0006", fake_extractor_always_hand)
                raised = False
            except RuntimeError:
                raised = True
            assert raised


class TestFindSignDirectoriesAndVideos:
    def test_zero_padded_sign_ids_are_preserved_as_strings(self):
        with tempfile.TemporaryDirectory() as tmp:
            os.makedirs(os.path.join(tmp, "0001"))
            os.makedirs(os.path.join(tmp, "0042"))
            sign_dirs = ek.find_sign_directories(tmp)
            assert sign_dirs == ["0001", "0042"]
            assert all(isinstance(s, str) for s in sign_dirs)
            # explicitly guard against the exact regression class of bug
            # found in import_karsl.py: no accidental int() coercion
            assert "1" not in sign_dirs
            assert "42" not in sign_dirs

    def test_finds_videos_by_supported_extension_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            sign_dir = os.path.join(tmp, "0001")
            os.makedirs(sign_dir)
            for name in ["a.mp4", "b.avi", "notes.txt", "c.MOV"]:
                open(os.path.join(sign_dir, name), "w").close()
            videos = ek.find_videos(sign_dir)
            names = [os.path.basename(v) for v in videos]
            assert "a.mp4" in names
            assert "b.avi" in names
            assert "c.MOV" in names  # extension matching is case-insensitive
            assert "notes.txt" not in names


class TestRunExtractionEndToEnd:
    def test_full_pipeline_on_a_synthetic_karsl_like_folder(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = os.path.join(tmp, "KArSL")
            output_dir = os.path.join(tmp, "processed-karsl")
            os.makedirs(os.path.join(input_dir, "0001"))
            os.makedirs(os.path.join(input_dir, "0002"))
            make_synthetic_video(os.path.join(input_dir, "0001", "video1.mp4"), num_frames=6)
            make_synthetic_video(os.path.join(input_dir, "0002", "video1.mp4"), num_frames=4)

            results = ek.run_extraction(input_dir, output_dir, lambda: fake_extractor_always_hand)

            assert len(results) == 2
            assert all(r.status == "success" for r in results)

            # Real files actually exist on disk with the documented structure.
            assert os.path.exists(os.path.join(output_dir, "0001", "video1.json"))
            assert os.path.exists(os.path.join(output_dir, "0002", "video1.json"))
            assert os.path.exists(os.path.join(output_dir, "manifest.json"))

            with open(os.path.join(output_dir, "0001", "video1.json")) as f:
                data = json.load(f)
            assert data["signId"] == "0001"  # string, zero-padding preserved
            assert data["frameCount"] == 6

            with open(os.path.join(output_dir, "manifest.json")) as f:
                manifest = json.load(f)
            assert manifest["videosFound"] == 2
            assert manifest["videosProcessed"] == 2
            assert manifest["videosFailed"] == 0

    def test_one_bad_video_does_not_abort_the_whole_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = os.path.join(tmp, "KArSL")
            output_dir = os.path.join(tmp, "processed-karsl")
            os.makedirs(os.path.join(input_dir, "0001"))
            os.makedirs(os.path.join(input_dir, "0002"))
            make_synthetic_video(os.path.join(input_dir, "0001", "good.mp4"), num_frames=5)
            # a corrupt "video" that will fail to open
            with open(os.path.join(input_dir, "0002", "bad.mp4"), "w") as f:
                f.write("not a real video")

            results = ek.run_extraction(input_dir, output_dir, lambda: fake_extractor_always_hand)

            statuses = {r.sign_id: r.status for r in results}
            assert statuses["0001"] == "success"
            assert statuses["0002"] == "failed"
            # the good video's output must still exist despite the other failure
            assert os.path.exists(os.path.join(output_dir, "0001", "good.json"))

    def test_empty_input_directory_reports_no_videos_found_without_crashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            input_dir = os.path.join(tmp, "KArSL")
            os.makedirs(input_dir)
            output_dir = os.path.join(tmp, "out")
            results = ek.run_extraction(input_dir, output_dir, lambda: fake_extractor_always_hand)
            assert results == []


class TestRealMediaPipeIntegration:
    def test_real_mediapipe_extractor_runs_without_crashing(self):
        """Uses the REAL MediaPipe HandLandmarker (Tasks API, not a fake)
        against a synthetic blank-frame video. Correctly expects no hand
        detected -- that's an honest result for a frame with no real hand in
        it, not a test failure.

        Requires downloading a ~10MB model file on first run. In network-
        restricted environments (e.g. this project's build sandbox) that
        download will fail -- this test skips cleanly in that case rather
        than reporting a false failure, since it's an environment
        limitation, not a code bug. It will run for real wherever normal
        internet access is available, e.g. on the machine this script is
        actually meant to run on.
        """
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "v.mp4")
            make_synthetic_video(video_path, num_frames=3)

            try:
                extractor = ek.build_real_mediapipe_extractor()
            except RuntimeError as e:
                if "download" in str(e).lower():
                    import pytest

                    pytest.skip(f"Model download unavailable in this environment: {e}")
                raise

            try:
                result = ek.process_video(video_path, "0001", extractor)
            finally:
                extractor.close()

            assert result["frameCount"] == 3
            assert all(f["handDetected"] is False for f in result["frames"])