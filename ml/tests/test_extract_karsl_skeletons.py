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

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
import extract_karsl_skeletons as ek
from handedness import assign_sides  # noqa: E402

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


# The extractor contract is now: frame -> list of detections, each
# {"landmarks": [...21 pts], "score": float, "rawLabel": "Left"|"Right"}.
# It changed deliberately when two-hand extraction was added; these fakes
# follow the new contract.

def _pose(offset=0.0):
    return [{"x": 0.1 * i + offset, "y": 0.2 * i, "z": 0.0} for i in range(21)]


def _det(offset=0.0, label="Right", score=0.99):
    return {"landmarks": _pose(offset), "score": score, "rawLabel": label}


def fake_extractor_always_hand(frame):
    """Deterministic fake: always 'detects' one fixed synthetic 21-point pose."""
    return [_det()]


def fake_extractor_two_hands(frame):
    """Deterministic fake: always 'detects' two well-separated hands."""
    return [_det(0.0, "Right"), _det(0.5, "Left")]


def fake_extractor_never_hand(frame):
    return []


def fake_extractor_alternating(state={"n": 0}):  # noqa: B006 -- deliberate mutable default for a closure counter
    def extractor(frame):
        state["n"] += 1
        return fake_extractor_always_hand(frame) if state["n"] % 2 == 0 else []
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

class TestHandSlotTracker:
    """The tracker exists to fix a real defect: with num_hands=1, MediaPipe's
    single returned hand was not pinned to one physical hand, so six of
    fourteen sampled real KArSL videos produced sequences that swapped hands
    part-way through. Measured on real video, nearest-wrist tracking cut the
    worst inter-frame wrist jump from 0.237 to 0.030 image units on sign 0227
    and from 0.214 to 0.032 on 0241 (a 0.2 jump is a quarter of the image in
    one frame -- physically impossible for a hand, i.e. a swap)."""

    @staticmethod
    def _hand(x, y, label="Right"):
        return {
            "landmarks": [{"x": x + 0.001 * i, "y": y + 0.001 * i, "z": 0.0} for i in range(21)],
            "score": 0.99,
            "rawLabel": label,
        }

    def test_keeps_each_hand_in_its_own_slot_when_mediapipe_reverses_order(self):
        t = ek.HandSlotTracker()
        # Two hands that stay put, but MediaPipe alternates its ordering.
        a, b = self._hand(0.2, 0.5, "Right"), self._hand(0.8, 0.5, "Left")
        first = t.step([a, b])
        assert first[0]["landmarks"][0]["x"] == pytest.approx(0.2)
        for i in range(10):
            slots = t.step([b, a] if i % 2 == 0 else [a, b])
            # Slot 0 must remain the hand near x=0.2 regardless of input order.
            assert slots[0]["landmarks"][0]["x"] == pytest.approx(0.2)
            assert slots[1]["landmarks"][0]["x"] == pytest.approx(0.8)

    def test_a_single_detection_goes_to_the_nearer_slot(self):
        t = ek.HandSlotTracker()
        t.step([self._hand(0.2, 0.5, "Right"), self._hand(0.8, 0.5, "Left")])
        slots = t.step([self._hand(0.79, 0.5, "Left")])
        assert slots[0] is None
        assert slots[1] is not None

    def test_does_not_snap_a_far_away_detection_into_an_existing_slot(self):
        t = ek.HandSlotTracker(gate=0.1)
        t.step([self._hand(0.2, 0.5, "Right")])
        # A detection way outside the gate must not be treated as slot 0
        # continuing -- that is exactly the splice this class prevents.
        slots = t.step([self._hand(0.9, 0.5, "Right")])
        assert slots[0] is None or slots[0]["landmarks"][0]["x"] != pytest.approx(0.2)

    def test_no_detections_yields_two_empty_slots(self):
        t = ek.HandSlotTracker()
        assert t.step([]) == [None, None]

    def test_slot_naming_uses_the_canonical_rule_not_a_convention_flag(self):
        """Slot naming moved out of this class into handedness.assign_sides().

        The old assertion was that a "camera" convention INVERTED MediaPipe's
        label, on the documented assumption that the label presumes a mirrored
        selfie image. Measured against geometry on 9,852 real two-hand KArSL
        frames, that inversion is wrong: the label maps to the signer's anatomy
        directly, 95.9% of the time. The assertion is therefore reversed rather
        than dropped -- a hand at the SMALLER image x is the signer's right,
        and MediaPipe's "Right" agrees with it.
        """
        t = ek.HandSlotTracker()
        slots = t.step([self._hand(0.2, 0.5, "Right"), self._hand(0.8, 0.5, "Left")])
        assert slots[0] is not None and slots[1] is not None

        a = assign_sides(
            [slots[0]["landmarks"]],
            [slots[1]["landmarks"]],
            [slots[0]["rawLabel"]],
            [slots[1]["rawLabel"]],
        )
        # One frame is below GEOMETRY_MIN_FRAMES, so the label decides -- and
        # the label is used as-is, not inverted.
        assert a.source == "handedness-label"
        assert list(a.slot_sides) == ["right", "left"]

    def test_geometry_outranks_the_label_once_there_are_enough_frames(self):
        """With enough both-hand frames the naming must come from geometry,
        which is a measurement rather than a model output -- even when the
        label disagrees with it."""
        t = ek.HandSlotTracker()
        left_slot, right_slot = [], []
        for i in range(6):
            slots = t.step(
                [
                    # Deliberately MISLABELLED: the hand at the smaller x is
                    # called "Left" by the model. Geometry must overrule it.
                    self._hand(0.2 + i * 0.001, 0.5, "Left"),
                    self._hand(0.8 - i * 0.001, 0.5, "Right"),
                ]
            )
            left_slot.append(slots[0]["landmarks"])
            right_slot.append(slots[1]["landmarks"])

        a = assign_sides(left_slot, right_slot, ["Left"] * 6, ["Right"] * 6)
        assert a.source == "geometry"
        assert a.confidence == pytest.approx(1.0)
        assert list(a.slot_sides) == ["right", "left"]

    def test_slot_names_are_never_both_the_same(self):
        t = ek.HandSlotTracker()
        s1 = t.step([self._hand(0.2, 0.5, "Right")])
        s2 = t.step([self._hand(0.8, 0.5, "Right")])
        a = assign_sides(
            [s1[0]["landmarks"] if s1[0] else None, s2[0]["landmarks"] if s2[0] else None],
            [s1[1]["landmarks"] if s1[1] else None, s2[1]["landmarks"] if s2[1] else None],
            ["Right", "Right"],
            ["Right", "Right"],
        )
        # Two slots are two different physical hands; naming both the same side
        # would be a contradiction, so they are always given opposite names.
        assert set(a.slot_sides) == {"left", "right"}


class TestTwoHandOutputSchema:
    def test_process_video_records_both_hands_and_keeps_legacy_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "two.mp4")
            make_synthetic_video(video_path, num_frames=8)
            result = ek.process_video(video_path, "0171", fake_extractor_two_hands, progress_every=0)

            assert result["handTracking"]["numHands"] == 2
            assert result["handTracking"]["framesWithTwoHands"] == result["frameCount"]

            for frame in result["frames"]:
                # New two-hand payload
                assert frame["hands"]["left"] is not None
                assert frame["hands"]["right"] is not None
                assert len(frame["hands"]["left"]["landmarks"]) == 21
                # Legacy fields preserved so existing consumers keep working
                assert frame["handDetected"] is True
                assert len(frame["landmarks"]) == 21

    def test_one_handed_video_leaves_the_other_side_null_not_fabricated(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "one.mp4")
            make_synthetic_video(video_path, num_frames=6)
            result = ek.process_video(video_path, "0339", fake_extractor_always_hand, progress_every=0)

            assert result["handTracking"]["framesWithTwoHands"] == 0
            sides = [(f["hands"]["left"], f["hands"]["right"]) for f in result["frames"]]
            # Exactly one side is ever populated; the other stays None.
            for left, right in sides:
                assert (left is None) != (right is None)


class TestFilenameProvenance:
    """The enlarged KArSL drop carries recording date and two numeric fields in
    every filename. Those are the only signer/session evidence the dataset
    actually provides, so they have to survive extraction -- and a name that
    does not follow the convention must produce null, never a guess."""

    def test_parses_a_real_karsl_filename_into_its_actual_fields(self):
        p = ek.parse_provenance("03_01_0071_(01_12_16_15_52_41)_c")
        assert p == {
            "field1": "03",
            "field2": "01",
            "signIdInName": "0071",
            "sessionDate": "01_12_16",
            "recordedAt": "01_12_16_15_52_41",
            "suffix": "c",
        }

    def test_takes_from_one_sitting_share_a_session_date(self):
        a = ek.parse_provenance("03_01_0071_(01_12_16_15_52_41)_c")
        b = ek.parse_provenance("03_01_0071_(01_12_16_15_54_22)_c")
        c = ek.parse_provenance("03_01_0071_(20_11_16_16_16_41)_c")
        assert a["sessionDate"] == b["sessionDate"]
        assert a["sessionDate"] != c["sessionDate"]

    def test_a_name_outside_the_convention_yields_null_not_a_guess(self):
        for stem in ["video1", "0071_take3", "03_01_0071_c", "03_01_0071_(bad)_c"]:
            assert ek.parse_provenance(stem) is None

    def test_processed_output_carries_the_stem_and_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            name = "03_01_0071_(01_12_16_15_52_41)_c"
            video_path = os.path.join(tmp, name + ".mp4")
            make_synthetic_video(video_path, num_frames=4)
            result = ek.process_video(video_path, "0071", fake_extractor_always_hand, progress_every=0)
            assert result["sourceVideoStem"] == name
            assert result["provenance"]["sessionDate"] == "01_12_16"
            assert result["signId"] == "0071"

    def test_provenance_is_null_when_the_filename_does_not_follow_the_convention(self):
        with tempfile.TemporaryDirectory() as tmp:
            video_path = os.path.join(tmp, "homemade.mp4")
            make_synthetic_video(video_path, num_frames=4)
            result = ek.process_video(video_path, "0071", fake_extractor_always_hand, progress_every=0)
            assert result["provenance"] is None
            assert result["sourceVideoStem"] == "homemade"


class TestResume:
    """A full pass over the enlarged dataset is tens of thousands of videos.
    Resume is what makes an interruption survivable -- and it must not trust a
    half-written file left behind by the interruption itself."""

    def _tree(self, tmp, count=3):
        input_dir = os.path.join(tmp, "KArSL", "0071")
        os.makedirs(input_dir)
        for i in range(count):
            make_synthetic_video(os.path.join(input_dir, f"take{i}.mp4"), num_frames=3)
        return os.path.join(tmp, "KArSL"), os.path.join(tmp, "out")

    def test_second_run_reuses_existing_output_instead_of_re_extracting(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = self._tree(tmp)
            first = ek.run_extraction(src, out, lambda: fake_extractor_always_hand, resume=True)
            assert [r.status for r in first] == ["success"] * 3

            calls = {"n": 0}

            def counting_factory():
                calls["n"] += 1
                return fake_extractor_always_hand

            second = ek.run_extraction(src, out, counting_factory, resume=True)
            assert [r.status for r in second] == ["skipped"] * 3
            assert calls["n"] == 0, "resume re-ran MediaPipe on already-extracted videos"

    def test_without_resume_everything_is_re_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = self._tree(tmp)
            ek.run_extraction(src, out, lambda: fake_extractor_always_hand)
            again = ek.run_extraction(src, out, lambda: fake_extractor_always_hand)
            assert [r.status for r in again] == ["success"] * 3

    def test_a_truncated_output_from_a_killed_run_is_redone_not_trusted(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = self._tree(tmp)
            ek.run_extraction(src, out, lambda: fake_extractor_always_hand, resume=True)
            victim = os.path.join(out, "0071", "take1.json")
            with open(victim, "w", encoding="utf-8") as f:
                f.write('{"signId": "0071", "frames": [')  # killed mid-write

            results = {r.source_video: r.status for r in
                       ek.run_extraction(src, out, lambda: fake_extractor_always_hand, resume=True)}
            assert results["take1.mp4"] == "success", "a truncated file was reused as if complete"
            assert results["take0.mp4"] == "skipped"

    def test_an_empty_frame_list_is_also_redone(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = self._tree(tmp)
            ek.run_extraction(src, out, lambda: fake_extractor_always_hand, resume=True)
            with open(os.path.join(out, "0071", "take2.json"), "w", encoding="utf-8") as f:
                json.dump({"signId": "0071", "frames": []}, f)
            results = {r.source_video: r.status for r in
                       ek.run_extraction(src, out, lambda: fake_extractor_always_hand, resume=True)}
            assert results["take2.mp4"] == "success"
