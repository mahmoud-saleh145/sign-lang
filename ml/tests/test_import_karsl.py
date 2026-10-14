"""
Tests the KArSL importer's parsing logic and Sign ID matching against
SYNTHETIC test fixtures -- not real KArSL files. These prove the tool works
correctly on well-formed input (including the REAL zero-padded-folder vs.
plain-integer-label ID mismatch found in the user's actual KARSL-502_Labels
file) and fails safely on malformed input.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))
import import_karsl as ik  # noqa: E402

IMPORTER_PATH = os.path.join(os.path.dirname(__file__), "..", "training", "import_karsl.py")


def write_labels_csv(root, rows):
    labels_path = os.path.join(root, "labels.csv")
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sign_id", "arabic_label"])
        for sign_id, label in rows:
            writer.writerow([sign_id, label])
    return labels_path


def make_synthetic_dataset(root):
    sign_points = {"0001": (0.1, 0.2, 0.0), "0002": (0.5, 0.5, 0.0)}
    for sign_id, point in sign_points.items():
        sign_dir = os.path.join(root, sign_id)
        os.makedirs(sign_dir, exist_ok=True)
        for i in range(3):
            with open(os.path.join(sign_dir, f"frame_{i:03d}.txt"), "w") as f:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

    labels_path = write_labels_csv(root, [("1", "\u0645\u0631\u062d\u0628\u0627")])
    return labels_path


def run_importer(input_dir, labels_path, output_path):
    return subprocess.run(
        [sys.executable, IMPORTER_PATH, "--input", input_dir, "--labels", labels_path, "--output", output_path],
        capture_output=True,
        text=True,
    )


def test_arabic_labels_do_not_kill_the_import_on_a_legacy_console():
    """A non-Unicode stdout must not cost you the registry.

    Real KArSL labels are Arabic and the importer prints each one as it goes.
    On Windows, `python -m pytest` inherits a cp1252 stdout, so that print
    raised UnicodeEncodeError -- and because the output JSON is written only
    after the import loop, the process died BEFORE writing it. Thirteen tests
    then failed with FileNotFoundError pointing at a missing output file
    instead of at the encoding, on code that passed on a UTF-8 machine.

    PYTHONIOENCODING reproduces that stdout on any platform, so this guards the
    real condition rather than a Windows-only approximation.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0001")
        os.makedirs(sign_dir)
        video_json = make_processed_karsl_video_json("0001", "v.mp4", (0.1, 0.2), [True, True])
        with open(os.path.join(sign_dir, "v.json"), "w", encoding="utf-8") as f:
            json.dump(video_json, f)

        arabic = "\u0625\u0633\u0639\u0627\u0641\u0627\u062a \u0623\u0648\u0644\u064a\u0629"
        labels_path = write_labels_csv(tmp, [("1", arabic)])
        output_path = os.path.join(tmp, "out.json")

        env = dict(os.environ, PYTHONIOENCODING="cp1252")
        result = subprocess.run(
            [sys.executable, IMPORTER_PATH, "--input", tmp, "--labels", labels_path, "--output", output_path],
            capture_output=True,
            text=True,
            env=env,
        )

        assert result.returncode == 0, result.stderr
        assert "UnicodeEncodeError" not in result.stderr
        assert os.path.exists(output_path), "the registry was not written"

        # The console may not be able to SHOW Arabic, but the registry must
        # still contain it -- the label is data, not console output.
        with open(output_path, encoding="utf-8") as f:
            entries = json.load(f)
        assert entries[0]["text"] == arabic


def test_canonicalize_sign_id_strips_leading_zeros():
    assert ik.canonicalize_sign_id("0001") == "1"
    assert ik.canonicalize_sign_id("0002") == "2"
    assert ik.canonicalize_sign_id("0010") == "10"
    assert ik.canonicalize_sign_id("0502") == "502"


def test_canonicalize_sign_id_leaves_already_canonical_ids_unchanged():
    assert ik.canonicalize_sign_id("1") == "1"
    assert ik.canonicalize_sign_id("502") == "502"


def test_canonicalize_sign_id_handles_all_zero_string():
    assert ik.canonicalize_sign_id("0") == "0"
    assert ik.canonicalize_sign_id("000") == "0"


def test_canonicalize_sign_id_leaves_non_numeric_ids_unchanged():
    assert ik.canonicalize_sign_id("abc") == "abc"
    assert ik.canonicalize_sign_id("sign_01") == "sign_01"


def test_canonical_label_index_matches_zero_padded_folder_to_plain_label():
    labels = {"1": "\u0645\u0631\u062d\u0628\u0627", "42": "\u0634\u0643\u0631\u0627"}
    index = ik.build_canonical_label_index(labels)
    assert index[ik.canonicalize_sign_id("0001")] == "\u0645\u0631\u062d\u0628\u0627"
    assert index[ik.canonicalize_sign_id("0042")] == "\u0634\u0643\u0631\u0627"


def test_canonical_label_index_warns_on_collision_and_does_not_silently_overwrite(capsys):
    labels = {"007": "first", "07": "second", "7": "third"}
    index = ik.build_canonical_label_index(labels)
    assert index["7"] in ("first", "second", "third")
    assert len(index) == 1
    captured = capsys.readouterr()
    assert "WARNING" in captured.out


def test_zero_padded_folder_0001_matches_plain_label_1():
    with tempfile.TemporaryDirectory() as tmp:
        labels_path = make_synthetic_dataset(tmp)
        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)

        assert result.returncode == 0, result.stdout + result.stderr
        with open(output_path, encoding="utf-8") as f:
            entries = json.load(f)

        assert len(entries) == 1
        assert entries[0]["text"] == "\u0645\u0631\u062d\u0628\u0627"


def test_zero_padded_folder_0002_matches_plain_label_2():
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0002")
        os.makedirs(sign_dir)
        with open(os.path.join(sign_dir, "frame_000.txt"), "w") as f:
            f.write("0.1 0.2 0.0\n")
        labels_path = write_labels_csv(tmp, [("2", "\u0634\u0643\u0631\u0627")])

        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode == 0, result.stdout + result.stderr
        with open(output_path, encoding="utf-8") as f:
            entries = json.load(f)
        assert entries[0]["text"] == "\u0634\u0643\u0631\u0627"
        assert entries[0]["sourceSignId"] == "0002"


def test_zero_padded_folder_0010_matches_plain_label_10():
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0010")
        os.makedirs(sign_dir)
        with open(os.path.join(sign_dir, "frame_000.txt"), "w") as f:
            f.write("0.1 0.2 0.0\n")
        labels_path = write_labels_csv(tmp, [("10", "\u0639\u0634\u0631\u0629")])

        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode == 0
        with open(output_path, encoding="utf-8") as f:
            entries = json.load(f)
        assert entries[0]["text"] == "\u0639\u0634\u0631\u0629"


def test_zero_padded_folder_0502_matches_plain_label_502():
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0502")
        os.makedirs(sign_dir)
        with open(os.path.join(sign_dir, "frame_000.txt"), "w") as f:
            f.write("0.1 0.2 0.0\n")
        labels_path = write_labels_csv(tmp, [("502", "\u0622\u062e\u0631 \u0631\u0642\u0645")])

        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode == 0
        with open(output_path, encoding="utf-8") as f:
            entries = json.load(f)
        assert entries[0]["text"] == "\u0622\u062e\u0631 \u0631\u0642\u0645"
        assert entries[0]["sourceSignId"] == "0502"


def test_invalid_or_missing_id_does_not_silently_match_another_sign():
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir_present = os.path.join(tmp, "0001")
        os.makedirs(sign_dir_present)
        with open(os.path.join(sign_dir_present, "frame_000.txt"), "w") as f:
            f.write("0.1 0.2 0.0\n")

        sign_dir_missing = os.path.join(tmp, "9999")
        os.makedirs(sign_dir_missing)
        with open(os.path.join(sign_dir_missing, "frame_000.txt"), "w") as f:
            f.write("0.9 0.9 0.0\n")

        labels_path = write_labels_csv(tmp, [("1", "\u0645\u0631\u062d\u0628\u0627")])
        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)

        assert result.returncode == 0
        with open(output_path, encoding="utf-8") as f:
            entries = json.load(f)
        assert len(entries) == 1
        assert entries[0]["sourceSignId"] == "0001"
        assert "9999 (no label found" in result.stdout


def test_sign_without_a_label_is_skipped_not_fabricated():
    with tempfile.TemporaryDirectory() as tmp:
        labels_path = make_synthetic_dataset(tmp)
        output_path = os.path.join(tmp, "out.json")
        run_importer(tmp, labels_path, output_path)

        with open(output_path, encoding="utf-8") as f:
            entries = json.load(f)
        assert len(entries) == 1


def test_fails_loudly_on_empty_input_directory():
    with tempfile.TemporaryDirectory() as tmp:
        empty_input = os.path.join(tmp, "empty")
        os.makedirs(empty_input)
        labels_path = write_labels_csv(tmp, [])
        output_path = os.path.join(tmp, "out.json")

        result = run_importer(empty_input, labels_path, output_path)
        assert result.returncode != 0
        assert "no subdirectories found" in result.stdout


def test_fails_loudly_when_no_signs_could_be_imported():
    with tempfile.TemporaryDirectory() as tmp:
        os.makedirs(os.path.join(tmp, "0001"))
        labels_path = write_labels_csv(tmp, [])
        output_path = os.path.join(tmp, "out.json")

        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode != 0
        assert "No signs were imported" in result.stdout


def make_processed_karsl_video_json(sign_id, source_video, points, hand_detected_flags):
    frames = []
    for i, detected in enumerate(hand_detected_flags):
        frames.append(
            {
                "frameIndex": i,
                "timestampMs": i * 33.3,
                "handDetected": detected,
                "landmarks": (
                    [{"x": points[0], "y": points[1], "z": 0.0} for _ in range(21)] if detected else None
                ),
            }
        )
    return {"signId": sign_id, "sourceVideo": source_video, "fps": 30.0, "frameCount": len(frames), "frames": frames}


def test_prefers_the_real_processed_karsl_format_over_the_fallback():
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0001")
        os.makedirs(sign_dir)
        video_json = make_processed_karsl_video_json("0001", "video1.mp4", (0.1, 0.2), [True, True, True])
        with open(os.path.join(sign_dir, "video1.json"), "w", encoding="utf-8") as f:
            json.dump(video_json, f)

        labels_path = write_labels_csv(tmp, [("1", "\u0645\u0631\u062d\u0628\u0627")])

        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode == 0, result.stdout + result.stderr

        with open(output_path, encoding="utf-8") as f:
            entries = json.load(f)
        assert len(entries) == 1
        assert len(entries[0]["landmarkSequence"]) == 3


def test_frames_with_no_detected_hand_are_dropped_not_fabricated():
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0001")
        os.makedirs(sign_dir)
        video_json = make_processed_karsl_video_json(
            "0001", "video1.mp4", (0.1, 0.2), [True, False, False, True, False]
        )
        with open(os.path.join(sign_dir, "video1.json"), "w", encoding="utf-8") as f:
            json.dump(video_json, f)

        labels_path = write_labels_csv(tmp, [("1", "\u0645\u0631\u062d\u0628\u0627")])
        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode == 0
        assert "dropped 3/5" in result.stdout

        with open(output_path, encoding="utf-8") as f:
            entries = json.load(f)
        assert len(entries[0]["landmarkSequence"]) == 2


def test_multiple_videos_per_sign_imports_one_take_and_says_which():
    """Exactly one take is imported per sign, and the choice is reported.

    This test previously asserted that the FIRST take alphabetically was used.
    That was encoding a defect, not a requirement: on the real dataset the
    alphabetically-first take is often the one where the second hand was never
    detected, which imported two-handed signs as one-handed. The requirement is
    that one take is chosen and the reason is visible -- see
    select_best_take() and the take-selection tests below.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0001")
        os.makedirs(sign_dir)
        for name in ["video1.mp4", "video2.mp4"]:
            video_json = make_processed_karsl_video_json("0001", name, (0.1, 0.2), [True, True])
            with open(os.path.join(sign_dir, name.replace(".mp4", ".json")), "w", encoding="utf-8") as f:
                json.dump(video_json, f)

        labels_path = write_labels_csv(tmp, [("1", "\u0645\u0631\u062d\u0628\u0627")])
        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode == 0
        assert "of 2 take(s)" in result.stdout

        with open(output_path, encoding="utf-8") as f:
            entries = json.load(f)
        assert len(entries) == 1


def test_source_sign_id_preserves_original_zero_padded_folder_name():
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0007")
        os.makedirs(sign_dir)
        video_json = make_processed_karsl_video_json("0007", "video1.mp4", (0.1, 0.2), [True])
        with open(os.path.join(sign_dir, "video1.json"), "w", encoding="utf-8") as f:
            json.dump(video_json, f)

        labels_path = write_labels_csv(tmp, [("7", "\u0634\u0643\u0631\u0627")])
        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode == 0
        with open(output_path, encoding="utf-8") as f:
            entries = json.load(f)
        assert entries[0]["text"] == "\u0634\u0643\u0631\u0627"
        assert entries[0]["sourceSignId"] == "0007"

def _take_json(sign_id, both_frames, right_only_frames, left_x=0.8, right_x=0.2):
    """A processed take: `both_frames` frames with two hands, then
    `right_only_frames` frames with just the hand at the smaller image x."""
    frames = []
    n = both_frames + right_only_frames
    for i in range(n):
        hands = {
            "right": {
                "landmarks": [{"x": right_x, "y": 0.5, "z": 0.0} for _ in range(21)],
                "score": 0.9,
                "rawLabel": "Right",
            },
            "left": None,
        }
        if i < both_frames:
            hands["left"] = {
                "landmarks": [{"x": left_x, "y": 0.5, "z": 0.0} for _ in range(21)],
                "score": 0.9,
                "rawLabel": "Left",
            }
        frames.append(
            {
                "frameIndex": i,
                "timestampMs": i * 33.0,
                "handDetected": True,
                "landmarks": [{"x": right_x, "y": 0.5, "z": 0.0} for _ in range(21)],
                "hands": hands,
            }
        )
    return {
        "signId": "0011",
        "sourceVideo": "v.mp4",
        "fps": 30.0,
        "frameCount": n,
        "handTracking": {
            "numHands": 2,
            "slotNames": ["left", "right"],
            "primaryHand": "left",
            "slotSwapCount": 0,
        },
        "frames": frames,
    }


def test_a_one_frame_second_hand_does_not_win_take_selection():
    """A second hand seen in one frame out of fifty is a detection flicker.

    Real defect this guards: signs 0076 and 0089 each picked a take on the
    strength of ONE both-hand frame ("most two-hand frames (1; next best 1)"),
    which made a genuinely one-handed sign look two-handed and left the legacy
    sequence with a single frame.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0011")
        os.makedirs(sign_dir)
        # flicker.json: 1 both-hand frame out of 50 -- 2%, below the floor.
        # solid.json:   0 both-hand frames but 60 frames of real motion.
        with open(os.path.join(sign_dir, "flicker.json"), "w", encoding="utf-8") as f:
            json.dump(_take_json("0011", 1, 49), f)
        with open(os.path.join(sign_dir, "solid.json"), "w", encoding="utf-8") as f:
            json.dump(_take_json("0011", 0, 60), f)

        labels_path = write_labels_csv(tmp, [("11", "\u0645\u0631\u062d\u0628\u0627")])
        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode == 0

        with open(output_path, encoding="utf-8") as f:
            entry = json.load(f)[0]
        assert entry["sourceTake"]["processedFile"] == "solid.json"
        assert "flicker" in entry["sourceTake"]["selectionReason"]


def test_a_real_second_hand_still_wins_take_selection():
    """The floor must not throw away a genuinely two-handed take."""
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0011")
        os.makedirs(sign_dir)
        with open(os.path.join(sign_dir, "twohanded.json"), "w", encoding="utf-8") as f:
            json.dump(_take_json("0011", 30, 10), f)
        with open(os.path.join(sign_dir, "longer.json"), "w", encoding="utf-8") as f:
            json.dump(_take_json("0011", 0, 60), f)

        labels_path = write_labels_csv(tmp, [("11", "\u0645\u0631\u062d\u0628\u0627")])
        output_path = os.path.join(tmp, "out.json")
        assert run_importer(tmp, labels_path, output_path).returncode == 0

        with open(output_path, encoding="utf-8") as f:
            entry = json.load(f)[0]
        assert entry["sourceTake"]["processedFile"] == "twohanded.json"
        assert entry["handedness"]["twoHanded"] is True


def test_primary_hand_names_the_side_that_actually_has_the_frames():
    """`primaryHand` is derived from the canonical frames, not translated.

    The previous version mapped the extractor's own primary through
    `slotNames.index()`, mixing an index-ordered list with a key-ordered one.
    On the real registry that inverted for signs 0076 and 0089, naming a side
    with ONE frame as primary against fifty on the other.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0011")
        os.makedirs(sign_dir)
        with open(os.path.join(sign_dir, "v.json"), "w", encoding="utf-8") as f:
            # One left frame, 40 right frames -- and the source file claims
            # "left" is primary, which it must not be believed about.
            json.dump(_take_json("0011", 1, 40), f)

        labels_path = write_labels_csv(tmp, [("11", "\u0645\u0631\u062d\u0628\u0627")])
        output_path = os.path.join(tmp, "out.json")
        assert run_importer(tmp, labels_path, output_path).returncode == 0

        with open(output_path, encoding="utf-8") as f:
            entry = json.load(f)[0]
        h = entry["handedness"]
        assert h["framesWithRight"] > h["framesWithLeft"]
        assert h["primaryHand"] == "right"


def test_source_take_traces_the_template_back_to_a_real_recording():
    """Which take a template came from must be recorded.

    Before this, an entry carried only `sourceSignId`, so a recognition error
    could be traced to a sign but not to the recording behind it -- and since
    select_best_take() chooses by content rather than position, it was not
    recoverable afterwards either.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0007")
        os.makedirs(sign_dir)
        for name in ["takeA.mp4", "takeB.mp4"]:
            video_json = make_processed_karsl_video_json("0007", name, (0.1, 0.2), [True, True])
            with open(os.path.join(sign_dir, name.replace(".mp4", ".json")), "w", encoding="utf-8") as f:
                json.dump(video_json, f)

        labels_path = write_labels_csv(tmp, [("7", "\u0634\u0643\u0631\u0627")])
        output_path = os.path.join(tmp, "out.json")
        assert run_importer(tmp, labels_path, output_path).returncode == 0

        with open(output_path, encoding="utf-8") as f:
            entries = json.load(f)
        take = entries[0]["sourceTake"]
        # The recorded file must be one that actually exists on disk -- not a
        # constructed or guessed name.
        assert take["processedFile"] in os.listdir(sign_dir)
        assert take["sourceVideoStem"] == os.path.splitext(take["processedFile"])[0]
        assert take["sourceFolder"] == "0007"
        assert take["takesAvailable"] == 2
        assert take["selectionReason"]


def test_sides_are_re_derived_from_the_data_not_copied_from_the_extractor():
    """The importer names the hands itself.

    A processed tree written by an extractor whose naming convention was wrong
    must be corrected on import rather than re-extracted -- 4,016 videos is
    hours of MediaPipe inference, and the raw evidence needed to re-decide is
    already in the file.
    """
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0009")
        os.makedirs(sign_dir)
        # The stored key "left" holds the hand at the SMALLER image x, which
        # geometry says is the signer's RIGHT. The names are therefore wrong
        # in the source file and must come out swapped.
        frames = []
        for i in range(6):
            frames.append(
                {
                    "frameIndex": i,
                    "timestampMs": i * 33.0,
                    "handDetected": True,
                    "landmarks": [{"x": 0.2, "y": 0.5, "z": 0.0} for _ in range(21)],
                    "hands": {
                        "left": {
                            "landmarks": [{"x": 0.2, "y": 0.5, "z": 0.0} for _ in range(21)],
                            "score": 0.9,
                            "rawLabel": "Right",
                        },
                        "right": {
                            "landmarks": [{"x": 0.8, "y": 0.5, "z": 0.0} for _ in range(21)],
                            "score": 0.9,
                            "rawLabel": "Left",
                        },
                    },
                }
            )
        with open(os.path.join(sign_dir, "v.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "signId": "0009",
                    "sourceVideo": "v.mp4",
                    "fps": 30.0,
                    "frameCount": len(frames),
                    "handTracking": {
                        "numHands": 2,
                        "slotNames": ["left", "right"],
                        "primaryHand": "left",
                        "slotSwapCount": 0,
                    },
                    "frames": frames,
                },
                f,
            )

        labels_path = write_labels_csv(tmp, [("9", "\u0645\u0631\u062d\u0628\u0627")])
        output_path = os.path.join(tmp, "out.json")
        assert run_importer(tmp, labels_path, output_path).returncode == 0

        with open(output_path, encoding="utf-8") as f:
            entry = json.load(f)[0]
        assert entry["handedness"]["sideSource"] == "geometry"
        # The hand at the smaller x now sits on the "right" key.
        assert entry["handFrames"][0]["right"][0]["x"] == pytest.approx(0.2)
        assert entry["handFrames"][0]["left"][0]["x"] == pytest.approx(0.8)
        # And the legacy primary-hand name was translated with it.
        assert entry["handedness"]["primaryHand"] == "right"


# ============================================================================
# TAKE SELECTION
# ============================================================================
# Regression guard for a real defect. The importer used to take the FIRST
# repetition alphabetically. On the real dataset that is frequently the worst
# one: sign 0001's eight takes extract as 0/13, 14/14, 15/15, 21/21, 24/24,
# 21/21, 21/21, 24/24 frames-with-both-hands, and alphabetical order picks the
# single take where the second hand was never detected -- so a two-handed sign
# was imported as one-handed. Two of the three signs in a 20-video sample were
# affected. select_best_take() picks by how completely a take captures the
# sign instead.

def _video_json(sign_id, frames_with_both, total_frames, name="v.json"):
    """A minimal but structurally real extractor output."""
    pts = [{"x": 0.5 + i * 0.001, "y": 0.5, "z": 0.0} for i in range(21)]
    frames = []
    for i in range(total_frames):
        both = i < frames_with_both
        frames.append(
            {
                "frameIndex": i,
                "timestampMs": i * 33.3,
                "handDetected": True,
                "landmarks": pts,
                "hands": {
                    "left": {"landmarks": pts, "score": 0.9, "rawLabel": "Left"} if both else None,
                    "right": {"landmarks": pts, "score": 0.9, "rawLabel": "Right"},
                },
            }
        )
    return {
        "signId": sign_id,
        "sourceVideo": name.replace(".json", ".mp4"),
        "fps": 30.0,
        "frameCount": total_frames,
        "handTracking": {
            "numHands": 2,
            "handednessConvention": "camera",
            "primaryHand": "right",
            "framesWithTwoHands": frames_with_both,
            "slotSwapCount": 0,
        },
        "frames": frames,
    }


def _write_takes(sign_dir, takes):
    os.makedirs(sign_dir, exist_ok=True)
    for name, both, total in takes:
        with open(os.path.join(sign_dir, name), "w", encoding="utf-8") as f:
            json.dump(_video_json(os.path.basename(sign_dir), both, total, name), f)
    return sorted(os.path.join(sign_dir, n) for n, _, _ in takes)


def test_take_selection_prefers_the_take_with_the_most_two_hand_frames():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "0001")
        # 'a_' sorts first and is the degenerate take -- the exact real-world shape.
        paths = _write_takes(d, [("a_first.json", 0, 13), ("b_second.json", 14, 14)])
        best = ik.select_best_take(paths)
        assert best is not None
        chosen, frames, hand_frames, stats, reason = best
        assert os.path.basename(chosen) == "b_second.json"
        assert stats["framesWithBoth"] == 14
        assert stats["twoHanded"] is True
        assert "most two-hand frames" in reason


def test_take_selection_does_not_penalise_a_genuinely_one_handed_sign():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "0353")
        # No take shows both hands -- a real one-handed sign. The longest
        # take should win, and it must stay marked one-handed.
        paths = _write_takes(d, [("a.json", 0, 12), ("b.json", 0, 29)])
        chosen, frames, hand_frames, stats, reason = ik.select_best_take(paths)
        assert os.path.basename(chosen) == "b.json"
        assert stats["framesWithBoth"] == 0
        assert stats["twoHanded"] is False
        assert "no take shows both hands" in reason


def test_take_selection_returns_none_when_nothing_is_parseable():
    with tempfile.TemporaryDirectory() as tmp:
        d = os.path.join(tmp, "0009")
        os.makedirs(d)
        with open(os.path.join(d, "broken.json"), "w", encoding="utf-8") as f:
            f.write("{not json")
        assert ik.select_best_take([os.path.join(d, "broken.json")]) is None


def test_importer_end_to_end_picks_the_two_handed_take():
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, "processed")
        _write_takes(os.path.join(root, "0001"), [("a_first.json", 0, 13), ("b_second.json", 14, 14)])
        labels = write_labels_csv(tmp, [("1", "صفر")])
        out = os.path.join(tmp, "registry.json")
        run_importer(root, labels, out)

        with open(out, encoding="utf-8") as f:
            entries = json.load(f)
        assert len(entries) == 1
        assert entries[0]["handedness"]["twoHanded"] is True
        assert entries[0]["handedness"]["framesWithBoth"] == 14
