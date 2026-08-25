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
        with open(output_path) as f:
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
        with open(output_path) as f:
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
        with open(output_path) as f:
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
        with open(output_path) as f:
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
        with open(output_path) as f:
            entries = json.load(f)
        assert len(entries) == 1
        assert entries[0]["sourceSignId"] == "0001"
        assert "9999 (no label found" in result.stdout


def test_sign_without_a_label_is_skipped_not_fabricated():
    with tempfile.TemporaryDirectory() as tmp:
        labels_path = make_synthetic_dataset(tmp)
        output_path = os.path.join(tmp, "out.json")
        run_importer(tmp, labels_path, output_path)

        with open(output_path) as f:
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

        with open(output_path) as f:
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

        with open(output_path) as f:
            entries = json.load(f)
        assert len(entries[0]["landmarkSequence"]) == 2


def test_multiple_videos_per_sign_uses_first_and_reports_others_unused():
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
        assert "other repetition(s) available but not used" in result.stdout

        with open(output_path) as f:
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
        with open(output_path) as f:
            entries = json.load(f)
        assert entries[0]["text"] == "\u0634\u0643\u0631\u0627"
        assert entries[0]["sourceSignId"] == "0007"