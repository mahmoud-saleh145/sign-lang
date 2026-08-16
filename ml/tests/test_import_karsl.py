"""
Tests the KArSL importer's parsing logic against SYNTHETIC test fixtures
built to match its documented assumptions -- not real KArSL files (which
were never available in this environment; see docs/SIGN_LANGUAGE_DATASETS.md
and the honesty note at the top of ml/training/import_karsl.py).

These tests prove the tool works correctly on well-formed input and fails
safely (skips + reports, never fabricates) on malformed input. They do NOT
prove the tool works against real KArSL files, since none were available.
"""
import csv
import json
import os
import subprocess
import sys
import tempfile

IMPORTER_PATH = os.path.join(os.path.dirname(__file__), "..", "training", "import_karsl.py")


def make_synthetic_dataset(root: str):
    """Builds a tiny synthetic dataset matching import_karsl.py's assumed
    layout: one directory per sign ID, containing per-frame skeleton files."""
    sign_points = {"0001": (0.1, 0.2, 0.0), "0002": (0.5, 0.5, 0.0)}
    for sign_id, point in sign_points.items():
        sign_dir = os.path.join(root, sign_id)
        os.makedirs(sign_dir, exist_ok=True)
        # 3 "frames" per sign, each a simple text file of one point per line
        for i in range(3):
            with open(os.path.join(sign_dir, f"frame_{i:03d}.txt"), "w") as f:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

    labels_path = os.path.join(root, "labels.csv")
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["sign_id", "arabic_label"])
        writer.writerow(["0001", "\u0645\u0631\u062d\u0628\u0627"])  # مرحبا (synthetic test mapping)
        # 0002 deliberately has no label -> must be skipped, not fabricated

    return labels_path


def run_importer(input_dir: str, labels_path: str, output_path: str):
    return subprocess.run(
        [sys.executable, IMPORTER_PATH, "--input", input_dir, "--labels", labels_path, "--output", output_path],
        capture_output=True,
        text=True,
    )


def test_imports_well_formed_synthetic_data_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        labels_path = make_synthetic_dataset(tmp)
        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)

        assert result.returncode == 0, result.stdout + result.stderr
        with open(output_path) as f:
            entries = json.load(f)

        # Only 0001 has a label; 0002 must be skipped, never fabricated.
        assert len(entries) == 1
        assert entries[0]["text"] == "\u0645\u0631\u062d\u0628\u0627"
        assert len(entries[0]["landmarkSequence"]) == 3  # 3 synthetic frames
        assert entries[0]["provenance"]["datasetName"] == "KArSL-502"
        assert "license" in entries[0]["provenance"]


def test_sign_without_a_label_is_skipped_not_fabricated():
    with tempfile.TemporaryDirectory() as tmp:
        labels_path = make_synthetic_dataset(tmp)
        output_path = os.path.join(tmp, "out.json")
        run_importer(tmp, labels_path, output_path)

        with open(output_path) as f:
            entries = json.load(f)
        texts = [e["text"] for e in entries]
        assert "0002" not in texts  # never given a placeholder label


def test_fails_loudly_on_empty_input_directory():
    with tempfile.TemporaryDirectory() as tmp:
        empty_input = os.path.join(tmp, "empty")
        os.makedirs(empty_input)
        labels_path = os.path.join(tmp, "labels.csv")
        with open(labels_path, "w") as f:
            f.write("sign_id,arabic_label\n")
        output_path = os.path.join(tmp, "out.json")

        result = run_importer(empty_input, labels_path, output_path)
        assert result.returncode != 0
        assert "no subdirectories found" in result.stdout


def test_fails_loudly_when_no_signs_could_be_imported():
    with tempfile.TemporaryDirectory() as tmp:
        # Directories exist but have no skeleton files and no labels at all.
        os.makedirs(os.path.join(tmp, "0001"))
        labels_path = os.path.join(tmp, "labels.csv")
        with open(labels_path, "w") as f:
            f.write("sign_id,arabic_label\n")  # no rows
        output_path = os.path.join(tmp, "out.json")

        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode != 0
        assert "No signs were imported" in result.stdout


def make_processed_karsl_video_json(sign_id: str, source_video: str, points, hand_detected_flags):
    """Builds one processed-karsl-format per-video JSON dict (the real
    output shape of ml/training/extract_karsl_skeletons.py), for testing
    the PREFERRED import path -- not real KArSL data."""
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

        labels_path = os.path.join(tmp, "labels.csv")
        with open(labels_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sign_id", "arabic_label"])
            writer.writerow(["0001", "\u0645\u0631\u062d\u0628\u0627"])

        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode == 0, result.stdout + result.stderr

        with open(output_path) as f:
            entries = json.load(f)
        assert len(entries) == 1
        assert len(entries[0]["landmarkSequence"]) == 3  # all 3 frames had a detected hand


def test_frames_with_no_detected_hand_are_dropped_not_fabricated():
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0001")
        os.makedirs(sign_dir)
        # 5 frames total, only 2 had a detected hand
        video_json = make_processed_karsl_video_json(
            "0001", "video1.mp4", (0.1, 0.2), [True, False, False, True, False]
        )
        with open(os.path.join(sign_dir, "video1.json"), "w", encoding="utf-8") as f:
            json.dump(video_json, f)

        labels_path = os.path.join(tmp, "labels.csv")
        with open(labels_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sign_id", "arabic_label"])
            writer.writerow(["0001", "\u0645\u0631\u062d\u0628\u0627"])

        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode == 0
        assert "dropped 3/5" in result.stdout

        with open(output_path) as f:
            entries = json.load(f)
        assert len(entries[0]["landmarkSequence"]) == 2  # only the 2 real detected frames


def test_multiple_videos_per_sign_uses_first_and_reports_others_unused():
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0001")
        os.makedirs(sign_dir)
        for name in ["video1.mp4", "video2.mp4"]:
            video_json = make_processed_karsl_video_json("0001", name, (0.1, 0.2), [True, True])
            with open(os.path.join(sign_dir, name.replace(".mp4", ".json")), "w", encoding="utf-8") as f:
                json.dump(video_json, f)

        labels_path = os.path.join(tmp, "labels.csv")
        with open(labels_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sign_id", "arabic_label"])
            writer.writerow(["0001", "\u0645\u0631\u062d\u0628\u0627"])

        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode == 0
        assert "other repetition(s) available but not used" in result.stdout

        with open(output_path) as f:
            entries = json.load(f)
        assert len(entries) == 1  # exactly one entry, not merged/duplicated


def test_zero_padded_sign_id_from_processed_karsl_stays_a_string():
    with tempfile.TemporaryDirectory() as tmp:
        sign_dir = os.path.join(tmp, "0007")
        os.makedirs(sign_dir)
        video_json = make_processed_karsl_video_json("0007", "video1.mp4", (0.1, 0.2), [True])
        with open(os.path.join(sign_dir, "video1.json"), "w", encoding="utf-8") as f:
            json.dump(video_json, f)

        labels_path = os.path.join(tmp, "labels.csv")
        with open(labels_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["sign_id", "arabic_label"])
            writer.writerow(["0007", "\u0634\u0643\u0631\u0627"])  # شكرا

        output_path = os.path.join(tmp, "out.json")
        result = run_importer(tmp, labels_path, output_path)
        assert result.returncode == 0
        with open(output_path) as f:
            entries = json.load(f)
        assert entries[0]["text"] == "\u0634\u0643\u0631\u0627"