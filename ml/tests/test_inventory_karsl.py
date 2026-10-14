"""
Tests for ml/training/inventory_karsl.py on SYNTHETIC trees only -- tiny
generated videos, no real KArSL data.

The inventory is what the whole re-import is planned from, so the things
tested here are the ones that would silently mislead that plan: a miscount,
a malformed file reported as fine, a duplicate missed, a filename guessed at
rather than reported as unparsed.
"""
import json
import os
import subprocess
import sys
import tempfile

import cv2
import numpy as np

TRAINING = os.path.join(os.path.dirname(__file__), "..", "training")
sys.path.insert(0, TRAINING)
import inventory_karsl as inv  # noqa: E402


def make_video(path: str, frames: int = 5, fps: float = 30.0, size=(64, 48), seed: int = 0):
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for i in range(frames):
        writer.write(np.full((size[1], size[0], 3), (i * 7 + seed) % 255, dtype=np.uint8))
    writer.release()


def build_tree(root: str):
    """Two well-formed signs plus every irregularity the inventory must catch."""
    os.makedirs(os.path.join(root, "0071"))
    os.makedirs(os.path.join(root, "0072"))
    os.makedirs(os.path.join(root, "notasign"))

    make_video(os.path.join(root, "0071", "03_01_0071_(01_12_16_15_52_41)_c.mp4"), seed=1)
    make_video(os.path.join(root, "0071", "03_01_0071_(01_12_16_15_52_44)_c.mp4"), seed=2)
    make_video(os.path.join(root, "0071", "03_01_0071_(20_11_16_16_16_41)_c.mp4"), seed=3)

    make_video(os.path.join(root, "0072", "03_01_0072_(01_12_16_15_53_41)_c.mp4"), seed=4)
    # A real duplicate: same bytes under a second name.
    with open(os.path.join(root, "0072", "03_01_0072_(01_12_16_15_53_41)_c.mp4"), "rb") as f:
        payload = f.read()
    with open(os.path.join(root, "0072", "copy_of_take.mp4"), "wb") as f:
        f.write(payload)
    # Not a video container at all.
    with open(os.path.join(root, "0072", "03_01_0072_(01_12_16_15_53_59)_c.mp4"), "wb") as f:
        f.write(b"not an mp4")
    with open(os.path.join(root, "0072", "notes.txt"), "w", encoding="utf-8") as f:
        f.write("hello")


def run(root: str, out: str, extra=()):
    result = subprocess.run(
        [sys.executable, os.path.join(TRAINING, "inventory_karsl.py"),
         "--input", root, "--out", out, *extra],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    with open(out, "r", encoding="utf-8") as f:
        return json.load(f)


class TestInventory:
    def test_counts_only_real_video_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "KArSL")
            build_tree(root)
            report = run(root, os.path.join(tmp, "inv.json"))
            # 3 in 0071, 3 in 0072 (the .txt is not a video), 0 in notasign.
            assert report["totalVideos"] == 6
            assert report["signs"]["0071"]["videos"] == 3
            assert report["signs"]["0072"]["videos"] == 3
            assert report["videosPerSign"]["max"] == 3
            assert report["signsWithNoVideos"] == ["notasign"]

    def test_reports_irregular_names_rather_than_normalising_them_away(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "KArSL")
            build_tree(root)
            report = run(root, os.path.join(tmp, "inv.json"))
            assert report["unexpectedFolderNames"] == ["notasign"]
            assert any(p.endswith("copy_of_take.mp4") for p in report["unparsedFilenames"])
            assert any(p.endswith("notes.txt") for p in report["nonVideoFiles"])

    def test_session_dates_are_recovered_per_sign(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "KArSL")
            build_tree(root)
            report = run(root, os.path.join(tmp, "inv.json"))
            assert report["signs"]["0071"]["sessionDates"] == {"01_12_16": 2, "20_11_16": 1}
            assert report["sessionsPerSign"]["max"] == 2

    def test_a_file_that_is_not_a_video_container_is_flagged_malformed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "KArSL")
            build_tree(root)
            report = run(root, os.path.join(tmp, "inv.json"))
            flagged = [b["path"] for b in report["malformedVideos"]]
            assert any(p.endswith("(01_12_16_15_53_59)_c.mp4") for p in flagged)
            # ...and the well-formed ones are NOT flagged.
            assert not any("15_52_41" in p for p in flagged)

    def test_duplicate_content_is_found_even_under_a_different_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "KArSL")
            build_tree(root)
            report = run(root, os.path.join(tmp, "inv.json"))
            groups = [sorted(os.path.basename(p) for p in g) for g in report["duplicateContentGroups"]]
            assert ["03_01_0072_(01_12_16_15_53_41)_c.mp4", "copy_of_take.mp4"] in groups

    def test_nothing_in_the_raw_tree_is_modified(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = os.path.join(tmp, "KArSL")
            build_tree(root)
            before = {
                os.path.join(dirpath, name): os.stat(os.path.join(dirpath, name)).st_mtime_ns
                for dirpath, _d, files in os.walk(root) for name in files
            }
            run(root, os.path.join(tmp, "inv.json"))
            after = {
                os.path.join(dirpath, name): os.stat(os.path.join(dirpath, name)).st_mtime_ns
                for dirpath, _d, files in os.walk(root) for name in files
            }
            assert before == after

    def test_canonicalization_matches_the_importer(self):
        # The join between "0071" on disk and SignID 71 in the labels file is
        # the thing that silently drops every sign when it goes wrong.
        assert inv.canonicalize_sign_id("0071") == "71"
        assert inv.canonicalize_sign_id("0502") == "502"
        assert inv.canonicalize_sign_id("notasign") == "notasign"
