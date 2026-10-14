"""
Tests for the provenance / split-feasibility part of
ml/datasets/audit_processed_karsl.py, on synthetic processed trees.

Why this is worth testing: the session-held-out split is the whole basis of
the leakage-safe evaluation, and "can this sign be split?" is not the same
question as "does this sign have two sessions?". A session whose every take
lost its hands to detection failure still appears in the filenames but
cannot be evaluated on. Getting that wrong would silently produce a split
where some classes are scored against nothing.
"""
import json
import os
import subprocess
import sys
import tempfile

DATASETS = os.path.join(os.path.dirname(__file__), "..", "datasets")
sys.path.insert(0, DATASETS)
import audit_processed_karsl as audit  # noqa: E402


def _frames(n: int, two_hands: bool, hands_present: bool):
    out = []
    for i in range(n):
        def lm(off):
            return [{"x": 0.3 + off + i * 0.001, "y": 0.4 + j * 0.01, "z": 0.0} for j in range(21)]

        if hands_present:
            payload = {
                "left": {"landmarks": lm(0.2), "score": 0.9, "rawLabel": "Left"} if two_hands else None,
                "right": {"landmarks": lm(0.0), "score": 0.9, "rawLabel": "Right"},
            }
        else:
            payload = {"left": None, "right": None}
        out.append({
            "frameIndex": i,
            "timestampMs": i * 33.3,
            "handDetected": hands_present,
            "landmarks": lm(0.0) if hands_present else None,
            "hands": payload,
        })
    return out


def write_take(root, sign, stem, n=12, two_hands=True, hands_present=True, provenance=True):
    os.makedirs(os.path.join(root, sign), exist_ok=True)
    date = stem.split("(")[1][:8]
    doc = {
        "signId": sign,
        "sourceVideo": stem + ".mp4",
        "sourceVideoStem": stem,
        "provenance": {
            "field1": "03", "field2": "01", "signIdInName": sign,
            "sessionDate": date, "recordedAt": date + "_12_00_00", "suffix": "c",
        } if provenance else None,
        "fps": 30.0,
        "frameCount": n,
        "handTracking": {
            "numHands": 2, "slotNames": ["right", "left"], "sideSource": "geometry",
            "sideConfidence": 1.0, "slotDetectionCounts": [n, n if two_hands else 0],
            "primaryHand": "right", "framesWithTwoHands": n if two_hands else 0,
            "framesWithAnyHand": n if hands_present else 0, "slotSwapCount": 0,
        },
        "frames": _frames(n, two_hands, hands_present),
    }
    with open(os.path.join(root, sign, stem + ".json"), "w", encoding="utf-8") as f:
        json.dump(doc, f)


def run(root):
    r = subprocess.run(
        [sys.executable, os.path.join(DATASETS, "audit_processed_karsl.py"), "--input", root],
        capture_output=True, text=True,
    )
    return r.stdout


class TestSplitFeasibility:
    def test_a_sign_with_two_usable_sessions_is_splittable(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                write_take(tmp, "0071", f"03_01_0071_(01_12_16_15_5{i}_41)_c")
                write_take(tmp, "0071", f"03_01_0071_(20_11_16_16_1{i}_41)_c")
            out = run(tmp)
            assert "signs with >=2 usable sessions: 1" in out
            assert "signs that CANNOT be split:     0" in out

    def test_a_session_whose_takes_all_lost_their_hands_does_not_count(self):
        # The failure this exists to prevent: sign 0105 in the real data has
        # two session dates, but every take in one of them detected no hand.
        # Counting dates would call it splittable; counting USABLE sessions
        # correctly does not.
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                write_take(tmp, "0105", f"03_01_0105_(04_12_16_18_2{i}_23)_c")
                write_take(tmp, "0105", f"03_01_0105_(24_11_16_18_2{i}_23)_c", hands_present=False)
            out = run(tmp)
            assert "session dates per sign:      {2: 1}" in out
            assert "signs with >=2 usable sessions: 0" in out
            assert "'0105'" in out

    def test_a_single_session_sign_is_reported_as_unsplittable(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                write_take(tmp, "0348", f"03_01_0348_(15_08_18_11_0{i}_00)_c")
            out = run(tmp)
            assert "signs that CANNOT be split:     1" in out
            assert "'0348'" in out

    def test_reports_the_smaller_session_size_that_bounds_the_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(5):
                write_take(tmp, "0071", f"03_01_0071_(01_12_16_15_5{i}_41)_c")
            for i in range(2):
                write_take(tmp, "0071", f"03_01_0071_(20_11_16_16_1{i}_41)_c")
            out = run(tmp)
            assert "smaller session's usable takes: min 2" in out


class TestProvenanceCoverage:
    def test_counts_files_carrying_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_take(tmp, "0071", "03_01_0071_(01_12_16_15_52_41)_c")
            write_take(tmp, "0071", "03_01_0071_(20_11_16_16_16_41)_c", provenance=False)
            out = run(tmp)
            assert "files with parsed provenance:1 / 2" in out
            assert "files with sourceVideoStem:  2 / 2" in out

    def test_a_file_without_provenance_is_not_credited_to_any_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_take(tmp, "0071", "03_01_0071_(01_12_16_15_52_41)_c")
            write_take(tmp, "0071", "03_01_0071_(20_11_16_16_16_41)_c", provenance=False)
            out = run(tmp)
            # One session is visible, so the sign is not splittable on evidence.
            assert "signs with >=2 usable sessions: 0" in out


class TestExistingChecksStillHold:
    def test_one_handed_and_two_handed_takes_are_counted_separately(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_take(tmp, "0071", "03_01_0071_(01_12_16_15_52_41)_c", two_hands=True)
            write_take(tmp, "0071", "03_01_0071_(20_11_16_16_16_41)_c", two_hands=False)
            out = run(tmp)
            assert "Two-handed (both >= 50% frames): 1" in out
            assert "One-handed (never both hands):   1" in out

    def test_a_handless_take_is_still_reported_as_a_blocking_issue(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_take(tmp, "0071", "03_01_0071_(01_12_16_15_52_41)_c", hands_present=False)
            out = run(tmp)
            assert "no hand detected in any frame" in out
            assert "BLOCKING ISSUES:" in out

    def test_impossible_jump_threshold_is_unchanged(self):
        # The audit's whole point is catching identity switches; if this
        # constant drifts, the check quietly stops working.
        assert audit.IMPOSSIBLE_JUMP == 0.15


class TestDuplicateDetection:
    """The duplicate check compares a hash of the primary hand's wrist track.
    Takes with no detected hand contribute nothing to that hash, so they all
    used to collide on sha1("") and get reported as one giant duplicate group
    -- 28 unrelated recordings in the real v2 tree. They are different videos;
    the emptiness is the only thing they share."""

    def test_handless_takes_are_not_reported_as_duplicates_of_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(3):
                write_take(tmp, "0105", f"03_01_0105_(24_11_16_18_2{i}_23)_c",
                           n=10 + i, hands_present=False)
            out = run(tmp)
            assert "Duplicate sequences:           0" in out
            # They are still reported, by the check that actually applies.
            assert "no hand detected in any frame" in out

    def test_genuinely_identical_tracks_are_still_caught(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_take(tmp, "0071", "03_01_0071_(01_12_16_15_52_41)_c", n=12)
            write_take(tmp, "0071", "03_01_0071_(20_11_16_16_16_41)_c", n=12)
            out = run(tmp)
            # Same synthetic landmark generator, same length -> same track.
            assert "Duplicate sequences:           1" in out

    def test_a_handless_take_gets_no_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_take(tmp, "0105", "03_01_0105_(24_11_16_18_21_23)_c", hands_present=False)
            path = os.path.join(tmp, "0105", "03_01_0105_(24_11_16_18_21_23)_c.json")
            r = audit.audit_file(path, "0105", 8)
            assert r["fingerprint"] is None


class TestExclusionsArtifact:
    """The exclusion list is what the leakage-safe split consumes, so its
    contents ARE the definition of the evaluation set. It has to name the
    right takes for the right reasons, and it must never be mistaken for
    permission to delete data."""

    def _run_with_exclusions(self, root, out):
        subprocess.run(
            [sys.executable, os.path.join(DATASETS, "audit_processed_karsl.py"),
             "--input", root, "--exclusions-out", out],
            capture_output=True, text=True,
        )
        with open(out, encoding="utf-8") as f:
            return json.load(f)

    def test_names_handless_takes_with_their_reason(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_take(tmp, "0105", "03_01_0105_(04_12_16_18_20_23)_c")
            write_take(tmp, "0105", "03_01_0105_(24_11_16_18_21_23)_c", hands_present=False)
            e = self._run_with_exclusions(tmp, os.path.join(tmp, "e.json"))
            assert e["excludedTakes"]["0105"]["03_01_0105_(24_11_16_18_21_23)_c"] == "no-hand"
            assert "03_01_0105_(04_12_16_18_20_23)_c" not in e["excludedTakes"]["0105"]
            assert e["counts"]["noHand"] == 1
            assert e["counts"]["identitySwitch"] == 0

    def test_names_signs_that_cannot_be_split_by_session(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(2):
                write_take(tmp, "0348", f"03_01_0348_(15_08_18_11_0{i}_00)_c")
                write_take(tmp, "0071", f"03_01_0071_(01_12_16_15_5{i}_41)_c")
                write_take(tmp, "0071", f"03_01_0071_(20_11_16_16_1{i}_41)_c")
            e = self._run_with_exclusions(tmp, os.path.join(tmp, "e.json"))
            assert e["signsWithoutSessionSplit"] == ["0348"]

    def test_records_the_threshold_the_decisions_were_made_under(self):
        # Without this, a later change to IMPOSSIBLE_JUMP would silently
        # invalidate an exclusion file that still looks current.
        with tempfile.TemporaryDirectory() as tmp:
            write_take(tmp, "0071", "03_01_0071_(01_12_16_15_52_41)_c")
            e = self._run_with_exclusions(tmp, os.path.join(tmp, "e.json"))
            assert e["impossibleJumpThreshold"] == audit.IMPOSSIBLE_JUMP
            assert e["version"] == 1

    def test_a_clean_tree_produces_an_empty_exclusion_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(2):
                write_take(tmp, "0071", f"03_01_0071_(01_12_16_15_5{i}_41)_c", n=10 + i)
                write_take(tmp, "0071", f"03_01_0071_(20_11_16_16_1{i}_41)_c", n=14 + i)
            e = self._run_with_exclusions(tmp, os.path.join(tmp, "e.json"))
            assert e["excludedTakes"] == {}
            assert e["signsWithoutSessionSplit"] == []
            assert e["counts"]["takesExcluded"] == 0

    def test_writing_exclusions_does_not_touch_the_processed_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_take(tmp, "0105", "03_01_0105_(24_11_16_18_21_23)_c", hands_present=False)
            before = {
                os.path.join(d, n): os.stat(os.path.join(d, n)).st_mtime_ns
                for d, _s, fs in os.walk(tmp) for n in fs
            }
            self._run_with_exclusions(tmp, os.path.join(tmp, "e.json"))
            after = {
                os.path.join(d, n): os.stat(os.path.join(d, n)).st_mtime_ns
                for d, _s, fs in os.walk(tmp) for n in fs if not n.endswith("e.json")
            }
            for path, mtime in before.items():
                assert after.get(path) == mtime
