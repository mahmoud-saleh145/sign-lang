"""Tests for the canonical left/right assignment (ml/training/handedness.py).

These pin down the rule that replaced two things that were never hand
identities: MediaPipe's slot ORDER, and "whichever slot was detected in more
frames". See the module docstring in handedness.py for the measurements.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

import handedness as h  # noqa: E402
from handedness import assign_sides, geometric_vote, label_vote  # noqa: E402


def hand(x, y=0.5):
    """21 points offset from (x, y). Only the wrist (index 0) matters here."""
    return [{"x": x + i * 0.001, "y": y + i * 0.001, "z": 0.0} for i in range(21)]


def track(xs):
    return [None if x is None else hand(x) for x in xs]


class TestGeometricVote:
    def test_counts_only_frames_where_both_hands_are_present(self):
        a = track([0.2, 0.2, None, 0.2])
        b = track([0.8, None, 0.8, 0.8])
        a_smaller, compared = geometric_vote(a, b)
        assert compared == 2
        assert a_smaller == 2

    def test_reports_no_comparison_when_the_hands_never_coincide(self):
        a = track([0.2, 0.2, None, None])
        b = track([None, None, 0.8, 0.8])
        assert geometric_vote(a, b) == (0, 2 - 2)


class TestLabelVote:
    def test_majority_wins_not_the_first_frame(self):
        # The old extractor seeded slot names from the FIRST frame's label.
        # Measured, that frame disagrees with the track majority in 8.1% of
        # tracks, which is why this is a vote.
        label, agreement = label_vote(["Left", "Right", "Right", "Right"])
        assert label == "Right"
        assert agreement == pytest.approx(0.75)

    def test_ignores_missing_labels(self):
        label, agreement = label_vote([None, "Left", None, "Left"])
        assert label == "Left"
        assert agreement == pytest.approx(1.0)

    def test_no_labels_at_all(self):
        assert label_vote([None, None]) == (None, 0.0)


class TestAssignSides:
    def test_geometry_decides_when_it_is_available_and_consistent(self):
        a = track([0.2] * 8)
        b = track([0.8] * 8)
        result = assign_sides(a, b, [None] * 8, [None] * 8)
        assert result.source == "geometry"
        # A signer faces the camera in an un-mirrored recording, so their RIGHT
        # hand is the one at the smaller image x.
        assert list(result.slot_sides) == ["right", "left"]
        assert result.confidence == pytest.approx(1.0)

    def test_geometry_overrules_a_disagreeing_label(self):
        a = track([0.2] * 8)
        b = track([0.8] * 8)
        result = assign_sides(a, b, ["Left"] * 8, ["Right"] * 8)
        assert result.source == "geometry"
        assert list(result.slot_sides) == ["right", "left"]

    def test_defers_to_the_label_when_the_hands_cross_too_often(self):
        # Alternating: geometry has no majority, so it must not decide.
        a = track([0.2, 0.8] * 5)
        b = track([0.8, 0.2] * 5)
        result = assign_sides(a, b, ["Right"] * 10, ["Left"] * 10)
        assert result.source == "handedness-label"
        assert list(result.slot_sides) == ["right", "left"]

    def test_uses_the_label_when_only_one_hand_is_ever_seen(self):
        a = track([0.4] * 6)
        b = [None] * 6
        result = assign_sides(a, b, ["Left"] * 6, [None] * 6)
        assert result.source == "handedness-label"
        assert result.slot_sides[0] == "left"
        assert result.slot_sides[1] == "right"

    def test_label_is_NOT_inverted_by_default(self):
        # The measured direction: MediaPipe's "Right" is the signer's right on
        # this (non-mirrored) source. An earlier extractor inverted it and
        # swapped the hands of every sign in the dataset.
        result = assign_sides(track([0.4] * 6), [None] * 6, ["Right"] * 6, [None] * 6)
        assert result.slot_sides[0] == "right"

    def test_inversion_is_available_for_a_genuinely_mirrored_source(self):
        original = h.INVERT_LABEL
        try:
            h.INVERT_LABEL = True
            result = assign_sides(track([0.4] * 6), [None] * 6, ["Right"] * 6, [None] * 6)
            assert result.slot_sides[0] == "left"
        finally:
            h.INVERT_LABEL = original

    def test_reports_unknown_rather_than_guessing(self):
        # One hand, no label, never two at once: nothing can name it.
        result = assign_sides(track([0.4] * 6), [None] * 6, [None] * 6, [None] * 6)
        assert result.source == "unknown"
        assert list(result.slot_sides) == ["unknown", "unknown"]
        assert result.confidence == 0.0

    def test_the_two_slots_never_get_the_same_side(self):
        # Both slots labelled the same way is a real MediaPipe output; naming
        # both hands "right" would be a contradiction, not a guess.
        result = assign_sides(track([0.4] * 6), track([0.6] * 6), ["Right"] * 6, ["Right"] * 6)
        assert set(result.slot_sides) == {"left", "right"}

    def test_serialises_the_evidence_not_just_the_answer(self):
        result = assign_sides(track([0.2] * 8), track([0.8] * 8), ["Right"] * 8, ["Left"] * 8)
        as_dict = result.as_dict()
        assert as_dict["sideSource"] == "geometry"
        assert as_dict["geometryFrames"] == 8
        assert as_dict["geometryAgreement"] == pytest.approx(1.0)
        assert as_dict["sideConfidence"] == pytest.approx(1.0)

    def test_below_the_frame_floor_geometry_does_not_decide(self):
        n = h.GEOMETRY_MIN_FRAMES - 1
        result = assign_sides(track([0.2] * n), track([0.8] * n), ["Left"] * n, ["Right"] * n)
        assert result.source != "geometry"
