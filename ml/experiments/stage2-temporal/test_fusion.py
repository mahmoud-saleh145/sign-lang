"""
Tests for the Stage 2.5 fusion logic.

These pin the DECISION RULES, not the measured accuracy. Each case is a tiny
hand-built scenario where the right answer is obvious by construction, so a
regression in a strategy shows up as a wrong decision rather than as a number
that merely drifted.

Run:  python -m pytest ml/experiments/stage2-temporal/test_fusion.py -q
"""
import numpy as np
import pytest

import fuse


def probs_for(rows, classes=4):
    """rows: list of (predicted_class, confidence). Remainder spread evenly."""
    p = np.zeros((len(rows), classes), dtype=np.float32)
    for i, (c, conf) in enumerate(rows):
        p[i] = (1.0 - conf) / (classes - 1)
        p[i, c] = conf
    return p


def dtw_for(preds, accepted, ratios=None):
    n = len(preds)
    return {
        "pred": np.array(preds, dtype=np.int64),
        "accepted": np.array(accepted, dtype=bool),
        "ratio": np.array(ratios if ratios is not None else [0.5] * n, dtype=float),
        "distance": np.full(n, 1.0),
        "missing": 0,
    }


class TestStrategyC:
    """DTW if accepted, else TCN."""

    def test_uses_dtw_when_it_accepted(self):
        p = probs_for([(0, 0.9)])
        d = dtw_for([1], [True])
        assert fuse.strat_dtw_then_tcn(p, d).predicted[0] == 1

    def test_falls_back_to_tcn_when_dtw_rejected(self):
        p = probs_for([(0, 0.9)])
        d = dtw_for([1], [False])
        assert fuse.strat_dtw_then_tcn(p, d).predicted[0] == 0

    def test_rescues_exactly_the_only_dtw_cases_dtw_accepted(self):
        # The headroom claim made concrete: C can only capture an
        # "only DTW correct" query when DTW's own gate accepted it.
        y = np.array([0, 0])
        p = probs_for([(1, 0.9), (1, 0.9)])      # TCN wrong on both
        d = dtw_for([0, 0], [True, False])        # DTW right on both, accepts one
        pred = fuse.strat_dtw_then_tcn(p, d).predicted
        assert (pred == y).tolist() == [True, False]

    def test_accepts_everything(self):
        p = probs_for([(0, 0.4)])
        assert fuse.strat_dtw_then_tcn(p, dtw_for([0], [False])).accepted.all()


class TestStrategyD:
    """Agreement gate: conservative, cannot change the prediction."""

    def test_accepts_only_on_agreement_with_an_accepted_dtw(self):
        p = probs_for([(0, 0.9), (0, 0.9), (0, 0.9)])
        d = dtw_for([0, 1, 0], [True, True, False])
        assert fuse.strat_agreement(p, d).accepted.tolist() == [True, False, False]

    def test_prediction_is_always_the_tcn(self):
        p = probs_for([(0, 0.9)])
        d = dtw_for([1], [True])
        assert fuse.strat_agreement(p, d).predicted[0] == 0

    def test_confidence_escape_hatch_widens_acceptance(self):
        p = probs_for([(0, 0.95)])
        d = dtw_for([1], [True])                  # disagreement
        assert not fuse.strat_agreement(p, d).accepted[0]
        assert fuse.strat_agreement(p, d, thresh=0.9).accepted[0]


class TestStrategyE:
    """Union reranking."""

    def test_weight_one_is_exactly_the_tcn(self):
        p = probs_for([(0, 0.6), (2, 0.8)])
        d = dtw_for([1, 1], [True, True])
        assert fuse.strat_union_rerank(p, d, 1.0).predicted.tolist() == [0, 2]

    def test_weight_zero_is_exactly_dtw(self):
        p = probs_for([(0, 0.6)])
        d = dtw_for([1], [True])
        assert fuse.strat_union_rerank(p, d, 0.0).predicted[0] == 1

    def test_dtw_candidate_can_win_at_an_intermediate_weight(self):
        # TCN only weakly prefers 0; DTW's vote carries 1 over the line.
        p = probs_for([(0, 0.30)], classes=4)
        d = dtw_for([1], [True])
        assert fuse.strat_union_rerank(p, d, 0.5).predicted[0] == 1

    def test_a_confident_tcn_survives_the_dtw_vote(self):
        p = probs_for([(0, 0.97)], classes=4)
        d = dtw_for([1], [True])
        assert fuse.strat_union_rerank(p, d, 0.7).predicted[0] == 0

    def test_union_cannot_reach_a_class_neither_model_proposed(self):
        p = probs_for([(0, 0.4)], classes=10)
        d = dtw_for([1], [True])
        pred = fuse.strat_union_rerank(p, d, 0.5).predicted[0]
        assert pred in set(np.argsort(-p[0])[:5].tolist()) | {1}


class TestStrategyF:
    def test_accepts_on_tcn_confidence_alone(self):
        p = probs_for([(0, 0.95)])
        d = dtw_for([0], [False], [0.99])
        assert fuse.strat_confidence_fusion(p, d, 0.9, 0.3).accepted[0]

    def test_accepts_on_a_comfortable_dtw_margin_alone(self):
        p = probs_for([(0, 0.2)])
        d = dtw_for([0], [True], [0.4])           # margin 0.6
        assert fuse.strat_confidence_fusion(p, d, 0.9, 0.3).accepted[0]

    def test_abstains_when_neither_is_confident(self):
        p = probs_for([(0, 0.2)])
        d = dtw_for([0], [True], [0.95])          # margin 0.05
        assert not fuse.strat_confidence_fusion(p, d, 0.9, 0.3).accepted[0]

    def test_an_unaccepted_dtw_cannot_carry_acceptance(self):
        p = probs_for([(0, 0.1)])
        d = dtw_for([0], [False], [0.1])          # huge margin, gate said no
        assert not fuse.strat_confidence_fusion(p, d, 0.9, 0.3).accepted[0]


class TestScoring:
    def test_top1_counts_all_queries_not_just_accepted(self):
        # A strategy must not be able to buy accuracy by abstaining.
        y = np.array([0, 1])
        o = fuse.Outcome("t", np.array([0, 0]), np.array([True, False]), False)
        assert fuse.score(o, y)["top1"] == pytest.approx(0.5)

    def test_precision_is_measured_among_accepted_only(self):
        y = np.array([0, 1])
        o = fuse.Outcome("t", np.array([0, 0]), np.array([True, False]), False)
        s = fuse.score(o, y)
        assert s["precisionAmongAccepted"] == pytest.approx(1.0)
        assert s["falseAccepts"] == 0
        assert s["rejectionRate"] == pytest.approx(0.5)

    def test_false_accepts_are_counted(self):
        y = np.array([0, 1])
        o = fuse.Outcome("t", np.array([0, 0]), np.array([True, True]), False)
        s = fuse.score(o, y)
        assert s["falseAccepts"] == 1
        assert s["precisionAmongAccepted"] == pytest.approx(0.5)


class TestTuning:
    def test_confidence_threshold_comes_from_validation_labels_only(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 4, 400)
        p = np.full((400, 4), 0.1, dtype=np.float32)
        for i in range(400):
            # High confidence exactly where the label is right.
            p[i, y[i] if i % 2 == 0 else (y[i] + 1) % 4] = 0.9 if i % 2 == 0 else 0.5
        p /= p.sum(1, keepdims=True)
        t = fuse.tune_confidence(p, y, target_precision=0.95, min_accepted=10)
        assert t is not None
        sel = p.max(1) >= t
        assert (p.argmax(1)[sel] == y[sel]).mean() >= 0.95

    def test_returns_none_when_no_threshold_reaches_the_target(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 4, 200)
        p = rng.random((200, 4)).astype(np.float32)
        p /= p.sum(1, keepdims=True)
        assert fuse.tune_confidence(p, y, target_precision=0.999, min_accepted=10) is None
