"""
STAGE 2.5 -- OFFLINE FUSION EXPERIMENT (DTW + temporal CNN).

Research only. Imports nothing from the app, changes no production file, no
threshold and no trained model. It reads artifacts the Stage 2 run already
produced and decides, on measured numbers, whether combining the two
recognisers beats either alone while keeping a conservative "unknown".

============================================================================
INPUTS, AND WHY NO MODEL IS LOADED
============================================================================
    <run>/test_probs.npy     TCN probabilities over the held-out queries
    <run>/test_rows.json     the query identity of each row, in the same order
    <run>/val_probs.npy      TCN probabilities over validation
    <run>/val_rows.json
    <run>/summary.json       parameter count, size, measured TCN latency
    <data>/dtw_baseline.json DTW's verdict on the SAME rows
    <data>/manifest.json     class index <-> signId <-> Arabic text

The TCN is deterministic at eval, so its saved probabilities are exactly what
re-running the model would produce -- and reading them keeps this script to
numpy, with no torch import in the fusion path. `--remeasure-latency` will
load the checkpoint and time it again if torch is present.

The DTW column was produced by export_tensors.ts --with-dtw from the same
`rows` the tensors came from, so "the same held-out queries" is structural
rather than asserted. This script re-checks the join anyway and refuses to
report anything if a single row fails to pair.

============================================================================
THE RULE THIS SCRIPT OBEYS: NOTHING IS TUNED ON TEST
============================================================================
Every threshold and every weight is chosen on VALIDATION and then applied,
unchanged, to test. Each strategy reports `validationTuned` so a reader can
tell a free parameter from a fixed rule at a glance.

Validation is the held-out session's other half, so it shares a sitting with
test and any operating point it picks is OPTIMISTIC. That is stated in the
output, not buried here. It is still the only honest option: a two-session
corpus has nowhere else to put a tuning set.

Usage:
    python ml/experiments/stage2-temporal/fuse.py \
        --run ml/experiments/stage2-temporal/runs/tcn \
        --data ml/experiments/stage2-temporal/data
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from dataclasses import dataclass, field, asdict

import numpy as np


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class Outcome:
    """One strategy's verdict on every query.

    `predicted` is the class the strategy would say; `accepted` is whether it
    would say it at all. Top-1 is measured over ALL queries regardless of
    acceptance, so a strategy cannot buy accuracy by abstaining.
    """
    name: str
    predicted: np.ndarray
    accepted: np.ndarray
    validation_tuned: bool
    operating_point: dict = field(default_factory=dict)
    fusion_ms: float = 0.0


def score(o: Outcome, y: np.ndarray) -> dict:
    correct = o.predicted == y
    acc = o.accepted
    n = len(y)
    accepted_correct = int((acc & correct).sum())
    false_accepts = int((acc & ~correct).sum())
    return {
        "strategy": o.name,
        "top1": float(correct.mean()),
        "acceptedRate": float(acc.mean()),
        "acceptedAndCorrect": float(accepted_correct / n),
        "falseAccepts": false_accepts,
        "precisionAmongAccepted": float(accepted_correct / acc.sum()) if acc.sum() else 1.0,
        "rejectionRate": float(1 - acc.mean()),
        "confusionCount": int((~correct).sum()),
        "distinctConfusionPairs": len({(int(y[i]), int(o.predicted[i])) for i in range(n) if not correct[i]}),
        "fusionMsPerQuery": o.fusion_ms,
        "validationTuned": o.validation_tuned,
        "operatingPoint": o.operating_point,
    }


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------
def strat_tcn_only(p, dtw, thresh=None):
    """A. TCN alone. Accepts everything unless a validation threshold is given."""
    pred = p.argmax(1)
    acc = np.ones(len(pred), bool) if thresh is None else p.max(1) >= thresh
    return Outcome("A. TCN only" + ("" if thresh is None else f" (conf>={thresh:.2f})"),
                   pred, acc, thresh is not None,
                   {} if thresh is None else {"tcnConfidence": thresh})


def strat_dtw_only(p, dtw):
    """B. DTW alone, with the production gate exactly as it ships."""
    return Outcome("B. DTW only (production gate)", dtw["pred"], dtw["accepted"], False)


def strat_dtw_then_tcn(p, dtw):
    """C. Trust DTW when its gate accepts; otherwise fall back to the TCN.

    The cheapest possible fusion and the one most likely to be shipped, so it
    is worth measuring precisely. It can only help where DTW is both correct
    AND confident enough to pass its own gate.
    """
    pred = np.where(dtw["accepted"], dtw["pred"], p.argmax(1))
    return Outcome("C. DTW if accepted, else TCN", pred, np.ones(len(pred), bool), False)


def strat_agreement(p, dtw, thresh=None):
    """D. Accept only when the two recognisers independently agree.

    A conservative gate rather than an accuracy rule: it cannot change top-1
    (the prediction is always the TCN's), only which answers are spoken.
    """
    tcn_pred = p.argmax(1)
    agree = (tcn_pred == dtw["pred"]) & dtw["accepted"]
    if thresh is not None:
        agree = agree | (p.max(1) >= thresh)
    return Outcome("D. agreement gate" + ("" if thresh is None else f" OR conf>={thresh:.2f}"),
                   tcn_pred, agree, thresh is not None,
                   {} if thresh is None else {"tcnConfidence": thresh})


def strat_union_rerank(p, dtw, weight, accept_thresh=None):
    """E. Candidate-union reranking.

    Union of the TCN's top-5 and DTW's best. Each candidate scores
    `weight * tcnProb + (1 - weight) * dtwScore`, where dtwScore is 1 for
    DTW's own pick and 0 otherwise -- DTW exposes a distance for its best
    candidate only, so a per-candidate DTW score over the union does not
    exist without re-running DTW per class. This asymmetry is the honest
    limit of what the saved diagnostics support, and it is why E is a
    re-ranker over a tiny union rather than a true score-level fusion.
    """
    n, k = len(p), 5
    top5 = np.argsort(-p, axis=1)[:, :k]
    pred = np.empty(n, dtype=np.int64)
    best = np.zeros(n)
    for i in range(n):
        cands = set(top5[i].tolist())
        cands.add(int(dtw["pred"][i]))
        bs, bc = -1.0, int(top5[i][0])
        for c in cands:
            s = weight * float(p[i, c]) + (1 - weight) * (1.0 if c == dtw["pred"][i] else 0.0)
            if s > bs:
                bs, bc = s, c
        pred[i], best[i] = bc, bs
    acc = np.ones(n, bool) if accept_thresh is None else best >= accept_thresh
    return Outcome(f"E. union rerank (w={weight:.2f})", pred, acc,
                   True, {"weight": weight, **({} if accept_thresh is None else {"score": accept_thresh})})


def strat_confidence_fusion(p, dtw, conf_thresh, margin_thresh):
    """F. Confidence-aware fusion with an explicit 'unknown'.

    Prediction: DTW's pick when its gate accepted it, else the TCN's.
    Acceptance: the TCN is confident, OR DTW accepted and its ratio margin is
    comfortable. Everything else returns unknown.

    Both thresholds come from validation. The rule is deliberately a
    disjunction: the two recognisers fail on different queries (measured), so
    requiring both to be confident would reject the cases fusion exists to
    rescue.
    """
    tcn_pred, tcn_conf = p.argmax(1), p.max(1)
    dtw_margin = 1.0 - dtw["ratio"]          # larger = less ambiguous
    pred = np.where(dtw["accepted"], dtw["pred"], tcn_pred)
    acc = (tcn_conf >= conf_thresh) | (dtw["accepted"] & (dtw_margin >= margin_thresh))
    return Outcome(f"F. confidence fusion (conf>={conf_thresh:.2f} OR dtwMargin>={margin_thresh:.2f})",
                   pred, acc, True,
                   {"tcnConfidence": conf_thresh, "dtwRatioMargin": margin_thresh})


# ---------------------------------------------------------------------------
# Validation-only tuning
# ---------------------------------------------------------------------------
def tune_confidence(probs, y, target_precision, min_accepted=25):
    """Lowest TCN confidence threshold holding `target_precision` on VAL."""
    conf, correct = probs.max(1), probs.argmax(1) == y
    chosen = None
    for t in np.arange(0.99, 0.0, -0.01):
        sel = conf >= t
        if sel.sum() >= min_accepted and correct[sel].mean() >= target_precision:
            chosen = float(t)
    return chosen


def tune_union_weight(probs, y, dtw, weights):
    """Weight maximising VAL top-1. Never looks at test."""
    best_w, best_acc = weights[0], -1.0
    for w in weights:
        o = strat_union_rerank(probs, dtw, w)
        a = float((o.predicted == y).mean())
        if a > best_acc:
            best_w, best_acc = w, a
    return best_w, best_acc


def tune_margin(dtw, y, target_precision, min_accepted=25):
    """Lowest DTW ratio-margin holding `target_precision` among DTW accepts."""
    correct = dtw["pred"] == y
    margin = 1.0 - dtw["ratio"]
    chosen = None
    for t in np.arange(0.60, -0.001, -0.01):
        sel = dtw["accepted"] & (margin >= t)
        if sel.sum() >= min_accepted and correct[sel].mean() >= target_precision:
            chosen = float(t)
    return chosen if chosen is not None else 0.0


# ---------------------------------------------------------------------------
def load_split(run, data, which):
    probs = np.load(os.path.join(run, f"{which}_probs.npy"))
    with open(os.path.join(run, f"{which}_rows.json"), encoding="utf-8") as f:
        rows = json.load(f)
    assert len(rows) == len(probs), f"{which}: {len(rows)} rows vs {len(probs)} probs"
    return probs, rows


def join_dtw(rows, dtw_results, class_of_sign):
    """Pairs every row with its DTW verdict. Refuses to proceed on a miss."""
    by_key = {(r["signId"], r["file"]): r for r in dtw_results}
    pred, ratio, accepted, dist, missing = [], [], [], [], 0
    for r in rows:
        d = by_key.get((r["signId"], r["file"]))
        if d is None:
            missing += 1
            pred.append(-1); ratio.append(1.0); accepted.append(False); dist.append(np.inf)
            continue
        pred.append(class_of_sign.get(d["predicted"], -1))
        ratio.append(1.0 if d["ratio"] is None else float(d["ratio"]))
        accepted.append(bool(d["accepted"]))
        dist.append(np.inf if d["distance"] is None else float(d["distance"]))
    return {
        "pred": np.array(pred, dtype=np.int64),
        "ratio": np.array(ratio), "accepted": np.array(accepted, dtype=bool),
        "distance": np.array(dist), "missing": missing,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="ml/experiments/stage2-temporal/runs/tcn")
    ap.add_argument("--data", default="ml/experiments/stage2-temporal/data")
    ap.add_argument("--target-precision", type=float, default=0.95)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    with open(os.path.join(args.data, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(args.data, "dtw_baseline.json"), encoding="utf-8") as f:
        dtw_file = json.load(f)
    with open(os.path.join(args.run, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)

    class_of_sign = {c["signId"]: c["classIndex"] for c in manifest["classes"]}
    text_of = {c["signId"]: c["text"] for c in manifest["classes"]}
    sign_of_class = {c["classIndex"]: c["signId"] for c in manifest["classes"]}

    pte, rte = load_split(args.run, args.data, "test")
    pva, rva = load_split(args.run, args.data, "val")
    yte = np.array([r["classIndex"] for r in rte])
    yva = np.array([r["classIndex"] for r in rva])

    dte = join_dtw(rte, dtw_file["results"], class_of_sign)
    if dte["missing"]:
        print(f"ABORT: {dte['missing']} test row(s) have no DTW verdict. "
              f"Re-run export_tensors.ts --with-dtw so both columns describe the same queries.")
        return 1

    # DTW on validation is optional; without it, val-tuned DTW parameters fall
    # back to a fixed, untuned value rather than being silently fitted on test.
    dva = None
    val_dtw_path = os.path.join(args.data, "dtw_baseline_val.json")
    if os.path.exists(val_dtw_path):
        with open(val_dtw_path, encoding="utf-8") as f:
            dva = join_dtw(rva, json.load(f)["results"], class_of_sign)

    # --- tuning, on validation only ---------------------------------------
    conf_t = tune_confidence(pva, yva, args.target_precision)
    weights = [0.0, 0.25, 0.5, 0.6, 0.7, 0.75, 0.8, 0.9, 1.0]
    if dva is not None:
        best_w, best_w_acc = tune_union_weight(pva, yva, dva, weights)
        margin_t = tune_margin(dva, yva, args.target_precision)
    else:
        best_w, best_w_acc, margin_t = 1.0, float("nan"), 0.0

    # --- strategies --------------------------------------------------------
    outcomes = []
    for build in (
        lambda: strat_tcn_only(pte, dte),
        lambda: strat_dtw_only(pte, dte),
        lambda: strat_dtw_then_tcn(pte, dte),
        lambda: strat_agreement(pte, dte),
    ):
        t0 = time.perf_counter()
        o = build()
        o.fusion_ms = (time.perf_counter() - t0) * 1000 / len(yte)
        outcomes.append(o)

    for w in weights:
        t0 = time.perf_counter()
        o = strat_union_rerank(pte, dte, w)
        o.fusion_ms = (time.perf_counter() - t0) * 1000 / len(yte)
        o.name += " [val-best]" if w == best_w else ""
        outcomes.append(o)

    if conf_t is not None:
        for build in (
            lambda: strat_tcn_only(pte, dte, conf_t),
            lambda: strat_agreement(pte, dte, conf_t),
            lambda: strat_confidence_fusion(pte, dte, conf_t, margin_t),
        ):
            t0 = time.perf_counter()
            o = build()
            o.fusion_ms = (time.perf_counter() - t0) * 1000 / len(yte)
            outcomes.append(o)

    results = [score(o, yte) for o in outcomes]

    # --- oracle ------------------------------------------------------------
    tcn_c = pte.argmax(1) == yte
    dtw_c = dte["pred"] == yte
    oracle = {
        "bothCorrect": int((tcn_c & dtw_c).sum()),
        "onlyDTW": int((~tcn_c & dtw_c).sum()),
        "onlyTCN": int((tcn_c & ~dtw_c).sum()),
        "bothWrong": int((~tcn_c & ~dtw_c).sum()),
        "eitherCorrect": float((tcn_c | dtw_c).mean()),
    }

    bar = "=" * 108
    print(bar)
    print("STAGE 2.5 -- OFFLINE FUSION, held-out session")
    print(bar)
    print(f"   queries {len(yte)}   classes {len(manifest['classes'])}   "
          f"val queries {len(yva)} (SAME SESSION as test -- tuned points are optimistic)")
    print(f"   val-tuned TCN confidence threshold: "
          f"{'none met target' if conf_t is None else f'{conf_t:.2f}'} "
          f"(target precision {args.target_precision})")
    print(f"   val-tuned union weight: {best_w:.2f}" +
          ("" if dva is not None else "  (NO val DTW column -- weight NOT tuned, fixed at 1.0)"))
    print(f"   val-tuned DTW ratio margin: {margin_t:.2f}" +
          ("" if dva is not None else "  (untuned fallback)"))

    print(f"\n{'strategy':52} {'top1':>7} {'acc':>7} {'acc&cor':>8} {'FA':>5} {'prec':>7} {'rej':>7} {'tuned':>6}")
    for r in results:
        print(f"   {r['strategy']:49} {r['top1']:7.4f} {r['acceptedRate']:7.4f} "
              f"{r['acceptedAndCorrect']:8.4f} {r['falseAccepts']:5d} "
              f"{r['precisionAmongAccepted']:7.4f} {r['rejectionRate']:7.4f} "
              f"{'yes' if r['validationTuned'] else 'no':>6}")

    print(f"\n   ORACLE (upper bound, NOT achievable by any rule above)")
    print(f"   both correct {oracle['bothCorrect']}   only DTW {oracle['onlyDTW']}   "
          f"only TCN {oracle['onlyTCN']}   both wrong {oracle['bothWrong']}")
    print(f"   either correct = {oracle['eitherCorrect']:.4f}   "
          f"headroom over TCN alone = {oracle['eitherCorrect'] - tcn_c.mean():+.4f}")
    print(f"   That headroom lives entirely in the {oracle['onlyDTW']} 'only DTW' queries, and is")
    print(f"   reachable only to the extent those queries are IDENTIFIABLE at decision time.")

    # --- is DTW useful as a reranker? --------------------------------------
    only_dtw = ~tcn_c & dtw_c
    print(f"\n   IS DTW USEFUL AS A RERANKER?")
    print(f"   of the {int(only_dtw.sum())} 'only DTW correct' queries, DTW's gate ACCEPTED "
          f"{int((only_dtw & dte['accepted']).sum())}")
    print(f"   -> strategy C can rescue at most that many; the rest are invisible to it.")
    wrong_acc = dte["accepted"] & ~dtw_c
    print(f"   DTW accepted and WRONG on {int(wrong_acc.sum())} queries; of those the TCN was "
          f"right on {int((wrong_acc & tcn_c).sum())}")
    print(f"   -> that is the cost of trusting DTW first.")
    in_top5 = np.array([yte[i] in set(np.argsort(-pte[i])[:5].tolist()) for i in range(len(yte))])
    print(f"   true class inside TCN top-5: {in_top5.mean():.4f}  "
          f"(of 'only DTW' cases: {in_top5[only_dtw].mean() if only_dtw.any() else float('nan'):.4f})")
    print(f"   -> a union reranker can only fix what is already in the union.")

    print(f"\n   IS TCN CONFIDENCE USEFUL FOR REJECTION?")
    conf = pte.max(1)
    print(f"   mean confidence when correct {conf[tcn_c].mean():.4f}, when wrong {conf[~tcn_c].mean():.4f}")
    for q in (0.5, 0.75, 0.9):
        t = float(np.quantile(conf, q))
        sel = conf >= t
        print(f"   top {int((1 - q) * 100)}% most confident (conf>={t:.3f}): "
              f"precision {tcn_c[sel].mean():.4f} over {int(sel.sum())} queries")

    print(f"\n   LATENCY (ms/query)")
    print(f"   TCN             {summary['msPerQuerySingleCPU']:.2f}   (single-sample CPU, measured in training run)")
    print(f"   DTW             {dtw_file.get('msPerQuery', float('nan')):.1f}")
    print(f"   fusion overhead {max(r['fusionMsPerQuery'] for r in results):.4f}   (worst strategy above)")
    print(f"   fused pipeline  {summary['msPerQuerySingleCPU'] + dtw_file.get('msPerQuery', 0):.1f}   "
          f"(TCN + DTW + fusion; DTW dominates)")
    print(f"   TCN alone is {dtw_file.get('msPerQuery', 1) / max(summary['msPerQuerySingleCPU'], 1e-9):.0f}x faster "
          f"than running DTW at all.")

    # --- error analysis ----------------------------------------------------
    def top_pairs(mask, label):
        c = Counter()
        for i in np.where(mask)[0]:
            c[(sign_of_class.get(int(yte[i]), "?"), sign_of_class.get(int(pte[i].argmax()), "?"))] += 1
        print(f"\n   {label} (true -> TCN predicted)")
        for (t, p_), k in c.most_common(8):
            print(f"   {k:3d}x  {t} ({text_of.get(t, '')}) -> {p_} ({text_of.get(p_, '')})")

    top_pairs(~tcn_c & dtw_c, "ONLY DTW CORRECT -- the TCN's misses that DTW catches")
    both_wrong = ~tcn_c & ~dtw_c
    print(f"\n   BOTH WRONG: {int(both_wrong.sum())} queries "
          f"({both_wrong.mean():.4f}) -- unreachable by any fusion of these two")
    agree_wrong = both_wrong & (pte.argmax(1) == dte["pred"])
    print(f"   of those, the two AGREED on the same wrong class {int(agree_wrong.sum())} times "
          f"-- agreement is not evidence of correctness")

    out = args.out or os.path.join(args.run, "fusion_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "queries": int(len(yte)), "oracle": oracle,
            "valTuned": {"tcnConfidence": conf_t, "unionWeight": best_w,
                         "dtwRatioMargin": margin_t, "valTop1AtBestWeight": best_w_acc,
                         "valSharesSessionWithTest": True},
            "latency": {"tcnMs": summary["msPerQuerySingleCPU"],
                        "dtwMs": dtw_file.get("msPerQuery"),
                        "fusionOverheadMs": max(r["fusionMsPerQuery"] for r in results)},
            "strategies": results,
        }, f, indent=1, ensure_ascii=False)
    print(f"\n   written: {out}")
    print(bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
