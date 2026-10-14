"""
STAGE 2 EXPERIMENT -- measurement, and the head-to-head against DTW.

Reads the trained model's test probabilities and, when present,
data/dtw_baseline.json -- which export_tensors.ts --with-dtw produced from the
SAME rows. The comparison is therefore built from two columns of one file
rather than from two separately assembled evaluations.

Reports, on the held-out session:
  top-1 / top-3 / top-5, per-class and macro accuracy
  confusion pairs, most-confused signs
  calibration (ECE + reliability bins)
  an abstention gate whose threshold is chosen on VAL and applied to TEST
  latency, parameter count, serialised size
  DTW vs model, same queries: top-1, accepted-and-correct, false accepts,
  rejection rate

The abstention threshold is deliberately chosen on val and never tuned on
test. Val shares a recording session with test, so the operating point it
picks is optimistic -- stated here and in the output rather than hidden.

Usage:
    python ml/experiments/stage2-temporal/evaluate.py \
        --run ml/experiments/stage2-temporal/runs/tcn \
        --data ml/experiments/stage2-temporal/data
"""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np


def expected_calibration_error(conf, correct, bins=10):
    """ECE plus the per-bin table. A model whose confidence means nothing is
    useless as a gate however good its top-1 is, so this is not optional."""
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if not m.any():
            continue
        acc, avg = correct[m].mean(), conf[m].mean()
        ece += m.mean() * abs(acc - avg)
        rows.append((edges[i], edges[i + 1], int(m.sum()), float(avg), float(acc)))
    return float(ece), rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="ml/experiments/stage2-temporal/runs/tcn")
    p.add_argument("--data", default="ml/experiments/stage2-temporal/data")
    p.add_argument("--target-precision", type=float, default=0.95,
                   help="Precision the abstention gate is asked to hold, chosen on VAL.")
    args = p.parse_args()

    with open(os.path.join(args.data, "manifest.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    with open(os.path.join(args.run, "summary.json"), encoding="utf-8") as f:
        summary = json.load(f)
    with open(os.path.join(args.run, "test_rows.json"), encoding="utf-8") as f:
        rows = json.load(f)
    probs = np.load(os.path.join(args.run, "test_probs.npy"))

    text_of = {c["signId"]: c["text"] for c in manifest["classes"]}
    sign_of_index = {c["classIndex"]: c["signId"] for c in manifest["classes"]}
    y = np.array([r["classIndex"] for r in rows])
    pred = probs.argmax(1)
    conf = probs.max(1)
    correct = pred == y
    n = len(y)

    order = np.argsort(-probs, axis=1)
    top3 = (order[:, :3] == y[:, None]).any(1).mean()
    top5 = (order[:, :5] == y[:, None]).any(1).mean()

    per_class = defaultdict(list)
    for i in range(n):
        per_class[y[i]].append(bool(correct[i]))
    class_acc = {k: float(np.mean(v)) for k, v in per_class.items()}
    macro = float(np.mean(list(class_acc.values())))

    confusions = Counter()
    for i in range(n):
        if not correct[i]:
            confusions[(sign_of_index[int(y[i])], sign_of_index[int(pred[i])])] += 1

    ece, bins = expected_calibration_error(conf, correct)

    # --- abstention: threshold chosen on VAL, applied to TEST --------------
    gate = None
    val_probs_path = os.path.join(args.run, "val_probs.npy")
    if os.path.exists(val_probs_path):
        vprobs = np.load(val_probs_path)
        with open(os.path.join(args.run, "val_rows.json"), encoding="utf-8") as f:
            vrows = json.load(f)
        vy = np.array([r["classIndex"] for r in vrows])
        vconf, vcorrect = vprobs.max(1), vprobs.argmax(1) == vy
        chosen = None
        for t in np.arange(0.99, 0.0, -0.01):
            sel = vconf >= t
            if sel.sum() >= 10 and vcorrect[sel].mean() >= args.target_precision:
                chosen = float(t)
        if chosen is not None:
            sel = conf >= chosen
            gate = {
                "threshold": chosen,
                "accepted": int(sel.sum()),
                "acceptedCorrect": int(correct[sel].sum()),
                "falseAccepts": int((~correct[sel]).sum()),
                "precision": float(correct[sel].mean()) if sel.sum() else 1.0,
                "acceptanceRate": float(sel.mean()),
                "abstentionRate": float(1 - sel.mean()),
            }

    bar = "=" * 78
    print(bar)
    print("STAGE 2 TEMPORAL MODEL -- HELD-OUT SESSION")
    print(bar)
    print(f"   queries {n}   classes {len(manifest['classes'])}   (queried {len(per_class)})")
    print(f"   top-1 {correct.mean():.4f}   top-3 {top3:.4f}   top-5 {top5:.4f}")
    print(f"   macro (per-class mean) {macro:.4f}")
    print(f"   classes at 0% {sum(1 for v in class_acc.values() if v == 0)}   "
          f"at 100% {sum(1 for v in class_acc.values() if v == 1)}")
    print(f"   distinct confusion pairs {len(confusions)}   total errors {int((~correct).sum())}")
    print(f"   parameters {summary['parameters']:,}   model {summary['modelSizeMB']:.2f} MB")
    print(f"   latency {summary['msPerQuerySingleCPU']:.2f} ms/query single-sample CPU")

    print(f"\n   CALIBRATION   ECE {ece:.4f}   mean confidence {conf.mean():.4f} "
          f"vs accuracy {correct.mean():.4f}")
    print("   conf range        n     mean conf   accuracy")
    for lo, hi, cnt, avg, acc in bins:
        print(f"   {lo:.2f}-{hi:.2f}  {cnt:7d}   {avg:9.4f}   {acc:8.4f}")

    if gate:
        print(f"\n   ABSTENTION GATE (threshold {gate['threshold']:.2f} chosen on VAL for "
              f"precision >= {args.target_precision})")
        print(f"   accepted {gate['accepted']}/{n} ({gate['acceptanceRate']:.4f})   "
              f"correct {gate['acceptedCorrect']}   false accepts {gate['falseAccepts']}   "
              f"precision {gate['precision']:.4f}")
        print(f"   abstained (\"unknown sign\") {gate['abstentionRate']:.4f}")
        print("   NOTE: val shares a recording session with test, so this operating")
        print("   point is optimistic. It is not a production threshold.")

    print("\n   MOST CONFUSED (true -> predicted, count)")
    for (t, pr), c in confusions.most_common(15):
        print(f"   {c:3d}x  {t} ({text_of.get(t, '')})  ->  {pr} ({text_of.get(pr, '')})")

    # --- head-to-head, same rows ------------------------------------------
    dtw_path = os.path.join(args.data, "dtw_baseline.json")
    if os.path.exists(dtw_path):
        with open(dtw_path, encoding="utf-8") as f:
            dtw = json.load(f)
        by_key = {(r["signId"], r["file"]): r for r in dtw["results"]}
        paired = [(i, by_key[(rows[i]["signId"], rows[i]["file"])])
                  for i in range(n) if (rows[i]["signId"], rows[i]["file"]) in by_key]
        if paired:
            idx = [i for i, _ in paired]
            d = [r for _, r in paired]
            dtw_correct = np.array([r["predicted"] == rows[i]["signId"] for i, r in paired])
            dtw_acc = np.array([bool(r["accepted"]) for r in d])
            m_correct = correct[idx]
            print(f"\n{bar}\n   HEAD TO HEAD on the SAME {len(paired)} held-out queries\n{bar}")
            print(f"   {'':26} {'DTW':>12} {'temporal model':>16}")
            print(f"   {'top-1':26} {dtw_correct.mean():12.4f} {m_correct.mean():16.4f}")
            def model_cell(value, fmt="{:16.4f}"):
                """Right-hand column, or an explicit 'no gate' when no
                threshold on val met the requested precision."""
                return fmt.format(value) if gate else f"{'no gate':>16}"

            print(f"   {'accepted':26} {dtw_acc.mean():12.4f} "
                  f"{model_cell(gate['acceptanceRate'] if gate else 0)}")
            print(f"   {'accepted AND correct':26} {(dtw_acc & dtw_correct).mean():12.4f} "
                  f"{model_cell((gate['acceptedCorrect'] / len(paired)) if gate else 0)}")
            print(f"   {'false accepts':26} {int((dtw_acc & ~dtw_correct).sum()):12d} "
                  f"{model_cell(gate['falseAccepts'] if gate else 0, '{:16d}')}")
            print(f"   {'rejection / abstention':26} {1 - dtw_acc.mean():12.4f} "
                  f"{model_cell(gate['abstentionRate'] if gate else 0)}")
            print(f"   {'ms per query':26} {dtw.get('msPerQuery', float('nan')):12.1f} "
                  f"{summary['msPerQuerySingleCPU']:16.2f}")
            print(f"   {'artifact size':26} {'prototypes':>12} {summary['modelSizeMB']:15.2f}MB")
            both = int((dtw_correct & m_correct).sum())
            print(f"\n   agreement: both right {both}   "
                  f"only DTW {int((dtw_correct & ~m_correct).sum())}   "
                  f"only model {int((~dtw_correct & m_correct).sum())}   "
                  f"both wrong {int((~dtw_correct & ~m_correct).sum())}")
            print("   'only model' plus 'only DTW' is the headroom a Stage-1+2 fusion could reach.")
    else:
        print(f"\n   (no {dtw_path} -- re-run the export with --with-dtw for the head-to-head)")
    print(bar)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
