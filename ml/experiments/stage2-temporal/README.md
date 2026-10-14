# Stage 2 — temporal model experiment

OFFLINE EXPERIMENT. Nothing here is imported by the app, and running it changes
no production behaviour, no threshold and no generated production artifact.

## Run order

```
# 1. Tensors + the DTW column, from the SAME rows. ~25 min for the DTW pass.
npx tsx ml/experiments/stage2-temporal/export_tensors.ts --with-dtw

# 2. Train. CPU is fine.
pip install -r ml/experiments/stage2-temporal/requirements.txt
python ml/experiments/stage2-temporal/train_tcn.py

# 3. Measure, including the head-to-head.
python ml/experiments/stage2-temporal/evaluate.py
```

## What the model sees

One take is `[T=30, C=95]` float32, built by the PRODUCTION
`prepareSidedSequence()` — not a Python reimplementation.

| channels | meaning |
|---|---|
| 0–43 | LEFT: 42 shape (21 landmarks x,y after `normalizeLandmarks`) + 2 position (wrist offset from that side's own mean wrist / own mean hand size × `POSITION_WEIGHT`) |
| 44–87 | RIGHT, same 44 |
| 88–89 | RELATION: left wrist − right wrist / mean hand size × `RELATION_WEIGHT` |
| 90–92 | presence flags: left, right, relation |
| 93–94 | coverage: detected frames / total frames, per side |

Presence flags exist because a missing side is written as zeros and zeros are a
legal landmark configuration — without a flag the model cannot tell "absent"
from "at its own reference point". DTW never needed this because a null side is
skipped and charged through coverage weighting; a fixed-width tensor has to say
it out loud. Nothing is interpolated or synthesised.

## Split

`splitTakes()` is called per sign, and the assignment is carried in the
manifest rather than recomputed:

- **train** = `split.prototypes` — the sessions DTW draws its prototypes from
- **val** = `split.calibration` — the held-out session's other half
- **test** = `split.benchmark` — exactly the queries the DTW benchmark uses

Takes named in `karsl-exclusions.json` are dropped from all three.

Two leakage guards run at export and are asserted again at load: no take in
both train and val/test, and no test take from a session that also supplied
training data for that sign. Both abort the run.

**Val is not a clean model-selection set.** It shares a recording session with
test, so any number measured on it — including the abstention threshold — is
optimistic. It is used for early stopping because a two-session corpus offers
nothing better, and it is labelled as such everywhere it appears.

## Why a temporal CNN rather than a BiLSTM

The deciding factor is export. `lib/ml/model.ts` already ships a hand-written
forward pass over `weights.json` with no inference runtime at all. Conv1d,
ReLU, pooling and a dense layer follow that precedent and would add **zero**
browser dependencies. A BiLSTM needs onnxruntime-web or TF.js — megabytes of
WASM before the model — for a 30-step sequence that dilations 1/2/4 already
span completely (receptive field 29 frames).
