# Architecture

## Overview

```
Phone browser
  Camera ──▶ MediaPipe HandLandmarker (WASM/GPU, local) ──▶ 21 hand landmarks
                                                                   │
                                                     lib/ml/features.ts
                                                     (normalize + 89-dim feature vector)
                                                                   │
                                                        lib/ml/model.ts
                                                (from-scratch MLP forward pass, local)
                                                                   │
                                                    lib/ml/segmentation.ts
                                              (state machine: commit or don't, per frame)
                                                                   │
                                                lib/ml/recognitionPipeline.ts
                                                  (ties it together, emits telemetry
                                                   + transcript entries)
                                                                   │
                                                          React UI (CameraView)
```

No camera frame ever leaves the device. Only session **summaries** (start/end
time, sign count, mode) optionally get POSTed to `/api/sessions`, and only if
`MONGODB_URI` is configured.

## Why the browser recomputes features instead of using the dataset's columns

The ArSL-31 dataset ships 47 pre-engineered columns (mean/angle/distance
features) alongside 42 raw landmark columns. We do **not** feed the
pre-engineered columns into the model. Their exact formulas aren't published,
so we can't prove a browser reimplementation would match bit-for-bit. Instead:

1. `ml/preprocessing/features.py` and `lib/ml/features.ts` each implement the
   *same, fully-specified* normalization + feature engineering from the raw
   21 landmarks.
2. `scripts/generate-parity-fixture.ts` runs the real TypeScript code on fixed
   input and saves the output.
3. `ml/tests/test_features_parity.py` loads that fixture and asserts the
   Python code produces identical numbers.

This is why `npm run ml:parity-fixture && npm run ml:test` passing is a
meaningful guarantee, not a formality: it's the only thing standing between
"the model works" and "the model works in Python but silently breaks in the
browser."

The same pattern applies one level up, for the trained model itself:
`ml/training/generate_model_parity_fixture.py` exports real held-out test
samples plus sklearn's exact output probabilities, and
`lib/ml/model.test.ts` asserts the from-scratch TypeScript forward pass in
`lib/ml/model.ts` reproduces them.

## Why there's no TensorFlow.js dependency

The trained model is a small MLP (89→64→31, ~7,700 parameters). Rather than
export to TensorFlow.js format (a real but heavier dependency with its own
conversion-fidelity risks), `ml/training/train.py` exports the raw weight
matrices as JSON, and `lib/ml/model.ts` implements the forward pass
(standardize → dense+ReLU → dense+softmax) directly. Total model payload is
~168KB.

## Sign segmentation state machine

See `docs/CONTINUOUS_RECOGNITION.md` for the full explanation of
`lib/ml/segmentation.ts`.

## Client/server boundary

- `lib/db/**` (MongoDB) is imported **only** by `app/api/**/route.ts` files.
  Never import it from a `"use client"` component.
- `lib/ml/**`, `lib/mediapipe/**`, `lib/speech/**` run entirely client-side.
- `app/dataset/page.tsx` and `app/translator/page.tsx` are client components
  (camera/mic access requires it); they call the API routes over `fetch`,
  never importing `lib/db` directly.

## Data storage philosophy (spec sections 28, 35)

MongoDB never receives raw video, raw audio, or individual camera frames.
Collections:

- `languages`, `signs` — static reference data, with in-app fallbacks if the
  DB isn't configured or seeded (see `app/api/languages`, `app/api/signs`).
- `modelMetadata` — optional cache of published model metadata.
- `translationSessions` — session **summaries** only (start/end time, mode,
  committed sign count, transcript character count). See
  `lib/db/types.ts::TranslationSessionDocument` — there is no field in the
  schema that could hold raw media even if a caller tried.
- `collectedSamples` — landmark sequences captured by the dataset collection
  tool (`app/dataset`), never raw video.
