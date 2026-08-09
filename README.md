# Sign ⇄ Speech Translator (Version 1)

A mobile-first web app for continuous Arabic Sign Language (alphabet)
recognition, plus speech-to-text in the reverse direction. All ML inference
runs locally in the browser — no camera frame or audio ever leaves the
device.

## What this actually is (read before assuming more)

- **Sign → Text**: continuous recognition of the **ArSL alphabet** (28
  letters + Delete/Finish/Space control signs — 31 classes total), displayed
  as real accumulated Arabic text (e.g. signing ب → ح → ب → ك produces
  "بحبك"), not full ArSL sentence/grammar translation. Text is built purely
  by concatenating committed letters in order — there is no word dictionary
  anywhere in this project. The underlying dataset contains isolated hand
  poses, not continuous sentences — see `docs/CONTINUOUS_RECOGNITION.md` for
  exactly what that does and doesn't support, and why.
- **Speech → Text**: browser-native speech recognition, swappable for a
  production service later (`lib/speech/speechRecognition.ts`).
- Model test accuracy is **99.43%**, but the dataset has no signer-ID
  column, so this is likely optimistic for a real, unseen user — see
  "Known limitations" below and `docs/MODEL_TRAINING.md`. This caveat is
  also visible live in the app's debug telemetry panel.

## Problem statement

A person communicating via sign language should be able to sign
continuously — without stopping, pressing a button, and waiting after every
sign — and have text accumulate live, the way a normal conversation would
feel. The reverse direction (someone speaking back to them) should also
produce live text.

## Architecture

See `docs/ARCHITECTURE.md` for the full data-flow diagram and rationale.
Short version:

```
Camera → MediaPipe (local) → landmarks → feature extraction (tested parity
with training code) → MLP classifier (from-scratch TS forward pass, tested
parity with the trained sklearn model) → segmentation state machine
(commits only after temporal stability) → transcript
```

## Technology stack

- Next.js 16 (App Router), React 19, TypeScript (strict, zero `any`)
- Tailwind CSS 4
- `@mediapipe/tasks-vision` for hand landmark detection
- scikit-learn (Python) for training; exported as raw JSON weights, no
  TensorFlow.js dependency
- MongoDB (optional — app degrades gracefully without it)
- Browser `SpeechRecognition` API, behind a swappable abstraction
- Vitest (TS tests) + pytest (Python tests)

## Dataset

**ArSL-31 Alphabet Landmark Dataset** — 7,010 real samples, 31 classes, CC BY
4.0. See `docs/DATASET_SETUP.md` for the exact source, license, and how it
was obtained in this build environment (manual upload — Zenodo wasn't
reachable from the sandbox's restricted network).

## Model

MLP, 89 input features → 64 hidden (ReLU) → 31 output (softmax), trained
with scikit-learn, ~7,700 parameters, exported as a 168KB JSON weights file.
See `docs/MODEL_TRAINING.md` for the full pipeline, real evaluation numbers,
and how to retrain.

## Continuous recognition & segmentation strategy

See `docs/CONTINUOUS_RECOGNITION.md`. Summary: a sign is only added to the
transcript after `minStableFrames` consecutive stable predictions, brief
pauses (under `maxGapFrames`) don't split one sign into two, and genuinely
repeated signs are still recognized as repeats. There is no `nothing`/idle
class in the dataset, so idle detection runs on MediaPipe hand-presence
instead of a model prediction — documented in code and in that doc.

## Installation

```bash
git clone <this repo>
cd sign-translator
npm install
pip install -r ml/requirements.txt --break-system-packages   # or use a venv
```

## Environment variables

Copy `.env.example` to `.env.local`:

```bash
cp .env.example .env.local
```

| Variable | Required? | Purpose |
|---|---|---|
| `MONGODB_URI` | No | Enables the dataset collection tool and session logging. App works without it (graceful fallback). |
| `DATABASE_NAME` | No | Defaults to `sign_translator`. |

## Running locally

```bash
npm run dev
```

Open `http://localhost:3000`. Camera/mic permissions work on `localhost`
without HTTPS (browsers treat localhost as a secure context) — but testing
on an actual phone requires HTTPS. See `docs/MOBILE_TESTING.md`.

## Retraining the model

```bash
npm run ml:validate       # validate the raw dataset
npm run ml:train          # train + evaluate + export
npm run ml:parity-fixture # regenerate cross-language parity fixtures
```

See `docs/MODEL_TRAINING.md`.

## Testing

```bash
npm run test         # TS: segmentation, model parity, end-to-end pipeline, API validation
npm run ml:test       # Python: cross-language feature parity
npm run typecheck
npm run lint
npm run build
npm run verify        # all of the above, in order
```

## Mobile testing

See `docs/MOBILE_TESTING.md` for the HTTPS requirement, permission-flow
checks, browser compatibility notes, and a real-device checklist.

## Deployment

Any Next.js host with HTTPS works; Vercel is the easiest fit for this stack.

1. Push to a Git repo, import into Vercel.
2. Set `MONGODB_URI` / `DATABASE_NAME` as Vercel environment variables if
   you want the dataset tool / session logging (e.g. a free MongoDB Atlas
   cluster — MongoDB itself doesn't need to be hosted on Vercel).
3. Deploy. Vercel gives you HTTPS automatically, satisfying the camera/mic
   secure-context requirement.
4. Optional: restore real Google Fonts — see `docs/TROUBLESHOOTING.md`
   ("Build fails fetching Google Fonts") for the exact diff, since this
   sandbox's network couldn't reach `fonts.googleapis.com` and a production
   host with normal internet access can.

## Known limitations

- **Not signer-independent evaluation** — no signer-ID column in the
  dataset; 99.43% test accuracy is likely optimistic for real unseen users.
- **Isolated-sign vocabulary only** — 31 ArSL alphabet/control signs, not
  sentence-level ArSL.
- **No `nothing`/idle class** — idle detection is hand-presence-based, not
  model-based; see `docs/CONTINUOUS_RECOGNITION.md`.
- **Not tested on a physical phone** — built and automatically tested in a
  sandboxed environment; see `docs/MOBILE_TESTING.md`'s honesty note and
  checklist.
- **Speech-to-text browser support varies**, especially on iOS Safari; the
  app detects and reports unsupported browsers rather than failing silently.
- **Offline behavior**: MediaPipe's WASM/model assets are fetched from a CDN
  (`cdn.jsdelivr.net` / `storage.googleapis.com`) on first load — see
  `lib/mediapipe/handLandmarker.ts`. After that initial load (and browser
  caching), the recognition pipeline itself runs fully offline; the app has
  not been verified to survive a fully offline *first* load, since that
  would require self-hosting those assets, which wasn't done here.

## Privacy

Camera frames and microphone audio are processed entirely on-device and
never uploaded. Only optional session **summaries** (timestamps, sign count,
mode — never transcript content, never media) may be sent to `/api/sessions`
if MongoDB is configured. See `docs/ARCHITECTURE.md`'s data storage section.

## Future roadmap

- Signer-independent re-evaluation once samples from new signers are
  collected via `app/dataset`.
- A real `nothing`/idle class trained on real negative samples.
- Sentence-level ArSL support, contingent on finding/creating a suitable
  continuous-annotation dataset.
- Self-hosted MediaPipe assets for fully offline first-load support.
