# Continuous Recognition Strategy

## What the dataset actually supports

The ArSL-31 dataset (7,010 samples, 31 classes: 28 alphabet letters + Delete,
Finish, Space) contains **isolated static hand poses**, one per sample. It
has no continuous-sentence annotations and no signer-ID column.

So Version 1 is honestly scoped as: **continuous recognition of a limited,
isolated-sign vocabulary** — not full continuous natural-language sentence
translation. A user can sign a sequence of alphabet letters continuously
without stopping between them, and the transcript accumulates letter by
letter. It cannot understand sentence-level ArSL grammar, because no dataset
here contains that information.

## No "nothing" / idle class exists

Check `public/models/arsl-31/metadata.json` yourself: all 31 classes are real
signs or control signs. None of them mean "no sign is being performed."

Consequently, idle/no-sign detection in `lib/ml/recognitionPipeline.ts` is
**not** based on a model prediction — it's based on MediaPipe's hand-presence
signal. If `HandLandmarkerService` detects zero hands in a frame, that frame
is fed to the segmenter as `{label: null, confidence: 0}`, identical to how a
below-confidence-threshold prediction is treated. This is a deliberate
design choice, not an oversight — training a "nothing" class would require
either fabricating negative examples (the honesty rule this project
follows against) or collecting real ones via the dataset collection tool.

## The segmentation state machine

`lib/ml/segmentation.ts` implements:

```
IDLE / COMMITTED
     │ confident prediction appears
     ▼
POSSIBLE_SIGN  ──(different label appears before stabilizing)──▶ restarts candidacy
     │ same label sustained for `minStableFrames` frames
     ▼
SIGN_ACTIVE
     │ prediction drops/changes for `maxGapFrames` consecutive frames
     ▼
POSSIBLE_END
     │
     ├─(same label returns within the gap tolerance)──▶ back to SIGN_ACTIVE
     │
     └─(gap confirmed)──▶ COMMIT to transcript ──▶ back to IDLE or POSSIBLE_SIGN
```

Default tuning (`DEFAULT_SEGMENTATION_CONFIG`, also driven by
`metadata.json`'s `temporalWindow`):
- `confidenceThreshold: 0.75` — predictions below this count as no-prediction.
- `minStableFrames: 8` — roughly ⅓ second at 24fps before a sign is
  considered "real" rather than noise.
- `maxGapFrames: 6` — how many consecutive idle/mismatched frames are
  tolerated as a brief pause before the sign is considered finished.
- `minFramesBetweenRepeats: 5` — after committing a sign, requires the
  segmenter to pass through idle/mismatch for at least this many frames
  before the *same* label can commit again. This is what lets a genuinely
  repeated sign ("HELLO HELLO") through while still preventing one long
  hold of a sign from spamming the transcript.

## What's actually verified, and how

`lib/ml/segmentation.test.ts` — 7 unit tests on the pure state machine logic
(no camera, no model): no-flicker commits, exactly-once commit for a clean
hold, brief-pause tolerance, repeated-sign support, multi-sign sequencing,
below-threshold predictions ignored, and clean reset.

`lib/ml/recognitionPipeline.test.ts` — 8 integration tests that load the
**actual trained model weights** (`public/models/arsl-31/weights.json`) and
**real held-out landmark data** (`ml/tests/fixtures/continuous_stream_fixture.json`,
generated from real CSV rows never used for training the exported fixture
sample), constructing simulated continuous streams (hand present/absent
segments) and asserting the pipeline behaves correctly end-to-end — including
a sanity check that the real model actually classifies real "Alef" landmarks
as "Alef" before trusting the rest of the test.

## Debug telemetry

`components/translator/DebugTelemetryPanel.tsx`, toggled from the camera
view, shows every frame's raw label, raw confidence, segmentation state,
current candidate, whether this frame committed, transcript length, the
accumulated Arabic text, and the active model's confidence threshold /
stability window / accuracy figures (with the signer-independence caveat
visible). This is what you'd use to diagnose real-device behavior that
differs from what's described here.

## Arabic text output (labels → characters → words)

The model's internal class names (`Alef`, `Ba2`, `7a2`, ...) never change —
they're fixed by the trained weights. Converting them to real Arabic text is
a presentation-layer concern, handled entirely downstream of segmentation:

- `lib/ml/arabicLabels.ts` — a static lookup table, `{English class name ->
  Arabic character}`, sourced directly from the dataset's own published
  documentation (Zenodo record 18363162), not guessed. Also identifies the
  3 control signs (`Space`, `Delete`, `Finish`).
- `lib/ml/arabicTextBuilder.ts` — a small stateful class that takes one
  **already-committed** label at a time (i.e. only labels that passed
  `lib/ml/segmentation.ts`'s temporal-stability check ever reach it) and
  applies it to a running text string:
  - a letter class appends its Arabic character
  - `Space` appends a literal space
  - `Delete` removes the last character (letter or space)
  - `Finish` is a no-op on the text — V1 defines no punctuation semantics,
    so we don't invent one. It still requires the same temporal stability
    to commit as any other sign.

**There is no dictionary anywhere in this layer.** `ArabicTextBuilder` has
no notion of Arabic vocabulary; "بحبك" only ever comes from four separate
commits — ب, ح, ب, ك — concatenated in the order they were signed. This is
deliberate and tested: `lib/ml/arabicTextBuilder.test.ts` proves word
formation is pure concatenation, and
`lib/ml/recognitionPipeline.test.ts` proves it end-to-end with the real
trained model and real landmark data, including a test that holds one real
sign for 60 raw frames and confirms exactly one character gets appended
(not one per raw frame).

`SignRecognitionPipeline.getArabicText()` exposes the accumulated text;
each `TranscriptEntry.sign` still holds the original English class name
(for debug/telemetry), while `TranscriptEntry.displayLabel` holds the
Arabic character or control-sign symbol (used for the live "current sign"
indicator in the camera view).
