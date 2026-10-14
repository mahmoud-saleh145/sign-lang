# Rebuilding the KArSL motion data from source video

Run these on the machine that holds `KArSL/`. The whole run is CPU-bound
MediaPipe inference over roughly 4,000 videos; budget a few hours.

Measured on the real dataset: **502 sign directories, 4,016 videos, 3.47 GB**.
At roughly 1.2 s per video the full extraction takes about **1.5-3 hours**
depending on CPU.

### Windows (cmd)

```cmd
pip install opencv-python mediapipe pandas openpyxl

:: 1. Verify the source tree BEFORE spending hours on extraction.
python ml/datasets/verify_karsl_source.py --input KArSL --expect-min 1 --expect-max 502

:: 2. Move the OLD single-hand output aside FIRST, before anything writes to
::    processed-karsl. Move, do not delete -- a failed run should not cost you
::    what you already had.
move processed-karsl processed-karsl.old-onehand

:: 3. SMOKE TEST -- three minutes to prove the pipeline works on this machine
::    before committing to the long run. Writes to a scratch directory so it
::    cannot disturb anything.
python ml/training/extract_karsl_skeletons.py --input KArSL --output processed-karsl-smoketest --limit 20
python ml/datasets/audit_processed_karsl.py --input processed-karsl-smoketest
::    Want: "impossible jumps (> 0.15): 0" and "OLD single-hand format: 0".
rmdir /s /q processed-karsl-smoketest

:: 4. Two-hand extraction over all 502 signs.
python ml/training/extract_karsl_skeletons.py --input KArSL --output processed-karsl

:: 5. Quality audit. Do not import if this reports blocking issues.
python ml/datasets/audit_processed_karsl.py --input processed-karsl --json-out karsl-audit.json

:: 6. Rebuild the registry from the new extraction only.
python ml/training/import_karsl.py --input processed-karsl --labels KARSL-502_Labels.xlsx --output public/models/sign-vocabulary/dedicated-signs.json

:: 7. Audit the result.
python ml/datasets/audit_registry.py --registry public/models/sign-vocabulary/dedicated-signs.json --labels KARSL-502_Labels.xlsx --expect-min 1 --expect-max 502

:: 8. Re-run the suite. The two-hand tests stop skipping once the registry
::    actually contains handFrames.
npm run typecheck && npm run lint && npm run build && npm test && npm run ml:test
```

### macOS / Linux

Same commands; replace `move` with `mv` and `rmdir /s /q` with `rm -rf`.

### What to watch in the extraction output

- **`N with BOTH hands`** should be non-zero for most signs -- that is the
  point of the rebuild.
- **`N slot swap(s) absorbed`** being high is GOOD. It means MediaPipe kept
  reordering its two detections and the tracker held the hand identities
  steady, which is precisely the splice bug being prevented.

`extract_karsl_skeletons.py` writes each video's output as it finishes, so an
interrupted run can be resumed by re-running the same command — already-written
files are simply rewritten.

## Registry size — read this before running

The browser fetches the whole registry on first load, so its size is a product
concern, not just a disk one. Measured on 14 real signs and extrapolated:

| variant | 502 signs |
|---|---|
| pretty-printed, full float precision | **~149 MB** |
| 4 dp + compact separators (default) | **~39 MB** |
| …and omitting the duplicated legacy field (default) | **~25 MB** |

Both savings are on by default:

- `--precision 4` rounds coordinates. MediaPipe coordinates are
  image-normalised, so 4 dp is ~0.06 px on a 640-wide frame — far below
  landmark noise. Pass `--precision 0` to disable.
- The legacy per-entry `landmarkSequence` is an exact duplicate of the primary
  hand already in `handFrames`, so it is omitted and rebuilt on load by
  `lib/sign-language/signRegistry.ts`. Every existing consumer, including the
  camera DTW recognizer, sees the field it has always seen. Pass
  `--keep-legacy-sequence` to write it anyway.

25 MB is still a large first-load fetch. If it matters, the next step would be
splitting the registry per sign and loading on demand — the loader is the only
place that would need to change.

## What the extractor produces

Per video, `processed-karsl/<signId>/<video>.json`:

```jsonc
{
  "signId": "0199",
  "handTracking": {
    "numHands": 2,
    "handednessConvention": "camera",
    "slotNames": ["right", "left"],
    "primaryHand": "right",
    "framesWithTwoHands": 35,
    "slotSwapCount": 0          // MediaPipe orderings the tracker absorbed
  },
  "frames": [{
    "frameIndex": 0,
    "handDetected": true,        // LEGACY: any hand this frame
    "landmarks": [ /* 21 */ ],   // LEGACY: the primary hand
    "hands": {                   // NEW: both hands, tracked
      "left":  { "landmarks": [ /* 21 */ ], "score": 0.97, "rawLabel": "Right" },
      "right": { "landmarks": [ /* 21 */ ], "score": 0.99, "rawLabel": "Left" }
    }
  }]
}
```

A hand not detected in a frame is `null`. It is never interpolated, copied
from a neighbour, or otherwise invented.

## Handedness

MediaPipe's `Left`/`Right` labels assume a **mirrored** selfie image. KArSL
videos are ordinary non-mirrored recordings of a signer facing the camera, so
the label is inverted with respect to the signer's anatomy.
`--handedness-convention camera` (default) flips it so the slots are named for
the signer's real hands; `raw` stores MediaPipe's label verbatim. The raw label
and score are kept per frame either way, so the choice is reversible without
re-extracting.

This is a convention, not a measurement — it cannot be verified from the data
alone. If signs come out with the hands swapped, flip this flag.

## Why the tracker exists

With `num_hands=1` the returned hand was not pinned to one physical hand. On
14 sampled real videos, six produced sequences that swapped hands mid-sign, so
the stored trajectory was one hand spliced onto the other. Measured worst
inter-frame wrist displacement, same videos, before and after nearest-wrist
tracking:

| sign | naive (MediaPipe order) | tracked |
|---|---|---|
| 0227 | 0.237 | **0.030** |
| 0241 | 0.214 | **0.032** |
| 0255 | 0.082 | **0.022** |

A 0.2 displacement is a quarter of the frame width in 1/30 s — impossible for
a hand, and therefore a swap. `audit_processed_karsl.py` flags anything above
0.15 for exactly this reason.
