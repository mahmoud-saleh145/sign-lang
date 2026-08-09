# Model Training

## Pipeline

```bash
pip install -r ml/requirements.txt --break-system-packages   # or use a venv
npm run ml:validate      # validate the raw dataset first
npm run ml:train         # trains + evaluates + exports the model
npm run ml:parity-fixture  # regenerate parity fixtures (only needed after
                            # changing the feature or model architecture)
npm run ml:test          # run Python-side tests, including cross-language parity
npm run test             # run TS-side tests, including model + pipeline parity
```

`ml/training/train.py`:
1. Loads `ArSL_dataset.csv`.
2. Recomputes the 89-dim feature vector for every row using
   `ml/preprocessing/features.py` (the same code path used for browser
   inference — see `docs/ARCHITECTURE.md`).
3. Splits 70/15/15 train/val/test, stratified by class. **Not** signer
   independent — see `docs/DATASET_SETUP.md`.
4. Standardizes features (mean/std fit on the training split only).
5. Trains an MLP: 89 → 64 (ReLU) → 31 (softmax), via
   `sklearn.neural_network.MLPClassifier`, Adam optimizer, early stopping.
6. Evaluates on the held-out test split.
7. Exports `public/models/arsl-31/weights.json` (raw weight matrices, for
   `lib/ml/model.ts`'s from-scratch forward pass) and
   `public/models/arsl-31/metadata.json` (spec section 29's model metadata).
8. Writes `ml/evaluation/eval_report.json` (full classification report +
   confusion matrix).

## Actual results (real run, not simulated)

- Train / val / test sizes: 4,907 / 1,051 / 1,052
- **Test accuracy: 99.43%**
- **Test macro F1: 0.9942**
- Converged in 27 iterations

**Read this before quoting that number anywhere:** this is very likely
optimistic. The dataset has no signer-ID column (see
`docs/DATASET_SETUP.md`), so near-duplicate frames from the same signing
session can end up on both sides of the train/test split, inflating
accuracy. Treat 99.43% as "the model fits this dataset's isolated poses
extremely well," not "the model is 99.43% accurate for a real, previously
unseen user on a real phone camera." There is no real-world/field accuracy
number in this project, because no field evaluation has been done. Anyone
extending this project should get real held-out data from *new* signers
(e.g. via the dataset collection tool, `app/dataset`) before trusting a
higher-confidence number.

Full per-class metrics: `ml/evaluation/eval_report.json`.

## Retraining after a feature/architecture change

If you touch `ml/preprocessing/features.py` (or its TS counterpart) or the
model architecture in `ml/training/train.py`:

1. Bump `PREPROCESSING_VERSION` in both `lib/ml/features.ts` and
   `ml/preprocessing/features.py` if the feature formulas changed.
2. Re-run `npm run ml:train`.
3. Re-run `npm run ml:parity-fixture` to regenerate both parity fixtures
   against the new code/weights.
4. Re-run `npm run test` and `npm run ml:test` — if either parity test
   fails, the browser and training pipelines have drifted apart and
   predictions in the app will be wrong. Do not ship until both pass.

## Adding a real "nothing" / idle class

Not currently possible without new data — see
`docs/CONTINUOUS_RECOGNITION.md` for why. If you collect real "no sign"
samples via `app/dataset` (recording a class called e.g. `Nothing` while
showing the camera an empty/relaxed hand or no hand), you could retrain with
a 32nd class and switch `lib/ml/recognitionPipeline.ts` to trust the model's
own idle prediction instead of relying purely on hand-presence.
