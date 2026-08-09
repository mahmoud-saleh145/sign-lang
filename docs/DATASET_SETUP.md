# Dataset Setup

## Dataset used

**ArSL-31 Alphabet Landmark Dataset** — 7,010 samples across 31 classes (28
Arabic alphabet letters + 3 control signs: Delete, Finish, Space), each
sample a single MediaPipe 21-point hand landmark reading. Licensed CC BY 4.0.

## Why manual download was required

This project was built in a sandboxed environment whose outbound network
access is restricted to a fixed allowlist (package registries, GitHub, etc.).
The dataset's host (Zenodo) was not reachable from that sandbox. The two CSV
files (`ArSL_dataset.csv`, `ArSL_dataset_encoded.csv`) were obtained by the
project owner and uploaded directly as chat attachments instead.

**If you're setting this up somewhere with normal internet access**, you can
likely fetch the dataset directly from its Zenodo record and skip the manual
upload step — check the dataset's Zenodo page for the current DOI/URL, since
this changes over time.

## Expected file

Only `ArSL_dataset.csv` is actually required by the training pipeline (it
recomputes features from the raw landmark columns itself — see
`docs/ARCHITECTURE.md` for why). Place it at:

```
/mnt/user-data/uploads/ArSL_dataset.csv
```

or update `RAW_PATH` in `ml/datasets/validate_dataset.py` and
`ml/training/train.py` to point wherever you've put it.

Expected schema: semicolon-delimited, first column `Sign` (the class label),
followed by 89 numeric feature columns (`x0..x20`, `y0..y20`, plus 47
pre-engineered columns we don't use — see Architecture doc).

## Validate before training

```bash
npm run ml:validate
```

This checks: row count, column count (89 feature columns expected), class
count (31 expected), class distribution, missing values, duplicate rows, and
that raw coordinates fall in the expected MediaPipe-normalized range. It
exits non-zero and prints every problem found if validation fails — training
should not proceed on a dataset that fails this check.

## Known dataset limitation: no signer independence

There is no signer-ID (or session-ID) column in this dataset. This means:

- We cannot verify whether samples in the test split come from a signer
  whose other samples appear in the training split.
- The reported test accuracy (see `docs/MODEL_TRAINING.md`) is likely
  **optimistic** relative to how the model will perform for a real user who
  wasn't one of the dataset's contributors.
- This is recorded in `public/models/arsl-31/metadata.json` as
  `evaluation.signerIndependent: false`, with a note, and surfaced live in
  the app's debug telemetry panel. Do not remove this caveat when retraining
  or updating the model.

If you obtain a version of this dataset (or a different one) with signer IDs,
update `ml/training/train.py`'s split logic to group by signer
(`GroupShuffleSplit` from scikit-learn) instead of the current stratified
random split, and update the metadata accordingly.
