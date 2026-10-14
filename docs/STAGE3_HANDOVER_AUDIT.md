# Stage 3 Handover Audit Report

## A. Repository Identity
- **Project Path:** `C:\Users\mahmoud\Desktop\sign-translator-v1_1`
- **Current Branch:** `main`
- **HEAD Commit:** `12e5da9d` (sign update)
- **Initial Git Status:** The repository contains significant uncommitted changes to core ML and pipeline files (`lib/ml/*.ts`, `ml/training/*.py`), and several untracked files related to the avatar system (`components/avatar/`, `lib/avatar/`).
- **Production Integrity:** No production files were modified during this audit.

## B. Architecture Map

### 1. Live Camera Path (Production)
`CameraView.tsx` $\rightarrow$ `HandLandmarkerService` $\rightarrow$ `SignRecognitionPipeline.processFrame()`
- **Static Path (Letters):** `ArSLClassifier` (MLP) $\rightarrow$ `SignSegmenter` (State Machine) $\rightarrow$ `ArabicTextBuilder` $\rightarrow$ Transcript.
- **Dynamic Path (KArSL Words):** `HandSlotTracker` (Identity) $\rightarrow$ `DynamicGestureBuffer` (Temporal) $\rightarrow$ `dynamicSignRecognizer` (DTW) $\rightarrow$ `SidedMatch` $\rightarrow$ Transcript.

### 2. Voice $\rightarrow$ Sign Path (Production)
`VoiceToSign.tsx` $\rightarrow$ `voiceToSignPipeline` $\rightarrow$ `SignerAvatar` $\rightarrow$ `glbRig` / `handRig` (Animation).

### 3. Offline Experiment Path (Non-Production)
`ml/experiments/stage2-temporal/`
- TCN training and evaluation scripts.
- `export_tensors.ts` (feature prep) $\rightarrow$ `train_tcn.py` $\rightarrow$ `evaluate.py`.

## C. Model and Dataset Inventory

### 1. Models
- **Static Model:** MLP (89 $\rightarrow$ 64 $\rightarrow$ 31). Weights in `public/models/arsl-31/weights.json`.
- **Dynamic Recognizer:** DTW against templates. Registry in `public/models/sign-vocabulary/dedicated-signs.json`.
- **TCN Model (Experimental):** Temporal Convolutional Network. Weights in `ml/experiments/stage2-temporal/runs/tcn/model.pt`.

### 2. Dataset
- **ArSL-31:** Isolated static poses for alphabet letters.
- **KArSL-502:** Isolated dynamic signs. The registry currently uses a subset (~432 signs) with a prototype store for variance.
- **Split Definition:** Defined in `lib/ml/karslTakeSplit.ts` (benchmark queries, calibration queries, and prototypes).

## D. Latest Verified Experimental Evidence

### Stage 2 (Temporal/TCN) - Verified via `summary.json`
- **Top-1 Accuracy:** $\approx 82.5\%$
- **Top-3 Accuracy:** $\approx 89.0\%$
- **Latency:** $\approx 1.43\text{ms}$ per single-query CPU inference.
- **Size:** $\approx 0.95\text{MB}$.

### Stage 2.5 (DTW vs TCN Fusion) - Verified via `fusion_results.json`
- **DTW Baseline Top-1:** $\approx 74.6\%$ (Precision among accepted $\approx 97.0\%$).
- **TCN vs DTW:** TCN is significantly faster and more accurate in isolated-sign top-1, but the existing production gate relies on DTW's high precision.
- **Finding:** A "fusion" approach exists in scripts but is not integrated into the production `SignRecognitionPipeline`.

### Stage 3 (Continuous Recognition)
- **Status:** **NOT IMPLEMENTED.**
- **Findings:** No `ml/experiments/stage3-continuous/` directory exists. No Stage 3 reports found. The current production pipeline uses a "gap-based" segmentation (`SignSegmenter`) which requires a break (no-hand or mismatched label) to commit a sign.

## E. Exact Unresolved Problem
**The Segmentation Bottleneck.**
Isolated-sign recognition (TCN/DTW) assumes the input is a pre-segmented "take". In continuous signing, there is no "idle" gap between words. The current system:
1. Relies on hand-absence or label-mismatch to trigger a commit.
2. Cannot distinguish between a "slow" sign and a "transition" between two signs.
3. Has no mechanism to predict boundaries based on model confidence (e.g., the "dip" in confidence when transitioning from Sign A to Sign B).

## F. Risks and Implementation Constraints
- **TCN Integration:** Moving from DTW (Template-based) to TCN (Weight-based) in the browser requires a lightweight runtime (the project currently uses hand-written forward passes for the MLP).
- **Overlapping Windows:** The current `DynamicGestureBuffer` is a "capture-then-classify" system. Continuous recognition requires sliding-window inference.
- **Boundary Jitter:** A recognition-driven boundary detector may trigger multiple commits for a single sign if not properly debounced.
- **Dataset Limitation:** KArSL is isolated. Validating "continuous" behavior requires synthetic concatenation or very limited real-world testing.

## G. Recommended Next Step
**Offline Prefix Analysis.**
Before changing the production pipeline, run an experiment on the KArSL dataset:
- Feed the TCN partial prefixes (first 10, 20, 30 frames) and full takes.
- Measure the "Confidence Curve" as a function of sign progress.
- Determine if the point where confidence exceeds a threshold can be used as a reliable "Start" and "End" marker.

## H. Suggested Future Implementation Plan
1. **Offline Evidence:** Build a "Continuous Simulation" by concatenating real KArSL takes and evaluating a sliding-window TCN.
2. **Runtime Integration:** Implement a lightweight TCN inference engine in `lib/ml/model.ts` (mirroring the MLP's dependency-free inode approach).
3. **Boundary Logic:** Replace the `SignSegmenter`'s gap-based logic with a "Confidence-Dip" or "Peak-Valley" detector.
4. **Performance Audit:** Verify that per-frame TCN inference maintains 30 FPS on mobile browsers.
5. **Validation:** Regression test against the isolated-sign benchmark to ensure no accuracy loss.

## I. Verification Commands
- `git status`: Verified modified and untracked files.
- `ls -R ml/experiments/`: Confirmed Stage 2 exists, Stage 3 is missing.
- `Read ml/experiments/stage2-temporal/runs/tcn/summary.json`: Verified TCN metrics.
- `Read ml/experiments/stage2-temporal/runs/tcn/fusion_results.json`: Verified DTW vs TCN comparison.
- `Read lib/ml/recognitionPipeline.ts`:algèbre Verified current production flow.
