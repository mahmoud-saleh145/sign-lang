import type { SegmentationState } from "@/types/features";

/**
 * Per-frame classifier output fed into the state machine.
 * `label: null` means "no hand detected" or "below confidence threshold for
 * every class" — i.e. this frame counts as idle, not as a prediction.
 */
export interface FramePrediction {
  label: string | null;
  confidence: number;
}

export interface SegmentationConfig {
  /** Minimum confidence for a prediction to count as a real candidate. */
  confidenceThreshold: number;
  /** Consecutive matching frames required before a sign is considered stable. */
  minStableFrames: number;
  /** Consecutive non-matching/idle frames allowed before we decide the sign ended. */
  maxGapFrames: number;
  /**
   * After committing a sign, require this many idle/non-matching frames
   * before the *same* label can be committed again. Prevents one long hold
   * of a sign from being committed multiple times, while still allowing a
   * genuinely repeated sign (spec section 18) once the hand clearly resets.
   */
  minFramesBetweenRepeats: number;
}

export const DEFAULT_SEGMENTATION_CONFIG: SegmentationConfig = {
  confidenceThreshold: 0.75,
  minStableFrames: 8, // at ~24fps, roughly 1/3 second of stable prediction
  maxGapFrames: 6,
  minFramesBetweenRepeats: 5,
};

export interface SegmentationResult {
  state: SegmentationState;
  /** Non-null exactly on the frame a sign is committed to the transcript. */
  committedLabel: string | null;
  committedConfidence: number | null;
  /** Best current candidate, for showing "current sign" in the UI live. */
  currentCandidate: string | null;
  currentCandidateConfidence: number;
}

/**
 * Stateful sign segmenter. Feed it one FramePrediction per processed video
 * frame; it returns a SegmentationResult telling the UI what to display and
 * whether a sign should be appended to the transcript this frame.
 *
 * State machine (spec section 17):
 *   IDLE -> POSSIBLE_SIGN -> SIGN_ACTIVE -> POSSIBLE_END -> COMMITTED -> IDLE
 */
export class SignSegmenter {
  private state: SegmentationState = "IDLE";
  private candidateLabel: string | null = null;
  private stableFrameCount = 0;
  private gapFrameCount = 0;
  private lastCommittedLabel: string | null = null;
  private framesSinceLastCommit = Number.POSITIVE_INFINITY;
  private lastConfidence = 0;

  constructor(private readonly config: SegmentationConfig = DEFAULT_SEGMENTATION_CONFIG) {}

  reset(): void {
    this.state = "IDLE";
    this.candidateLabel = null;
    this.stableFrameCount = 0;
    this.gapFrameCount = 0;
    this.lastCommittedLabel = null;
    this.framesSinceLastCommit = Number.POSITIVE_INFINITY;
    this.lastConfidence = 0;
  }

  step(frame: FramePrediction): SegmentationResult {
    this.framesSinceLastCommit += 1;

    const isCandidate =
      frame.label !== null && frame.confidence >= this.config.confidenceThreshold;

    switch (this.state) {
      case "IDLE":
      case "COMMITTED": {
        if (isCandidate) {
          this.state = "POSSIBLE_SIGN";
          this.candidateLabel = frame.label;
          this.stableFrameCount = 1;
          this.lastConfidence = frame.confidence;
        } else {
          this.state = "IDLE";
          this.candidateLabel = null;
          this.stableFrameCount = 0;
        }
        this.gapFrameCount = 0;
        break;
      }

      case "POSSIBLE_SIGN": {
        if (isCandidate && frame.label === this.candidateLabel) {
          this.stableFrameCount += 1;
          this.lastConfidence = frame.confidence;
          if (this.stableFrameCount >= this.config.minStableFrames) {
            this.state = "SIGN_ACTIVE";
          }
        } else if (isCandidate) {
          // Prediction flipped to a different class before stabilizing —
          // restart the candidacy window on the new label.
          this.candidateLabel = frame.label;
          this.stableFrameCount = 1;
          this.lastConfidence = frame.confidence;
        } else {
          // Lost the candidate before it stabilized; treat as idle/noise.
          this.state = "IDLE";
          this.candidateLabel = null;
          this.stableFrameCount = 0;
        }
        break;
      }

      case "SIGN_ACTIVE": {
        if (isCandidate && frame.label === this.candidateLabel) {
          this.gapFrameCount = 0;
          this.lastConfidence = frame.confidence;
        } else {
          this.gapFrameCount += 1;
          if (this.gapFrameCount >= this.config.maxGapFrames) {
            this.state = "POSSIBLE_END";
          }
        }
        break;
      }

      case "POSSIBLE_END": {
        if (isCandidate && frame.label === this.candidateLabel) {
          // Hand came back within the gap tolerance — still the same sign.
          this.state = "SIGN_ACTIVE";
          this.gapFrameCount = 0;
          this.lastConfidence = frame.confidence;
        } else {
          // Confirmed end of sign. Commit it, unless this exact label was
          // just committed too recently (debounce a single long hold).
          const canCommit =
            this.candidateLabel !== this.lastCommittedLabel ||
            this.framesSinceLastCommit >= this.config.minFramesBetweenRepeats;

          const committed = canCommit ? this.candidateLabel : null;
          const committedConfidence = canCommit ? this.lastConfidence : null;

          if (canCommit && committed !== null) {
            this.lastCommittedLabel = committed;
            this.framesSinceLastCommit = 0;
          }

          const result: SegmentationResult = {
            state: "COMMITTED",
            committedLabel: committed,
            committedConfidence,
            currentCandidate: this.candidateLabel,
            currentCandidateConfidence: this.lastConfidence,
          };

          // Now resolve the state for the *next* frame based on this frame.
          if (isCandidate) {
            this.state = "POSSIBLE_SIGN";
            this.candidateLabel = frame.label;
            this.stableFrameCount = 1;
            this.lastConfidence = frame.confidence;
          } else {
            this.state = "IDLE";
            this.candidateLabel = null;
            this.stableFrameCount = 0;
          }
          this.gapFrameCount = 0;

          return result;
        }
        break;
      }
    }

    return {
      state: this.state,
      committedLabel: null,
      committedConfidence: null,
      currentCandidate: this.candidateLabel,
      currentCandidateConfidence: this.lastConfidence,
    };
  }
}
