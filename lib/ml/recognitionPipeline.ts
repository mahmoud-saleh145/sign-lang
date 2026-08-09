import type { FrameDetection } from "@/types/landmarks";
import type { TranscriptEntry } from "@/types/features";
import { extractFeatures } from "./features";
import { ArSLClassifier } from "./model";
import { SignSegmenter, DEFAULT_SEGMENTATION_CONFIG } from "./segmentation";
import type { SegmentationConfig } from "./segmentation";
import { ArabicTextBuilder } from "./arabicTextBuilder";
import { toArabicChar, isControlSign, controlSignSymbol, type ControlSign } from "./arabicLabels";

/**
 * Debug telemetry for one processed frame (spec item 8: raw prediction,
 * confidence, segmentation state, committed signs). Intended for a
 * developer-facing overlay, not the end-user UI.
 */
export interface FrameTelemetry {
  timestampMs: number;
  handDetected: boolean;
  rawLabel: string | null;
  rawConfidence: number;
  segmentationState: string;
  currentCandidate: string | null;
  committedThisFrame: string | null;
  transcriptLength: number;
  /** Full accumulated Arabic text after this frame (see ArabicTextBuilder). */
  arabicText: string;
}

export interface PipelineTick {
  telemetry: FrameTelemetry;
  newTranscriptEntry: TranscriptEntry | null;
}

let entryCounter = 0;

/** Display form for a committed label: the Arabic letter, or a symbol for control signs. */
function displayLabelFor(label: string): string {
  if (isControlSign(label)) return controlSignSymbol(label as ControlSign);
  return toArabicChar(label) ?? label; // fall back to raw label if somehow unmapped
}

/**
 * Stateful per-session pipeline. One instance per active camera session.
 *
 * IMPORTANT — no "nothing"/idle class exists in the ArSL-31 dataset (verify
 * yourself: public/models/arsl-31/metadata.json "classes" has 31 entries,
 * all real signs/control signs, none meaning "no sign"). So idle detection
 * is NOT model-based here — it's driven by MediaPipe hand-presence: no
 * detected hand this frame == idle, fed to the segmenter as
 * {label: null, confidence: 0}, exactly like a below-threshold prediction.
 *
 * TEXT OUTPUT: committed labels (English model class names, e.g. "Ba2")
 * are converted to Arabic characters only at this layer, via
 * ArabicTextBuilder — the model's own class names never change. There is
 * no dictionary here: words form purely by concatenating committed
 * characters in the order they were signed.
 */
export class SignRecognitionPipeline {
  private readonly segmenter: SignSegmenter;
  private readonly textBuilder = new ArabicTextBuilder();
  private transcript: TranscriptEntry[] = [];

  constructor(
    private readonly classifier: ArSLClassifier,
    segmentationConfig: SegmentationConfig = DEFAULT_SEGMENTATION_CONFIG
  ) {
    this.segmenter = new SignSegmenter(segmentationConfig);
  }

  getTranscript(): readonly TranscriptEntry[] {
    return this.transcript;
  }

  /** The accumulated, user-facing Arabic text built from committed signs so far. */
  getArabicText(): string {
    return this.textBuilder.getText();
  }

  clearTranscript(): void {
    this.transcript = [];
    this.textBuilder.reset();
  }

  /** Reset segmentation state (e.g. user pressed pause/resume) without clearing transcript. */
  resetSegmentation(): void {
    this.segmenter.reset();
  }

  processFrame(detection: FrameDetection): PipelineTick {
    const hand = detection.hands[0]; // numHands=1; see HandLandmarkerServiceOptions

    let rawLabel: string | null = null;
    let rawConfidence = 0;

    if (hand) {
      const features = extractFeatures(hand.landmarks);
      const prediction = this.classifier.predict(features);
      rawLabel = prediction.label;
      rawConfidence = prediction.confidence;
    }

    const segResult = this.segmenter.step({ label: rawLabel, confidence: rawConfidence });

    let newTranscriptEntry: TranscriptEntry | null = null;
    if (segResult.committedLabel !== null && segResult.committedConfidence !== null) {
      // Exactly one call per genuine commit — never per raw frame — is what
      // guarantees raw per-frame predictions can't spam the Arabic text.
      this.textBuilder.applyCommittedLabel(segResult.committedLabel);

      newTranscriptEntry = {
        id: `entry-${entryCounter++}`,
        sign: segResult.committedLabel, // unchanged internal model class name
        displayLabel: displayLabelFor(segResult.committedLabel), // Arabic char or control symbol
        confidence: segResult.committedConfidence,
        committedAtMs: detection.timestampMs,
      };
      this.transcript = [...this.transcript, newTranscriptEntry];
    }

    const telemetry: FrameTelemetry = {
      timestampMs: detection.timestampMs,
      handDetected: Boolean(hand),
      rawLabel,
      rawConfidence,
      segmentationState: segResult.state,
      currentCandidate: segResult.currentCandidate,
      committedThisFrame: segResult.committedLabel,
      transcriptLength: this.transcript.length,
      arabicText: this.textBuilder.getText(),
    };

    return { telemetry, newTranscriptEntry };
  }
}
