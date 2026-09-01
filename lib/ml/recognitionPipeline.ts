import type { FrameDetection } from "@/types/landmarks";
import type { TranscriptEntry } from "@/types/features";
import { extractFeatures } from "./features";
import { ArSLClassifier } from "./model";
import { SignSegmenter, DEFAULT_SEGMENTATION_CONFIG } from "./segmentation";
import type { SegmentationConfig } from "./segmentation";
import { ArabicTextBuilder } from "./arabicTextBuilder";
import { toArabicChar, isControlSign, controlSignSymbol, type ControlSign } from "./arabicLabels";
import { DEFAULT_GESTURE_BUFFER_CONFIG, DynamicGestureBuffer, GestureBufferConfig } from "./dynamicGestureBuffer";
import { recognizeDynamicSignSync } from "./dynamicSignRecognizer";

export interface FrameTelemetry {
  timestampMs: number;
  handDetected: boolean;
  rawLabel: string | null;
  rawConfidence: number;
  segmentationState: string;
  currentCandidate: string | null;
  committedThisFrame: string | null;
  transcriptLength: number;
  arabicText: string;
  /** State of the SEPARATE dynamic-gesture buffer (lib/ml/dynamicGestureBuffer.ts).
   * Independent of segmentationState -- see that module's docstring for why
   * these are deliberately two different state machines. */
  dynamicGestureState: string;
  /** Number of frames currently buffered for dynamic-sign matching. */
  dynamicGestureFrameCount: number;
  /** Non-null exactly on the frame a KArSL dedicated sign was recognized this frame. */
  dynamicMatchThisFrame: string | null;
}

export interface PipelineTick {
  telemetry: FrameTelemetry;
  /** A committed static letter/control sign this frame, if any (unchanged behavior). */
  newTranscriptEntry: TranscriptEntry | null;
  /** A recognized KArSL dedicated word/phrase this frame, if any. Separate
   * from newTranscriptEntry so a letter commit and a dynamic-sign commit
   * can never silently overwrite each other if they happened to land on
   * the same frame. */
  newDynamicSignEntry: TranscriptEntry | null;
}

let entryCounter = 0;

function displayLabelFor(label: string): string {
  if (isControlSign(label)) return controlSignSymbol(label as ControlSign);
  return toArabicChar(label) ?? label;
}

export class SignRecognitionPipeline {
  private readonly segmenter: SignSegmenter;
  private readonly textBuilder = new ArabicTextBuilder();
  private readonly gestureBuffer: DynamicGestureBuffer;
  private transcript: TranscriptEntry[] = [];

  constructor(
    private readonly classifier: ArSLClassifier,
    segmentationConfig: SegmentationConfig = DEFAULT_SEGMENTATION_CONFIG,
    gestureBufferConfig: GestureBufferConfig = DEFAULT_GESTURE_BUFFER_CONFIG
  ) {
    this.segmenter = new SignSegmenter(segmentationConfig);
    this.gestureBuffer = new DynamicGestureBuffer(gestureBufferConfig);
  }

  getTranscript(): readonly TranscriptEntry[] {
    return this.transcript;
  }

  getArabicText(): string {
    return this.textBuilder.getText();
  }

  clearTranscript(): void {
    this.transcript = [];
    this.textBuilder.reset();
  }

  resetSegmentation(): void {
    this.segmenter.reset();
    this.gestureBuffer.reset();
  }

  processFrame(detection: FrameDetection): PipelineTick {
    const hand = detection.hands[0];

    let rawLabel: string | null = null;
    let rawConfidence = 0;

    if (hand) {
      const features = extractFeatures(hand.landmarks);
      const prediction = this.classifier.predict(features);
      rawLabel = prediction.label;
      rawConfidence = prediction.confidence;
    }

    // --- Static letter/control path: UNCHANGED from before this feature. ---
    const segResult = this.segmenter.step({ label: rawLabel, confidence: rawConfidence });

    let newTranscriptEntry: TranscriptEntry | null = null;
    if (segResult.committedLabel !== null && segResult.committedConfidence !== null) {
      this.textBuilder.applyCommittedLabel(segResult.committedLabel);

      newTranscriptEntry = {
        id: `entry-${entryCounter++}`,
        sign: segResult.committedLabel,
        displayLabel: displayLabelFor(segResult.committedLabel),
        confidence: segResult.committedConfidence,
        committedAtMs: detection.timestampMs,
      };
      this.transcript = [...this.transcript, newTranscriptEntry];
    }

    // --- Dynamic KArSL path: runs independently, alongside the static path. ---
    const gestureResult = this.gestureBuffer.step(hand ? hand.landmarks : null);

    let newDynamicSignEntry: TranscriptEntry | null = null;
    let dynamicMatchThisFrame: string | null = null;
    if (gestureResult.completedGesture) {
      const match = recognizeDynamicSignSync(gestureResult.completedGesture);
      if (match) {
        this.textBuilder.appendLiteralText(match.entry.text);
        dynamicMatchThisFrame = match.entry.text;
        newDynamicSignEntry = {
          id: `entry-${entryCounter++}`,
          sign: match.entry.sourceSignId ? `KArSL:${match.entry.sourceSignId}` : `KArSL:${match.entry.text}`,
          displayLabel: match.entry.text,
          confidence: Math.max(0, 1 - match.distance),
          committedAtMs: detection.timestampMs,
        };
        this.transcript = [...this.transcript, newDynamicSignEntry];
      }
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
      dynamicGestureState: gestureResult.state,
      dynamicGestureFrameCount: gestureResult.bufferedFrameCount,
      dynamicMatchThisFrame,
    };

    return { telemetry, newTranscriptEntry, newDynamicSignEntry };
  }
}