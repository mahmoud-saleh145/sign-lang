import type { FrameDetection } from "@/types/landmarks";
import type { TranscriptEntry } from "@/types/features";
import { extractFeatures } from "./features";
import { ArSLClassifier } from "./model";
import { SignSegmenter, DEFAULT_SEGMENTATION_CONFIG } from "./segmentation";
import type { SegmentationConfig } from "./segmentation";
import { ArabicTextBuilder } from "./arabicTextBuilder";
import { toArabicChar, isControlSign, controlSignSymbol, type ControlSign } from "./arabicLabels";
import { DynamicGestureBuffer, DEFAULT_GESTURE_BUFFER_CONFIG } from "./dynamicGestureBuffer";
import type {
  GestureBufferConfig,
  GestureBufferResult,
  GestureDiscardReason,
  GestureEndReason,
} from "./dynamicGestureBuffer";
import { HandSlotTracker } from "./handSlotTracker";
import type { SideSource, SidedSequence, HandLandmarks } from "@/types/landmarks";
import {
  recognizeDynamicSignSyncWithDiagnostics,
  recognizeUnknownSideHandSyncWithDiagnostics,
  isDynamicSignRegistryLoaded,
  getDynamicSignRegistryEntryCount,
  MAX_NORMALIZED_DISTANCE,
  MAX_BEST_TO_SECOND_BEST_RATIO,
} from "./dynamicSignRecognizer";
import type {
  DynamicSignRejectionReason,
  DynamicSignDiagnostics,
  QueryOrientation,
} from "./dynamicSignRecognizer";
import { isLetterPathActive, isWordPathActive, type RecognitionMode } from "./recognitionMode";

/** High-level phase of the dynamic path this frame, for the debug panel. */
export type DynamicPhase = "off" | "idle" | "buffering" | "complete" | "classifying";

/**
 * WHAT ACTUALLY HAPPENED TO THE LAST COMPLETED GESTURE.
 *
 * This exists because the panel could previously report "classifier running
 * no" beside a perfectly good 45-frame gesture and offer no reason at all --
 * the reader could not tell a gesture the segmenter threw away from one the
 * classifier rejected from one that was classified a frame ago and whose
 * result had already been dropped. Every one of those looked identical.
 *
 * Exactly one of these is true of the most recent finalized gesture, and it
 * is latched until the next one finalizes:
 *
 *   not-yet-run                no gesture has finalized since the pipeline
 *                              started, so there is nothing to report
 *   discarded-by-segmentation  the buffer threw it out before classification
 *                              (see gestureDiscardReason: too-few-frames or
 *                              no-motion). The classifier never saw it.
 *   sides-unresolved           frames and motion were fine, but neither hand
 *                              could be named and no single unnamed hand
 *                              could be salvaged. Nothing to classify.
 *   no-usable-landmarks        the classifier ran and found nothing it could
 *                              compare -- see lastDynamicRejectionReason
 *                              ("no-comparable-side" / "sequence-too-short")
 *   classified-no-candidates   the classifier ran against an empty or
 *                              unloaded registry
 *   classified-rejected        real candidates were scored and the gate
 *                              turned them down -- the reason says which gate
 *   classified-accepted        a word was produced
 */
export type DynamicOutcome =
  | "not-yet-run"
  | "discarded-by-segmentation"
  | "sides-unresolved"
  | "no-usable-landmarks"
  | "classified-no-candidates"
  | "classified-rejected"
  | "classified-accepted";

export interface FrameTelemetry {
  timestampMs: number;
  handDetected: boolean;
  mode: RecognitionMode;

  // --- Static letter model ---
  rawLabel: string | null;
  rawConfidence: number;
  segmentationState: string;
  currentCandidate: string | null;
  /** Non-null only when a letter was actually COMMITTED. In "word" mode this
   * is always null by construction -- see letterCommitSuppressed. */
  committedThisFrame: string | null;
  /** True on a frame where the static path produced a commit that was
   * deliberately dropped because the mode is "word". Makes the suppression
   * visible instead of looking like the letter model stopped working. */
  letterCommitSuppressed: boolean;
  transcriptLength: number;
  arabicText: string;

  // --- Dynamic KArSL gesture buffer ---
  dynamicPhase: DynamicPhase;
  dynamicGestureState: string;
  dynamicGestureFrameCount: number;
  dynamicManualCapture: boolean;
  lastGestureFrameCount: number;
  lastGestureDurationMs: number | null;
  lastGestureMotionEnergy: number | null;
  lastGestureEndReason: GestureEndReason;
  lastGestureEndedByFrameCap: boolean;
  /** Set on the frame a buffer was finalized and thrown away before
   * classification, with the reason. This is what previously made the system
   * "silently do nothing". */
  gestureDiscardedThisFrame: boolean;
  gestureDiscardReason: GestureDiscardReason;

  // --- Dynamic KArSL registry + matching ---
  dynamicRegistryLoaded: boolean;
  dynamicRegistryEntryCount: number;
  lastDynamicCandidateCount: number | null;
  lastDynamicBestSignId: string | null;
  lastDynamicBestText: string | null;
  lastDynamicBestDistance: number | null;
  /** Per-side DTW distance behind the winning candidate. */
  lastDynamicBestPerSideDistance: { left: number | null; right: number | null };
  /** The inter-hand track's DTW cost -- the third component of the score. */
  lastDynamicBestRelationDistance: number | null;
  /** Denominator for all coverage fractions: frames in the captured gesture. */
  lastDynamicQueryTotalFrames: number | null;
  /** Observed frames on the winning template's sides, and its own total. */
  lastDynamicBestTemplateSideFrames: { left: number | null; right: number | null };
  lastDynamicBestTemplateTotalFrames: number | null;
  /**
   * The LANDMARK COORDINATES of the last completed gesture, for offline
   * replay. Diagnostic export only -- numeric landmark arrays and nothing
   * else. No camera frames, no imagery, no identity, no auth material: a
   * HandLandmarks entry is {x, y, z} in normalised image space, which is not
   * a picture of anything.
   *
   * Kept because the gate's inputs cannot otherwise be recovered after the
   * fact: the rejected gesture is gone the moment the buffer closes, and every
   * question about WHY it was rejected needs exactly these numbers.
   */
  lastCompletedGestureSided: SidedSequence | null;
  /** Landmarks of a completed gesture whose single hand had no determinable
   *  side. Mutually exclusive with lastCompletedGestureSided. */
  lastCompletedUnknownSideHand: HandLandmarks[] | null;
  /** Stable per-session id of the most recently COMPLETED gesture, or null. */
  lastGestureSequence: number | null;
  /** Timestamp of that gesture's completion, or null before any gesture. */
  lastCompletedGestureAtMs: number | null;
  /** Which gesture the DISPLAYED classification belongs to, or null if none. */
  lastClassificationSequence: number | null;
  /** Sides the query and the winning candidate both had. */
  lastDynamicBestSharedSides: number | null;
  /** Sides present in exactly one of them, each charged a penalty. */
  lastDynamicBestPresenceMismatches: number | null;
  /**
   * Whether the winning candidate was scored against the query's hands as
   * named, or against them mirrored. Diagnostic only.
   *
   * `dynamicSignRecognizer` has always computed this and it is the one field
   * that separates "the capture was left/right mirrored" from "the capture was
   * simply poor" -- two causes of a bad distance that look identical in every
   * other telemetry field. It was computed, carried in the diagnostics object,
   * and then dropped on the floor before reaching the panel, so a mirrored
   * capture was indistinguishable from a noisy one when reading a rejection.
   */
  lastDynamicBestOrientation: QueryOrientation | null;
  /** Real detected frames per side in the captured gesture. */
  lastDynamicQuerySideFrames: { left: number; right: number };
  /** How the captured gesture's hands were named, and how strongly. */
  lastDynamicSideSource: SideSource | null;
  lastDynamicSideConfidence: number | null;
  /** True when the gesture was one hand whose side could not be determined. */
  lastDynamicSideUnknown: boolean | null;
  /** Times the slot tracker had to reorder MediaPipe's detections to keep
   * hand identity stable across the captured gesture. */
  lastDynamicTrackerSwitches: number | null;
  lastDynamicSecondBestSignId: string | null;
  lastDynamicSecondBestDistance: number | null;
  lastDynamicBestToSecondBestRatio: number | null;
  dynamicDistanceThreshold: number;
  dynamicRatioThreshold: number;
  lastDynamicDistanceThresholdPassed: boolean | null;
  lastDynamicRatioThresholdPassed: boolean | null;
  lastDynamicAccepted: boolean | null;
  lastDynamicRejectionReason: DynamicSignRejectionReason;
  /** What happened to the most recent finalized gesture. Latched. */
  dynamicOutcome: DynamicOutcome;
  /** Frames since the displayed classification was produced. 0 means "this
   * frame"; null means nothing has been classified yet. */
  framesSinceLastClassification: number | null;
  /** Tracker slot swaps during the LAST gesture only. */
  lastGestureTrackerSwitches: number | null;
  /** Non-null exactly on the frame a KArSL sign was recognized. */
  dynamicMatchThisFrame: string | null;
  /** True exactly on the frame the recognized text was written to a
   * transcript (word transcript always; Arabic text builder only in
   * "both" mode). */
  dynamicResultSentToTranscript: boolean;
}

export interface PipelineTick {
  telemetry: FrameTelemetry;
  /** A committed static letter/control sign this frame, if any. Always null
   * in "word" mode. */
  newTranscriptEntry: TranscriptEntry | null;
  /** A recognized KArSL word/phrase this frame, if any. A separate field so
   * a letter commit and a sign match can never overwrite each other. */
  newDynamicSignEntry: TranscriptEntry | null;
}

let entryCounter = 0;

function displayLabelFor(label: string): string {
  if (isControlSign(label)) return controlSignSymbol(label as ControlSign);
  return toArabicChar(label) ?? label;
}

/**
 * Maps a completed classification onto the single outcome that describes it.
 *
 * Order matters: "no candidates" and "nothing comparable" are both reported
 * by the recogniser as rejections, but they are NOT the gate turning a real
 * candidate down, and telling them apart is the difference between "the
 * registry did not load" and "your sign was too close to another one".
 */
export function outcomeOf(d: DynamicSignDiagnostics): DynamicOutcome {
  if (d.accepted) return "classified-accepted";
  if (d.rejectionReason === "no-comparable-side" || d.rejectionReason === "sequence-too-short") {
    return "no-usable-landmarks";
  }
  if (
    d.rejectionReason === "registry-not-loaded" ||
    d.rejectionReason === "no-candidates-in-registry" ||
    d.candidateCount === 0
  ) {
    return "classified-no-candidates";
  }
  return "classified-rejected";
}

export class SignRecognitionPipeline {
  private readonly segmenter: SignSegmenter;
  private readonly textBuilder = new ArabicTextBuilder();
  private readonly gestureBuffer: DynamicGestureBuffer;
  private transcript: TranscriptEntry[] = [];
  private wordTranscript: TranscriptEntry[] = [];
  private mode: RecognitionMode;
  /** Keeps each physical hand pinned to one slot across the gesture. */
  private readonly slotTracker = new HandSlotTracker();
  private lastSideSource: SideSource | null = null;
  private lastSideConfidence: number | null = null;
  /**
   * THE LAST CLASSIFICATION, KEPT.
   *
   * Every lastDynamic* field used to be read from a local that is non-null
   * only on the single frame a gesture completes, so ~33ms later the panel
   * showed "-" for all of them and 0/0 captured frames -- a hardcoded default,
   * not a measurement. The operator saw the answer for one frame and then an
   * empty panel that looked exactly like "the classifier never ran".
   */
  private lastDiagnostics: DynamicSignDiagnostics | null = null;
  /**
   * Monotonic per-session gesture counter. Incremented the moment a gesture
   * is FINALIZED, so every snapshot, its landmarks and its classification can
   * be tied to one stable identifier instead of being matched up afterwards by
   * proximity in time.
   */
  private gestureSequence = 0;
  /**
   * The completed gesture's landmarks, LATCHED.
   *
   * GestureBufferResult.completedGesture is non-null on exactly ONE frame --
   * the frame the buffer closes -- and null on every frame after it. Reading it
   * straight off `gestureResult` therefore makes the coordinates available for
   * a single 16 ms tick and unavailable by the time a human could press a
   * button, which is what an earlier version of this telemetry did. Latched
   * here alongside the classification instead, which is what makes the export
   * usable at all.
   */
  private lastCompletedSided: SidedSequence | null = null;
  /**
   * The same latch for the OTHER completion path: a single hand whose side
   * could not be determined. It arrives as a bare frame list rather than a
   * sided timeline, and is kept in its own field rather than being forced
   * into `left` or `right` -- doing that would assert a side the data does not
   * support, which is exactly what the recognizer refuses to do elsewhere.
   */
  private lastCompletedUnknownHand: HandLandmarks[] | null = null;
  private lastCompletedSequence: number | null = null;
  private lastCompletedAtMs: number | null = null;
  /**
   * Which gesture the LATCHED classification belongs to.
   *
   * Classification is SYNCHRONOUS -- recognizeDynamicSignSyncWithDiagnostics
   * runs inline on the closing frame -- so there is no asynchronous window in
   * which a result could be pending. This field exists so the panel can prove
   * that rather than assume it: a shown result whose sequence is older than
   * the latest completed gesture is reported as stale instead of being
   * presented as the latest gesture's verdict.
   */
  private lastDiagnosticsSequence: number | null = null;
  private lastOutcome: DynamicOutcome = "not-yet-run";
  /** Frames elapsed since lastDiagnostics was set, so a stale reading is
   * visibly stale rather than silently mistaken for the current gesture. */
  private framesSinceClassification: number | null = null;
  /** Tracker swaps at the START of the current gesture, so the per-gesture
   * figure can be reported alongside the session-cumulative one. */
  private swapsAtGestureStart = 0;
  private lastGestureSwaps: number | null = null;
  /** Whether the buffer was idle on the previous tick, to detect gesture start. */
  private wasIdle = true;
  /** Latched alongside lastOutcome: the discard reason was ALSO per-tick, so
   * "discarded-by-segmentation" arrived with a blank reason one frame later.
   * Same defect as the classification fields, found by its own test. */
  private lastDiscardReason: GestureDiscardReason = null;

  constructor(
    private readonly classifier: ArSLClassifier,
    segmentationConfig: SegmentationConfig = DEFAULT_SEGMENTATION_CONFIG,
    gestureBufferConfig: GestureBufferConfig = DEFAULT_GESTURE_BUFFER_CONFIG,
    mode: RecognitionMode = "both"
  ) {
    this.segmenter = new SignSegmenter(segmentationConfig);
    this.gestureBuffer = new DynamicGestureBuffer(gestureBufferConfig);
    this.mode = mode;
  }

  getMode(): RecognitionMode {
    return this.mode;
  }

  /** Switching modes resets BOTH state machines, so partial state captured
   * under one mode can never leak into the other's results. */
  setMode(mode: RecognitionMode): void {
    if (mode === this.mode) return;
    this.mode = mode;
    this.segmenter.reset();
    this.gestureBuffer.reset();
  }

  /** Letter/control transcript (static path only). */
  getTranscript(): readonly TranscriptEntry[] {
    return this.transcript;
  }

  /** Recognized KArSL word/phrase transcript (dynamic path only). Kept
   * separate from the letter transcript so word results are never mixed
   * into, or rendered as, individual letters. */
  getWordTranscript(): readonly TranscriptEntry[] {
    return this.wordTranscript;
  }

  getArabicText(): string {
    return this.textBuilder.getText();
  }

  clearTranscript(): void {
    this.transcript = [];
    this.wordTranscript = [];
    this.textBuilder.reset();
  }

  resetSegmentation(): void {
    this.segmenter.reset();
    this.gestureBuffer.reset();
  }

  /** Opens a deterministic manual capture window (word/both modes only). */
  beginManualCapture(): void {
    if (!isWordPathActive(this.mode)) return;
    this.gestureBuffer.beginManualCapture();
  }

  /** Closes a manual capture window. The completed gesture is classified on
   * this call and returned exactly like an automatic completion would be. */
  endManualCapture(timestampMs: number): PipelineTick {
    const gestureResult = this.gestureBuffer.endManualCapture();
    return this.finishTick(
      { timestampMs, hands: [] },
      null,
      0,
      { state: "IDLE", currentCandidate: null, committedLabel: null, committedConfidence: null },
      null,
      false,
      gestureResult
    );
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

    // --- Static letter/control path ---
    // The segmenter is always stepped so segmentation telemetry stays live,
    // but in "word" mode its commits are DISCARDED rather than applied.
    const segResult = this.segmenter.step({ label: rawLabel, confidence: rawConfidence });
    const hasCommit = segResult.committedLabel !== null && segResult.committedConfidence !== null;
    const letterActive = isLetterPathActive(this.mode);

    let newTranscriptEntry: TranscriptEntry | null = null;
    let letterCommitSuppressed = false;

    if (hasCommit && letterActive) {
      this.textBuilder.applyCommittedLabel(segResult.committedLabel!);
      newTranscriptEntry = {
        id: `entry-${entryCounter++}`,
        sign: segResult.committedLabel!,
        displayLabel: displayLabelFor(segResult.committedLabel!),
        confidence: segResult.committedConfidence!,
        committedAtMs: detection.timestampMs,
      };
      this.transcript = [...this.transcript, newTranscriptEntry];
    } else if (hasCommit && !letterActive) {
      letterCommitSuppressed = true;
    }

    // --- Dynamic KArSL path ---
    // In "letter" mode the buffer is not stepped AT ALL, so it cannot
    // consume frames, allocate, or emit anything.
    // TWO HANDS, tracked. MediaPipe's detection ORDER is not a hand identity
    // (measured: it swapped mid-sign in 6 of 14 sampled real videos), so the
    // detections go through the slot tracker first and are named as sides
    // once, over the whole gesture, when the buffer finalizes.
    const gestureResult = isWordPathActive(this.mode)
      ? this.gestureBuffer.step(
        this.slotTracker.step(
          detection.hands.map((h) => ({
            landmarks: h.landmarks,
            handedness: h.handedness,
            score: h.score,
          }))
        ),
        detection.timestampMs
      )
      : null;

    return this.finishTick(
      detection,
      rawLabel,
      rawConfidence,
      segResult,
      newTranscriptEntry,
      letterCommitSuppressed,
      gestureResult
    );
  }

  private finishTick(
    detection: FrameDetection,
    rawLabel: string | null,
    rawConfidence: number,
    segResult: {
      state: string;
      currentCandidate: string | null;
      committedLabel: string | null;
      committedConfidence: number | null;
    },
    newTranscriptEntry: TranscriptEntry | null,
    letterCommitSuppressed: boolean,
    gestureResult: GestureBufferResult | null
  ): PipelineTick {
    let newDynamicSignEntry: TranscriptEntry | null = null;
    let dynamicMatchThisFrame: string | null = null;
    let dynamicResultSentToTranscript = false;
    let diagnostics: ReturnType<typeof recognizeDynamicSignSyncWithDiagnostics> | null = null;
    let phase: DynamicPhase = isWordPathActive(this.mode) ? "idle" : "off";

    // Age the previously displayed classification by one frame BEFORE this
    // frame can replace it, so "0" always means "produced on this frame".
    if (this.framesSinceClassification !== null) this.framesSinceClassification += 1;

    if (gestureResult) {
      if (gestureResult.state === "buffering") phase = "buffering";
      // Remember where the swap counter stood when this gesture began, so the
      // per-gesture figure can be separated from the session total.
      if (this.wasIdle && gestureResult.state === "buffering") {
        this.swapsAtGestureStart = this.slotTracker.switchCount;
      }
      this.wasIdle = gestureResult.state !== "buffering";

      if (gestureResult.completedSideAssignment) {
        this.lastSideSource = gestureResult.completedSideAssignment.source;
        this.lastSideConfidence = gestureResult.completedSideAssignment.confidence;
      }
      if (gestureResult.completedGesture || gestureResult.completedUnknownSideHand) {
        phase = "classifying";
        this.lastGestureSwaps = this.slotTracker.switchCount - this.swapsAtGestureStart;
        // Bump the gesture id and latch the landmarks BEFORE classifying, so
        // the sequence number already names this gesture even if the
        // classification below throws.
        this.gestureSequence += 1;
        this.lastCompletedSequence = this.gestureSequence;
        this.lastCompletedAtMs = detection.timestampMs;
        this.lastCompletedSided =
            gestureResult.completedGesture ?? null;
        this.lastCompletedUnknownHand =
            gestureResult.completedUnknownSideHand ?? null;
        diagnostics = gestureResult.completedGesture
          ? recognizeDynamicSignSyncWithDiagnostics(gestureResult.completedGesture)
          // One hand, side undetermined. Recognised as exactly that: compared
          // against whichever side of each template fits, and charged for the
          // side it cannot account for. It is never assigned a side by guess.
          : recognizeUnknownSideHandSyncWithDiagnostics(gestureResult.completedUnknownSideHand!);
        if (diagnostics.accepted && diagnostics.match) {
          const match = diagnostics.match;
          dynamicMatchThisFrame = match.entry.text;

          // The registered text is committed VERBATIM as one unit. A
          // multi-word KArSL label ("إسعافات أولية") stays one entry and is
          // never split into words or letters.
          newDynamicSignEntry = {
            id: `entry-${entryCounter++}`,
            sign: match.entry.sourceSignId ? `KArSL:${match.entry.sourceSignId}` : `KArSL:${match.entry.text}`,
            displayLabel: match.entry.text,
            confidence: Math.max(0, 1 - match.distance / MAX_NORMALIZED_DISTANCE),
            committedAtMs: detection.timestampMs,
          };
          this.wordTranscript = [...this.wordTranscript, newDynamicSignEntry];
          dynamicResultSentToTranscript = true;

          // Only "both" mode merges the word into the shared Arabic text
          // stream. In "word" mode the letter text builder is left untouched
          // so word results never appear as, or alongside, letters.
          if (this.mode === "both") {
            this.textBuilder.appendLiteralText(match.entry.text);
            this.transcript = [...this.transcript, newDynamicSignEntry];
          }
        }

        // LATCH. This is the whole point of the change: the result stays
        // readable after the one frame it was produced on.
        this.lastDiagnostics = diagnostics;
        this.lastDiagnosticsSequence = this.gestureSequence;
        this.framesSinceClassification = 0;
        this.lastOutcome = outcomeOf(diagnostics);
        this.lastDiscardReason = null;
      } else if (gestureResult.discardedAsNonDynamic) {
        phase = "complete";
        this.lastGestureSwaps = this.slotTracker.switchCount - this.swapsAtGestureStart;
        // A newly finalized gesture replaces the previous one's verdict
        // wholesale. Keeping stale diagnostics beside a fresh discard would
        // let the panel describe two different gestures at once.
        this.lastDiagnostics = null;
        this.lastDiagnosticsSequence = null;
        this.framesSinceClassification = null;
        this.lastDiscardReason = gestureResult.discardReason;
        this.lastOutcome =
          gestureResult.discardReason === "sides-unresolved"
            ? "sides-unresolved"
            : "discarded-by-segmentation";
      }
    }

    // The classification the panel should DISPLAY: the one just produced if
    // there is one, otherwise the latched previous result. Never a default.
    const shown: DynamicSignDiagnostics | null = diagnostics ?? this.lastDiagnostics;

    const telemetry: FrameTelemetry = {
      timestampMs: detection.timestampMs,
      handDetected: detection.hands.length > 0,
      mode: this.mode,

      rawLabel,
      rawConfidence,
      segmentationState: segResult.state,
      currentCandidate: segResult.currentCandidate,
      committedThisFrame: newTranscriptEntry ? segResult.committedLabel : null,
      letterCommitSuppressed,
      transcriptLength: this.transcript.length,
      arabicText: this.textBuilder.getText(),

      dynamicPhase: phase,
      dynamicGestureState: gestureResult?.state ?? "off",
      dynamicGestureFrameCount: gestureResult?.bufferedFrameCount ?? 0,
      dynamicManualCapture: this.gestureBuffer.isManualCapture(),
      lastGestureFrameCount: gestureResult?.lastGestureFrameCount ?? 0,
      lastGestureDurationMs: gestureResult?.lastGestureDurationMs ?? null,
      lastGestureMotionEnergy: gestureResult?.lastGestureMotionEnergy ?? null,
      lastGestureEndReason: gestureResult?.lastGestureEndReason ?? null,
      lastGestureEndedByFrameCap: gestureResult?.lastGestureEndedByFrameCap ?? false,
      gestureDiscardedThisFrame: gestureResult?.discardedAsNonDynamic ?? false,
      gestureDiscardReason: gestureResult?.discardReason ?? this.lastDiscardReason,

      dynamicRegistryLoaded: isDynamicSignRegistryLoaded(),
      dynamicRegistryEntryCount: getDynamicSignRegistryEntryCount(),
      lastDynamicCandidateCount: shown?.candidateCount ?? null,
      lastDynamicBestSignId: shown?.bestSignId ?? null,
      lastDynamicBestText: shown?.bestText ?? null,
      lastDynamicBestDistance: shown?.bestDistance ?? null,
      lastDynamicBestPerSideDistance: shown?.bestPerSideDistance ?? { left: null, right: null },
      lastDynamicBestRelationDistance: shown?.bestRelationDistance ?? null,
      lastDynamicQueryTotalFrames: shown?.queryTotalFrames ?? null,
      lastDynamicBestTemplateSideFrames: shown?.bestTemplateSideFrames ?? { left: null, right: null },
      lastDynamicBestTemplateTotalFrames: shown?.bestTemplateTotalFrames ?? null,
      lastCompletedGestureSided: this.lastCompletedSided,
      lastCompletedUnknownSideHand: this.lastCompletedUnknownHand,
      lastGestureSequence: this.lastCompletedSequence,
      lastCompletedGestureAtMs: this.lastCompletedAtMs,
      lastClassificationSequence: shown ? this.lastDiagnosticsSequence : null,
      lastDynamicBestSharedSides: shown?.bestSharedSides ?? null,
      lastDynamicBestPresenceMismatches: shown?.bestPresenceMismatches ?? null,
      lastDynamicBestOrientation: shown?.bestOrientation ?? null,
      lastDynamicQuerySideFrames: shown?.querySideFrames ?? { left: 0, right: 0 },
      lastDynamicSideSource: gestureResult?.completedSideAssignment?.source ?? this.lastSideSource,
      lastDynamicSideConfidence:
        gestureResult?.completedSideAssignment?.confidence ?? this.lastSideConfidence,
      lastDynamicSideUnknown: shown ? shown.querySideUnknown : null,
      lastDynamicTrackerSwitches: this.slotTracker.switchCount,
      lastDynamicSecondBestSignId: shown?.secondBestSignId ?? null,
      lastDynamicSecondBestDistance: shown?.secondBestDistance ?? null,
      lastDynamicBestToSecondBestRatio: shown?.bestToSecondBestRatio ?? null,
      dynamicDistanceThreshold: MAX_NORMALIZED_DISTANCE,
      dynamicRatioThreshold: MAX_BEST_TO_SECOND_BEST_RATIO,
      lastDynamicDistanceThresholdPassed: shown?.distanceThresholdPassed ?? null,
      lastDynamicRatioThresholdPassed: shown?.ratioThresholdPassed ?? null,
      lastDynamicAccepted: shown?.accepted ?? null,
      lastDynamicRejectionReason: shown?.rejectionReason ?? null,
      dynamicOutcome: this.lastOutcome,
      framesSinceLastClassification: this.framesSinceClassification,
      lastGestureTrackerSwitches: this.lastGestureSwaps,
      dynamicMatchThisFrame,
      dynamicResultSentToTranscript,
    };

    return { telemetry, newTranscriptEntry, newDynamicSignEntry };
  }
}
