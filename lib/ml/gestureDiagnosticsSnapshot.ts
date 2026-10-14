/**
 * ONE TYPED SHAPE for an exported KArSL gesture snapshot.
 *
 * Two jobs:
 *   1. Decide FRESHNESS -- is the classification on screen actually the verdict
 *      for the gesture whose landmarks are being exported, or is it a previous
 *      gesture's result still sitting in the latched slot?
 *   2. Serialise it, so the browser download and any offline replay read the
 *      same JSON rather than two ad-hoc shapes.
 *
 * Privacy: landmark COORDINATES ({x,y,z} per point, normalised image space)
 * and numeric recognition metrics only. No camera frames, no video, no audio,
 * no credentials, no user identifiers. A landmark array is numbers, not an
 * image of anything.
 */
import type { FrameTelemetry } from "./recognitionPipeline";
import type { SidedSequence, SignerSide, HandLandmarks } from "@/types/landmarks";

/**
 * How the classification relates to the exported landmarks.
 *
 * `pending` is representable even though the recognizer is synchronous: it
 * means a gesture has completed and been identified but no classification is
 * available for it -- which is exactly the state after a gesture was DISCARDED
 * by segmentation, where lastDiagnostics is deliberately cleared.
 */
export type GestureSnapshotFreshness =
  | "fresh" // shown classification is this gesture's verdict
  | "stale" // shown classification belongs to an EARLIER gesture
  | "pending" // this gesture completed; no verdict yet
  | "no-gesture"; // nothing completed yet this session

export interface GestureCoverageSide {
  /** Frames this side was actually detected in. */
  observed: number | null;
  /** Frames on the gesture's shared timeline -- the coverage DENOMINATOR.
   *  Deliberately NOT the resampled length of 30. */
  total: number | null;
}

export interface GestureDiagnosticsSnapshot {
  /** Bumped by the app when the snapshot schema changes. */
  schemaVersion: 1;
  exportedAt: string;
  freshness: GestureSnapshotFreshness;
  /** Stable per-session gesture id, or null before any gesture completes. */
  gestureSequence: number | null;
  /** Which gesture the classification below belongs to. */
  classificationSequence: number | null;
  completedAtMs: number | null;
  privacy: "landmark-coordinates-and-metrics-only";

  gesture: {
    reason: string | null;
    frameCount: number | null;
    durationMs: number | null;
    motionEnergy: number | null;
    discarded: boolean | null;
    discardReason: string | null;
  };

  registry: { loaded: boolean | null; entryCount: number | null };

  classification: {
    outcome: string | null;
    accepted: boolean | null;
    rejectionReason: string | null;
    bestSignId: string | null;
    bestText: string | null;
    bestDistance: number | null;
    secondBestSignId: string | null;
    secondBestDistance: number | null;
    bestToSecondBestRatio: number | null;
    distanceThreshold: number | null;
    ratioThreshold: number | null;
    distancePassed: boolean | null;
    ratioPassed: boolean | null;
    sentToTranscript: boolean | null;
  };

  perSide: {
    left: number | null;
    right: number | null;
    relation: number | null;
    sharedSides: number | null;
    presenceMismatches: number | null;
    queryOrientation: string | null;
  };

  coverage: {
    query: { left: GestureCoverageSide; right: GestureCoverageSide };
    template: { left: GestureCoverageSide; right: GestureCoverageSide };
  };

  sides: {
    source: string | null;
    confidence: number | null;
    swapsThisGesture: number | null;
    swapsSessionTotal: number | null;
  };

  freshnessDetail: {
    framesSinceLastClassification: number | null;
    phase: string | null;
  };

  /** x/y/z per point, normalised, temporal order preserved. Null before any
   *  gesture completes, and null when the gesture took the unknown-side path.
   *  A frame is {left, right}; either may be null. */
  landmarks: SidedSequence | null;
  /**
   * Landmarks for the OTHER completion path: one hand, side undetermined.
   * Null whenever `landmarks` is set. Kept separate rather than forced into a
   * left/right slot, so the export never claims a side the data does not have.
   */
  landmarksUnknownSide: HandLandmarks[] | null;
}

/**
 * Freshness is decided by comparing two SEQUENCE NUMBERS, never by elapsed
 * frames. Frame age says "this reading is old"; sequence says "this reading is
 * about a different gesture", which is the question that actually matters when
 * the landmarks being exported are the newest ones.
 */
export function resolveFreshness(t: FrameTelemetry | null): {
  freshness: GestureSnapshotFreshness;
  gestureSequence: number | null;
  classificationSequence: number | null;
} {
  const gestureSequence = t?.lastGestureSequence ?? null;
  const classificationSequence = t?.lastClassificationSequence ?? null;

  if (gestureSequence === null) {
    return { freshness: "no-gesture", gestureSequence: null, classificationSequence };
  }
  if (classificationSequence === null) {
    // A gesture completed and was identified, but carries no verdict: either
    // discarded before classification, or not yet classified.
    return { freshness: "pending", gestureSequence, classificationSequence: null };
  }
  if (classificationSequence !== gestureSequence) {
    return { freshness: "stale", gestureSequence, classificationSequence };
  }
  return { freshness: "fresh", gestureSequence, classificationSequence };
}

function side(
  observed: number | null | undefined,
  total: number | null | undefined
): GestureCoverageSide {
  // Both terms are reported as measured. observed 0 with total 40 is a real
  // and useful reading -- that side was never detected -- so it is NOT
  // collapsed to null the way an unknown would be. `null` means "not
  // available", never "zero" and never a fabricated fraction.
  return { observed: observed ?? null, total: total ?? null };
}

/**
 * Builds the snapshot. Every field is read from telemetry; nothing is derived,
 * estimated or defaulted. Unavailable values are `null` and carry no filler.
 */
export function buildGestureDiagnosticsSnapshot(t: FrameTelemetry | null): GestureDiagnosticsSnapshot {
  const resolved = resolveFreshness(t);
  const qFrames = t?.lastDynamicQuerySideFrames;
  const tFrames = t?.lastDynamicBestTemplateSideFrames;
  const sides: Record<SignerSide, number | null> = {
    left: t?.lastDynamicBestPerSideDistance?.left ?? null,
    right: t?.lastDynamicBestPerSideDistance?.right ?? null,
  };

  return {
    schemaVersion: 1,
    exportedAt: new Date().toISOString(),
    freshness: resolved.freshness,
    gestureSequence: resolved.gestureSequence,
    classificationSequence: resolved.classificationSequence,
    completedAtMs: t?.lastCompletedGestureAtMs ?? null,
    privacy: "landmark-coordinates-and-metrics-only",

    gesture: {
      reason: t?.lastGestureEndReason ?? null,
      frameCount: t?.lastGestureFrameCount || null,
      durationMs: t?.lastGestureDurationMs ?? null,
      motionEnergy: t?.lastGestureMotionEnergy ?? null,
      discarded: t?.gestureDiscardedThisFrame ?? null,
      discardReason: t?.gestureDiscardReason ?? null,
    },

    registry: {
      loaded: t?.dynamicRegistryLoaded ?? null,
      entryCount: t?.dynamicRegistryEntryCount ?? null,
    },

    classification: {
      outcome: t?.dynamicOutcome ?? null,
      accepted: t?.lastDynamicAccepted ?? null,
      rejectionReason: t?.lastDynamicRejectionReason ?? null,
      bestSignId: t?.lastDynamicBestSignId ?? null,
      bestText: t?.lastDynamicBestText ?? null,
      bestDistance: t?.lastDynamicBestDistance ?? null,
      secondBestSignId: t?.lastDynamicSecondBestSignId ?? null,
      secondBestDistance: t?.lastDynamicSecondBestDistance ?? null,
      bestToSecondBestRatio: t?.lastDynamicBestToSecondBestRatio ?? null,
      distanceThreshold: t?.dynamicDistanceThreshold ?? null,
      ratioThreshold: t?.dynamicRatioThreshold ?? null,
      distancePassed: t?.lastDynamicDistanceThresholdPassed ?? null,
      ratioPassed: t?.lastDynamicRatioThresholdPassed ?? null,
      sentToTranscript: t?.dynamicResultSentToTranscript ?? null,
    },

    perSide: {
      left: sides.left,
      right: sides.right,
      relation: t?.lastDynamicBestRelationDistance ?? null,
      sharedSides: t?.lastDynamicBestSharedSides ?? null,
      presenceMismatches: t?.lastDynamicBestPresenceMismatches ?? null,
      queryOrientation: t?.lastDynamicBestOrientation ?? null,
    },

    coverage: {
      query: {
        left: side(qFrames?.left, t?.lastDynamicQueryTotalFrames),
        right: side(qFrames?.right, t?.lastDynamicQueryTotalFrames),
      },
      template: {
        left: side(tFrames?.left, t?.lastDynamicBestTemplateTotalFrames),
        right: side(tFrames?.right, t?.lastDynamicBestTemplateTotalFrames),
      },
    },

    sides: {
      source: t?.lastDynamicSideSource ?? null,
      confidence: t?.lastDynamicSideConfidence ?? null,
      swapsThisGesture: t?.lastGestureTrackerSwitches ?? null,
      swapsSessionTotal: t?.lastDynamicTrackerSwitches ?? null,
    },

    freshnessDetail: {
      framesSinceLastClassification: t?.framesSinceLastClassification ?? null,
      phase: t?.dynamicPhase ?? null,
    },

    landmarks: t?.lastCompletedGestureSided ?? null,
    landmarksUnknownSide: t?.lastCompletedUnknownSideHand ?? null,
  };
}

/** Suggested download filename. Sortable, and names the gesture it came from. */
export function snapshotFilename(snap: GestureDiagnosticsSnapshot): string {
  const seq = snap.gestureSequence ?? "none";
  const stamp = snap.exportedAt.replace(/[:.]/g, "-").slice(0, 19);
  return `karsl-gesture-${seq}-${stamp}.json`;
}