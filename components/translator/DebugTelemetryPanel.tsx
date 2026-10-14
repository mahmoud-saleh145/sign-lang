import type { FrameTelemetry } from "@/lib/ml/recognitionPipeline";
import type { ModelMetadata } from "@/types/features";
import {
  buildGestureDiagnosticsSnapshot,
  snapshotFilename,
  resolveFreshness,
} from "@/lib/ml/gestureDiagnosticsSnapshot";

/** Freshness of the reading, stated in words rather than inferred from numbers. */
function freshnessLabel(t: FrameTelemetry | null): string {
  const { freshness, gestureSequence, classificationSequence } = resolveFreshness(t);
  switch (freshness) {
    case "no-gesture":
      return "no gesture completed yet";
    case "pending":
      return `pending (gesture ${gestureSequence} completed, no verdict yet)`;
    case "stale":
      return `STALE (verdict is for gesture ${classificationSequence}, latest is ${gestureSequence})`;
    case "fresh":
      return `fresh (gesture ${gestureSequence})`;
  }
}

/**
 * Downloads the snapshot as a file. A browser download is the honest channel
 * here: the landmarks exist only in the page's runtime, so a Node script could
 * not reach them. Blob + object URL, no new dependency.
 */
function downloadGestureDiagnostics(t: FrameTelemetry | null): void {
  const snap = buildGestureDiagnosticsSnapshot(t);
  const blob = new Blob([JSON.stringify(snap, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = snapshotFilename(snap);
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  // Revoke on the next tick: revoking synchronously can cancel the download in
  // some browsers before it has started reading the blob.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}

/** Every value here is a REAL runtime measurement taken from the pipeline on
 * the current frame -- nothing is derived or estimated for display. The
 * point of the KArSL section is that when a sign is not recognized, the
 * panel says which stage stopped it, rather than showing a blank result. */
export default function DebugTelemetryPanel({
  telemetry,
  metadata,
}: {
  telemetry: FrameTelemetry | null;
  metadata: ModelMetadata | null;
}) {
  const t = telemetry;
  const num = (v: number | null | undefined, digits = 5) =>
    v === null || v === undefined ? "-" : v.toFixed(digits);
  const bool = (v: boolean | null | undefined) => (v === null || v === undefined ? "-" : String(v));

  return (
    <section
      className="rounded-2xl bg-black/60 border border-white/10 p-4 font-mono text-xs text-[var(--color-text-dim)] space-y-1 overflow-x-auto"
      aria-label="Developer telemetry"
      aria-live="polite"
    >
      <Row label="mode" value={t?.mode.toUpperCase() ?? "-"} highlight />
      <Row label="hand detected" value={bool(t?.handDetected)} />

      <Divider label="Static letter model" />
      <Row label="raw label" value={t?.rawLabel ?? "null"} />
      <Row label="raw confidence" value={num(t?.rawConfidence, 4)} />
      <Row label="segmentation state" value={t?.segmentationState ?? "-"} />
      <Row label="current candidate" value={t?.currentCandidate ?? "null"} />
      <Row label="committed this frame" value={t?.committedThisFrame ?? "null"} />
      <Row
        label="letter commit suppressed (word mode)"
        value={bool(t?.letterCommitSuppressed)}
        highlight={t?.letterCommitSuppressed}
      />
      <Row label="letter transcript length" value={String(t?.transcriptLength ?? 0)} />
      <Row label="arabic text" value={t?.arabicText || "(empty)"} />

      <Divider label="KArSL word mode — registry" />
      <Row label="registry loaded" value={bool(t?.dynamicRegistryLoaded)} />
      <Row label="registry entries" value={String(t?.dynamicRegistryEntryCount ?? 0)} />

      <Divider label="KArSL word mode — gesture" />
      <Row label="phase" value={t?.dynamicPhase ?? "-"} highlight />
      <Row label="buffer state" value={t?.dynamicGestureState ?? "-"} />
      <Row label="manual capture open" value={bool(t?.dynamicManualCapture)} />
      <Row label="live buffered frames" value={String(t?.dynamicGestureFrameCount ?? 0)} />
      <Row label="completed gesture frames" value={String(t?.lastGestureFrameCount ?? 0)} />
      <Row
        label="completed gesture duration"
        value={t?.lastGestureDurationMs != null ? `${Math.round(t.lastGestureDurationMs)}ms` : "-"}
      />
      <Row label="completed gesture motion energy" value={num(t?.lastGestureMotionEnergy)} />
      <Row label="gesture end reason" value={t?.lastGestureEndReason ?? "-"} />
      <Row
        label="truncated by frame cap"
        value={bool(t?.lastGestureEndedByFrameCap)}
        highlight={t?.lastGestureEndedByFrameCap}
      />
      <Row
        label="gesture DISCARDED before classify"
        value={bool(t?.gestureDiscardedThisFrame)}
        highlight={t?.gestureDiscardedThisFrame}
      />
      <Row
        label="discard reason"
        value={t?.gestureDiscardReason ?? "-"}
        highlight={Boolean(t?.gestureDiscardReason)}
      />

      <Divider label="KArSL word mode — classification" />
      {/* OUTCOME FIRST. This row answers "what happened to my sign?" on its
          own. Before it existed the panel could show "classifier running no"
          beside a good 45-frame gesture, which read as "the classifier never
          ran" when in fact it had run, rejected, and had its result dropped
          one frame later. */}
      <Row
        label="OUTCOME (last gesture)"
        value={t?.dynamicOutcome ?? "-"}
        highlight
      />
      <Row
        label="result age"
        value={
          t?.framesSinceLastClassification == null
            ? "no classification yet"
            : t.framesSinceLastClassification === 0
              ? "this frame"
              : `${t.framesSinceLastClassification} frames ago (previous gesture)`
        }
        highlight={(t?.framesSinceLastClassification ?? 0) > 0}
      />
      <Row
        label="classifier running"
        value={t?.dynamicPhase === "classifying" ? "yes (this frame)" : "no"}
      />
      <Row label="candidate count" value={t?.lastDynamicCandidateCount != null ? String(t.lastDynamicCandidateCount) : "-"} />
      <Row label="best sign ID" value={t?.lastDynamicBestSignId ?? "-"} />
      <Row label="best sign text" value={t?.lastDynamicBestText ?? "-"} />
      <Row label="best DTW distance" value={num(t?.lastDynamicBestDistance)} />
      <Row label="2nd-best sign ID" value={t?.lastDynamicSecondBestSignId ?? "-"} />
      <Row label="2nd-best DTW distance" value={num(t?.lastDynamicSecondBestDistance)} />
      <Row label="best/2nd-best ratio" value={num(t?.lastDynamicBestToSecondBestRatio, 4)} />
      <Row label="distance threshold" value={num(t?.dynamicDistanceThreshold, 3)} />
      <Row label="ratio threshold" value={num(t?.dynamicRatioThreshold, 3)} />
      <Row label="distance passed" value={bool(t?.lastDynamicDistanceThresholdPassed)} />
      <Row label="ratio passed" value={bool(t?.lastDynamicRatioThresholdPassed)} />
      <Row
        label="FINAL DECISION"
        value={
          t?.lastDynamicAccepted === null || t?.lastDynamicAccepted === undefined
            ? "-"
            : t.lastDynamicAccepted
              ? "ACCEPTED"
              : "REJECTED"
        }
        highlight
      />
      <Row label="rejection reason" value={t?.lastDynamicRejectionReason ?? "-"} />
      <Row
        label="hand sides from"
        value={
          t?.lastDynamicSideSource
            ? `${t.lastDynamicSideSource} (${(t.lastDynamicSideConfidence ?? 0).toFixed(2)})`
            : "-"
        }
      />
      <Row label="side undetermined" value={bool(t?.lastDynamicSideUnknown)} />
      <Row
        label="captured frames L/R"
        value={
          t ? `${t.lastDynamicQuerySideFrames.left} / ${t.lastDynamicQuerySideFrames.right}` : "-"
        }
      />
      <Row
        label="best distance L/R"
        value={
          t
            ? `${t.lastDynamicBestPerSideDistance.left?.toFixed(3) ?? "-"} / ${t.lastDynamicBestPerSideDistance.right?.toFixed(3) ?? "-"}`
            : "-"
        }
      />
      <Row label="sides compared" value={t?.lastDynamicBestSharedSides != null ? String(t.lastDynamicBestSharedSides) : "-"} />
      <Row label="sides only one had" value={t?.lastDynamicBestPresenceMismatches != null ? String(t.lastDynamicBestPresenceMismatches) : "-"} />
      {/* Diagnostic only. "swapped" means the capture's left/right naming was
          mirrored relative to the stored prototypes, which looks exactly like
          a poor match in every other field. Surfaced so the two are not
          confused when reading a rejection. */}
      <Row label="query orientation" value={t?.lastDynamicBestOrientation ?? "-"} />
      {/* The relation track is one of the THREE terms in the combined distance
          (left, right, relation) but was the only one never displayed. */}
      <Row
        label="best distance RELATION"
        value={num(t?.lastDynamicBestRelationDistance)}
      />
      {/* Coverage, with both terms named. Numerator = frames that side was
          actually detected in. Denominator = frames in the captured gesture's
          shared timeline (queryTotalFrames), NOT the resampled length of 30. */}
      <Row
        label="query coverage L (obs/total)"
        value={
          t?.lastDynamicQuerySideFrames && t.lastDynamicQueryTotalFrames
            ? `${t.lastDynamicQuerySideFrames.left}/${t.lastDynamicQueryTotalFrames}`
            : "-"
        }
      />
      <Row
        label="query coverage R (obs/total)"
        value={
          t?.lastDynamicQuerySideFrames && t.lastDynamicQueryTotalFrames
            ? `${t.lastDynamicQuerySideFrames.right}/${t.lastDynamicQueryTotalFrames}`
            : "-"
        }
      />
      {/* Template-side coverage is the DENOMINATOR the shipped weighting
          saturates against (min(query, template)), so both sides are shown. */}
      <Row
        label="template coverage L/R (obs/total)"
        value={
          t?.lastDynamicBestTemplateTotalFrames
            ? `${t.lastDynamicBestTemplateSideFrames.left ?? "-"}/${t.lastDynamicBestTemplateTotalFrames}` +
              `  ${t.lastDynamicBestTemplateSideFrames.right ?? "-"}/${t.lastDynamicBestTemplateTotalFrames}`
            : "-"
        }
      />
      <Row label="tracker swaps (last gesture)" value={t?.lastGestureTrackerSwitches != null ? String(t.lastGestureTrackerSwitches) : "-"} />
      <Row label="tracker swaps (session total)" value={t?.lastDynamicTrackerSwitches != null ? String(t.lastDynamicTrackerSwitches) : "-"} />
      <Row label="match this frame" value={t?.dynamicMatchThisFrame ?? "null"} />
      <Row label="result sent to transcript" value={bool(t?.dynamicResultSentToTranscript)} />

      {/* Association, stated before any verdict: is the reading below about
          THIS gesture or an earlier one? Sequence numbers, not frame age. */}
      <Row
        label="gesture / verdict association"
        value={freshnessLabel(t)}
        highlight={resolveFreshness(t).freshness === "stale"}
      />
      <Row
        label="landmarks available for export"
        value={
          t?.lastCompletedGestureSided
            ? `${t.lastCompletedGestureSided.length} frames (two-hand)`
            : t?.lastCompletedUnknownSideHand
              ? `${t.lastCompletedUnknownSideHand.length} frames (one hand, side undetermined)`
              : "none yet"
        }
      />

      {/* Writes the typed snapshot to a file. The landmarks exist only in this
          page's runtime, so a browser download is the honest channel -- a Node
          script could not reach them. Landmarks and metrics only: no camera
          frames, no video, no audio, no credentials, no identifiers. */}
      <button
        type="button"
        onClick={() => downloadGestureDiagnostics(t)}
        disabled={!t?.lastCompletedGestureSided && !t?.lastCompletedUnknownSideHand}
        style={{ marginTop: 12, padding: "4px 8px", fontSize: 11, cursor: "pointer" }}
      >
        Export latest gesture diagnostics (JSON)
      </button>

      <Divider label="Static model metadata" />
      <Row label="confidence threshold" value={metadata ? String(metadata.confidenceThreshold) : "-"} />
      <Row
        label="stability window"
        value={
          metadata
            ? `min ${metadata.temporalWindow.minStableFrames}f / gap ${metadata.temporalWindow.maxGapFrames}f`
            : "-"
        }
      />
      <Row label="model version" value={metadata?.modelVersion ?? "-"} />
      <Row
        label="test accuracy (not real-world)"
        value={metadata ? `${(metadata.evaluation.testAccuracy ?? 0) * 100}%` : "-"}
      />
      <Row label="signer independent" value={metadata ? String(metadata.evaluation.signerIndependent) : "-"} />
    </section>
  );
}

function Row({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
  return (
    <div className="flex justify-between gap-4">
      <span>{label}</span>
      <span className={highlight ? "text-[var(--color-accent-active)] font-semibold" : "text-[var(--color-text)]"}>
        {value}
      </span>
    </div>
  );
}

function Divider({ label }: { label: string }) {
  return (
    <div className="pt-2 mt-2 border-t border-white/10 text-[var(--color-accent-active)] uppercase tracking-wide">
      {label}
    </div>
  );
}
