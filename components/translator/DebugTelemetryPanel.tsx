import type { FrameTelemetry } from "@/lib/ml/recognitionPipeline";
import type { ModelMetadata } from "@/types/features";

export default function DebugTelemetryPanel({
  telemetry,
  metadata,
}: {
  telemetry: FrameTelemetry | null;
  metadata: ModelMetadata | null;
}) {
  return (
    <section
      className="rounded-2xl bg-black/60 border border-white/10 p-4 font-mono text-xs text-[var(--color-text-dim)] space-y-1"
      aria-label="Developer telemetry"
    >
      <Row label="hand detected" value={String(telemetry?.handDetected ?? "-")} />
      <Row label="raw label" value={telemetry?.rawLabel ?? "null"} />
      <Row
        label="raw confidence"
        value={telemetry ? telemetry.rawConfidence.toFixed(4) : "-"}
      />
      <Row label="segmentation state" value={telemetry?.segmentationState ?? "-"} />
      <Row label="current candidate" value={telemetry?.currentCandidate ?? "null"} />
      <Row label="committed this frame" value={telemetry?.committedThisFrame ?? "null"} />
      <Row label="transcript length" value={String(telemetry?.transcriptLength ?? 0)} />
      <Row label="arabic text" value={telemetry?.arabicText || "(empty)"} />
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

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-4">
      <span>{label}</span>
      <span className="text-[var(--color-text)]">{value}</span>
    </div>
  );
}
