"use client";

/**
 * CAMERA-FREE DEVELOPER TEST PAGE for the KArSL dynamic recognizer.
 *
 * Purpose: separate "the classifier/data pipeline is broken" from "the live
 * camera capture/segmentation is broken". Everything on this page runs the
 * EXACT production classifier (lib/ml/dynamicSignRecognizer.ts) and the
 * EXACT production gesture-buffer gates (lib/ml/dynamicGestureBuffer.ts)
 * against REAL stored KArSL landmark sequences from
 * public/models/sign-vocabulary/dedicated-signs.json.
 *
 * NOTHING HERE IS SYNTHESIZED. The variants are re-timings and crops of the
 * real recorded sequence, plus a horizontal mirror -- all transformations a
 * live camera genuinely introduces. No motion is invented, and no landmark
 * value is authored by hand.
 *
 * HONEST CAVEAT, stated in the UI too: the "identity" case feeds a stored
 * sequence back against a registry that contains that same sequence, so a
 * distance of ~0 is expected and proves only that loading, normalization,
 * resampling and DTW are wired up. The speed/crop variants are the ones
 * that say something about robustness.
 */

import { useCallback, useEffect, useState } from "react";
import {
  listDynamicSignRegistry,
  getStoredSidedSequence,
  diagnoseDynamicSign,
  MAX_NORMALIZED_DISTANCE,
  MAX_BEST_TO_SECOND_BEST_RATIO,
  type DynamicSignRegistrySummary,
  type DynamicSignDiagnostics,
} from "@/lib/ml/dynamicSignRecognizer";
import {
  DEFAULT_GESTURE_BUFFER_CONFIG,
  computeMotionEnergy,
} from "@/lib/ml/dynamicGestureBuffer";
import type { HandLandmarks, SidedSequence } from "@/types/landmarks";

/** Linear-interpolated re-timing of one hand of a REAL sequence. Changes frame
 * density only -- the same recorded path, walked at a different speed. */
function retimeHand(seq: HandLandmarks[], target: number): HandLandmarks[] {
  const out: HandLandmarks[] = [];
  for (let i = 0; i < target; i++) {
    const t = (i / (target - 1)) * (seq.length - 1);
    const lo = Math.floor(t);
    const hi = Math.min(lo + 1, seq.length - 1);
    const f = t - lo;
    out.push(seq[lo].map((p, k) => ({
      x: p.x + (seq[hi][k].x - p.x) * f,
      y: p.y + (seq[hi][k].y - p.y) * f,
      z: p.z + (seq[hi][k].z - p.z) * f,
    })));
  }
  return out;
}

/**
 * Re-times a two-hand sequence, keeping both hands on ONE shared timeline.
 *
 * A side is re-timed only where it was actually detected, and stays null
 * everywhere it was not -- resampling never fills a gap with an invented hand.
 */
function retime(seq: SidedSequence, factor: number): SidedSequence {
  const target = Math.max(2, Math.round(seq.length * factor));
  const out: SidedSequence = [];
  const perSide: Record<"left" | "right", HandLandmarks[] | null> = { left: null, right: null };
  const presence: Record<"left" | "right", boolean[]> = { left: [], right: [] };
  for (const side of ["left", "right"] as const) {
    const present = seq.map((f) => f[side]).filter((f): f is HandLandmarks => f !== null);
    perSide[side] = present.length >= 2 ? retimeHand(present, target) : null;
    presence[side] = seq.map((f) => f[side] !== null);
  }
  for (let i = 0; i < target; i++) {
    const src = Math.min(seq.length - 1, Math.round((i / (target - 1)) * (seq.length - 1)));
    out.push({
      left: presence.left[src] && perSide.left ? perSide.left[i] : null,
      right: presence.right[src] && perSide.right ? perSide.right[i] : null,
    });
  }
  return out;
}

/**
 * SWAPS the two hands of a real sequence.
 *
 * This replaces the old "mirrored" variant. Mirroring tested whether the
 * matcher could cope with not knowing which hand it was looking at; now that
 * both hands carry an anatomical identity, the useful check is the opposite
 * one -- a sign performed with the hands the wrong way round must NOT be
 * accepted as that sign. Both hands here are real recorded motion; only their
 * names are exchanged.
 */
function swapHands(seq: SidedSequence): SidedSequence {
  return seq.map((f) => ({ left: f.right, right: f.left }));
}

interface VariantResult {
  name: string;
  note: string;
  frames: number;
  motionEnergy: number;
  passesBufferGates: boolean;
  bufferGateReason: string;
  diag: DynamicSignDiagnostics | null;
  correct: boolean;
}

export default function KarslDevPage() {
  const [registry, setRegistry] = useState<DynamicSignRegistrySummary[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [results, setResults] = useState<VariantResult[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sweep, setSweep] = useState<string | null>(null);

  useEffect(() => {
    listDynamicSignRegistry()
      .then((r) => {
        setRegistry(r);
        if (r.length > 0) setSelected(r[0].sourceSignId ?? "");
      })
      .catch((e) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  const run = useCallback(async () => {
    if (!selected) return;
    setBusy(true);
    setError(null);
    try {
      const base = await getStoredSidedSequence(selected);
      if (!base) throw new Error(`No stored sequence for sign ${selected}`);

      const variants: Array<{ name: string; note: string; seq: SidedSequence }> = [
        { name: "identity", note: "the stored sequence itself (sanity check only)", seq: base },
        { name: "0.5x speed", note: "same real path, half the frame density", seq: retime(base, 0.5) },
        { name: "2x speed", note: "same real path, double the frame density", seq: retime(base, 2) },
        { name: "first 80%", note: "gesture captured slightly early", seq: base.slice(0, Math.max(2, Math.floor(base.length * 0.8))) },
        { name: "last 80%", note: "gesture captured slightly late", seq: base.slice(Math.floor(base.length * 0.2)) },
        { name: "hands swapped", note: "real motion, left/right exchanged -- must NOT be accepted", seq: swapHands(base) },
      ];

      const out: VariantResult[] = [];
      for (const v of variants) {
        const energy = computeMotionEnergy(v.seq);
        const tooFew = v.seq.length < DEFAULT_GESTURE_BUFFER_CONFIG.minGestureFrames;
        const tooStill = energy < DEFAULT_GESTURE_BUFFER_CONFIG.minMotionEnergy;
        const diag = await diagnoseDynamicSign(v.seq);
        out.push({
          name: v.name,
          note: v.note,
          frames: v.seq.length,
          motionEnergy: energy,
          passesBufferGates: !tooFew && !tooStill,
          bufferGateReason: tooFew ? "too-few-frames" : tooStill ? "no-motion" : "passes",
          diag,
          correct: diag.bestSignId === selected,
        });
      }
      setResults(out);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }, [selected]);

  /** Runs every registry sign through the buffer gates, reporting how many
   * real signs would be thrown away before classification ever happens.
   * This is the check that exposed the original bug. */
  const runGateSweep = useCallback(async () => {
    setBusy(true);
    try {
      let tooStill = 0;
      let tooFew = 0;
      const energies: number[] = [];
      for (const r of registry) {
        if (!r.sourceSignId) continue;
        const seq = await getStoredSidedSequence(r.sourceSignId);
        if (!seq) continue;
        const e = computeMotionEnergy(seq);
        energies.push(e);
        if (seq.length < DEFAULT_GESTURE_BUFFER_CONFIG.minGestureFrames) tooFew++;
        else if (e < DEFAULT_GESTURE_BUFFER_CONFIG.minMotionEnergy) tooStill++;
      }
      energies.sort((a, b) => a - b);
      const q = (p: number) => energies[Math.min(energies.length - 1, Math.floor((p / 100) * (energies.length - 1)))];
      setSweep(
        `${energies.length} real sequences · motion energy min=${q(0).toFixed(6)} ` +
        `p50=${q(50).toFixed(6)} max=${q(100).toFixed(6)} · ` +
        `would be discarded by current gates: ${tooFew} too-few-frames, ${tooStill} no-motion ` +
        `(minMotionEnergy=${DEFAULT_GESTURE_BUFFER_CONFIG.minMotionEnergy}, ` +
        `minGestureFrames=${DEFAULT_GESTURE_BUFFER_CONFIG.minGestureFrames})`
      );
    } finally {
      setBusy(false);
    }
  }, [registry]);

  const selectedEntry = registry.find((r) => r.sourceSignId === selected);

  return (
    <main className="min-h-dvh max-w-3xl mx-auto p-4 flex flex-col gap-4 font-mono text-sm">
      <h1 className="font-display text-xl font-semibold">KArSL dynamic recognizer — camera-free test</h1>
      <p className="text-xs text-[var(--color-text-dim)]">
        Replays REAL stored KArSL landmark sequences through the production classifier. If a sign fails
        here, the classifier or the data is at fault. If it passes here but fails on camera, the live
        capture/segmentation is at fault.
      </p>

      {error && <p className="text-[var(--color-accent-error)]">{error}</p>}

      <div className="flex flex-wrap gap-2 items-center">
        <label htmlFor="sign" className="text-xs text-[var(--color-text-dim)]">
          registry ({registry.length} entries):
        </label>
        <select
          id="sign"
          value={selected}
          onChange={(e) => setSelected(e.target.value)}
          className="bg-[var(--color-surface)] border border-white/10 rounded px-2 py-1 max-w-full"
        >
          {registry.map((r) => (
            <option key={r.sourceSignId ?? r.text} value={r.sourceSignId ?? ""}>
              {r.sourceSignId} — {r.text} ({r.frameCount}f)
            </option>
          ))}
        </select>
        <button
          onClick={run}
          disabled={busy || !selected}
          className="px-4 py-2 rounded-full bg-[var(--color-accent-active)] text-[#0B1220] font-semibold disabled:opacity-40"
        >
          {busy ? "running…" : "Run classifier"}
        </button>
        <button
          onClick={runGateSweep}
          disabled={busy || registry.length === 0}
          className="px-4 py-2 rounded-full border border-white/20 disabled:opacity-40"
        >
          Sweep buffer gates over all signs
        </button>
      </div>

      {selectedEntry && (
        <p className="text-xs text-[var(--color-text-dim)]">
          expected label: <span dir="rtl" lang="ar" className="text-[var(--color-text)]">{selectedEntry.text}</span>
          {" · "}thresholds: distance ≤ {MAX_NORMALIZED_DISTANCE}, best/2nd ratio ≤ {MAX_BEST_TO_SECOND_BEST_RATIO}
        </p>
      )}

      {sweep && <pre className="text-xs whitespace-pre-wrap bg-black/40 p-3 rounded-xl">{sweep}</pre>}

      {results.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead className="text-[var(--color-text-dim)]">
              <tr className="text-left">
                <th className="p-2">variant</th>
                <th className="p-2">frames</th>
                <th className="p-2">motion</th>
                <th className="p-2">gates</th>
                <th className="p-2">best</th>
                <th className="p-2">dist</th>
                <th className="p-2">2nd</th>
                <th className="p-2">ratio</th>
                <th className="p-2">L dist</th>
                <th className="p-2">R dist</th>
                <th className="p-2">decision</th>
              </tr>
            </thead>
            <tbody>
              {results.map((r) => (
                <tr key={r.name} className="border-t border-white/10">
                  <td className="p-2" title={r.note}>{r.name}</td>
                  <td className="p-2">{r.frames}</td>
                  <td className="p-2">{r.motionEnergy.toFixed(5)}</td>
                  <td className={`p-2 ${r.passesBufferGates ? "" : "text-[var(--color-accent-error)]"}`}>
                    {r.bufferGateReason}
                  </td>
                  <td className={`p-2 ${r.correct ? "" : "text-[var(--color-accent-error)]"}`}>
                    {r.diag?.bestSignId ?? "-"}
                  </td>
                  <td className="p-2">{r.diag?.bestDistance?.toFixed(4) ?? "-"}</td>
                  <td className="p-2">{r.diag?.secondBestSignId ?? "-"}</td>
                  <td className="p-2">{r.diag?.bestToSecondBestRatio?.toFixed(3) ?? "-"}</td>
                  <td className="p-2">{r.diag?.bestPerSideDistance.left?.toFixed(3) ?? "-"}</td>
                  <td className="p-2">{r.diag?.bestPerSideDistance.right?.toFixed(3) ?? "-"}</td>
                  <td className={`p-2 ${r.diag?.accepted ? "" : "text-[var(--color-accent-error)]"}`}>
                    {r.diag?.accepted ? "ACCEPTED" : `REJECTED (${r.diag?.rejectionReason})`}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-xs text-[var(--color-text-dim)] mt-2">
            The <em>identity</em> row matches a stored sequence against a registry that contains it, so
            distance ≈ 0 is expected and only proves the pipeline is wired up. The speed and crop rows are
            the meaningful robustness signal.
          </p>
        </div>
      )}
    </main>
  );
}
