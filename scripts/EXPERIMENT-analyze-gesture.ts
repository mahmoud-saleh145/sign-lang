/**
 * EXPERIMENT -- NOT PRODUCTION. Read-only analysis of an exported gesture
 * snapshot. Produces no scoring changes and touches no production code.
 */
import { readFileSync } from "fs";
import path from "path";
import type { HandLandmarks } from "../types/landmarks";

const FILE =
    process.argv[2] ??
    "C:/Users/mahmoud/Downloads/karsl-gesture-30-2026-10-13T21-57-01.json";

interface Frame {
    left: HandLandmarks | null;
    right: HandLandmarks | null;
}
interface Snap {
    gestureSequence: number | null;
    freshness: string;
    gesture: Record<string, unknown>;
    perSide: Record<string, number | null>;
    coverage: Record<string, Record<string, Record<string, number | null>>>;
    landmarks: Frame[] | null;
    landmarksUnknownSide: HandLandmarks[] | null;
}

const snap = JSON.parse(readFileSync(FILE, "utf-8")) as Snap;
const frames = snap.landmarks ?? [];

const wrist = (h: HandLandmarks): { x: number; y: number } => ({ x: h[0].x, y: h[0].y });
const dist = (a: { x: number; y: number }, b: { x: number; y: number }): number =>
    Math.hypot(a.x - b.x, a.y - b.y);
const size = (h: HandLandmarks): number =>
    Math.hypot(h[9].x - h[0].x, h[9].y - h[0].y);

console.log(`file      ${path.basename(FILE)}`);
console.log(`gesture   #${snap.gestureSequence} freshness=${snap.freshness} frames=${frames.length}`);
console.log(`perSide   ${JSON.stringify(snap.perSide)}`);
console.log(`coverage  ${JSON.stringify(snap.coverage)}`);

for (const side of ["left", "right"] as const) {
    const present = frames.map((f, i) => (f[side] ? i : -1)).filter((i) => i >= 0);
    console.log(`\n--- ${side.toUpperCase()} ---`);
    if (present.length === 0) {
        console.log("  never present");
        continue;
    }
    console.log(`  present in ${present.length}/${frames.length} frames`);
    console.log(`  first=${present[0]} last=${present[present.length - 1]}`);
    const gaps: string[] = [];
    let run = 1;
    for (let i = 1; i < present.length; i++) {
        if (present[i] === present[i - 1] + 1) run++;
        else {
            gaps.push(`${present[i - 1]}+${run}`);
            run = 1;
        }
    }
    gaps.push(`${present[present.length - 1]}+${run}`);
    console.log(`  contiguous runs (end+len): ${gaps.join(" ")}`);
    console.log(`  -> ${gaps.length === 1 ? "ONE contiguous block" : `SPLIT into ${gaps.length} blocks`}`);

    // Frame-to-frame continuity among consecutive PRESENT frames.
    const steps: number[] = [];
    for (let i = 1; i < present.length; i++) {
        const a = wrist(frames[present[i - 1]]![side]!);
        const b = wrist(frames[present[i]]![side]!);
        steps.push(dist(a, b));
    }
    steps.sort((a, b) => a - b);
    const p = (q: number): string => (steps.length ? steps[Math.floor(q * (steps.length - 1))].toFixed(4) : "n/a");
    console.log(`  wrist step: min=${steps[0]?.toFixed(4)} p50=${p(0.5)} p90=${p(0.9)} max=${steps[steps.length - 1]?.toFixed(4)}`);
    const big = steps.filter((s) => s > 0.05).length;
    console.log(`  steps > 0.05 (discontinuity scale): ${big}/${steps.length}`);

    // Hand size over time -- a real hand keeps a roughly constant apparent size.
    const sizes = present.map((i) => size(frames[i]![side]!));
    const smin = Math.min(...sizes);
    const smax = Math.max(...sizes);
    const mean = sizes.reduce((a, b) => a + b, 0) / sizes.length;
    console.log(`  hand size: min=${smin.toFixed(4)} mean=${mean.toFixed(4)} max=${smax.toFixed(4)} ratio=${(smax / (smin || 1e-9)).toFixed(2)}`);

    // Overall travel -- how far the wrist moves across the whole gesture.
    const all = present.map((i) => wrist(frames[i]![side]!));
    let travel = 0;
    for (let i = 1; i < all.length; i++) travel += dist(all[i - 1], all[i]);
    console.log(`  total wrist travel: ${travel.toFixed(4)} (mean step ${(travel / (all.length || 1)).toFixed(4)})`);
}

// Inter-hand geometry over the frames where BOTH were seen.
const both = frames.map((f, i) => (f.left && f.right ? i : -1)).filter((i) => i >= 0);
console.log(`\n--- RELATION (frames with both hands: ${both.length}) ---`);
if (both.length > 1) {
    const seps = both.map((i) => dist(wrist(frames[i]!.left!), wrist(frames[i]!.right!)));
    const smin = Math.min(...seps);
    const smax = Math.max(...seps);
    const mean = seps.reduce((a, b) => a + b, 0) / seps.length;
    console.log(`  wrist separation: min=${smin.toFixed(4)} mean=${mean.toFixed(4)} max=${smax.toFixed(4)}`);
    // A crossing hands would drive separation through zero.
    console.log(`  ever crossed (sep < 0.02): ${seps.some((s) => s < 0.02) ? "YES" : "no"}`);
    // Identity check: are the two tracks ever the SAME hand duplicated?
    let dup = 0;
    for (const i of both) {
        const a = frames[i]!.left!;
        const b = frames[i]!.right!;
        if (Math.abs(a[0].x - b[0].x) < 0.01 && Math.abs(a[0].y - b[0].y) < 0.01) dup++;
    }
    console.log(`  frames where L and R wrists coincide: ${dup}/${both.length}`);
}