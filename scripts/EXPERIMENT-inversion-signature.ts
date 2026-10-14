/**
 * EXPERIMENT -- NOT PRODUCTION. Reproduces the SHAPE of the reported failure
 * on REAL registry data, without touching scoring, gates or thresholds.
 *
 * The reported gesture showed one side at 0.791 (near-perfect) and the other
 * at 3.290 -- worse than the documented different-sign p10 of 3.051. One side
 * excellent + one side catastrophic is the signature of a LEFT/RIGHT INVERSION,
 * not of a poor capture (poor capture degrades both sides).
 *
 * But lib/ml/blindQueryOrientation.test.ts:176 pins that a geometry-backed
 * query is NEVER re-oriented, so `bestOrientation` reports "as-named" whether
 * or not the capture is mirrored -- the inversion is invisible to telemetry.
 *
 * This measures what an inverted two-handed gesture ACTUALLY costs, using the
 * stored template of the very sign that was reported (0154) against its own
 * prototypes. Deterministic, read-only, seconds to run.
 */
import { readFileSync } from "fs";
import path from "path";
import {
    prepareSidedSequence,
    sidedDistance,
} from "../lib/ml/twoHandMatch";
import { decodePrototypeFile } from "../lib/ml/prototypeStore";
import { entryToSidedSequence, withLegacySequence } from "../lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "../lib/sign-language/signRegistry";
import type { SidedSequence } from "../types/landmarks";

const ROOT = path.resolve(__dirname, "..");

/** Mirrors dynamicSignRecognizer.ts swapSides(). */
function swapSides(frames: SidedSequence): SidedSequence {
    return frames.map((f) => ({ left: f.right, right: f.left }));
}

const registry = (
    JSON.parse(
        readFileSync(path.join(ROOT, "public/models/sign-vocabulary/dedicated-signs.json"), "utf-8")
    ) as DedicatedSignEntry[]
).map(withLegacySequence);

const proto = decodePrototypeFile(
    JSON.parse(
        readFileSync(path.join(ROOT, "public/models/sign-vocabulary/sign-prototypes.eval.json"), "utf-8")
    )
);

function report(signId: string): void {
    const entry = registry.find((e) => e.sourceSignId === signId);
    if (!entry) {
        console.log(`${signId}: not in registry`);
        return;
    }
    const sided = entryToSidedSequence(entry);
    const takes = proto?.bySign.get(signId) ?? [];
    if (!sided || takes.length === 0) {
        console.log(`${signId}: no usable sided sequence or no prototypes`);
        return;
    }

    console.log("=".repeat(96));
    console.log(`SIGN ${signId}  "${entry.text}"`);
    console.log(
        `  stored template: ${sided.length} frames, ` +
            `L in ${sided.filter((f) => f.left).length}, R in ${sided.filter((f) => f.right).length}` +
            `   prototypes: ${takes.length}`
    );

    for (const [label, seq] of [
        ["as named", sided],
        ["LEFT/RIGHT SWAPPED", swapSides(sided)],
    ] as Array<[string, SidedSequence]>) {
        const q = prepareSidedSequence(seq);
        let best = Infinity;
        let bestProto = -1;
        const bests: Array<{ d: number; l: number | null; r: number | null }> = [];
        for (let i = 0; i < takes.length; i++) {
            const d = sidedDistance(q, takes[i]);
            bests.push({ d: d.distance, l: d.perSide.left, r: d.perSide.right });
            if (d.distance < best) {
                best = d.distance;
                bestProto = i;
            }
        }
        bests.sort((a, b) => a.d - b.d);
        const b = bests[0];
        console.log(
            `  ${label.padEnd(20)} best=${b.d.toFixed(5)}  L=${b.l?.toFixed(3) ?? "n/a"}  R=${b.r?.toFixed(3) ?? "n/a"}` +
                `  sharedSides=${sidedDistance(q, takes[bestProto]).sharedSides}`
        );
        if (label !== "as named") {
            console.log(
                `  ${"".padEnd(20)} 3 best prototype distances: ${bests.slice(0, 3).map((x) => x.d.toFixed(4)).join(", ")}`
            );
        }
    }
    console.log();
}

console.log("INVERSION-SIGNATURE PROBE (read-only, production scoring unchanged)\n");
for (const id of process.argv.slice(2).length ? process.argv.slice(2) : ["0154", "0302"]) {
    report(id);
}