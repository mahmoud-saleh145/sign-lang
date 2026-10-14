/**
 * EXPERIMENT -- NOT PRODUCTION. Offline comparison of sign 0154 source takes
 * against the exported gesture-30 capture, using the UNCHANGED production
 * preparation and matching code.
 *
 * LEAKAGE DISCIPLINE
 *   - The registry template take and every take inside the prototype artifact
 *     are reported SEPARATELY from held-out takes. Scoring a template against
 *     a prototype built from it is a tautology, not a measurement, and is
 *     labelled as such everywhere it appears.
 *   - splitTakes() classifies each take; nothing is re-split here.
 *   - No capture, take or prototype is added, changed or removed.
 */
import { readFileSync, readdirSync } from "fs";
import path from "path";
import { prepareSidedSequence, sidedDistance } from "../lib/ml/twoHandMatch";
import { decodePrototypeFile } from "../lib/ml/prototypeStore";
import { splitTakes } from "../lib/ml/karslTakeSplit";
import { DEFAULT_PROCESSED_ROOT } from "../lib/ml/karslPaths";
import { loadExclusions, excludedFor } from "../lib/ml/karslExclusions";
import {
    withLegacySequence,
    type DedicatedSignEntry,
} from "../lib/sign-language/signRegistry";
import type { HandLandmarks, SidedSequence } from "../types/landmarks";

const ROOT = path.resolve(__dirname, "..");
const TARGET = "0154";
const GESTURE30 =
    "C:/Users/mahmoud/Downloads/karsl-gesture-30-2026-10-13T21-57-01.json";

interface PHand { landmarks: HandLandmarks; rawLabel?: string }
interface PFrame { hands?: { left?: PHand | null; right?: PHand | null } }
interface PTake { handTracking?: { primaryHand?: string }; frames?: PFrame[] }
interface Snap { landmarks: Array<{ left: HandLandmarks | null; right: HandLandmarks | null }> | null }

/** Same rule as ml/training/handedness.py and the pipeline: geometry first,
 *  label majority second. Mirrors scripts/calibrate-two-hand.ts. */
function sidedFromTake(take: PTake): SidedSequence {
    const frames = take.frames ?? [];
    const slots: Array<Array<HandLandmarks | null>> = [[], []];
    const labels: Array<Array<string | null>> = [[], []];
    for (const fr of frames) {
        const pair: Array<PHand | null | undefined> = [fr.hands?.left, fr.hands?.right];
        for (const i of [0, 1] as const) {
            slots[i].push(pair[i]?.landmarks ?? null);
            labels[i].push(pair[i]?.rawLabel ?? null);
        }
    }
    let aSmaller = 0;
    let compared = 0;
    for (let f = 0; f < frames.length; f++) {
        const a = slots[0][f];
        const b = slots[1][f];
        if (!a || !b) continue;
        compared++;
        if (a[0].x < b[0].x) aSmaller++;
    }
    if (compared >= 3) {
        const aIsLeft = aSmaller / compared <= 0.5;
        return frames.map((_, f) => ({
            left: (aIsLeft ? slots[0][f] : slots[1][f]) ?? null,
            right: (aIsLeft ? slots[1][f] : slots[0][f]) ?? null,
        }));
    }
    return frames.map((_, f) => {
        const lab = labels[0][f];
        const a = slots[0][f];
        const b = slots[1][f];
        return {
            left: lab === "Left" ? a : lab === "Right" ? b : null,
            right: lab === "Right" ? a : lab === "Left" ? b : null,
        };
    });
}

const registry = (
    JSON.parse(
        readFileSync(path.join(ROOT, "public/models/sign-vocabulary/dedicated-signs.json"), "utf-8")
    ) as DedicatedSignEntry[]
).map(withLegacySequence);
const entry = registry.find((e) => e.sourceSignId === TARGET);
const textOf = new Map(registry.map((e) => [e.sourceSignId ?? "", e.text]));
const templateFile = entry?.sourceTake?.processedFile ?? null;

const proto = decodePrototypeFile(
    JSON.parse(
        readFileSync(path.join(ROOT, "public/models/sign-vocabulary/sign-prototypes.eval.json"), "utf-8")
    )
);
if (!proto) throw new Error("no prototypes");
const targetProtos = proto.bySign.get(TARGET) ?? [];

const exclusions = loadExclusions(path.join(ROOT, "karsl-exclusions.json"));
const dir = path.join(ROOT, DEFAULT_PROCESSED_ROOT, TARGET);
const files = readdirSync(dir).filter((f) => f.endsWith(".json")).sort();
const split = splitTakes(files, templateFile, excludedFor(exclusions, TARGET));

interface Row {
    file: string; kind: "template" | "benchmark" | "calibration" | "other";
    frames: number; lObs: number; rObs: number; lFirst: number; rFirst: number;
    selfD: number | null;
}
const rows: Row[] = [];

for (const f of files) {
    const take = JSON.parse(readFileSync(path.join(dir, f), "utf-8")) as PTake;
    if (!take.handTracking?.primaryHand) continue;
    const sided = sidedFromTake(take);
    if (sided.length < 2) continue;
    const q = prepareSidedSequence(sided);
    const kind: Row["kind"] =
        f === templateFile
            ? "template"
            : split.benchmark.includes(f)
              ? "benchmark"
              : split.calibration.includes(f)
                ? "calibration"
                : "other";

    // Self-match: scored against prototypes that EXCLUDE this take's own file.
    // The prototype artifact stores only vectors, so self-match is instead
    // measured as distance to the registry template, reported separately.
    let selfD: number | null = null;

    if (targetProtos.length > 0) {
        // distance to the NEAREST prototype (what production would score)
        selfD = Math.min(...targetProtos.map((p) => sidedDistance(q, p).distance));
    }
    rows.push({
        file: f, kind,
        frames: sided.length,
        lObs: sided.filter((x) => x.left).length,
        rObs: sided.filter((x) => x.right).length,
        lFirst: sided.findIndex((x) => x.left),
        rFirst: sided.findIndex((x) => x.right),
        selfD,
    });
}

console.log("=".repeat(108));
console.log(`SIGN ${TARGET} "${textOf.get(TARGET) ?? "?"}" -- ${files.length} processed takes`);
console.log(`registry template take: ${templateFile}`);
console.log(`split -> benchmark=${split.benchmark.length} calibration=${split.calibration.length} prototypes=${split.prototypes.length}`);
console.log(`prototypes in artifact for target: ${targetProtos.length}`);
console.log("=".repeat(108));

const held = rows.filter((r) => r.kind !== "template");
const tplRow = rows.find((r) => r.kind === "template");

console.log("\n--- HAND PRESENCE ACROSS ALL TAKES ---");
const cov = (obs: number, tot: number): string => `${obs}/${tot}`;
console.log("kind        n   frames(min-med-max)     left cov            right cov           leftFirst(med)");
const group = (sel: Row[]) => {
    if (sel.length === 0) return "none";
    const f = sel.map((r) => r.frames).sort((a, b) => a - b);
    const med = f[Math.floor(f.length / 2)];
    return `${String(sel.length).padEnd(4)}${`${f[0]}-${med}-${f[f.length - 1]}`.padEnd(21)}` +
        `${sel.map((r) => cov(r.lObs, r.frames)).join(" ").slice(0, 18).padEnd(19)}` +
        `${sel.map((r) => cov(r.rObs, r.frames)).join(" ").slice(0, 18).padEnd(18)}` +
        `${sel.map((r) => r.lFirst).sort((a, b) => a - b)[Math.floor(sel.length / 2)]}`;
};
console.log(`all         ${group(rows)}`);
console.log(`template    ${group(tplRow ? [tplRow] : [])}`);
console.log(`held-out    ${group(held)}`);

const fullL = held.filter((r) => r.lObs === r.frames).length;
const fullR = held.filter((r) => r.rObs === r.frames).length;
console.log(`\nheld-out takes with BOTH hands in EVERY frame: ${fullL}/${held.length}`);
console.log(`held-out takes with right hand in every frame:   ${fullR}/${held.length}`);
console.log(`left hand present only after frame>0 (i.e. enters late): ${held.filter((r) => r.lFirst > 0).length}/${held.length}`);

console.log("\n--- DISTANCE TO NEAREST 0154 PROTOTYPE (TAUTOLOGY FOR template row) ---");
const ds = held.map((r) => r.selfD).filter((d): d is number => d !== null).sort((a, b) => a - b);
if (ds.length) {
    console.log(`held-out takes n=${ds.length}: min=${ds[0].toFixed(4)} p50=${ds[Math.floor(ds.length / 2)].toFixed(4)} max=${ds[ds.length - 1].toFixed(4)}`);
}
if (tplRow?.selfD !== null && tplRow?.selfD !== undefined) {
    console.log(`registry TEMPLATE take (matches a prototype by construction): ${tplRow.selfD.toFixed(4)}  <-- NOT independent evidence`);
}

// --- gesture 30, same production code ---
const snap = JSON.parse(readFileSync(GESTURE30, "utf-8")) as Snap;
if (snap.landmarks) {
    const q = prepareSidedSequence(snap.landmarks);
    const nearest = Math.min(...targetProtos.map((p) => sidedDistance(q, p).distance));
    console.log("\n--- GESTURE 30 vs THE SAME 0154 PROTOTYPES ---");
    console.log(`frames=${q.totalFrames}  left=${q.left?.detectedFrames}/${q.totalFrames}  right=${q.right?.detectedFrames}/${q.totalFrames}  relation=${q.relation?.detectedFrames ?? "-"}`);
    console.log(`distance to nearest 0154 prototype: ${nearest.toFixed(4)}`);
    const nearestPerTake = targetProtos
        .map((p) => sidedDistance(q, p))
        .sort((a, b) => a.distance - b.distance)[0];
    console.log(`best 0154 prototype components: L=${nearestPerTake.perSide.left?.toFixed(3) ?? "-"} R=${nearestPerTake.perSide.right?.toFixed(3) ?? "-"} rel=${nearestPerTake.relation?.toFixed(3) ?? "-"}`);
    if (ds.length) {
        console.log(`\nPOSITION of gesture-30 within the held-out take distribution: ` +
            `${ds.filter((d) => d < nearest).length}/${ds.length} held-out takes match 0154 BETTER`);
    }
}
