/**
 * EXPERIMENT -- NOT PRODUCTION. Replays an exported gesture through the REAL
 * production matching path (prepareSidedSequence + sidedDistance + the shipped
 * gate) and ablates one side at a time. Deterministic. No production change.
 */
import { readFileSync } from "fs";
import path from "path";
import {
    prepareSidedSequence,
    sidedDistance,
} from "../lib/ml/twoHandMatch";
import { decodePrototypeFile } from "../lib/ml/prototypeStore";
import {
    MAX_NORMALIZED_DISTANCE,
    MAX_BEST_TO_SECOND_BEST_RATIO,
} from "../lib/ml/dynamicSignRecognizer";
import {
    withLegacySequence,
    type DedicatedSignEntry,
} from "../lib/sign-language/signRegistry";
import type { HandLandmarks, SidedSequence } from "../types/landmarks";

const ROOT = path.resolve(__dirname, "..");
const FILE =
    process.argv[2] ??
    "C:/Users/mahmoud/Downloads/karsl-gesture-30-2026-10-13T21-57-01.json";

interface Frame { left: HandLandmarks | null; right: HandLandmarks | null }
interface Snap { landmarks: Frame[] | null; coverage: Record<string, unknown> }

const snap = JSON.parse(readFileSync(FILE, "utf-8")) as Snap;
const frames = snap.landmarks ?? [];

const registry = (
    JSON.parse(
        readFileSync(path.join(ROOT, "public/models/sign-vocabulary/dedicated-signs.json"), "utf-8")
    ) as DedicatedSignEntry[]
).map(withLegacySequence);
const textOf = new Map(registry.map((e) => [e.sourceSignId, e.text]));

const proto = decodePrototypeFile(
    JSON.parse(
        readFileSync(path.join(ROOT, "public/models/sign-vocabulary/sign-prototypes.eval.json"), "utf-8")
    )
);

interface Hit { id: string; d: number; l: number | null; r: number | null; rel: number | null }

/** Scores one sequence against every sign's nearest prototype, applies the
 *  SHIPPED gate, and returns the ranked top three. */
function rank(seq: SidedSequence, label: string): Hit[] {
    const q = prepareSidedSequence(seq);
    const hits: Hit[] = [];
    if (!proto) {
        throw new Error("prototype artifact unavailable");
    }
    for (const [id, takes] of proto.bySign) {
        let best: { d: number; l: number | null; r: number | null; rel: number | null } | null = null;
        for (const t of takes) {
            const r = sidedDistance(q, t);
            if (!best || r.distance < best.d) {
                best = { d: r.distance, l: r.perSide.left, r: r.perSide.right, rel: r.relation };
            }
        }
        if (best) hits.push({ id, ...best });
    }
    hits.sort((a, b) => a.d - b.d);
    const [b, s] = hits;
    const ratio = b && Number.isFinite(s!.d) ? b.d / s!.d : null;
    const accepted = b!.d <= MAX_NORMALIZED_DISTANCE && (ratio === null || b!.d <= s!.d * MAX_BEST_TO_SECOND_BEST_RATIO);
    console.log(
        `\n[${label}]  frames=${seq.length} ` +
            `qLeft=${q.left?.detectedFrames ?? "-"}/${q.totalFrames} ` +
            `qRight=${q.right?.detectedFrames ?? "-"}/${q.totalFrames} ` +
            `qRel=${q.relation?.detectedFrames ?? "-"}`
    );
    console.log(
        `   best=${b!.id} "${textOf.get(b!.id) ?? "?"}" d=${b!.d.toFixed(5)} ` +
            `(L=${b!.l?.toFixed(3) ?? "-"} R=${b!.r?.toFixed(3) ?? "-"} rel=${b!.rel?.toFixed(3) ?? "-"})`
    );
    console.log(`   2nd =${s!.id} "${textOf.get(s!.id) ?? "?"}" d=${s!.d.toFixed(5)}  ratio=${ratio?.toFixed(4)}`);
    console.log(
        `   GATE dist<=${MAX_NORMALIZED_DISTANCE}:${b!.d <= MAX_NORMALIZED_DISTANCE} ` +
            `ratio<=${MAX_BEST_TO_SECOND_BEST_RATIO}:${ratio !== null && b!.d <= s!.d * MAX_BEST_TO_SECOND_BEST_RATIO} ` +
            `=> ${accepted ? "ACCEPTED" : "REJECTED"}`
    );
    return hits;
}

console.log("=".repeat(104));
console.log("REPLAY OF EXPORTED GESTURE -- production matcher, shipped gate, no changes");
console.log("=".repeat(104));

// 1) EXACT replay -- must reproduce the exported numbers.
rank(frames, "as exported (54 frames, both sides)");

// 2) Drop the left hand entirely.
rank(frames.map((f) => ({ left: null, right: f.right })), "RIGHT hand only");

// 3) Drop the right hand entirely.
rank(frames.map((f) => ({ left: f.left, right: null })), "LEFT hand only");

// 4) The first 23 frames, where only the right hand exists at all.
rank(frames.slice(0, 23), "RIGHT only, frames 0-22 (before L enters)");

// 6) LEFT/RIGHT SWAPPED. Tests a camera-convention inversion against REAL
// landmarks rather than inferring one from `query orientation`, which for a
// geometry-backed query is always "as-named" and therefore carries no
// information (blindQueryOrientation.test.ts:176 pins that).
rank(
    frames.map((f) => ({ left: f.right, right: f.left })),
    "LEFT/RIGHT SWAPPED (mirrored convention)"
);