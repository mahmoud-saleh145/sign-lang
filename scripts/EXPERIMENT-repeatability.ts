/**
 * EXPERIMENT B -- REPEATABILITY, label-agnostic. NOT PRODUCTION.
 *
 * Replays one or more exported gesture snapshots through the REAL production
 * matching path (prepareSidedSequence + sidedDistance + the shipped gate) and
 * reports, for a TARGET SIGN ID given on the command line:
 *
 *   - top-1 candidate and distance
 *   - the TARGET's rank and distance  (NOT present in the export, so it is
 *     computed here -- the export only ever carries best and 2nd-best)
 *   - runner-up, best/2nd ratio, and both gate outcomes
 *   - left / right / relation distances and per-side temporal coverage
 *
 * Captures are replayed AGAINST THE EXISTING TEMPLATES ONLY. New captures are
 * never added to the registry, the prototypes or any training data, and no
 * scoring weight, threshold or ranking rule is altered: sidedDistance, the
 * prototype artifact and MAX_* constants all come from production unchanged.
 *
 * Usage:
 *   npx tsx scripts/EXPERIMENT-repeatability.ts --target 0302 -- \
 *     <export.json> [<export.json> ...]
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
import { withLegacySequence, type DedicatedSignEntry } from "../lib/sign-language/signRegistry";
import type { HandLandmarks } from "../types/landmarks";

const ROOT = path.resolve(__dirname, "..");

interface Frame { left: HandLandmarks | null; right: HandLandmarks | null }
interface Snap {
    gestureSequence: number | null;
    classificationSequence: number | null;
    freshness: string;
    exportedAt: string;
    landmarks: Frame[] | null;
    landmarksUnknownSide: HandLandmarks[] | null;
}

interface Hit {
    id: string; d: number; l: number | null; r: number | null; rel: number | null;
}

function parseArgs(): { target: string; files: string[] } {
    const argv = process.argv.slice(2);
    const ti = argv.indexOf("--target");
    const target = ti >= 0 && argv[ti + 1] ? argv[ti + 1] : "";
    const files = argv.filter((a, i) => !a.startsWith("--") && i !== ti + 1);
    return { target, files };
}

const { target, files } = parseArgs();
if (!target || files.length === 0) {
    console.error("usage: --target <signId> <export.json> [<export.json> ...]");
    process.exit(2);
}

const registry = (
    JSON.parse(
        readFileSync(path.join(ROOT, "public/models/sign-vocabulary/dedicated-signs.json"), "utf-8")
    ) as DedicatedSignEntry[]
).map(withLegacySequence);
const textOf = new Map(registry.map((e) => [e.sourceSignId ?? "", e.text]));
const label = textOf.get(target) ?? "<NOT IN REGISTRY>";

const proto = decodePrototypeFile(
    JSON.parse(
        readFileSync(path.join(ROOT, "public/models/sign-vocabulary/sign-prototypes.eval.json"), "utf-8")
    )
);
if (!proto) {
    console.error("prototype artifact unavailable");
    process.exit(1);
}

console.log("=".repeat(108));
console.log(`REPEATABILITY -- target ${target} "${label}"`);
console.log(
    `prototypes for target: ${proto.bySign.get(target)?.length ?? 0}  |  registry signs: ${proto.bySign.size}`
);
console.log(`gate (imported, unchanged): dist<=${MAX_NORMALIZED_DISTANCE} ratio<=${MAX_BEST_TO_SECOND_BEST_RATIO}`);
console.log("=".repeat(108));

interface Row {
    file: string; seq: number | null; fresh: string;
    topId: string; topD: number; rank: number; targetD: number | null;
    secondId: string; secondD: number; ratio: number | null;
    distPass: boolean; ratioPass: boolean; accepted: boolean;
    l: number | null; r: number | null; rel: number | null;
    qL: string; qR: string;
}
const rows: Row[] = [];

for (const file of files) {
    const snap = JSON.parse(readFileSync(file, "utf-8")) as Snap;
    const seq = snap.landmarks;
    if (!seq || seq.length === 0) {
        console.log(`\n${path.basename(file)}: NO SIDED LANDMARKS -- skipped`);
        console.log(
            "  (a capture taken on the unknown-side path exports landmarksUnknownSide instead; " +
            "replay needs a two-handed gesture)"
        );
        continue;
    }
    const q = prepareSidedSequence(seq);

    const hits: Hit[] = [];
    for (const [id, takes] of proto.bySign) {
        let best: Hit | null = null;
        for (const t of takes) {
            const r = sidedDistance(q, t);
            if (!best || r.distance < best.d) {
                best = { id, d: r.distance, l: r.perSide.left, r: r.perSide.right, rel: r.relation };
            }
        }
        if (best) hits.push(best);
    }
    hits.sort((a, b) => a.d - b.d);

    const top = hits[0];
    const second = hits[1];
    const tIdx = hits.findIndex((h) => h.id === target);
    const tHit = tIdx >= 0 ? hits[tIdx] : null;
    const ratio = second ? top.d / second.d : null;
    const distPass = top.d <= MAX_NORMALIZED_DISTANCE;
    const ratioPass = ratio === null || top.d <= second.d * MAX_BEST_TO_SECOND_BEST_RATIO;

    rows.push({
        file: path.basename(file), seq: snap.gestureSequence, fresh: snap.freshness,
        topId: top.id, topD: top.d, rank: tIdx >= 0 ? tIdx + 1 : -1,
        targetD: tHit?.d ?? null,
        secondId: second.id, secondD: second.d, ratio,
        distPass, ratioPass, accepted: distPass && ratioPass,
        l: top.l, r: top.r, rel: top.rel,
        qL: `${q.left?.detectedFrames ?? "-"}/${q.totalFrames}`,
        qR: `${q.right?.detectedFrames ?? "-"}/${q.totalFrames}`,
    });
}

const f = (v: number | null, n = 4): string => (v === null ? "-" : v.toFixed(n));
console.log(
    ["capture".padEnd(38), "seq".padEnd(5), "fresh".padEnd(7), "top-1".padEnd(22),
     "target rank".padEnd(11), "tgt d".padEnd(9), "2nd".padEnd(20),
     "ratio".padEnd(8), "gate"].join(" ")
);
console.log("-".repeat(108));
for (const r of rows) {
    const verdict = r.accepted ? "ACCEPTED" : r.distPass ? "REJECTED(ratio)" : "REJECTED(dist)";
    console.log(
        [
            r.file.slice(0, 37).padEnd(38),
            String(r.seq).padEnd(5),
            r.fresh.padEnd(7),
            `${r.topId} "${(textOf.get(r.topId) ?? "?").slice(0, 10)}"`.padEnd(22),
            `${r.rank}${r.rank === 1 ? " (top)" : ""}`.padEnd(11),
            f(r.targetD).padEnd(9),
            `${r.secondId} "${(textOf.get(r.secondId) ?? "?").slice(0, 8)}"`.padEnd(20),
            f(r.ratio).padEnd(8),
            verdict,
        ].join(" ")
    );
}
console.log("-".repeat(108));
console.log("\nper-capture detail (top-1 candidate's component distances / query coverage):");
for (const r of rows) {
    console.log(
        `  ${r.file.slice(0, 40).padEnd(42)} L=${f(r.l, 3)} R=${f(r.r, 3)} rel=${f(r.rel, 3)}` +
            `  query L=${r.qL.padEnd(7)} R=${r.qR}`
    );
}

if (rows.length > 1) {
    const ranks = rows.map((r) => r.rank);
    const agree = ranks.every((r) => r === ranks[0]);
    console.log(
        `\nAGREEMENT across ${rows.length} captures: target ranks ${ranks.join(", ")}` +
            ` -> ${agree ? "CONSISTENT" : "VARIES between repetitions"}`
    );
}
console.log("\nNOTE: replays against EXISTING templates only. No capture was added to the registry,");
console.log("      prototypes or training data; no threshold or scoring rule was altered.");