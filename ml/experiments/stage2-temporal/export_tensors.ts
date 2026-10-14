/**
 * STAGE 2 EXPERIMENT -- tensor export.
 *
 * ============================================================================
 * WHY THIS IS TYPESCRIPT AND NOT PYTHON
 * ============================================================================
 * The temporal model has to be compared against DTW on EXACTLY the same
 * queries with EXACTLY the same features, or the comparison measures a
 * reimplementation rather than an architecture. Rewriting
 * prepareSidedSequence() and splitTakes() in Python would put a second copy of
 * both in the tree, and this project has already been burned twice by that
 * shape of duplication -- the index-arithmetic split that leaked prototypes
 * into the query set, and the processed-karsl path literal that survived the
 * corpus re-extraction.
 *
 * So the export calls the PRODUCTION functions directly. Python only trains.
 * If the features change, this export changes with them automatically.
 *
 * ============================================================================
 * WHAT GOES INTO THE MODEL, CHANNEL BY CHANNEL
 * ============================================================================
 * One take becomes a [T=30, C=95] float32 matrix. T is DTW_SEQUENCE_LENGTH,
 * so the temporal resampling is identical to what the matcher consumes.
 *
 *    0..43    LEFT side, per frame: 42 shape values (21 landmarks x, y after
 *             normalizeLandmarks) + 2 position values (wrist offset from that
 *             SIDE's own mean wrist, over that side's own mean hand size,
 *             times POSITION_WEIGHT=2).
 *   44..87    RIGHT side, same 44 values.
 *   88..89    RELATION: left wrist minus right wrist over mean hand size,
 *             times RELATION_WEIGHT=1. Zero when the two hands were never
 *             seen together.
 *   90        leftPresent    1.0 / 0.0
 *   91        rightPresent   1.0 / 0.0
 *   92        relationPresent
 *   93        left coverage  = detectedFrames / totalFrames
 *   94        right coverage
 *
 * WHY THE PRESENCE CHANNELS EXIST. A missing side is written as zeros, and
 * zeros are a legal landmark configuration -- without an explicit flag the
 * model cannot tell "this hand was absent" from "this hand sat exactly at its
 * own reference point". DTW never had this problem because a null side is
 * skipped and charged through coverage weighting; a fixed-width tensor has to
 * say it out loud. Channels 93-94 carry the same coverage information DTW
 * uses, so the model has access to the same evidence and no more.
 *
 * NOTHING IS INVENTED. A side that was never detected contributes zeros and a
 * zero flag. No landmark is interpolated, mirrored or synthesised.
 *
 * ============================================================================
 * SPLIT, AND THE LEAKAGE GUARANTEE
 * ============================================================================
 * splitTakes() is called for every sign, and:
 *
 *   train = split.prototypes   (the sessions that supply DTW's prototypes)
 *   test  = split.benchmark    (the held-out session's queries)
 *   val   = split.calibration  (the held-out session's OTHER half)
 *
 * The model therefore trains on exactly the recordings DTW matches against,
 * and is tested on exactly the recordings DTW is tested on. Neither sees the
 * held-out session during training.
 *
 * VAL IS NOT A CLEAN MODEL-SELECTION SET and is labelled as such: it comes
 * from the same sitting as the test set. It is exported for early stopping
 * because there is nothing better in a two-session corpus, and any number
 * measured on it must be reported as optimistic. Test is never touched during
 * training.
 *
 * Takes named in karsl-exclusions.json are dropped from all three, exactly as
 * they are for DTW.
 *
 * ============================================================================
 * --with-dtw
 * ============================================================================
 * Runs the PRODUCTION recogniser over the test rows and records its
 * prediction, distances, ratio and accept/reject for each one. That is what
 * makes "same held-out queries" a fact rather than a claim: the comparison
 * table is built from two columns of the same file.
 *
 * Usage:
 *   npx tsx ml/experiments/stage2-temporal/export_tensors.ts \
 *       [--processed processed-karsl-v2] [--out ml/experiments/stage2-temporal/data] \
 *       [--with-dtw] [--dtw-split test|val|both]
 *       [--dtw-only] [--dtw-from N] [--dtw-to M]
 *
 * --dtw-only scores the DTW column WITHOUT re-exporting tensors: the rows come
 * from the EXISTING data/manifest.json instead of being re-derived from the
 * corpus. The split is therefore read, never recomputed, so it cannot drift
 * from the tensors the TCN was trained on and tensors.f32 is left untouched.
 * --dtw-from/--dtw-to take a half-open slice of that split's rows, in manifest
 * order, and write data/dtw_baseline_<split>.part-<from>-<to>.json; the parts
 * concatenate back in row order into the single-file schema. This exists only
 * because scoring 4,288 val queries takes ~1 h at ~0.9 s/query and this
 * environment caps a process at 180 s. Nothing about the DTW computation,
 * the prototypes, the thresholds or the recorded fields changes.
 */
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync, mkdirSync } from "fs";
import path from "path";
import {
    prepareSidedSequence,
    FEATURE_VERSION,
    DTW_SEQUENCE_LENGTH,
    type PreparedSidedSequence,
} from "../../../lib/ml/twoHandMatch";
import { assignSides } from "../../../lib/ml/handSides";
import { splitTakes } from "../../../lib/ml/karslTakeSplit";
import { loadExclusions, excludedFor } from "../../../lib/ml/karslExclusions";
import { DEFAULT_PROCESSED_ROOT } from "../../../lib/ml/karslPaths";
import {
    clearDynamicSignCache,
    preloadDynamicSignRegistry,
    recognizeDynamicSignSyncWithDiagnostics,
    MAX_NORMALIZED_DISTANCE,
    MAX_BEST_TO_SECOND_BEST_RATIO,
} from "../../../lib/ml/dynamicSignRecognizer";
import { withLegacySequence } from "../../../lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "../../../lib/sign-language/signRegistry";
import type { HandLandmarks, SidedSequence } from "../../../types/landmarks";

const CHANNELS = 95;
const SIDE_DIM = 44;
const RELATION_DIM = 2;

function arg(name: string, fallback: string): string {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

interface ProcessedHand { landmarks: HandLandmarks; rawLabel?: "Left" | "Right" }
interface ProcessedFrame { hands?: { left?: ProcessedHand | null; right?: ProcessedHand | null } }
interface ProcessedTake { frames?: ProcessedFrame[] }

function sidedFromTake(take: ProcessedTake): SidedSequence {
    const slotA: Array<HandLandmarks | null> = [];
    const slotB: Array<HandLandmarks | null> = [];
    const labelA: Array<"Left" | "Right" | null> = [];
    const labelB: Array<"Left" | "Right" | null> = [];
    for (const frame of take.frames ?? []) {
        slotA.push(frame.hands?.left?.landmarks ?? null);
        slotB.push(frame.hands?.right?.landmarks ?? null);
        labelA.push(frame.hands?.left?.rawLabel ?? null);
        labelB.push(frame.hands?.right?.rawLabel ?? null);
    }
    const assignment = assignSides(slotA, slotB, labelA, labelB);
    const out: SidedSequence = [];
    for (let i = 0; i < slotA.length; i++) {
        const frame: { left: HandLandmarks | null; right: HandLandmarks | null } = { left: null, right: null };
        frame[assignment.sides[0] as "left" | "right"] = slotA[i];
        frame[assignment.sides[1] as "left" | "right"] = slotB[i];
        if (frame.left || frame.right) out.push(frame);
    }
    return out;
}

/** [T, C] row-major, exactly as documented at the top of this file. */
function toTensor(prepared: PreparedSidedSequence): Float32Array | null {
    const { left, right, relation, totalFrames } = prepared;
    if (!left && !right) return null;
    const out = new Float32Array(DTW_SEQUENCE_LENGTH * CHANNELS);
    for (let t = 0; t < DTW_SEQUENCE_LENGTH; t++) {
        const base = t * CHANNELS;
        if (left) for (let c = 0; c < SIDE_DIM; c++) out[base + c] = left.vectors[t][c];
        if (right) for (let c = 0; c < SIDE_DIM; c++) out[base + SIDE_DIM + c] = right.vectors[t][c];
        if (relation) {
            for (let c = 0; c < RELATION_DIM; c++) out[base + 2 * SIDE_DIM + c] = relation.vectors[t][c];
        }
        out[base + 90] = left ? 1 : 0;
        out[base + 91] = right ? 1 : 0;
        out[base + 92] = relation ? 1 : 0;
        const denom = totalFrames || 1;
        out[base + 93] = left ? left.detectedFrames / denom : 0;
        out[base + 94] = right ? right.detectedFrames / denom : 0;
    }
    return out;
}

interface Row {
    signId: string;
    classIndex: number;
    file: string;
    session: string | null;
    split: "train" | "val" | "test";
}

async function main(): Promise<void> {
    const processedRoot = arg("processed", DEFAULT_PROCESSED_ROOT);
    const registryPath = arg("registry", "public/models/sign-vocabulary/dedicated-signs.json");
    const outDir = arg("out", "ml/experiments/stage2-temporal/data");
    const withDtw = process.argv.includes("--with-dtw");
    const dtwOnly = process.argv.includes("--dtw-only");

    if (!existsSync(processedRoot)) {
        console.error(`No processed data at "${processedRoot}".`);
        process.exit(1);
    }
    mkdirSync(outDir, { recursive: true });

    // DTW-ONLY: read the split that is already on disk and score it. No tensor
    // export, no manifest rewrite, no re-derivation of the split.
    if (dtwOnly) {
        const manifestPath = path.join(outDir, "manifest.json");
        if (!existsSync(manifestPath)) {
            console.error(`--dtw-only needs an existing ${manifestPath}. Run the full export first.`);
            process.exit(1);
        }
        const existing = JSON.parse(readFileSync(manifestPath, "utf-8")) as {
            featureVersion: string;
            rows: Row[];
        };
        if (existing.featureVersion !== FEATURE_VERSION) {
            console.error(
                `--dtw-only refused: manifest featureVersion "${existing.featureVersion}" != ` +
                `"${FEATURE_VERSION}". The rows were built for a different representation.`
            );
            process.exit(1);
        }
        console.log(`   --dtw-only: ${existing.rows.length} rows read from ${manifestPath}`);
        await runDtwBaseline(existing.rows, processedRoot, outDir);
        return;
    }

    const registry = (JSON.parse(readFileSync(registryPath, "utf-8")) as DedicatedSignEntry[]).map(
        withLegacySequence
    );
    const labelOf = new Map<string, string>();
    const templateFileOf = new Map<string, string>();
    for (const e of registry) {
        if (!e.sourceSignId) continue;
        // Multi-word Arabic labels stay ONE class. The text is carried
        // verbatim and never split on whitespace.
        labelOf.set(e.sourceSignId, e.text);
        if (e.sourceTake?.processedFile) templateFileOf.set(e.sourceSignId, e.sourceTake.processedFile);
    }

    const exclusions = loadExclusions(arg("exclusions", "karsl-exclusions.json"));
    const signIds = readdirSync(processedRoot)
        .filter((d) => statSync(path.join(processedRoot, d)).isDirectory() && labelOf.has(d))
        .sort();

    const rows: Row[] = [];
    const tensors: Float32Array[] = [];
    let skippedUnusable = 0;

    const sessionOf = (f: string): string | null => {
        const m = /\((\d{2}_\d{2}_\d{2})_\d{2}_\d{2}_\d{2}\)/.exec(f);
        return m ? m[1] : null;
    };

    for (let classIndex = 0; classIndex < signIds.length; classIndex++) {
        const signId = signIds[classIndex];
        const dir = path.join(processedRoot, signId);
        const files = readdirSync(dir).filter((f) => f.endsWith(".json")).sort();
        const split = splitTakes(files, templateFileOf.get(signId) ?? null, excludedFor(exclusions, signId));

        const groups: Array<[ReadonlyArray<string>, Row["split"]]> = [
            [split.prototypes, "train"],
            [split.calibration, "val"],
            [split.benchmark, "test"],
        ];
        for (const [group, name] of groups) {
            for (const file of group) {
                const take = JSON.parse(readFileSync(path.join(dir, file), "utf-8")) as ProcessedTake;
                const sided = sidedFromTake(take);
                if (sided.length < 2) { skippedUnusable++; continue; }
                const tensor = toTensor(prepareSidedSequence(sided));
                if (!tensor) { skippedUnusable++; continue; }
                tensors.push(tensor);
                rows.push({ signId, classIndex, file, session: sessionOf(file), split: name });
            }
        }
        if ((classIndex + 1) % 50 === 0) {
            console.log(`   ...${classIndex + 1}/${signIds.length} signs, ${rows.length} takes`);
        }
    }

    // --- leakage guard, asserted rather than assumed -----------------------
    const trainFiles = new Set(rows.filter((r) => r.split === "train").map((r) => `${r.signId}/${r.file}`));
    const overlap = rows
        .filter((r) => r.split !== "train")
        .filter((r) => trainFiles.has(`${r.signId}/${r.file}`));
    if (overlap.length > 0) {
        console.error(`LEAKAGE: ${overlap.length} take(s) appear in train AND in val/test.`);
        process.exit(1);
    }
    const trainSessions = new Map<string, Set<string>>();
    for (const r of rows.filter((x) => x.split === "train")) {
        if (!r.session) continue;
        if (!trainSessions.has(r.signId)) trainSessions.set(r.signId, new Set());
        trainSessions.get(r.signId)!.add(r.session);
    }
    const sessionLeaks = rows
        .filter((r) => r.split === "test" && r.session)
        .filter((r) => trainSessions.get(r.signId)?.has(r.session!));
    if (sessionLeaks.length > 0) {
        console.error(
            `LEAKAGE: ${sessionLeaks.length} test take(s) come from a session that also supplied ` +
            `training data for the same sign. Example: ${sessionLeaks[0].signId}/${sessionLeaks[0].file}`
        );
        process.exit(1);
    }

    // --- write ------------------------------------------------------------
    const flat = new Float32Array(tensors.length * DTW_SEQUENCE_LENGTH * CHANNELS);
    tensors.forEach((t, i) => flat.set(t, i * DTW_SEQUENCE_LENGTH * CHANNELS));
    writeFileSync(path.join(outDir, "tensors.f32"), Buffer.from(flat.buffer));

    const manifest = {
        featureVersion: FEATURE_VERSION,
        sequenceLength: DTW_SEQUENCE_LENGTH,
        channels: CHANNELS,
        channelLayout: {
            left: [0, 43], right: [44, 87], relation: [88, 89],
            presence: [90, 92], coverage: [93, 94],
        },
        processedRoot,
        generatedAt: new Date().toISOString(),
        classes: signIds.map((s, i) => ({ classIndex: i, signId: s, text: labelOf.get(s) ?? "" })),
        exclusionsLoaded: exclusions.loaded,
        skippedUnusable,
        counts: {
            train: rows.filter((r) => r.split === "train").length,
            val: rows.filter((r) => r.split === "val").length,
            test: rows.filter((r) => r.split === "test").length,
        },
        rows,
    };
    writeFileSync(path.join(outDir, "manifest.json"), JSON.stringify(manifest));

    console.log("=".repeat(76));
    console.log("STAGE 2 TENSORS EXPORTED");
    console.log("=".repeat(76));
    console.log(`   feature version   ${FEATURE_VERSION}`);
    console.log(`   shape             [${rows.length}, ${DTW_SEQUENCE_LENGTH}, ${CHANNELS}]`);
    console.log(`   classes           ${signIds.length}`);
    console.log(`   train / val / test  ${manifest.counts.train} / ${manifest.counts.val} / ${manifest.counts.test}`);
    console.log(`   exclusions        ${exclusions.loaded ? "loaded" : "NOT FOUND"}`);
    console.log(`   takes skipped     ${skippedUnusable}`);
    console.log(`   leakage guards    PASS (no shared take, no shared session)`);
    console.log(`   out               ${outDir}`);

    if (withDtw) {
        console.log("\n   --with-dtw: scoring the SAME test rows with the production recogniser...");
        await runDtwBaseline(rows, processedRoot, outDir);
    }
    console.log("=".repeat(76));
}

/**
 * The DTW column of the comparison, produced from the same `rows` the tensors
 * were produced from, so "the same held-out queries" is structural.
 */
async function runDtwBaseline(rows: Row[], processedRoot: string, outDir: string): Promise<void> {
    const registryJson = readFileSync("public/models/sign-vocabulary/dedicated-signs.json", "utf-8");
    const protoPath = arg("prototypes", "public/models/sign-vocabulary/sign-prototypes.eval.json");
    const protoJson = existsSync(protoPath) ? readFileSync(protoPath, "utf-8") : null;
    if (!protoJson) {
        console.error(`   MISSING ${protoPath} -- build it with --held-out first. DTW column skipped.`);
        return;
    }
    (globalThis as unknown as { fetch: unknown }).fetch = async (input: unknown) => {
        const url = String(input);
        const body = url.includes("sign-prototypes") ? protoJson : registryJson;
        return { ok: true, status: 200, json: async () => JSON.parse(body) };
    };
    clearDynamicSignCache();
    // MUST be awaited. Firing this and scoring immediately measured an empty
    // registry and reported DTW top-1 of exactly 0.0000 -- a number that looks
    // like a finding and is a bug.
    await preloadDynamicSignRegistry();

    // BOTH val and test. Val is scored because the fusion weights and the DTW
    // margin must be tuned somewhere that is not test -- without a val DTW
    // column those parameters cannot be chosen honestly at all, and
    // ml/experiments/stage2-temporal/fuse.py falls back to untuned defaults
    // rather than quietly fitting them on the test labels.
    const splitName = (arg("dtw-split", "test") as "test" | "val" | "both");
    const wanted = splitName === "both" ? ["val", "test"] : [splitName];
    for (const which of wanted) {
        scoreSplit(rows.filter((r) => r.split === which), which, processedRoot, outDir);
    }
}

function scoreSplit(all: Row[], which: string, processedRoot: string, outDir: string): void {
    // Half-open slice of this split's rows, IN MANIFEST ORDER, so the parts of
    // a chunked run concatenate back to exactly the unchunked result.
    const from = Number(arg("dtw-from", "0"));
    const toRaw = arg("dtw-to", "");
    const to = toRaw === "" ? all.length : Math.min(all.length, Number(toRaw));
    if (!Number.isFinite(from) || from < 0 || !(to > from)) {
        console.error(`Invalid --dtw-from/--dtw-to range: [${from}, ${toRaw || "end"})`);
        process.exit(1);
    }
    const partial = from !== 0 || to !== all.length;
    const test = all.slice(from, to);
    const out: unknown[] = [];
    const started = Date.now();
    for (let i = 0; i < test.length; i++) {
        const r = test[i];
        const take = JSON.parse(
            readFileSync(path.join(processedRoot, r.signId, r.file), "utf-8")
        ) as ProcessedTake;
        const d = recognizeDynamicSignSyncWithDiagnostics(sidedFromTake(take));
        out.push({
            signId: r.signId, file: r.file,
            predicted: d.bestSignId, distance: d.bestDistance,
            secondBest: d.secondBestSignId, secondDistance: d.secondBestDistance,
            ratio: d.bestToSecondBestRatio, accepted: d.accepted,
            rejectionReason: d.rejectionReason,
        });
        if ((i + 1) % 200 === 0) console.log(`   ...DTW ${which} ${i + 1}/${test.length}`);
    }
    const elapsed = (Date.now() - started) / 1000;
    const base = which === "test" ? "dtw_baseline" : `dtw_baseline_${which}`;
    const name = partial ? `${base}.part-${from}-${to}.json` : `${base}.json`;
    writeFileSync(
        path.join(outDir, name),
        JSON.stringify({
            thresholds: { distance: MAX_NORMALIZED_DISTANCE, ratio: MAX_BEST_TO_SECOND_BEST_RATIO },
            elapsedSeconds: elapsed,
            msPerQuery: (elapsed * 1000) / Math.max(test.length, 1),
            results: out,
        })
    );
    console.log(`   DTW baseline (${which}${partial ? ` rows [${from},${to})` : ""}) written to ${name}: ` +
        `${test.length} queries in ${elapsed.toFixed(1)}s ` +
        `(${((elapsed * 1000) / Math.max(test.length, 1)).toFixed(1)} ms/query)`);
}

void main();
