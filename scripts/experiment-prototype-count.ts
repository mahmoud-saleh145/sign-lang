/**
 * PROTOTYPE-COUNT CURVE -- how much does each additional real KArSL take buy?
 *
 * The question this answers: is the DTW/prototype architecture short of data
 * or short of capability? If accuracy is still climbing steeply with the
 * number of stored examples, more data is the cheap answer. If it has
 * flattened, the metric itself is the ceiling and no amount of extra takes
 * will move it.
 *
 * ============================================================================
 * METHOD, and why each choice is the one that keeps the numbers comparable
 * ============================================================================
 * THE QUERY POPULATION IS FIXED. Every prototype count is measured on exactly
 * the same queries -- the benchmark takes from lib/ml/karslTakeSplit.ts. Only
 * the template side changes. Varying the queries as well would confound "more
 * prototypes" with "different, possibly easier, queries".
 *
 * THE CANDIDATE SET IS FIXED, and restricted to signs that have enough takes
 * to supply every count under test. A sign that can only offer 2 prototypes
 * would silently stay at 2 while others grew to 4, so the "4 prototypes"
 * condition would not be a 4-prototype system. Restricting the vocabulary
 * keeps every condition honest and keeps the task N-way at the same N
 * throughout. N is reported.
 *
 * NO LEAKAGE, three independent ways:
 *   1. Queries come from split.benchmark; prototypes only ever from
 *      split.prototypes. Those sets are disjoint by construction -- one
 *      function produces both.
 *   2. Calibration takes are never used as prototypes, so the thresholds
 *      (unchanged here) were never derived on anything a prototype saw.
 *   3. An explicit guard asserts no query file appears among the prototype
 *      files, and a distance check flags any match nearer than 0.02 -- the
 *      same tautology threshold lib/ml/karslHeldOut.test.ts uses. Both are
 *      reported per condition and a violation aborts the run.
 *
 * THE REAL RECOGNISER DOES THE WORK. This drives
 * recognizeDynamicSignSyncWithDiagnostics through a stubbed fetch rather than
 * reimplementing scoring, so blind-query orientation, the nearest-prototype
 * rule and both gates are exactly what production applies. A reimplementation
 * would eventually drift from it and the numbers would quietly stop meaning
 * what they claim.
 *
 * THRESHOLDS ARE UNTOUCHED. MAX_NORMALIZED_DISTANCE and the ratio are read
 * from the recogniser and printed, never set.
 *
 * WHICH TAKES BECOME PROTOTYPES MATTERS AS MUCH AS HOW MANY. `splitTakes` now
 * holds out a whole recording session, so every prototype comes from a
 * different sitting than every query, for every sign. That is the honest
 * split -- and it is also why the count curve flattens early: takes recorded
 * minutes apart in one session carry far less new information than takes
 * spread across sessions. `--prototype-source calibration` re-runs the same
 * queries against the calibration takes, which come from the SAME session as
 * the benchmark queries, and the difference between the two isolates session
 * diversity from raw count. That second mode is a deliberate leak, used only
 * to measure the size of the effect; it is never how the system is scored.
 *
 * Usage: npx tsx scripts/experiment-prototype-count.ts [--counts 1,2,3,4]
 *                                                      [--processed processed-karsl]
 *                                                      [--match-vocabulary] [--prototype-source prototypes|calibration]
 */
import { readFileSync, readdirSync, statSync, existsSync } from "fs";
import path from "path";
import {
    clearDynamicSignCache,
    preloadDynamicSignRegistry,
    recognizeDynamicSignSyncWithDiagnostics,
    MAX_NORMALIZED_DISTANCE,
    MAX_BEST_TO_SECOND_BEST_RATIO,
} from "../lib/ml/dynamicSignRecognizer";
import { prepareSidedSequence, MIN_SIDE_FRAMES, FEATURE_VERSION, DTW_SEQUENCE_LENGTH } from "../lib/ml/twoHandMatch";
import { assignSides, hasGeometricSideEvidence, GEOMETRY_MIN_FRAMES } from "../lib/ml/handSides";
import { encodeTake, type EncodedTake, type PrototypeFile } from "../lib/ml/prototypeStore";
import { splitTakes } from "../lib/ml/karslTakeSplit";
import { DEFAULT_PROCESSED_ROOT } from "../lib/ml/karslPaths";
import { loadExclusions, excludedFor } from "../lib/ml/karslExclusions";
import { withLegacySequence, entryToSidedSequence } from "../lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "../lib/sign-language/signRegistry";
import type { HandLandmarks, SidedSequence } from "../types/landmarks";

function arg(name: string, fallback: string): string {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

interface ProcessedHand { landmarks: HandLandmarks; rawLabel?: "Left" | "Right" }
interface ProcessedFrame { hands?: { left?: ProcessedHand | null; right?: ProcessedHand | null } }
interface ProcessedTake { frames?: ProcessedFrame[] }

/** Canonical sides, derived exactly as the importer and the camera derive them. */
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

/**
 * The recording date embedded in a KArSL filename, e.g.
 * "01_02_0003_(15_11_16_16_44_06)_c.json" -> "15_11_16". Null when the name
 * does not carry one, in which case session structure is simply unknown and
 * the sign is not counted either way.
 */
function sessionDay(file: string): string | null {
    const m = /\((\d\d_\d\d_\d\d)_/.exec(file);
    return m ? m[1] : null;
}

const usableSides = (frames: SidedSequence): number =>
    (frames.filter((f) => f.left).length >= MIN_SIDE_FRAMES ? 1 : 0) +
    (frames.filter((f) => f.right).length >= MIN_SIDE_FRAMES ? 1 : 0);

interface Query {
    signId: string;
    file: string;
    sided: SidedSequence;
    sides: number;
    geometryBacked: boolean;
}

interface SignData {
    signId: string;
    entry: DedicatedSignEntry;
    queries: Query[];
    /** Encoded prototypes, in the deterministic order of split.prototypes. */
    prototypes: EncodedTake[];
    prototypeFiles: string[];
    templateSides: number;
}

async function main(): Promise<void> {
    const processedRoot = arg("processed", DEFAULT_PROCESSED_ROOT);
    const registryPath = arg("registry", "public/models/sign-vocabulary/dedicated-signs.json");
    const counts = arg("counts", "1,2,3,4").split(",").map((c) => Number(c.trim())).filter((c) => c > 0);
    const maxCount = Math.max(...counts);
    const source = arg("prototype-source", "prototypes");
    const matchVocabulary = process.argv.includes("--match-vocabulary");
    if (source !== "prototypes" && source !== "calibration") {
        console.error(`--prototype-source must be "prototypes" or "calibration"`);
        process.exit(1);
    }

    if (!existsSync(processedRoot)) {
        console.error(`No processed data at "${processedRoot}".`);
        process.exit(1);
    }

    const registry = (JSON.parse(readFileSync(registryPath, "utf-8")) as DedicatedSignEntry[]).map(withLegacySequence);
    const entryOf = new Map<string, DedicatedSignEntry>();
    const templateFileOf = new Map<string, string>();
    for (const e of registry) {
        if (!e.sourceSignId) continue;
        entryOf.set(e.sourceSignId, e);
        if (e.sourceTake?.processedFile) templateFileOf.set(e.sourceSignId, e.sourceTake.processedFile);
    }

    // --- assemble, keeping only signs that can supply EVERY count ---------
    const exclusions = loadExclusions(arg("exclusions", "karsl-exclusions.json"));
    const signs: SignData[] = [];
    let skippedTooFewPrototypes = 0;
    for (const signId of readdirSync(processedRoot).sort()) {
        const dir = path.join(processedRoot, signId);
        if (!statSync(dir).isDirectory()) continue;
        const entry = entryOf.get(signId);
        if (!entry) continue;
        const files = readdirSync(dir).filter((f) => f.endsWith(".json")).sort();
        const split = splitTakes(files, templateFileOf.get(signId) ?? null, excludedFor(exclusions, signId));
        // Both pools are disjoint from split.benchmark by construction; the
        // calibration pool is the same-session comparison, not a second split.
        const pool = source === "calibration" ? split.calibration : split.prototypes;
        // --match-vocabulary requires BOTH pools to supply the count, whichever
        // one is in use, so the two sources are measured on an identical
        // vocabulary and the same-session comparison is not also a change of
        // task. It caps the usable count at the calibration pool's size, so it
        // is off for the count curve itself.
        const insufficient = matchVocabulary
            ? split.calibration.length < maxCount || split.prototypes.length < maxCount
            : pool.length < maxCount;
        if (insufficient) {
            skippedTooFewPrototypes++;
            continue;
        }

        const queries: Query[] = [];
        for (const file of split.benchmark) {
            const take = JSON.parse(readFileSync(path.join(dir, file), "utf-8")) as ProcessedTake;
            const sided = sidedFromTake(take);
            if (sided.length < 2) continue;
            queries.push({
                signId, file, sided,
                sides: usableSides(sided),
                geometryBacked: hasGeometricSideEvidence(sided),
            });
        }
        if (queries.length === 0) continue;

        const prototypes: EncodedTake[] = [];
        const prototypeFiles: string[] = [];
        for (const file of pool) {
            if (prototypes.length >= maxCount) break;
            const take = JSON.parse(readFileSync(path.join(dir, file), "utf-8")) as ProcessedTake;
            const sided = sidedFromTake(take);
            if (sided.length < 2) continue;
            const encoded = encodeTake(file, prepareSidedSequence(sided));
            if (!encoded) continue;
            prototypes.push(encoded);
            prototypeFiles.push(file);
        }
        if (prototypes.length < maxCount) {
            skippedTooFewPrototypes++;
            continue;
        }

        const templateSided = entryToSidedSequence(entry);
        signs.push({
            signId, entry, queries, prototypes, prototypeFiles,
            templateSides: templateSided ? usableSides(templateSided) : 0,
        });
    }

    const allQueries = signs.flatMap((s) => s.queries);
    console.log("=".repeat(104));
    console.log("PROTOTYPE-COUNT CURVE");
    console.log("=".repeat(104));
    console.log(`   feature version            ${FEATURE_VERSION}  (sequence length ${DTW_SEQUENCE_LENGTH})`);
    console.log(`   thresholds (UNCHANGED)     distance <= ${MAX_NORMALIZED_DISTANCE}, ratio <= ${MAX_BEST_TO_SECOND_BEST_RATIO}`);
    console.log(`   signs in the vocabulary    ${signs.length}   (${skippedTooFewPrototypes} skipped: fewer than ${maxCount} prototype takes)`);
    console.log(`   queries (FIXED across all prototype counts)  ${allQueries.length}`);
    console.log(`   counts under test          ${counts.join(", ")}`);
    console.log(`   prototype source           split.${source}${matchVocabulary ? "  (vocabulary matched across sources)" : ""}`);

    // Session structure: a prototype recorded in the same session as the query
    // is a much easier match than one from another day, so the mix has to be
    // reported or the curve cannot be read.
    let sameSession = 0;
    let unknownSession = 0;
    for (const s of signs) {
        const qDays = new Set(s.queries.map((q) => sessionDay(q.file)).filter((d) => d !== null));
        const pDays = new Set(s.prototypeFiles.map(sessionDay).filter((d) => d !== null));
        if (qDays.size === 0 || pDays.size === 0) unknownSession++;
        else if ([...pDays].some((d) => qDays.has(d))) sameSession++;
    }
    console.log(
        `   session overlap            ${sameSession}/${signs.length} signs have a prototype from a query's own ` +
        `recording session${unknownSession ? ` (${unknownSession} unknown)` : ""}`
    );

    // --- leakage guard 1: file-level, before any scoring ------------------
    const overlaps: string[] = [];
    for (const s of signs) {
        const protoSet = new Set(s.prototypeFiles);
        for (const q of s.queries) if (protoSet.has(q.file)) overlaps.push(`${s.signId}/${q.file}`);
    }
    if (overlaps.length > 0) {
        console.error(`\nLEAKAGE: ${overlaps.length} query take(s) are also prototypes: ${overlaps.slice(0, 5).join(", ")}`);
        process.exit(1);
    }
    console.log(`   leakage guard (file)       PASS: no query take appears among any sign's prototypes`);

    // The registry the recogniser sees is restricted to this vocabulary, so
    // the task is N-way at the same N in every condition.
    const restrictedRegistry = signs.map((s) => s.entry);
    let currentPrototypeFile: PrototypeFile | null = null;
    const fetchStub = async (input: unknown) => {
        const url = String(input);
        if (url.includes("sign-prototypes")) {
            if (!currentPrototypeFile) return { ok: false, status: 404, json: async () => ({}) };
            return { ok: true, status: 200, json: async () => currentPrototypeFile };
        }
        return { ok: true, status: 200, json: async () => restrictedRegistry };
    };
    (globalThis as unknown as { fetch: unknown }).fetch = fetchStub;

    interface Result {
        k: number; totalPrototypes: number; queries: number;
        top1: number; acceptedCorrect: number; accepted: number; falseAccepts: number;
        rejectedCorrect: number; rejectedWrong: number; tautologies: number;
        slices: Record<string, { n: number; hits: number; accepted: number; fa: number }>;
    }
    const results: Result[] = [];

    for (const k of counts) {
        const signsOut: Record<string, EncodedTake[]> = {};
        let totalPrototypes = 0;
        for (const s of signs) {
            signsOut[s.signId] = s.prototypes.slice(0, k);
            totalPrototypes += Math.min(k, s.prototypes.length);
        }
        currentPrototypeFile = {
            version: 1,
            featureVersion: FEATURE_VERSION,
            sequenceLength: DTW_SEQUENCE_LENGTH,
            generatedAt: new Date().toISOString(),
            signs: signsOut,
        };
        clearDynamicSignCache();
        await preloadDynamicSignRegistry();

        const slices: Result["slices"] = {
            "one-hand query": { n: 0, hits: 0, accepted: 0, fa: 0 },
            "two-hand query": { n: 0, hits: 0, accepted: 0, fa: 0 },
            "geometry-backed query": { n: 0, hits: 0, accepted: 0, fa: 0 },
            "blind query": { n: 0, hits: 0, accepted: 0, fa: 0 },
            "one-handed template": { n: 0, hits: 0, accepted: 0, fa: 0 },
            "two-handed template": { n: 0, hits: 0, accepted: 0, fa: 0 },
        };
        const templateSidesOf = new Map(signs.map((s) => [s.signId, s.templateSides]));
        let top1 = 0, accepted = 0, acceptedCorrect = 0, falseAccepts = 0;
        let rejectedCorrect = 0, rejectedWrong = 0, tautologies = 0;

        for (const q of allQueries) {
            const d = recognizeDynamicSignSyncWithDiagnostics(q.sided);
            const hit = d.bestSignId === q.signId;
            if (hit) top1++;
            if (hit && (d.bestDistance ?? 1) < 0.02) tautologies++;
            if (d.accepted) {
                accepted++;
                if (hit) acceptedCorrect++;
                else falseAccepts++;
            } else if (hit) rejectedCorrect++;
            else rejectedWrong++;

            const bump = (name: string) => {
                const s = slices[name];
                s.n++;
                if (hit) s.hits++;
                if (d.accepted) { s.accepted++; if (!hit) s.fa++; }
            };
            bump(q.sides === 1 ? "one-hand query" : "two-hand query");
            bump(q.geometryBacked ? "geometry-backed query" : "blind query");
            bump(templateSidesOf.get(q.signId) === 1 ? "one-handed template" : "two-handed template");
        }

        results.push({
            k, totalPrototypes, queries: allQueries.length,
            top1, acceptedCorrect, accepted, falseAccepts,
            rejectedCorrect, rejectedWrong, tautologies, slices,
        });

        const pct = (x: number) => (x / allQueries.length).toFixed(4);
        console.log(`\n${"-".repeat(104)}`);
        console.log(`k = ${k} prototype(s)/sign   total prototypes ${totalPrototypes}   queries ${allQueries.length}`);
        console.log(`   top-1 ${top1}/${allQueries.length} = ${pct(top1)}   accepted&correct ${pct(acceptedCorrect)}   acceptance ${pct(accepted)}   falseAccepts ${falseAccepts}   precision ${accepted ? (acceptedCorrect / accepted).toFixed(4) : "n/a"}`);
        console.log(`   rejected-but-correct ${rejectedCorrect}   rejected-and-wrong ${rejectedWrong}   leakage guard (distance<0.02) ${tautologies === 0 ? "PASS" : `FAIL (${tautologies})`}`);
        for (const [name, s] of Object.entries(slices)) {
            if (s.n === 0) continue;
            console.log(`   ${name.padEnd(24)} n=${String(s.n).padStart(4)}  top1=${(s.hits / s.n).toFixed(3)}  accepted=${(s.accepted / s.n).toFixed(3)}  falseAccepts=${s.fa}`);
        }
        if (tautologies > 0) {
            console.error("\nLEAKAGE: a query matched a stored prototype at near-zero distance. Numbers void.");
            process.exit(1);
        }
    }

    console.log(`\n${"=".repeat(104)}`);
    console.log("SUMMARY");
    console.log("   prototypes/sign | total prototypes | queries | top1    | accepted_correct | acceptance | false_accepts | precision");
    for (const r of results) {
        console.log(
            `   ${String(r.k).padStart(15)} | ${String(r.totalPrototypes).padStart(16)} | ${String(r.queries).padStart(7)} | ` +
            `${(r.top1 / r.queries).toFixed(4)}  | ${(r.acceptedCorrect / r.queries).toFixed(4)}           | ` +
            `${(r.accepted / r.queries).toFixed(4)}     | ${String(r.falseAccepts).padStart(13)} | ` +
            `${r.accepted ? (r.acceptedCorrect / r.accepted).toFixed(4) : "n/a"}`
        );
    }
    console.log("\n   MARGINAL GAIN IN TOP-1 PER ADDED PROTOTYPE");
    for (let i = 1; i < results.length; i++) {
        const a = results[i - 1], b = results[i];
        const delta = (b.top1 - a.top1) / b.queries;
        const perProto = delta / (b.k - a.k);
        console.log(`   k=${a.k} -> k=${b.k}:  ${(delta * 100 >= 0 ? "+" : "")}${(delta * 100).toFixed(2)} pp total, ${(perProto * 100).toFixed(2)} pp per added prototype`);
    }
    console.log(`   GEOMETRY_MIN_FRAMES=${GEOMETRY_MIN_FRAMES}`);
    console.log("=".repeat(104));
}

void main();
