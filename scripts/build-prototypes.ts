/**
 * BUILDS public/models/sign-vocabulary/sign-prototypes.json from the real
 * processed KArSL takes.
 *
 * Every usable take of every sign becomes a prototype. The recogniser scores a
 * query against all of them and keeps the nearest, which is worth +21.5 points
 * of top-1 over the single stored template -- see lib/ml/prototypeStore.ts for
 * the measurement.
 *
 * THE PREPARATION IS THE PRODUCTION PREPARATION. This calls
 * prepareSidedSequence() and encodeTake() from lib/ml, not a copy of them, so
 * a prototype cannot be built with features the matcher does not use. The
 * artifact records FEATURE_VERSION and the loader rejects a mismatch.
 *
 * SIDE NAMING is re-derived here with lib/ml/handSides.assignSides, the same
 * rule the importer and the live camera use -- geometry first, handedness
 * label second -- rather than trusting the slot names the extractor wrote.
 *
 * LEAKAGE: this builds TEMPLATES, not an evaluation. It reads every take,
 * including the ones the held-out benchmark uses as queries. That is not
 * leakage in itself -- a deployed system is entitled to every recording it
 * has -- but it means the benchmark MUST exclude, for each query, the
 * prototype built from that same take. lib/ml/karslHeldOut.test.ts does this
 * by `file`, which is recorded on every prototype for exactly this purpose,
 * and its "never scores a take against itself" guard fails loudly otherwise.
 *
 * Usage:
 *   npx tsx scripts/build-prototypes.ts [--processed processed-karsl]
 *                                       [--out ...sign-prototypes.json]
 *                                       [--max-takes N] [--held-out]
 *                                       [--sign-from I] [--sign-to J]
 *
 * --sign-from / --sign-to restrict the build to a half-open slice [I, J) of the
 * SORTED sign-folder list, so the whole corpus can be built in several bounded
 * passes and the per-slice outputs concatenated. The slices are disjoint by
 * construction and every other decision -- the split, the exclusions, the
 * template, the feature preparation -- is made per sign, so a sliced build and
 * a single build produce the same `signs` map. Nothing else about the artifact
 * depends on which signs happen to be in the same pass. This exists because
 * reading ~3.8 GB of takes over a slow/mounted filesystem can exceed a single
 * process's time budget; a normal build omits both flags and is unchanged.
 *
 * --held-out builds the EVALUATION artifact: it omits every take that
 * lib/ml/karslTakeSplit.ts assigns to the benchmark or to calibration, so
 * prototypes and queries are disjoint by construction. Both this script and
 * the benchmark call that one function, because two files agreeing on an
 * arithmetic rule is not a split.
 *
 * The artifact the app ships uses EVERY take -- a deployed system is entitled
 * to all its data; only the measurement has to hold some back.
 */
import { readFileSync, writeFileSync, readdirSync, statSync, existsSync, mkdirSync } from "fs";
import path from "path";
import { prepareSidedSequence, FEATURE_VERSION, DTW_SEQUENCE_LENGTH } from "../lib/ml/twoHandMatch";
import { assignSides } from "../lib/ml/handSides";
import { withLegacySequence } from "../lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "../lib/sign-language/signRegistry";
import { encodeTake, type EncodedTake, type PrototypeFile } from "../lib/ml/prototypeStore";
import { splitTakes } from "../lib/ml/karslTakeSplit";
import { DEFAULT_PROCESSED_ROOT } from "../lib/ml/karslPaths";
import { loadExclusions, excludedFor } from "../lib/ml/karslExclusions";
import type { HandLandmarks, SidedSequence } from "../types/landmarks";

function arg(name: string, fallback: string): string {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

interface ProcessedHand {
    landmarks: HandLandmarks;
    rawLabel?: "Left" | "Right";
}
interface ProcessedFrame {
    hands?: { left?: ProcessedHand | null; right?: ProcessedHand | null };
}
interface ProcessedTake {
    frames?: ProcessedFrame[];
}

/** Canonical sides, derived the way the importer and the camera derive them. */
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
        const frame: { left: HandLandmarks | null; right: HandLandmarks | null } = {
            left: null,
            right: null,
        };
        frame[assignment.sides[0] as "left" | "right"] = slotA[i];
        frame[assignment.sides[1] as "left" | "right"] = slotB[i];
        if (frame.left || frame.right) out.push(frame);
    }
    return out;
}

function main(): void {
    const processedRoot = arg("processed", DEFAULT_PROCESSED_ROOT);
    const registryPath = arg("registry", "public/models/sign-vocabulary/dedicated-signs.json");
    const outPath = arg("out", "public/models/sign-vocabulary/sign-prototypes.json");
    const maxTakes = Number(arg("max-takes", "0")) || Infinity;
    const heldOut = process.argv.includes("--held-out");
    const signFrom = Number(arg("sign-from", "0"));
    const signToRaw = arg("sign-to", "");
    const signTo = signToRaw === "" ? Infinity : Number(signToRaw);
    if (!Number.isFinite(signFrom) || signFrom < 0 || (signTo !== Infinity && !(signTo > signFrom))) {
        console.error(`Invalid --sign-from/--sign-to range: [${signFrom}, ${signToRaw || "end"})`);
        process.exit(1);
    }

    if (!existsSync(processedRoot)) {
        console.error(
            `No processed data at "${processedRoot}". Run ml/training/extract_karsl_skeletons.py first.`
        );
        process.exit(1);
    }

    // Only signs that are actually in the registry can be recognised, so only
    // those get prototypes.
    const registry = (JSON.parse(readFileSync(registryPath, "utf-8")) as DedicatedSignEntry[]).map(
        withLegacySequence
    );
    const known = new Set<string>();
    const templateFileOf = new Map<string, string>();
    for (const entry of registry) {
        if (!entry.sourceSignId) continue;
        known.add(entry.sourceSignId);
        if (entry.sourceTake?.processedFile) {
            templateFileOf.set(entry.sourceSignId, entry.sourceTake.processedFile);
        }
    }

    // Takes the audit flagged are dropped from BOTH builds. A spliced
    // trajectory is not better training data just because it ships.
    const exclusions = loadExclusions(arg("exclusions", "karsl-exclusions.json"));
    let skippedExcluded = 0;

    const signs: Record<string, EncodedTake[]> = {};
    let takeCount = 0;
    let skippedUnusable = 0;
    let skippedUnknownSign = 0;

    // Slice the SORTED DIRECTORY list, not the raw readdir listing: the corpus
    // root also holds files (manifest.json), and indices have to mean the same
    // thing in every pass for the slices to be disjoint and exhaustive.
    const signDirs = readdirSync(processedRoot)
        .sort()
        .filter((name) => statSync(path.join(processedRoot, name)).isDirectory());
    const selectedDirs = signDirs.slice(
        signFrom,
        signTo === Infinity ? undefined : signTo
    );

    for (const signId of selectedDirs) {
        const dir = path.join(processedRoot, signId);
        if (!known.has(signId)) {
            skippedUnknownSign++;
            continue;
        }
        const encoded: EncodedTake[] = [];
        const all = readdirSync(dir).filter((f) => f.endsWith(".json")).sort();
        const excluded = excludedFor(exclusions, signId);
        const split = splitTakes(all, templateFileOf.get(signId) ?? null, excluded);
        const files = heldOut
            ? split.prototypes
            : all.filter((f) => !excluded.has(f.replace(/\.[^.]+$/, "")));
        skippedExcluded += split.excluded.length;
        for (const file of files) {
            if (encoded.length >= maxTakes) break;
            const take = JSON.parse(readFileSync(path.join(dir, file), "utf-8")) as ProcessedTake;
            const sided = sidedFromTake(take);
            if (sided.length < 2) {
                skippedUnusable++;
                continue;
            }
            const entry = encodeTake(file, prepareSidedSequence(sided));
            if (!entry) {
                skippedUnusable++;
                continue;
            }
            encoded.push(entry);
        }
        if (encoded.length > 0) {
            signs[signId] = encoded;
            takeCount += encoded.length;
        }
    }

    const out: PrototypeFile = {
        version: 1,
        featureVersion: FEATURE_VERSION,
        sequenceLength: DTW_SEQUENCE_LENGTH,
        generatedAt: new Date().toISOString(),
        signs,
    };
    mkdirSync(path.dirname(outPath), { recursive: true });
    writeFileSync(outPath, JSON.stringify(out), "utf-8");

    const bytes = statSync(outPath).size;
    const signCount = Object.keys(signs).length;
    console.log("=".repeat(72));
    console.log("SIGN PROTOTYPES BUILT");
    console.log("=".repeat(72));
    console.log(`   feature version        ${FEATURE_VERSION}`);
    if (heldOut) {
        console.log(`   HELD-OUT BUILD: benchmark and calibration takes omitted`);
    }
    if (signFrom !== 0 || signTo !== Infinity) {
        console.log(
            `   PARTIAL BUILD: sign slice [${signFrom}, ${signTo === Infinity ? signDirs.length : signTo})` +
            ` of ${signDirs.length} sign folders -- concatenate the slices`
        );
    }
    console.log(`   signs with prototypes  ${signCount} / ${known.size} in the registry`);
    console.log(`   prototypes (takes)     ${takeCount}`);
    console.log(`   mean per sign          ${signCount ? (takeCount / signCount).toFixed(2) : "0"}`);
    console.log(`   takes skipped, unusable        ${skippedUnusable}`);
    console.log(
        `   takes skipped, audit-excluded  ${skippedExcluded}` +
        `  (exclusion list ${exclusions.loaded ? "loaded" : "NOT FOUND"})`
    );
    console.log(`   sign folders not in the registry ${skippedUnknownSign}`);
    console.log(`   output                 ${outPath}  (${(bytes / 1024 / 1024).toFixed(1)} MB)`);
    const isPartial = signFrom !== 0 || signTo !== Infinity;
    if (signCount < known.size && !isPartial) {
        console.log(
            `\n   NOTE: ${known.size - signCount} registry signs have no processed takes here and will\n` +
            `   keep using their single stored template. That is supported, not an error.`
        );
    }
    console.log("=".repeat(72));
}

main();
