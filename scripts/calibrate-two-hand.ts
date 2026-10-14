/**
 * CALIBRATION for the side-aware KArSL matcher.
 *
 * Sets, from real data rather than by tuning against the benchmark:
 *   - UNMATCHED_SIDE_COST  (lib/ml/twoHandMatch.ts)
 *   - SIDE_WEIGHTING       (lib/ml/twoHandMatch.ts)
 *   - MAX_NORMALIZED_DISTANCE / MAX_BEST_TO_SECOND_BEST_RATIO
 *                          (lib/ml/dynamicSignRecognizer.ts)
 *
 * Thresholds picked here are never re-picked on the benchmark: the benchmark
 * is run once, afterwards, with whatever this produced.
 *
 * HOW DISJOINT THE TWO SETS ACTUALLY ARE, since this used to claim more than
 * it could deliver. When the benchmark covered 40 signs the two sign lists
 * could be kept apart, and were. The benchmark now runs on every sign the
 * processed tree holds -- all 502 -- so the 30 signs passed to --signs are in
 * it too, and about 6% of its cases are not strictly held out from the
 * threshold choice. What IS still guaranteed is that no individual take is
 * ever scored against a template built from itself, on either side. Slice 10
 * of the benchmark report prints the calibration signs and everything else
 * separately, which is the check that the gate did not memorise them.
 *
 * Queries are REAL repetitions, and the take the importer stored as the
 * template is excluded two independent ways -- by `sourceTake.processedFile`,
 * and by a fingerprint of the SIDED timeline as a fallback for a registry old
 * enough to lack it. This is the same exclusion lib/ml/karslHeldOut.test.ts
 * uses, and it replaces a legacy-sequence fingerprint that did not work:
 * measured, 9 of this set's 60 takes were the stored template scoring against
 * itself, which is why calibration reported zero false accepts at an operating
 * point the benchmark showed admitting nine.
 *
 * --gate-only derives ONLY the acceptance gate. Sections 1 and 2 exist to set
 * UNMATCHED_SIDE_COST and the side weighting; they sweep 16 configurations
 * over every query and every sign, which at full corpus scale runs for hours.
 * When those constants are not being re-derived that work is wasted, so
 * --gate-only scores the shipped configuration once and goes straight to the
 * gate. It changes no formula: sections 3 and 4 receive exactly the scores
 * they would have received from the sweep.
 *
 * USAGE:
 *   npx tsx scripts/calibrate-two-hand.ts \
 *     --registry public/models/sign-vocabulary/dedicated-signs.json \
 *     --processed processed-karsl \
 *     --signs 0006,0016,0026,...
 */
import { readFileSync, existsSync, readdirSync, statSync } from "fs";
import path from "path";
import {
    prepareSidedSequence,
    sidedDistance,
    prepareSide,
    setUnmatchedSideCost,
    SIDE_WEIGHTING,
    UNMATCHED_SIDE_COST,
    type PreparedSidedSequence,
    type SideWeighting,
} from "../lib/ml/twoHandMatch";

/** Candidate prices for a side that only one of the two sequences has. */
const COST_CANDIDATES = [0, 1.0, 1.3, 1.58, 1.9, 2.2, 2.6, 3.0];
import { decodePrototypeFile } from "../lib/ml/prototypeStore";
import { splitTakes } from "../lib/ml/karslTakeSplit";
import { DEFAULT_PROCESSED_ROOT } from "../lib/ml/karslPaths";
import { loadExclusions, excludedFor } from "../lib/ml/karslExclusions";
import { entryToSidedSequence, withLegacySequence } from "../lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "../lib/sign-language/signRegistry";
import type { HandLandmarks, SidedSequence } from "../types/landmarks";

interface ProcessedHand {
    landmarks: HandLandmarks;
    score?: number;
    rawLabel?: string;
}
interface ProcessedFrame {
    landmarks?: HandLandmarks | null;
    hands?: { left?: ProcessedHand | null; right?: ProcessedHand | null };
}
interface ProcessedTake {
    signId?: string;
    handTracking?: { primaryHand?: string; slotNames?: string[] };
    frames?: ProcessedFrame[];
}

interface Query {
    signId: string;
    file: string;
    sided: SidedSequence;
    legacy: HandLandmarks[];
    /** Resampled once: the sweep below scores every query many times over. */
    prepared: PreparedSidedSequence;
}

function arg(name: string, fallback: string): string {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

/**
 * Content fingerprint of a TWO-HAND timeline -- per frame, each side's wrist at
 * 4 dp or "-" when absent. Must stay identical to the benchmark's: comparing
 * the LEGACY sequence instead compares the registry's primary-side
 * reconstruction against the extractor's slot-based one, two different
 * selections that usually disagree, so the template goes unrecognised.
 */
function fingerprint(frames: SidedSequence): string {
    const wrist = (h: HandLandmarks | null) =>
        h ? `${h[0].x.toFixed(4)},${h[0].y.toFixed(4)}` : "-";
    return `${frames.length}|${frames.map((f) => `${wrist(f.left)}/${wrist(f.right)}`).join(";")}`;
}

/**
 * Re-derives the canonical sides of a processed take the same way
 * ml/training/handedness.py does: geometry first, label majority second.
 * Kept here so the calibration reads the source tree exactly as the importer
 * would, rather than trusting whatever names the extractor happened to write.
 */
function sidedFromTake(take: ProcessedTake): SidedSequence {
    const frames = take.frames ?? [];
    const slots: Array<Array<HandLandmarks | null>> = [[], []];
    const labels: Array<Array<string | null>> = [[], []];
    for (const fr of frames) {
        const hands = fr.hands;
        const pair: Array<ProcessedHand | null | undefined> = [hands?.left, hands?.right];
        for (const i of [0, 1]) {
            const h = pair[i];
            slots[i].push(h?.landmarks ?? null);
            labels[i].push(h?.rawLabel ?? null);
        }
    }
    let aSmaller = 0;
    let compared = 0;
    for (let i = 0; i < slots[0].length; i++) {
        const a = slots[0][i];
        const b = slots[1][i];
        if (!a || !b) continue;
        compared++;
        if (a[0].x < b[0].x) aSmaller++;
    }
    let slot0: "left" | "right";
    if (compared >= 3 && Math.max(aSmaller, compared - aSmaller) / compared >= 0.9) {
        slot0 = aSmaller * 2 >= compared ? "right" : "left";
    } else {
        const seen = labels[0].filter((l): l is string => l === "Left" || l === "Right");
        let left = 0;
        for (const l of seen) if (l === "Left") left++;
        slot0 = seen.length === 0 ? "right" : left >= seen.length - left ? "left" : "right";
    }
    const slot1 = slot0 === "left" ? "right" : "left";
    const out: SidedSequence = [];
    for (let i = 0; i < slots[0].length; i++) {
        const frame: { left: HandLandmarks | null; right: HandLandmarks | null } = {
            left: null,
            right: null,
        };
        frame[slot0] = slots[0][i];
        frame[slot1] = slots[1][i];
        if (frame.left || frame.right) out.push(frame);
    }
    return out;
}

function main(): void {
    const registryPath = arg("registry", "public/models/sign-vocabulary/dedicated-signs.json");
    const processedRoot = arg("processed", DEFAULT_PROCESSED_ROOT);
    const gateOnly = process.argv.includes("--gate-only");
    /** The configuration production actually runs, so --gate-only measures it. */
    const shippedKey = `${SIDE_WEIGHTING}:${UNMATCHED_SIDE_COST}`;
    const explicitSigns = arg("signs", "")
        .split(",")
        .map((s) => s.trim())
        .filter(Boolean);
    // No --signs means EVERY sign in the processed tree. The take-level split
    // keeps calibration queries disjoint from benchmark queries regardless of
    // how many signs take part, so a full-corpus calibration is not a wider
    // net over the benchmark -- it is the same net over more evidence.
    const signList =
        explicitSigns.length > 0
            ? explicitSigns
            : existsSync(processedRoot)
              ? readdirSync(processedRoot)
                    .filter((d) => statSync(path.join(processedRoot, d)).isDirectory())
                    .sort()
              : [];

    const registry = (JSON.parse(readFileSync(registryPath, "utf-8")) as DedicatedSignEntry[]).map(
        withLegacySequence
    );

    const templates = new Map<string, PreparedSidedSequence>();
    const prototypesBySign = new Map<string, PreparedSidedSequence[]>();
    const templateFingerprints = new Map<string, string>();
    const templateFiles = new Map<string, string>();
    for (const entry of registry) {
        const id = entry.sourceSignId;
        if (!id) continue;
        const sided = entryToSidedSequence(entry);
        if (!sided) continue;
        templates.set(id, prepareSidedSequence(sided));
        if (sided.length > 0) templateFingerprints.set(id, fingerprint(sided));
        const file = entry.sourceTake?.processedFile;
        if (file) templateFiles.set(id, file);
    }

    // PROTOTYPES, and the three-way split that keeps this honest.
    //
    // Takes are sorted by filename and partitioned once:
    //     benchmark    queries for lib/ml/karslHeldOut.test.ts
    //     calibration  queries HERE, where the gate is derived
    //     prototypes   everything else, shared by both
    //
    // So the gate is never derived on a take the benchmark scores, and no
    // query is ever compared against a prototype built from itself. Build the
    // matching artifact with:
    //     npx tsx scripts/build-prototypes.ts --skip-first 4 \
    //       --out public/models/sign-vocabulary/sign-prototypes.eval.json
    const prototypePath = arg("prototypes", "public/models/sign-vocabulary/sign-prototypes.eval.json");
    const prototypes = existsSync(prototypePath)
        ? decodePrototypeFile(JSON.parse(readFileSync(prototypePath, "utf-8")))
        : null;
    if (prototypes) {
        for (const [id, takes] of prototypes.bySign) {
            if (takes.length > 0) templates.set(id, takes[0]);
            prototypesBySign.set(id, takes);
        }
    }

    const exclusions = loadExclusions(arg("exclusions", "karsl-exclusions.json"));
    const queries: Query[] = [];
    for (const signId of signList) {
        const dir = path.join(processedRoot, signId);
        if (!existsSync(dir) || !statSync(dir).isDirectory()) continue;
        const split = splitTakes(
            readdirSync(dir).filter((f) => f.endsWith(".json")),
            templateFiles.get(signId) ?? null,
            excludedFor(exclusions, signId)
        );
        for (const file of split.calibration) {
            const take = JSON.parse(readFileSync(path.join(dir, file), "utf-8")) as ProcessedTake;
            if (!take.handTracking?.primaryHand) continue;
            const legacy: HandLandmarks[] = [];
            for (const fr of take.frames ?? []) if (fr.landmarks) legacy.push(fr.landmarks);
            if (legacy.length < 2) continue;

            // Belt and braces: the split above already excludes the stored
            // template and every prototype, but the original two exclusions
            // stay so a changed split cannot silently reintroduce a tautology.
            const templateFile = templateFiles.get(signId);
            if (templateFile && file === templateFile) continue;
            const sided = sidedFromTake(take);
            if (sided.length < 2) continue;
            const tmpl = templateFingerprints.get(signId);
            if (tmpl && fingerprint(sided) === tmpl) continue;

            queries.push({ signId, file, sided, legacy, prepared: prepareSidedSequence(sided) });
        }
    }

    console.log("=".repeat(72));
    console.log("TWO-HAND MATCHER CALIBRATION");
    console.log("=".repeat(72));
    console.log(`Registry templates:            ${templates.size}`);
    console.log(`Calibration signs requested:   ${signList.length}`);
    // Not "takes 2-3" any more: the split holds out a whole recording session
    // and gives calibration every other take of it, so this is roughly half a
    // sitting per sign. Printing the stale description was how the last
    // corpus mismatch stayed invisible for a full run.
    console.log(
        `Held-out calibration queries:  ${queries.length}   ` +
        `(interleaved half of each sign's held-out session, from ${processedRoot})`
    );
    console.log(`Prototype signs loaded:        ${prototypesBySign.size}`);

    // --- 1. What does an UNINFORMATIVE side cost? -------------------------
    // Skipped by --gate-only: this distribution exists to price
    // UNMATCHED_SIDE_COST, which --gate-only does not re-derive.
    // Per-side DTW distance between a query and the template of a DIFFERENT
    // sign. That is the price of a side that tells you nothing, and therefore
    // the correct price for a side that is simply absent.
    const wrongPairDistances: number[] = [];
    const correctPairDistances: number[] = [];
    for (const q of gateOnly ? [] : queries) {
        const prepared = q.prepared;
        for (const [id, t] of templates) {
            if (wrongPairDistances.length > 200000) break;
            for (const side of ["left", "right"] as const) {
                const a = prepared[side];
                const b = t[side];
                if (!a || !b) continue;
                const d = sidedDistance(
                    { left: side === "left" ? a : null, right: side === "right" ? a : null, unknown: null, relation: null, totalFrames: 0 },
                    { left: side === "left" ? b : null, right: side === "right" ? b : null, unknown: null, relation: null, totalFrames: 0 }
                ).distance;
                if (id === q.signId) correctPairDistances.push(d);
                else wrongPairDistances.push(d);
            }
        }
    }
    const quantile = (xs: number[], p: number): number => {
        if (xs.length === 0) return NaN;
        const s = [...xs].sort((a, b) => a - b);
        return s[Math.min(s.length - 1, Math.floor(p * s.length))];
    };
    if (gateOnly) {
        console.log("\n1) PER-SIDE DTW DISTANCE DISTRIBUTION -- skipped (--gate-only)");
    } else {
    console.log("\n1) PER-SIDE DTW DISTANCE DISTRIBUTION");
    console.log(
        `   same sign  (n=${correctPairDistances.length}): p10 ${quantile(correctPairDistances, 0.1).toFixed(3)}  ` +
        `median ${quantile(correctPairDistances, 0.5).toFixed(3)}  p90 ${quantile(correctPairDistances, 0.9).toFixed(3)}`
    );
    console.log(
        `   other sign (n=${wrongPairDistances.length}): p10 ${quantile(wrongPairDistances, 0.1).toFixed(3)}  ` +
        `median ${quantile(wrongPairDistances, 0.5).toFixed(3)}  p90 ${quantile(wrongPairDistances, 0.9).toFixed(3)}`
    );
    console.log(
        `   => UNMATCHED_SIDE_COST should be the "other sign" median: ${quantile(wrongPairDistances, 0.5).toFixed(2)}`
    );
    }

    // --- 2. Which side weighting matches best? ---------------------------
    interface Scored {
        best: string | null;
        bestDistance: number;
        secondDistance: number;
    }
    const scoreAll = (q: Query, weighting: SideWeighting): Scored => {
        const prepared = q.prepared;
        const results: Array<{ id: string; d: number }> = [];
        for (const [id, t] of templates) {
            const takes = prototypesBySign.get(id);
            if (takes && takes.length > 0) {
                let best = Infinity;
                for (const take of takes) {
                    const d = sidedDistance(prepared, take, weighting).distance;
                    if (d < best) best = d;
                }
                results.push({ id, d: best });
            } else {
                results.push({ id, d: sidedDistance(prepared, t, weighting).distance });
            }
        }
        results.sort((a, b) => a.d - b.d);
        return {
            best: results[0]?.id ?? null,
            bestDistance: results[0]?.d ?? Infinity,
            secondDistance: results[1]?.d ?? Infinity,
        };
    };

    const scoredByKey = new Map<string, Scored[]>();
    if (gateOnly) {
        // ONE configuration, the shipped one, scored exactly as section 2
        // would have scored it. No sweep, no alternative weightings.
        const [weighting, costText] = shippedKey.split(":");
        setUnmatchedSideCost(Number(costText));
        const scored = queries.map((q) => scoreAll(q, weighting as SideWeighting));
        scoredByKey.set(shippedKey, scored);
        const hits = scored.filter((s, i) => s.best === queries[i].signId).length;
        console.log(`\n2) SWEEP -- skipped (--gate-only). Scored the shipped configuration only: ${shippedKey}`);
        console.log(`   top-1 on the calibration queries: ${hits}/${queries.length} = ${(hits / queries.length).toFixed(4)}`);
    } else {
    console.log("\n2) SIDE WEIGHTING x UNMATCHED_SIDE_COST (top-1 on the calibration set)");
    for (const weighting of ["equal", "coverage"] as SideWeighting[]) {
        for (const cost of COST_CANDIDATES) {
            setUnmatchedSideCost(cost);
            const scored = queries.map((q) => scoreAll(q, weighting));
            scoredByKey.set(`${weighting}:${cost}`, scored);
            const hits = scored.filter((s, i) => s.best === queries[i].signId).length;
            // False accepts at a FIXED reference gate, so the effect of the
            // cost on precision is visible separately from its effect on top-1.
            let fa = 0;
            let acc = 0;
            scored.forEach((sc, i) => {
                if (sc.bestDistance <= 1.5 && sc.bestDistance <= sc.secondDistance * 0.7) {
                    acc++;
                    if (sc.best !== queries[i].signId) fa++;
                }
            });
            console.log(
                `   ${weighting.padEnd(9)} cost=${String(cost).padStart(5)}  top1 = ${hits}/${queries.length} = ${(hits / queries.length).toFixed(4)}` +
                `   @gate(1.5,0.70): accepted ${acc}, falseAccepts ${fa}`
            );
        }
    }
    }

    // --- 3. Threshold sweep, precision first ------------------------------
    console.log("\n3) ACCEPTANCE GATES (calibration set; precision must stay 1.000)");
    const chosen = arg("chosen", shippedKey);
    console.log(`   (using ${chosen})`);
    const best = scoredByKey.get(chosen) ?? [];
    console.log("   dist<=   ratio<=  accepted  correct  falseAccepts  recall");
    for (const maxDist of [0.75, 1.0, 1.25, 1.5, 2.0, Infinity]) {
        for (const maxRatio of [0.6, 0.65, 0.7, 0.75, 0.8]) {
            let accepted = 0;
            let correct = 0;
            let falseAccepts = 0;
            best.forEach((s, i) => {
                const passes =
                    s.bestDistance <= maxDist && s.bestDistance <= s.secondDistance * maxRatio;
                if (!passes) return;
                accepted++;
                if (s.best === queries[i].signId) correct++;
                else falseAccepts++;
            });
            console.log(
                `   ${String(maxDist).padStart(6)}   ${maxRatio.toFixed(2)}    ${String(accepted).padStart(6)}   ` +
                `${String(correct).padStart(6)}   ${String(falseAccepts).padStart(10)}   ${(accepted / queries.length).toFixed(3)}`
            );
        }
    }
    // --- 4. The gate itself, derived rather than picked off the grid ------
    // For each ratio, the binding constraint is the NEAREST WRONG match that
    // already survives the ratio test: the distance gate has to sit below it,
    // or that wrong word is spoken. So read that distance off the data and
    // stop just short of it, instead of hoping a round number lands there.
    console.log("\n4) DERIVED GATE (zero false accepts is the hard constraint)");
    console.log("   ratio<=  nearest wrong  => dist<=   accepted  recall  precision   @0.90x  accepted");
    for (const maxRatio of [0.6, 0.65, 0.7, 0.75, 0.8]) {
        const survives = (s: Scored) => s.bestDistance <= s.secondDistance * maxRatio;
        let nearestWrong = Infinity;
        best.forEach((s, i) => {
            if (!survives(s)) return;
            if (s.best !== queries[i].signId) nearestWrong = Math.min(nearestWrong, s.bestDistance);
        });
        // Two decimal places, strictly below the nearest wrong match. With no
        // wrong match at all the ratio alone is already sufficient.
        const gate = Number.isFinite(nearestWrong)
            ? Math.max(0, Math.floor(nearestWrong * 100 - 1) / 100)
            : Infinity;
        let accepted = 0;
        let correct = 0;
        best.forEach((s, i) => {
            if (!survives(s) || s.bestDistance > gate) return;
            accepted++;
            if (s.best === queries[i].signId) correct++;
        });
        // A gate resting on the single nearest wrong match is fitted to one
        // sample. The 0.90x column is what a 10% margin below it costs, and
        // is the value to prefer unless the margin is free.
        const safe = Number.isFinite(nearestWrong)
            ? Math.floor(nearestWrong * 90) / 100
            : Infinity;
        let safeAccepted = 0;
        best.forEach((s, i) => {
            if (!survives(s) || s.bestDistance > safe) return;
            safeAccepted++;
            void i;
        });
        console.log(
            `   ${maxRatio.toFixed(2)}    ${(Number.isFinite(nearestWrong) ? nearestWrong.toFixed(4) : "none").padStart(13)}` +
            `   ${(Number.isFinite(gate) ? gate.toFixed(2) : "inf").padStart(7)}   ${String(accepted).padStart(8)}` +
            `   ${(accepted / queries.length).toFixed(3)}   ${accepted ? (correct / accepted).toFixed(3) : "n/a"}` +
            `     ${(Number.isFinite(safe) ? safe.toFixed(2) : "inf").padStart(5)}   ${String(safeAccepted).padStart(7)}`
        );
    }

    // --- INTEGRITY GUARD --------------------------------------------------
    // Every number above is void if a query was scored against a prototype
    // built from itself. Nothing that is genuinely a DIFFERENT recording of a
    // sign lands anywhere near zero distance, so a near-zero best distance on
    // the correct sign means the split broke. Same threshold and the same
    // reasoning as lib/ml/karslHeldOut.test.ts; two exclusion bugs have
    // already done exactly this, and neither was visible in the numbers.
    const tautologies: string[] = [];
    best.forEach((s, i) => {
        if (s.best === queries[i].signId && s.bestDistance < 0.02) {
            tautologies.push(`${queries[i].signId}/${queries[i].file} at ${s.bestDistance.toFixed(5)}`);
        }
    });
    console.log("\n5) INTEGRITY GUARD -- no take scored against itself");
    if (tautologies.length === 0) {
        console.log(`   PASS: ${queries.length} queries, none within 0.02 of a stored prototype.`);
    } else {
        console.log(`   FAIL: ${tautologies.length} query/prototype pair(s) are the same take:`);
        for (const t of tautologies.slice(0, 20)) console.log(`      ${t}`);
        console.log("   The thresholds above are NOT valid. Rebuild the held-out artifact:");
        console.log("   npm run prototypes:eval");
    }
    console.log("=".repeat(72));
    if (tautologies.length > 0) process.exitCode = 1;
    void prepareSide;
}

main();
