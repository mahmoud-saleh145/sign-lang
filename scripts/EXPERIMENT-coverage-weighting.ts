/**
 * EXPERIMENTAL HARNESS -- NOT PRODUCTION. Nothing in lib/ml/ imports this.
 *
 * QUESTION
 *   Does replacing twoHandMatch.ts's SATURATING frame-count weight
 *       w_i = min(q.detectedFrames_i, t.detectedFrames_i)
 *   with a normalized COVERAGE FRACTION raise accepted-and-correct without
 *   raising false accepts?
 *
 * BOUNDARY
 *   sidedDistance() accepts only "equal" | "coverage", so the candidate cannot
 *   be expressed without editing the production matcher. This file therefore
 *   imports the PRODUCTION preparation and DTW primitives UNCHANGED and
 *   re-implements only the final COMBINATION step. Gates and thresholds are
 *   imported from production and never varied.
 *
 * FIDELITY NOTES (each verified against production before writing)
 *   - A sign is scored by its NEAREST prototype, as dynamicSignRecognizer.ts
 *     does -- not by an arbitrary one.
 *   - The swapped orientation is tried ONLY for blind queries
 *     (prepareQuery, dynamicSignRecognizer.ts:503-516). Calibration queries
 *     come from stored takes with geometry-decided sides, so the named
 *     orientation alone is used here, exactly as production would.
 *   - The UNMATCHED_SIDE_COST term and the "relation track joins the average
 *     on the same footing" rule are reproduced from sidedDistance().
 *
 * SPLIT
 *   Queries are splitTakes(...).calibration only. The benchmark split
 *   (split.benchmark) is never read. Stored templates are excluded by filename
 *   AND by sided fingerprint.
 *
 * COST
 *   The expensive part -- dtwDistance per side per prototype -- is independent
 *   of the combination formula, so every candidate is scored from ONE sweep.
 */
import { readFileSync, existsSync, readdirSync, statSync } from "fs";
import path from "path";
import {
    prepareSidedSequence,
    type PreparedSidedSequence,
    type PreparedSide,
} from "../lib/ml/twoHandMatch";
import { UNMATCHED_SIDE_COST } from "../lib/ml/twoHandMatch";
import { dtwDistance } from "../lib/ml/dtw";
import { decodePrototypeFile } from "../lib/ml/prototypeStore";
import { splitTakes } from "../lib/ml/karslTakeSplit";
import { DEFAULT_PROCESSED_ROOT } from "../lib/ml/karslPaths";
import { loadExclusions, excludedFor } from "../lib/ml/karslExclusions";
import {
    entryToSidedSequence,
    withLegacySequence,
    type DedicatedSignEntry,
} from "../lib/sign-language/signRegistry";
import {
    MAX_NORMALIZED_DISTANCE,
    MAX_BEST_TO_SECOND_BEST_RATIO,
} from "../lib/ml/dynamicSignRecognizer";
import type { HandLandmarks, SidedSequence } from "../types/landmarks";

const ROOT = path.resolve(__dirname, "..");

function arg(name: string, fallback: string): string {
    const i = process.argv.indexOf(`--${name}`);
    return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

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
    handTracking?: { primaryHand?: string; slotNames?: string[] };
    frames?: ProcessedFrame[];
}

/** Mirrors ml/training/handedness.py: geometry first, label majority second. */
function sidedFromTake(take: ProcessedTake): SidedSequence {
    const frames = take.frames ?? [];
    const slots: Array<Array<HandLandmarks | null>> = [[], []];
    const labels: Array<Array<string | null>> = [[], []];
    for (const fr of frames) {
        const pair: Array<ProcessedHand | null | undefined> = [fr.hands?.left, fr.hands?.right];
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

/** MUST stay identical to scripts/calibrate-two-hand.ts. */
function fingerprint(frames: SidedSequence): string {
    const wrist = (h: HandLandmarks | null) =>
        h ? `${h[0].x.toFixed(4)},${h[0].y.toFixed(4)}` : "-";
    return `${frames.length}|${frames.map((f) => `${wrist(f.left)}/${wrist(f.right)}`).join(";")}`;
}

type TrackSide = "left" | "right" | "relation";
interface Track {
    side: TrackSide;
    d: number;
    qDet: number;
    tDet: number;
    qTot: number;
    tTot: number;
}
interface PairCost {
    tracks: Track[];
    presenceMismatches: number;
}

const cov = (det: number, tot: number): number => (tot > 0 ? Math.min(1, det / tot) : 1);

type Formula = (p: PairCost) => number;

/** Shared combination, identical to sidedDistance() except the weight function
 *  is supplied by the caller instead of by SIDE_WEIGHTING. */
function combine(p: PairCost, weightOf: (t: Track) => number): number {
    if (p.tracks.length === 0) return Infinity;
    let num = 0;
    let den = 0;
    for (const t of p.tracks) {
        const w = weightOf(t);
        num += t.d * w;
        den += w;
    }
    const mean =
        den > 0
            ? num / den
            : p.tracks.reduce((a, t) => a + t.d, 0) / p.tracks.length;
    return mean + (UNMATCHED_SIDE_COST * p.presenceMismatches) / 2;
}

const CANDIDATES: Array<[string, Formula]> = [
    ["A equal (other shipped)", (p) => combine(p, () => 1)],
    ["B min-count (SHIPPED)", (p) =>
        combine(p, (t) => Math.min(t.qDet, t.tDet))],
    ["C query-cov x min-count", (p) =>
        combine(p, (t) => Math.min(t.qDet, t.tDet) * cov(t.qDet, t.qTot))],
    ["D both-cov x min-count", (p) =>
        combine(
            p,
            (t) =>
                Math.min(t.qDet, t.tDet) *
                Math.min(cov(t.qDet, t.qTot), cov(t.tDet, t.tTot))
        )],
    ["E relative query cov", (p) => {
        const best = p.tracks.reduce((m, t) => Math.max(m, cov(t.qDet, t.qTot)), 0);
        const norm = best > 0 ? best : 1;
        return combine(
            p,
            (t) => Math.min(t.qDet, t.tDet) * (cov(t.qDet, t.qTot) / norm)
        );
    }],
];

function pairCosts(q: PreparedSidedSequence, t: PreparedSidedSequence): PairCost {
    const tracks: Track[] = [];
    let presenceMismatches = 0;
    const sides: Array<[TrackSide, PreparedSide | null, PreparedSide | null]> = [
        ["left", q.left, t.left],
        ["right", q.right, t.right],
    ];
    for (const [name, qs, ts] of sides) {
        if (qs && ts) {
            tracks.push({
                side: name,
                d: dtwDistance(qs.vectors, ts.vectors).normalizedCost,
                qDet: qs.detectedFrames,
                tDet: ts.detectedFrames,
                qTot: q.totalFrames,
                tTot: t.totalFrames,
            });
        } else if (qs && !ts) presenceMismatches++;
    }
    if (q.relation && t.relation) {
        tracks.push({
            side: "relation",
            d: dtwDistance(q.relation.vectors, t.relation.vectors).normalizedCost,
            qDet: q.relation.detectedFrames,
            tDet: t.relation.detectedFrames,
            qTot: q.totalFrames,
            tTot: t.totalFrames,
        });
    }
    return { tracks, presenceMismatches };
}

interface Metrics {
    n: number; top1: number; accCorrect: number; falseAcc: number; acc: number;
    rejButCorrect: number; rejWrong: number;
    oneHand: number; twoHand: number; oneHandOK: number; twoHandOK: number;
}
const fresh = (): Metrics => ({
    n: 0, top1: 0, accCorrect: 0, falseAcc: 0, acc: 0, rejButCorrect: 0, rejWrong: 0,
    oneHand: 0, twoHand: 0, oneHandOK: 0, twoHandOK: 0,
});

function scoreQuery(
    m: Metrics, ranked: Array<{ id: string; d: number }>, truthId: string, oneHand: boolean
): void {
    if (ranked.length === 0) return;
    ranked.sort((a, b) => a.d - b.d);
    const best = ranked[0];
    const second = ranked[1]?.d ?? Infinity;
    const correct = best.id === truthId;
    m.n++;
    if (correct) m.top1++;
    if (oneHand) m.oneHand++;
    else m.twoHand++;
    const ok = best.d <= MAX_NORMALIZED_DISTANCE &&
        (!Number.isFinite(second) || best.d <= second * MAX_BEST_TO_SECOND_BEST_RATIO);
    if (ok) {
        m.acc++;
        if (correct) {
            m.accCorrect++;
            if (oneHand) m.oneHandOK++;
            else m.twoHandOK++;
        } else m.falseAcc++;
    } else if (correct) m.rejButCorrect++;
    else m.rejWrong++;
}

function main(): void {
    const processedRoot = arg("processed", DEFAULT_PROCESSED_ROOT);
    const registryPath = arg("registry", "public/models/sign-vocabulary/dedicated-signs.json");
    const prototypePath = arg("prototypes", "public/models/sign-vocabulary/sign-prototypes.eval.json");
    const stride = Math.max(1, Number(arg("stride", "7")));

    const registry = (JSON.parse(readFileSync(path.join(ROOT, registryPath), "utf-8")) as DedicatedSignEntry[]).map(
        withLegacySequence
    );
    const templateFingerprints = new Map<string, string>();
    const templateFiles = new Map<string, string>();
    for (const e of registry) {
        if (!e.sourceSignId) continue;
        const sided = entryToSidedSequence(e);
        if (sided) templateFingerprints.set(e.sourceSignId, fingerprint(sided));
        if (e.sourceTake?.processedFile) templateFiles.set(e.sourceSignId, e.sourceTake.processedFile);
    }

    const templates = new Map<string, PreparedSidedSequence[]>();
    const proto = decodePrototypeFile(
        JSON.parse(readFileSync(path.join(ROOT, prototypePath), "utf-8"))
    );
    if (proto) {
        for (const [id, takes] of proto.bySign) templates.set(id, takes);
    }
    if (templates.size === 0) {
        for (const e of registry) {
            if (!e.sourceSignId) continue;
            const sided = entryToSidedSequence(e);
            if (sided) templates.set(e.sourceSignId, [prepareSidedSequence(sided)]);
        }
    }
    let totalPrototypes = 0;
    for (const t of templates.values()) totalPrototypes += t.length;

    const exclusions = loadExclusions(path.join(ROOT, arg("exclusions", "karsl-exclusions.json")));
    const signList = readdirSync(path.join(ROOT, processedRoot))
        .filter((d) => statSync(path.join(ROOT, processedRoot, d)).isDirectory())
        .sort();

    const all: Array<{ id: string; prepared: PreparedSidedSequence; oneHand: boolean }> = [];
    for (const signId of signList) {
        const dir = path.join(ROOT, processedRoot, signId);
        if (!existsSync(dir) || !statSync(dir).isDirectory()) continue;
        const split = splitTakes(
            readdirSync(dir).filter((f) => f.endsWith(".json")),
            templateFiles.get(signId) ?? null,
            excludedFor(exclusions, signId)
        );
        for (const file of split.calibration) {
            const take = JSON.parse(readFileSync(path.join(dir, file), "utf-8")) as ProcessedTake;
            if (!take.handTracking?.primaryHand) continue;
            if ((take.frames ?? []).filter((f) => f.landmarks).length < 2) continue;
            const tf = templateFiles.get(signId);
            if (tf && file === tf) continue;
            const sided = sidedFromTake(take);
            if (sided.length < 2) continue;
            const fp = templateFingerprints.get(signId);
            if (fp && fingerprint(sided) === fp) continue;
            const prepared = prepareSidedSequence(sided);
            all.push({
                id: signId,
                prepared,
                oneHand: prepared.left === null || prepared.right === null,
            });
        }
    }
    const queries = all.filter((_, i) => i % stride === 0);

    console.log("=".repeat(104));
    console.log("EXPERIMENTAL COVERAGE-WEIGHTING CALIBRATION   (NOT PRODUCTION CODE)");
    console.log("=".repeat(104));
    console.log(`processed tree       ${processedRoot}`);
    console.log(`prototypes           ${path.basename(prototypePath)}  (${totalPrototypes} takes / ${templates.size} signs)`);
    console.log(`calibration queries  ${queries.length} of ${all.length} (deterministic stride ${stride})`);
    console.log(`split                splitTakes().calibration ONLY -- benchmark split never read`);
    console.log(`gates (imported)     dist<=${MAX_NORMALIZED_DISTANCE}  ratio<=${MAX_BEST_TO_SECOND_BEST_RATIO}  UNMATCHED_SIDE_COST=${UNMATCHED_SIDE_COST}`);

    const metrics = CANDIDATES.map(() => fresh());
    const ranked = CANDIDATES.map(() => [] as Array<{ id: string; d: number }>);
    const bestPerFormula = CANDIDATES.map(() => new Map<string, number>());

    const t0 = Date.now();
    for (let qi = 0; qi < queries.length; qi++) {
        const q = queries[qi];
        for (let f = 0; f < CANDIDATES.length; f++) bestPerFormula[f].clear();
        for (const [signId, takes] of templates) {
            for (const t of takes) {
                const cost = pairCosts(q.prepared, t);
                for (let f = 0; f < CANDIDATES.length; f++) {
                    const d = CANDIDATES[f][1](cost);
                    const prev = bestPerFormula[f].get(signId);
                    if (prev === undefined || d < prev) bestPerFormula[f].set(signId, d);
                }
            }
        }
        for (let f = 0; f < CANDIDATES.length; f++) {
            ranked[f] = [...bestPerFormula[f]].map(([id, d]) => ({ id, d }));
            scoreQuery(metrics[f], ranked[f], q.id, q.oneHand);
        }
        if ((qi + 1) % 50 === 0) {
            const el = (Date.now() - t0) / 1000;
            const eta = (el / (qi + 1)) * (queries.length - qi - 1);
            process.stdout.write(
                `  ${qi + 1}/${queries.length}  ${el.toFixed(0)}s elapsed  eta ${eta.toFixed(0)}s\n`
            );
        }
    }

    const pct = (a: number, b: number) => (b > 0 ? `${((a / b) * 100).toFixed(2)}%` : "n/a");
    console.log("\n" + "-".repeat(104));
    console.log(
        ["configuration".padEnd(26), "acc+correct".padEnd(13), "rate".padEnd(8),
         "fAcc".padEnd(6), "prec".padEnd(8), "accept".padEnd(9), "top1".padEnd(8),
         "rejOK".padEnd(8)].join(" ")
    );
    console.log("-".repeat(104));
    for (let f = 0; f < CANDIDATES.length; f++) {
        const m = metrics[f];
        const prec = m.acc > 0 ? (m.accCorrect / m.acc).toFixed(4) : "n/a";
        console.log(
            [
                CANDIDATES[f][0].padEnd(26),
                `${m.accCorrect}/${m.n}`.padEnd(13),
                pct(m.accCorrect, m.n).padEnd(8),
                String(m.falseAcc).padEnd(6),
                prec.padEnd(8),
                pct(m.acc, m.n).padEnd(9),
                pct(m.top1, m.n).padEnd(8),
                pct(m.rejButCorrect, m.n).padEnd(8),
            ].join(" ")
        );
        const oh = m.oneHand > 0 ? (m.oneHandOK / m.oneHand).toFixed(4) : "n/a";
        const th = m.twoHand > 0 ? (m.twoHandOK / m.twoHand).toFixed(4) : "n/a";
        console.log(`   one-hand n=${String(m.oneHand).padEnd(5)} acc+correct=${oh}   two-hand n=${String(m.twoHand).padEnd(5)} acc+correct=${th}`);
    }
    console.log("-".repeat(104));
    console.log(`sweep ${((Date.now() - t0) / 1000).toFixed(0)}s   (no benchmark take was read)`);
}

main();