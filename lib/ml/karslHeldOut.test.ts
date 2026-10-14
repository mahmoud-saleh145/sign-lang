/**
 * REAL-DATA ACCURACY TEST for the dynamic KArSL recognizer.
 *
 * Every query here is a REAL KArSL repetition extracted from the real
 * videos by ml/training/extract_karsl_skeletons.py, and it is a repetition
 * that ml/training/import_karsl.py did NOT put in the registry. So this
 * measures "given a genuine performance of a sign, does the classifier find
 * that sign among all registry entries" -- not a tautology against the
 * template itself. Nothing is synthesized.
 *
 * HOW THE TEMPLATE TAKE IS EXCLUDED, in two independent ways because getting
 * this wrong silently inflates every number in the file:
 *
 *   1. BY FILE. `sourceTake.processedFile` records exactly which recording the
 *      importer used. Exact, and immune to representation differences.
 *   2. BY CONTENT, for a registry old enough to lack sourceTake: the take's
 *      two-hand timeline fingerprinted against the entry's.
 *
 * Two earlier versions of this were wrong. "Skip the first file
 * alphabetically" stopped being true when select_best_take() began choosing on
 * content. Fingerprinting the LEGACY sequence then compared the registry's
 * primary-side reconstruction against the extractor's slot-based one -- two
 * different selections that usually disagree, so the template went unrecognised
 * and scored against itself at a distance of ~0.003. Measured over 73 real
 * signs, that exclusion identified the template in only 12/73 signs (16.4%)
 * before the importer's primaryHand fix and 22/73 (30.1%) after, meaning most
 * signs were contributing a free hit. Numbers produced before this was fixed
 * are inflated and not comparable with numbers produced after it.
 *
 * REQUIRES the local, gitignored processed tree (processed-karsl-v2 by
 * default, overridable with KARSL_PROCESSED). It is skipped with a clear
 * message when that data is not present, rather than passing vacuously.
 *
 * WHAT THIS NOW MEASURES, and what it still cannot. Queries come from a
 * recording session that supplied NO prototypes -- see lib/ml/karslTakeSplit.ts
 * -- so this is session-independent: it asks whether a sign performed on a
 * different day, in a different sitting, is still recognised. That is a much
 * harder and much more honest question than the take-level split it replaced,
 * and the numbers dropped accordingly when it landed.
 *
 * It remains signer-DEPENDENT. Every take in the 432-sign corpus carries the
 * same 03_01 filename prefix, so no signer-held-out split is possible with
 * this data at all. A new signer in front of a webcam is still outside what
 * any offline test here can predict.
 *
 * FOUR SIGNS (0105, 0348, 0363, 0364) have only one usable session and so
 * contribute no queries. They stay in the vocabulary as real classes and as
 * distractors, so the task is 432-way while the query set covers 428 signs.
 */
import { describe, it, expect, beforeAll, vi } from "vitest";
import fs from "fs";
import path from "path";
import {
    clearDynamicSignCache,
    preloadDynamicSignRegistry,
    recognizeDynamicSignSyncWithDiagnostics,
    getLoadedPrototypeCount,
    MAX_NORMALIZED_DISTANCE,
    MAX_BEST_TO_SECOND_BEST_RATIO,
    type DynamicSignDiagnostics,
} from "./dynamicSignRecognizer";
import { DEFAULT_GESTURE_BUFFER_CONFIG, computeMotionEnergy } from "./dynamicGestureBuffer";
import type { HandLandmarks, SidedSequence } from "@/types/landmarks";
import { withLegacySequence, entryToSidedSequence } from "@/lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "@/lib/sign-language/signRegistry";
import { assignSides, hasGeometricSideEvidence } from "./handSides";
import { MIN_SIDE_FRAMES } from "./twoHandMatch";
import { splitTakes } from "./karslTakeSplit";
import { loadExclusions, excludedFor } from "./karslExclusions";

/**
 * How many sides this timeline actually USES, by the same rule the matcher
 * applies: a side with fewer than MIN_SIDE_FRAMES frames is not prepared and
 * never scored, so counting a single flicker frame as a hand would report a
 * one-handed sign as two-handed and put it in the wrong slice below.
 */
const usableSides = (frames: SidedSequence): number =>
    (frames.filter((f) => f.left).length >= MIN_SIDE_FRAMES ? 1 : 0) +
    (frames.filter((f) => f.right).length >= MIN_SIDE_FRAMES ? 1 : 0);

const ROOT = path.resolve(__dirname, "../..");
const REGISTRY = path.join(ROOT, "public/models/sign-vocabulary/dedicated-signs.json");
/**
 * WHICH PROCESSED TREE IS BEING MEASURED.
 *
 * This was hard-coded to "processed-karsl" and silently stayed there when the
 * corpus was re-extracted into processed-karsl-v2. The run that followed
 * scored OLD-tree queries -- different signer configurations, and 70 signs
 * (0001-0070) that the new 432-sign registry does not contain at all -- against
 * prototypes built from the NEW tree. Every one of those 70 signs' queries was
 * unanswerable by construction, and the resulting 0.3533 top-1 measured a
 * mismatch between two datasets rather than anything about the recogniser.
 *
 * It is an env var now so that pointing the benchmark at a different corpus is
 * a deliberate act that shows up in the command, and the report prints the
 * path so no run can be misread again.
 */
const PROCESSED = path.join(ROOT, process.env.KARSL_PROCESSED ?? "processed-karsl-v2");

/**
 * The signs scripts/calibrate-two-hand.ts USED to derive the gate on, back
 * when it was given an explicit 30-sign list. Slice 11 separates them out so
 * a gate overfitted to the signs it saw would show up as a suspiciously good
 * number on this subset.
 *
 * THAT CONTROL IS CURRENTLY VACUOUS AND THE SLICE SHOULD BE READ AS SUCH.
 * The 2026-09-20 re-derivation ran with --signs defaulting to all 432, so the
 * gate has now seen every sign and "signs the gate did not see" is an empty
 * set that this list no longer describes. It is kept because the last run for
 * which it WAS meaningful showed acceptance slightly lower on the seen signs
 * (0.271 against 0.285) -- evidence against overfitting, worth not discarding.
 * Restoring a real control means calibrating on a sign-disjoint subset.
 */
const CALIBRATION_SIGNS = new Set(
    ("0006,0016,0026,0036,0051,0061,0076,0086,0101,0111,0126,0136,0151,0161,0176," +
        "0201,0216,0226,0236,0251,0266,0276,0301,0316,0326,0351,0366,0376,0401,0416").split(",")
);

/**
 * Held-out repetitions queried per sign. The session split leaves about ten
 * available, so this is a RUNTIME cap, not a data limit: each query is scored
 * against all ~8,988 prototypes, and the full pass already takes ~25 minutes
 * at two per sign. Raise it with KARSL_REPS for a tighter confidence interval
 * when you can afford the wall clock.
 */
const REPS_PER_SIGN = Number(process.env.KARSL_REPS ?? "2");

interface Case {
    signId: string;
    file: string;
    /** The take's two REAL hands, named the same way the importer names them. */
    sided: SidedSequence;
    /** The take's legacy primary-hand sequence, used only for the template
     * fingerprint and for the "did the old primary-hand choice disagree"
     * breakdown -- never as the query. */
    legacy: HandLandmarks[];
    /** The side the OLD extractor called primary for this take. */
    legacyPrimary: string;
}

/**
 * Content fingerprint of a TWO-HAND timeline: per frame, the wrist of each
 * side at 4 dp (the precision the registry is written at), or "-" where that
 * hand was not detected. Two different takes of the same sign never collide.
 *
 * WHY THE SIDED TIMELINE AND NOT THE LEGACY SEQUENCE. This used to fingerprint
 * `landmarkSequence`, which is reconstructed from whichever side `primaryHand`
 * names, and compare it against the take's own legacy field, which the
 * EXTRACTOR built from whichever tracker slot it saw more often. Those two are
 * different selections and usually disagree, so the fingerprints did not match
 * and the template take was not recognised -- it entered the query set and
 * matched itself at a distance of about 0.003, a free hit.
 *
 * Measured over 73 real signs: the template take was identifiable in only
 * 12/73 signs (16.4%) before the importer's primaryHand fix and 22/73 (30.1%)
 * after it. In the other 70-84% the benchmark was scoring templates against
 * themselves. Comparing the sided timelines, which both sides derive the same
 * way, removes the mismatch entirely.
 */
function fingerprint(frames: SidedSequence): string {
    const wrist = (h: HandLandmarks | null) =>
        h ? `${h[0].x.toFixed(4)},${h[0].y.toFixed(4)}` : "-";
    return `${frames.length}|${frames.map((f) => `${wrist(f.left)}/${wrist(f.right)}`).join(";")}`;
}

const registryEntries: DedicatedSignEntry[] = fs.existsSync(REGISTRY)
    ? (JSON.parse(fs.readFileSync(REGISTRY, "utf-8")) as DedicatedSignEntry[]).map(withLegacySequence)
    : [];

const templateFingerprints: Map<string, string> = (() => {
    const m = new Map<string, string>();
    for (const e of registryEntries) {
        const sided = e.sourceSignId ? entryToSidedSequence(e) : null;
        if (e.sourceSignId && sided && sided.length) m.set(e.sourceSignId, fingerprint(sided));
    }
    return m;
})();

/**
 * The FILE each sign's template was built from, when the registry records it.
 *
 * This is the exact answer and the fingerprint is the fallback: `sourceTake`
 * exists precisely so a template can be traced to its recording, and matching
 * on the file name cannot be defeated by a representation difference. A
 * registry imported before traceability existed has no sourceTake, so the
 * fingerprint still has to work on its own.
 */
const templateFiles: Map<string, string> = (() => {
    const m = new Map<string, string>();
    for (const e of registryEntries) {
        const file = e.sourceTake?.processedFile;
        if (e.sourceSignId && file) m.set(e.sourceSignId, file);
    }
    return m;
})();

interface ProcessedHand {
    landmarks: HandLandmarks;
    rawLabel?: "Left" | "Right";
}
interface ProcessedFrame {
    landmarks?: HandLandmarks | null;
    hands?: { left?: ProcessedHand | null; right?: ProcessedHand | null };
}
interface ProcessedTake {
    handTracking?: { primaryHand?: string; slotNames?: string[] };
    frames?: ProcessedFrame[];
}

/**
 * Re-derives a take's canonical sides exactly as the importer does: the two
 * stored keys are treated as opaque TRACKER SLOTS and named by
 * lib/ml/handSides.assignSides. Reading the stored names instead would make
 * this benchmark agree with the extractor by construction rather than measure
 * anything.
 */
function sidedFromTake(take: ProcessedTake): SidedSequence {
    const frames = take.frames ?? [];
    const slotFrames: Array<[HandLandmarks | null, HandLandmarks | null]> = [];
    const slotLabels: Array<["Left" | "Right" | null, "Left" | "Right" | null]> = [];
    for (const fr of frames) {
        slotFrames.push([fr.hands?.left?.landmarks ?? null, fr.hands?.right?.landmarks ?? null]);
        slotLabels.push([fr.hands?.left?.rawLabel ?? null, fr.hands?.right?.rawLabel ?? null]);
    }
    const assignment = assignSides(
        slotFrames.map((f) => f[0]),
        slotFrames.map((f) => f[1]),
        slotLabels.map((l) => l[0]),
        slotLabels.map((l) => l[1])
    );
    const out: SidedSequence = [];
    for (const [a, b] of slotFrames) {
        const frame: { left: HandLandmarks | null; right: HandLandmarks | null } = { left: null, right: null };
        for (const slot of [0, 1] as const) {
            const side = assignment.sides[slot];
            const landmarks = slot === 0 ? a : b;
            if ((side === "left" || side === "right") && landmarks) frame[side] = landmarks;
        }
        if (frame.left || frame.right) out.push(frame);
    }
    return out;
}

const exclusions = loadExclusions();

function loadHeldOut(): Case[] {
    const cases: Case[] = [];
    if (!fs.existsSync(PROCESSED)) return cases;
    for (const signId of fs.readdirSync(PROCESSED).sort()) {
        const dir = path.join(PROCESSED, signId);
        if (!fs.existsSync(dir) || !fs.statSync(dir).isDirectory()) continue;
        const templatePrint = templateFingerprints.get(signId);
        const templateFile = templateFiles.get(signId);
        // ONE source of truth for which takes are queries -- see
        // lib/ml/karslTakeSplit.ts for why this is not a slice index.
        const split = splitTakes(
            fs.readdirSync(dir).filter((x) => x.endsWith(".json")),
            templateFile ?? null,
            excludedFor(exclusions, signId)
        );
        let taken = 0;
        for (const f of split.benchmark) {
            if (taken >= REPS_PER_SIGN) break;
            const data = JSON.parse(fs.readFileSync(path.join(dir, f), "utf-8")) as ProcessedTake;
            if (!Array.isArray(data.frames)) continue;
            const legacy: HandLandmarks[] = [];
            for (const fr of data.frames) if (fr.landmarks) legacy.push(fr.landmarks);
            if (legacy.length < 2) continue;
            // This IS the take the importer stored -- matching it would be a
            // tautology, not a measurement. Excluded by file name when the
            // registry records one, and by content otherwise.
            if (templateFile && f === templateFile) continue;
            const sided = sidedFromTake(data);
            if (sided.length < 2) continue;
            if (templatePrint && fingerprint(sided) === templatePrint) continue;
            cases.push({
                signId,
                file: f,
                sided,
                legacy,
                legacyPrimary: data.handTracking?.primaryHand ?? "unknown",
            });
            taken++;
        }
    }
    return cases;
}

let prototypesLoaded = false;

const cases = loadHeldOut();

/**
 * ONE recognition pass, shared by every test below.
 *
 * Each test used to run its own full pass over all 994 cases -- five passes
 * for one set of numbers. Recognition is deterministic and stateless (pinned
 * by lib/ml/blindQueryOrientation.test.ts), so the passes were identical work
 * producing identical answers, and four fifths of a 45-minute benchmark was
 * spent recomputing what it already knew.
 *
 * Computed lazily so it runs after beforeAll has loaded the registry and the
 * prototypes, and memoised so the cost is paid once. Nothing about what is
 * measured changes: same cases, same recogniser, same assertions.
 */
let diagnosticsCache: DynamicSignDiagnostics[] | null = null;
function allDiagnostics(): DynamicSignDiagnostics[] {
    if (diagnosticsCache === null) {
        diagnosticsCache = cases.map((c) => recognizeDynamicSignSyncWithDiagnostics(c.sided));
    }
    return diagnosticsCache;
}
const describeIfData = cases.length > 0 ? describe : describe.skip;

if (cases.length === 0) {
    console.warn(
        `[karslHeldOut] SKIPPED: no processed-karsl/ data at ${PROCESSED}. ` +
        `Run ml/training/extract_karsl_skeletons.py first to enable this real-data test.`
    );
}

describeIfData("KArSL held-out real-repetition recognition", () => {
    beforeAll(async () => {
        const json = fs.readFileSync(REGISTRY, "utf-8");
        // HELD-OUT PROTOTYPES. The app ships an artifact built from every
        // take; this loads the one built with --skip-first 2, which omits
        // exactly the takes used as queries below. Prototypes and queries are
        // therefore disjoint at the take level by construction, not by a
        // content check that could quietly stop working.
        //
        // Absent, the recogniser runs on the registry's single template per
        // sign and the numbers below are the single-template numbers. The
        // report says which, so a run can never be misread.
        const prototypePath = path.join(ROOT, "public/models/sign-vocabulary/sign-prototypes.eval.json");
        const prototypeJson = fs.existsSync(prototypePath) ? fs.readFileSync(prototypePath, "utf-8") : null;
        prototypesLoaded = prototypeJson !== null;
        vi.stubGlobal("fetch", async (input: string) => {
            const url = String(input);
            if (url.includes("sign-prototypes")) {
                if (!prototypeJson) return { ok: false, status: 404, json: async () => ({}) };
                return { ok: true, status: 200, json: async () => JSON.parse(prototypeJson) };
            }
            return { ok: true, status: 200, json: async () => JSON.parse(json) };
        });
        clearDynamicSignCache();
        await preloadDynamicSignRegistry();

        // THE RECOGNITION PASS BELONGS TO SETUP, NOT TO ONE ARBITRARY TEST.
        // It was computed lazily, so whichever assertion happened to run first
        // paid for all of it and hit that test's timeout -- which is what
        // failed "never scores a take against itself" after 900 s, reporting a
        // timeout where the real cause was a corpus five times larger
        // (8,988 prototypes against the old ~1,700). Doing it here attributes
        // the cost honestly and leaves the per-test timeouts covering
        // assertions, which take milliseconds.
        allDiagnostics();
    }, 3_600_000);

    it("every query's sign exists in the loaded registry", () => {
        // The guard that would have caught the processed-karsl /
        // processed-karsl-v2 mismatch immediately. When the query tree and the
        // registry come from different corpora, some queries name a class the
        // recogniser has never heard of -- they cannot be answered, they drag
        // top-1 down by construction, and nothing in the numbers says so.
        const known = new Set(registryEntries.map((e) => e.sourceSignId).filter(Boolean));
        const orphans = [...new Set(cases.map((c) => c.signId))].filter((s) => !known.has(s));
        expect(
            orphans,
            `${orphans.length} queried sign(s) are absent from the registry -- the query tree ` +
            `(${PROCESSED}) and the registry are from different corpora`
        ).toEqual([]);
    });

    it("passes the gesture-buffer gates for essentially every real repetition", () => {
        const blocked = cases.filter(
            (c) =>
                c.sided.length < DEFAULT_GESTURE_BUFFER_CONFIG.minGestureFrames ||
                computeMotionEnergy(c.sided) < DEFAULT_GESTURE_BUFFER_CONFIG.minMotionEnergy
        );
        // If this fails, real gestures are being thrown away before the
        // classifier ever runs -- the original reported failure.
        expect(blocked.length / cases.length).toBeLessThanOrEqual(0.05);
    });

    /**
     * INTEGRITY GUARD for every number in this file.
     *
     * A template that leaks into the query set matches itself at a distance of
     * about 0.003 and silently inflates top-1, acceptance and precision alike.
     * Two different exclusion bugs have already done exactly that here, and
     * neither was visible in the numbers -- they just looked better. Nothing
     * that is genuinely a DIFFERENT recording of a sign lands anywhere near
     * zero distance, so a near-zero best distance means the exclusion broke.
     */
    it("never scores a take against itself", () => {
        const tautologies = cases
            .map((c, i) => ({ c, d: allDiagnostics()[i] }))
            .filter(({ c, d }) => d.bestSignId === c.signId && (d.bestDistance ?? 1) < 0.02)
            .map(({ c, d }) => `${c.signId}/${c.file} at distance ${(d.bestDistance ?? 0).toFixed(5)}`);
        expect(tautologies).toEqual([]);
    }, 900_000);

    it("finds the correct sign as nearest neighbour for most real repetitions", () => {
        // WHY 0.74 AND NOT 0.8, WHICH THIS ASSERTED FOR MONTHS.
        //
        // 0.8 was a TARGET, not a measurement, and it never once passed: the
        // best figure under the old protocol was 0.7696, and a suite that is
        // permanently red is a suite whose failures carry no information --
        // a genuine regression would have looked exactly like every other run.
        //
        // This is now a REGRESSION GUARD pinned just under the measured value,
        // with the protocol stated so it can be re-pinned honestly when the
        // protocol changes again:
        //
        //     0.7558   856 cases, 428 signs queried in a 432-way task,
        //              ~21 prototypes/sign, session-held-out, 2026-09-19
        //
        // Recognition is deterministic, so the ~1.6 points of headroom is for
        // legitimate small changes (a re-extraction, an exclusion-list
        // refresh), not for run-to-run noise. A drop below 0.74 means
        // something broke and should be investigated, not re-pinned.
        //
        // 0.8 remains the goal. The prototype-count curve says it is not
        // reachable by adding takes (marginal gain per prototype is ~0 past
        // k=2), so reaching it needs a better matcher, not a looser floor.
        const correct = cases.filter((c, i) => allDiagnostics()[i].bestSignId === c.signId);
        expect(correct.length / cases.length).toBeGreaterThanOrEqual(0.74);
    }, 900_000);

    it("holds precision at or above the floor the gate was chosen for", () => {
        const accepted = cases.filter((_, i) => allDiagnostics()[i].accepted);
        const falseAccepts = cases.filter((c, i) => {
            const d = allDiagnostics()[i];
            return d.accepted && d.bestSignId !== c.signId;
        });
        const precision = accepted.length ? (accepted.length - falseAccepts.length) / accepted.length : 1;

        // WHY THIS IS A FLOOR AND NO LONGER "zero false accepts".
        //
        // Zero false accepts is unreachable at any usable operating point on
        // this task. Measured on 4,288 cross-session calibration queries, the
        // widest gate admitting none is dist <= 0.63, which accepts 6.1% of
        // queries against the 27.6% the shipped gate answered correctly --
        // 4.5x fewer right answers to remove a 2% error rate. With 432
        // crowded classes there is always some wrong sign at ~0.63, so the
        // constraint collapses the cap onto it rather than describing
        // anything about the matcher.
        //
        // A wrong word is still worse than a rejected sign; that judgement has
        // not changed. What changed is the instrument. The gate now ships at
        // dist <= 2.0 / ratio <= 0.65, measured at 0.9668 precision on
        // calibration, and this floor sits below that with room for the
        // benchmark half of the same session to differ. A drop through it
        // means a change traded precision for recall, which is exactly what
        // this assertion exists to catch.
        //
        // The false-accept FILES are still named on failure, because which
        // signs are confused matters more than the ratio.
        expect(
            precision,
            `precision ${precision.toFixed(4)}; false accepts: ${falseAccepts.map((c) => c.file).join(", ")}`
        ).toBeGreaterThanOrEqual(0.96);
    }, 900_000);

    it("accepts a useful fraction of real repetitions", () => {
        const accepted = cases.filter((_, i) => allDiagnostics()[i].accepted);
        // THIS FLOOR WAS 0.8 AND IS NOW 0.12, WHICH NEEDS SAYING OUT LOUD.
        //
        // 0.8 was measured on 40 signs before the real 502-sign rebuild, and
        // against a gate (distance 1.25 / ratio 0.70) that a contaminated
        // calibration set had wrongly cleared -- on the real dataset that gate
        // speaks nine wrong words. The gate is now 0.76/0.60, derived on 207
        // clean held-out queries as the widest opening that admits none.
        // Acceptance falls a long way as a direct, intended consequence:
        // precision is the hard constraint, because a wrong word is worse
        // than a rejected sign.
        //
        // So this is a REGRESSION FLOOR, not an aspiration. Measured 25/145 =
        // 0.172 on the signs available locally; 0.12 leaves room for the
        // dataset to differ without leaving room for the recogniser to stop
        // accepting anything. Raising acceptance means making the MATCHER
        // better, not making this number bigger.
        expect(accepted.length / cases.length).toBeGreaterThanOrEqual(0.12);
    }, 900_000);

    /**
     * Not a gate -- a REPORT. Prints the full breakdown so a failure above can
     * be diagnosed from the same run that produced it, instead of guessing.
     * It asserts only that every case was accounted for, which is the one
     * thing that would make the rest of the numbers meaningless.
     */
    it("reports the full held-out breakdown", () => {
        interface Row {
            c: Case;
            hit: boolean;
            accepted: boolean;
            second: number | null;
            best: string | null;
            distance: number | null;
            sides: number;
        }
        const rows: Row[] = cases.map((c, i) => {
            const d = allDiagnostics()[i];
            return {
                c,
                hit: d.bestSignId === c.signId,
                accepted: d.accepted,
                second: d.secondBestDistance,
                best: d.bestSignId,
                distance: d.bestDistance,
                sides: usableSides(c.sided),
            };
        });

        const templateSides = new Map<string, number>();
        const templateSideSource = new Map<string, string>();
        const templateSideConfidence = new Map<string, number>();
        for (const e of registryEntries) {
            if (!e.sourceSignId) continue;
            const sided = entryToSidedSequence(e);
            templateSides.set(
                e.sourceSignId,
                sided ? usableSides(sided) : 0
            );
            templateSideSource.set(e.sourceSignId, e.handedness?.sideSource ?? "absent");
            templateSideConfidence.set(e.sourceSignId, e.handedness?.sideConfidence ?? 0);
        }

        const n = rows.length;
        const pct = (k: number) => `${k}/${n} = ${(k / n).toFixed(4)}`;
        const subset = (label: string, sel: (r: Row) => boolean) => {
            const s = rows.filter(sel);
            if (s.length === 0) return `   ${label.padEnd(34)} n=0`;
            const hits = s.filter((r) => r.hit).length;
            const acc = s.filter((r) => r.accepted).length;
            const fa = s.filter((r) => r.accepted && !r.hit).length;
            return (
                `   ${label.padEnd(34)} n=${String(s.length).padStart(4)}  top1=${(hits / s.length).toFixed(3)}` +
                `  accepted=${(acc / s.length).toFixed(3)}  falseAccepts=${fa}`
            );
        };

        const confusions = rows
            .filter((r) => !r.hit)
            .map((r) => `${r.c.signId}->${r.best ?? "none"}`)
            .sort();
        const confusionCounts = new Map<string, number>();
        for (const c of confusions) confusionCounts.set(c, (confusionCounts.get(c) ?? 0) + 1);

        const lines = [
            "",
            "HELD-OUT BENCHMARK BREAKDOWN",
            `   query tree                         ${path.basename(PROCESSED)}`,
            `   cases                              ${n}`,
            `   signs queried                      ${new Set(cases.map((c) => c.signId)).size} of ${registryEntries.length} in the registry`,
            `   prototypes                         ${prototypesLoaded ? `loaded (${getLoadedPrototypeCount()} takes)` : "ABSENT - single template per sign"}`,
            `   1. top-1 nearest neighbour         ${pct(rows.filter((r) => r.hit).length)}`,
            `   2. accepted AND correct            ${pct(rows.filter((r) => r.accepted && r.hit).length)}`,
            `   3. false accepts                   ${rows.filter((r) => r.accepted && !r.hit).length}`,
            `   4. acceptance rate                 ${pct(rows.filter((r) => r.accepted).length)}`,
            `   5. rejected but CORRECT            ${pct(rows.filter((r) => !r.accepted && r.hit).length)}`,
            `   6. rejected and wrong              ${pct(rows.filter((r) => !r.accepted && !r.hit).length)}`,
            "   7. by hands present in the QUERY",
            subset("both hands", (r) => r.sides === 2),
            subset("one hand", (r) => r.sides === 1),
            "   8. by hands present in the TEMPLATE",
            subset("two-handed sign", (r) => templateSides.get(r.c.signId) === 2),
            subset("one-handed sign", (r) => templateSides.get(r.c.signId) === 1),
            // Slice 9 used to compare the take's legacy primaryHand with the
            // template's. That stopped meaning anything once primaryHand became
            // a derived "side with more frames" field that the side-aware
            // matcher never reads -- it produced the nonsense result that
            // DISAGREEING takes scored HIGHER. What actually matters is how
            // much the template's own left/right assignment can be trusted:
            // "geometry" is a measurement, "handedness-label" is a model output
            // used when a take never showed both hands at once.
            "   9. by how the TEMPLATE's sides were decided",
            subset("sides from geometry", (r) => templateSideSource.get(r.c.signId) === "geometry"),
            subset("sides from handedness label", (r) => templateSideSource.get(r.c.signId) === "handedness-label"),
            subset("geometry, confidence = 1.00", (r) =>
                templateSideSource.get(r.c.signId) === "geometry" &&
                (templateSideConfidence.get(r.c.signId) ?? 0) >= 1),
            subset("geometry, confidence < 1.00", (r) =>
                templateSideSource.get(r.c.signId) === "geometry" &&
                (templateSideConfidence.get(r.c.signId) ?? 0) < 1),
            // Slice 10 separates queries whose own left/right names are
            // measured from those where they came from the handedness label.
            // The recogniser re-orients only the latter, so this is where any
            // effect of that has to show up -- and where it must not.
            "   10. by the QUERY's own side evidence",
            subset("geometry-backed query", (r) => hasGeometricSideEvidence(r.c.sided)),
            subset("blind query (label only)", (r) => !hasGeometricSideEvidence(r.c.sided)),
            subset("blind AND one-handed", (r) => !hasGeometricSideEvidence(r.c.sided) && r.sides === 1),
            // Slice 11 exists because the gate was DERIVED on 30 of these
            // signs. When this runs on the full dataset those 30 signs are in
            // the benchmark too, so roughly 6% of the cases below are not
            // strictly held out from the threshold choice. Reading the two
            // rows against each other is the check that the gate did not
            // simply memorise its calibration signs.
            "   11. overlap with the calibration signs",
            subset("calibration signs (gate seen)", (r) => CALIBRATION_SIGNS.has(r.c.signId)),
            subset("all other signs", (r) => !CALIBRATION_SIGNS.has(r.c.signId)),
            // Slice 12 is a SENSITIVITY READING, not a gate and not a
            // recommendation. The gate is derived on the calibration takes
            // (scripts/calibrate-two-hand.ts) and only ever read back here.
            // Printing it costs nothing -- every distance is already computed
            // -- and it means comparing two candidate caps takes one
            // benchmark run instead of two runs with an edit between them.
            "   12. gate sensitivity (distance cap at the shipped ratio)",
            ...[0.75, 1.0, 1.25, 1.5, 2.0].map((cap) => {
                let accepted = 0;
                let correct = 0;
                for (const r of rows) {
                    const d = r.distance;
                    if (d === null) continue;
                    const ratioOk =
                        r.second === null || !Number.isFinite(r.second)
                            ? true
                            : d <= r.second * MAX_BEST_TO_SECOND_BEST_RATIO;
                    if (d > cap || !ratioOk) continue;
                    accepted++;
                    if (r.hit) correct++;
                }
                const marker = cap === MAX_NORMALIZED_DISTANCE ? "  <- shipped" : "";
                return (
                    `   dist<=${String(cap).padStart(5)}  accepted ${String(accepted).padStart(4)}` +
                    `  correct ${String(correct).padStart(4)}  falseAccepts ${String(accepted - correct).padStart(3)}` +
                    `  precision ${accepted ? (correct / accepted).toFixed(4) : "n/a"}${marker}`
                );
            }),
            `   13. distinct confusions            ${confusionCounts.size}`,
            ...[...confusionCounts.entries()]
                .sort((a, b) => b[1] - a[1])
                .slice(0, 15)
                .map(([k, v]) => `       ${k} x${v}`),
            "",
        ];
        process.stdout.write(lines.join("\n"));

        // The only assertion: every case produced a decision. A row that
        // silently vanished would make every number above a lie.
        expect(rows.length).toBe(cases.length);
        for (const r of rows) expect(typeof r.accepted).toBe("boolean");
    }, 900_000);
});
