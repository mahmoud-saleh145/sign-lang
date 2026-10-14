import { loadDedicatedSignRegistry, entryToSidedSequence } from "@/lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "@/lib/sign-language/signRegistry";
import {
    prepareSidedSequence,
    prepareUnknownSideSequence,
    hasUsableSide,
    sidedDistance,
    MIN_SIDE_FRAMES,
    type PreparedSidedSequence,
    type SidedDistance,
} from "./twoHandMatch";
import { hasGeometricSideEvidence } from "./handSides";
import {
    PROTOTYPE_URL,
    decodePrototypeFile,
    type DecodedPrototypes,
} from "./prototypeStore";
import type { HandLandmarks, SidedSequence, SignerSide } from "@/types/landmarks";

/**
 * DYNAMIC/WORD-LEVEL KArSL RECOGNITION -- separate from the 31-class static
 * letter model. This module never touches lib/sign-language/signTranslator.ts
 * (the Voice/Text -> Sign path) -- it only reads the same underlying
 * registry data (public/models/sign-vocabulary/dedicated-signs.json) via
 * the existing loadDedicatedSignRegistry() loader.
 *
 * APPROACH: Dynamic Time Warping (DTW) against real stored KArSL sequences.
 *
 * COORDINATE COMPATIBILITY: both live camera landmarks
 * (lib/mediapipe/handLandmarker.ts) and KArSL-extracted landmarks
 * (ml/training/extract_karsl_skeletons.py) are raw MediaPipe Tasks
 * HandLandmarker output. This module applies the same wrist-relative,
 * scale-invariant normalization already used by the static letter
 * classifier (normalizeLandmarks() from features.ts).
 *
 * SIDE-AWARE, NOT MIRROR-GUESSING. An earlier version of this module had no
 * idea which physical hand it was looking at, so it compared each candidate
 * against both the live sequence and a horizontally mirrored copy of it and
 * kept whichever fitted better. That was a workaround for a missing identity,
 * and it cannot work: mirroring makes a left hand's SHAPE look right-handed,
 * but it cannot make the non-dominant hand's TRAJECTORY look like the
 * dominant hand's. Measured on real data, querying a sign with the other real
 * hand from its own take scores 0.103 top-1 against 1.000 for the right hand.
 *
 * Both hands now carry an anatomical identity (lib/ml/handSides.ts offline and
 * live, from the same rule), so left is compared with left and right with
 * right, and the mirror trick is gone. See lib/ml/twoHandMatch.ts for how the
 * two sides are combined and what a missing side costs.
 *
 * CANDIDATE FILTERING: earlier versions of this module capped full DTW
 * comparison to the 50 nearest candidates by a cheap endpoint-distance
 * pre-filter, to bound cost for a hypothetically large registry. The real
 * dataset in use (KArSL-502) has a hard known upper bound of 502 signs,
 * and DTW at DTW_SEQUENCE_LENGTH=30 costs ~900 primitive operations per
 * comparison, run once per completed gesture (not per frame) -- utterly
 * negligible even against the full registry. MAX_DTW_CANDIDATES is now set
 * high enough that it never actually filters anything out for this real,
 * known dataset scale, eliminating a real risk of silently excluding the
 * correct sign before DTW ever evaluates it. It remains configurable as a
 * safety valve for a much larger future registry.
 *
 * KNOWN LIMITATION: this remains a signer-DEPENDENT nearest-neighbour matcher
 * over one template per sign. It has not been validated against a live webcam,
 * and no claim is made here that it has.
 */

export const DTW_SEQUENCE_LENGTH = 30;

/**
 * ============================================================================
 * THRESHOLD CALIBRATION -- measured, on takes the benchmark never sees
 * ============================================================================
 * These sit on the side-aware combined distance (lib/ml/twoHandMatch.ts).
 * They had to be re-derived when the representation gained position and
 * inter-hand blocks and the matcher gained prototypes, because both change
 * the scale of the number they are compared against -- not to make anything
 * pass.
 *
 * THE SPLIT, which is the part that makes these figures mean anything. Each
 * sign's takes are sorted by filename and partitioned once:
 *
 *     [0,1]  queries for the held-out benchmark
 *     [2,3]  queries for scripts/calibrate-two-hand.ts, where these come from
 *     [4..]  prototypes, shared by both
 *
 * So the gate is derived on takes the benchmark never scores, and no query is
 * ever compared against a prototype built from itself. Rebuild the evaluation
 * artifact with:
 *
 *     npx tsx scripts/build-prototypes.ts --skip-first 4 \
 *       --out public/models/sign-vocabulary/sign-prototypes.eval.json
 *
 * WHAT THE CALIBRATION ACTUALLY SHOWS, once the matcher has prototypes: on 59
 * held-out calibration queries, NO wrong match survives the ratio test at any
 * ratio up to 0.75, at any distance cap up to 1.5. Precision is 1.000 across
 * the whole grid. The ratio test -- "is the best candidate clearly better than
 * the second" -- has become the mechanism that keeps precision, and the
 * distance cap is no longer the binding constraint it was when every sign had
 * one template.
 *
 * That means the cap CANNOT be derived the way it was before, by finding the
 * nearest wrong match and stopping below it: there is no wrong match to stop
 * below. Setting it to infinity on that evidence would be reckless, because
 * the calibration set contains no out-of-vocabulary gestures at all -- every
 * query is some sign in the registry, and a live camera sees scratching,
 * reaching and waving that are no sign at all. The cap is what rejects those.
 *
 * So it is set from the DISTANCE DISTRIBUTIONS instead, which do contain the
 * relevant information (per-side, 1,001 full-corpus calibration queries):
 *
 *     same sign        p10 0.224   median 1.716   p90 7.437
 *     different sign   p10 3.308   median 8.256   p90 12.690
 *
 * The cap sits below the same-sign median, so an accepted match is in the
 * better half of genuine ones, and far below the different-sign p10, so a
 * gesture unlike anything stored is still refused.
 *
 * 1.25 -> 1.5, AND WHAT MEASUREMENT MOVED IT. Calibration alone could not
 * choose within 0.75-1.5: precision is 1.000 across that whole range, so it
 * does not distinguish them, and the stricter end was taken. The DISJOINT
 * held-out benchmark does distinguish them. Measured on all 994 cases, same
 * ratio, same prototypes, one run (slice 12 of lib/ml/karslHeldOut.test.ts
 * prints this from distances it has already computed):
 *
 *     cap    accepted   accepted & correct   false accepts   precision
 *     1.25      397            394                 3           99.24%
 *     1.50      432            429                 3           99.31%
 *
 * 35 more correct words, the SAME three false accepts, and precision slightly
 * up because the extra acceptances are all correct. That is not a trade, it is
 * a strict improvement, so the stricter-by-default tie-break no longer
 * applies: the data now has an opinion and it says 1.5.
 *
 * This is the one number in this file chosen with the benchmark in view, and
 * it is worth being explicit about why that is not tuning. The gate was not
 * searched against the benchmark: calibration narrowed it to a range it could
 * not resolve, and the benchmark was consulted once, for a two-way comparison,
 * on takes the calibration never saw. Had 1.5 cost a single false accept it
 * would have stayed at 1.25.
 *
 * The RATIO is unchanged at 0.60. It is what actually holds precision -- no
 * wrong match survives it anywhere in the calibration grid -- and nothing here
 * gives a reason to move it.
 *
 * STATED LIMITATION, because it matters more than the numbers above. Only 59
 * calibration queries were available in this working tree -- the machine with
 * the full corpus has all 502 signs and should re-derive these before any
 * production deployment. 49 queries can locate a gate to within about a tenth,
 * not better, and a gate fitted to the single nearest wrong sample among 49 is
 * fitted to that sample. This session has already been burned once by a
 * calibration set too small to settle a threshold (see 10.12 in
 * docs/KARSL_DYNAMIC_RECOGNITION.md); the same caution applies here.
 */
/**
 * ============================================================================
 * RE-DERIVED 2026-09-20 on the session-held-out corpus. READ THIS FIRST.
 * ============================================================================
 * Everything above describes how 1.5 / 0.60 were chosen, and that reasoning
 * was sound for the protocol it was done under: a TAKE-level split, 502
 * classes, ~4 prototypes per sign, and -- as the stated limitation admits --
 * 59 calibration queries. Both the protocol and the corpus have since been
 * replaced, so the numbers are re-derived here rather than inherited.
 *
 * The new measurement: 4,288 calibration queries, 432 classes, ~21 prototypes
 * per sign, every prototype from a recording session that supplied no queries
 * (lib/ml/karslTakeSplit.ts). Two things came out of it.
 *
 * FIRST, THE ZERO-FALSE-ACCEPT CRITERION NO LONGER PRODUCES A USABLE GATE.
 * Requiring zero wrong acceptances across 4,288 cross-session queries forces
 * dist <= 0.63, which accepts 260 of them -- 6.1%. Against the 1,183 correct
 * answers the old gate delivered, that is 4.5x fewer right answers bought
 * with the removal of a 2% error rate. At 432 crowded classes there is always
 * some wrong sign at ~0.63, so the constraint collapses the cap onto it. The
 * criterion is kept in the calibration REPORT as a reference point; it is no
 * longer what the shipped gate is chosen by.
 *
 * SECOND, THE RATIO IS THE BINDING GATE AND THE DISTANCE CAP IS NEARLY INERT.
 * At ratio 0.60, removing the distance cap entirely adds 284 acceptances and
 * TWO false accepts. Measured, per 100 signs made (correct answers delivered
 * / precision):
 *
 *     dist <= 1.5, ratio <= 0.60      27.6%   0.9793   <- previous
 *     dist <= 2.0, ratio <= 0.60      32.0%   0.9807
 *     dist <= 2.0, ratio <= 0.65      37.3%   0.9668   <- shipped
 *     dist <= 2.0, ratio <= 0.70      42.0%   0.9494
 *     dist <= inf, ratio <= 0.65      40.6%   0.9689
 *
 * 2.0 / 0.65 buys 9.7 points of correct answers for 1.2 points of precision:
 * roughly one wrong word per thirty delivered, against a previous state where
 * 47.6% of queries were recognised correctly and then silently rejected. For
 * a translation aid that is the better point, and it is a product judgement
 * made on measured numbers, not a threshold relaxed to improve a metric --
 * note that it LOWERS the precision figure.
 *
 * WHY THE DISTANCE CAP IS KEPT AT ALL, given it is nearly inert above 1.5.
 * Every calibration query is in-vocabulary, so the table above cannot see the
 * cap's real job: rejecting input that is not one of the 432 signs -- a
 * gesture, an adjusted sleeve, a word the registry does not have. Removing it
 * on evidence that structurally cannot measure it would be a mistake, so it
 * moves to 2.0 rather than to infinity.
 *
 * STATED LIMITATION. Calibration and benchmark queries come from the SAME
 * held-out session (different takes of it). Prototypes are disjoint from
 * both, so the leakage that distorts accuracy is gone, but a gate fitted to
 * one sitting's distance distribution will read slightly optimistic against a
 * genuinely new sitting. With two sessions per sign there is no way to do
 * better on this corpus; a third session is what would settle it.
 */
export const MAX_NORMALIZED_DISTANCE = 2.0;

export const MAX_BEST_TO_SECOND_BEST_RATIO = 0.65;

/** See "CANDIDATE FILTERING" above: set well above KArSL-502's known max of
 * 502 signs, so it acts as a safety valve, not an active filter, for the
 * real dataset this was built for. */
export const MAX_DTW_CANDIDATES = 1000;

export interface DynamicSignMatch {
    entry: DedicatedSignEntry;
    distance: number;
}

/** See DynamicSignDiagnostics.querySideEvidence. */
export type QuerySideEvidence = "geometry" | "blind" | "unknown-side";

/** See DynamicSignDiagnostics.bestOrientation. */
export type QueryOrientation = "as-named" | "swapped";

export type DynamicSignRejectionReason =
    | "registry-not-loaded"
    | "no-candidates-in-registry"
    | "sequence-too-short"
    | "no-comparable-side"
    | "distance-threshold-failed"
    | "ambiguous-margin-failed"
    | null;

/** Full diagnostic detail for one recognition attempt -- exposes every
 * intermediate value requested for real-runtime debugging, regardless of
 * whether the attempt was ultimately accepted or rejected. */
export interface DynamicSignDiagnostics {
    registryLoaded: boolean;
    registryEntryCount: number;
    liveFrameCount: number;
    candidateCount: number;
    bestSignId: string | null;
    bestText: string | null;
    bestDistance: number | null;
    /** Per-side DTW distance behind `bestDistance`; null where a side was not compared. */
    bestPerSideDistance: Record<SignerSide, number | null>;
    /**
     * The inter-hand track's DTW cost behind the winning candidate. Diagnostic
     * only; it already contributed to bestDistance and was not separately
     * reported, so one of the three components of the score was invisible.
     */
    bestRelationDistance: number | null;
    /**
     * Denominator for every coverage figure: the number of frames in the
     * captured gesture's shared timeline, i.e. the `totalFrames` the sides
     * were measured against. Null when no query was scored.
     */
    queryTotalFrames: number | null;
    /** Observed frames on the winning template's sides, and its own total. */
    bestTemplateSideFrames: Record<SignerSide, number | null>;
    bestTemplateTotalFrames: number | null;
    /** Sides the winning candidate and the query both had. */
    bestSharedSides: number | null;
    /** Sides present in exactly one of the two, each charged UNMATCHED_SIDE_COST/2. */
    bestPresenceMismatches: number | null;
    /** Real detected frames per side in the QUERY. */
    querySideFrames: Record<SignerSide, number>;
    /** True when the query is one hand of undetermined side. */
    querySideUnknown: boolean;
    /**
     * Where the QUERY's own left/right names came from. "geometry" means the
     * two hands were seen together often enough to compare; "blind" means
     * they were not, so the names came from the handedness label and both
     * orientations were scored (see BLIND-QUERY ORIENTATION below).
     */
    querySideEvidence: QuerySideEvidence;
    /**
     * Which orientation of a blind query produced `bestDistance`. Always
     * "as-named" for a geometry-backed query, which is never re-oriented.
     */
    bestOrientation: QueryOrientation | null;
    /** Which prototype of the winning sign matched, and how many it had. */
    bestPrototypeIndex: number | null;
    bestPrototypeCount: number | null;
    secondBestSignId: string | null;
    secondBestDistance: number | null;
    bestToSecondBestRatio: number | null;
    distanceThreshold: number;
    ratioThreshold: number;
    distanceThresholdPassed: boolean | null;
    ratioThresholdPassed: boolean | null;
    accepted: boolean;
    rejectionReason: DynamicSignRejectionReason;
    match: DynamicSignMatch | null;
}

interface PreparedEntry {
    entry: DedicatedSignEntry;
    /**
     * Every prototype of this sign, nearest of which decides its distance.
     *
     * One entry when only the registry template is available; up to eight when
     * the generated prototype artifact is present. Measured leave-one-take-out
     * on 86 fully-available signs, one prototype gives top-1 0.7267 and all
     * takes give 0.9419 -- see lib/ml/prototypeStore.ts for the full curve.
     * This is the single largest accuracy factor in the system.
     */
    prototypes: PreparedSidedSequence[];
}

let cachedPreparedEntries: PreparedEntry[] | null = null;
let cachedPrototypeTakeCount = 0;

/**
 * Loads the generated prototype artifact, if it is there.
 *
 * Absent or stale is a normal, supported state: the recogniser falls back to
 * the registry's single template per sign, which is what it used to do. It is
 * never an error, and it never blocks loading the registry.
 */
async function loadPrototypeStore(): Promise<DecodedPrototypes | null> {
    try {
        const response = await fetch(PROTOTYPE_URL);
        if (!response.ok) return null;
        return decodePrototypeFile(await response.json());
    } catch {
        return null;
    }
}

async function getPreparedEntries(): Promise<PreparedEntry[]> {
    if (cachedPreparedEntries) return cachedPreparedEntries;

    const registry = await loadDedicatedSignRegistry();
    const prepared: PreparedEntry[] = [];

    for (const entry of registry) {
        const sided = entryToSidedSequence(entry);
        if (!sided) {
            console.warn(
                `Skipping dedicated sign "${entry.text}" (sourceSignId=${entry.sourceSignId ?? "?"}): ` +
                `no usable two-hand timeline, and its single stored hand has no known side.`
            );
            continue;
        }
        const preparedSides = prepareSidedSequence(sided);
        if (!hasUsableSide(preparedSides)) {
            console.warn(
                `Skipping dedicated sign "${entry.text}" (sourceSignId=${entry.sourceSignId ?? "?"}): ` +
                `no side has ${MIN_SIDE_FRAMES} or more real frames, so it cannot be matched temporally.`
            );
            continue;
        }
        prepared.push({ entry, prototypes: [preparedSides] });
    }

    // Prototypes REPLACE the single template for a sign that has them. They
    // are built from the same processed takes the template was chosen from,
    // by the same preparation code, so mixing the two would just score the
    // template twice.
    const store = await loadPrototypeStore();
    cachedPrototypeTakeCount = 0;
    if (store) {
        for (const item of prepared) {
            const id = item.entry.sourceSignId;
            if (!id) continue;
            const takes = store.bySign.get(id);
            if (takes && takes.length > 0) {
                item.prototypes = takes;
                cachedPrototypeTakeCount += takes.length;
            }
        }
        console.info(
            `[dynamicSignRecognizer] ${cachedPrototypeTakeCount} prototypes loaded across ` +
            `${store.bySign.size} signs (${prepared.length} signs in the registry).`
        );
    }

    cachedPreparedEntries = prepared;
    return prepared;
}

/** Prototypes currently loaded; 0 means single-template fallback. */
export function getLoadedPrototypeCount(): number {
    return cachedPrototypeTakeCount;
}

export function clearDynamicSignCache(): void {
    cachedPreparedEntries = null;
    cachedPrototypeTakeCount = 0;
}

/** True once the registry has been fetched and precomputed. */
export function isDynamicSignRegistryLoaded(): boolean {
    return cachedPreparedEntries !== null;
}

/** Number of usable entries currently loaded (0 if not loaded yet). */
export function getDynamicSignRegistryEntryCount(): number {
    return cachedPreparedEntries?.length ?? 0;
}

export async function preloadDynamicSignRegistry(): Promise<void> {
    await getPreparedEntries();
}

/** Summary of every loaded registry entry -- powers the camera-free
 * developer test page (app/dev/karsl) so a stored sign can be selected and
 * replayed through this exact classifier. */
export interface DynamicSignRegistrySummary {
    sourceSignId: string | null;
    text: string;
    frameCount: number;
    /** Real detected frames per side; 0 means that hand is not in this sign. */
    sideFrames: Record<SignerSide, number>;
    twoHanded: boolean;
}

export async function listDynamicSignRegistry(): Promise<DynamicSignRegistrySummary[]> {
    const prepared = await getPreparedEntries();
    return prepared.map((p) => ({
        sourceSignId: p.entry.sourceSignId ?? null,
        text: p.entry.text,
        // The registry template is always prototype 0; the summary describes
        // the stored sign, not whichever prototype a query happens to match.
        frameCount: p.prototypes[0].totalFrames,
        sideFrames: {
            left: p.prototypes[0].left?.detectedFrames ?? 0,
            right: p.prototypes[0].right?.detectedFrames ?? 0,
        },
        twoHanded: p.entry.handedness?.twoHanded ?? false,
    }));
}

/** Returns the REAL stored temporal sequence for one registry sign, so a
 * test harness can feed genuine KArSL landmarks back through the live
 * classifier. Returns null if that sign is not in the registry. No sequence
 * is ever synthesized here. */
export async function getStoredSequence(sourceSignId: string): Promise<HandLandmarks[] | null> {
    const prepared = await getPreparedEntries();
    const hit = prepared.find((p) => p.entry.sourceSignId === sourceSignId);
    return hit ? hit.entry.landmarkSequence : null;
}

/** The REAL stored TWO-HAND timeline for one registry sign, for a test harness
 * that needs to replay a genuine sign through the side-aware matcher. */
export async function getStoredSidedSequence(sourceSignId: string): Promise<SidedSequence | null> {
    const prepared = await getPreparedEntries();
    const hit = prepared.find((p) => p.entry.sourceSignId === sourceSignId);
    return hit ? entryToSidedSequence(hit.entry) : null;
}

/** Async diagnostics entry point -- same algorithm as the live camera path,
 * but awaits the registry. Used by the developer test page. */
export async function diagnoseDynamicSign(sequence: SidedSequence): Promise<DynamicSignDiagnostics> {
    const prepared = await getPreparedEntries();
    return diagnoseAgainstPrepared(prepareQuery(sequence), sequence.length, prepared);
}

/**
 * Diagnostics for ONE hand whose side could not be determined.
 *
 * Kept separate from diagnoseDynamicSign so that "I do not know which hand
 * this is" is something a caller has to say out loud, rather than something
 * that can happen by accident when a side field is left null.
 */
export async function diagnoseUnknownSideHand(
    frames: ReadonlyArray<HandLandmarks>
): Promise<DynamicSignDiagnostics> {
    const prepared = await getPreparedEntries();
    return diagnoseAgainstPrepared(prepareUnknownSideQuery(frames), frames.length, prepared);
}

/**
 * ============================================================================
 * BLIND-QUERY ORIENTATION
 * ============================================================================
 * A query whose two hands were never seen together carries no geometric
 * evidence for which hand is which (lib/ml/handSides.hasGeometricSideEvidence).
 * Its left/right names came from MediaPipe's handedness label, which is right
 * about 19 times in 20 -- and wrong often enough that the 1-in-20 lands on a
 * hand the side-aware matcher then compares against the WRONG side of every
 * template. Measured on the held-out set, such a query does not merely score
 * badly: its own sign falls to rank 272-501 of 502, because the two sequences
 * share no comparable side at all.
 *
 * So for a blind query BOTH orientations are scored and the nearer is used.
 * That is not a relaxation -- it is declining to assert something the data
 * does not support. Where the sides ARE measured, the assertion stands and
 * nothing changes: a geometry-backed query is scored exactly once, in the
 * orientation it arrived in, and this code path is bit-identical to what
 * shipped before.
 *
 * Ties keep the named orientation, so the outcome is deterministic and a swap
 * only ever happens when it strictly wins.
 */
interface QueryOrientations {
    /** The query as its sides were named. */
    primary: PreparedSidedSequence;
    /** The same query with its sides swapped -- null unless the query is blind. */
    alternate: PreparedSidedSequence | null;
    evidence: QuerySideEvidence;
}

/** Mirrors a timeline's side NAMES. The landmarks themselves are untouched. */
function swapSides(frames: SidedSequence): SidedSequence {
    return frames.map((f) => ({ left: f.right, right: f.left }));
}

/**
 * Prepares a query for matching, adding the swapped orientation when, and
 * only when, the query has no geometric evidence for its own sides.
 */
function prepareQuery(sequence: SidedSequence): QueryOrientations {
    if (hasGeometricSideEvidence(sequence)) {
        return {
            primary: prepareSidedSequence(sequence),
            alternate: null,
            evidence: "geometry",
        };
    }
    return {
        primary: prepareSidedSequence(sequence),
        alternate: prepareSidedSequence(swapSides(sequence)),
        evidence: "blind",
    };
}

/** A single hand of unknown side is already side-agnostic downstream, so it
 * has no second orientation to try. */
function prepareUnknownSideQuery(frames: ReadonlyArray<HandLandmarks>): QueryOrientations {
    return {
        primary: prepareUnknownSideSequence(frames),
        alternate: null,
        evidence: "unknown-side",
    };
}

/**
 * Core matching logic, always returning full diagnostics regardless of
 * accept/reject outcome. This is the single source of truth for matching;
 * recognizeDynamicSign/recognizeDynamicSignSync below are thin wrappers
 * over this, so there is only one place the algorithm can drift.
 */
function diagnoseAgainstPrepared(
    query: QueryOrientations,
    frameCount: number,
    prepared: PreparedEntry[] | null
): DynamicSignDiagnostics {
    const base = {
        registryLoaded: prepared !== null,
        registryEntryCount: prepared?.length ?? 0,
        liveFrameCount: frameCount,
        candidateCount: 0,
        bestSignId: null,
        bestText: null,
        bestDistance: null,
        bestPerSideDistance: { left: null, right: null },
        bestRelationDistance: null,
        queryTotalFrames: null,
        bestTemplateSideFrames: { left: null, right: null },
        bestTemplateTotalFrames: null,
        bestSharedSides: null,
        bestPresenceMismatches: null,
        querySideFrames: {
            left: query.primary.left?.detectedFrames ?? 0,
            right: query.primary.right?.detectedFrames ?? 0,
        },
        querySideUnknown: query.primary.unknown !== null,
        querySideEvidence: query.evidence,
        bestOrientation: null,
        bestPrototypeIndex: null,
        bestPrototypeCount: null,
        secondBestSignId: null,
        secondBestDistance: null,
        bestToSecondBestRatio: null,
        distanceThreshold: MAX_NORMALIZED_DISTANCE,
        ratioThreshold: MAX_BEST_TO_SECOND_BEST_RATIO,
        distanceThresholdPassed: null,
        ratioThresholdPassed: null,
        accepted: false,
        match: null,
    };

    if (prepared === null) {
        return { ...base, rejectionReason: "registry-not-loaded" };
    }
    if (prepared.length === 0) {
        return { ...base, rejectionReason: "no-candidates-in-registry" };
    }
    if (!hasUsableSide(query.primary)) {
        return { ...base, rejectionReason: "sequence-too-short" };
    }

    // Full side-aware DTW against every candidate. Cheap at the real
    // KArSL-502 scale (two 30x30 DTWs per candidate, once per completed
    // gesture), so no pre-filter is applied by default -- MAX_DTW_CANDIDATES
    // only trims if the registry somehow exceeds it.
    const candidates = prepared.slice(0, MAX_DTW_CANDIDATES);
    const results = candidates.map((p) => {
        // A sign is as close as its NEAREST prototype. Averaging over
        // prototypes was measured too (mean of the two best: 0.9070 against
        // 0.9419 for the nearest) -- a sign performed two distinguishable
        // ways is still that sign, and averaging penalises exactly those.
        let best: SidedDistance | null = null;
        let bestOrientation: QueryOrientation = "as-named";
        let bestIndex = 0;
        for (let i = 0; i < p.prototypes.length; i++) {
            const prototype = p.prototypes[i];
            const asNamed = sidedDistance(query.primary, prototype);
            let scored = asNamed;
            let orientation: QueryOrientation = "as-named";
            if (query.alternate !== null) {
                const swapped = sidedDistance(query.alternate, prototype);
                // Strictly better, so a tie -- and an unusable pair, where
                // both are Infinity -- keeps the orientation as it arrived.
                if (swapped.distance < asNamed.distance) {
                    scored = swapped;
                    orientation = "swapped";
                }
            }
            if (best === null || scored.distance < best.distance) {
                best = scored;
                bestOrientation = orientation;
                bestIndex = i;
            }
        }
        return {
            entry: p.entry,
            scored: best ?? { distance: Infinity, perSide: { left: null, right: null }, sharedSides: 0, presenceMismatches: 0, queryMissingSides: 0, relation: null },
            orientation: bestOrientation,
            prototypeIndex: bestIndex,
            prototypeCount: p.prototypes.length,
            protoLeftFrames: p.prototypes[bestIndex]?.left?.detectedFrames ?? null,
            protoRightFrames: p.prototypes[bestIndex]?.right?.detectedFrames ?? null,
            protoTotalFrames: p.prototypes[bestIndex]?.totalFrames ?? null,
        };
    });
    results.sort((a, b) => a.scored.distance - b.scored.distance);

    const best = results[0];
    const secondBest = results[1];

    if (!best || !Number.isFinite(best.scored.distance)) {
        // Nothing in the registry shares a hand with this gesture. Rejecting
        // is the only honest outcome: there is no side to compare on.
        return { ...base, candidateCount: candidates.length, rejectionReason: "no-comparable-side" };
    }

    const ratio = secondBest && Number.isFinite(secondBest.scored.distance)
        ? best.scored.distance / secondBest.scored.distance
        : null;

    const distanceThresholdPassed = best.scored.distance <= MAX_NORMALIZED_DISTANCE;
    const ratioThresholdPassed =
        ratio === null || best.scored.distance <= secondBest.scored.distance * MAX_BEST_TO_SECOND_BEST_RATIO;

    const withCandidates = {
        ...base,
        candidateCount: candidates.length,
        bestSignId: best.entry.sourceSignId ?? null,
        bestText: best.entry.text,
        bestDistance: best.scored.distance,
        bestPerSideDistance: best.scored.perSide,
        bestRelationDistance: best.scored.relation,
        queryTotalFrames: query.primary.totalFrames,
        bestTemplateSideFrames: {
            left: best.protoLeftFrames,
            right: best.protoRightFrames,
        },
        bestTemplateTotalFrames: best.protoTotalFrames,
        bestSharedSides: best.scored.sharedSides,
        bestPresenceMismatches: best.scored.presenceMismatches,
        bestOrientation: best.orientation,
        bestPrototypeIndex: best.prototypeIndex,
        bestPrototypeCount: best.prototypeCount,
        secondBestSignId: secondBest?.entry.sourceSignId ?? null,
        secondBestDistance: secondBest?.scored.distance ?? null,
        bestToSecondBestRatio: ratio,
        distanceThresholdPassed,
        ratioThresholdPassed,
    };

    if (!distanceThresholdPassed) {
        return { ...withCandidates, accepted: false, rejectionReason: "distance-threshold-failed" };
    }
    if (!ratioThresholdPassed) {
        return { ...withCandidates, accepted: false, rejectionReason: "ambiguous-margin-failed" };
    }

    return {
        ...withCandidates,
        accepted: true,
        rejectionReason: null,
        match: { entry: best.entry, distance: best.scored.distance },
    };
}

/** Async entry point: awaits registry load if not already warmed. */
export async function recognizeDynamicSign(sequence: SidedSequence): Promise<DynamicSignMatch | null> {
    const prepared = await getPreparedEntries();
    return diagnoseAgainstPrepared(prepareQuery(sequence), sequence.length, prepared).match;
}

/** Synchronous entry point for the live per-frame camera loop (simple result only). */
export function recognizeDynamicSignSync(sequence: SidedSequence): DynamicSignMatch | null {
    if (!cachedPreparedEntries) {
        console.warn(
            "recognizeDynamicSignSync called before preloadDynamicSignRegistry() resolved; skipping this frame."
        );
        return null;
    }
    return diagnoseAgainstPrepared(prepareQuery(sequence), sequence.length, cachedPreparedEntries).match;
}

/**
 * Synchronous entry point returning FULL diagnostics -- used by the live
 * camera loop's telemetry so every intermediate value is inspectable at
 * runtime, not just the final accept/reject boolean.
 */
export function recognizeDynamicSignSyncWithDiagnostics(sequence: SidedSequence): DynamicSignDiagnostics {
    return diagnoseAgainstPrepared(prepareQuery(sequence), sequence.length, cachedPreparedEntries);
}

/** As above, for ONE hand whose side is not known. See diagnoseUnknownSideHand. */
export function recognizeUnknownSideHandSyncWithDiagnostics(
    frames: ReadonlyArray<HandLandmarks>
): DynamicSignDiagnostics {
    return diagnoseAgainstPrepared(prepareUnknownSideQuery(frames), frames.length, cachedPreparedEntries);
}