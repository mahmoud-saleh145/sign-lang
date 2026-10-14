/**
 * SIDE-AWARE COMPARISON of two two-hand gestures.
 *
 * Each hand is resampled and matched against the SAME hand of the candidate:
 * left against left, right against right, never mirrored into each other.
 * That is the whole point -- measured on real held-out KArSL takes, querying
 * a sign with the other real hand from its own take drops nearest-neighbour
 * accuracy from 1.000 to 0.103, so the two hands of a sign carry almost
 * entirely different information and must not be substituted for one another.
 *
 * WHY NOT ONE CONCATENATED VECTOR
 * -------------------------------
 * Stacking left and right into a single 84-dimensional frame and running one
 * DTW over it is the obvious shortcut and it is wrong here for two reasons.
 * It forces both hands onto ONE warping path, so a sign where the hands move
 * at different rates (very common: a static base hand while the dominant hand
 * traces a shape) cannot be aligned; and it has no representation for a
 * missing hand, so every null frame would need a fabricated value. Per-side
 * DTW has neither problem: each hand gets its own warp, and a hand that was
 * never seen simply has no track.
 *
 * COMBINING THE TWO SIDES
 * -----------------------
 * A candidate is scored on the sides that BOTH sequences actually have, and is
 * charged UNMATCHED_SIDE_COST for each side the QUERY has that the TEMPLATE
 * lacks -- a stored sign that does not use a hand the signer just used is
 * evidence against that sign. The reverse is NOT charged: a hand missing from
 * the capture says nothing about the sign. That asymmetry is load-bearing; see
 * the constant's own comment for the measurement that forced it.
 *
 * See scripts/calibrate-two-hand.ts for the measurements that set the
 * constants below, and docs/KARSL_DYNAMIC_RECOGNITION.md for the results.
 */
import { dtwDistance, resampleSequence } from "./dtw";
import { normalizeLandmarks } from "./features";
import type { HandLandmarks, SidedSequence, SignerSide } from "@/types/landmarks";
import { SIGNER_SIDES } from "@/types/landmarks";

/** Frames each side is resampled to before DTW. */
export const DTW_SEQUENCE_LENGTH = 30;

/**
 * Identifies the per-frame representation produced by this module.
 *
 * lib/ml/prototypeStore.ts ships PREPARED vectors, so a prototype artifact is
 * only meaningful for the pipeline that built it. Bump this whenever the
 * feature layout, weights or sequence length change; a stale artifact is then
 * rejected on load instead of being scored as if it matched.
 */
export const FEATURE_VERSION = "shape42+pos2+rel1@30";

/** Fewest real detected frames on a side before that side is usable at all. */
export const MIN_SIDE_FRAMES = 2;

/**
 * Cost charged for a side the QUERY has and the TEMPLATE lacks.
 *
 * THE PENALTY IS ASYMMETRIC, and the symmetric version was measurably wrong.
 * It used to be charged whenever either sequence had a side the other lacked,
 * which meant a capture that missed a hand penalised the CORRECT two-handed
 * sign while leaving every wrong one-handed sign untouched. Traced on real
 * held-out data, sign 0033 ("ب", template L14/R14 of 14 frames) captured with
 * only one hand (L14/R0):
 *
 *     vs 0356 (wrong, one-handed L37/R0): shared 1, penalty 0.00 -> 0.6466  ACCEPTED
 *     vs 0033 (right, two-handed):        shared 1, penalty 0.79 -> not even 2nd
 *
 * A correct-pair per-side distance has a median of 1.483, so a 0.79 charge is
 * comparable to the whole useful range: presence alone decided the outcome.
 * One-hand queries were 10.7% of the benchmark and carried 3 of its 6 false
 * accepts, at top-1 0.542 against 0.696 for two-hand queries.
 *
 * The direction that survives scrutiny: a hand missing from the QUERY is a
 * capture failure and says nothing about the sign, so it costs nothing. A hand
 * missing from the TEMPLATE means the stored sign genuinely does not use a hand
 * the signer just used, which is evidence against it, so it is charged.
 *
 * DERIVED, and the first derivation was wrong -- recording both because the
 * second only makes sense against the first. Per-side DTW distances on the
 * calibration set (30 signs, disjoint from the benchmark; 95 same-sign pairs,
 * 47,421 different-sign pairs), re-measured after the asymmetry fix:
 *
 *     same sign       p10 0.068   median 1.416   p90  6.154
 *     different sign  p10 3.051   median 7.208   p90 12.401
 *
 * The tidy argument is that an ABSENT side should cost what an UNINFORMATIVE
 * side costs, i.e. the different-sign median, ~7.2. Measured under the old
 * symmetric rule that was much too expensive: top-1 fell from 0.646 to 0.521.
 *
 * The quantity that works is the MARGIN separating a correct side from the
 * nearest wrong one: different-sign p10 minus same-sign median, 3.051 - 1.416
 * = 1.635, which the shipped 1.58 sits just under.
 *
 * Re-swept after the asymmetry fix, coverage weighting, top-1 on the
 * calibration set: 0 -> 0.577, and 1.0, 1.3, 1.58, 1.9, 2.2, 2.6, 3.0 ALL ->
 * 0.615. The plateau is now completely flat from 1.0 to 3.0, where before it
 * ran 1.0-2.2 and fell away on either side. That is the expected consequence
 * of the asymmetry: the penalty now fires only when the query has two hands
 * and the template has one, which is both rarer and less ambiguous, so the
 * exact price matters far less than the fact that it is not zero. 1.58 is kept
 * because it is the derived margin and sits mid-plateau; no value in the
 * plateau would measure differently.
 *
 * LIMITATION, stated: this calibration runs through
 * scripts/calibrate-two-hand.ts, which still identifies the template take by
 * the LEGACY fingerprint -- the same defect that was fixed in
 * lib/ml/karslHeldOut.test.ts. Some calibration queries are therefore still
 * templates scoring against themselves. It does not affect the choice made
 * here (the sweep is flat), but the absolute calibration accuracy figures are
 * optimistic and should not be compared with benchmark figures.
 */
export const UNMATCHED_SIDE_COST = 1.58;

let unmatchedSideCost: number = UNMATCHED_SIDE_COST;

/**
 * Overrides the unmatched-side cost. Exists ONLY for
 * scripts/calibrate-two-hand.ts, which has to sweep candidate values to pick
 * the shipped one. Production code never calls this.
 */
export function setUnmatchedSideCost(cost: number): void {
    unmatchedSideCost = cost;
}

export function getUnmatchedSideCost(): number {
    return unmatchedSideCost;
}

/**
 * How the per-side distances are averaged.
 *
 * "equal"    - plain mean over shared sides.
 * "coverage" - weighted by how many real frames each side contributed, so a
 *              side seen in three frames does not count as much as one seen
 *              throughout.
 *
 * Measured on the calibration set, "coverage" was equal or better at every
 * candidate cost (e.g. 0.646 vs 0.625 at the chosen cost), so it is the
 * default. The margin is small; it is not claimed to be more than that.
 */
export type SideWeighting = "equal" | "coverage";
export const SIDE_WEIGHTING: SideWeighting = "coverage";

/** One side of a gesture, prepared for DTW. */
export interface PreparedSide {
    vectors: number[][];
    /** Real frames this side was actually detected in (before resampling). */
    detectedFrames: number;
}

/** Both sides of a gesture, prepared for DTW. A side absent from the data is null. */
export interface PreparedSidedSequence {
    left: PreparedSide | null;
    right: PreparedSide | null;
    /**
     * A single hand that could NOT be named (lib/ml/handSides.ts returned
     * "unknown"). Set only when `left` and `right` are both null. Kept as its
     * own field rather than being copied into both sides, because copying it
     * would assert something about the hand that the data does not support.
     */
    unknown: PreparedSide | null;
    /**
     * Inter-hand geometry over the frames that held both hands. Null whenever
     * the gesture never showed two hands at once, which is honest: there is no
     * relationship to describe.
     */
    relation: PreparedSide | null;
    /** Frames on the shared timeline, whether or not a hand was seen. */
    totalFrames: number;
}

export interface SidedDistance {
    /** Combined distance, comparable across candidates. Infinity = incomparable. */
    distance: number;
    perSide: Record<SignerSide, number | null>;
    /** Sides present in BOTH sequences and therefore actually compared. */
    sharedSides: number;
    /**
     * Sides the QUERY has that the TEMPLATE lacks. These are the penalised
     * ones: the template is demonstrably missing a hand the signer used.
     */
    presenceMismatches: number;
    /**
     * Sides the TEMPLATE has that the QUERY lacks. NOT penalised -- see the
     * asymmetry note on UNMATCHED_SIDE_COST. Reported so a caller can tell a
     * partial capture from a genuine one-handed sign.
     */
    queryMissingSides: number;
    /** The inter-hand track's own DTW cost, or null when there is none to compare.
     *
     * DIAGNOSTIC ONLY. This value already participates in `distance` -- it is
     * one of the terms in the weighted mean -- but was then discarded, so the
     * panel could show left and right while hiding the third component of the
     * score. Reporting it changes no arithmetic. */
    relation: number | null;
}

/**
 * ============================================================================
 * WHAT A FRAME IS MADE OF
 * ============================================================================
 * This used to be `normalizeLandmarks()` alone: every frame translated to its
 * own wrist and divided by its own hand size. That is the right way to
 * describe a hand SHAPE and the wrong way to describe a SIGN, because it is
 * applied per frame -- so a hand tracing a circle across the body and a hand
 * held still produce the identical vector sequence whenever the fingers
 * match. Position, trajectory and the relationship between the two hands were
 * all discarded before DTW ever ran. For a language whose words ARE movement,
 * that removes the primary signal.
 *
 * Measured on 223 held-out KArSL takes against all 502 templates, top-1:
 *
 *   SHAPE only (what shipped)          0.6188
 *   SHAPE + REL                        0.6547
 *   SHAPE + 2*POS                      0.6323
 *   SHAPE + 2*POS + REL                0.6592   <- shipped
 *
 * and the two additions help different cases, which is why both are kept:
 * REL can only apply when both hands are present (two-hand 0.6235 -> 0.6728,
 * geometry-backed 0.6494 -> 0.7013), POS is what one-hand and blind queries
 * gain from (one-hand 0.6066 -> 0.6885, blind 0.5507 -> 0.6232).
 *
 * Things that were measured and REJECTED, so they are not tried again:
 *   POS in absolute image coordinates   0.5516  (ties the model to one camera)
 *   POS against the take's centroid     0.6054  (degenerate for one hand: the
 *                                        centroid IS the hand, and a one-hand
 *                                        query then cannot match a two-hand
 *                                        template of its own sign)
 *   SHAPE + derivative of SHAPE         0.6099
 *   Sakoe-Chiba band 4 / 6 / 8          0.6099 / 0.6188 / 0.6188 (no gain;
 *                                        the warps were not the problem)
 */

/** Weight on the positional block, relative to shape. Swept: 1 -> 0.6457,
 * 2 -> 0.6592, and 2 is also the best on the one-hand and blind slices. */
const POSITION_WEIGHT = 2;

/** Weight on the inter-hand block. Swept: 0.5 -> 0.6502, 1 -> 0.6592,
 * 2 -> 0.6502. Flat around 1, which is taken as "present and not delicate". */
const RELATION_WEIGHT = 1;

const WRIST_INDEX = 0;
const MIDDLE_MCP_INDEX = 9;

/** Distance wrist -> middle-finger MCP: one hand's apparent size in frame. */
function handSize(hand: HandLandmarks): number {
    return (
        Math.hypot(
            hand[MIDDLE_MCP_INDEX].x - hand[WRIST_INDEX].x,
            hand[MIDDLE_MCP_INDEX].y - hand[WRIST_INDEX].y
        ) || 1e-6
    );
}

/**
 * The frame a hand's POSITION is expressed in: centred on that hand's own mean
 * wrist position over the gesture and scaled by its own mean apparent size.
 *
 * Per SIDE, deliberately, and not per take. A per-take centroid is a different
 * quantity depending on how many hands were seen -- for a one-hand take the
 * centroid is the hand itself, so the feature collapses to zero and a
 * one-hand capture can no longer match the two-hand template of its own sign.
 * Measured, that cost 11 points of one-hand top-1 (0.6066 -> 0.4918). Per-side
 * centring treats one-hand and two-hand takes identically and is invariant to
 * where the signer sits and how far from the camera they are, while keeping
 * the movement WITHIN the gesture, which is the part that identifies it.
 */
interface SideReference {
    centreX: number;
    centreY: number;
    scale: number;
}

function referenceFrom(hands: ReadonlyArray<HandLandmarks>): SideReference {
    let sx = 0;
    let sy = 0;
    let ss = 0;
    for (const hand of hands) {
        sx += hand[WRIST_INDEX].x;
        sy += hand[WRIST_INDEX].y;
        ss += handSize(hand);
    }
    const n = hands.length || 1;
    return { centreX: sx / n, centreY: sy / n, scale: ss / n || 1e-6 };
}

function landmarksToVector(landmarks: HandLandmarks, reference: SideReference): number[] {
    const vector: number[] = [];
    for (const point of normalizeLandmarks(landmarks)) {
        vector.push(point.x, point.y);
    }
    vector.push(
        ((landmarks[WRIST_INDEX].x - reference.centreX) / reference.scale) * POSITION_WEIGHT,
        ((landmarks[WRIST_INDEX].y - reference.centreY) / reference.scale) * POSITION_WEIGHT
    );
    return vector;
}

/**
 * Builds the DTW-ready form of one side.
 *
 * MISSING FRAMES INSIDE A SIDE: frames where this hand was not detected are
 * dropped, and the frames that remain keep their order. DTW is invariant to
 * time warping, so removing a gap changes the alignment far less than filling
 * it would -- and filling it would mean inventing coordinates, which is not
 * allowed. `detectedFrames` records how much of the timeline was real, so a
 * sparse side can be recognised as such by the caller.
 */
export function prepareSide(frames: ReadonlyArray<HandLandmarks | null>): PreparedSide | null {
    const present: HandLandmarks[] = [];
    for (const frame of frames) {
        if (frame && frame.length > 0) present.push(frame);
    }
    if (present.length < MIN_SIDE_FRAMES) return null;
    const reference = referenceFrom(present);
    try {
        return {
            vectors: resampleSequence(
                present.map((hand) => landmarksToVector(hand, reference)),
                DTW_SEQUENCE_LENGTH
            ),
            detectedFrames: present.length,
        };
    } catch {
        return null;
    }
}

/**
 * The INTER-HAND track: the vector from the right wrist to the left, over the
 * frames where both hands were seen, scaled by their mean apparent size.
 *
 * It has to be its own track. Per-side DTW gives each hand an independent
 * warping path -- which is the whole reason it beats one concatenated vector
 * -- and a relational quantity has no single side to live on. Built only from
 * frames that really contain both hands, so nothing is invented for a hand
 * that was not there.
 *
 * Two signs can use identical handshapes moving identically and differ only in
 * whether the hands meet, cross or stay apart. Nothing else in the
 * representation can tell those apart.
 */
export function prepareRelation(sequence: SidedSequence): PreparedSide | null {
    const pairs: Array<[HandLandmarks, HandLandmarks]> = [];
    for (const frame of sequence) {
        if (frame.left && frame.right) pairs.push([frame.left, frame.right]);
    }
    if (pairs.length < MIN_SIDE_FRAMES) return null;
    let scaleSum = 0;
    for (const [l, r] of pairs) scaleSum += (handSize(l) + handSize(r)) / 2;
    const scale = scaleSum / pairs.length || 1e-6;
    try {
        return {
            vectors: resampleSequence(
                pairs.map(([l, r]) => [
                    ((l[WRIST_INDEX].x - r[WRIST_INDEX].x) / scale) * RELATION_WEIGHT,
                    ((l[WRIST_INDEX].y - r[WRIST_INDEX].y) / scale) * RELATION_WEIGHT,
                ]),
                DTW_SEQUENCE_LENGTH
            ),
            detectedFrames: pairs.length,
        };
    } catch {
        return null;
    }
}

export function prepareSidedSequence(sequence: SidedSequence): PreparedSidedSequence {
    return {
        left: prepareSide(sequence.map((f) => f.left)),
        right: prepareSide(sequence.map((f) => f.right)),
        unknown: null,
        relation: prepareRelation(sequence),
        totalFrames: sequence.length,
    };
}

/**
 * Prepares ONE hand whose side could not be determined.
 *
 * This is what an unnameable hand looks like: a real track with no side, not a
 * hand assigned to a side by a coin flip and not the same track written into
 * both sides. `sidedDistance` compares it against whichever side of a template
 * fits best and charges for every template side it therefore leaves unmatched,
 * so an unnamed hand is always a weaker match than a named one.
 */
export function prepareUnknownSideSequence(
    frames: ReadonlyArray<HandLandmarks | null>
): PreparedSidedSequence {
    return {
        left: null,
        right: null,
        unknown: prepareSide(frames),
        relation: null,
        totalFrames: frames.length,
    };
}

/** True when this gesture has at least one usable hand track. */
export function hasUsableSide(prepared: PreparedSidedSequence): boolean {
    return prepared.left !== null || prepared.right !== null || prepared.unknown !== null;
}

/**
 * Distance between a query gesture and a candidate template.
 *
 * Cases, all handled explicitly:
 *   A) both sides in both -> both compared, averaged, no penalty
 *   B) one side in QUERY, two in template -> the shared side is compared and
 *      NOTHING is charged. The capture missed a hand; that is not evidence
 *      against the sign, and charging it used to hand the match to whichever
 *      wrong one-handed sign happened to fit.
 *   C) two in QUERY, one in template -> the shared side is compared and one
 *      UNMATCHED_SIDE_COST is charged: the stored sign does not use a hand the
 *      signer demonstrably just used.
 *   D) one side in both, same side -> compared, no penalty
 *      one side in both, DIFFERENT sides -> nothing shared, Infinity
 *   E) gaps inside a side -> handled in prepareSide()
 */
export function sidedDistance(
    query: PreparedSidedSequence,
    template: PreparedSidedSequence,
    weighting: SideWeighting = SIDE_WEIGHTING
): SidedDistance {
    const perSide: Record<SignerSide, number | null> = { left: null, right: null };
    const weights: number[] = [];
    const distances: number[] = [];
    let presenceMismatches = 0;
    let queryMissingSides = 0;

    // An unnamed query hand: try it against every side the template has, keep
    // the best fit, and pay for each template side it could not account for.
    const unknown = query.unknown ?? template.unknown;
    if (unknown) {
        const other = query.unknown ? template : query;
        let best: number | null = null;
        let sidesPresent = 0;
        for (const side of SIGNER_SIDES) {
            const s = other[side];
            if (!s) continue;
            sidesPresent++;
            const d = dtwDistance(unknown.vectors, s.vectors).normalizedCost;
            perSide[side] = d;
            if (best === null || d < best) best = d;
        }
        if (best === null) {
            const bothUnknown = query.unknown && template.unknown;
            if (!bothUnknown) {
                return { distance: Infinity, perSide, sharedSides: 0, presenceMismatches: 0, queryMissingSides: 0, relation: null };
            }
            const d = dtwDistance(query.unknown!.vectors, template.unknown!.vectors).normalizedCost;
            return { distance: d, perSide, sharedSides: 1, presenceMismatches: 0, queryMissingSides: 0, relation: null };
        }
        // An UNNAMED QUERY hand is a query-side deficiency: it is charged
        // nothing, exactly as a missing query side is. An unnamed TEMPLATE
        // against a two-sided query is the template lacking a named side, so
        // that direction is charged.
        const templateIsTheUnknownOne = template.unknown !== null;
        const unmatched = templateIsTheUnknownOne ? Math.max(0, sidesPresent - 1) : 0;
        return {
            distance: best + (unmatchedSideCost * unmatched) / 2,
            perSide,
            sharedSides: 1,
            presenceMismatches: unmatched,
            queryMissingSides: templateIsTheUnknownOne ? 0 : Math.max(0, sidesPresent - 1),
            relation: null,
        };
    }

    for (const side of SIGNER_SIDES) {
        const q = query[side];
        const t = template[side];
        if (q && t) {
            const d = dtwDistance(q.vectors, t.vectors).normalizedCost;
            perSide[side] = d;
            distances.push(d);
            weights.push(
                weighting === "coverage" ? Math.min(q.detectedFrames, t.detectedFrames) : 1
            );
        } else if (q && !t) {
            // The template does not use a hand the signer just used.
            presenceMismatches++;
        } else if (t && !q) {
            // The capture missed a hand the template uses. Not the sign's
            // fault, and not charged -- see UNMATCHED_SIDE_COST.
            queryMissingSides++;
        }
    }

    if (distances.length === 0) {
        return { distance: Infinity, perSide, sharedSides: 0, presenceMismatches, queryMissingSides, relation: null };
    }

    // The inter-hand track joins the average on the same footing as a side,
    // and only when BOTH gestures have one. A query that showed both hands
    // against a template that did not is already handled by the presence
    // rules above; adding a relation penalty on top would charge twice for
    // the same fact.
    const sharedSideCount = distances.length;
    let relationDistance: number | null = null;
    if (query.relation && template.relation) {
        relationDistance = dtwDistance(query.relation.vectors, template.relation.vectors).normalizedCost;
        distances.push(relationDistance);
        weights.push(
            weighting === "coverage"
                ? Math.min(query.relation.detectedFrames, template.relation.detectedFrames)
                : 1
        );
    }

    const totalWeight = weights.reduce((a, b) => a + b, 0);
    const weightedMean =
        totalWeight > 0
            ? distances.reduce((sum, d, i) => sum + d * weights[i], 0) / totalWeight
            : distances.reduce((a, b) => a + b, 0) / distances.length;

    // Divided by two so that the penalty for "one side missing on one hand" is
    // half the price of a fully uninformative side, keeping the combined score
    // on the same scale as a per-side distance.
    const penalty = (unmatchedSideCost * presenceMismatches) / 2;

    return {
        distance: weightedMean + penalty,
        perSide,
        sharedSides: sharedSideCount,
        presenceMismatches,
        queryMissingSides,
        relation: relationDistance,
    };
}

/** Builds a sided sequence from one named hand, leaving the other side absent. */
export function singleSideSequence(
    side: SignerSide,
    frames: ReadonlyArray<HandLandmarks>
): SidedSequence {
    return frames.map((f) => (side === "left" ? { left: f, right: null } : { left: null, right: f }));
}
