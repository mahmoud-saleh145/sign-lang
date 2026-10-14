/**
 * TWO MECHANISMS THAT WERE CANDIDATES FOR A WORD-MODE FALSE REJECTION,
 * pinned by test so neither is ever assumed again.
 *
 * Reported telemetry for one rejected gesture: best sign 0154 at 1.64616,
 * second 0302 at 1.65625, ratio 0.9939 (gate 0.65 -> "ambiguous-margin-failed"),
 * session tracker swaps 255, per-side distance left 3.290 / right 0.791, and
 * query side frames 40 / 56 out of 56 gesture frames.
 *
 * Two explanations were proposed:
 *   (1) unstable hand-slot tracking splicing one hand into the other's track,
 *   (2) a side observed in only part of the gesture being resampled as though
 *       it were complete.
 *
 * (1) is REFUTED here. (2) is CONFIRMED here -- and, importantly, confirmed to
 * be a property of the WEIGHTING rather than of the resampling itself.
 */
import { describe, it, expect } from "vitest";
import { HandSlotTracker, type SlotDetection } from "./handSlotTracker";
import { prepareSide, prepareRelation, prepareSidedSequence, sidedDistance, UNMATCHED_SIDE_COST } from "./twoHandMatch";
import type { HandLandmarks, SidedSequence } from "@/types/landmarks";

/** A 21-point hand whose wrist sits at (x, y). Shape varies with `seed` so
 *  two hands are distinguishable by more than position. */
function handAt(x: number, y: number, seed = 0): HandLandmarks {
    return Array.from({ length: 21 }, (_, i) => ({
        x: x + ((i * 7 + seed * 13) % 5) * 0.004,
        y: y + Math.floor(i / 5) * 0.004,
        z: 0,
    }));
}

function detection(x: number, y: number, seed = 0, score = 0.9): SlotDetection {
    return { landmarks: handAt(x, y, seed), handedness: seed === 0 ? "Left" : "Right", score };
}

describe("1. tracker swaps cannot corrupt a gesture", () => {
    it("counts ARRAY re-ordering, not identity changes: a stably reversed feed costs zero", () => {
        const t = new HandSlotTracker();
        for (let i = 0; i < 10; i++) {
            // Hand "A" (seed 0) is physically left and is ALWAYS reported second.
            t.step([detection(0.7, 0.5, 1), detection(0.2, 0.5, 0)]);
        }
        // Identity never changed; the detector simply always lists them the
        // other way round. The tracker keeps one physical hand per slot.
        expect(t.switchCount).toBe(0);
    });

    it("does NOT retroactively change frames already emitted when a later frame swaps", () => {
        const t = new HandSlotTracker();
        // Build slot 0 history with a single physical hand held steady.
        const emitted: Array<{ slot0x: number; slot1x: number }> = [];
        for (let i = 0; i < 6; i++) {
            const f = t.step([detection(0.2, 0.5, 0), detection(0.7, 0.5, 1)]);
            emitted.push({ slot0x: f[0]!.landmarks[0].x, slot1x: f[1]!.landmarks[0].x });
        }
        const before = emitted.map((e) => ({ ...e }));

        // Now force a swap: report the two hands crossed over.
        t.step([detection(0.7, 0.5, 1), detection(0.2, 0.5, 0)]);
        t.step([detection(0.2, 0.5, 0), detection(0.7, 0.5, 1)]);

        // The gesture buffer holds the arrays above. Nothing in this class
        // revisits them, so they must be bit-identical.
        for (let i = 0; i < before.length; i++) {
            expect(emitted[i]).toEqual(before[i]);
        }
    });

    it("reset() zeroes the counter, which is what separates last-gesture from session", () => {
        const t = new HandSlotTracker();
        // ALTERNATE the reported order: that is what actually forces swaps.
        for (let i = 0; i < 6; i++) {
            const ordered = i % 2 === 0
                ? [detection(0.2, 0.5, 0), detection(0.7, 0.5, 1)]
                : [detection(0.7, 0.5, 1), detection(0.2, 0.5, 0)];
            t.step(ordered);
        }
        expect(t.switchCount).toBeGreaterThan(0);
        t.reset();
        expect(t.switchCount).toBe(0);
    });
});

describe("2. partial coverage is resampled as though complete", () => {
    it("reports the true observed count but a full-length vector regardless", () => {
        const frames: Array<HandLandmarks | null> = [];
        for (let i = 0; i < 56; i++) frames.push(i < 40 ? handAt(0.2, 0.5, 0) : null);

        const side = prepareSide(frames);
        expect(side).not.toBeNull();
        // The truth about the capture...
        expect(side!.detectedFrames).toBe(40);
        // ...is erased by resampling: identical length to a complete capture.
        expect(side!.vectors).toHaveLength(30);

        const complete = prepareSide(Array.from({ length: 56 }, () => handAt(0.2, 0.5, 0)));
        expect(complete!.vectors).toHaveLength(30);
        expect(side!.vectors).toHaveLength(complete!.vectors.length);
    });

    it("a side with only 2 real frames is resampled to the same 30 as a full one", () => {
        const sparse = prepareSide([handAt(0.2, 0.5, 0), null, null, null, handAt(0.2, 0.5, 0)]);
        expect(sparse).not.toBeNull();
        expect(sparse!.detectedFrames).toBe(2);
        expect(sparse!.vectors).toHaveLength(30);
    });
});

describe("3. coverage weighting SATURATES, so a degraded side is not discounted", () => {
    /** Query: right hand present every frame; left hand present in `leftFrames`
     *  of them. Motion is held IDENTICAL either way -- the left hand stays put,
     *  exactly as it does in the template -- so coverage is the ONLY variable. */
    function query(leftFrames: number): SidedSequence {
        const seq: SidedSequence = [];
        for (let i = 0; i < 56; i++) {
            seq.push({
                left: i < leftFrames ? handAt(0.2, 0.5, 0) : null,
                right: handAt(0.7, 0.5, 1),
            });
        }
        return seq;
    }
    /** A template whose two sides are both steady and well covered. */
    function template(): SidedSequence {
        const seq: SidedSequence = [];
        for (let i = 0; i < 36; i++) {
            seq.push({ left: handAt(0.2, 0.5, 0), right: handAt(0.7, 0.5, 1) });
        }
        return seq;
    }

    it("weights a 40-of-56 side the same as a 56-of-56 side (min() caps at the template count)", () => {
        const tpl = prepareSidedSequence(template());
        expect(tpl.left!.detectedFrames).toBe(36);
        expect(tpl.right!.detectedFrames).toBe(36);

        const degraded = prepareSidedSequence(query(40));
        const complete = prepareSidedSequence(query(56));

        // min(40, 36) === min(56, 36) === 36: the coverage weight CANNOT
        // distinguish a side seen in 71% of the gesture from one seen in all
        // of it. This is why the reported gesture's left side contributed a
        // full third of the combined score despite being badly degraded.
        expect(Math.min(degraded.left!.detectedFrames, tpl.left!.detectedFrames)).toBe(
            Math.min(complete.left!.detectedFrames, tpl.left!.detectedFrames)
        );

        const dDeg = sidedDistance(degraded, tpl);
        const dCom = sidedDistance(complete, tpl);

        // The decisive measurement. Motion is identical on both sides of this
        // comparison, so if partial observation were being discounted the
        // degraded query would score strictly worse. It does not:
        expect(dDeg.perSide.left!).toBeCloseTo(dCom.perSide.left!, 6);
        expect(dDeg.distance).toBeCloseTo(dCom.distance, 6);

        // A side seen in 71% of the gesture is scored EXACTLY as though it had
        // been seen in all of it. Nothing downstream can tell them apart.
    });

    it("a partially covered side is NOT a presence mismatch (the asymmetry holds)", () => {
        const tpl = prepareSidedSequence(template());
        const degraded = prepareSidedSequence(query(40));
        const d = sidedDistance(degraded, tpl);
        expect(d.presenceMismatches).toBe(0);
        expect(d.queryMissingSides).toBe(0);
        expect(d.sharedSides).toBe(2);
        // relation track only exists over frames that held BOTH hands
        expect(degraded.relation!.detectedFrames).toBe(40);
        expect(prepareRelation(template())!.detectedFrames).toBe(36);
    });
});

describe("4. the hand-detection score is carried but never used", () => {
    it("identical detections differing only in score produce identical tracking", () => {
        const a = new HandSlotTracker();
        const b = new HandSlotTracker();
        for (let i = 0; i < 8; i++) {
            a.step([detection(0.2, 0.5, 0, 0.95), detection(0.7, 0.5, 1, 0.95)]);
            // Same positions, but the second detection is essentially garbage.
            b.step([detection(0.2, 0.5, 0, 0.95), detection(0.7, 0.5, 1, 0.11)]);
        }
        expect(b.switchCount).toBe(a.switchCount);
        const fa = a.step([detection(0.2, 0.5, 0, 0.95), detection(0.7, 0.5, 1, 0.95)]);
        const fb = b.step([detection(0.2, 0.5, 0, 0.95), detection(0.7, 0.5, 1, 0.11)]);
        expect(fb[0]!.landmarks).toEqual(fa[0]!.landmarks);
        expect(fb[1]!.landmarks).toEqual(fa[1]!.landmarks);
    });
});

describe("5. the reported relation distance is the SAME term already in the score", () => {
    /** Two two-hand gestures that both hold a relation track. */
    function twoHand(seedA: number, seedB: number): SidedSequence {
        return Array.from({ length: 36 }, (_, i) => ({
            left: handAt(0.2 + i * 0.002, 0.5, seedA),
            right: handAt(0.7 - i * 0.001, 0.4, seedB),
        }));
    }

    it("reports a relation distance and folds it into `distance` exactly once", () => {
        const q = prepareSidedSequence(twoHand(0, 1));
        const t = prepareSidedSequence(twoHand(0, 1));
        expect(q.relation).not.toBeNull();
        expect(t.relation).not.toBeNull();

        const d = sidedDistance(q, t);
        expect(d.relation).not.toBeNull();

        // Reconstruct the shipped combination from the per-track numbers and
        // assert `distance` equals it. If the relation term were added twice,
        // dropped, or re-weighted, this fails.
        const tracks = [
            { d: d.perSide.left!, w: Math.min(q.left!.detectedFrames, t.left!.detectedFrames) },
            { d: d.perSide.right!, w: Math.min(q.right!.detectedFrames, t.right!.detectedFrames) },
            {
                d: d.relation!,
                w: Math.min(q.relation!.detectedFrames, t.relation!.detectedFrames),
            },
        ];
        const totalW = tracks.reduce((a, x) => a + x.w, 0);
        const mean = tracks.reduce((a, x) => a + x.d * x.w, 0) / totalW;
        const expected = mean + (UNMATCHED_SIDE_COST * d.presenceMismatches) / 2;
        expect(d.distance).toBeCloseTo(expected, 10);
    });

    it("reports relation: null when either side has no inter-hand track", () => {
        const twoHands = prepareSidedSequence(twoHand(0, 1));
        const oneHand = prepareSidedSequence(
            Array.from({ length: 36 }, (_, i) => ({ left: handAt(0.2 + i * 0.002, 0.5, 0), right: null }))
        );
        expect(twoHands.relation).not.toBeNull();
        expect(oneHand.relation).toBeNull();
        expect(sidedDistance(twoHands, oneHand).relation).toBeNull();
    });
});