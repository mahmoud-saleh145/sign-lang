/**
 * REGRESSION TESTS for side-aware (two-hand) KArSL recognition.
 *
 * Each test here pins down one property that was either absent before, or was
 * broken in a way that only showed up as an accuracy number. They use REAL
 * registry entries wherever the property is about real signs; the few
 * synthetic fixtures are clearly marked and exist only to construct a
 * situation the real data does not contain (a permuted detection order, a
 * hand that is never nameable).
 */
import { describe, it, expect, beforeAll, vi } from "vitest";
import { readFileSync, existsSync, readdirSync } from "fs";
import path from "path";
import {
    clearDynamicSignCache,
    preloadDynamicSignRegistry,
    recognizeDynamicSignSyncWithDiagnostics,
    recognizeUnknownSideHandSyncWithDiagnostics,
} from "./dynamicSignRecognizer";
import { HandSlotTracker, type SlotDetection } from "./handSlotTracker";
import { assignSides } from "./handSides";
import { DEFAULT_PROCESSED_ROOT } from "./karslPaths";
import { prepareSidedSequence, sidedDistance, UNMATCHED_SIDE_COST } from "./twoHandMatch";
import { DynamicGestureBuffer, DEFAULT_GESTURE_BUFFER_CONFIG } from "./dynamicGestureBuffer";
import { withLegacySequence, entryToSidedSequence } from "@/lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "@/lib/sign-language/signRegistry";
import type { HandLandmarks, SidedSequence } from "@/types/landmarks";

const ROOT = path.resolve(__dirname, "../..");
const REGISTRY_PATH = path.join(ROOT, "public/models/sign-vocabulary/dedicated-signs.json");
const registryJson = readFileSync(REGISTRY_PATH, "utf-8");
const entries = (JSON.parse(registryJson) as DedicatedSignEntry[]).map(withLegacySequence);

const twoHanded = entries.filter(
    (e) => e.handedness?.twoHanded && (e.handFrames?.length ?? 0) > 10
);
const oneHanded = entries.filter(
    (e) => e.handFrames?.length && !e.handedness?.twoHanded && (e.handedness?.framesWithBoth ?? 0) === 0
);

/** A hand at a chosen image position, with MediaPipe's own 21-point topology
 * standing in as a plain offset cloud. Synthetic ON PURPOSE: it exists to
 * construct detection ORDER, which real recorded data cannot be made to do. */
function handAt(x: number, y: number): HandLandmarks {
    return Array.from({ length: 21 }, (_, i) => ({
        x: x + (i % 5) * 0.005,
        y: y + Math.floor(i / 5) * 0.005,
        z: 0,
    }));
}

beforeAll(async () => {
    vi.stubGlobal("fetch", async () => ({
        ok: true,
        status: 200,
        json: async () => JSON.parse(registryJson),
    }));
    clearDynamicSignCache();
    await preloadDynamicSignRegistry();
}, 300_000);

describe("1. MediaPipe hand-order permutation does not change hand identity", () => {
    it("keeps each physical hand in the same slot when detections are reordered", () => {
        const leftPath = Array.from({ length: 12 }, (_, i) => handAt(0.6 + i * 0.002, 0.5));
        const rightPath = Array.from({ length: 12 }, (_, i) => handAt(0.3 - i * 0.002, 0.5));

        const tracker = new HandSlotTracker();
        const slot0: Array<HandLandmarks | null> = [];
        for (let i = 0; i < 12; i++) {
            const a: SlotDetection = { landmarks: leftPath[i], handedness: "Left", score: 0.9 };
            const b: SlotDetection = { landmarks: rightPath[i], handedness: "Right", score: 0.9 };
            // Flip the order MediaPipe reports on every other frame.
            const slots = tracker.step(i % 2 === 0 ? [a, b] : [b, a]);
            slot0.push(slots[0]?.landmarks ?? null);
        }

        // Slot 0 must be ONE hand the whole way, not an alternation of two.
        for (let i = 1; i < slot0.length; i++) {
            const prev = slot0[i - 1]!;
            const cur = slot0[i]!;
            expect(Math.hypot(cur[0].x - prev[0].x, cur[0].y - prev[0].y)).toBeLessThan(0.05);
        }
        expect(tracker.switchCount).toBeGreaterThan(0); // it really did have to reorder
    });

    it("names the sides the same way regardless of the order they arrived in", () => {
        const a = Array.from({ length: 10 }, (_, i) => handAt(0.65, 0.5 + i * 0.001));
        const b = Array.from({ length: 10 }, (_, i) => handAt(0.25, 0.5 + i * 0.001));
        const labels = Array.from({ length: 10 }, () => "Left" as const);
        const other = Array.from({ length: 10 }, () => "Right" as const);

        const forward = assignSides(a, b, labels, other);
        const reversed = assignSides(b, a, other, labels);

        expect(forward.source).toBe("geometry");
        // The hand at the SMALLER image x is the signer's right in both orders.
        expect(forward.sides[1]).toBe("right");
        expect(reversed.sides[0]).toBe("right");
    });
});

describe("2. two-handed signs match on both real hands", () => {
    it("has real two-handed signs to test with", () => {
        expect(twoHanded.length).toBeGreaterThan(0);
    });

    it("compares BOTH sides and finds the sign itself", () => {
        const entry = twoHanded[0];
        const sided = entryToSidedSequence(entry)!;
        const diag = recognizeDynamicSignSyncWithDiagnostics(sided);
        expect(diag.bestSignId).toBe(entry.sourceSignId);
        expect(diag.bestSharedSides).toBe(2);
        expect(diag.bestPresenceMismatches).toBe(0);
        expect(diag.bestPerSideDistance.left).not.toBeNull();
        expect(diag.bestPerSideDistance.right).not.toBeNull();
    });
});

describe("3. one-handed signs still work", () => {
    it("matches a genuinely one-handed real sign without a presence penalty", () => {
        if (oneHanded.length === 0) return; // the dataset may contain none
        const entry = oneHanded[0];
        const sided = entryToSidedSequence(entry)!;
        const diag = recognizeDynamicSignSyncWithDiagnostics(sided);
        expect(diag.bestSignId).toBe(entry.sourceSignId);
        expect(diag.bestSharedSides).toBe(1);
        expect(diag.bestPresenceMismatches).toBe(0);
    });
});

describe("4. the presence penalty is charged in one direction only", () => {
    /**
     * This block used to assert the opposite, and the opposite was wrong.
     *
     * The penalty was charged whenever either sequence had a side the other
     * lacked. Measured on real held-out data that handed one-hand captures to
     * whichever wrong ONE-HANDED sign happened to fit, because the correct
     * two-handed template was the only one paying: sign 0033 ("ب", template
     * L14/R14) captured as L14/R0 scored 0.6466 against the wrong one-handed
     * 0356 and was not even second against its own template. One-hand queries
     * were 10.7% of the benchmark and carried 3 of its 6 false accepts.
     *
     * The contract now: a hand missing from the QUERY is a capture failure and
     * costs nothing; a hand missing from the TEMPLATE is evidence against that
     * sign and is charged. Both directions are asserted, because getting either
     * one wrong is what produced the bug.
     */
    it("does NOT charge a query that is missing a hand the template uses", () => {
        const entry = twoHanded[0];
        const sided = entryToSidedSequence(entry)!;
        const full = prepareSidedSequence(sided);
        const rightOnly = prepareSidedSequence(sided.map((f) => ({ left: null, right: f.right })));

        const partial = sidedDistance(rightOnly, full);

        expect(partial.sharedSides).toBe(1);
        expect(partial.presenceMismatches).toBe(0);
        expect(partial.queryMissingSides).toBe(1);
        // No penalty component at all: the distance is the shared side's own.
        expect(partial.distance).toBeCloseTo(partial.perSide.right!, 10);
    });

    it("DOES charge a template that is missing a hand the query used", () => {
        const entry = twoHanded[0];
        const sided = entryToSidedSequence(entry)!;
        const full = prepareSidedSequence(sided);
        const templateRightOnly = prepareSidedSequence(sided.map((f) => ({ left: null, right: f.right })));

        const complete = sidedDistance(full, full);
        const charged = sidedDistance(full, templateRightOnly);

        expect(complete.presenceMismatches).toBe(0);
        expect(charged.presenceMismatches).toBe(1);
        expect(charged.queryMissingSides).toBe(0);
        // Exactly the documented charge, not an arbitrary fudge.
        expect(charged.distance - charged.perSide.right!).toBeCloseTo(UNMATCHED_SIDE_COST / 2, 10);
        expect(charged.distance).toBeGreaterThan(complete.distance);
    });

    it("keeps the correct two-handed sign reachable from a one-hand capture", () => {
        // The property the whole change exists for: a capture that lost a hand
        // must not be pushed behind a one-handed template it does not belong to.
        const entry = twoHanded[0];
        const sided = entryToSidedSequence(entry)!;
        const rightOnly = prepareSidedSequence(sided.map((f) => ({ left: null, right: f.right })));

        const own = sidedDistance(rightOnly, prepareSidedSequence(sided));
        // A one-handed template carrying the SAME right-hand motion is the
        // strongest possible rival; it must not win by presence alone.
        const rivalOneHanded = prepareSidedSequence(sided.map((f) => ({ left: null, right: f.right })));
        const rival = sidedDistance(rightOnly, rivalOneHanded);

        expect(own.distance).toBeCloseTo(rival.distance, 10);
    });
});

describe("5. a query on the WRONG hand does not win", () => {
    it("scores far worse than the same motion on the correct hand", () => {
        const entry = twoHanded[0];
        const sided = entryToSidedSequence(entry)!;
        const template = prepareSidedSequence(sided);

        const rightOnly = prepareSidedSequence(sided.map((f) => ({ left: null, right: f.right })));
        // The SAME real motion, presented as if it were the other hand.
        const asLeft = prepareSidedSequence(sided.map((f) => ({ left: f.right, right: null })));

        const correct = sidedDistance(rightOnly, template).distance;
        const wrongHand = sidedDistance(asLeft, template).distance;
        expect(wrongHand).toBeGreaterThan(correct);
    });
});

describe("6. symmetric two-handed signs keep their hands apart", () => {
    it("swapping the two real hands is not accepted as the same sign", () => {
        // Symmetric signs are exactly where the old detection-count rule
        // degenerated into a coin flip, so they are the ones to check.
        const symmetric = twoHanded.filter(
            (e) => (e.handedness?.framesWithLeft ?? 0) === (e.handedness?.framesWithRight ?? 0)
        );
        expect(symmetric.length).toBeGreaterThan(0);

        let checked = 0;
        for (const entry of symmetric.slice(0, 5)) {
            const sided = entryToSidedSequence(entry)!;
            const swapped: SidedSequence = sided.map((f) => ({ left: f.right, right: f.left }));
            const self = recognizeDynamicSignSyncWithDiagnostics(sided);
            const flipped = recognizeDynamicSignSyncWithDiagnostics(swapped);
            expect(self.bestSignId).toBe(entry.sourceSignId);
            // Swapping real hands must cost something: either it stops being
            // the nearest neighbour, or it gets further away.
            const degraded =
                flipped.bestSignId !== entry.sourceSignId ||
                (flipped.bestDistance ?? 0) > (self.bestDistance ?? 0);
            expect(degraded).toBe(true);
            checked++;
        }
        expect(checked).toBeGreaterThan(0);
    });
});

describe("7. sourceVideo traceability survives into the registry", () => {
    const withTake = entries.filter((e) => e.sourceTake);

    it("records which real recording each template came from", () => {
        // Entries built before traceability existed have no sourceTake; this
        // asserts the field's SHAPE wherever it is present, and reports rather
        // than passing vacuously when the registry predates the change.
        if (withTake.length === 0) {
            console.warn(
                "[twoHandRecognition] No entry carries sourceTake: this registry predates " +
                "traceability. Re-run ml/training/import_karsl.py to populate it."
            );
            return;
        }
        for (const e of withTake.slice(0, 50)) {
            expect(e.sourceTake!.processedFile).toMatch(/\.json$/);
            expect(e.sourceTake!.sourceVideoStem.length).toBeGreaterThan(0);
            expect(e.sourceTake!.sourceFolder).toBe(e.sourceSignId);
            expect(e.sourceTake!.takesAvailable).toBeGreaterThanOrEqual(1);
            expect(e.sourceTake!.selectionReason.length).toBeGreaterThan(0);
        }
    });

    it("points at a file that really exists, when the source tree is present", () => {
        const processed = path.join(ROOT, DEFAULT_PROCESSED_ROOT);
        if (withTake.length === 0 || !existsSync(processed)) return;
        let checked = 0;
        for (const e of withTake) {
            const dir = path.join(processed, e.sourceTake!.sourceFolder);
            if (!existsSync(dir)) continue;
            expect(readdirSync(dir)).toContain(e.sourceTake!.processedFile);
            checked++;
            if (checked >= 10) break;
        }
    });
});

describe("8. template-take exclusion is by content, not by position", () => {
    it("the stored template matches itself at ~zero distance", () => {
        // This is the property the held-out benchmark relies on to EXCLUDE the
        // template: a take that is the template is identifiable by content.
        const entry = twoHanded[0];
        const diag = recognizeDynamicSignSyncWithDiagnostics(entryToSidedSequence(entry)!);
        expect(diag.bestSignId).toBe(entry.sourceSignId);
        expect(diag.bestDistance).toBeLessThan(0.02);
    });
});

describe("9. left/right identity is stable across a whole captured gesture", () => {
    it("names the sides once for the gesture, not per frame", () => {
        const buffer = new DynamicGestureBuffer();
        const tracker = new HandSlotTracker();
        const frames = Math.max(DEFAULT_GESTURE_BUFFER_CONFIG.minGestureFrames + 4, 12);

        for (let i = 0; i < frames; i++) {
            const left: SlotDetection = {
                landmarks: handAt(0.62 + i * 0.003, 0.5),
                handedness: "Left",
                score: 0.9,
            };
            const right: SlotDetection = {
                landmarks: handAt(0.28 - i * 0.003, 0.5),
                handedness: "Right",
                score: 0.9,
            };
            // Detection order flips constantly, as it does in real output.
            buffer.step(tracker.step(i % 2 === 0 ? [left, right] : [right, left]));
        }
        let result = buffer.step([null, null]);
        for (let i = 0; i < DEFAULT_GESTURE_BUFFER_CONFIG.endGapFrames && !result.completedGesture; i++) {
            result = buffer.step([null, null]);
        }

        expect(result.completedGesture).not.toBeNull();
        expect(result.completedSideAssignment?.source).toBe("geometry");
        const gesture = result.completedGesture!;
        // Every frame carries both hands, and the hand on each side moves
        // continuously -- i.e. the side names did not swap part-way through.
        for (let i = 1; i < gesture.length; i++) {
            for (const side of ["left", "right"] as const) {
                const prev = gesture[i - 1][side];
                const cur = gesture[i][side];
                expect(prev).not.toBeNull();
                expect(cur).not.toBeNull();
                expect(Math.hypot(cur![0].x - prev![0].x, cur![0].y - prev![0].y)).toBeLessThan(0.05);
            }
        }
        // The signer's right hand is the one at the smaller image x.
        expect(gesture[0].right![0].x).toBeLessThan(gesture[0].left![0].x);
    });

    it("reports an unnameable single hand as unknown instead of guessing a side", () => {
        const buffer = new DynamicGestureBuffer();
        const frames = Math.max(DEFAULT_GESTURE_BUFFER_CONFIG.minGestureFrames + 4, 12);
        for (let i = 0; i < frames; i++) {
            // One hand, never two, and MediaPipe gave no label: neither the
            // geometric vote nor the label vote can name it.
            buffer.step([{ landmarks: handAt(0.5 + i * 0.003, 0.5), handedness: null, score: 0.9 }, null]);
        }
        let result = buffer.step([null, null]);
        for (
            let i = 0;
            i < DEFAULT_GESTURE_BUFFER_CONFIG.endGapFrames && !result.completedUnknownSideHand;
            i++
        ) {
            result = buffer.step([null, null]);
        }

        expect(result.completedSideAssignment?.source).toBe("unknown");
        expect(result.completedGesture).toBeNull();
        expect(result.completedUnknownSideHand).not.toBeNull();

        // And it is recognised AS an unnamed hand -- compared against a side,
        // but charged for the side it cannot account for.
        const diag = recognizeUnknownSideHandSyncWithDiagnostics(result.completedUnknownSideHand!);
        expect(diag.querySideUnknown).toBe(true);
        expect(diag.bestSharedSides).toBe(1);
    });
});
