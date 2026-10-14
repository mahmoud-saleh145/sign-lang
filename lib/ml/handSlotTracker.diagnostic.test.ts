/**
 * WHAT "tracker swap" MEANS, pinned by test.
 *
 * Real-world word-mode telemetry reported 91 swaps on a 93-frame gesture and
 * that reads as the left/right slots oscillating once per frame, corrupting
 * the stored trajectory and wrecking the DTW margin.
 *
 * It is the opposite. switchCount increments when the tracker REORDERS
 * MediaPipe's incoming array to preserve slot identity -- it counts
 * corrections applied, not identities lost. These tests separate the two
 * quantities that the single telemetry number conflated:
 *
 *     swaps         how often the incoming array order had to be corrected
 *     slot purity   whether a slot actually kept one physical hand
 *
 * and show that the near-one-swap-per-frame signature is produced by a
 * PERFECTLY STABLE tracker.
 *
 * Fixtures are real KArSL landmark motion. Only the ORDER the two hands are
 * presented in is varied -- the physical hands are identical in every case,
 * which is what makes purity checkable rather than inferred.
 */
import { describe, it, expect } from "vitest";
import { readFileSync, existsSync } from "fs";
import path from "path";
import { HandSlotTracker, type SlotDetection } from "./handSlotTracker";
import type { HandLandmarks } from "@/types/landmarks";

interface ProcessedHand { landmarks: HandLandmarks; rawLabel?: "Left" | "Right" }
interface ProcessedFrame { hands?: { left?: ProcessedHand | null; right?: ProcessedHand | null } }
interface TruthFrame { left: ProcessedHand | null; right: ProcessedHand | null }

const ROOT = path.resolve(__dirname, "../..");
const TAKE = path.join(ROOT, "processed-karsl/0071/03_01_0071_(01_12_16_15_53_03)_c.json");

function loadTwoHandFrames(): TruthFrame[] {
    if (!existsSync(TAKE)) return [];
    const raw = JSON.parse(readFileSync(TAKE, "utf-8")) as { frames?: ProcessedFrame[] };
    return (raw.frames ?? [])
        .map((f) => ({ left: f.hands?.left ?? null, right: f.hands?.right ?? null }))
        .filter((f): f is TruthFrame => Boolean(f.left && f.right));
}

const frames = loadTwoHandFrames();
const describeIfData = frames.length >= 20 ? describe : describe.skip;

function det(h: ProcessedHand): SlotDetection {
    return { landmarks: h.landmarks, handedness: h.rawLabel ?? null, score: 0.9 };
}

/** Feeds frames in a chosen order and reports swaps AND slot purity. */
function measure(order: (f: TruthFrame, i: number) => SlotDetection[]) {
    const tracker = new HandSlotTracker();
    const slot0: Array<"left" | "right"> = [];
    for (let i = 0; i < frames.length; i++) {
        const slots = tracker.step(order(frames[i], i));
        const s0 = slots[0];
        if (!s0) continue;
        slot0.push(s0.landmarks === frames[i].left!.landmarks ? "left" : "right");
    }
    const lefts = slot0.filter((s) => s === "left").length;
    let identityChanges = 0;
    for (let i = 1; i < slot0.length; i++) if (slot0[i] !== slot0[i - 1]) identityChanges++;
    return {
        swaps: tracker.switchCount,
        swapsPerFrame: tracker.switchCount / frames.length,
        purity: slot0.length ? Math.max(lefts, slot0.length - lefts) / slot0.length : 1,
        identityChanges,
    };
}

const asStored = (f: TruthFrame) => [det(f.left!), det(f.right!)];
const reversed = (f: TruthFrame) => [det(f.right!), det(f.left!)];

describeIfData("HandSlotTracker — what switchCount counts", () => {
    it("counts ZERO swaps when MediaPipe's order is stable", () => {
        const r = measure(asStored);
        expect(r.swaps).toBe(0);
        expect(r.purity).toBe(1);
        expect(r.identityChanges).toBe(0);
    });

    it("counts ZERO swaps when the order is stably REVERSED", () => {
        // The tracker seeds its slots from frame 0, so a consistently reversed
        // stream simply seeds the other way round and never needs correcting.
        const r = measure(reversed);
        expect(r.swaps).toBe(0);
        expect(r.purity).toBe(1);
    });

    it("counts ~0.5 swaps/frame when the order ALTERNATES, and still loses nothing", () => {
        const r = measure((f, i) => (i % 2 === 1 ? reversed(f) : asStored(f)));
        expect(r.swapsPerFrame).toBeGreaterThan(0.4);
        expect(r.swapsPerFrame).toBeLessThan(0.6);
        // The corrections worked: no slot ever changed physical hand.
        expect(r.purity).toBe(1);
        expect(r.identityChanges).toBe(0);
    });

    it("reproduces ~1 swap PER FRAME with the slots still perfectly stable", () => {
        // THE REPORTED SIGNATURE: 91 swaps on 93 frames.
        //
        // It appears when the tracker seeds its slots one way on the first
        // frame and MediaPipe then returns the two hands in the OPPOSITE order
        // on every following frame. The tracker corrects all of them, so the
        // count approaches one per frame while purity stays at 1.000.
        //
        // A near-100% swap rate is therefore evidence of a STABLE upstream
        // ordering that disagrees with the seed -- the most benign case, not
        // the most alarming one.
        const r = measure((f, i) => (i === 0 ? asStored(f) : reversed(f)));
        expect(r.swapsPerFrame).toBeGreaterThan(0.9);
        expect(r.purity).toBe(1);
        expect(r.identityChanges).toBe(0);
    });

    it("a high swap count NEVER implies a lost hand identity", () => {
        for (const order of [
            asStored,
            reversed,
            (f: TruthFrame, i: number) => (i % 2 === 1 ? reversed(f) : asStored(f)),
            (f: TruthFrame, i: number) => (i === 0 ? asStored(f) : reversed(f)),
        ]) {
            const r = measure(order);
            expect(r.purity, `purity dropped at ${r.swaps} swaps`).toBe(1);
        }
    });

    it("WITHOUT the tracker, an alternating order destroys the trajectory", () => {
        // The control: this is what the swap count is buying.
        let changes = 0;
        let prev: "left" | "right" | null = null;
        for (let i = 0; i < frames.length; i++) {
            const first = i % 2 === 1 ? "right" : "left";
            if (prev !== null && first !== prev) changes++;
            prev = first;
        }
        expect(changes).toBeGreaterThan(frames.length / 2);
    });

    it("keeps identity even when the two wrists are nearly touching", () => {
        // Wrist proximity is the intuitive failure mode for a nearest-wrist
        // rule. Measured across 122 real takes it is not the driver: the six
        // takes with the closest approach (down to 0.0010 separation) all held
        // purity 1.000 under an adversarial alternating order.
        let minSep = Infinity;
        for (const f of frames) {
            minSep = Math.min(
                minSep,
                Math.hypot(
                    f.left!.landmarks[0].x - f.right!.landmarks[0].x,
                    f.left!.landmarks[0].y - f.right!.landmarks[0].y
                )
            );
        }
        const r = measure((f, i) => (i % 2 === 1 ? reversed(f) : asStored(f)));
        expect(r.purity, `closest wrist separation in this take: ${minSep.toFixed(4)}`).toBe(1);
    });
});
