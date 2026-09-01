import { describe, it, expect } from "vitest";
import { DynamicGestureBuffer, DEFAULT_GESTURE_BUFFER_CONFIG } from "./dynamicGestureBuffer";
import type { HandLandmarks } from "@/types/landmarks";

// SYNTHETIC fixtures only -- no real KArSL motion data was available when
// these were written. staticPose() and movingPose() are just numeric
// stand-ins for "not moving" vs "moving," not real hand shapes.

function staticPose(): HandLandmarks {
    return Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
}

function movingPose(step: number): HandLandmarks {
    return Array.from({ length: 21 }, (_, i) => ({ x: 0.1 + 0.05 * step + i * 0.001, y: 0.2 + 0.05 * step, z: 0 }));
}

describe("DynamicGestureBuffer", () => {
    it("starts idle and stays idle while no hand is present", () => {
        const buf = new DynamicGestureBuffer();
        for (let i = 0; i < 10; i++) {
            const result = buf.step(null);
            expect(result.state).toBe("idle");
            expect(result.completedGesture).toBeNull();
        }
    });

    it("begins buffering as soon as a hand appears", () => {
        const buf = new DynamicGestureBuffer();
        const result = buf.step(movingPose(0));
        expect(result.state).toBe("buffering");
    });

    it("finalizes and returns a completed gesture after the hand disappears for endGapFrames", () => {
        const buf = new DynamicGestureBuffer({ ...DEFAULT_GESTURE_BUFFER_CONFIG, minGestureFrames: 5 });
        for (let i = 0; i < 15; i++) buf.step(movingPose(i));
        let last;
        for (let i = 0; i < DEFAULT_GESTURE_BUFFER_CONFIG.endGapFrames; i++) {
            last = buf.step(null);
        }
        expect(last!.completedGesture).not.toBeNull();
        expect(last!.completedGesture!.length).toBe(15);
    });

    it("discards a completed buffer that is too short (below minGestureFrames)", () => {
        const buf = new DynamicGestureBuffer({ ...DEFAULT_GESTURE_BUFFER_CONFIG, minGestureFrames: 20 });
        for (let i = 0; i < 5; i++) buf.step(movingPose(i)); // well below 20
        let last;
        for (let i = 0; i < DEFAULT_GESTURE_BUFFER_CONFIG.endGapFrames; i++) {
            last = buf.step(null);
        }
        expect(last!.completedGesture).toBeNull();
        expect(last!.discardedAsNonDynamic).toBe(true);
    });

    it("discards a completed buffer that shows too little motion (looks like a held static letter)", () => {
        const buf = new DynamicGestureBuffer({ ...DEFAULT_GESTURE_BUFFER_CONFIG, minGestureFrames: 5 });
        for (let i = 0; i < 20; i++) buf.step(staticPose()); // held perfectly still
        let last;
        for (let i = 0; i < DEFAULT_GESTURE_BUFFER_CONFIG.endGapFrames; i++) {
            last = buf.step(null);
        }
        expect(last!.completedGesture).toBeNull();
        expect(last!.discardedAsNonDynamic).toBe(true);
    });

    it("does NOT finalize on brief gaps shorter than endGapFrames -- absorbs brief tracking dropout", () => {
        const buf = new DynamicGestureBuffer({ ...DEFAULT_GESTURE_BUFFER_CONFIG, minGestureFrames: 5 });
        for (let i = 0; i < 10; i++) buf.step(movingPose(i));
        const briefGap = buf.step(null); // just 1 frame of gap, well under endGapFrames
        expect(briefGap.completedGesture).toBeNull();
        expect(briefGap.state).toBe("buffering"); // still buffering, not reset
        const resumed = buf.step(movingPose(10));
        expect(resumed.state).toBe("buffering");
    });

    it("one continuous hold produces at most one completed gesture, never a repeated stream (duplicate prevention)", () => {
        const buf = new DynamicGestureBuffer({ ...DEFAULT_GESTURE_BUFFER_CONFIG, minGestureFrames: 5, maxGestureFrames: 1000 });
        const results = [];
        for (let i = 0; i < 200; i++) {
            results.push(buf.step(movingPose(i)));
        }
        const completions = results.filter((r) => r.completedGesture !== null);
        expect(completions).toHaveLength(0); // hand never dropped -- no finalize yet
    });

    it("force-finalizes at maxGestureFrames so a hand that never leaves frame doesn't buffer forever", () => {
        const buf = new DynamicGestureBuffer({ ...DEFAULT_GESTURE_BUFFER_CONFIG, minGestureFrames: 5, maxGestureFrames: 30 });
        let completed = null;
        for (let i = 0; i < 30; i++) {
            const r = buf.step(movingPose(i));
            if (r.completedGesture) completed = r.completedGesture;
        }
        expect(completed).not.toBeNull();
        expect(completed!.length).toBe(30);
    });

    it("allows a new gesture to begin immediately after one completes -- no artificial extra delay", () => {
        const config = { ...DEFAULT_GESTURE_BUFFER_CONFIG, minGestureFrames: 5 };
        const buf = new DynamicGestureBuffer(config);
        for (let i = 0; i < 10; i++) buf.step(movingPose(i));
        for (let i = 0; i < config.endGapFrames; i++) buf.step(null); // ends gesture 1

        // Very next frame with a hand should start buffering again immediately.
        const result = buf.step(movingPose(100));
        expect(result.state).toBe("buffering");
    });

    it("reset() clears any in-progress buffer", () => {
        const buf = new DynamicGestureBuffer();
        buf.step(movingPose(0));
        buf.step(movingPose(1));
        buf.reset();
        const result = buf.step(null);
        expect(result.state).toBe("idle");
    });
});