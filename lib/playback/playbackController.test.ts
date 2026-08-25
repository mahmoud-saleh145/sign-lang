import { describe, it, expect } from "vitest";
import { PlaybackController } from "./playbackController";

const CONFIG = { stepDurationMs: 1000 };

describe("PlaybackController", () => {
    it("starts idle with step index 0", () => {
        const c = new PlaybackController(3, CONFIG);
        expect(c.getSnapshot()).toEqual({
            state: "idle",
            currentStepIndex: 0,
            totalSteps: 3,
            speed: 1,
            progressWithinStep: 0,
        });
    });

    it("play() transitions to playing", () => {
        const c = new PlaybackController(3, CONFIG);
        c.play();
        expect(c.getSnapshot().state).toBe("playing");
    });

    it("advances to the next step in order as time passes", () => {
        const c = new PlaybackController(3, CONFIG);
        c.play();
        c.tick(999);
        expect(c.getSnapshot().currentStepIndex).toBe(0); // not yet a full step
        c.tick(2);
        expect(c.getSnapshot().currentStepIndex).toBe(1); // crossed the threshold
    });

    it("pause() stops advancing, resume() continues from where it left off", () => {
        const c = new PlaybackController(3, CONFIG);
        c.play();
        c.tick(500);
        c.pause();
        expect(c.getSnapshot().state).toBe("paused");
        c.tick(10000); // should have zero effect while paused
        expect(c.getSnapshot().currentStepIndex).toBe(0);
        c.resume();
        expect(c.getSnapshot().state).toBe("playing");
        c.tick(600); // 500 + 600 = 1100 > 1000 -> advances
        expect(c.getSnapshot().currentStepIndex).toBe(1);
    });

    it("reaches 'finished' state at the end of the sequence, not before", () => {
        const c = new PlaybackController(2, CONFIG);
        c.play();
        c.tick(1000); // -> step 1
        expect(c.getSnapshot().state).toBe("playing");
        c.tick(1000); // -> past step 1, sequence has only 2 steps (0,1) -> finished
        const snap = c.getSnapshot();
        expect(snap.state).toBe("finished");
        expect(snap.currentStepIndex).toBe(1); // stays on the last valid step
    });

    it("tick() after finished is a no-op", () => {
        const c = new PlaybackController(1, CONFIG);
        c.play();
        c.tick(1000);
        expect(c.getSnapshot().state).toBe("finished");
        c.tick(5000);
        expect(c.getSnapshot()).toEqual({
            state: "finished",
            currentStepIndex: 0,
            totalSteps: 1,
            speed: 1,
            progressWithinStep: 1,
        });
    });

    it("replay() restarts from step 0 regardless of current state", () => {
        const c = new PlaybackController(3, CONFIG);
        c.play();
        c.tick(2500); // -> step 2
        c.replay();
        const snap = c.getSnapshot();
        expect(snap.currentStepIndex).toBe(0);
        expect(snap.state).toBe("playing");
    });

    it("play() on an empty sequence goes straight to finished, never crashes", () => {
        const c = new PlaybackController(0, CONFIG);
        c.play();
        expect(c.getSnapshot().state).toBe("finished");
    });

    it("speed multiplies effective elapsed time", () => {
        const c = new PlaybackController(2, CONFIG);
        c.setSpeed(1.25);
        c.play();
        c.tick(800); // 800 * 1.25x = 1000ms -> crosses the 1000ms threshold
        expect(c.getSnapshot().currentStepIndex).toBe(1);
    });

    it("setSequenceLength() resets playback for a new sequence", () => {
        const c = new PlaybackController(2, CONFIG);
        c.play();
        c.tick(1000);
        expect(c.getSnapshot().currentStepIndex).toBe(1);
        c.setSequenceLength(5);
        expect(c.getSnapshot()).toEqual({
            state: "idle",
            currentStepIndex: 0,
            totalSteps: 5,
            speed: 1,
            progressWithinStep: 0,
        });
    });

    it("tick() while idle or paused does not change state", () => {
        const c = new PlaybackController(3, CONFIG);
        c.tick(5000);
        expect(c.getSnapshot().state).toBe("idle");
    });

    it("progressWithinStep tracks 0..1 progress through the current step, for animating multi-frame signs", () => {
        const c = new PlaybackController(2, CONFIG);
        c.play();
        expect(c.getSnapshot().progressWithinStep).toBe(0);
        c.tick(500);
        expect(c.getSnapshot().progressWithinStep).toBeCloseTo(0.5);
        c.tick(500);
        // crossed into step 1, progress resets for the new step
        expect(c.getSnapshot().currentStepIndex).toBe(1);
        expect(c.getSnapshot().progressWithinStep).toBeCloseTo(0);
    });
});