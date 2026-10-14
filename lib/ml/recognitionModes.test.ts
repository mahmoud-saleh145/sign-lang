/**
 * MODE ISOLATION TESTS.
 *
 * Verifies the three recognition modes are actually independent at the
 * pipeline level -- specifically that WORD mode cannot write letters into
 * the transcript, which is the interference the user observed (the static
 * model confidently reporting "Waw" at 0.97 while a dynamic sign was being
 * performed).
 *
 * The static letter side uses REAL landmark samples
 * (ml/tests/fixtures/continuous_stream_fixture.json) and the REAL trained
 * weights (public/models/arsl-31/weights.json), so "the letter path still
 * commits" is proven against the real model, not a stub.
 *
 * The dynamic side here is exercised only for PLUMBING (does the buffer get
 * stepped, are commits routed to the right transcript). Real-data accuracy
 * of dynamic recognition is covered by karslHeldOut.test.ts instead.
 */
import { describe, it, expect, beforeEach } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { ArSLClassifier } from "./model";
import type { ModelWeights } from "./model";
import { SignRecognitionPipeline } from "./recognitionPipeline";
import { clearDynamicSignCache } from "./dynamicSignRecognizer";
import type { HandLandmarks, FrameDetection } from "@/types/landmarks";

const weights = JSON.parse(
    readFileSync(path.resolve(__dirname, "../../public/models/arsl-31/weights.json"), "utf-8")
) as ModelWeights;

const streamFixture = JSON.parse(
    readFileSync(path.resolve(__dirname, "../../ml/tests/fixtures/continuous_stream_fixture.json"), "utf-8")
) as Record<string, HandLandmarks[]>;

function classifier() {
    return new ArSLClassifier(weights);
}

function frame(landmarks: HandLandmarks | null, t: number): FrameDetection {
    return {
        timestampMs: t,
        hands: landmarks ? [{ landmarks, handedness: "Right", score: 0.99 }] : [],
    };
}

/** Feeds one real letter pose long enough to stabilize, then a run of
 * no-hand frames. The static segmenter commits on sign END (gap frames or a
 * different confident sign), never while a pose is merely held -- so the
 * trailing idle frames are what actually produce the commit. */
function holdLetter(pipeline: SignRecognitionPipeline, label: string, frames = 20, startT = 0, idleFrames = 12) {
    const samples = streamFixture[label];
    if (!samples || samples.length === 0) throw new Error(`No fixture samples for ${label}`);
    const ticks = [];
    for (let i = 0; i < frames; i++) {
        ticks.push(pipeline.processFrame(frame(samples[i % samples.length], startT + i * 33)));
    }
    for (let i = 0; i < idleFrames; i++) {
        ticks.push(pipeline.processFrame(frame(null, startT + (frames + i) * 33)));
    }
    return ticks;
}

describe("recognition mode isolation", () => {
    beforeEach(() => {
        clearDynamicSignCache();
    });

    it("LETTER mode still commits letters (existing behaviour unchanged)", () => {
        const p = new SignRecognitionPipeline(classifier(), undefined, undefined, "letter");
        const ticks = holdLetter(p, "Alef");
        const commits = ticks.filter((t) => t.newTranscriptEntry !== null);
        expect(commits.length).toBeGreaterThan(0);
        expect(p.getTranscript().length).toBeGreaterThan(0);
        expect(p.getArabicText().length).toBeGreaterThan(0);
    });

    it("LETTER mode never steps the dynamic gesture buffer", () => {
        const p = new SignRecognitionPipeline(classifier(), undefined, undefined, "letter");
        const ticks = holdLetter(p, "Alef", 30);
        for (const t of ticks) {
            expect(t.telemetry.dynamicPhase).toBe("off");
            expect(t.telemetry.dynamicGestureFrameCount).toBe(0);
            expect(t.newDynamicSignEntry).toBeNull();
        }
        expect(p.getWordTranscript().length).toBe(0);
    });

    it("WORD mode never commits a letter, even when the static model is confident", () => {
        const p = new SignRecognitionPipeline(classifier(), undefined, undefined, "word");
        const ticks = holdLetter(p, "Alef", 40);

        // The static model must still be RUNNING and confident -- this test is
        // meaningless if it simply stopped predicting.
        const confident = ticks.filter((t) => t.telemetry.rawConfidence > 0.5);
        expect(confident.length).toBeGreaterThan(0);

        // ...but nothing may reach the transcript or the Arabic text.
        for (const t of ticks) {
            expect(t.newTranscriptEntry).toBeNull();
            expect(t.telemetry.committedThisFrame).toBeNull();
        }
        expect(p.getTranscript().length).toBe(0);
        expect(p.getArabicText()).toBe("");
    });

    it("WORD mode reports suppression rather than hiding it", () => {
        const p = new SignRecognitionPipeline(classifier(), undefined, undefined, "word");
        const ticks = holdLetter(p, "Alef", 40);
        // At least one frame must have had a commit that was dropped, or the
        // fixture never reached a commit and the test above proves nothing.
        expect(ticks.some((t) => t.telemetry.letterCommitSuppressed)).toBe(true);
    });

    it("WORD mode does step the dynamic gesture buffer", () => {
        const p = new SignRecognitionPipeline(classifier(), undefined, undefined, "word");
        const ticks = holdLetter(p, "Alef", 20, 0, 0);
        expect(ticks.some((t) => t.telemetry.dynamicPhase === "buffering")).toBe(true);
        expect(Math.max(...ticks.map((t) => t.telemetry.dynamicGestureFrameCount))).toBeGreaterThan(0);
    });

    it("BOTH mode commits letters and keeps a separate word transcript", () => {
        const p = new SignRecognitionPipeline(classifier(), undefined, undefined, "both");
        const ticks = holdLetter(p, "Alef", 25);
        expect(ticks.some((t) => t.newTranscriptEntry !== null)).toBe(true);
        expect(ticks.some((t) => t.telemetry.dynamicPhase === "buffering")).toBe(true);
        // No dynamic match is expected from a held letter pose, and crucially
        // the word transcript must not have been polluted by letter commits.
        expect(p.getWordTranscript().length).toBe(0);
    });

    it("switching modes resets both state machines so state cannot leak", () => {
        const p = new SignRecognitionPipeline(classifier(), undefined, undefined, "word");
        holdLetter(p, "Alef", 15);
        p.setMode("letter");
        const t = p.processFrame(frame(null, 10_000));
        expect(t.telemetry.dynamicGestureFrameCount).toBe(0);
        expect(t.telemetry.segmentationState).toBe("IDLE");
        expect(t.telemetry.mode).toBe("letter");
    });

    it("a manual capture window is ignored in LETTER mode", () => {
        const p = new SignRecognitionPipeline(classifier(), undefined, undefined, "letter");
        p.beginManualCapture();
        const t = p.processFrame(frame(streamFixture.Alef[0], 0));
        expect(t.telemetry.dynamicManualCapture).toBe(false);
    });

    it("a manual capture window survives dropped detections in WORD mode", () => {
        const p = new SignRecognitionPipeline(classifier(), undefined, undefined, "word");
        p.beginManualCapture();
        const samples = streamFixture.Alef;
        let last;
        for (let i = 0; i < 20; i++) {
            // every third frame has no detected hand -- an automatic buffer
            // would have ended the gesture; a manual window must not.
            last = p.processFrame(frame(i % 3 === 2 ? null : samples[i % samples.length], i * 33));
        }
        expect(last!.telemetry.dynamicManualCapture).toBe(true);
        expect(last!.telemetry.dynamicGestureFrameCount).toBeGreaterThan(10);
    });
});
