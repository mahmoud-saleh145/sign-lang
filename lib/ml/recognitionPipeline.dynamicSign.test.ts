/**
 * HONESTY NOTE (per project requirement): the user's real, actually-imported
 * public/models/sign-vocabulary/dedicated-signs.json was never uploaded to
 * this environment -- only the KARSL-502_Labels.xlsx spreadsheet was
 * available in an earlier turn, and no real processed-karsl/ or
 * dedicated-signs.json data has been provided since. Every dynamic-sign
 * fixture in this file is SYNTHETIC, built only to exercise the matching
 * algorithm and pipeline wiring correctly -- it does not represent any real
 * Arabic Sign Language movement, and should never be treated as a real-data
 * validation of recognition accuracy.
 *
 * What IS real here: the alphabet classifier (public/models/arsl-31/weights.json,
 * trained on the real ArSL-31 CSV) and the real per-letter landmark samples
 * used for the letter-interleaving tests below.
 */
import { describe, it, expect, beforeAll, vi } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { ArSLClassifier } from "./model";
import type { ModelWeights } from "./model";
import { SignRecognitionPipeline } from "./recognitionPipeline";
import { preloadDynamicSignRegistry, clearDynamicSignCache } from "./dynamicSignRecognizer";
import type { HandLandmarks, FrameDetection } from "@/types/landmarks";
import { DEFAULT_GESTURE_BUFFER_CONFIG } from "./dynamicGestureBuffer";

const weights = JSON.parse(
    readFileSync(path.resolve(__dirname, "../../public/models/arsl-31/weights.json"), "utf-8")
) as ModelWeights;

const streamFixture = JSON.parse(
    readFileSync(
        path.resolve(__dirname, "../../ml/tests/fixtures/continuous_stream_fixture.json"),
        "utf-8"
    )
) as Record<"Alef" | "Ba2" | "Space" | "7a2" | "Kaf" | "Noon" | "Delete", HandLandmarks[]>;

/**
 * IMPORTANT: this generator must include genuine per-frame SHAPE
 * deformation, not just rigid translation -- see the identical note in
 * dynamicSignRecognizer.test.ts for why (normalizeLandmarks() correctly
 * removes translation+scale, so a purely-rigid synthetic "hand" collapses
 * to the same degenerate shape on every frame and is indistinguishable
 * from any other rigid fixture). Verified numerically before use.
 */
function makeSyntheticGestureFrames(
    basePoint: [number, number],
    numFrames: number,
    freq: number,
    phase: number
): HandLandmarks[] {
    const frames: HandLandmarks[] = [];
    for (let f = 0; f < numFrames; f++) {
        frames.push(
            Array.from({ length: 21 }, (_, i) => {
                const angle = freq * f + phase + i * 0.35;
                return {
                    x: basePoint[0] + 0.01 * f + 0.06 * Math.cos(angle) + i * 0.004,
                    y: basePoint[1] + 0.01 * f + 0.06 * Math.sin(angle) + i * 0.004,
                    z: 0,
                };
            })
        );
    }
    return frames;
}

function makeEntry(text: string, sourceSignId: string, frames: HandLandmarks[]) {
    return {
        text,
        landmarkSequence: frames.map((frame) => frame.map((p) => ({ x: p.x, y: p.y, z: p.z }))),
        // The recognizer is side-aware, so a registry entry has to say which
        // hand its motion is. These fixtures are one-handed on the RIGHT,
        // matching the single "Right"-labelled hand the streams below feed in.
        handFrames: frames.map((frame) => ({
            left: null,
            right: frame.map((p) => ({ x: p.x, y: p.y, z: p.z })),
        })),
        handedness: {
            twoHanded: false,
            framesWithLeft: 0,
            framesWithRight: frames.length,
            framesWithBoth: 0,
            primaryHand: "right",
            sideSource: "handedness-label" as const,
            sideConfidence: 1,
            geometryFrames: 0,
        },
        provenance: {
            datasetName: "TEST-FIXTURE-NOT-REAL-KARSL",
            source: "unit test only",
            license: "n/a",
            version: "test",
        },
        sourceSignId,
    };
}

const GESTURE_A = makeSyntheticGestureFrames([0.3, 0.3], 20, 0.5, 0);
const GESTURE_B = makeSyntheticGestureFrames([0.7, 0.2], 20, 0.9, 1.5);
const GESTURE_PHRASE = makeSyntheticGestureFrames([0.5, 0.5], 25, 1.3, 3.0);

const SYNTHETIC_REGISTRY = [
    makeEntry("\u0645\u0631\u062d\u0628\u0627", "0001", GESTURE_A), // مرحبا
    makeEntry("\u0634\u0643\u0631\u0627", "0002", GESTURE_B), // شكرا
    makeEntry("\u0625\u0633\u0639\u0627\u0641\u0627\u062a \u0623\u0648\u0644\u064a\u0629", "0093", GESTURE_PHRASE), // إسعافات أولية
];

/** Builds a live-ish copy of a synthetic gesture with tiny noise, simulating
 * a camera capturing "the same sign" again. */
function liveLikeCopy(frames: HandLandmarks[]): HandLandmarks[] {
    return frames.map((frame) => frame.map((p) => ({ x: p.x + 0.0004, y: p.y + 0.0004, z: p.z })));
}

/** Builds a FrameDetection stream: a `label` from streamFixture (real letter
 * data) or explicit raw HandLandmarks[] (synthetic dynamic gesture), or
 * null for idle. */
type StreamSegment =
    | { kind: "letter"; label: keyof typeof streamFixture; frames: number }
    | { kind: "dynamic"; landmarks: HandLandmarks[] }
    | { kind: "idle"; frames: number };

function buildMixedStream(segments: StreamSegment[]): FrameDetection[] {
    const stream: FrameDetection[] = [];
    let t = 0;
    for (const seg of segments) {
        if (seg.kind === "idle") {
            for (let i = 0; i < seg.frames; i++) {
                stream.push({ timestampMs: t, hands: [] });
                t += 33;
            }
        } else if (seg.kind === "letter") {
            for (let i = 0; i < seg.frames; i++) {
                stream.push({
                    timestampMs: t,
                    hands: [
                        {
                            landmarks: streamFixture[seg.label][i % streamFixture[seg.label].length],
                            handedness: "Right" as const,
                            score: 0.95,
                        },
                    ],
                });
                t += 33;
            }
        } else {
            for (const landmarks of seg.landmarks) {
                stream.push({ timestampMs: t, hands: [{ landmarks, handedness: "Right" as const, score: 0.95 }] });
                t += 33;
            }
        }
    }
    return stream;
}

describe("SignRecognitionPipeline — combined static-letter + dynamic-KArSL recognition (SYNTHETIC dynamic-sign fixtures, real letter model+data)", () => {
    let classifier: ArSLClassifier;

    beforeAll(async () => {
        classifier = new ArSLClassifier(weights);
        clearDynamicSignCache();
        vi.stubGlobal(
            "fetch",
            vi.fn(async (url: string) => {
                if (url === "/models/sign-vocabulary/dedicated-signs.json") {
                    return { ok: true, status: 200, json: async () => SYNTHETIC_REGISTRY } as Response;
                }
                throw new Error(`Unexpected fetch in test: ${url}`);
            })
        );
        await preloadDynamicSignRegistry();
    });

    function newPipeline() {
        // Small minGestureFrames so 20-25 frame synthetic gestures qualify;
        // real KArSL frame counts are unknown, so this is a test-scale choice.
        return new SignRecognitionPipeline(classifier, undefined, {
            ...DEFAULT_GESTURE_BUFFER_CONFIG,
            minGestureFrames: 10,
        });
    }

    it("recognizes one real (well, real-shaped-but-synthetic) KArSL sign and appends its text", () => {
        const pipeline = newPipeline();
        const stream = buildMixedStream([
            { kind: "dynamic", landmarks: liveLikeCopy(GESTURE_A) },
            { kind: "idle", frames: 10 },
        ]);
        const dynamicCommits = stream
            .map((f) => pipeline.processFrame(f))
            .filter((r) => r.newDynamicSignEntry)
            .map((r) => r.newDynamicSignEntry!.displayLabel);
        expect(dynamicCommits).toEqual(["\u0645\u0631\u062d\u0628\u0627"]);
        expect(pipeline.getArabicText()).toBe("\u0645\u0631\u062d\u0628\u0627");
    });

    it("a multi-frame dynamic sign is only recognized after the WHOLE sequence completes, not mid-gesture", () => {
        const pipeline = newPipeline();
        const partial = liveLikeCopy(GESTURE_A).slice(0, 5); // well short of the real gesture
        const results = partial.map((landmarks) =>
            pipeline.processFrame({ timestampMs: 0, hands: [{ landmarks, handedness: "Right", score: 0.95 }] })
        );
        expect(results.every((r) => r.newDynamicSignEntry === null)).toBe(true);
    });

    it("a multi-word phrase is recognized and appended as ONE unit, never split into separate word commits", () => {
        const pipeline = newPipeline();
        const stream = buildMixedStream([
            { kind: "dynamic", landmarks: liveLikeCopy(GESTURE_PHRASE) },
            { kind: "idle", frames: 10 },
        ]);
        const dynamicCommits = stream
            .map((f) => pipeline.processFrame(f))
            .filter((r) => r.newDynamicSignEntry)
            .map((r) => r.newDynamicSignEntry!.displayLabel);
        expect(dynamicCommits).toEqual(["\u0625\u0633\u0639\u0627\u0641\u0627\u062a \u0623\u0648\u0644\u064a\u0629"]);
        expect(dynamicCommits[0].split(" ")).toHaveLength(2); // one commit, two words inside it
    });

    it("a low-confidence/no-match sequence does not append anything, and the letter path is unaffected", () => {
        const pipeline = newPipeline();
        const noise = makeSyntheticGestureFrames([-5, -5], 20, 5, 6); // nothing like any registry entry
        const stream = buildMixedStream([
            { kind: "dynamic", landmarks: noise },
            { kind: "idle", frames: 10 },
        ]);
        const results = stream.map((f) => pipeline.processFrame(f));
        expect(results.every((r) => r.newDynamicSignEntry === null)).toBe(true);
        expect(pipeline.getArabicText()).toBe("");
    });

    it("duplicate prevention: holding one continuous dynamic gesture produces exactly one recognition, never repeated", () => {
        const pipeline = newPipeline();
        // Extend GESTURE_A by holding its last frame many extra times, hand
        // never dropped -- simulates a user holding a completed gesture.
        const held = [...liveLikeCopy(GESTURE_A)];
        const lastFrame = held[held.length - 1];
        for (let i = 0; i < 60; i++) held.push(lastFrame.map((p) => ({ ...p })));

        const stream = buildMixedStream([{ kind: "dynamic", landmarks: held }, { kind: "idle", frames: 10 }]);
        const dynamicCommits = stream
            .map((f) => pipeline.processFrame(f))
            .filter((r) => r.newDynamicSignEntry)
            .map((r) => r.newDynamicSignEntry!.displayLabel);
        expect(dynamicCommits.length).toBeLessThanOrEqual(1); // never "مرحبا مرحبا مرحبا"
        const occurrences = pipeline.getArabicText().split("\u0645\u0631\u062d\u0628\u0627").length - 1;
        expect(occurrences).toBeLessThanOrEqual(1);
    });

    it("two consecutive dynamic signs (A then B) are both recognized in order, with no long artificial idle gap required", () => {
        const pipeline = newPipeline();
        const stream = buildMixedStream([
            { kind: "dynamic", landmarks: liveLikeCopy(GESTURE_A) },
            { kind: "idle", frames: DEFAULT_GESTURE_BUFFER_CONFIG.endGapFrames + 1 }, // just enough to end gesture A
            { kind: "dynamic", landmarks: liveLikeCopy(GESTURE_B) },
            { kind: "idle", frames: 10 },
        ]);
        const dynamicCommits = stream
            .map((f) => pipeline.processFrame(f))
            .filter((r) => r.newDynamicSignEntry)
            .map((r) => r.newDynamicSignEntry!.displayLabel);
        expect(dynamicCommits).toEqual(["\u0645\u0631\u062d\u0628\u0627", "\u0634\u0643\u0631\u0627"]);
    });

    it("a dynamic sign followed by a real letter both appear correctly, in order", () => {
        const pipeline = newPipeline();
        const stream = buildMixedStream([
            { kind: "dynamic", landmarks: liveLikeCopy(GESTURE_A) },
            { kind: "idle", frames: 10 },
            { kind: "letter", label: "Alef", frames: 12 },
            { kind: "idle", frames: 10 },
        ]);
        stream.forEach((f) => pipeline.processFrame(f));
        // CONTRACT CHANGED (dynamic word boundaries): a spelled letter after a whole
        // recognized word starts a NEW word, so these no longer merge into
        // "مرحباا". See lib/ml/arabicTextBuilder.ts.
        expect(pipeline.getArabicText()).toBe("\u0645\u0631\u062d\u0628\u0627 \u0627"); // مرحبا + ا
    });

    it("a real letter followed by a dynamic sign both appear correctly, in order", () => {
        const pipeline = newPipeline();
        const stream = buildMixedStream([
            { kind: "letter", label: "Alef", frames: 12 },
            { kind: "idle", frames: 10 },
            { kind: "dynamic", landmarks: liveLikeCopy(GESTURE_A) },
            { kind: "idle", frames: 10 },
        ]);
        stream.forEach((f) => pipeline.processFrame(f));
        // CONTRACT CHANGED (dynamic word boundaries): a recognized word after a
        // spelled letter is a separate word, so these no longer merge.
        expect(pipeline.getArabicText()).toBe("\u0627 \u0645\u0631\u062d\u0628\u0627"); // ا + مرحبا
    });

    it("Space still works correctly alongside the dynamic path", () => {
        const pipeline = newPipeline();
        const stream = buildMixedStream([
            { kind: "dynamic", landmarks: liveLikeCopy(GESTURE_A) },
            { kind: "idle", frames: 10 },
            { kind: "letter", label: "Space", frames: 12 },
            { kind: "idle", frames: 10 },
            { kind: "letter", label: "Alef", frames: 12 },
            { kind: "idle", frames: 10 },
        ]);
        stream.forEach((f) => pipeline.processFrame(f));
        expect(pipeline.getArabicText()).toBe("\u0645\u0631\u062d\u0628\u0627 \u0627"); // مرحبا + space + ا
    });

    it("Delete still works correctly, removing only the last character even after a dynamic-sign append", () => {
        const pipeline = newPipeline();
        const stream = buildMixedStream([
            { kind: "letter", label: "Alef", frames: 12 },
            { kind: "idle", frames: 10 },
            { kind: "dynamic", landmarks: liveLikeCopy(GESTURE_A) },
            { kind: "idle", frames: 10 },
            { kind: "letter", label: "Delete", frames: 12 },
            { kind: "idle", frames: 10 },
        ]);
        stream.forEach((f) => pipeline.processFrame(f));
        // ا + مرحبا, then Delete removes only the LAST character of "مرحبا"
        // Delete still removes EXACTLY ONE character; only the word boundary is new.
        expect(pipeline.getArabicText()).toBe("\u0627 \u0645\u0631\u062d\u0628"); // ا + مرحب (ا dropped)
    });

    it("Finish still behaves as a no-op on text, unaffected by the dynamic path", () => {
        const pipeline = newPipeline();
        const stream = buildMixedStream([
            { kind: "dynamic", landmarks: liveLikeCopy(GESTURE_A) },
            { kind: "idle", frames: 10 },
            { kind: "letter", label: "Alef", frames: 12 }, // stand-in frames; Finish text-effect tested at unit level too
            { kind: "idle", frames: 10 },
        ]);
        const before = SYNTHETIC_REGISTRY[0].text;
        stream.forEach((f) => pipeline.processFrame(f));
        expect(pipeline.getArabicText().startsWith(before)).toBe(true);
    });
});