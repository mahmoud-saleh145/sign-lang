import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { ArSLClassifier } from "./model";
import type { ModelWeights } from "./model";
import { SignRecognitionPipeline } from "./recognitionPipeline";
import { DEFAULT_SEGMENTATION_CONFIG } from "./segmentation";
import { DEFAULT_GESTURE_BUFFER_CONFIG } from "./dynamicGestureBuffer";
import { extractFeatures } from "./features";
import type { FrameDetection, HandLandmarks } from "@/types/landmarks";

const weights = JSON.parse(
  readFileSync(path.resolve(__dirname, "../../public/models/arsl-31/weights.json"), "utf-8")
) as ModelWeights;

const streamFixture = JSON.parse(
  readFileSync(
    path.resolve(__dirname, "../../ml/tests/fixtures/continuous_stream_fixture.json"),
    "utf-8"
  )
) as Record<"Alef" | "Ba2" | "Space" | "7a2" | "Kaf" | "Noon" | "Delete", HandLandmarks[]>;

/** Build a FrameDetection sequence: `null` = no hand (idle), otherwise cycles
 * through the given class's real landmark samples. */
function buildStream(
  spec: Array<{ label: keyof typeof streamFixture | null; frames: number }>
): FrameDetection[] {
  const stream: FrameDetection[] = [];
  let t = 0;
  for (const segment of spec) {
    for (let i = 0; i < segment.frames; i++) {
      const hands =
        segment.label === null
          ? []
          : [
            {
              landmarks: streamFixture[segment.label][i % streamFixture[segment.label].length],
              handedness: "Right" as const,
              score: 0.95,
            },
          ];
      stream.push({ timestampMs: t, hands });
      t += 33; // ~30fps
    }
  }
  return stream;
}

describe("SignRecognitionPipeline — continuous stream, real trained model", () => {
  let classifier: ArSLClassifier;

  beforeAll(() => {
    classifier = new ArSLClassifier(weights);
    // Sanity: the real model must actually classify real Alef/Ba2 landmarks
    // as Alef/Ba2 for this test to be meaningful (not just testing plumbing).
    const alefFeatureCheck = classifier.predict(extractFeatures(streamFixture.Alef[0]));
    expect(alefFeatureCheck.label).toBe("Alef");
  });

  it("1. does not commit the same sign on every frame", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([{ label: "Alef", frames: 30 }]);
    const commits = stream.map((f) => pipeline.processFrame(f)).filter((r) => r.newTranscriptEntry);
    // 30 frames of a held sign must NOT produce 30 commits.
    expect(commits.length).toBeLessThan(5);
  });

  it("2. idle periods (no hand detected — dataset has no 'nothing' class) never commit", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([{ label: null, frames: 40 }]);
    const results = stream.map((f) => pipeline.processFrame(f));
    expect(results.every((r) => r.newTranscriptEntry === null)).toBe(true);
    expect(results.every((r) => r.telemetry.handDetected === false)).toBe(true);
  });

  it("3. a sign commits only after temporal stability, not instantly", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([{ label: "Alef", frames: 3 }]); // too short to stabilize
    const results = stream.map((f) => pipeline.processFrame(f));
    expect(results.every((r) => r.newTranscriptEntry === null)).toBe(true);
  });

  it("4. a short pause mid-sign does not create a duplicate commit", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([
      { label: "Alef", frames: 12 },
      { label: null, frames: 3 }, // brief pause, shorter than maxGapFrames
      { label: "Alef", frames: 6 },
      { label: null, frames: 10 }, // real end
    ]);
    const results = stream.map((f) => pipeline.processFrame(f));
    const commits = results.filter((r) => r.newTranscriptEntry).map((r) => r.newTranscriptEntry!.sign);
    expect(commits).toEqual(["Alef"]);
  });

  it("5. a genuinely repeated sign is recognized twice", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const oneSign = [
      { label: "Alef" as const, frames: 12 },
      { label: null, frames: 10 },
    ];
    const stream = buildStream([...oneSign, ...oneSign]);
    const results = stream.map((f) => pipeline.processFrame(f));
    const commits = results.filter((r) => r.newTranscriptEntry).map((r) => r.newTranscriptEntry!.sign);
    expect(commits).toEqual(["Alef", "Alef"]);
  });

  it("6. multiple distinct signs are recognized continuously with no manual stops between them", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([
      { label: "Alef", frames: 12 },
      { label: null, frames: 8 },
      { label: "Ba2", frames: 12 },
      { label: null, frames: 8 },
    ]);
    // Simulating one uninterrupted stream — no pipeline.reset() or button
    // press occurs anywhere between the two signs.
    const results = stream.map((f) => pipeline.processFrame(f));
    const commits = results.filter((r) => r.newTranscriptEntry).map((r) => r.newTranscriptEntry!.sign);
    expect(commits).toEqual(["Alef", "Ba2"]);
  });

  it("7. transcript reflects only committed signs, never raw per-frame predictions", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([{ label: "Alef", frames: 12 }, { label: null, frames: 8 }]);
    stream.forEach((f) => pipeline.processFrame(f));
    const transcript = pipeline.getTranscript();
    expect(transcript).toHaveLength(1);
    expect(transcript[0].sign).toBe("Alef");
    // 12 raw Alef-predicting frames occurred, but exactly 1 transcript entry.
  });

  it("8. every frame carries full debug telemetry", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([{ label: "Alef", frames: 5 }]);
    const results = stream.map((f) => pipeline.processFrame(f));
    for (const r of results) {
      expect(r.telemetry).toHaveProperty("rawLabel");
      expect(r.telemetry).toHaveProperty("rawConfidence");
      expect(r.telemetry).toHaveProperty("segmentationState");
      expect(r.telemetry).toHaveProperty("committedThisFrame");
      expect(r.telemetry).toHaveProperty("transcriptLength");
      expect(r.telemetry).toHaveProperty("arabicText");
    }
  });

  it("9. a real committed sign appends the correct Arabic character to accumulated text", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([{ label: "Alef", frames: 12 }, { label: null, frames: 10 }]);
    stream.forEach((f) => pipeline.processFrame(f));
    expect(pipeline.getArabicText()).toBe("\u0627"); // ا
  });

  it("10. two real distinct signs form a two-letter word by concatenation, not a dictionary lookup", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([
      { label: "Alef", frames: 12 },
      { label: null, frames: 8 },
      { label: "Ba2", frames: 12 },
      { label: null, frames: 8 },
    ]);
    stream.forEach((f) => pipeline.processFrame(f));
    expect(pipeline.getArabicText()).toBe("\u0627\u0628"); // اب
  });

  it("11. holding one real sign for many raw frames does not append repeated characters — only genuine commits do", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    // 60 raw frames all predicting "Alef" (well beyond minStableFrames),
    // followed by idle to force exactly one commit.
    const stream = buildStream([{ label: "Alef", frames: 60 }, { label: null, frames: 10 }]);
    const results = stream.map((f) => pipeline.processFrame(f));

    const rawAlefFrames = results.filter((r) => r.telemetry.rawLabel === "Alef").length;
    expect(rawAlefFrames).toBeGreaterThan(30); // plenty of raw per-frame predictions

    // But the Arabic text must contain exactly one ا, not dozens.
    expect(pipeline.getArabicText()).toBe("\u0627");
    const alefCount = [...pipeline.getArabicText()].filter((c) => c === "\u0627").length;
    expect(alefCount).toBe(1);
  });

  it("12. clearTranscript() also resets the accumulated Arabic text", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([{ label: "Alef", frames: 12 }, { label: null, frames: 8 }]);
    stream.forEach((f) => pipeline.processFrame(f));
    expect(pipeline.getArabicText()).not.toBe("");
    pipeline.clearTranscript();
    expect(pipeline.getArabicText()).toBe("");
    expect(pipeline.getTranscript()).toHaveLength(0);
  });

  it("13. transcript entries keep the original English model class name in `sign`, and the Arabic form only in `displayLabel`", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([{ label: "Alef", frames: 12 }, { label: null, frames: 8 }]);
    stream.forEach((f) => pipeline.processFrame(f));
    const [entry] = pipeline.getTranscript();
    expect(entry.sign).toBe("Alef"); // internal model class name, unchanged
    expect(entry.displayLabel).toBe("\u0627"); // ا, user-facing only
  });
});

describe("SignRecognitionPipeline — word spacing (real Space class, real model)", () => {
  let classifier: ArSLClassifier;

  beforeAll(() => {
    classifier = new ArSLClassifier(weights);
    // Sanity: the real model must actually classify real Space landmarks as
    // "Space" for these tests to be meaningful.
    const check = classifier.predict(extractFeatures(streamFixture.Space[0]));
    expect(check.label).toBe("Space");
  });

  it("letters accumulate into a word (real model, with idle gaps between signs)", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([
      { label: "Ba2", frames: 12 }, { label: null, frames: 8 },
      { label: "7a2", frames: 12 }, { label: null, frames: 8 },
      { label: "Ba2", frames: 12 }, { label: null, frames: 8 },
      { label: "Kaf", frames: 12 }, { label: null, frames: 8 },
    ]);
    stream.forEach((f) => pipeline.processFrame(f));
    expect(pipeline.getArabicText()).toBe("\u0628\u062d\u0628\u0643"); // بحبك
  });

  it("word -> Space -> word produces a real space between them (with idle gaps)", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([
      { label: "Ba2", frames: 12 }, { label: null, frames: 8 },
      { label: "Kaf", frames: 12 }, { label: null, frames: 8 },
      { label: "Space", frames: 12 }, { label: null, frames: 8 },
      { label: "Alef", frames: 12 }, { label: null, frames: 8 },
    ]);
    stream.forEach((f) => pipeline.processFrame(f));
    expect(pipeline.getArabicText()).toBe("\u0628\u0643 \u0627"); // بك ا
  });

  it("REGRESSION: word -> Space -> word with ZERO idle gap between signs (direct continuous transitions) still produces a real space — this was the reported bug", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    // Exact literal example from the bug report: ب ح ب ك Space ا ن ا -> بحبك انا
    // No idle/null frames anywhere between signs — this is what continuous
    // signing without pausing actually looks like.
    const stream = buildStream([
      { label: "Ba2", frames: 12 },
      { label: "7a2", frames: 12 },
      { label: "Ba2", frames: 12 },
      { label: "Kaf", frames: 12 },
      { label: "Space", frames: 12 },
      { label: "Alef", frames: 12 },
      { label: "Noon", frames: 12 },
      { label: "Alef", frames: 12 },
      { label: null, frames: 10 }, // natural trailing pause so the last sign commits
    ]);
    const commits = stream
      .map((f) => pipeline.processFrame(f))
      .filter((r) => r.newTranscriptEntry)
      .map((r) => r.newTranscriptEntry!.sign);

    expect(commits).toEqual(["Ba2", "7a2", "Ba2", "Kaf", "Space", "Alef", "Noon", "Alef"]);
    expect(pipeline.getArabicText()).toBe("\u0628\u062d\u0628\u0643 \u0627\u0646\u0627"); // بحبك انا
  });

  it("Space held/predicted across many raw frames commits exactly once — no duplicate spaces", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([
      { label: "Alef", frames: 12 }, { label: null, frames: 8 },
      { label: "Space", frames: 60 }, // sustained well beyond minStableFrames
      { label: null, frames: 10 },
    ]);
    const results = stream.map((f) => pipeline.processFrame(f));

    const rawSpaceFrames = results.filter((r) => r.telemetry.rawLabel === "Space").length;
    expect(rawSpaceFrames).toBeGreaterThan(30); // plenty of raw per-frame Space predictions

    const spaceCommits = results.filter((r) => r.newTranscriptEntry?.sign === "Space").length;
    expect(spaceCommits).toBe(1); // but only ONE genuine commit
    expect(pipeline.getArabicText()).toBe("\u0627 "); // ا + exactly one space
  });

  it("Space performed twice for real (with a genuine gap) produces two spaces", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([
      { label: "Alef", frames: 12 }, { label: null, frames: 8 },
      { label: "Space", frames: 12 }, { label: null, frames: 8 },
      { label: "Space", frames: 12 }, { label: null, frames: 8 },
      { label: "Ba2", frames: 12 }, { label: null, frames: 8 },
    ]);
    stream.forEach((f) => pipeline.processFrame(f));
    expect(pipeline.getArabicText()).toBe("\u0627  \u0628"); // ا + two spaces + ب
  });

  it("Delete correctly removes a trailing space, revealing the previous word", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([
      { label: "Alef", frames: 12 }, { label: null, frames: 8 },
      { label: "Space", frames: 12 }, { label: null, frames: 8 },
      { label: "Delete", frames: 12 }, { label: null, frames: 8 },
    ]);
    stream.forEach((f) => pipeline.processFrame(f));
    expect(pipeline.getArabicText()).toBe("\u0627"); // trailing space removed, ا remains
  });

  it("multiple words in one continuous session, no manual stops or resets anywhere", () => {
    const pipeline = new SignRecognitionPipeline(classifier);
    const stream = buildStream([
      { label: "Alef", frames: 12 },
      { label: "Noon", frames: 12 },
      { label: "Alef", frames: 12 },
      { label: "Space", frames: 12 },
      { label: "Ba2", frames: 12 },
      { label: "Alef", frames: 12 },
      { label: null, frames: 10 },
    ]);
    // One uninterrupted stream: no pipeline.reset(), no clearTranscript(),
    // no button press simulated anywhere between signs or words.
    stream.forEach((f) => pipeline.processFrame(f));
    expect(pipeline.getArabicText()).toBe("\u0627\u0646\u0627 \u0628\u0627"); // انا با
  });
});
/**
 * WORD-MODE OBSERVABILITY.
 *
 * These reproduce the exact field pattern reported from real-world testing:
 * a completed 45-frame gesture sitting beside "classifier running no",
 * "candidate count -", "FINAL DECISION -" and "captured frames L/R 0 / 0".
 *
 * None of that meant the classifier had not run. Every lastDynamic* field was
 * read from a local that is non-null only on the single frame a gesture
 * finalizes, so one frame later they all collapsed -- and "captured frames
 * L/R 0 / 0" was a hardcoded default, never a measurement. The one field with
 * a persisted fallback (side source) kept its value, which is what made the
 * panel look self-contradictory.
 *
 * The tests below pin the LATCHING, not the recognition. No threshold, gate
 * or matcher behaviour is asserted here.
 */
describe("SignRecognitionPipeline — word-mode classification telemetry", () => {
  let classifier: ArSLClassifier;
  beforeAll(() => {
    classifier = new ArSLClassifier(weights);
  });

  /** Runs a gesture long enough to finalize, then idle frames after it. */
  function runGestureThenIdle(gestureFrames: number, idleFrames: number) {
    const pipeline = new SignRecognitionPipeline(classifier, DEFAULT_SEGMENTATION_CONFIG, DEFAULT_GESTURE_BUFFER_CONFIG, "word");
    const stream = buildStream([
      { label: "Alef", frames: gestureFrames },
      { label: null, frames: idleFrames },
    ]);
    return stream.map((f) => pipeline.processFrame(f).telemetry);
  }

  it("keeps the classification visible after the gesture ends (two-tick)", () => {
    const ticks = runGestureThenIdle(45, 20);
    const at = ticks.findIndex((t) => t.framesSinceLastClassification === 0);
    expect(at, "no frame ever reported a fresh classification").toBeGreaterThan(-1);

    const produced = ticks[at];
    const later = ticks[ticks.length - 1];

    // THE REGRESSION: every one of these used to be null/"-"/0 one frame on.
    expect(later.dynamicOutcome).toBe(produced.dynamicOutcome);
    expect(later.lastDynamicCandidateCount).toBe(produced.lastDynamicCandidateCount);
    expect(later.lastDynamicBestSignId).toBe(produced.lastDynamicBestSignId);
    expect(later.lastDynamicBestDistance).toBe(produced.lastDynamicBestDistance);
    expect(later.lastDynamicSecondBestDistance).toBe(produced.lastDynamicSecondBestDistance);
    expect(later.lastDynamicBestToSecondBestRatio).toBe(produced.lastDynamicBestToSecondBestRatio);
    expect(later.lastDynamicAccepted).toBe(produced.lastDynamicAccepted);
    expect(later.lastDynamicRejectionReason).toBe(produced.lastDynamicRejectionReason);
    expect(later.lastDynamicQuerySideFrames).toEqual(produced.lastDynamicQuerySideFrames);
  });

  it("never reports 0/0 captured frames for a gesture that was classified", () => {
    // "captured frames L/R 0 / 0" was the `?? { left: 0, right: 0 }` default,
    // indistinguishable from a real measurement of zero.
    const ticks = runGestureThenIdle(45, 20);
    const classified = ticks.filter((t) => t.framesSinceLastClassification === 0);
    for (const t of classified) {
      const { left, right } = t.lastDynamicQuerySideFrames;
      expect(left + right, "a classified gesture reported no captured frames").toBeGreaterThan(0);
    }
  });

  it("ages the displayed result so a stale reading is visibly stale", () => {
    const ticks = runGestureThenIdle(45, 20);
    const at = ticks.findIndex((t) => t.framesSinceLastClassification === 0);
    expect(ticks[at + 1]?.framesSinceLastClassification).toBe(1);
    expect(ticks[at + 2]?.framesSinceLastClassification).toBe(2);
    // ...and it is null before anything has been classified.
    expect(ticks[0].framesSinceLastClassification).toBeNull();
  });

  it("reports not-yet-run before any gesture finalizes, and never after one does", () => {
    const ticks = runGestureThenIdle(45, 20);
    expect(ticks[0].dynamicOutcome).toBe("not-yet-run");
    const at = ticks.findIndex((t) => t.framesSinceLastClassification === 0);
    for (const t of ticks.slice(at)) {
      expect(t.dynamicOutcome).not.toBe("not-yet-run");
    }
  });

  it("an outcome always accompanies a non-running classifier", () => {
    // The reported symptom: "classifier running no" with nothing to explain it.
    const ticks = runGestureThenIdle(45, 20);
    for (const t of ticks.slice(-5)) {
      expect(t.dynamicPhase).not.toBe("classifying");
      expect(t.dynamicOutcome).toBeTruthy();
      expect(t.dynamicOutcome).not.toBe("not-yet-run");
    }
  });

  it("a gesture too short to classify is reported as discarded, not as a rejection", () => {
    const ticks = runGestureThenIdle(3, 15);
    const final = ticks[ticks.length - 1];
    expect(final.dynamicOutcome).toBe("discarded-by-segmentation");
    expect(final.gestureDiscardReason).toBe("too-few-frames");
    // The classifier genuinely never ran, so there is nothing to display.
    expect(final.framesSinceLastClassification).toBeNull();
    expect(final.lastDynamicCandidateCount).toBeNull();
  });

  it("a newly discarded gesture does not leave the previous verdict on screen", () => {
    // Otherwise the panel describes two different gestures at once.
    const pipeline = new SignRecognitionPipeline(classifier, DEFAULT_SEGMENTATION_CONFIG, DEFAULT_GESTURE_BUFFER_CONFIG, "word");
    const stream = buildStream([
      { label: "Alef", frames: 45 },
      { label: null, frames: 15 },
      { label: "Alef", frames: 3 },
      { label: null, frames: 15 },
    ]);
    const ticks = stream.map((f) => pipeline.processFrame(f).telemetry);
    const final = ticks[ticks.length - 1];
    expect(final.dynamicOutcome).toBe("discarded-by-segmentation");
    expect(final.lastDynamicBestSignId).toBeNull();
    expect(final.framesSinceLastClassification).toBeNull();
  });

  it("separates per-gesture tracker swaps from the session total", () => {
    // 299 swaps beside a 45-frame gesture looked like a defect; it was the
    // session-cumulative counter with no label saying so.
    const ticks = runGestureThenIdle(45, 20);
    const final = ticks[ticks.length - 1];
    expect(final.lastGestureTrackerSwitches).not.toBeNull();
    expect(final.lastGestureTrackerSwitches!).toBeLessThanOrEqual(final.lastDynamicTrackerSwitches!);
  });

  it("word mode leaves the letter transcript untouched", () => {
    // Guards the "recognition behaviour unchanged" claim from the side the
    // telemetry work could plausibly have disturbed.
    const pipeline = new SignRecognitionPipeline(classifier, DEFAULT_SEGMENTATION_CONFIG, DEFAULT_GESTURE_BUFFER_CONFIG, "word");
    const stream = buildStream([{ label: "Alef", frames: 45 }, { label: null, frames: 15 }]);
    const results = stream.map((f) => pipeline.processFrame(f));
    expect(results.every((r) => r.newTranscriptEntry === null)).toBe(true);
    expect(pipeline.getArabicText()).toBe("");
  });
});
