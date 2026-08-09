import { describe, it, expect, beforeAll } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { ArSLClassifier } from "./model";
import type { ModelWeights } from "./model";
import { SignRecognitionPipeline } from "./recognitionPipeline";
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
) as { Alef: HandLandmarks[]; Ba2: HandLandmarks[] };

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
