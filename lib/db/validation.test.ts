import { describe, it, expect } from "vitest";
import { validateCollectedSample, validateTranslationSession } from "./validation";

function fakeFrame() {
  return Array.from({ length: 21 }, (_, i) => ({ x: 0.1 * i, y: 0.2 * i, z: 0 }));
}

describe("validateCollectedSample", () => {
  it("accepts a well-formed sample", () => {
    const result = validateCollectedSample({
      label: "Alef",
      languageCode: "ArSL",
      signerSessionId: "session-1",
      landmarkSequence: [fakeFrame(), fakeFrame(), fakeFrame()],
    });
    expect(result.valid).toBe(true);
    expect(result.data?.frameCount).toBe(3);
  });

  it("rejects a non-object body", () => {
    expect(validateCollectedSample("not an object").valid).toBe(false);
    expect(validateCollectedSample(null).valid).toBe(false);
  });

  it("rejects a missing label", () => {
    const result = validateCollectedSample({
      languageCode: "ArSL",
      signerSessionId: "s1",
      landmarkSequence: [fakeFrame(), fakeFrame(), fakeFrame()],
    });
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("label"))).toBe(true);
  });

  it("rejects a sequence that is too short", () => {
    const result = validateCollectedSample({
      label: "Alef",
      languageCode: "ArSL",
      signerSessionId: "s1",
      landmarkSequence: [fakeFrame()],
    });
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("at least"))).toBe(true);
  });

  it("rejects a frame without exactly 21 landmarks", () => {
    const result = validateCollectedSample({
      label: "Alef",
      languageCode: "ArSL",
      signerSessionId: "s1",
      landmarkSequence: [fakeFrame().slice(0, 10), fakeFrame(), fakeFrame()],
    });
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("21 landmarks"))).toBe(true);
  });

  it("rejects a sequence exceeding the max frame count", () => {
    const longSeq = Array.from({ length: 500 }, () => fakeFrame());
    const result = validateCollectedSample({
      label: "Alef",
      languageCode: "ArSL",
      signerSessionId: "s1",
      landmarkSequence: longSeq,
    });
    expect(result.valid).toBe(false);
    expect(result.errors.some((e) => e.includes("at most"))).toBe(true);
  });
});

describe("validateTranslationSession", () => {
  it("accepts a well-formed session summary", () => {
    const result = validateTranslationSession({
      sessionId: "abc123",
      mode: "sign",
      languageCode: "ArSL",
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      committedSignCount: 5,
      transcriptCharCount: 20,
    });
    expect(result.valid).toBe(true);
  });

  it("rejects an invalid mode", () => {
    const result = validateTranslationSession({
      sessionId: "abc123",
      mode: "video-upload", // not a real mode — this app never uploads raw media
      languageCode: "ArSL",
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      committedSignCount: 5,
      transcriptCharCount: 20,
    });
    expect(result.valid).toBe(false);
  });

  it("rejects a negative committedSignCount", () => {
    const result = validateTranslationSession({
      sessionId: "abc123",
      mode: "sign",
      languageCode: "ArSL",
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      committedSignCount: -1,
      transcriptCharCount: 20,
    });
    expect(result.valid).toBe(false);
  });

  it("rejects malformed date strings", () => {
    const result = validateTranslationSession({
      sessionId: "abc123",
      mode: "sign",
      languageCode: "ArSL",
      startedAt: "not-a-date",
      endedAt: new Date().toISOString(),
      committedSignCount: 5,
      transcriptCharCount: 20,
    });
    expect(result.valid).toBe(false);
  });

  it("never accepts raw media fields — schema has no video/audio field to validate", () => {
    // This test documents intent: there is no rawVideo/rawAudio field in the
    // TranslationSessionDocument type at all, so there's nothing to strip —
    // the type system itself prevents raw media from ever being stored here.
    const result = validateTranslationSession({
      sessionId: "abc123",
      mode: "sign",
      languageCode: "ArSL",
      startedAt: new Date().toISOString(),
      endedAt: new Date().toISOString(),
      committedSignCount: 5,
      transcriptCharCount: 20,
      rawVideoBase64: "should be ignored, not a real field",
    });
    expect(result.valid).toBe(true);
    expect(result.data).not.toHaveProperty("rawVideoBase64");
  });
});
