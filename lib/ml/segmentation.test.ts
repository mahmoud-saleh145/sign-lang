import { describe, it, expect } from "vitest";
import { SignSegmenter, DEFAULT_SEGMENTATION_CONFIG } from "./segmentation";
import type { FramePrediction } from "./segmentation";

function run(segmenter: SignSegmenter, frames: FramePrediction[]) {
  return frames.map((f) => segmenter.step(f));
}

function idle(n: number): FramePrediction[] {
  return Array.from({ length: n }, () => ({ label: null, confidence: 0 }));
}

function hold(label: string, confidence: number, n: number): FramePrediction[] {
  return Array.from({ length: n }, () => ({ label, confidence }));
}

describe("SignSegmenter", () => {
  it("does not commit a flickering single-frame prediction", () => {
    const seg = new SignSegmenter();
    const frames: FramePrediction[] = [
      { label: "HELLO", confidence: 0.9 },
      { label: null, confidence: 0 },
      { label: "HELLO", confidence: 0.9 },
      { label: null, confidence: 0 },
    ];
    const results = run(seg, frames);
    expect(results.every((r) => r.committedLabel === null)).toBe(true);
  });

  it("commits exactly once for a single clean sign hold", () => {
    const seg = new SignSegmenter();
    const frames = [
      ...hold("HELLO", 0.92, DEFAULT_SEGMENTATION_CONFIG.minStableFrames + 5),
      ...idle(DEFAULT_SEGMENTATION_CONFIG.maxGapFrames + 2),
    ];
    const results = run(seg, frames);
    const commits = results.filter((r) => r.committedLabel !== null);
    expect(commits).toHaveLength(1);
    expect(commits[0].committedLabel).toBe("HELLO");
  });

  it("tolerates a brief pause mid-sign without ending it (spec: handle brief pauses)", () => {
    const seg = new SignSegmenter();
    const briefGap = DEFAULT_SEGMENTATION_CONFIG.maxGapFrames - 1;
    const frames = [
      ...hold("SCHOOL", 0.9, DEFAULT_SEGMENTATION_CONFIG.minStableFrames + 3),
      ...idle(briefGap), // shorter than maxGapFrames: should NOT end the sign
      ...hold("SCHOOL", 0.9, 5),
      ...idle(DEFAULT_SEGMENTATION_CONFIG.maxGapFrames + 2), // now really end it
    ];
    const results = run(seg, frames);
    const commits = results.filter((r) => r.committedLabel !== null);
    // Exactly one commit: the brief pause was absorbed, not treated as two signs.
    expect(commits).toHaveLength(1);
    expect(commits[0].committedLabel).toBe("SCHOOL");
  });

  it("allows a genuinely repeated sign to appear twice (spec section 18)", () => {
    const seg = new SignSegmenter();
    const oneSign = [
      ...hold("HELLO", 0.9, DEFAULT_SEGMENTATION_CONFIG.minStableFrames + 3),
      ...idle(DEFAULT_SEGMENTATION_CONFIG.maxGapFrames + 2),
    ];
    const frames = [...oneSign, ...oneSign];
    const results = run(seg, frames);
    const commits = results
      .filter((r) => r.committedLabel !== null)
      .map((r) => r.committedLabel);
    expect(commits).toEqual(["HELLO", "HELLO"]);
  });

  it("produces a sequential transcript for multiple distinct signs with no manual stops", () => {
    const seg = new SignSegmenter();
    const seq = [
      ...hold("I", 0.9, 12),
      ...idle(8),
      ...hold("WANT", 0.9, 12),
      ...idle(8),
      ...hold("SCHOOL", 0.9, 12),
      ...idle(8),
    ];
    const results = run(seg, seq);
    const commits = results
      .filter((r) => r.committedLabel !== null)
      .map((r) => r.committedLabel);
    expect(commits).toEqual(["I", "WANT", "SCHOOL"]);
  });

  it("ignores low-confidence predictions entirely (below threshold = idle)", () => {
    const seg = new SignSegmenter();
    const frames = hold("HELLO", 0.3, 30); // below default 0.75 threshold
    const results = run(seg, frames);
    expect(results.every((r) => r.state === "IDLE")).toBe(true);
    expect(results.every((r) => r.committedLabel === null)).toBe(true);
  });

  it("reset() returns the segmenter to a clean IDLE state", () => {
    const seg = new SignSegmenter();
    run(seg, hold("HELLO", 0.9, DEFAULT_SEGMENTATION_CONFIG.minStableFrames + 3));
    seg.reset();
    const result = seg.step({ label: null, confidence: 0 });
    expect(result.state).toBe("IDLE");
    expect(result.currentCandidate).toBeNull();
  });
});
