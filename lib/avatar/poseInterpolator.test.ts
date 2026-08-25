import { describe, it, expect } from "vitest";
import { interpolatePose } from "./poseInterpolator";
import type { HandLandmarks } from "@/types/landmarks";

function makeFrame(value: number): HandLandmarks {
    return Array.from({ length: 21 }, () => ({ x: value, y: value, z: value }));
}

describe("interpolatePose", () => {
    it("returns null for an empty frame list", () => {
        expect(interpolatePose([], 0.5)).toBeNull();
    });

    it("holds a single static frame regardless of progress -- no fabricated motion", () => {
        const frame = makeFrame(0.5);
        expect(interpolatePose([frame], 0)).toEqual(frame);
        expect(interpolatePose([frame], 0.5)).toEqual(frame);
        expect(interpolatePose([frame], 1)).toEqual(frame);
    });

    it("returns the first frame exactly at progress 0", () => {
        const frames = [makeFrame(0), makeFrame(1)];
        const result = interpolatePose(frames, 0);
        expect(result![0].x).toBeCloseTo(0);
    });

    it("returns the last frame exactly at progress 1", () => {
        const frames = [makeFrame(0), makeFrame(1)];
        const result = interpolatePose(frames, 1);
        expect(result![0].x).toBeCloseTo(1);
    });

    it("linearly interpolates between two frames at progress 0.5", () => {
        const frames = [makeFrame(0), makeFrame(10)];
        const result = interpolatePose(frames, 0.5);
        expect(result![0].x).toBeCloseTo(5);
    });

    it("interpolates correctly across more than two frames", () => {
        const frames = [makeFrame(0), makeFrame(10), makeFrame(20)];
        // progress 0.25 of the whole sequence -> halfway between frame 0 and frame 1
        const result = interpolatePose(frames, 0.25);
        expect(result![0].x).toBeCloseTo(5);
    });

    it("clamps out-of-range progress instead of crashing", () => {
        const frames = [makeFrame(0), makeFrame(10)];
        expect(interpolatePose(frames, -1)![0].x).toBeCloseTo(0);
        expect(interpolatePose(frames, 2)![0].x).toBeCloseTo(10);
    });
});