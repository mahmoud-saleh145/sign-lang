import { describe, it, expect } from "vitest";
import { dtwDistance, resampleSequence } from "./dtw";

describe("dtwDistance", () => {
    it("returns zero cost for two identical sequences", () => {
        const seq = [[0, 0], [1, 1], [2, 2]];
        const result = dtwDistance(seq, seq);
        expect(result.totalCost).toBeCloseTo(0, 6);
        expect(result.normalizedCost).toBeCloseTo(0, 6);
    });

    it("returns low cost for a time-shifted/warped version of the same shape", () => {
        const seqA = [[0, 0], [1, 1], [2, 2], [3, 3]];
        // Same shape, but frame 1 is "held" an extra step -- a warp, not a
        // different gesture. DTW should absorb this via the alignment path.
        const seqB = [[0, 0], [1, 1], [1, 1], [2, 2], [3, 3]];
        const result = dtwDistance(seqA, seqB);
        expect(result.normalizedCost).toBeCloseTo(0, 6);
    });

    it("returns high cost for genuinely different sequences", () => {
        const seqA = [[0, 0], [1, 1], [2, 2]];
        const seqB = [[10, 10], [11, 11], [12, 12]];
        const result = dtwDistance(seqA, seqB);
        expect(result.normalizedCost).toBeGreaterThan(5);
    });

    it("handles sequences of different lengths", () => {
        const short = [[0, 0], [5, 5]];
        const long = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]];
        const result = dtwDistance(short, long);
        expect(result.pathLength).toBeGreaterThanOrEqual(long.length);
        expect(Number.isFinite(result.totalCost)).toBe(true);
    });

    it("throws on empty sequences rather than silently returning a meaningless result", () => {
        expect(() => dtwDistance([], [[0, 0]])).toThrow();
        expect(() => dtwDistance([[0, 0]], [])).toThrow();
    });

    it("distance is symmetric", () => {
        const seqA = [[0, 0], [1, 2], [3, 1]];
        const seqB = [[0, 1], [2, 2], [3, 0]];
        const ab = dtwDistance(seqA, seqB);
        const ba = dtwDistance(seqB, seqA);
        expect(ab.totalCost).toBeCloseTo(ba.totalCost, 6);
    });
});

describe("resampleSequence", () => {
    it("returns the same sequence unchanged if already the target length", () => {
        const seq = [[0, 0], [1, 1], [2, 2]];
        expect(resampleSequence(seq, 3)).toEqual(seq);
    });

    it("downsamples a longer sequence while preserving endpoints", () => {
        const seq = [[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]];
        const result = resampleSequence(seq, 3);
        expect(result).toHaveLength(3);
        expect(result[0]).toEqual([0, 0]);
        expect(result[2]).toEqual([4, 4]);
    });

    it("upsamples a shorter sequence while preserving endpoints", () => {
        const seq = [[0, 0], [10, 10]];
        const result = resampleSequence(seq, 5);
        expect(result).toHaveLength(5);
        expect(result[0]).toEqual([0, 0]);
        expect(result[4]).toEqual([10, 10]);
        expect(result[2][0]).toBeCloseTo(5, 1); // midpoint, linearly interpolated
    });

    it("throws on an empty sequence", () => {
        expect(() => resampleSequence([], 5)).toThrow();
    });
});