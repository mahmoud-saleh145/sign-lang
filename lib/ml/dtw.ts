/**
 * Generic Dynamic Time Warping (DTW) for sequences of equal-dimension
 * numeric vectors. No sign-language-specific logic lives here -- this is a
 * pure algorithm module, reused by lib/ml/dynamicSignRecognizer.ts.
 *
 * Classic O(n*m) DTW with Euclidean per-frame distance. Returns the total
 * warping cost AND the cost normalized by path length (roughly comparable
 * across sequence-pairs of different lengths), since raw DTW cost grows
 * with sequence length and isn't meaningfully comparable on its own.
 */

export interface DtwResult {
    /** Raw accumulated warping cost along the optimal alignment path. */
    totalCost: number;
    /** totalCost divided by the alignment path length -- roughly length-invariant. */
    normalizedCost: number;
    /** Number of steps in the optimal alignment path (>= max(n, m)). */
    pathLength: number;
}

function euclideanDistance(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return Math.sqrt(sum);
}

/**
 * Computes DTW distance between two sequences of numeric vectors.
 * Both sequences must contain vectors of the same dimensionality (not
 * necessarily the same length as each other).
 */
export function dtwDistance(seqA: number[][], seqB: number[][]): DtwResult {
    const n = seqA.length;
    const m = seqB.length;

    if (n === 0 || m === 0) {
        throw new Error("dtwDistance: both sequences must be non-empty");
    }

    // cost[i][j] = min accumulated cost aligning seqA[0..i) with seqB[0..j)
    const cost: number[][] = Array.from({ length: n + 1 }, () =>
        new Array<number>(m + 1).fill(Infinity)
    );
    // pathLen[i][j] tracks how many steps the optimal path to (i,j) took, so
    // we can report a length-normalized cost at the end.
    const pathLen: number[][] = Array.from({ length: n + 1 }, () =>
        new Array<number>(m + 1).fill(0)
    );
    cost[0][0] = 0;

    for (let i = 1; i <= n; i++) {
        for (let j = 1; j <= m; j++) {
            const d = euclideanDistance(seqA[i - 1], seqB[j - 1]);
            const candidates: Array<[number, number]> = [
                [cost[i - 1][j], pathLen[i - 1][j]], // insertion
                [cost[i][j - 1], pathLen[i][j - 1]], // deletion
                [cost[i - 1][j - 1], pathLen[i - 1][j - 1]], // match
            ];
            let best = candidates[0];
            for (const c of candidates) {
                if (c[0] < best[0]) best = c;
            }
            cost[i][j] = d + best[0];
            pathLen[i][j] = best[1] + 1;
        }
    }

    const totalCost = cost[n][m];
    const pathLength = pathLen[n][m];
    return {
        totalCost,
        normalizedCost: totalCost / pathLength,
        pathLength,
    };
}

/**
 * Resamples a sequence of numeric vectors to exactly `targetLength` frames
 * via linear interpolation. Used to bound DTW cost for very long sequences
 * (see dynamicSignRecognizer.ts) -- this does NOT change the shape of the
 * gesture, just its frame density, so it doesn't fabricate motion.
 */
export function resampleSequence(seq: number[][], targetLength: number): number[][] {
    if (seq.length === 0) {
        throw new Error("resampleSequence: sequence must be non-empty");
    }
    if (seq.length === targetLength) return seq;
    if (targetLength < 2) {
        throw new Error("resampleSequence: targetLength must be >= 2");
    }

    const result: number[][] = [];
    const dim = seq[0].length;
    for (let i = 0; i < targetLength; i++) {
        const t = (i / (targetLength - 1)) * (seq.length - 1);
        const lower = Math.floor(t);
        const upper = Math.min(lower + 1, seq.length - 1);
        const frac = t - lower;
        const frame = new Array<number>(dim);
        for (let d = 0; d < dim; d++) {
            frame[d] = seq[lower][d] + (seq[upper][d] - seq[lower][d]) * frac;
        }
        result.push(frame);
    }
    return result;
}