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

/**
 * Computes DTW distance between two sequences of numeric vectors.
 * Both sequences must contain vectors of the same dimensionality (not
 * necessarily the same length as each other).
 */
/**
 * SCRATCH BUFFERS, reused across calls.
 *
 * This used to allocate `cost` and `pathLen` as fresh arrays-of-arrays on
 * every call: 2 + 2*(n+1) array objects each time. Measured on the real
 * corpus, one held-out query runs 2,002 DTW comparisons and allocates 128,128
 * arrays; at 2,000 prototypes that is ~360,000 per query, and a 994-case
 * benchmark pass allocates hundreds of millions. The arithmetic was never the
 * problem -- the garbage was. Once the heap is under pressure V8 spends its
 * time collecting rather than computing, which is how the same benchmark on
 * the same data went from 45 minutes to 15 hours.
 *
 * Flat typed arrays, indexed i*(m+1)+j, grown on demand and never shrunk. DTW
 * here is synchronous and never re-entrant -- one call completes before the
 * next begins -- so a single shared pair is safe. Float64Array holds exactly
 * the doubles JS numbers hold, so every value and every comparison is
 * identical to the previous implementation.
 */
let costBuffer = new Float64Array(0);
let pathBuffer = new Int32Array(0);

function ensureCapacity(cells: number): void {
    if (costBuffer.length < cells) {
        costBuffer = new Float64Array(cells);
        pathBuffer = new Int32Array(cells);
    }
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

    const width = m + 1;
    ensureCapacity((n + 1) * width);
    const cost = costBuffer;
    const pathLen = pathBuffer;

    // Only the cells this call uses are initialised, so a larger buffer left
    // over from an earlier call cannot leak stale values into this one.
    for (let j = 0; j <= m; j++) {
        cost[j] = Infinity;
        pathLen[j] = 0;
    }
    cost[0] = 0;

    for (let i = 1; i <= n; i++) {
        const row = i * width;
        const prevRow = row - width;
        cost[row] = Infinity;
        pathLen[row] = 0;
        const va = seqA[i - 1];
        const dim = va.length;
        for (let j = 1; j <= m; j++) {
            const vb = seqB[j - 1];
            let sum = 0;
            for (let k = 0; k < dim; k++) {
                const diff = va[k] - vb[k];
                sum += diff * diff;
            }
            const d = Math.sqrt(sum);

            // insertion, deletion, match -- same order and the same strict
            // "<" comparison as before, so ties resolve identically.
            let bestCost = cost[prevRow + j];
            let bestPath = pathLen[prevRow + j];
            const deletion = cost[row + j - 1];
            if (deletion < bestCost) {
                bestCost = deletion;
                bestPath = pathLen[row + j - 1];
            }
            const match = cost[prevRow + j - 1];
            if (match < bestCost) {
                bestCost = match;
                bestPath = pathLen[prevRow + j - 1];
            }
            cost[row + j] = d + bestCost;
            pathLen[row + j] = bestPath + 1;
        }
    }

    const last = n * width + m;
    const totalCost = cost[last];
    const pathLength = pathLen[last];
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