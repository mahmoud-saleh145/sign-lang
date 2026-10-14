/**
 * MULTI-PROTOTYPE STORE for KArSL word recognition.
 *
 * ============================================================================
 * WHY THIS EXISTS
 * ============================================================================
 * The registry holds ONE take per sign. KArSL recorded eight. A 502-way
 * nearest-neighbour classifier built on a single example per class is
 * discarding seven eighths of the evidence, and the one example it keeps was
 * picked by a heuristic (`select_best_take`) rather than for being
 * representative.
 *
 * Measured by leave-one-take-out on the 86 signs whose takes are all present,
 * same queries and same classes in every condition, with the shipped
 * representation:
 *
 *     single stored template      0.7267      <- what the app did
 *     medoid take                 0.7093
 *     2 most representative       0.8140
 *     3 most representative       0.8430
 *     4 most representative       0.8605
 *     ALL takes, nearest wins     0.9419      <- what it does now
 *     ALL takes, mean of 2 best   0.9070
 *
 * +21.5 points, monotone in the number of prototypes, with no knee. Every
 * feature and matcher change measured alongside it was worth a few points at
 * most. The bottleneck was never DTW, handedness or the gate: it was having
 * one example of each word.
 *
 * ============================================================================
 * WHY A SEPARATE FILE, AND WHY QUANTIZED
 * ============================================================================
 * `dedicated-signs.json` stores full-resolution landmark timelines because the
 * AVATAR needs them: it replays the real recording. Recognition does not --
 * it only ever sees the resampled feature vectors. Storing eight full-
 * resolution takes per sign would multiply a 30 MB artifact by eight for data
 * the recogniser immediately throws away.
 *
 * So prototypes ship as their own artifact, holding exactly what the matcher
 * consumes: the prepared vectors, 16-bit quantized. That is ~5 KB per take
 * against ~60 KB for the landmarks it was derived from, and it removes all
 * per-load preparation work -- the vectors are used as decoded.
 *
 * The cost of storing prepared vectors is that the file is only valid for the
 * feature pipeline that produced it, so it carries FEATURE_VERSION and a
 * mismatch is REJECTED rather than silently mis-scored. Regenerate with:
 *
 *     npx tsx scripts/build-prototypes.ts
 *
 * The file is generated, not committed. When it is absent the recogniser runs
 * on the registry's single template exactly as before -- degraded, never
 * broken.
 */
import type { PreparedSide, PreparedSidedSequence } from "./twoHandMatch";
import { FEATURE_VERSION, DTW_SEQUENCE_LENGTH } from "./twoHandMatch";

/** Where the generated artifact is served from. */
export const PROTOTYPE_URL = "/models/sign-vocabulary/sign-prototypes.json";

/**
 * Fixed-point scale for quantization. Feature values are normalized hand
 * coordinates and hand-size-relative offsets, which sit within a few units of
 * zero; 1/1000 of a unit is far below the noise in a landmark estimate, and
 * the widest value observed across the corpus is ~20, so nothing clips at the
 * Int16 bound of 32.767.
 */
const QUANT_SCALE = 1000;

export interface EncodedTrack {
    /** Real detected frames before resampling -- drives coverage weighting. */
    frames: number;
    /** Vector dimensionality per frame. */
    dim: number;
    /** Int16 little-endian, base64: frames * dim values, row-major. */
    data: string;
}

export interface EncodedTake {
    /** Processed file this prototype came from, for traceability. */
    file: string;
    left: EncodedTrack | null;
    right: EncodedTrack | null;
    relation: EncodedTrack | null;
    totalFrames: number;
}

export interface PrototypeFile {
    version: 1;
    /** Must equal FEATURE_VERSION or the file is rejected. */
    featureVersion: string;
    sequenceLength: number;
    generatedAt: string;
    /** sourceSignId -> every usable take of that sign. */
    signs: Record<string, EncodedTake[]>;
}

/** Node has Buffer, the browser has btoa. Encoding only ever runs in the
 * generator, but the module must not break a browser bundle by assuming. */
function toBase64(bytes: Uint8Array): string {
    if (typeof Buffer !== "undefined") return Buffer.from(bytes).toString("base64");
    let binary = "";
    for (const byte of bytes) binary += String.fromCharCode(byte);
    return btoa(binary);
}

function encodeTrack(side: PreparedSide | null): EncodedTrack | null {
    if (!side || side.vectors.length === 0) return null;
    const dim = side.vectors[0].length;
    const out = new Int16Array(side.vectors.length * dim);
    let i = 0;
    for (const vector of side.vectors) {
        for (const value of vector) {
            const q = Math.round(value * QUANT_SCALE);
            out[i++] = q > 32767 ? 32767 : q < -32768 ? -32768 : q;
        }
    }
    return { frames: side.detectedFrames, dim, data: toBase64(new Uint8Array(out.buffer)) };
}

function decodeTrack(track: EncodedTrack | null): PreparedSide | null {
    if (!track) return null;
    const bytes =
        typeof Buffer !== "undefined"
            ? Uint8Array.from(Buffer.from(track.data, "base64"))
            : Uint8Array.from(atob(track.data), (c) => c.charCodeAt(0));
    const values = new Int16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
    const vectors: number[][] = [];
    for (let i = 0; i + track.dim <= values.length; i += track.dim) {
        const row: number[] = new Array<number>(track.dim);
        for (let k = 0; k < track.dim; k++) row[k] = values[i + k] / QUANT_SCALE;
        vectors.push(row);
    }
    if (vectors.length === 0) return null;
    return { vectors, detectedFrames: track.frames };
}

/** Serializes one prepared gesture for the generated artifact. */
export function encodeTake(
    file: string,
    prepared: PreparedSidedSequence
): EncodedTake | null {
    const left = encodeTrack(prepared.left);
    const right = encodeTrack(prepared.right);
    if (!left && !right) return null;
    return {
        file,
        left,
        right,
        relation: encodeTrack(prepared.relation),
        totalFrames: prepared.totalFrames,
    };
}

export function decodeTake(take: EncodedTake): PreparedSidedSequence {
    return {
        left: decodeTrack(take.left),
        right: decodeTrack(take.right),
        unknown: null,
        relation: decodeTrack(take.relation),
        totalFrames: take.totalFrames,
    };
}

export interface DecodedPrototypes {
    /** sourceSignId -> prepared prototypes, ready to score. */
    bySign: Map<string, PreparedSidedSequence[]>;
    takeCount: number;
}

/**
 * Validates and decodes a prototype file.
 *
 * Returns null, loudly, when the file was produced by a different feature
 * pipeline. Scoring 16-bit vectors from one representation against another is
 * not a degraded result, it is a meaningless one, and the single-template
 * fallback is strictly better than that.
 */
export function decodePrototypeFile(raw: unknown): DecodedPrototypes | null {
    if (typeof raw !== "object" || raw === null) return null;
    const file = raw as Partial<PrototypeFile>;
    if (file.version !== 1 || typeof file.signs !== "object" || file.signs === null) return null;
    if (file.featureVersion !== FEATURE_VERSION) {
        console.warn(
            `[prototypeStore] ignoring sign-prototypes.json: built for feature version ` +
            `"${String(file.featureVersion)}", this build needs "${FEATURE_VERSION}". ` +
            `Regenerate with: npx tsx scripts/build-prototypes.ts`
        );
        return null;
    }
    if (file.sequenceLength !== DTW_SEQUENCE_LENGTH) {
        console.warn(
            `[prototypeStore] ignoring sign-prototypes.json: sequence length ` +
            `${String(file.sequenceLength)} != ${DTW_SEQUENCE_LENGTH}.`
        );
        return null;
    }
    const bySign = new Map<string, PreparedSidedSequence[]>();
    let takeCount = 0;
    for (const [signId, takes] of Object.entries(file.signs)) {
        if (!Array.isArray(takes)) continue;
        const decoded: PreparedSidedSequence[] = [];
        for (const take of takes) {
            const prepared = decodeTake(take);
            if (prepared.left || prepared.right) {
                decoded.push(prepared);
                takeCount++;
            }
        }
        if (decoded.length > 0) bySign.set(signId, decoded);
    }
    return bySign.size > 0 ? { bySign, takeCount } : null;
}
