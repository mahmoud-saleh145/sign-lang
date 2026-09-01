import { loadDedicatedSignRegistry } from "@/lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "@/lib/sign-language/signRegistry";
import { normalizeLandmarks } from "./features";
import { dtwDistance, resampleSequence } from "./dtw";
import type { HandLandmarks } from "@/types/landmarks";

/**
 * DYNAMIC/WORD-LEVEL KArSL RECOGNITION -- separate from the 31-class static
 * letter model. This module never touches lib/sign-language/signTranslator.ts
 * (the Voice/Text -> Sign path) -- it only reads the same underlying
 * registry data (public/models/sign-vocabulary/dedicated-signs.json) via
 * the existing loadDedicatedSignRegistry() loader, per the architecture
 * constraint that these two directions stay independent.
 *
 * APPROACH: Dynamic Time Warping (DTW) against real stored KArSL sequences.
 * Why DTW and not a trained temporal model: the registry stores real,
 * already-imported landmark sequences per sign (see import_karsl.py). DTW
 * compares a live sequence to those real sequences directly -- no training,
 * no risk of an undertrained deep model fabricating confidence on what may
 * be very few samples per sign. This is a nearest-neighbor classifier in
 * sequence space, not a synthesized or hand-authored gesture model.
 *
 * COORDINATE COMPATIBILITY: both live camera landmarks
 * (lib/mediapipe/handLandmarker.ts) and KArSL-extracted landmarks
 * (ml/training/extract_karsl_skeletons.py) are raw MediaPipe Tasks
 * HandLandmarker output -- same model family, same [0,1]-normalized (x,y)
 * convention, single hand (numHands=1 on both sides). Neither side applies
 * any further normalization at capture/extraction time. This module
 * applies the SAME wrist-relative, scale-invariant normalization already
 * used by the static letter classifier (normalizeLandmarks() from
 * features.ts) to both live and stored sequences before comparing --
 * reusing tested code rather than inventing new normalization. Per-frame
 * vectors used for DTW are the resulting 42 numbers (21 points x, y);
 * z is intentionally excluded, matching normalizeLandmarks()'s existing
 * convention (it does not normalize z either).
 *
 * KNOWN LIMITATION: single-hand only, on both sides. Any real KArSL sign
 * that is inherently two-handed cannot be fully represented by this data,
 * since extraction only ever kept one hand's landmarks. Documented, not
 * silently ignored.
 */

export const DTW_SEQUENCE_LENGTH = 30;

export const MAX_NORMALIZED_DISTANCE = 0.35;

export const MAX_BEST_TO_SECOND_BEST_RATIO = 0.85;

export const MAX_DTW_CANDIDATES = 50;

export interface DynamicSignMatch {
    entry: DedicatedSignEntry;
    distance: number;
}

interface PreparedEntry {
    entry: DedicatedSignEntry;
    vectors: number[][];
}

let cachedPreparedEntries: PreparedEntry[] | null = null;

function landmarksToVector(landmarks: HandLandmarks): number[] {
    const norm = normalizeLandmarks(landmarks);
    const vec: number[] = [];
    for (const p of norm) {
        vec.push(p.x, p.y);
    }
    return vec;
}

function prepareSequence(frames: HandLandmarks[]): number[][] {
    const vectors = frames.map(landmarksToVector);
    return resampleSequence(vectors, DTW_SEQUENCE_LENGTH);
}

async function getPreparedEntries(): Promise<PreparedEntry[]> {
    if (cachedPreparedEntries) return cachedPreparedEntries;

    const registry = await loadDedicatedSignRegistry();
    const prepared: PreparedEntry[] = [];

    for (const entry of registry) {
        if (entry.landmarkSequence.length < 2) {
            console.warn(
                `Skipping dedicated sign "${entry.text}" (sourceSignId=${entry.sourceSignId ?? "?"}): ` +
                `fewer than 2 real frames, cannot be used for temporal matching.`
            );
            continue;
        }
        try {
            const vectors = prepareSequence(entry.landmarkSequence);
            prepared.push({ entry, vectors });
        } catch (err) {
            console.warn(`Skipping dedicated sign "${entry.text}": ${err instanceof Error ? err.message : err}`);
        }
    }

    cachedPreparedEntries = prepared;
    return prepared;
}

export function clearDynamicSignCache(): void {
    cachedPreparedEntries = null;
}

/**
 * Call this once (e.g. when the camera starts) to fetch and precompute the
 * registry before any live frames need matching. recognitionPipeline.ts's
 * per-frame processing is synchronous and has existing tests relying on
 * that contract, so per-frame matching itself (recognizeDynamicSignSync)
 * must not require an await -- this function is what makes that possible.
 */
export async function preloadDynamicSignRegistry(): Promise<void> {
    await getPreparedEntries();
}

function matchAgainstPrepared(frames: HandLandmarks[], prepared: PreparedEntry[]): DynamicSignMatch | null {
    if (frames.length < 2 || prepared.length === 0) return null;

    const liveVectors = prepareSequence(frames);
    const liveStart = liveVectors[0];
    const liveEnd = liveVectors[liveVectors.length - 1];

    const ranked = prepared
        .map((p) => {
            const start = p.vectors[0];
            const end = p.vectors[p.vectors.length - 1];
            const endpointDist = euclidean(liveStart, start) + euclidean(liveEnd, end);
            return { p, endpointDist };
        })
        .sort((a, b) => a.endpointDist - b.endpointDist)
        .slice(0, MAX_DTW_CANDIDATES);

    const results = ranked.map(({ p }) => ({
        entry: p.entry,
        distance: dtwDistance(liveVectors, p.vectors).normalizedCost,
    }));
    results.sort((a, b) => a.distance - b.distance);

    const best = results[0];
    if (!best || best.distance > MAX_NORMALIZED_DISTANCE) return null;

    const secondBest = results[1];
    if (secondBest && best.distance > secondBest.distance * MAX_BEST_TO_SECOND_BEST_RATIO) {
        return null;
    }

    return { entry: best.entry, distance: best.distance };
}

/**
 * Async entry point: awaits registry load if not already warmed, then
 * matches. Suitable for tests and one-off calls; NOT used by the live
 * per-frame camera loop (see recognizeDynamicSignSync below).
 */
export async function recognizeDynamicSign(frames: HandLandmarks[]): Promise<DynamicSignMatch | null> {
    const prepared = await getPreparedEntries();
    return matchAgainstPrepared(frames, prepared);
}

/**
 * Synchronous entry point for the live per-frame camera loop. Returns null
 * (with a console.warn) if preloadDynamicSignRegistry() hasn't resolved
 * yet -- a safe degradation (that frame simply isn't checked for a dynamic
 * match) rather than a thrown error interrupting the static letter path.
 */
export function recognizeDynamicSignSync(frames: HandLandmarks[]): DynamicSignMatch | null {
    if (!cachedPreparedEntries) {
        console.warn(
            "recognizeDynamicSignSync called before preloadDynamicSignRegistry() resolved; skipping this frame."
        );
        return null;
    }
    return matchAgainstPrepared(frames, cachedPreparedEntries);
}

function euclidean(a: number[], b: number[]): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        const d = a[i] - b[i];
        sum += d * d;
    }
    return Math.sqrt(sum);
}