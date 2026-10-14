import { describe, it, expect, beforeAll, vi } from "vitest";
import type { HandLandmarks } from "@/types/landmarks";
import { sidedFramesFromSingleHand } from "./dynamicGestureBuffer";

/**
 * IMPORTANT HONESTY NOTE: the real public/models/sign-vocabulary/
 * dedicated-signs.json produced by the user's actual KArSL import was never
 * available in this environment (only the labels spreadsheet was). Every
 * fixture below is SYNTHETIC -- built to exercise the matching algorithm
 * correctly, not to represent any real Arabic Sign Language movement.
 */

/**
 * IMPORTANT: this generator must include genuine per-frame SHAPE
 * deformation (the cos/sin term below), not just a rigid whole-hand
 * translation. An earlier version of this fixture translated all 21
 * points identically every frame; since dynamicSignRecognizer.ts's
 * matching normalizes out translation and scale (reusing
 * normalizeLandmarks() from features.ts, correctly), a purely rigid
 * "hand" collapses to an IDENTICAL degenerate shape on every frame and
 * every synthetic sequence -- which silently made every fixture
 * indistinguishable from every other. Verified numerically before writing
 * these tests; see the freq/phase parameters below for how distinct
 * synthetic "signs" are made genuinely distinct after normalization.
 */
function makeSequence(basePoint: [number, number], numFrames: number, freq: number, phase: number): HandLandmarks[] {
    const frames: HandLandmarks[] = [];
    for (let f = 0; f < numFrames; f++) {
        const frame = Array.from({ length: 21 }, (_, i) => {
            const angle = freq * f + phase + i * 0.35;
            return {
                x: basePoint[0] + 0.01 * f + 0.06 * Math.cos(angle) + i * 0.004,
                y: basePoint[1] + 0.01 * f + 0.06 * Math.sin(angle) + i * 0.004,
                z: 0,
            };
        });
        frames.push(frame);
    }
    return frames;
}

function makeEntry(text: string, sourceSignId: string, frames: HandLandmarks[]) {
    return {
        text,
        landmarkSequence: frames.map((frame) => frame.map((p) => ({ x: p.x, y: p.y, z: p.z }))),
        // The registry is side-aware now: an entry with no `handFrames` and no
        // recorded side has no hand the matcher can name, and is skipped on
        // load. These synthetic signs are one right hand.
        handFrames: frames.map((frame) => ({
            left: null,
            right: frame.map((p) => ({ x: p.x, y: p.y, z: p.z })),
        })),
        provenance: {
            datasetName: "TEST-FIXTURE-NOT-REAL-KARSL",
            source: "unit test only",
            license: "n/a",
            version: "test",
        },
        sourceSignId,
    };
}

const SEQUENCE_A = makeSequence([0.3, 0.3], 20, 0.5, 0);
const SEQUENCE_B = makeSequence([0.7, 0.2], 20, 0.9, 1.5);
const SEQUENCE_PHRASE = makeSequence([0.5, 0.5], 25, 1.3, 3.0);

const SYNTHETIC_REGISTRY = [
    makeEntry("\u0645\u0631\u062d\u0628\u0627", "0001", SEQUENCE_A),
    makeEntry("\u0634\u0643\u0631\u0627", "0002", SEQUENCE_B),
    makeEntry("\u0625\u0633\u0639\u0627\u0641\u0627\u062a \u0623\u0648\u0644\u064a\u0629", "0093", SEQUENCE_PHRASE),
];

beforeAll(() => {
    vi.stubGlobal(
        "fetch",
        vi.fn(async (url: string) => {
            if (url === "/models/sign-vocabulary/dedicated-signs.json") {
                return { ok: true, status: 200, json: async () => SYNTHETIC_REGISTRY } as Response;
            }
            throw new Error(`Unexpected fetch in test: ${url}`);
        })
    );
});

describe("recognizeDynamicSign (SYNTHETIC registry fixture -- real KArSL data unavailable)", () => {
    it("recognizes a live sequence that closely matches a real registry entry", async () => {
        const { recognizeDynamicSign, clearDynamicSignCache } = await import("./dynamicSignRecognizer");
        clearDynamicSignCache();

        const live = SEQUENCE_A.map((frame) =>
            frame.map((p) => ({ x: p.x + 0.0005, y: p.y + 0.0005, z: p.z }))
        );

        const match = await recognizeDynamicSign(sidedFramesFromSingleHand("right", live));
        expect(match).not.toBeNull();
        expect(match!.entry.text).toBe("\u0645\u0631\u062d\u0628\u0627");
    });

    it("recognizes a multi-word phrase entry as ONE result, text returned verbatim", async () => {
        const { recognizeDynamicSign, clearDynamicSignCache } = await import("./dynamicSignRecognizer");
        clearDynamicSignCache();

        const live = SEQUENCE_PHRASE.map((frame) =>
            frame.map((p) => ({ x: p.x + 0.0003, y: p.y + 0.0003, z: p.z }))
        );

        const match = await recognizeDynamicSign(sidedFramesFromSingleHand("right", live));
        expect(match).not.toBeNull();
        expect(match!.entry.text).toBe("\u0625\u0633\u0639\u0627\u0641\u0627\u062a \u0623\u0648\u0644\u064a\u0629");
        expect(match!.entry.text.split(" ")).toHaveLength(2);
    });

    it("returns null for a sequence that matches nothing confidently (below threshold)", async () => {
        const { recognizeDynamicSign, clearDynamicSignCache } = await import("./dynamicSignRecognizer");
        clearDynamicSignCache();

        const noise = makeSequence([-5, -5], 20, 5, 6);
        const match = await recognizeDynamicSign(sidedFramesFromSingleHand("right", noise));
        expect(match).toBeNull();
    });

    it("returns null when too few frames are provided", async () => {
        const { recognizeDynamicSign, clearDynamicSignCache } = await import("./dynamicSignRecognizer");
        clearDynamicSignCache();
        const match = await recognizeDynamicSign(sidedFramesFromSingleHand("right", SEQUENCE_A.slice(0, 1)));
        expect(match).toBeNull();
    });

    it("distinguishes between two different real registry entries correctly", async () => {
        const { recognizeDynamicSign, clearDynamicSignCache } = await import("./dynamicSignRecognizer");
        clearDynamicSignCache();

        const liveB = SEQUENCE_B.map((frame) => frame.map((p) => ({ x: p.x + 0.0004, y: p.y + 0.0004, z: p.z })));
        const match = await recognizeDynamicSign(sidedFramesFromSingleHand("right", liveB));
        expect(match).not.toBeNull();
        expect(match!.entry.text).toBe("\u0634\u0643\u0631\u0627");
    });
});

describe("recognizeDynamicSignSync (used by the live per-frame camera loop)", () => {
    it("returns null and warns if called before preloadDynamicSignRegistry() resolves", async () => {
        const { recognizeDynamicSignSync, clearDynamicSignCache } = await import("./dynamicSignRecognizer");
        clearDynamicSignCache();
        const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => { });
        const result = recognizeDynamicSignSync(sidedFramesFromSingleHand("right", SEQUENCE_A));
        expect(result).toBeNull();
        expect(warnSpy).toHaveBeenCalled();
        warnSpy.mockRestore();
    });

    it("matches synchronously once the registry has been preloaded", async () => {
        const { recognizeDynamicSignSync, preloadDynamicSignRegistry, clearDynamicSignCache } = await import(
            "./dynamicSignRecognizer"
        );
        clearDynamicSignCache();
        await preloadDynamicSignRegistry();

        const live = SEQUENCE_A.map((frame) => frame.map((p) => ({ x: p.x + 0.0005, y: p.y + 0.0005, z: p.z })));
        const match = recognizeDynamicSignSync(sidedFramesFromSingleHand("right", live));
        expect(match).not.toBeNull();
        expect(match!.entry.text).toBe("\u0645\u0631\u062d\u0628\u0627");
    });
});

/** Mirrors a real (raw, un-normalized) landmark sequence around each
 * frame's own wrist x-position. In wrist-relative normalized space this is
 * mathematically equivalent to negating x -- used here to simulate a capture
 * of the gesture performed with the other hand. */
function mirrorRawLandmarks(frames: HandLandmarks[]): HandLandmarks[] {
    return frames.map((frame) => {
        const wristX = frame[0].x;
        return frame.map((p) => ({ x: 2 * wristX - p.x, y: p.y, z: p.z }));
    });
}

/**
 * WHAT THIS BLOCK USED TO ASSERT, AND WHY IT NOW ASSERTS THE OPPOSITE.
 *
 * The recognizer used to have no idea which physical hand it was looking at,
 * so it compared every candidate against both the live sequence and a
 * horizontally mirrored copy and kept whichever fitted better. These tests
 * asserted that mirror-invariance: a mirrored capture still matched.
 *
 * Mirror-invariance was a workaround for a missing identity, and it was
 * measurably harmful. Both hands now carry an anatomical side, decided from
 * the data (lib/ml/handSides.ts), so left is compared only with left and
 * right only with right.
 *
 * A horizontally mirrored capture is therefore no longer the same gesture: it
 * is the gesture performed with the hands the other way round, and accepting
 * it would be accepting a sign that was not performed. The assertion is
 * inverted rather than deleted, so the removal of mirror-invariance is a
 * property this suite still pins down, in the direction the design now
 * requires -- and the second test keeps asserting, unchanged, that a genuine
 * capture with the same hand still matches.
 */
describe("side-aware matching (a mirrored capture is a DIFFERENT gesture, not the same one)", () => {
    it("does NOT accept a horizontally mirrored capture as the same sign", async () => {
        const { recognizeDynamicSignSyncWithDiagnostics, preloadDynamicSignRegistry, clearDynamicSignCache } =
            await import("./dynamicSignRecognizer");
        clearDynamicSignCache();
        await preloadDynamicSignRegistry();

        const mirroredLive = mirrorRawLandmarks(SEQUENCE_A);
        const diag = recognizeDynamicSignSyncWithDiagnostics(sidedFramesFromSingleHand("right", mirroredLive));

        // A wrong word is worse than a rejected sign: a mirrored performance
        // must not be committed as the un-mirrored one.
        expect(diag.accepted).toBe(false);
        // And it must be rejected by a THRESHOLD, i.e. because it genuinely
        // does not fit -- not because the pipeline failed to reach a decision.
        expect(["distance-threshold-failed", "ambiguous-margin-failed"]).toContain(
            diag.rejectionReason
        );
        // The comparison really did happen, on the side the query carries.
        expect(diag.bestPerSideDistance.right).not.toBeNull();
        expect(Number.isFinite(diag.bestPerSideDistance.right!)).toBe(true);
    });

    it("still accepts a genuine capture of the same sign with the same hand", async () => {
        const { recognizeDynamicSignSyncWithDiagnostics, preloadDynamicSignRegistry, clearDynamicSignCache } =
            await import("./dynamicSignRecognizer");
        clearDynamicSignCache();
        await preloadDynamicSignRegistry();

        const live = SEQUENCE_A.map((frame) => frame.map((p) => ({ x: p.x + 0.0004, y: p.y + 0.0004, z: p.z })));
        const diag = recognizeDynamicSignSyncWithDiagnostics(sidedFramesFromSingleHand("right", live));

        expect(diag.accepted).toBe(true);
        // `bestUsedMirror` was REMOVED with mirroring itself. Equivalent-strength
        // replacement for "the unmirrored orientation won": the correct sign is
        // the nearest neighbour, and the query's own side carries a real,
        // finite per-side distance.
        expect(diag.bestText).toBe("مرحبا");
        expect(diag.bestPerSideDistance.right).not.toBeNull();
        expect(Number.isFinite(diag.bestPerSideDistance.right!)).toBe(true);
    });
});

describe("recognizeDynamicSignSyncWithDiagnostics (full diagnostic detail for real-runtime debugging)", () => {
    it("reports registry-not-loaded before preload completes", async () => {
        const { recognizeDynamicSignSyncWithDiagnostics, clearDynamicSignCache } = await import(
            "./dynamicSignRecognizer"
        );
        clearDynamicSignCache();
        const diag = recognizeDynamicSignSyncWithDiagnostics(sidedFramesFromSingleHand("right", SEQUENCE_A));
        expect(diag.registryLoaded).toBe(false);
        expect(diag.rejectionReason).toBe("registry-not-loaded");
        expect(diag.accepted).toBe(false);
    });

    it("reports sequence-too-short for a near-empty live sequence", async () => {
        const { recognizeDynamicSignSyncWithDiagnostics, preloadDynamicSignRegistry, clearDynamicSignCache } =
            await import("./dynamicSignRecognizer");
        clearDynamicSignCache();
        await preloadDynamicSignRegistry();
        const diag = recognizeDynamicSignSyncWithDiagnostics(sidedFramesFromSingleHand("right", SEQUENCE_A.slice(0, 1)));
        expect(diag.rejectionReason).toBe("sequence-too-short");
    });

    it("reports distance-threshold-failed with the real measured distance for a genuinely unmatched sequence", async () => {
        const { recognizeDynamicSignSyncWithDiagnostics, preloadDynamicSignRegistry, clearDynamicSignCache } =
            await import("./dynamicSignRecognizer");
        clearDynamicSignCache();
        await preloadDynamicSignRegistry();
        const noise = makeSequence([-5, -5], 20, 5, 6);
        const diag = recognizeDynamicSignSyncWithDiagnostics(sidedFramesFromSingleHand("right", noise));
        expect(diag.accepted).toBe(false);
        expect(diag.rejectionReason).toBe("distance-threshold-failed");
        expect(diag.bestDistance).not.toBeNull();
        expect(diag.bestDistance!).toBeGreaterThan(diag.distanceThreshold);
    });

    it("on acceptance, reports the full real distance/ratio numbers, not just a boolean", async () => {
        const { recognizeDynamicSignSyncWithDiagnostics, preloadDynamicSignRegistry, clearDynamicSignCache } =
            await import("./dynamicSignRecognizer");
        clearDynamicSignCache();
        await preloadDynamicSignRegistry();
        const live = SEQUENCE_A.map((frame) => frame.map((p) => ({ x: p.x + 0.0004, y: p.y + 0.0004, z: p.z })));
        const diag = recognizeDynamicSignSyncWithDiagnostics(sidedFramesFromSingleHand("right", live));

        expect(diag.accepted).toBe(true);
        expect(diag.bestSignId).toBe("0001");
        expect(diag.candidateCount).toBe(3); // all 3 registry entries evaluated -- no candidate pre-filtering
        expect(diag.secondBestDistance).not.toBeNull();
        expect(diag.bestToSecondBestRatio).not.toBeNull();
        expect(diag.distanceThresholdPassed).toBe(true);
        expect(diag.ratioThresholdPassed).toBe(true);
    });
});