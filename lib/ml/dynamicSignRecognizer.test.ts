import { describe, it, expect, beforeAll, vi } from "vitest";
import type { HandLandmarks } from "@/types/landmarks";

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

        const match = await recognizeDynamicSign(live);
        expect(match).not.toBeNull();
        expect(match!.entry.text).toBe("\u0645\u0631\u062d\u0628\u0627");
    });

    it("recognizes a multi-word phrase entry as ONE result, text returned verbatim", async () => {
        const { recognizeDynamicSign, clearDynamicSignCache } = await import("./dynamicSignRecognizer");
        clearDynamicSignCache();

        const live = SEQUENCE_PHRASE.map((frame) =>
            frame.map((p) => ({ x: p.x + 0.0003, y: p.y + 0.0003, z: p.z }))
        );

        const match = await recognizeDynamicSign(live);
        expect(match).not.toBeNull();
        expect(match!.entry.text).toBe("\u0625\u0633\u0639\u0627\u0641\u0627\u062a \u0623\u0648\u0644\u064a\u0629");
        expect(match!.entry.text.split(" ")).toHaveLength(2);
    });

    it("returns null for a sequence that matches nothing confidently (below threshold)", async () => {
        const { recognizeDynamicSign, clearDynamicSignCache } = await import("./dynamicSignRecognizer");
        clearDynamicSignCache();

        const noise = makeSequence([-5, -5], 20, 5, 6);
        const match = await recognizeDynamicSign(noise);
        expect(match).toBeNull();
    });

    it("returns null when too few frames are provided", async () => {
        const { recognizeDynamicSign, clearDynamicSignCache } = await import("./dynamicSignRecognizer");
        clearDynamicSignCache();
        const match = await recognizeDynamicSign(SEQUENCE_A.slice(0, 1));
        expect(match).toBeNull();
    });

    it("distinguishes between two different real registry entries correctly", async () => {
        const { recognizeDynamicSign, clearDynamicSignCache } = await import("./dynamicSignRecognizer");
        clearDynamicSignCache();

        const liveB = SEQUENCE_B.map((frame) => frame.map((p) => ({ x: p.x + 0.0004, y: p.y + 0.0004, z: p.z })));
        const match = await recognizeDynamicSign(liveB);
        expect(match).not.toBeNull();
        expect(match!.entry.text).toBe("\u0634\u0643\u0631\u0627");
    });
});

describe("recognizeDynamicSignSync (used by the live per-frame camera loop)", () => {
    it("returns null and warns if called before preloadDynamicSignRegistry() resolves", async () => {
        const { recognizeDynamicSignSync, clearDynamicSignCache } = await import("./dynamicSignRecognizer");
        clearDynamicSignCache();
        const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => { });
        const result = recognizeDynamicSignSync(SEQUENCE_A);
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
        const match = recognizeDynamicSignSync(live);
        expect(match).not.toBeNull();
        expect(match!.entry.text).toBe("\u0645\u0631\u062d\u0628\u0627");
    });
});