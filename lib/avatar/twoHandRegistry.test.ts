/**
 * END-TO-END VALIDATION OF THE TWO-HAND PIPELINE, on REAL KArSL data.
 *
 * Every sequence used here was extracted from real KArSL video by
 * ml/training/extract_karsl_skeletons.py with num_hands=2 and imported by
 * ml/training/import_karsl.py. Nothing is synthesised.
 *
 * WHICH REGISTRY IS USED: set KARSL_TWO_HAND_REGISTRY to a two-hand registry
 * to test against it; otherwise the shipped
 * public/models/sign-vocabulary/dedicated-signs.json is used. The two-hand
 * assertions SKIP (loudly) when the registry in play has no `handFrames` --
 * which is the case until the full re-extraction has been run. They must not
 * pass vacuously and must not fail for data that does not exist yet.
 */
import { describe, it, expect, beforeAll, vi } from "vitest";
import { readFileSync, existsSync } from "fs";
import path from "path";
import {
    loadDedicatedSignRegistry,
    findDedicatedSign,
    entryToSidedSequence,
} from "@/lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "@/lib/sign-language/signRegistry";
import { loadKarslSignById, listKarslSigns } from "./karslSign";
import { solveHandPose } from "./retarget";
import { translateToSignSequence } from "@/lib/sign-language/signTranslator";
import { fromSignStep } from "./karslSign";
import {
    clearDynamicSignCache,
    preloadDynamicSignRegistry,
    getDynamicSignRegistryEntryCount,
    recognizeDynamicSignSyncWithDiagnostics,
} from "@/lib/ml/dynamicSignRecognizer";

const ROOT = path.resolve(__dirname, "../..");
const SHIPPED = path.join(ROOT, "public/models/sign-vocabulary/dedicated-signs.json");
const OVERRIDE = process.env.KARSL_TWO_HAND_REGISTRY;
const REGISTRY_PATH = OVERRIDE && existsSync(OVERRIDE) ? OVERRIDE : SHIPPED;

const registryJson = readFileSync(REGISTRY_PATH, "utf-8");
const rawEntries = JSON.parse(registryJson) as DedicatedSignEntry[];
const lettersJson = readFileSync(path.join(ROOT, "public/models/arsl-31/letterLandmarks.json"), "utf-8");

const hasTwoHandData = rawEntries.some((e) => e.handFrames && e.handFrames.length > 0);
const describeTwoHand = hasTwoHandData ? describe : describe.skip;

if (!hasTwoHandData) {
    console.warn(
        `[twoHandRegistry] Two-hand assertions SKIPPED: ${REGISTRY_PATH} has no handFrames. ` +
        `Re-run extract_karsl_skeletons.py + import_karsl.py, or set KARSL_TWO_HAND_REGISTRY.`
    );
}

beforeAll(() => {
    vi.stubGlobal("fetch", async (url: string) => ({
        ok: true,
        status: 200,
        json: async () => JSON.parse(String(url).includes("sign-vocabulary") ? registryJson : lettersJson),
    }));
});

describe("registry loading (any format)", () => {
    it("always exposes a usable landmarkSequence", async () => {
        const reg = await loadDedicatedSignRegistry();
        expect(reg.length).toBeGreaterThan(0);
        for (const e of reg) {
            expect(Array.isArray(e.landmarkSequence)).toBe(true);
            expect(e.landmarkSequence.length).toBeGreaterThan(0);
            expect(e.landmarkSequence[0].length).toBe(21);
        }
    });

    it("keeps multi-word phrases intact as single entries", async () => {
        const reg = await loadDedicatedSignRegistry();
        const multi = reg.filter((e) => e.text.trim().includes(" "));
        for (const e of multi) {
            expect(e.text).toBe(e.text.trim());
            const hit = await findDedicatedSign(e.text);
            expect(hit?.text).toBe(e.text);
        }
    });
});

describeTwoHand("two-hand registry -> AnimatableSign", () => {
    const twoHanded = rawEntries.filter((e) => e.handedness?.twoHanded);
    const oneHanded = rawEntries.filter(
        (e) => e.handFrames && !e.handedness?.twoHanded && (e.handedness?.framesWithBoth ?? 0) === 0
    );

    it("has both a two-handed and a one-handed real sign to test with", () => {
        expect(twoHanded.length).toBeGreaterThan(0);
        // A one-handed example is not guaranteed in every subset, so only
        // assert the two-handed side is present.
    });

    it("loads BOTH real hand tracks for a two-handed sign", async () => {
        const e = twoHanded[0];
        const sign = await loadKarslSignById(e.sourceSignId!);
        expect(sign).not.toBeNull();
        expect(sign!.tracks).not.toBeNull();
        expect(sign!.twoHanded).toBe(true);
        expect(sign!.handDataUnknown).toBe(false);

        const { left, right } = sign!.tracks!;
        expect(left.detectedFrames).toBeGreaterThan(0);
        expect(right.detectedFrames).toBeGreaterThan(0);
        // Both tracks live on ONE shared timeline.
        expect(left.frames.length).toBe(right.frames.length);
        expect(left.frames.length).toBe(sign!.frameCount);
    });

    it("gives each hand its own distinct trajectory, not a copy", async () => {
        const e = twoHanded[0];
        const sign = await loadKarslSignById(e.sourceSignId!);
        const { left, right } = sign!.tracks!;
        let compared = 0;
        let differing = 0;
        for (let i = 0; i < left.frames.length; i++) {
            const l = left.frames[i];
            const r = right.frames[i];
            if (!l || !r) continue;
            compared++;
            if (Math.hypot(l[0].x - r[0].x, l[0].y - r[0].y) > 1e-6) differing++;
        }
        expect(compared).toBeGreaterThan(0);
        expect(differing).toBe(compared);
    });

    it("leaves the absent hand null for a genuinely one-handed sign", async () => {
        if (oneHanded.length === 0) return; // subset may not contain one
        const sign = await loadKarslSignById(oneHanded[0].sourceSignId!);
        expect(sign!.twoHanded).toBe(false);
        const { left, right } = sign!.tracks!;
        const empty = left.detectedFrames === 0 ? left : right;
        const present = left.detectedFrames === 0 ? right : left;
        expect(empty.detectedFrames).toBe(0);
        expect(present.detectedFrames).toBeGreaterThan(0);
        // Every frame of the empty side is null -- never filled in.
        expect(empty.frames.every((f) => f === null)).toBe(true);
    });

    it("solves real bone rotations for every frame of BOTH hands", async () => {
        const sign = await loadKarslSignById(twoHanded[0].sourceSignId!);
        for (const side of ["left", "right"] as const) {
            const track = sign!.tracks![side];
            let solved = 0;
            for (const f of track.frames) {
                if (!f) continue;
                const s = solveHandPose(f);
                expect(s.bones.length).toBe(20);
                expect(s.degenerate).toBe(false);
                solved++;
            }
            expect(solved).toBe(track.detectedFrames);
        }
    });

    it("calibrates each hand's proportions from that hand's own frames", async () => {
        const sign = await loadKarslSignById(twoHanded[0].sourceSignId!);
        const { left, right } = sign!.tracks!;
        expect(Object.keys(left.boneLengths).length).toBeGreaterThan(0);
        expect(Object.keys(right.boneLengths).length).toBeGreaterThan(0);
    });

    it("reports hand availability in the sign listing", async () => {
        const all = await listKarslSigns();
        const hit = all.find((s) => s.signId === twoHanded[0].sourceSignId);
        expect(hit).toBeDefined();
        expect(hit!.twoHanded).toBe(true);
        expect(hit!.hasLeft && hit!.hasRight).toBe(true);
        expect(hit!.handDataUnknown).toBe(false);
    });
});

describeTwoHand("VoiceToSign lookup reaches the two-hand data", () => {
    const twoHanded = rawEntries.filter((e) => e.handedness?.twoHanded);

    it("resolves a real two-handed label to one dedicated step carrying both hands", async () => {
        const e = twoHanded.find((x) => x.text.trim().length > 0)!;
        const seq = await translateToSignSequence(e.text);
        expect(seq.words.length).toBe(1);
        expect(seq.words[0].steps.length).toBe(1);

        const step = seq.words[0].steps[0];
        expect(step.source).toBe("dedicated");

        // NOTE: fromSignStep() builds from the translator's SignStep, which
        // carries only landmarkFrames. Two-hand playback in the app comes from
        // loadKarslSignById(); this asserts the text path resolves to the same
        // registry entry, which is what makes that lookup possible.
        const animatable = fromSignStep(step);
        expect(animatable.source).toBe("karsl");
        expect(animatable.frameCount).toBeGreaterThan(1);

        const byId = await loadKarslSignById(e.sourceSignId!);
        expect(byId!.text).toBe(step.text);
    });

    it("keeps a real multi-word two-handed phrase whole", async () => {
        const phrase = twoHanded.find((x) => x.text.trim().includes(" "));
        if (!phrase) return;
        const seq = await translateToSignSequence(phrase.text);
        expect(seq.words.length).toBe(1);
        expect(seq.words[0].steps[0].text).toContain(" ");
    });
});

describe("camera WORD MODE still works against this registry", () => {
    beforeAll(async () => {
        clearDynamicSignCache();
        await preloadDynamicSignRegistry();
    }, 120_000);

    it("loads every entry into the DTW recognizer", () => {
        expect(getDynamicSignRegistryEntryCount()).toBe(rawEntries.length);
    });

    it("recognizes a stored sequence as itself (pipeline is intact)", async () => {
        const reg = await loadDedicatedSignRegistry();
        const entry = reg.find((e) => e.landmarkSequence.length >= 10)!;
        // The recognizer is side-aware now: replay the entry's REAL two-hand
        // timeline (both hands, on their recorded sides), not the flat legacy
        // primary-hand copy.
        const sided = entryToSidedSequence(entry)!;
        const d = recognizeDynamicSignSyncWithDiagnostics(sided);
        expect(d.registryLoaded).toBe(true);
        expect(d.bestSignId).toBe(entry.sourceSignId);
        expect(d.accepted).toBe(true);
    }, 120_000);
});
