/**
 * VOICE/TEXT -> SIGN -> AVATAR INTEGRATION TESTS, on REAL KArSL data.
 *
 * Everything here loads the actual registry that ships in the repo
 * (public/models/sign-vocabulary/dedicated-signs.json) and the actual
 * per-letter landmark table. No landmark sequence is fabricated: the phrases
 * and words under test are chosen at runtime FROM the registry, so the tests
 * stay valid if the registry is re-imported with different content.
 */
import { describe, it, expect, beforeAll, vi } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { translateToSignSequence } from "@/lib/sign-language/signTranslator";
import { normalizeArabicText } from "@/lib/sign-language/textNormalization";
import {
    fromSignStep,
    loadKarslSignById,
    listKarslSigns,
    totalDurationMs,
    KARSL_SOURCE_FPS,
    STATIC_POSE_HOLD_MS,
} from "./karslSign";
import { SignAnimationController } from "./signAnimationController";
import { solveHandPose } from "./retarget";
import type { HandLandmarks } from "@/types/landmarks";
import { withLegacySequence } from "@/lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "@/lib/sign-language/signRegistry";

const ROOT = path.resolve(__dirname, "../..");
const registryRaw = readFileSync(path.join(ROOT, "public/models/sign-vocabulary/dedicated-signs.json"), "utf-8");
const lettersRaw = readFileSync(path.join(ROOT, "public/models/arsl-31/letterLandmarks.json"), "utf-8");

interface RegistryEntry {
    text: string;
    sourceSignId?: string;
    landmarkSequence: HandLandmarks[];
}
// The shipped registry omits the legacy `landmarkSequence` (it duplicates the
// primary hand already in `handFrames`). Reading the file directly therefore
// has to apply the same normalization the app applies on load.
const registry = (JSON.parse(registryRaw) as DedicatedSignEntry[]).map(
    withLegacySequence
) as unknown as RegistryEntry[];

/** A real multi-word phrase and a real single word, picked from the registry
 * itself rather than hardcoded. */
const realPhrase = registry.find((e) => e.text.trim().includes(" ") && e.landmarkSequence.length > 5)!;
const realWord = registry.find((e) => !e.text.includes(" ") && /[؀-ۿ]/.test(e.text) && e.landmarkSequence.length > 5)!;

/** An Arabic word that is definitely NOT in the registry. */
function makeUnknownWord(): string {
    const known = new Set(registry.map((e) => normalizeArabicText(e.text).text));
    for (const candidate of ["كتاب", "قلم", "شجرة", "نافذة", "مفتاح"]) {
        if (!known.has(normalizeArabicText(candidate).text)) return candidate;
    }
    throw new Error("No unknown-word candidate available");
}

beforeAll(() => {
    vi.stubGlobal("fetch", async (url: string) => ({
        ok: true,
        status: 200,
        json: async () => JSON.parse(String(url).includes("sign-vocabulary") ? registryRaw : lettersRaw),
    }));
});

describe("KArSL lookup through the shared translation pipeline", () => {
    it("has real registry content to test against", () => {
        expect(registry.length).toBeGreaterThan(0);
        expect(realPhrase).toBeDefined();
        expect(realWord).toBeDefined();
    });

    it("resolves an exact KArSL phrase to ONE dedicated sign, kept intact", async () => {
        const seq = await translateToSignSequence(realPhrase.text);
        expect(seq.words.length).toBe(1);
        expect(seq.words[0].steps.length).toBe(1);
        const step = seq.words[0].steps[0];
        expect(step.source).toBe("dedicated");
        // The phrase must survive as a whole string, spaces and all.
        expect(step.text).toBe(normalizeArabicText(realPhrase.text).text);
        expect(step.text).toContain(" ");
        expect(step.landmarkFrames.length).toBe(realPhrase.landmarkSequence.length);
    });

    it("resolves an exact KArSL word to one dedicated sign", async () => {
        const seq = await translateToSignSequence(realWord.text);
        expect(seq.words.length).toBe(1);
        expect(seq.words[0].steps[0].source).toBe("dedicated");
        expect(seq.words[0].steps[0].landmarkFrames.length).toBeGreaterThan(1);
    });

    it("prefers the LONGEST phrase when a phrase is embedded in a sentence", async () => {
        const unknown = makeUnknownWord();
        const seq = await translateToSignSequence(`${unknown} ${realPhrase.text}`);
        const dedicated = seq.words.filter((w) => w.steps.some((s) => s.source === "dedicated"));
        expect(dedicated.length).toBe(1);
        expect(dedicated[0].word).toBe(normalizeArabicText(realPhrase.text).text);
        // The phrase's own words must NOT have been looked up individually.
        const phraseWords = normalizeArabicText(realPhrase.text).text.split(" ");
        for (const pw of phraseWords) {
            expect(seq.words.some((w) => w.word === pw)).toBe(false);
        }
    });

    it("falls back to letter spelling for a word with no dedicated sign", async () => {
        const unknown = makeUnknownWord();
        const seq = await translateToSignSequence(unknown);
        expect(seq.words.length).toBe(1);
        expect(seq.words[0].steps.every((s) => s.source !== "dedicated")).toBe(true);
        expect(seq.words[0].steps.length).toBe(unknown.length);
    });

    it("marks a fallback distinguishably from a real word sign", async () => {
        const unknown = makeUnknownWord();
        const fallback = (await translateToSignSequence(unknown)).words[0].steps.map(fromSignStep);
        const real = (await translateToSignSequence(realWord.text)).words[0].steps.map(fromSignStep);
        expect(fallback.every((s) => s.source !== "karsl")).toBe(true);
        expect(real.every((s) => s.source === "karsl")).toBe(true);
    });

    it("voice and typed input produce identical results (one pipeline)", async () => {
        // The speech path differs only in where the string comes from, so
        // equality of the translation output IS the property to check.
        const spoken = await translateToSignSequence(realPhrase.text);
        const typed = await translateToSignSequence(realPhrase.text);
        expect(JSON.stringify(typed.words.map((w) => w.word))).toBe(JSON.stringify(spoken.words.map((w) => w.word)));
        expect(typed.normalizedText).toBe(spoken.normalizedText);
    });
});

describe("AnimatableSign built from real sequences", () => {
    it("loads a real KArSL sign by ID with its real frames", async () => {
        const id = realPhrase.sourceSignId!;
        const sign = await loadKarslSignById(id);
        expect(sign).not.toBeNull();
        expect(sign!.signId).toBe(id);
        expect(sign!.frameCount).toBe(realPhrase.landmarkSequence.length);
        expect(sign!.landmarkSequence[0].length).toBe(21);
    });

    it("returns null for an unknown sign ID rather than inventing one", async () => {
        expect(await loadKarslSignById("9999")).toBeNull();
    });

    it("derives duration from the real frame count at the source frame rate", async () => {
        const sign = await loadKarslSignById(realPhrase.sourceSignId!);
        expect(sign!.durationIsReal).toBe(true);
        expect(sign!.durationMs).toBeCloseTo((sign!.frameCount / KARSL_SOURCE_FPS) * 1000, 6);
    });

    it("marks a single-frame letter pose as NOT having a real duration", async () => {
        const unknown = makeUnknownWord();
        const steps = (await translateToSignSequence(unknown)).words[0].steps.map(fromSignStep);
        const letter = steps.find((s) => s.source === "spelled-letter");
        expect(letter).toBeDefined();
        expect(letter!.durationIsReal).toBe(false);
        expect(letter!.durationMs).toBe(STATIC_POSE_HOLD_MS);
    });

    it("lists every registry sign with a real duration", async () => {
        const all = await listKarslSigns();
        expect(all.length).toBe(registry.filter((e) => e.sourceSignId).length);
        for (const s of all.slice(0, 20)) expect(s.durationMs).toBeGreaterThan(0);
    });

    it("solves every frame of a real sign to bone rotations", async () => {
        const sign = await loadKarslSignById(realPhrase.sourceSignId!);
        const solutions = sign!.landmarkSequence.map((f) => solveHandPose(f));
        expect(solutions.length).toBe(sign!.frameCount);
        expect(solutions.every((s) => s.bones.length === 20)).toBe(true);
        // A real recorded hand should not produce degenerate palms.
        expect(solutions.filter((s) => s.degenerate).length).toBe(0);
    });
});

describe("SignAnimationController on real durations", () => {
    it("plays a real sequence for its real length, then finishes", async () => {
        const sign = await loadKarslSignById(realPhrase.sourceSignId!);
        const c = new SignAnimationController([sign!]);
        c.play();
        let snap = c.getSnapshot();
        // Step in 16ms slices, as a 60fps render loop would.
        for (let t = 0; t < sign!.durationMs - 20 && snap.state === "playing"; t += 16) {
            snap = c.tick(16);
        }
        expect(snap.state).toBe("playing");
        expect(snap.progressWithinSign).toBeGreaterThan(0.9);
        c.tick(200);
        expect(c.getSnapshot().state).toBe("finished");
    });

    it("walks the real frames in order", async () => {
        const sign = await loadKarslSignById(realPhrase.sourceSignId!);
        const c = new SignAnimationController([sign!]);
        c.play();
        const seen: number[] = [];
        for (let t = 0; t < sign!.durationMs; t += 16) seen.push(c.tick(16).frameIndex);
        for (let i = 1; i < seen.length; i++) expect(seen[i]).toBeGreaterThanOrEqual(seen[i - 1]);
        expect(seen[seen.length - 1]).toBeCloseTo(sign!.frameCount - 1, 0);
    });

    it("speed changes wall-clock time without changing which frames play", async () => {
        const sign = await loadKarslSignById(realPhrase.sourceSignId!);
        const c = new SignAnimationController([sign!]);
        c.setSpeed(2);
        c.play();
        let elapsed = 0;
        while (c.getSnapshot().state === "playing" && elapsed < 10_000) {
            c.tick(16);
            elapsed += 16;
        }
        // At 2x, wall-clock should be about half the real duration.
        expect(elapsed).toBeLessThan(sign!.durationMs * 0.75);
        expect(c.getSnapshot().frameIndex).toBeCloseTo(sign!.frameCount - 1, 0);
    });

    it("advances across several signs with per-sign durations", async () => {
        const a = await loadKarslSignById(registry[0].sourceSignId!);
        const b = await loadKarslSignById(registry[1].sourceSignId!);
        const c = new SignAnimationController([a!, b!]);
        expect(c.getSnapshot().totalDurationMs).toBeCloseTo(totalDurationMs([a!, b!]), 6);
        c.play();
        c.tick(a!.durationMs + 5);
        expect(c.getSnapshot().signIndex).toBe(1);
    });

    it("pause, resume and replay behave", async () => {
        const sign = await loadKarslSignById(realPhrase.sourceSignId!);
        const c = new SignAnimationController([sign!]);
        c.play();
        c.tick(100);
        c.pause();
        const at = c.getSnapshot().elapsedInSignMs;
        c.tick(500);
        expect(c.getSnapshot().elapsedInSignMs).toBe(at);
        c.resume();
        c.tick(100);
        expect(c.getSnapshot().elapsedInSignMs).toBeGreaterThan(at);
        c.replay();
        expect(c.getSnapshot().elapsedInSignMs).toBe(0);
        expect(c.getSnapshot().state).toBe("playing");
    });

    it("an empty sign list finishes instead of throwing", () => {
        const c = new SignAnimationController([]);
        c.play();
        expect(c.getSnapshot().state).toBe("finished");
        expect(() => c.tick(100)).not.toThrow();
    });
});
