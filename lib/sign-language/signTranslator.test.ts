import { describe, it, expect, beforeAll, vi } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { translateToSignSequence } from "./signTranslator";
import { findDedicatedSign, loadDedicatedSignRegistry } from "./signRegistry";

const REAL_LETTER_LANDMARKS = JSON.parse(
    readFileSync(
        path.resolve(__dirname, "../../public/models/arsl-31/letterLandmarks.json"),
        "utf-8"
    )
) as Record<string, unknown>;

beforeAll(() => {
    vi.stubGlobal(
        "fetch",
        vi.fn(async (url: string) => {
            if (url === "/models/arsl-31/letterLandmarks.json") {
                return { ok: true, status: 200, json: async () => REAL_LETTER_LANDMARKS } as Response;
            }
            if (url === "/models/sign-vocabulary/dedicated-signs.json") {
                return { ok: true, status: 200, json: async () => [] } as Response;
            }
            throw new Error(`Unexpected fetch in test: ${url}`);
        })
    );
});

describe("signRegistry", () => {
    it("is currently empty -- documenting the real, honest state (no dedicated word dataset is integrated yet)", async () => {
        const registry = await loadDedicatedSignRegistry();
        expect(registry).toHaveLength(0);
    });

    it("findDedicatedSign returns null for any word, since the registry has no real entries", async () => {
        expect(await findDedicatedSign("\u0645\u0631\u062d\u0628\u0627")).toBeNull();
    });
});

describe("translateToSignSequence", () => {
    it("spells a single word letter-by-letter using real landmark data, explicitly marked as fallback", async () => {
        const result = await translateToSignSequence("\u0627\u0646\u0627");
        expect(result.words).toHaveLength(1);
        const [word] = result.words;
        expect(word.word).toBe("\u0627\u0646\u0627");
        expect(word.steps).toHaveLength(3);
        for (const step of word.steps) {
            expect(step.source).toBe("spelled-letter");
            expect(step.landmarkFrames.length).toBeGreaterThan(0);
            expect(step.landmarkFrames[0]).toHaveLength(21);
        }
    });

    it("translates multiple words, splitting correctly on spaces", async () => {
        const result = await translateToSignSequence("\u0627\u0646\u0627 \u0628\u062d\u0628\u0643");
        expect(result.words).toHaveLength(2);
        expect(result.words[0].word).toBe("\u0627\u0646\u0627");
        expect(result.words[1].word).toBe("\u0628\u062d\u0628\u0643");
    });

    it("marks a character with no corresponding real sign data as unsupported, never silently dropped", async () => {
        const result = await translateToSignSequence("\u0627\u0661\u0646");
        const steps = result.words[0].steps;
        expect(steps).toHaveLength(3);
        expect(steps[0].source).toBe("spelled-letter");
        expect(steps[1].source).toBe("unsupported");
        expect(steps[1].landmarkFrames).toHaveLength(0);
        expect(steps[2].source).toBe("spelled-letter");
    });

    it("phrase example from the product spec translates without crashing and every step carries real data or an explicit unsupported marker", async () => {
        const result = await translateToSignSequence("\u0645\u0631\u062d\u0628\u0627 \u0628\u0643");
        for (const word of result.words) {
            for (const step of word.steps) {
                expect(["dedicated", "spelled-letter", "unsupported"]).toContain(step.source);
                if (step.source !== "unsupported") {
                    expect(step.landmarkFrames.length).toBeGreaterThan(0);
                }
            }
        }
    });

    it("applies normalization before translation (diacritics, hamza-alef, teh marbuta) and reports it", async () => {
        const result = await translateToSignSequence("\u0623\u0646\u0627");
        expect(result.normalizedText).toBe("\u0627\u0646\u0627");
        expect(result.normalizationNotes.length).toBeGreaterThan(0);
    });

    it("empty/whitespace-only input produces an empty word list, not an error", async () => {
        const result = await translateToSignSequence("   ");
        expect(result.words).toHaveLength(0);
    });
});

describe("translateToSignSequence -- dedicated-sign mechanism (SYNTHETIC test fixture, not real ArSL data)", () => {
    const FAKE_TEST_ONLY_ENTRY = {
        text: "\u0645\u0631\u062d\u0628\u0627",
        landmarkSequence: [
            Array.from({ length: 21 }, (_, i) => ({ x: 0.1 * i, y: 0.2 * i, z: 0 })),
        ],
        provenance: {
            datasetName: "TEST-FIXTURE-NOT-REAL",
            source: "unit test only",
            license: "n/a",
            version: "test",
        },
    };

    beforeAll(() => {
        vi.stubGlobal(
            "fetch",
            vi.fn(async (url: string) => {
                if (url === "/models/arsl-31/letterLandmarks.json") {
                    return { ok: true, status: 200, json: async () => REAL_LETTER_LANDMARKS } as Response;
                }
                if (url === "/models/sign-vocabulary/dedicated-signs.json") {
                    return { ok: true, status: 200, json: async () => [FAKE_TEST_ONLY_ENTRY] } as Response;
                }
                throw new Error(`Unexpected fetch in test: ${url}`);
            })
        );
        vi.resetModules();
    });

    it("uses the dedicated sign when the registry has a matching entry, instead of spelling it", async () => {
        const { translateToSignSequence: translate } = await import("./signTranslator");
        const result = await translate("\u0645\u0631\u062d\u0628\u0627");
        const [word] = result.words;
        expect(word.steps).toHaveLength(1);
        expect(word.steps[0].source).toBe("dedicated");
        expect(word.steps[0].datasetName).toBe("TEST-FIXTURE-NOT-REAL");
        expect(word.steps[0].landmarkFrames).toHaveLength(1);
    });
});

describe("translateToSignSequence -- longest-match-first multi-word phrase matching (SYNTHETIC test fixtures, not real ArSL data)", () => {
    // Uses short, single-Arabic-character "words" purely as distinct, easy-to-
    // read tokens for testing the matching ALGORITHM -- none of this is real
    // KArSL data, dataset content, or a claim about any real sign. The real
    // registry (public/models/sign-vocabulary/dedicated-signs.json) stays
    // empty; these entries exist only inside this test file's fetch stub.
    const ALEF = "\u0627"; // ا -- plain Alef, unaffected by normalization (span-matching tests only)
    const BA = "\u0628"; // ب
    const JIM = "\u062c"; // ج
    const SEEN = "\u0633"; // س -- deliberately real (Sin) so it can fall back to letter spelling

    function fakeEntry(text: string, datasetName: string) {
        return {
            text,
            landmarkSequence: [Array.from({ length: 21 }, (_, i) => ({ x: 0.05 * i, y: 0.05 * i, z: 0 }))],
            provenance: { datasetName, source: "unit test only", license: "n/a", version: "test" },
        };
    }

    // Registry: "أ" (1 word), "أ ب" (2 words), "أ ب ج" (3 words) -- exactly the
    // longest-match-first scenario from the product spec.
    const PHRASE_REGISTRY = [
        fakeEntry(ALEF, "TEST-1WORD"),
        fakeEntry(`${ALEF} ${BA}`, "TEST-2WORD"),
        fakeEntry(`${ALEF} ${BA} ${JIM}`, "TEST-3WORD"),
    ];

    async function translateWithRegistry(text: string) {
        vi.stubGlobal(
            "fetch",
            vi.fn(async (url: string) => {
                if (url === "/models/arsl-31/letterLandmarks.json") {
                    return { ok: true, status: 200, json: async () => REAL_LETTER_LANDMARKS } as Response;
                }
                if (url === "/models/sign-vocabulary/dedicated-signs.json") {
                    return { ok: true, status: 200, json: async () => PHRASE_REGISTRY } as Response;
                }
                throw new Error(`Unexpected fetch in test: ${url}`);
            })
        );
        vi.resetModules();
        const { translateToSignSequence: translate } = await import("./signTranslator");
        return translate(text);
    }

    it("1. single-word dedicated sign", async () => {
        const result = await translateWithRegistry(ALEF);
        expect(result.words).toHaveLength(1);
        expect(result.words[0].word).toBe(ALEF);
        expect(result.words[0].steps[0].source).toBe("dedicated");
        expect(result.words[0].steps[0].datasetName).toBe("TEST-1WORD");
    });

    it("2. two-word dedicated phrase", async () => {
        const result = await translateWithRegistry(`${ALEF} ${BA}`);
        expect(result.words).toHaveLength(1);
        expect(result.words[0].word).toBe(`${ALEF} ${BA}`);
        expect(result.words[0].steps[0].source).toBe("dedicated");
        expect(result.words[0].steps[0].datasetName).toBe("TEST-2WORD");
    });

    it("3. three-word dedicated phrase", async () => {
        const result = await translateWithRegistry(`${ALEF} ${BA} ${JIM}`);
        expect(result.words).toHaveLength(1);
        expect(result.words[0].word).toBe(`${ALEF} ${BA} ${JIM}`);
        expect(result.words[0].steps[0].source).toBe("dedicated");
        expect(result.words[0].steps[0].datasetName).toBe("TEST-3WORD");
    });

    it("4. a phrase followed by another word", async () => {
        const result = await translateWithRegistry(`${ALEF} ${BA} ${JIM} ${SEEN}`);
        expect(result.words).toHaveLength(2);
        expect(result.words[0].word).toBe(`${ALEF} ${BA} ${JIM}`);
        expect(result.words[0].steps[0].source).toBe("dedicated");
        expect(result.words[1].word).toBe(SEEN);
        expect(result.words[1].steps[0].source).toBe("spelled-letter");
    });

    it("5. a word followed by a phrase", async () => {
        const result = await translateWithRegistry(`${SEEN} ${ALEF} ${BA} ${JIM}`);
        expect(result.words).toHaveLength(2);
        expect(result.words[0].word).toBe(SEEN);
        expect(result.words[0].steps[0].source).toBe("spelled-letter");
        expect(result.words[1].word).toBe(`${ALEF} ${BA} ${JIM}`);
        expect(result.words[1].steps[0].source).toBe("dedicated");
    });

    it("6. phrase not found (but a shorter dedicated word is) -> uses the shorter dedicated match, not spelling", async () => {
        // "أ ك" isn't in the registry, but "أ" alone is -- longest-match-first
        // should fall back from span=2 to span=1 and find the dedicated "أ".
        const KAF = "\u0643";
        const result = await translateWithRegistry(`${ALEF} ${KAF}`);
        expect(result.words).toHaveLength(2);
        expect(result.words[0].word).toBe(ALEF);
        expect(result.words[0].steps[0].source).toBe("dedicated");
        expect(result.words[1].word).toBe(KAF);
        expect(result.words[1].steps[0].source).toBe("spelled-letter");
    });

    it("7. a word with no dedicated entry at all falls back to real letter-by-letter spelling", async () => {
        const result = await translateWithRegistry(SEEN);
        expect(result.words).toHaveLength(1);
        expect(result.words[0].steps[0].source).toBe("spelled-letter");
        expect(result.words[0].steps[0].landmarkFrames.length).toBeGreaterThan(0);
    });

    it("8. multiple phrases in the same sentence", async () => {
        const result = await translateWithRegistry(`${ALEF} ${BA} ${JIM} ${SEEN} ${ALEF} ${BA}`);
        expect(result.words).toHaveLength(3);
        expect(result.words[0].word).toBe(`${ALEF} ${BA} ${JIM}`);
        expect(result.words[0].steps[0].source).toBe("dedicated");
        expect(result.words[1].word).toBe(SEEN);
        expect(result.words[1].steps[0].source).toBe("spelled-letter");
        expect(result.words[2].word).toBe(`${ALEF} ${BA}`);
        expect(result.words[2].steps[0].source).toBe("dedicated");
        expect(result.words[2].steps[0].datasetName).toBe("TEST-2WORD");
    });

    it("9. longest-match-first: 'أ ب ج' with أ/أب/أبج all registered chooses the full 3-word phrase, never the shorter ones", async () => {
        const result = await translateWithRegistry(`${ALEF} ${BA} ${JIM}`);
        expect(result.words).toHaveLength(1); // NOT 2 or 3 separate matches
        expect(result.words[0].steps[0].datasetName).toBe("TEST-3WORD");
    });

    it("10. normal single-letter fallback spelling is unaffected by the presence of a phrase registry (existing behavior preserved)", async () => {
        const result = await translateWithRegistry("\u0627\u0646\u0627"); // انا, none of which is in PHRASE_REGISTRY
        expect(result.words).toHaveLength(1);
        expect(result.words[0].steps).toHaveLength(3);
        for (const step of result.words[0].steps) {
            expect(step.source).toBe("spelled-letter");
        }
    });

    it("11. registry entries with un-normalized text (e.g. hamza-alef, as real Arabic labels often are) still match normalized input", async () => {
        const HAMZA_ALEF = "\u0623"; // أ -- normalizeArabicText maps this to plain ا
        const hamzaEntry = fakeEntry(HAMZA_ALEF, "TEST-HAMZA");
        vi.stubGlobal(
            "fetch",
            vi.fn(async (url: string) => {
                if (url === "/models/arsl-31/letterLandmarks.json") {
                    return { ok: true, status: 200, json: async () => REAL_LETTER_LANDMARKS } as Response;
                }
                if (url === "/models/sign-vocabulary/dedicated-signs.json") {
                    return { ok: true, status: 200, json: async () => [hamzaEntry] } as Response;
                }
                throw new Error(`Unexpected fetch in test: ${url}`);
            })
        );
        vi.resetModules();
        const { translateToSignSequence: translate } = await import("./signTranslator");

        // Input is plain Alef (ا) -- what speech recognition + normalization
        // would actually produce -- and must still match the hamza-form entry.
        const result = await translate(ALEF);
        expect(result.words).toHaveLength(1);
        expect(result.words[0].steps[0].source).toBe("dedicated");
        expect(result.words[0].steps[0].datasetName).toBe("TEST-HAMZA");
        expect(result.words[0].word).toBe(ALEF);
    });
});