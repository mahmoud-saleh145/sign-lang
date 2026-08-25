import { normalizeArabicText } from "./textNormalization";
import { loadDedicatedSignRegistry } from "./signRegistry";
import type { DedicatedSignEntry } from "./signRegistry";
import { fromArabicChar } from "@/lib/ml/arabicLabels";
import type { SignSequence, SignStep, WordTranslation } from "./types";
import type { HandLandmarks } from "@/types/landmarks";

/** Real per-letter representative landmark samples, keyed by model class name
 * (e.g. "Ba2"). Loaded once and injected -- see loadLetterLandmarks(). */
export type LetterLandmarkTable = Readonly<Record<string, HandLandmarks>>;

let cachedTable: LetterLandmarkTable | null = null;

/** Fetches public/models/arsl-31/letterLandmarks.json (real dataset rows,
 * see ml/training/export_letter_landmarks.py) and caches it. */
export async function loadLetterLandmarks(): Promise<LetterLandmarkTable> {
    if (cachedTable) return cachedTable;
    const res = await fetch("/models/arsl-31/letterLandmarks.json");
    if (!res.ok) {
        throw new Error(`Failed to load letter landmarks: ${res.status}`);
    }
    cachedTable = (await res.json()) as LetterLandmarkTable;
    return cachedTable;
}

function dedicatedStepFor(entry: DedicatedSignEntry, matchedText: string): SignStep {
    return {
        text: matchedText,
        source: "dedicated",
        landmarkFrames: entry.landmarkSequence.map((frame) =>
            frame.map((p) => ({ x: p.x, y: p.y, z: p.z }))
        ),
        datasetName: entry.provenance.datasetName,
    };
}

function spellWord(word: string, letterLandmarks: LetterLandmarkTable): SignStep[] {
    const steps: SignStep[] = [];
    for (const char of word) {
        const classId = fromArabicChar(char);
        const landmarks = classId ? letterLandmarks[classId] : undefined;
        if (classId && landmarks) {
            steps.push({ text: char, source: "spelled-letter", landmarkFrames: [landmarks] });
        } else {
            // A real character with no corresponding real sign data -- surfaced
            // explicitly, never dropped silently.
            steps.push({ text: char, source: "unsupported", landmarkFrames: [] });
        }
    }
    return steps;
}

/**
 * Translates normalized Arabic text into a full SignSequence using
 * LONGEST-MATCH-FIRST phrase matching against the dedicated registry.
 *
 * At each position in the (already whitespace-split) word list, this tries
 * progressively shorter word spans -- starting from the longest phrase
 * length that actually exists anywhere in the registry, down to a single
 * word -- and takes the first exact match found. This means a 3-word
 * dedicated phrase is always preferred over a 2-word or 1-word match
 * starting at the same position, and a 2-word phrase over a 1-word match,
 * etc. If no span at a position matches anything in the registry, that
 * single word falls back to letter-by-letter spelling with real alphabet
 * data, and the search continues from the next word.
 *
 * Nothing here is hardcoded to any specific phrase -- the registry (loaded
 * from the real, importer-populated public/models/sign-vocabulary/
 * dedicated-signs.json) is the only source of dedicated entries, whatever
 * they happen to be, single-word or multi-word.
 *
 * Registry entry text is normalized the same way input text is before
 * matching (diacritics stripped, hamza-alef variants collapsed, etc. -- see
 * textNormalization.ts) so a real label like "أهلاً" matches normalized
 * speech input "اهلا" correctly, rather than silently never matching
 * because the registry's raw text and the input's normalized text differ.
 */
export async function translateToSignSequence(rawText: string): Promise<SignSequence> {
    const { text, notes } = normalizeArabicText(rawText);
    const letterLandmarks = await loadLetterLandmarks();
    const registry = await loadDedicatedSignRegistry();

    const registryByText = new Map(
        registry.map((e) => [normalizeArabicText(e.text).text, e])
    );
    const maxPhraseWords = registry.reduce(
        (max, e) => Math.max(max, normalizeArabicText(e.text).text.split(" ").length),
        1
    );

    const inputWords = text.split(" ").filter((w) => w.length > 0);
    const words: WordTranslation[] = [];

    let i = 0;
    while (i < inputWords.length) {
        const upperBound = Math.min(maxPhraseWords, inputWords.length - i);
        let matchedSpan = 0;

        for (let span = upperBound; span >= 1; span--) {
            const candidate = inputWords.slice(i, i + span).join(" ");
            const entry = registryByText.get(candidate);
            if (entry) {
                words.push({ word: candidate, steps: [dedicatedStepFor(entry, candidate)] });
                matchedSpan = span;
                break;
            }
        }

        if (matchedSpan > 0) {
            i += matchedSpan;
            continue;
        }

        // No dedicated match at any span starting here: spell this one word.
        const word = inputWords[i];
        words.push({ word, steps: spellWord(word, letterLandmarks) });
        i += 1;
    }

    return { normalizedText: text, words, normalizationNotes: notes };
}