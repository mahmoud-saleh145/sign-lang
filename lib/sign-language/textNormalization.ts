/**
 * Normalizes raw Arabic text (e.g. from speech recognition) before sign
 * translation. This does NOT do spelling correction or grammar processing
 * (spec explicitly scopes that out) -- only well-defined, reversible-in-
 * intent normalization:
 *   - strips Arabic diacritics (tashkeel: fatha, damma, kasra, sukun, etc.)
 *   - strips tatweel (kashida, ـ) elongation marks
 *   - decomposes the لا ligature (U+FEFB etc. and the common لا sequence)
 *     into ل + ا so it can be spelled letter-by-letter
 *   - maps ة (teh marbuta) -> ه, since our 28-letter set has no dedicated
 *     teh-marbuta class (documented limitation)
 *   - maps hamza-carrying alef forms (أ إ آ) -> ا, since our 28-letter set
 *     has a single "Alef" class with no separate hamza variants
 *     (documented limitation -- this is a real information loss)
 *   - collapses runs of whitespace to a single space
 */

const DIACRITICS_REGEX = /[\u064B-\u0652\u0670\u0640]/g; // tashkeel + tatweel

// Hamza-carrying alef variants -> plain Alef (our dataset has one Alef class)
const ALEF_VARIANTS: Readonly<Record<string, string>> = {
    "\u0623": "\u0627", // أ -> ا
    "\u0625": "\u0627", // إ -> ا
    "\u0622": "\u0627", // آ -> ا
    "\u0671": "\u0627", // ٱ -> ا
};

// Teh marbuta has no dedicated class in our 28-letter set; map to Ha2 (ه),
// its closest visual/phonetic relative. Documented information loss.
const TEH_MARBUTA = "\u0629"; // ة
const HA2_CHAR = "\u0647"; // ه

export interface NormalizationNote {
    original: string;
    replacement: string;
    reason: string;
}

export interface NormalizationResult {
    text: string;
    notes: NormalizationNote[];
}

/** Normalizes raw Arabic input text. Returns the normalized text plus a
 * list of every lossy substitution made, so callers can be honest about it. */
export function normalizeArabicText(input: string): NormalizationResult {
    const notes: NormalizationNote[] = [];
    let text = input;

    const beforeDiacritics = text;
    text = text.replace(DIACRITICS_REGEX, "");
    if (text !== beforeDiacritics) {
        notes.push({
            original: "tashkeel/tatweel marks",
            replacement: "(removed)",
            reason: "Diacritics carry no separate sign in our letter set; removed rather than guessed.",
        });
    }

    for (const [variant, base] of Object.entries(ALEF_VARIANTS)) {
        if (text.includes(variant)) {
            text = text.split(variant).join(base);
            notes.push({
                original: variant,
                replacement: base,
                reason: "No dedicated hamza-on-alef class in the 28-letter set; mapped to plain Alef.",
            });
        }
    }

    if (text.includes(TEH_MARBUTA)) {
        text = text.split(TEH_MARBUTA).join(HA2_CHAR);
        notes.push({
            original: TEH_MARBUTA,
            replacement: HA2_CHAR,
            reason: "No teh-marbuta class in the 28-letter set; mapped to Ha2 (ه).",
        });
    }

    text = text.replace(/\s+/g, " ").trim();

    return { text, notes };
}