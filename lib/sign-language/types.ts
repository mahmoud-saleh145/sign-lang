import type { HandLandmarks } from "@/types/landmarks";

/**
 * How a SignStep's visual representation was determined. This distinction
 * is user-facing (spec: "Do NOT silently present alphabet spelling as a
 * native sign") -- the UI must be able to show it, not just log it.
 */
export type SignSource = "dedicated" | "spelled-letter" | "unsupported";

export interface SignStep {
    /** The Arabic text this step represents: a whole word (dedicated) or one letter (spelled). */
    text: string;
    source: SignSource;
    /** Real landmark frame(s) to render. Empty for "unsupported" steps. */
    landmarkFrames: HandLandmarks[];
    /** Only set for source === "dedicated": which dataset backs this sign. */
    datasetName?: string;
}

export interface WordTranslation {
    word: string;
    /** "dedicated" if one SignStep represents the whole word; "spelled-letter"
     * sequence if it had to be spelled out letter by letter; "unsupported" if
     * some/all letters have no real sign data at all. */
    steps: SignStep[];
}

export interface SignSequence {
    /** The normalized Arabic text this sequence was built from. */
    normalizedText: string;
    words: WordTranslation[];
    /** Every normalization change made before translation (see textNormalization.ts). */
    normalizationNotes: Array<{ original: string; replacement: string; reason: string }>;
}