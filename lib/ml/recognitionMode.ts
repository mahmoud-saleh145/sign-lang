/**
 * The three independently testable recognition modes.
 *
 *  - "letter": ONLY the existing static per-frame Arabic letter model runs.
 *              The dynamic KArSL path is not stepped at all, so it can never
 *              interfere with the letter transcript. Behaviour here is
 *              byte-for-byte the pre-existing letter pipeline.
 *  - "word":   ONLY dynamic KArSL word/phrase recognition commits. The
 *              static model still runs so its raw label/confidence remain
 *              visible in telemetry, but nothing it produces is ever
 *              committed to the transcript or the Arabic text builder.
 *              A recognized sign is emitted as ONE complete registered
 *              string (multi-word KArSL labels such as "إسعافات أولية" stay
 *              intact) and is never decomposed into letters.
 *  - "both":   Both paths run. They write to two separate transcripts and
 *              two separate result fields, so neither can overwrite the
 *              other even if a letter commit and a sign match land on the
 *              same frame.
 */
export type RecognitionMode = "letter" | "word" | "both";

export const RECOGNITION_MODES: readonly RecognitionMode[] = ["letter", "word", "both"];

export function isLetterPathActive(mode: RecognitionMode): boolean {
    return mode === "letter" || mode === "both";
}

export function isWordPathActive(mode: RecognitionMode): boolean {
    return mode === "word" || mode === "both";
}
