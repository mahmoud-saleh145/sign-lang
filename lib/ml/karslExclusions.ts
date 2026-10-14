/**
 * Loads the exclusion list that ml/datasets/audit_processed_karsl.py writes.
 *
 * Two kinds of take cannot honestly take part in the evaluation, and both are
 * named in that file rather than hard-coded here:
 *
 *   no-hand          MediaPipe never found a hand, so there is nothing to
 *                    match. 28 of these in the 432-sign corpus, all of them
 *                    in sign 0076 and sign 0105, both recorded at camera
 *                    angles the detector cannot work with.
 *   identity-switch  the wrist track jumps further in one frame than a hand
 *                    can move, which means the stored trajectory is very
 *                    likely one hand's path spliced onto the other's. 82 of
 *                    these, spread thinly over 47 signs.
 *
 * Neither is usable as a prototype either, so lib/ml/karslTakeSplit.ts drops
 * them from all three groups.
 *
 * THIS FILE IS NODE-ONLY on purpose. karslTakeSplit.ts stays pure so it can be
 * imported anywhere; the filesystem read lives here, where only scripts and
 * tests reach it.
 *
 * A MISSING FILE IS NOT AN ERROR. Returning an empty map means "exclude
 * nothing", which reproduces the pre-exclusion behaviour exactly. It IS
 * reported, because silently evaluating on takes that were meant to be
 * dropped is the failure this whole mechanism exists to prevent.
 */
import { readFileSync, existsSync } from "fs";

export const DEFAULT_EXCLUSIONS_PATH = "karsl-exclusions.json";

interface ExclusionFile {
    version?: number;
    generatedAt?: string;
    impossibleJumpThreshold?: number;
    signsWithoutSessionSplit?: string[];
    /** sourceSignId -> { take stem -> reason } */
    excludedTakes?: Record<string, Record<string, string>>;
}

export interface Exclusions {
    /** sourceSignId -> excluded take stems. Empty set for an unlisted sign. */
    bySign: Map<string, Set<string>>;
    /** Signs the audit found cannot be split by session, for reporting. */
    signsWithoutSessionSplit: Set<string>;
    takeCount: number;
    generatedAt: string | null;
    /** False when no exclusion file was found. */
    loaded: boolean;
}

export const NO_EXCLUSIONS: Exclusions = {
    bySign: new Map(),
    signsWithoutSessionSplit: new Set(),
    takeCount: 0,
    generatedAt: null,
    loaded: false,
};

export function excludedFor(exclusions: Exclusions, signId: string): ReadonlySet<string> {
    return exclusions.bySign.get(signId) ?? new Set<string>();
}

/**
 * Reads the exclusion list. Returns NO_EXCLUSIONS, with a warning, when the
 * file is absent or unreadable -- the caller then evaluates on everything,
 * which is a weaker measurement but never a wrong one.
 */
export function loadExclusions(path: string = DEFAULT_EXCLUSIONS_PATH): Exclusions {
    if (!existsSync(path)) {
        console.warn(
            `[karslExclusions] no ${path}; evaluating on every take, including any the ` +
            `audit flagged. Regenerate with: python ml/datasets/audit_processed_karsl.py ` +
            `--input processed-karsl-v2 --exclusions-out ${path}`
        );
        return NO_EXCLUSIONS;
    }

    let parsed: ExclusionFile;
    try {
        parsed = JSON.parse(readFileSync(path, "utf-8")) as ExclusionFile;
    } catch (error) {
        console.warn(`[karslExclusions] ${path} is not readable JSON (${String(error)}); ignoring it.`);
        return NO_EXCLUSIONS;
    }

    if (parsed.version !== 1) {
        console.warn(
            `[karslExclusions] ${path} is version ${String(parsed.version)}, expected 1; ignoring it.`
        );
        return NO_EXCLUSIONS;
    }

    const bySign = new Map<string, Set<string>>();
    let takeCount = 0;
    for (const [signId, takes] of Object.entries(parsed.excludedTakes ?? {})) {
        const stems = new Set(Object.keys(takes));
        if (stems.size === 0) continue;
        bySign.set(signId, stems);
        takeCount += stems.size;
    }

    return {
        bySign,
        signsWithoutSessionSplit: new Set(parsed.signsWithoutSessionSplit ?? []),
        takeCount,
        generatedAt: parsed.generatedAt ?? null,
        loaded: true,
    };
}
