/**
 * Registry of DEDICATED signs -- whole words OR multi-word phrases that
 * have their own single sign, as opposed to being spelled letter-by-letter.
 * An entry's `text` may contain spaces (e.g. "إسعافات أولية") -- this
 * registry does exact matching only; longest-match-first phrase search
 * across multiple words happens in signTranslator.ts, not here.
 *
 * ============================================================================
 * STATUS: EMPTY by default. Real entries come only from a real importer
 * (e.g. ml/training/import_karsl.py) writing to
 * public/models/sign-vocabulary/dedicated-signs.json. Never hand-author
 * entries into that file without verifiable dataset provenance.
 * ============================================================================
 */

export interface DatasetProvenance {
    datasetName: string;
    source: string;
    license: string;
    version: string;
}

export interface DedicatedSignEntry {
    /** The Arabic word or multi-word phrase this sign represents, normalized form. */
    text: string;
    landmarkSequence: Array<Array<{ x: number; y: number; z: number }>>;
    provenance: DatasetProvenance;
    /** Original source sign ID (e.g. the zero-padded KArSL folder name
     * "0093"), preserved for traceability back to the source video, distinct
     * from any canonicalized form used for matching during import. */
    sourceSignId?: string;
}

const REGISTRY_URL = "/models/sign-vocabulary/dedicated-signs.json";

let cachedRegistry: DedicatedSignEntry[] | null = null;

export async function loadDedicatedSignRegistry(): Promise<DedicatedSignEntry[]> {
    if (cachedRegistry) return cachedRegistry;
    const res = await fetch(REGISTRY_URL);
    if (!res.ok) {
        throw new Error(`Failed to load dedicated sign registry: ${res.status}`);
    }
    cachedRegistry = (await res.json()) as DedicatedSignEntry[];
    return cachedRegistry;
}

/** Looks up a dedicated sign for EXACT normalized text (single word or a
 * full multi-word phrase, matched verbatim). Returns null if none exists. */
export async function findDedicatedSign(normalizedText: string): Promise<DedicatedSignEntry | null> {
    const registry = await loadDedicatedSignRegistry();
    return registry.find((e) => e.text === normalizedText) ?? null;
}