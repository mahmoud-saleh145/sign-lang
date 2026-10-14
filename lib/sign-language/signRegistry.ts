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

import type { SidedSequence, SideSource } from "@/types/landmarks";

export interface DatasetProvenance {
    datasetName: string;
    source: string;
    license: string;
    version: string;
}

/** One frame of the two-hand timeline. Either side may be null when that
 * hand was not detected in that frame -- never interpolated or copied. */
export interface HandFrame {
    left: Array<{ x: number; y: number; z: number }> | null;
    right: Array<{ x: number; y: number; z: number }> | null;
}

/** Which real recording a template's motion came from. */
export interface SourceTake {
    /** File name inside processed-karsl/<sourceSignId>/. */
    processedFile: string;
    /** The source video's file name without its extension. */
    sourceVideoStem: string;
    /** The KArSL folder the take came from (same as sourceSignId). */
    sourceFolder: string;
    /** How many takes of this sign the importer could choose between. */
    takesAvailable: number;
    /** Why this take was chosen over the others. */
    selectionReason: string;
}

export interface HandednessInfo {
    /** True when BOTH hands are present together in a clear majority of
     * frames. A second hand that flickers in briefly does not qualify. */
    twoHanded: boolean;
    framesWithLeft: number;
    framesWithRight: number;
    framesWithBoth: number;
    /**
     * LEGACY CONVENIENCE ONLY: which side the flat `landmarkSequence` carries.
     * It is NOT the identity of the sign. Side-aware matching reads
     * `handFrames`; nothing may branch on this to decide which hand a gesture
     * "is". See ml/training/handedness.py for why the old
     * detection-count-based choice was not an identity.
     */
    primaryHand: string;
    /**
     * How left/right was decided for this entry. "geometry" is a measurement
     * (the signer's right hand sits at the smaller image x in an un-mirrored
     * recording); "handedness-label" is a MediaPipe model output used when a
     * take never shows both hands at once; "unknown" means neither signal was
     * usable and the sides must not be trusted.
     */
    sideSource: SideSource;
    /** Agreement fraction of whichever vote decided. 0 when nothing decided. */
    sideConfidence: number;
    /** Frames in which both hands were present, i.e. usable by the geometry vote. */
    geometryFrames: number;
}

export interface DedicatedSignEntry {
    /** The Arabic word or multi-word phrase this sign represents, normalized form. */
    text: string;
    /** LEGACY, always present: the primary hand only, null frames dropped.
     * Every consumer written before two-hand support keeps working on this. */
    landmarkSequence: Array<Array<{ x: number; y: number; z: number }>>;
    /** Both hands on one shared timeline. Absent on entries imported before
     * two-hand extraction existed -- absence means "single-hand data", it
     * does NOT mean the sign is one-handed. */
    handFrames?: HandFrame[];
    handedness?: HandednessInfo;
    provenance: DatasetProvenance;
    /** Original source sign ID (e.g. the zero-padded KArSL folder name
     * "0093"), preserved for traceability back to the source video, distinct
     * from any canonicalized form used for matching during import. */
    sourceSignId?: string;
    /** Which real recording this template was built from. Absent on entries
     * imported before traceability existed, or from the per-frame-file
     * fallback path where there is no single source video. */
    sourceTake?: SourceTake;
}

/**
 * The entry's two hands as a shared timeline, in the form the side-aware
 * matcher and the avatar both consume. Entries that predate two-hand
 * extraction have no `handFrames`; their single stored hand is returned on the
 * side `primaryHand` names, or as `null` when even that is unknown -- it is
 * never placed on a side by guesswork.
 */
export function entryToSidedSequence(entry: DedicatedSignEntry): SidedSequence | null {
    if (entry.handFrames && entry.handFrames.length > 0) {
        return entry.handFrames.map((f) => ({ left: f.left, right: f.right }));
    }
    const legacy = entry.landmarkSequence;
    if (!legacy || legacy.length === 0) return null;
    const side = entry.handedness?.primaryHand;
    if (side !== "left" && side !== "right") return null;
    return legacy.map((frame) =>
        side === "left" ? { left: frame, right: null } : { left: null, right: frame }
    );
}

const REGISTRY_URL = "/models/sign-vocabulary/dedicated-signs.json";

let cachedRegistry: DedicatedSignEntry[] | null = null;

/**
 * Reconstructs the legacy `landmarkSequence` for an entry that only carries
 * `handFrames`.
 *
 * WHY: `landmarkSequence` is, by construction, exactly the primary hand's
 * frames with the empty ones dropped -- the same coordinates already present
 * inside `handFrames`. Writing both doubles that data in a file the browser
 * downloads whole (about 14 MB across 502 signs), so import_karsl.py omits it
 * and it is rebuilt here. Every existing consumer -- notably the camera DTW
 * recognizer in lib/ml/dynamicSignRecognizer.ts -- keeps seeing the field it
 * has always seen, so nothing downstream needed changing.
 *
 * Entries that predate two-hand extraction still carry a real
 * `landmarkSequence` and are passed through untouched.
 */
export function withLegacySequence(entry: DedicatedSignEntry): DedicatedSignEntry {
    if (entry.landmarkSequence && entry.landmarkSequence.length > 0) return entry;
    if (!entry.handFrames || entry.handFrames.length === 0) {
        return { ...entry, landmarkSequence: entry.landmarkSequence ?? [] };
    }
    const framesOn = (side: "left" | "right") =>
        entry.handFrames!
            .map((f) => f[side])
            .filter((f): f is Array<{ x: number; y: number; z: number }> => Boolean(f));

    const primary = entry.handedness?.primaryHand === "left" ? "left" : "right";
    const other = primary === "left" ? "right" : "left";
    const primaryFrames = framesOn(primary);
    const otherFrames = framesOn(other);

    // Fall back whenever the recorded primary side is DEGENERATE, not only when
    // it is empty. A side seen in a single frame is not a temporal sequence,
    // and handing one back is worse than handing back the hand that was
    // actually recorded: signs 0076 and 0089 in the real registry had
    // `primaryHand: "left"` with one left frame against fifty right ones, and
    // every legacy consumer saw a one-frame sign. The importer now derives
    // `primaryHand` from the frame counts so this should not arise, but the
    // loader must not depend on a field it does not control being right.
    const frames =
        primaryFrames.length >= 2 || primaryFrames.length >= otherFrames.length
            ? primaryFrames
            : otherFrames;
    return { ...entry, landmarkSequence: frames };
}

export async function loadDedicatedSignRegistry(): Promise<DedicatedSignEntry[]> {
    if (cachedRegistry) return cachedRegistry;
    const res = await fetch(REGISTRY_URL);
    if (!res.ok) {
        throw new Error(`Failed to load dedicated sign registry: ${res.status}`);
    }
    const raw = (await res.json()) as DedicatedSignEntry[];
    cachedRegistry = raw.map(withLegacySequence);
    return cachedRegistry;
}

/** Looks up a dedicated sign for EXACT normalized text (single word or a
 * full multi-word phrase, matched verbatim). Returns null if none exists. */
export async function findDedicatedSign(normalizedText: string): Promise<DedicatedSignEntry | null> {
    const registry = await loadDedicatedSignRegistry();
    return registry.find((e) => e.text === normalizedText) ?? null;
}