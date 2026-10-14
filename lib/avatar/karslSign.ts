import type { HandLandmarks } from "@/types/landmarks";
import { loadDedicatedSignRegistry } from "@/lib/sign-language/signRegistry";
import type { SignStep } from "@/lib/sign-language/types";
import { calibrateBoneLengths } from "./retarget";
import type { BoneLengths } from "./retarget";
import type { HandFrame, HandednessInfo } from "@/lib/sign-language/signRegistry";

/**
 * ONE ANIMATABLE SIGN -- the single abstraction shared by every consumer
 * (VoiceToSign, the /dev/avatar page, and anything added later). Nothing
 * downstream of this needs to know whether a sign came from the KArSL
 * registry or from the alphabet fallback; it only needs frames and a
 * duration.
 *
 * The registry is the COMMON SOURCE OF TRUTH with the camera recognizer:
 * lib/ml/dynamicSignRecognizer.ts matches live landmarks against exactly the
 * same public/models/sign-vocabulary/dedicated-signs.json entries that
 * fromDedicatedStep() reads here. A sign the camera can recognize is a sign
 * the avatar can perform, by construction.
 */

/** Frame rate of the source KArSL videos. Confirmed from the extraction
 * output: every processed-karsl/*.json records "fps": 30.0. */
export const KARSL_SOURCE_FPS = 30;

/** How long a single-frame pose (an alphabet letter) is held. Letters have no
 * recorded duration -- there is only one real frame -- so this is a display
 * choice, and it is labelled as such wherever it is used. */
export const STATIC_POSE_HOLD_MS = 900;

export type AnimatableSource = "karsl" | "spelled-letter" | "unsupported";

/** Per-hand animation track. `frames` is aligned index-for-index with the
 * sign's shared timeline; a null entry means that hand was NOT detected in
 * that frame and is never filled in. */
export interface HandTrack {
    frames: Array<HandLandmarks | null>;
    /** How many frames actually contain this hand. */
    detectedFrames: number;
    boneLengths: BoneLengths;
}

export interface AnimatableSign {
    /** KArSL sign ID (e.g. "0071") when source === "karsl", else null. */
    signId: string | null;
    /** The Arabic word or complete multi-word phrase this sign represents.
     * Multi-word KArSL labels stay intact -- never split. */
    text: string;
    source: AnimatableSource;
    landmarkSequence: HandLandmarks[];
    frameCount: number;
    /** Real duration for a KArSL sign (frameCount / 30 fps). For a static
     * letter this is STATIC_POSE_HOLD_MS, which is a display choice, not a
     * recorded duration -- see `durationIsReal`. */
    durationMs: number;
    /** True only when durationMs came from the actual recorded frame count. */
    durationIsReal: boolean;
    /** Bone proportions calibrated to this specific recorded hand. */
    boneLengths: BoneLengths;
    /** Both hands on the sign's shared timeline, when the registry entry has
     * two-hand data. Null when the entry predates two-hand extraction --
     * which means "we only have single-hand data", NOT "this sign is
     * one-handed". The two cases are reported separately so the UI never
     * claims a one-handed performance it cannot back up. */
    tracks: Record<"left" | "right", HandTrack> | null;
    /** True only when the source data shows both hands together in a clear
     * majority of frames. */
    twoHanded: boolean;
    /** True when the entry carries no two-hand information at all. */
    handDataUnknown: boolean;
    handedness?: HandednessInfo;
    datasetName?: string;
}

function buildTracks(handFrames: HandFrame[] | undefined): Record<"left" | "right", HandTrack> | null {
    if (!handFrames || handFrames.length === 0) return null;
    const out = {} as Record<"left" | "right", HandTrack>;
    for (const side of ["left", "right"] as const) {
        const frames = handFrames.map((f) => (f[side] ? (f[side] as HandLandmarks) : null));
        const present = frames.filter((f): f is HandLandmarks => f !== null);
        out[side] = {
            frames,
            detectedFrames: present.length,
            // Calibrate on the frames that actually contain this hand.
            boneLengths: calibrateBoneLengths(present),
        };
    }
    return out;
}

function build(
    text: string,
    source: AnimatableSource,
    frames: HandLandmarks[],
    signId: string | null,
    datasetName?: string,
    handFrames?: HandFrame[],
    handedness?: HandednessInfo
): AnimatableSign {
    const tracks = buildTracks(handFrames);
    // The shared timeline is the two-hand timeline when we have one, because
    // that is the only frame index both hands agree on.
    const timelineLength = tracks ? tracks.left.frames.length : frames.length;
    const frameCount = timelineLength;
    const isTemporal = source === "karsl" && frameCount >= 2;
    return {
        signId,
        text,
        source,
        landmarkSequence: frames,
        frameCount,
        durationMs: isTemporal ? (frameCount / KARSL_SOURCE_FPS) * 1000 : STATIC_POSE_HOLD_MS,
        durationIsReal: isTemporal,
        boneLengths: calibrateBoneLengths(frames),
        tracks,
        twoHanded: Boolean(handedness?.twoHanded),
        handDataUnknown: source === "karsl" && !tracks,
        handedness,
        datasetName,
    };
}

/** Wraps one SignStep from the translator. The translator already decided
 * dedicated-vs-spelled; this only adds timing and rig calibration. */
export function fromSignStep(step: SignStep): AnimatableSign {
    if (step.source === "dedicated") {
        return build(step.text, "karsl", step.landmarkFrames, null, step.datasetName);
    }
    if (step.source === "spelled-letter") {
        return build(step.text, "spelled-letter", step.landmarkFrames, null);
    }
    return build(step.text, "unsupported", [], null);
}

/** Loads one registry sign directly by its KArSL sign ID. Used by the
 * camera-free /dev/avatar page so a real sequence can be replayed without
 * going through text matching at all. */
export async function loadKarslSignById(signId: string): Promise<AnimatableSign | null> {
    const registry = await loadDedicatedSignRegistry();
    const entry = registry.find((e) => e.sourceSignId === signId);
    if (!entry) return null;
    return build(
        entry.text,
        "karsl",
        entry.landmarkSequence.map((f) => f.map((p) => ({ x: p.x, y: p.y, z: p.z }))),
        entry.sourceSignId ?? null,
        entry.provenance.datasetName,
        entry.handFrames,
        entry.handedness
    );
}

export interface KarslSignSummary {
    signId: string;
    text: string;
    frameCount: number;
    durationMs: number;
    twoHanded: boolean;
    hasLeft: boolean;
    hasRight: boolean;
    handDataUnknown: boolean;
}

export async function listKarslSigns(): Promise<KarslSignSummary[]> {
    const registry = await loadDedicatedSignRegistry();
    return registry
        .filter((e) => e.sourceSignId)
        .map((e) => ({
            signId: e.sourceSignId!,
            text: e.text,
            frameCount: e.handFrames?.length ?? e.landmarkSequence.length,
            durationMs: ((e.handFrames?.length ?? e.landmarkSequence.length) / KARSL_SOURCE_FPS) * 1000,
            twoHanded: Boolean(e.handedness?.twoHanded),
            hasLeft: (e.handedness?.framesWithLeft ?? 0) > 0,
            hasRight: (e.handedness?.framesWithRight ?? 0) > 0,
            handDataUnknown: !e.handFrames,
        }));
}

/** Total wall-clock duration of a list of signs played back to back. */
export function totalDurationMs(signs: readonly AnimatableSign[]): number {
    return signs.reduce((sum, s) => sum + s.durationMs, 0);
}
