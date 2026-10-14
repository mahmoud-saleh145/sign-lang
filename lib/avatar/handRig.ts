/**
 * Skeletal hand rig definition.
 *
 * The bone hierarchy mirrors MediaPipe's 21-landmark hand topology EXACTLY
 * (one bone per landmark-to-landmark segment, 20 bones under a wrist root),
 * which is what makes retargeting exact rather than approximate: every bone
 * has a defined source segment in the real KArSL landmark data, and no bone
 * needs a name-matching heuristic against some third-party rig.
 *
 * BONE CONVENTION: every bone's local +Y axis points down its own length,
 * and each bone sits at (0, parentLength, 0) in its parent's space. So a
 * bone's local rest direction is always (0, 1, 0), and retargeting reduces
 * to "rotate (0,1,0) onto the measured direction, expressed in parent
 * space". See lib/avatar/retarget.ts.
 *
 * BONE LENGTHS ARE MEASURED, NOT INVENTED. Every ratio below is the median
 * of that segment's length across 1,447 frames sampled from all 170 real
 * sequences in public/models/sign-vocabulary/dedicated-signs.json,
 * normalized by that frame's wrist->middle-MCP distance (so the unit is
 * "middle metacarpal = 1.0"). Reproduce with the script quoted in
 * docs/AVATAR_RETARGETING.md. The avatar therefore has the average
 * proportions of the actual dataset's hands.
 *
 * WHY FIXED LENGTHS AT ALL: retargeting drives bone ROTATION only. Bone
 * length stays constant, so the avatar's hand keeps a stable size and shape
 * while its pose comes entirely from the data. Rebuilding the hand from
 * raw landmark positions every frame would make the hand visibly grow and
 * shrink with MediaPipe's per-frame noise, and would be point-placement
 * rather than skeletal animation.
 */

export type FingerName = "thumb" | "index" | "middle" | "ring" | "pinky";

export interface HandBoneDef {
    name: string;
    /** Parent bone name, or null for a direct child of the wrist root. */
    parent: string | null;
    /** MediaPipe landmark index this bone starts at. */
    fromLandmark: number;
    /** MediaPipe landmark index this bone points to. */
    toLandmark: number;
    /** Length in units of the wrist->middle-MCP distance. Median of real data. */
    length: number;
    finger: FingerName;
    /** Human-readable joint name, matching standard hand anatomy. */
    joint: string;
}

/** Wrist->middle-MCP length in avatar world units. Everything else scales
 * from this, so changing it resizes the whole hand consistently. */
export const HAND_UNIT = 0.085;

export const HAND_BONES: readonly HandBoneDef[] = [
    // Thumb: CMC -> MCP -> IP -> tip
    { name: "thumb_meta", parent: null, fromLandmark: 0, toLandmark: 1, length: 0.4585, finger: "thumb", joint: "CMC" },
    { name: "thumb_prox", parent: "thumb_meta", fromLandmark: 1, toLandmark: 2, length: 0.4616, finger: "thumb", joint: "MCP" },
    { name: "thumb_mid", parent: "thumb_prox", fromLandmark: 2, toLandmark: 3, length: 0.3734, finger: "thumb", joint: "IP" },
    { name: "thumb_dist", parent: "thumb_mid", fromLandmark: 3, toLandmark: 4, length: 0.2931, finger: "thumb", joint: "tip" },

    // Index: MCP -> PIP -> DIP -> tip
    { name: "index_meta", parent: null, fromLandmark: 0, toLandmark: 5, length: 1.0435, finger: "index", joint: "MCP" },
    { name: "index_prox", parent: "index_meta", fromLandmark: 5, toLandmark: 6, length: 0.5355, finger: "index", joint: "PIP" },
    { name: "index_mid", parent: "index_prox", fromLandmark: 6, toLandmark: 7, length: 0.3048, finger: "index", joint: "DIP" },
    { name: "index_dist", parent: "index_mid", fromLandmark: 7, toLandmark: 8, length: 0.2335, finger: "index", joint: "tip" },

    { name: "middle_meta", parent: null, fromLandmark: 0, toLandmark: 9, length: 1.0, finger: "middle", joint: "MCP" },
    { name: "middle_prox", parent: "middle_meta", fromLandmark: 9, toLandmark: 10, length: 0.5755, finger: "middle", joint: "PIP" },
    { name: "middle_mid", parent: "middle_prox", fromLandmark: 10, toLandmark: 11, length: 0.3290, finger: "middle", joint: "DIP" },
    { name: "middle_dist", parent: "middle_mid", fromLandmark: 11, toLandmark: 12, length: 0.2408, finger: "middle", joint: "tip" },

    { name: "ring_meta", parent: null, fromLandmark: 0, toLandmark: 13, length: 0.9501, finger: "ring", joint: "MCP" },
    { name: "ring_prox", parent: "ring_meta", fromLandmark: 13, toLandmark: 14, length: 0.5352, finger: "ring", joint: "PIP" },
    { name: "ring_mid", parent: "ring_prox", fromLandmark: 14, toLandmark: 15, length: 0.3002, finger: "ring", joint: "DIP" },
    { name: "ring_dist", parent: "ring_mid", fromLandmark: 15, toLandmark: 16, length: 0.2223, finger: "ring", joint: "tip" },

    { name: "pinky_meta", parent: null, fromLandmark: 0, toLandmark: 17, length: 0.9227, finger: "pinky", joint: "MCP" },
    { name: "pinky_prox", parent: "pinky_meta", fromLandmark: 17, toLandmark: 18, length: 0.4285, finger: "pinky", joint: "PIP" },
    { name: "pinky_mid", parent: "pinky_prox", fromLandmark: 18, toLandmark: 19, length: 0.2254, finger: "pinky", joint: "DIP" },
    { name: "pinky_dist", parent: "pinky_mid", fromLandmark: 19, toLandmark: 20, length: 0.1831, finger: "pinky", joint: "tip" },
] as const;

/** Bones in parent-before-child order. HAND_BONES is already authored that
 * way; this asserts it rather than trusting it. */
export function bonesInHierarchyOrder(): readonly HandBoneDef[] {
    const seen = new Set<string>();
    for (const b of HAND_BONES) {
        if (b.parent !== null && !seen.has(b.parent)) {
            throw new Error(`HAND_BONES is not in hierarchy order: ${b.name} precedes its parent ${b.parent}`);
        }
        seen.add(b.name);
    }
    return HAND_BONES;
}

export function boneByName(name: string): HandBoneDef | undefined {
    return HAND_BONES.find((b) => b.name === name);
}

/** Every bone whose `joint` the user listed as required coverage. */
export const REQUIRED_JOINT_COVERAGE: Readonly<Record<FingerName, readonly string[]>> = {
    thumb: ["CMC", "MCP", "IP", "tip"],
    index: ["MCP", "PIP", "DIP", "tip"],
    middle: ["MCP", "PIP", "DIP", "tip"],
    ring: ["MCP", "PIP", "DIP", "tip"],
    pinky: ["MCP", "PIP", "DIP", "tip"],
};
