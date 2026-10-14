import { Bone, Object3D, Quaternion, Vector3 } from "three";
import { HAND_BONES } from "./handRig";
import type { FingerName } from "./handRig";

/**
 * GLB HUMANOID RIG BINDING.
 *
 * Loads a rigged humanoid .glb and works out which of its bones correspond
 * to MediaPipe's 21 hand landmarks, for BOTH hands, without the caller
 * having to hand-write a bone map.
 *
 * WHY AUTO-DETECTION RATHER THAN A HARDCODED MAP: rigged humanoids in the
 * wild use a handful of well-known naming conventions and they disagree on
 * everything except structure. Mixamo emits "mixamorig:LeftHandIndex1",
 * Ready Player Me "LeftHandIndex1", VRM "J_Bip_L_Index1", Blender/Rigify
 * "hand.L" / "f_index.01.L", Unity-style rigs "Index Proximal.L". Hardcoding
 * one convention means any other model silently animates nothing. The
 * matcher below normalises a bone name and scores it against side, finger
 * and joint-depth tokens, so a model only has to be structurally sane.
 *
 * NOTHING HERE FABRICATES MOTION. If a bone cannot be found, it is reported
 * as missing in the validation result and simply never driven; the UI shows
 * the rig as invalid rather than pretending the sign was performed.
 */

export const AVATAR_GLB_URL = "/models/avatar/signer.glb";

export type HandSide = "left" | "right";

/** Joint depth along a finger: 0 = metacarpal/CMC, 1 = proximal, 2 = middle,
 * 3 = distal. MediaPipe's per-finger chain is exactly four segments deep. */
export type JointDepth = 0 | 1 | 2 | 3;

export interface BoundHandBones {
    /** Wrist / hand root bone. */
    wrist: Object3D | null;
    /** handRig bone name -> the GLB bone that plays that role. */
    bones: Map<string, Object3D>;
    /** Rest-pose local quaternion of each bound bone, captured at load time.
     * Retargeting is applied RELATIVE to this, so a model authored in an
     * A-pose or with non-identity rest rotations still works. */
    restLocal: Map<string, Quaternion>;
    /** Rest direction of each bone in its own parent's space, measured from
     * the model itself rather than assumed to be +Y. */
    restDirection: Map<string, Vector3>;
}

export interface BoundAvatar {
    root: Object3D;
    hands: Record<HandSide, BoundHandBones>;
    /** Upper-arm and forearm per side, for IK. Null when not found. */
    arms: Record<HandSide, { upperArm: Object3D | null; forearm: Object3D | null }>;
    head: Object3D | null;
    validation: RigValidation;
}

export interface RigValidation {
    ok: boolean;
    totalBonesInModel: number;
    boundPerSide: Record<HandSide, number>;
    missing: string[];
    /** Names actually matched, so a mis-binding is auditable rather than
     * showing up only as a strangely-posed hand. */
    matchedNames: Record<string, string>;
    notes: string[];
}

const FINGER_TOKENS: Record<FingerName, string[]> = {
    thumb: ["thumb"],
    index: ["index", "point"],
    middle: ["middle", "mid"],
    ring: ["ring"],
    pinky: ["pinky", "little"],
};

/** A single-letter side marker must be its OWN segment, delimited by a
 * separator or a string boundary. A naive substring test is wrong in a way
 * that is easy to miss: "_l" matches inside "J_Bip_R_LowerArm", which bound
 * the RIGHT forearm to the left side and left the right arm with no forearm
 * at all. Caught by the VRM-convention test. */
const LEFT_SEGMENT = /(^|[._\-|:])l([._\-|:]|$)/;
const RIGHT_SEGMENT = /(^|[._\-|:])r([._\-|:]|$)/;

function normalise(name: string): string {
    return name.toLowerCase().replace(/^mixamorig:?/, "").replace(/\s+/g, "");
}

export function sideOf(raw: string): HandSide | null {
    const n = normalise(raw);
    // Whole words first, then single-letter segments.
    if (n.includes("left")) return "left";
    if (n.includes("right")) return "right";
    if (LEFT_SEGMENT.test(n)) return "left";
    if (RIGHT_SEGMENT.test(n)) return "right";
    return null;
}

function fingerOf(raw: string): FingerName | null {
    const n = normalise(raw);
    for (const [finger, tokens] of Object.entries(FINGER_TOKENS) as Array<[FingerName, string[]]>) {
        for (const t of tokens) if (n.includes(t)) return finger;
    }
    return null;
}

/**
 * Extracts the joint depth from a bone name. Conventions differ:
 *  - Mixamo / RPM append 1,2,3 ("Index1" = proximal .. "Index3" = distal)
 *  - Rigify uses ".01" / ".02" / ".03"
 *  - Some rigs spell it out ("Proximal", "Intermediate", "Distal")
 * Returns null when no depth marker is present.
 */
export function jointDepthOf(raw: string, finger: FingerName): JointDepth | null {
    const n = normalise(raw);
    if (n.includes("metacarpal") || n.includes("carpal")) return 0;
    if (n.includes("proximal")) return finger === "thumb" ? 1 : 1;
    if (n.includes("intermediate") || n.includes("middlephalanx")) return 2;
    if (n.includes("distal")) return 3;
    if (n.includes("tip") || n.includes("end")) return 3;

    const m = n.match(/(\d+)\s*$/);
    if (m) {
        const d = parseInt(m[1], 10);
        // "Index1..Index4": 1 is the knuckle-most driven bone. Map 1->1, 2->2,
        // 3->3 and treat a 4th as the tip (not separately driven).
        if (d >= 1 && d <= 3) return d as JointDepth;
        if (d === 4) return 3;
    }
    return null;
}

/** handRig bone name -> (finger, depth) so the two sides can be matched up. */
function handRigDepth(boneName: string): { finger: FingerName; depth: JointDepth } | null {
    const def = HAND_BONES.find((b) => b.name === boneName);
    if (!def) return null;
    const depth =
        def.name.endsWith("_meta") ? 0 : def.name.endsWith("_prox") ? 1 : def.name.endsWith("_mid") ? 2 : 3;
    return { finger: def.finger, depth: depth as JointDepth };
}

function isWristName(raw: string): boolean {
    const n = normalise(raw);
    if (fingerOf(raw)) return false; // "LeftHandIndex1" is not the wrist
    return n.includes("hand") || n.includes("wrist");
}

function isForearmName(raw: string): boolean {
    const n = normalise(raw);
    return n.includes("forearm") || n.includes("lowerarm") || n.includes("elbow");
}

function isUpperArmName(raw: string): boolean {
    const n = normalise(raw);
    return (n.includes("upperarm") || n.includes("arm") || n.includes("shoulder")) && !isForearmName(raw);
}

/**
 * Walks a loaded GLB scene and binds it to the MediaPipe hand topology.
 * Pure function over an Object3D tree, so it is unit-testable against a
 * synthetic skeleton without loading a real file.
 */
export function bindAvatarRig(root: Object3D): BoundAvatar {
    const all: Object3D[] = [];
    root.traverse((o) => {
        if ((o as Bone).isBone || o.type === "Bone") all.push(o);
    });
    // Some exporters emit plain Object3D nodes for joints.
    if (all.length === 0) root.traverse((o) => all.push(o));

    const hands: Record<HandSide, BoundHandBones> = {
        left: { wrist: null, bones: new Map(), restLocal: new Map(), restDirection: new Map() },
        right: { wrist: null, bones: new Map(), restLocal: new Map(), restDirection: new Map() },
    };
    const arms: Record<HandSide, { upperArm: Object3D | null; forearm: Object3D | null }> = {
        left: { upperArm: null, forearm: null },
        right: { upperArm: null, forearm: null },
    };
    let head: Object3D | null = null;
    const matchedNames: Record<string, string> = {};
    const notes: string[] = [];

    for (const bone of all) {
        const raw = bone.name;
        if (!raw) continue;
        const n = normalise(raw);
        if (!head && n.includes("head") && !n.includes("headtop")) head = bone;

        const side = sideOf(raw);
        if (!side) continue;

        const finger = fingerOf(raw);
        if (finger) {
            const depth = jointDepthOf(raw, finger);
            if (depth === null) continue;
            const target = HAND_BONES.find((b) => {
                const d = handRigDepth(b.name);
                return d && d.finger === finger && d.depth === depth;
            });
            if (target && !hands[side].bones.has(target.name)) {
                hands[side].bones.set(target.name, bone);
                matchedNames[`${side}.${target.name}`] = raw;
            }
            continue;
        }

        if (isWristName(raw) && !hands[side].wrist) {
            hands[side].wrist = bone;
            matchedNames[`${side}.wrist`] = raw;
        } else if (isForearmName(raw) && !arms[side].forearm) {
            arms[side].forearm = bone;
            matchedNames[`${side}.forearm`] = raw;
        } else if (isUpperArmName(raw) && !arms[side].upperArm) {
            arms[side].upperArm = bone;
            matchedNames[`${side}.upperArm`] = raw;
        }
    }

    // Capture the model's REST pose. Retargeting composes onto this instead of
    // assuming an identity rest, so an A-pose or T-pose model both work.
    for (const side of ["left", "right"] as HandSide[]) {
        for (const [name, bone] of hands[side].bones) {
            hands[side].restLocal.set(name, bone.quaternion.clone());
            const child = bone.children.find((c) => (c as Bone).isBone || c.type === "Bone");
            const dir = child
                ? child.position.clone().normalize()
                : new Vector3(0, 1, 0);
            hands[side].restDirection.set(name, dir.lengthSq() > 1e-9 ? dir : new Vector3(0, 1, 0));
        }
        if (hands[side].wrist) {
            hands[side].restLocal.set("wrist", hands[side].wrist!.quaternion.clone());
        }
    }

    const missing: string[] = [];
    const boundPerSide: Record<HandSide, number> = { left: 0, right: 0 };
    for (const side of ["left", "right"] as HandSide[]) {
        boundPerSide[side] = hands[side].bones.size;
        if (!hands[side].wrist) missing.push(`${side}.wrist`);
        // Metacarpals are frequently absent in humanoid rigs (most riggers
        // start the finger chain at the knuckle). That is expected and NOT an
        // error: the retargeter folds the metacarpal rotation into the wrist.
        for (const def of HAND_BONES) {
            const d = handRigDepth(def.name)!;
            if (d.depth === 0) continue;
            if (!hands[side].bones.has(def.name)) missing.push(`${side}.${def.name}`);
        }
    }

    const perSideDrivable = HAND_BONES.filter((b) => handRigDepth(b.name)!.depth !== 0).length;
    const ok =
        hands.left.wrist !== null &&
        hands.right.wrist !== null &&
        boundPerSide.left >= perSideDrivable * 0.8 &&
        boundPerSide.right >= perSideDrivable * 0.8;

    if (!ok) {
        notes.push(
            "Rig binding incomplete. The model needs individually named finger bones for BOTH hands " +
            "(e.g. LeftHandIndex1/2/3 or J_Bip_L_Index1/2/3). Bones that could not be identified are " +
            "listed in `missing` and are never driven."
        );
    }
    if (all.length === 0) notes.push("No bones found in the model at all -- is it actually rigged?");

    return {
        root,
        hands,
        arms,
        head,
        validation: { ok, totalBonesInModel: all.length, boundPerSide, missing, matchedNames, notes },
    };
}

/**
 * Applies a solved local rotation to a bound GLB bone, composed onto that
 * bone's captured rest pose.
 *
 * The retargeter solves rotations for a canonical rig whose bones point
 * along local +Y. A GLB bone's rest direction is whatever the artist
 * authored, so the solved rotation is first re-expressed in that frame:
 * restLocal * align(+Y -> restDirection)^-1 * solved * align(...). Without
 * this, a model whose fingers rest along -X would bend sideways.
 */
export function applySolvedRotation(
    hand: BoundHandBones,
    boneName: string,
    solved: Quaternion,
    out?: Quaternion
): boolean {
    const bone = hand.bones.get(boneName);
    if (!bone) return false;
    const rest = hand.restLocal.get(boneName) ?? new Quaternion();
    const restDir = hand.restDirection.get(boneName) ?? new Vector3(0, 1, 0);

    const align = new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), restDir);
    const q = (out ?? new Quaternion())
        .copy(rest)
        .multiply(align)
        .multiply(solved)
        .multiply(align.clone().invert());

    bone.quaternion.copy(q);
    return true;
}
