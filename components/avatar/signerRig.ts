import {
    Bone,
    CapsuleGeometry,
    Color,
    Group,
    Mesh,
    MeshStandardMaterial,
    Object3D,
    Quaternion,
    SphereGeometry,
    Vector3,
} from "three";
import { HAND_BONES, HAND_UNIT } from "@/lib/avatar/handRig";
import type { BoneLengths } from "@/lib/avatar/retarget";
import type { HandSide } from "@/lib/avatar/glbRig";

/**
 * PROCEDURAL FALLBACK SIGNER.
 *
 * This is the avatar used when no rigged .glb is present at
 * public/models/avatar/signer.glb. It is a real Object3D/Bone hierarchy with
 * solid geometry attached to the bones -- not spheres at landmark positions:
 * each finger segment is a capsule parented to a Bone and inherits whatever
 * rotation the retargeter writes.
 *
 * It is a rigid-segment rig, NOT a smooth-skinned character, and it will not
 * look photoreal. That is the honest trade: it ships with the repo, carries
 * no third-party licence, and guarantees both hands are fully rigged. Drop a
 * rigged humanoid GLB in and the app uses that instead (see glbRig.ts).
 *
 * BOTH ARMS ARE FULLY RIGGED HERE. Whether a hand actually moves depends on
 * whether the KArSL data for that sign contains that hand -- a hand with no
 * recorded data stays in its rest pose rather than being animated from
 * invented motion.
 */

export interface HandRigHandles {
    /** Wrist root -- the retargeter writes its quaternion here. */
    handRoot: Object3D;
    bones: Map<string, Bone>;
    upperArm: Object3D;
    forearm: Object3D;
    shoulderWorld: Vector3;
}

export interface SignerRigHandles {
    root: Group;
    hands: Record<HandSide, HandRigHandles>;
    upperArmLength: number;
    forearmLength: number;
    dispose(): void;

    // --- Back-compat accessors for the single-hand call sites and tests
    // that predate two-hand support. They alias the RIGHT hand. ---
    handRoot: Object3D;
    bones: Map<string, Bone>;
    upperArm: Object3D;
    forearm: Object3D;
    shoulderWorld: Vector3;
}

const SKIN = new Color("#d8b9a0");
const CLOTH = new Color("#2f6f8f");
const JOINT = new Color("#c79f84");
const EYE = new Color("#20303c");
const HAIR = new Color("#3a2b24");

export const BODY = {
    headRadius: 0.098,
    headY: 1.63,
    neckY: 1.44,
    neckLength: 0.1,
    torsoBottomY: 0.94,
    torsoHeight: 0.52,
    torsoRadius: 0.145,
    shoulderY: 1.38,
    shoulderX: 0.175,
    upperArm: 0.29,
    forearm: 0.26,
    /** Orbit-controls look-at height: mid-chest, where signing happens. */
    chestTarget: 1.24,
} as const;

/**
 * A capsule spanning 0..length along its own local +Y.
 *
 * Returns a GROUP, not the Mesh: CapsuleGeometry is centred on the origin, so
 * the mesh must be pushed up by length/2. If that offset lived on the mesh,
 * any caller doing `part.position.set(...)` would silently overwrite it. That
 * exact bug detached the head from the torso in an earlier build.
 */
function capsule(
    length: number,
    radius: number,
    color: Color,
    materials: MeshStandardMaterial[],
    geometries: Array<{ dispose(): void }>
): Group {
    const mat = new MeshStandardMaterial({ color, roughness: 0.72, metalness: 0.02 });
    materials.push(mat);
    const geo = new CapsuleGeometry(radius, Math.max(length - radius * 2, 0.001), 6, 14);
    geometries.push(geo);
    const mesh = new Mesh(geo, mat);
    mesh.castShadow = true;
    mesh.position.y = length / 2;
    const group = new Group();
    group.add(mesh);
    return group;
}

function sphere(
    radius: number,
    color: Color,
    materials: MeshStandardMaterial[],
    geometries: Array<{ dispose(): void }>
): Mesh {
    const mat = new MeshStandardMaterial({ color, roughness: 0.68, metalness: 0.02 });
    materials.push(mat);
    const geo = new SphereGeometry(radius, 20, 14);
    geometries.push(geo);
    const mesh = new Mesh(geo, mat);
    mesh.castShadow = true;
    return mesh;
}

function buildHand(
    parent: Object3D,
    boneLengths: BoneLengths | undefined,
    materials: MeshStandardMaterial[],
    geometries: Array<{ dispose(): void }>
): { handRoot: Object3D; bones: Map<string, Bone> } {
    const handRoot = new Object3D();
    handRoot.position.set(0, BODY.forearm, 0);
    parent.add(handRoot);

    const palm = capsule(HAND_UNIT * 0.95, HAND_UNIT * 0.42, SKIN, materials, geometries);
    palm.scale.set(1.0, 1, 0.45);
    handRoot.add(palm);
    handRoot.add(sphere(HAND_UNIT * 0.34, JOINT, materials, geometries));

    const bones = new Map<string, Bone>();
    for (const def of HAND_BONES) {
        const bone = new Bone();
        const boneParent = def.parent === null ? handRoot : bones.get(def.parent)!;
        if (def.parent !== null) {
            const parentDef = HAND_BONES.find((b) => b.name === def.parent)!;
            const parentLength = (boneLengths?.[parentDef.name] ?? parentDef.length) * HAND_UNIT;
            bone.position.set(0, parentLength, 0);
        }
        boneParent.add(bone);
        bones.set(def.name, bone);

        const length = (boneLengths?.[def.name] ?? def.length) * HAND_UNIT;
        const radius = def.finger === "thumb" ? HAND_UNIT * 0.135 : HAND_UNIT * 0.115;
        bone.add(
            capsule(
                Math.max(length, HAND_UNIT * 0.08),
                radius * (def.joint === "tip" ? 0.85 : 1),
                SKIN,
                materials,
                geometries
            )
        );
        if (def.joint !== "CMC" && def.parent !== null) {
            bone.add(sphere(radius * 1.05, JOINT, materials, geometries));
        }
    }

    return { handRoot, bones };
}

function buildArm(
    root: Group,
    side: HandSide,
    boneLengths: BoneLengths | undefined,
    materials: MeshStandardMaterial[],
    geometries: Array<{ dispose(): void }>
): HandRigHandles {
    const sx = side === "right" ? BODY.shoulderX : -BODY.shoulderX;

    const shoulder = new Object3D();
    shoulder.position.set(sx, BODY.shoulderY, 0);
    root.add(shoulder);
    shoulder.add(sphere(0.052, CLOTH, materials, geometries));

    const upperArm = new Object3D();
    shoulder.add(upperArm);
    upperArm.add(capsule(BODY.upperArm, 0.044, SKIN, materials, geometries));

    const forearm = new Object3D();
    forearm.position.set(0, BODY.upperArm, 0);
    upperArm.add(forearm);
    forearm.add(sphere(0.04, SKIN, materials, geometries));
    forearm.add(capsule(BODY.forearm, 0.039, SKIN, materials, geometries));

    const { handRoot, bones } = buildHand(forearm, boneLengths, materials, geometries);

    return { handRoot, bones, upperArm, forearm, shoulderWorld: new Vector3(sx, BODY.shoulderY, 0) };
}

/**
 * @param boneLengths per-sign calibrated proportions, applied to both hands.
 */
export function buildSignerRig(boneLengths?: BoneLengths): SignerRigHandles {
    const materials: MeshStandardMaterial[] = [];
    const geometries: Array<{ dispose(): void }> = [];
    const root = new Group();

    // ---- Body: static. The KArSL data contains NO torso, head or facial
    // motion, so none is animated.
    const torso = capsule(BODY.torsoHeight, BODY.torsoRadius, CLOTH, materials, geometries);
    torso.position.set(0, BODY.torsoBottomY, 0);
    torso.scale.set(1.05, 1, 0.6);
    root.add(torso);

    const hips = capsule(0.16, 0.14, CLOTH, materials, geometries);
    hips.position.set(0, BODY.torsoBottomY - 0.14, 0);
    hips.scale.set(1, 1, 0.62);
    root.add(hips);

    const neck = capsule(BODY.neckLength, 0.042, SKIN, materials, geometries);
    neck.position.set(0, BODY.neckY, 0);
    root.add(neck);

    const head = sphere(BODY.headRadius, SKIN, materials, geometries);
    head.position.set(0, BODY.headY, 0);
    head.scale.set(0.92, 1.12, 0.96);
    root.add(head);

    // Minimal facial features, so a viewer can tell which way the avatar
    // faces. Nothing here is animated: the dataset has no facial data.
    for (const s of [-1, 1]) {
        const eye = sphere(0.016, EYE, materials, geometries);
        eye.position.set(s * 0.036, BODY.headY + 0.015, BODY.headRadius * 0.9);
        root.add(eye);
    }
    const brow = capsule(0.05, 0.008, HAIR, materials, geometries);
    brow.position.set(0, BODY.headY + 0.055, BODY.headRadius * 0.78);
    brow.rotation.z = Math.PI / 2;
    brow.scale.set(1, 1.6, 1);
    root.add(brow);

    const hands: Record<HandSide, HandRigHandles> = {
        right: buildArm(root, "right", boneLengths, materials, geometries),
        left: buildArm(root, "left", boneLengths, materials, geometries),
    };

    return {
        root,
        hands,
        upperArmLength: BODY.upperArm,
        forearmLength: BODY.forearm,
        // Back-compat aliases (right hand).
        handRoot: hands.right.handRoot,
        bones: hands.right.bones,
        upperArm: hands.right.upperArm,
        forearm: hands.right.forearm,
        shoulderWorld: hands.right.shoulderWorld,
        dispose() {
            for (const m of materials) m.dispose();
            for (const g of geometries) g.dispose();
        },
    };
}

/** Puts one arm in a neutral resting pose beside the body. Used for a hand
 * the current sign has NO recorded data for -- resting is the honest
 * representation of "no motion was captured for this hand". */
export function restArm(hand: HandRigHandles, side: HandSide): void {
    hand.upperArm.quaternion.setFromAxisAngle(
        new Vector3(0, 0, 1),
        side === "right" ? Math.PI - 0.14 : Math.PI + 0.14
    );
    hand.forearm.quaternion.setFromAxisAngle(new Vector3(1, 0, 0), 0.22);
    hand.handRoot.quaternion.identity();
    for (const bone of hand.bones.values()) bone.quaternion.identity();
}

/**
 * Two-bone IK: orients the upper arm and bends the elbow so the wrist lands
 * on `target` (in root space).
 *
 * HONESTY: this arm pose is DERIVED, not recorded. KArSL stores hand
 * landmarks only -- no shoulder, no elbow. What is real is the wrist position
 * the target comes from; the elbow angle is the geometric consequence of
 * reaching it with fixed arm lengths, and the elbow's swivel about the
 * shoulder-wrist axis is a fixed choice.
 */
export function solveArmIK(
    rig: SignerRigHandles,
    target: Vector3,
    side: HandSide = "right"
): { reachable: boolean; elbowAngle: number } {
    const hand = rig.hands[side];
    const toTarget = target.clone().sub(hand.shoulderWorld);
    const distance = toTarget.length();
    const a = rig.upperArmLength;
    const b = rig.forearmLength;
    const maxReach = a + b;

    const reachable = distance <= maxReach;
    const clamped = Math.min(distance, maxReach * 0.999);

    const dir = toTarget.clone().normalize();
    const aim = new Quaternion().setFromUnitVectors(new Vector3(0, 1, 0), dir);

    const cosShoulder = (a * a + clamped * clamped - b * b) / (2 * a * clamped);
    const shoulderAngle = Math.acos(Math.min(1, Math.max(-1, cosShoulder)));
    const cosElbow = (a * a + b * b - clamped * clamped) / (2 * a * b);
    const elbowAngle = Math.acos(Math.min(1, Math.max(-1, cosElbow)));

    const swivelAxis = new Vector3().crossVectors(dir, new Vector3(0, 0, 1)).normalize();
    if (swivelAxis.lengthSq() < 1e-8) swivelAxis.set(1, 0, 0);

    const bend = new Quaternion().setFromAxisAngle(swivelAxis, -shoulderAngle);
    hand.upperArm.quaternion.copy(aim.clone().premultiply(bend));
    hand.forearm.quaternion.setFromAxisAngle(
        swivelAxis.clone().applyQuaternion(hand.upperArm.quaternion.clone().invert()),
        Math.PI - elbowAngle
    );

    return { reachable, elbowAngle };
}
