import { Matrix4, Quaternion, Vector3 } from "three";
import { HAND_BONES, HAND_UNIT, bonesInHierarchyOrder } from "./handRig";
import type { HandLandmarks } from "@/types/landmarks";

/**
 * LANDMARK -> BONE ROTATION RETARGETING.
 *
 * Input: one frame of REAL MediaPipe/KArSL landmarks (21 points).
 * Output: a local-space quaternion per bone, plus a wrist root orientation
 * and position. Nothing is placed at a landmark coordinate; every bone gets
 * an ORIENTATION computed from the direction between two real landmarks,
 * and the avatar's own fixed bone lengths do the rest.
 *
 * ============================================================================
 * ALGORITHM
 * ============================================================================
 * 1. Wrist root orientation: build an orthonormal basis from two real
 *    directions in the palm --
 *      up      = normalize(L9 - L0)     wrist -> middle MCP
 *      across  = normalize(L17 - L5)    index MCP -> pinky MCP
 *      forward = normalize(up x across) palm normal
 *      side    = up x forward           re-orthogonalized (right-handed)
 *    The rotation matrix [side, up, forward] is the hand's world orientation.
 *    Using two measured directions (rather than one) is what fixes the roll
 *    of the palm; a single direction leaves it undetermined.
 *
 * 2. Each bone, parent before child:
 *      worldDir  = normalize(L[to] - L[from])
 *      localDir  = parentWorldQuat^-1 * worldDir
 *      local q   = rotation taking (0,1,0) onto localDir
 *      thisWorld = parentWorldQuat * q
 *    Because every bone's rest direction is its own local +Y (see
 *    handRig.ts), `local q` is exactly the value to assign to that bone's
 *    quaternion in a Three.js hierarchy.
 *
 * ============================================================================
 * COORDINATE CONVERSION AND ITS LIMITS -- stated, not glossed over
 * ============================================================================
 * MediaPipe image landmarks use x right, y DOWN, z toward the camera-ish,
 * all in image-normalized units. Three.js uses y UP. So y is negated and z
 * is negated to get a right-handed, y-up frame.
 *
 * The z channel of THIS dataset is weak. Measured across the real registry:
 * within-frame z span is p50 = 0.026 while the hand scale (wrist->middle
 * MCP) is p50 = 0.066 -- depth varies over roughly 40% of hand size, and
 * MediaPipe's image-landmark z is only loosely metric to begin with (the
 * extraction stored `landmarks`, not `worldLandmarks`). The reconstruction
 * is therefore well-conditioned in the image plane and weakly conditioned in
 * depth. Poses distinguished mainly by depth will look flatter than the
 * original. This is a property of the stored data, not something the
 * retargeter can recover.
 */

const REST_DIR = new Vector3(0, 1, 0);

export interface BoneRotation {
    name: string;
    /** Local-space quaternion to assign to this bone, as [x, y, z, w]. */
    quaternion: [number, number, number, number];
}

export interface HandPoseSolution {
    /** Wrist root orientation in avatar world space, as [x, y, z, w]. */
    rootQuaternion: [number, number, number, number];
    /** Wrist position in avatar world space, derived from the REAL recorded
     * wrist landmark mapped into the signing volume (see landmarkToAvatar). */
    rootPosition: [number, number, number];
    bones: BoneRotation[];
    /** True when the frame was unusable (degenerate palm) and a rest pose was
     * returned instead of a solved one. Surfaced rather than hidden. */
    degenerate: boolean;
}

/** Converts one MediaPipe image-space landmark to the avatar's y-up frame,
 * scaled so that dataset hand-size maps to the rig's HAND_UNIT. */
export function landmarkToAvatar(p: { x: number; y: number; z: number }, scale: number): Vector3 {
    return new Vector3(p.x * scale, -p.y * scale, -p.z * scale);
}

/** Hand scale of a frame: the real wrist->middle-MCP distance, which is what
 * handRig.ts's bone lengths are expressed relative to. */
export function frameHandScale(frame: HandLandmarks): number {
    const dx = frame[9].x - frame[0].x;
    const dy = frame[9].y - frame[0].y;
    const dz = frame[9].z - frame[0].z;
    return Math.hypot(dx, dy, dz);
}

function quatFromTo(from: Vector3, to: Vector3): Quaternion {
    const q = new Quaternion();
    // setFromUnitVectors handles the antiparallel case internally.
    q.setFromUnitVectors(from.clone().normalize(), to.clone().normalize());
    return q;
}

/**
 * Solves the wrist orientation from the palm's two measured directions.
 * Returns null when the palm is degenerate (collinear or zero-length
 * directions), which can happen on a badly-detected frame.
 */
export function solveWristOrientation(points: Vector3[]): Quaternion | null {
    const up = points[9].clone().sub(points[0]);
    const across = points[17].clone().sub(points[5]);
    if (up.lengthSq() < 1e-12 || across.lengthSq() < 1e-12) return null;
    up.normalize();
    across.normalize();

    const forward = new Vector3().crossVectors(up, across);
    if (forward.lengthSq() < 1e-10) return null; // palm directions collinear
    forward.normalize();

    // side = up x forward, NOT forward x up. The basis (side, up, forward)
    // must be RIGHT-handed: with s = u x f, s x u = f, so det = +1 and
    // Matrix4.makeBasis yields a true rotation. Using forward x up gives
    // s x u = -f, i.e. a reflection with det = -1, and
    // Quaternion.setFromRotationMatrix silently returns a meaningless
    // quaternion for a reflection -- which showed up as the wrist frame
    // being flipped ~180 degrees about Y.
    const side = new Vector3().crossVectors(up, forward).normalize();

    const m = new Matrix4().makeBasis(side, up, forward);
    return new Quaternion().setFromRotationMatrix(m);
}

/**
 * Solves every bone rotation for one real landmark frame.
 *
 * `positionScale` maps dataset image units into avatar world units for the
 * WRIST POSITION only; bone orientations are scale-invariant.
 */
export function solveHandPose(
    frame: HandLandmarks,
    options: { positionScale?: number; mirror?: boolean } = {}
): HandPoseSolution {
    const { positionScale = 1, mirror = false } = options;

    if (frame.length !== 21) {
        throw new Error(`solveHandPose expects 21 landmarks, received ${frame.length}`);
    }

    const scale = frameHandScale(frame) || 1e-6;
    // Normalize the frame to rig units so bone directions are computed in a
    // consistent space. Orientation is unaffected by uniform scale; this just
    // keeps magnitudes sane.
    const points = frame.map((p) => {
        const v = landmarkToAvatar(p, HAND_UNIT / scale);
        if (mirror) v.x = -v.x;
        return v;
    });

    const root = solveWristOrientation(points);
    const rootPosition: [number, number, number] = [
        (mirror ? -frame[0].x : frame[0].x) * positionScale,
        -frame[0].y * positionScale,
        -frame[0].z * positionScale,
    ];

    if (!root) {
        return {
            rootQuaternion: [0, 0, 0, 1],
            rootPosition,
            bones: HAND_BONES.map((b) => ({ name: b.name, quaternion: [0, 0, 0, 1] as [number, number, number, number] })),
            degenerate: true,
        };
    }

    const worldQuats = new Map<string, Quaternion>();
    const bones: BoneRotation[] = [];

    for (const bone of bonesInHierarchyOrder()) {
        const parentWorld = bone.parent === null ? root : worldQuats.get(bone.parent)!;

        const worldDir = points[bone.toLandmark].clone().sub(points[bone.fromLandmark]);
        if (worldDir.lengthSq() < 1e-14) {
            // Two coincident landmarks: keep the bone at rest rather than
            // producing a NaN quaternion.
            const identity = new Quaternion();
            worldQuats.set(bone.name, parentWorld.clone());
            bones.push({ name: bone.name, quaternion: [identity.x, identity.y, identity.z, identity.w] });
            continue;
        }
        worldDir.normalize();

        const localDir = worldDir.clone().applyQuaternion(parentWorld.clone().invert());
        const local = quatFromTo(REST_DIR, localDir);

        worldQuats.set(bone.name, parentWorld.clone().multiply(local));
        bones.push({ name: bone.name, quaternion: [local.x, local.y, local.z, local.w] });
    }

    return {
        rootQuaternion: [root.x, root.y, root.z, root.w],
        rootPosition,
        bones,
        degenerate: false,
    };
}

/** Per-bone length ratios (same units as handRig's `length`): the rig's
 * proportions, optionally calibrated to one specific recorded hand. */
export type BoneLengths = Readonly<Record<string, number>>;

/**
 * Calibrates bone lengths to the hand in ONE real sequence.
 *
 * WHY PER-SEQUENCE AT ALL: a segment's length measured from image landmarks
 * is a PROJECTION, so it shrinks whenever that segment points toward or away
 * from the camera. Across the real registry the per-sign median of
 * thumb_meta ranges from 0.24 to 1.08 -- a 4.5x spread that is mostly
 * foreshortening and signer variation, not a fixed anatomy. One global set of
 * medians therefore fits no individual hand well.
 *
 * WHICH PERCENTILE: this was measured, not argued. The intuition that a HIGH
 * percentile is better (the least-foreshortened frames should reveal true
 * length) turned out to be wrong for this purpose. Reconstruction error
 * against 40 real frames spread across the registry, in avatar world units
 * where HAND_UNIT = 0.085:
 *
 *   calibration            p50 err   p90 err   max err
 *   global medians         0.0177    0.0463    0.1046
 *   per-sequence p50       0.0012    0.0105    0.0379   <- chosen
 *   per-sequence p75       0.0016    0.0262    0.0753
 *   per-sequence p90       0.0021    0.0366    0.1095
 *   per-sequence max       0.0054    0.0592    0.2163
 *
 * The median wins because the target being reproduced IS the projected
 * recording, not an unobservable true anatomy: with only weak depth in this
 * data, matching the typical projected length reproduces what the camera
 * actually saw. A high percentile would be the right choice given real metric
 * 3D landmarks, which this dataset does not contain.
 *
 * Lengths are computed once per sequence and then held constant, so the
 * avatar's hand never changes shape mid-sign.
 */
export function calibrateBoneLengths(frames: HandLandmarks[], percentile = 0.5): BoneLengths {
    const samples = new Map<string, number[]>();
    for (const bone of HAND_BONES) samples.set(bone.name, []);

    for (const frame of frames) {
        if (frame.length !== 21) continue;
        const scale = frameHandScale(frame);
        if (scale < 1e-9) continue;
        for (const bone of HAND_BONES) {
            const a = frame[bone.fromLandmark];
            const b = frame[bone.toLandmark];
            samples.get(bone.name)!.push(Math.hypot(b.x - a.x, b.y - a.y, b.z - a.z) / scale);
        }
    }

    const out: Record<string, number> = {};
    for (const bone of HAND_BONES) {
        const s = samples.get(bone.name)!.sort((x, y) => x - y);
        out[bone.name] = s.length === 0 ? bone.length : s[Math.min(s.length - 1, Math.floor(percentile * (s.length - 1)))];
    }
    return out;
}

/**
 * Forward-kinematics reconstruction of the 21 joint positions implied by a
 * solved pose. This exists so tests can verify that the solved rotations
 * actually reproduce the input hand shape -- i.e. that retargeting is
 * correct rather than merely plausible-looking.
 *
 * Pass `lengths` from calibrateBoneLengths() to use the source hand's own
 * proportions; omit it to use handRig.ts's dataset-median defaults.
 */
export function forwardKinematics(solution: HandPoseSolution, lengths?: BoneLengths): Vector3[] {
    const out: Vector3[] = new Array(21);
    const root = new Quaternion(...solution.rootQuaternion);
    out[0] = new Vector3(0, 0, 0);

    const worldQuats = new Map<string, Quaternion>();
    const worldOrigins = new Map<string, Vector3>();
    const byName = new Map(solution.bones.map((b) => [b.name, b]));

    for (const bone of bonesInHierarchyOrder()) {
        const local = byName.get(bone.name);
        if (!local) continue;
        const parentQuat = bone.parent === null ? root : worldQuats.get(bone.parent)!;
        const parentOrigin = bone.parent === null ? out[0] : worldOrigins.get(bone.parent)!;

        const world = parentQuat.clone().multiply(new Quaternion(...local.quaternion));
        const boneLength = lengths?.[bone.name] ?? bone.length;
        const offset = new Vector3(0, boneLength * HAND_UNIT, 0).applyQuaternion(world);
        const end = parentOrigin.clone().add(offset);

        worldQuats.set(bone.name, world);
        worldOrigins.set(bone.name, end);
        out[bone.toLandmark] = end;
    }

    return out;
}
