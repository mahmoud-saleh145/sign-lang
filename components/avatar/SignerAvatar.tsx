"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, ContactShadows } from "@react-three/drei";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";
import { Box3, Object3D, Quaternion, Vector3 } from "three";
import { buildSignerRig, solveArmIK, restArm, BODY } from "./signerRig";
import type { SignerRigHandles } from "./signerRig";
import { bindAvatarRig, applySolvedRotation, AVATAR_GLB_URL } from "@/lib/avatar/glbRig";
import type { BoundAvatar, HandSide, RigValidation } from "@/lib/avatar/glbRig";
import { solveHandPose } from "@/lib/avatar/retarget";
import type { HandPoseSolution } from "@/lib/avatar/retarget";
import type { AnimatableSign } from "@/lib/avatar/karslSign";
import type { SignAnimationController, AnimationSnapshot } from "@/lib/avatar/signAnimationController";

/**
 * 3D avatar view.
 *
 * AVATAR SOURCE: if a rigged humanoid exists at public/models/avatar/signer.glb
 * it is loaded and bound (see lib/avatar/glbRig.ts). Otherwise the procedural
 * rig is used. The active source and the rig-validation result are both
 * reported in diagnostics, so "the avatar looks wrong" is always traceable to
 * either the model, the binding, or the data -- never a silent fallback.
 *
 * BOTH HANDS: when the sign's registry entry carries two-hand data, each hand
 * is driven from its OWN recorded track. A hand with no data for a frame is
 * left where it is; a hand with no data for the whole sign is put in a resting
 * pose. No second-hand motion is ever synthesised.
 *
 * PERFORMANCE:
 *  - The model loads once and stays mounted; it is never rebuilt per frame.
 *  - Per-frame bone rotations are PRECOMPUTED per sign, then SLERPed at
 *    render time. Slerp stays on the rotation manifold, so a fast rotation
 *    never collapses through the middle.
 *  - useFrame writes straight to Three.js objects. No React state per frame;
 *    the UI gets a throttled snapshot (~12 Hz) for the progress readout only.
 */

/** Maps dataset image coordinates into the avatar's signing volume. Derived
 * from the registry's real wrist distribution (x p5..p95 = 0.48..0.70,
 * y = 0.50..0.90) and the scale relation between dataset hand size
 * (wrist->middle MCP p50 = 0.066) and the rig's HAND_UNIT (0.085 m). */
const SIGNING_VOLUME = {
    scale: 1.29,
    originX: 0.59,
    originY: 0.68,
    chestY: 1.16,
    forwardZ: 0.34,
} as const;

export function wristTargetFromLandmark(
    wrist: { x: number; y: number; z: number },
    mirror: boolean
): Vector3 {
    const x = mirror ? 2 * SIGNING_VOLUME.originX - wrist.x : wrist.x;
    return new Vector3(
        (x - SIGNING_VOLUME.originX) * SIGNING_VOLUME.scale,
        (SIGNING_VOLUME.originY - wrist.y) * SIGNING_VOLUME.scale + SIGNING_VOLUME.chestY,
        SIGNING_VOLUME.forwardZ - wrist.z * SIGNING_VOLUME.scale
    );
}

export type AvatarSource = "glb" | "procedural";

export interface AvatarDiagnostics {
    avatarSource: AvatarSource;
    avatarLoaded: boolean;
    avatarLoadError: string | null;
    rigValid: boolean;
    rigValidation: RigValidation | null;
    handBoneCount: number;
    boundPerSide: Record<HandSide, number>;
    solvedFrameCount: number;
    degenerateFrameCount: number;
    currentFrameIndex: number;
    /** Whether the CURRENT sign's data contains each hand. */
    leftHandDetected: boolean;
    rightHandDetected: boolean;
    twoHanded: boolean;
    handDataUnknown: boolean;
    armReachable: boolean;
}

type SideSolutions = Record<HandSide, Array<HandPoseSolution | null>>;

interface DriverProps {
    controller: SignAnimationController;
    mirror: boolean;
    glb: Object3D | null;
    glbError: string | null;
    onSnapshot?: (s: AnimationSnapshot) => void;
    onDiagnostics?: (d: AvatarDiagnostics) => void;
}

function RigDriver({ controller, mirror, glb, glbError, onSnapshot, onDiagnostics }: DriverProps) {
    const { scene } = useThree();
    const proceduralRef = useRef<SignerRigHandles | null>(null);
    const boundRef = useRef<BoundAvatar | null>(null);
    const solvedRef = useRef<{ key: string; solutions: SideSolutions; degenerate: number } | null>(null);
    const lastEmitRef = useRef(0);
    const signKeyRef = useRef<string | null>(null);

    const usingGlb = glb !== null;

    // Mount the avatar ONCE (per model swap), never per frame.
    useEffect(() => {
        if (glb) {
            scene.add(glb);
            boundRef.current = bindAvatarRig(glb);
            return () => {
                scene.remove(glb);
                boundRef.current = null;
            };
        }
        const rig = buildSignerRig();
        proceduralRef.current = rig;
        scene.add(rig.root);
        return () => {
            scene.remove(rig.root);
            rig.dispose();
            proceduralRef.current = null;
        };
    }, [glb, scene]);

    useFrame((_state, delta) => {
        const snapshot = controller.tick(delta * 1000);
        const sign = controller.getCurrentSign();

        const key = sign ? `${sign.source}:${sign.signId ?? sign.text}:${sign.frameCount}:${mirror}` : "none";
        if (key !== signKeyRef.current) {
            signKeyRef.current = key;
            solvedRef.current = precompute(sign, mirror);
        }
        const solved = solvedRef.current;

        const proc = proceduralRef.current;
        const bound = boundRef.current;
        if (!solved || (!proc && !bound)) return;

        const frames = Math.max(solved.solutions.left.length, solved.solutions.right.length);
        if (frames === 0) return;

        const fi = Math.min(snapshot.frameIndex, frames - 1);
        const lo = Math.floor(fi);
        const hi = Math.min(lo + 1, frames - 1);
        const t = fi - lo;

        let anyReachable = true;
        for (const side of ["left", "right"] as HandSide[]) {
            const A = solved.solutions[side][lo] ?? null;
            const B = solved.solutions[side][hi] ?? A;
            if (!A) {
                // No recorded data for this hand at this frame.
                if (proc && solved.solutions[side].every((s) => s === null)) restArm(proc.hands[side], side);
                continue;
            }
            const r = applyPose(side, A, B ?? A, t, sign!, lo, hi, mirror, proc, bound);
            anyReachable = anyReachable && r;
        }

        const now = performance.now();
        if (now - lastEmitRef.current > 80) {
            lastEmitRef.current = now;
            onSnapshot?.(snapshot);
            const validation = bound?.validation ?? null;
            onDiagnostics?.({
                avatarSource: usingGlb ? "glb" : "procedural",
                avatarLoaded: true,
                avatarLoadError: glbError,
                rigValid: usingGlb ? Boolean(validation?.ok) : true,
                rigValidation: validation,
                handBoneCount: usingGlb
                    ? (validation?.boundPerSide.left ?? 0) + (validation?.boundPerSide.right ?? 0)
                    : (proc?.hands.left.bones.size ?? 0) + (proc?.hands.right.bones.size ?? 0),
                boundPerSide: usingGlb
                    ? (validation?.boundPerSide ?? { left: 0, right: 0 })
                    : { left: proc?.hands.left.bones.size ?? 0, right: proc?.hands.right.bones.size ?? 0 },
                solvedFrameCount: frames,
                degenerateFrameCount: solved.degenerate,
                currentFrameIndex: lo,
                leftHandDetected: solved.solutions.left.some((s) => s !== null),
                rightHandDetected: solved.solutions.right.some((s) => s !== null),
                twoHanded: Boolean(sign?.twoHanded),
                handDataUnknown: Boolean(sign?.handDataUnknown),
                armReachable: anyReachable,
            });
        }
    });

    return null;
}

/** Solves every frame of a sign, per hand, once. */
function precompute(sign: AnimatableSign | null, mirror: boolean) {
    const solutions: SideSolutions = { left: [], right: [] };
    let degenerate = 0;
    if (!sign) return { key: "none", solutions, degenerate };

    if (sign.tracks) {
        for (const side of ["left", "right"] as HandSide[]) {
            solutions[side] = sign.tracks[side].frames.map((f) => {
                if (!f) return null;
                const s = solveHandPose(f, { mirror });
                if (s.degenerate) degenerate++;
                return s;
            });
        }
    } else {
        // Single-hand data (pre two-hand extraction). Drive the right hand and
        // leave the left at rest; do NOT invent a second track.
        solutions.right = sign.landmarkSequence.map((f) => {
            const s = solveHandPose(f, { mirror });
            if (s.degenerate) degenerate++;
            return s;
        });
        solutions.left = solutions.right.map(() => null);
    }
    return { key: "", solutions, degenerate };
}

function applyPose(
    side: HandSide,
    A: HandPoseSolution,
    B: HandPoseSolution,
    t: number,
    sign: AnimatableSign,
    lo: number,
    hi: number,
    mirror: boolean,
    proc: SignerRigHandles | null,
    bound: BoundAvatar | null
): boolean {
    const qa = new Quaternion();
    const qb = new Quaternion();

    if (bound) {
        const hand = bound.hands[side];
        for (let i = 0; i < A.bones.length; i++) {
            qa.set(...A.bones[i].quaternion);
            qb.set(...B.bones[i].quaternion);
            applySolvedRotation(hand, A.bones[i].name, qa.slerp(qb, t));
        }
        if (hand.wrist) {
            qa.set(...A.rootQuaternion);
            qb.set(...B.rootQuaternion);
            // GLB arms are left in their authored pose; only the hand is driven,
            // because the dataset has no shoulder or elbow data to drive them with.
            hand.wrist.quaternion.copy(qa.slerp(qb, t));
        }
        return true;
    }

    if (!proc) return true;
    const hand = proc.hands[side];
    for (let i = 0; i < A.bones.length; i++) {
        const bone = hand.bones.get(A.bones[i].name);
        if (!bone) continue;
        qa.set(...A.bones[i].quaternion);
        qb.set(...B.bones[i].quaternion);
        bone.quaternion.copy(qa.slerp(qb, t));
    }

    qa.set(...A.rootQuaternion);
    qb.set(...B.rootQuaternion);
    const worldWrist = qa.slerp(qb, t);

    const track = sign.tracks ? sign.tracks[side].frames : sign.landmarkSequence;
    const wa = track[lo] ?? null;
    const wb = track[hi] ?? wa;
    if (!wa) return true;
    const w0 = wa[0];
    const w1 = (wb ?? wa)[0];
    const target = wristTargetFromLandmark(
        { x: w0.x + (w1.x - w0.x) * t, y: w0.y + (w1.y - w0.y) * t, z: w0.z + (w1.z - w0.z) * t },
        mirror
    );
    const ik = solveArmIK(proc, target, side);

    hand.forearm.updateWorldMatrix(true, false);
    const forearmWorld = new Quaternion();
    hand.forearm.getWorldQuaternion(forearmWorld);
    hand.handRoot.quaternion.copy(forearmWorld.invert().multiply(worldWrist));

    return ik.reachable;
}

/** Frames the camera on the avatar's upper body once the model is known. */
function AutoFrame({ target }: { target: Object3D | null }) {
    const { camera } = useThree();
    useEffect(() => {
        if (!target) return;
        const box = new Box3().setFromObject(target);
        if (!box.isEmpty()) {
            const size = new Vector3();
            const centre = new Vector3();
            box.getSize(size);
            box.getCenter(centre);
            // Frame the upper body: from mid-torso to just above the head,
            // with headroom on both sides so hands stay in shot while signing.
            const top = box.max.y;
            const focusY = top - size.y * 0.28;
            const dist = Math.max(size.x, size.y * 0.55) * 1.9 + 0.35;
            camera.position.set(0, focusY, dist);
            camera.lookAt(0, focusY, 0);
            camera.updateProjectionMatrix();
        }
    }, [target, camera]);
    return null;
}

export interface SignerAvatarProps {
    controller: SignAnimationController;
    mirror?: boolean;
    onSnapshot?: (s: AnimationSnapshot) => void;
    onDiagnostics?: (d: AvatarDiagnostics) => void;
    className?: string;
}

export default function SignerAvatar({
    controller,
    mirror = false,
    onSnapshot,
    onDiagnostics,
    className,
}: SignerAvatarProps) {
    const [glb, setGlb] = useState<Object3D | null>(null);
    const [glbError, setGlbError] = useState<string | null>(null);

    // Load the optional rigged humanoid once. A missing file is the normal
    // case, not an error -- the procedural rig takes over.
    useEffect(() => {
        let cancelled = false;
        const loader = new GLTFLoader();
        loader.load(
            AVATAR_GLB_URL,
            (gltf) => {
                if (cancelled) return;
                gltf.scene.traverse((o) => {
                    o.castShadow = true;
                    o.receiveShadow = true;
                });
                setGlb(gltf.scene);
                setGlbError(null);
            },
            undefined,
            () => {
                if (cancelled) return;
                setGlb(null);
                setGlbError(`No rigged avatar at ${AVATAR_GLB_URL} — using the built-in procedural signer.`);
            }
        );
        return () => {
            cancelled = true;
        };
    }, []);

    const focusY = useMemo(() => BODY.chestTarget, []);

    return (
        <div
            className={
                className ??
                "w-full aspect-[4/3] rounded-2xl overflow-hidden bg-[var(--color-surface)] border border-white/10"
            }
        >
            <Canvas
                camera={{ position: [0, 1.30, 2.15], fov: 32 }}
                dpr={[1, 2]}
                shadows
                gl={{ antialias: true }}
            >
                <color attach="background" args={["#0d1620"]} />

                {/* Studio-style three-point lighting, all local -- drei's
                    <Environment> presets fetch an HDRI from a CDN at runtime. */}
                <hemisphereLight intensity={0.5} groundColor="#0a1018" />
                <directionalLight
                    position={[1.7, 2.6, 2.3]}
                    intensity={2.0}
                    castShadow
                    shadow-mapSize={[1024, 1024]}
                />
                <directionalLight position={[-2.0, 1.4, 1.0]} intensity={0.7} color="#9dc4e8" />
                <directionalLight position={[0, 1.2, -2.2]} intensity={0.5} color="#ffd9b0" />

                <RigDriver
                    controller={controller}
                    mirror={mirror}
                    glb={glb}
                    glbError={glbError}
                    onSnapshot={onSnapshot}
                    onDiagnostics={onDiagnostics}
                />

                <AutoFrame target={glb} />

                <ContactShadows position={[0, 0.62, 0]} opacity={0.45} scale={3} blur={2.4} far={1.2} />

                <OrbitControls
                    target={[0, focusY, 0]}
                    enablePan
                    minDistance={0.5}
                    maxDistance={4}
                    maxPolarAngle={Math.PI * 0.85}
                />
            </Canvas>
        </div>
    );
}
