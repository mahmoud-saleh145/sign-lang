/**
 * Proves the avatar is a real skeletal rig rather than geometry positioned
 * at landmark coordinates:
 *  - the bone hierarchy exists and matches handRig's parent/child structure,
 *  - every bone carries mesh geometry as a CHILD (so it is driven by the
 *    bone, not placed independently),
 *  - rotating a bone moves its descendants' world positions,
 *  - the whole chain reproduces a real KArSL frame when fed that frame's
 *    solved rotations.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { Mesh, Quaternion, Vector3 } from "three";
import { buildSignerRig, solveArmIK, BODY } from "./signerRig";
import { HAND_BONES } from "@/lib/avatar/handRig";
import { solveHandPose, calibrateBoneLengths } from "@/lib/avatar/retarget";
import type { HandLandmarks } from "@/types/landmarks";
import { withLegacySequence } from "@/lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "@/lib/sign-language/signRegistry";

// The shipped registry omits the legacy `landmarkSequence` (it duplicates the
// primary hand already in `handFrames`). Reading the file directly therefore
// has to apply the same normalization the app applies on load.
const registry = (
    JSON.parse(
        readFileSync(path.resolve(__dirname, "../../public/models/sign-vocabulary/dedicated-signs.json"), "utf-8")
    ) as DedicatedSignEntry[]
).map(withLegacySequence) as Array<{ sourceSignId?: string; landmarkSequence: HandLandmarks[] }>;

const realSign = registry.find((e) => e.landmarkSequence.length > 10)!;

describe("procedural signer rig", () => {
    it("builds every hand bone with the right parent", () => {
        const rig = buildSignerRig();
        expect(rig.bones.size).toBe(HAND_BONES.length);
        for (const def of HAND_BONES) {
            const bone = rig.bones.get(def.name);
            expect(bone, `${def.name} missing`).toBeDefined();
            const expectedParent = def.parent === null ? rig.handRoot : rig.bones.get(def.parent);
            expect(bone!.parent).toBe(expectedParent);
        }
        rig.dispose();
    });

    it("attaches geometry to bones rather than positioning it independently", () => {
        const rig = buildSignerRig();
        for (const def of HAND_BONES) {
            const bone = rig.bones.get(def.name)!;
            // Capsules are wrapped in a Group (see capsule()), so look through
            // descendants rather than only direct children.
            let meshCount = 0;
            bone.traverse((o) => {
                if ((o as Mesh).isMesh) meshCount++;
            });
            expect(meshCount, `${def.name} has no geometry`).toBeGreaterThan(0);
        }
        rig.dispose();
    });

    it("has the hand parented through forearm -> upper arm -> root", () => {
        const rig = buildSignerRig();
        expect(rig.handRoot.parent).toBe(rig.forearm);
        expect(rig.forearm.parent).toBe(rig.upperArm);
        rig.dispose();
    });

    it("rotating a bone moves its descendants (the chain is real)", () => {
        const rig = buildSignerRig();
        rig.root.updateWorldMatrix(true, true);
        const tip = rig.bones.get("index_dist")!;
        const before = new Vector3();
        tip.getWorldPosition(before);

        rig.bones.get("index_prox")!.quaternion.setFromAxisAngle(new Vector3(1, 0, 0), 1.2);
        rig.root.updateWorldMatrix(true, true);
        const after = new Vector3();
        tip.getWorldPosition(after);

        expect(before.distanceTo(after)).toBeGreaterThan(0.01);
        rig.dispose();
    });

    it("reproduces a real KArSL frame's finger directions through the rig", () => {
        const frame = realSign.landmarkSequence[Math.floor(realSign.landmarkSequence.length / 2)];
        const lengths = calibrateBoneLengths(realSign.landmarkSequence);
        const rig = buildSignerRig(lengths);
        const solution = solveHandPose(frame);

        for (const b of solution.bones) {
            rig.bones.get(b.name)!.quaternion.set(...b.quaternion);
        }
        rig.handRoot.quaternion.set(...solution.rootQuaternion);
        rig.root.updateWorldMatrix(true, true);

        // Compare each bone's world direction in the rig against the real
        // frame's direction, both expressed relative to the wrist frame.
        const wristQuat = new Quaternion();
        rig.handRoot.getWorldQuaternion(wristQuat);
        const toWrist = wristQuat.clone().invert();

        const worldOf = (name: string) => {
            const v = new Vector3();
            rig.bones.get(name)!.getWorldPosition(v);
            return v;
        };
        const handRootWorld = new Vector3();
        rig.handRoot.getWorldPosition(handRootWorld);

        for (const def of HAND_BONES) {
            const start = def.parent === null ? handRootWorld : worldOf(def.name);
            // A bone's own origin is its start; its end is where its child sits,
            // or start + length along its own +Y for a leaf.
            const boneObj = rig.bones.get(def.name)!;
            const boneQuat = new Quaternion();
            boneObj.getWorldQuaternion(boneQuat);
            const end = start.clone().add(new Vector3(0, 1, 0).applyQuaternion(boneQuat));

            const rigDir = end.sub(start).applyQuaternion(toWrist).normalize();

            const a = frame[def.fromLandmark];
            const bpt = frame[def.toLandmark];
            const realDir = new Vector3(bpt.x - a.x, -(bpt.y - a.y), -(bpt.z - a.z)).normalize();
            const realLocal = realDir.applyQuaternion(new Quaternion(...solution.rootQuaternion).invert());

            expect(rigDir.dot(realLocal), `${def.name}`).toBeGreaterThan(0.9);
        }
        rig.dispose();
    });
});

describe("arm IK", () => {
    it("reaches a target inside the arm's range", () => {
        const rig = buildSignerRig();
        const target = new Vector3(0, BODY.chestTarget, 0.25);
        const r = solveArmIK(rig, target);
        expect(r.reachable).toBe(true);
        expect(Number.isFinite(r.elbowAngle)).toBe(true);
        rig.dispose();
    });

    it("reports an out-of-range target instead of stretching silently", () => {
        const rig = buildSignerRig();
        const r = solveArmIK(rig, new Vector3(0, BODY.chestTarget, 5));
        expect(r.reachable).toBe(false);
        rig.dispose();
    });

    it("keeps the arm pose finite for every wrist position in a real sequence", () => {
        const rig = buildSignerRig();
        for (const frame of realSign.landmarkSequence) {
            const w = frame[0];
            const target = new Vector3((w.x - 0.59) * 1.29, (0.68 - w.y) * 1.29 + 1.16, 0.2 - w.z * 1.29);
            solveArmIK(rig, target);
            expect(Number.isFinite(rig.upperArm.quaternion.x)).toBe(true);
            expect(Number.isFinite(rig.forearm.quaternion.x)).toBe(true);
        }
        rig.dispose();
    });
});
