/**
 * Rig-binding tests against SYNTHETIC skeletons in each of the naming
 * conventions a real rigged humanoid is likely to use. These are structure
 * tests, not motion tests: they prove that when you drop in a model named
 * the Mixamo way, the RPM way or the VRM way, the correct bones get bound.
 *
 * They are explicitly NOT a claim that any particular downloaded avatar
 * works -- that can only be confirmed against the real file.
 */
import { describe, it, expect } from "vitest";
import { Bone, Object3D, Quaternion, Vector3 } from "three";
import { bindAvatarRig, applySolvedRotation, jointDepthOf, sideOf } from "./glbRig";
import { HAND_BONES } from "./handRig";

type Naming = (side: "Left" | "Right", finger: string, depth: number) => string;

const CONVENTIONS: Record<string, { wrist: (s: string) => string; finger: Naming; forearm: (s: string) => string; upper: (s: string) => string }> = {
    mixamo: {
        wrist: (s) => `mixamorig:${s}Hand`,
        finger: (s, f, d) => `mixamorig:${s}Hand${f}${d}`,
        forearm: (s) => `mixamorig:${s}ForeArm`,
        upper: (s) => `mixamorig:${s}Arm`,
    },
    readyPlayerMe: {
        wrist: (s) => `${s}Hand`,
        finger: (s, f, d) => `${s}Hand${f}${d}`,
        forearm: (s) => `${s}ForeArm`,
        upper: (s) => `${s}Arm`,
    },
    vrm: {
        wrist: (s) => `J_Bip_${s[0]}_Hand`,
        finger: (s, f, d) => `J_Bip_${s[0]}_${f}${["", "Proximal", "Intermediate", "Distal"][d]}`,
        forearm: (s) => `J_Bip_${s[0]}_LowerArm`,
        upper: (s) => `J_Bip_${s[0]}_UpperArm`,
    },
};

const FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"];

function buildSkeleton(convention: keyof typeof CONVENTIONS): Object3D {
    const c = CONVENTIONS[convention];
    const root = new Object3D();
    root.name = "Armature";
    for (const side of ["Left", "Right"] as const) {
        const upper = new Bone();
        upper.name = c.upper(side);
        root.add(upper);
        const fore = new Bone();
        fore.name = c.forearm(side);
        fore.position.set(0, 0.3, 0);
        upper.add(fore);
        const wrist = new Bone();
        wrist.name = c.wrist(side);
        wrist.position.set(0, 0.26, 0);
        fore.add(wrist);
        for (const f of FINGERS) {
            let parent: Object3D = wrist;
            for (let d = 1; d <= 3; d++) {
                const b = new Bone();
                b.name = c.finger(side, f, d);
                b.position.set(0, 0.03, 0);
                parent.add(b);
                parent = b;
            }
        }
    }
    const head = new Bone();
    head.name = "Head";
    root.add(head);
    return root;
}

describe("sideOf", () => {
    it("does not mistake a substring for a side marker", () => {
        // The bug this guards: "_l" inside "_LowerArm" made a RIGHT bone bind left.
        expect(sideOf("J_Bip_R_LowerArm")).toBe("right");
        expect(sideOf("J_Bip_L_LowerArm")).toBe("left");
        expect(sideOf("mixamorig:RightForeArm")).toBe("right");
    });
    it("handles Blender-style .L / .R suffixes", () => {
        expect(sideOf("hand.L")).toBe("left");
        expect(sideOf("f_index.01.R")).toBe("right");
    });
    it("returns null for a bone with no side", () => {
        expect(sideOf("Spine")).toBeNull();
        expect(sideOf("Head")).toBeNull();
    });
});

describe("jointDepthOf", () => {
    it("reads numeric suffixes (Mixamo / Ready Player Me)", () => {
        expect(jointDepthOf("mixamorig:LeftHandIndex1", "index")).toBe(1);
        expect(jointDepthOf("LeftHandIndex3", "index")).toBe(3);
    });
    it("reads spelled-out depths (VRM)", () => {
        expect(jointDepthOf("J_Bip_L_IndexProximal", "index")).toBe(1);
        expect(jointDepthOf("J_Bip_L_IndexIntermediate", "index")).toBe(2);
        expect(jointDepthOf("J_Bip_L_IndexDistal", "index")).toBe(3);
    });
    it("returns null when there is no depth marker at all", () => {
        expect(jointDepthOf("LeftHandIndex", "index")).toBeNull();
    });
});

describe("bindAvatarRig across naming conventions", () => {
    for (const name of Object.keys(CONVENTIONS)) {
        it(`binds both hands for a ${name}-style rig`, () => {
            const bound = bindAvatarRig(buildSkeleton(name as keyof typeof CONVENTIONS));
            expect(bound.validation.ok, JSON.stringify(bound.validation.notes)).toBe(true);
            expect(bound.hands.left.wrist).not.toBeNull();
            expect(bound.hands.right.wrist).not.toBeNull();
            // 15 drivable bones per side (5 fingers x proximal/middle/distal).
            expect(bound.validation.boundPerSide.left).toBeGreaterThanOrEqual(15);
            expect(bound.validation.boundPerSide.right).toBeGreaterThanOrEqual(15);
        });

        it(`never binds the same GLB bone to both sides for a ${name}-style rig`, () => {
            const bound = bindAvatarRig(buildSkeleton(name as keyof typeof CONVENTIONS));
            const left = new Set([...bound.hands.left.bones.values()].map((b) => b.name));
            const right = [...bound.hands.right.bones.values()].map((b) => b.name);
            for (const r of right) expect(left.has(r)).toBe(false);
        });

        it(`finds arm bones for a ${name}-style rig`, () => {
            const bound = bindAvatarRig(buildSkeleton(name as keyof typeof CONVENTIONS));
            expect(bound.arms.left.forearm).not.toBeNull();
            expect(bound.arms.right.forearm).not.toBeNull();
        });
    }

    it("reports an unrigged model as invalid instead of silently doing nothing", () => {
        const lone = new Object3D();
        lone.name = "Mesh";
        const bound = bindAvatarRig(lone);
        expect(bound.validation.ok).toBe(false);
        expect(bound.validation.notes.length).toBeGreaterThan(0);
    });

    it("reports missing finger bones by name", () => {
        const root = buildSkeleton("mixamo");
        // Collect first, then remove -- mutating during traverse breaks it.
        const doomed: Object3D[] = [];
        root.traverse((o) => {
            if (o.name === "mixamorig:LeftHandIndex1") doomed.push(o);
        });
        for (const d of doomed) d.removeFromParent();
        const bound = bindAvatarRig(root);
        expect(bound.validation.missing.some((m) => m.startsWith("left.index"))).toBe(true);
    });

    it("records which GLB bone was matched to each role, for auditing", () => {
        const bound = bindAvatarRig(buildSkeleton("mixamo"));
        expect(bound.validation.matchedNames["left.wrist"]).toBe("mixamorig:LeftHand");
        expect(bound.validation.matchedNames["right.index_prox"]).toBe("mixamorig:RightHandIndex1");
    });

    it("captures the model's rest pose rather than assuming identity", () => {
        const root = buildSkeleton("mixamo");
        const target = "mixamorig:LeftHandIndex1";
        const rest = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), 0.4);
        root.traverse((o) => {
            if (o.name === target) o.quaternion.copy(rest);
        });
        const bound = bindAvatarRig(root);
        const captured = bound.hands.left.restLocal.get("index_prox")!;
        expect(captured.angleTo(rest)).toBeLessThan(1e-6);
    });
});

describe("applySolvedRotation", () => {
    it("composes onto the rest pose instead of overwriting it", () => {
        const root = buildSkeleton("mixamo");
        const rest = new Quaternion().setFromAxisAngle(new Vector3(0, 0, 1), 0.4);
        root.traverse((o) => {
            if (o.name === "mixamorig:LeftHandIndex1") o.quaternion.copy(rest);
        });
        const bound = bindAvatarRig(root);

        // Identity solved rotation must leave the bone exactly at rest.
        expect(applySolvedRotation(bound.hands.left, "index_prox", new Quaternion())).toBe(true);
        const bone = bound.hands.left.bones.get("index_prox")!;
        expect(bone.quaternion.angleTo(rest)).toBeLessThan(1e-6);
    });

    it("actually rotates the bone for a non-identity solution", () => {
        const bound = bindAvatarRig(buildSkeleton("mixamo"));
        const bone = bound.hands.left.bones.get("index_prox")!;
        const before = bone.quaternion.clone();
        applySolvedRotation(bound.hands.left, "index_prox", new Quaternion().setFromAxisAngle(new Vector3(1, 0, 0), 1.0));
        expect(bone.quaternion.angleTo(before)).toBeGreaterThan(0.5);
    });

    it("returns false for a bone the model does not have", () => {
        const bound = bindAvatarRig(buildSkeleton("mixamo"));
        expect(applySolvedRotation(bound.hands.left, "nonexistent_bone", new Quaternion())).toBe(false);
    });

    it("covers every drivable handRig bone name", () => {
        const bound = bindAvatarRig(buildSkeleton("mixamo"));
        const drivable = HAND_BONES.filter((b) => !b.name.endsWith("_meta"));
        for (const d of drivable) {
            expect(bound.hands.right.bones.has(d.name), `${d.name} unbound`).toBe(true);
        }
    });
});
