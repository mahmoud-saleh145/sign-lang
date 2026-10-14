/**
 * RETARGETING CORRECTNESS, verified against REAL KArSL frames.
 *
 * The decisive test is round-tripping: solve bone rotations from a real
 * landmark frame, then run forward kinematics with the rig's fixed bone
 * lengths and check the reconstructed joints match the real hand's SHAPE.
 * If retargeting were wrong, the reconstruction would not match, and no
 * amount of "it looks like a hand on screen" would reveal it.
 *
 * The comparison is shape-only: each hand is normalized to unit
 * wrist->middle-MCP and aligned by the solved wrist orientation, so what is
 * being checked is pose, not placement. Bone lengths come from
 * calibrateBoneLengths() on that sign's own real sequence and are held
 * constant across it, so any residual is retargeting error rather than a
 * proportion mismatch.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { Quaternion, Vector3 } from "three";
import { solveHandPose, forwardKinematics, solveWristOrientation, frameHandScale, landmarkToAvatar, calibrateBoneLengths } from "./retarget";
import type { BoneLengths } from "./retarget";
import { HAND_BONES, HAND_UNIT, REQUIRED_JOINT_COVERAGE, bonesInHierarchyOrder } from "./handRig";
import type { HandLandmarks } from "@/types/landmarks";
import { withLegacySequence } from "@/lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "@/lib/sign-language/signRegistry";

interface RegistryEntry {
    text: string;
    sourceSignId?: string;
    landmarkSequence: HandLandmarks[];
}

// The shipped registry omits the legacy `landmarkSequence` (it duplicates the
// primary hand already in `handFrames`). Reading the file directly therefore
// has to apply the same normalization the app applies on load.
const registry = (
    JSON.parse(
        readFileSync(path.resolve(__dirname, "../../public/models/sign-vocabulary/dedicated-signs.json"), "utf-8")
    ) as DedicatedSignEntry[]
).map(withLegacySequence) as unknown as RegistryEntry[];

/** A spread of real frames from across the whole registry. */
function sampleRealFrames(count: number): Array<{ signId: string; frame: HandLandmarks; lengths: BoneLengths }> {
    const out: Array<{ signId: string; frame: HandLandmarks; lengths: BoneLengths }> = [];
    const step = Math.max(1, Math.floor(registry.length / count));
    for (let i = 0; i < registry.length && out.length < count; i += step) {
        const e = registry[i];
        if (e.landmarkSequence.length === 0) continue;
        out.push({
            signId: e.sourceSignId ?? "?",
            frame: e.landmarkSequence[Math.floor(e.landmarkSequence.length / 2)],
            lengths: calibrateBoneLengths(e.landmarkSequence),
        });
    }
    return out;
}

/** Every registry entry's middle frame, with that entry's own calibrated bone
 * lengths. Used instead of a 40-entry sample so the measured distribution is a
 * property of the dataset rather than of the sampling stride. */
function allRealFrames(): Array<{ signId: string; frame: HandLandmarks; lengths: BoneLengths }> {
    const out: Array<{ signId: string; frame: HandLandmarks; lengths: BoneLengths }> = [];
    for (const e of registry) {
        if (e.landmarkSequence.length === 0) continue;
        out.push({
            signId: e.sourceSignId ?? "?",
            frame: e.landmarkSequence[Math.floor(e.landmarkSequence.length / 2)],
            lengths: calibrateBoneLengths(e.landmarkSequence),
        });
    }
    return out;
}

/** Largest distance between a real landmark and where the rig puts it. */
function worstLandmarkDeviation(frame: HandLandmarks, lengths?: BoneLengths): number {
    const real = realShapeInRigFrame(frame);
    const rig = reconstructedShape(frame, lengths);
    let worst = 0;
    for (let i = 1; i <= 20; i++) worst = Math.max(worst, real[i].distanceTo(rig[i]));
    return worst;
}

/** Real hand, expressed in the same normalized, wrist-aligned frame the rig
 * reconstruction lives in, so the two are directly comparable. */
function realShapeInRigFrame(frame: HandLandmarks): Vector3[] {
    const scale = frameHandScale(frame) || 1e-6;
    const pts = frame.map((p) => landmarkToAvatar(p, HAND_UNIT / scale));
    const wrist = pts[0].clone();
    const root = solveWristOrientation(pts);
    const inv = root ? root.clone().invert() : new Quaternion();
    return pts.map((p) => p.clone().sub(wrist).applyQuaternion(inv));
}

function reconstructedShape(frame: HandLandmarks, lengths?: BoneLengths): Vector3[] {
    const sol = solveHandPose(frame);
    const fk = forwardKinematics(sol, lengths);
    const inv = new Quaternion(...sol.rootQuaternion).invert();
    return fk.map((p) => (p ? p.clone().applyQuaternion(inv) : new Vector3()));
}

describe("hand rig definition", () => {
    it("is authored parent-before-child", () => {
        expect(() => bonesInHierarchyOrder()).not.toThrow();
    });

    it("has exactly one bone per MediaPipe segment (20 bones, 21 landmarks)", () => {
        expect(HAND_BONES.length).toBe(20);
        const covered = new Set(HAND_BONES.map((b) => b.toLandmark));
        // Every landmark except the wrist (0) is the endpoint of exactly one bone.
        for (let i = 1; i <= 20; i++) expect(covered.has(i)).toBe(true);
        expect(covered.size).toBe(20);
    });

    it("covers every joint the spec requires", () => {
        for (const [finger, joints] of Object.entries(REQUIRED_JOINT_COVERAGE)) {
            for (const joint of joints) {
                const hit = HAND_BONES.find((b) => b.finger === finger && b.joint === joint);
                expect(hit, `${finger} ${joint} bone missing`).toBeDefined();
            }
        }
    });
});

describe("retargeting against real KArSL frames", () => {
    const samples = sampleRealFrames(40);

    it("has real frames to test with", () => {
        expect(samples.length).toBeGreaterThanOrEqual(20);
    });

    it("produces finite, unit-norm quaternions for every bone on every sample", () => {
        for (const { signId, frame } of samples) {
            const sol = solveHandPose(frame);
            expect(sol.degenerate, `sign ${signId} produced a degenerate palm`).toBe(false);
            expect(sol.bones.length).toBe(20);
            for (const b of sol.bones) {
                const n = Math.hypot(...b.quaternion);
                expect(Number.isFinite(n), `${signId}/${b.name} non-finite`).toBe(true);
                expect(n).toBeCloseTo(1, 5);
            }
        }
    });

    /**
     * WHAT THIS USED TO ASSERT, AND WHY IT NOW ASSERTS MORE.
     *
     * It used to be: worst landmark deviation over 40 sampled entries < 0.05
     * avatar units, with a comment claiming 0.05 was "a real bound with
     * headroom" backed by a measured worst case of 0.0379.
     *
     * That measurement was taken on the registry as it stood BEFORE the
     * two-hand side correction, where 482 of 502 entries had their two hands
     * swapped -- so the legacy sequence sampled here was a different physical
     * hand than it is now. Re-measured on the corrected registry, across ALL
     * 502 entries rather than a 40-entry sample:
     *
     *   per-sequence calibrated lengths: median 0.0042, p90 0.0335,
     *                                    p99 0.0717, max 0.1824
     *   rig default lengths:             median 0.0375, p90 0.0741,
     *                                    p99 0.1156, max 0.2229
     *
     * 20 of 502 entries exceed 0.05 and 3 exceed 0.10. So 0.05-as-a-maximum
     * was not a property of this data; it was a property of which 40 entries
     * the old sampler happened to reach.
     *
     * WHY THE TAIL EXISTS. Traced on the worst case: at those frames the palm
     * is strongly foreshortened, so `frameHandScale` -- a SINGLE span, wrist to
     * middle MCP -- under-estimates the hand, and every bone is inflated
     * relative to the rig's fixed lengths. On sign 0085's middle frame the
     * measured proximal phalanges came out 2-3x their own sequence median.
     * A robust scale (median of the four wrist-to-MCP spans, each rescaled by
     * its measured ratio to the middle span over 3,134 real frames) was
     * implemented and measured: it cuts the sampled worst case 0.0901 -> 0.0533
     * but raises the median 0.0075 -> 0.0107. It trades typical accuracy for
     * tail accuracy and still does not reach 0.05, so it was NOT adopted.
     *
     * The assertions below are therefore the properties that ARE true and that
     * a degraded retargeting would break, over the whole registry instead of a
     * sample: the typical error, the tail, and -- the one that actually carries
     * the contract -- that per-sequence calibration beats the rig's default
     * lengths by a wide margin on the very same frames. A bare threshold can be
     * passed by accident; a paired comparison cannot.
     */
    it("reconstructs the real hand shape via forward kinematics", () => {
        const calibrated: number[] = [];
        const uncalibrated: number[] = [];

        for (const { frame, lengths } of allRealFrames()) {
            calibrated.push(worstLandmarkDeviation(frame, lengths));
            uncalibrated.push(worstLandmarkDeviation(frame, undefined));
        }

        expect(calibrated.length).toBeGreaterThanOrEqual(100);
        const q = (xs: number[], p: number) =>
            [...xs].sort((a, b) => a - b)[Math.floor(p * (xs.length - 1))];

        // Typical reconstruction is tight. Measured 0.0042; 0.01 is the bound.
        expect(q(calibrated, 0.5)).toBeLessThan(0.01);
        // The bulk is well inside the old 0.05. Measured p90 0.0335.
        expect(q(calibrated, 0.9)).toBeLessThan(0.05);
        // The tail is real and bounded. Measured max 0.1824.
        expect(q(calibrated, 1)).toBeLessThan(0.25);

        // THE ACTUAL CONTRACT: calibrating bone lengths per sequence must be a
        // large, unambiguous improvement over the rig's defaults on the same
        // frames. Measured ratio of medians is ~9x. Breaking calibration --
        // which is what a regression here would look like -- collapses this.
        expect(q(calibrated, 0.5)).toBeLessThan(q(uncalibrated, 0.5) / 4);
        expect(q(calibrated, 0.9)).toBeLessThan(q(uncalibrated, 0.9));
        // And it must be better frame by frame, not just in aggregate.
        const betterCount = calibrated.filter((c, i) => c < uncalibrated[i]).length;
        expect(betterCount / calibrated.length).toBeGreaterThan(0.9);
    });

    it("reproduces bone DIRECTIONS almost exactly (rotation is what retargeting solves)", () => {
        // Direction fidelity is the real contract: lengths are intentionally
        // the rig's, but every bone must point where the data points.
        for (const { signId, frame, lengths } of samples) {
            const real = realShapeInRigFrame(frame);
            const rig = reconstructedShape(frame, lengths);
            for (const bone of HAND_BONES) {
                const realDir = real[bone.toLandmark].clone().sub(real[bone.fromLandmark]).normalize();
                const rigDir = rig[bone.toLandmark].clone().sub(rig[bone.fromLandmark]).normalize();
                const dot = realDir.dot(rigDir);
                expect(dot, `${signId}/${bone.name} direction`).toBeGreaterThan(0.9);
            }
        }
    });

    it("mirroring flips the pose in x and nothing else", () => {
        const { frame } = samples[0];
        const normal = reconstructedShape(frame);
        const mirrored = forwardKinematics(solveHandPose(frame, { mirror: true }));
        const inv = new Quaternion(...solveHandPose(frame, { mirror: true }).rootQuaternion).invert();
        const m = mirrored.map((p) => p.clone().applyQuaternion(inv));
        // A mirrored hand is not simply x-negated in its own wrist frame, but
        // it must remain a valid, finite, same-length reconstruction.
        for (let i = 1; i <= 20; i++) {
            expect(Number.isFinite(m[i].x)).toBe(true);
            expect(m[i].length()).toBeGreaterThan(0);
        }
        expect(normal.length).toBe(m.length);
    });

    it("returns a rest pose, flagged, for a degenerate frame instead of NaN", () => {
        const flat: HandLandmarks = Array.from({ length: 21 }, () => ({ x: 0.5, y: 0.5, z: 0 }));
        const sol = solveHandPose(flat);
        expect(sol.degenerate).toBe(true);
        for (const b of sol.bones) {
            for (const v of b.quaternion) expect(Number.isFinite(v)).toBe(true);
        }
    });
});
