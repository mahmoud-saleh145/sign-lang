/**
 * CANONICAL LEFT/RIGHT ASSIGNMENT for the LIVE camera path.
 *
 * This is the browser twin of ml/training/handedness.py, and it must stay
 * behaviourally identical to it: the registry's stored sides and the live
 * sides have to mean the same physical hand, or side-aware matching compares
 * a left hand against a right template and scores zero. The thresholds below
 * are the same values, derived from the same measurements on real KArSL data
 * (204 takes, 11,975 detected hand-frames, 4,926 with both hands) -- see that
 * file's docstring for the numbers.
 *
 * THE RULE, in priority order:
 *
 *  1. GEOMETRY. A signer faces the camera and the frames MediaPipe receives
 *     are NOT mirrored (the app mirrors only the CSS preview -- see
 *     components/camera/CameraView.tsx), so the signer's RIGHT hand sits at
 *     the SMALLER image x. On frames where both hands are present this needs
 *     no model output at all. Measured per-take agreement on KArSL: median
 *     1.000, mean 0.972, 147/170 takes perfectly consistent.
 *
 *  2. MEDIAPIPE'S HANDEDNESS LABEL, as a per-track majority -- never from a
 *     single frame. Measured 95.9% agreement with geometry, but 30.4% of
 *     tracks contain at least one disagreeing frame, so a single-frame read is
 *     unreliable. Used when a track never shows both hands at once, which is
 *     the genuinely one-handed case (34/204 takes).
 *
 *  3. UNKNOWN. Reported, not guessed. A caller that cannot name a hand must
 *     decide what to do about it rather than being handed a coin flip.
 *
 * On the label mapping: MediaPipe documents handedness as assuming a mirrored
 * (selfie) image, which would mean inverting it for a non-mirrored frame.
 * Measured against geometry on this data the label maps to anatomy DIRECTLY
 * -- "Right" is the signer's right hand -- so no inversion is applied.
 * INVERT_HANDEDNESS_LABEL exists for a genuinely mirrored source.
 *
 * NOT VERIFIED AGAINST A LIVE WEBCAM. The measurement above is from KArSL
 * video. The live camera feeds MediaPipe an un-mirrored frame of a signer
 * facing it, which is the same geometry, but that has not been tested with a
 * real person in front of a real camera and this comment does not claim it
 * has. Rule 1 is self-correcting if the label convention differs live; rule 2
 * is the part that would be wrong, and `SideAssignment.source` says when a
 * result depended on it.
 */
import type {
    HandLandmarks,
    Handedness,
    SideAssignment,
    SidedSequence,
    SignerSide,
} from "@/types/landmarks";
import { oppositeSide } from "@/types/landmarks";

/** Fewest both-hand frames before the geometric vote is trusted at all. */
export const GEOMETRY_MIN_FRAMES = 3;
/**
 * Fraction of both-hand frames that must agree for geometry to decide.
 * From the measured per-take distribution: p10 = 0.913 and 147/170 takes are
 * 100% consistent, so 0.90 takes the clear cases and defers on the 15 takes
 * where the hands genuinely cross.
 */
export const GEOMETRY_MIN_AGREEMENT = 0.9;
/** Fraction of a track's frames that must share a label for it to decide. */
export const LABEL_MIN_AGREEMENT = 0.6;

/**
 * Whether the image MediaPipe sees is horizontally mirrored.
 *
 * ONE SWITCH FOR THE WHOLE CONVENTION. Both signals below depend on it: in a
 * mirrored image the signer's right hand moves to the LARGER x, and
 * MediaPipe's label flips with it. Setting this true inverts both together, so
 * they can never end up disagreeing.
 *
 * False here because the app hands MediaPipe the raw video element and mirrors
 * only the CSS preview (components/camera/CameraView.tsx), and because KArSL
 * source video is an ordinary non-mirrored recording.
 *
 * SYMPTOM IF THIS IS WRONG for a given deployment: every gesture's hands are
 * named the wrong way round, so nothing matches and almost everything is
 * rejected -- a uniform failure, not a subtle degradation. The "hand sides
 * from" and "captured frames L/R" rows in the debug panel are what to look at.
 */
export const CAMERA_IMAGE_IS_MIRRORED = false;

/** Kept as a separate name because it is what the offline twin calls it. */
export const INVERT_HANDEDNESS_LABEL = CAMERA_IMAGE_IS_MIRRORED;

/** One tracked hand's observations over time; `null` where it was not seen. */
export type SlotTrack = ReadonlyArray<HandLandmarks | null>;
export type SlotLabels = ReadonlyArray<Handedness | null>;

function wristX(frame: HandLandmarks | null): number | null {
    if (!frame || frame.length === 0) return null;
    return frame[0].x;
}

/** (frames where slot A sits at the smaller image x, frames comparable). */
export function geometricVote(slotA: SlotTrack, slotB: SlotTrack): { aSmaller: number; compared: number } {
    let aSmaller = 0;
    let compared = 0;
    const n = Math.min(slotA.length, slotB.length);
    for (let i = 0; i < n; i++) {
        const xa = wristX(slotA[i]);
        const xb = wristX(slotB[i]);
        if (xa === null || xb === null) continue;
        compared++;
        if (xa < xb) aSmaller++;
    }
    return { aSmaller, compared };
}

/** (majority label for one slot, agreement fraction). */
export function labelVote(labels: SlotLabels): { label: Handedness | null; agreement: number } {
    const seen = labels.filter((l): l is Handedness => l === "Left" || l === "Right");
    if (seen.length === 0) return { label: null, agreement: 0 };
    let left = 0;
    for (const l of seen) if (l === "Left") left++;
    const right = seen.length - left;
    const label: Handedness = left >= right ? "Left" : "Right";
    return { label, agreement: Math.max(left, right) / seen.length };
}

function anatomicalOf(label: Handedness): SignerSide {
    const direct: SignerSide = label === "Left" ? "left" : "right";
    return INVERT_HANDEDNESS_LABEL ? oppositeSide(direct) : direct;
}

/**
 * Names two tracked hands as the signer's left and right.
 *
 * The two slots always receive OPPOSITE names when either can be named: by
 * construction of the slot tracker they hold two different physical hands, so
 * naming both the same side would be a contradiction rather than a guess.
 */
export function assignSides(
    slotA: SlotTrack,
    slotB: SlotTrack,
    labelsA: SlotLabels,
    labelsB: SlotLabels
): SideAssignment {
    const { aSmaller, compared } = geometricVote(slotA, slotB);
    const geometryAgreement = compared > 0 ? Math.max(aSmaller, compared - aSmaller) / compared : 0;

    if (compared >= GEOMETRY_MIN_FRAMES && geometryAgreement >= GEOMETRY_MIN_AGREEMENT) {
        // Smaller image x is the signer's right hand -- unless the image is
        // mirrored, in which case it is their left.
        const aIsSmaller = aSmaller * 2 >= compared;
        const aIsRight = CAMERA_IMAGE_IS_MIRRORED ? !aIsSmaller : aIsSmaller;
        return {
            sides: aIsRight ? ["right", "left"] : ["left", "right"],
            source: "geometry",
            confidence: geometryAgreement,
            geometryFrames: compared,
        };
    }

    const voteA = labelVote(labelsA);
    const voteB = labelVote(labelsB);
    const bestAgreement = Math.max(voteA.agreement, voteB.agreement);

    if (bestAgreement >= LABEL_MIN_AGREEMENT && (voteA.label || voteB.label)) {
        let sideA: SignerSide | null = null;
        // Trust whichever slot the classifier was more consistent about and
        // derive the other from it, so the two can never collide.
        if (voteA.agreement >= voteB.agreement && voteA.label) {
            sideA = anatomicalOf(voteA.label);
        } else if (voteB.label) {
            sideA = oppositeSide(anatomicalOf(voteB.label));
        }
        if (sideA) {
            return {
                sides: [sideA, oppositeSide(sideA)],
                source: "handedness-label",
                confidence: bestAgreement,
                geometryFrames: compared,
            };
        }
    }

    return {
        sides: ["unknown", "unknown"],
        source: "unknown",
        confidence: 0,
        geometryFrames: compared,
    };
}

/**
 * Frames where BOTH hands were detected together. Only these can carry the
 * geometric vote -- the rule is a COMPARISON between two hands, so a frame
 * holding one hand tells it nothing.
 */
export function twoHandFrameCount(frames: SidedSequence): number {
    let both = 0;
    for (const f of frames) if (f.left && f.right) both++;
    return both;
}

/**
 * True when this timeline carries NO geometric evidence for its own sides:
 * the two hands were never seen together often enough to compare, so
 * whichever left/right names it wears came from rule 2, the handedness label.
 *
 * The label is right about 19 times in 20 (measured 760/798 against geometry
 * on the takes where geometry can vote), so these names are usually correct
 * -- but "usually" is not "measured", and a caller matching against a
 * side-aware template is entitled to know the difference. Same threshold as
 * assignSides uses to decide whether to trust geometry, so the two can never
 * disagree about what counts as evidence.
 */
export function hasGeometricSideEvidence(frames: SidedSequence): boolean {
    return twoHandFrameCount(frames) >= GEOMETRY_MIN_FRAMES;
}
