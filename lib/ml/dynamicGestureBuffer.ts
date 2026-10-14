import type {
    HandLandmarks,
    SidedLandmarkFrame,
    SidedSequence,
    SignerSide,
} from "@/types/landmarks";
import { SIGNER_SIDES } from "@/types/landmarks";
import type { Handedness, SideAssignment } from "@/types/landmarks";
import { assignSides } from "./handSides";
import type { SlotFrame } from "./handSlotTracker";

/** A frame in which neither hand was detected. */
export const EMPTY_SIDED_FRAME: SidedLandmarkFrame = { left: null, right: null };

/**
 * Buffers consecutive frames into a candidate "gesture" for dynamic KArSL
 * recognition. DELIBERATELY separate from lib/ml/segmentation.ts: that
 * state machine debounces a repeated per-frame classification label (static
 * letter model); this one accumulates a whole sequence of raw landmarks to
 * hand to a temporal recognizer (lib/ml/dynamicSignRecognizer.ts).
 *
 * ============================================================================
 * GESTURE BOUNDARIES -- and why there is no motion-based "gesture end"
 * ============================================================================
 * GESTURE START: any frame with a detected hand while idle, or an explicit
 * beginManualCapture() call.
 * GESTURE END: `endGapFrames` consecutive no-hand frames, OR the buffer
 * reaching `maxGestureFrames`, OR an explicit endManualCapture() call.
 *
 * A stillness/velocity-based end detector was evaluated against the REAL
 * imported KArSL sequences in public/models/sign-vocabulary/dedicated-signs.json
 * and rejected on the evidence. Measured over all 170 real sequences
 * (avg per-frame displacement of the 21 landmarks, MediaPipe [0,1] image
 * coordinates):
 *
 *   per-frame displacement:   p10=0.00023  p50=0.00081  p90=0.00937
 *   per-SEQUENCE peak disp.:  p0 =0.00102  p50=0.00722  p100=0.53284
 *
 * The quietest real signs never exceed 0.001 at their peak, while the median
 * frame of an active sign sits at 0.0008. The two distributions overlap
 * almost completely, so no absolute stillness threshold can separate "signer
 * paused" from "signer performing a low-motion sign" without cutting real
 * signs in half. Hand presence is used instead because it is the same
 * boundary the source data itself uses (one KArSL video = one sign), and
 * manual capture is offered for deterministic testing.
 *
 * ============================================================================
 * WHY minMotionEnergy IS 0.0001 AND NOT 0.02
 * ============================================================================
 * This filter exists only to drop a literally frozen buffer (a held static
 * letter pose) before it reaches DTW. The previous default of 0.02 was a
 * guess made before real data existed, and it is catastrophically wrong:
 * measured against the real registry, 164 of 170 sequences (96.5%) have an
 * average motion energy BELOW 0.02 -- median 0.00155, minimum 0.000193.
 * A 0.02 gate therefore discards 96.5% of the dataset's own real signs
 * before classification, which is exactly the failure this module was
 * reported to have. 0.0001 sits below the real observed minimum
 * (0.000193 sequence average; 0.000116 single-frame p1), so it excludes a
 * frozen buffer and nothing real. Rejection of wrong matches is handled
 * where it belongs -- the DTW distance/margin test in
 * dynamicSignRecognizer.ts -- not by a motion heuristic.
 *
 * ============================================================================
 * WHY THE FRAME CAP NO LONGER DISCARDS
 * ============================================================================
 * Previously, hitting `maxGestureFrames` finalized the buffer and then let
 * the motion filter throw it away. A signer who simply keeps their hand in
 * frame therefore produced an endless cycle of buffer -> cap -> discard,
 * with the classifier never invoked. The cap now finalizes and CLASSIFIES,
 * and `endedByFrameCap` is reported so a truncated capture is visible
 * rather than silent.
 *
 * DUPLICATE PREVENTION: a gesture is buffered continuously while the hand
 * stays present and finalized once, so one continuous gesture produces at
 * most one completed-gesture event.
 */

export type GestureBufferState = "idle" | "buffering";

/** Why a finalized buffer was thrown away instead of classified. */
/**
 * Why a finalized buffer was thrown away instead of classified.
 *
 * "sides-unresolved" is distinct from "too-few-frames" on purpose. A gesture
 * can have plenty of frames and real motion and still produce nothing to
 * classify, because assignSides() could name neither tracked hand and no
 * single unnamed hand could be salvaged either. Reporting that as
 * "too-few-frames" told the operator to sign for longer, which does not help
 * and is not what happened.
 */
export type GestureDiscardReason = "too-few-frames" | "no-motion" | "sides-unresolved" | null;

/** What ended the most recent gesture. */
export type GestureEndReason = "hand-absent" | "frame-cap" | "manual" | null;

export interface GestureBufferConfig {
    /** Consecutive no-hand frames required to consider a gesture ended. */
    endGapFrames: number;
    /** Minimum buffered frames for a completed gesture to be considered.
     * Real registry sequences run 11-90 frames (p5 = 18) at 30 fps video;
     * the browser loop runs faster, so this is a floor against noise, not a
     * model of sign duration. */
    minGestureFrames: number;
    /** Safety cap: force-finalize (and classify) if the hand never leaves
     * frame. At a typical ~50 fps rAF loop 180 frames is ~3.6 s, comfortably
     * longer than the longest real registry sequence (90 frames of 30 fps
     * video = 3.0 s). The old value of 90 truncated real signs mid-gesture. */
    maxGestureFrames: number;
    /** Floor for "this buffer is not literally frozen". See module docstring
     * for the measured justification of this value. */
    minMotionEnergy: number;
}

export const DEFAULT_GESTURE_BUFFER_CONFIG: GestureBufferConfig = {
    endGapFrames: 6,
    minGestureFrames: 8,
    maxGestureFrames: 180,
    minMotionEnergy: 0.0001,
};

export interface GestureBufferResult {
    state: GestureBufferState;
    /** Non-null exactly on the frame a candidate gesture is ready for recognition. */
    completedGesture: SidedSequence | null;
    /**
     * How the completed gesture's two hands were named, decided once over the
     * WHOLE gesture rather than per frame -- a single frame cannot settle which
     * hand is which. Null except on the frame `completedGesture` is non-null.
     */
    completedSideAssignment: SideAssignment | null;
    /**
     * Set instead of `completedGesture` when exactly one hand was seen and its
     * side could not be determined. The caller must recognise it as a hand of
     * UNKNOWN side rather than pretending it was the dominant one.
     */
    completedUnknownSideHand: HandLandmarks[] | null;
    /** True when a buffer was finalized but discarded by the coarse filters. */
    discardedAsNonDynamic: boolean;
    /** Which filter discarded it, or null. Surfaced so the UI can say WHY
     * nothing happened instead of showing nothing. */
    discardReason: GestureDiscardReason;
    /** What ended the most recently finalized gesture. */
    lastGestureEndReason: GestureEndReason;
    /** True if the most recent gesture was cut off by maxGestureFrames
     * rather than a natural boundary -- a truncated, possibly partial sign. */
    lastGestureEndedByFrameCap: boolean;
    /** Frames buffered right now (0 when idle). */
    bufferedFrameCount: number;
    /** Frame count of the most recently finalized buffer (accepted or discarded). */
    lastGestureFrameCount: number;
    /** Wall-clock duration (ms) of the most recently finalized buffer. */
    lastGestureDurationMs: number | null;
    /** Measured average per-frame landmark displacement of the most recently
     * finalized buffer -- the actual number compared against minMotionEnergy. */
    lastGestureMotionEnergy: number | null;
}

export class DynamicGestureBuffer {
    private state: GestureBufferState = "idle";
    private slotFrames: Array<[HandLandmarks | null, HandLandmarks | null]> = [];
    private slotLabels: Array<[Handedness | null, Handedness | null]> = [];
    private timestamps: number[] = [];
    private gapCount = 0;
    private manualCapture = false;
    private lastGestureFrameCount = 0;
    private lastGestureDurationMs: number | null = null;
    private lastGestureMotionEnergy: number | null = null;
    private lastGestureEndReason: GestureEndReason = null;
    private lastGestureEndedByFrameCap = false;

    constructor(private readonly config: GestureBufferConfig = DEFAULT_GESTURE_BUFFER_CONFIG) { }

    reset(): void {
        this.state = "idle";
        this.slotFrames = [];
        this.slotLabels = [];
        this.timestamps = [];
        this.gapCount = 0;
        this.manualCapture = false;
    }

    /** True while an explicit manual capture window is open. */
    isManualCapture(): boolean {
        return this.manualCapture;
    }

    /**
     * Opens a deterministic capture window: hand-absence no longer ends the
     * gesture, only endManualCapture() (or the frame cap) does. This exists
     * so a tester can isolate recognition from automatic segmentation --
     * the single most useful thing when deciding whether a failure is in the
     * classifier or in the boundary detection.
     */
    beginManualCapture(): void {
        this.slotFrames = [];
        this.slotLabels = [];
        this.timestamps = [];
        this.gapCount = 0;
        this.manualCapture = true;
        this.state = "buffering";
    }

    /** Closes a manual capture window and finalizes whatever was collected. */
    endManualCapture(): GestureBufferResult {
        if (!this.manualCapture) return this.snapshot(null, false, null);
        this.manualCapture = false;
        return this.finalize("manual");
    }

    /** `timestampMs` is optional (defaults to Date.now()) and only used to
     * report lastGestureDurationMs -- it never affects recognition logic. */
    /**
     * Feeds one frame of TRACKED SLOTS (see lib/ml/handSlotTracker.ts).
     *
     * Slots, not sides, on purpose: which physical hand a slot holds is a
     * property of the whole gesture, so it is decided once at finalize() by
     * lib/ml/handSides.ts and never guessed frame by frame.
     */
    step(slots: SlotFrame, timestampMs?: number): GestureBufferResult {
        const ts = timestampMs ?? Date.now();
        const anyHand = slots[0] !== null || slots[1] !== null;

        if (anyHand) {
            this.slotFrames.push([slots[0]?.landmarks ?? null, slots[1]?.landmarks ?? null]);
            this.slotLabels.push([slots[0]?.handedness ?? null, slots[1]?.handedness ?? null]);
            this.timestamps.push(ts);
            this.gapCount = 0;
            this.state = "buffering";

            if (this.slotFrames.length >= this.config.maxGestureFrames) {
                const wasManual = this.manualCapture;
                this.manualCapture = false;
                return this.finalize(wasManual ? "manual" : "frame-cap", true);
            }
            return this.snapshot(null, false, null);
        }

        // No hand this frame.
        if (this.manualCapture) {
            // A manual window deliberately tolerates dropped detections --
            // the tester, not the detector, decides where the sign ends.
            return this.snapshot(null, false, null);
        }

        if (this.state === "buffering") {
            this.gapCount += 1;
            if (this.gapCount >= this.config.endGapFrames) {
                return this.finalize("hand-absent");
            }
            return this.snapshot(null, false, null);
        }

        return this.snapshot(null, false, null, 0);
    }

    private snapshot(
        completedGesture: SidedSequence | null,
        discardedAsNonDynamic: boolean,
        discardReason: GestureDiscardReason,
        bufferedOverride?: number,
        completedSideAssignment: SideAssignment | null = null,
        completedUnknownSideHand: HandLandmarks[] | null = null
    ): GestureBufferResult {
        return {
            state: this.state,
            completedGesture,
            completedSideAssignment,
            completedUnknownSideHand,
            discardedAsNonDynamic,
            discardReason,
            lastGestureEndReason: this.lastGestureEndReason,
            lastGestureEndedByFrameCap: this.lastGestureEndedByFrameCap,
            bufferedFrameCount: bufferedOverride ?? this.slotFrames.length,
            lastGestureFrameCount: this.lastGestureFrameCount,
            lastGestureDurationMs: this.lastGestureDurationMs,
            lastGestureMotionEnergy: this.lastGestureMotionEnergy,
        };
    }

    private finalize(endReason: GestureEndReason, byFrameCap = false): GestureBufferResult {
        const slotFrames = this.slotFrames;
        const slotLabels = this.slotLabels;
        const timestamps = this.timestamps;
        this.slotFrames = [];
        this.slotLabels = [];
        this.timestamps = [];
        this.gapCount = 0;
        this.state = "idle";

        this.lastGestureFrameCount = slotFrames.length;
        this.lastGestureDurationMs =
            timestamps.length >= 2 ? timestamps[timestamps.length - 1] - timestamps[0] : null;
        this.lastGestureEndReason = endReason;
        this.lastGestureEndedByFrameCap = byFrameCap;

        // Name the two tracked hands ONCE, over the whole gesture.
        const assignment = assignSides(
            slotFrames.map((f) => f[0]),
            slotFrames.map((f) => f[1]),
            slotLabels.map((l) => l[0]),
            slotLabels.map((l) => l[1])
        );

        const sided: SidedSequence = [];
        for (const [a, b] of slotFrames) {
            const frame: SidedLandmarkFrame = { left: null, right: null };
            for (const slot of [0, 1] as const) {
                const side = assignment.sides[slot];
                const landmarks = slot === 0 ? a : b;
                if ((side === "left" || side === "right") && landmarks) frame[side] = landmarks;
            }
            if (frame.left || frame.right) sided.push(frame);
        }

        // Neither hand could be named. If exactly one hand was seen, hand it
        // back as an explicitly UNNAMED hand rather than inventing a side.
        let unknownHand: HandLandmarks[] | null = null;
        if (assignment.source === "unknown") {
            const slotWithData = slotFrames.some((f) => f[0]) ? 0 : slotFrames.some((f) => f[1]) ? 1 : null;
            if (slotWithData !== null) {
                unknownHand = slotFrames
                    .map((f) => f[slotWithData])
                    .filter((f): f is HandLandmarks => f !== null);
            }
        }

        const energySource: SidedSequence =
            sided.length > 0
                ? sided
                : (unknownHand ?? []).map((f) => ({ left: f, right: null }));
        this.lastGestureMotionEnergy =
            energySource.length >= 2 ? computeMotionEnergy(energySource) : null;

        if (slotFrames.length < this.config.minGestureFrames) {
            return this.snapshot(null, true, "too-few-frames", 0);
        }

        if (this.lastGestureMotionEnergy === null || this.lastGestureMotionEnergy < this.config.minMotionEnergy) {
            return this.snapshot(null, true, "no-motion", 0);
        }

        // Frames and motion were both fine; there was simply no hand this
        // sequence could be named with. Say so.
        if (sided.length === 0 && unknownHand === null) {
            return this.snapshot(null, true, "sides-unresolved", 0, assignment);
        }

        return this.snapshot(
            sided.length > 0 ? sided : null,
            false,
            null,
            0,
            assignment,
            sided.length > 0 ? null : unknownHand
        );
    }
}

/**
 * Average per-frame raw landmark displacement across a whole sequence.
 *
 * TWO HANDS: a frame pair contributes the mean displacement over the sides
 * present in BOTH of its frames. A hand that appears or disappears between two
 * frames contributes nothing rather than an enormous fake displacement, and a
 * frame pair that shares no hand is skipped entirely. This keeps the measured
 * energy on the same scale as the single-hand version the gate was calibrated
 * against (real KArSL sequences: min 0.000193, median 0.001553), so
 * minMotionEnergy did not have to move.
 *
 * Exported so telemetry and tests can inspect the actual number.
 */
export function computeMotionEnergy(frames: SidedSequence): number {
    if (frames.length < 2) return 0;
    let totalDisplacement = 0;
    let comparedPairs = 0;
    for (let i = 1; i < frames.length; i++) {
        const perSide: number[] = [];
        for (const side of SIGNER_SIDES) {
            const a = frames[i - 1][side];
            const b = frames[i][side];
            if (a && b) perSide.push(averagePointDisplacement(a, b));
        }
        if (perSide.length === 0) continue;
        totalDisplacement += perSide.reduce((x, y) => x + y, 0) / perSide.length;
        comparedPairs++;
    }
    return comparedPairs === 0 ? 0 : totalDisplacement / comparedPairs;
}

/** Convenience for a single hand of known side, e.g. a stored one-hand sequence. */
export function sidedFramesFromSingleHand(
    side: SignerSide,
    frames: ReadonlyArray<HandLandmarks>
): SidedSequence {
    return frames.map((f) => (side === "left" ? { left: f, right: null } : { left: null, right: f }));
}

function averagePointDisplacement(a: HandLandmarks, b: HandLandmarks): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y);
    }
    return sum / a.length;
}
