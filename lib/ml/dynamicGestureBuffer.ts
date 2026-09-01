import type { HandLandmarks } from "@/types/landmarks";

/**
 * Buffers consecutive frames into a candidate "gesture" for dynamic KArSL
 * recognition. This is DELIBERATELY separate from lib/ml/segmentation.ts:
 * that state machine debounces a single repeated per-frame classification
 * label (for the static letter model); this one accumulates a whole
 * sequence of raw landmarks to hand to a temporal recognizer
 * (lib/ml/dynamicSignRecognizer.ts). They serve different purposes and
 * operate on different signals -- reusing segmentation.ts here would
 * conflate "is this the same label repeating" with "is a gesture in
 * progress," which are not the same question.
 *
 * GESTURE START: any frame with a detected hand while idle.
 * GESTURE END: `endGapFrames` consecutive frames with no detected hand
 * (the signer's hand leaving frame/dropping is treated as the natural
 * boundary between one sign and the next -- see the module-level "why not
 * motion velocity" note below), OR the buffer reaching `maxGestureFrames`
 * (a safety cap so a hand that never leaves frame doesn't buffer forever).
 *
 * WHY HAND-PRESENCE, NOT MOTION-VELOCITY, FOR SEGMENTATION: a motion-based
 * "gesture start/stop" detector (e.g. thresholding landmark velocity) would
 * need real KArSL motion-profile statistics to tune reliably, which were
 * not available when this was built. Hand-presence boundaries are simpler,
 * verifiable without that data, and match how continuous letter signing
 * already works in this app (idle = no hand). This is a real, disclosed
 * design trade-off, not a claim that it's the only valid approach.
 *
 * COARSE FILTERING BEFORE RECOGNITION: a completed buffer is discarded
 * (never sent to the recognizer) if it's too short (`minGestureFrames`) or
 * shows too little motion (`minMotionEnergy`) -- both are proxies for "this
 * was probably just a held static letter pose," which the existing
 * unaffected static pipeline already owns. This is what prevents every
 * letter hold from wastefully (and riskily) reaching DTW matching.
 *
 * DUPLICATE PREVENTION: because a gesture is buffered continuously while
 * the hand stays present and only finalized once (on hand-drop or the
 * safety cap), holding one continuous gesture produces AT MOST one
 * completed-gesture event, never a repeated stream of the same result.
 *
 * CONSECUTIVE GESTURES: as soon as one gesture finalizes, the buffer
 * resets to idle and immediately accepts a new gesture on the very next
 * frame with a detected hand -- no additional artificial delay is imposed
 * beyond `endGapFrames`, which is the same order of magnitude as
 * segmentation.ts's existing maxGapFrames default.
 */

export type GestureBufferState = "idle" | "buffering";

export interface GestureBufferConfig {
    /** Consecutive no-hand frames required to consider a gesture ended. */
    endGapFrames: number;
    /** Minimum buffered frames for a completed gesture to be considered at
     * all -- shorter buffers are discarded as noise/transition, not sent to
     * the recognizer. */
    minGestureFrames: number;
    /** Safety cap: force-finalize if the hand never leaves frame, so memory
     * and recognition latency stay bounded. */
    maxGestureFrames: number;
    /** Minimum average per-frame raw landmark displacement required to treat
     * a completed buffer as "dynamic" rather than a held static pose. This
     * is a coarse heuristic, not calibrated against real KArSL motion data
     * (unavailable at build time) -- a reasonable conservative default. */
    minMotionEnergy: number;
}

export const DEFAULT_GESTURE_BUFFER_CONFIG: GestureBufferConfig = {
    endGapFrames: 6,
    minGestureFrames: 10,
    maxGestureFrames: 90,
    minMotionEnergy: 0.02,
};

export interface GestureBufferResult {
    state: GestureBufferState;
    /** Non-null exactly on the frame a candidate gesture is ready for recognition. */
    completedGesture: HandLandmarks[] | null;
    /** True when a buffer was finalized but discarded by the coarse filters
     * (too short or too static) -- exposed for debug telemetry, not acted on. */
    discardedAsNonDynamic: boolean;
    /** Current number of frames buffered right now (0 when idle or just after
     * a gesture finalized/was discarded). For debug telemetry. */
    bufferedFrameCount: number;
}

export class DynamicGestureBuffer {
    private state: GestureBufferState = "idle";
    private frames: HandLandmarks[] = [];
    private gapCount = 0;

    constructor(private readonly config: GestureBufferConfig = DEFAULT_GESTURE_BUFFER_CONFIG) { }

    reset(): void {
        this.state = "idle";
        this.frames = [];
        this.gapCount = 0;
    }

    step(landmarks: HandLandmarks | null): GestureBufferResult {
        if (landmarks) {
            this.frames.push(landmarks);
            this.gapCount = 0;
            this.state = "buffering";

            if (this.frames.length >= this.config.maxGestureFrames) {
                return this.finalize();
            }
            return {
                state: this.state,
                completedGesture: null,
                discardedAsNonDynamic: false,
                bufferedFrameCount: this.frames.length,
            };
        }

        if (this.state === "buffering") {
            this.gapCount += 1;
            if (this.gapCount >= this.config.endGapFrames) {
                return this.finalize();
            }
            return {
                state: this.state,
                completedGesture: null,
                discardedAsNonDynamic: false,
                bufferedFrameCount: this.frames.length,
            };
        }

        return { state: "idle", completedGesture: null, discardedAsNonDynamic: false, bufferedFrameCount: 0 };
    }

    private finalize(): GestureBufferResult {
        const frames = this.frames;
        this.frames = [];
        this.gapCount = 0;
        this.state = "idle";

        if (frames.length < this.config.minGestureFrames) {
            return { state: "idle", completedGesture: null, discardedAsNonDynamic: true, bufferedFrameCount: 0 };
        }

        if (!hasEnoughMotion(frames, this.config.minMotionEnergy)) {
            return { state: "idle", completedGesture: null, discardedAsNonDynamic: true, bufferedFrameCount: 0 };
        }

        return { state: "idle", completedGesture: frames, discardedAsNonDynamic: false, bufferedFrameCount: 0 };
    }
}

function hasEnoughMotion(frames: HandLandmarks[], threshold: number): boolean {
    if (frames.length < 2) return false;
    let totalDisplacement = 0;
    for (let i = 1; i < frames.length; i++) {
        totalDisplacement += averagePointDisplacement(frames[i - 1], frames[i]);
    }
    const avgPerFrame = totalDisplacement / (frames.length - 1);
    return avgPerFrame >= threshold;
}

function averagePointDisplacement(a: HandLandmarks, b: HandLandmarks): number {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y);
    }
    return sum / a.length;
}