/**
 * TEMPORAL HAND IDENTITY for the LIVE camera path.
 *
 * MediaPipe returns detections in an order that is not stable between frames.
 * Measured on real KArSL video (the same model family, ml/training/
 * extract_karsl_skeletons.py), tracking which physical hand
 * `hand_landmarks[0]` was, frame by frame:
 *
 *     0171  LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL RRRRRRRRRRRRRRRRRRRRRRRRRR
 *     0241  RRRRRRRRLRRRRRRRRLLRRLLLLLLLLLLLLLLL
 *     0255  LLLLLLLLLLLLLLLLLLLLLLLLL RRRRRRRRRRRRRR
 *
 * Six of fourteen sampled videos swapped mid-sign. Following detection order
 * therefore produces a "trajectory" that is one hand spliced onto the other,
 * which is not a real motion and cannot be matched or animated.
 *
 * This is the browser twin of HandSlotTracker in
 * ml/training/extract_karsl_skeletons.py and uses the same rule and the same
 * gate: hands are two persistent SLOTS, assigned frame to frame by whichever
 * pairing minimises total wrist displacement. Slots carry no left/right
 * meaning -- naming happens once, over the whole gesture, in
 * lib/ml/handSides.ts.
 */
import type { HandLandmarks, Handedness } from "@/types/landmarks";

/**
 * Largest normalised wrist movement between consecutive frames that can still
 * be the same hand continuing. 0.25 of image width dwarfs real inter-frame
 * hand motion (measured p90 per-frame landmark displacement ~0.009) while
 * still rejecting a jump across to the other hand (measured swaps produced
 * 0.21-0.24). Same value as the offline tracker.
 */
export const SLOT_CONTINUITY_GATE = 0.25;

export interface SlotDetection {
    landmarks: HandLandmarks;
    handedness: Handedness | null;
    score: number;
}

/** Slot 0 and slot 1 for one frame; `null` where that slot saw nothing. */
export type SlotFrame = [SlotDetection | null, SlotDetection | null];

type Wrist = { x: number; y: number };

function wristOf(landmarks: HandLandmarks): Wrist {
    return { x: landmarks[0].x, y: landmarks[0].y };
}

function distance(a: Wrist, b: Wrist): number {
    return Math.hypot(a.x - b.x, a.y - b.y);
}

export class HandSlotTracker {
    private readonly gate: number;
    private last: [Wrist | null, Wrist | null] = [null, null];
    private seedLabels: [Handedness | null, Handedness | null] = [null, null];
    /** How often MediaPipe's ordering had to be swapped to keep identity. */
    public switchCount = 0;

    constructor(gate: number = SLOT_CONTINUITY_GATE) {
        this.gate = gate;
    }

    reset(): void {
        this.last = [null, null];
        this.seedLabels = [null, null];
        this.switchCount = 0;
    }

    step(detections: readonly SlotDetection[]): SlotFrame {
        const slots: SlotFrame = [null, null];

        if (detections.length >= 2) {
            let a = detections[0];
            let b = detections[1];
            let wa = wristOf(a.landmarks);
            let wb = wristOf(b.landmarks);

            const [l0, l1] = this.last;
            if (l0 && l1) {
                const keep = distance(wa, l0) + distance(wb, l1);
                const swap = distance(wb, l0) + distance(wa, l1);
                if (swap < keep) {
                    [a, b] = [b, a];
                    [wa, wb] = [wb, wa];
                    this.switchCount++;
                }
            }
            slots[0] = a;
            slots[1] = b;
            this.last = [wa, wb];
            for (const i of [0, 1] as const) {
                const d = i === 0 ? a : b;
                if (this.seedLabels[i] === null) this.seedLabels[i] = d.handedness;
            }
        } else if (detections.length === 1) {
            const d = detections[0];
            const w = wristOf(d.landmarks);
            let target: 0 | 1 | null = null;

            const candidates: Array<{ i: 0 | 1; dist: number }> = [];
            for (const i of [0, 1] as const) {
                const l = this.last[i];
                if (l) candidates.push({ i, dist: distance(w, l) });
            }
            if (candidates.length > 0) {
                const nearest = candidates.reduce((m, c) => (c.dist < m.dist ? c : m));
                if (nearest.dist <= this.gate) target = nearest.i;
            }
            if (target === null) {
                // No usable history: prefer the slot already associated with
                // this label, else the free slot.
                for (const i of [0, 1] as const) {
                    if (d.handedness !== null && this.seedLabels[i] === d.handedness) {
                        target = i;
                        break;
                    }
                }
                if (target === null) target = this.last[0] === null ? 0 : 1;
            }

            slots[target] = d;
            this.last[target] = w;
            if (this.seedLabels[target] === null) this.seedLabels[target] = d.handedness;
        }

        return slots;
    }
}
