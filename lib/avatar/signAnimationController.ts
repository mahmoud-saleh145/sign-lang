import type { AnimatableSign } from "./karslSign";
import { totalDurationMs } from "./karslSign";

/**
 * Playback across a list of AnimatableSigns, driven by real per-sign
 * durations.
 *
 * WHY THIS REPLACES THE FIXED-DURATION MODEL: the previous
 * lib/playback/playbackController.ts gave every step the same
 * stepDurationMs (1200 ms). Real registry sequences run 11 to 90 frames at
 * the source 30 fps -- 0.37 s to 3.00 s. A fixed step therefore played every
 * sign at the wrong speed, compressing a 3 s sign by 2.5x and stretching a
 * 0.37 s one by 3.2x. Since the whole point is to reproduce real temporal
 * motion, duration has to come from the data. That controller is untouched
 * and still used by its existing tests; this one is what the avatar uses.
 *
 * Like its predecessor, this class owns no timer. The caller feeds it
 * elapsed time via tick(), which keeps it unit-testable and lets the 3D
 * layer drive it from its own render loop without React re-rendering.
 */

export type AnimationState = "idle" | "playing" | "paused" | "finished";

export type PlaybackSpeed = 0.25 | 0.5 | 0.75 | 1 | 1.5 | 2;

export const PLAYBACK_SPEEDS: readonly PlaybackSpeed[] = [0.25, 0.5, 0.75, 1, 1.5, 2];

export interface AnimationSnapshot {
    state: AnimationState;
    /** Index into the sign list. */
    signIndex: number;
    signCount: number;
    speed: PlaybackSpeed;
    /** 0..1 through the CURRENT sign. Feed to the pose sampler. */
    progressWithinSign: number;
    /** Elapsed ms within the current sign, at 1x-equivalent source time. */
    elapsedInSignMs: number;
    /** Elapsed ms across the whole list. */
    elapsedTotalMs: number;
    totalDurationMs: number;
    /** Fractional frame index into the current sign's real frames. */
    frameIndex: number;
}

export class SignAnimationController {
    private state: AnimationState = "idle";
    private signIndex = 0;
    private elapsedInSignMs = 0;
    private speed: PlaybackSpeed = 1;
    private signs: AnimatableSign[] = [];

    constructor(signs: readonly AnimatableSign[] = []) {
        this.signs = [...signs];
    }

    setSigns(signs: readonly AnimatableSign[]): void {
        this.signs = [...signs];
        this.reset();
    }

    getSigns(): readonly AnimatableSign[] {
        return this.signs;
    }

    getCurrentSign(): AnimatableSign | null {
        return this.signs[this.signIndex] ?? null;
    }

    setSpeed(speed: PlaybackSpeed): void {
        this.speed = speed;
    }

    play(): void {
        if (this.signs.length === 0) {
            this.state = "finished";
            return;
        }
        if (this.state === "finished" || this.state === "idle") {
            this.signIndex = 0;
            this.elapsedInSignMs = 0;
        }
        this.state = "playing";
    }

    pause(): void {
        if (this.state === "playing") this.state = "paused";
    }

    resume(): void {
        if (this.state === "paused") this.state = "playing";
    }

    replay(): void {
        this.signIndex = 0;
        this.elapsedInSignMs = 0;
        this.state = this.signs.length === 0 ? "finished" : "playing";
    }

    stop(): void {
        this.reset();
    }

    reset(): void {
        this.state = "idle";
        this.signIndex = 0;
        this.elapsedInSignMs = 0;
    }

    /** Jump to a specific sign (used by the dev page's sign picker). */
    seekToSign(index: number): void {
        if (index < 0 || index >= this.signs.length) return;
        this.signIndex = index;
        this.elapsedInSignMs = 0;
    }

    /** Advance by `deltaMs` of REAL time. Speed scales how much source time
     * that consumes, so 2x speed plays a 3.0 s sign in 1.5 s of wall clock
     * while still walking the same real frames. */
    tick(deltaMs: number): AnimationSnapshot {
        if (this.state !== "playing") return this.getSnapshot();

        this.elapsedInSignMs += deltaMs * this.speed;

        // `while`, not `if`: a long frame gap (a backgrounded tab) can span
        // more than one short sign, and skipping that would desynchronize
        // the index from elapsed time.
        while (this.state === "playing") {
            const current = this.signs[this.signIndex];
            if (!current) {
                this.state = "finished";
                break;
            }
            if (this.elapsedInSignMs < current.durationMs) break;

            this.elapsedInSignMs -= current.durationMs;
            this.signIndex += 1;
            if (this.signIndex >= this.signs.length) {
                this.signIndex = Math.max(this.signs.length - 1, 0);
                const last = this.signs[this.signIndex];
                this.elapsedInSignMs = last ? last.durationMs : 0;
                this.state = "finished";
            }
        }

        return this.getSnapshot();
    }

    getSnapshot(): AnimationSnapshot {
        const current = this.signs[this.signIndex] ?? null;
        const duration = current?.durationMs ?? 0;
        const progress = duration > 0 ? Math.min(this.elapsedInSignMs / duration, 1) : 0;

        let elapsedTotal = 0;
        for (let i = 0; i < this.signIndex; i++) elapsedTotal += this.signs[i].durationMs;
        elapsedTotal += Math.min(this.elapsedInSignMs, duration);

        const frameCount = current?.frameCount ?? 0;
        const frameIndex = frameCount > 1 ? progress * (frameCount - 1) : 0;

        return {
            state: this.state,
            signIndex: this.signIndex,
            signCount: this.signs.length,
            speed: this.speed,
            progressWithinSign: progress,
            elapsedInSignMs: Math.min(this.elapsedInSignMs, duration),
            elapsedTotalMs: elapsedTotal,
            totalDurationMs: totalDurationMs(this.signs),
            frameIndex,
        };
    }
}
