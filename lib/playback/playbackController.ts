export type PlaybackState = "idle" | "playing" | "paused" | "finished";

export type PlaybackSpeed = 0.5 | 0.75 | 1 | 1.25;

export interface PlaybackConfig {
    /** How long each step is shown at 1x speed. Default prioritizes clarity over speed. */
    stepDurationMs: number;
}

export const DEFAULT_PLAYBACK_CONFIG: PlaybackConfig = {
    stepDurationMs: 1200,
};

export interface PlaybackSnapshot {
    state: PlaybackState;
    currentStepIndex: number;
    totalSteps: number;
    speed: PlaybackSpeed;
    /** 0..1 progress through the current step's display duration -- feed this
     * into interpolatePose() (lib/avatar/poseInterpolator.ts) to animate a
     * real multi-frame sign smoothly rather than jumping between frames. */
    progressWithinStep: number;
}

/**
 * Drives sequential playback through a flat list of steps (typically the
 * flattened SignStep[] of a SignSequence). Caller supplies elapsed time via
 * tick(); this class has no setInterval/requestAnimationFrame of its own,
 * which keeps it trivially unit-testable and reusable outside React.
 */
export class PlaybackController {
    private state: PlaybackState = "idle";
    private currentStepIndex = 0;
    private elapsedInStepMs = 0;
    private speed: PlaybackSpeed = 1;

    constructor(
        private totalSteps: number,
        private readonly config: PlaybackConfig = DEFAULT_PLAYBACK_CONFIG
    ) { }

    getSnapshot(): PlaybackSnapshot {
        return {
            state: this.state,
            currentStepIndex: this.currentStepIndex,
            totalSteps: this.totalSteps,
            speed: this.speed,
            progressWithinStep:
                this.state === "finished"
                    ? 1
                    : Math.min(this.elapsedInStepMs / this.config.stepDurationMs, 1),
        };
    }

    setSequenceLength(totalSteps: number): void {
        this.totalSteps = totalSteps;
        this.reset();
    }

    setSpeed(speed: PlaybackSpeed): void {
        this.speed = speed;
    }

    play(): void {
        if (this.totalSteps === 0) {
            this.state = "finished";
            return;
        }
        if (this.state === "finished" || this.state === "idle") {
            this.currentStepIndex = 0;
            this.elapsedInStepMs = 0;
        }
        this.state = "playing";
    }

    pause(): void {
        if (this.state === "playing") {
            this.state = "paused";
        }
    }

    resume(): void {
        if (this.state === "paused") {
            this.state = "playing";
        }
    }

    replay(): void {
        this.currentStepIndex = 0;
        this.elapsedInStepMs = 0;
        this.state = this.totalSteps === 0 ? "finished" : "playing";
    }

    reset(): void {
        this.state = "idle";
        this.currentStepIndex = 0;
        this.elapsedInStepMs = 0;
    }

    /** Advance playback by `deltaMs` of real time. Call this on every animation
     * frame / interval tick while playing; it's a no-op unless state is "playing". */
    tick(deltaMs: number): PlaybackSnapshot {
        if (this.state !== "playing") return this.getSnapshot();

        this.elapsedInStepMs += deltaMs * this.speed;
        const stepDuration = this.config.stepDurationMs;

        while (this.elapsedInStepMs >= stepDuration && this.state === "playing") {
            this.elapsedInStepMs -= stepDuration;
            this.currentStepIndex += 1;
            if (this.currentStepIndex >= this.totalSteps) {
                this.currentStepIndex = Math.max(this.totalSteps - 1, 0);
                this.state = "finished";
            }
        }

        return this.getSnapshot();
    }
}