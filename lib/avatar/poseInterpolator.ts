import type { HandLandmarks, Point3D } from "@/types/landmarks";

/**
 * Computes the interpolated hand pose at a point in time within one sign's
 * animation. `progress` is 0..1 across the full frame sequence.
 *
 * - 0 frames -> null (nothing to show)
 * - 1 frame -> that frame, always (a static sign; this is what ALL our
 *   current real data looks like, since the alphabet dataset has no
 *   temporal/dynamic samples)
 * - 2+ frames -> linear interpolation between the two frames straddling
 *   `progress`, so a real dynamic sequence (once one exists) animates
 *   smoothly instead of jumping between poses.
 *
 * This never fabricates motion: with fewer than 2 real frames there is
 * nothing to interpolate, so it simply holds the single real pose.
 */
export function interpolatePose(frames: HandLandmarks[], progress: number): HandLandmarks | null {
    if (frames.length === 0) return null;
    if (frames.length === 1) return frames[0];

    const clamped = Math.min(Math.max(progress, 0), 1);
    const scaled = clamped * (frames.length - 1);
    const lowerIndex = Math.floor(scaled);
    const upperIndex = Math.min(lowerIndex + 1, frames.length - 1);
    const t = scaled - lowerIndex;

    const lower = frames[lowerIndex];
    const upper = frames[upperIndex];

    return lower.map((p, i) => lerpPoint(p, upper[i], t));
}

function lerpPoint(a: Point3D, b: Point3D, t: number): Point3D {
    return {
        x: a.x + (b.x - a.x) * t,
        y: a.y + (b.y - a.y) * t,
        z: a.z + (b.z - a.z) * t,
    };
}