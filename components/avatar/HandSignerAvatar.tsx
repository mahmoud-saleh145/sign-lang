import type { HandLandmarks } from "@/types/landmarks";
import { interpolatePose } from "@/lib/avatar/poseInterpolator";

// Standard MediaPipe hand topology (21 landmarks).
const HAND_CONNECTIONS: ReadonlyArray<readonly [number, number]> = [
    [0, 1], [1, 2], [2, 3], [3, 4], // thumb
    [0, 5], [5, 6], [6, 7], [7, 8], // index
    [5, 9], [9, 10], [10, 11], [11, 12], // middle
    [9, 13], [13, 14], [14, 15], [15, 16], // ring
    [13, 17], [17, 18], [18, 19], [19, 20], // pinky
    [0, 17], // palm base
];

const VIEWBOX = 320;
const PADDING = 40;

function toSvgCoords(landmarks: HandLandmarks): Array<{ x: number; y: number }> {
    // Landmarks are MediaPipe-normalized [0,1]. Fit them into the viewbox with
    // padding, preserving aspect ratio so the hand shape isn't distorted.
    const xs = landmarks.map((p) => p.x);
    const ys = landmarks.map((p) => p.y);
    const minX = Math.min(...xs);
    const maxX = Math.max(...xs);
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const spanX = maxX - minX || 1e-6;
    const spanY = maxY - minY || 1e-6;
    const span = Math.max(spanX, spanY);
    const drawable = VIEWBOX - PADDING * 2;

    return landmarks.map((p) => ({
        x: PADDING + ((p.x - minX) / span) * drawable,
        y: PADDING + ((p.y - minY) / span) * drawable,
    }));
}

export interface HandSignerAvatarProps {
    /** All real frames for the current sign. A static sign (all our current
     * data) has exactly 1; a real dynamic sign (once one exists) would have
     * many, and this component will animate smoothly through them. */
    frames: HandLandmarks[];
    /** 0..1 progress through this sign's display duration (from
     * PlaybackController.getSnapshot().progressWithinStep). Ignored/harmless
     * when frames.length <= 1. */
    progress: number;
    /** Shown below the hand: the letter/word this pose represents. */
    caption?: string;
}

export default function HandSignerAvatar({ frames, progress, caption }: HandSignerAvatarProps) {
    const pose = interpolatePose(frames, progress);
    const points = pose ? toSvgCoords(pose) : null;

    return (
        <div className="flex flex-col items-center gap-2">
            <svg
                viewBox={`0 0 ${VIEWBOX} ${VIEWBOX}`}
                className="w-full max-w-xs aspect-square"
                role="img"
                aria-label={caption ? `Hand sign for ${caption}` : "No sign to display"}
            >
                <rect width={VIEWBOX} height={VIEWBOX} rx={24} fill="var(--color-surface)" />
                {points ? (
                    <>
                        {HAND_CONNECTIONS.map(([a, b], i) => (
                            <line
                                key={i}
                                x1={points[a].x}
                                y1={points[a].y}
                                x2={points[b].x}
                                y2={points[b].y}
                                stroke="var(--color-accent-active)"
                                strokeWidth={6}
                                strokeLinecap="round"
                            />
                        ))}
                        {points.map((p, i) => (
                            <circle
                                key={i}
                                cx={p.x}
                                cy={p.y}
                                r={i === 0 ? 10 : 7}
                                fill={i === 0 ? "var(--color-accent-candidate)" : "#ffffff"}
                                stroke="var(--color-bg)"
                                strokeWidth={2}
                            />
                        ))}
                    </>
                ) : (
                    <text
                        x={VIEWBOX / 2}
                        y={VIEWBOX / 2}
                        textAnchor="middle"
                        fill="var(--color-text-dim)"
                        fontSize={16}
                    >
                        No sign available
                    </text>
                )}
            </svg>
            {caption && (
                <p className="font-display text-2xl" dir="rtl" lang="ar">
                    {caption}
                </p>
            )}
        </div>
    );
}