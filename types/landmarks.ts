/**
 * Types for MediaPipe hand landmark data.
 *
 * MediaPipe Hand Landmarker returns 21 keypoints per detected hand, each with
 * normalized (x, y, z) coordinates in [0, 1] relative to the image, plus a
 * handedness classification.
 */

export interface Point2D {
  x: number;
  y: number;
}

export interface Point3D extends Point2D {
  z: number;
}

/** A single hand's 21 landmarks, as returned directly by MediaPipe. */
export type HandLandmarks = Point3D[];

export type Handedness = "Left" | "Right";

export interface DetectedHand {
  landmarks: HandLandmarks;
  /**
   * MediaPipe's raw per-frame label, or null when it did not produce one.
   * This is a MODEL OUTPUT in the camera's frame of reference, NOT the
   * signer's anatomical side -- see SignerSide below and lib/ml/handSides.ts.
   */
  handedness: Handedness | null;
  score: number;
}

/** Raw output of one frame of MediaPipe hand detection. */
export interface FrameDetection {
  timestampMs: number;
  hands: DetectedHand[];
}

/**
 * The SIGNER's own left/right hand. This is an anatomical identity, not
 * MediaPipe's output order and not the `Handedness` label above -- see
 * lib/ml/handSides.ts for how a detection is assigned one, and
 * ml/training/handedness.py for the same rule on the offline side.
 */
export type SignerSide = "left" | "right";

export const SIGNER_SIDES: readonly SignerSide[] = ["left", "right"] as const;

export function oppositeSide(side: SignerSide): SignerSide {
  return side === "left" ? "right" : "left";
}

/**
 * One frame of a two-hand timeline. A side is `null` when that hand was not
 * detected in that frame. It is NEVER interpolated, copied from the other
 * hand, or carried over from a neighbouring frame -- absence is real data.
 */
export interface SidedLandmarkFrame {
  left: HandLandmarks | null;
  right: HandLandmarks | null;
}

/** Both hands on one shared timeline. */
export type SidedSequence = SidedLandmarkFrame[];

/** How a left/right assignment was arrived at. */
export type SideSource = "geometry" | "handedness-label" | "unknown";

/** The outcome of naming two tracked hands. `sides[i]` names tracker slot i. */
export interface SideAssignment {
  sides: [SignerSide | "unknown", SignerSide | "unknown"];
  source: SideSource;
  /** Agreement fraction of whichever vote decided; 0 when nothing decided. */
  confidence: number;
  /** Frames in which both hands were present, i.e. usable by the geometry vote. */
  geometryFrames: number;
}
