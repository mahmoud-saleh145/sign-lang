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
  handedness: Handedness;
  score: number;
}

/** Raw output of one frame of MediaPipe hand detection. */
export interface FrameDetection {
  timestampMs: number;
  hands: DetectedHand[];
}
