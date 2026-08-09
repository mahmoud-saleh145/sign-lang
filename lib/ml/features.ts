import type { HandLandmarks, Point3D } from "@/types/landmarks";
import type { FeatureVector } from "@/types/features";
import { FEATURE_COUNT } from "@/types/features";

/**
 * Landmark normalization + feature engineering.
 *
 * IMPORTANT: This file must stay mathematically identical to
 * ml/preprocessing/features.py. If you change a formula here, change it
 * there too, bump PREPROCESSING_VERSION in both places, and retrain.
 *
 * Why we don't use the ArSL-31-Pilot dataset's pre-computed "engineered"
 * columns as-is: that CSV's mean/angle/distance formulas aren't published,
 * so we cannot guarantee we could reproduce them bit-for-bit in the browser.
 * Instead we take the dataset's raw 21-landmark (x, y) columns as source
 * data and define our own normalization + feature pipeline below, which we
 * control end-to-end and can therefore prove is identical at train time and
 * inference time. This also adds translation/scale invariance (spec section
 * 15) that the dataset's raw frame-normalized coordinates lack on their own.
 */

export const PREPROCESSING_VERSION = "1.0.0";

const WRIST = 0;
const MIDDLE_MCP = 9; // used as the scale-reference bone: wrist -> middle MCP

const MEAN_PAIRS: ReadonlyArray<readonly [number, number]> = [
  [0, 4], [0, 8], [0, 12], [0, 16], [0, 20],
  [4, 8], [4, 12], [4, 16], [4, 20],
];

const ANGLE_TRIPLETS: ReadonlyArray<readonly [number, number, number]> = [
  [1, 2, 3], [2, 3, 4],
  [0, 5, 6], [5, 6, 7], [6, 7, 8],
  [0, 9, 10], [9, 10, 11], [10, 11, 12],
  [0, 13, 14], [13, 14, 15], [14, 15, 16],
  [0, 17, 18], [17, 18, 19], [18, 19, 20],
];

const DISTANCE_PAIRS: ReadonlyArray<readonly [number, number]> = [
  [0, 4], [0, 8], [4, 8],
  [0, 12], [4, 12], [0, 16], [4, 16], [0, 20], [4, 20],
  [8, 12], [8, 16], [12, 16], [8, 20], [12, 20], [16, 20],
];

export interface NormalizedPoint {
  x: number;
  y: number;
}

/**
 * Translate so the wrist is the origin, then scale so the wrist->middle-MCP
 * bone has unit length. This makes the feature vector invariant to the
 * signer's hand position and distance from the camera.
 */
export function normalizeLandmarks(landmarks: HandLandmarks): NormalizedPoint[] {
  if (landmarks.length !== 21) {
    throw new Error(`Expected 21 landmarks, received ${landmarks.length}`);
  }
  const wrist = landmarks[WRIST];
  const ref = landmarks[MIDDLE_MCP];
  const dx = ref.x - wrist.x;
  const dy = ref.y - wrist.y;
  const scale = Math.hypot(dx, dy) || 1e-6; // avoid division by zero

  return landmarks.map((p: Point3D) => ({
    x: (p.x - wrist.x) / scale,
    y: (p.y - wrist.y) / scale,
  }));
}

function angleAtVertex(a: NormalizedPoint, vertex: NormalizedPoint, b: NormalizedPoint): number {
  const v1x = a.x - vertex.x;
  const v1y = a.y - vertex.y;
  const v2x = b.x - vertex.x;
  const v2y = b.y - vertex.y;
  const cross = v1x * v2y - v1y * v2x;
  const dot = v1x * v2x + v1y * v2y;
  return Math.atan2(cross, dot); // signed angle in radians, range [-pi, pi]
}

function distance(a: NormalizedPoint, b: NormalizedPoint): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Compute the full 89-dimensional feature vector from one hand's 21 raw
 * MediaPipe landmarks. Order matches FEATURE_NAMES in types/features.ts.
 */
export function extractFeatures(landmarks: HandLandmarks): FeatureVector {
  const norm = normalizeLandmarks(landmarks);
  const features: number[] = [];

  // 42 normalized raw coordinates
  for (const p of norm) {
    features.push(p.x, p.y);
  }

  // 18 mean-coordinate features
  for (const [a, b] of MEAN_PAIRS) {
    features.push((norm[a].x + norm[b].x) / 2);
    features.push((norm[a].y + norm[b].y) / 2);
  }

  // 14 angular features
  for (const [a, vertex, b] of ANGLE_TRIPLETS) {
    features.push(angleAtVertex(norm[a], norm[vertex], norm[b]));
  }

  // 15 distance features
  for (const [a, b] of DISTANCE_PAIRS) {
    features.push(distance(norm[a], norm[b]));
  }

  if (features.length !== FEATURE_COUNT) {
    throw new Error(
      `Feature extraction produced ${features.length} features, expected ${FEATURE_COUNT}`
    );
  }
  return features;
}
