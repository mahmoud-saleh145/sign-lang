/**
 * Types describing the engineered feature vector consumed by the classifier.
 *
 * Schema matches the ArSL-31-Pilot dataset (Zenodo, CC BY 4.0, Ait Madi &
 * Farhan, Ibn Tofail University) exactly:
 *   - 42 raw coordinates:  x0,y0 ... x20,y20  (21 right-hand landmarks)
 *   - 47 engineered features: 18 mean-coordinate + 14 angular + 15 distance
 * Total: 89 features. This exact order/definition MUST be reproduced
 * identically in ml/preprocessing (Python, training) and lib/ml (TypeScript,
 * browser inference), or the trained model's weights will be meaningless
 * when fed browser-extracted features.
 */

/** Fixed feature order. Index in this array === column index fed to the model. */
export const FEATURE_NAMES: readonly string[] = [
  // 42 raw coordinates
  ...Array.from({ length: 21 }, (_, i) => [`x${i}`, `y${i}`]).flat(),
  // 18 mean-coordinate features
  "mean(x0_x4)", "mean(y0_y4)",
  "mean(x0_x8)", "mean(y0_y8)",
  "mean(x0_x12)", "mean(y0_y12)",
  "mean(x0_x16)", "mean(y0_y16)",
  "mean(x0_x20)", "mean(y0_y20)",
  "mean(x4_x8)", "mean(y4_y8)",
  "mean(x4_x12)", "mean(y4_y12)",
  "mean(x4_x16)", "mean(y4_y16)",
  "mean(x4_x20)", "mean(y4_y20)",
  // 14 angular features (named by the vertex/middle landmark of the triplet)
  "Angle_2", "Angle_3", "Angle_5", "Angle_6", "Angle_7",
  "Angle_9", "Angle_10", "Angle_11", "Angle_13", "Angle_14",
  "Angle_15", "Angle_17", "Angle_18", "Angle_19",
  // 15 distance features
  "Dis(x0_x4)", "Dis(x0_x8)", "Dis(x4_x8)",
  "Dis(x0_x12)", "Dis(x4_x12)", "Dis(x0_x16)", "Dis(x4_x16)",
  "Dis(x0_x20)", "Dis(x4_x20)", "Dis(x8_x12)", "Dis(x8_x16)",
  "Dis(x12_x16)", "Dis(x8_x20)", "Dis(x12_x20)", "Dis(x16_x20)",
] as const;

export const FEATURE_COUNT = 89;

/** A single 89-dimensional feature vector, in FEATURE_NAMES order. */
export type FeatureVector = number[];

/** Model metadata loaded at runtime (never hardcoded in UI). Section 29. */
export interface ModelMetadata {
  modelVersion: string;
  language: "ASL" | "ArSL";
  languageDisplayName: string;
  dataset: {
    name: string;
    source: string;
    license: string;
    sampleCount: number;
    classCount: number;
  };
  classes: string[];
  classDisplayLabels: Record<string, string>;
  featureFormat: {
    count: number;
    order: readonly string[];
  };
  preprocessingVersion: string;
  confidenceThreshold: number;
  temporalWindow: {
    minStableFrames: number;
    maxGapFrames: number;
  };
  modelArchitecture: string;
  evaluation: {
    testAccuracy: number | null;
    macroF1: number | null;
    signerIndependent: boolean;
    notes: string;
  };
  trainingDate: string | null;
}

export interface TranscriptEntry {
  id: string;
  sign: string;
  displayLabel: string;
  confidence: number;
  committedAtMs: number;
}

/**
 * Sign-commit state machine (spec section 17).
 *
 * IDLE            -> no hand / no motion detected
 * POSSIBLE_SIGN   -> a candidate class is being predicted, not yet stable
 * SIGN_ACTIVE     -> the same class has been predicted for minStableFrames
 * POSSIBLE_END    -> confidence/motion dropped, waiting to confirm sign end
 * COMMITTED       -> transcript updated this tick; transitions back to IDLE
 */
export type SegmentationState =
  | "IDLE"
  | "POSSIBLE_SIGN"
  | "SIGN_ACTIVE"
  | "POSSIBLE_END"
  | "COMMITTED";
