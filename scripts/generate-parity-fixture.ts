/**
 * Generates a deterministic fixture: 21 fake landmarks + the feature vector
 * our TS pipeline computes for them. ml/tests/test_features_parity.py loads
 * this same fixture and asserts the Python pipeline produces the same
 * numbers (within floating point tolerance). This is how we prove
 * lib/ml/features.ts and ml/preprocessing/features.py stay in sync.
 */
import { writeFileSync } from "fs";
import { extractFeatures } from "../lib/ml/features";
import type { HandLandmarks } from "../types/landmarks";

// Deterministic pseudo-hand: not a real gesture, just fixed numbers so both
// languages compute from identical input.
function fixedLandmarks(): HandLandmarks {
  const landmarks = [];
  for (let i = 0; i < 21; i++) {
    landmarks.push({
      x: 0.3 + 0.02 * i + 0.001 * i * i,
      y: 0.6 - 0.015 * i + 0.0005 * i * i,
      z: 0,
    });
  }
  return landmarks;
}

const landmarks = fixedLandmarks();
const features = extractFeatures(landmarks);

writeFileSync(
  "ml/tests/fixtures/parity_fixture.json",
  JSON.stringify({ landmarks, features }, null, 2)
);

console.log(`Wrote fixture with ${features.length} features.`);
