import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { ArSLClassifier } from "./model";
import type { ModelWeights } from "./model";

const weights = JSON.parse(
  readFileSync(path.resolve(__dirname, "../../public/models/arsl-31/weights.json"), "utf-8")
) as ModelWeights;

const fixture = JSON.parse(
  readFileSync(
    path.resolve(__dirname, "../../ml/tests/fixtures/model_parity_fixture.json"),
    "utf-8"
  )
) as {
  classes: string[];
  samples: { rawFeatures: number[]; trueLabel: string; expectedProbabilities: number[] }[];
};

describe("ArSLClassifier parity with trained sklearn model", () => {
  const classifier = new ArSLClassifier(weights);

  it("has weight dimensions consistent with the metadata", () => {
    expect(weights.classes).toHaveLength(31);
    expect(weights.featureMean).toHaveLength(89);
    expect(weights.featureStd).toHaveLength(89);
  });

  it("reproduces sklearn's probability distribution on real held-out test samples", () => {
    expect(fixture.samples.length).toBeGreaterThan(0);
    for (const sample of fixture.samples) {
      const result = classifier.predict(sample.rawFeatures);
      expect(result.probabilities).toHaveLength(sample.expectedProbabilities.length);
      for (let i = 0; i < result.probabilities.length; i++) {
        expect(result.probabilities[i]).toBeCloseTo(sample.expectedProbabilities[i], 6);
      }
    }
  });

  it("predicts the correct label on real held-out test samples", () => {
    let correct = 0;
    for (const sample of fixture.samples) {
      const result = classifier.predict(sample.rawFeatures);
      if (result.label === sample.trueLabel) correct += 1;
    }
    // Not asserting 100% — this just proves predictions are being computed
    // for real, not hardcoded, by checking against ground truth.
    expect(correct).toBeGreaterThan(0);
  });

  it("probabilities always sum to 1", () => {
    for (const sample of fixture.samples) {
      const result = classifier.predict(sample.rawFeatures);
      const sum = result.probabilities.reduce((a, b) => a + b, 0);
      expect(sum).toBeCloseTo(1, 6);
    }
  });
});
