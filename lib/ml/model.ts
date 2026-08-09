import type { FeatureVector } from "@/types/features";

/**
 * Raw weights exported by ml/training/train.py. Loaded once from
 * /models/arsl-31/weights.json and reused for every prediction.
 *
 * Architecture must match ml/training/train.py's MLPClassifier exactly:
 * standardize -> Dense(89,64) + ReLU -> Dense(64,31) + softmax.
 */
export interface ModelWeights {
  W1: number[][]; // 89 x 64
  b1: number[]; // 64
  W2: number[][]; // 64 x 31
  b2: number[]; // 31
  featureMean: number[]; // 89
  featureStd: number[]; // 89
  classes: string[]; // 31, in the order used by the model's output layer
}

export interface Prediction {
  label: string;
  confidence: number;
  /** Full probability distribution, same order as ModelWeights.classes. */
  probabilities: number[];
}

function relu(x: number): number {
  return x > 0 ? x : 0;
}

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function matVecAdd(W: number[][], x: number[], b: number[]): number[] {
  // W is (inputDim x outputDim); x is (inputDim); result is (outputDim)
  const outDim = b.length;
  const inDim = x.length;
  const out = new Array<number>(outDim).fill(0);
  for (let j = 0; j < outDim; j++) {
    let sum = b[j];
    for (let i = 0; i < inDim; i++) {
      sum += x[i] * W[i][j];
    }
    out[j] = sum;
  }
  return out;
}

export class ArSLClassifier {
  constructor(private readonly weights: ModelWeights) {}

  static async loadFromUrl(url: string): Promise<ArSLClassifier> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`Failed to load model weights from ${url}: ${res.status}`);
    }
    const weights = (await res.json()) as ModelWeights;
    return new ArSLClassifier(weights);
  }

  predict(features: FeatureVector): Prediction {
    const { W1, b1, W2, b2, featureMean, featureStd, classes } = this.weights;

    if (features.length !== featureMean.length) {
      throw new Error(
        `Expected ${featureMean.length} features, received ${features.length}`
      );
    }

    const standardized = features.map((v, i) => (v - featureMean[i]) / featureStd[i]);

    const hidden = matVecAdd(W1, standardized, b1).map(relu);
    const logits = matVecAdd(W2, hidden, b2);
    const probabilities = softmax(logits);

    let bestIdx = 0;
    for (let i = 1; i < probabilities.length; i++) {
      if (probabilities[i] > probabilities[bestIdx]) bestIdx = i;
    }

    return {
      label: classes[bestIdx],
      confidence: probabilities[bestIdx],
      probabilities,
    };
  }
}
