"""
Landmark normalization + feature engineering for training.

MUST stay mathematically identical to lib/ml/features.ts (browser inference).
If you change a formula here, change it there too, bump PREPROCESSING_VERSION
in both files, and retrain. See lib/ml/features.ts for the full rationale.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Sequence, Tuple

PREPROCESSING_VERSION = "1.0.0"

WRIST = 0
MIDDLE_MCP = 9

MEAN_PAIRS: List[Tuple[int, int]] = [
    (0, 4), (0, 8), (0, 12), (0, 16), (0, 20),
    (4, 8), (4, 12), (4, 16), (4, 20),
]

ANGLE_TRIPLETS: List[Tuple[int, int, int]] = [
    (1, 2, 3), (2, 3, 4),
    (0, 5, 6), (5, 6, 7), (6, 7, 8),
    (0, 9, 10), (9, 10, 11), (10, 11, 12),
    (0, 13, 14), (13, 14, 15), (14, 15, 16),
    (0, 17, 18), (17, 18, 19), (18, 19, 20),
]

DISTANCE_PAIRS: List[Tuple[int, int]] = [
    (0, 4), (0, 8), (4, 8),
    (0, 12), (4, 12), (0, 16), (4, 16), (0, 20), (4, 20),
    (8, 12), (8, 16), (12, 16), (8, 20), (12, 20), (16, 20),
]

FEATURE_COUNT = 89


@dataclass(frozen=True)
class Point2D:
    x: float
    y: float


def normalize_landmarks(landmarks: Sequence[Point2D]) -> List[Point2D]:
    """Translate so wrist=origin, scale so wrist->middle-MCP bone has unit length."""
    if len(landmarks) != 21:
        raise ValueError(f"Expected 21 landmarks, got {len(landmarks)}")
    wrist = landmarks[WRIST]
    ref = landmarks[MIDDLE_MCP]
    dx = ref.x - wrist.x
    dy = ref.y - wrist.y
    scale = math.hypot(dx, dy) or 1e-6
    return [Point2D((p.x - wrist.x) / scale, (p.y - wrist.y) / scale) for p in landmarks]


def _angle_at_vertex(a: Point2D, vertex: Point2D, b: Point2D) -> float:
    v1x, v1y = a.x - vertex.x, a.y - vertex.y
    v2x, v2y = b.x - vertex.x, b.y - vertex.y
    cross = v1x * v2y - v1y * v2x
    dot = v1x * v2x + v1y * v2y
    return math.atan2(cross, dot)


def _distance(a: Point2D, b: Point2D) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def extract_features(landmarks: Sequence[Point2D]) -> List[float]:
    """Compute the 89-dim feature vector from 21 raw (x, y) landmarks.

    Order: 42 normalized raw coords, 18 mean-coordinate, 14 angular,
    15 distance. Must match FEATURE_NAMES in types/features.ts exactly.
    """
    norm = normalize_landmarks(landmarks)
    features: List[float] = []

    for p in norm:
        features.append(p.x)
        features.append(p.y)

    for a, b in MEAN_PAIRS:
        features.append((norm[a].x + norm[b].x) / 2)
        features.append((norm[a].y + norm[b].y) / 2)

    for a, vertex, b in ANGLE_TRIPLETS:
        features.append(_angle_at_vertex(norm[a], norm[vertex], norm[b]))

    for a, b in DISTANCE_PAIRS:
        features.append(_distance(norm[a], norm[b]))

    if len(features) != FEATURE_COUNT:
        raise ValueError(f"Produced {len(features)} features, expected {FEATURE_COUNT}")
    return features
