"""
CANONICAL LEFT/RIGHT ASSIGNMENT for a tracked pair of hands.

WHY THIS MODULE EXISTS
----------------------
Recognition compares a live pair of hands against a stored pair. That only
works if "left" means the same physical hand on both sides of the comparison.
Two things that were previously used as the hand's identity are not identities
at all:

  1. MediaPipe's slot ORDER (`result.hand_landmarks[0]`). It is the model's
     output order and can permute between frames. HandSlotTracker already
     fixes this WITHIN a video by nearest-wrist continuity, but a tracker slot
     is still just "slot 0" -- it carries no left/right meaning.

  2. "Whichever slot was detected in more frames" (the old `primary_slot =
     0 if slot_hits[0] >= slot_hits[1] else 1`). For a symmetric two-handed
     sign the two counts are routinely EQUAL, so this decayed to slot order.

THE SIGNALS, AND WHICH ONE WINS
-------------------------------
Measured over 204 real extracted KArSL takes (73 signs, 4 signer folders,
11,975 detected hand-frames, 4,926 of them with both hands):

  GEOMETRY -- a KArSL signer faces the camera and the recording is not
  mirrored, so the signer's RIGHT hand appears at the SMALLER image x. On
  frames where both hands are present this is decidable without any model
  output. It is highly consistent within a take: per-take agreement with the
  take's own majority was median 1.000, mean 0.972, and 147/170 takes were
  100% consistent. Only 15/170 fell below 0.90 (hands genuinely crossing).

  MEDIAPIPE'S `rawLabel` -- agrees with the geometric answer 95.9% of the
  time (9,450/9,852 hand-frames; 97.6% when both handedness scores >= 0.95):

      geometric RIGHT -> rawLabel "Right" 4673, "Left"  253
      geometric LEFT  -> rawLabel "Left"  4777, "Right" 149

  Note the direction. The label maps to anatomy DIRECTLY on this data --
  "Right" really is the signer's right hand. An earlier version of the
  extractor inverted it (on the documented assumption that MediaPipe's
  handedness presumes a mirrored selfie image) and that inversion is
  measurably wrong here: the inverted names agreed with geometry only
  452/4926 = 9.2% of frames, and in only 19 of 170 takes. Every sign built by
  that extractor therefore has its two hands swapped. `INVERT_LABEL` exists
  for a genuinely mirrored source, but it is off by default because the
  measurement says so.

  The label is also not stable frame to frame: 30.4% of tracks (113/372)
  contain at least one frame whose label disagrees with the rest of the track,
  and in 8.1% of tracks the FIRST frame -- the one the old extractor seeded
  the slot name from -- disagrees with the track majority. A single-frame seed
  is therefore the worst available choice; a majority vote is much better.

So: geometry decides when it is available and consistent; otherwise the
per-track label majority decides; otherwise the side is recorded as UNKNOWN
and the ambiguity is carried forward rather than guessed. A one-handed take
has no both-hand frames at all (34/204 takes here), which is exactly the case
where geometry cannot help and the label must.

NOTHING HERE INVENTS A HAND. It only names the hands that were detected.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

Point = dict
LandmarkFrame = Optional[Sequence[Point]]

#: Fewest both-hand frames before the geometric vote is trusted at all.
GEOMETRY_MIN_FRAMES = 3
#: Fraction of both-hand frames that must agree for geometry to decide.
#: Chosen from the measured per-take distribution: 147/170 takes are 100%
#: consistent and the 10th percentile is 0.913, so 0.90 admits the takes where
#: the answer is clear and defers on the 15 takes where the hands genuinely
#: cross.
GEOMETRY_MIN_AGREEMENT = 0.90
#: Fraction of a track's frames that must share a label for the label vote to
#: decide. Measured frame-level agreement with the track majority is 0.962, so
#: anything below 0.60 is a track the classifier could not make up its mind on.
LABEL_MIN_AGREEMENT = 0.60

#: Set True only for a source whose video really is mirrored (a selfie-style
#: recording). Measured false for KArSL -- see the module docstring.
INVERT_LABEL = False

SIDES = ("right", "left")
UNKNOWN = "unknown"


@dataclass
class SideAssignment:
    """Which physical hand each tracker slot holds."""

    slot_sides: tuple[str, str]  # side name for slot 0, slot 1
    source: str  # "geometry" | "handedness-label" | "unknown"
    confidence: float  # agreement fraction of whichever vote decided
    geometry_frames: int
    geometry_agreement: Optional[float]
    label_agreement: Optional[float]

    def as_dict(self) -> dict:
        return {
            "slotSides": list(self.slot_sides),
            "sideSource": self.source,
            "sideConfidence": round(self.confidence, 4),
            "geometryFrames": self.geometry_frames,
            "geometryAgreement": None if self.geometry_agreement is None else round(self.geometry_agreement, 4),
            "labelAgreement": None if self.label_agreement is None else round(self.label_agreement, 4),
        }


def _wrist_x(frame: LandmarkFrame) -> Optional[float]:
    if not frame:
        return None
    return float(frame[0]["x"])


def geometric_vote(slot_a: Sequence[LandmarkFrame], slot_b: Sequence[LandmarkFrame]) -> tuple[int, int]:
    """(frames where slot A is at the smaller image x, frames compared).

    Only frames where BOTH slots have a detection are counted -- a frame with
    one hand says nothing about which hand it is.
    """
    a_smaller = 0
    compared = 0
    for fa, fb in zip(slot_a, slot_b):
        xa, xb = _wrist_x(fa), _wrist_x(fb)
        if xa is None or xb is None:
            continue
        compared += 1
        if xa < xb:
            a_smaller += 1
    return a_smaller, compared


def label_vote(labels: Sequence[Optional[str]]) -> tuple[Optional[str], float]:
    """(majority raw label for one slot, agreement fraction)."""
    seen = [lb for lb in labels if lb]
    if not seen:
        return None, 0.0
    counts: dict[str, int] = {}
    for lb in seen:
        counts[lb] = counts.get(lb, 0) + 1
    top = max(counts, key=lambda k: counts[k])
    return top, counts[top] / len(seen)


def _anatomical(raw_label: str) -> str:
    side = raw_label.strip().lower()
    if side not in ("left", "right"):
        return UNKNOWN
    if INVERT_LABEL:
        return "left" if side == "right" else "right"
    return side


def assign_sides(
    slot_a_frames: Sequence[LandmarkFrame],
    slot_b_frames: Sequence[LandmarkFrame],
    slot_a_labels: Sequence[Optional[str]],
    slot_b_labels: Sequence[Optional[str]],
) -> SideAssignment:
    """Names slot 0 and slot 1 as the signer's "left"/"right", or "unknown".

    The two slots are always given OPPOSITE names when either can be named:
    they are, by construction of HandSlotTracker, two different physical hands.
    """
    a_smaller, compared = geometric_vote(slot_a_frames, slot_b_frames)
    geom_agreement: Optional[float] = None
    if compared:
        geom_agreement = max(a_smaller, compared - a_smaller) / compared

    if compared >= GEOMETRY_MIN_FRAMES and geom_agreement is not None and geom_agreement >= GEOMETRY_MIN_AGREEMENT:
        # Smaller image x is the signer's right hand.
        a_is_right = a_smaller * 2 >= compared
        return SideAssignment(
            slot_sides=("right", "left") if a_is_right else ("left", "right"),
            source="geometry",
            confidence=geom_agreement,
            geometry_frames=compared,
            geometry_agreement=geom_agreement,
            label_agreement=None,
        )

    label_a, agree_a = label_vote(slot_a_labels)
    label_b, agree_b = label_vote(slot_b_labels)
    best_agreement = max(agree_a, agree_b)

    if best_agreement >= LABEL_MIN_AGREEMENT and (label_a or label_b):
        # Trust whichever slot the classifier was more consistent about, and
        # give the other slot the opposite side. This is what keeps the two
        # slots from ever being named the same hand.
        if agree_a >= agree_b and label_a:
            side_a = _anatomical(label_a)
        elif label_b:
            side_a = _opposite(_anatomical(label_b))
        else:
            side_a = UNKNOWN
        if side_a != UNKNOWN:
            return SideAssignment(
                slot_sides=(side_a, _opposite(side_a)),
                source="handedness-label",
                confidence=best_agreement,
                geometry_frames=compared,
                geometry_agreement=geom_agreement,
                label_agreement=best_agreement,
            )

    return SideAssignment(
        slot_sides=(UNKNOWN, UNKNOWN),
        source="unknown",
        confidence=0.0,
        geometry_frames=compared,
        geometry_agreement=geom_agreement,
        label_agreement=best_agreement or None,
    )


def _opposite(side: str) -> str:
    if side == "left":
        return "right"
    if side == "right":
        return "left"
    return UNKNOWN
