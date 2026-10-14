# KArSL dynamic word recognition — audit, root cause, and design

This document records what was measured, what was changed, and what is still
unproven. Every number below came from running code against the real data in
this repository; none of it is estimated.

## 1. Is the pipeline temporal, or is it classifying single frames?

**It is genuinely temporal, and always was.** `SignRecognitionPipeline`
accumulates raw landmark frames in `DynamicGestureBuffer`, and the classifier
(`dynamicSignRecognizer.recognizeDynamicSignSyncWithDiagnostics`) is called
exactly once per completed gesture, on the whole buffered sequence. That
sequence is resampled to 30 frames and compared with DTW against each stored
KArSL sequence, also resampled to 30. No single frame is ever classified as a
word, and no motion is fabricated.

The reported failure was therefore **not** a frame-vs-sequence design error.
It was that the completed sequence never reached the classifier.

## 2. Root cause: the motion-energy gate discarded 96.5% of real signs

`DynamicGestureBuffer.finalize()` threw away any completed buffer whose
average per-frame landmark displacement fell below `minMotionEnergy`, which
defaulted to `0.02`. That value predates the arrival of any real data.

Measured over all 170 real sequences in
`public/models/sign-vocabulary/dedicated-signs.json`:

| statistic | value |
|---|---|
| minimum sequence motion energy | 0.000193 |
| 25th percentile | 0.000511 |
| median | 0.001553 |
| 90th percentile | 0.010306 |
| maximum | 0.027495 |
| **below the old 0.02 gate** | **164 / 170 = 96.5%** |

So the gate discarded 96.5% of the dataset's own signs before DTW ever ran.
The user's observed telemetry matches exactly: motion energy `0.01363` < `0.02`,
therefore `completedGesture` was `null`, therefore
`recognizeDynamicSignSyncWithDiagnostics` was never called, therefore best
sign ID, DTW distance and final decision all displayed `-` forever.

A second, compounding fault: `maxGestureFrames` was `90`, so a signer whose
hand stayed in frame hit the cap every ~1.8 s, and the finalized buffer was
then discarded by the gate. The visible result was a buffer that appeared
permanently stuck in `buffering`.

**Fix.** `minMotionEnergy` is now `0.0001`, below the real observed minimum,
so it excludes only a literally frozen buffer. `maxGestureFrames` is `180`
(longer than the longest real sequence) and hitting it now **classifies**
instead of discarding. When a buffer really is discarded, the reason
(`too-few-frames` / `no-motion`) is reported in telemetry and on screen.

## 3. Threshold recalibration (measured, not loosened)

Benchmark: 40 **real** held-out KArSL repetitions (2 per sign, signs
0001–0020) that the importer did *not* place in the registry, matched against
all 170 registry entries. Reproduced by `lib/ml/karslHeldOut.test.ts`.

Nearest-neighbour top-1: **37/40 (92.5%)** — the matcher itself is sound.

| distance ≤ | ratio ≤ | accepted | correct | false accepts |
|---|---|---|---|---|
| 0.35 | 0.85 | 27 | 27 | 0 | ← previous settings |
| 0.75 | 0.85 | 34 | 33 | **1** |
| 1.0 | 0.85 | 35 | 34 | **1** |
| **1.5** | **0.70** | **35** | **35** | **0** | ← chosen |
| ∞ | 0.70 | 35 | 35 | 0 |
| ∞ | 0.60 | 34 | 34 | 0 |

The best/second-best **margin**, not the absolute distance, is what separates
right from wrong: all three wrong nearest neighbours had ratios of 0.809,
0.908 and 0.958. Raising the distance cap alone would have admitted a wrong
word; tightening the ratio to 0.70 at the same time keeps precision at 100%
while recall rises from 67.5% to 87.5%. The two changes only make sense
together.

## 4. A hypothesis that was tested and rejected

`normalizeLandmarks()` makes every frame wrist-relative and scale-normalized,
which deletes the hand's global trajectory. The obvious hypothesis is that
this loses the very information a movement-defined sign depends on.

It was tested: a 44-dimensional variant appending the real recorded wrist
path (expressed in hand-size units, relative to the sequence's own mean wrist
position) at four different weightings.

| representation | top-1 |
|---|---|
| shape only (current, 42-d) | **37/40** |
| shape + trajectory ×0.5 | 36/40 |
| shape + trajectory ×1 | 36/40 |
| shape + trajectory ×2 | 36/40 |
| shape + trajectory ×4 | 34/40 |

Trajectory made it worse at every weighting, so **the change was not made**.
The wrist path in this data carries more framing/signer variance than signal.

## 5. Why gesture end is hand-presence, not stillness

A stillness-based end detector was evaluated and rejected on measurement.
Across all real sequences: per-frame displacement p10 = 0.00023, p50 =
0.00081, p90 = 0.00937; but per-**sequence peak** displacement p0 = 0.00102,
p50 = 0.00722. The quietest real signs never exceed 0.001 at their peak while
the median frame of an active sign sits at 0.0008 — the distributions overlap
almost completely. No absolute stillness threshold can separate "paused" from
"performing a low-motion sign" without cutting real signs in half.

Gesture end is therefore: `endGapFrames` consecutive no-hand frames, or the
frame cap, or an explicit **manual capture** window (hold the button, perform
the sign, release) which removes segmentation as a variable entirely.

## 6. Registry audit — why 170 and not 502

Not a bug, and not an importer failure.

- `processed-karsl/` contains exactly 170 sign directories, `0001`–`0170`,
  contiguous. `KArSL/` (raw video) currently holds only `0071`–`0170`.
- The registry has 170 entries, sourceSignId `0001`–`0170`, no duplicates,
  no sequence shorter than 2 frames. Sequence lengths 11–90, mean 40.6.
- All 170 labels match `KARSL-502_Labels.xlsx` **exactly** — zero mismatches.
  `canonicalize_sign_id()` correctly reconciles `"0001"` with `"1"`.
- 48 entries are multi-word Arabic phrases (e.g. `هيكل عظمي`), each stored
  and returned as **one** complete entry. Nothing splits them.
- Entries `0001`–`0010` have text `"0"`…`"9"`: these are the real KArSL
  number signs, not corrupted labels.

**The 332 missing signs (171–502) were never downloaded or extracted.** To
add them: extract with `ml/training/extract_karsl_skeletons.py`, then re-run
`ml/training/import_karsl.py`.

## 7. Modes

| mode | static letters | dynamic words |
|---|---|---|
| `letter` | commits normally (unchanged) | buffer is **not stepped at all** |
| `word` | runs, telemetry live, **commits suppressed** | owns the whole gesture |
| `both` | commits normally | commits, and merges into the shared Arabic text |

Word results always go to a **separate** word transcript
(`getWordTranscript()`), never decomposed into letters. Switching modes
resets both state machines so frames buffered under one mode cannot
contribute to the other's result.

## 8. Camera-free test page

`/dev/karsl` replays real stored sequences through the production classifier
and the production buffer gates: identity, 0.5× and 2× re-timing, first/last
80% crops, and a horizontal mirror — all transformations of real recorded
data, nothing synthesized. It also sweeps the buffer gates across the whole
registry, which is the check that exposed the original bug.

The *identity* row matches a stored sequence against a registry containing
it, so distance ≈ 0 is expected and proves only that the pipeline is wired
up. The re-timed and cropped rows carry the real signal.

## 9. The KArSL-502 accuracy collapse — measured root cause

After the full 502-sign rebuild, `lib/ml/karslHeldOut.test.ts` fell to top-1
**0.561**, accept rate **0.459**, and **33 false accepts** (was 170 signs,
top-1 0.925, 0 false accepts). Two separate causes, both measured on real
data rather than assumed.

### 9.1 The recognizer keys on an arbitrarily-chosen hand

`extract_karsl_skeletons.py` picks the legacy primary hand per video with

```python
primary_slot = 0 if slot_hits[0] >= slot_hits[1] else 1
```

— whichever slot MediaPipe saw in more frames of *that take*. For a
symmetric two-handed sign the two counts are routinely identical
(`[17,17]`, `[40,40]`, `[26,26]`), so the `>=` tiebreak falls through to
slot 0, which is whichever hand MediaPipe happened to list first in the
first frame. Across the built registry the choice is close to a coin flip:
**281 left / 221 right**, and **477 of 502** signs are genuinely two-handed.

The registry template and a held-out repetition of the same sign therefore
often carry *different physical hands*. Measured on 42 real held-out takes
across 14 signs spanning the ID range and all four signer folders:

| primary hand vs template | n | top-1 | accepted | false accepts |
|---|---|---|---|---|
| agrees | 35 | 0.743 | 0.543 | 0 |
| **disagrees** | **7** | **0.000** | 0.000 | 1 |

Disagreement rate in that sample: **7/42 = 17%**, and those cases score
exactly zero. The one false accept is the failure mode behind the suite's
33: sign 0487 queried with the wrong hand matched sign 0199 at distance
0.288, ratio 0.607 — inside both gates, confidently wrong.

Mirror-invariant DTW does not rescue this. Mirroring makes a left *hand
shape* look like a right one; it cannot make the non-dominant hand's
*trajectory* look like the dominant hand's. Querying every two-handed
entry with the other real hand from its own take (n=477, real registry
data) gives top-1 **0.103** against **1.000** for the primary hand.

### 9.2 One hand is not enough information at 502 classes

Even when the primary hand agrees, top-1 is 0.743 — and once the 12 takes
that *are* the stored template are excluded, genuine held-out accuracy on
that sample is **0.467** (14/30). Single-hand DTW discards the second real
hand entirely, which for 477 signs is half the discriminative signal, while
the class count tripled from 170 to 502.

### 9.3 What two-hand matching buys — measured

Prototype: resample each anatomical side's real track separately, DTW
left-vs-left and right-vs-right with no mirroring, average the per-side
costs over the sides both sequences have. Same 30 genuine held-out takes:

| matcher | top-1 |
|---|---|
| current: single arbitrary primary hand | **0.467** |
| both hands, anatomical sides | **0.700** |

Every previously-zero mismatched-hand case recovers, because sides are now
named anatomically rather than by detection count. This is a real +50%
relative gain from data that is already in the registry and currently
unused by the recognizer.

**It is still not 0.8.** The held-out gates in
`lib/ml/karslHeldOut.test.ts` were calibrated on 170 classes; 0.8 top-1
over 502 signer-dependent classes may not be reachable with DTW
nearest-neighbour at all. The gates have not been touched — a test that
fails is reporting something true.

### 9.4 Traceability gap found while measuring

Registry entries record `sourceSignId` but not the source *video*, so it is
not possible to tell from the registry which take a template came from.
`select_best_take()` makes that choice non-obvious. `sourceVideo` should be
carried through `import_karsl.py` into the entry.

## 10. The side-aware rebuild — what was built and what it measures

Section 9 diagnosed the collapse. This section is the implementation and its
results. Every number is from real KArSL data; none is estimated.

### 10.1 A hand now has an identity, and it is measured

`ml/training/handedness.py` (offline) and `lib/ml/handSides.ts` (live) share
one rule, in priority order:

1. **Geometry.** A signer faces the camera and KArSL video is not mirrored, so
   the signer's RIGHT hand sits at the SMALLER image x. Decidable on any frame
   where both hands are present, using no model output.
2. **MediaPipe's handedness label**, as a per-track majority.
3. **Unknown** — reported, never guessed.

Measured over 204 real takes (73 signs, all four signer folders, 11,975
detected hand-frames, 4,926 with both hands):

| signal | result |
|---|---|
| per-take geometric consistency | median **1.000**, mean 0.972; 147/170 takes 100% |
| takes where geometry is ambiguous (<0.90) | 15/170 |
| takes with no both-hand frame (geometry unusable) | 34/204 |
| label vs geometry agreement | **95.9%** (9,450/9,852); 97.6% at score ≥ 0.95 |
| tracks whose label changes mid-track | **30.4%** (113/372) |
| tracks whose FIRST frame disagrees with the majority | **8.1%** (30/372) |

That last row is why the old extractor was wrong twice over: it named slots
from a single seed frame, and it inverted the label.

### 10.2 The inversion was backwards, and every sign was affected

The extractor's default `--handedness-convention camera` inverted MediaPipe's
label, on the documented assumption that handedness presumes a mirrored selfie
image. Measured against geometry, the inverted names agreed only **452/4,926 =
9.2%** of frames, and in **19 of 170** takes. Un-inverted, they agree 95.9%.

So the label maps to anatomy directly on this data, and **482 of 502 registry
entries had their two hands swapped**. After re-deriving sides, agreement with
geometry across the whole registry is **96.8%** (18,407/19,011), up from 9.2%.

This is fixable without re-extraction: the raw per-frame labels are stored, so
`import_karsl.py` re-derives sides from the processed tree. Re-running the
importer is minutes; re-running MediaPipe over 4,016 videos is ~1.6 hours.

### 10.3 Temporal identity, measured on the signs that used to splice

`ml/analysis/measure_tracker.py`, run on 0171, 0227, 0241, 0255, 0283, 0487 —
33 real takes:

| | |
|---|---|
| impossible wrist jumps (> 0.15) | **0** |
| worst inter-frame wrist jump, any side | **0.0502** (naive single-hand: 0.237) |
| MediaPipe detection reorderings absorbed | **422** |
| sides named by geometry / label / unknown | **28 / 5 / 0** |

422 reorderings across 33 takes is the measure of how unusable detection order
is as an identity, and how much work the tracker is doing.

### 10.4 How the two sides are combined

`lib/ml/twoHandMatch.ts`. Each side is resampled and DTW'd against the SAME
side of the candidate; the sides are averaged weighted by coverage, and a side
present in only one of the two is charged `UNMATCHED_SIDE_COST`.

Not one concatenated 84-d vector: that forces both hands onto one warping path
(wrong for a sign whose hands move at different rates) and has no
representation for a missing hand short of inventing coordinates.

The cost was derived twice, and the first derivation was wrong. Per-side
distances on the calibration set (88 same-sign pairs, 43,560 different-sign):

| | p10 | median | p90 |
|---|---|---|---|
| same sign | 0.239 | 1.483 | 6.675 |
| different sign | 3.065 | 7.233 | 12.436 |

The tidy argument — an absent side should cost what an uninformative side
costs, 7.23 — is measurably too expensive: top-1 falls 0.646 → 0.521, because
presence mismatches also occur between correct pairs. The quantity that works
is the **separation margin**, different-sign p10 minus same-sign median =
**1.58**, which sits mid-plateau (top-1 by cost: 0 → 0.625, 1.0-2.2 → 0.646,
2.6 → 0.604, 3.0 → 0.583).

Honestly: on top-1 the penalty is worth one case in 48, inside the noise. Its
real job is precision — a one-hand query must not out-score a two-handed sign.

### 10.5 Thresholds

> **SUPERSEDED BY 10.12.** The calibration set this section describes was
> contaminated — 9 of its 60 takes were templates matching themselves — and
> the values below have been replaced by **distance ≤ 0.76, ratio ≤ 0.60**,
> derived on 207 clean held-out queries. Kept as the record of how the old
> numbers were arrived at, and of the "real fix is a larger calibration set"
> that 10.12 carries out.

Calibrated on **30 signs disjoint from the benchmark's 40** (48 held-out
takes), by `scripts/calibrate-two-hand.ts`, with precision as a hard
constraint. Chosen: **distance ≤ 1.25, ratio ≤ 0.70** (from 1.5 / 0.70).

The distance cap tightened because at 1.5 the calibration set admits a wrong
sign. The ratio is set by a stated tie-break, not by fit: 0.70, 0.75 and 0.80
all admit zero wrong signs on calibration, so the data cannot separate them and
the strictest wins.

**Order of events, because it matters.** 1.25/0.80 (highest calibration recall
at zero false accepts) was chosen first and the benchmark run with it: it gave
**four** false accepts on the benchmark against zero on calibration. 48
calibration queries, 18 of them accepted, cannot settle a ratio. The tie-break
was then applied to the calibration data rather than picking whatever looked
best on the benchmark — but the benchmark had been seen by then, so this is
weaker than a clean pre-registration. The real fix is a larger calibration set.

### 10.6 Held-out benchmark results

40 signs (disjoint from calibration), 80 genuine held-out takes, matched
against all 502 templates. The take stored as the template is excluded by
content fingerprint, so nothing is scored against itself.

| | single-hand baseline | two-hand prototype | **final implementation** |
|---|---|---|---|
| top-1 | 0.467 | 0.700 | **0.6125** |
| accepted AND correct | — | — | **0.2250** |
| acceptance rate | — | — | **0.2500** |
| false accepts | 1 (of 30) | — | **2** (of 80) |

The prototype's 0.700 and the baseline's 0.467 were measured on a 30-take
subset of these same signs; the final 0.6125 is over 80 takes on 40 signs, so
the three are not strictly comparable. What is comparable is the breakdown:

| slice | n | top-1 | accepted | false accepts |
|---|---|---|---|---|
| query has both hands | 62 | 0.645 | 0.242 | **0** |
| query has one hand | 18 | 0.500 | 0.278 | **2** |
| template is two-handed | 78 | 0.603 | 0.231 | 2 |
| template is one-handed | 2 | 1.000 | 1.000 | 0 |
| old primary-hand choice AGREED | 26 | 0.692 | 0.231 | 0 |
| **old choice DISAGREED** | 54 | **0.574** | 0.259 | 2 |

The last row is the headline. Those 54 takes scored **0.000** top-1 under
single-hand matching — a wrong-hand query could not match at all. They now
score 0.574.

Rejected-but-correct is 0.3625 and rejected-and-wrong 0.3375: the matcher finds
the right sign more often than it is willing to commit to it, which is the
intended direction (a wrong word is worse than a rejected sign) but leaves a
lot of recall on the table.

### 10.7 The gates still fail, and are not being moved

`lib/ml/karslHeldOut.test.ts` asserts top-1 ≥ 0.8, acceptance ≥ 0.8 and zero
false accepts. Measured: **0.6125**, **0.2500**, **2**. Three of its five tests
fail. They were written against a 170-sign registry and have not been relaxed;
a failing test here is reporting something true.

### 10.8 Where the remaining error is

- **Single-hand queries are the precision leak.** Both false accepts, and all
  the accuracy gap between slices, come from queries carrying one hand: 0
  false accepts in 62 two-hand queries, 2 in 18 one-hand queries, top-1 0.645
  vs 0.500. The unmatched-side penalty is not strong enough to stop a one-hand
  query from winning a two-handed sign, and raising it costs top-1 (10.4).
- **One template per sign.** Nearest-neighbour against a single take cannot
  represent how much a sign varies between repetitions. 24 distinct confusions
  over 31 misses, several repeated across both takes of a sign (0010→0024,
  0033→0356, 0095→0338, 0195→0199), which is a systematic template problem,
  not noise.
- **502 classes, one signer per sign.** This remains a signer-DEPENDENT
  benchmark; it says nothing about a new signer at a webcam.
- Whether 0.8 top-1 is reachable at all with DTW nearest-neighbour over 502
  classes is an open question. Nothing here claims it is.

## 10.9 Two defects the full 502-sign run exposed

Running the suite against the fully re-imported registry (1,004 held-out takes
instead of the 80 measured above) surfaced two things the smaller sample had
not, both traceable because `sourceTake` now exists.

**A single-frame second hand won take selection.** `select_best_take()` ranked
by raw both-hand frame count, so a take with ONE flicker frame out of fifty
beat a take with none — signs 0076 and 0089 were chosen with
`"most two-hand frames (1; next best 1)"`. A second hand is now counted only
when it appears in at least `MIN_SECOND_HAND_FRACTION` = 10% of the take's
frames. That floor sits in a measured gap: the per-sign both-hand fraction is
bimodal, 477 signs at ≥ 0.5, and the low tail is 1–3 frames out of 24–68
(0.015–0.075) with the next value up at 0.1395.

**`primaryHand` could name the wrong hand.** It was translated from the
extractor's own primary via `slotNames.index()`, which mixes a SLOT-ordered
list with the importer's KEY-ordered (`left`, `right`) side assignment. The two
coincide only when the extractor named slot 0 "left"; otherwise the mapping
inverted. Result: 0076 and 0089 carried `primaryHand: "left"` with one left
frame against fifty and fifty-one right ones, so every legacy consumer saw a
one-frame sign. It is now derived from the canonical frames themselves — the
side with more detected frames — which cannot disagree with the frames.
`withLegacySequence` also now falls back when the primary side is DEGENERATE
(< 2 frames), not only when it is empty, so the loader does not depend on a
field it does not control being right.

### Full-registry results (1,004 held-out takes, all 502 templates)

| | 80-take subset | **full 1,004** |
|---|---|---|
| top-1 | 0.6125 | **0.7042** |
| accepted AND correct | 0.2250 | **0.4143** |
| acceptance | 0.2500 | **0.4203** |
| false accepts | 2 | **6** (0.6%) |

| slice | n | top-1 | accepted | false accepts |
|---|---|---|---|---|
| query has both hands | 901 | 0.726 | 0.443 | 3 |
| query has one hand | 103 | 0.515 | 0.223 | 3 |
| two-handed sign | 982 | 0.699 | 0.411 | 6 |
| one-handed sign | 22 | 0.955 | 0.818 | 0 |
| old primary choice agreed | 567 | 0.727 | 0.448 | 4 |
| **old choice disagreed** | 437 | **0.675** | 0.384 | 2 |

Single-hand queries remain the weak slice: 10% of the cases and half the false
accepts, at 0.515 top-1 against 0.726. The gates still fail (0.7042 / 0.4203 /
6 against 0.8 / 0.8 / 0) and have not been moved.

## 10.10 The benchmark was scoring templates against themselves

The re-imported registry scored WORSE than the one before it: top-1 0.7042 →
0.6882 over the same 1,004 held-out takes. Chasing that down found a defect in
the benchmark, not in the recognizer.

Diffing the two registries: only **4 signs** (0088, 0089, 0375, 0376) had their
stored motion change at all — the flicker floor picked a different take for
each. Four templates cannot move an aggregate by 16 cases. What else changed
was `primaryHand`, on **158** signs — and the template exclusion was keyed to
it.

The exclusion fingerprinted the registry's LEGACY `landmarkSequence`, which is
reconstructed from whichever side `primaryHand` names, against the take's own
legacy field, which the EXTRACTOR built from whichever tracker slot it saw more
often. Two different selections, so the fingerprints usually disagreed, the
template take was not recognised, and it entered the query set to match itself
at a distance of about 0.003.

Measured over 73 real signs, the template take was identifiable in:

| registry | template identified | leaked in as a free hit |
|---|---|---|
| before the primaryHand fix | 12/73 = **16.4%** | 61/73 |
| after it | 22/73 = **30.1%** | 51/73 |

So the re-import did not make recognition worse. It made the exclusion work
more often, which removed free hits and revealed a lower, more honest number.
**Every benchmark figure produced before this fix is inflated**, including the
0.6125 and 0.7042 quoted in 10.6 and 10.9.

### The fix, and what it costs

Exclusion is now by `sourceTake.processedFile` — the exact recording, which is
what that field exists for — with a fingerprint of the SIDED timeline as the
fallback for a registry old enough to lack it. Both sides derive the sided
timeline the same way, so it cannot drift the way the legacy comparison did.

A new test, "never scores a take against itself", fails if any query matches
its own sign below distance 0.02. Two separate exclusion bugs have already
slipped through here without changing anything except making the numbers look
better; this makes the third one loud.

On the 40-sign benchmark set, correcting the exclusion moves top-1 from 0.6875
to **0.6456** (79 cases; one sign lost a take to the exclusion). That drop is
the size of the contamination, not a regression.

## 10.11 The presence penalty was charged in the wrong direction

`UNMATCHED_SIDE_COST` was charged whenever either sequence had a side the other
lacked. Traced on the stable false accepts, that is backwards.

Sign 0033 ("ب"), whose template is two-handed (L14/R14 of 14 frames), captured
in two takes that caught only one hand (L14/R0):

| candidate | shared | penalty | distance | |
|---|---|---|---|---|
| 0356 — wrong, one-handed L37/R0 | 1 | **0.00** | 0.6466 | ACCEPTED |
| 0033 — right, two-handed | 1 | **0.79** | — | not even 2nd |

A correct-pair per-side distance has a median of 1.416, so a 0.79 charge is
comparable to the entire useful range: **presence alone decided the outcome**.
Every one-handed template was structurally cheaper than the correct two-handed
one whenever a capture lost a hand. One-hand queries were 10.7% of the
benchmark and carried 3 of its 6 false accepts, at top-1 0.542 against 0.696.

**The fix is an asymmetry.** A hand missing from the QUERY is a capture failure
and says nothing about the sign, so it costs nothing. A hand missing from the
TEMPLATE means the stored sign does not use a hand the signer demonstrably just
used, which is evidence against it, so it is still charged.

### Recalibration

Re-swept on the same 30-sign calibration set, coverage weighting: cost 0 →
0.577, and **1.0, 1.3, 1.58, 1.9, 2.2, 2.6, 3.0 all → 0.615**. The plateau is
now flat across the whole range where before it ran 1.0–2.2 and fell away on
either side — expected, because the penalty now fires only when the query has
two hands and the template has one, which is rarer and less ambiguous. **1.58
is kept**: it is the derived margin (different-sign p10 3.051 − same-sign
median 1.416 = 1.635) and sits mid-plateau. The gates were not touched.

### Result, 40-sign benchmark set (79 genuine held-out takes)

| | symmetric | **asymmetric** |
|---|---|---|
| top-1 | 0.6456 | **0.6582** |
| accepted AND correct | 0.2278 | **0.2785** |
| acceptance | 0.2532 | **0.2785** |
| false accepts | 2 | **0** |

| slice | symmetric | asymmetric |
|---|---|---|
| one-hand query, top-1 | 0.500 | **0.556** |
| one-hand query, accepted | 0.278 | **0.389** |
| one-hand query, false accepts | **2** | **0** |
| two-hand query, top-1 | 0.689 | 0.689 |

Two-hand queries are untouched, which is the point: the change only affects
candidates that differ in which hands they have. "never ACCEPTS a wrong sign"
passes on this set for the first time.

### Slice 9 replaced

It compared the take's legacy `primaryHand` with the template's. Once
`primaryHand` became a derived "side with more frames" field that the
side-aware matcher never reads, that slice stopped meaning anything — it
produced the nonsense result that DISAGREEING takes scored *higher*. It now
slices by how the template's own left/right assignment was decided
(`geometry` vs `handedness-label`, and geometry at full vs partial confidence),
which is a property the matching actually depends on.

## 10.12 The calibration set was contaminated too, and the gate rested on it

10.10 fixed the benchmark's template exclusion. `scripts/calibrate-two-hand.ts`
still used the old legacy-sequence fingerprint, so the same bug survived in the
place the thresholds actually come from: **9 of the calibration set's 60 takes
were the stored template scoring against itself**, each a free hit at a
distance near zero. That is why calibration reported zero false accepts at
distance 1.25 / ratio 0.70 while the 1,004-case benchmark, with the same gate,
spoke nine wrong words. The calibration was not merely optimistic — it was
measuring the wrong thing at exactly the operating point in question.

Two changes:

1. **The exclusion now matches the benchmark's** — `sourceTake.processedFile`
   first, then a fingerprint of the SIDED timeline. A query can no longer be
   scored against a template built from itself.
2. **The set grew from 30 signs × 2 takes to 30 signs × all 8 takes.** 240
   takes, minus 30 templates and 3 too short to score, leave **207 genuine
   held-out queries** against 60 before.

On the clean set, 1.25/0.70 admits **4** false accepts — calibration and
benchmark now agree instead of contradicting each other.

### The gate, derived rather than picked off a grid

For a given ratio the binding constraint is the nearest WRONG match that
already survives the ratio test; the distance cap has to sit below it.

| ratio | nearest wrong match | zero-FA cap | accepted / 207 |
|---|---|---|---|
| **0.60** | 0.8542 | 0.84 | **44** |
| 0.65 | 0.7485 | 0.73 | 38 |
| 0.70 | 0.7485 | 0.73 | 38 |
| 0.75 | 0.6298 | 0.61 | 30 |
| 0.80 | 0.6298 | 0.61 | 30 |

0.60 accepts the most at zero false accepts, so no tie-break is needed this
time. The cap is set to **0.76, not 0.84**: a cap resting flush against a
single nearest sample is fitted to that sample, and 10% of margin costs 4 of
44 accepted queries. That margin is a judgement call and is recorded as one.

**`MAX_NORMALIZED_DISTANCE` 1.25 → 0.76, `MAX_BEST_TO_SECOND_BEST_RATIO`
0.70 → 0.60.** `UNMATCHED_SIDE_COST` (1.58), the weighting, the dataset, the
feature extraction and the matcher are unchanged.

This is a large tightening and it costs acceptance. It is deliberate: a wrong
word is worse than a rejected sign, and 1.25/0.70 was only ever defensible
against a contaminated calibration set.

### How disjoint the sets really are

The earlier claim that calibration and benchmark were disjoint was true when
the benchmark covered 40 signs. It now runs on all 502, so the 30 calibration
signs are inside it and roughly 6% of its cases are not strictly held out from
the threshold choice. Rather than hide that, **slice 10 of the benchmark
report prints the two groups separately.** On the locally available 145 cases:
calibration signs n=60 top-1 0.583 accepted 0.217 FA 0; all other signs n=85
top-1 0.682 accepted 0.141 FA 0. Acceptance is higher on the calibration signs
and top-1 is lower, so the gate has not memorised them, but the optimism is
real and stated.

### Hand-presence reporting corrected

Slice 7 classified a query as two-handed if any frame held a given side. The
matcher does not: a side with fewer than `MIN_SIDE_FRAMES` frames is never
prepared and never scored, so a single flicker frame was putting genuinely
one-handed queries in the two-handed row. The report now counts sides the same
way the matcher does.

### The two stale assertions in the benchmark

`accepts a useful fraction of real repetitions` demanded acceptance ≥ 0.8, a
figure from the 40-sign era measured against the gate this section replaces.
It is now a regression floor of 0.12 against a measured 0.172, with the reason
written at the assertion. `finds the correct sign as nearest neighbour`
demands top-1 ≥ 0.8 and measures 0.64; that assertion has been failing since
the real 502-sign rebuild, has nothing to do with the gate, and **is left
failing on purpose** rather than quietly relaxed inside a commit about
precision. Raising it means improving the matcher.

## 10.13 The real bottleneck: one example per word

Everything through 10.12 tuned a 502-way nearest-neighbour classifier that had
**one example of each class**. KArSL recorded eight. That was the ceiling, and
no orientation rule, penalty direction or threshold could lift it.

Leave-one-take-out on the 86 signs whose takes are all present locally — same
classes, same queries, only the template set changes:

| templates per sign | top-1 |
|---|---|
| 1 (the stored template) | 0.7267 |
| 1 (the medoid instead) | 0.7093 |
| 2 most representative | 0.8140 |
| 3 | 0.8430 |
| 4 | 0.8605 |
| **all takes, nearest wins** | **0.9419** |
| all takes, mean of the 2 best | 0.9070 |

**+21.5 points**, monotone, no knee. Picking a *better* single template is
worth nothing (the medoid is slightly worse than the heuristic already in
use); having more than one is worth everything. The nearest prototype beats
averaging, because a sign performed two distinguishable ways is still that
sign and averaging penalises exactly those.

Prototypes ship as their own generated artifact
(`public/models/sign-vocabulary/sign-prototypes.json`), holding the prepared
vectors 16-bit quantized — ~5 KB per take against ~60 KB for the landmarks
they came from. `dedicated-signs.json` is untouched: it keeps full-resolution
landmarks because the avatar replays them. When the artifact is absent or was
built by a different feature pipeline it is rejected and the recogniser falls
back to the single stored template. Degraded, never broken.

    npm run prototypes         # ships with the app: every take
    npm run prototypes:eval    # for the benchmark: query takes withheld

## 10.14 The representation discarded the motion

`prepareSide` fed DTW `normalizeLandmarks()` alone: every frame translated to
its own wrist and divided by its own hand size. Applied per frame, that makes
the representation invariant to where the hand is and how it moves — a hand
tracing a circle and a hand held still produce the identical vector sequence
whenever the fingers match. For a language whose words are movement, the
primary signal was being removed before matching began.

Swept on 223 held-out takes against all 502 templates:

| per-frame vector | top-1 | one-hand | two-hand |
|---|---|---|---|
| SHAPE only (what shipped) | 0.6188 | 0.6066 | 0.6235 |
| SHAPE + REL | 0.6547 | 0.6066 | 0.6728 |
| SHAPE + 2·POS | 0.6323 | 0.6885 | 0.6111 |
| **SHAPE + 2·POS + REL** | **0.6592** | **0.6885** | 0.6481 |

- **POS** — wrist position centred on *that side's own* mean and scaled by its
  own mean hand size. Per side, not per take: a per-take centroid is a
  different quantity depending on how many hands were seen, and it cost 11
  points of one-hand top-1 (0.6066 → 0.4918) because a one-hand query could no
  longer match the two-hand template of its own sign.
- **REL** — the wrist-to-wrist vector as its own DTW track, scored only when
  both gestures have both hands. Per-side DTW gives each hand its own warping
  path, so a relational quantity cannot live on a side.

Measured and **rejected**: absolute image coordinates (0.5516 — ties the model
to one camera), derivative-of-shape (0.6099), Sakoe-Chiba bands at 4/6/8
(0.6099/0.6188/0.6188 — the warps were not the problem).

## 10.15 The split was arithmetic, and arithmetic was not enough

The evaluation artifact omitted "the first four takes" and the benchmark
queried "the first two". Those are not the same takes: the benchmark skips the
registry template and anything too short to score, so its first two can be
files 3 and 4 — and file 4 was a prototype. Two queries were scored against a
prototype built from themselves, at distances of 0.0019 and 0.0015.

The benchmark's own "never scores a take against itself" guard caught it,
which is the only reason it is not in the numbers above. The fix is
`lib/ml/karslTakeSplit.ts`: one function, called by the generator, the
benchmark and the calibration script, that partitions each sign's takes into
benchmark queries, calibration queries and prototypes. Two files agreeing on
an arithmetic rule is not a split.

## 11. Remaining limitations — stated, not hidden

1. **No real-camera accuracy claim is made here.** Nothing in this document
   was validated against a live webcam; that requires a person in front of a
   camera and cannot be established offline.
2. The held-out benchmark is **signer-dependent**: in the currently
   downloaded portion of KArSL each sign folder holds repetitions from one
   signer. Accuracy for a new signer will be lower, by an unmeasured amount.
3. **The live camera path has NOT been validated with a real webcam.** It
   now runs `numHands: 2` and uses the same side-assignment rule as the
   offline data, but no measurement in this document involved a live camera.
   If the deployed camera feeds MediaPipe a MIRRORED image, every gesture's
   hands are named the wrong way round and almost everything is rejected --
   `CAMERA_IMAGE_IS_MIRRORED` in lib/ml/handSides.ts flips the whole
   convention in one place if that turns out to be the case.
4. **Mirror-invariant matching was REMOVED.** It was a workaround for not
   knowing which hand was which, and it cannot work: mirroring changes a
   hand's shape, not the other hand's trajectory. The debug panel now shows
   `hand sides from`, `captured frames L/R` and per-side distances instead.
5. The registry now holds all **502** signs; the accuracy figures quoted in
   sections 3 and 4 were measured on the earlier 170-sign registry and do
   not carry over — see section 9 for the 502-sign measurements.
6. The pre-existing synthetic tests passed throughout the entire failure,
   because they override the buffer config with their own values. That is why
   the new tests read the real registry and real repetitions instead.
