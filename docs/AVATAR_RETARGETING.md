# Voice/Text → Arabic Sign Language → 3D avatar

What was audited, what was built, and what this system does not do. Every
number here came from running code against the data in this repository.

## 1. Audit: where the old implementation stopped

The **translation layer was already correct** and was left alone.
`lib/sign-language/signTranslator.ts` already did longest-match-first phrase
matching against the registry, normalizing registry text and input the same
way. Verified against the real registry:

```
"هيكل عظمي"        → 1 step, dedicated, 47 real frames
"أريد هيكل عظمي"   → "اريد" spelled (4 letters) + "هيكل عظمي" as ONE sign
"تحليل دم"         → 1 step, dedicated, 62 real frames
"كتاب"             → 4 spelled letters (correctly, not in the registry)
```

48 of the 170 registry entries are multi-word phrases; the longest label is
5 words; `إسعافات أولية` is present. Three things were actually broken:

1. **`components/avatar/HandSignerAvatar.tsx` was a 2D SVG** — `<line>`
   elements between 21 `<circle>` elements. That was the entire visual layer.
2. **`lib/playback/playbackController.ts` destroyed real timing.**
   `stepDurationMs` was a fixed 1200 ms. Real registry sequences run 11–90
   frames at the source 30 fps, i.e. **0.37 s to 3.00 s**, so every sign
   played at the wrong speed — a 3 s sign compressed 2.5×, a 0.37 s sign
   stretched 3.2×.
3. **No text input** — voice only. No debug panel, no `/dev/avatar`.

The project had **no 3D engine at all** before this work.

## 2. The avatar asset: there isn't one

No GLB, no downloaded model, no runtime CDN. The entire signer is
constructed from Three.js primitives in `components/avatar/signerRig.ts`, so
there is **no third-party licence attached to the avatar**. `drei`'s
`<Environment>` was deliberately not used because its presets fetch an HDRI
from a CDN at runtime.

This was chosen over shipping a rigged humanoid GLB because the hand rig can
then match MediaPipe's 21-landmark topology *exactly* — one bone per
landmark segment — which removes bone-name mapping as a failure mode
entirely.

It is a **rigid-segment rig, not a smooth-skinned mesh**: each phalanx is a
capsule parented to a `Bone`, each joint a sphere. Described accurately
rather than called "skinning", because it isn't.

## 3. Hand proportions come from the data

Every bone length in `lib/avatar/handRig.ts` is the median of that segment
across **1,447 frames sampled from all 170 real sequences**, normalized by
each frame's wrist→middle-MCP distance. At playback, `calibrateBoneLengths()`
re-derives proportions from the specific sign being played and holds them
constant for that sign.

Which percentile to calibrate on was **measured, not argued**. The intuition
that a high percentile recovers true (unforeshortened) length turned out to
be wrong for this purpose. Reconstruction error against 40 real frames, in
avatar units where wrist→middle-MCP = 0.085:

| calibration | p50 err | p90 err | max err |
|---|---|---|---|
| global medians | 0.0177 | 0.0463 | 0.1046 |
| **per-sequence p50** | **0.0012** | **0.0105** | **0.0379** |
| per-sequence p75 | 0.0016 | 0.0262 | 0.0753 |
| per-sequence p90 | 0.0021 | 0.0366 | 0.1095 |
| per-sequence max | 0.0054 | 0.0592 | 0.2163 |

The median wins because the target being reproduced *is* the projected
recording, not an unobservable true anatomy.

## 4. Retargeting

`lib/avatar/retarget.ts`. Per frame:

1. **Wrist orientation** from two measured palm directions — `up = L9 − L0`,
   `across = L17 − L5`, `forward = up × across`, `side = up × forward`.
   Two directions, not one, because a single direction leaves palm roll
   undetermined.
2. **Each bone, parent before child**: `worldDir = normalize(L[to] − L[from])`,
   `localDir = parentWorldQuat⁻¹ · worldDir`, and the bone's local quaternion
   is the rotation taking `(0,1,0)` onto `localDir`. Every bone's rest
   direction is its own local +Y, so that quaternion is assigned directly.

Only **rotations** are retargeted; bone lengths stay fixed. Rebuilding the
hand from raw landmark positions each frame would make it grow and shrink
with MediaPipe noise, and would be point-placement rather than skeletal
animation.

**A real bug found and fixed here:** the wrist basis was initially built as
`side = forward × up`, which is **left-handed** (det = −1).
`Quaternion.setFromRotationMatrix` silently returns a meaningless quaternion
for a reflection, and the whole hand came out flipped ~180° about Y. Caught
by the forward-kinematics round-trip test, not by looking at the screen.

**Verification** (`lib/avatar/retarget.test.ts`): solve rotations from a real
frame, run FK with calibrated lengths, compare against the real hand. Bone
direction agreement `dot > 0.9` for all 20 bones across 40 real frames from
across the registry; worst joint position deviation 0.0379.

### Depth is weak, and that is a data property

Within-frame z span is p50 = 0.026 while hand scale is p50 = 0.066 — depth
varies over ~40% of hand size, and the extraction stored MediaPipe
`landmarks` (image space), not `worldLandmarks` (metric). The reconstruction
is well-conditioned in the image plane and **weakly conditioned in depth**.
Poses distinguished mainly by depth will look flatter than the original. The
retargeter cannot recover what was not recorded.

## 5. Timing

`lib/avatar/signAnimationController.ts`. A KArSL sign's duration is
`frameCount / 30 fps` — its real recorded length. A spelled letter has one
real frame and therefore no recorded duration, so it gets a 900 ms display
hold, and `durationIsReal: false` says so in the debug panel. The old fixed
1200 ms controller is untouched and still passes its own tests; it is simply
no longer what the avatar uses.

## 6. What is real and what is derived — the honest split

| Part | Source |
|---|---|
| Finger and wrist **rotations** | **Real** — solved from KArSL landmarks |
| Wrist **position** | **Real** — the recorded wrist path, mapped into a signing volume |
| Hand **proportions** | **Real** — measured from the sign's own frames |
| Sign **duration** | **Real** — frame count ÷ 30 fps |
| Elbow and shoulder | **Derived** — two-bone IK reaching the real wrist. KArSL stores no shoulder or elbow data. The elbow swivel is a fixed choice. |
| Torso, head, left arm | **Static** — no motion at all, because the dataset has none |
| Face | **Static** — two eyes and a brow, purely so the viewer can tell which way the avatar faces. No facial data exists in KArSL. |

Nothing on this list is invented motion presented as recorded motion.

## 7. Shared vocabulary with the camera

`lib/ml/dynamicSignRecognizer.ts` (camera) and `lib/avatar/karslSign.ts`
(avatar) both read the same
`public/models/sign-vocabulary/dedicated-signs.json`. A sign the camera can
recognize is a sign the avatar can perform, by construction.

## 8. Limitations — stated, not hidden

1. **KArSL is isolated word data.** Playing several signs in sequence is not
   Arabic Sign Language grammar. There is no sentence-level ordering, no
   facial expression, no non-manual marker, no body language.
2. **One hand only.** The dataset stores a single hand per sign and no
   handedness field. The left arm is static; genuinely two-handed signs
   cannot be represented.
3. **Handedness/mirroring of the source is unverified.** The registry records
   no handedness, so a `mirror` toggle is exposed on both `/dev/avatar` and
   VoiceToSign rather than guessing.
4. **170 of 502 signs.** Everything else falls back to letter spelling, and
   the UI labels that in red as *not* a real word sign.
5. **Depth fidelity** — see §4.
6. **Arm pose is not recorded** — see §6.
7. The avatar has been verified rendering in headless Chromium with
   SwiftShader. It has not been profiled on a low-end mobile GPU.

## 9. How to test

```bash
npm run dev
```

- **`/dev/avatar`** — pick any of the 170 signs, Play. No camera, no speech,
  no text matching. If a sign works here, the data and retargeting are fine.
- **`/translator` → Voice → Sign** — type `أريد هيكل عظمي` and press Sign.
  Expected: 5 steps, where `هيكل عظمي` is **one** chip (KArSL, 47 frames,
  1.57 s) and `ا ر ي د` are four separate letter-fallback chips. Turn on
  `debug` to see matched type, sequence frames, bone count and animation
  state live.
- Type a word that is not in the registry to see the Arabic
  "not available in KArSL" message plus the labelled letter fallback.

```bash
npm run lint && npm run typecheck && npm run build && npm test && npm run ml:test
```
