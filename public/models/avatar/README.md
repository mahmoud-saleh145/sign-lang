# Rigged avatar slot

Put a rigged humanoid here as:

```
public/models/avatar/signer.glb
```

The app picks it up automatically on next load. **No code change is needed.**
If the file is absent, the built-in procedural signer is used instead and the
UI says so (`3D Avatar (built-in signer)` / `avatar source: procedural`).

## Requirements

| Requirement | Why |
|---|---|
| `.glb` (or `.gltf` renamed to `.glb`) | loaded with three's `GLTFLoader` |
| Humanoid skeleton | arms and hands must be bones, not baked geometry |
| **Individual finger bones on BOTH hands** | this is the hard requirement — 3 bones per finger × 5 fingers × 2 hands. A model with mitten hands cannot sign. |
| Bone names in a recognised convention | see below |
| Licence permitting web distribution | you are shipping the file to every visitor |

## Bone naming

`lib/avatar/glbRig.ts` auto-detects these conventions — you do not have to
rename anything if your model uses one of them:

| Convention | Wrist | Index finger |
|---|---|---|
| Mixamo | `mixamorig:LeftHand` | `mixamorig:LeftHandIndex1/2/3` |
| Ready Player Me | `LeftHand` | `LeftHandIndex1/2/3` |
| VRM | `J_Bip_L_Hand` | `J_Bip_L_IndexProximal/Intermediate/Distal` |
| Blender / Rigify | `hand.L` | `f_index.01.L` … |

Detection is by *segment*, so `Left`/`Right`, `_L_`/`_R_`, `.L`/`.R` all work.
Metacarpal bones are optional — most humanoid rigs start the finger chain at
the knuckle, and that is handled.

After dropping the file in, open `/dev/avatar` and check:

```
avatar                  Loaded (glb)
hand rig                Valid
bound hand bones (L/R)  15 / 15     <- 15 or more per side
```

If it says **Invalid**, the panel lists exactly which bones could not be
bound, and `matchedNames` in the diagnostics shows what *was* matched, so a
mis-binding is visible rather than showing up only as a strangely posed hand.

## Where to get one

Anything meeting the requirements works. Two common routes:

- **Ready Player Me** — free, browser-based, full-body avatars with finger
  bones. Check their current terms for commercial use.
- **Mixamo** (Adobe) — characters have complete finger rigs. Free to use, but
  read Adobe's terms on redistributing the raw asset inside an application.

**Record the licence.** Fill this in when you add the file, so the project
has an auditable answer:

```
Asset:            <file name>
Source / URL:     <where it came from>
Author:           <creator>
Licence:          <e.g. CC0 / CC-BY 4.0 / vendor terms + link>
Commercial use:   <yes / no>
Redistribution
inside this app:  <permitted? under what conditions>
Date added:       <YYYY-MM-DD>
```

Do not commit an asset whose licence you have not verified. The app works
without one.

## What the model does and does not drive

Driven from **real KArSL landmark data**: every finger bone and the wrist, on
each hand the source sequence actually contains.

Not driven, because the dataset contains no such data: torso, head, facial
expression, and — on a GLB — the shoulder and elbow, which keep the model's
authored pose. Nothing here is animated from invented motion.
