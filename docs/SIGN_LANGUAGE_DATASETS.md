# Sign Language Dataset & Avatar Research

This documents the real research done before building the Voice → Sign
feature, per the project's "no fabricated sign mappings" rule. Nothing here
is assumed — every dataset was actually looked up and its access method
verified (or found to be blocked) before any code was written.

## Word/phrase-level Arabic Sign Language datasets found

| Dataset | Source | License | Samples | Data type | Accessible from this environment? |
|---|---|---|---|---|---|
| **KArSL** | hamzah-luqman.github.io/KArSL (KFUPM) | Not explicitly stated — "please cite" only | 502 isolated words, 75,300 clips (3 signers × 50 reps) | Kinect V2 RGB + depth + skeleton video | **No.** Files are hosted on a gated SharePoint link (`kfupmedusa-my.sharepoint.com`), not a domain this sandbox can reach, and not a license that clearly permits redistribution/reuse in a derivative product. |
| **ArabSign** | hamzah-luqman.github.io/ArabSign (KFUPM) | Not stated | 9,335 samples, 50 continuous sentences, 6 signers | Kinect V2 color + depth + skeleton | **No.** No download link at all — the page says to email the author (hluqman@kfupm.edu.sa) directly. Requires human correspondence and an indeterminate approval process; not something I can obtain within this session. |
| **"Arabic words sign language video dataset"** | Zenodo, record 8035320 (Yahia, Allam, Naguib, Othman, Attia — Egyptian Chinese University) | **CC BY 4.0** (genuinely open) | 3,000 videos, 30 signs, 5 volunteers × 20 reps | Smartphone RGB video, 23 MB compressed | **Not yet.** This is the most promising real option — openly licensed and small — but `zenodo.org` isn't in this sandbox's outbound network allowlist, and the file is a binary `.rar` archive of videos, which isn't practical to retrieve through the text-oriented `web_fetch` tool (unlike the earlier CSV dataset, this isn't UTF-8 text). **See "How to unblock this" below.** |
| **ArabicSL-Net** | Zenodo 7771372 / Kaggle mirror | CC BY (stated) | ~30,000 videos, 307 words | Video | **No.** Same network reachability problem, and at this size, downloading and processing it isn't practical in this environment even if reachable. |
| Roboflow "Arabic Sign Language Recognition" | universe.roboflow.com | Unclear/varies per Roboflow dataset | 2,889 images | Static images (object-detection annotated) | **Not used even if accessible.** Many ArSL words are dynamic (motion matters); representing them with a single static image would misrepresent the sign, which the project's honesty rules don't allow. |

**Bottom line: no word- or phrase-level Arabic Sign Language dataset is both openly licensed and technically reachable from this environment right now.** This is why `lib/sign-language/signRegistry.ts` (the "dedicated sign" lookup table) is deliberately empty, and why Voice → Sign currently falls back to spelling every word letter-by-letter using the real, already-integrated 28-letter alphabet dataset.

### How to unblock this

The Zenodo 8035320 dataset is the realistic path forward: it's small (23MB), openly licensed (CC BY 4.0), and just needs to get onto a domain I can reach — the exact same pattern used to get the original alphabet CSVs into this project:

1. Download `https://zenodo.org/records/8035320/files/Dataset.rar?download=1` (23 MB) yourself.
2. Either upload it directly as a chat attachment, or push it to a public GitHub repo (I can reach `raw.githubusercontent.com`/`codeload.github.com` directly).
3. From there, real additional work is required before it's usable: extracting `.rar`, running MediaPipe on each video to get real per-frame landmark sequences, and building a small temporal model (the current 89-feature MLP only handles static poses — a word sign spans many frames, so it needs a sequence-aware approach, e.g. DTW matching against a handful of real reference sequences per word, which is the simplest thing that would actually work for only 30 classes). This is a real, separate, substantial task — flagging it honestly rather than starting it speculatively without knowing if you want to invest in it.

### Real video → landmark extraction pipeline (for KArSL or similarly-structured local video datasets)

Two scripts now exist for processing a real local video dataset end-to-end,
designed to run on your own machine (e.g. Windows) where you have the
actual video files and normal internet access:

1. **`ml/training/extract_karsl_skeletons.py`** — `VIDEO → per-frame landmarks`.
   Reads real video files frame-by-frame (streaming, no full-video RAM
   load), runs MediaPipe's real `HandLandmarker` (Tasks API — the same
   MediaPipe family the browser already uses) on each frame, and writes one
   JSON file per video with the full temporal sequence. Frames with no
   detected hand get `"landmarks": null` — never fabricated coordinates.
   This does **not** recognize or label signs; the sign ID comes only from
   the input folder name (e.g. `KArSL/0001/` → `signId: "0001"`, preserved
   as a string, never coerced to an integer).

2. **`ml/training/import_karsl.py`** — updated to consume that real output
   format directly: for each sign, it takes the first available processed
   video (reporting any other repetitions as available-but-unused, never
   silently merging them), drops any frames with no detected hand
   (reporting exactly how many), and writes
   `public/models/sign-vocabulary/dedicated-signs.json` in the exact shape
   `lib/sign-language/signRegistry.ts` expects. It still falls back to its
   original per-frame-file assumption for a differently-structured dataset.

**Real, verified limitation:** the `HandLandmarker` model asset
(`hand_landmarker.task`, ~10MB) is downloaded automatically from
`storage.googleapis.com` on first run. That host is blocked from this
project's build sandbox, so the real end-to-end MediaPipe path was verified
by direct Python API introspection (confirming exact method/field names)
plus a real dry run against synthetic no-hand video frames, but not against
a real hand video — it needs to run on a machine with normal internet
access, e.g. yours.

## Avatar / visual signer options investigated

| Option | Finding |
|---|---|
| **3D avatar (Three.js/React Three Fiber + a human model)** | Real open-source sign-avatar systems exist (DFKI's MMS-Player, the SignON project) but they require a Blender-based rigging/animation production pipeline and real motion-capture or hand-authored animation data per sign. None of them ship pre-made Arabic Sign Language animations I could just drop in — building this from scratch is a multi-week production effort (3D modeling/rigging + either mocap or manual keyframing per sign), not something buildable in this session, and there's no free ArSL-specific animation asset library to reuse. |
| **Pre-recorded sign videos** | Would need real video per supported sign — same dataset-access blocker as above (KArSL/ArabSign gated, Zenodo option not yet downloaded). Not implemented for the same reason. |
| **Pose/landmark-driven avatar (chosen)** | We already have real, licensed, per-letter hand landmark data (the same 28-letter dataset powering Sign→Text). A 2D hand-skeleton renderer driven directly by these real coordinates needs no new assets, no license risk, and — per the spec's own priority ("hand signs must be extremely clear," "prioritize clarity over graphics effects") — is arguably more legible on a small phone screen than a distant 3D figure would be. This is what `components/avatar/HandSignerAvatar.tsx` implements. |

**This is why Voice → Sign currently renders a real 2D hand-skeleton (SVG, driven by actual dataset coordinates) instead of a 3D avatar or video** — it's the only option that's both honest (real data, no placeholder/fabricated motion) and actually shippable right now.

## What this means concretely for V1 of Voice → Sign

- Every word is spelled out letter-by-letter, using the real per-letter landmark data.
- The UI explicitly labels each sign as `spelled-letter` (fallback) — never presented as if it were a native dedicated sign. See the caption under the hand renderer in `components/translator/VoiceToSign.tsx`.
- Characters with no corresponding class in the 28-letter set (e.g. Arabic-Indic digits, unmapped punctuation) are shown as an explicit "no sign data available" state — never silently dropped.
- No dedicated word/phrase sign is currently displayed anywhere in the app, because none currently exist in the codebase. `lib/sign-language/signRegistry.ts` is the single place this would change once real data is integrated — the translator logic already supports it (`findDedicatedSign` is checked first, before falling back to spelling).