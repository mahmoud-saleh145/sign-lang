# Mobile Testing

## HTTPS requirement

Browsers only grant camera/microphone access on a "secure context": HTTPS,
or `http://localhost` specifically. A phone on your LAN hitting your dev
machine's `http://192.168.x.x:3000` is **not** secure context and camera/mic
permission prompts will not work. Options:

1. **Deploy it** (e.g. to Vercel — see `docs/DEPLOYMENT.md` in the README)
   and test against the real HTTPS URL. Simplest and most representative of
   production.
2. **Tunnel your local dev server** with a tool that gives you an HTTPS URL
   (e.g. `ngrok http 3000`, or Vercel's own `vercel dev` tunnel). Point your
   phone at the HTTPS tunnel URL, not the raw LAN IP.
3. **Chrome's "insecure origins treated as secure" flag** — set
   `chrome://flags/#unsafely-treat-insecure-origin-as-secure` on the
   *desktop* Chrome you use to remote-debug, listing your LAN dev URL. This
   does not help Safari on iPhone.

## Camera/microphone permission flow

- The app never requests camera/mic permission on page load — only when the
  person taps "Start camera" / "Start speaking" (spec section 25). Confirm
  this holds on a real device: the permission prompt should appear only
  after your tap, not before.
- Permission denial is handled explicitly (`CameraView`'s
  `permission-denied` state) — verify by denying the prompt and confirming
  you see a clear message, not a blank screen or console error.
- After denying once, most mobile browsers won't re-prompt automatically on
  a second tap — verify the shown message makes it clear the person needs to
  go into browser site settings to re-enable it.

## Supported browsers

- **Camera pipeline (MediaPipe + WebGL/WASM):** modern Chrome/Edge on
  Android, Safari 16+ on iOS. Test on both — MediaPipe's GPU delegate
  (`delegate: "GPU"` in `lib/mediapipe/handLandmarker.ts`) has historically
  had rougher edges on iOS Safari than desktop/Android Chrome; if you see
  degraded performance there, try switching `delegate` to `"CPU"` as a
  fallback and compare.
- **Speech recognition:** relies on the non-standard `webkitSpeechRecognition`
  / `SpeechRecognition` API. Chrome on Android supports it well. Safari on
  iOS has partial/inconsistent support depending on iOS version —
  `components/speech/SpeechToText.tsx` detects unavailability and shows a
  clear "not supported in this browser" message rather than failing silently
  (spec section 22's "browser compatibility detection" requirement).

## What to actually check on a real phone

1. Open the deployed HTTPS URL on the phone.
2. Tap "Start translating" → Sign → Text tab.
3. Tap "Start camera" — permission prompt appears, grant it.
4. Sign a few ArSL alphabet letters continuously, without pausing/pressing
   anything between them. Confirm the transcript accumulates letters and the
   stability bar visibly builds up before each commit.
5. Toggle "debug: on" and confirm telemetry updates live (raw label,
   confidence, segmentation state).
6. Tap Pause, confirm the camera feed freezes/stops processing; Resume,
   confirm it picks back up.
7. Tap Clear, confirm the transcript empties.
8. Switch to Speech → Text, tap "Start speaking," say something, confirm
   live interim + final text appears.
9. Rotate the phone / lock the screen and return — note whatever actually
   happens (this project has not been hardened against those cases; see
   `docs/TROUBLESHOOTING.md`).

## Honesty note

This project was built and automatically tested in a sandboxed environment
with no physical phone attached. Every claim above about *what should
happen* is backed by the actual code and the automated test suite
(`npm run verify`), but no one has physically run this on a phone yet. Do
step-by-step testing above before considering Version 1 verified end-to-end.
