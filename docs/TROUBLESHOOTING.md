# Troubleshooting

## Build fails fetching Google Fonts

`app/layout.tsx` intentionally does **not** use `next/font/google` — the
sandbox this was built in can't reach `fonts.googleapis.com`. If you're
deploying somewhere with normal internet access and want the real Space
Grotesk / Inter / JetBrains Mono fonts instead of the system-font fallback
stack in `app/globals.css`:

```diff
- import "./globals.css";
+ import { Space_Grotesk, Inter, JetBrains_Mono } from "next/font/google";
+ import "./globals.css";
+
+ const spaceGrotesk = Space_Grotesk({ variable: "--font-display", subsets: ["latin"], weight: ["500","600","700"] });
+ const inter = Inter({ variable: "--font-body", subsets: ["latin"], weight: ["400","500","600"] });
+ const jetbrainsMono = JetBrains_Mono({ variable: "--font-mono", subsets: ["latin"], weight: ["400","500"] });
```

and restore the `${spaceGrotesk.variable} ${inter.variable} ${jetbrainsMono.variable}`
classes on `<body>`. Then either delete or leave the fallback
`--font-display`/`--font-body`/`--font-mono` declarations in
`app/globals.css` — `next/font`'s CSS variables will take precedence.

## "Failed to load model weights" in the browser

`ArSLClassifier.loadFromUrl` fetches `/models/arsl-31/weights.json`, which
must exist under `public/models/arsl-31/`. If you retrained and the file is
missing, run `npm run ml:train` — it writes both `weights.json` and
`metadata.json` there.

## Camera works but nothing ever gets recognized

1. Toggle debug telemetry (top-right button in the camera view). Check
   `raw label` / `raw confidence` — if `handDetected` is `false`, MediaPipe
   isn't finding a hand; check lighting/framing.
2. If a hand is detected but `raw confidence` stays below 0.75
   (`confidenceThreshold` in metadata), the model isn't confident enough for
   any commit to happen — this is by design (spec: don't classify every
   frame), but check whether the sign being performed is actually in the
   31-class ArSL alphabet vocabulary (`public/models/arsl-31/metadata.json`
   → `classes`).

## MongoDB errors / dataset tool says "Database not configured"

Copy `.env.example` to `.env.local`, set `MONGODB_URI` to a real connection
string (e.g. a free MongoDB Atlas cluster), restart the dev server. The rest
of the app (translator, camera, speech) works fine without this — only
`/api/dataset/samples` (POST) and session logging require it.

## `npm run verify` fails on `ml:test` with "fixture missing"

Run `npm run ml:parity-fixture` first — it needs to generate
`ml/tests/fixtures/parity_fixture.json` and
`ml/tests/fixtures/model_parity_fixture.json` before the Python tests that
consume them can pass. `ml:parity-fixture` in turn needs
`public/models/arsl-31/weights.json` to already exist (i.e. run
`npm run ml:train` first).

## iOS Safari: camera works but is very slow / drops frames

Try switching MediaPipe's delegate from GPU to CPU in
`lib/mediapipe/handLandmarker.ts` (`delegate: "GPU"` → `"CPU"`) — iOS
Safari's WebGL delegate support for MediaPipe Tasks has been inconsistent
across versions. This is a real, not-yet-applied fix; it wasn't changed by
default because it trades battery/CPU load for reliability and should be
decided per real-device testing (see `docs/MOBILE_TESTING.md`).
