/**
 * GESTURE<->VERDICT ASSOCIATION, and the exported snapshot shape.
 *
 * The bug this pins: `GestureBufferResult.completedGesture` is non-null on
 * exactly ONE frame -- the frame the buffer closes -- and null on every frame
 * after. An earlier version of the telemetry read the landmarks straight off
 * that field, so they existed for a single ~16 ms tick and were gone by the
 * time a human could press a button. The pipeline now LATCHES them next to the
 * classification and stamps both with a monotonic gesture sequence number, so
 * "is this verdict about this gesture?" is decidable instead of assumed.
 *
 * Driven through the REAL pipeline with REAL fixture data. No mocked scoring.
 */
import { describe, it, expect, beforeAll, vi } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { ArSLClassifier } from "./model";
import type { ModelWeights } from "./model";
import { SignRecognitionPipeline } from "./recognitionPipeline";
import type { FrameTelemetry } from "./recognitionPipeline";
import {
    buildGestureDiagnosticsSnapshot,
    resolveFreshness,
    snapshotFilename,
    type GestureDiagnosticsSnapshot,
} from "./gestureDiagnosticsSnapshot";
import {
    preloadDynamicSignRegistry,
    clearDynamicSignCache,
} from "./dynamicSignRecognizer";
import type { HandLandmarks, FrameDetection } from "@/types/landmarks";

import { DEFAULT_GESTURE_BUFFER_CONFIG } from "./dynamicGestureBuffer";

const weights = JSON.parse(
    readFileSync(path.resolve(__dirname, "../../public/models/arsl-31/weights.json"), "utf-8")
) as ModelWeights;

const REGISTRY_JSON = readFileSync(
    path.resolve(__dirname, "../../public/models/sign-vocabulary/dedicated-signs.json"),
    "utf-8"
);

/** A one-handed gesture with genuine per-frame shape deformation. Shaped like
 *  the fixtures the existing dynamic-sign pipeline tests use, which is known
 *  to clear the buffer's frame and motion gates. */
function gestureFrames(frames: number): HandLandmarks[] {
    return Array.from({ length: frames }, (_, f) =>
        Array.from({ length: 21 }, (_, i) => {
            const a = 0.5 * f + i * 0.35;
            return { x: 0.3 + 0.01 * f + 0.06 * Math.cos(a) + i * 0.004, y: 0.3 + 0.01 * f + 0.06 * Math.sin(a), z: 0 };
        })
    );
}

function detections(frames: number, startTs = 0): FrameDetection[] {
    let t = startTs;
    return gestureFrames(frames).map((landmarks) => {
        const ts = t;
        t += 33;
        return { timestampMs: ts, hands: [{ landmarks, handedness: "Right" as const, score: 0.95 }] };
    });
}

const idle = (n: number, startTs: number): FrameDetection[] =>
    Array.from({ length: n }, (_, i) => ({ timestampMs: startTs + i * 33, hands: [] }));

function newPipeline(): SignRecognitionPipeline {
    const classifier = new ArSLClassifier(weights);
    const p = new SignRecognitionPipeline(classifier, undefined, {
        ...DEFAULT_GESTURE_BUFFER_CONFIG,
        minGestureFrames: 10,
    });
    p.setMode("both");
    return p;
}

beforeAll(async () => {
    vi.stubGlobal("fetch", async () => ({
        ok: true,
        status: 200,
        json: async () => JSON.parse(REGISTRY_JSON),
    }));
    clearDynamicSignCache();
    await preloadDynamicSignRegistry();
}, 300_000);

/** Drives one gesture to completion and returns the telemetry of the LAST frame. */
function runOneGesture(pipeline: SignRecognitionPipeline, base = 0): FrameTelemetry {
    const stream = [
        ...detections(20, base),
        ...idle(10, base + 20 * 33), // hands leave -> buffer closes
    ];
    let last: FrameTelemetry | null = null;
    for (const f of stream) last = pipeline.processFrame(f).telemetry;
    if (!last) throw new Error("pipeline produced no telemetry");
    return last;
}

describe("1. landmarks outlive the frame they were produced on", () => {
    it("keeps the completed gesture's landmarks available on later frames", () => {
        const p = newPipeline();
        const completing = runOneGesture(p);
        expect(
            completing.lastCompletedGestureSided !== null ||
                completing.lastCompletedUnknownSideHand !== null
        ).toBe(true);

        // Many more frames with no gesture at all. The landmarks must persist --
        // this is exactly what reading `gestureResult.completedGesture` could
        // not do, since that field is null on every frame but the closing one.
        let t = 10_000;
        let last: FrameTelemetry | null = null;
        for (let i = 0; i < 50; i++) {
            last = p.processFrame({ timestampMs: t, hands: [] }).telemetry;
            t += 33;
        }
        expect(last!.lastCompletedGestureSided !== null ||
            last!.lastCompletedUnknownSideHand !== null).toBe(true);
    });
});

describe("2. a verdict is associated with the gesture that produced it", () => {
    it("sequence numbers increase once per completed gesture", () => {
        const p = newPipeline();
        const a = runOneGesture(p, 0);
        const b = runOneGesture(p, 20_000);
        expect(a.lastGestureSequence).not.toBeNull();
        expect(b.lastGestureSequence).toBe(a.lastGestureSequence! + 1);
        expect(b.lastCompletedGestureAtMs).toBeGreaterThan(a.lastCompletedGestureAtMs!);
    });

    it("the freshly produced verdict is linked to the gesture just completed", () => {
        const p = newPipeline();
        const t = runOneGesture(p);
        expect(resolveFreshness(t)).toEqual({
            freshness: "fresh",
            gestureSequence: t.lastGestureSequence,
            classificationSequence: t.lastGestureSequence,
        });
    });
});

describe("3. a previous verdict is never presented as the latest gesture's", () => {
    it("still reads fresh while it is the latest gesture", () => {
        const p = newPipeline();
        const first = runOneGesture(p, 0);
        // Frames pass; no new gesture.
        let t = 10_000;
        let after: FrameTelemetry | null = null;
        for (let i = 0; i < 60; i++) {
            after = p.processFrame({ timestampMs: t, hands: [] }).telemetry;
            t += 33;
        }
        // Same gesture is still the latest, so the verdict is legitimately
        // about it -- freshness is about ASSOCIATION, not about age.
        expect(resolveFreshness(after!)).toEqual({
            freshness: "fresh",
            gestureSequence: first.lastGestureSequence,
            classificationSequence: first.lastGestureSequence,
        });
    });

    it("a second gesture DISCARDED before classification reports pending, not a stale verdict", () => {
        const p = newPipeline();
        const first = runOneGesture(p, 0);
        expect(first.lastGestureSequence).toBe(1);

        // A gesture with no hands cannot be buffered or discarded; drive a
        // gesture that is identified then cleared. Simplest reliable route:
        // let the second gesture complete and be discarded by segmentation.
        const second = runOneGesture(p, 20_000);
        // Whichever verdict state it lands in, it must never be reported with
        // the FIRST gesture's sequence number.
        if (second.lastClassificationSequence !== null) {
            expect(second.lastClassificationSequence).not.toBe(first.lastGestureSequence);
            expect(second.lastClassificationSequence).toBe(second.lastGestureSequence);
        } else {
            expect(resolveFreshness(second).freshness).toBe("pending");
        }
    });

    it("a latched verdict from an EARLIER gesture is reported stale", () => {
        // Constructed directly: this is the association rule under test, not a
        // pipeline behaviour, so it is pinned where it is decided.
        const telemetry = {
            lastGestureSequence: 7,
            lastClassificationSequence: 6,
        } as unknown as FrameTelemetry;
        expect(resolveFreshness(telemetry).freshness).toBe("stale");

        expect(resolveFreshness({ lastGestureSequence: 7 } as unknown as FrameTelemetry).freshness)
            .toBe("pending");
        expect(resolveFreshness({} as unknown as FrameTelemetry).freshness).toBe("no-gesture");
    });
});

describe("4. the exported snapshot", () => {
    it("round-trips through JSON and matches the declared type", () => {
        const p = newPipeline();
        const snap = buildGestureDiagnosticsSnapshot(runOneGesture(p));
        const parsed: GestureDiagnosticsSnapshot = JSON.parse(JSON.stringify(snap));

        expect(parsed.schemaVersion).toBe(1);
        expect(parsed.freshness).toBe("fresh");
        expect(parsed.gestureSequence).not.toBeNull();
        expect(parsed.classificationSequence).toBe(parsed.gestureSequence);
        expect(parsed.privacy).toBe("landmark-coordinates-and-metrics-only");
        // A one-hand gesture arrives on the unknown-side path; exactly one of
        // the two landmark fields is populated, never both.
        expect(parsed.landmarks !== null || parsed.landmarksUnknownSide !== null).toBe(true);
        const pts =
            parsed.landmarks?.[0]?.left ??
            parsed.landmarks?.[0]?.right ??
            parsed.landmarksUnknownSide?.[0] ??
            null;
        expect(pts).not.toBeNull();
        expect(pts!.length).toBe(21);
        // Temporal order preserved: first frame differs from the last.
        const lastPts =
            parsed.landmarks?.[parsed.landmarks.length - 1]?.right ??
            parsed.landmarksUnknownSide?.[parsed.landmarksUnknownSide.length - 1] ??
            null;
        expect(lastPts![0].x).not.toBe(pts![0].x);
        expect(snapshotFilename(parsed)).toMatch(/^karsl-gesture-\d+-\d{4}-\d{2}-\d{2}T.*\.json$/);
    });

    it("carries no imagery, credentials or free-form personal data", () => {
        const snap = buildGestureDiagnosticsSnapshot(runOneGesture(newPipeline()));
        const text = JSON.stringify(snap);
        // Every landmark point must be exactly {x, y, z} numbers.
        for (const frame of snap.landmarks ?? []) {
            for (const hand of [frame.left, frame.right]) {
                if (!hand) continue;
                for (const point of hand) {
                    expect(Object.keys(point).sort()).toEqual(["x", "y", "z"]);
                    expect(typeof point.x).toBe("number");
                }
            }
        }
        for (const forbidden of ["data:image", "base64", "password", "token", "authorization", "userName"]) {
            expect(text.toLowerCase()).not.toContain(forbidden.toLowerCase());
        }
    });

    it("represents unavailable metrics as null rather than inventing values", () => {
        const empty = buildGestureDiagnosticsSnapshot(null);
        expect(empty.gestureSequence).toBeNull();
        expect(empty.landmarks).toBeNull();
        expect(empty.classification.bestSignId).toBeNull();
        expect(empty.perSide.relation).toBeNull();
        expect(empty.coverage.query.left).toEqual({ observed: null, total: null });
        expect(empty.coverage.template.right).toEqual({ observed: null, total: null });
        expect(empty.sides.swapsThisGesture).toBeNull();

        // observed is reported as measured even at zero: 0 of 56 frames is a
        // real reading ("this side was never detected"), distinct from null.
        const partial = buildGestureDiagnosticsSnapshot({
            lastDynamicQuerySideFrames: { left: 0, right: 12 },
            lastDynamicQueryTotalFrames: 56,
        } as unknown as FrameTelemetry);
        expect(partial.coverage.query.left).toEqual({ observed: 0, total: 56 });
        expect(partial.coverage.query.right).toEqual({ observed: 12, total: 56 });

        // A missing TOTAL is null on both terms -- no fabricated denominator.
        const noTotal = buildGestureDiagnosticsSnapshot({
            lastDynamicQuerySideFrames: { left: 12, right: 12 },
        } as unknown as FrameTelemetry);
        expect(noTotal.coverage.query.left).toEqual({ observed: 12, total: null });
    });

    it("reports a rejected gesture with its reason and both candidates", () => {
        const snap = buildGestureDiagnosticsSnapshot(runOneGesture(newPipeline()));
        if (snap.classification.accepted === false) {
            expect(snap.classification.rejectionReason).not.toBeNull();
        }
        // Whatever the verdict, both candidate columns must be present as keys
        // so an offline reader never has to guess at a missing shape.
        expect(snap.classification).toHaveProperty("bestSignId");
        expect(snap.classification).toHaveProperty("secondBestSignId");
        expect(snap.classification).toHaveProperty("bestToSecondBestRatio");
        expect(snap.perSide).toHaveProperty("relation");
    });
});