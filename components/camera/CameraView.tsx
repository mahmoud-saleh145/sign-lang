"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { HandLandmarkerService } from "@/lib/mediapipe/handLandmarker";
import { ArSLClassifier } from "@/lib/ml/model";
import { SignRecognitionPipeline } from "@/lib/ml/recognitionPipeline";
import type { FrameTelemetry } from "@/lib/ml/recognitionPipeline";
import type { TranscriptEntry, ModelMetadata } from "@/types/features";
import TranscriptPanel from "@/components/translator/TranscriptPanel";
import DebugTelemetryPanel from "@/components/translator/DebugTelemetryPanel";
import StabilityBar from "@/components/translator/StabilityBar";
import { toArabicChar, isControlSign, controlSignSymbol, type ControlSign } from "@/lib/ml/arabicLabels";

function candidateDisplayLabel(label: string): string {
  if (isControlSign(label)) return controlSignSymbol(label as ControlSign);
  return toArabicChar(label) ?? label;
}

type CameraState =
  | "idle"
  | "requesting-permission"
  | "loading-model"
  | "streaming"
  | "paused"
  | "permission-denied"
  | "unavailable"
  | "error";

export default function CameraView() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<HandLandmarkerService | null>(null);
  const pipelineRef = useRef<SignRecognitionPipeline | null>(null);
  const rafRef = useRef<number | null>(null);
  const pausedRef = useRef(false);

  const [state, setState] = useState<CameraState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<ModelMetadata | null>(null);
  const [transcript, setTranscript] = useState<TranscriptEntry[]>([]);
  const [arabicText, setArabicText] = useState("");
  const [telemetry, setTelemetry] = useState<FrameTelemetry | null>(null);
  const [showDebug, setShowDebug] = useState(false);

  const stopEverything = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    landmarkerRef.current?.close();
    streamRef.current = null;
    landmarkerRef.current = null;
    rafRef.current = null;
  }, []);

  useEffect(() => stopEverything, [stopEverything]);

  const loopRef = useRef<() => void>(() => {});
  useEffect(() => {
    loopRef.current = () => {
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;
      const pipeline = pipelineRef.current;
      if (!video || !landmarker || !pipeline) return;

      if (!pausedRef.current && video.readyState >= 2) {
        const detection = landmarker.detectForVideo(video, performance.now());
        const tick = pipeline.processFrame(detection);
        setTelemetry(tick.telemetry);
        setArabicText(tick.telemetry.arabicText);
        if (tick.newTranscriptEntry) {
          setTranscript((prev) => [...prev, tick.newTranscriptEntry!]);
        }
      }
      rafRef.current = requestAnimationFrame(() => loopRef.current());
    };
  });
  const loop = useCallback(() => loopRef.current(), []);

  const start = useCallback(async () => {
    setErrorMessage(null);
    setState("requesting-permission");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
        audio: false,
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setState("loading-model");
      const [landmarker, metaRes] = await Promise.all([
        HandLandmarkerService.create({ numHands: 1 }),
        fetch("/models/arsl-31/metadata.json"),
      ]);
      if (!metaRes.ok) throw new Error("Failed to load model metadata.");
      const meta = (await metaRes.json()) as ModelMetadata;
      const classifier = await ArSLClassifier.loadFromUrl("/models/arsl-31/weights.json");

      landmarkerRef.current = landmarker;
      pipelineRef.current = new SignRecognitionPipeline(classifier, {
        confidenceThreshold: meta.confidenceThreshold,
        minStableFrames: meta.temporalWindow.minStableFrames,
        maxGapFrames: meta.temporalWindow.maxGapFrames,
        minFramesBetweenRepeats: 5,
      });
      setMetadata(meta);

      pausedRef.current = false;
      setState("streaming");
      rafRef.current = requestAnimationFrame(loop);
    } catch (err) {
      stopEverything();
      if (err instanceof DOMException && err.name === "NotAllowedError") {
        setState("permission-denied");
      } else if (err instanceof DOMException && err.name === "NotFoundError") {
        setState("unavailable");
      } else {
        setState("error");
        setErrorMessage(err instanceof Error ? err.message : "Something went wrong starting the camera.");
      }
    }
  }, [loop, stopEverything]);

  const stop = useCallback(() => {
    stopEverything();
    setState("idle");
  }, [stopEverything]);

  const togglePause = useCallback(() => {
    pausedRef.current = !pausedRef.current;
    setState(pausedRef.current ? "paused" : "streaming");
  }, []);

  const clearTranscript = useCallback(() => {
    setTranscript([]);
    setArabicText("");
    pipelineRef.current?.clearTranscript();
  }, []);

  return (
    <div className="flex flex-col gap-4">
      <div className="relative w-full aspect-[3/4] max-h-[60vh] rounded-2xl overflow-hidden bg-[var(--color-surface)] border border-white/10">
        <video
          ref={videoRef}
          className="w-full h-full object-cover -scale-x-100"
          playsInline
          muted
          aria-label="Live camera preview for sign language recognition"
        />

        {state !== "streaming" && state !== "paused" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-4 p-6 text-center bg-[var(--color-bg)]/90">
            <CameraStatusMessage state={state} errorMessage={errorMessage} />
            {(state === "idle" || state === "error" || state === "permission-denied" || state === "unavailable") && (
              <button
                onClick={start}
                className="font-display font-semibold text-lg px-8 py-4 rounded-full bg-[var(--color-accent-active)] text-[#0B1220] active:scale-95 transition-transform"
              >
                {state === "idle" ? "Start camera" : "Try again"}
              </button>
            )}
          </div>
        )}

        {(state === "streaming" || state === "paused") && (
          <>
            <div className="absolute top-3 left-3 right-3 flex items-center justify-between gap-2">
              <span className="font-mono text-xs px-2 py-1 rounded bg-black/50 text-[var(--color-text-dim)]">
                {metadata?.languageDisplayName ?? "ArSL"}
              </span>
              <button
                onClick={() => setShowDebug((v) => !v)}
                className="font-mono text-xs px-2 py-1 rounded bg-black/50 text-[var(--color-text-dim)]"
                aria-pressed={showDebug}
              >
                debug: {showDebug ? "on" : "off"}
              </button>
            </div>

            <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/70 to-transparent">
              <StabilityBar state={telemetry?.segmentationState ?? "IDLE"} />
              <p className="font-display text-3xl mt-2 min-h-[2rem]" dir="rtl" lang="ar">
                {telemetry?.currentCandidate ? (
                  candidateDisplayLabel(telemetry.currentCandidate)
                ) : (
                  <span dir="ltr" className="text-[var(--color-text-dim)] text-base">
                    No sign detected
                  </span>
                )}
              </p>
              {telemetry && telemetry.rawConfidence > 0 && (
                <p className="font-mono text-xs text-[var(--color-text-dim)]">
                  confidence {(telemetry.rawConfidence * 100).toFixed(0)}%
                </p>
              )}
            </div>
          </>
        )}
      </div>

      {(state === "streaming" || state === "paused") && (
        <div className="flex gap-3">
          <button
            onClick={togglePause}
            className="flex-1 py-3 rounded-full border border-white/20 font-medium"
          >
            {state === "paused" ? "Resume" : "Pause"}
          </button>
          <button
            onClick={stop}
            className="flex-1 py-3 rounded-full border border-white/20 font-medium"
          >
            Stop camera
          </button>
        </div>
      )}

      <TranscriptPanel transcript={transcript} arabicText={arabicText} onClear={clearTranscript} />

      {showDebug && <DebugTelemetryPanel telemetry={telemetry} metadata={metadata} />}
    </div>
  );
}

function CameraStatusMessage({
  state,
  errorMessage,
}: {
  state: CameraState;
  errorMessage: string | null;
}) {
  switch (state) {
    case "idle":
      return (
        <p className="text-[var(--color-text-dim)] max-w-xs">
          Sign into the camera and the transcript below fills in as you go — no
          need to stop between signs.
        </p>
      );
    case "requesting-permission":
      return <p className="font-mono text-sm">Requesting camera access…</p>;
    case "loading-model":
      return <p className="font-mono text-sm">Loading recognition model…</p>;
    case "permission-denied":
      return (
        <p className="text-[var(--color-accent-error)] max-w-xs">
          Camera access was denied. Enable camera permission for this site in
          your browser settings, then try again.
        </p>
      );
    case "unavailable":
      return (
        <p className="text-[var(--color-accent-error)] max-w-xs">
          No camera was found on this device.
        </p>
      );
    case "error":
      return (
        <p className="text-[var(--color-accent-error)] max-w-xs">
          {errorMessage ?? "Something went wrong."}
        </p>
      );
    default:
      return null;
  }
}
