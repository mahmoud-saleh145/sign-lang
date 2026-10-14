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
import { preloadDynamicSignRegistry } from "@/lib/ml/dynamicSignRecognizer";
import { RECOGNITION_MODES, type RecognitionMode } from "@/lib/ml/recognitionMode";


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
  const [mode, setModeState] = useState<RecognitionMode>("letter");
  const [wordTranscript, setWordTranscript] = useState<TranscriptEntry[]>([]);
  const [manualCapture, setManualCapture] = useState(false);
  const modeRef = useRef<RecognitionMode>("letter");

  const stopEverything = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    landmarkerRef.current?.close();
    streamRef.current = null;
    landmarkerRef.current = null;
    rafRef.current = null;
  }, []);

  useEffect(() => stopEverything, [stopEverything]);

  const loopRef = useRef<() => void>(() => { });
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
        if (tick.newDynamicSignEntry) {
          // WORD results live in their own list. Only "both" mode also
          // merges them into the letter transcript (the pipeline decides;
          // this mirrors it for display).
          setWordTranscript((prev) => [...prev, tick.newDynamicSignEntry!]);
          if (modeRef.current === "both") {
            setTranscript((prev) => [...prev, tick.newDynamicSignEntry!]);
          }
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
        // TWO hands. Most KArSL signs are two-handed (477 of 502 in the
        // registry), and the second hand is half the discriminative signal.
        HandLandmarkerService.create({ numHands: 2 }),
        fetch("/models/arsl-31/metadata.json"),
      ]);
      // Dynamic KArSL recognition is additive: if this fails (e.g. no
      // dedicated-signs.json yet), letter recognition must still work.
      preloadDynamicSignRegistry().catch((err) => {
        console.warn("Dynamic KArSL sign recognition unavailable:", err);
      });
      if (!metaRes.ok) throw new Error("Failed to load model metadata.");
      const meta = (await metaRes.json()) as ModelMetadata;
      const classifier = await ArSLClassifier.loadFromUrl("/models/arsl-31/weights.json");

      landmarkerRef.current = landmarker;
      pipelineRef.current = new SignRecognitionPipeline(classifier, {
        confidenceThreshold: meta.confidenceThreshold,
        minStableFrames: meta.temporalWindow.minStableFrames,
        maxGapFrames: meta.temporalWindow.maxGapFrames,
        minFramesBetweenRepeats: 5,
      }, undefined, modeRef.current);
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
    setWordTranscript([]);
    setArabicText("");
    pipelineRef.current?.clearTranscript();
  }, []);

  /** Switching mode resets both state machines inside the pipeline, so
   * frames buffered under one mode never contribute to the other's result. */
  const setMode = useCallback((next: RecognitionMode) => {
    modeRef.current = next;
    setModeState(next);
    setManualCapture(false);
    pipelineRef.current?.setMode(next);
  }, []);

  const startManualCapture = useCallback(() => {
    pipelineRef.current?.beginManualCapture();
    setManualCapture(true);
  }, []);

  const finishManualCapture = useCallback(() => {
    const pipeline = pipelineRef.current;
    setManualCapture(false);
    if (!pipeline) return;
    const tick = pipeline.endManualCapture(performance.now());
    setTelemetry(tick.telemetry);
    if (tick.newDynamicSignEntry) {
      setWordTranscript((prev) => [...prev, tick.newDynamicSignEntry!]);
      if (modeRef.current === "both") setTranscript((prev) => [...prev, tick.newDynamicSignEntry!]);
    }
  }, []);

  const wordModeActive = mode === "word" || mode === "both";
  const lastWord = wordTranscript[wordTranscript.length - 1] ?? null;

  return (
    <div className="flex flex-col gap-4">
      <div
        role="radiogroup"
        aria-label="Recognition mode"
        className="flex rounded-full bg-[var(--color-surface)] p-1 border border-white/10"
      >
        {RECOGNITION_MODES.map((m) => (
          <button
            key={m}
            role="radio"
            aria-checked={mode === m}
            onClick={() => setMode(m)}
            className={`flex-1 py-2 rounded-full font-mono text-xs uppercase tracking-wide transition-colors ${mode === m
              ? "bg-[var(--color-accent-active)] text-[#0B1220] font-semibold"
              : "text-[var(--color-text-dim)]"
              }`}
          >
            {m === "letter" ? "Letters" : m === "word" ? "KArSL Words" : "Both"}
          </button>
        ))}
      </div>
      <p className="font-mono text-[11px] text-[var(--color-text-dim)] -mt-2">
        {mode === "letter" && "Static letter recognition only. Dynamic word matching is not running."}
        {mode === "word" && "KArSL word mode: letters are never committed. One complete gesture = one sign."}
        {mode === "both" && "Both paths run; letter and word results are kept in separate transcripts."}
      </p>
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
              {mode === "word" ? (
                <>
                  <div className="font-mono text-xs text-[var(--color-text-dim)]">
                    {telemetry?.dynamicPhase === "buffering"
                      ? `capturing gesture — ${telemetry.dynamicGestureFrameCount} frames${telemetry.dynamicManualCapture ? " (manual)" : ""}`
                      : telemetry?.dynamicPhase === "classifying"
                        ? "classifying complete gesture…"
                        : "waiting for a sign"}
                  </div>
                  <p className="font-display text-3xl mt-1 min-h-[2rem]" dir="rtl" lang="ar">
                    {lastWord ? (
                      lastWord.displayLabel
                    ) : (
                      <span dir="ltr" className="text-[var(--color-text-dim)] text-base">
                        No sign recognized yet
                      </span>
                    )}
                  </p>
                  {telemetry?.lastDynamicAccepted === false && (
                    <p className="font-mono text-xs text-[var(--color-accent-error)]">
                      rejected: {telemetry.lastDynamicRejectionReason} (best {telemetry.lastDynamicBestText ?? "-"} @{" "}
                      {telemetry.lastDynamicBestDistance?.toFixed(3)}, ratio{" "}
                      {telemetry.lastDynamicBestToSecondBestRatio?.toFixed(3)})
                    </p>
                  )}
                  {telemetry?.gestureDiscardedThisFrame && (
                    <p className="font-mono text-xs text-[var(--color-accent-error)]">
                      gesture discarded before classification: {telemetry.gestureDiscardReason}
                    </p>
                  )}
                </>
              ) : (
                <>
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
                </>
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

      {(state === "streaming" || state === "paused") && wordModeActive && (
        <button
          onPointerDown={startManualCapture}
          onPointerUp={finishManualCapture}
          onPointerLeave={() => manualCapture && finishManualCapture()}
          className={`py-4 rounded-full font-semibold transition-colors ${manualCapture
            ? "bg-[var(--color-accent-error)] text-[#0B1220]"
            : "border border-white/20 text-[var(--color-text)]"
            }`}
        >
          {manualCapture ? "Recording sign — release to classify" : "Hold to capture one sign"}
        </button>
      )}

      {wordModeActive && (
        <section
          className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4"
          aria-label="Recognized KArSL signs"
        >
          <div className="flex items-center justify-between">
            <h2 className="font-mono text-xs uppercase tracking-wide text-[var(--color-text-dim)]">
              Recognized KArSL signs
            </h2>
            <button onClick={clearTranscript} className="font-mono text-xs text-[var(--color-text-dim)] underline">
              clear ({wordTranscript.length})
            </button>
          </div>
          {wordTranscript.length === 0 ? (
            <p className="font-mono text-xs text-[var(--color-text-dim)] mt-2">
              Nothing recognized yet. Turn the debug panel on to see which stage is stopping.
            </p>
          ) : (
            <ul className="mt-2 space-y-1" dir="rtl" lang="ar">
              {wordTranscript.map((e) => (
                <li key={e.id} className="flex items-baseline justify-between gap-3">
                  <span className="font-display text-2xl">{e.displayLabel}</span>
                  <span dir="ltr" className="font-mono text-xs text-[var(--color-text-dim)]">
                    {e.sign} · conf {(e.confidence * 100).toFixed(0)}%
                  </span>
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {mode !== "word" && (
        <TranscriptPanel transcript={transcript} arabicText={arabicText} onClear={clearTranscript} />
      )}

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