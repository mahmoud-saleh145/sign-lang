"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { HandLandmarkerService } from "@/lib/mediapipe/handLandmarker";
import type { HandLandmarks } from "@/types/landmarks";

type Stage = "idle" | "loading" | "ready" | "countdown" | "recording" | "reviewing" | "saving";

const COUNTDOWN_SECONDS = 3;
const RECORD_FRAMES = 45; // ~1.5s at 30fps
const MIN_VALID_FRAMES = 20; // require the hand to be visible for most of it

function newSessionId(): string {
  return `session-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`;
}

export default function DatasetCollectionPage() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const landmarkerRef = useRef<HandLandmarkerService | null>(null);
  const rafRef = useRef<number | null>(null);
  const recordedRef = useRef<HandLandmarks[]>([]);
  const recordingRef = useRef(false);
  const sessionIdRef = useRef(newSessionId());

  const [stage, setStage] = useState<Stage>("idle");
  const [classes, setClasses] = useState<string[]>([]);
  const [selectedLabel, setSelectedLabel] = useState<string>("");
  const [countdown, setCountdown] = useState(COUNTDOWN_SECONDS);
  const [lastCapturedFrames, setLastCapturedFrames] = useState(0);
  const [sampleCounts, setSampleCounts] = useState<Record<string, number>>({});
  const [recordedCount, setRecordedCount] = useState(0);
  const [message, setMessage] = useState<string | null>(null);

  const refreshCounts = useCallback(async () => {
    try {
      const res = await fetch("/api/dataset/samples");
      const data = await res.json();
      setSampleCounts(data.counts ?? {});
    } catch {
      // Non-fatal — progress display just stays empty.
    }
  }, []);

  useEffect(() => {
    fetch("/api/signs")
      .then((r) => r.json())
      .then((data: { signs: { classId: string }[] }) => {
        const ids = data.signs.map((s) => s.classId);
        setClasses(ids);
        setSelectedLabel(ids[0] ?? "");
      })
      .catch(() => setMessage("Could not load sign list."));
    const timeoutId = setTimeout(() => refreshCounts(), 0);
    return () => clearTimeout(timeoutId);
  }, [refreshCounts]);

  const stopCamera = useCallback(() => {
    if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    streamRef.current?.getTracks().forEach((t) => t.stop());
    landmarkerRef.current?.close();
    streamRef.current = null;
    landmarkerRef.current = null;
    rafRef.current = null;
  }, []);

  useEffect(() => stopCamera, [stopCamera]);

  const detectionLoopRef = useRef<() => void>(() => {});
  useEffect(() => {
    detectionLoopRef.current = () => {
      const video = videoRef.current;
      const landmarker = landmarkerRef.current;
      if (!video || !landmarker) return;

      if (video.readyState >= 2) {
        const detection = landmarker.detectForVideo(video, performance.now());
        if (recordingRef.current) {
          const hand = detection.hands[0];
          if (hand) recordedRef.current.push(hand.landmarks);
        }
      }
      rafRef.current = requestAnimationFrame(() => detectionLoopRef.current());
    };
  });
  const detectionLoop = useCallback(() => detectionLoopRef.current(), []);

  const startCamera = useCallback(async () => {
    setStage("loading");
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
      landmarkerRef.current = await HandLandmarkerService.create({ numHands: 1 });
      setStage("ready");
      rafRef.current = requestAnimationFrame(detectionLoop);
    } catch {
      setMessage("Could not access the camera. Check permissions and try again.");
      setStage("idle");
    }
  }, [detectionLoop]);

  const beginCountdown = useCallback(() => {
    setMessage(null);
    setStage("countdown");
    setCountdown(COUNTDOWN_SECONDS);
  }, []);

  useEffect(() => {
    if (stage !== "countdown") return;
    if (countdown === 0) {
      const timeoutId = setTimeout(() => {
        recordedRef.current = [];
        recordingRef.current = true;
        setRecordedCount(0);
        setStage("recording");
      }, 0);
      return () => clearTimeout(timeoutId);
    }
    const t = setTimeout(() => setCountdown((c) => c - 1), 1000);
    return () => clearTimeout(t);
  }, [stage, countdown]);

  useEffect(() => {
    if (stage !== "recording") return;
    const t = setInterval(() => {
      setRecordedCount(recordedRef.current.length);
      if (recordedRef.current.length >= RECORD_FRAMES) {
        recordingRef.current = false;
        setLastCapturedFrames(recordedRef.current.length);
        setStage("reviewing");
        clearInterval(t);
      }
    }, 33);
    return () => clearInterval(t);
  }, [stage]);

  const discard = useCallback(() => {
    recordedRef.current = [];
    setStage("ready");
  }, []);

  const save = useCallback(async () => {
    setStage("saving");
    try {
      const res = await fetch("/api/dataset/samples", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          label: selectedLabel,
          languageCode: "ArSL",
          signerSessionId: sessionIdRef.current,
          landmarkSequence: recordedRef.current,
        }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.error ?? "Save failed.");
      }
      setMessage(`Saved ${recordedRef.current.length} frames for "${selectedLabel}".`);
      await refreshCounts();
    } catch (err) {
      setMessage(err instanceof Error ? err.message : "Save failed.");
    } finally {
      recordedRef.current = [];
      setStage("ready");
    }
  }, [selectedLabel, refreshCounts]);

  const quality = lastCapturedFrames >= MIN_VALID_FRAMES ? "good" : "low";

  return (
    <main className="min-h-dvh max-w-lg mx-auto p-4 flex flex-col gap-4">
      <header>
        <h1 className="font-display text-xl font-semibold">Dataset collection</h1>
        <p className="text-sm text-[var(--color-text-dim)]">
          Developer tool — not part of the end-user translator. Captures real
          landmark sequences (no video is saved) for future training.
        </p>
      </header>

      <div className="relative w-full aspect-[3/4] max-h-[45vh] rounded-2xl overflow-hidden bg-[var(--color-surface)] border border-white/10">
        <video ref={videoRef} className="w-full h-full object-cover -scale-x-100" playsInline muted />
        {stage === "idle" && (
          <div className="absolute inset-0 flex items-center justify-center">
            <button
              onClick={startCamera}
              className="font-display font-semibold px-6 py-3 rounded-full bg-[var(--color-accent-active)] text-[#0B1220]"
            >
              Start camera
            </button>
          </div>
        )}
        {stage === "loading" && (
          <div className="absolute inset-0 flex items-center justify-center font-mono text-sm">
            Loading camera + model…
          </div>
        )}
        {stage === "countdown" && (
          <div className="absolute inset-0 flex items-center justify-center">
            <span className="font-display text-6xl">{countdown || "Go!"}</span>
          </div>
        )}
        {stage === "recording" && (
          <div className="absolute top-3 left-3 px-3 py-1 rounded-full bg-[var(--color-accent-error)] text-white text-xs font-mono">
            REC {recordedCount}/{RECORD_FRAMES}
          </div>
        )}
      </div>

      {(stage === "ready" || stage === "countdown" || stage === "recording") && (
        <>
          <label className="flex flex-col gap-1">
            <span className="text-sm text-[var(--color-text-dim)]">Sign to record</span>
            <select
              value={selectedLabel}
              onChange={(e) => setSelectedLabel(e.target.value)}
              disabled={stage !== "ready"}
              className="rounded-lg bg-[var(--color-surface)] border border-white/10 p-3 font-mono"
            >
              {classes.map((c) => (
                <option key={c} value={c}>
                  {c} ({sampleCounts[c] ?? 0} saved)
                </option>
              ))}
            </select>
          </label>
          <button
            onClick={beginCountdown}
            disabled={stage !== "ready" || !selectedLabel}
            className="py-4 rounded-full font-display font-semibold text-lg bg-[var(--color-accent-active)] text-[#0B1220] disabled:opacity-40"
          >
            Record sample
          </button>
        </>
      )}

      {stage === "reviewing" && (
        <div className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4 flex flex-col gap-3">
          <p className="font-mono text-sm">
            Captured {lastCapturedFrames} frames with a hand detected.{" "}
            <span className={quality === "good" ? "text-[var(--color-accent-active)]" : "text-[var(--color-accent-error)]"}>
              Quality: {quality}
            </span>
          </p>
          {quality === "low" && (
            <p className="text-sm text-[var(--color-accent-error)]">
              Hand wasn&apos;t detected for enough of the recording. Consider discarding and retrying.
            </p>
          )}
          <div className="flex gap-3">
            <button onClick={discard} className="flex-1 py-3 rounded-full border border-white/20">
              Discard
            </button>
            <button
              onClick={save}
              className="flex-1 py-3 rounded-full bg-[var(--color-accent-active)] text-[#0B1220] font-medium"
            >
              Save sample
            </button>
          </div>
        </div>
      )}

      {stage === "saving" && <p className="font-mono text-sm">Saving…</p>}

      {message && <p className="font-mono text-sm text-[var(--color-text-dim)]">{message}</p>}
    </main>
  );
}
