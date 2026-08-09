"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { BrowserSpeechRecognitionService } from "@/lib/speech/speechRecognition";

type MicState = "idle" | "listening" | "error" | "unsupported";

export default function SpeechToText() {
  const serviceRef = useRef<BrowserSpeechRecognitionService | null>(null);
  const [state, setState] = useState<MicState>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [finalText, setFinalText] = useState("");
  const [interimText, setInterimText] = useState("");

  useEffect(() => {
    const service = new BrowserSpeechRecognitionService();
    serviceRef.current = service;

    if (!service.isSupported()) {
      const timeoutId = setTimeout(() => setState("unsupported"), 0);
      return () => clearTimeout(timeoutId);
    }

    service.onResult((text, isFinal) => {
      if (isFinal) {
        setFinalText((prev) => (prev ? `${prev} ${text}` : text));
        setInterimText("");
      } else {
        setInterimText(text);
      }
    });
    service.onError((message) => {
      setErrorMessage(message);
      setState("error");
    });
    service.onEnd(() => {
      setState((s) => (s === "listening" ? "idle" : s));
    });

    return () => service.stop();
  }, []);

  const start = useCallback(() => {
    setErrorMessage(null);
    serviceRef.current?.start();
    setState("listening");
  }, []);

  const stop = useCallback(() => {
    serviceRef.current?.stop();
    setState("idle");
  }, []);

  const clear = useCallback(() => {
    setFinalText("");
    setInterimText("");
  }, []);

  if (state === "unsupported") {
    return (
      <div className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-6 text-center">
        <p className="text-[var(--color-accent-error)]">
          Speech recognition isn&apos;t supported in this browser. Try Chrome
          on Android or Safari on iOS.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-6 min-h-[40vh] flex flex-col">
        <div className="flex items-center justify-between mb-4">
          <h2 className="font-display text-sm uppercase tracking-wide text-[var(--color-text-dim)]">
            {state === "listening" ? "Listening…" : "Transcript"}
          </h2>
          <button
            onClick={clear}
            disabled={!finalText && !interimText}
            className="text-sm text-[var(--color-text-dim)] disabled:opacity-40"
          >
            Clear
          </button>
        </div>
        <p className="font-display text-2xl leading-relaxed flex-1" aria-live="polite">
          {finalText}
          {interimText && (
            <span className="text-[var(--color-text-dim)]"> {interimText}</span>
          )}
          {!finalText && !interimText && (
            <span className="text-[var(--color-text-dim)] text-base font-body">
              Tap the mic and start speaking — text appears here live.
            </span>
          )}
        </p>
        {state === "error" && errorMessage && (
          <p className="text-[var(--color-accent-error)] text-sm mt-2">{errorMessage}</p>
        )}
      </div>

      <button
        onClick={state === "listening" ? stop : start}
        className={`py-4 rounded-full font-display font-semibold text-lg active:scale-95 transition-transform ${
          state === "listening"
            ? "bg-[var(--color-accent-error)] text-white"
            : "bg-[var(--color-accent-active)] text-[#0B1220]"
        }`}
      >
        {state === "listening" ? "Stop listening" : "Start speaking"}
      </button>
    </div>
  );
}
