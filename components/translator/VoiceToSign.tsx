"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { BrowserSpeechRecognitionService } from "@/lib/speech/speechRecognition";
import { translateToSignSequence } from "@/lib/sign-language/signTranslator";
import type { SignSequence, SignStep } from "@/lib/sign-language/types";
import { PlaybackController, DEFAULT_PLAYBACK_CONFIG } from "@/lib/playback/playbackController";
import type { PlaybackSnapshot, PlaybackSpeed } from "@/lib/playback/playbackController";
import HandSignerAvatar from "@/components/avatar/HandSignerAvatar";

type MicState = "idle" | "listening" | "translating" | "error" | "unsupported";

const SPEEDS: PlaybackSpeed[] = [0.5, 0.75, 1, 1.25];

function flattenSteps(sequence: SignSequence | null): SignStep[] {
    if (!sequence) return [];
    return sequence.words.flatMap((w) => w.steps);
}

export default function VoiceToSign() {
    const serviceRef = useRef<BrowserSpeechRecognitionService | null>(null);
    const controllerRef = useRef<PlaybackController>(new PlaybackController(0, DEFAULT_PLAYBACK_CONFIG));
    const rafRef = useRef<number | null>(null);
    const lastTsRef = useRef<number | null>(null);

    const [state, setState] = useState<MicState>("idle");
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [heardText, setHeardText] = useState("");
    const [sequence, setSequence] = useState<SignSequence | null>(null);
    const [playback, setPlayback] = useState<PlaybackSnapshot>({
        state: "idle",
        currentStepIndex: 0,
        totalSteps: 0,
        speed: 1,
        progressWithinStep: 0,
    });

    const steps = flattenSteps(sequence);

    const runTranslation = useCallback(async (text: string) => {
        setState("translating");
        try {
            const result = await translateToSignSequence(text);
            setSequence(result);
            controllerRef.current.setSequenceLength(flattenSteps(result).length);
            setPlayback(controllerRef.current.getSnapshot());
            setState("idle");
        } catch (err) {
            setErrorMessage(err instanceof Error ? err.message : "Translation failed.");
            setState("error");
        }
    }, []);

    useEffect(() => {
        const service = new BrowserSpeechRecognitionService(); // defaults to ar-EG
        serviceRef.current = service;
        if (!service.isSupported()) {
            const t = setTimeout(() => setState("unsupported"), 0);
            return () => clearTimeout(t);
        }
        service.onResult((text, isFinal) => {
            setHeardText(text);
            if (isFinal && text.trim().length > 0) {
                void runTranslation(text);
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
    }, [runTranslation]);

    const loopRef = useRef<(ts: number) => void>(() => { });
    useEffect(() => {
        loopRef.current = (ts: number) => {
            if (lastTsRef.current !== null) {
                const delta = ts - lastTsRef.current;
                const snapshot = controllerRef.current.tick(delta);
                setPlayback(snapshot);
            }
            lastTsRef.current = ts;
            rafRef.current = requestAnimationFrame((t) => loopRef.current(t));
        };
    });
    useEffect(() => {
        rafRef.current = requestAnimationFrame((t) => loopRef.current(t));
        return () => {
            if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
        };
    }, []);

    const startListening = useCallback(() => {
        setErrorMessage(null);
        setHeardText("");
        serviceRef.current?.start();
        setState("listening");
    }, []);

    const stopListening = useCallback(() => {
        serviceRef.current?.stop();
        setState("idle");
    }, []);

    const play = useCallback(() => {
        controllerRef.current.play();
        setPlayback(controllerRef.current.getSnapshot());
    }, []);
    const pause = useCallback(() => {
        controllerRef.current.pause();
        setPlayback(controllerRef.current.getSnapshot());
    }, []);
    const resume = useCallback(() => {
        controllerRef.current.resume();
        setPlayback(controllerRef.current.getSnapshot());
    }, []);
    const replay = useCallback(() => {
        controllerRef.current.replay();
        setPlayback(controllerRef.current.getSnapshot());
    }, []);
    const setSpeed = useCallback((speed: PlaybackSpeed) => {
        controllerRef.current.setSpeed(speed);
        setPlayback(controllerRef.current.getSnapshot());
    }, []);

    if (state === "unsupported") {
        return (
            <div className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-6 text-center">
                <p className="text-[var(--color-accent-error)]">
                    Speech recognition isn&apos;t supported in this browser.
                </p>
            </div>
        );
    }

    const currentStep = steps[playback.currentStepIndex] ?? null;

    return (
        <div className="flex flex-col gap-4">
            <HandSignerAvatar
                frames={currentStep?.landmarkFrames ?? []}
                progress={playback.progressWithinStep}
                caption={currentStep?.text}
            />

            {currentStep && (
                <p className="text-center text-xs font-mono text-[var(--color-text-dim)]">
                    {currentStep.source === "dedicated" &&
                        `Dedicated sign (${currentStep.datasetName ?? "dataset"})`}
                    {currentStep.source === "spelled-letter" && "Spelled letter (fallback \u2014 no dedicated word sign)"}
                    {currentStep.source === "unsupported" && "No sign data available for this character"}
                </p>
            )}

            <div className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4">
                <h2 className="font-display text-sm uppercase tracking-wide text-[var(--color-text-dim)] mb-2">
                    {state === "listening" ? "Listening\u2026" : "You said"}
                </h2>
                <p className="font-display text-2xl min-h-[2rem]" dir="rtl" lang="ar">
                    {heardText || (
                        <span dir="ltr" className="text-[var(--color-text-dim)] text-base font-body">
                            Tap the mic and speak in Arabic.
                        </span>
                    )}
                </p>
                {state === "error" && errorMessage && (
                    <p className="text-[var(--color-accent-error)] text-sm mt-2">{errorMessage}</p>
                )}
            </div>

            <button
                onClick={state === "listening" ? stopListening : startListening}
                className={`py-4 rounded-full font-display font-semibold text-lg active:scale-95 transition-transform ${state === "listening"
                    ? "bg-[var(--color-accent-error)] text-white"
                    : "bg-[var(--color-accent-active)] text-[#0B1220]"
                    }`}
            >
                {state === "listening" ? "Stop listening" : "\ud83c\udf99 Start speaking"}
            </button>

            {steps.length > 0 && (
                <div className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4 flex flex-col gap-3">
                    <div className="flex gap-3">
                        <button
                            onClick={playback.state === "playing" ? pause : playback.state === "paused" ? resume : play}
                            className="flex-1 py-3 rounded-full border border-white/20 font-medium"
                        >
                            {playback.state === "playing" ? "\u23f8 Pause" : playback.state === "paused" ? "\u25b6 Resume" : "\u25b6 Play"}
                        </button>
                        <button onClick={replay} className="flex-1 py-3 rounded-full border border-white/20 font-medium">
                            {"\u21bb Replay"}
                        </button>
                    </div>
                    <div className="flex items-center justify-center gap-2">
                        <span className="text-xs font-mono text-[var(--color-text-dim)]">Speed:</span>
                        {SPEEDS.map((s) => (
                            <button
                                key={s}
                                onClick={() => setSpeed(s)}
                                className={`px-3 py-1 rounded-full text-xs font-mono ${playback.speed === s
                                    ? "bg-[var(--color-accent-active)] text-[#0B1220]"
                                    : "border border-white/20 text-[var(--color-text-dim)]"
                                    }`}
                            >
                                {s}x
                            </button>
                        ))}
                    </div>
                    <p className="text-center font-mono text-xs text-[var(--color-text-dim)]">
                        {playback.currentStepIndex + 1} / {playback.totalSteps}
                    </p>
                </div>
            )}
        </div>
    );
}