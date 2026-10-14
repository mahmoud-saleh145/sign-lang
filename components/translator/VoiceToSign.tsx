"use client";

/**
 * VOICE / TEXT -> ARABIC SIGN LANGUAGE -> 3D AVATAR.
 *
 * ONE PIPELINE FOR BOTH INPUTS. Speech and typing both call submit(), which
 * calls translateToSignSequence(). Speech recognition only produces the
 * string that typing would have produced -- there is no separate translation
 * path for voice.
 *
 *   Arabic speech OR typed Arabic
 *     -> normalizeArabicText()                (textNormalization.ts)
 *     -> longest-phrase-first registry match  (signTranslator.ts)
 *     -> AnimatableSign per step              (karslSign.ts)
 *     -> real per-sign durations              (signAnimationController.ts)
 *     -> bone rotations from real landmarks   (retarget.ts)
 *     -> 3D avatar                            (SignerAvatar.tsx)
 *
 * A multi-word KArSL label such as "هيكل عظمي" is matched and played as ONE
 * sign; a registered phrase is never split into its words.
 *
 * WHAT THIS DOES NOT DO, and does not claim to: KArSL is isolated word data.
 * Playing several signs in a row is not Arabic Sign Language grammar. There
 * are no facial expressions, no non-manual markers and no sentence-level
 * reordering, because the dataset contains none of those signals.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { BrowserSpeechRecognitionService } from "@/lib/speech/speechRecognition";
import { translateToSignSequence } from "@/lib/sign-language/signTranslator";
import type { SignSequence } from "@/lib/sign-language/types";
import SignerAvatar from "@/components/avatar/SignerAvatar";
import type { AvatarDiagnostics } from "@/components/avatar/SignerAvatar";
import { SignAnimationController, PLAYBACK_SPEEDS } from "@/lib/avatar/signAnimationController";
import type { AnimationSnapshot, PlaybackSpeed } from "@/lib/avatar/signAnimationController";
import { fromSignStep, KARSL_SOURCE_FPS } from "@/lib/avatar/karslSign";
import type { AnimatableSign } from "@/lib/avatar/karslSign";
import { normalizeArabicText } from "@/lib/sign-language/textNormalization";

type MicState = "idle" | "listening" | "translating" | "error" | "unsupported";
type InputMode = "voice" | "text";

const UNAVAILABLE_AR = "هذا اللفظ غير متوفر حاليًا في بيانات KArSL";

function toAnimatable(sequence: SignSequence | null): AnimatableSign[] {
    if (!sequence) return [];
    return sequence.words.flatMap((w) => w.steps).map(fromSignStep);
}

export default function VoiceToSign() {
    const serviceRef = useRef<BrowserSpeechRecognitionService | null>(null);
    // useState with a lazy initializer, not useRef: the controller is a stable
    // object read during render to wire up the transport controls.
    const [controller] = useState(() => new SignAnimationController([]));

    const [state, setState] = useState<MicState>("idle");
    const [inputMode, setInputMode] = useState<InputMode>("text");
    const [errorMessage, setErrorMessage] = useState<string | null>(null);
    const [inputText, setInputText] = useState("");
    const [submittedText, setSubmittedText] = useState("");
    const [sequence, setSequence] = useState<SignSequence | null>(null);
    const [snapshot, setSnapshot] = useState<AnimationSnapshot | null>(null);
    const [diag, setDiag] = useState<AvatarDiagnostics | null>(null);
    const [showDebug, setShowDebug] = useState(false);
    const [mirror, setMirror] = useState(false);

    const animatable = useMemo(() => toAnimatable(sequence), [sequence]);

    const submit = useCallback(
        async (text: string, autoplay: boolean) => {
            const trimmed = text.trim();
            if (!trimmed) return;
            setSubmittedText(trimmed);
            setState("translating");
            setErrorMessage(null);
            try {
                const result = await translateToSignSequence(trimmed);
                setSequence(result);
                const signs = toAnimatable(result);
                controller.setSigns(signs);
                if (autoplay && signs.some((s) => s.landmarkSequence.length > 0)) controller.play();
                setSnapshot(controller.getSnapshot());
                setState("idle");
            } catch (err) {
                setErrorMessage(err instanceof Error ? err.message : "Translation failed.");
                setState("error");
            }
        },
        [controller]
    );

    useEffect(() => {
        const service = new BrowserSpeechRecognitionService(); // defaults to ar-EG
        serviceRef.current = service;
        if (!service.isSupported()) {
            const t = setTimeout(() => setState("unsupported"), 0);
            return () => clearTimeout(t);
        }
        service.onResult((text, isFinal) => {
            setInputText(text);
            if (isFinal && text.trim().length > 0) void submit(text, true);
        });
        service.onError((message) => {
            setErrorMessage(message);
            setState("error");
        });
        service.onEnd(() => setState((s) => (s === "listening" ? "idle" : s)));
        return () => service.stop();
    }, [submit]);

    const currentSign = animatable[snapshot?.signIndex ?? 0] ?? null;
    const hasAnyKarsl = animatable.some((s) => s.source === "karsl");
    const allUnsupported = animatable.length > 0 && animatable.every((s) => s.source === "unsupported");
    const speechSupported = state !== "unsupported";
    const playing = snapshot?.state === "playing";

    const startListening = useCallback(() => {
        setErrorMessage(null);
        setInputText("");
        serviceRef.current?.start();
        setState("listening");
    }, []);

    const stopListening = useCallback(() => {
        serviceRef.current?.stop();
        setState("idle");
    }, []);

    const pct =
        snapshot && snapshot.totalDurationMs > 0
            ? (snapshot.elapsedTotalMs / snapshot.totalDurationMs) * 100
            : 0;

    return (
        <div className="flex flex-col gap-4">
            <header className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4">
                <h1 className="font-display text-2xl font-semibold">Voice to Sign</h1>
                <p className="text-xs text-[var(--color-text-dim)] mt-1">
                    Convert Arabic speech or text to KArSL, performed by a 3D avatar.
                </p>
            </header>

            <div className="grid lg:grid-cols-[minmax(0,1.35fr)_minmax(0,1fr)] gap-4 items-start">
                {/* ---------- Avatar stage ---------- */}
                <div className="relative">
                    <SignerAvatar
                        controller={controller}
                        mirror={mirror}
                        onSnapshot={setSnapshot}
                        onDiagnostics={setDiag}
                        className="w-full aspect-[3/4] sm:aspect-[4/5] lg:aspect-[3/4] rounded-2xl overflow-hidden bg-[var(--color-surface)] border border-white/10"
                    />
                    <div className="absolute top-3 left-3 flex items-center gap-2 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur">
                        <span
                            className={`w-2 h-2 rounded-full ${diag?.avatarLoaded ? "bg-[var(--color-accent-active)]" : "bg-[var(--color-accent-error)]"
                                }`}
                        />
                        <span className="font-mono text-xs">
                            {diag?.avatarLoaded ? "Avatar Ready" : "Loading avatar…"}
                        </span>
                    </div>
                    <div className="absolute bottom-3 left-3 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur font-mono text-xs text-[var(--color-text-dim)]">
                        {diag?.avatarSource === "glb" ? "3D Avatar (rigged model)" : "3D Avatar (built-in signer)"}
                    </div>
                    <button
                        onClick={() => setMirror((m) => !m)}
                        className="absolute top-3 right-3 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur font-mono text-xs"
                        aria-pressed={mirror}
                        title="Flip the hand orientation. The registry does not record which hand the source signer used."
                    >
                        mirror {mirror ? "on" : "off"}
                    </button>
                </div>

                {/* ---------- Right column ---------- */}
                <div className="flex flex-col gap-4">
                    <section className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4">
                        <h2 className="font-display text-lg mb-3" dir="rtl" lang="ar">
                            إدخال الصوت أو النص
                        </h2>

                        <div role="tablist" className="flex gap-1 p-1 rounded-full bg-black/30 mb-3">
                            <Tab active={inputMode === "voice"} onClick={() => setInputMode("voice")} label="🎙 Voice Input" />
                            <Tab active={inputMode === "text"} onClick={() => setInputMode("text")} label="⌨ Text Input" />
                        </div>

                        {inputMode === "voice" ? (
                            <div className="flex flex-col items-center gap-3 py-4">
                                <button
                                    onClick={state === "listening" ? stopListening : startListening}
                                    disabled={!speechSupported}
                                    className={`w-20 h-20 rounded-full flex items-center justify-center text-3xl transition-transform active:scale-95 disabled:opacity-40 ${state === "listening"
                                        ? "bg-[var(--color-accent-error)] text-white animate-pulse"
                                        : "border-2 border-[var(--color-accent-active)] text-[var(--color-accent-active)]"
                                        }`}
                                >
                                    🎙
                                </button>
                                <p className="font-mono text-xs text-[var(--color-text-dim)]">
                                    {!speechSupported
                                        ? "Speech recognition isn't supported here — use Text Input."
                                        : state === "listening"
                                            ? "Listening… speak in Arabic"
                                            : "Tap to speak…"}
                                </p>
                                {inputText && (
                                    <p className="font-display text-xl" dir="rtl" lang="ar">
                                        {inputText}
                                    </p>
                                )}
                            </div>
                        ) : (
                            <div className="flex gap-2">
                                <input
                                    id="ar-input"
                                    aria-label="Arabic text input"
                                    value={inputText}
                                    onChange={(e) => setInputText(e.target.value)}
                                    onKeyDown={(e) => {
                                        if (e.key === "Enter") void submit(inputText, true);
                                    }}
                                    dir="rtl"
                                    lang="ar"
                                    placeholder="اكتب كلمة أو عبارة"
                                    className="flex-1 min-w-0 bg-[var(--color-bg)] border border-white/15 rounded-xl px-3 py-3 font-display text-xl"
                                />
                                <button
                                    onClick={() => void submit(inputText, true)}
                                    disabled={!inputText.trim() || state === "translating"}
                                    className="px-5 rounded-xl bg-[var(--color-accent-active)] text-[#0B1220] font-semibold disabled:opacity-40"
                                >
                                    Sign
                                </button>
                            </div>
                        )}

                        {state === "error" && errorMessage && (
                            <p className="text-[var(--color-accent-error)] text-sm mt-3">{errorMessage}</p>
                        )}
                    </section>

                    {/* ---------- Translation result ---------- */}
                    <section className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4 space-y-3">
                        <h2 className="font-display text-lg">Translation Result</h2>

                        <Field label="Detected text">
                            <span dir="rtl" lang="ar" className="font-display text-xl">
                                {submittedText || "—"}
                            </span>
                        </Field>

                        <Field
                            label={currentSign?.source === "karsl" ? "Matched KArSL sign" : "Current step"}
                            accent={currentSign?.source === "karsl"}
                        >
                            <span dir="rtl" lang="ar" className="font-display text-xl">
                                {currentSign?.text ?? "—"}
                            </span>
                        </Field>

                        <Field label="Source">
                            <span
                                className={`font-mono text-xs ${currentSign?.source === "karsl"
                                    ? "text-[var(--color-accent-active)]"
                                    : "text-[var(--color-accent-error)]"
                                    }`}
                            >
                                {!currentSign
                                    ? "—"
                                    : currentSign.source === "karsl"
                                        ? `KArSL dedicated sign (${currentSign.datasetName ?? "KArSL-502"})`
                                        : currentSign.source === "spelled-letter"
                                            ? "Letter fallback — NOT a real word sign"
                                            : "No sign data for this character"}
                            </span>
                        </Field>

                        {submittedText && !hasAnyKarsl && (
                            <div className="pt-2 border-t border-white/10">
                                <p dir="rtl" lang="ar" className="text-[var(--color-accent-error)]">
                                    {UNAVAILABLE_AR}
                                </p>
                                <p className="font-mono text-xs text-[var(--color-text-dim)] mt-1">
                                    {allUnsupported
                                        ? "No sign data at all for this input."
                                        : "Fallback: spelling by letters. This is alphabet spelling, not a real word sign."}
                                </p>
                            </div>
                        )}
                    </section>

                    {/* ---------- Animation controls ---------- */}
                    {animatable.length > 0 && (
                        <section className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4 flex flex-col gap-3">
                            <h2 className="font-display text-lg">Animation Controls</h2>

                            <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                                <div className="h-full bg-[var(--color-accent-active)]" style={{ width: `${pct}%` }} />
                            </div>
                            <div className="flex justify-between font-mono text-xs text-[var(--color-text-dim)]">
                                <span>{((snapshot?.elapsedTotalMs ?? 0) / 1000).toFixed(1)}s</span>
                                <span>
                                    {(snapshot?.signIndex ?? 0) + 1} / {snapshot?.signCount ?? 0}
                                </span>
                                <span>{((snapshot?.totalDurationMs ?? 0) / 1000).toFixed(1)}s</span>
                            </div>

                            <div className="flex gap-2">
                                <button
                                    onClick={() => {
                                        if (playing) controller.pause();
                                        else if (snapshot?.state === "paused") controller.resume();
                                        else controller.play();
                                        setSnapshot(controller.getSnapshot());
                                    }}
                                    className="flex-1 py-3 rounded-xl bg-[var(--color-accent-active)] text-[#0B1220] font-semibold"
                                >
                                    {playing ? "⏸ Pause" : snapshot?.state === "paused" ? "▶ Resume" : "▶ Play"}
                                </button>
                                <button
                                    onClick={() => {
                                        controller.replay();
                                        setSnapshot(controller.getSnapshot());
                                    }}
                                    className="flex-1 py-3 rounded-xl border border-white/20 font-medium"
                                >
                                    ↻ Replay
                                </button>
                            </div>

                            <div className="flex items-center gap-3">
                                <span className="font-mono text-xs text-[var(--color-text-dim)]">Speed</span>
                                <input
                                    type="range"
                                    min={0}
                                    max={PLAYBACK_SPEEDS.length - 1}
                                    step={1}
                                    value={Math.max(0, PLAYBACK_SPEEDS.indexOf((snapshot?.speed ?? 1) as PlaybackSpeed))}
                                    onChange={(e) => {
                                        controller.setSpeed(PLAYBACK_SPEEDS[Number(e.target.value)] as PlaybackSpeed);
                                        setSnapshot(controller.getSnapshot());
                                    }}
                                    className="flex-1 accent-[var(--color-accent-active)]"
                                    aria-label="Playback speed"
                                />
                                <span className="font-mono text-xs w-10 text-right">{snapshot?.speed ?? 1}x</span>
                            </div>

                            <ol className="flex flex-wrap gap-1 justify-end" dir="rtl">
                                {animatable.map((s, i) => (
                                    <li
                                        key={`${s.text}-${i}`}
                                        className={`px-2 py-1 rounded-lg font-display text-lg ${i === snapshot?.signIndex
                                            ? "bg-[var(--color-accent-active)] text-[#0B1220]"
                                            : "bg-white/5"
                                            } ${s.source === "karsl" ? "" : "opacity-60"}`}
                                        title={s.source === "karsl" ? `KArSL sign, ${s.frameCount} frames` : s.source}
                                    >
                                        {s.text}
                                    </li>
                                ))}
                            </ol>
                        </section>
                    )}

                    {/* ---------- System status ---------- */}
                    <section className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4 space-y-1.5">
                        <h2 className="font-display text-lg mb-2">System Status</h2>
                        <Status label="KArSL registry" ok value={`Loaded`} />
                        <Status
                            label="Avatar model"
                            ok={Boolean(diag?.avatarLoaded)}
                            value={
                                diag?.avatarLoaded
                                    ? diag.avatarSource === "glb"
                                        ? "Loaded (rigged GLB)"
                                        : "Loaded (built-in)"
                                    : "Loading"
                            }
                        />
                        <Status
                            label="Hand rig"
                            ok={Boolean(diag?.rigValid)}
                            value={diag ? (diag.rigValid ? "Valid" : "Invalid") : "—"}
                        />
                        <Status
                            label="Animation"
                            ok
                            value={playing ? "Playing" : snapshot?.state === "finished" ? "Finished" : "Ready"}
                        />
                        <Status
                            label="Current frame"
                            ok
                            value={
                                currentSign ? `${(diag?.currentFrameIndex ?? 0) + 1} / ${currentSign.frameCount}` : "—"
                            }
                        />
                    </section>
                </div>
            </div>

            {/* ---------- KArSL sign information ---------- */}
            {currentSign && (
                <section className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4">
                    <h2 className="font-display text-lg mb-3">KArSL Sign Information</h2>
                    <dl className="grid sm:grid-cols-2 gap-x-8 gap-y-2 font-mono text-xs">
                        <Info label="Sign ID" value={currentSign.signId ? `#${currentSign.signId}` : "—"} />
                        <Info label="Arabic text" value={currentSign.text} rtl />
                        <Info
                            label="Type"
                            value={
                                currentSign.source === "karsl"
                                    ? currentSign.text.includes(" ")
                                        ? "KArSL phrase"
                                        : "KArSL word"
                                    : currentSign.source === "spelled-letter"
                                        ? "Letter fallback"
                                        : "Unsupported"
                            }
                        />
                        <Info
                            label="Hands in source data"
                            value={
                                currentSign.handDataUnknown
                                    ? "unknown (single-hand extraction)"
                                    : currentSign.twoHanded
                                        ? "2 — two-handed sign"
                                        : "1 — one-handed sign"
                            }
                        />
                        <Info label="Sequence length" value={`${currentSign.frameCount} frames`} />
                        <Info
                            label="Duration"
                            value={
                                currentSign.durationIsReal
                                    ? `${(currentSign.durationMs / 1000).toFixed(2)}s at ${KARSL_SOURCE_FPS} fps (real)`
                                    : `${(currentSign.durationMs / 1000).toFixed(2)}s (display default)`
                            }
                        />
                    </dl>
                </section>
            )}

            <div className="flex items-center justify-between font-mono text-xs">
                <button
                    onClick={() => setShowDebug((v) => !v)}
                    className="px-3 py-1 rounded bg-black/40 text-[var(--color-text-dim)]"
                    aria-pressed={showDebug}
                >
                    debug: {showDebug ? "on" : "off"}
                </button>
            </div>

            {showDebug && (
                <VoiceToSignDebugPanel
                    inputText={submittedText}
                    sequence={sequence}
                    currentSign={currentSign}
                    snapshot={snapshot}
                    diag={diag}
                    mirror={mirror}
                />
            )}
        </div>
    );
}

function Tab({ active, onClick, label }: { active: boolean; onClick: () => void; label: string }) {
    return (
        <button
            role="tab"
            aria-selected={active}
            onClick={onClick}
            className={`flex-1 py-2 rounded-full text-sm font-medium transition-colors ${active ? "bg-[var(--color-accent-active)] text-[#0B1220]" : "text-[var(--color-text-dim)]"
                }`}
        >
            {label}
        </button>
    );
}

function Field({ label, children, accent }: { label: string; children: React.ReactNode; accent?: boolean }) {
    return (
        <div
            className={`rounded-xl px-3 py-2 border ${accent ? "border-[var(--color-accent-active)]/40 bg-[var(--color-accent-active)]/5" : "border-white/10 bg-black/20"
                }`}
        >
            <div className="font-mono text-[10px] uppercase tracking-wide text-[var(--color-text-dim)]">{label}</div>
            <div className="mt-0.5">{children}</div>
        </div>
    );
}

function Status({ label, value, ok }: { label: string; value: string; ok: boolean }) {
    return (
        <div className="flex justify-between items-center gap-3 font-mono text-xs">
            <span className="text-[var(--color-text-dim)]">{label}</span>
            <span className={ok ? "text-[var(--color-accent-active)]" : "text-[var(--color-accent-error)]"}>{value}</span>
        </div>
    );
}

function Info({ label, value, rtl }: { label: string; value: string; rtl?: boolean }) {
    return (
        <div className="flex justify-between gap-3 border-b border-white/5 py-1">
            <dt className="text-[var(--color-text-dim)]">{label}</dt>
            <dd className="text-[var(--color-text)]" dir={rtl ? "rtl" : undefined} lang={rtl ? "ar" : undefined}>
                {value}
            </dd>
        </div>
    );
}

function VoiceToSignDebugPanel({
    inputText,
    sequence,
    currentSign,
    snapshot,
    diag,
    mirror,
}: {
    inputText: string;
    sequence: SignSequence | null;
    currentSign: AnimatableSign | null;
    snapshot: AnimationSnapshot | null;
    diag: AvatarDiagnostics | null;
    mirror: boolean;
}) {
    const matchedType = currentSign
        ? currentSign.source === "karsl"
            ? currentSign.text.includes(" ")
                ? "KArSL phrase"
                : "KArSL word"
            : currentSign.source === "spelled-letter"
                ? "letter fallback"
                : "unsupported"
        : "-";

    return (
        <section className="rounded-2xl bg-black/60 border border-white/10 p-4 font-mono text-xs text-[var(--color-text-dim)] space-y-1">
            <Divider label="Text matching" />
            <Row label="input text" value={inputText || "-"} />
            <Row
                label="normalized text"
                value={sequence?.normalizedText ?? (inputText ? normalizeArabicText(inputText).text : "-")}
            />
            <Row label="normalization changes" value={String(sequence?.normalizationNotes.length ?? 0)} />
            <Row label="matched type" value={matchedType} highlight />
            <Row label="step count" value={String(snapshot?.signCount ?? 0)} />

            <Divider label="KArSL sequence" />
            <Row label="KArSL sign ID" value={currentSign?.signId ?? "-"} />
            <Row label="sequence frames" value={String(currentSign?.frameCount ?? 0)} />
            <Row
                label="sequence duration"
                value={
                    currentSign
                        ? `${currentSign.durationMs.toFixed(0)}ms (${currentSign.durationIsReal ? "real" : "display default"})`
                        : "-"
                }
            />
            <Row
                label="two-hand data"
                value={
                    currentSign
                        ? currentSign.handDataUnknown
                            ? "absent (single-hand extraction)"
                            : currentSign.twoHanded
                                ? "yes"
                                : "present, one-handed sign"
                        : "-"
                }
            />
            <Row label="left hand detected" value={String(diag?.leftHandDetected ?? "-")} />
            <Row label="right hand detected" value={String(diag?.rightHandDetected ?? "-")} />

            <Divider label="Avatar" />
            <Row label="avatar source" value={diag?.avatarSource ?? "-"} highlight />
            <Row label="avatar loaded" value={String(diag?.avatarLoaded ?? false)} />
            <Row label="hand rig valid" value={String(diag?.rigValid ?? "-")} />
            <Row
                label="bound hand bones (L / R)"
                value={diag ? `${diag.boundPerSide.left} / ${diag.boundPerSide.right}` : "-"}
            />
            <Row label="frames solved to rotations" value={String(diag?.solvedFrameCount ?? 0)} />
            <Row label="degenerate frames" value={String(diag?.degenerateFrameCount ?? 0)} />
            <Row label="arm target reachable" value={String(diag?.armReachable ?? "-")} />
            <Row label="mirror used" value={String(mirror)} />
            {diag?.avatarLoadError && (
                <p className="text-[10px] text-[var(--color-accent-error)]">{diag.avatarLoadError}</p>
            )}
            {diag?.rigValidation?.notes.map((n, i) => (
                <p key={i} className="text-[10px] text-[var(--color-accent-error)]">
                    {n}
                </p>
            ))}

            <Divider label="Animation" />
            <Row label="animation playing" value={String(snapshot?.state === "playing")} />
            <Row label="animation state" value={snapshot?.state ?? "-"} />
            <Row label="current source frame" value={String(diag?.currentFrameIndex ?? 0)} />
            <Row label="progress within sign" value={snapshot ? snapshot.progressWithinSign.toFixed(3) : "-"} />
            <Row label="speed" value={snapshot ? `${snapshot.speed}x` : "-"} />

            {sequence && sequence.normalizationNotes.length > 0 && (
                <>
                    <Divider label="Normalization notes" />
                    {sequence.normalizationNotes.map((n, i) => (
                        <p key={i} className="text-[10px]">
                            {n.original} → {n.replacement}: {n.reason}
                        </p>
                    ))}
                </>
            )}
        </section>
    );
}

function Row({ label, value, highlight }: { label: string; value: string; highlight?: boolean }) {
    return (
        <div className="flex justify-between gap-4">
            <span>{label}</span>
            <span className={highlight ? "text-[var(--color-accent-active)] font-semibold" : "text-[var(--color-text)]"}>
                {value}
            </span>
        </div>
    );
}

function Divider({ label }: { label: string }) {
    return (
        <div className="pt-2 mt-2 border-t border-white/10 text-[var(--color-accent-active)] uppercase tracking-wide">
            {label}
        </div>
    );
}
