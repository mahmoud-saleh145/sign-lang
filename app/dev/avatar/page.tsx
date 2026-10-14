"use client";

/**
 * CAMERA-FREE, TEXT-FREE AVATAR TEST PAGE.
 *
 * Picks a real KArSL sign straight out of the registry by ID, loads its real
 * temporal landmark sequence, and plays it on the 3D avatar. Nothing here
 * touches speech recognition or text matching, so a failure on this page is
 * unambiguously a data/retargeting/animation problem, and a success here
 * means any VoiceToSign failure is a text-matching problem.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import SignerAvatar from "@/components/avatar/SignerAvatar";
import type { AvatarDiagnostics } from "@/components/avatar/SignerAvatar";
import { SignAnimationController, PLAYBACK_SPEEDS } from "@/lib/avatar/signAnimationController";
import type { AnimationSnapshot, PlaybackSpeed } from "@/lib/avatar/signAnimationController";
import { listKarslSigns, loadKarslSignById, KARSL_SOURCE_FPS } from "@/lib/avatar/karslSign";
import type { AnimatableSign, KarslSignSummary } from "@/lib/avatar/karslSign";

export default function DevAvatarPage() {
    // useState with a lazy initializer, not useRef: this controller is read
    // during render to wire up the transport controls.
    const [controller] = useState(() => new SignAnimationController([]));
    const [signs, setSigns] = useState<KarslSignSummary[]>([]);
    const [selected, setSelected] = useState("");
    const [current, setCurrent] = useState<AnimatableSign | null>(null);
    const [snapshot, setSnapshot] = useState<AnimationSnapshot | null>(null);
    const [diag, setDiag] = useState<AvatarDiagnostics | null>(null);
    const [mirror, setMirror] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [loading, setLoading] = useState(true);

    const load = useCallback(async (signId: string, autoplay: boolean) => {
        setError(null);
        try {
            const sign = await loadKarslSignById(signId);
            if (!sign) {
                setError(`Sign ${signId} is not in the registry.`);
                return;
            }
            setCurrent(sign);
            controller.setSigns([sign]);
            if (autoplay) controller.play();
            setSnapshot(controller.getSnapshot());
        } catch (e) {
            setError(e instanceof Error ? e.message : String(e));
        }
    }, [controller]);

    useEffect(() => {
        listKarslSigns()
            .then(async (s) => {
                setSigns(s);
                setLoading(false);
                if (s.length > 0) {
                    setSelected(s[0].signId);
                    await load(s[0].signId, false);
                }
            })
            .catch((e) => {
                setError(e instanceof Error ? e.message : String(e));
                setLoading(false);
            });
    }, [load]);

    const state = snapshot?.state ?? "idle";

    const summary = useMemo(() => signs.find((s) => s.signId === selected) ?? null, [signs, selected]);

    return (
        <main className="min-h-dvh max-w-3xl mx-auto p-4 flex flex-col gap-4">
            <header>
                <h1 className="font-display text-xl font-semibold">Avatar test — real KArSL sequences</h1>
                <p className="text-xs text-[var(--color-text-dim)] mt-1">
                    Plays a real stored KArSL landmark sequence on the 3D avatar. No camera, no speech, no
                    text matching involved.
                </p>
            </header>

            {error && <p className="text-[var(--color-accent-error)] font-mono text-sm">{error}</p>}

            <SignerAvatar
                controller={controller}
                mirror={mirror}
                onSnapshot={setSnapshot}
                onDiagnostics={setDiag}
            />

            <div className="flex flex-wrap gap-2 items-center font-mono text-xs">
                <label htmlFor="sign" className="text-[var(--color-text-dim)]">
                    sign ({signs.length}):
                </label>
                <select
                    id="sign"
                    value={selected}
                    onChange={(e) => {
                        // Load directly from the event handler rather than from an
                        // effect keyed on `selected`: this is a user action, not
                        // derived state.
                        setSelected(e.target.value);
                        void load(e.target.value, false);
                    }}
                    disabled={loading}
                    className="bg-[var(--color-surface)] border border-white/10 rounded px-2 py-1 max-w-[60%]"
                >
                    {signs.map((s) => (
                        <option key={s.signId} value={s.signId}>
                            {s.signId} — {s.text} ({s.frameCount}f / {(s.durationMs / 1000).toFixed(2)}s{s.twoHanded ? ", 2 hands" : ""})
                        </option>
                    ))}
                </select>
                <label className="flex items-center gap-1 text-[var(--color-text-dim)]">
                    <input type="checkbox" checked={mirror} onChange={(e) => setMirror(e.target.checked)} />
                    mirror
                </label>
            </div>

            {current && (
                <div className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4">
                    <p className="font-display text-3xl" dir="rtl" lang="ar">
                        {current.text}
                    </p>
                    <p className="font-mono text-xs text-[var(--color-text-dim)] mt-1">
                        KArSL {current.signId} · {current.frameCount} real frames ·{" "}
                        {(current.durationMs / 1000).toFixed(2)}s at the source {KARSL_SOURCE_FPS} fps
                    </p>
                </div>
            )}

            <Controls
                controller={controller}
                state={state}
                snapshot={snapshot}
                onChange={() => setSnapshot(controller.getSnapshot())}
            />

            <section className="rounded-2xl bg-black/60 border border-white/10 p-4 font-mono text-xs text-[var(--color-text-dim)] space-y-1">
                <Divider label="Sign" />
                <Row label="registry entries" value={String(signs.length)} />
                <Row label="sign ID" value={selected || "-"} />
                <Row label="arabic text" value={current?.text ?? summary?.text ?? "-"} />
                <Row
                    label="hands in this sign's data"
                    value={
                        current
                            ? current.handDataUnknown
                                ? "unknown (single-hand extraction)"
                                : current.twoHanded
                                    ? "2 (two-handed sign)"
                                    : "1 (one-handed sign)"
                            : "-"
                    }
                    highlight
                />
                <Row label="left hand" value={handStatus(current, diag, "left")} />
                <Row label="right hand" value={handStatus(current, diag, "right")} />
                <Row label="frame count" value={String(current?.frameCount ?? 0)} />
                <Row
                    label="duration"
                    value={
                        current
                            ? `${current.durationMs.toFixed(0)}ms (${current.durationIsReal ? "real" : "display default"})`
                            : "-"
                    }
                />

                <Divider label="Avatar" />
                <Row
                    label="avatar"
                    value={diag ? (diag.avatarLoaded ? `Loaded (${diag.avatarSource})` : "Failed") : "-"}
                    highlight
                />
                <Row
                    label="hand rig"
                    value={diag ? (diag.rigValid ? "Valid" : "Invalid") : "-"}
                    highlight={diag ? !diag.rigValid : false}
                />
                <Row
                    label="bound hand bones (L / R)"
                    value={diag ? `${diag.boundPerSide.left} / ${diag.boundPerSide.right}` : "-"}
                />
                {diag?.avatarLoadError && <Note text={diag.avatarLoadError} />}
                {diag?.rigValidation?.notes.map((n, i) => <Note key={i} text={n} />)}
                {diag?.rigValidation && diag.rigValidation.missing.length > 0 && (
                    <Note text={`unbound: ${diag.rigValidation.missing.slice(0, 8).join(", ")}${diag.rigValidation.missing.length > 8 ? " …" : ""}`} />
                )}

                <Divider label="Animation" />
                <Row
                    label="animation"
                    value={state === "playing" ? "Playing" : state === "finished" ? "Finished" : "Ready"}
                    highlight
                />
                <Row label="frames solved to bone rotations" value={String(diag?.solvedFrameCount ?? 0)} />
                <Row label="degenerate frames (unsolvable palm)" value={String(diag?.degenerateFrameCount ?? 0)} />
                <Row label="current source frame" value={String(diag?.currentFrameIndex ?? 0)} />
                <Row label="arm target reachable" value={String(diag?.armReachable ?? "-")} />
                <Row label="mirror" value={String(mirror)} />
            </section>
        </main>
    );
}

export function Controls({
    controller,
    state,
    snapshot,
    onChange,
}: {
    controller: SignAnimationController;
    state: string;
    snapshot: AnimationSnapshot | null;
    onChange: () => void;
}) {
    const pct = snapshot && snapshot.totalDurationMs > 0
        ? (snapshot.elapsedTotalMs / snapshot.totalDurationMs) * 100
        : 0;

    return (
        <div className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4 flex flex-col gap-3">
            <div className="h-2 rounded-full bg-white/10 overflow-hidden">
                <div
                    className="h-full bg-[var(--color-accent-active)] transition-[width] duration-100"
                    style={{ width: `${pct}%` }}
                />
            </div>
            <div className="flex justify-between font-mono text-xs text-[var(--color-text-dim)]">
                <span>{((snapshot?.elapsedTotalMs ?? 0) / 1000).toFixed(2)}s</span>
                <span>
                    sign {(snapshot?.signIndex ?? 0) + 1} / {snapshot?.signCount ?? 0}
                </span>
                <span>{((snapshot?.totalDurationMs ?? 0) / 1000).toFixed(2)}s</span>
            </div>

            <div className="flex gap-2">
                <button
                    onClick={() => {
                        if (state === "playing") controller.pause();
                        else if (state === "paused") controller.resume();
                        else controller.play();
                        onChange();
                    }}
                    className="flex-1 py-3 rounded-full bg-[var(--color-accent-active)] text-[#0B1220] font-semibold"
                >
                    {state === "playing" ? "⏸ Pause" : state === "paused" ? "▶ Resume" : "▶ Play"}
                </button>
                <button
                    onClick={() => {
                        controller.replay();
                        onChange();
                    }}
                    className="flex-1 py-3 rounded-full border border-white/20 font-medium"
                >
                    ↻ Replay
                </button>
                <button
                    onClick={() => {
                        controller.stop();
                        onChange();
                    }}
                    className="px-5 py-3 rounded-full border border-white/20 font-medium"
                >
                    ■ Stop
                </button>
            </div>

            <div className="flex items-center justify-center gap-2 flex-wrap">
                <span className="font-mono text-xs text-[var(--color-text-dim)]">speed:</span>
                {PLAYBACK_SPEEDS.map((s) => (
                    <button
                        key={s}
                        onClick={() => {
                            controller.setSpeed(s as PlaybackSpeed);
                            onChange();
                        }}
                        className={`px-3 py-1 rounded-full text-xs font-mono ${snapshot?.speed === s
                            ? "bg-[var(--color-accent-active)] text-[#0B1220]"
                            : "border border-white/20 text-[var(--color-text-dim)]"
                            }`}
                    >
                        {s}x
                    </button>
                ))}
            </div>
        </div>
    );
}

function handStatus(
    current: AnimatableSign | null,
    diag: AvatarDiagnostics | null,
    side: "left" | "right"
): string {
    if (!current) return "-";
    if (current.handDataUnknown) return "no two-hand data in this entry";
    const track = current.tracks?.[side];
    if (!track || track.detectedFrames === 0) return "Not detected";
    return `Detected (${track.detectedFrames}/${current.frameCount} frames)`;
}

function Note({ text }: { text: string }) {
    return <p className="text-[10px] text-[var(--color-accent-error)] leading-snug">{text}</p>;
}

function Divider({ label }: { label: string }) {
    return (
        <div className="pt-2 mt-2 border-t border-white/10 text-[var(--color-accent-active)] uppercase tracking-wide">
            {label}
        </div>
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
