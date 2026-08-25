"use client";

import { useState } from "react";
import CameraView from "@/components/camera/CameraView";
import SpeechToText from "@/components/speech/SpeechToText";
import VoiceToSign from "@/components/translator/VoiceToSign";

type Mode = "sign" | "speech" | "voice-to-sign";

export default function TranslatorPage() {
  const [mode, setMode] = useState<Mode>("sign");

  return (
    <main className="min-h-dvh flex flex-col max-w-lg mx-auto p-4 gap-4">
      <header className="flex items-center justify-between pt-2">
        <h1 className="font-display text-xl font-semibold">Translator</h1>
      </header>

      <div
        role="tablist"
        aria-label="Translation direction"
        className="flex flex-wrap rounded-full bg-[var(--color-surface)] p-1 border border-white/10"
      >
        <ModeTab label={"Sign \u2192 Text"} active={mode === "sign"} onClick={() => setMode("sign")} />
        <ModeTab label={"Speech \u2192 Text"} active={mode === "speech"} onClick={() => setMode("speech")} />
        <ModeTab label={"Voice \u2192 Sign"} active={mode === "voice-to-sign"} onClick={() => setMode("voice-to-sign")} />
      </div>

      {mode === "sign" && <CameraView />}
      {mode === "speech" && <SpeechToText />}
      {mode === "voice-to-sign" && <VoiceToSign />}
    </main>
  );
}

function ModeTab({
  label,
  active,
  onClick,
}: {
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      role="tab"
      aria-selected={active}
      onClick={onClick}
      className={`flex-1 min-w-[30%] py-3 rounded-full font-medium text-sm transition-colors ${active ? "bg-[var(--color-accent-active)] text-[#0B1220]" : "text-[var(--color-text-dim)]"
        }`}
    >
      {label}
    </button>
  );
}