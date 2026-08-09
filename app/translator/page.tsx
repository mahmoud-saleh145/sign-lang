"use client";

import { useState } from "react";
import CameraView from "@/components/camera/CameraView";
import SpeechToText from "@/components/speech/SpeechToText";

type Mode = "sign" | "speech";

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
        className="flex rounded-full bg-[var(--color-surface)] p-1 border border-white/10"
      >
        <ModeTab label={"Sign \u2192 Text"} active={mode === "sign"} onClick={() => setMode("sign")} />
        <ModeTab label={"Speech \u2192 Text"} active={mode === "speech"} onClick={() => setMode("speech")} />
      </div>

      {mode === "sign" ? <CameraView /> : <SpeechToText />}
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
      className={`flex-1 py-3 rounded-full font-medium text-sm transition-colors ${
        active ? "bg-[var(--color-accent-active)] text-[#0B1220]" : "text-[var(--color-text-dim)]"
      }`}
    >
      {label}
    </button>
  );
}
