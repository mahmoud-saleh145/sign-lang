import type { TranscriptEntry } from "@/types/features";

export default function TranscriptPanel({
  transcript,
  arabicText,
  onClear,
}: {
  transcript: TranscriptEntry[];
  arabicText: string;
  onClear: () => void;
}) {
  return (
    <section
      className="rounded-2xl bg-[var(--color-surface)] border border-white/10 p-4"
      aria-label="Transcript"
    >
      <div className="flex items-center justify-between mb-2">
        <h2 className="font-display text-sm uppercase tracking-wide text-[var(--color-text-dim)]">
          Transcript
        </h2>
        <button
          onClick={onClear}
          disabled={transcript.length === 0}
          className="text-sm text-[var(--color-text-dim)] disabled:opacity-40"
        >
          Clear
        </button>
      </div>
      <p
        className="font-display text-3xl leading-snug min-h-[3rem]"
        dir="rtl"
        lang="ar"
        aria-live="polite"
      >
        {arabicText.length === 0 ? (
          <span dir="ltr" className="text-[var(--color-text-dim)] text-base font-body block">
            Recognized signs will appear here as Arabic text as you sign.
          </span>
        ) : (
          arabicText
        )}
      </p>
    </section>
  );
}
