import Link from "next/link";

export default function HomePage() {
  return (
    <main className="min-h-dvh flex flex-col items-center justify-center max-w-lg mx-auto p-6 text-center gap-6">
      <h1 className="font-display text-3xl font-semibold leading-tight">
        Sign <span className="text-[var(--color-accent-active)]">{"\u21c4"}</span> Speech
        <br />
        Translator
      </h1>
      <p className="text-[var(--color-text-dim)] max-w-xs">
        Continuous Arabic Sign Language alphabet recognition and speech
        recognition, running locally on your phone.
      </p>
      <Link
        href="/translator"
        className="font-display font-semibold text-lg px-10 py-4 rounded-full bg-[var(--color-accent-active)] text-[#0B1220] active:scale-95 transition-transform"
      >
        Start translating
      </Link>
      <p className="font-mono text-xs text-[var(--color-text-dim)] max-w-xs">
        v1 supports the ArSL alphabet (31 signs). Camera and microphone
        processing stay on this device {"\u2014"} nothing is uploaded.
      </p>
    </main>
  );
}
