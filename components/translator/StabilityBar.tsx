const STATE_ORDER = ["IDLE", "POSSIBLE_SIGN", "SIGN_ACTIVE", "POSSIBLE_END", "COMMITTED"];

export default function StabilityBar({ state }: { state: string }) {
  const index = Math.max(STATE_ORDER.indexOf(state), 0);
  const fraction = state === "COMMITTED" ? 1 : index / (STATE_ORDER.length - 2);
  const color =
    state === "COMMITTED"
      ? "var(--color-accent-active)"
      : state === "SIGN_ACTIVE" || state === "POSSIBLE_END"
        ? "var(--color-accent-active)"
        : "var(--color-accent-candidate)";

  return (
    <div
      className="h-1.5 w-full rounded-full bg-white/10 overflow-hidden"
      role="progressbar"
      aria-label="Sign stability"
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(fraction * 100)}
    >
      <div
        className="h-full rounded-full transition-all duration-150 ease-out"
        style={{
          width: `${Math.min(fraction, 1) * 100}%`,
          background: color,
        }}
      />
    </div>
  );
}
