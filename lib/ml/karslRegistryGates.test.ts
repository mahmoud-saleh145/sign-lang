/**
 * REGRESSION TEST FOR THE ORIGINAL ROOT CAUSE.
 *
 * The dynamic path used to reach the classifier for almost no real gesture,
 * because DynamicGestureBuffer discarded any completed buffer whose average
 * per-frame landmark displacement was below minMotionEnergy = 0.02. Measured
 * against the project's OWN real imported KArSL data, 164 of 170 sequences
 * (96.5%) fall below that value -- so the gate was throwing away almost the
 * entire dataset before DTW ever ran.
 *
 * These tests use the REAL registry file that ships in the repo
 * (public/models/sign-vocabulary/dedicated-signs.json), not synthetic poses,
 * specifically so a future change that re-tightens the gate fails here
 * instead of silently disabling word recognition again.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { DEFAULT_GESTURE_BUFFER_CONFIG, computeMotionEnergy } from "./dynamicGestureBuffer";
import type { SidedSequence } from "@/types/landmarks";
import { withLegacySequence, entryToSidedSequence } from "@/lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "@/lib/sign-language/signRegistry";

// The shipped registry omits the legacy `landmarkSequence` (it duplicates the
// primary hand already in `handFrames`). Reading the file directly therefore
// has to apply the same normalization the app applies on load.
const registry = (
    JSON.parse(
        readFileSync(path.resolve(__dirname, "../../public/models/sign-vocabulary/dedicated-signs.json"), "utf-8")
    ) as DedicatedSignEntry[]
).map(withLegacySequence);

const usable = registry.filter((e) => e.landmarkSequence.length >= 2);

/**
 * The entry's REAL two-hand timeline, which is what the side-aware
 * computeMotionEnergy() now consumes. A registry entry that cannot produce one
 * is an error rather than a silently skipped sample -- skipping would change
 * which data the gates below are measured against.
 */
function sidedOf(entry: DedicatedSignEntry): SidedSequence {
    const sided = entryToSidedSequence(entry);
    if (!sided) {
        throw new Error(
            `Registry entry ${entry.sourceSignId ?? entry.text} has no usable two-hand timeline.`
        );
    }
    return sided;
}

describe("KArSL registry vs. gesture-buffer gates (real data)", () => {
    it("has real entries to test against", () => {
        expect(usable.length).toBeGreaterThan(0);
    });

    it("passes at least 95% of real sequences through the motion-energy gate", () => {
        const passing = usable.filter(
            (e) => computeMotionEnergy(sidedOf(e)) >= DEFAULT_GESTURE_BUFFER_CONFIG.minMotionEnergy
        );
        const rate = passing.length / usable.length;
        // A gate that rejects real KArSL signs is the bug this suite exists
        // to prevent. 0.02 scored 0.035 here; the calibrated 0.0001 scores 1.0.
        expect(rate).toBeGreaterThanOrEqual(0.95);
    });

    it("passes at least 95% of real sequences through the minimum-frame gate", () => {
        const passing = usable.filter(
            (e) => e.landmarkSequence.length >= DEFAULT_GESTURE_BUFFER_CONFIG.minGestureFrames
        );
        expect(passing.length / usable.length).toBeGreaterThanOrEqual(0.95);
    });

    it("does not truncate real sequences at the frame cap", () => {
        const longest = Math.max(...usable.map((e) => e.landmarkSequence.length));
        expect(DEFAULT_GESTURE_BUFFER_CONFIG.maxGestureFrames).toBeGreaterThan(longest);
    });

    it("documents the measured motion-energy range of the real data", () => {
        const energies = usable.map((e) => computeMotionEnergy(sidedOf(e))).sort((a, b) => a - b);
        const min = energies[0];
        const median = energies[Math.floor(energies.length / 2)];
        // These are assertions about the DATA, kept so that if the registry
        // is ever re-imported with a different representation, the change is
        // caught rather than silently shifting every threshold's meaning.
        expect(min).toBeGreaterThan(0);
        expect(min).toBeLessThan(0.01);
        expect(median).toBeLessThan(0.02);
    });

    it("keeps multi-word KArSL labels intact as single entries", () => {
        const multiWord = registry.filter((e) => e.text.trim().includes(" "));
        // The dataset genuinely contains multi-word phrases; if it does, each
        // must be ONE entry whose text is the whole phrase.
        for (const e of multiWord) {
            expect(e.text).toBe(e.text.trim());
            expect(e.landmarkSequence.length).toBeGreaterThanOrEqual(2);
        }
    });
});
