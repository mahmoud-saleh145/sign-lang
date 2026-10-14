/**
 * BLIND-QUERY ORIENTATION -- regression tests.
 *
 * A query whose two hands were never seen together has no geometric evidence
 * for which hand is which; its left/right names came from MediaPipe's
 * handedness label. The recogniser therefore scores both orientations of such
 * a query and keeps the nearer, and scores a geometry-backed query exactly
 * once, as named.
 *
 * Every fixture here is a REAL registry sequence. The only manipulations are
 * renaming the two sides of a timeline -- the same thing a reversed handedness
 * label does to a real recording -- and, for the blind TWO-hand fixture,
 * removing the two hands' temporal OVERLAP (see disjoinSides). Both move or
 * drop whole real frames; no landmark value is ever invented or synthesised.
 *
 * WHY THE BLIND TWO-HAND FIXTURE IS CONSTRUCTED AND NOT SELECTED.
 * It used to be picked straight out of the registry. After the
 * processed-karsl-v2 re-import it cannot be: the importer now chooses each
 * sign's template by "most two-hand frames", so every two-handed entry in the
 * shipped registry has co-presence well above GEOMETRY_MIN_FRAMES and is
 * geometry-backed BY CONSTRUCTION. Measured on the 432-sign registry:
 * sideSource is "geometry" for 388 entries and "handedness-label" for 44, and
 * no entry is a two-handed sequence whose hands were never seen together.
 * That is a property of template selection, not of the recogniser, and it is
 * desirable -- so the production registry is expected never to contain this
 * shape again. The blind two-hand CODE PATH still runs live, whenever the
 * camera sees each hand only separately, so it still has to be tested; the
 * fixture is therefore derived from a real entry here, in the test, and the
 * assertion that it is genuinely blind is made explicitly below rather than
 * assumed.
 */
import { describe, it, expect, beforeAll, vi } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import {
    clearDynamicSignCache,
    preloadDynamicSignRegistry,
    recognizeDynamicSignSyncWithDiagnostics,
} from "./dynamicSignRecognizer";
import { hasGeometricSideEvidence, twoHandFrameCount, GEOMETRY_MIN_FRAMES } from "./handSides";
import { MIN_SIDE_FRAMES } from "./twoHandMatch";
import { withLegacySequence, entryToSidedSequence } from "@/lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "@/lib/sign-language/signRegistry";
import type { SidedSequence } from "@/types/landmarks";

const ROOT = path.resolve(__dirname, "../..");
const REGISTRY_PATH = path.join(ROOT, "public/models/sign-vocabulary/dedicated-signs.json");
const registryJson = readFileSync(REGISTRY_PATH, "utf-8");
const entries = (JSON.parse(registryJson) as DedicatedSignEntry[]).map(withLegacySequence);

/** Renames the two sides. Landmarks are untouched -- this is exactly what a
 * reversed handedness label produces from a real recording. */
const swapSides = (frames: SidedSequence): SidedSequence =>
    frames.map((f) => ({ left: f.right, right: f.left }));

/**
 * Removes the two hands' temporal overlap, keeping only real frames.
 *
 * The left hand keeps the first half of the timeline and the right hand the
 * second half, so both hands are still present for plenty of frames but are
 * NEVER present in the same frame. That is precisely a blind two-hand query:
 * each hand is real and unmodified, and there is no frame in which their
 * relative position could be compared, so geometry cannot name the sides.
 *
 * This is the same category of manipulation as swapSides -- it rearranges
 * which real frame is attributed to which side and when, and invents nothing.
 */
const disjoinSides = (frames: SidedSequence): SidedSequence => {
    const mid = Math.floor(frames.length / 2);
    return frames.map((f, i) => (i < mid ? { left: f.left, right: null } : { left: null, right: f.right }));
};

const usableSides = (frames: SidedSequence): number =>
    (frames.filter((f) => f.left).length >= MIN_SIDE_FRAMES ? 1 : 0) +
    (frames.filter((f) => f.right).length >= MIN_SIDE_FRAMES ? 1 : 0);

interface Fixture {
    signId: string;
    sided: SidedSequence;
}

function pick(predicate: (sided: SidedSequence) => boolean): Fixture | null {
    for (const e of entries) {
        if (!e.sourceSignId) continue;
        const sided = entryToSidedSequence(e);
        if (!sided || sided.length < 8) continue;
        if (predicate(sided)) return { signId: e.sourceSignId, sided };
    }
    return null;
}

/** One hand only, and therefore never two hands in a frame: blind by construction. */
const blindOneHand = pick((s) => usableSides(s) === 1 && !hasGeometricSideEvidence(s));
/**
 * Both hands used, but never seen together enough to compare: blind,
 * two-handed. DERIVED, for the reason given in the module docstring: the
 * source is a real two-handed registry sequence long enough that each half
 * still clears MIN_SIDE_FRAMES, with its co-presence removed.
 */
const blindTwoHand = ((): Fixture | null => {
    for (const e of entries) {
        if (!e.sourceSignId) continue;
        const sided = entryToSidedSequence(e);
        if (!sided || sided.length < 4 * MIN_SIDE_FRAMES) continue;
        if (usableSides(sided) !== 2) continue;
        const disjoined = disjoinSides(sided);
        if (usableSides(disjoined) !== 2) continue;
        if (hasGeometricSideEvidence(disjoined)) continue;
        return { signId: e.sourceSignId, sided: disjoined };
    }
    return null;
})();
/** Both hands, seen together throughout: sides are measured, not guessed. */
const geometryTwoHand = pick((s) => usableSides(s) === 2 && twoHandFrameCount(s) >= 10);

beforeAll(async () => {
    vi.stubGlobal("fetch", async () => ({
        ok: true,
        status: 200,
        json: async () => JSON.parse(registryJson),
    }));
    clearDynamicSignCache();
    await preloadDynamicSignRegistry();
}, 300_000);

describe("blind-query orientation", () => {
    it("has real fixtures of each shape in the registry", () => {
        // If the registry ever stops containing one of these, the tests below
        // would silently pass on nothing. Fail loudly instead.
        expect(blindOneHand, "no blind one-hand entry in the registry").not.toBeNull();
        expect(blindTwoHand, "no blind two-hand entry in the registry").not.toBeNull();
        expect(geometryTwoHand, "no geometry-backed two-hand entry in the registry").not.toBeNull();
        expect(twoHandFrameCount(blindOneHand!.sided)).toBeLessThan(GEOMETRY_MIN_FRAMES);
        // The DERIVED blind two-hand fixture must really be both: two usable
        // hands, and zero frames in which they appear together. Asserted, not
        // assumed, because it is constructed rather than selected.
        expect(usableSides(blindTwoHand!.sided)).toBe(2);
        expect(twoHandFrameCount(blindTwoHand!.sided)).toBe(0);
        expect(hasGeometricSideEvidence(blindTwoHand!.sided)).toBe(false);
        expect(twoHandFrameCount(geometryTwoHand!.sided)).toBeGreaterThanOrEqual(GEOMETRY_MIN_FRAMES);
    });

    it("still finds the sign when a blind one-hand query's handedness is reversed", () => {
        // THE FAILURE THIS EXISTS FOR: measured on the held-out set, a blind
        // one-hand query named on the wrong side does not merely score badly
        // -- it shares NO side with its own template, so its sign falls to
        // rank 272-501 of 502 and some unrelated sign is returned instead.
        const reversed = swapSides(blindOneHand!.sided);
        const d = recognizeDynamicSignSyncWithDiagnostics(reversed);
        expect(d.querySideEvidence).toBe("blind");
        expect(d.bestSignId).toBe(blindOneHand!.signId);
        expect(d.bestOrientation).toBe("swapped");
    });

    it("keeps the named orientation for a blind query that is already right", () => {
        const d = recognizeDynamicSignSyncWithDiagnostics(blindOneHand!.sided);
        expect(d.querySideEvidence).toBe("blind");
        expect(d.bestSignId).toBe(blindOneHand!.signId);
        expect(d.bestOrientation).toBe("as-named");
    });

    it("reports the swapped orientation for a blind two-hand query that is mirrored", () => {
        const asNamed = recognizeDynamicSignSyncWithDiagnostics(blindTwoHand!.sided);
        expect(asNamed.querySideEvidence).toBe("blind");
        expect(asNamed.bestOrientation).toBe("as-named");
        expect(asNamed.bestSignId).toBe(blindTwoHand!.signId);

        const mirrored = recognizeDynamicSignSyncWithDiagnostics(swapSides(blindTwoHand!.sided));
        expect(mirrored.querySideEvidence).toBe("blind");
        expect(mirrored.bestSignId).toBe(blindTwoHand!.signId);
        expect(mirrored.bestOrientation).toBe("swapped");
        // Recovered to the same distance, because it is the same recording.
        expect(mirrored.bestDistance).toBeCloseTo(asNamed.bestDistance!, 6);
    });

    it("never re-orients a geometry-backed two-hand query", () => {
        const asNamed = recognizeDynamicSignSyncWithDiagnostics(geometryTwoHand!.sided);
        expect(asNamed.querySideEvidence).toBe("geometry");
        expect(asNamed.bestOrientation).toBe("as-named");
        expect(asNamed.bestSignId).toBe(geometryTwoHand!.signId);

        // Mirrored, this query is genuinely a different gesture and must be
        // scored as one. If the swap were applied here it would recover the
        // original distance -- the whole point is that it does not.
        const mirrored = recognizeDynamicSignSyncWithDiagnostics(swapSides(geometryTwoHand!.sided));
        expect(mirrored.querySideEvidence).toBe("geometry");
        expect(mirrored.bestOrientation).toBe("as-named");
        expect(mirrored.bestDistance).toBeGreaterThan(asNamed.bestDistance! + 0.01);
    });

    it("leaves geometry-backed recognition bit-identical across repeated runs", () => {
        // Determinism, and a guard that nothing in the blind path leaks into
        // the geometry path through shared state.
        const a = recognizeDynamicSignSyncWithDiagnostics(geometryTwoHand!.sided);
        const b = recognizeDynamicSignSyncWithDiagnostics(geometryTwoHand!.sided);
        expect(b.bestSignId).toBe(a.bestSignId);
        expect(b.bestDistance).toBe(a.bestDistance);
        expect(b.secondBestDistance).toBe(a.secondBestDistance);
        expect(b.accepted).toBe(a.accepted);

        const blindRun1 = recognizeDynamicSignSyncWithDiagnostics(blindOneHand!.sided);
        const blindRun2 = recognizeDynamicSignSyncWithDiagnostics(blindOneHand!.sided);
        expect(blindRun2.bestDistance).toBe(blindRun1.bestDistance);
        expect(blindRun2.bestOrientation).toBe(blindRun1.bestOrientation);
    });
});
