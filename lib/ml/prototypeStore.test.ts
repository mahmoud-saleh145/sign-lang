/**
 * Regression tests for the multi-prototype store and the take-level split.
 *
 * Fixtures are REAL registry timelines. What is checked is the machinery that
 * carries them -- quantization, version gating, the split -- not the data.
 */
import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import {
    encodeTake,
    decodeTake,
    decodePrototypeFile,
    type PrototypeFile,
} from "./prototypeStore";
import { prepareSidedSequence, FEATURE_VERSION, DTW_SEQUENCE_LENGTH, sidedDistance } from "./twoHandMatch";
import { splitTakes, sessionKey } from "./karslTakeSplit";
import { withLegacySequence, entryToSidedSequence } from "@/lib/sign-language/signRegistry";
import type { DedicatedSignEntry } from "@/lib/sign-language/signRegistry";

const ROOT = path.resolve(__dirname, "../..");
const registryJson = readFileSync(
    path.join(ROOT, "public/models/sign-vocabulary/dedicated-signs.json"),
    "utf-8"
);
const entries = (JSON.parse(registryJson) as DedicatedSignEntry[]).map(withLegacySequence);

function firstTwoHandEntry() {
    for (const e of entries) {
        if (!e.sourceSignId) continue;
        const sided = entryToSidedSequence(e);
        if (!sided || sided.length < 8) continue;
        if (sided.some((f) => f.left && f.right)) return { entry: e, sided };
    }
    throw new Error("registry has no two-hand entry");
}

describe("prototype store", () => {
    it("round-trips a prepared gesture within quantization error", () => {
        const { sided } = firstTwoHandEntry();
        const prepared = prepareSidedSequence(sided);
        const encoded = encodeTake("take.json", prepared);
        expect(encoded).not.toBeNull();
        const decoded = decodeTake(encoded!);

        expect(decoded.left?.vectors.length).toBe(prepared.left?.vectors.length);
        expect(decoded.right?.vectors.length).toBe(prepared.right?.vectors.length);
        expect(decoded.relation?.vectors.length).toBe(prepared.relation?.vectors.length);
        expect(decoded.left?.detectedFrames).toBe(prepared.left?.detectedFrames);

        // 16-bit at 1/1000 of a unit: every value must survive to that.
        const a = prepared.left!.vectors;
        const b = decoded.left!.vectors;
        for (let i = 0; i < a.length; i++) {
            for (let k = 0; k < a[i].length; k++) {
                expect(Math.abs(a[i][k] - b[i][k])).toBeLessThan(0.001);
            }
        }
    });

    it("scores a decoded prototype as the original, within the quantization floor", () => {
        // The point of the artifact: what comes back has to behave like what
        // went in, or every stored distance is quietly wrong.
        const { sided } = firstTwoHandEntry();
        const prepared = prepareSidedSequence(sided);
        const decoded = decodeTake(encodeTake("take.json", prepared)!);
        const self = sidedDistance(prepared, prepared).distance;
        const viaStore = sidedDistance(prepared, decoded).distance;

        // Not exactly equal, and it cannot be: 16-bit quantization at 1/1000
        // leaves up to 0.0005 per component, which a 44-dimensional Euclidean
        // frame distance accumulates to at most sqrt(44) * 0.0005 = 0.0033.
        // Measured here: ~0.0014.
        expect(self).toBe(0);
        expect(viaStore).toBeLessThan(0.0033);

        // And that floor must stay far below the benchmark's tautology guard,
        // or a take scored against a prototype built from itself would stop
        // looking like the leak it is.
        expect(viaStore).toBeLessThan(0.02);
    });

    it("rejects an artifact built by a different feature pipeline", () => {
        const { sided } = firstTwoHandEntry();
        const take = encodeTake("take.json", prepareSidedSequence(sided))!;
        const stale: PrototypeFile = {
            version: 1,
            featureVersion: "shape42@30",
            sequenceLength: DTW_SEQUENCE_LENGTH,
            generatedAt: new Date().toISOString(),
            signs: { "0001": [take] },
        };
        // Scoring vectors from one representation against another is not a
        // degraded answer, it is a meaningless one. Rejecting sends the
        // recogniser back to its single-template fallback, which is correct.
        expect(decodePrototypeFile(stale)).toBeNull();

        const good: PrototypeFile = { ...stale, featureVersion: FEATURE_VERSION };
        const decoded = decodePrototypeFile(good);
        expect(decoded?.bySign.get("0001")?.length).toBe(1);
        expect(decoded?.takeCount).toBe(1);
    });

    it("rejects malformed and empty artifacts rather than throwing", () => {
        expect(decodePrototypeFile(null)).toBeNull();
        expect(decodePrototypeFile("nope")).toBeNull();
        expect(decodePrototypeFile({ version: 2 })).toBeNull();
        expect(
            decodePrototypeFile({
                version: 1,
                featureVersion: FEATURE_VERSION,
                sequenceLength: DTW_SEQUENCE_LENGTH,
                generatedAt: "",
                signs: {},
            })
        ).toBeNull();
    });
});

describe("session-held-out take split", () => {
    // Real KArSL names: two sittings, the later one (01_12_16) chronologically
    // last despite sorting first as a string, which is exactly the trap the
    // YY_MM_DD reordering exists to avoid.
    const early = [
        "03_01_0071_(20_11_16_16_16_41)_c.json",
        "03_01_0071_(20_11_16_16_17_01)_c.json",
        "03_01_0071_(20_11_16_16_17_25)_c.json",
    ];
    const late = [
        "03_01_0071_(01_12_16_15_52_41)_c.json",
        "03_01_0071_(01_12_16_15_53_08)_c.json",
        "03_01_0071_(01_12_16_15_54_03)_c.json",
        "03_01_0071_(01_12_16_15_54_51)_c.json",
    ];
    const files = [...early, ...late];

    it("queries come from one session and prototypes from the other", () => {
        const split = splitTakes(files, null);
        const queries = [...split.benchmark, ...split.calibration];
        expect(queries.length).toBe(late.length);
        for (const q of queries) expect(late).toContain(q);
        for (const p of split.prototypes) expect(early).toContain(p);
    });

    it("no take is ever both a query and a prototype", () => {
        for (const template of [null, ...files]) {
            const split = splitTakes(files, template);
            const prototypes = new Set(split.prototypes);
            for (const f of [...split.benchmark, ...split.calibration]) {
                expect(prototypes.has(f), `${f} is both a query and a prototype`).toBe(false);
            }
        }
    });

    it("no prototype shares a recording session with any query", () => {
        // This is the property the whole split exists for. A same-session
        // prototype measured 0.8309 top-1 against 0.8209 for four
        // cross-session ones, so leakage here inflates the headline number.
        const split = splitTakes(files, null);
        const querySessions = new Set(
            [...split.benchmark, ...split.calibration].map((f) => sessionKey(f))
        );
        for (const p of split.prototypes) {
            expect(querySessions.has(sessionKey(p)), `${p} shares a session with a query`).toBe(false);
        }
    });

    it("holds out the chronologically last session, not the last string", () => {
        // "01_12_16" sorts before "20_11_16" as text but is a later date.
        const split = splitTakes(files, null);
        expect(split.evaluationSession).toBe("01_12_16");
    });

    it("keeps benchmark and calibration disjoint and interleaved", () => {
        const split = splitTakes(files, null);
        for (const f of split.calibration) expect(split.benchmark).not.toContain(f);
        // Interleaved rather than halved, because a sitting drifts as it runs.
        expect(split.benchmark).toEqual([late[0], late[2]]);
        expect(split.calibration).toEqual([late[1], late[3]]);
    });

    it("never queries with the registry's own template take", () => {
        const split = splitTakes(files, late[0]);
        expect(split.benchmark).not.toContain(late[0]);
        expect(split.calibration).not.toContain(late[0]);
        // And it is NOT promoted into prototypes either: it belongs to the
        // held-out session, and moving it across would break the split.
        expect(split.prototypes).not.toContain(late[0]);
    });

    it("keeps a template from a prototype session as training data", () => {
        const split = splitTakes(files, early[0]);
        expect(split.prototypes).toContain(early[0]);
    });

    it("drops excluded takes from all three groups", () => {
        // A take with no detected hand or a spliced trajectory is no better
        // as a prototype than as a query.
        const excluded = new Set([
            "03_01_0071_(01_12_16_15_52_41)_c",
            "03_01_0071_(20_11_16_16_16_41)_c",
        ]);
        const split = splitTakes(files, null, excluded);
        const everywhere = [...split.benchmark, ...split.calibration, ...split.prototypes];
        for (const name of everywhere) expect(excluded.has(name.replace(/\.json$/, ""))).toBe(false);
        expect(split.excluded).toHaveLength(2);
    });

    it("yields no queries for a sign with only one usable session", () => {
        // 0348, 0363 and 0364 in the real corpus. The sign stays in the
        // vocabulary as a prototype and a distractor; it is simply not scored.
        const split = splitTakes(early, null);
        expect(split.benchmark).toEqual([]);
        expect(split.calibration).toEqual([]);
        expect(split.prototypes).toEqual(early);
        expect(split.evaluationSession).toBeNull();
    });

    it("yields no queries when exclusions empty out the second session", () => {
        // Sign 0105: two session dates, but every take of one of them lost
        // its hands. Counting dates would call it splittable.
        const excluded = new Set(late.map((f) => f.replace(/\.json$/, "")));
        const split = splitTakes(files, null, excluded);
        expect(split.benchmark).toEqual([]);
        expect(split.evaluationSession).toBeNull();
        expect(split.prototypes).toEqual(early);
    });

    it("is deterministic regardless of directory read order", () => {
        expect(splitTakes([...files].reverse(), null)).toEqual(splitTakes(files, null));
    });

    it("degrades safely on empty and single-take input", () => {
        expect(splitTakes([], null)).toEqual({
            benchmark: [], calibration: [], prototypes: [],
            evaluationSession: null, excluded: [],
        });
        const one = splitTakes(["03_01_0071_(01_12_16_15_52_41)_c.json"], null);
        expect(one.benchmark).toEqual([]);
        expect(one.prototypes).toHaveLength(1);
    });

    it("treats an unreadable date as its own session rather than guessing", () => {
        const split = splitTakes(["homemade.json", ...early], null);
        // "homemade" cannot pair with anything, so it is its own bucket and
        // the sign is splittable -- but no unknown-date take is ever grouped
        // with a dated one as "the same sitting".
        const queries = [...split.benchmark, ...split.calibration];
        expect(queries.length + split.prototypes.length).toBe(4);
        for (const q of queries) {
            for (const p of split.prototypes) {
                expect(sessionKey(q) === sessionKey(p) && sessionKey(q) !== null).toBe(false);
            }
        }
    });
});
