/**
 * THE TAKE-LEVEL SPLIT, in one place.
 *
 * KArSL records each sign several times. Those takes have to be divided
 * between three jobs that must not overlap:
 *
 *   benchmark    queries for lib/ml/karslHeldOut.test.ts
 *   calibration  queries for scripts/calibrate-two-hand.ts, where the
 *                acceptance gate is derived
 *   prototypes   what the recogniser matches against
 *
 * WHY THIS IS A MODULE AND NOT A SLICE INDEX. It was an index: the generator
 * skipped the first four files and the benchmark queried "the first two". Those
 * are not the same takes. The benchmark skips the registry template and any
 * take too short to score, so its "first two" can be files 3 and 4 -- and file
 * 4 was a prototype. Two queries were scored against a prototype built from
 * themselves, at distances of 0.0019 and 0.0015.
 *
 * The benchmark's own integrity guard caught it, which is the only reason it is
 * not in the numbers. The lesson is that two files agreeing on an arithmetic
 * rule is not a split; a split is one function both of them call.
 *
 * ============================================================================
 * WHY THE SPLIT IS BY RECORDING SESSION
 * ============================================================================
 * It used to be by position in a sorted list, which quietly measured the wrong
 * thing. KArSL recorded each sign across two sittings months apart, and takes
 * from one sitting are far more alike than takes from another day -- same
 * lighting, same distance, same warm-up. Measured on the 68-sign subset with
 * an otherwise identical protocol:
 *
 *     1 prototype from the query's OWN session      0.8309 top-1
 *     4 prototypes from a DIFFERENT session         0.8209 top-1
 *
 * One same-session example beat four cross-session ones. A split that lets
 * same-session takes sit on both sides therefore reports a number the live
 * camera will never reproduce, because a live user is always "another
 * session". Holding out a whole session removes that, and it is the harder,
 * honest measurement.
 *
 * So: ONE session is held out whole and supplies the queries; every other
 * session supplies the prototypes. No take, and no sitting, appears on both
 * sides.
 *
 * WHICH SESSION IS HELD OUT: the chronologically last one, which is both
 * deterministic and the natural reading of "does what we built still work on
 * the next recording". Dates in KArSL filenames are DD_MM_YY, so they are
 * reordered to YY_MM_DD before sorting -- a plain string sort on the raw date
 * puts 03_10_18 before 16_05_18 and would pick the wrong session.
 *
 * BENCHMARK vs CALIBRATION both come from that held-out session, interleaved
 * rather than split down the middle, because a sitting drifts as it goes on
 * and the first half is not exchangeable with the second. They are disjoint
 * from each other and both are disjoint from every prototype, which is the
 * property that matters: the gate is never derived on anything a prototype
 * has seen.
 *
 * A SIGN WITH ONLY ONE USABLE SESSION cannot take part in this at all. Rather
 * than fake it with a within-session split, such a sign yields NO queries and
 * keeps all of its takes as prototypes. It stays in the vocabulary -- it is
 * still a class the recogniser can answer with, and still a distractor every
 * other query has to beat -- but it is not scored. In the 432-sign corpus
 * that is four signs (0105, 0348, 0363, 0364), so the benchmark covers 428
 * classes out of a 432-way task, and reporting has to say so.
 *
 * EXCLUDED TAKES are named by ml/datasets/audit_processed_karsl.py in
 * karsl-exclusions.json, loaded by lib/ml/karslExclusions.ts. They are takes
 * with no detected hand, or with a wrist jump too large to be real motion
 * (one hand's path spliced onto the other's). They are dropped from ALL
 * THREE groups: corrupt input is no better as a prototype than as a query.
 */

/** One sign's takes, partitioned so that no take has two jobs. */
export interface TakeSplit {
    /** Queries for the held-out benchmark. Never prototypes. */
    benchmark: string[];
    /** Queries for threshold calibration. Never prototypes, never benchmark. */
    calibration: string[];
    /** Every take from the sessions that were NOT held out. */
    prototypes: string[];
    /**
     * The session held out for queries, or null when the sign had fewer than
     * two usable sessions and so yields no queries at all.
     */
    evaluationSession: string | null;
    /** Takes dropped before splitting, by name, for reporting. */
    excluded: string[];
}

/**
 * The recording date in a KArSL filename stem, as a sortable YY_MM_DD key.
 *
 * "03_01_0071_(01_12_16_15_52_41)_c" -> "16_12_01"
 *
 * Returns null when the name does not carry one. A take whose session cannot
 * be read is not guessed at: it is treated as its own unknown session, which
 * keeps it out of any cross-session pairing.
 */
export function sessionKey(file: string): string | null {
    const match = /\((\d{2})_(\d{2})_(\d{2})_\d{2}_\d{2}_\d{2}\)/.exec(file);
    if (!match) return null;
    const [, dd, mm, yy] = match;
    return `${yy}_${mm}_${dd}`;
}

/** The session date exactly as the filename spells it, for reporting. */
function rawSessionDate(file: string): string | null {
    const match = /\((\d{2}_\d{2}_\d{2})_\d{2}_\d{2}_\d{2}\)/.exec(file);
    return match ? match[1] : null;
}

function stemOf(file: string): string {
    return file.replace(/\.[^./\\]+$/, "");
}

/**
 * Partitions one sign's processed take filenames by recording session.
 *
 * Deterministic: the input is sorted first, so the same corpus always
 * produces the same split regardless of directory-read order.
 *
 * Takes the caller later discards (too short to score, unreadable) simply
 * shrink the query groups. They never migrate into `prototypes`, so the
 * failure direction is "fewer queries", never "a query that is also a
 * prototype".
 *
 * @param files      every take of one sign, as filenames
 * @param templateFile  the registry's own take, kept out of both query groups
 * @param excluded   take STEMS (no extension) to drop entirely
 */
export function splitTakes(
    files: ReadonlyArray<string>,
    templateFile: string | null,
    excluded: ReadonlySet<string> = new Set<string>()
): TakeSplit {
    const sorted = [...files].sort();
    const dropped = sorted.filter((f) => excluded.has(stemOf(f)));
    const usable = sorted.filter((f) => !excluded.has(stemOf(f)));

    // Group by session. An unreadable date becomes its own bucket keyed by the
    // filename, so it can never be paired with anything as "the same sitting".
    const bySession = new Map<string, string[]>();
    for (const file of usable) {
        const key = sessionKey(file) ?? `unknown:${file}`;
        const bucket = bySession.get(key);
        if (bucket) bucket.push(file);
        else bySession.set(key, [file]);
    }

    const empty = {
        benchmark: [] as string[],
        calibration: [] as string[],
        prototypes: usable,
        evaluationSession: null,
        excluded: dropped,
    };
    if (bySession.size < 2) return empty;

    // Chronologically last session is the held-out one.
    const keys = [...bySession.keys()].sort();
    const evalKey = keys[keys.length - 1];
    const evalTakes = bySession.get(evalKey) ?? [];

    // The registry template is never a query. If it happens to fall in the
    // held-out session it is also NOT promoted to a prototype -- doing that
    // would put one of the evaluation session's recordings on the prototype
    // side and undo the whole point of the split.
    const queryPool = evalTakes.filter((f) => f !== templateFile);

    // Interleaved, not halved: a sitting drifts as it goes on.
    const benchmark: string[] = [];
    const calibration: string[] = [];
    queryPool.forEach((file, i) => (i % 2 === 0 ? benchmark : calibration).push(file));

    const prototypes = keys
        .filter((k) => k !== evalKey)
        .flatMap((k) => bySession.get(k) ?? []);

    // A held-out session that yields no usable query leaves the sign
    // unscored rather than silently borrowing a prototype session's takes.
    if (queryPool.length === 0) return empty;

    return {
        benchmark,
        calibration,
        prototypes,
        evaluationSession: rawSessionDate(evalTakes[0]) ?? evalKey,
        excluded: dropped,
    };
}
