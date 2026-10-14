/**
 * REGRESSION: the sign-vocabulary API route must not 500.
 *
 * The route called loadDedicatedSignRegistry(), which fetches
 * "/models/sign-vocabulary/dedicated-signs.json" with a ROOT-RELATIVE url.
 * That resolves in a browser and nowhere else: on the server fetch() has no
 * base url and throws "Failed to parse URL", so every GET to this endpoint
 * returned 500 with an empty body. The registry was never even read.
 *
 * This drives the route handler directly under Node -- the same runtime Next
 * uses for App Router route handlers -- so a regression to a browser-only
 * loader fails here instead of only in production.
 *
 * NOTE ON TIMING: the registry is ~35 MB of JSON and the route parses all of
 * it. That is fine in production (verified: the endpoint answers in well under
 * a second) but it exceeds Vitest's 5 s default once this file runs alongside
 * the 61 MB prototype/benchmark suites, so the timeout is set explicitly.
 * Deliberately ONE test, not one per assertion: parsing 35 MB twice for two
 * assertions doubled the wall time and still only re-checked the same file.
 */
import { describe, it, expect } from "vitest";

const REGISTRY_PARSE_TIMEOUT_MS = 120_000;

describe("GET /api/sign-vocabulary", () => {
    it(
        "reads the registry off disk and returns a real count, without throwing",
        async () => {
            const { GET } = await import("@/app/api/sign-vocabulary/route");
            const res = await GET();

            // The regression: this was 500 with an empty body.
            expect(res.status).toBe(200);

            const body = (await res.json()) as {
                dedicatedSignCount: number;
                storedDatasets: unknown[];
                note?: string;
            };

            // The registry currently holds 432 KArSL word/phrase signs. Asserted
            // as a floor, not an equality, so adding signs upstream does not
            // turn this into a change-detector -- what matters is that the
            // route read a real, non-empty registry rather than throwing.
            expect(body.dedicatedSignCount).toBeGreaterThan(0);

            // The "no dataset integrated" note is only for an EMPTY registry;
            // asserting it is absent here is what proves the branch was not
            // taken by accident.
            expect(body.note).toBeUndefined();
            expect(Array.isArray(body.storedDatasets)).toBe(true);
        },
        REGISTRY_PARSE_TIMEOUT_MS
    );
});