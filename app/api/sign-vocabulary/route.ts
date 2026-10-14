import { readFile } from "node:fs/promises";
import path from "node:path";
import { NextResponse } from "next/server";
import { isDbConfigured } from "@/lib/db/mongodb";
import { signVocabularyCollection } from "@/lib/db/collections";
import type { DedicatedSignEntry } from "@/lib/sign-language/signRegistry";

/**
 * Reads the registry off disk instead of through loadDedicatedSignRegistry().
 *
 * That loader fetches "/models/sign-vocabulary/dedicated-signs.json" with a
 * ROOT-RELATIVE url, which only resolves in a browser. On the server, fetch()
 * has no base url and throws "Failed to parse URL from /models/...", which
 * propagated out of this route and turned every GET into a 500 with an empty
 * body. The asset lives in public/, so on the server it is a file read -- no
 * self-HTTP round trip, no dependence on a known origin or host header.
 *
 * Every other caller of loadDedicatedSignRegistry() is client-side, so that
 * function is left exactly as it is.
 */
const REGISTRY_FILE = path.join(
    process.cwd(),
    "public",
    "models",
    "sign-vocabulary",
    "dedicated-signs.json"
);

async function loadRegistryOnServer(): Promise<DedicatedSignEntry[]> {
    try {
        const raw = await readFile(REGISTRY_FILE, "utf-8");
        const parsed: unknown = JSON.parse(raw);
        return Array.isArray(parsed) ? (parsed as DedicatedSignEntry[]) : [];
    } catch (err) {
        // Reported rather than swallowed: an empty vocabulary and a missing
        // one are different facts, and only one of them is a bug.
        console.error(`[api/sign-vocabulary] could not read ${REGISTRY_FILE}:`, err);
        throw err;
    }
}

export async function GET() {
    // public/models/sign-vocabulary/dedicated-signs.json is the source of
    // truth for what the app actually uses (see
    // lib/sign-language/signRegistry.ts). It holds the KArSL word/phrase
    // signs; the ArSL-31 alphabet is served separately from public/models/arsl-31.
    // This endpoint also reports any dataset metadata records in MongoDB, if
    // configured, for future dataset-management tooling.
    const registry = await loadRegistryOnServer();

    let storedDatasets: unknown[] = [];
    if (isDbConfigured()) {
        try {
            const col = await signVocabularyCollection();
            storedDatasets = await col.find({}).toArray();
        } catch {
            storedDatasets = [];
        }
    }

    return NextResponse.json({
        dedicatedSignCount: registry.length,
        storedDatasets,
        note:
            registry.length === 0
                ? "No dedicated word/phrase sign dataset is currently integrated. All words are spelled letter-by-letter as an explicit fallback. See docs/SIGN_LANGUAGE_DATASETS.md."
                : undefined,
    });
}