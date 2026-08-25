import { NextResponse } from "next/server";
import { isDbConfigured } from "@/lib/db/mongodb";
import { signVocabularyCollection } from "@/lib/db/collections";
import { loadDedicatedSignRegistry } from "@/lib/sign-language/signRegistry";

export async function GET() {
    // public/models/sign-vocabulary/dedicated-signs.json is the source of
    // truth for what the app actually uses (see
    // lib/sign-language/signRegistry.ts) -- it is currently empty, honestly,
    // because no licensed+reachable word-level dataset has been integrated
    // yet. This endpoint also reports any dataset metadata records in
    // MongoDB, if configured, for future dataset-management tooling.
    const registry = await loadDedicatedSignRegistry();

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