import { NextResponse } from "next/server";
import { isDbConfigured } from "@/lib/db/mongodb";
import { collectedSamplesCollection } from "@/lib/db/collections";
import { validateCollectedSample } from "@/lib/db/validation";

export async function POST(request: Request) {
  if (!isDbConfigured()) {
    return NextResponse.json(
      { error: "Database not configured. Set MONGODB_URI to use the dataset collection tool." },
      { status: 503 }
    );
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const result = validateCollectedSample(body);
  if (!result.valid || !result.data) {
    return NextResponse.json({ error: "Validation failed.", details: result.errors }, { status: 400 });
  }

  try {
    const col = await collectedSamplesCollection();
    const doc = { ...result.data, capturedAt: new Date() };
    const inserted = await col.insertOne(doc);
    return NextResponse.json({ stored: true, id: inserted.insertedId }, { status: 201 });
  } catch (err) {
    return NextResponse.json(
      { error: "Failed to store sample.", message: err instanceof Error ? err.message : "unknown" },
      { status: 500 }
    );
  }
}

/** Returns per-label sample counts, for the collection tool's progress display. */
export async function GET() {
  if (!isDbConfigured()) {
    return NextResponse.json({ counts: {}, source: "unavailable" });
  }
  try {
    const col = await collectedSamplesCollection();
    const results = await col
      .aggregate<{ _id: string; count: number }>([{ $group: { _id: "$label", count: { $sum: 1 } } }])
      .toArray();
    const counts: Record<string, number> = {};
    for (const r of results) counts[r._id] = r.count;
    return NextResponse.json({ counts, source: "database" });
  } catch (err) {
    return NextResponse.json(
      { error: "Failed to fetch counts.", message: err instanceof Error ? err.message : "unknown" },
      { status: 500 }
    );
  }
}
