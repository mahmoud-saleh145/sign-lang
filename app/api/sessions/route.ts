import { NextResponse } from "next/server";
import { isDbConfigured } from "@/lib/db/mongodb";
import { translationSessionsCollection } from "@/lib/db/collections";
import { validateTranslationSession } from "@/lib/db/validation";

export async function POST(request: Request) {
  if (!isDbConfigured()) {
    // Not fatal — session logging is telemetry, not core functionality.
    return NextResponse.json({ stored: false, reason: "Database not configured." }, { status: 202 });
  }

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body." }, { status: 400 });
  }

  const result = validateTranslationSession(body);
  if (!result.valid || !result.data) {
    return NextResponse.json({ error: "Validation failed.", details: result.errors }, { status: 400 });
  }

  try {
    const col = await translationSessionsCollection();
    await col.insertOne(result.data);
    return NextResponse.json({ stored: true }, { status: 201 });
  } catch (err) {
    return NextResponse.json(
      { error: "Failed to store session.", message: err instanceof Error ? err.message : "unknown" },
      { status: 500 }
    );
  }
}

export async function GET(request: Request) {
  if (!isDbConfigured()) {
    return NextResponse.json({ sessions: [], source: "unavailable" });
  }
  const { searchParams } = new URL(request.url);
  const limit = Math.min(Number(searchParams.get("limit") ?? 50), 200);

  try {
    const col = await translationSessionsCollection();
    const sessions = await col.find({}).sort({ startedAt: -1 }).limit(limit).toArray();
    return NextResponse.json({ sessions, source: "database" });
  } catch (err) {
    return NextResponse.json(
      { error: "Failed to fetch sessions.", message: err instanceof Error ? err.message : "unknown" },
      { status: 500 }
    );
  }
}
