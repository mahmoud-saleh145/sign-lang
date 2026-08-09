import { NextResponse } from "next/server";
import { readFile } from "fs/promises";
import path from "path";
import { isDbConfigured } from "@/lib/db/mongodb";
import { signsCollection } from "@/lib/db/collections";

const CONTROL_SIGNS = new Set(["Delete", "Finish", "Space"]);

async function fallbackFromModelMetadata() {
  const metadataPath = path.join(process.cwd(), "public", "models", "arsl-31", "metadata.json");
  const raw = await readFile(metadataPath, "utf-8");
  const metadata = JSON.parse(raw) as { classes: string[]; language: string };
  return metadata.classes.map((classId) => ({
    languageCode: metadata.language,
    classId,
    displayLabel: classId,
    category: CONTROL_SIGNS.has(classId) ? "control" : "letter",
  }));
}

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const language = searchParams.get("language") ?? "ArSL";

  if (isDbConfigured()) {
    try {
      const col = await signsCollection();
      const signs = await col.find({ languageCode: language }).toArray();
      if (signs.length > 0) {
        return NextResponse.json({ signs, source: "database" });
      }
    } catch {
      // fall through to metadata fallback below
    }
  }

  try {
    const signs = await fallbackFromModelMetadata();
    return NextResponse.json({ signs, source: "model-metadata" });
  } catch {
    return NextResponse.json({ error: "No sign data available." }, { status: 503 });
  }
}
