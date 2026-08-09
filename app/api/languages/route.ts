import { NextResponse } from "next/server";
import { isDbConfigured } from "@/lib/db/mongodb";
import { languagesCollection } from "@/lib/db/collections";

// Static fallback so the app remains usable (spec 38: "MongoDB unavailable")
// even before a database is provisioned.
const FALLBACK_LANGUAGES = [
  { code: "ArSL", displayName: "Arabic Sign Language (alphabet)", isActive: true },
];

export async function GET() {
  if (!isDbConfigured()) {
    return NextResponse.json({ languages: FALLBACK_LANGUAGES, source: "fallback" });
  }
  try {
    const col = await languagesCollection();
    const languages = await col.find({ isActive: true }).toArray();
    if (languages.length === 0) {
      return NextResponse.json({ languages: FALLBACK_LANGUAGES, source: "fallback" });
    }
    return NextResponse.json({ languages, source: "database" });
  } catch {
    return NextResponse.json({ languages: FALLBACK_LANGUAGES, source: "fallback" });
  }
}
