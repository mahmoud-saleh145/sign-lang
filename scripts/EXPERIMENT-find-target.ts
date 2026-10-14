/**
 * EXPERIMENT -- NOT PRODUCTION. Registry lookup for a target phrase.
 * Reports exact stored label, codepoints, aliases/gloss and template counts.
 */
import { readFileSync } from "fs";
import path from "path";
import { decodePrototypeFile } from "../lib/ml/prototypeStore";
import { withLegacySequence, type DedicatedSignEntry } from "../lib/sign-language/signRegistry";
import { normalizeArabicText } from "../lib/sign-language/textNormalization";
/** The production lookup key. */
const norm = (s: string): string => normalizeArabicText(s).text;

const ROOT = path.resolve(__dirname, "..");
const PHRASE = process.argv[2] ?? "إعاقة بصرية";

const registry = (
    JSON.parse(
        readFileSync(path.join(ROOT, "public/models/sign-vocabulary/dedicated-signs.json"), "utf-8")
    ) as DedicatedSignEntry[]
).map(withLegacySequence);

const proto = decodePrototypeFile(
    JSON.parse(
        readFileSync(path.join(ROOT, "public/models/sign-vocabulary/sign-prototypes.eval.json"), "utf-8")
    )
);

const codes = (s: string): string =>
    [...s].map((ch) => ch.codePointAt(0)!.toString(16).toUpperCase().padStart(4, "0")).join(" ");

console.log("=".repeat(104));
console.log(`QUERY PHRASE : "${PHRASE}"`);
console.log(`codepoints    : ${codes(PHRASE)}`);
console.log(`normalizeArabic(query) = "${norm(PHRASE)}"  [${codes(norm(PHRASE))}]`);
console.log("=".repeat(104));

// 1. exact raw match
const exact = registry.filter((e) => e.text === PHRASE);
console.log(`\n1. RAW EXACT MATCH (text === phrase): ${exact.length}`);
for (const e of exact) {
    console.log(`   ${e.sourceSignId}  "${e.text}"  prototypes=${proto?.bySign.get(e.sourceSignId ?? "")?.length ?? 0}`);
}

// 2. match AFTER the app's own normalisation
const targetNorm = norm(PHRASE);
const byNorm = registry.filter((e) => norm(e.text ?? "") === targetNorm);
console.log(`\n2. MATCH AFTER normalizeArabic(): ${byNorm.length}`);
for (const e of byNorm) {
    console.log(`   ${e.sourceSignId}  stored="${e.text}"  [${codes(e.text ?? "")}]  prototypes=${proto?.bySign.get(e.sourceSignId ?? "")?.length ?? 0}`);
}

// 3. loose match: same length, differing only by hamza-like marks
const strip = (s: string): string => s.replace(/[ً-ٰٟـ]/g, "").replace(/[إأآٱ]/g, "ا").replace(/ى/g, "ي").replace(/ة/g, "ه").replace(/\s+/g, " ").trim();
const loose = registry.filter((e) => strip(e.text ?? "") === strip(PHRASE));
console.log(`\n3. LOOSE MATCH (hamza/alef/taa-marbuta/ya normalised): ${loose.length}`);
for (const e of loose) {
    console.log(`   ${e.sourceSignId}  stored="${e.text}"  prototypes=${proto?.bySign.get(e.sourceSignId ?? "")?.length ?? 0}`);
    console.log(`      codepoints: ${codes(e.text ?? "")}`);
    console.log(`      == query?  ${e.text === PHRASE}`);
}

// 4. any other field carrying the phrase (english gloss etc.)
const fields = ["text", "english", "gloss", "alias", "aliases", "englishLabel"];
console.log(`\n4. SEARCH OF OTHER LABEL FIELDS: ${fields.join(", ")}`);
for (const e of registry) {
    const rec = e as unknown as Record<string, unknown>;
    for (const f of fields) {
        const v = rec[f];
        if (typeof v === "string" && norm(v).includes(targetNorm)) {
            console.log(`   ${e.sourceSignId}.${f} = "${v}"`);
        }
    }
}

// 5. registry key list -- what fields actually exist on an entry
const sample = registry[0] as unknown as Record<string, unknown>;
console.log(`\n5. FIELDS PRESENT ON A REGISTRY ENTRY: ${Object.keys(sample).join(", ")}`);

