import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { ARABIC_LETTER_MAP, toArabicChar, isControlSign } from "./arabicLabels";
import { ArabicTextBuilder } from "./arabicTextBuilder";

describe("ARABIC_LETTER_MAP", () => {
  it("maps every English model label to the correct Arabic character (spot checks against dataset docs)", () => {
    expect(toArabicChar("Alef")).toBe("\u0627"); // ا
    expect(toArabicChar("Ba2")).toBe("\u0628"); // ب
    expect(toArabicChar("Ta2")).toBe("\u062a"); // ت
    expect(toArabicChar("Tha2")).toBe("\u062b"); // ث
    expect(toArabicChar("7a2")).toBe("\u062d"); // ح
    expect(toArabicChar("Chin")).toBe("\u0634"); // ش
    expect(toArabicChar("9af")).toBe("\u0642"); // ق
    expect(toArabicChar("Ha2")).toBe("\u0647"); // ه
    expect(toArabicChar("Ya2")).toBe("\u064a"); // ي
  });

  it("covers exactly the 28 letter classes from the model's metadata, no more, no less", () => {
    const metadata = JSON.parse(
      readFileSync(
        path.resolve(__dirname, "../../public/models/arsl-31/metadata.json"),
        "utf-8"
      )
    ) as { classes: string[] };
    const controlSigns = new Set(["Space", "Delete", "Finish"]);
    const expectedLetters = metadata.classes.filter((c) => !controlSigns.has(c));

    expect(expectedLetters.sort()).toEqual(Object.keys(ARABIC_LETTER_MAP).sort());
  });

  it("every mapped value is a single, distinct Arabic character", () => {
    const values = Object.values(ARABIC_LETTER_MAP);
    expect(new Set(values).size).toBe(values.length); // no duplicates
    for (const v of values) {
      expect(v.length).toBe(1);
      expect(v.codePointAt(0)).toBeGreaterThanOrEqual(0x0600); // Arabic block
      expect(v.codePointAt(0)).toBeLessThanOrEqual(0x06ff);
    }
  });

  it("recognizes control signs as control, not letters", () => {
    expect(isControlSign("Space")).toBe(true);
    expect(isControlSign("Delete")).toBe(true);
    expect(isControlSign("Finish")).toBe(true);
    expect(isControlSign("Alef")).toBe(false);
    expect(toArabicChar("Space")).toBeNull();
  });
});

describe("ArabicTextBuilder", () => {
  it("accumulates letters into a word purely by concatenation (no dictionary)", () => {
    const builder = new ArabicTextBuilder();
    // ب -> ح -> ب -> ك should genuinely concatenate to بحبك, not look up a
    // word table. Prove it by checking each intermediate state too.
    expect(builder.applyCommittedLabel("Ba2").text).toBe("\u0628"); // ب
    expect(builder.applyCommittedLabel("7a2").text).toBe("\u0628\u062d"); // بح
    expect(builder.applyCommittedLabel("Ba2").text).toBe("\u0628\u062d\u0628"); // بحب
    expect(builder.applyCommittedLabel("Kaf").text).toBe("\u0628\u062d\u0628\u0643"); // بحبك
  });

  it("Space inserts a literal space", () => {
    const builder = new ArabicTextBuilder();
    builder.applyCommittedLabel("Alef");
    const result = builder.applyCommittedLabel("Space");
    expect(result.action).toBe("space");
    expect(result.text).toBe("\u0627 ");
  });

  it("Delete removes the last committed character", () => {
    const builder = new ArabicTextBuilder();
    builder.applyCommittedLabel("Alef");
    builder.applyCommittedLabel("Ba2");
    const result = builder.applyCommittedLabel("Delete");
    expect(result.action).toBe("delete");
    expect(result.text).toBe("\u0627"); // ب removed, ا remains
  });

  it("Delete on empty text is a safe no-op", () => {
    const builder = new ArabicTextBuilder();
    const result = builder.applyCommittedLabel("Delete");
    expect(result.text).toBe("");
  });

  it("Delete can remove a trailing space", () => {
    const builder = new ArabicTextBuilder();
    builder.applyCommittedLabel("Alef");
    builder.applyCommittedLabel("Space");
    const result = builder.applyCommittedLabel("Delete");
    expect(result.text).toBe("\u0627"); // trailing space removed
  });

  it("Finish does not modify the text (no punctuation semantics defined for V1)", () => {
    const builder = new ArabicTextBuilder();
    builder.applyCommittedLabel("Alef");
    const before = builder.getText();
    const result = builder.applyCommittedLabel("Finish");
    expect(result.action).toBe("finish");
    expect(result.text).toBe(before);
  });

  it("repeated identical letters both get appended when each is a genuine separate commit", () => {
    const builder = new ArabicTextBuilder();
    builder.applyCommittedLabel("Lam");
    const result = builder.applyCommittedLabel("Lam");
    expect(result.text).toBe("\u0644\u0644"); // لل
  });

  it("builds multiple words separated by Space, matching the spec's worked example", () => {
    const builder = new ArabicTextBuilder();
    // أ -> ن -> ا -> Space -> ب -> ح -> ب -> ك  should give "انا بحبك"
    // Note: dataset has "Alef" (ا) only, no separate hamza-on-alef class,
    // so we use Alef for all three "ا"/"أ"-looking letters here, matching
    // what the model can actually output.
    for (const label of ["Alef", "Noon", "Alef", "Space", "Ba2", "7a2", "Ba2", "Kaf"]) {
      builder.applyCommittedLabel(label);
    }
    expect(builder.getText()).toBe("\u0627\u0646\u0627 \u0628\u062d\u0628\u0643"); // "انا بحبك"
  });

  it("reset() clears accumulated text", () => {
    const builder = new ArabicTextBuilder();
    builder.applyCommittedLabel("Alef");
    builder.reset();
    expect(builder.getText()).toBe("");
  });

  it("unknown labels are a safe no-op", () => {
    const builder = new ArabicTextBuilder();
    builder.applyCommittedLabel("Alef");
    const result = builder.applyCommittedLabel("NotARealClass");
    expect(result.action).toBe("unknown");
    expect(result.text).toBe("\u0627");
  });
});
