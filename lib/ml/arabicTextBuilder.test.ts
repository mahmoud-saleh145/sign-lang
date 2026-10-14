import { describe, it, expect } from "vitest";
import { readFileSync } from "fs";
import path from "path";
import { ARABIC_LETTER_MAP, toArabicChar, isControlSign } from "./arabicLabels";
import { ArabicTextBuilder } from "./arabicTextBuilder";

describe("ARABIC_LETTER_MAP", () => {
  it("maps every English model label to the correct Arabic character (spot checks against dataset docs)", () => {
    expect(toArabicChar("Alef")).toBe("\u0627");
    expect(toArabicChar("Ba2")).toBe("\u0628");
    expect(toArabicChar("Ta2")).toBe("\u062a");
    expect(toArabicChar("Tha2")).toBe("\u062b");
    expect(toArabicChar("7a2")).toBe("\u062d");
    expect(toArabicChar("Chin")).toBe("\u0634");
    expect(toArabicChar("9af")).toBe("\u0642");
    expect(toArabicChar("Ha2")).toBe("\u0647");
    expect(toArabicChar("Ya2")).toBe("\u064a");
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
    expect(new Set(values).size).toBe(values.length);
    for (const v of values) {
      expect(v.length).toBe(1);
      expect(v.codePointAt(0)).toBeGreaterThanOrEqual(0x0600);
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
    expect(builder.applyCommittedLabel("Ba2").text).toBe("\u0628");
    expect(builder.applyCommittedLabel("7a2").text).toBe("\u0628\u062d");
    expect(builder.applyCommittedLabel("Ba2").text).toBe("\u0628\u062d\u0628");
    expect(builder.applyCommittedLabel("Kaf").text).toBe("\u0628\u062d\u0628\u0643");
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
    expect(result.text).toBe("\u0627");
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
    expect(result.text).toBe("\u0627");
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
    expect(result.text).toBe("\u0644\u0644");
  });

  it("builds multiple words separated by Space, matching the spec's worked example", () => {
    const builder = new ArabicTextBuilder();
    for (const label of ["Alef", "Noon", "Alef", "Space", "Ba2", "7a2", "Ba2", "Kaf"]) {
      builder.applyCommittedLabel(label);
    }
    expect(builder.getText()).toBe("\u0627\u0646\u0627 \u0628\u062d\u0628\u0643");
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

  // CONTRACT CHANGED DELIBERATELY. This case used to assert that no spacing was
  // ever added, which produced "\u0645\u0631\u062d\u0628\u0627\u0634\u0643\u0631\u0627" for two consecutively recognized
  // dynamic signs -- the reported bug. The phrase is still committed VERBATIM
  // and is still never split; what changed is that a separator is inserted
  // BEFORE it when the text does not already end at a word boundary.
  it("appendLiteralText commits a KArSL phrase verbatim, and separates it from preceding text", () => {
    const builder = new ArabicTextBuilder();
    builder.applyCommittedLabel("Alef");
    const result = builder.appendLiteralText("\u0625\u0633\u0639\u0627\u0641\u0627\u062a \u0623\u0648\u0644\u064a\u0629");
    expect(result.action).toBe("append");
    // "\u0627" + boundary + the phrase, whose own internal space is untouched.
    expect(result.text).toBe("\u0627 \u0625\u0633\u0639\u0627\u0641\u0627\u062a \u0623\u0648\u0644\u064a\u0629");
    // The phrase survives as one contiguous substring: it was not split.
    expect(result.text).toContain("\u0625\u0633\u0639\u0627\u0641\u0627\u062a \u0623\u0648\u0644\u064a\u0629");
  });

  // ==========================================================================
  // DYNAMIC WORD BOUNDARIES (Phase 3). The distinction these pin down is
  // "two separate dynamic signs" vs "one multi-word KArSL label", decided by
  // HOW MANY append calls arrive -- never by inspecting the label text.
  // ==========================================================================

  it("a single dynamic word stands alone with no stray separator", () => {
    const builder = new ArabicTextBuilder();
    const r = builder.appendLiteralText("\u0645\u0631\u062d\u0628\u0627");
    expect(r.text).toBe("\u0645\u0631\u062d\u0628\u0627");
  });

  it("TWO dynamic words are separated: \u0645\u0631\u062d\u0628\u0627 + \u0634\u0643\u0631\u0627 -> '\u0645\u0631\u062d\u0628\u0627 \u0634\u0643\u0631\u0627', never '\u0645\u0631\u062d\u0628\u0627\u0634\u0643\u0631\u0627'", () => {
    const builder = new ArabicTextBuilder();
    builder.appendLiteralText("\u0645\u0631\u062d\u0628\u0627");
    const r = builder.appendLiteralText("\u0634\u0643\u0631\u0627");
    expect(r.text).toBe("\u0645\u0631\u062d\u0628\u0627 \u0634\u0643\u0631\u0627");
    expect(r.text).not.toBe("\u0645\u0631\u062d\u0628\u0627\u0634\u0643\u0631\u0627");
    expect(r.text.split(" ")).toHaveLength(2);
  });

  it("a multi-word dedicated KArSL label stays ONE sign and is not split", () => {
    const builder = new ArabicTextBuilder();
    const phrase = "\u0625\u0633\u0639\u0627\u0641\u0627\u062a \u0623\u0648\u0644\u064a\u0629";
    const r = builder.appendLiteralText(phrase);
    // Exactly the label, byte for byte -- no leading separator, no re-spacing.
    expect(r.text).toBe(phrase);
  });

  it("a multi-word label followed by another dynamic word yields three words, not a merge", () => {
    const builder = new ArabicTextBuilder();
    const phrase = "\u0625\u0633\u0639\u0627\u0641\u0627\u062a \u0623\u0648\u0644\u064a\u0629";
    builder.appendLiteralText(phrase);
    const r = builder.appendLiteralText("\u0634\u0643\u0631\u0627");
    expect(r.text).toBe(`${phrase} \u0634\u0643\u0631\u0627`);
    expect(r.text).toContain(phrase);
    expect(r.text.split(" ")).toHaveLength(3);
  });

  it("the SAME dynamic word recognized twice appears twice, separated", () => {
    const builder = new ArabicTextBuilder();
    builder.appendLiteralText("\u0645\u0631\u062d\u0628\u0627");
    const r = builder.appendLiteralText("\u0645\u0631\u062d\u0628\u0627");
    expect(r.text).toBe("\u0645\u0631\u062d\u0628\u0627 \u0645\u0631\u062d\u0628\u0627");
  });

  it("a spelled letter after a dynamic word starts a new word; letters after letters do not", () => {
    const builder = new ArabicTextBuilder();
    builder.appendLiteralText("\u0645\u0631\u062d\u0628\u0627");
    builder.applyCommittedLabel("Alef");
    // boundary inserted once, before the first letter of the new word
    expect(builder.getText()).toBe("\u0645\u0631\u062d\u0628\u0627 \u0627");
    builder.applyCommittedLabel("Ba2");
    // the second letter continues the SAME word
    expect(builder.getText()).toBe("\u0645\u0631\u062d\u0628\u0627 \u0627\u0628");
  });

  it("an explicit Space before a dynamic word does not produce a double separator", () => {
    const builder = new ArabicTextBuilder();
    builder.applyCommittedLabel("Alef");
    builder.applyCommittedLabel("Space");
    const r = builder.appendLiteralText("\u0645\u0631\u062d\u0628\u0627");
    expect(r.text).toBe("\u0627 \u0645\u0631\u062d\u0628\u0627");
    expect(r.text).not.toContain("  ");
  });

  it("reset clears the word-boundary state as well as the text", () => {
    const builder = new ArabicTextBuilder();
    builder.appendLiteralText("\u0645\u0631\u062d\u0628\u0627");
    builder.reset();
    const r = builder.appendLiteralText("\u0634\u0643\u0631\u0627");
    expect(r.text).toBe("\u0634\u0643\u0631\u0627");
  });

  it("appendLiteralText composes correctly with Space and letters (existing behavior unaffected)", () => {
    const builder = new ArabicTextBuilder();
    builder.applyCommittedLabel("Alef");
    builder.applyCommittedLabel("Space");
    builder.appendLiteralText("\u0645\u0631\u062d\u0628\u0627");
    builder.applyCommittedLabel("Space");
    builder.applyCommittedLabel("Ba2");
    expect(builder.getText()).toBe("\u0627 \u0645\u0631\u062d\u0628\u0627 \u0628");
  });

  it("Delete after appendLiteralText removes only the last character, not the whole phrase", () => {
    const builder = new ArabicTextBuilder();
    builder.appendLiteralText("\u0645\u0631\u062d\u0628\u0627");
    const result = builder.applyCommittedLabel("Delete");
    expect(result.text).toBe("\u0645\u0631\u062d\u0628"); // last character removed, not the whole phrase
  });
});