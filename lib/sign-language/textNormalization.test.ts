import { describe, it, expect } from "vitest";
import { normalizeArabicText } from "./textNormalization";

describe("normalizeArabicText", () => {
    it("passes through plain, already-normal Arabic text unchanged", () => {
        const result = normalizeArabicText("\u0645\u0631\u062d\u0628\u0627"); // مرحبا
        expect(result.text).toBe("\u0645\u0631\u062d\u0628\u0627");
        expect(result.notes).toHaveLength(0);
    });

    it("strips tashkeel diacritics", () => {
        // مَرْحَبًا with fatha/sukun/tanween marks -> مرحبا
        const withDiacritics = "\u0645\u064e\u0631\u0652\u062d\u064e\u0628\u064b\u0627";
        const result = normalizeArabicText(withDiacritics);
        expect(result.text).toBe("\u0645\u0631\u062d\u0628\u0627");
        expect(result.notes.some((n) => n.reason.includes("Diacritics"))).toBe(true);
    });

    it("maps hamza-carrying alef variants to plain Alef, with a documented note", () => {
        const result = normalizeArabicText("\u0623\u0646\u0627"); // أنا
        expect(result.text).toBe("\u0627\u0646\u0627"); // انا
        expect(result.notes.some((n) => n.original === "\u0623")).toBe(true);
    });

    it("maps teh marbuta to Ha2 (ه), with a documented note", () => {
        const result = normalizeArabicText("\u0645\u062f\u0631\u0633\u0629"); // مدرسة
        expect(result.text).toBe("\u0645\u062f\u0631\u0633\u0647"); // مدرسه
        expect(result.notes.some((n) => n.original === "\u0629")).toBe(true);
    });

    it("collapses whitespace and trims", () => {
        const result = normalizeArabicText("  \u0627\u0646\u0627    \u0628\u062d\u0628\u0643  ");
        expect(result.text).toBe("\u0627\u0646\u0627 \u0628\u062d\u0628\u0643");
    });

    it("empty input normalizes to empty text with no notes", () => {
        const result = normalizeArabicText("");
        expect(result.text).toBe("");
        expect(result.notes).toHaveLength(0);
    });
});