/**
 * Maps the trained model's internal class names to their real Arabic
 * characters, for display only. The model itself, its weights, and its
 * class names (public/models/arsl-31/metadata.json "classes") are NOT
 * changed by this file — this is purely a presentation-layer lookup.
 *
 * Source: the dataset's own published documentation (Zenodo record
 * 18363162, "Sign Inventory" section), not an assumption:
 *   Alef (ا), Ba2 (ب), Ta2 (ت), Tha2 (ث), Jim (ج), 7a2 (ح), Kha2 (خ),
 *   Dal (د), Thal (ذ), Ra2 (ر), Zayn (ز), Sin (س), Chin (ش), SSad (ص),
 *   DDad (ض), TTa2 (ط), TTha2 (ظ), 3ayn (ع), Ghayn (غ), Fa2 (ف), 9af (ق),
 *   Kaf (ك), Lam (ل), Mim (م), Noon (ن), Ha2 (ه), Waw (و), Ya2 (ي)
 * Verified against public/models/arsl-31/metadata.json's 31 classes: exact
 * match, 28 letters + Space/Delete/Finish control signs.
 */
export const ARABIC_LETTER_MAP: Readonly<Record<string, string>> = {
  Alef: "\u0627", // ا
  Ba2: "\u0628", // ب
  Ta2: "\u062a", // ت
  Tha2: "\u062b", // ث
  Jim: "\u062c", // ج
  "7a2": "\u062d", // ح
  Kha2: "\u062e", // خ
  Dal: "\u062f", // د
  Thal: "\u0630", // ذ
  Ra2: "\u0631", // ر
  Zayn: "\u0632", // ز
  Sin: "\u0633", // س
  Chin: "\u0634", // ش
  SSad: "\u0635", // ص
  DDad: "\u0636", // ض
  TTa2: "\u0637", // ط
  TTha2: "\u0638", // ظ
  "3ayn": "\u0639", // ع
  Ghayn: "\u063a", // غ
  Fa2: "\u0641", // ف
  "9af": "\u0642", // ق
  Kaf: "\u0643", // ك
  Lam: "\u0644", // ل
  Mim: "\u0645", // م
  Noon: "\u0646", // ن
  Ha2: "\u0647", // ه
  Waw: "\u0648", // و
  Ya2: "\u064a", // ي
};

export const CONTROL_SIGNS = ["Space", "Delete", "Finish"] as const;
export type ControlSign = (typeof CONTROL_SIGNS)[number];

export function isControlSign(label: string): label is ControlSign {
  return (CONTROL_SIGNS as readonly string[]).includes(label);
}

/** Returns the Arabic character for a letter class, or null for control/unknown labels. */
export function toArabicChar(label: string): string | null {
  return ARABIC_LETTER_MAP[label] ?? null;
}

/** Short symbol used to represent a control sign in debug/telemetry UI. */
export function controlSignSymbol(label: ControlSign): string {
  switch (label) {
    case "Space":
      return "\u23b5"; // ⎵
    case "Delete":
      return "\u232b"; // ⌫
    case "Finish":
      return "\u2713"; // ✓
  }
}
