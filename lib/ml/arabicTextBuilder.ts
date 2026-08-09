import { toArabicChar, isControlSign } from "./arabicLabels";

export type TextBuilderAction = "append" | "space" | "delete" | "finish" | "unknown";

export interface TextBuilderResult {
  text: string;
  action: TextBuilderAction;
}

/**
 * Builds Arabic text from a stream of COMMITTED sign labels (i.e. labels
 * that already passed through lib/ml/segmentation.ts's temporal stability
 * checks — this class has no opinion about frame-level noise, only about
 * turning one committed label into one character-level edit).
 *
 * Deliberately NOT a dictionary: there is no table mapping letter sequences
 * to whole words. "ب" + "ح" + "ب" + "ك" becomes "بحبك" purely because each
 * committed letter is appended to the running string, in order. This class
 * has no knowledge of Arabic vocabulary at all.
 */
export class ArabicTextBuilder {
  private text = "";

  getText(): string {
    return this.text;
  }

  reset(): void {
    this.text = "";
  }

  /**
   * Apply one committed class label.
   *  - A letter class (e.g. "Ba2") appends its Arabic character.
   *  - "Space" appends a literal space.
   *  - "Delete" removes the last character (letter or space) from the text.
   *  - "Finish" ends the current word/phrase per the segmentation state
   *    machine (it still requires normal temporal stability to commit like
   *    any other sign) but is a no-op on the text itself — there is no
   *    punctuation semantics defined for V1, so we don't invent one.
   *  - Anything else (shouldn't happen with a valid trained model, but
   *    defensively handled) is a no-op, reported as "unknown".
   */
  applyCommittedLabel(label: string): TextBuilderResult {
    if (label === "Space") {
      this.text += " ";
      return { text: this.text, action: "space" };
    }
    if (label === "Delete") {
      this.text = this.text.slice(0, -1);
      return { text: this.text, action: "delete" };
    }
    if (label === "Finish") {
      return { text: this.text, action: "finish" };
    }
    if (isControlSign(label)) {
      // Exhaustive per CONTROL_SIGNS, but keep a safe fallback.
      return { text: this.text, action: "unknown" };
    }

    const char = toArabicChar(label);
    if (char === null) {
      return { text: this.text, action: "unknown" };
    }
    this.text += char;
    return { text: this.text, action: "append" };
  }
}
