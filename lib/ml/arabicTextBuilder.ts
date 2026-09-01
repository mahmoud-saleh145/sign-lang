import { toArabicChar, isControlSign } from "./arabicLabels";

export type TextBuilderAction = "append" | "space" | "delete" | "finish" | "unknown";

export interface TextBuilderResult {
  text: string;
  action: TextBuilderAction;
}

export class ArabicTextBuilder {
  private text = "";

  getText(): string {
    return this.text;
  }

  reset(): void {
    this.text = "";
  }

  /**
   * Appends an already-resolved literal string verbatim -- used for a
   * recognized KArSL dedicated word/phrase (e.g. "إسعافات أولية"), which
   * arrives as one complete string from dynamicSignRecognizer.ts, not as
   * individual English class labels like applyCommittedLabel() expects.
   * No spacing is added automatically around it, consistent with the rest
   * of this builder: Space is the only thing that ever inserts a space.
   */
  appendLiteralText(text: string): TextBuilderResult {
    this.text += text;
    return { text: this.text, action: "append" };
  }

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