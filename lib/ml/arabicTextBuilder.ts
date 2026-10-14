import { toArabicChar, isControlSign } from "./arabicLabels";

export type TextBuilderAction = "append" | "space" | "delete" | "finish" | "unknown";

export interface TextBuilderResult {
  text: string;
  action: TextBuilderAction;
}

export class ArabicTextBuilder {
  private text = "";
  /**
   * True when the most recent thing appended was a whole recognized word or
   * phrase (appendLiteralText), as opposed to a single spelled letter.
   *
   * This is what lets the builder tell "two separate dynamic signs" from "one
   * multi-word KArSL label" WITHOUT inspecting the label itself. A dedicated
   * phrase such as "إسعافات أولية" arrives as ONE string from ONE recognition
   * event, so its internal space is part of the label and never a decision
   * this class makes; a boundary is only ever inserted BETWEEN two separate
   * append calls. No label is special-cased, and no label is ever split.
   */
  private lastWasWholeWord = false;

  getText(): string {
    return this.text;
  }

  reset(): void {
    this.text = "";
    this.lastWasWholeWord = false;
  }

  /** True when the text already ends at a word boundary (or is empty), so no
   * separator is needed. */
  private atWordBoundary(): boolean {
    return this.text.length === 0 || /\s$/.test(this.text);
  }

  /**
   * Appends an already-resolved literal string verbatim -- used for a
   * recognized KArSL dedicated word/phrase (e.g. "إسعافات أولية"), which
   * arrives as one complete string from dynamicSignRecognizer.ts, not as
   * individual English class labels like applyCommittedLabel() expects.
   *
   * WORD BOUNDARIES. A separator is inserted BEFORE the literal when the text
   * so far does not already end at one, so two consecutively recognized
   * dynamic signs read as "مرحبا شكرا" rather than "مرحباشكرا". The literal
   * itself is still committed verbatim and is never split, so a multi-word
   * KArSL label keeps its own internal spacing and remains one sign. The same
   * boundary is inserted when a spelled letter follows a whole word, which is
   * why applyCommittedLabel() consults `lastWasWholeWord` too.
   *
   * An explicit Space sign still works exactly as before; it simply finds the
   * text already at a boundary and does not double the separator.
   */
  appendLiteralText(text: string): TextBuilderResult {
    if (!this.atWordBoundary()) this.text += " ";
    this.text += text;
    this.lastWasWholeWord = true;
    return { text: this.text, action: "append" };
  }

  applyCommittedLabel(label: string): TextBuilderResult {
    if (label === "Space") {
      this.text += " ";
      this.lastWasWholeWord = false;
      return { text: this.text, action: "space" };
    }
    if (label === "Delete") {
      this.text = this.text.slice(0, -1);
      this.lastWasWholeWord = false;
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
    // A letter that follows a whole recognized word starts a NEW word; letters
    // that follow letters keep spelling the current one.
    if (this.lastWasWholeWord && !this.atWordBoundary()) this.text += " ";
    this.text += char;
    this.lastWasWholeWord = false;
    return { text: this.text, action: "append" };
  }
}