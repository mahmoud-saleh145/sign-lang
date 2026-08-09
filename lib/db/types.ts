import type { ObjectId } from "mongodb";

/** Languages supported by the app (spec section 28). */
export interface LanguageDocument {
  _id?: ObjectId;
  code: "ArSL" | "ASL";
  displayName: string;
  isActive: boolean;
}

/** One recognizable sign/class within a language. */
export interface SignDocument {
  _id?: ObjectId;
  languageCode: string;
  classId: string; // matches a class name in model weights.json, e.g. "Alef"
  displayLabel: string;
  category: "letter" | "control";
}

/** Cached copy of the currently-deployed model's metadata, for API consumers. */
export interface ModelMetadataDocument {
  _id?: ObjectId;
  modelVersion: string;
  language: string;
  isActive: boolean;
  metadata: Record<string, unknown>; // the full metadata.json contents
  publishedAt: Date;
}

/**
 * A translation session SUMMARY only — never raw video/audio (spec section
 * 28/35). Recorded when a session ends, from data already visible client-side
 * (transcript length, duration), not from camera/mic streams themselves.
 */
export interface TranslationSessionDocument {
  _id?: ObjectId;
  sessionId: string;
  mode: "sign" | "speech";
  languageCode: string;
  startedAt: Date;
  endedAt: Date;
  committedSignCount: number;
  transcriptCharCount: number;
}

/**
 * A collected training sample from the dataset collection tool (spec
 * sections 12/13). Stores landmark sequences + metadata only — never raw
 * video, per spec section 12 ("Do NOT save raw video unless necessary").
 */
export interface CollectedSampleDocument {
  _id?: ObjectId;
  label: string;
  languageCode: string;
  landmarkSequence: Array<Array<{ x: number; y: number; z: number }>>; // frames x 21 landmarks
  frameCount: number;
  signerSessionId: string;
  capturedAt: Date;
}
