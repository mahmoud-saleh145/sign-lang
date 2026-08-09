import type { CollectedSampleDocument, TranslationSessionDocument } from "./types";

export interface ValidationResult<T> {
  valid: boolean;
  errors: string[];
  data?: T;
}

const MAX_SEQUENCE_FRAMES = 300; // ~10s at 30fps; generous upper bound against abuse
const MIN_SEQUENCE_FRAMES = 3;

/** Validates a POST /api/dataset/samples request body. */
export function validateCollectedSample(
  body: unknown
): ValidationResult<Omit<CollectedSampleDocument, "_id" | "capturedAt">> {
  const errors: string[] = [];

  if (typeof body !== "object" || body === null) {
    return { valid: false, errors: ["Request body must be a JSON object."] };
  }
  const b = body as Record<string, unknown>;

  if (typeof b.label !== "string" || b.label.trim().length === 0) {
    errors.push("label must be a non-empty string.");
  }
  if (typeof b.languageCode !== "string" || b.languageCode.trim().length === 0) {
    errors.push("languageCode must be a non-empty string.");
  }
  if (typeof b.signerSessionId !== "string" || b.signerSessionId.trim().length === 0) {
    errors.push("signerSessionId must be a non-empty string.");
  }

  if (!Array.isArray(b.landmarkSequence)) {
    errors.push("landmarkSequence must be an array of frames.");
  } else {
    const seq = b.landmarkSequence;
    if (seq.length < MIN_SEQUENCE_FRAMES) {
      errors.push(`landmarkSequence must have at least ${MIN_SEQUENCE_FRAMES} frames.`);
    }
    if (seq.length > MAX_SEQUENCE_FRAMES) {
      errors.push(`landmarkSequence must have at most ${MAX_SEQUENCE_FRAMES} frames.`);
    }
    for (const frame of seq) {
      if (!Array.isArray(frame) || frame.length !== 21) {
        errors.push("Each frame must contain exactly 21 landmarks.");
        break;
      }
      for (const p of frame) {
        if (
          typeof p !== "object" ||
          p === null ||
          typeof (p as Record<string, unknown>).x !== "number" ||
          typeof (p as Record<string, unknown>).y !== "number" ||
          typeof (p as Record<string, unknown>).z !== "number"
        ) {
          errors.push("Each landmark must be {x, y, z} numbers.");
          break;
        }
      }
    }
  }

  if (errors.length > 0) return { valid: false, errors };

  return {
    valid: true,
    errors: [],
    data: {
      label: (b.label as string).trim(),
      languageCode: (b.languageCode as string).trim(),
      signerSessionId: (b.signerSessionId as string).trim(),
      landmarkSequence: b.landmarkSequence as CollectedSampleDocument["landmarkSequence"],
      frameCount: (b.landmarkSequence as unknown[]).length,
    },
  };
}

/** Validates a POST /api/sessions request body. Summary metadata only — never raw media. */
export function validateTranslationSession(
  body: unknown
): ValidationResult<Omit<TranslationSessionDocument, "_id">> {
  const errors: string[] = [];

  if (typeof body !== "object" || body === null) {
    return { valid: false, errors: ["Request body must be a JSON object."] };
  }
  const b = body as Record<string, unknown>;

  if (typeof b.sessionId !== "string" || b.sessionId.trim().length === 0) {
    errors.push("sessionId must be a non-empty string.");
  }
  if (b.mode !== "sign" && b.mode !== "speech") {
    errors.push("mode must be 'sign' or 'speech'.");
  }
  if (typeof b.languageCode !== "string" || b.languageCode.trim().length === 0) {
    errors.push("languageCode must be a non-empty string.");
  }
  if (typeof b.startedAt !== "string" || Number.isNaN(Date.parse(b.startedAt))) {
    errors.push("startedAt must be an ISO date string.");
  }
  if (typeof b.endedAt !== "string" || Number.isNaN(Date.parse(b.endedAt))) {
    errors.push("endedAt must be an ISO date string.");
  }
  if (typeof b.committedSignCount !== "number" || b.committedSignCount < 0) {
    errors.push("committedSignCount must be a non-negative number.");
  }
  if (typeof b.transcriptCharCount !== "number" || b.transcriptCharCount < 0) {
    errors.push("transcriptCharCount must be a non-negative number.");
  }

  if (errors.length > 0) return { valid: false, errors };

  return {
    valid: true,
    errors: [],
    data: {
      sessionId: (b.sessionId as string).trim(),
      mode: b.mode as "sign" | "speech",
      languageCode: (b.languageCode as string).trim(),
      startedAt: new Date(b.startedAt as string),
      endedAt: new Date(b.endedAt as string),
      committedSignCount: b.committedSignCount as number,
      transcriptCharCount: b.transcriptCharCount as number,
    },
  };
}
