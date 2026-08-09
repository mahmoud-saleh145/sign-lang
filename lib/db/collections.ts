import { getDb } from "./mongodb";
import type {
  LanguageDocument,
  SignDocument,
  ModelMetadataDocument,
  TranslationSessionDocument,
  CollectedSampleDocument,
} from "./types";

export async function languagesCollection() {
  return (await getDb()).collection<LanguageDocument>("languages");
}

export async function signsCollection() {
  return (await getDb()).collection<SignDocument>("signs");
}

export async function modelMetadataCollection() {
  return (await getDb()).collection<ModelMetadataDocument>("modelMetadata");
}

export async function translationSessionsCollection() {
  return (await getDb()).collection<TranslationSessionDocument>("translationSessions");
}

export async function collectedSamplesCollection() {
  return (await getDb()).collection<CollectedSampleDocument>("collectedSamples");
}
