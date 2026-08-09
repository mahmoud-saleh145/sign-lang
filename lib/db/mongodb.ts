import { MongoClient, type Db } from "mongodb";

/**
 * SERVER-ONLY. Never import this file from a "use client" component — doing
 * so would either fail to build or, worse, leak DB logic into the client
 * bundle. Only app/api/** route handlers should import this.
 *
 * Uses the standard Next.js dev-mode-safe singleton pattern: in dev, hot
 * reload re-executes this module, so we stash the client promise on
 * globalThis to avoid opening a new connection on every reload.
 */

const MONGODB_URI = process.env.MONGODB_URI;
const DATABASE_NAME = process.env.DATABASE_NAME ?? "sign_translator";

declare global {
  var _mongoClientPromise: Promise<MongoClient> | undefined;
}

function getClientPromise(): Promise<MongoClient> {
  if (!MONGODB_URI) {
    throw new Error(
      "MONGODB_URI is not set. Copy .env.example to .env.local and fill it in."
    );
  }

  if (process.env.NODE_ENV === "development") {
    if (!global._mongoClientPromise) {
      const client = new MongoClient(MONGODB_URI);
      global._mongoClientPromise = client.connect();
    }
    return global._mongoClientPromise;
  }

  // Production: one client per server instance, no global stash needed.
  const client = new MongoClient(MONGODB_URI);
  return client.connect();
}

let cachedDb: Db | null = null;

/** Get the app's database. Throws a clear error if MONGODB_URI is missing. */
export async function getDb(): Promise<Db> {
  if (cachedDb) return cachedDb;
  const client = await getClientPromise();
  cachedDb = client.db(DATABASE_NAME);
  return cachedDb;
}

/** True if a MongoDB URI is configured. Lets routes fail gracefully instead of throwing. */
export function isDbConfigured(): boolean {
  return Boolean(MONGODB_URI);
}
