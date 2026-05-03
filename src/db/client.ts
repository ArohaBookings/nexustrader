import { neon } from "@neondatabase/serverless";
import { drizzle } from "drizzle-orm/neon-http";
import * as schema from "./schema";

type Db = ReturnType<typeof drizzle<typeof schema>>;

let cachedDb: Db | null = null;

export function getDb() {
  const url = process.env.DATABASE_URL;
  if (!url) {
    throw new Error("DATABASE_URL is not configured");
  }
  if (!cachedDb) {
    cachedDb = drizzle(neon(url), { schema });
  }
  return cachedDb;
}

export function databaseConfigured() {
  return Boolean(process.env.DATABASE_URL);
}
