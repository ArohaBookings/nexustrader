import { z } from "zod";

const serverEnvSchema = z.object({
  DATABASE_URL: z.string().min(1).optional(),
  APP_ADMIN_PASSWORD: z.string().min(8).optional(),
  SESSION_SECRET: z.string().min(16).optional(),
  INGEST_API_KEY: z.string().min(8).optional(),
  TELEGRAM_BOT_TOKEN: z.string().min(1).optional(),
  TELEGRAM_WEBHOOK_SECRET: z.string().min(8).optional(),
  OPENAI_API_KEY: z.string().min(1).optional(),
  BRIDGE_API_BASE_URL: z.string().url().optional(),
  BRIDGE_TOKEN: z.string().optional(),
  BRIDGE_DASHBOARD_PASSWORD: z.string().optional(),
  JANITOR_SECRET: z.string().min(8).optional(),
});

export type ServerEnv = z.infer<typeof serverEnvSchema>;

export function getEnv(): ServerEnv {
  return serverEnvSchema.parse(process.env);
}

export function requiredEnv(name: keyof ServerEnv) {
  const value = getEnv()[name];
  if (!value) {
    throw new Error(`Missing required env var: ${name}`);
  }
  return value;
}

export function hasDatabaseUrl() {
  return Boolean(getEnv().DATABASE_URL);
}
