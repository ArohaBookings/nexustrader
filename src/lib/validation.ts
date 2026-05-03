import { z } from "zod";

export const botSnapshotSchema = z.object({
  observedAt: z.string().datetime().optional(),
  source: z.string().default("bridge"),
  equity: z.number().optional(),
  balance: z.number().optional(),
  pnlToday: z.number().optional(),
  drawdownPct: z.number().optional(),
  queueDepth: z.number().int().nonnegative().optional(),
  session: z.string().optional(),
  killState: z.string().optional(),
  openRiskPct: z.number().optional(),
  payload: z.record(z.string(), z.unknown()).default({}),
});

export const symbolSnapshotSchema = z.object({
  symbol: z.string().min(1),
  observedAt: z.string().datetime().optional(),
  strategy: z.string().optional(),
  state: z.string().optional(),
  confidence: z.number().optional(),
  spread: z.number().optional(),
  openRiskPct: z.number().optional(),
  blocker: z.string().optional(),
  thinking: z.string().optional(),
  payload: z.record(z.string(), z.unknown()).default({}),
});

export const eventSchema = z.object({
  id: z.string().optional(),
  symbol: z.string().optional(),
  type: z.string().min(1),
  status: z.string().optional(),
  side: z.string().optional(),
  quantity: z.number().optional(),
  price: z.number().optional(),
  fee: z.number().optional(),
  slippageBps: z.number().optional(),
  pnl: z.number().optional(),
  reason: z.string().optional(),
  occurredAt: z.string().datetime().optional(),
  payload: z.record(z.string(), z.unknown()).default({}),
});

export const ingestSchema = z.object({
  source: z.string().default("collector"),
  observedAt: z.string().datetime().optional(),
  bot: botSnapshotSchema.optional(),
  symbols: z.array(symbolSnapshotSchema).default([]),
  trades: z.array(eventSchema).default([]),
  orders: z.array(eventSchema).default([]),
  risks: z.array(eventSchema).default([]),
  dataIntegrity: z.array(eventSchema).default([]),
});

export type IngestPayload = z.infer<typeof ingestSchema>;
export type BotSnapshotInput = z.infer<typeof botSnapshotSchema>;
export type SymbolSnapshotInput = z.infer<typeof symbolSnapshotSchema>;
export type EventInput = z.infer<typeof eventSchema>;

export const fundedConfigSchema = z.object({
  enabled: z.boolean().default(false),
  group: z.string().min(1).default("custom"),
  phase: z.string().min(1).default("evaluation"),
  startingBalance: z.number().positive().default(100),
  profitTargetPct: z.number().min(0).max(5).default(0.08),
  dailyDrawdownPct: z.number().min(0).max(1).default(0.05),
  maxDrawdownPct: z.number().min(0).max(1).default(0.10),
  trailingDrawdown: z.boolean().default(false),
  baseRiskPct: z.number().min(0).max(0.05).default(0.005),
  maxOpenRiskPct: z.number().min(0).max(0.20).default(0.015),
  dailyResetTimezone: z.string().min(1).default("Australia/Sydney"),
});

export type FundedConfigInput = z.infer<typeof fundedConfigSchema>;

export const telegramUpdateSchema = z.object({
  update_id: z.number().optional(),
  message: z
    .object({
      message_id: z.number().optional(),
      text: z.string().optional(),
      date: z.number().optional(),
      chat: z.object({ id: z.union([z.string(), z.number()]), type: z.string().optional() }),
      from: z
        .object({
          id: z.union([z.string(), z.number()]),
          username: z.string().optional(),
          first_name: z.string().optional(),
        })
        .optional(),
    })
    .optional(),
});

export type TelegramUpdate = z.infer<typeof telegramUpdateSchema>;
