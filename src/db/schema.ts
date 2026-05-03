import {
  boolean,
  index,
  integer,
  jsonb,
  numeric,
  pgTable,
  serial,
  text,
  timestamp,
  uniqueIndex,
} from "drizzle-orm/pg-core";

export const botSnapshots = pgTable(
  "bot_snapshots",
  {
    id: serial("id").primaryKey(),
    observedAt: timestamp("observed_at", { withTimezone: true }).notNull().defaultNow(),
    source: text("source").notNull().default("bridge"),
    equity: numeric("equity", { precision: 18, scale: 6 }),
    balance: numeric("balance", { precision: 18, scale: 6 }),
    pnlToday: numeric("pnl_today", { precision: 18, scale: 6 }),
    drawdownPct: numeric("drawdown_pct", { precision: 12, scale: 8 }),
    queueDepth: integer("queue_depth"),
    session: text("session"),
    killState: text("kill_state"),
    openRiskPct: numeric("open_risk_pct", { precision: 12, scale: 8 }),
    payload: jsonb("payload").notNull().default({}),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => ({
    observedIdx: index("bot_snapshots_observed_idx").on(table.observedAt),
  }),
);

export const symbolSnapshots = pgTable(
  "symbol_snapshots",
  {
    id: serial("id").primaryKey(),
    observedAt: timestamp("observed_at", { withTimezone: true }).notNull().defaultNow(),
    symbol: text("symbol").notNull(),
    strategy: text("strategy"),
    state: text("state"),
    confidence: numeric("confidence", { precision: 12, scale: 8 }),
    spread: numeric("spread", { precision: 18, scale: 8 }),
    openRiskPct: numeric("open_risk_pct", { precision: 12, scale: 8 }),
    blocker: text("blocker"),
    thinking: text("thinking"),
    payload: jsonb("payload").notNull().default({}),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => ({
    symbolObservedIdx: index("symbol_snapshots_symbol_observed_idx").on(table.symbol, table.observedAt),
  }),
);

export const tradeEvents = pgTable("trade_events", {
  id: serial("id").primaryKey(),
  externalId: text("external_id"),
  occurredAt: timestamp("occurred_at", { withTimezone: true }).notNull().defaultNow(),
  symbol: text("symbol"),
  type: text("type").notNull(),
  status: text("status"),
  side: text("side"),
  quantity: numeric("quantity", { precision: 18, scale: 8 }),
  price: numeric("price", { precision: 18, scale: 8 }),
  fee: numeric("fee", { precision: 18, scale: 8 }),
  pnl: numeric("pnl", { precision: 18, scale: 8 }),
  payload: jsonb("payload").notNull().default({}),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const orderEvents = pgTable("order_events", {
  id: serial("id").primaryKey(),
  externalId: text("external_id"),
  occurredAt: timestamp("occurred_at", { withTimezone: true }).notNull().defaultNow(),
  symbol: text("symbol"),
  type: text("type").notNull(),
  status: text("status"),
  side: text("side"),
  quantity: numeric("quantity", { precision: 18, scale: 8 }),
  price: numeric("price", { precision: 18, scale: 8 }),
  fee: numeric("fee", { precision: 18, scale: 8 }),
  slippageBps: numeric("slippage_bps", { precision: 12, scale: 6 }),
  reason: text("reason"),
  payload: jsonb("payload").notNull().default({}),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const riskEvents = pgTable("risk_events", {
  id: serial("id").primaryKey(),
  occurredAt: timestamp("occurred_at", { withTimezone: true }).notNull().defaultNow(),
  symbol: text("symbol"),
  type: text("type").notNull(),
  status: text("status"),
  reason: text("reason"),
  payload: jsonb("payload").notNull().default({}),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const telegramMessages = pgTable("telegram_messages", {
  id: serial("id").primaryKey(),
  chatId: text("chat_id").notNull(),
  telegramUserId: text("telegram_user_id"),
  username: text("username"),
  direction: text("direction").notNull(),
  text: text("text").notNull(),
  payload: jsonb("payload").notNull().default({}),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const commandRequests = pgTable(
  "command_requests",
  {
    id: serial("id").primaryKey(),
    commandId: text("command_id").notNull(),
    chatId: text("chat_id").notNull(),
    telegramUserId: text("telegram_user_id"),
    action: text("action").notNull(),
    status: text("status").notNull().default("pending_confirmation"),
    requestedText: text("requested_text").notNull(),
    modelResponse: text("model_response"),
    confirmationRequired: boolean("confirmation_required").notNull().default(true),
    executedAt: timestamp("executed_at", { withTimezone: true }),
    executionResult: jsonb("execution_result").notNull().default({}),
    payload: jsonb("payload").notNull().default({}),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => ({
    commandIdUnique: uniqueIndex("command_requests_command_id_unique").on(table.commandId),
  }),
);

export const rollups = pgTable("rollups", {
  id: serial("id").primaryKey(),
  bucket: text("bucket").notNull(),
  bucketStart: timestamp("bucket_start", { withTimezone: true }).notNull(),
  metric: text("metric").notNull(),
  value: numeric("value", { precision: 18, scale: 8 }),
  payload: jsonb("payload").notNull().default({}),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
});

export const janitorRuns = pgTable("janitor_runs", {
  id: serial("id").primaryKey(),
  startedAt: timestamp("started_at", { withTimezone: true }).notNull().defaultNow(),
  finishedAt: timestamp("finished_at", { withTimezone: true }),
  rawDeleted: integer("raw_deleted").notNull().default(0),
  rollupsWritten: integer("rollups_written").notNull().default(0),
  status: text("status").notNull().default("ok"),
  payload: jsonb("payload").notNull().default({}),
});
