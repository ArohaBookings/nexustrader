import { and, desc, eq, lt, sql } from "drizzle-orm";
import { databaseConfigured, getDb } from "@/db/client";
import {
  botSnapshots,
  commandRequests,
  janitorRuns,
  orderEvents,
  riskEvents,
  rollups,
  symbolSnapshots,
  telegramMessages,
  tradeEvents,
} from "@/db/schema";
import { demoOverview } from "@/lib/demo-data";
import type { EventInput, IngestPayload } from "@/lib/validation";

type Json = Record<string, unknown>;

type MemoryStore = {
  bots: Json[];
  symbols: Json[];
  trades: Json[];
  orders: Json[];
  risks: Json[];
  telegram: Json[];
  commands: Json[];
  janitor: Json[];
};

declare global {
  var __nexusTraderStore: MemoryStore | undefined;
}

function memory() {
  globalThis.__nexusTraderStore ??= {
    bots: [demoOverview.bot],
    symbols: demoOverview.symbols,
    trades: demoOverview.trades,
    orders: demoOverview.orders,
    risks: demoOverview.risks,
    telegram: [],
    commands: demoOverview.commands,
    janitor: [],
  };
  return globalThis.__nexusTraderStore;
}

function numeric(value: number | null | undefined) {
  return value === undefined || value === null ? null : String(value);
}

function dateOrNow(value: string | undefined) {
  return value ? new Date(value) : new Date();
}

function eventRow(event: EventInput) {
  return {
    externalId: event.id ?? null,
    occurredAt: dateOrNow(event.occurredAt),
    symbol: event.symbol ?? null,
    type: event.type,
    status: event.status ?? null,
    side: event.side ?? null,
    quantity: numeric(event.quantity),
    price: numeric(event.price),
    fee: numeric(event.fee),
    slippageBps: numeric(event.slippageBps),
    pnl: numeric(event.pnl),
    reason: event.reason ?? null,
    payload: event.payload,
  };
}

export async function ingestTelemetry(payload: IngestPayload) {
  const observedAt = dateOrNow(payload.observedAt);
  if (!databaseConfigured()) {
    const store = memory();
    if (payload.bot) store.bots.push({ ...payload.bot, observedAt: payload.bot.observedAt ?? observedAt.toISOString(), source: payload.source });
    store.symbols.push(...payload.symbols.map((item) => ({ ...item, observedAt: item.observedAt ?? observedAt.toISOString() })));
    store.trades.push(...payload.trades.map((item) => ({ ...item, occurredAt: item.occurredAt ?? observedAt.toISOString() })));
    store.orders.push(...payload.orders.map((item) => ({ ...item, occurredAt: item.occurredAt ?? observedAt.toISOString() })));
    store.risks.push(...payload.risks.map((item) => ({ ...item, occurredAt: item.occurredAt ?? observedAt.toISOString() })));
    store.risks.push(...payload.dataIntegrity.map((item) => ({ ...item, occurredAt: item.occurredAt ?? observedAt.toISOString() })));
    return {
      botSnapshots: payload.bot ? 1 : 0,
      symbolSnapshots: payload.symbols.length,
      trades: payload.trades.length,
      orders: payload.orders.length,
      risks: payload.risks.length + payload.dataIntegrity.length,
      mode: "memory",
    };
  }

  const db = getDb();
  if (payload.bot) {
    await db.insert(botSnapshots).values({
      observedAt: dateOrNow(payload.bot.observedAt ?? payload.observedAt),
      source: payload.source,
      equity: numeric(payload.bot.equity),
      balance: numeric(payload.bot.balance),
      pnlToday: numeric(payload.bot.pnlToday),
      drawdownPct: numeric(payload.bot.drawdownPct),
      queueDepth: payload.bot.queueDepth ?? null,
      session: payload.bot.session ?? null,
      killState: payload.bot.killState ?? null,
      openRiskPct: numeric(payload.bot.openRiskPct),
      payload: payload.bot.payload,
    });
  }
  if (payload.symbols.length) {
    await db.insert(symbolSnapshots).values(
      payload.symbols.map((item) => ({
        observedAt: dateOrNow(item.observedAt ?? payload.observedAt),
        symbol: item.symbol.toUpperCase(),
        strategy: item.strategy ?? null,
        state: item.state ?? null,
        confidence: numeric(item.confidence),
        spread: numeric(item.spread),
        openRiskPct: numeric(item.openRiskPct),
        blocker: item.blocker ?? null,
        thinking: item.thinking ?? null,
        payload: item.payload,
      })),
    );
  }
  if (payload.trades.length) await db.insert(tradeEvents).values(payload.trades.map(eventRow));
  if (payload.orders.length) await db.insert(orderEvents).values(payload.orders.map(eventRow));
  const riskRows = [...payload.risks, ...payload.dataIntegrity].map(eventRow);
  if (riskRows.length) await db.insert(riskEvents).values(riskRows);
  return {
    botSnapshots: payload.bot ? 1 : 0,
    symbolSnapshots: payload.symbols.length,
    trades: payload.trades.length,
    orders: payload.orders.length,
    risks: riskRows.length,
    mode: "database",
  };
}

export async function getOverview() {
  if (!databaseConfigured()) {
    const store = memory();
    return {
      ...demoOverview,
      bot: store.bots.at(-1) ?? demoOverview.bot,
      symbols: latestBySymbol(store.symbols),
      trades: store.trades.slice(-50).reverse(),
      orders: store.orders.slice(-50).reverse(),
      risks: store.risks.slice(-50).reverse(),
      commands: store.commands.slice(-20).reverse(),
    };
  }

  const db = getDb();
  const [bot] = await db.select().from(botSnapshots).orderBy(desc(botSnapshots.observedAt)).limit(1);
  const symbols = await db.select().from(symbolSnapshots).orderBy(desc(symbolSnapshots.observedAt)).limit(200);
  const trades = await db.select().from(tradeEvents).orderBy(desc(tradeEvents.occurredAt)).limit(50);
  const orders = await db.select().from(orderEvents).orderBy(desc(orderEvents.occurredAt)).limit(50);
  const risks = await db.select().from(riskEvents).orderBy(desc(riskEvents.occurredAt)).limit(50);
  const commands = await db.select().from(commandRequests).orderBy(desc(commandRequests.createdAt)).limit(20);
  return {
    bot: bot ?? demoOverview.bot,
    symbols: latestBySymbol(symbols as unknown as Json[]),
    trades,
    orders,
    risks,
    commands,
    equityCurve: demoOverview.equityCurve,
  };
}

export async function getSymbols() {
  return (await getOverview()).symbols;
}

export async function getTrades() {
  const overview = await getOverview();
  return { trades: overview.trades, orders: overview.orders };
}

export async function getRisk() {
  const overview = await getOverview();
  return { bot: overview.bot, risks: overview.risks, commands: overview.commands };
}

export async function recordTelegramMessage(input: {
  chatId: string;
  telegramUserId?: string;
  username?: string;
  direction: "in" | "out";
  text: string;
  payload?: Json;
}) {
  if (!databaseConfigured()) {
    memory().telegram.push({ ...input, createdAt: new Date().toISOString() });
    return;
  }
  await getDb().insert(telegramMessages).values({
    chatId: input.chatId,
    telegramUserId: input.telegramUserId ?? null,
    username: input.username ?? null,
    direction: input.direction,
    text: input.text,
    payload: input.payload ?? {},
  });
}

export async function createCommandRequest(input: {
  commandId: string;
  chatId: string;
  telegramUserId?: string;
  action: string;
  requestedText: string;
  modelResponse?: string;
  confirmationRequired: boolean;
  payload?: Json;
}) {
  const row = {
    commandId: input.commandId,
    chatId: input.chatId,
    telegramUserId: input.telegramUserId ?? null,
    action: input.action,
    requestedText: input.requestedText,
    modelResponse: input.modelResponse ?? null,
    confirmationRequired: input.confirmationRequired,
    payload: input.payload ?? {},
  };
  if (!databaseConfigured()) {
    memory().commands.push({ ...row, status: input.confirmationRequired ? "pending_confirmation" : "executed", createdAt: new Date().toISOString() });
    return row;
  }
  await getDb().insert(commandRequests).values(row);
  return row;
}

export async function findPendingCommand(commandId: string) {
  if (!databaseConfigured()) {
    return memory().commands.find((item) => item.commandId === commandId && item.status === "pending_confirmation") ?? null;
  }
  const [row] = await getDb()
    .select()
    .from(commandRequests)
    .where(and(eq(commandRequests.commandId, commandId), eq(commandRequests.status, "pending_confirmation")))
    .limit(1);
  return row ?? null;
}

export async function markCommandExecuted(commandId: string, result: Json) {
  if (!databaseConfigured()) {
    const row = memory().commands.find((item) => item.commandId === commandId);
    if (row) {
      row.status = "executed";
      row.executedAt = new Date().toISOString();
      row.executionResult = result;
    }
    return;
  }
  await getDb()
    .update(commandRequests)
    .set({ status: "executed", executedAt: new Date(), executionResult: result })
    .where(eq(commandRequests.commandId, commandId));
}

export async function runJanitor() {
  const now = new Date();
  const rawCutoff = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
  if (!databaseConfigured()) {
    const store = memory();
    const before = store.risks.length + store.orders.length + store.trades.length;
    store.risks = store.risks.filter((item) => new Date(String(item.occurredAt ?? item.createdAt ?? now)).getTime() >= rawCutoff.getTime());
    store.orders = store.orders.filter((item) => new Date(String(item.occurredAt ?? item.createdAt ?? now)).getTime() >= rawCutoff.getTime());
    store.trades = store.trades.filter((item) => new Date(String(item.occurredAt ?? item.createdAt ?? now)).getTime() >= rawCutoff.getTime());
    const after = store.risks.length + store.orders.length + store.trades.length;
    const result = { status: "ok", rawDeleted: before - after, rollupsWritten: 0, mode: "memory" };
    store.janitor.push({ ...result, finishedAt: now.toISOString() });
    return result;
  }

  const db = getDb();
  const [botRollup] = await db
    .select({
      avgEquity: sql<string>`avg(${botSnapshots.equity})`,
      maxQueue: sql<string>`max(${botSnapshots.queueDepth})`,
      count: sql<string>`count(*)`,
    })
    .from(botSnapshots)
    .where(lt(botSnapshots.observedAt, rawCutoff));
  let rollupsWritten = 0;
  if (botRollup && Number(botRollup.count) > 0) {
    await db.insert(rollups).values({
      bucket: "daily",
      bucketStart: rawCutoff,
      metric: "compressed_bot_snapshot",
      value: botRollup.avgEquity,
      payload: { maxQueue: botRollup.maxQueue, compressedRows: botRollup.count },
    });
    rollupsWritten = 1;
  }
  const deletedBots = await db.delete(botSnapshots).where(lt(botSnapshots.observedAt, rawCutoff));
  await db.delete(symbolSnapshots).where(lt(symbolSnapshots.observedAt, rawCutoff));
  await db.delete(orderEvents).where(and(lt(orderEvents.occurredAt, rawCutoff), sql`${orderEvents.status} not in ('filled', 'rejected')`));
  await db.delete(riskEvents).where(lt(riskEvents.occurredAt, rawCutoff));
  await db.insert(janitorRuns).values({
    finishedAt: now,
    rawDeleted: Number(deletedBots.rowCount ?? 0),
    rollupsWritten,
    status: "ok",
    payload: { rawCutoff: rawCutoff.toISOString() },
  });
  return { status: "ok", rawDeleted: Number(deletedBots.rowCount ?? 0), rollupsWritten, mode: "database" };
}

function latestBySymbol(rows: Json[]) {
  const latest = new Map<string, Json>();
  for (const row of rows) {
    const symbol = String(row.symbol ?? "").toUpperCase();
    if (!symbol) continue;
    const current = latest.get(symbol);
    const rowTime = new Date(String(row.observedAt ?? row.observed_at ?? row.createdAt ?? 0)).getTime();
    const currentTime = current ? new Date(String(current.observedAt ?? current.observed_at ?? current.createdAt ?? 0)).getTime() : -1;
    if (!current || rowTime >= currentTime) latest.set(symbol, row);
  }
  return [...latest.values()].sort((a, b) => String(a.symbol).localeCompare(String(b.symbol)));
}
