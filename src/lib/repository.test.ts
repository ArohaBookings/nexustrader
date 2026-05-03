import { beforeEach, describe, expect, it } from "vitest";
import { ingestTelemetry, getOverview, runJanitor } from "@/lib/repository";
import { resetMemoryStoreForTests } from "@/lib/test-store";

describe("repository memory fallback", () => {
  beforeEach(() => {
    delete process.env.DATABASE_URL;
    resetMemoryStoreForTests();
  });

  it("ingests compact snapshots and events deterministically", async () => {
    const observedAt = "2026-01-01T00:00:00.000Z";
    const result = await ingestTelemetry({
      source: "test",
      observedAt,
      bot: { observedAt, source: "test", equity: 125, payload: {} },
      symbols: [{ symbol: "btc", observedAt, state: "paper", confidence: 0.7, payload: {} }],
      trades: [{ type: "trade_close", status: "closed", symbol: "BTC", pnl: 2, occurredAt: observedAt, payload: {} }],
      orders: [{ type: "order_fill", status: "filled", symbol: "BTC", fee: 0.1, slippageBps: 1.2, occurredAt: observedAt, payload: {} }],
      risks: [],
      dataIntegrity: [],
    });
    expect(result).toMatchObject({ mode: "memory", botSnapshots: 1, symbolSnapshots: 1, trades: 1, orders: 1 });
    const overview = await getOverview();
    expect(overview.bot.equity).toBe(125);
    expect(overview.symbols.some((symbol) => symbol.symbol === "btc")).toBe(true);
  });

  it("janitor deletes old raw memory events without deleting key demo state", async () => {
    const old = "2020-01-01T00:00:00.000Z";
    await ingestTelemetry({
      source: "test",
      observedAt: old,
      trades: [{ type: "trade_close", status: "closed", symbol: "ETH", occurredAt: old, payload: {} }],
      orders: [{ type: "order_open", status: "working", symbol: "ETH", occurredAt: old, payload: {} }],
      risks: [{ type: "stale_data", status: "blocked", occurredAt: old, payload: {} }],
      symbols: [],
      dataIntegrity: [],
    });
    const result = await runJanitor();
    expect(result.rawDeleted).toBeGreaterThanOrEqual(3);
  });
});

