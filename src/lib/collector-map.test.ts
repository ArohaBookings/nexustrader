import { describe, expect, it } from "vitest";
import { bridgePayloadToIngest } from "@/lib/collector-map";

describe("bridge collector mapping", () => {
  it("maps bridge telemetry into lean ingest payload", () => {
    const payload = bridgePayloadToIngest({
      observedAt: "2026-01-01T00:00:00.000Z",
      health: { status: "ok" },
      stats: { equity: 101, daily_pnl: 1.5 },
      dashboard: {
        account: { equity: 101, balance: 100, free_margin: 95 },
        pairs: [{ symbol: "BTC", strategy: "baseline", state: "paper", confidence: 0.55 }],
        recent_events: [{ type: "order_fill", symbol: "BTC", status: "filled", fee: 0.02 }],
      },
    });
    expect(payload.bot?.equity).toBe(101);
    expect(payload.symbols).toHaveLength(1);
    expect(payload.orders).toHaveLength(1);
    expect(payload.bot?.payload.mt5_account).toMatchObject({ equity: 101, balance: 100 });
  });
});
