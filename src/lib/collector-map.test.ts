import { describe, expect, it } from "vitest";
import { bridgePayloadToIngest } from "@/lib/collector-map";

describe("bridge collector mapping", () => {
  it("maps bridge telemetry into lean ingest payload", () => {
    const payload = bridgePayloadToIngest({
      observedAt: "2026-01-01T00:00:00.000Z",
      health: { status: "ok" },
      stats: { equity: 101, daily_pnl: 1.5 },
      dashboard: {
        summary: { equity: 111, balance: 110, daily_pnl: 2.5, queue_depth: 3, session: "LONDON" },
        account: { equity: 101, balance: 100, free_margin: 95 },
        institutional_intelligence: { policy: "edge_gated_no_forced_live_frequency" },
        training_bootstrap_status: { status: "trained_observe_validate", live_risk_expansion_allowed: false },
        data_quality: { status: "tradable_native_backed", proxy_history_labeled: true },
        promotion_audit: { promotion_allowed: false, reason: "validation_pending" },
        trajectory_forecast: { forecast_type: "speculative_target_path", current_equity: 101 },
        xau_btc_opportunity_pipeline: { live_frequency_forced: false, priority_symbols: [{ symbol: "XAUUSD" }] },
        live_shadow_gap: { status: "collecting_or_aligned", max_gap_score: 0 },
        pairs: [{ symbol: "BTC", strategy: "baseline", state: "paper", confidence: 0.55 }],
        recent_events: [{ type: "order_fill", symbol: "BTC", status: "filled", fee: 0.02 }],
      },
    });
    expect(payload.bot?.equity).toBe(101);
    expect(payload.bot?.balance).toBe(110);
    expect(payload.bot?.queueDepth).toBe(3);
    expect(payload.bot?.session).toBe("LONDON");
    expect(payload.symbols).toHaveLength(1);
    expect(payload.orders).toHaveLength(1);
    expect(payload.bot?.payload.mt5_account).toMatchObject({ equity: 101, balance: 100 });
    expect(payload.bot?.payload.institutional_intelligence).toMatchObject({ policy: "edge_gated_no_forced_live_frequency" });
    expect(payload.bot?.payload.xau_btc_opportunity_pipeline).toMatchObject({ live_frequency_forced: false });
    expect(payload.bot?.payload.trajectory_forecast).toMatchObject({ forecast_type: "speculative_target_path" });
  });
});
