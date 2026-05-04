import { beforeEach, describe, expect, it, vi } from "vitest";
import { handleTelegramCommand } from "@/lib/commands";
import { ingestTelemetry } from "@/lib/repository";
import { resetMemoryStoreForTests } from "@/lib/test-store";

vi.mock("@/lib/bridge", () => ({
  executeBridgeControl: vi.fn(async (action: string) => ({ ok: true, action })),
}));

describe("Telegram command policy", () => {
  beforeEach(() => {
    delete process.env.DATABASE_URL;
    resetMemoryStoreForTests();
  });

  it("blocks trade placement and aggression requests", async () => {
    const reply = await handleTelegramCommand({ chatId: "1", text: "buy BTC now and increase leverage" });
    expect(reply).toContain("Blocked");
  });

  it("requires confirmation for resume", async () => {
    const reply = await handleTelegramCommand({ chatId: "1", text: "/resume" });
    expect(reply).toContain("Confirmation required");
    expect(reply).toContain("/confirm");
  });

  it("answers readonly status", async () => {
    const reply = await handleTelegramCommand({ chatId: "1", text: "/status" });
    expect(reply).toContain("Nexus Status");
    expect(reply).toContain("Equity");
  });

  it("reports edge-gated trajectory, blockers, and frequency without executing controls", async () => {
    await ingestTelemetry({
      source: "test_bridge",
      observedAt: "2026-01-01T00:00:00.000Z",
      bot: {
        observedAt: "2026-01-01T00:00:00.000Z",
        source: "bridge",
        equity: 101,
        payload: {
          institutional_intelligence: {
            policy: "edge_gated_no_forced_live_frequency",
            self_repair: {
              status: "soft_repair_available",
              soft_blockers: [{ symbol: "XAUUSD", reason: "stale quote" }],
              hard_rails: [],
              recommended_bridge_action: "refresh_state",
            },
          },
          trajectory_forecast: {
            current_equity: 101,
            short_goal_equity: 100000,
            short_goal_on_track: false,
            forecast_type: "speculative_target_path",
          },
          xau_btc_opportunity_pipeline: {
            live_frequency_forced: false,
            priority_symbols: [
              {
                symbol: "XAUUSD",
                shadow_target_10m: { low: 6, high: 8 },
                actual_candidates_last_10m: 2,
                actual_live_trades_last_10m: 0,
                live_gate: "blocked_by_edge_or_risk_gate",
              },
            ],
          },
          live_shadow_gap: {
            status: "collecting_or_aligned",
            max_gap_score: 0.12,
            priority_symbols: [{ symbol: "XAUUSD", status: "collecting" }],
          },
        },
      },
      symbols: [],
      trades: [],
      orders: [],
      risks: [],
      dataIntegrity: [],
    });

    const trajectory = await handleTelegramCommand({ chatId: "1", text: "/trajectory" });
    const blockers = await handleTelegramCommand({ chatId: "1", text: "/blockers" });
    const frequency = await handleTelegramCommand({ chatId: "1", text: "increase frequency on XAU" });

    expect(trajectory).toContain("Trajectory");
    expect(trajectory).toContain("not a sizing input");
    expect(blockers).toContain("Blockers");
    expect(blockers).toContain("refresh_state");
    expect(frequency).toContain("Frequency Policy");
    expect(frequency).toContain("cannot force entries");
    expect(frequency).toContain("Forced live frequency: false");
  });
});
