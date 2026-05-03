import { describe, expect, it } from "vitest";
import { calculateFundedStatus, DEFAULT_FUNDED_CONFIG } from "@/lib/funded-mode";

describe("funded mode calculations", () => {
  it("uses MT5 day-start and day-high equity for funded guard buffers", () => {
    const status = calculateFundedStatus(
      {
        ...DEFAULT_FUNDED_CONFIG,
        enabled: true,
        startingBalance: 100,
        profitTargetPct: 0.08,
        dailyDrawdownPct: 0.05,
        maxDrawdownPct: 0.10,
      },
      {
        equity: 104,
        balance: 102,
        payload: {
          stats: {
            active_bridge_context: { account: "MT5-1", magic: 42 },
            account_scaling: { equity: 104, balance: 102, free_margin: 99, high_watermark_equity: 106 },
            latest_account_snapshot: { account: "MT5-1", magic: 42, equity: 104, balance: 102, total_open_positions: 2 },
            risk_state: { day_start_equity: 103, day_high_equity: 106, daily_dd_pct_live: 0.0188 },
          },
        },
      },
    );

    expect(status.mt5Derived).toBe(true);
    expect(status.account.account).toBe("MT5-1");
    expect(status.dailyLossFloor).toBeCloseTo(97.85);
    expect(status.dailyLossRemainingUsd).toBeCloseTo(6.15);
    expect(status.neededToPass).toBeCloseTo(4);
    expect(status.maxRiskPerTradeUsd).toBeGreaterThan(0);
  });

  it("hard stops when the live MT5 equity breaches a funded floor", () => {
    const status = calculateFundedStatus(
      {
        ...DEFAULT_FUNDED_CONFIG,
        enabled: true,
        startingBalance: 100,
        dailyDrawdownPct: 0.05,
        maxDrawdownPct: 0.10,
      },
      {
        equity: 94.5,
        payload: {
          stats: {
            account_scaling: { equity: 94.5, balance: 100 },
            latest_account_snapshot: { equity: 94.5, balance: 100 },
            risk_state: { day_start_equity: 100, day_high_equity: 101 },
          },
        },
      },
    );

    expect(status.status).toBe("hard_stop");
    expect(status.riskThrottle).toBe(0);
    expect(status.maxRiskPerTradeUsd).toBe(0);
  });
});
