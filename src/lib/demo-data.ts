import { subMinutes } from "date-fns";

const now = new Date();

export const demoEquityCurve = Array.from({ length: 48 }).map((_, index) => {
  const t = subMinutes(now, (47 - index) * 30);
  const base = 100 + index * 1.9;
  const wave = Math.sin(index / 3) * 5 + Math.cos(index / 5) * 3;
  const equity = Math.max(68, base + wave);
  return {
    timestamp: t.toISOString(),
    equity,
    target: 100 * Math.pow(1000, index / 47),
    downside: equity * (0.74 + index * 0.002),
    upside: equity * (1.16 + index * 0.018),
  };
});

export const demoOverview = {
  bot: {
    observedAt: now.toISOString(),
    source: "demo",
    equity: 189.42,
    balance: 184.11,
    pnlToday: 14.33,
    drawdownPct: -0.042,
    queueDepth: 2,
    session: "TOKYO",
    killState: "NONE",
    openRiskPct: 0.018,
    payload: {
      watchdog_state: "OK",
      bridge_status: "UP",
      current_daily_state: "DAILY_NORMAL",
      live_control_scope: "ops_controls_only",
    },
  },
  symbols: [
    {
      symbol: "BTC",
      strategy: "moving_average_crossover",
      state: "paper_observe",
      confidence: 0.64,
      spread: 0.7,
      openRiskPct: 0.008,
      blocker: "",
      thinking: "Trend bias positive, depth acceptable, no live order permission.",
      payload: { native_data: true, stale: false },
    },
    {
      symbol: "ETH",
      strategy: "baseline_watch",
      state: "hold",
      confidence: 0.52,
      spread: 0.9,
      openRiskPct: 0.004,
      blocker: "awaiting_walk_forward_confirmation",
      thinking: "Momentum is mixed; simulator holding until out-of-sample confirms.",
      payload: { native_data: true, stale: false },
    },
    {
      symbol: "XAUUSD",
      strategy: "existing_mT5_bridge",
      state: "managed",
      confidence: 0.71,
      spread: 22,
      openRiskPct: 0.006,
      blocker: "",
      thinking: "Existing bridge healthy; ops controls only from Nexus dashboard.",
      payload: { bridge_source: true },
    },
  ],
  trades: [
    { id: "tr-1", symbol: "BTC", type: "paper_close", status: "closed", side: "buy", pnl: 4.2, fee: 0.09, occurredAt: subMinutes(now, 64).toISOString() },
    { id: "tr-2", symbol: "ETH", type: "paper_reject", status: "rejected", side: "buy", pnl: 0, fee: 0, occurredAt: subMinutes(now, 44).toISOString() },
    { id: "tr-3", symbol: "XAUUSD", type: "bridge_status", status: "managed", side: "", pnl: 0, fee: 0, occurredAt: subMinutes(now, 12).toISOString() },
  ],
  orders: [
    { id: "ord-1", symbol: "BTC", type: "market_sim", status: "filled", side: "buy", quantity: 0.012, price: 67200, fee: 0.36, slippageBps: 1.7, occurredAt: subMinutes(now, 66).toISOString() },
    { id: "ord-2", symbol: "ETH", type: "market_sim", status: "rejected", side: "buy", quantity: 0.2, price: 3450, fee: 0, slippageBps: 0, reason: "min_fill_ratio_not_met", occurredAt: subMinutes(now, 44).toISOString() },
  ],
  risks: [
    { id: "risk-1", symbol: "BTC", type: "circuit_breaker", status: "clear", reason: "normal_volatility", occurredAt: subMinutes(now, 20).toISOString() },
    { id: "risk-2", symbol: "ETH", type: "paper_gate", status: "blocked", reason: "paper_observation_period", occurredAt: subMinutes(now, 13).toISOString() },
  ],
  commands: [
    { commandId: "cmd-demo-1", action: "pause_trading", status: "pending_confirmation", requestedText: "/pause", createdAt: subMinutes(now, 8).toISOString() },
  ],
  equityCurve: demoEquityCurve,
};

export type DemoOverview = typeof demoOverview;
