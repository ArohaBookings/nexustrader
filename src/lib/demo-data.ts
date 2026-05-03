import { subMinutes } from "date-fns";
import { calculateFundedStatus, DEFAULT_FUNDED_CONFIG } from "@/lib/funded-mode";

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
      stats: {
        active_bridge_context: { account: "DEMO-MT5", magic: 4254 },
        account_scaling: { equity: 189.42, balance: 184.11, free_margin: 177.2, high_watermark_equity: 194.4 },
        latest_account_snapshot: { account: "DEMO-MT5", magic: 4254, equity: 189.42, balance: 184.11, free_margin: 177.2, floating_pnl: 2.1, total_open_positions: 2 },
        risk_state: { day_start_equity: 181.2, day_high_equity: 194.4, daily_dd_pct_live: 0.0256, daily_pnl_pct: 0.0453, open_positions: 2 },
        data_sources: {
          mt5: { status: "active", latency_ms: 22 },
          hyperliquid: { status: "active", latency_ms: 48 },
          binance: { status: "active", latency_ms: 61 },
          bybit: { status: "degraded", latency_ms: 140 },
          cryptopanic: { status: "active", latency_ms: 88 },
        },
        learning: {
          recent_sample_size: 74,
          validation_sample_size: 0,
          recent_expectancy: 0.41,
          validation_expectancy: 0,
          recent_expectancy_delta: 0.018,
          validation_expectancy_delta: 0,
        },
      },
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
      payload: {
        native_data: true,
        source: "hyperliquid",
        stale: false,
        regime: "trend",
        candle_score: 0.76,
        smc_score: 0.69,
        order_flow_score: 0.72,
        microstructure_score: 0.81,
        mtf_confluence: 0.74,
        volume_profile_score: 0.66,
        vwap_score: 0.7,
        cross_asset_score: 0.58,
        liquidity_sweep: true,
        fvg_state: "partial_fill",
        bos_choch: "bos_up",
        delta_pressure: "bid_absorption",
      },
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
      payload: {
        native_data: true,
        source: "hyperliquid",
        stale: false,
        regime: "range",
        candle_score: 0.55,
        smc_score: 0.49,
        order_flow_score: 0.52,
        microstructure_score: 0.64,
        mtf_confluence: 0.5,
        volume_profile_score: 0.6,
        vwap_score: 0.57,
        cross_asset_score: 0.43,
        fvg_state: "none",
        bos_choch: "range",
      },
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
      payload: {
        bridge_source: true,
        source: "mt5",
        regime: "london_premarket",
        candle_score: 0.71,
        smc_score: 0.66,
        order_flow_score: 0.62,
        microstructure_score: 0.68,
        mtf_confluence: 0.73,
        volume_profile_score: 0.61,
        vwap_score: 0.64,
        cross_asset_score: 0.7,
        order_block: "discount_retest",
        delta_pressure: "neutral",
      },
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
  funded: {
    config: DEFAULT_FUNDED_CONFIG,
    status: calculateFundedStatus(DEFAULT_FUNDED_CONFIG, {
      equity: 189.42,
      balance: 184.11,
      pnlToday: 14.33,
      payload: {
        stats: {
          active_bridge_context: { account: "DEMO-MT5", magic: 4254 },
          account_scaling: { equity: 189.42, balance: 184.11, free_margin: 177.2, high_watermark_equity: 194.4 },
          latest_account_snapshot: { account: "DEMO-MT5", magic: 4254, equity: 189.42, balance: 184.11, free_margin: 177.2, floating_pnl: 2.1, total_open_positions: 2 },
          risk_state: { day_start_equity: 181.2, day_high_equity: 194.4, daily_dd_pct_live: 0.0256, daily_pnl_pct: 0.0453, open_positions: 2 },
        },
      },
    }),
  },
};

export type DemoOverview = typeof demoOverview;
