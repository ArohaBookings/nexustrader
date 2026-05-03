import type { IngestPayload } from "@/lib/validation";

type AnyRecord = Record<string, unknown>;

function numberFrom(...values: unknown[]) {
  for (const value of values) {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function textFrom(...values: unknown[]) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value;
  }
  return undefined;
}

function asRecord(value: unknown): AnyRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as AnyRecord) : {};
}

function asArray(value: unknown): AnyRecord[] {
  return Array.isArray(value) ? value.filter((item): item is AnyRecord => Boolean(item && typeof item === "object")) : [];
}

export function bridgePayloadToIngest(input: {
  health?: unknown;
  stats?: unknown;
  dashboard?: unknown;
  observedAt?: string;
}): IngestPayload {
  const observedAt = input.observedAt ?? new Date().toISOString();
  const health = asRecord(input.health);
  const stats = asRecord(input.stats);
  const dashboard = asRecord(input.dashboard);
  const runtime = asRecord(dashboard.runtime);
  const account = asRecord(dashboard.account);
  const control = asRecord(dashboard.control_state ?? dashboard.operator_control);
  const pairs = asArray(dashboard.pairs ?? dashboard.symbols ?? dashboard.strategy_pool);
  const recentEvents = asArray(dashboard.recent_events ?? dashboard.events);

  return {
    source: "bridge_collector",
    observedAt,
    bot: {
      observedAt,
      source: "bridge",
      equity: numberFrom(stats.equity, stats.current_equity, dashboard.equity, account.equity, runtime.equity),
      balance: numberFrom(stats.balance, dashboard.balance, account.balance),
      pnlToday: numberFrom(stats.pnl_today, stats.daily_pnl, dashboard.pnl_today, runtime.pnl_today),
      drawdownPct: numberFrom(stats.drawdown_pct, dashboard.drawdown_pct, runtime.drawdown_pct),
      queueDepth: numberFrom(stats.queue_depth, runtime.queue_depth),
      session: textFrom(String(stats.session ?? ""), String(dashboard.session ?? ""), String(runtime.session ?? "")),
      killState: textFrom(String(control.kill_switch ?? ""), String(dashboard.kill_state ?? ""), String(health.status ?? "")),
      openRiskPct: numberFrom(stats.open_risk_pct, dashboard.open_risk_pct, runtime.open_risk_pct),
      payload: {
        health,
        stats,
        dashboard_summary: dashboard.summary ?? dashboard.status ?? null,
        control,
        mt5_account: asRecord(stats.latest_account_snapshot ?? account),
        account_scaling: asRecord(stats.account_scaling),
        risk_state: asRecord(stats.risk_state),
        data_sources: asRecord(stats.data_sources ?? dashboard.data_sources ?? dashboard.providers),
        market_context: asRecord(stats.market_context ?? dashboard.market_context),
        learning: asRecord(stats.learning ?? stats.self_evolution ?? dashboard.learning),
      },
    },
    symbols: pairs.map((pair) => ({
      symbol: String(pair.symbol ?? pair.name ?? pair.pair ?? "UNKNOWN").toUpperCase(),
      observedAt,
      strategy: textFrom(pair.strategy, pair.strategy_id, pair.mode),
      state: textFrom(pair.state, pair.status, pair.gate_state),
      confidence: numberFrom(pair.confidence, pair.score, pair.day_score),
      spread: numberFrom(pair.spread, pair.spread_bps),
      openRiskPct: numberFrom(pair.open_risk_pct, pair.risk_pct),
      blocker: textFrom(pair.blocker, pair.block_reason, pair.reason),
      thinking: textFrom(pair.thinking, pair.diagnostic, pair.explanation),
      payload: pair,
    })),
    trades: recentEvents
      .filter((event) => String(event.type ?? event.event ?? "").toLowerCase().includes("trade"))
      .map((event) => eventToIngest(event, observedAt)),
    orders: recentEvents
      .filter((event) => String(event.type ?? event.event ?? "").toLowerCase().includes("order"))
      .map((event) => eventToIngest(event, observedAt)),
    risks: recentEvents
      .filter((event) => /risk|guard|breaker|stale|pause|kill/i.test(String(event.type ?? event.event ?? event.reason ?? "")))
      .map((event) => eventToIngest(event, observedAt)),
    dataIntegrity: [],
  };
}

function eventToIngest(event: AnyRecord, observedAt: string) {
  return {
    id: textFrom(event.id, event.event_id, event.order_id),
    symbol: textFrom(event.symbol, event.pair),
    type: textFrom(event.type, event.event) ?? "event",
    status: textFrom(event.status, event.state),
    side: textFrom(event.side, event.direction),
    quantity: numberFrom(event.quantity, event.qty, event.size),
    price: numberFrom(event.price, event.fill_price),
    fee: numberFrom(event.fee, event.commission),
    slippageBps: numberFrom(event.slippage_bps, event.slippageBps),
    pnl: numberFrom(event.pnl, event.realized_pnl),
    reason: textFrom(event.reason, event.error),
    occurredAt: textFrom(event.occurred_at, event.timestamp, event.time) ?? observedAt,
    payload: event,
  };
}
