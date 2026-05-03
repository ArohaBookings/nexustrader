#!/usr/bin/env node

const bridgeBaseUrl = (process.env.BRIDGE_API_BASE_URL || "").replace(/\/+$/, "");
const ingestUrl = process.env.NEXUS_INGEST_URL || "http://127.0.0.1:3000/api/ingest";
const ingestKey = process.env.INGEST_API_KEY || "";
const bridgeToken = process.env.BRIDGE_TOKEN || "";
const intervalMs = Number(process.env.COLLECTOR_INTERVAL_MS || 15000);

if (!bridgeBaseUrl) {
  throw new Error("BRIDGE_API_BASE_URL is required");
}
if (!ingestKey) {
  throw new Error("INGEST_API_KEY is required");
}

async function fetchJson(path) {
  const response = await fetch(`${bridgeBaseUrl}${path}`, {
    headers: bridgeToken ? { "x-bridge-token": bridgeToken } : {},
  });
  if (!response.ok) {
    throw new Error(`Bridge ${path} failed with HTTP ${response.status}`);
  }
  return response.json();
}

function record(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function array(value) {
  return Array.isArray(value) ? value.filter((item) => item && typeof item === "object") : [];
}

function numberFrom(...values) {
  for (const value of values) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return undefined;
}

function textFrom(...values) {
  for (const value of values) {
    if (typeof value === "string" && value.trim()) return value;
  }
  return undefined;
}

function eventToIngest(event, observedAt) {
  return {
    id: textFrom(event.id, event.event_id, event.order_id),
    symbol: textFrom(event.symbol, event.pair),
    type: textFrom(event.type, event.event) || "event",
    status: textFrom(event.status, event.state),
    side: textFrom(event.side, event.direction),
    quantity: numberFrom(event.quantity, event.qty, event.size),
    price: numberFrom(event.price, event.fill_price),
    fee: numberFrom(event.fee, event.commission),
    slippageBps: numberFrom(event.slippage_bps, event.slippageBps),
    pnl: numberFrom(event.pnl, event.realized_pnl),
    reason: textFrom(event.reason, event.error),
    occurredAt: textFrom(event.occurred_at, event.timestamp, event.time) || observedAt,
    payload: event,
  };
}

function toIngest({ health, stats, dashboard }) {
  const observedAt = new Date().toISOString();
  const runtime = record(dashboard.runtime);
  const account = record(dashboard.account);
  const control = record(dashboard.control_state || dashboard.operator_control);
  const pairs = array(dashboard.pairs || dashboard.symbols || dashboard.strategy_pool);
  const recentEvents = array(dashboard.recent_events || dashboard.events);

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
      session: textFrom(stats.session, dashboard.session, runtime.session),
      killState: textFrom(String(control.kill_switch ?? ""), dashboard.kill_state, health.status),
      openRiskPct: numberFrom(stats.open_risk_pct, dashboard.open_risk_pct, runtime.open_risk_pct),
      payload: {
        health,
        stats,
        dashboard_summary: dashboard.summary || dashboard.status || null,
        control,
        mt5_account: record(stats.latest_account_snapshot || account),
        account_scaling: record(stats.account_scaling),
        risk_state: record(stats.risk_state),
        data_sources: record(stats.data_sources || dashboard.data_sources || dashboard.providers),
        market_context: record(stats.market_context || dashboard.market_context),
        learning: record(stats.learning || stats.self_evolution || dashboard.learning),
      },
    },
    symbols: pairs.map((pair) => ({
      symbol: String(pair.symbol || pair.name || pair.pair || "UNKNOWN").toUpperCase(),
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
    trades: recentEvents.filter((event) => String(event.type || event.event || "").toLowerCase().includes("trade")).map((event) => eventToIngest(event, observedAt)),
    orders: recentEvents.filter((event) => String(event.type || event.event || "").toLowerCase().includes("order")).map((event) => eventToIngest(event, observedAt)),
    risks: recentEvents.filter((event) => /risk|guard|breaker|stale|pause|kill/i.test(String(event.type || event.event || event.reason || ""))).map((event) => eventToIngest(event, observedAt)),
    dataIntegrity: [],
  };
}

async function collectOnce() {
  const [health, stats, dashboard] = await Promise.all([
    fetchJson("/health").catch((error) => ({ status: "error", error: String(error.message || error) })),
    fetchJson("/stats").catch((error) => ({ status: "error", error: String(error.message || error) })),
    fetchJson("/dashboard/data").catch((error) => ({ status: "error", error: String(error.message || error) })),
  ]);
  const payload = toIngest({ health: record(health), stats: record(stats), dashboard: record(dashboard) });
  const response = await fetch(ingestUrl, {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-ingest-key": ingestKey,
    },
    body: JSON.stringify(payload),
  });
  if (!response.ok) {
    throw new Error(`Ingest failed with HTTP ${response.status}: ${await response.text()}`);
  }
  console.log(`[collector] ${new Date().toISOString()} posted telemetry`);
}

await collectOnce();
setInterval(() => {
  collectOnce().catch((error) => {
    console.error(`[collector] ${new Date().toISOString()} ${error.stack || error.message || error}`);
  });
}, intervalMs);
