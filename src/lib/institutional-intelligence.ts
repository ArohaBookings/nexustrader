import { percent, usd } from "@/lib/utils";

type Json = Record<string, unknown>;

type OverviewLike = {
  bot?: Json;
  symbols?: Json[];
  trades?: Json[];
  orders?: Json[];
  risks?: Json[];
  funded?: {
    config?: Json;
    status?: Json;
  };
};

const DATA_PROVIDERS = [
  { id: "polygon", label: "Polygon.io", aliases: ["polygon", "polygon.io"] },
  { id: "twelve_data", label: "Twelve Data", aliases: ["twelvedata", "twelve_data", "twelve data"] },
  { id: "finnhub", label: "Finnhub", aliases: ["finnhub"] },
  { id: "alpha_vantage", label: "Alpha Vantage", aliases: ["alphavantage", "alpha_vantage", "alpha vantage"] },
  { id: "binance", label: "Binance", aliases: ["binance"] },
  { id: "bybit", label: "Bybit", aliases: ["bybit"] },
  { id: "tradingeconomics", label: "TradingEconomics", aliases: ["tradingeconomics", "trading_economics"] },
  { id: "newsapi", label: "NewsAPI", aliases: ["newsapi", "news_api"] },
  { id: "cryptopanic", label: "CryptoPanic", aliases: ["cryptopanic", "crypto_panic"] },
  { id: "hyperliquid", label: "Hyperliquid", aliases: ["hyperliquid"] },
  { id: "mt5", label: "MT5 Bridge", aliases: ["mt5", "metatrader", "bridge"] },
] as const;

const MARKET_DIMENSIONS = [
  "candle",
  "smc",
  "orderFlow",
  "microstructure",
  "confluence",
  "regime",
  "volumeProfile",
  "vwap",
  "crossAsset",
] as const;

export function buildInstitutionalIntelligence(overview: OverviewLike) {
  const bot = record(overview.bot);
  const symbols = Array.isArray(overview.symbols) ? overview.symbols : [];
  const trades = Array.isArray(overview.trades) ? overview.trades : [];
  const orders = Array.isArray(overview.orders) ? overview.orders : [];
  const risks = Array.isArray(overview.risks) ? overview.risks : [];
  const fundedStatus = record(overview.funded?.status);
  const botPayload = record(bot.payload);
  const market = buildMarketMastery(symbols);
  const blockers = buildRepairPlan(symbols, risks, fundedStatus, bot);
  const antiOverfit = buildAntiOverfitGate(bot, trades);
  const dataFusion = buildDataFusion(bot, symbols);
  const scaling = buildScalingPlan(bot, fundedStatus, antiOverfit, blockers);
  const fundedMission = buildFundedMission(fundedStatus);
  const execution = buildExecutionQuality(orders, trades);
  const grade = weightedGrade([
    [market.score, 0.25],
    [dataFusion.consensusScore, 0.18],
    [antiOverfit.score, 0.18],
    [scaling.score, 0.18],
    [blockers.score, 0.13],
    [execution.score, 0.08],
  ]);
  const readiness = readinessState(fundedStatus, blockers, antiOverfit, dataFusion, grade);

  return {
    generatedAt: new Date().toISOString(),
    systemGrade: grade,
    readiness,
    summary: summaryFor(readiness, fundedMission, blockers, antiOverfit),
    fundedMission,
    marketMastery: market,
    dataFusion,
    antiOverfit,
    selfRepair: blockers,
    scaling,
    execution,
    telegramBrief: buildTelegramBrief(readiness, fundedMission, blockers, antiOverfit, dataFusion, scaling),
    bridgeApex: record(botPayload.institutional_apex),
    edgePolicy: record(botPayload.institutional_intelligence),
    trainingBootstrap: record(botPayload.training_bootstrap_status),
    dataQualityOverlay: record(botPayload.data_quality),
    promotionAudit: record(botPayload.promotion_audit),
    trajectoryForecast: record(botPayload.trajectory_forecast),
    opportunityPipeline: record(botPayload.xau_btc_opportunity_pipeline),
    liveShadowGap: record(botPayload.live_shadow_gap),
    operatorLimits: [
      "No direct Telegram trade placement.",
      "No AI-driven risk/aggression increases.",
      "No parameter promotion unless both recent and validation gates clear.",
      "Drawdown, stale data, exchange errors, and funded floors override frequency.",
    ],
  };
}

export type InstitutionalIntelligence = ReturnType<typeof buildInstitutionalIntelligence>;

function buildMarketMastery(symbols: Json[]) {
  const rows = symbols.map((symbol) => {
    const payload = record(symbol.payload);
    const confidence = clamp(numberValue(symbol.confidence, 0.45), 0, 1);
    const dimensions = {
      candle: metric(payload, ["candle_score", "candleStructureScore", "body_wick_score", "candle_mastery"], confidence),
      smc: metric(payload, ["smc_score", "order_block_score", "liquidity_sweep_score", "fvg_score", "bos_choch_score"], confidence),
      orderFlow: metric(payload, ["order_flow_score", "delta_pressure_score", "absorption_score", "imbalance_score"], confidence),
      microstructure: metric(payload, ["microstructure_score", "depth_score", "spread_quality", "book_quality"], confidence),
      confluence: metric(payload, ["confluence_score", "mtf_confluence", "multi_timeframe_score"], confidence),
      regime: metric(payload, ["regime_score", "regime_confidence", "trend_regime_score"], confidence),
      volumeProfile: metric(payload, ["volume_profile_score", "vp_score", "poc_score"], confidence),
      vwap: metric(payload, ["vwap_score", "anchored_vwap_score", "session_vwap_score"], confidence),
      crossAsset: metric(payload, ["cross_asset_score", "lead_lag_score", "correlation_score"], confidence),
    };
    const score = average(Object.values(dimensions));
    return {
      symbol: String(symbol.symbol ?? "UNKNOWN").toUpperCase(),
      strategy: stringValue(symbol.strategy, "strategy_pending"),
      state: stringValue(symbol.state, "unknown"),
      confidence,
      score,
      regime: stringValue(payload.regime ?? payload.market_regime, "unclassified"),
      session: stringValue(payload.session ?? payload.active_session, "unknown"),
      blocker: stringValue(symbol.blocker, ""),
      thinking: stringValue(symbol.thinking, "No diagnostic text received."),
      dimensions,
      detected: {
        liquiditySweep: booleanValue(payload.liquidity_sweep ?? payload.sweep_detected, false),
        fvg: stringValue(payload.fvg_state ?? payload.fvg, "unknown"),
        bosChoch: stringValue(payload.bos_choch ?? payload.structure_shift, "unknown"),
        absorption: booleanValue(payload.absorption ?? payload.absorption_detected, false),
        deltaPressure: stringValue(payload.delta_pressure ?? payload.delta, "unknown"),
        orderBlock: stringValue(payload.order_block ?? payload.order_block_state, "unknown"),
      },
    };
  });
  const score = rows.length ? average(rows.map((row) => row.score)) : 0.35;
  const regimeVotes = mode(rows.map((row) => row.regime).filter((value) => value !== "unclassified"));
  return {
    score,
    regime: regimeVotes || "unclassified",
    rows,
    dimensions: Object.fromEntries(MARKET_DIMENSIONS.map((key) => [key, average(rows.map((row) => row.dimensions[key])) || 0])) as Record<
      (typeof MARKET_DIMENSIONS)[number],
      number
    >,
  };
}

function buildRepairPlan(symbols: Json[], risks: Json[], fundedStatus: Json, bot: Json) {
  const symbolBlockers = symbols
    .filter((symbol) => stringValue(symbol.blocker, ""))
    .map((symbol) => ({
      symbol: String(symbol.symbol ?? "UNKNOWN").toUpperCase(),
      reason: stringValue(symbol.blocker, "unknown_blocker"),
      ageMinutes: ageMinutes(symbol.observedAt),
    }));
  const riskBlockers = risks
    .filter((risk) => /blocked|breach|stale|error|halt|kill|pause/i.test(`${risk.status ?? ""} ${risk.type ?? ""} ${risk.reason ?? ""}`))
    .map((risk) => ({
      symbol: stringValue(risk.symbol, "SYSTEM").toUpperCase(),
      reason: stringValue(risk.reason ?? risk.type, "risk_event"),
      ageMinutes: ageMinutes(risk.occurredAt ?? risk.createdAt),
    }));
  const all = [...symbolBlockers, ...riskBlockers];
  const hardRails = all.filter((item) => isHardRail(item.reason));
  if (isHardRail(stringValue(fundedStatus.fundedGuardReason, "")) || stringValue(fundedStatus.status, "") === "hard_stop") {
    hardRails.push({ symbol: "FUNDED", reason: stringValue(fundedStatus.fundedGuardReason, "funded_hard_stop"), ageMinutes: 0 });
  }
  if (isHardRail(stringValue(bot.killState, ""))) {
    hardRails.push({ symbol: "SYSTEM", reason: `kill_state_${bot.killState}`, ageMinutes: 0 });
  }
  const softBlockers = all.filter((item) => !isHardRail(item.reason));
  const softRepairable = softBlockers.filter((item) => isSoftRepairable(item.reason));
  const needsRefresh = softRepairable.length > 0;
  const status = hardRails.length ? "hard_rail_holds" : needsRefresh ? "repair_refresh_required" : softBlockers.length ? "wait_for_validation" : "clear";
  return {
    status,
    score: hardRails.length ? 0 : softBlockers.length ? 0.62 : 1,
    slaMinutes: 5,
    softBlockers,
    hardRails,
    actions: [
      ...(needsRefresh ? [{ action: "refresh_state", reason: "soft data/api/stale blocker inside repair SLA", allowed: true }] : []),
      ...softBlockers
        .filter((item) => !isSoftRepairable(item.reason))
        .map((item) => ({ action: "observe_or_shadow_validate", reason: `${item.symbol}: ${item.reason}`, allowed: true })),
      ...hardRails.map((item) => ({ action: "do_not_override_hard_rail", reason: `${item.symbol}: ${item.reason}`, allowed: false })),
    ],
    recommendedBridgeAction: hardRails.length ? "none" : needsRefresh ? "refresh_state" : "none",
  };
}

function buildAntiOverfitGate(bot: Json, trades: Json[]) {
  const payload = record(bot.payload);
  const stats = record(payload.stats);
  const learning = record(stats.learning ?? stats.self_evolution ?? payload.learning);
  const recentSample = Math.trunc(numberValue(learning.recent_sample_size ?? learning.recent_trades, trades.length));
  const validationSample = Math.trunc(numberValue(learning.validation_sample_size ?? learning.validation_trades, 0));
  const recentDelta = numberValue(learning.recent_expectancy_delta ?? learning.recent_edge_delta, 0);
  const validationDelta = numberValue(learning.validation_expectancy_delta ?? learning.validation_edge_delta, 0);
  const recentExpectancy = numberValue(learning.recent_expectancy, expectancy(trades.slice(0, 200)));
  const validationExpectancy = numberValue(learning.validation_expectancy, 0);
  const promotionAllowed = recentSample >= 200 && validationSample >= 100 && recentDelta > 0.03 && validationDelta > 0.03;
  const reason = promotionAllowed
    ? "promotion_gate_cleared"
    : recentSample < 200 || validationSample < 100
      ? "insufficient_out_of_sample_evidence"
      : "expectancy_delta_below_3pct_gate";
  return {
    score: promotionAllowed ? 1 : recentSample >= 200 && validationSample >= 100 ? 0.72 : 0.38,
    promotionAllowed,
    improvementGatePct: 0.03,
    recentSample,
    validationSample,
    recentExpectancy,
    validationExpectancy,
    recentDelta,
    validationDelta,
    reason,
  };
}

function buildDataFusion(bot: Json, symbols: Json[]) {
  const payload = record(bot.payload);
  const stats = record(payload.stats);
  const sourceRoot = record(stats.data_sources ?? payload.data_sources ?? stats.providers ?? payload.providers);
  const symbolSources = symbols.flatMap((symbol) => sourceNames(record(symbol.payload)));
  const providers = DATA_PROVIDERS.map((provider) => {
    const explicit = provider.aliases.map((alias) => sourceRoot[alias]).find((value) => value !== undefined);
    const sourceText = JSON.stringify(explicit ?? "").toLowerCase();
    const implied = provider.aliases.some((alias) => symbolSources.includes(alias));
    const active = implied || explicit === true || /online|active|ok|healthy|connected|native|ready/.test(sourceText);
    const stale = /stale|degraded|down|error|missing|timeout/.test(sourceText);
    return {
      id: provider.id,
      label: provider.label,
      status: active && !stale ? "active" : stale ? "degraded" : "missing",
      latencyMs: finiteOrNull(record(explicit).latency_ms ?? record(explicit).latencyMs),
      lastSeen: stringValue(record(explicit).last_seen ?? record(explicit).lastSeen, ""),
    };
  });
  const activeCount = providers.filter((item) => item.status === "active").length;
  const degradedCount = providers.filter((item) => item.status === "degraded").length;
  const consensusScore = clamp((activeCount + degradedCount * 0.45) / providers.length, 0, 1);
  return {
    consensusScore,
    activeSources: activeCount,
    degradedSources: degradedCount,
    missingSources: providers.length - activeCount - degradedCount,
    providers,
    fallbackReady: activeCount >= 2 || providers.some((item) => item.id === "mt5" && item.status === "active"),
  };
}

function buildScalingPlan(bot: Json, fundedStatus: Json, antiOverfit: ReturnType<typeof buildAntiOverfitGate>, repair: ReturnType<typeof buildRepairPlan>) {
  const equity = numberValue(bot.equity, numberValue(fundedStatus.account && record(fundedStatus.account).equity, 0));
  const throttle = numberValue(fundedStatus.riskThrottle, 1);
  const maxRiskPerTradeUsd = numberValue(fundedStatus.maxRiskPerTradeUsd, equity * 0.003);
  const maxOpenRiskUsd = numberValue(fundedStatus.maxOpenRiskUsd, equity * 0.01);
  const openRiskPct = numberValue(bot.openRiskPct, 0);
  const fundedStatusText = stringValue(fundedStatus.status, "disabled");
  const hardBlocked = fundedStatusText === "hard_stop" || repair.hardRails.length > 0;
  const score = hardBlocked ? 0 : antiOverfit.promotionAllowed ? Math.min(1, throttle) : Math.min(0.72, throttle);
  const maxOpenTrades = maxRiskPerTradeUsd > 0 ? Math.max(0, Math.floor(maxOpenRiskUsd / maxRiskPerTradeUsd)) : 0;
  const aggression =
    hardBlocked ? "locked" : fundedStatusText === "passed" || fundedStatusText === "pass_ready" ? "protect_pass" : antiOverfit.promotionAllowed ? "expand" : "measured";
  return {
    score,
    aggression,
    equity,
    throttle,
    openRiskPct,
    maxRiskPerTradeUsd,
    maxOpenRiskUsd,
    maxOpenTrades,
    baseCapitalProtectedUsd: Math.min(equity, numberValue(fundedStatus.startingBalance, equity)),
    fundingChange: equity > numberValue(fundedStatus.startingBalance, equity) * 1.25 ? "capital_increase_detected" : "no_material_top_up_detected",
    notes: scalingNotes(aggression, antiOverfit, fundedStatus, repair),
  };
}

function buildFundedMission(fundedStatus: Json) {
  return {
    enabled: Boolean(fundedStatus.enabled),
    group: stringValue(fundedStatus.group, "custom"),
    phase: stringValue(fundedStatus.phase, "evaluation"),
    status: stringValue(fundedStatus.status, "disabled"),
    targetEquity: numberValue(fundedStatus.targetEquity, 0),
    neededToPass: numberValue(fundedStatus.neededToPass, 0),
    passProgressPct: numberValue(fundedStatus.passProgressPct, 0),
    dailyBufferUsd: numberValue(fundedStatus.dailyLossRemainingUsd, 0),
    dailyBufferPct: numberValue(fundedStatus.dailyBufferPct, 0),
    maxBufferUsd: numberValue(fundedStatus.maxLossRemainingUsd, 0),
    maxBufferPct: numberValue(fundedStatus.maxBufferPct, 0),
    riskThrottle: numberValue(fundedStatus.riskThrottle, 1),
    guardReason: stringValue(fundedStatus.fundedGuardReason, "funded_mode_disabled"),
    mt5Derived: Boolean(fundedStatus.mt5Derived),
  };
}

function buildExecutionQuality(orders: Json[], trades: Json[]) {
  const filled = orders.filter((order) => /fill|filled|closed/i.test(String(order.status ?? order.type ?? "")));
  const rejected = orders.filter((order) => /reject|rejected/i.test(String(order.status ?? order.type ?? "")));
  const avgSlippageBps = average(filled.map((order) => numberValue(order.slippageBps, 0)));
  const totalFees = filled.reduce((sum, order) => sum + numberValue(order.fee, 0), 0);
  const winRate = winRateFrom(trades);
  const rejectRate = orders.length ? rejected.length / orders.length : 0;
  const score = clamp(0.72 + winRate * 0.18 - rejectRate * 0.25 - Math.max(0, avgSlippageBps - 3) * 0.03, 0, 1);
  return {
    score,
    fills: filled.length,
    rejections: rejected.length,
    rejectRate,
    avgSlippageBps,
    totalFees,
    winRate,
  };
}

function buildTelegramBrief(
  readiness: string,
  funded: ReturnType<typeof buildFundedMission>,
  repair: ReturnType<typeof buildRepairPlan>,
  antiOverfit: ReturnType<typeof buildAntiOverfitGate>,
  dataFusion: ReturnType<typeof buildDataFusion>,
  scaling: ReturnType<typeof buildScalingPlan>,
) {
  return [
    `Readiness: ${readiness}.`,
    `Funded: ${funded.status}; needed ${usd(funded.neededToPass)}; daily buffer ${usd(funded.dailyBufferUsd)}; throttle ${percent(funded.riskThrottle)}.`,
    `Repair: ${repair.status}; soft blockers ${repair.softBlockers.length}; hard rails ${repair.hardRails.length}; action ${repair.recommendedBridgeAction}.`,
    `Overfit gate: ${antiOverfit.reason}; recent ${antiOverfit.recentSample}; validation ${antiOverfit.validationSample}.`,
    `Data fusion: ${percent(dataFusion.consensusScore)} consensus across ${dataFusion.activeSources} active sources.`,
    `Scaling: ${scaling.aggression}; max risk/trade ${usd(scaling.maxRiskPerTradeUsd)}; max open trades ${scaling.maxOpenTrades}.`,
  ].join("\n");
}

function summaryFor(
  readiness: string,
  funded: ReturnType<typeof buildFundedMission>,
  repair: ReturnType<typeof buildRepairPlan>,
  antiOverfit: ReturnType<typeof buildAntiOverfitGate>,
) {
  if (funded.status === "hard_stop" || repair.hardRails.length) return "Hard rail active. Preserve account, do not override funded/risk controls.";
  if (funded.status === "passed") return "Target reached. Protect the pass and reduce mistake risk before any scale-up.";
  if (!antiOverfit.promotionAllowed) return "Trading can be observed, but new optimization is blocked until out-of-sample evidence clears.";
  if (readiness === "expand") return "Evidence and buffers support measured expansion inside current funded caps.";
  return "Operate inside current caps while telemetry continues to validate edge.";
}

function readinessState(
  funded: Json,
  repair: ReturnType<typeof buildRepairPlan>,
  antiOverfit: ReturnType<typeof buildAntiOverfitGate>,
  dataFusion: ReturnType<typeof buildDataFusion>,
  grade: number,
) {
  const status = stringValue(funded.status, "disabled");
  if (status === "hard_stop" || repair.hardRails.length) return "hard_stop";
  if (status === "passed" || status === "pass_ready") return "protect_funded_pass";
  if (!dataFusion.fallbackReady || repair.status === "repair_refresh_required") return "repair_first";
  if (antiOverfit.promotionAllowed && grade >= 0.72) return "expand";
  return "observe_validate";
}

function scalingNotes(
  aggression: string,
  antiOverfit: ReturnType<typeof buildAntiOverfitGate>,
  fundedStatus: Json,
  repair: ReturnType<typeof buildRepairPlan>,
) {
  const notes = [];
  if (aggression === "locked") notes.push("Hard rails active; all scaling remains disabled.");
  if (aggression === "protect_pass") notes.push("Funded target state reached/near; protect pass before scaling.");
  if (!antiOverfit.promotionAllowed) notes.push("Optimization promotion blocked by anti-overfit gate.");
  if (repair.softBlockers.length) notes.push("Soft blockers require refresh/validation before frequency expansion.");
  if (numberValue(fundedStatus.riskThrottle, 1) < 1) notes.push("Funded throttle is reducing risk from configured base.");
  if (!notes.length) notes.push("Expansion remains bounded by funded max risk and open-risk caps.");
  return notes;
}

function weightedGrade(items: Array<[number, number]>) {
  const totalWeight = items.reduce((sum, [, weight]) => sum + weight, 0);
  return clamp(items.reduce((sum, [score, weight]) => sum + clamp(score, 0, 1) * weight, 0) / totalWeight, 0, 1);
}

function metric(payload: Json, keys: string[], fallback: number) {
  for (const key of keys) {
    const value = numberValue(payload[key], Number.NaN);
    if (Number.isFinite(value)) return clamp(value > 1 ? value / 100 : value, 0, 1);
  }
  return fallback;
}

function sourceNames(payload: Json) {
  const values = [
    payload.source,
    payload.venue,
    payload.exchange,
    payload.primary_source,
    payload.secondary_source,
    ...(Array.isArray(payload.sources) ? payload.sources : []),
  ];
  return values.map((value) => String(value ?? "").toLowerCase().trim()).filter(Boolean);
}

function expectancy(trades: Json[]) {
  const pnls = trades.map((trade) => numberValue(trade.pnl, Number.NaN)).filter(Number.isFinite);
  return pnls.length ? average(pnls) : 0;
}

function winRateFrom(trades: Json[]) {
  const pnls = trades.map((trade) => numberValue(trade.pnl, Number.NaN)).filter(Number.isFinite);
  return pnls.length ? pnls.filter((pnl) => pnl > 0).length / pnls.length : 0;
}

function isHardRail(reason: string) {
  return /drawdown|daily.*floor|overall.*floor|kill|hard_stop|breach|funded_hard|loss_limit/i.test(reason);
}

function isSoftRepairable(reason: string) {
  return /stale|api|timeout|disconnect|gap|data|exchange|bridge|sync|book|latency/i.test(reason);
}

function average(values: number[]) {
  const clean = values.filter(Number.isFinite);
  return clean.length ? clean.reduce((sum, value) => sum + value, 0) / clean.length : 0;
}

function mode(values: string[]) {
  const counts = new Map<string, number>();
  for (const value of values) counts.set(value, (counts.get(value) ?? 0) + 1);
  return [...counts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "";
}

function ageMinutes(value: unknown) {
  const time = new Date(String(value ?? "")).getTime();
  if (!Number.isFinite(time)) return 0;
  return Math.max(0, Math.round((Date.now() - time) / 60_000));
}

function record(value: unknown): Json {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Json) : {};
}

function numberValue(value: unknown, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function finiteOrNull(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function stringValue(value: unknown, fallback: string) {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function booleanValue(value: unknown, fallback: boolean) {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return ["1", "true", "yes", "on"].includes(value.toLowerCase());
  return fallback;
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}
