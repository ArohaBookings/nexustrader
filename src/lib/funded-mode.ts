export type JsonRecord = Record<string, unknown>;

export type FundedConfig = {
  enabled: boolean;
  group: string;
  phase: string;
  startingBalance: number;
  profitTargetPct: number;
  dailyDrawdownPct: number;
  maxDrawdownPct: number;
  trailingDrawdown: boolean;
  baseRiskPct: number;
  maxOpenRiskPct: number;
  dailyResetTimezone: string;
};

export type Mt5AccountSource = {
  account: string;
  magic: number | null;
  equity: number;
  balance: number;
  freeMargin: number;
  floatingPnl: number;
  dayStartEquity: number;
  dayHighEquity: number;
  dailyPnl: number;
  dailyPnlPct: number;
  dailyDrawdownPctLive: number;
  openPositions: number;
  highWatermarkEquity: number;
  source: string;
  observedAt: string;
};

export type FundedStatus = {
  enabled: boolean;
  group: string;
  phase: string;
  status: "disabled" | "normal" | "caution" | "defensive" | "hard_stop" | "pass_ready" | "passed";
  account: Mt5AccountSource;
  startingBalance: number;
  targetEquity: number;
  profitTargetUsd: number;
  profitFromStart: number;
  neededToPass: number;
  passProgressPct: number;
  dailyLossLimitUsd: number;
  dailyLossFloor: number;
  dailyLossUsedUsd: number;
  dailyLossRemainingUsd: number;
  dailyBufferPct: number;
  maxLossLimitUsd: number;
  maxLossFloor: number;
  maxLossUsedUsd: number;
  maxLossRemainingUsd: number;
  maxBufferPct: number;
  riskThrottle: number;
  maxRiskPerTradeUsd: number;
  maxOpenRiskUsd: number;
  fundedGuardReason: string;
  mt5Derived: boolean;
};

export const FUNDED_PRESETS: Record<string, Partial<FundedConfig>> = {
  custom: {},
  ftmo: {
    group: "ftmo",
    profitTargetPct: 0.10,
    dailyDrawdownPct: 0.05,
    maxDrawdownPct: 0.10,
    trailingDrawdown: false,
  },
  fundednext: {
    group: "fundednext",
    profitTargetPct: 0.10,
    dailyDrawdownPct: 0.05,
    maxDrawdownPct: 0.10,
    trailingDrawdown: false,
  },
  the5ers: {
    group: "the5ers",
    profitTargetPct: 0.08,
    dailyDrawdownPct: 0.03,
    maxDrawdownPct: 0.06,
    trailingDrawdown: false,
  },
  apex: {
    group: "apex",
    profitTargetPct: 0.06,
    dailyDrawdownPct: 0.025,
    maxDrawdownPct: 0.06,
    trailingDrawdown: true,
  },
  topstep: {
    group: "topstep",
    profitTargetPct: 0.06,
    dailyDrawdownPct: 0.03,
    maxDrawdownPct: 0.06,
    trailingDrawdown: true,
  },
};

export const DEFAULT_FUNDED_CONFIG: FundedConfig = {
  enabled: false,
  group: "custom",
  phase: "evaluation",
  startingBalance: 100,
  profitTargetPct: 0.08,
  dailyDrawdownPct: 0.05,
  maxDrawdownPct: 0.10,
  trailingDrawdown: false,
  baseRiskPct: 0.005,
  maxOpenRiskPct: 0.015,
  dailyResetTimezone: "Australia/Sydney",
};

export function applyFundedPreset(config: FundedConfig): FundedConfig {
  const preset = FUNDED_PRESETS[String(config.group || "custom").toLowerCase()] ?? {};
  return sanitizeFundedConfig({ ...config, ...preset, group: config.group || String(preset.group ?? "custom") });
}

export function sanitizeFundedConfig(input: Partial<FundedConfig> | JsonRecord | null | undefined): FundedConfig {
  const source = record(input);
  const group = stringValue(source.group, DEFAULT_FUNDED_CONFIG.group).toLowerCase();
  const preset = FUNDED_PRESETS[group] ?? {};
  return {
    enabled: booleanValue(source.enabled, DEFAULT_FUNDED_CONFIG.enabled),
    group,
    phase: stringValue(source.phase, DEFAULT_FUNDED_CONFIG.phase),
    startingBalance: clamp(numberValue(source.startingBalance, DEFAULT_FUNDED_CONFIG.startingBalance), 1, 100_000_000),
    profitTargetPct: clamp(numberValue(source.profitTargetPct, preset.profitTargetPct ?? DEFAULT_FUNDED_CONFIG.profitTargetPct), 0, 5),
    dailyDrawdownPct: clamp(numberValue(source.dailyDrawdownPct, preset.dailyDrawdownPct ?? DEFAULT_FUNDED_CONFIG.dailyDrawdownPct), 0, 1),
    maxDrawdownPct: clamp(numberValue(source.maxDrawdownPct, preset.maxDrawdownPct ?? DEFAULT_FUNDED_CONFIG.maxDrawdownPct), 0, 1),
    trailingDrawdown: booleanValue(source.trailingDrawdown, Boolean(preset.trailingDrawdown ?? DEFAULT_FUNDED_CONFIG.trailingDrawdown)),
    baseRiskPct: clamp(numberValue(source.baseRiskPct, DEFAULT_FUNDED_CONFIG.baseRiskPct), 0, 0.05),
    maxOpenRiskPct: clamp(numberValue(source.maxOpenRiskPct, DEFAULT_FUNDED_CONFIG.maxOpenRiskPct), 0, 0.20),
    dailyResetTimezone: stringValue(source.dailyResetTimezone, DEFAULT_FUNDED_CONFIG.dailyResetTimezone),
  };
}

export function extractMt5AccountSource(bot: JsonRecord | null | undefined, config: FundedConfig = DEFAULT_FUNDED_CONFIG): Mt5AccountSource {
  const botRecord = record(bot);
  const payload = record(botRecord.payload);
  const stats = record(payload.stats);
  const riskState = record(stats.risk_state);
  const accountScaling = record(stats.account_scaling);
  const snapshot = record(stats.latest_account_snapshot);
  const activeContext = record(stats.active_bridge_context);
  const now = stringValue(botRecord.observedAt ?? botRecord.observed_at ?? stats.time, new Date().toISOString());
  const equity = firstNumber(
    botRecord.equity,
    accountScaling.equity,
    snapshot.equity,
    config.startingBalance,
  );
  const balance = firstNumber(botRecord.balance, accountScaling.balance, snapshot.balance, equity);
  const dayStartEquity = firstNumber(
    riskState.day_start_equity,
    snapshot.day_start_equity,
    accountScaling.day_start_equity,
    equity,
  );
  const dayHighEquity = Math.max(
    dayStartEquity,
    firstNumber(riskState.day_high_equity, snapshot.day_high_equity, accountScaling.high_watermark_equity, equity),
  );
  const highWatermarkEquity = Math.max(
    config.startingBalance,
    dayHighEquity,
    firstNumber(accountScaling.high_watermark_equity, snapshot.high_watermark_equity, equity),
  );
  return {
    account: stringValue(activeContext.account ?? snapshot.account, ""),
    magic: finiteOrNull(activeContext.magic ?? snapshot.magic),
    equity,
    balance,
    freeMargin: firstNumber(accountScaling.free_margin, snapshot.free_margin, equity),
    floatingPnl: firstNumber(snapshot.floating_pnl, botRecord.floatingPnl, 0),
    dayStartEquity,
    dayHighEquity,
    dailyPnl: firstNumber(botRecord.pnlToday, stats.daily_realized_pnl, snapshot.daily_pnl, equity - dayStartEquity),
    dailyPnlPct: firstNumber(stats.daily_pnl_pct, riskState.daily_pnl_pct, safeRatio(equity - dayStartEquity, dayStartEquity)),
    dailyDrawdownPctLive: firstNumber(stats.daily_dd_pct_live, riskState.daily_dd_pct_live, safeRatio(dayHighEquity - equity, dayHighEquity)),
    openPositions: Math.max(0, Math.trunc(firstNumber(stats.open_positions, riskState.open_positions, snapshot.total_open_positions, 0))),
    highWatermarkEquity,
    source: snapshot.equity !== undefined || accountScaling.equity !== undefined ? "mt5_bridge" : "dashboard_fallback",
    observedAt: now,
  };
}

export function calculateFundedStatus(configInput: Partial<FundedConfig> | JsonRecord | null | undefined, bot: JsonRecord | null | undefined): FundedStatus {
  const config = sanitizeFundedConfig(configInput);
  const account = extractMt5AccountSource(bot, config);
  const startingBalance = config.startingBalance;
  const targetEquity = startingBalance * (1 + config.profitTargetPct);
  const profitTargetUsd = targetEquity - startingBalance;
  const profitFromStart = account.equity - startingBalance;
  const neededToPass = Math.max(0, targetEquity - account.equity);
  const passProgressPct = profitTargetUsd > 0 ? clamp(profitFromStart / profitTargetUsd, 0, 1.5) : 1;

  const dailyLossBasis = Math.max(1, account.dayStartEquity || startingBalance);
  const dailyLossLimitUsd = dailyLossBasis * config.dailyDrawdownPct;
  const dailyLossFloor = dailyLossBasis - dailyLossLimitUsd;
  const dailyLossUsedUsd = Math.max(0, dailyLossBasis - account.equity);
  const dailyLossRemainingUsd = account.equity - dailyLossFloor;
  const dailyBufferPct = dailyLossLimitUsd > 0 ? clamp(dailyLossRemainingUsd / dailyLossLimitUsd, -1, 3) : 1;

  const maxLossBasis = config.trailingDrawdown ? Math.max(startingBalance, account.highWatermarkEquity) : startingBalance;
  const maxLossLimitUsd = maxLossBasis * config.maxDrawdownPct;
  const maxLossFloor = maxLossBasis - maxLossLimitUsd;
  const maxLossUsedUsd = Math.max(0, maxLossBasis - account.equity);
  const maxLossRemainingUsd = account.equity - maxLossFloor;
  const maxBufferPct = maxLossLimitUsd > 0 ? clamp(maxLossRemainingUsd / maxLossLimitUsd, -1, 3) : 1;
  const closestBufferPct = Math.min(dailyBufferPct, maxBufferPct);

  let status: FundedStatus["status"] = "normal";
  let riskThrottle = 1;
  let fundedGuardReason = "funded_guard_normal";
  if (!config.enabled) {
    status = "disabled";
    riskThrottle = 1;
    fundedGuardReason = "funded_mode_disabled";
  } else if (dailyLossRemainingUsd <= 0 || maxLossRemainingUsd <= 0) {
    status = "hard_stop";
    riskThrottle = 0;
    fundedGuardReason = dailyLossRemainingUsd <= 0 ? "daily_drawdown_floor_breached" : "overall_drawdown_floor_breached";
  } else if (neededToPass <= 0) {
    status = "passed";
    riskThrottle = 0.35;
    fundedGuardReason = "target_reached_preserve_pass";
  } else if (passProgressPct >= 0.9) {
    status = "pass_ready";
    riskThrottle = Math.min(0.55, bufferThrottle(closestBufferPct));
    fundedGuardReason = "near_pass_target_reduce_mistake_risk";
  } else if (closestBufferPct <= 0.25) {
    status = "defensive";
    riskThrottle = Math.min(0.25, bufferThrottle(closestBufferPct));
    fundedGuardReason = "funded_buffer_defensive";
  } else if (closestBufferPct <= 0.50) {
    status = "caution";
    riskThrottle = Math.min(0.55, bufferThrottle(closestBufferPct));
    fundedGuardReason = "funded_buffer_caution";
  } else {
    riskThrottle = bufferThrottle(closestBufferPct);
  }

  const rawRiskPerTrade = account.equity * config.baseRiskPct;
  const bufferRiskCap = Math.max(0, Math.min(dailyLossRemainingUsd * 0.08, maxLossRemainingUsd * 0.06));
  const maxRiskPerTradeUsd = config.enabled
    ? Math.max(0, Math.min(rawRiskPerTrade * riskThrottle, bufferRiskCap))
    : rawRiskPerTrade;
  const maxOpenRiskUsd = config.enabled
    ? Math.max(0, Math.min(account.equity * config.maxOpenRiskPct * riskThrottle, dailyLossRemainingUsd * 0.22, maxLossRemainingUsd * 0.16))
    : account.equity * config.maxOpenRiskPct;

  return {
    enabled: config.enabled,
    group: config.group,
    phase: config.phase,
    status,
    account,
    startingBalance,
    targetEquity,
    profitTargetUsd,
    profitFromStart,
    neededToPass,
    passProgressPct,
    dailyLossLimitUsd,
    dailyLossFloor,
    dailyLossUsedUsd,
    dailyLossRemainingUsd,
    dailyBufferPct,
    maxLossLimitUsd,
    maxLossFloor,
    maxLossUsedUsd,
    maxLossRemainingUsd,
    maxBufferPct,
    riskThrottle,
    maxRiskPerTradeUsd,
    maxOpenRiskUsd,
    fundedGuardReason,
    mt5Derived: account.source === "mt5_bridge",
  };
}

function bufferThrottle(bufferPct: number) {
  if (bufferPct <= 0) return 0;
  if (bufferPct <= 0.25) return 0.25;
  if (bufferPct <= 0.50) return 0.55;
  if (bufferPct <= 0.80) return 0.78;
  return 1.0;
}

function record(value: unknown): JsonRecord {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as JsonRecord) : {};
}

function numberValue(value: unknown, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function firstNumber(...values: unknown[]): number {
  for (const value of values) {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return 0;
}

function finiteOrNull(value: unknown): number | null {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function stringValue(value: unknown, fallback: string): string {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function booleanValue(value: unknown, fallback: boolean): boolean {
  if (typeof value === "boolean") return value;
  if (typeof value === "string") return ["1", "true", "yes", "on"].includes(value.toLowerCase());
  return fallback;
}

function safeRatio(numerator: number, denominator: number): number {
  return denominator > 0 ? numerator / denominator : 0;
}

function clamp(value: number, low: number, high: number): number {
  return Math.max(low, Math.min(high, value));
}
