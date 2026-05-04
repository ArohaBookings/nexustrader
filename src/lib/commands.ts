import { executeBridgeControl, type BridgeAction } from "@/lib/bridge";
import { explainBotStatus } from "@/lib/ai";
import {
  createCommandRequest,
  findPendingCommand,
  getOverview,
  getRisk,
  getTrades,
  markCommandExecuted,
} from "@/lib/repository";
import { runSelfRepairAudit } from "@/lib/self-repair";
import { usd, percent } from "@/lib/utils";

type CommandContext = {
  chatId: string;
  telegramUserId?: string;
  text: string;
};

const ACTIONS: Record<string, { action: BridgeAction; confirmationRequired: boolean }> = {
  "/pause": { action: "pause_trading", confirmationRequired: false },
  "/resume": { action: "resume_trading", confirmationRequired: true },
  "/kill": { action: "kill_switch", confirmationRequired: true },
  "/refresh": { action: "refresh_state", confirmationRequired: false },
};

const BLOCKED_INTENTS = /\b(buy|sell|long|short|market order|place order|open position|increase leverage|raise leverage|increase risk|ignore risk|bypass|yolo|all in)\b/i;

function commandId() {
  return `cmd_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
}

export async function handleTelegramCommand(context: CommandContext) {
  const text = context.text.trim();
  const lower = text.toLowerCase();

  if (lower.startsWith("/confirm")) {
    const id = text.split(/\s+/)[1];
    if (!id) return "Send `/confirm <command_id>`.";
    return executePendingCommand(id);
  }

  if (lower === "/status" || lower === "status") {
    return formatStatus(await getOverview());
  }

  if (lower === "/funded" || lower === "funded" || /\bfunded\b.*\b(pass|target|account|status)\b/i.test(text)) {
    return formatFunded(await getOverview());
  }

  if (lower === "/trajectory" || /\b(trajectory|100k|forecast)\b/i.test(text)) {
    return formatTrajectory(await getOverview());
  }

  if (lower === "/losses" || /\b(losses|loss review|why losing|losing too much)\b/i.test(text)) {
    return formatLosses(await getOverview());
  }

  if (lower === "/blockers" || /\b(blockers|blocked|why blocked|not trading)\b/i.test(text)) {
    return formatBlockers(await getOverview());
  }

  if (lower === "/intel" || lower === "/apex" || /\b(intel|intelligence|thinking|edge|scale|100k)\b/i.test(text)) {
    return formatIntel(await getOverview());
  }

  if (/\b(increase|more|raise|boost).*\b(frequency|trades|entries)\b|\bfrequency\b.*\b(xau|btc|gold)\b/i.test(text)) {
    return formatFrequencyPolicy(await getOverview());
  }

  if (lower === "/repair" || /\b(repair|fix|self[- ]?heal|unstick)\b/i.test(text)) {
    return formatRepair(await runSelfRepairAudit("telegram"));
  }

  if (lower === "/risk" || lower === "risk") {
    return formatRisk(await getRisk());
  }

  if (lower === "/trades" || lower === "trades") {
    return formatTrades(await getTrades());
  }

  if (lower in ACTIONS) {
    const spec = ACTIONS[lower];
    const id = commandId();
    await createCommandRequest({
      commandId: id,
      chatId: context.chatId,
      telegramUserId: context.telegramUserId,
      action: spec.action,
      requestedText: text,
      confirmationRequired: spec.confirmationRequired,
      payload: { source: "telegram" },
    });
    if (spec.confirmationRequired) {
      return `Confirmation required for \`${spec.action}\`.\nSend \`/confirm ${id}\` to execute.`;
    }
    const result = await executeBridgeControl(spec.action);
    await markCommandExecuted(id, result);
    return `Executed \`${spec.action}\`.\nResult: \`${result.ok ? "ok" : result.reason ?? "not_ok"}\``;
  }

  if (BLOCKED_INTENTS.test(text)) {
    return "Blocked: Telegram AI cannot place trades, increase aggression, bypass risk, or change strategy parameters. Use `/status`, `/risk`, `/trades`, `/pause`, `/refresh`, `/resume`, or `/kill`.";
  }

  return explainBotStatus(text);
}

export async function executePendingCommand(commandId: string) {
  const pending = await findPendingCommand(commandId);
  if (!pending) return `No pending command found for \`${commandId}\`.`;
  const action = String((pending as { action?: unknown }).action ?? "") as BridgeAction;
  if (!(action in ACTIONS_BY_BRIDGE_ACTION)) {
    return `Command \`${commandId}\` has unsupported action \`${action}\`.`;
  }
  const result = await executeBridgeControl(action);
  await markCommandExecuted(commandId, result);
  return `Executed \`${action}\`.\nResult: \`${result.ok ? "ok" : result.reason ?? "not_ok"}\``;
}

const ACTIONS_BY_BRIDGE_ACTION: Record<BridgeAction, true> = {
  pause_trading: true,
  resume_trading: true,
  kill_switch: true,
  refresh_state: true,
};

function formatStatus(overview: Awaited<ReturnType<typeof getOverview>>) {
  const bot = overview.bot as Record<string, unknown>;
  const symbols = overview.symbols as Record<string, unknown>[];
  const funded = (overview.funded?.status ?? {}) as Record<string, unknown>;
  const intelligence = (overview.intelligence ?? {}) as Record<string, unknown>;
  return [
    "*Nexus Status*",
    `Equity: ${usd(Number(bot.equity ?? 0))}`,
    `Daily PnL: ${usd(Number(bot.pnlToday ?? 0))}`,
    `Drawdown: ${percent(Number(bot.drawdownPct ?? 0))}`,
    `Funded: ${funded.status ?? "disabled"} | Needed: ${usd(Number(funded.neededToPass ?? 0))} | Daily buffer: ${usd(Number(funded.dailyLossRemainingUsd ?? 0))}`,
    `Apex readiness: ${intelligence.readiness ?? "unknown"} | Grade: ${percent(Number(intelligence.systemGrade ?? 0))}`,
    `Session: ${bot.session ?? "unknown"} | Kill: ${bot.killState ?? "unknown"} | Queue: ${bot.queueDepth ?? 0}`,
    `Symbols: ${symbols.length}`,
  ].join("\n");
}

function formatRisk(risk: Awaited<ReturnType<typeof getRisk>>) {
  const bot = risk.bot as Record<string, unknown>;
  const risks = risk.risks as Record<string, unknown>[];
  const recent = risks
    .slice(0, 5)
    .map((event) => `${event.type ?? "risk"}:${event.status ?? "unknown"} ${event.reason ?? ""}`.trim())
    .join("\n");
  return [
    "*Risk*",
    `Open risk: ${percent(Number(bot.openRiskPct ?? 0))}`,
    `Drawdown: ${percent(Number(bot.drawdownPct ?? 0))}`,
    recent || "No recent risk events.",
  ].join("\n");
}

function formatFunded(overview: Awaited<ReturnType<typeof getOverview>>) {
  const status = (overview.funded?.status ?? {}) as Record<string, unknown>;
  const account = (status.account ?? {}) as Record<string, unknown>;
  const intelligence = (overview.intelligence ?? {}) as Record<string, unknown>;
  const mission = (intelligence.fundedMission ?? {}) as Record<string, unknown>;
  return [
    "*Funded Mission*",
    `Account: ${account.account ?? "not linked"} | Source: ${status.mt5Derived ? "MT5 bridge" : "fallback"}`,
    `Status: ${status.status ?? "disabled"} | Guard: ${status.fundedGuardReason ?? "unknown"}`,
    `Equity: ${usd(Number(account.equity ?? 0))} | Target: ${usd(Number(status.targetEquity ?? 0))} | Needed: ${usd(Number(status.neededToPass ?? 0))}`,
    `Daily buffer: ${usd(Number(status.dailyLossRemainingUsd ?? 0))} (${percent(Number(status.dailyBufferPct ?? 0))})`,
    `Overall buffer: ${usd(Number(status.maxLossRemainingUsd ?? 0))} (${percent(Number(status.maxBufferPct ?? 0))})`,
    `Risk throttle: ${percent(Number(status.riskThrottle ?? 1))} | Max risk/trade: ${usd(Number(status.maxRiskPerTradeUsd ?? 0))}`,
    `Mission: ${mission.guardReason ?? "waiting"}`,
  ].join("\n");
}

function formatIntel(overview: Awaited<ReturnType<typeof getOverview>>) {
  const intelligence = overview.intelligence as Record<string, unknown>;
  const funded = record(intelligence.fundedMission);
  const repair = record(intelligence.selfRepair);
  const anti = record(intelligence.antiOverfit);
  const data = record(intelligence.dataFusion);
  const scaling = record(intelligence.scaling);
  return [
    "*Institutional Apex Intel*",
    `Readiness: ${intelligence.readiness ?? "unknown"} | Grade: ${percent(Number(intelligence.systemGrade ?? 0))}`,
    String(intelligence.summary ?? "No summary available."),
    `Funded needed: ${usd(Number(funded.neededToPass ?? 0))} | Daily buffer: ${usd(Number(funded.dailyBufferUsd ?? 0))}`,
    `Repair: ${repair.status ?? "unknown"} | Soft: ${array(repair.softBlockers).length} | Hard: ${array(repair.hardRails).length} | Action: ${repair.recommendedBridgeAction ?? "none"}`,
    `Overfit gate: ${anti.reason ?? "unknown"} | Recent: ${anti.recentSample ?? 0} | Validation: ${anti.validationSample ?? 0}`,
    `Data consensus: ${percent(Number(data.consensusScore ?? 0))} | Active sources: ${data.activeSources ?? 0}`,
    `Scaling: ${scaling.aggression ?? "unknown"} | Max risk/trade: ${usd(Number(scaling.maxRiskPerTradeUsd ?? 0))} | Max open trades: ${scaling.maxOpenTrades ?? 0}`,
  ].join("\n");
}

function formatTrajectory(overview: Awaited<ReturnType<typeof getOverview>>) {
  const intelligence = overview.intelligence as Record<string, unknown>;
  const trajectory = record(intelligence.trajectoryForecast);
  return [
    "*Trajectory*",
    `Current equity: ${usd(Number(trajectory.current_equity ?? trajectory.currentEquity ?? 0))}`,
    `Short goal: ${usd(Number(trajectory.short_goal_equity ?? trajectory.shortGoalEquity ?? 100000))} | On track: ${Boolean(trajectory.short_goal_on_track ?? trajectory.shortGoalOnTrack)}`,
    `Medium goal: ${usd(Number(trajectory.medium_goal_equity ?? trajectory.mediumGoalEquity ?? 1000000))} | On track: ${Boolean(trajectory.medium_goal_on_track ?? trajectory.mediumGoalOnTrack)}`,
    `Forecast type: ${trajectory.forecast_type ?? trajectory.forecastType ?? "speculative_target_path"}`,
    "This forecast is not a sizing input and does not authorize higher risk.",
  ].join("\n");
}

function formatLosses(overview: Awaited<ReturnType<typeof getOverview>>) {
  const intelligence = overview.intelligence as Record<string, unknown>;
  const liveShadow = record(intelligence.liveShadowGap);
  const priority = array(liveShadow.priority_symbols ?? liveShadow.prioritySymbols).map(record);
  const bridgeApex = record(intelligence.bridgeApex);
  const execution = record(intelligence.execution);
  return [
    "*Loss Review*",
    `Readiness: ${bridgeApex.readiness ?? intelligence.readiness ?? "unknown"}`,
    `Execution score: ${percent(Number(execution.score ?? 0))} | Floating win rate: ${percent(Number(execution.winRate ?? execution.floating_win_rate ?? 0))}`,
    `Live-shadow status: ${liveShadow.status ?? "collecting_or_aligned"} | Max gap: ${percent(Number(liveShadow.max_gap_score ?? liveShadow.maxGapScore ?? 0))}`,
    `Priority symbols: ${priority.map((item) => `${item.symbol}:${item.status ?? "collecting"}`).join(", ") || "insufficient samples"}`,
    "Next safe action: collect more BTCUSD/XAUUSD live-shadow evidence before promoting risk or frequency.",
  ].join("\n");
}

function formatBlockers(overview: Awaited<ReturnType<typeof getOverview>>) {
  const intelligence = overview.intelligence as Record<string, unknown>;
  const repair = record(intelligence.selfRepair);
  const edgeRepair = record(record(intelligence.edgePolicy).self_repair ?? record(intelligence.edgePolicy).selfRepair);
  const activeRepair = Object.keys(edgeRepair).length ? edgeRepair : repair;
  const opportunity = record(intelligence.opportunityPipeline);
  const priority = array(opportunity.priority_symbols ?? opportunity.prioritySymbols).map(record);
  return [
    "*Blockers*",
    `Repair: ${activeRepair.status ?? "unknown"} | Soft: ${array(activeRepair.soft_blockers ?? activeRepair.softBlockers).length} | Hard: ${array(activeRepair.hard_rails ?? activeRepair.hardRails).length}`,
    `Recommended action: ${activeRepair.recommended_bridge_action ?? activeRepair.recommendedBridgeAction ?? "none"}`,
    `BTC/XAU: ${priority.map((item) => `${item.symbol}:${item.live_gate ?? item.liveGate ?? "edge_gated"}`).join(", ") || "waiting for priority telemetry"}`,
    "Hard rails are locked and will not be auto-repaired.",
  ].join("\n");
}

function formatFrequencyPolicy(overview: Awaited<ReturnType<typeof getOverview>>) {
  const intelligence = overview.intelligence as Record<string, unknown>;
  const opportunity = record(intelligence.opportunityPipeline);
  const priority = array(opportunity.priority_symbols ?? opportunity.prioritySymbols).map(record);
  return [
    "*Frequency Policy*",
    "Live frequency is edge-gated; Telegram cannot force entries, raise risk, or change parameters.",
    ...priority.slice(0, 4).map((item) => {
      const target = record(item.shadow_target_10m ?? item.shadowTarget10m);
      return `${item.symbol}: shadow ${target.low ?? 0}-${target.high ?? 0}/10m | candidates ${item.actual_candidates_last_10m ?? item.actualCandidatesLast10m ?? 0} | live ${item.actual_live_trades_last_10m ?? item.actualLiveTradesLast10m ?? 0} | gate ${item.live_gate ?? item.liveGate ?? "edge_gated"}`;
    }),
    `Forced live frequency: ${Boolean(opportunity.live_frequency_forced ?? opportunity.liveFrequencyForced)}`,
  ].join("\n");
}

function formatRepair(result: Awaited<ReturnType<typeof runSelfRepairAudit>>) {
  const repair = result.repair as Record<string, unknown>;
  return [
    "*Self-Repair Audit*",
    `Mode: ${result.mode} | Action: ${result.action}`,
    `Status: ${repair.status ?? "unknown"} | SLA: ${repair.slaMinutes ?? 5}m`,
    `Soft blockers: ${array(repair.softBlockers).length} | Hard rails: ${array(repair.hardRails).length}`,
    result.action === "refresh_state" ? `Refresh command: ${result.commandId ?? "issued"}` : `Reason: ${result.reason ?? "no repair needed"}`,
    "Hard drawdown/funded/kill rails are never auto-overridden.",
  ].join("\n");
}

function formatTrades(data: Awaited<ReturnType<typeof getTrades>>) {
  const trades = data.trades as Record<string, unknown>[];
  if (!trades.length) return "*Trades*\nNo recent trades recorded.";
  return [
    "*Trades*",
    ...trades.slice(0, 6).map((trade) => {
      const pnl = trade.pnl === undefined || trade.pnl === null ? "" : ` pnl ${usd(Number(trade.pnl))}`;
      return `${trade.symbol ?? "?"} ${trade.status ?? trade.type ?? "event"}${pnl}`;
    }),
  ].join("\n");
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function array(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}
