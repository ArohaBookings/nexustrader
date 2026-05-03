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
  return [
    "*Nexus Status*",
    `Equity: ${usd(Number(bot.equity ?? 0))}`,
    `Daily PnL: ${usd(Number(bot.pnlToday ?? 0))}`,
    `Drawdown: ${percent(Number(bot.drawdownPct ?? 0))}`,
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

