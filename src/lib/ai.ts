import OpenAI from "openai";
import { getOverview } from "@/lib/repository";

let client: OpenAI | null = null;

function getOpenAIClient() {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) return null;
  client ??= new OpenAI({ apiKey });
  return client;
}

export async function explainBotStatus(question: string) {
  const overview = await getOverview();
  const bot = overview.bot as Record<string, unknown>;
  const symbols = (overview.symbols as Record<string, unknown>[]).slice(0, 20);
  const risks = (overview.risks as Record<string, unknown>[]).slice(0, 10);
  const trades = (overview.trades as Record<string, unknown>[]).slice(0, 10);
  const funded = overview.funded;
  const intelligence = overview.intelligence;
  const client = getOpenAIClient();

  if (!client) {
    return localExplanation(question, bot, symbols, risks, trades, funded, intelligence);
  }

  const response = await client.responses.create({
    model: "gpt-4.1-mini",
    input: [
      {
        role: "system",
        content:
          "You are Nexus Trader's funded-account operator assistant. Understand normal English, answer from telemetry only, and be direct. Focus on funded pass risk, account protection, blockers, data quality, anti-overfit gates, self-repair actions, and safe scaling caps. Do not reveal hidden chain-of-thought. Do not place trades. Do not recommend increasing leverage, bypassing risk controls, or changing strategy parameters. Allowed operational actions are pause, resume with confirmation, kill switch with confirmation, refresh state, and self-repair refresh for soft blockers only.",
      },
      {
        role: "user",
        content: JSON.stringify({
          question,
          bot,
          symbols,
          risks,
          trades,
          funded,
          intelligence,
        }),
      },
    ],
  });
  return response.output_text || localExplanation(question, bot, symbols, risks, trades, funded, intelligence);
}

function localExplanation(
  question: string,
  bot: Record<string, unknown>,
  symbols: Record<string, unknown>[],
  risks: Record<string, unknown>[],
  trades: Record<string, unknown>[],
  funded?: unknown,
  intelligence?: unknown,
) {
  const blocked = symbols.filter((symbol) => String(symbol.blocker ?? "").length > 0);
  const fundedStatus = record(record(funded).status);
  const intel = record(intelligence);
  const repair = record(intel.selfRepair);
  const anti = record(intel.antiOverfit);
  const lines = [
    `Nexus status for: ${question}`,
    `Equity: ${bot.equity ?? "unknown"} | Daily PnL: ${bot.pnlToday ?? "unknown"} | Kill state: ${bot.killState ?? "unknown"}`,
    `Funded: ${fundedStatus.status ?? "disabled"} | needed ${fundedStatus.neededToPass ?? "unknown"} | daily buffer ${fundedStatus.dailyLossRemainingUsd ?? "unknown"} | throttle ${fundedStatus.riskThrottle ?? "unknown"}`,
    `Apex readiness: ${intel.readiness ?? "unknown"} | self-repair ${repair.status ?? "unknown"} | overfit gate ${anti.reason ?? "unknown"}`,
    `Active symbols: ${symbols.length}; blocked: ${blocked.length}; recent risk events: ${risks.length}; recent trades: ${trades.length}.`,
  ];
  if (blocked.length) {
    lines.push(`Main blockers: ${blocked.slice(0, 4).map((item) => `${item.symbol}: ${item.blocker}`).join("; ")}`);
  }
  lines.push("Ops scope: telemetry explanations plus pause, resume confirmation, kill confirmation, and refresh only.");
  return lines.join("\n");
}

function record(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}
