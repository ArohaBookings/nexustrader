"use client";

import {
  Activity,
  AlertTriangle,
  Bot,
  BrainCircuit,
  CircleDollarSign,
  DatabaseZap,
  Gauge,
  LogOut,
  RadioTower,
  ShieldAlert,
  ShieldCheck,
  Siren,
  Sparkles,
  Terminal,
  TimerReset,
  TrendingUp,
  Zap,
} from "lucide-react";
import type { LucideIcon } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import type { ReactNode } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { DEFAULT_FUNDED_CONFIG, FUNDED_PRESETS } from "@/lib/funded-mode";
import { compactNumber, percent, usd } from "@/lib/utils";

type Overview = {
  bot: Record<string, unknown>;
  symbols: Record<string, unknown>[];
  trades: Record<string, unknown>[];
  orders: Record<string, unknown>[];
  risks: Record<string, unknown>[];
  commands: Record<string, unknown>[];
  equityCurve: Record<string, unknown>[];
  funded?: {
    config: Record<string, unknown>;
    status: Record<string, unknown>;
  };
  intelligence?: Record<string, unknown>;
};

const tabs = [
  { id: "overview", label: "Live", icon: Activity },
  { id: "thinking", label: "Thinking", icon: BrainCircuit },
  { id: "apex", label: "Apex", icon: Sparkles },
  { id: "symbols", label: "Symbols", icon: RadioTower },
  { id: "orders", label: "Orders", icon: CircleDollarSign },
  { id: "risk", label: "Risk", icon: ShieldAlert },
  { id: "funded", label: "Funded", icon: ShieldCheck },
  { id: "data", label: "Data", icon: DatabaseZap },
  { id: "ops", label: "Ops", icon: Terminal },
  { id: "trajectory", label: "$100K", icon: TrendingUp },
] as const;

type TabId = (typeof tabs)[number]["id"];

function n(value: unknown, fallback = 0) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function text(value: unknown, fallback = "unknown") {
  return typeof value === "string" && value.trim() ? value : fallback;
}

function time(value: unknown) {
  const parsed = new Date(String(value ?? ""));
  if (Number.isNaN(parsed.getTime())) return "unknown";
  return parsed.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function shortDate(value: unknown) {
  const parsed = new Date(String(value ?? ""));
  if (Number.isNaN(parsed.getTime())) return "unknown";
  return parsed.toLocaleString([], { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function NexusDashboard({ initialOverview }: { initialOverview: Overview }) {
  const [overview, setOverview] = useState<Overview>(initialOverview);
  const [activeTab, setActiveTab] = useState<TabId>("overview");
  const [lastSyncLabel, setLastSyncLabel] = useState("initial");
  const [chartsReady, setChartsReady] = useState(false);

  useEffect(() => {
    let alive = true;
    const frame = window.requestAnimationFrame(() => {
      setChartsReady(true);
      setLastSyncLabel(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    });
    const refresh = async () => {
      const response = await fetch("/api/overview", { cache: "no-store" });
      if (!response.ok || !alive) return;
      setOverview(await response.json());
      setLastSyncLabel(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" }));
    };
    const id = window.setInterval(refresh, 8000);
    return () => {
      alive = false;
      window.cancelAnimationFrame(frame);
      window.clearInterval(id);
    };
  }, []);

  const bot = overview.bot ?? {};
  const symbols = overview.symbols ?? [];
  const trades = overview.trades ?? [];
  const orders = overview.orders ?? [];
  const risks = overview.risks ?? [];
  const commands = overview.commands ?? [];
  const equityCurve = overview.equityCurve ?? [];
  const intelligence = (overview.intelligence ?? {}) as Record<string, unknown>;
  const funded = (overview.funded ?? { config: DEFAULT_FUNDED_CONFIG, status: {} }) as {
    config: Record<string, unknown>;
    status: Record<string, unknown>;
  };
  const openBlockers = symbols.filter((symbol) => text(symbol.blocker, "")).length;
  const killState = text(bot.killState, "NONE");
  const fundedStatus = text(funded.status?.status, "disabled");

  return (
    <main className="min-h-screen px-4 py-4 sm:px-6 lg:px-8">
      <header className="mx-auto flex max-w-7xl flex-col gap-4 border-b border-cyan-300/15 pb-5 md:flex-row md:items-end md:justify-between">
        <div>
          <div className="mb-3 flex items-center gap-3">
            <div className="grid size-10 place-items-center border border-cyan-300/40 bg-cyan-300/10 text-cyan-200 shadow-[0_0_28px_rgba(34,211,238,0.22)]">
              <Bot size={22} />
            </div>
            <div>
              <p className="text-xs uppercase tracking-[0.32em] text-cyan-200/75">Private Ops Console</p>
              <h1 className="text-2xl font-black uppercase tracking-[0.16em] text-white sm:text-4xl">Nexus Trader</h1>
            </div>
          </div>
          <p className="max-w-3xl text-sm leading-6 text-zinc-300">
            Live telemetry, risk state, Telegram command audit, and speculative trajectory tracking. Ops controls are limited to pause,
            resume confirmation, kill confirmation, and refresh.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2 text-xs uppercase tracking-[0.18em]">
          <StatusPill label="Sync" value={lastSyncLabel} tone="cyan" />
          <StatusPill label="Funded" value={fundedStatus} tone={fundedStatus === "hard_stop" ? "rose" : fundedStatus === "disabled" ? "cyan" : "lime"} />
          <StatusPill label="Kill" value={killState} tone={killState === "NONE" || killState === "false" ? "lime" : "rose"} />
          <a
            className="inline-flex h-10 items-center gap-2 border border-white/15 px-3 text-zinc-200 transition hover:border-cyan-300/50 hover:text-cyan-100"
            href="/logout"
          >
            <LogOut size={15} />
            Logout
          </a>
        </div>
      </header>

      <section className="mx-auto mt-5 grid max-w-7xl grid-cols-2 gap-3 lg:grid-cols-4">
        <Metric icon={CircleDollarSign} label="Equity" value={usd(n(bot.equity))} sub={`Balance ${usd(n(bot.balance))}`} tone="cyan" />
        <Metric icon={Zap} label="Daily PnL" value={usd(n(bot.pnlToday))} sub="Observed bridge/lab delta" tone={n(bot.pnlToday) >= 0 ? "lime" : "rose"} />
        <Metric icon={Gauge} label="Drawdown" value={percent(n(bot.drawdownPct))} sub={`Open risk ${percent(n(bot.openRiskPct))}`} tone="amber" />
        <Metric
          icon={TimerReset}
          label="Funded Buffer"
          value={usd(n(funded.status?.dailyLossRemainingUsd))}
          sub={`Queue ${String(bot.queueDepth ?? 0)} / ${text(bot.session)}`}
          tone={fundedStatus === "hard_stop" || n(funded.status?.dailyLossRemainingUsd) < 0 ? "rose" : "violet"}
        />
      </section>

      <nav className="mx-auto mt-5 flex max-w-7xl gap-2 overflow-x-auto border-y border-white/10 py-2">
        {tabs.map((tab) => {
          const Icon = tab.icon;
          const active = activeTab === tab.id;
          return (
            <button
              key={tab.id}
              type="button"
              aria-label={tab.label}
              title={tab.label}
              onClick={() => setActiveTab(tab.id)}
              className={`flex h-10 shrink-0 items-center gap-2 border px-3 text-sm transition ${
                active ? "border-cyan-300 bg-cyan-300 text-black" : "border-white/10 bg-black/30 text-zinc-300 hover:border-cyan-300/50"
              }`}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </nav>

      <section className="mx-auto mt-5 max-w-7xl pb-10">
        {activeTab === "overview" ? <OverviewPanel overview={overview} openBlockers={openBlockers} chartsReady={chartsReady} /> : null}
        {activeTab === "thinking" ? <ThinkingPanel symbols={symbols} risks={risks} /> : null}
        {activeTab === "apex" ? <ApexPanel intelligence={intelligence} /> : null}
        {activeTab === "symbols" ? <SymbolsPanel symbols={symbols} /> : null}
        {activeTab === "orders" ? <OrdersPanel orders={orders} trades={trades} chartsReady={chartsReady} /> : null}
        {activeTab === "risk" ? <RiskPanel bot={bot} risks={risks} /> : null}
        {activeTab === "funded" ? <FundedPanel funded={funded} onFundedUpdate={(next) => setOverview((current) => ({ ...current, funded: next }))} /> : null}
        {activeTab === "data" ? <DataPanel risks={risks} symbols={symbols} /> : null}
        {activeTab === "ops" ? <OpsPanel commands={commands} /> : null}
        {activeTab === "trajectory" ? <TrajectoryPanel equityCurve={equityCurve} chartsReady={chartsReady} /> : null}
      </section>
    </main>
  );
}

function OverviewPanel({ overview, openBlockers, chartsReady }: { overview: Overview; openBlockers: number; chartsReady: boolean }) {
  const bot = overview.bot ?? {};
  const symbols = overview.symbols ?? [];
  const orders = overview.orders ?? [];
  const equityCurve = overview.equityCurve ?? [];
  const recentFills = orders.filter((order) => text(order.status, "").toLowerCase() === "filled").length;
  return (
    <div className="grid gap-4 lg:grid-cols-[1.45fr_0.9fr]">
      <Panel title="Live Equity" icon={Activity}>
        <ChartFrame ready={chartsReady}>
          {({ width, height }) => (
            <AreaChart data={equityCurve} width={width} height={height}>
              <defs>
                <linearGradient id="equityFill" x1="0" x2="0" y1="0" y2="1">
                  <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.45} />
                  <stop offset="95%" stopColor="#22d3ee" stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke="rgba(255,255,255,0.07)" vertical={false} />
              <XAxis dataKey="timestamp" tickFormatter={time} stroke="#94a3b8" tickLine={false} axisLine={false} minTickGap={30} />
              <YAxis tickFormatter={compactNumber} stroke="#94a3b8" tickLine={false} axisLine={false} width={48} />
              <Tooltip contentStyle={{ background: "#05070a", border: "1px solid rgba(34,211,238,.25)", color: "#fff" }} labelFormatter={shortDate} formatter={(value) => usd(Number(value))} />
              <Area type="monotone" dataKey="equity" stroke="#22d3ee" fill="url(#equityFill)" strokeWidth={2.4} />
            </AreaChart>
          )}
        </ChartFrame>
      </Panel>
      <Panel title="Runtime Pulse" icon={Sparkles}>
        <div className="grid gap-3">
          <Pulse label="Symbols online" value={String(symbols.length)} tone="cyan" />
          <Pulse label="Open blockers" value={String(openBlockers)} tone={openBlockers ? "amber" : "lime"} />
          <Pulse label="Recent fills" value={String(recentFills)} tone="lime" />
          <Pulse label="Kill state" value={text(bot.killState, "NONE")} tone={text(bot.killState, "NONE") === "NONE" ? "lime" : "rose"} />
        </div>
      </Panel>
    </div>
  );
}

function ThinkingPanel({ symbols, risks }: { symbols: Record<string, unknown>[]; risks: Record<string, unknown>[] }) {
  return (
    <div className="grid gap-4 lg:grid-cols-[1fr_0.8fr]">
      <Panel title="Bot Thinking" icon={BrainCircuit}>
        <div className="grid gap-3">
          {symbols.map((symbol) => (
            <div key={String(symbol.symbol)} className="border border-white/10 bg-white/[0.03] p-4">
              <div className="mb-2 flex items-center justify-between gap-3">
                <p className="font-mono text-lg font-semibold text-white">{String(symbol.symbol)}</p>
                <span className="text-xs uppercase tracking-[0.18em] text-cyan-200">{percent(n(symbol.confidence))}</span>
              </div>
              <p className="text-sm leading-6 text-zinc-300">{text(symbol.thinking, "No diagnostic text received yet.")}</p>
              {text(symbol.blocker, "") ? <p className="mt-3 text-sm text-amber-200">Blocker: {text(symbol.blocker, "")}</p> : null}
            </div>
          ))}
        </div>
      </Panel>
      <Panel title="Recent Guards" icon={ShieldCheck}>
        <EventList rows={risks.slice(0, 8)} empty="No recent risk guards." />
      </Panel>
    </div>
  );
}

function ApexPanel({ intelligence }: { intelligence: Record<string, unknown> }) {
  const funded = asRecord(intelligence.fundedMission);
  const market = asRecord(intelligence.marketMastery);
  const data = asRecord(intelligence.dataFusion);
  const anti = asRecord(intelligence.antiOverfit);
  const repair = asRecord(intelligence.selfRepair);
  const scaling = asRecord(intelligence.scaling);
  const execution = asRecord(intelligence.execution);
  const rows = asArray(market.rows).map(asRecord);
  const providers = asArray(data.providers).map(asRecord);
  const softBlockers = asArray(repair.softBlockers).map(asRecord);
  const hardRails = asArray(repair.hardRails).map(asRecord);
  const notes = asArray(scaling.notes).map(String);

  return (
    <div className="grid gap-4">
      <div className="grid gap-4 xl:grid-cols-[0.9fr_1.1fr]">
        <Panel title="Institutional Readiness" icon={Sparkles}>
          <div className="grid gap-3 md:grid-cols-2">
            <Pulse label="Apex Grade" value={percent(n(intelligence.systemGrade))} tone={toneFromScore(n(intelligence.systemGrade))} />
            <Pulse label="Readiness" value={text(intelligence.readiness, "unknown")} tone={readinessTone(text(intelligence.readiness, ""))} />
            <Pulse label="Funded Mission" value={text(funded.status, "disabled")} tone={fundedTone(text(funded.status, "disabled"))} />
            <Pulse label="Data Consensus" value={percent(n(data.consensusScore))} tone={toneFromScore(n(data.consensusScore))} />
            <Pulse label="Repair SLA" value={`${String(repair.slaMinutes ?? 5)}m / ${text(repair.status, "unknown")}`} tone={softBlockers.length ? "amber" : hardRails.length ? "rose" : "lime"} />
            <Pulse label="Scaling State" value={text(scaling.aggression, "unknown")} tone={text(scaling.aggression, "") === "locked" ? "rose" : "cyan"} />
          </div>
          <p className="mt-4 border border-cyan-300/20 bg-cyan-300/10 p-4 text-sm leading-6 text-cyan-100">
            {text(intelligence.summary, "Telemetry is waiting for a live bridge snapshot.")}
          </p>
        </Panel>

        <Panel title="Funded Scaling Brain" icon={ShieldCheck}>
          <div className="grid gap-3 md:grid-cols-3">
            <MiniStat label="Needed to pass" value={usd(n(funded.neededToPass))} />
            <MiniStat label="Pass progress" value={percent(n(funded.passProgressPct))} />
            <MiniStat label="Risk throttle" value={percent(n(funded.riskThrottle, 1))} />
            <MiniStat label="Daily buffer" value={usd(n(funded.dailyBufferUsd))} />
            <MiniStat label="Overall buffer" value={usd(n(funded.maxBufferUsd))} />
            <MiniStat label="Max risk/trade" value={usd(n(scaling.maxRiskPerTradeUsd))} />
            <MiniStat label="Max open risk" value={usd(n(scaling.maxOpenRiskUsd))} />
            <MiniStat label="Max open trades" value={String(scaling.maxOpenTrades ?? 0)} />
            <MiniStat label="Funding change" value={text(scaling.fundingChange, "unknown")} />
          </div>
          <div className="mt-4 grid gap-2">
            {notes.map((note) => (
              <p key={note} className="border border-white/10 bg-white/[0.03] p-3 text-sm text-zinc-300">
                {note}
              </p>
            ))}
          </div>
        </Panel>
      </div>

      <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
        <Panel title="Market Mastery Matrix" icon={BrainCircuit}>
          <div className="grid gap-3">
            {rows.map((row) => {
              const dimensions = asRecord(row.dimensions);
              return (
                <div key={text(row.symbol, "symbol")} className="border border-white/10 bg-white/[0.03] p-4">
                  <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
                    <div>
                      <p className="font-mono text-lg font-black text-white">{text(row.symbol, "UNKNOWN")}</p>
                      <p className="text-xs uppercase tracking-[0.16em] text-zinc-500">
                        {text(row.regime, "unclassified")} / {text(row.session, "unknown")}
                      </p>
                    </div>
                    <StateBadge state={`${percent(n(row.score))} mastery`} />
                  </div>
                  <div className="grid gap-2 md:grid-cols-3">
                    <ProgressLine label="Candle" value={n(dimensions.candle)} tone={toneFromScore(n(dimensions.candle))} />
                    <ProgressLine label="SMC" value={n(dimensions.smc)} tone={toneFromScore(n(dimensions.smc))} />
                    <ProgressLine label="Order flow" value={n(dimensions.orderFlow)} tone={toneFromScore(n(dimensions.orderFlow))} />
                    <ProgressLine label="Microstructure" value={n(dimensions.microstructure)} tone={toneFromScore(n(dimensions.microstructure))} />
                    <ProgressLine label="MTF confluence" value={n(dimensions.confluence)} tone={toneFromScore(n(dimensions.confluence))} />
                    <ProgressLine label="Cross-asset" value={n(dimensions.crossAsset)} tone={toneFromScore(n(dimensions.crossAsset))} />
                  </div>
                  <p className="mt-3 text-sm leading-5 text-zinc-300">{text(row.thinking, "No diagnostic text received.")}</p>
                </div>
              );
            })}
          </div>
        </Panel>

        <Panel title="Self-Repair / Anti-Overfit" icon={Siren}>
          <div className="grid gap-3">
            <Pulse label="Repair State" value={text(repair.status, "unknown")} tone={hardRails.length ? "rose" : softBlockers.length ? "amber" : "lime"} />
            <Pulse label="Soft Blockers" value={String(softBlockers.length)} tone={softBlockers.length ? "amber" : "lime"} />
            <Pulse label="Hard Rails" value={String(hardRails.length)} tone={hardRails.length ? "rose" : "lime"} />
            <Pulse label="Promotion Gate" value={anti.promotionAllowed ? "cleared" : text(anti.reason, "blocked")} tone={anti.promotionAllowed ? "lime" : "amber"} />
            <Pulse label="Recent Window" value={`${String(anti.recentSample ?? 0)} trades / ${percent(n(anti.recentDelta))}`} tone={n(anti.recentSample) >= 200 ? "lime" : "amber"} />
            <Pulse label="Validation Window" value={`${String(anti.validationSample ?? 0)} trades / ${percent(n(anti.validationDelta))}`} tone={n(anti.validationSample) >= 100 ? "lime" : "amber"} />
            <Pulse label="Execution" value={`${percent(n(execution.winRate))} win / ${n(execution.avgSlippageBps).toFixed(2)} bps`} tone={toneFromScore(n(execution.score))} />
          </div>
        </Panel>
      </div>

      <Panel title="Multi-Source Data Fusion" icon={DatabaseZap}>
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {providers.map((provider) => (
            <div key={text(provider.id, "provider")} className="border border-white/10 bg-black/35 p-4">
              <div className="mb-3 flex items-center justify-between gap-3">
                <p className="text-sm font-semibold text-white">{text(provider.label, "Provider")}</p>
                <StateBadge state={text(provider.status, "missing")} />
              </div>
              <p className="font-mono text-sm text-zinc-400">
                Latency {provider.latencyMs === null || provider.latencyMs === undefined ? "n/a" : `${String(provider.latencyMs)}ms`}
              </p>
            </div>
          ))}
        </div>
      </Panel>
    </div>
  );
}

function SymbolsPanel({ symbols }: { symbols: Record<string, unknown>[] }) {
  return (
    <Panel title="Per-Pair Runtime" icon={RadioTower}>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {symbols.map((symbol) => (
          <div key={String(symbol.symbol)} className="border border-white/10 bg-zinc-950/70 p-4">
            <div className="mb-4 flex items-center justify-between gap-2">
              <div>
                <p className="font-mono text-xl font-black text-white">{String(symbol.symbol)}</p>
                <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">{text(symbol.strategy, "strategy_pending")}</p>
              </div>
              <StateBadge state={text(symbol.state, "unknown")} />
            </div>
            <div className="grid grid-cols-3 gap-2 text-sm">
              <MiniStat label="Conf" value={percent(n(symbol.confidence))} />
              <MiniStat label="Spread" value={String(symbol.spread ?? "n/a")} />
              <MiniStat label="Risk" value={percent(n(symbol.openRiskPct))} />
            </div>
            <p className="mt-4 min-h-10 text-sm leading-5 text-zinc-300">{text(symbol.blocker, "No active blocker.")}</p>
          </div>
        ))}
      </div>
    </Panel>
  );
}

function OrdersPanel({ orders, trades, chartsReady }: { orders: Record<string, unknown>[]; trades: Record<string, unknown>[]; chartsReady: boolean }) {
  const slippageData = orders.slice(0, 14).reverse().map((order, index) => ({
    name: `${String(order.symbol ?? "?")}-${index + 1}`,
    slippage: n(order.slippageBps),
    fee: n(order.fee),
  }));
  return (
    <div className="grid gap-4 lg:grid-cols-[0.9fr_1.1fr]">
      <Panel title="Fees / Slippage" icon={CircleDollarSign}>
        <ChartFrame short ready={chartsReady}>
          {({ width, height }) => (
            <BarChart data={slippageData} width={width} height={height}>
              <CartesianGrid stroke="rgba(255,255,255,0.07)" vertical={false} />
              <XAxis dataKey="name" hide />
              <YAxis stroke="#94a3b8" tickLine={false} axisLine={false} width={38} />
              <Tooltip contentStyle={{ background: "#05070a", border: "1px solid rgba(34,211,238,.25)", color: "#fff" }} />
              <Bar dataKey="slippage" fill="#f59e0b" radius={0} />
              <Bar dataKey="fee" fill="#22d3ee" radius={0} />
            </BarChart>
          )}
        </ChartFrame>
      </Panel>
      <Panel title="Order / Trade Audit" icon={Terminal}>
        <div className="grid gap-3">
          <EventList rows={[...orders.slice(0, 8), ...trades.slice(0, 6)]} empty="No order or trade events recorded." />
        </div>
      </Panel>
    </div>
  );
}

function RiskPanel({ bot, risks }: { bot: Record<string, unknown>; risks: Record<string, unknown>[] }) {
  return (
    <div className="grid gap-4 lg:grid-cols-[0.75fr_1.25fr]">
      <Panel title="Risk Envelope" icon={ShieldAlert}>
        <div className="grid gap-3">
          <Pulse label="Daily PnL" value={usd(n(bot.pnlToday))} tone={n(bot.pnlToday) >= 0 ? "lime" : "rose"} />
          <Pulse label="Open Risk" value={percent(n(bot.openRiskPct))} tone="amber" />
          <Pulse label="Drawdown" value={percent(n(bot.drawdownPct))} tone="rose" />
          <Pulse label="Kill State" value={text(bot.killState, "NONE")} tone={text(bot.killState, "NONE") === "NONE" ? "lime" : "rose"} />
        </div>
      </Panel>
      <Panel title="Circuit Breakers" icon={Siren}>
        <EventList rows={risks} empty="No circuit-breaker events recorded." />
      </Panel>
    </div>
  );
}

function FundedPanel({
  funded,
  onFundedUpdate,
}: {
  funded: { config: Record<string, unknown>; status: Record<string, unknown> };
  onFundedUpdate: (funded: { config: Record<string, unknown>; status: Record<string, unknown> }) => void;
}) {
  const [form, setForm] = useState<Record<string, unknown>>({ ...DEFAULT_FUNDED_CONFIG, ...(funded.config ?? {}) });
  const [saveState, setSaveState] = useState<"idle" | "saving" | "saved" | "error">("idle");
  const status = funded.status ?? {};
  const account = (status.account ?? {}) as Record<string, unknown>;

  const setField = (key: string, value: unknown) => {
    setForm((current) => ({ ...current, [key]: value }));
    setSaveState("idle");
  };

  const setGroup = (group: string) => {
    const preset = FUNDED_PRESETS[group] ?? {};
    setForm((current) => ({ ...current, ...preset, group }));
    setSaveState("idle");
  };

  const save = async () => {
    setSaveState("saving");
    const body = {
      enabled: Boolean(form.enabled),
      group: text(form.group, DEFAULT_FUNDED_CONFIG.group),
      phase: text(form.phase, DEFAULT_FUNDED_CONFIG.phase),
      startingBalance: n(form.startingBalance, DEFAULT_FUNDED_CONFIG.startingBalance),
      profitTargetPct: n(form.profitTargetPct, DEFAULT_FUNDED_CONFIG.profitTargetPct),
      dailyDrawdownPct: n(form.dailyDrawdownPct, DEFAULT_FUNDED_CONFIG.dailyDrawdownPct),
      maxDrawdownPct: n(form.maxDrawdownPct, DEFAULT_FUNDED_CONFIG.maxDrawdownPct),
      trailingDrawdown: Boolean(form.trailingDrawdown),
      baseRiskPct: n(form.baseRiskPct, DEFAULT_FUNDED_CONFIG.baseRiskPct),
      maxOpenRiskPct: n(form.maxOpenRiskPct, DEFAULT_FUNDED_CONFIG.maxOpenRiskPct),
      dailyResetTimezone: text(form.dailyResetTimezone, DEFAULT_FUNDED_CONFIG.dailyResetTimezone),
    };
    const response = await fetch("/api/funded", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      setSaveState("error");
      return;
    }
    onFundedUpdate(await response.json());
    setSaveState("saved");
  };

  const statusTone = fundedTone(text(status.status, "disabled"));
  const nearStop = n(status.dailyLossRemainingUsd) <= 0 || n(status.maxLossRemainingUsd) <= 0;

  return (
    <div className="grid gap-4 xl:grid-cols-[1.2fr_0.8fr]">
      <Panel title="Funded Mode Guard" icon={ShieldCheck}>
        <div className="grid gap-4">
          <div className="grid gap-3 md:grid-cols-4">
            <Pulse label="Status" value={text(status.status, "disabled")} tone={statusTone} />
            <Pulse label="MT5 Source" value={status.mt5Derived ? "live bridge" : "fallback"} tone={status.mt5Derived ? "lime" : "amber"} />
            <Pulse label="Account" value={text(account.account, "not linked")} tone="cyan" />
            <Pulse label="Risk Throttle" value={percent(n(status.riskThrottle, 1))} tone={nearStop ? "rose" : statusTone} />
          </div>

          <div className="grid gap-3 md:grid-cols-3">
            <MiniStat label="Equity" value={usd(n(account.equity))} />
            <MiniStat label="Balance" value={usd(n(account.balance))} />
            <MiniStat label="Free margin" value={usd(n(account.freeMargin))} />
            <MiniStat label="Start balance" value={usd(n(status.startingBalance))} />
            <MiniStat label="Target equity" value={usd(n(status.targetEquity))} />
            <MiniStat label="Needed to pass" value={usd(n(status.neededToPass))} />
            <MiniStat label="Daily floor" value={usd(n(status.dailyLossFloor))} />
            <MiniStat label="Daily buffer" value={usd(n(status.dailyLossRemainingUsd))} />
            <MiniStat label="Overall buffer" value={usd(n(status.maxLossRemainingUsd))} />
            <MiniStat label="Max risk/trade" value={usd(n(status.maxRiskPerTradeUsd))} />
            <MiniStat label="Max open risk" value={usd(n(status.maxOpenRiskUsd))} />
            <MiniStat label="Open positions" value={String(account.openPositions ?? 0)} />
          </div>

          <div className="grid gap-3">
            <ProgressLine label="Pass progress" value={n(status.passProgressPct)} tone="lime" />
            <ProgressLine label="Daily buffer" value={n(status.dailyBufferPct)} tone={n(status.dailyBufferPct) <= 0.25 ? "rose" : "cyan"} />
            <ProgressLine label="Overall buffer" value={n(status.maxBufferPct)} tone={n(status.maxBufferPct) <= 0.25 ? "rose" : "violet"} />
          </div>

          <p className="border border-cyan-300/20 bg-cyan-300/10 p-4 text-sm leading-6 text-cyan-100">
            Guard reason: {text(status.fundedGuardReason, "funded_mode_waiting")}. The funded panel calculates limits from MT5 bridge equity, balance,
            day-start equity, day-high equity, floating PnL, and open-position state. It does not place trades.
          </p>
        </div>
      </Panel>

      <Panel title="Funding Rules" icon={Gauge}>
        <div className="grid gap-3">
          <label className="flex items-center justify-between gap-3 border border-white/10 bg-white/[0.03] p-3 text-sm text-zinc-300">
            <span>Funded mode</span>
            <input
              type="checkbox"
              checked={Boolean(form.enabled)}
              onChange={(event) => setField("enabled", event.target.checked)}
              className="size-5 accent-cyan-300"
            />
          </label>
          <Field label="Funded group">
            <select value={text(form.group, "custom")} onChange={(event) => setGroup(event.target.value)} className={fieldClass}>
              <option value="custom">Custom</option>
              <option value="ftmo">FTMO-style</option>
              <option value="fundednext">FundedNext-style</option>
              <option value="the5ers">The5ers-style</option>
              <option value="apex">Apex futures-style</option>
              <option value="topstep">Topstep-style</option>
            </select>
          </Field>
          <Field label="Phase">
            <select value={text(form.phase, "evaluation")} onChange={(event) => setField("phase", event.target.value)} className={fieldClass}>
              <option value="evaluation">Evaluation</option>
              <option value="verification">Verification</option>
              <option value="funded_live">Funded live</option>
              <option value="personal_100">Personal $100 build</option>
            </select>
          </Field>
          <NumberField label="Starting balance" value={n(form.startingBalance)} onChange={(value) => setField("startingBalance", value)} prefix="$" step="1" />
          <PercentField label="Profit target" value={n(form.profitTargetPct)} onChange={(value) => setField("profitTargetPct", value)} />
          <PercentField label="Daily drawdown" value={n(form.dailyDrawdownPct)} onChange={(value) => setField("dailyDrawdownPct", value)} />
          <PercentField label="Overall drawdown" value={n(form.maxDrawdownPct)} onChange={(value) => setField("maxDrawdownPct", value)} />
          <PercentField label="Base risk/trade" value={n(form.baseRiskPct)} onChange={(value) => setField("baseRiskPct", value)} />
          <PercentField label="Max open risk" value={n(form.maxOpenRiskPct)} onChange={(value) => setField("maxOpenRiskPct", value)} />
          <label className="flex items-center justify-between gap-3 border border-white/10 bg-white/[0.03] p-3 text-sm text-zinc-300">
            <span>Trailing drawdown</span>
            <input
              type="checkbox"
              checked={Boolean(form.trailingDrawdown)}
              onChange={(event) => setField("trailingDrawdown", event.target.checked)}
              className="size-5 accent-cyan-300"
            />
          </label>
          <button
            type="button"
            onClick={() => void save()}
            disabled={saveState === "saving"}
            className="h-11 border border-cyan-300/50 bg-cyan-300 px-4 text-sm font-black uppercase tracking-[0.18em] text-black transition hover:bg-cyan-200 disabled:cursor-wait disabled:opacity-70"
          >
            {saveState === "saving" ? "Saving" : "Save Funded Mode"}
          </button>
          <p className={`text-sm ${saveState === "error" ? "text-rose-200" : saveState === "saved" ? "text-lime-200" : "text-zinc-500"}`}>
            {saveState === "error" ? "Save failed." : saveState === "saved" ? "Saved." : "Changes apply to the Nexus funded calculator."}
          </p>
        </div>
      </Panel>
    </div>
  );
}

function DataPanel({ risks, symbols }: { risks: Record<string, unknown>[]; symbols: Record<string, unknown>[] }) {
  const dataRows = risks.filter((risk) => /data|gap|stale|proxy|book|ohlcv/i.test(`${risk.type ?? ""} ${risk.reason ?? ""}`));
  return (
    <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
      <Panel title="Data Integrity" icon={DatabaseZap}>
        <EventList rows={dataRows} empty="No data integrity alerts recorded." />
      </Panel>
      <Panel title="Source Status" icon={ShieldCheck}>
        <div className="grid gap-3">
          {symbols.map((symbol) => {
            const payload = (symbol.payload ?? {}) as Record<string, unknown>;
            const native = payload.native_data === true || payload.source === "hyperliquid";
            const stale = payload.stale === true;
            return (
              <Pulse
                key={String(symbol.symbol)}
                label={String(symbol.symbol)}
                value={`${native ? "native" : "proxy/bridge"}${stale ? " stale" : ""}`}
                tone={stale ? "rose" : native ? "lime" : "amber"}
              />
            );
          })}
        </div>
      </Panel>
    </div>
  );
}

function OpsPanel({ commands }: { commands: Record<string, unknown>[] }) {
  return (
    <Panel title="Telegram / Ops Audit" icon={Terminal}>
      <div className="grid gap-3">
        {commands.length ? (
          commands.map((command) => (
            <div key={String(command.commandId ?? command.id)} className="border border-white/10 bg-white/[0.03] p-4">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <p className="font-mono text-sm text-cyan-200">{String(command.commandId ?? "command")}</p>
                <StateBadge state={text(command.status, "unknown")} />
              </div>
              <p className="mt-2 text-sm text-white">{text(command.action, "unknown_action")}</p>
              <p className="mt-1 text-xs uppercase tracking-[0.14em] text-zinc-500">{shortDate(command.createdAt)}</p>
              {text(command.status, "") === "pending_confirmation" ? (
                <p className="mt-3 text-sm text-amber-200">Pending explicit confirmation before execution.</p>
              ) : null}
            </div>
          ))
        ) : (
          <p className="text-sm text-zinc-400">No command requests recorded.</p>
        )}
      </div>
    </Panel>
  );
}

function TrajectoryPanel({ equityCurve, chartsReady }: { equityCurve: Record<string, unknown>[]; chartsReady: boolean }) {
  const latest = equityCurve.at(-1) ?? {};
  const ratio = n(latest.equity) / 100000;
  return (
    <div className="grid gap-4 lg:grid-cols-[1.4fr_0.8fr]">
      <Panel title="$100 to $100K Trajectory" icon={TrendingUp}>
        <ChartFrame ready={chartsReady}>
          {({ width, height }) => (
            <ComposedChart data={equityCurve} width={width} height={height}>
              <CartesianGrid stroke="rgba(255,255,255,0.07)" vertical={false} />
              <XAxis dataKey="timestamp" tickFormatter={time} stroke="#94a3b8" tickLine={false} axisLine={false} minTickGap={30} />
              <YAxis tickFormatter={compactNumber} stroke="#94a3b8" tickLine={false} axisLine={false} width={56} scale="log" domain={["auto", "auto"]} />
              <Tooltip contentStyle={{ background: "#05070a", border: "1px solid rgba(34,211,238,.25)", color: "#fff" }} labelFormatter={shortDate} formatter={(value) => usd(Number(value))} />
              <Legend />
              <Area dataKey="upside" fill="rgba(34,211,238,.10)" stroke="#22d3ee" strokeWidth={1.4} />
              <Line type="monotone" dataKey="target" stroke="#a3e635" strokeWidth={2} dot={false} />
              <Line type="monotone" dataKey="equity" stroke="#ffffff" strokeWidth={2.5} dot={false} />
              <Line type="monotone" dataKey="downside" stroke="#fb7185" strokeWidth={1.5} dot={false} />
            </ComposedChart>
          )}
        </ChartFrame>
      </Panel>
      <Panel title="Trajectory Guardrails" icon={AlertTriangle}>
        <div className="grid gap-3">
          <Pulse label="Progress" value={percent(ratio)} tone="cyan" />
          <Pulse label="Forecast Type" value="speculative" tone="amber" />
          <Pulse label="Risk Input" value="not used" tone="lime" />
          <p className="border border-amber-300/25 bg-amber-300/10 p-4 text-sm leading-6 text-amber-100">
            This path is a visual target scenario only. It does not authorize higher risk, live trade placement, leverage changes, or parameter tuning.
          </p>
        </div>
      </Panel>
    </div>
  );
}

const fieldClass =
  "h-11 w-full border border-white/10 bg-black/55 px-3 font-mono text-sm text-white outline-none transition focus:border-cyan-300";

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="grid gap-2 text-sm">
      <span className="text-xs uppercase tracking-[0.18em] text-zinc-500">{label}</span>
      {children}
    </label>
  );
}

function NumberField({
  label,
  value,
  onChange,
  prefix = "",
  step = "0.01",
}: {
  label: string;
  value: number;
  onChange: (value: number) => void;
  prefix?: string;
  step?: string;
}) {
  return (
    <Field label={label}>
      <div className="flex items-center border border-white/10 bg-black/55 focus-within:border-cyan-300">
        {prefix ? <span className="px-3 font-mono text-sm text-zinc-500">{prefix}</span> : null}
        <input
          type="number"
          value={Number.isFinite(value) ? value : 0}
          step={step}
          min="0"
          onChange={(event) => onChange(n(event.target.value))}
          className="h-11 min-w-0 flex-1 bg-transparent px-3 font-mono text-sm text-white outline-none"
        />
      </div>
    </Field>
  );
}

function PercentField({ label, value, onChange }: { label: string; value: number; onChange: (value: number) => void }) {
  return (
    <NumberField
      label={label}
      value={Number((value * 100).toFixed(3))}
      onChange={(next) => onChange(next / 100)}
      prefix="%"
      step="0.1"
    />
  );
}

function ProgressLine({ label, value, tone }: { label: string; value: number; tone: "cyan" | "lime" | "rose" | "amber" | "violet" }) {
  const colors = {
    cyan: "bg-cyan-300",
    lime: "bg-lime-300",
    rose: "bg-rose-300",
    amber: "bg-amber-300",
    violet: "bg-violet-300",
  };
  const width = `${Math.max(0, Math.min(100, value * 100))}%`;
  return (
    <div className="border border-white/10 bg-white/[0.03] p-3">
      <div className="mb-2 flex items-center justify-between gap-3">
        <p className="text-sm text-zinc-400">{label}</p>
        <p className="font-mono text-sm font-bold text-white">{percent(value)}</p>
      </div>
      <div className="h-2 bg-white/10">
        <div className={`h-full ${colors[tone]}`} style={{ width }} />
      </div>
    </div>
  );
}

function fundedTone(status: string): "cyan" | "lime" | "rose" | "amber" | "violet" {
  if (status.includes("hard") || status.includes("breach")) return "rose";
  if (status.includes("defensive") || status.includes("caution") || status.includes("ready")) return "amber";
  if (status.includes("passed") || status.includes("normal")) return "lime";
  return "cyan";
}

function readinessTone(status: string): "cyan" | "lime" | "rose" | "amber" | "violet" {
  if (status.includes("hard")) return "rose";
  if (status.includes("repair") || status.includes("observe")) return "amber";
  if (status.includes("expand")) return "lime";
  if (status.includes("protect")) return "violet";
  return "cyan";
}

function toneFromScore(score: number): "cyan" | "lime" | "rose" | "amber" | "violet" {
  if (score >= 0.72) return "lime";
  if (score >= 0.5) return "amber";
  if (score <= 0.25) return "rose";
  return "cyan";
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === "object" && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function Metric({
  icon: Icon,
  label,
  value,
  sub,
  tone,
}: {
  icon: LucideIcon;
  label: string;
  value: string;
  sub: string;
  tone: "cyan" | "lime" | "rose" | "amber" | "violet";
}) {
  const toneClass = {
    cyan: "text-cyan-200 border-cyan-300/30 bg-cyan-300/10",
    lime: "text-lime-200 border-lime-300/30 bg-lime-300/10",
    rose: "text-rose-200 border-rose-300/30 bg-rose-300/10",
    amber: "text-amber-200 border-amber-300/30 bg-amber-300/10",
    violet: "text-violet-200 border-violet-300/30 bg-violet-300/10",
  }[tone];
  return (
    <div className="border border-white/10 bg-black/45 p-4 shadow-[0_0_24px_rgba(0,0,0,0.25)] backdrop-blur">
      <div className={`mb-4 inline-grid size-9 place-items-center border ${toneClass}`}>
        <Icon size={18} />
      </div>
      <p className="text-xs uppercase tracking-[0.22em] text-zinc-500">{label}</p>
      <p className="mt-1 break-words font-mono text-2xl font-black text-white sm:text-3xl">{value}</p>
      <p className="mt-2 text-sm text-zinc-400">{sub}</p>
    </div>
  );
}

function Panel({ title, icon: Icon, children }: { title: string; icon: LucideIcon; children: ReactNode }) {
  return (
    <section className="border border-white/10 bg-black/55 p-4 shadow-[0_0_60px_rgba(0,0,0,0.28)] backdrop-blur">
      <div className="mb-4 flex items-center gap-2 border-b border-white/10 pb-3">
        <Icon size={17} className="text-cyan-200" />
        <h2 className="text-sm font-bold uppercase tracking-[0.2em] text-white">{title}</h2>
      </div>
      {children}
    </section>
  );
}

function ChartFrame({
  children,
  short = false,
  ready,
}: {
  children: (size: { width: number; height: number }) => ReactNode;
  short?: boolean;
  ready: boolean;
}) {
  const ref = useRef<HTMLDivElement | null>(null);
  const height = short ? 260 : 360;
  const [width, setWidth] = useState(0);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const update = () => setWidth(Math.max(1, Math.floor(node.getBoundingClientRect().width)));
    update();
    const observer = new ResizeObserver(update);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  return (
    <div
      ref={ref}
      className={`${short ? "h-[260px]" : "h-[360px]"} w-full min-w-0 overflow-hidden`}
      style={{ minWidth: 1, minHeight: height }}
    >
      {ready && width > 1 ? children({ width, height }) : <div className="h-full w-full border border-cyan-300/10 bg-cyan-300/[0.03]" />}
    </div>
  );
}

function Pulse({ label, value, tone }: { label: string; value: string; tone: "cyan" | "lime" | "rose" | "amber" | "violet" }) {
  const colors = {
    cyan: "border-cyan-300/25 text-cyan-100",
    lime: "border-lime-300/25 text-lime-100",
    rose: "border-rose-300/25 text-rose-100",
    amber: "border-amber-300/25 text-amber-100",
    violet: "border-violet-300/25 text-violet-100",
  };
  return (
    <div className={`flex items-center justify-between gap-4 border bg-white/[0.03] p-4 ${colors[tone]}`}>
      <p className="text-sm text-zinc-400">{label}</p>
      <p className="break-words text-right font-mono text-lg font-bold">{value}</p>
    </div>
  );
}

function StatusPill({ label, value, tone }: { label: string; value: string; tone: "cyan" | "lime" | "rose" }) {
  const colors = {
    cyan: "border-cyan-300/25 bg-cyan-300/10 text-cyan-100",
    lime: "border-lime-300/25 bg-lime-300/10 text-lime-100",
    rose: "border-rose-300/25 bg-rose-300/10 text-rose-100",
  };
  return (
    <span className={`inline-flex h-10 items-center gap-2 border px-3 ${colors[tone]}`}>
      <span className="text-zinc-400">{label}</span>
      <span className="font-mono font-bold">{value}</span>
    </span>
  );
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return (
    <div className="border border-white/10 bg-black/40 p-2">
      <p className="text-[10px] uppercase tracking-[0.18em] text-zinc-500">{label}</p>
      <p className="mt-1 break-words font-mono text-sm text-white">{value}</p>
    </div>
  );
}

function StateBadge({ state }: { state: string }) {
  const lower = state.toLowerCase();
  const tone = lower.includes("block") || lower.includes("reject") || lower.includes("kill") ? "#fb7185" : lower.includes("hold") || lower.includes("pending") ? "#f59e0b" : "#a3e635";
  return (
    <span
      className="inline-flex min-h-8 items-center border px-2 text-xs font-bold uppercase tracking-[0.16em]"
      style={{ borderColor: `${tone}66`, color: tone, backgroundColor: `${tone}18` }}
    >
      {state}
    </span>
  );
}

function EventList({ rows, empty }: { rows: Record<string, unknown>[]; empty: string }) {
  if (!rows.length) return <p className="text-sm text-zinc-400">{empty}</p>;
  return (
    <div className="grid gap-2">
      {rows.slice(0, 14).map((row, index) => {
        const status = text(row.status, text(row.state, "event"));
        return (
          <div
            key={String(row.id ?? row.externalId ?? row.commandId ?? `${row.type}-${index}`)}
            className="grid gap-2 border border-white/10 bg-white/[0.03] p-3 sm:grid-cols-[1fr_auto] sm:items-center"
          >
            <div>
              <p className="font-mono text-sm text-white">
                {String(row.symbol ?? row.action ?? row.type ?? "event")} <span className="text-zinc-500">{String(row.side ?? "")}</span>
              </p>
              <p className="mt-1 text-sm text-zinc-400">{text(row.reason, text(row.requestedText, text(row.type, "")))}</p>
            </div>
            <div className="flex items-center gap-2 sm:justify-end">
              <StateBadge state={status} />
              <span className="text-xs text-zinc-500">{shortDate(row.occurredAt ?? row.createdAt)}</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}
