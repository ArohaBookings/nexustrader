# APEX

APEX is a local, MT5-native autonomous trading daemon for a tight symbol roster: XAUUSD, EURUSD, GBPUSD, USDJPY, and BTCUSD. It runs a hybrid engine:

- fast scalping entries (M1 trigger + M5 confirmation),
- set-and-forget trend legs (H1/H4 alignment),
- multi-position scaling (up to configured caps),
- per-position BE/trailing/partials plus basket controls.

This build is intentionally safety-first:

- Default mode is `DEMO`.
- `LIVE` requires explicit `system.trading_enabled: true` and `system.live_trading_enabled: true`.
- Max positions are capped by `system.max_positions_total` and `system.max_positions_per_symbol` (default `50` total / `10` per symbol).
- Max positions per symbol are capped separately.
- Every order requires SL and TP at submission time.
- High-impact news is fail-closed by default.
- Friday 20:00 GMT activates a soft no-new-trades state.
- Rolling drawdown and daily-loss circuit breakers can trip a soft or hard kill switch.

## Layout

```text
apex_bot/
  config/
  data/
  models/
  scripts/
  src/
  tests/
```

Key modules:

- `src/main.py`: daemon entrypoint, training entrypoint, backtest entrypoint
- `src/bridge_server.py`: local HTTP bridge API for MT5 EA pull/report flow
- `src/mt5_client.py`: MT5 connection, symbol resolution, order operations
- `src/feature_engineering.py`: multi-timeframe feature builder (50+ features)
- `src/strategy_engine.py`: trend continuation, breakout-retest, range reversal
- `src/grid_scalper.py`: bounded `XAUUSD_M5_GRID_SCALPER` cycle engine
- `src/news_engine.py`: API-backed news calendar + fail-closed fallback schedule
- `src/ai_gate.py`: trade scorer plus final execution approval layer
- `src/risk_engine.py`: hard caps, per-symbol throttles, volatility shock pauses, kill-switch decisions
- `src/execution.py`: SQLite trade journal, idempotent execution path
- `src/position_manager.py`: per-position partials, trailing, time-stop planning
- `src/backtest.py`: walk-forward-friendly event simulation and Monte Carlo
- `src/train.py`: local model training and artifact persistence
- `mt5_bridge/ApexBridgeEA.mq5`: MT5 EA bridge (WebRequest pull + execution reports)
- `config/presets/high_frequency_scalp.yaml`: preset tuned for frequent entries

## Install

From `/Users/leobons/Library/Mobile Documents/com~apple~CloudDocs/Kimi_Agent_Set & Forget Trade Bot/apex_bot`:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If your MT5 Python package is already installed globally, keep the same interpreter the MT5 terminal is configured for.
On macOS, `MetaTrader5` may not be available on PyPI for your interpreter. The default install remains usable because APEX safely falls back to simulated mode when that package is missing.

## Configure

1. Copy values from `config/secrets.env` into your shell environment or a local `.env` loader.
2. Review `config/settings.yaml` before running live logic, especially `system.symbols`, `system.max_positions_per_symbol`, and the `news` section.
3. `xau_grid_scalper.enabled` controls the dedicated XAUUSD M5 bounded grid cycle logic.
4. `grid.enabled` is still the old optional module and remains separate from `xau_grid_scalper`.

Startup automatically loads `config/secrets.env` then `secrets.env` (repo root) for any missing env vars. Existing shell env values always take precedence.

Important environment variables:

- `MT5_LOGIN`
- `MT5_PASSWORD`
- `MT5_SERVER`
- `MT5_TERMINAL_PATH`
- `LIVE_TRADING`
- `APEX_MODE`
- `APEX_SYMBOLS`
- `NEWS_API_KEY`
- `BRIDGE_AUTH_TOKEN` (optional)

Default symbol roster in `config/settings.yaml`:

```yaml
system:
  symbols:
    - XAUUSD
    - EURUSD
    - GBPUSD
    - USDJPY
    - BTCUSD
  always_on_symbols:
    - BTCUSD
```

Override it from the shell with:

```bash
export APEX_SYMBOLS="XAUUSD,EURUSD"
```

`always_on_symbols` is for instruments intended to run 24/7 (for example BTCUSD). For non-`always_on_symbols`, Friday cutoff and weekend blocks stay active.

To apply the high-frequency preset quickly, copy values from `config/presets/high_frequency_scalp.yaml` into `config/settings.yaml`.

News engine defaults:

- `news.provider: newsapi`
- `news.block_high_impact: true`
- `news.block_window_minutes_before: 30`
- `news.block_window_minutes_after: 15`
- `news.fail_open: false`

`NEWS_API_KEY` is read from the environment first, then `news.api_key` in `config/settings.yaml`.
If the API is unavailable and `news.fail_open` is still `false`, APEX marks news as unknown and blocks new entries only during the configured main-session fallback windows.

Bridge defaults (`config/settings.yaml`):

```yaml
bridge:
  enabled: true
  host: 127.0.0.1
  port: 8000
  poll_ttl_seconds: 10
  magic_number: 20260304
```

Action Orchestrator defaults:

- `/v1/pull` returns at most one action.
- Actions are persisted with statuses: `QUEUED`, `DELIVERED`, `ACKED`, `EXPIRED`, `CANCELLED`, `FAILED_VALIDATION`, `CLOSED`.
- Duplicate actions are blocked by a persistent `dedupe_key`.
- New `OPEN_MARKET` actions are delivery-locked per `(account, magic, symbol)` to prevent spray/replay.
- Delivered actions are leased (`lease_seconds`) and are not re-served while leased.
- Re-delivery is disabled by default (`allow_redelivery: false`); if enabled, only expired leases for the same `(account, magic)` can be re-served.
- `/v1/ack` can acknowledge delivered actions explicitly (`ACKED`, `CANCELLED`, `REJECTED`).
- Bridge keeps persistent per-symbol runtime state (grid cycle id/legs, last entry, cycle risk used, last block reason).
- Pull-time reality sync is supported with optional query params: `open_count`, `net_lots`, `avg_entry`, `floating_pnl`.
- If reality fields are missing, bridge falls back to execution reports + persisted estimates and tracks confidence (`high|medium|low`) and drift flags.
- XAU grid entries use cycle-aware gating (M5-only option, step-distance adds, max legs, cycle risk budget).
- Grid block reasons are explicit (`grid_step_not_reached`, `spread_too_wide`, `cycle_risk_exhausted`, `max_legs_reached`, `cooldown_active`, `recovery_mode_active`, `session_filter_block`).
- Grid cycle recycle is enforced (`EXIT -> COOLDOWN -> IDLE`) so profit-taking cycles restart cleanly.
- NAS and OIL are supported with broker alias mapping and dedicated strategy keys (`NAS_SESSION_SCALPER`, `OIL_INVENTORY_SCALPER`).
- OIL has an explicit `NEWS_ARMED` window mode for inventory/event trading with stricter confluence/spread controls.
- Stop levels are normalized and validated server-side against `config/symbol_rules.yaml`.
- If validation fails, action status becomes `FAILED_VALIDATION` with a snapshot in storage.
- Optional fallback (`allow_open_without_stops_fallback=true`): if broker-compatible stops are not immediately possible, the bridge can send `OPEN_MARKET` with `sl=0,tp=0` and queue a `MODIFY_SLTP` after execution report.
- Risk/confluence gates run in bridge before delivery (probability, EV, confluence, risk budget, cooldown, hourly limits, drawdown guards).
- Session-aware dynamic spread thresholds + bounded undertrading governor are supported (loosen spread/confluence slightly only when safe).
- `NEWS_ARMED` mode is separate from normal flow and only allows high-confidence event-window trading with stricter limits.
- Omega runtime adds local regime state (`TRENDING_UP`, `TRENDING_DOWN`, `RANGING`, `VOLATILE`, `NEWS_SHOCK`) and bounded Kelly attenuation for entry sizing.
- Rule change governance exists server-side (`bridge.orchestrator.rule_change`) and only permits bounded parameter updates after minimum samples, shadow validation, and cooldown.

XAUUSD bounded grid scalper defaults (`config/settings.yaml`):

```yaml
xau_grid_scalper:
  enabled: true
  symbol: XAUUSD
  timeframe: M5
  allowed_sessions: [LONDON, OVERLAP, NEW_YORK]
  max_levels: 6
  max_open_positions_symbol: 10
  max_open_cycles: 1
  base_lot: 0.01
  profit_target_usd: 0.6
  stop_atr_k: 2.5
  step_atr_k: 0.35
```

The grid scalper starts a cycle on EMA50 stretch + deceleration/extreme confirmation, adds bounded levels on adverse ATR steps, and exits quickly on small net-profit mean reversion, hard stop, or time limit.
When enabled, XAUUSD M5 uses this grid cycle engine for scalp entries while higher-timeframe `SET_FORGET_H1_H4` / scale-ins can still run in parallel.

NAS/OIL additions (`config/settings.yaml`):

- `nas_strategy.*`: session-scalper with NY cash-open + power-hour emphasis.
- `oil_strategy.*`: inventory-aware NY/overlap scalper with stricter `news_armed` windows.
- `system.symbol_mapping`: includes aliases (`US100/USTEC/NAS*`, `USOIL/XTIUSD/OILUSD/WTI/CL`).

## Run

Safe local preview without MT5 login:

Set `system.mode: DRY_RUN` in `config/settings.yaml`, then run:

```bash
PYTHONPATH=. python3 -m src.main --once
```

DRY_RUN uses cached candles when available, never places orders, and prints the multi-symbol dashboard. If a symbol has no cached candles yet, the dashboard shows `no cache yet`.
The dashboard prints both UTC and local time plus the active session state.

Connectivity verification only:

```bash
PYTHONPATH=. python3 -m src.main --verify
```

`--verify` resolves symbols, attempts MT5 connectivity, and prints `account_info`, `terminal_info`, and `version`. It never places orders.

Bridge server only (EA pulls signals, strategy loop disabled):

```bash
PYTHONPATH=. python3 -m src.main --bridge-only
```

Bridge + strategy loop (recommended for EA execution):

```bash
PYTHONPATH=. python3 -m src.main --bridge-serve
```

Bridge smoke check (no broker required):

```bash
PYTHONPATH=. ./.venv/bin/python scripts/smoke_pull_action.py
```

Paper simulation (no MT5 orders, internal fills with spread/slippage assumptions):

```bash
PYTHONPATH=. python3 -m src.main --paper-sim --once
```

Evaluate recent closed-trade performance:

```bash
PYTHONPATH=. python3 -m src.main --eval-last 200
```

Detailed grouped report (overall + per symbol + per session + per setup):

```bash
PYTHONPATH=. python3 -m src.main --report
```

`--report` includes:

- grouped performance (overall, symbol, session, setup)
- `micro_survivability_last_20` (max consecutive losses, max intraday USD drawdown, fee estimate)
- `blocked_counts_total` and `blocked_counts_last_24h` (entry denials by reason)

OpenAI connectivity self-test:

```bash
PYTHONPATH=. python3 -m src.main --ai-test
```

Bounded demo smoke test:

```bash
PYTHONPATH=. python3 -m src.main --smoke-demo
```

`--smoke-demo` first runs `--verify`, then executes a bounded number of scan loops in `DEMO` or `PAPER` mode and may place demo orders only if every gate approves. Use `--smoke-loops 20` (or another value) to control loop count.

## MT5 EA Bridge Setup

1. For local Mac MT5/WebRequest mode, keep these values in `config/secrets.env`:

```bash
LIVE_TRADING=true
APEX_MT5_RUNTIME_MODE=EA_BRIDGE
```

`APEX_MT5_RUNTIME_MODE=EA_BRIDGE` means the Python bridge does not require the `MetaTrader5` Python package. MT5 executes through the EA in `mt5_bridge/ApexBridgeEA.mq5`, which polls the local bridge.

2. Start Python engine with bridge enabled:

```bash
PYTHONPATH=. python3 scripts/start_bridge_prod.py
```

3. In MT5 terminal, open `Tools -> Options -> Expert Advisors` and add this URL to allowed WebRequest list:
`http://127.0.0.1:8000`

4. Open `mt5_bridge/ApexBridgeEA.mq5` in MetaEditor, compile it, then attach to each symbol chart you want traded (`XAUUSD`, `EURUSD`, `GBPUSD`, `USDJPY`, `BTCUSD`).

5. Suggested chart timeframe for the EA bridge: `M5` (the Python strategy still uses multi-timeframe features internally).

6. Confirm bridge health:
   - Browser/curl: `http://127.0.0.1:8000/health`
   - Stats endpoint: `http://127.0.0.1:8000/stats`
   - AI health endpoint: `http://127.0.0.1:8000/ai/health`
   - Omega status endpoint: `http://127.0.0.1:8000/omega/status`
   - Symbol debug endpoint: `http://127.0.0.1:8000/debug/symbol?symbol=XAUUSD`
   - MT5 Experts tab should show `[ApexBridgeEA]` pull/execution log lines.

Bridge account feed note:

- EA calls `/v1/pull` with account state, symbol execution metadata, and MT5 permission flags.
- Required live sign-off flags come from the compiled EA: `terminal_connected`, `terminal_trade_allowed`, and `mql_trade_allowed`.
- When these fields are present, dashboard equity is labeled `LIVE_BRIDGE_FEED`.
- If MT5 equity feed is unavailable, dashboard shows `INTERNAL (NO MT5 EQUITY FEED)` so internal fallback is explicit.
- In bridge mode without MT5 account metrics, risk uses a conservative internal estimate (`micro_account_mode.internal_equity_estimate`, default `50`) and never labels account state as live feed.
- Optional (non-breaking) snapshot ingestion endpoint is available:
  - `POST /v1/mt5_snapshot`
  - Payload may include `symbol`, `account`, `magic`, `bid`, `ask`, `spread_points`, `open_count`, `net_lots`, `avg_entry`, `floating_pnl`, `balance`, `equity`, `free_margin`.
  - If snapshots are not provided, bridge management still runs with pull/query + journal state.

No `npm dev` or Next.js runtime is required for live trading. The Python process is the engine; the EA is the execution bridge.

## Local Telegram Polling

For this Mac setup, Telegram should use local polling instead of a webhook. Telegram webhooks require a public HTTPS URL, but the local bridge is only on `127.0.0.1`.

One-command local Mac startup:

```bash
./scripts/start_local_mac.sh
```

This starts the bridge listener and Telegram polling sidecar in the background, with logs in `logs/local_bridge.log` and `logs/telegram_poll.log`. By default it starts the bridge listener only; set `APEX_LOCAL_START_STRATEGY=true` only after live sign-off passes and you want the strategy loop running.

For a persistent Mac setup that survives shell/Codex session exits, install the launch agents:

```bash
./scripts/install_local_mac_launch_agents.sh
```

This registers `com.apexbot.bridge` and `com.apexbot.telegram` with `launchd`, keeps them alive, and writes to the same local log files. Stop them with:

```bash
./scripts/stop_local_mac.sh
```

1. Open `https://t.me/Nexus_vantage_trader_bot` and send `/start`.

2. Claim the owner chat ID from the first incoming message:

```bash
PYTHONPATH=. python3 scripts/apex_telegram_poll.py --claim-owner --once
```

3. Run the polling sidecar while the bridge is running:

```bash
PYTHONPATH=. python3 scripts/apex_telegram_poll.py --claim-owner
```

4. Send a test message after `TELEGRAM_CHAT_ID` is present:

```bash
PYTHONPATH=. python3 scripts/apex_telegram_check.py --send-test
```

The polling sidecar forwards every Telegram update to the same local `/telegram/webhook` bridge handler, using `TELEGRAM_WEBHOOK_SECRET`. It stores its update cursor in `data/telegram_poll_offset.json` so restarts do not replay old commands.

Stop local background services:

```bash
./scripts/stop_local_mac.sh
```

AI adaptation note:

- The final AI gate now uses recent closed-trade outcomes per symbol+setup to adapt probability thresholds and size multipliers.
- Weak recent performance raises the required probability and shrinks size.
- Persistent losses tighten anti-overtrading by reducing the effective max trades/hour.
- Micro-account mode applies trade-count ramp sizing by default: very small risk at startup, then linear ramp toward configured risk after enough closed trades.
- Micro survival clamps enforce fixed USD limits for tiny accounts:
  - `micro_account_mode.min_lot`
  - `micro_account_mode.micro_max_loss_usd`
  - `micro_account_mode.micro_total_risk_usd`
  - `micro_account_mode.max_positions_total_micro`
  - `micro_account_mode.max_positions_per_symbol_micro`
  - `micro_account_mode.cooldown_minutes_after_loss`
  - `micro_account_mode.cooldown_minutes_after_win`

## Tuning: More Trades vs Safer

More trades (higher frequency):

- lower `strategy.min_atr_pct` (for example `0.65-0.75`)
- lower `strategy.min_volume_ratio` (for example `0.9-1.0`)
- increase `strategy.max_entries_per_symbol_loop` (for example `2-3`)
- increase `risk.max_trades_per_hour` (for example `8-12`)

Safer / lower churn:

- raise `strategy.min_atr_pct` and `strategy.min_volume_ratio`
- reduce `strategy.max_entries_per_symbol_loop` to `1`
- reduce `risk.max_trades_per_hour`
- increase `exits.break_even_trigger_r`

Recommended focus:

- primary: `XAUUSD` (scalp + set-and-forget)
- secondary: `EURUSD`, `GBPUSD`, `USDJPY`
- 24/7 optional: `BTCUSD` using `always_on_symbols`

Demo daemon:

```bash
./scripts/start_demo.sh
```

One-cycle health check:

```bash
PYTHONPATH=. python3 -m src.main --once
```

`--once` runs exactly one scan cycle. It prints:

- configured symbol to MT5 symbol resolution

## CLI Flags (from `src.main`)

- `--once`
- `--verify`
- `--bridge-serve`
- `--bridge-only`
- `--paper-sim`
- `--smoke-demo`
- `--smoke-loops <N>`
- `--eval-last <N>`
- `--report`
- `--ai-test`
- `--train`
- `--backtest --preset realistic|frictionless`
- per-symbol news state
- per-symbol trading allowed/blocked
- the exact reason the symbol is allowed or blocked

Backtest:

```bash
PYTHONPATH=. python -m src.main --backtest --preset realistic
PYTHONPATH=. python -m src.main --backtest --preset frictionless
```

`realistic` enables spread/slippage/commission/latency and plausibility checks.
`frictionless` is comparison-only and disables cost/friction assumptions.

Train local models:

```bash
./scripts/train_models.sh
```

Live:

```bash
./scripts/start_live.sh
```

`start_live.sh` loads local env values from `config/secrets.env`, requires `LIVE_TRADING=true`, requires Telegram/OpenAI operator telemetry keys, and then runs the fail-closed production bridge launcher. The Python process is the bot. You do not run a Next.js dev server to place trades.

## Tests

```bash
PYTHONPATH=. python3 -m unittest discover -s tests
```

Equivalent short form:

```bash
PYTHONPATH=. python -m unittest
```

The included tests focus on safety-critical logic:

- News blocking and cache-backed fallback behavior
- AI final approval behavior
- Risk cap enforcement and volatility pauses
- Per-symbol and total cap behavior used by the main loop
- Position management rules
- Execution idempotency

## Operational Notes

- `data/trades_db.sqlite` is the source of truth for executions and trade state.
- `data/kill_switch.lock` is the manual reset gate. Delete it only after the cause is fixed.
- `data/regime_history.json` stores the recent regime classification log.
- `data/news_cache.json` stores the latest calendar snapshot and TTL.
- The backtester uses conservative bar-order handling and configurable spread/slippage assumptions.
- The training pipeline uses local scikit-learn models by default as the baseline local-AI implementation. If you want to swap in XGBoost or LightGBM later, keep the same `predict` / `predict_proba` interfaces and artifact paths.
- The terminal dashboard is now multi-symbol and shows per-symbol regime, news status, open count, and gate reason each loop.
- If MT5 initialization fails on macOS, use DRY_RUN locally and run the live MT5-connected bot on a Windows VM or VPS. The official MT5 Python bridge expects a Windows terminal executable such as `metatrader64.exe`.

## Recommended Rollout

1. For a local preview, set `system.mode: DRY_RUN` and run `PYTHONPATH=. python3 -m src.main --once`.
2. Run `PYTHONPATH=. python3 -m src.main --verify` to test MT5 connectivity without placing orders.
3. Run demo mode and verify broker execution quality before increasing risk.
4. Run `PYTHONPATH=. python3 -m src.main --smoke-demo` for a bounded demo-trading check after connectivity is confirmed.
5. Run `./scripts/backtest_report.sh` and verify win rate, PF, expectancy, drawdown, and consecutive-loss behavior.
6. Train models only after enough broker-side history is available.
7. Keep live disabled until broker execution quality is confirmed.

## LIVE Checklist

Before switching to live:

1. Set `system.mode: LIVE` only when you are ready for live execution.
2. Set `system.live_trading_enabled: true` in `config/settings.yaml`.
3. Run `PYTHONPATH=. python3 -m src.main --ai-test` (fallback mode is safe if remote API is rate-limited).
4. Run `PYTHONPATH=. python3 -m src.main --verify` and confirm symbol/account checks are OK.
5. Confirm bridge endpoints respond (`/health`, `/stats`, `/ai/health`) and EA pull logs show no auth/connect errors.
6. Start with micro-account defaults enabled and review `--report` + `--eval-last 200` before increasing risk.
7. Green state looks like:
   - `/health` returns `{"ok":true,...}`
   - `/stats` returns non-error JSON and queue counts
   - `/ai/health` returns `ok=true` with mode `remote` or `fallback` (fallback is safe local scoring)

If no trades occur:
1. Check dashboard `reason` per symbol (session, spread, news, risk lock, or confluence).
2. Check `/ai/health` and `--ai-test`.
3. Check news cache and session windows in `config/settings.yaml`.
4. Verify MT5 symbol mapping resolves to broker symbols.
