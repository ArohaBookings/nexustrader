# APEX Bridge Runbook

## Start bridge + strategy engine

```bash
cd /Users/leobons/Library/Mobile\ Documents/com~apple~CloudDocs/Kimi_Agent_Set\ \&\ Forget\ Trade\ Bot/apex_bot\ V2\ BOT
PYTHONPATH=. ./.venv/bin/python -m src.main --bridge-serve
```

## Verify endpoints

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/stats
curl http://127.0.0.1:8000/ai/health
curl http://127.0.0.1:8000/omega/status
curl http://127.0.0.1:8000/telegram/status
curl "http://127.0.0.1:8000/debug/symbol?symbol=XAUUSD&limit=20"
```

Check `/stats` for:

- `action_status_counts_by_symbol`
- `symbol_state` (grid cycle/leg/risk state)
- `risk_state`
- `rule_change.summary`
- `symbol_runtime` (last candidate/action summary, starvation/adaptive state)
- `ai_runtime` (`last_ai_ok`, `last_ai_error`, `last_ai_latency_ms`)
- `omega_runtime` + `/omega/status` (regime state, Kelly scaled fraction, next expected trade time)
- `strategy_optimizer` (per-strategy rolling metrics + adaptive soft-knob multipliers)
- `symbol_training_mode` (unknown-symbol discovery/training/activation state)

## Telegram owner bot

Telegram secrets stay outside source files. Put real values in `config/secrets.env`:

```bash
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
TELEGRAM_WEBHOOK_SECRET=...
OPENAI_API_KEY=...
NEWS_API_KEY=...
```

Send `/start` to `@Nexus_vantage_trader_bot`, then verify:

```bash
PYTHONPATH=. python3 scripts/apex_telegram_check.py --get-me
PYTHONPATH=. python3 scripts/apex_telegram_check.py --discover-chat
PYTHONPATH=. python3 scripts/apex_telegram_check.py --send-test
```

The bot answers `/status`, `/funded`, `/risk`, `/trades`, `/apex`, and `/intel`. Owner controls are limited to `/pause`, `/refresh`, `/resume`, and `/kill`; resume and kill require `/confirm <id>`. It refuses direct trade placement, risk bypasses, and live parameter changes.

## Live sign-off gate

Run this on the actual MT5 VPS/VM before calling the system live-ready:

```bash
PYTHONPATH=. python3 scripts/live_signoff.py
```

The sign-off gate checks:

- MT5 Python initialize/account/symbol resolution.
- Bridge `/health` and live MT5 `terminal_trade_allowed` feed.
- Telegram token, owner chat, webhook secret, and optional test send.
- GitHub CLI/auth + repo remote for `ArohaBookings/nexustrader`.
- Vercel CLI/project readiness for dashboard/webhook deployment.
- Required env vars and public dashboard guardrails.

This workspace is wired as the active git root for `https://github.com/ArohaBookings/nexustrader.git`. The old nested `nexus-trader/` app is ignored so it cannot be pushed by accident. The Vercel CLI is installed in the user-local npm prefix on this Mac; add it to your shell when needed:

```bash
export PATH="$HOME/.npm-global/bin:$PATH"
```

Authenticate deployment tooling:

```bash
gh auth login
vercel login
vercel link --yes --project nexus-trader
```

Useful variants:

```bash
PYTHONPATH=. python3 scripts/live_signoff.py --send-telegram-test
PYTHONPATH=. python3 scripts/live_signoff.py --no-deploy
```

Researched platform constraints:

- MT5 Python `initialize()` connects to a MetaTrader terminal executable such as `metatrader64.exe`; this is why the live execution worker belongs on the Windows MT5 VPS/VM, not Vercel serverless.
- Telegram webhook requests should be protected with `secret_token`, which Telegram sends as `X-Telegram-Bot-Api-Secret-Token`.
- Vercel can deploy dashboard/webhook surfaces after CLI auth/project linking, but it must not replace the long-running MT5 bridge process.

Production bridge startup is fail-closed in live mode. If MT5 verification does not turn green, `scripts/start_bridge_prod.py` exits instead of starting a live bridge. Use `APEX_START_BRIDGE_WITHOUT_MT5=1` only for explicit dashboard observation or recovery mode, not for live execution sign-off.

## Configure broker symbol/stop rules

Edit:

- `config/symbol_rules.yaml`

Per symbol keys:

- `digits`
- `tick_size`
- `point`
- `min_stop_points`
- `freeze_points`
- `typical_spread_points`
- `max_slippage_points`
- `tick_value`
- `contract_size`

## Reset bridge state (queue DB only)

```bash
rm -f data/bridge_actions.sqlite
```

Then restart `--bridge-serve` to recreate schema.

## Simulate EA pull with curl

```bash
curl "http://127.0.0.1:8000/v1/pull?symbol=XAUUSD&tf=M5&account=123456&magic=20260304&balance=50.0&equity=50.0&free_margin=48.0&spread_points=30"
```

Expected response shape:

```json
{"actions":[{...}]}
```

The server returns at most one action per call. If no eligible action passes stop validation + risk/confluence gates, it returns:

```json
{"actions":[]}
```

By default, delivered actions are not re-sent (`allow_redelivery=false`) and are protected by a short lease window.

## Optional MT5 snapshot ingestion (no EA contract break)

```bash
curl -X POST http://127.0.0.1:8000/v1/mt5_snapshot \
  -H "Content-Type: application/json" \
  -d '{"symbol":"XAUUSD","account":"123456","magic":20260304,"timeframe":"M5","bid":2200.10,"ask":2200.35,"spread_points":25,"open_count":1,"net_lots":0.01,"avg_entry":2199.80,"floating_pnl":0.42,"equity":51.10,"free_margin":48.90}'
```

This endpoint is optional. If no snapshots are posted, bridge still operates from pull/query params + execution reports.

Key bridge knobs live in `config/settings.yaml` under `bridge.orchestrator`:

- `max_actions_per_pull`
- `lease_seconds`
- `delivery_cooldown_seconds`
- `xau_grid_step_points`, `xau_grid_max_legs`, `xau_grid_cycle_risk_pct`
- `news_mode` (`SAFE`, `ALLOW_HIGH_CONF`, `OFF`)
- `rule_change.*`
- `strategy_optimizer.*`
- `symbol_auto_discovery.*`

Additional instrument strategy knobs:

- `nas_strategy.*` (session scalper, NY cash-open + power-hour focus)
- `oil_strategy.*` (inventory/news-aware NY scalper)
- `bridge.orchestrator.undertrading_governor.*` (bounded soft-parameter adjustments)

Market-hours rationale:

- NAS/index products and WTI are near-24h with exchange maintenance windows, so the bridge uses session-aware spread/risk gating rather than single-session hard blocks.
- OIL `NEWS_ARMED` mode is designed around the weekly EIA inventory release window (Wednesday 10:30 ET).

## Smoke test (no broker required)

```bash
PYTHONPATH=. ./.venv/bin/python scripts/smoke_pull_action.py
```

Expected output:

- `SMOKE PASS`
- One returned action with valid normalized `sl` and `tp`.

## Optional explicit ack (future-safe)

```bash
curl -X POST http://127.0.0.1:8000/v1/ack \
  -H "Content-Type: application/json" \
  -d '{"signal_id":"<signal_id>","status":"ACKED"}'
```

## Invalid-stops fallback behavior

If `bridge.orchestrator.allow_open_without_stops_fallback` is enabled and stop normalization cannot produce broker-safe values on the initial open, the bridge can:

1. deliver `OPEN_MARKET` with `sl=0,tp=0`
2. after execution report, queue `MODIFY_SLTP` with normalized SL/TP

## Why no trade? (quick checks)

Check `/stats`:

- `symbol_state.<account:magic:symbol:tf>.last_block_reason`
- `symbol_state.*.state_confidence` and `drift_flag`
- `undertrading_governor` / `undertrading_by_symbol`

Common block reasons:

- `low_confidence_reality_sync`
- `spread_too_wide_session`
- `confluence_below_threshold`
- `cooldown_active`
- `cycle_risk_exhausted`
- `symbol_training_mode` (unknown symbol still in data collection phase)
