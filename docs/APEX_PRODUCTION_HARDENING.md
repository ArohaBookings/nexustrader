# APEX Production Hardening

## What changed
- Rich regime state is now exposed alongside the legacy regime label.
- Every candidate gets a unified trade quality score and band.
- Risk now uses phase base risk plus bounded adaptive modifiers.
- Daily risk is now resolved through a staged governor:
  - `DAILY_NORMAL`
  - `DAILY_CAUTION`
  - `DAILY_DEFENSIVE`
  - `DAILY_HARD_STOP`
- Elite overflow risk is allowed only for top-tier setups.
- Portfolio correlation is cluster-aware across FX, risk-on, and crypto groupings.
- Queue rows persist richer context for bridge delivery, debug, and post-entry management.
- `/health` now exposes bridge, broker, kill, queue, AI, and price freshness state.
- A read-only operator dashboard is now served from the same FastAPI app with cookie auth.

## New config
The primary new knobs are in [/Users/leobons/Library/Mobile Documents/com~apple~CloudDocs/Kimi_Agent_Set & Forget Trade Bot/apex_bot/config/settings.yaml](/Users/leobons/Library/Mobile%20Documents/com~apple~CloudDocs/Kimi_Agent_Set%20%26%20Forget%20Trade%20Bot/apex_bot/config/settings.yaml):

```yaml
risk:
  daily_caution_threshold_pct: 0.02
  daily_defensive_threshold_pct: 0.035
  daily_hard_stop_threshold_pct: 0.05
  daily_normal_quality_floor: 0.58
  daily_caution_quality_floor: 0.70
  daily_defensive_quality_floor: 0.85
  daily_normal_risk_multiplier: 1.0
  daily_caution_risk_multiplier: 0.75
  daily_defensive_risk_multiplier: 0.45
  daily_hard_stop_risk_multiplier: 0.0

risk_scaling:
  trade_quality_floor: 0.58
  trade_quality_exception_floor: 0.60
  trade_quality_exception_probability_floor: 0.60
  trade_quality_exception_ev_floor: 0.20
  trade_quality_exception_confluence_floor: 3.0

dashboard:
  enabled: true
  host: "127.0.0.1"
  port: 8000
  public_enabled: true
  password: ""
  session_secret: ""
  session_timeout_minutes: 240
  read_only: true
  allowed_ips: []
  mobile_refresh_seconds: 8
  desktop_refresh_seconds: 5
```

## Phase scaling
- `PHASE_1`: base `3%`, overflow `5%`, `4` trades/day.
- `PHASE_2`: base `4%`, overflow `6%`, `6` trades/day.
- `PHASE_3`: base `5%`, overflow `7%`, `8` trades/day.

Overflow risk is disabled unless the trade is elite and execution/news/correlation conditions are clean.

## Health
`/health` now includes:
- `bridge_status`
- `broker_connectivity`
- `news_engine_status`
- `current_kill_state`
- `current_daily_state`
- `current_daily_state_reason`
- `queue_depth`
- `open_risk_pct`
- `last_successful_ai_call`
- `last_price_update_age_seconds`
- `xau_grid_override_state`
- `execution_quality_state`
- `watchdog_state`
- `stale_poll_warning`

## Dashboard
Routes:
- `/dashboard`
- `/dashboard/login`
- `/dashboard/logout`
- `/dashboard/data`

Behavior:
- read-only by default
- password is loaded from `dashboard.password` or `APEX_DASHBOARD_PASSWORD`
- session secret is loaded from `dashboard.session_secret` or `APEX_DASHBOARD_SESSION_SECRET`
- the app should stay bound to `127.0.0.1:8000`
- expose the dashboard externally only through reverse proxy
- do not expose `/v1/*`, `/stats`, or `/debug/*` publicly

Change the dashboard password later:
- set `APEX_DASHBOARD_PASSWORD` in the environment, or
- set `dashboard.password` in [/Users/leobons/Library/Mobile Documents/com~apple~CloudDocs/Kimi_Agent_Set & Forget Trade Bot/apex_bot/config/settings.yaml](/Users/leobons/Library/Mobile%20Documents/com~apple~CloudDocs/Kimi_Agent_Set%20%26%20Forget%20Trade%20Bot/apex_bot/config/settings.yaml)

Disable public dashboard access later:
- leave the app bound to `127.0.0.1`
- stop proxying `/dashboard*`
- or set `dashboard.public_enabled: false`

Sample Nginx reverse proxy:
```nginx
server {
    listen 80;
    server_name 31.44.5.163;

    location /dashboard/ {
        proxy_pass http://127.0.0.1:8000/dashboard/;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location = /dashboard {
        proxy_pass http://127.0.0.1:8000/dashboard;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /dashboard/data {
        proxy_pass http://127.0.0.1:8000/dashboard/data;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /dashboard/login {
        proxy_pass http://127.0.0.1:8000/dashboard/login;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /dashboard/logout {
        proxy_pass http://127.0.0.1:8000/dashboard/logout;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    location /health {
        proxy_pass http://127.0.0.1:8000/health;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Recommended reverse-proxy scope:
- `/dashboard`
- `/dashboard/login`
- `/dashboard/logout`
- `/dashboard/data`
- optional `/health`

## Local run
```bash
cd "/Users/leobons/Library/Mobile Documents/com~apple~CloudDocs/Kimi_Agent_Set & Forget Trade Bot/apex_bot"
PYTHONPATH=. ./.venv/bin/python -m src.main --bridge-serve
```

## VPS later
Keep using the existing production launcher:
```bash
PYTHONPATH=. ./.venv/bin/python scripts/start_bridge_prod.py
```

## Logs to watch
- `regime_transition`
- `risk_gate_decision`
- `daily_state`
- `TRADE_QUEUED_FOR_EA`
- `manage_trade_decision`
- `SIGNAL_STATE_PROTECTED`
- `proof_exception_allowed`
- `XAU_GRID_OVERRIDE_ALLOWED`
- `bridge_execution_report`
- `bridge_close_report`

## Migration notes
- Bridge queue DB will auto-add the new `context_json` column.
- Existing rows remain valid.
- No EA contract or endpoint shape change is required.
