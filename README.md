# Nexus Trader

Private Next.js dashboard and telemetry backend for the existing APEX/Hyperliquid research stack.

This project is intentionally isolated from the MT5/APEX live bot paths. It reads existing bridge telemetry, stores lean snapshots/events in Postgres, renders owner-only dashboards, and exposes Telegram AI status/ops controls with no direct trade placement.

## Local Setup

```bash
npm install
cp .env.example .env.local
npm run dev
```

Minimum local env for dashboard login:

```bash
APP_ADMIN_PASSWORD="change-this-owner-password"
SESSION_SECRET="change-this-very-long-random-session-secret"
INGEST_API_KEY="change-this-ingest-key"
JANITOR_SECRET="change-this-janitor-secret"
TELEGRAM_WEBHOOK_SECRET="change-this-telegram-webhook-secret"
```

Without `DATABASE_URL`, the app uses an in-memory demo store so the UI and tests remain runnable. Production should use Neon pooled `DATABASE_URL`.

## Database

Apply the included SQL migration or run Drizzle after Neon is configured:

```bash
npm run db:migrate
```

Tables are lean by design: bot/symbol snapshots, trade/order/risk events, Telegram messages, audited command requests, rollups, and janitor runs.

## Collector

The collector polls the existing bridge without mutating it, then posts compact telemetry to `/api/ingest`.

```bash
BRIDGE_API_BASE_URL="http://127.0.0.1:8000" \
NEXUS_INGEST_URL="http://127.0.0.1:3000/api/ingest" \
INGEST_API_KEY="change-this-ingest-key" \
npm run collector
```

## Telegram Webhook

After production deploy, configure Telegram with a secret header:

```bash
curl -sS "https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/setWebhook" \
  -d "url=https://YOUR_VERCEL_URL/api/telegram/webhook" \
  -d "secret_token=$TELEGRAM_WEBHOOK_SECRET"
```

Supported commands:

```text
/status
/risk
/trades
/pause
/refresh
/resume
/kill
/confirm <command_id>
```

`/resume` and `/kill` require explicit confirmation. Telegram AI cannot place trades, increase aggression, bypass risk controls, or tune parameters.

## Verification

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

