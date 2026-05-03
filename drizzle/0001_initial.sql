CREATE TABLE IF NOT EXISTS bot_snapshots (
  id serial PRIMARY KEY,
  observed_at timestamptz NOT NULL DEFAULT now(),
  source text NOT NULL DEFAULT 'bridge',
  equity numeric(18,6),
  balance numeric(18,6),
  pnl_today numeric(18,6),
  drawdown_pct numeric(12,8),
  queue_depth integer,
  session text,
  kill_state text,
  open_risk_pct numeric(12,8),
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS bot_snapshots_observed_idx ON bot_snapshots (observed_at);

CREATE TABLE IF NOT EXISTS symbol_snapshots (
  id serial PRIMARY KEY,
  observed_at timestamptz NOT NULL DEFAULT now(),
  symbol text NOT NULL,
  strategy text,
  state text,
  confidence numeric(12,8),
  spread numeric(18,8),
  open_risk_pct numeric(12,8),
  blocker text,
  thinking text,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS symbol_snapshots_symbol_observed_idx ON symbol_snapshots (symbol, observed_at);

CREATE TABLE IF NOT EXISTS trade_events (
  id serial PRIMARY KEY,
  external_id text,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  symbol text,
  type text NOT NULL,
  status text,
  side text,
  quantity numeric(18,8),
  price numeric(18,8),
  fee numeric(18,8),
  pnl numeric(18,8),
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS order_events (
  id serial PRIMARY KEY,
  external_id text,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  symbol text,
  type text NOT NULL,
  status text,
  side text,
  quantity numeric(18,8),
  price numeric(18,8),
  fee numeric(18,8),
  slippage_bps numeric(12,6),
  reason text,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS risk_events (
  id serial PRIMARY KEY,
  occurred_at timestamptz NOT NULL DEFAULT now(),
  symbol text,
  type text NOT NULL,
  status text,
  reason text,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS telegram_messages (
  id serial PRIMARY KEY,
  chat_id text NOT NULL,
  telegram_user_id text,
  username text,
  direction text NOT NULL,
  text text NOT NULL,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS command_requests (
  id serial PRIMARY KEY,
  command_id text NOT NULL UNIQUE,
  chat_id text NOT NULL,
  telegram_user_id text,
  action text NOT NULL,
  status text NOT NULL DEFAULT 'pending_confirmation',
  requested_text text NOT NULL,
  model_response text,
  confirmation_required boolean NOT NULL DEFAULT true,
  executed_at timestamptz,
  execution_result jsonb NOT NULL DEFAULT '{}'::jsonb,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS rollups (
  id serial PRIMARY KEY,
  bucket text NOT NULL,
  bucket_start timestamptz NOT NULL,
  metric text NOT NULL,
  value numeric(18,8),
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS janitor_runs (
  id serial PRIMARY KEY,
  started_at timestamptz NOT NULL DEFAULT now(),
  finished_at timestamptz,
  raw_deleted integer NOT NULL DEFAULT 0,
  rollups_written integer NOT NULL DEFAULT 0,
  status text NOT NULL DEFAULT 'ok',
  payload jsonb NOT NULL DEFAULT '{}'::jsonb
);
