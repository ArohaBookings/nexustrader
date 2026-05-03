CREATE TABLE IF NOT EXISTS funded_configs (
  id serial PRIMARY KEY,
  name text NOT NULL DEFAULT 'default',
  enabled boolean NOT NULL DEFAULT false,
  "group" text NOT NULL DEFAULT 'custom',
  phase text NOT NULL DEFAULT 'evaluation',
  starting_balance numeric(18,6) NOT NULL DEFAULT 100,
  profit_target_pct numeric(12,8) NOT NULL DEFAULT 0.08,
  daily_drawdown_pct numeric(12,8) NOT NULL DEFAULT 0.05,
  max_drawdown_pct numeric(12,8) NOT NULL DEFAULT 0.10,
  trailing_drawdown boolean NOT NULL DEFAULT false,
  base_risk_pct numeric(12,8) NOT NULL DEFAULT 0.005,
  max_open_risk_pct numeric(12,8) NOT NULL DEFAULT 0.015,
  daily_reset_timezone text NOT NULL DEFAULT 'Australia/Sydney',
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  updated_at timestamptz NOT NULL DEFAULT now(),
  created_at timestamptz NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX IF NOT EXISTS funded_configs_name_unique ON funded_configs (name);
