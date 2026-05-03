# Backtest Reliability Audit

## Scope
Audit date: 2026-03-05  
Repo: `./apex_bot`  
Focus: remove inflated/synthetic performance artifacts and enforce realistic execution assumptions.

## What Was Wrong

1. Partial-close accounting inflated win rate:
- Partial closes were pushed directly into `r_results` as standalone positive outcomes.
- This made `trade_count` and `win_rate` reflect partial events, not true closed trades.

2. Execution model was too optimistic:
- Entries/exits used a fixed cost shortcut and did not consistently model bid/ask behavior.
- Stop/TP checks did not account for side-specific bid/ask trigger logic.
- No latency or partial-fill constraints in trade execution simulation.

3. Feature leakage risk:
- Swing/fractal flags used centered rolling windows (`center=True`) in feature engineering.
- With precomputed full frames, centered windows can inject future-bar information.

4. No plausibility guard:
- Backtests could report unrealistic win rates over large sample sizes without any hard failure.

## Fixes Implemented

### 1) Accounting and PnL fixes
- `src/backtest.py`
  - Added per-position `realized_r` and removed partial-close entries from standalone trade outcomes.
  - `trade_count` now maps to closed positions/trades (not partial fills).
  - Final trade `r_multiple` now includes partial realizations plus final close.

### 2) Realistic execution model
- `src/backtest.py`
  - Added/used:
    - spread points
    - slippage points
    - commission per lot
    - latency (ms -> bar delay)
    - deterministic partial-fill ratio with minimum fill threshold
  - Entry:
    - BUY at ask + slippage
    - SELL at bid - slippage
  - Exit:
    - BUY closes on bid (with adverse slippage)
    - SELL closes on ask (with adverse slippage)
  - Stop/TP trigger checks now use bid/ask-adjusted candle extremes.

### 3) Leakage mitigation
- `src/feature_engineering.py`
  - Replaced centered swing windows with trailing windows only.

### 4) Plausibility enforcement
- `src/backtest.py`
  - Added `enforce_plausibility(...)`.
  - If `win_rate > 0.85` over `>200` trades, backtest fails unless explicitly whitelisted.

### 5) Preset-driven backtest CLI
- `src/main.py`
  - Added `--preset realistic|frictionless`.
  - `realistic`: costs + latency + plausibility checks on.
  - `frictionless`: comparison-only (costs off, plausibility check off).

### 6) New guardrail tests
- `tests/test_backtest_audit.py`
  - A) No-lookahead shift test
  - B) Forced-loss PnL sanity test
  - C) Costs-matter test
  - D) Win-rate plausibility guard test

## Before/After (Same Local Data Context)

### Before audit (previous implementation)
Observed output (`--backtest`, pre-fix):
- `trade_count`: 280
- `win_rate`: 1.00
- `profit_factor`: Infinity
- `expectancy_r`: `+0.6906`
- `max_drawdown_r`: `0.0`

### After audit (`--preset realistic`)
Observed output:
- `trade_count`: 120
- `win_rate`: 0.00
- `profit_factor`: 0.0
- `expectancy_r`: `-2.0364`
- `net_r`: `-244.36`

### Comparison (`--preset frictionless`)
Observed output:
- `trade_count`: 120
- `win_rate`: 0.00
- `profit_factor`: 0.0
- `expectancy_r`: `-1.0000`
- `net_r`: `-120.00`

Interpretation:
- The prior 100% profile was not credible.
- With realistic execution assumptions and corrected accounting, results degrade materially as expected.
- Costs/friction now worsen results versus frictionless baseline.

## Audit Conclusion
- Backtest is now fail-closed for suspiciously high win-rate runs (unless explicitly whitelisted).
- Accounting and execution modeling are materially more realistic.
- Leakage risk from centered rolling swing features has been removed.

## Run Commands

```bash
PYTHONPATH=. python -m unittest
PYTHONPATH=. python -m src.main --backtest --preset realistic
PYTHONPATH=. python -m src.main --backtest --preset frictionless
```
