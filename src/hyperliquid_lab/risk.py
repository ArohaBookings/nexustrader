from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class RiskConfig:
    atr_window: int = 14
    volatility_risk_fraction: float = 0.01
    max_position_pct: float = 0.20
    stop_loss_pct: float = 0.01
    max_daily_loss_pct: float = 0.03
    kelly_fraction_cap: float = 0.25
    circuit_breaker_volatility_multiple: float = 3.0
    stale_data_seconds: float = 60.0
    max_api_errors: int = 3


@dataclass
class CircuitBreakerState:
    api_errors: int = 0
    halted: bool = False
    reasons: list[str] = field(default_factory=list)


def compute_atr(data: pd.DataFrame, window: int = 14) -> pd.Series:
    required = {"high", "low", "close"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"data frame missing columns: {sorted(missing)}")
    frame = data.copy()
    high = frame["high"].astype(float)
    low = frame["low"].astype(float)
    close = frame["close"].astype(float)
    previous_close = close.shift(1)
    true_range = pd.concat([(high - low), (high - previous_close).abs(), (low - previous_close).abs()], axis=1).max(axis=1)
    return true_range.rolling(int(window), min_periods=1).mean()


def volatility_sized_quantity(equity: float, price: float, atr: float, config: RiskConfig) -> float:
    if float(equity) <= 0.0 or float(price) <= 0.0:
        return 0.0
    if float(atr) <= 0.0:
        return 0.0
    risk_dollars = float(equity) * float(config.volatility_risk_fraction)
    raw_qty = risk_dollars / float(atr)
    max_notional = float(equity) * float(config.max_position_pct)
    max_qty = max_notional / float(price)
    return float(max(0.0, min(raw_qty, max_qty)))


def capped_kelly_fraction(win_rate: float, avg_win_loss_ratio: float, config: RiskConfig) -> float:
    if avg_win_loss_ratio <= 0.0:
        return 0.0
    raw = float(win_rate) - ((1.0 - float(win_rate)) / float(avg_win_loss_ratio))
    return float(max(0.0, min(raw, float(config.kelly_fraction_cap))))


def stop_loss_triggered(entry_price: float, current_price: float, side: str, config: RiskConfig) -> bool:
    if float(entry_price) <= 0.0:
        return False
    threshold = float(config.stop_loss_pct)
    if str(side).lower() == "buy":
        return float(current_price) <= float(entry_price) * (1.0 - threshold)
    if str(side).lower() == "sell":
        return float(current_price) >= float(entry_price) * (1.0 + threshold)
    raise ValueError(f"Unsupported side: {side}")


def daily_loss_limit_hit(day_start_equity: float, current_equity: float, config: RiskConfig) -> bool:
    if float(day_start_equity) <= 0.0:
        return False
    loss = (float(day_start_equity) - float(current_equity)) / float(day_start_equity)
    return loss >= float(config.max_daily_loss_pct)


def evaluate_circuit_breaker(
    data: pd.DataFrame,
    *,
    last_data_time_utc: pd.Timestamp,
    now_utc: pd.Timestamp,
    api_errors: int,
    config: RiskConfig,
) -> CircuitBreakerState:
    reasons: list[str] = []
    if int(api_errors) >= int(config.max_api_errors):
        reasons.append("exchange_api_errors")
    age = (pd.Timestamp(now_utc).tz_convert("UTC") - pd.Timestamp(last_data_time_utc).tz_convert("UTC")).total_seconds()
    if age > float(config.stale_data_seconds):
        reasons.append("stale_data")
    atr = compute_atr(data, config.atr_window)
    if len(atr.dropna()) >= 2:
        recent = float(atr.iloc[-1])
        baseline = float(atr.iloc[:-1].median())
        if baseline > 0.0 and recent >= baseline * float(config.circuit_breaker_volatility_multiple):
            reasons.append("volatility_spike")
    return CircuitBreakerState(api_errors=int(api_errors), halted=bool(reasons), reasons=reasons)
