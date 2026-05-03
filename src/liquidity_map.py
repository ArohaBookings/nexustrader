from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

from src.utils import clamp


@dataclass(frozen=True)
class LiquidityDecision:
    allowed: bool
    reason: str
    score: float
    target_side: str
    nearest_above_points: float
    nearest_below_points: float
    nearest_liquidity_above: float = 0.0
    nearest_liquidity_below: float = 0.0
    liquidity_sweep_detected: bool = False
    session_high: float = 0.0
    session_low: float = 0.0
    prev_day_high: float = 0.0
    prev_day_low: float = 0.0
    equal_highs: float = 0.0
    equal_lows: float = 0.0


def _round_step(symbol_key: str, entry_price: float) -> float:
    key = str(symbol_key or "").upper()
    if key == "XAUUSD":
        return 5.0
    if key == "NAS100":
        return 50.0
    if key == "USOIL":
        return 1.0
    if key == "BTCUSD":
        return 100.0
    if key == "USDJPY":
        return 0.50
    # EURUSD/GBPUSD and other FX majors.
    return 0.005 if entry_price < 10 else 0.50


def _coerce_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def _collect_levels(entry_price: float, round_step: float, context: dict[str, Any] | None) -> tuple[list[float], list[float]]:
    above: list[float] = []
    below: list[float] = []
    if context:
        for key in (
            "asian_high",
            "prev_day_high",
            "weekly_open_high",
            "vwap_upper",
            "session_high",
            "rolling_high",
            "fvg_high",
        ):
            level = _coerce_float(context.get(key))
            if level is None:
                continue
            if level >= entry_price:
                above.append(level)
            else:
                below.append(level)
        for key in (
            "asian_low",
            "prev_day_low",
            "weekly_open_low",
            "vwap_lower",
            "session_low",
            "rolling_low",
            "fvg_low",
        ):
            level = _coerce_float(context.get(key))
            if level is None:
                continue
            if level >= entry_price:
                above.append(level)
            else:
                below.append(level)

    rounded = round(entry_price / max(round_step, 1e-9)) * round_step
    for offset in range(-4, 5):
        if offset == 0:
            continue
        level = rounded + (offset * round_step)
        if level > entry_price:
            above.append(level)
        else:
            below.append(level)
    return above, below


def _liquidity_snapshot(context: dict[str, Any] | None) -> dict[str, float]:
    payload = dict(context or {})
    return {
        "session_high": _coerce_float(payload.get("session_high")) or 0.0,
        "session_low": _coerce_float(payload.get("session_low")) or 0.0,
        "prev_day_high": _coerce_float(payload.get("prev_day_high")) or 0.0,
        "prev_day_low": _coerce_float(payload.get("prev_day_low")) or 0.0,
        "equal_highs": _coerce_float(payload.get("equal_highs")) or _coerce_float(payload.get("rolling_high")) or 0.0,
        "equal_lows": _coerce_float(payload.get("equal_lows")) or _coerce_float(payload.get("rolling_low")) or 0.0,
    }


def evaluate_liquidity_map(
    *,
    symbol_key: str,
    side: str,
    entry_price: float,
    point_size: float,
    context: dict[str, Any] | None = None,
    now_utc: datetime | None = None,
) -> LiquidityDecision:
    del now_utc  # reserved for time-aware expansion (session-specific maps)
    if entry_price <= 0 or point_size <= 0:
        return LiquidityDecision(True, "liquidity_unknown", 0.5, "UNKNOWN", 0.0, 0.0)

    step = _round_step(symbol_key, entry_price)
    above, below = _collect_levels(entry_price, step, context)
    snapshot = _liquidity_snapshot(context)
    nearest_above = min((level - entry_price for level in above), default=step)
    nearest_below = min((entry_price - level for level in below), default=step)
    nearest_above_points = nearest_above / point_size
    nearest_below_points = nearest_below / point_size
    nearest_above_level = entry_price + nearest_above
    nearest_below_level = entry_price - nearest_below
    liquidity_sweep_detected = bool(
        int(float((context or {}).get("m5_liquidity_sweep", (context or {}).get("m5_sweep", 0)) or 0.0)) == 1
        or int(float((context or {}).get("sweep_flag", 0) or 0.0)) == 1
    )

    side_text = str(side or "").upper()
    min_buffer_points = 20.0
    score = 0.5
    target_side = "UNKNOWN"

    if nearest_above_points <= nearest_below_points:
        target_side = "BUY_SIDE_LIQUIDITY"
    else:
        target_side = "SELL_SIDE_LIQUIDITY"

    if side_text == "BUY":
        score += clamp((nearest_above_points - nearest_below_points) / 200.0, -0.2, 0.2)
        if nearest_above_points < min_buffer_points:
            return LiquidityDecision(
                False,
                "liquidity_wall_above",
                clamp(score, 0.0, 1.0),
                target_side,
                nearest_above_points,
                nearest_below_points,
                nearest_liquidity_above=nearest_above_level,
                nearest_liquidity_below=nearest_below_level,
                liquidity_sweep_detected=liquidity_sweep_detected,
                **snapshot,
            )
        if target_side == "BUY_SIDE_LIQUIDITY":
            score += 0.05
    elif side_text == "SELL":
        score += clamp((nearest_below_points - nearest_above_points) / 200.0, -0.2, 0.2)
        if nearest_below_points < min_buffer_points:
            return LiquidityDecision(
                False,
                "liquidity_wall_below",
                clamp(score, 0.0, 1.0),
                target_side,
                nearest_above_points,
                nearest_below_points,
                nearest_liquidity_above=nearest_above_level,
                nearest_liquidity_below=nearest_below_level,
                liquidity_sweep_detected=liquidity_sweep_detected,
                **snapshot,
            )
        if target_side == "SELL_SIDE_LIQUIDITY":
            score += 0.05
    else:
        return LiquidityDecision(
            False,
            "liquidity_invalid_side",
            0.0,
            target_side,
            nearest_above_points,
            nearest_below_points,
            nearest_liquidity_above=nearest_above_level,
            nearest_liquidity_below=nearest_below_level,
            liquidity_sweep_detected=liquidity_sweep_detected,
            **snapshot,
        )

    return LiquidityDecision(
        True,
        "liquidity_aligned",
        clamp(score, 0.0, 1.0),
        target_side,
        nearest_above_points,
        nearest_below_points,
        nearest_liquidity_above=nearest_above_level,
        nearest_liquidity_below=nearest_below_level,
        liquidity_sweep_detected=liquidity_sweep_detected,
        **snapshot,
    )
