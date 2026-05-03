from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable


UTC = timezone.utc


def _metric_series(rows: list[dict[str, Any]]) -> list[float]:
    pnl_r = [float(row.get("pnl_r", 0.0) or 0.0) for row in rows]
    if any(abs(value) > 1e-9 for value in pnl_r):
        return pnl_r
    pnl_amount = [float(row.get("pnl_amount", 0.0) or 0.0) for row in rows]
    if any(abs(value) > 1e-9 for value in pnl_amount):
        return pnl_amount
    return pnl_r


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "trades": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_r": 0.0,
            "max_drawdown_r": 0.0,
            "expectancy_r": 0.0,
        }
    pnl = _metric_series(rows)
    wins = [value for value in pnl if value > 0]
    losses = [abs(value) for value in pnl if value < 0]
    win_rate = sum(1 for value in pnl if value >= 0) / len(pnl)
    avg_r = sum(pnl) / len(pnl)
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    equity_curve = []
    running = 0.0
    for value in pnl:
        running += value
        equity_curve.append(running)
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        max_dd = max(max_dd, peak - value)
    return {
        "trades": float(len(rows)),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_r": float(avg_r),
        "max_drawdown_r": float(max_dd),
        "expectancy_r": float(avg_r),
    }


def build_performance_report(
    rows: list[dict[str, Any]],
    session_name_resolver: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    resolver = session_name_resolver or default_session_name
    by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_setup: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_session: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in rows:
        symbol = str(row.get("symbol", "UNKNOWN")).upper()
        setup = str(row.get("setup", "UNKNOWN")).upper()
        closed_at = str(row.get("closed_at", ""))
        session = resolver(closed_at)
        by_symbol[symbol].append(row)
        by_setup[setup].append(row)
        by_session[session].append(row)

    return {
        "overall": _metrics(rows),
        "by_symbol": {key: _metrics(values) for key, values in by_symbol.items()},
        "by_setup": {key: _metrics(values) for key, values in by_setup.items()},
        "by_session": {key: _metrics(values) for key, values in by_session.items()},
    }


def default_session_name(timestamp_iso: str) -> str:
    try:
        current = datetime.fromisoformat(str(timestamp_iso).replace("Z", "+00:00"))
    except ValueError:
        return "UNKNOWN"
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    hour = current.astimezone(UTC).hour
    if 0 <= hour < 7:
        return "TOKYO"
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 17:
        return "OVERLAP"
    if 17 <= hour < 21:
        return "NEW_YORK"
    return "SYDNEY"
