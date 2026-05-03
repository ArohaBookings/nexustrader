#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any
from statistics import pstdev

import pandas as pd

from src.backtest import Backtester
from src.execution import trading_day_key_for_timestamp
from src.grid_scalper import XAUGridScalper
from src.main import (
    _append_candidate_verification_log,
    _candidate_strategy_pool_rankings,
    _is_xau_grid_setup,
    _is_xau_higher_tf_candidate_setup,
    _strategy_exit_profile,
    _strategy_pool_winner_reason,
    _trade_actual_rr_achieved,
    build_runtime,
)
from src.session_calendar import dominant_session_name, is_weekend_market_mode, market_open_tuple
from src.strategy_engine import SignalCandidate, StrategyEngine
from src.trade_idea_lifecycle import TradeIdeaLifecycle
from src.utils import clamp, deterministic_id


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "trades": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "avg_r": 0.0,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "winner_loss_ratio": 0.0,
            "avg_mae_r": 0.0,
            "avg_mfe_r": 0.0,
            "capture_efficiency": 0.0,
            "avg_duration_minutes": 0.0,
            "top_exit_reasons": [],
        }
    pnl = [_safe_float(row.get("r_multiple"), 0.0) for row in rows]
    wins = [value for value in pnl if value > 0]
    losses = [abs(value) for value in pnl if value < 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    avg_win_r = gross_profit / max(len(wins), 1)
    avg_loss_r = gross_loss / max(len(losses), 1)
    winner_loss_ratio = avg_win_r / avg_loss_r if avg_loss_r > 0 else (avg_win_r if avg_win_r > 0 else 0.0)
    mae_values = [_safe_float(row.get("mae_r"), 0.0) for row in rows]
    mfe_values = [_safe_float(row.get("mfe_r"), 0.0) for row in rows]
    duration_values = [_safe_float(row.get("duration_minutes"), 0.0) for row in rows]
    capture_values = [
        min(1.5, max(0.0, _safe_float(row.get("r_multiple"), 0.0)) / max(_safe_float(row.get("mfe_r"), 0.0), 1e-6))
        for row in rows
        if _safe_float(row.get("mfe_r"), 0.0) > 0.0
    ]
    exit_reason_counts: Counter[str] = Counter(str(row.get("exit_reason") or "") for row in rows if str(row.get("exit_reason") or "").strip())
    return {
        "trades": float(len(rows)),
        "win_rate": float(sum(1 for value in pnl if value >= 0) / len(pnl)),
        "profit_factor": float(profit_factor),
        "expectancy_r": float(sum(pnl) / len(pnl)),
        "avg_r": float(sum(pnl) / len(pnl)),
        "avg_win_r": float(avg_win_r),
        "avg_loss_r": float(avg_loss_r),
        "winner_loss_ratio": float(winner_loss_ratio),
        "avg_mae_r": float(sum(mae_values) / len(mae_values)),
        "avg_mfe_r": float(sum(mfe_values) / len(mfe_values)),
        "capture_efficiency": float(sum(capture_values) / len(capture_values)) if capture_values else 0.0,
        "avg_duration_minutes": float(sum(duration_values) / len(duration_values)),
        "top_exit_reasons": [
            {"reason": reason, "count": count}
            for reason, count in exit_reason_counts.most_common(5)
        ],
    }


def _group_report(rows: list[dict[str, Any]], keys: tuple[str, ...], *, min_trades: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "") or "") for key in keys)].append(row)
    report: list[dict[str, Any]] = []
    for group_key, group_rows in grouped.items():
        if len(group_rows) < min_trades:
            continue
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(_metrics(group_rows))
        report.append(entry)
    report.sort(
        key=lambda item: (
            float(item.get("expectancy_r", 0.0) or 0.0),
            float(item.get("profit_factor", 0.0) or 0.0),
            -float(item.get("trades", 0.0) or 0.0),
        )
    )
    return report


def _normalize_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def _maybe_timestamp(value: Any) -> pd.Timestamp | None:
    if value in {None, ""}:
        return None
    try:
        return _normalize_timestamp(value)
    except Exception:
        return None


def _mean(values: list[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return 0.0
    return float(sum(finite_values) / len(finite_values))


def _stddev(values: list[float]) -> float:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if len(finite_values) <= 1:
        return 0.0
    try:
        return float(pstdev(finite_values))
    except Exception:
        return 0.0


def _trade_timestamp(row: dict[str, Any]) -> pd.Timestamp | None:
    for key in ("opened_at", "timestamp", "closed_at", "created_at"):
        parsed = _maybe_timestamp(row.get(key))
        if parsed is not None:
            return parsed
    return None


def _daily_projection(rows: list[dict[str, Any]]) -> dict[str, Any]:
    daily: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        trade_ts = _trade_timestamp(row)
        if trade_ts is None:
            continue
        day_key = str(trading_day_key_for_timestamp(trade_ts.to_pydatetime()))
        daily[day_key].append(row)
    if not daily:
        return {
            "days": 0,
            "avg_trades_per_day": 0.0,
            "median_trades_per_day": 0.0,
            "max_trades_per_day": 0.0,
            "avg_daily_win_rate": 0.0,
        }
    counts = [float(len(items)) for items in daily.values()]
    win_rates = []
    for items in daily.values():
        metrics = _metrics(items)
        win_rates.append(float(metrics.get("win_rate", 0.0) or 0.0))
    counts_sorted = sorted(counts)
    mid = len(counts_sorted) // 2
    if len(counts_sorted) % 2 == 0:
        median_count = (counts_sorted[mid - 1] + counts_sorted[mid]) / 2.0
    else:
        median_count = counts_sorted[mid]
    return {
        "days": int(len(daily)),
        "avg_trades_per_day": float(_mean(counts)),
        "median_trades_per_day": float(median_count),
        "max_trades_per_day": float(max(counts)),
        "avg_daily_win_rate": float(_mean(win_rates)),
    }


def _burst_density_report(rows: list[dict[str, Any]], *, window_minutes: int = 10) -> dict[str, Any]:
    windows: dict[tuple[str, str], int] = defaultdict(int)
    per_session: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        trade_ts = _trade_timestamp(row)
        if trade_ts is None:
            continue
        session_name = str(row.get("session_name") or "UNKNOWN").upper()
        window_key = trade_ts.floor(f"{max(1, int(window_minutes))}min").isoformat()
        windows[(session_name, window_key)] += 1
        per_session[session_name][window_key] += 1
    if not windows:
        return {
            "window_minutes": int(window_minutes),
            "active_windows": 0,
            "avg_actions_per_active_window": 0.0,
            "max_actions_per_window": 0.0,
            "by_session": [],
        }
    counts = [float(value) for value in windows.values()]
    by_session = []
    for session_name, session_windows in sorted(per_session.items()):
        session_counts = [float(value) for value in session_windows.values()]
        by_session.append(
            {
                "session_name": str(session_name),
                "active_windows": int(len(session_counts)),
                "avg_actions_per_active_window": float(_mean(session_counts)),
                "max_actions_per_window": float(max(session_counts) if session_counts else 0.0),
                "trades": int(sum(int(value) for value in session_windows.values())),
            }
        )
    return {
        "window_minutes": int(window_minutes),
        "active_windows": int(len(counts)),
        "avg_actions_per_active_window": float(_mean(counts)),
        "max_actions_per_window": float(max(counts) if counts else 0.0),
        "by_session": by_session,
    }


def _slice_report(
    rows: list[dict[str, Any]],
    *,
    symbol_key: str = "",
    strategy_key: str = "",
    weekend_only: bool = False,
) -> dict[str, Any]:
    filtered: list[dict[str, Any]] = []
    symbol_norm = str(symbol_key or "").upper()
    strategy_norm = str(strategy_key or "").upper()
    for row in rows:
        row_symbol = str(row.get("symbol") or "").upper()
        row_strategy = str(row.get("strategy_key") or "").upper()
        if symbol_norm and row_symbol != symbol_norm:
            continue
        if strategy_norm and row_strategy != strategy_norm:
            continue
        if weekend_only:
            trade_ts = _trade_timestamp(row)
            if trade_ts is None or not bool(is_weekend_market_mode(trade_ts.to_pydatetime())):
                continue
        filtered.append(row)
    report = _metrics(filtered)
    report["daily_projection"] = _daily_projection(filtered)
    report["by_session"] = _group_report(filtered, ("session_name",), min_trades=1)
    report["burst_density"] = _burst_density_report(filtered)
    return report


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        json.dumps(value)
        return value
    except Exception:
        return str(value)


def _snapshot_payload(
    runtime: dict[str, Any],
    *,
    closed_limit: int = 250,
    window_id: str = "",
    window_start: Any = None,
    window_end: Any = None,
) -> dict[str, Any]:
    journal = runtime["journal"]
    baseline_stats = journal.stats(current_equity=100.0)
    closed_trades = list(journal.closed_trades(closed_limit))
    start_ts = _maybe_timestamp(window_start)
    if start_ts is not None:
        historical = [
            trade
            for trade in closed_trades
            if (_maybe_timestamp(trade.get("closed_at") or trade.get("opened_at")) or start_ts) < start_ts
        ]
        if historical:
            closed_trades = historical[:closed_limit]
    return {
        "snapshot_timestamp": pd.Timestamp.utcnow().isoformat(),
        "trading_day_key": str(getattr(baseline_stats, "trading_day_key", "") or ""),
        "window_id": str(window_id or ""),
        "window_start": start_ts.isoformat() if start_ts is not None else "",
        "window_end": _maybe_timestamp(window_end).isoformat() if _maybe_timestamp(window_end) is not None else "",
        "baseline_stats": _json_safe(baseline_stats),
        "closed_trades": _json_safe(closed_trades),
        "xau_proof_state": _json_safe(runtime.get("xau_proof_state") or {}),
    }


def _snapshot_matches_window(snapshot: dict[str, Any], window_spec: dict[str, Any] | None) -> bool:
    window_payload = dict(window_spec or {})
    if not window_payload:
        return True
    expected_id = str(window_payload.get("window_id") or "")
    snapshot_id = str(snapshot.get("window_id") or "")
    if expected_id and snapshot_id and snapshot_id != expected_id:
        return False
    expected_start = _maybe_timestamp(window_payload.get("start"))
    expected_end = _maybe_timestamp(window_payload.get("end"))
    snapshot_start = _maybe_timestamp(snapshot.get("window_start"))
    snapshot_end = _maybe_timestamp(snapshot.get("window_end"))
    if expected_start is not None and snapshot_start != expected_start:
        return False
    if expected_end is not None and snapshot_end != expected_end:
        return False
    return True


def _resolve_snapshot(
    runtime: dict[str, Any],
    *,
    use_snapshot_path: str = "",
    refresh_snapshot: bool = False,
    window_spec: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], Path]:
    bridge_orchestrator_config = dict(runtime.get("bridge_orchestrator_config") or {})
    base_path = Path(
        use_snapshot_path
        or runtime["settings"].resolve_path_value(
            str(bridge_orchestrator_config.get("replay_snapshot_file", "data/replay_snapshot.json"))
        )
    )
    window_payload = dict(window_spec or {})
    if window_payload:
        window_id = str(window_payload.get("window_id") or "window")
        path = base_path.with_name(f"{base_path.stem}_{window_id}{base_path.suffix}")
    else:
        path = base_path
    existing_snapshot: dict[str, Any] | None = None
    if not refresh_snapshot and path.exists():
        try:
            existing_snapshot = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing_snapshot = None
    if use_snapshot_path and existing_snapshot is not None and not refresh_snapshot:
        return existing_snapshot, path
    if existing_snapshot is not None and _snapshot_matches_window(existing_snapshot, window_payload):
        return existing_snapshot, path
    if refresh_snapshot or not path.exists() or existing_snapshot is None or not _snapshot_matches_window(existing_snapshot, window_payload):
        snapshot = _snapshot_payload(
            runtime,
            window_id=str(window_payload.get("window_id") or ""),
            window_start=window_payload.get("start"),
            window_end=window_payload.get("end"),
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8")
        return snapshot, path
    return json.loads(path.read_text(encoding="utf-8")), path


def _apply_xau_profile(
    runtime: dict[str, Any],
    *,
    xau_profile: str = "",
    proof_mode: str = "",
) -> dict[str, Any]:
    settings = runtime.get("settings")
    raw_settings = getattr(settings, "raw", {}) if settings is not None else {}
    xau_config = dict(raw_settings.get("xau_grid_scalper") or {}) if isinstance(raw_settings, dict) else {}
    requested_profile = str(xau_profile or xau_config.get("active_profile") or "").strip()
    requested_proof_mode = str(proof_mode or xau_config.get("proof_mode") or requested_profile or "").strip()
    if xau_config:
        if requested_profile:
            xau_config["active_profile"] = requested_profile
        if requested_proof_mode:
            xau_config["proof_mode"] = requested_proof_mode
        grid_scalper = XAUGridScalper.from_config(
            xau_config,
            logger=runtime.get("logger"),
        )
        runtime["grid_scalper"] = grid_scalper
        runtime["xau_proof_state"] = dict(grid_scalper.profile_state() or {})
        return dict(runtime["xau_proof_state"])
    existing_scalper = runtime.get("grid_scalper")
    if existing_scalper is not None and hasattr(existing_scalper, "profile_state"):
        try:
            runtime["xau_proof_state"] = dict(existing_scalper.profile_state() or {})
        except Exception:
            runtime["xau_proof_state"] = {}
    else:
        runtime["xau_proof_state"] = {}
    return dict(runtime.get("xau_proof_state") or {})


def _load_snapshot_metadata(snapshot_path: str) -> dict[str, Any]:
    path = Path(str(snapshot_path or "").strip())
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _prime_session_window_specs(base_window: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    start = _maybe_timestamp((base_window or {}).get("window_start") or (base_window or {}).get("start"))
    end = _maybe_timestamp((base_window or {}).get("window_end") or (base_window or {}).get("end"))
    specs = [
        {
            "window_id": "london",
            "label": "xau_prime_london",
            "session_names": ["LONDON"],
        },
        {
            "window_id": "overlap",
            "label": "xau_prime_overlap",
            "session_names": ["OVERLAP"],
        },
        {
            "window_id": "new_york",
            "label": "xau_prime_new_york",
            "session_names": ["NEW_YORK"],
        },
    ]
    if start is not None or end is not None:
        for spec in specs:
            if start is not None:
                spec["start"] = start
            if end is not None:
                spec["end"] = end
    return specs


def _serialize_window_bound(value: Any) -> str:
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            return str(value)
    return str(value or "")


def _xau_prime_session_window_reports(
    base_report: dict[str, Any],
    *,
    base_snapshot: dict[str, Any],
    base_snapshot_path: Path,
    xau_proof_state: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    mapping = {
        "london": "LONDON",
        "overlap": "OVERLAP",
        "new_york": "NEW_YORK",
    }
    xau_slice = dict(base_report.get("xau_grid_pure_slice") or {})
    burst_density = dict(xau_slice.get("burst_density") or {})
    by_session = {
        str(item.get("session_name") or "").upper(): dict(item)
        for item in list(xau_slice.get("by_session") or [])
        if str(item.get("session_name") or "").strip()
    }
    density_by_session = {
        str(item.get("session_name") or "").upper(): dict(item)
        for item in list(burst_density.get("by_session") or [])
        if str(item.get("session_name") or "").strip()
    }
    window_minutes = int(burst_density.get("window_minutes") or 10)
    snapshot_suite: dict[str, Any] = {}
    window_reports: dict[str, Any] = {}
    window_activity_state: dict[str, Any] = {}
    snapshot_timestamp = str(base_snapshot.get("snapshot_timestamp") or "")
    trading_day_key = str(base_snapshot.get("trading_day_key") or "")
    window_start = str(base_snapshot.get("window_start") or "")
    window_end = str(base_snapshot.get("window_end") or "")
    source_window_id = str(base_snapshot.get("window_id") or "")
    resolved_proof_state = dict(xau_proof_state or base_snapshot.get("xau_proof_state") or {})

    for window_id, session_name in mapping.items():
        session_stats = dict(by_session.get(session_name) or {})
        session_density = dict(density_by_session.get(session_name) or {})
        if session_density:
            session_stats["burst_density"] = {
                "active_windows": int(float(session_density.get("active_windows", 0) or 0)),
                "avg_actions_per_active_window": float(session_density.get("avg_actions_per_active_window", 0.0) or 0.0),
                "max_actions_per_window": float(session_density.get("max_actions_per_window", 0.0) or 0.0),
                "window_minutes": window_minutes,
                "by_session": [dict(session_density)],
            }
        elif "burst_density" not in session_stats:
            session_stats["burst_density"] = {
                "active_windows": 0,
                "avg_actions_per_active_window": 0.0,
                "max_actions_per_window": 0.0,
                "window_minutes": window_minutes,
                "by_session": [],
            }
        session_snapshot = {
            "path": str(base_snapshot_path),
            "snapshot_timestamp": snapshot_timestamp,
            "trading_day_key": trading_day_key,
            "window_id": str(window_id),
            "window_start": window_start,
            "window_end": window_end,
            "session_name": session_name,
            "source_window_id": source_window_id,
            "xau_proof_state": dict(resolved_proof_state),
        }
        window_reports[window_id] = {
            "snapshot": session_snapshot,
            "xau_grid_pure_slice": session_stats,
        }
        snapshot_suite[window_id] = dict(session_snapshot)
        trades = float(session_stats.get("trades", 0.0) or 0.0)
        window_activity_state[window_id] = {
            "window_id": str(window_id),
            "active": bool(trades > 0.0),
            "reason": "" if trades > 0.0 else "no_xau_grid_trades_in_window",
            "xau_grid_trades": int(trades),
            "window_start": window_start,
            "window_end": window_end,
        }
    return window_reports, snapshot_suite, window_activity_state


def _windowed_features(features: pd.DataFrame, window_spec: dict[str, Any] | None) -> pd.DataFrame:
    if features.empty or not window_spec:
        return features
    start_ts = _maybe_timestamp(window_spec.get("start"))
    end_ts = _maybe_timestamp(window_spec.get("end"))
    session_names = {str(item).upper() for item in list(window_spec.get("session_names") or []) if str(item).strip()}
    weekend_only = bool(window_spec.get("weekend_only", False))
    weekdays_only = bool(window_spec.get("weekdays_only", False))
    timestamps = features["time"].apply(_normalize_timestamp)
    mask = pd.Series(True, index=features.index)
    if start_ts is not None:
        mask &= timestamps >= start_ts
    if end_ts is not None:
        mask &= timestamps < end_ts
    if weekend_only:
        mask &= timestamps.apply(lambda ts: bool(is_weekend_market_mode(ts.to_pydatetime())))
    if weekdays_only:
        mask &= timestamps.apply(lambda ts: not bool(is_weekend_market_mode(ts.to_pydatetime())))
    if session_names:
        mask &= timestamps.apply(lambda ts: str(dominant_session_name(ts.to_pydatetime()) or "").upper() in session_names)
    return features.loc[mask].reset_index(drop=True)


def _market_open_features(features: pd.DataFrame, symbol_key: str) -> pd.DataFrame:
    if features.empty or "time" not in features:
        return features
    symbol = str(symbol_key or "").upper()
    timestamps = features["time"].apply(_normalize_timestamp)
    mask = timestamps.apply(lambda ts: bool(market_open_tuple(symbol, ts.to_pydatetime())[0]))
    return features.loc[mask].reset_index(drop=True)


def _derive_window_specs(runtime: dict[str, Any], symbols: list[str]) -> list[dict[str, Any]]:
    preferred = []
    for item in ("BTCUSD", "XAUUSD", *symbols):
        normalized = str(item or "").upper()
        if normalized and normalized not in preferred:
            preferred.append(normalized)
    anchor_times: list[pd.Timestamp] = []
    for symbol_key in preferred:
        try:
            frames = _load_frames(runtime, symbol_key)
        except Exception:
            frames = {}
        frame = frames.get("M5")
        if frame is None or len(frame) < 200 or "time" not in frame:
            continue
        anchor_times.extend(
            _normalize_timestamp(value)
            for value in list(frame["time"])
            if _maybe_timestamp(value) is not None
        )
    if not anchor_times:
        return []
    anchor_times = sorted(set(anchor_times))
    earliest = min(anchor_times)
    latest = max(anchor_times) + pd.Timedelta(minutes=5)
    total_span = latest - earliest
    if total_span <= pd.Timedelta(days=3):
        split_one = earliest + (total_span / 3.0)
        split_two = earliest + ((total_span * 2.0) / 3.0)
    else:
        window_span = total_span / 3.0
        split_one = earliest + window_span
        split_two = earliest + (window_span * 2.0)
    mixed_start = earliest
    mixed_end = split_one
    weekday_start = split_one
    weekday_end = split_two
    weekend_heavy_start = split_two
    weekend_end = latest

    return [
        {
            "window_id": "window_a",
            "label": "recent_weekend_heavy",
            "start": weekend_heavy_start,
            "end": weekend_end,
        },
        {
            "window_id": "window_b",
            "label": "recent_weekday_london_ny",
            "start": weekday_start,
            "end": weekday_end,
        },
        {
            "window_id": "window_c",
            "label": "prior_mixed",
            "start": mixed_start,
            "end": mixed_end,
        },
    ]


def _rr_band_midpoint(band: str, default: float = 2.0) -> float:
    raw = str(band or "").strip()
    if "-" not in raw:
        return float(default)
    lower_raw, _, upper_raw = raw.partition("-")
    lower = _safe_float(lower_raw, default)
    upper = _safe_float(upper_raw, lower)
    if upper < lower:
        upper = lower
    return float((lower + upper) / 2.0)


def _symbol_backtest_overrides(symbol_key: str) -> dict[str, float]:
    symbol = str(symbol_key or "").upper()
    fx_5_digit = {
        "EURUSD",
        "GBPUSD",
        "AUDNZD",
    }
    jpy_pairs = {
        "USDJPY",
        "AUDJPY",
        "NZDJPY",
        "EURJPY",
        "GBPJPY",
    }
    if symbol in fx_5_digit:
        return {
            "instrument_point_size": 0.00001,
            "spread_points": 8.0,
            "slippage_points": 2.0,
            "contract_size": 100000.0,
            "be_trigger_r": 0.70,
            "trail_start_r": 1.40,
            "partial1_r": 0.90,
            "partial1_fraction": 0.20,
            "partial2_r": 1.80,
            "partial2_fraction": 0.20,
            "close_on_signal_flip": False,
        }
    if symbol in jpy_pairs:
        return {
            "instrument_point_size": 0.001,
            "spread_points": 10.0,
            "slippage_points": 2.0,
            "contract_size": 100000.0,
            "be_trigger_r": 0.70,
            "trail_start_r": 1.40,
            "partial1_r": 0.90,
            "partial1_fraction": 0.20,
            "partial2_r": 1.80,
            "partial2_fraction": 0.20,
            "close_on_signal_flip": False,
        }
    if symbol == "XAUUSD":
        return {
            "instrument_point_size": 0.01,
            "spread_points": 20.0,
            "slippage_points": 4.0,
            "contract_size": 100.0,
            "be_trigger_r": 0.80,
            "trail_start_r": 1.80,
            "partial1_r": 1.00,
            "partial1_fraction": 0.20,
            "partial2_r": 2.40,
            "partial2_fraction": 0.20,
            "close_on_signal_flip": False,
        }
    if symbol == "BTCUSD":
        return {
            "instrument_point_size": 0.01,
            "spread_points": 150.0,
            "slippage_points": 20.0,
            "contract_size": 1.0,
            "commission_per_lot": 0.0,
            "be_trigger_r": 0.70,
            "trail_start_r": 1.50,
            "partial1_r": 1.00,
            "partial1_fraction": 0.15,
            "partial2_r": 2.20,
            "partial2_fraction": 0.20,
            "close_on_signal_flip": False,
        }
    if symbol == "NAS100":
        return {
            "instrument_point_size": 0.01,
            "spread_points": 60.0,
            "slippage_points": 10.0,
            "contract_size": 1.0,
            "commission_per_lot": 0.0,
            "be_trigger_r": 0.70,
            "trail_start_r": 1.30,
            "partial1_r": 0.90,
            "partial1_fraction": 0.15,
            "partial2_r": 1.80,
            "partial2_fraction": 0.20,
            "close_on_signal_flip": False,
        }
    if symbol == "USOIL":
        return {
            "instrument_point_size": 0.01,
            "spread_points": 8.0,
            "slippage_points": 2.0,
            "contract_size": 1000.0,
            "be_trigger_r": 0.70,
            "trail_start_r": 1.30,
            "partial1_r": 0.90,
            "partial1_fraction": 0.15,
            "partial2_r": 1.80,
            "partial2_fraction": 0.20,
            "close_on_signal_flip": False,
        }
    return {
        "instrument_point_size": 0.01,
        "spread_points": 25.0,
        "slippage_points": 5.0,
        "contract_size": 100.0,
        "be_trigger_r": 0.70,
        "trail_start_r": 1.30,
        "partial1_r": 0.90,
        "partial1_fraction": 0.20,
        "partial2_r": 1.80,
        "partial2_fraction": 0.20,
        "close_on_signal_flip": False,
    }


class RouterStrategyEngineAdapter:
    def __init__(
        self,
        *,
        symbol_key: str,
        strategy_engine: StrategyEngine,
        strategy_router: Any,
        session_profile: Any,
        regime_detector: Any,
        max_spread_points: float,
        top_n: int = 1,
        closed_trades: list[dict[str, Any]] | None = None,
        current_day_key: str = "",
        candidate_tier_config: dict[str, Any] | None = None,
        orchestrator_config: dict[str, Any] | None = None,
        grid_scalper: Any | None = None,
        exits_config: dict[str, Any] | None = None,
        window_id: str = "",
        idea_lifecycle_config: dict[str, Any] | None = None,
    ) -> None:
        self.symbol_key = str(symbol_key)
        self.symbol = self.symbol_key
        self.strategy_engine = strategy_engine
        self.strategy_router = strategy_router
        self.session_profile = session_profile
        self.regime_detector = regime_detector
        self.max_spread_points = float(max_spread_points)
        self.top_n = max(1, int(top_n))
        self.closed_trades = list(closed_trades or [])
        self.current_day_key = str(current_day_key or "")
        self.candidate_tier_config = dict(candidate_tier_config or {})
        self.orchestrator_config = dict(orchestrator_config or {})
        self.grid_scalper = grid_scalper
        self.exits_config = dict(exits_config or {})
        self.window_id = str(window_id or "")
        lifecycle_config = dict(idea_lifecycle_config or {})
        self.idea_lifecycle = TradeIdeaLifecycle(
            archive_minutes=int(lifecycle_config.get("archive_minutes", 10)),
            max_rechecks_per_idea=int(lifecycle_config.get("max_rechecks_per_idea", 5)),
            max_active_ideas_per_symbol=int(lifecycle_config.get("max_active_ideas_per_symbol", 3)),
            recheck_seconds_default=int(lifecycle_config.get("recheck_seconds_default", 60)),
            recheck_seconds_by_session={
                str(key).upper(): int(value)
                for key, value in (lifecycle_config.get("recheck_seconds_by_session", {}) or {}).items()
            },
            cooldown_seconds_by_session={
                str(key).upper(): int(value)
                for key, value in (lifecycle_config.get("cooldown_seconds_by_session", {}) or {}).items()
            },
        )
        self.approved_reason_counts: Counter[str] = Counter()
        self.quality_tier_counts: Counter[str] = Counter()
        self.verified_trade_count = 0
        self.recycled_trade_count = 0
        self.verification_entries: list[dict[str, Any]] = []
        self.replay_signal_memory: dict[str, dict[str, Any]] = {}

    def _selection_limit(
        self,
        *,
        ranked: list[tuple[SignalCandidate, dict[str, Any]]],
        session_name: str,
    ) -> int:
        limit = max(1, int(self.top_n))
        if self.symbol_key != "XAUUSD":
            return min(limit, len(ranked))
        session_key = str(session_name or "").upper()
        if session_key not in {"LONDON", "OVERLAP", "NEW_YORK"}:
            return min(limit, len(ranked))
        burst_candidates = [
            candidate
            for candidate, _ in ranked
            if _is_xau_grid_setup(str(candidate.setup or ""))
            and str((candidate.meta or {}).get("grid_action") or "").upper() in {"START", "ADD"}
        ]
        if len(burst_candidates) < 2:
            return min(limit, len(ranked))
        cycle_burst_sizes: dict[str, int] = {}
        for candidate in burst_candidates:
            meta = candidate.meta or {}
            cycle_key = str(meta.get("grid_cycle_id") or candidate.signal_id or "")
            cycle_burst_sizes[cycle_key] = max(
                int(cycle_burst_sizes.get(cycle_key, 0) or 0),
                int(meta.get("grid_burst_size", 1) or 1),
            )
        total_burst_size = max(1, sum(cycle_burst_sizes.values()))
        limit = max(limit, min(len(burst_candidates), max(2, total_burst_size)))
        return min(limit, len(ranked))

    def generate(
        self,
        features: pd.DataFrame,
        regime: str,
        open_positions: list[dict] | None = None,
        max_positions_per_symbol: int = 1,
    ) -> list[SignalCandidate]:
        if features.empty:
            return []
        row = features.iloc[-1]
        timestamp = StrategyEngine._resolve_timestamp(features, row)
        current_time = _normalize_timestamp(timestamp).to_pydatetime()
        regime_obj = self.regime_detector.classify(row)
        session_ctx = self.session_profile.classify(current_time)
        all_open_positions = list(open_positions or [])
        grid_open_positions = [
            position for position in all_open_positions
            if _is_xau_grid_setup(str(position.get("setup", "")))
        ]
        router_candidates = self.strategy_router.generate(
            symbol=self.symbol_key,
            features=features,
            regime=regime_obj,
            session=session_ctx,
            strategy_engine=self.strategy_engine,
            open_positions=all_open_positions,
            max_positions_per_symbol=max_positions_per_symbol,
            current_time=current_time,
        )
        candidates = list(router_candidates)
        if self.symbol_key == "XAUUSD" and getattr(self.grid_scalper, "enabled", False):
            grid_candidates: list[SignalCandidate] = []
            try:
                grid_decision = self.grid_scalper.evaluate(
                    symbol=self.symbol_key,
                    features=features,
                    row=row,
                    open_positions=grid_open_positions,
                    session_name=session_ctx.session_name,
                    news_safe=True,
                    now_utc=current_time,
                    spread_points=float(row.get("m5_spread", 0.0) or 0.0),
                    contract_size=100.0,
                    approver=None,
                )
                grid_candidates = list(grid_decision.candidates or [])
            except Exception:
                grid_candidates = []
            usable_grid_candidates = [
                candidate
                for candidate in grid_candidates
                if (
                    not str(((candidate.meta or {}).get("grid_entry_profile") or "")).startswith("grid_stretch_reversion")
                    and not self.grid_scalper.native_candidate_is_unsafe_extreme_bucket(
                        candidate=candidate,
                        row=row,
                        session_name=session_ctx.session_name,
                        regime_state=str(regime_obj.state_label or regime_obj.label),
                    )
                )
            ]
            router_candidates = [
                candidate for candidate in router_candidates if _is_xau_higher_tf_candidate_setup(candidate.setup)
            ]
            grid_overlay_candidates: list[SignalCandidate] = []
            grid_overlay_source_ids: set[str] = set()
            mirror_extension_side_counts: Counter[str] = Counter(
                str(candidate.side).upper()
                for candidate in router_candidates
                if str(candidate.setup).upper().startswith(
                    (
                        "XAUUSD_ATR_EXPANSION_SCALPER",
                        "XAUUSD_M15_STRUCTURED_PULLBACK",
                        "XAUUSD_M15_STRUCTURED_SWEEP_RETEST",
                        "XAUUSD_M15_STRUCTURED_BREAKOUT",
                        "XAUUSD_M15_FIX_FLOW",
                        "XAU_BREAKOUT_RETEST",
                        "XAUUSD_M1_MICRO_SCALPER",
                    )
                )
                and float(getattr(candidate, "confluence_score", 0.0) or 0.0) >= 3.7
                and float(getattr(candidate, "score_hint", 0.0) or 0.0) >= 0.70
            )
            mirror_overlay_capacity = max(
                2,
                int(self.orchestrator_config.get("xau_grid_max_entries_per_symbol_loop", 6) or 6),
            )
            if (
                len(grid_open_positions) < mirror_overlay_capacity
                and session_ctx.session_name in set(getattr(self.grid_scalper, "allowed_sessions", ()))
                and not usable_grid_candidates
                and (
                    not grid_open_positions
                    or bool(mirror_extension_side_counts and max(mirror_extension_side_counts.values()) >= 2)
                )
            ):
                preferred_grid_source = sorted(
                    (
                        candidate for candidate in router_candidates
                        if str(candidate.setup).upper().startswith(
                            (
                                "XAUUSD_ATR_EXPANSION_SCALPER",
                                "XAUUSD_M15_STRUCTURED_PULLBACK",
                                "XAUUSD_M15_STRUCTURED_SWEEP_RETEST",
                                "XAUUSD_M15_STRUCTURED_BREAKOUT",
                                "XAUUSD_M15_FIX_FLOW",
                                "XAU_BREAKOUT_RETEST",
                                "XAUUSD_M1_MICRO_SCALPER",
                            )
                        )
                    ),
                    key=lambda item: (
                        float(getattr(item, "confluence_score", 0.0) or 0.0),
                        float(getattr(item, "score_hint", 0.0) or 0.0),
                    ),
                    reverse=True,
                )
                if preferred_grid_source:
                    source_candidate = preferred_grid_source[0]
                    source_setup = str(source_candidate.setup).upper()
                    primary_side = str(source_candidate.side).upper()
                    source_meta = dict(getattr(source_candidate, "meta", {}) or {})
                    support_candidates: list[SignalCandidate] = []
                    for support_candidate in preferred_grid_source[1:]:
                        if len(support_candidates) >= 2:
                            break
                        if str(support_candidate.side).upper() != primary_side:
                            continue
                        support_setup = str(support_candidate.setup).upper()
                        if support_setup == source_setup:
                            continue
                        if float(getattr(support_candidate, "confluence_score", 0.0) or 0.0) < 3.7:
                            continue
                        if float(getattr(support_candidate, "score_hint", 0.0) or 0.0) < 0.62:
                            continue
                        support_candidates.append(support_candidate)
                    support_source_count = len(support_candidates)
                    support_source_setups = [str(candidate.setup) for candidate in support_candidates]
                    mirror_profile = "grid_directional_flow_long" if str(source_candidate.side).upper() == "BUY" else "grid_directional_flow_short"
                    if source_setup.startswith("XAUUSD_M15_STRUCTURED_PULLBACK"):
                        mirror_profile = "grid_m15_pullback_reclaim_long" if str(source_candidate.side).upper() == "BUY" else "grid_m15_pullback_reclaim_short"
                    elif source_setup.startswith("XAUUSD_M15_STRUCTURED_SWEEP_RETEST"):
                        mirror_profile = "grid_liquidity_reclaim_long" if str(source_candidate.side).upper() == "BUY" else "grid_liquidity_reclaim_short"
                    elif source_setup.startswith(("XAUUSD_M15_STRUCTURED_BREAKOUT", "XAU_BREAKOUT_RETEST", "XAUUSD_ATR_EXPANSION_SCALPER")):
                        mirror_profile = "grid_breakout_reclaim_long" if str(source_candidate.side).upper() == "BUY" else "grid_breakout_reclaim_short"
                    atr = max(float(row.get("m5_atr_14", 0.0) or 0.0), 1e-6)
                    atr_avg = max(float(row.get("m5_atr_avg_20", atr) or atr), 1e-6)
                    session_profile = self.grid_scalper._session_profile(
                        session_name=session_ctx.session_name,
                        now_utc=current_time,
                        atr_ratio=max(atr / atr_avg, 0.0),
                        spread_points=float(row.get("m5_spread", 0.0) or 0.0),
                    )
                    mirror_grid_mode = "ATTACK_GRID" if session_profile == "AGGRESSIVE" else "NORMAL_GRID"
                    mirror_step_points = self.grid_scalper._step_points(
                        atr=atr,
                        multiplier=self.grid_scalper._grid_spacing_multiplier(mirror_grid_mode),
                    )
                    mirror_stop_atr = max(0.85, float(getattr(source_candidate, "stop_atr", 0.95) or 0.95))
                    alignment_score = clamp(
                        float(source_meta.get("multi_tf_alignment_score") or row.get("multi_tf_alignment_score") or 0.5),
                        0.0,
                        1.0,
                    )
                    seasonality_score = clamp(
                        float(source_meta.get("seasonality_edge_score") or row.get("seasonality_edge_score") or 0.5),
                        0.0,
                        1.0,
                    )
                    fractal_score = clamp(
                        float(
                            source_meta.get("fractal_persistence_score")
                            or row.get("fractal_persistence_score")
                            or row.get("hurst_persistence_score")
                            or row.get("m5_hurst_proxy_64")
                            or 0.5
                        ),
                        0.0,
                        1.0,
                    )
                    instability_score = clamp(
                        float(
                            source_meta.get("market_instability_score")
                            or row.get("market_instability_score")
                            or row.get("feature_drift_score")
                            or 0.0
                        ),
                        0.0,
                        1.0,
                    )
                    feature_drift_score = clamp(
                        float(source_meta.get("feature_drift_score") or row.get("feature_drift_score") or 0.0),
                        0.0,
                        1.0,
                    )
                    compression_expansion_score = clamp(
                        float(source_meta.get("compression_expansion_score") or row.get("compression_expansion_score") or 0.0),
                        0.0,
                        1.0,
                    )
                    body_efficiency = clamp(
                        float(
                            row.get("m5_body_efficiency")
                            or row.get("m5_candle_efficiency")
                            or row.get("body_efficiency")
                            or 0.5
                        ),
                        0.0,
                        1.0,
                    )
                    volume_ratio = max(float(row.get("m5_volume_ratio", 1.0) or 1.0), 0.0)
                    volume_score = clamp(volume_ratio / 1.25, 0.0, 1.0)
                    trend_efficiency = clamp(
                        float(
                            source_meta.get("trend_efficiency_score")
                            or row.get("trend_efficiency_score")
                            or row.get("m5_trend_efficiency")
                            or 0.5
                        ),
                        0.0,
                        1.0,
                    )
                    spread_score = clamp(
                        1.0 - (float(row.get("m5_spread", 0.0) or 0.0) / max(float(self.grid_scalper.spread_max_points), 1.0)),
                        0.0,
                        1.0,
                    )
                    mirror_session_fit = clamp(
                        max(
                            float(source_meta.get("session_fit", 0.0) or 0.0),
                            0.88 if session_profile == "AGGRESSIVE" else (0.76 if session_profile == "MODERATE" else 0.54),
                        ),
                        0.0,
                        1.0,
                    )
                    mirror_volatility_fit = clamp(
                        max(
                            float(source_meta.get("volatility_fit", 0.0) or 0.0),
                            0.44
                            + (0.20 * compression_expansion_score)
                            + (0.16 * spread_score)
                            + (0.12 * clamp(1.0 - instability_score, 0.0, 1.0))
                            + (0.08 * float(session_profile == "AGGRESSIVE")),
                        ),
                        0.0,
                        1.0,
                    )
                    mirror_pair_behavior_fit = clamp(
                        max(
                            float(source_meta.get("pair_behavior_fit", 0.0) or 0.0),
                            0.34
                            + (0.24 * alignment_score)
                            + (0.18 * fractal_score)
                            + (0.12 * seasonality_score)
                            + (0.10 * trend_efficiency),
                        ),
                        0.0,
                        1.0,
                    )
                    mirror_execution_quality_fit = clamp(
                        max(
                            float(source_meta.get("execution_quality_fit", 0.0) or 0.0),
                            0.36
                            + (0.18 * spread_score)
                            + (0.16 * body_efficiency)
                            + (0.14 * volume_score)
                            + (0.12 * clamp(1.0 - instability_score, 0.0, 1.0))
                            + (0.06 * clamp(1.0 - feature_drift_score, 0.0, 1.0)),
                        ),
                        0.0,
                        1.0,
                    )
                    mirror_entry_timing_score = clamp(
                        max(
                            float(source_meta.get("entry_timing_score", 0.0) or 0.0),
                            0.30
                            + (0.16 * body_efficiency)
                            + (0.14 * volume_score)
                            + (0.14 * compression_expansion_score)
                            + (0.12 * alignment_score)
                            + (0.08 * clamp(1.0 - instability_score, 0.0, 1.0)),
                        ),
                        0.0,
                        1.0,
                    )
                    mirror_structure_cleanliness_score = clamp(
                        max(
                            float(source_meta.get("structure_cleanliness_score", 0.0) or 0.0),
                            0.28
                            + (0.24 * alignment_score)
                            + (0.18 * fractal_score)
                            + (0.14 * trend_efficiency)
                            + (0.08 * body_efficiency)
                            + (0.08 * compression_expansion_score)
                            - (0.08 * instability_score),
                        ),
                        0.0,
                        1.0,
                    )
                    mirror_regime_fit = clamp(
                        max(
                            float(source_meta.get("regime_fit", 0.0) or 0.0),
                            0.30
                            + (0.18 * alignment_score)
                            + (0.16 * fractal_score)
                            + (0.12 * compression_expansion_score)
                            + (0.08 * trend_efficiency)
                            + (0.08 * float(mirror_profile.startswith(("grid_breakout_reclaim", "grid_directional_flow", "grid_m15_pullback_reclaim"))))
                            - (0.08 * instability_score)
                            - (0.04 * feature_drift_score),
                        ),
                        0.0,
                        1.0,
                    )
                    mirror_router_rank_score = clamp(
                        max(
                            float(source_meta.get("router_rank_score", getattr(source_candidate, "score_hint", 0.0)) or 0.0),
                            0.58
                            + (0.10 * alignment_score)
                            + (0.08 * compression_expansion_score)
                            + (0.06 * body_efficiency)
                            + (0.04 * float(session_profile == "AGGRESSIVE"))
                            - (0.04 * instability_score),
                        ),
                        0.58,
                        0.92,
                    )
                    mirror_quality_tier = "A"
                    if not (
                        mirror_regime_fit >= 0.60
                        and mirror_entry_timing_score >= 0.58
                        and mirror_structure_cleanliness_score >= 0.58
                        and mirror_execution_quality_fit >= 0.56
                    ):
                        mirror_quality_tier = "C"
                    mirror_cycle_id = deterministic_id(
                        self.symbol_key,
                        "xau-grid-mirror-cycle",
                        source_candidate.side,
                        source_candidate.setup,
                        row["time"],
                    )
                    prime_recovery_active = bool(
                        session_ctx.session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
                        and (
                            self.grid_scalper._last_signal_emitted_at is None
                            or (current_time - self.grid_scalper._last_signal_emitted_at).total_seconds()
                            >= (int(self.grid_scalper.prime_recovery_idle_minutes) * 60)
                        )
                    )
                    mirror_confluence = max(3.2, float(getattr(source_candidate, "confluence_score", 0.0) or 0.0))
                    score_bonus = 0.05
                    if source_setup.startswith(("XAUUSD_M15_FIX_FLOW", "XAU_BREAKOUT_RETEST", "XAUUSD_M15_STRUCTURED_BREAKOUT", "XAUUSD_ATR_EXPANSION_SCALPER")):
                        score_bonus = 0.08
                    base_score_hint = clamp(
                        float(getattr(source_candidate, "score_hint", 0.0) or 0.0)
                        + score_bonus
                        + (0.01 * support_source_count),
                        0.60,
                        0.92,
                    )
                    burst_count = self.grid_scalper._burst_start_count(
                        session_profile=session_profile,
                        grid_mode=mirror_grid_mode,
                        entry_profile=mirror_profile,
                        confluence=mirror_confluence,
                        alignment_score=alignment_score,
                        fractal_score=fractal_score,
                        trend_efficiency=trend_efficiency,
                        body_efficiency=body_efficiency,
                        compression_expansion_score=compression_expansion_score,
                        instability_score=instability_score,
                        prime_recovery_active=prime_recovery_active,
                        support_sources=support_source_count,
                        grid_max_levels=int(self.grid_scalper.max_levels),
                    )
                    if (
                        session_profile == "AGGRESSIVE"
                        and mirror_quality_tier == "A"
                        and source_setup.startswith(
                            (
                                "XAUUSD_M1_MICRO_SCALPER",
                                "XAUUSD_ATR_EXPANSION_SCALPER",
                                "XAUUSD_M15_STRUCTURED_BREAKOUT",
                                "XAU_BREAKOUT_RETEST",
                                "XAUUSD_M15_FIX_FLOW",
                            )
                        )
                        and mirror_confluence >= 3.30
                        and mirror_router_rank_score >= 0.83
                        and instability_score <= 0.10
                        and mirror_entry_timing_score >= 0.94
                        and mirror_structure_cleanliness_score >= 0.84
                    ):
                        burst_count = max(int(burst_count), 3)
                    if (
                        session_profile == "AGGRESSIVE"
                        and mirror_quality_tier == "A"
                        and source_setup.startswith(
                            (
                                "XAUUSD_M1_MICRO_SCALPER",
                                "XAUUSD_ATR_EXPANSION_SCALPER",
                                "XAUUSD_M15_STRUCTURED_BREAKOUT",
                                "XAU_BREAKOUT_RETEST",
                            )
                        )
                        and mirror_confluence >= 3.38
                        and mirror_router_rank_score >= 0.84
                        and instability_score <= 0.08
                        and mirror_entry_timing_score >= 0.95
                        and mirror_structure_cleanliness_score >= 0.84
                    ):
                        burst_count = max(int(burst_count), 4)
                    overlap_session = session_ctx.session_name == "OVERLAP"
                    london_session = session_ctx.session_name == "LONDON"
                    prime_mirror_strong_ready = bool(
                        mirror_quality_tier == "A"
                        and support_source_count >= 1
                        and source_setup.startswith(
                            (
                                "XAUUSD_ATR_EXPANSION_SCALPER",
                                "XAUUSD_M15_STRUCTURED_BREAKOUT",
                                "XAU_BREAKOUT_RETEST",
                                "XAUUSD_M15_FIX_FLOW",
                            )
                        )
                        and mirror_confluence >= 3.55
                        and mirror_router_rank_score >= 0.85
                        and mirror_entry_timing_score >= 0.95
                        and mirror_structure_cleanliness_score >= 0.86
                        and alignment_score >= 0.68
                        and instability_score <= 0.08
                    )
                    overlap_prime_mirror_ready = bool(
                        mirror_quality_tier == "A"
                        and source_setup.startswith(
                            (
                                "XAUUSD_ATR_EXPANSION_SCALPER",
                                "XAUUSD_M15_STRUCTURED_BREAKOUT",
                                "XAU_BREAKOUT_RETEST",
                                "XAUUSD_M15_FIX_FLOW",
                            )
                        )
                        and mirror_confluence >= 3.46
                        and mirror_router_rank_score >= 0.84
                        and mirror_entry_timing_score >= 0.95
                        and mirror_structure_cleanliness_score >= 0.84
                        and alignment_score >= 0.60
                        and instability_score <= 0.09
                    )
                    overlap_mirror_ready = bool((not overlap_session) or overlap_prime_mirror_ready)
                    if not overlap_mirror_ready:
                        burst_count = 0
                    elif overlap_session:
                        burst_count = min(int(burst_count), 1)
                    elif (
                        london_session
                        and prime_mirror_strong_ready
                        and mirror_confluence >= 3.72
                        and mirror_router_rank_score >= 0.87
                        and alignment_score >= 0.72
                        and instability_score <= 0.05
                    ):
                        burst_count = min(max(int(burst_count), 2), 2)
                    elif london_session and not prime_mirror_strong_ready:
                        burst_count = min(int(burst_count), 1)
                    if burst_count <= 0:
                        return list(router_candidates)
                    grid_overlay_source_ids.add(str(source_candidate.signal_id))
                    for support_candidate in support_candidates:
                        grid_overlay_source_ids.add(str(support_candidate.signal_id))
                    source_meta.update(
                        {
                            "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                            "setup_family": "GRID",
                            "grid_cycle": True,
                            "grid_action": "START",
                            "grid_cycle_id": str(mirror_cycle_id),
                            "grid_burst_size": int(burst_count),
                            "grid_max_levels": int(self.grid_scalper.max_levels),
                            "grid_probe": False,
                            "grid_source_setup": str(source_candidate.setup),
                            "grid_support_source_count": int(support_source_count),
                            "grid_support_source_setups": list(support_source_setups),
                            "session_profile": str(session_profile),
                            "grid_entry_profile": str(mirror_profile),
                            "grid_mode": str(mirror_grid_mode),
                            "grid_volatility_multiplier": float(self.grid_scalper._grid_spacing_multiplier(mirror_grid_mode)),
                            "grid_stop_atr_k": float(mirror_stop_atr),
                            "xau_engine": "GRID_DIRECTIONAL_MIRROR",
                            "mirror_live_enabled": True,
                            "compression_proxy_state": str(
                                source_meta.get("compression_proxy_state")
                                or row.get("compression_proxy_state")
                                or row.get("compression_state")
                                or "NEUTRAL"
                            ),
                            "compression_expansion_score": float(compression_expansion_score),
                            "multi_tf_alignment_score": float(alignment_score),
                            "seasonality_edge_score": float(seasonality_score),
                            "fractal_persistence_score": float(fractal_score),
                            "market_instability_score": float(instability_score),
                            "feature_drift_score": float(feature_drift_score),
                            "trend_efficiency_score": float(trend_efficiency),
                            "regime_fit": float(mirror_regime_fit),
                            "session_fit": float(mirror_session_fit),
                            "volatility_fit": float(mirror_volatility_fit),
                            "pair_behavior_fit": float(mirror_pair_behavior_fit),
                            "execution_quality_fit": float(mirror_execution_quality_fit),
                            "entry_timing_score": float(mirror_entry_timing_score),
                            "structure_cleanliness_score": float(mirror_structure_cleanliness_score),
                            "router_rank_score": float(mirror_router_rank_score),
                            "quality_tier": str(mirror_quality_tier),
                            "strategy_recent_performance_seed": float(max(0.58, float(source_meta.get("strategy_recent_performance_seed", 0.0) or 0.0))),
                            "prime_session_recovery_active": bool(prime_recovery_active),
                        }
                    )
                    for leg_index in range(1, int(burst_count) + 1):
                        leg_step_points = float(mirror_step_points)
                        if leg_index > 1 and session_profile == "AGGRESSIVE":
                            leg_step_points *= 0.96 if mirror_grid_mode == "ATTACK_GRID" else 0.98
                        leg_stop_points = float(self.grid_scalper._entry_stop_points(step_points=leg_step_points, probe_candidate=False))
                        leg_lot = max(
                            0.01,
                            float(self.grid_scalper._lot_for_level(leg_index))
                            * (
                                self.grid_scalper.attack_grid_lot_multiplier
                                if mirror_grid_mode == "ATTACK_GRID"
                                else 1.0
                            ),
                        )
                        if leg_index > 1:
                            leg_lot *= 1.0 if mirror_grid_mode == "ATTACK_GRID" else 0.95
                        leg_meta = dict(source_meta)
                        leg_meta.update(
                            {
                                "grid_level": int(leg_index),
                                "grid_lot": float(leg_lot),
                                "grid_step_atr_k": float(self.grid_scalper.step_atr_k),
                                "grid_step_points": float(leg_step_points),
                                "chosen_spacing_points": float(leg_step_points),
                                "stop_points": float(leg_stop_points),
                                "grid_burst_index": int(leg_index),
                            }
                        )
                        grid_overlay_candidates.append(
                            SignalCandidate(
                                signal_id=deterministic_id(
                                    self.symbol_key,
                                    "xau-grid-mirror",
                                    source_candidate.side,
                                    source_candidate.setup,
                                    row["time"],
                                    mirror_cycle_id,
                                    leg_index,
                                ),
                                setup="XAUUSD_M5_GRID_SCALPER_START",
                                side=str(source_candidate.side).upper(),
                                score_hint=clamp(base_score_hint - ((leg_index - 1) * 0.01), 0.60, 0.92),
                                reason=f"grid_directional_mirror:{source_candidate.setup}",
                                stop_atr=mirror_stop_atr,
                                tp_r=max(1.8, float(getattr(source_candidate, "tp_r", 2.0) or 2.0)),
                                entry_kind="GRID_START",
                                strategy_family="GRID",
                                confluence_score=mirror_confluence,
                                confluence_required=max(3.0, float(getattr(source_candidate, "confluence_required", 0.0) or 0.0)),
                                meta=leg_meta,
                            )
                        )
                    support_cycle_room = max(
                        0,
                        max(2, int(self.orchestrator_config.get("xau_grid_max_entries_per_symbol_loop", 6) or 6))
                        - int(burst_count),
                    )
                    if overlap_session or london_session:
                        support_cycle_room = 0
                    support_cycle_candidate: SignalCandidate | None = support_candidates[0] if support_candidates else None
                    support_cycle_from_primary = False
                    if (
                        support_cycle_candidate is None
                        and support_cycle_room > 0
                        and session_profile == "AGGRESSIVE"
                        and mirror_quality_tier == "A"
                        and int(burst_count) >= 3
                        and mirror_confluence >= 3.30
                        and mirror_router_rank_score >= 0.83
                        and instability_score <= 0.10
                        and mirror_entry_timing_score >= 0.94
                        and mirror_structure_cleanliness_score >= 0.84
                        and source_setup.startswith(
                            (
                                "XAUUSD_M1_MICRO_SCALPER",
                                "XAUUSD_ATR_EXPANSION_SCALPER",
                                "XAUUSD_M15_STRUCTURED_BREAKOUT",
                                "XAU_BREAKOUT_RETEST",
                                "XAUUSD_M15_FIX_FLOW",
                            )
                        )
                    ):
                        support_cycle_candidate = source_candidate
                        support_cycle_from_primary = True
                    if (
                        support_cycle_room > 0
                        and session_profile == "AGGRESSIVE"
                        and support_cycle_candidate is not None
                    ):
                        support_candidate = support_cycle_candidate
                        support_setup = str(support_candidate.setup).upper()
                        support_meta = dict(getattr(support_candidate, "meta", {}) or {})
                        support_profile = (
                            "grid_directional_flow_long"
                            if str(support_candidate.side).upper() == "BUY"
                            else "grid_directional_flow_short"
                        )
                        if support_setup.startswith("XAUUSD_M15_STRUCTURED_PULLBACK"):
                            support_profile = (
                                "grid_m15_pullback_reclaim_long"
                                if str(support_candidate.side).upper() == "BUY"
                                else "grid_m15_pullback_reclaim_short"
                            )
                        elif support_setup.startswith("XAUUSD_M15_STRUCTURED_SWEEP_RETEST"):
                            support_profile = (
                                "grid_liquidity_reclaim_long"
                                if str(support_candidate.side).upper() == "BUY"
                                else "grid_liquidity_reclaim_short"
                            )
                        elif support_setup.startswith(
                            (
                                "XAUUSD_M15_STRUCTURED_BREAKOUT",
                                "XAU_BREAKOUT_RETEST",
                                "XAUUSD_ATR_EXPANSION_SCALPER",
                            )
                        ):
                            support_profile = (
                                "grid_breakout_reclaim_long"
                                if str(support_candidate.side).upper() == "BUY"
                                else "grid_breakout_reclaim_short"
                            )
                        support_cycle_id = deterministic_id(
                            self.symbol_key,
                            "xau-grid-support-cycle",
                            support_candidate.side,
                            support_candidate.setup,
                            row["time"],
                        )
                        support_confluence = max(
                            3.2,
                            float(getattr(support_candidate, "confluence_score", 0.0) or 0.0),
                        )
                        support_stop_atr = max(
                            0.85,
                            float(getattr(support_candidate, "stop_atr", mirror_stop_atr) or mirror_stop_atr),
                        )
                        support_score_bonus = 0.04
                        if support_setup.startswith(
                            (
                                "XAUUSD_M15_FIX_FLOW",
                                "XAU_BREAKOUT_RETEST",
                                "XAUUSD_M15_STRUCTURED_BREAKOUT",
                                "XAUUSD_ATR_EXPANSION_SCALPER",
                            )
                        ):
                            support_score_bonus = 0.07
                        support_burst_count = 1
                        if (
                            not support_cycle_from_primary
                            and
                            mirror_quality_tier == "A"
                            and support_confluence >= 3.9
                            and alignment_score >= 0.48
                            and fractal_score >= 0.44
                            and trend_efficiency >= 0.38
                            and mirror_router_rank_score >= 0.70
                            and instability_score <= 0.55
                        ):
                            support_burst_count = 2
                        elif (
                            support_cycle_from_primary
                            and mirror_confluence >= 3.38
                            and mirror_router_rank_score >= 0.84
                            and instability_score <= 0.08
                            and mirror_entry_timing_score >= 0.95
                        ):
                            support_burst_count = 2
                        support_burst_count = min(
                            int(support_cycle_room),
                            int(support_burst_count),
                        )
                        if support_burst_count > 0:
                            support_base_score_hint = clamp(
                                float(getattr(support_candidate, "score_hint", 0.0) or 0.0)
                                + support_score_bonus
                                + (0.005 * max(0, support_source_count - 1)),
                                0.60,
                                0.90,
                            )
                            support_meta.update(
                                {
                                    "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                                    "setup_family": "GRID",
                                    "grid_cycle": True,
                                    "grid_action": "START",
                                    "grid_cycle_id": str(support_cycle_id),
                                    "grid_burst_size": int(support_burst_count),
                                    "grid_max_levels": int(self.grid_scalper.max_levels),
                                    "grid_probe": False,
                                    "grid_source_setup": str(support_candidate.setup),
                                    "grid_source_role": "SECONDARY_PRIMARY" if support_cycle_from_primary else "SECONDARY_SUPPORT",
                                    "grid_primary_cycle_id": str(mirror_cycle_id),
                                    "grid_primary_source_setup": str(source_candidate.setup),
                                    "grid_support_source_count": int(support_source_count),
                                    "grid_support_source_setups": list(support_source_setups),
                                    "session_profile": str(session_profile),
                                    "grid_entry_profile": str(support_profile),
                                    "grid_mode": str(mirror_grid_mode),
                                    "grid_volatility_multiplier": float(self.grid_scalper._grid_spacing_multiplier(mirror_grid_mode)),
                                    "grid_stop_atr_k": float(support_stop_atr),
                                    "xau_engine": "GRID_DIRECTIONAL_MIRROR_SECONDARY_PRIMARY" if support_cycle_from_primary else "GRID_DIRECTIONAL_MIRROR_SUPPORT",
                                    "mirror_live_enabled": True,
                                    "compression_proxy_state": str(
                                        support_meta.get("compression_proxy_state")
                                        or source_meta.get("compression_proxy_state")
                                        or row.get("compression_proxy_state")
                                        or row.get("compression_state")
                                        or "NEUTRAL"
                                    ),
                                    "compression_expansion_score": float(compression_expansion_score),
                                    "multi_tf_alignment_score": float(alignment_score),
                                    "seasonality_edge_score": float(seasonality_score),
                                    "fractal_persistence_score": float(fractal_score),
                                    "market_instability_score": float(instability_score),
                                    "feature_drift_score": float(feature_drift_score),
                                    "trend_efficiency_score": float(trend_efficiency),
                                    "regime_fit": float(mirror_regime_fit),
                                    "session_fit": float(mirror_session_fit),
                                    "volatility_fit": float(mirror_volatility_fit),
                                    "pair_behavior_fit": float(mirror_pair_behavior_fit),
                                    "execution_quality_fit": float(mirror_execution_quality_fit),
                                    "entry_timing_score": float(mirror_entry_timing_score),
                                    "structure_cleanliness_score": float(mirror_structure_cleanliness_score),
                                    "router_rank_score": float(mirror_router_rank_score),
                                    "quality_tier": str(mirror_quality_tier),
                                    "strategy_recent_performance_seed": float(
                                        max(0.58, float(support_meta.get("strategy_recent_performance_seed", 0.0) or 0.0))
                                    ),
                                    "prime_session_recovery_active": bool(prime_recovery_active),
                                }
                            )
                            for support_leg_index in range(1, int(support_burst_count) + 1):
                                support_step_points = float(mirror_step_points)
                                if support_leg_index > 1:
                                    support_step_points *= 0.95 if mirror_grid_mode == "ATTACK_GRID" else 0.98
                                support_stop_points = float(
                                    self.grid_scalper._entry_stop_points(
                                        step_points=support_step_points,
                                        probe_candidate=False,
                                    )
                                )
                                support_lot = max(
                                    0.01,
                                    float(self.grid_scalper._lot_for_level(support_leg_index))
                                    * (
                                        self.grid_scalper.attack_grid_lot_multiplier
                                        if mirror_grid_mode == "ATTACK_GRID"
                                        else 1.0
                                    ),
                                )
                                support_leg_meta = dict(support_meta)
                                support_leg_meta.update(
                                    {
                                        "grid_level": int(support_leg_index),
                                        "grid_lot": float(support_lot),
                                        "grid_step_atr_k": float(self.grid_scalper.step_atr_k),
                                        "grid_step_points": float(support_step_points),
                                        "chosen_spacing_points": float(support_step_points),
                                        "stop_points": float(support_stop_points),
                                        "grid_burst_index": int(support_leg_index),
                                    }
                                )
                                grid_overlay_candidates.append(
                                    SignalCandidate(
                                        signal_id=deterministic_id(
                                            self.symbol_key,
                                            "xau-grid-support",
                                            support_candidate.side,
                                            support_candidate.setup,
                                            row["time"],
                                            support_cycle_id,
                                            support_leg_index,
                                        ),
                                        setup="XAUUSD_M5_GRID_SCALPER_START",
                                        side=str(support_candidate.side).upper(),
                                        score_hint=clamp(
                                            support_base_score_hint - ((support_leg_index - 1) * 0.01),
                                            0.60,
                                            0.90,
                                        ),
                                        reason=f"grid_directional_support:{support_candidate.setup}",
                                        stop_atr=support_stop_atr,
                                        tp_r=max(1.8, float(getattr(support_candidate, "tp_r", 2.0) or 2.0)),
                                        entry_kind="GRID_START",
                                        strategy_family="GRID",
                                        confluence_score=support_confluence,
                                        confluence_required=max(
                                            3.0,
                                            float(getattr(support_candidate, "confluence_required", 0.0) or 0.0),
                                        ),
                                        meta=support_leg_meta,
                                    )
                                )
                    if grid_overlay_candidates:
                        self.grid_scalper._last_signal_emitted_at = current_time
            if grid_overlay_source_ids:
                router_candidates = [
                    candidate for candidate in router_candidates if str(candidate.signal_id) not in grid_overlay_source_ids
                ]
            if grid_overlay_candidates:
                router_candidates = []
            if usable_grid_candidates or grid_overlay_candidates:
                candidates = list(usable_grid_candidates) + list(grid_overlay_candidates) + list(router_candidates)
            else:
                candidates = list(router_candidates)
        if not candidates:
            return []

        ranked = _candidate_strategy_pool_rankings(
            symbol_key=self.symbol_key,
            candidates=candidates,
            session_name=session_ctx.session_name,
            row=row,
            regime=regime_obj,
            max_spread_points=self.max_spread_points,
            closed_trades=self.closed_trades,
            current_day_key=self.current_day_key,
            candidate_tier_config=self.candidate_tier_config,
            orchestrator_config=self.orchestrator_config,
        )
        ranking_summary = [dict(entry) for _, entry in ranked]
        winning_reason = _strategy_pool_winner_reason(ranking_summary)
        selected: list[SignalCandidate] = []
        selected_limit = self._selection_limit(ranked=ranked, session_name=session_ctx.session_name)
        for rank, (candidate, entry) in enumerate(ranked[: selected_limit], start=1):
            setattr(candidate, "symbol", self.symbol_key)
            is_grid_candidate = bool((candidate.meta or {}).get("grid_cycle", False))
            atr_field = str((candidate.meta or {}).get("atr_field", "m5_atr_14"))
            atr_for_candidate = max(float(row.get(atr_field, row.get("m5_atr_14", 0.0)) or 0.0), 1e-6)
            lifecycle_atr_multiplier = 10.0
            if self.symbol_key in {"NAS100", "USOIL"}:
                lifecycle_atr_multiplier = 5.0
            if self.symbol_key == "BTCUSD":
                lifecycle_atr_multiplier = 2.5
            if self.symbol_key == "XAUUSD":
                lifecycle_atr_multiplier = 1.8
            lifecycle_atr = atr_for_candidate * lifecycle_atr_multiplier
            entry_price = max(float(row.get("m5_close", 0.0) or 0.0), 0.0)
            idea_structure = TradeIdeaLifecycle.build_structure_snapshot(
                row=row,
                confluence_score=float(candidate.confluence_score),
                entry_price=entry_price,
                atr=lifecycle_atr,
            )
            idea = self.idea_lifecycle.upsert(
                symbol=self.symbol_key,
                setup_type=str(candidate.setup),
                side=str(candidate.side),
                confidence=float(entry.get("rank_score", candidate.score_hint or 0.0) or 0.0),
                confluence_score=float(candidate.confluence_score),
                entry_price=entry_price,
                atr=lifecycle_atr,
                now=current_time,
                structure=idea_structure,
            )
            if is_grid_candidate:
                allow_eval = True
            else:
                allow_eval, _ = self.idea_lifecycle.can_evaluate(
                    idea=idea,
                    now=current_time,
                    session_name=session_ctx.session_name,
                    structure=idea_structure,
                )
            if not allow_eval:
                continue
            self.idea_lifecycle.mark_evaluated(idea=idea, now=current_time, structure=idea_structure)
            move_atr_multiplier = 6.0
            min_repeat_minutes = 240
            strict_repeat_window = True
            if self.symbol_key in {"EURUSD", "GBPUSD"}:
                move_atr_multiplier = 3.0
                min_repeat_minutes = 60
                strict_repeat_window = False
            if self.symbol_key == "USDJPY":
                move_atr_multiplier = 2.8
                min_repeat_minutes = 45 if session_ctx.session_name in {"SYDNEY", "TOKYO"} else 60
                strict_repeat_window = False
            if self.symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD"}:
                move_atr_multiplier = 2.4
                min_repeat_minutes = 25 if session_ctx.session_name in {"SYDNEY", "TOKYO"} else 45
                strict_repeat_window = False
            if self.symbol_key in {"NAS100", "USOIL"}:
                move_atr_multiplier = 3.5
                min_repeat_minutes = 45 if self.symbol_key == "NAS100" else 60
                strict_repeat_window = False
            if self.symbol_key == "BTCUSD":
                move_atr_multiplier = 2.0
                min_repeat_minutes = 10
                strict_repeat_window = False
            if self.symbol_key == "XAUUSD":
                move_atr_multiplier = 2.0
                min_repeat_minutes = 10
                strict_repeat_window = False
            memory_key = "|".join(
                [
                    self.symbol_key,
                    str(candidate.setup),
                    str(candidate.side),
                    str(session_ctx.session_name),
                    str(entry.get("regime_state") or regime_obj.state_label or regime_obj.label),
                ]
            )
            memory = self.replay_signal_memory.get(memory_key)
            move_threshold = atr_for_candidate * move_atr_multiplier
            if (not is_grid_candidate) and memory is not None:
                last_price = float(memory.get("entry_price", 0.0) or 0.0)
                last_time = _maybe_timestamp(memory.get("timestamp"))
                if (
                    last_time is not None
                    and current_time <= last_time.to_pydatetime() + timedelta(hours=8)
                    and current_time <= last_time.to_pydatetime() + timedelta(minutes=min_repeat_minutes)
                    and (strict_repeat_window or abs(entry_price - last_price) < move_threshold)
                ):
                    continue
            exit_profile = _strategy_exit_profile(
                symbol_key=self.symbol_key,
                strategy_key=str(entry.get("strategy_key") or candidate.meta.get("strategy_key") or ""),
                quality_tier=str(entry.get("quality_tier") or "B"),
                exits_config=self.exits_config,
                streak_adjust_mode=str(entry.get("streak_adjust_mode") or "NEUTRAL"),
                session_name=str(session_ctx.session_name),
                regime_state=str(entry.get("regime_state") or regime_obj.state_label or regime_obj.label),
                weekend_mode=bool(entry.get("btc_weekend_mode", False)),
            )
            rr_target_band = str(exit_profile.get("approved_rr_target", "2.0-2.3") or "2.0-2.3")
            rr_midpoint = _rr_band_midpoint(rr_target_band, default=max(float(candidate.tp_r or 2.0), 1.5))
            partials = list(exit_profile.get("partials") or [])
            partial_1 = partials[0] if len(partials) >= 1 else {}
            partial_2 = partials[1] if len(partials) >= 2 else {}
            candidate.tp_r = max(float(candidate.tp_r or 0.0), rr_midpoint)
            candidate.meta["approved_rr_target"] = rr_target_band
            candidate.meta["breakeven_trigger_r"] = float(exit_profile.get("breakeven_trigger_r", 0.5) or 0.5)
            candidate.meta["trail_activation_r"] = float(exit_profile.get("trail_activation_r", 1.0) or 1.0)
            candidate.meta["trail_atr"] = float(exit_profile.get("trail_atr", 1.0) or 1.0)
            candidate.meta["basket_take_profit_r"] = float(exit_profile.get("basket_take_profit_r", 0.0) or 0.0)
            candidate.meta["be_buffer_r"] = float(exit_profile.get("be_buffer_r", 0.05) or 0.05)
            candidate.meta["min_profit_protection_r"] = float(exit_profile.get("min_profit_protection_r", 0.0) or 0.0)
            candidate.meta["trail_backoff_r"] = float(exit_profile.get("trail_backoff_r", 0.55) or 0.55)
            candidate.meta["trail_requires_partial1"] = bool(exit_profile.get("trail_requires_partial1", False))
            candidate.meta["no_progress_bars"] = int(exit_profile.get("no_progress_bars", 0) or 0)
            candidate.meta["no_progress_mfe_r"] = float(exit_profile.get("no_progress_mfe_r", 0.0) or 0.0)
            candidate.meta["early_invalidation_r"] = float(exit_profile.get("early_invalidation_r", -1.0) or -1.0)
            candidate.meta["time_stop_bars"] = int(exit_profile.get("time_stop_bars", 0) or 0)
            candidate.meta["time_stop_max_r"] = float(exit_profile.get("time_stop_max_r", 0.0) or 0.0)
            candidate.meta["partial1_r"] = float(partial_1.get("triggerR", 0.9) or 0.9)
            candidate.meta["partial1_fraction"] = float(partial_1.get("closeFraction", 0.2) or 0.2)
            candidate.meta["partial2_r"] = float(partial_2.get("triggerR", max(candidate.meta["partial1_r"], 1.8)) or max(candidate.meta["partial1_r"], 1.8))
            candidate.meta["partial2_fraction"] = float(partial_2.get("closeFraction", 0.2) or 0.2)
            candidate.meta["strategy_pool_ranking"] = ranking_summary
            candidate.meta["strategy_pool_winner"] = str(ranking_summary[0].get("strategy_key") or "")
            candidate.meta["winning_strategy_reason"] = str(winning_reason)
            candidate.meta["strategy_key"] = str(entry.get("strategy_key") or candidate.meta.get("strategy_key") or "")
            candidate.meta["strategy_state"] = str(entry.get("strategy_state") or candidate.meta.get("strategy_state") or "NORMAL")
            candidate.meta["strategy_score"] = float(entry.get("strategy_score", 0.0) or 0.0)
            candidate.meta["rank_score"] = float(entry.get("rank_score", 0.0) or 0.0)
            candidate.meta["session_name"] = str(session_ctx.session_name)
            candidate.meta["regime_state"] = str(entry.get("regime_state") or regime_obj.state_label or regime_obj.label)
            candidate.meta["regime_fit"] = float(entry.get("regime_fit", 0.0) or 0.0)
            candidate.meta["entry_timing_score"] = float(entry.get("entry_timing_score", 0.0) or 0.0)
            candidate.meta["structure_cleanliness_score"] = float(entry.get("structure_cleanliness_score", 0.0) or 0.0)
            candidate.meta["execution_quality_fit"] = float(entry.get("execution_quality_fit", 0.0) or 0.0)
            candidate.meta["strategy_pool_rank"] = int(rank)
            reason_code = str(entry.get("verified_reason_code") or "")
            self.approved_reason_counts[reason_code or "combined_strategy_score"] += 1
            self.quality_tier_counts[str(entry.get("quality_tier") or "UNK")] += 1
            self.verified_trade_count += 1
            if bool(entry.get("recycle_session", False)):
                self.recycled_trade_count += 1
            self.verification_entries.append(
                {
                    "window_id": str(self.window_id),
                    "timestamp": str(row.get("time") or ""),
                    "signal_id": str(candidate.signal_id),
                    "symbol": str(self.symbol_key),
                    "session": str(session_ctx.session_name),
                    "regime": str(entry.get("regime_state") or regime_obj.state_label or regime_obj.label),
                    "strategy_key": str(entry.get("strategy_key") or ""),
                    "setup_family": str(candidate.strategy_family),
                    "quality_tier": str(entry.get("quality_tier") or ""),
                    "tier_size_multiplier": float(entry.get("tier_size_multiplier", 1.0) or 1.0),
                    "recycle_session": bool(entry.get("recycle_session", False)),
                    "recycle_origin_session": str(entry.get("recycle_origin_session") or ""),
                    "family_rotation_penalty": float(entry.get("family_rotation_penalty", 0.0) or 0.0),
                    "delta_proxy_score": float(entry.get("delta_proxy_score", 0.0) or 0.0),
                    "compression_state": str(entry.get("compression_proxy_state") or ""),
                    "compression_expansion_score": float(entry.get("compression_expansion_score", 0.0) or 0.0),
                    "transition_momentum": float(entry.get("transition_momentum", 0.0) or 0.0),
                    "velocity_decay": float(entry.get("velocity_decay", 1.0) or 1.0),
                    "velocity_decay_score_penalty": float(entry.get("velocity_decay_score_penalty", 0.0) or 0.0),
                    "velocity_trades_per_10_bars": float(entry.get("velocity_trades_per_10_bars", 0.0) or 0.0),
                    "velocity": float(entry.get("velocity_trades_per_10_bars", 0.0) or 0.0),
                    "correlation_penalty": float(entry.get("correlation_penalty", 0.0) or 0.0),
                    "btc_weekend_mode": bool(entry.get("btc_weekend_mode", False)),
                    "btc_velocity_decay": float(entry.get("btc_velocity_decay", 1.0) or 1.0),
                    "session_loosen_factor": float(entry.get("session_loosen_factor", 1.0) or 1.0),
                    "equity_momentum_mode": str(entry.get("equity_momentum_mode") or "NEUTRAL"),
                    "streak_adjust_mode": str(entry.get("streak_adjust_mode") or "NEUTRAL"),
                    "manager_reason": str(entry.get("manager_reason") or ""),
                    "final_score": float(entry.get("strategy_score", 0.0) or 0.0),
                    "final_score_reason": str(entry.get("verified_reason_text") or winning_reason),
                    "verified_reason_code": reason_code,
                    "verified_reason_text": str(entry.get("verified_reason_text") or ""),
                    "approved_rr_target": rr_target_band,
                    "dynamic_rr_band": rr_target_band,
                    "actual_rr_achieved": None,
                }
            )
            selected.append(candidate)
            self.idea_lifecycle.mark_trade_sent(idea=idea, now=current_time)
            if not is_grid_candidate:
                self.replay_signal_memory[memory_key] = {
                    "entry_price": float(entry_price),
                    "timestamp": pd.Timestamp(current_time).isoformat(),
                }
        return selected


def _load_frames(runtime: dict[str, Any], configured_symbol: str) -> dict[str, pd.DataFrame]:
    market_data = runtime["market_data"]
    resolved_symbols: dict[str, str] = runtime["resolved_symbols"]
    resolved_symbol = resolved_symbols[configured_symbol]
    counts = {"M1": 12000, "M5": 20000, "M15": 10000, "H1": 4000, "H4": 2000}
    frames: dict[str, pd.DataFrame] = {}
    for timeframe, count in counts.items():
        cached = market_data.load_cached(resolved_symbol, timeframe)
        cached_ok = cached is not None and len(cached) > 100
        preferred = cached if cached_ok else None
        # Replay/backtest evaluation must prefer stable cached history over
        # live refresh attempts. Some providers only return a short recent
        # slice, which can overwrite a deeper cached parquet and collapse the
        # next replay pass into an "insufficient_data" false negative.
        if preferred is not None:
            frames[timeframe] = preferred
            continue
        try:
            fetched = market_data.fetch(resolved_symbol, timeframe, count)
        except Exception:
            fetched = None
        if fetched is not None and len(fetched) > 100:
            preferred = fetched
        if preferred is not None and len(preferred) > 100:
            frames[timeframe] = preferred
        elif timeframe in {"M1", "H4"}:
            continue
    return frames


def _trade_rows(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for trade in metrics.get("trades", []):
        row = dict(trade)
        row["mae_r"] = _safe_float(row.get("mae_r"), 0.0)
        row["mfe_r"] = _safe_float(row.get("mfe_r"), 0.0)
        duration_minutes = _safe_float(row.get("duration_minutes"), 0.0)
        if duration_minutes <= 0.0:
            try:
                opened_at = _normalize_timestamp(row.get("opened_at"))
                closed_at = _normalize_timestamp(row.get("closed_at"))
                duration_minutes = max(0.0, (closed_at - opened_at).total_seconds() / 60.0)
            except Exception:
                duration_minutes = 0.0
        row["duration_minutes"] = duration_minutes
        rows.append(row)
    return rows


def _finalize_verification_entries(
    entries: list[dict[str, Any]],
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    trade_index = {str(row.get("signal_id") or ""): row for row in rows if str(row.get("signal_id") or "").strip()}
    finalized: list[dict[str, Any]] = []
    for entry in entries:
        trade = trade_index.get(str(entry.get("signal_id") or ""))
        payload = dict(entry)
        payload["actual_rr_achieved"] = _trade_actual_rr_achieved(trade or {})
        finalized.append(payload)
    return finalized


def _verification_report(entries: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: Counter[str] = Counter()
    ancestry_counts: Counter[str] = Counter()
    quality_tier_counts: Counter[str] = Counter()
    rr_distribution: Counter[str] = Counter()
    recycled = 0
    velocity_decay_counts: Counter[str] = Counter()
    for entry in entries:
        reason_code = str(entry.get("verified_reason_code") or "combined_strategy_score")
        reason_counts[reason_code] += 1
        quality_tier_counts[str(entry.get("quality_tier") or "UNK")] += 1
        rr_distribution[str(entry.get("approved_rr_target") or "")] += 1
        if bool(entry.get("recycle_session")):
            recycled += 1
        velocity_active = float(entry.get("velocity_decay", 1.0) or 1.0) < 0.999
        velocity_decay_counts["active" if velocity_active else "inactive"] += 1
        ancestry_counts[
            json.dumps(
                {
                    "reason": reason_code,
                    "recycle_session": bool(entry.get("recycle_session")),
                    "velocity_decay": velocity_active,
                    "transition_momentum": bool(float(entry.get("transition_momentum", 0.0) or 0.0) > 0.0),
                    "correlation_penalty": bool(float(entry.get("correlation_penalty", 0.0) or 0.0) > 0.0),
                    "streak_adjust_mode": str(entry.get("streak_adjust_mode") or "NEUTRAL"),
                    "btc_weekend_mode": bool(entry.get("btc_weekend_mode", False)),
                },
                sort_keys=True,
            )
        ] += 1
    return {
        "verified_trade_count": int(len(entries)),
        "approved_reason_counts": dict(reason_counts),
        "top_20_approved_reasons": [
            {"reason": reason, "count": count}
            for reason, count in reason_counts.most_common(20)
        ],
        "top_20_approved_reasons_with_ancestry": [
            {**json.loads(raw), "count": count}
            for raw, count in ancestry_counts.most_common(20)
        ],
        "recycled_trade_count": int(recycled),
        "quality_tier_counts": dict(quality_tier_counts),
        "rr_distribution": dict(rr_distribution),
        "velocity_decay_counts": dict(velocity_decay_counts),
        "verification_by_symbol": _verification_group_report(entries, ("symbol",), min_entries=1),
        "verification_by_symbol_strategy": _verification_group_report(entries, ("symbol", "strategy_key"), min_entries=1),
        "verification_by_symbol_session": _verification_group_report(entries, ("symbol", "session"), min_entries=1),
        "verification_by_symbol_strategy_session": _verification_group_report(entries, ("symbol", "strategy_key", "session"), min_entries=1),
    }


def _verification_group_report(
    entries: list[dict[str, Any]],
    keys: tuple[str, ...],
    *,
    min_entries: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[tuple(str(entry.get(key, "") or "") for key in keys)].append(entry)
    report: list[dict[str, Any]] = []
    for group_key, group_entries in grouped.items():
        if len(group_entries) < min_entries:
            continue
        reason_counts: Counter[str] = Counter(
            str(item.get("verified_reason_code") or "combined_strategy_score")
            for item in group_entries
        )
        quality_counts: Counter[str] = Counter(
            str(item.get("quality_tier") or "UNK")
            for item in group_entries
        )
        rr_midpoints = [
            _rr_band_midpoint(str(item.get("approved_rr_target") or ""), 0.0)
            for item in group_entries
            if str(item.get("approved_rr_target") or "").strip()
        ]
        actual_rr_values = [
            _safe_float(item.get("actual_rr_achieved"), 0.0)
            for item in group_entries
            if item.get("actual_rr_achieved") is not None
        ]
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(
            {
                "verified_candidates": int(len(group_entries)),
                "recycled_trade_count": int(sum(1 for item in group_entries if bool(item.get("recycle_session")))),
                "avg_rr_target_midpoint": float(_mean(rr_midpoints)) if rr_midpoints else 0.0,
                "avg_actual_rr_achieved": float(_mean(actual_rr_values)) if actual_rr_values else 0.0,
                "top_reasons": [
                    {"reason": reason, "count": count}
                    for reason, count in reason_counts.most_common(5)
                ],
                "quality_tier_counts": dict(quality_counts),
            }
        )
        report.append(entry)
    report.sort(
        key=lambda item: (
            -int(item.get("verified_candidates", 0) or 0),
            -float(item.get("avg_rr_target_midpoint", 0.0) or 0.0),
        )
    )
    return report


def _by_pair_rr(rows: list[dict[str, Any]], *, min_trades: int) -> list[dict[str, Any]]:
    return [
        {
            "symbol": str(item.get("symbol") or ""),
            "trades": float(item.get("trades", 0.0) or 0.0),
            "win_rate": float(item.get("win_rate", 0.0) or 0.0),
            "winner_loss_ratio": float(item.get("winner_loss_ratio", 0.0) or 0.0),
            "avg_win_r": float(item.get("avg_win_r", 0.0) or 0.0),
            "avg_loss_r": float(item.get("avg_loss_r", 0.0) or 0.0),
            "expectancy_r": float(item.get("expectancy_r", 0.0) or 0.0),
            "avg_mae_r": float(item.get("avg_mae_r", 0.0) or 0.0),
            "avg_mfe_r": float(item.get("avg_mfe_r", 0.0) or 0.0),
            "capture_efficiency": float(item.get("capture_efficiency", 0.0) or 0.0),
            "top_exit_reasons": list(item.get("top_exit_reasons") or []),
        }
        for item in _group_report(rows, ("symbol",), min_trades=min_trades)
    ]


def _aggregate_overall_suite(reports: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_keys = (
        "trades",
        "win_rate",
        "profit_factor",
        "expectancy_r",
        "avg_r",
        "avg_win_r",
        "avg_loss_r",
        "winner_loss_ratio",
        "avg_mae_r",
        "avg_mfe_r",
        "capture_efficiency",
        "avg_duration_minutes",
    )
    mean_payload: dict[str, Any] = {}
    std_payload: dict[str, Any] = {}
    for key in metric_keys:
        values = [float((report.get("overall") or {}).get(key, 0.0) or 0.0) for report in reports]
        mean_payload[key] = _mean(values)
        std_payload[key] = _stddev(values)
    mean_payload["window_count"] = int(len(reports))
    std_payload["window_count"] = int(len(reports))
    return mean_payload, std_payload


def _aggregate_group_suite(
    reports: list[dict[str, Any]],
    *,
    group_key: str,
    report_key: str = "by_pair",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_keys = (
        "trades",
        "win_rate",
        "profit_factor",
        "expectancy_r",
        "avg_r",
        "avg_win_r",
        "avg_loss_r",
        "winner_loss_ratio",
        "avg_mae_r",
        "avg_mfe_r",
        "capture_efficiency",
        "avg_duration_minutes",
    )
    key_space = sorted(
        {
            str(entry.get(group_key) or "")
            for report in reports
            for entry in list(report.get(report_key) or [])
            if str(entry.get(group_key) or "").strip()
        }
    )
    mean_rows: list[dict[str, Any]] = []
    std_rows: list[dict[str, Any]] = []
    for group_value in key_space:
        matches_by_window: list[dict[str, Any] | None] = []
        for report in reports:
            match = next(
                (entry for entry in list(report.get(report_key) or []) if str(entry.get(group_key) or "") == group_value),
                None,
            )
            matches_by_window.append(match)
        active = [entry for entry in matches_by_window if entry and float(entry.get("trades", 0.0) or 0.0) > 0.0]
        if not active:
            continue
        mean_row: dict[str, Any] = {group_key: group_value, "active_windows": int(len(active))}
        std_row: dict[str, Any] = {group_key: group_value, "active_windows": int(len(active))}
        trade_values = [float(entry.get("trades", 0.0) or 0.0) if entry else 0.0 for entry in matches_by_window]
        mean_row["trades"] = _mean(trade_values)
        std_row["trades"] = _stddev(trade_values)
        for key in metric_keys:
            if key == "trades":
                continue
            values = [float(entry.get(key, 0.0) or 0.0) for entry in active]
            mean_row[key] = _mean(values)
            std_row[key] = _stddev(values)
        mean_rows.append(mean_row)
        std_rows.append(std_row)
    mean_rows.sort(
        key=lambda item: (
            float(item.get("expectancy_r", 0.0) or 0.0),
            float(item.get("profit_factor", 0.0) or 0.0),
            -float(item.get("trades", 0.0) or 0.0),
        )
    )
    std_rows.sort(key=lambda item: str(item.get(group_key) or ""))
    return mean_rows, std_rows


def _aggregate_slice_suite(reports: list[dict[str, Any]], key: str) -> tuple[dict[str, Any], dict[str, Any]]:
    metric_keys = (
        "trades",
        "win_rate",
        "profit_factor",
        "expectancy_r",
        "avg_r",
        "avg_win_r",
        "avg_loss_r",
        "winner_loss_ratio",
        "avg_mae_r",
        "avg_mfe_r",
        "capture_efficiency",
        "avg_duration_minutes",
    )
    active_reports = [
        report
        for report in reports
        if float(((report.get(key) or {}).get("trades", 0.0) or 0.0)) > 0.0
    ]
    source_reports = active_reports or reports
    mean_payload: dict[str, Any] = {}
    std_payload: dict[str, Any] = {}
    for metric in metric_keys:
        values = [float(((report.get(key) or {}).get(metric, 0.0) or 0.0)) for report in source_reports]
        mean_payload[metric] = _mean(values)
        std_payload[metric] = _stddev(values)
    mean_payload["active_windows"] = int(len(active_reports))
    std_payload["active_windows"] = int(len(active_reports))
    projection_keys = ("days", "avg_trades_per_day", "median_trades_per_day", "max_trades_per_day", "avg_daily_win_rate")
    daily_mean: dict[str, Any] = {}
    daily_std: dict[str, Any] = {}
    for metric in projection_keys:
        values = [
            float((((report.get(key) or {}).get("daily_projection") or {}).get(metric, 0.0) or 0.0))
            for report in source_reports
        ]
        daily_mean[metric] = _mean(values)
        daily_std[metric] = _stddev(values)
    mean_payload["daily_projection"] = daily_mean
    std_payload["daily_projection"] = daily_std
    mean_payload["by_session"] = _aggregate_group_suite(
        [{"by_pair": list((report.get(key) or {}).get("by_session") or [])} for report in source_reports],
        group_key="session_name",
        report_key="by_pair",
    )[0]
    std_payload["by_session"] = _aggregate_group_suite(
        [{"by_pair": list((report.get(key) or {}).get("by_session") or [])} for report in source_reports],
        group_key="session_name",
        report_key="by_pair",
    )[1]
    burst_reports = [dict((report.get(key) or {}).get("burst_density") or {}) for report in source_reports]
    mean_payload["burst_density"] = {
        "window_minutes": int(next((report.get("window_minutes") for report in burst_reports if report.get("window_minutes")), 10) or 10),
        "active_windows": int(_mean([float(report.get("active_windows", 0) or 0) for report in burst_reports])),
        "avg_actions_per_active_window": float(_mean([float(report.get("avg_actions_per_active_window", 0.0) or 0.0) for report in burst_reports])),
        "max_actions_per_window": float(_mean([float(report.get("max_actions_per_window", 0.0) or 0.0) for report in burst_reports])),
    }
    std_payload["burst_density"] = {
        "window_minutes": int(mean_payload["burst_density"]["window_minutes"]),
        "active_windows": int(_mean([float(report.get("active_windows", 0) or 0) for report in burst_reports])),
        "avg_actions_per_active_window": float(_stddev([float(report.get("avg_actions_per_active_window", 0.0) or 0.0) for report in burst_reports])),
        "max_actions_per_window": float(_stddev([float(report.get("max_actions_per_window", 0.0) or 0.0) for report in burst_reports])),
    }
    return mean_payload, std_payload


def _protected_regressions(
    runtime: dict[str, Any],
    suite_by_pair_mean: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_path = Path(
        runtime["settings"].resolve_path_value("data/replay_baselines/candidate3_frozen.json")
    )
    if not baseline_path.exists():
        return []
    try:
        baseline_payload = json.loads(baseline_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    protected_pairs = {"GBPUSD", "AUDJPY", "XAUUSD"}
    baseline_map = {
        str(entry.get("symbol") or ""): entry
        for entry in list(baseline_payload.get("by_pair") or [])
        if str(entry.get("symbol") or "") in protected_pairs
    }
    current_map = {
        str(entry.get("symbol") or ""): entry
        for entry in suite_by_pair_mean
        if str(entry.get("symbol") or "") in protected_pairs
    }
    regressions: list[dict[str, Any]] = []
    for symbol in sorted(protected_pairs):
        baseline = baseline_map.get(symbol)
        current = current_map.get(symbol)
        if not baseline or not current:
            continue
        if (
            float(current.get("win_rate", 0.0) or 0.0) + 1e-9 < float(baseline.get("win_rate", 0.0) or 0.0)
            or float(current.get("profit_factor", 0.0) or 0.0) + 1e-9 < float(baseline.get("profit_factor", 0.0) or 0.0)
        ):
            regressions.append(
                {
                    "symbol": symbol,
                    "baseline_win_rate": float(baseline.get("win_rate", 0.0) or 0.0),
                    "current_win_rate": float(current.get("win_rate", 0.0) or 0.0),
                    "baseline_profit_factor": float(baseline.get("profit_factor", 0.0) or 0.0),
                    "current_profit_factor": float(current.get("profit_factor", 0.0) or 0.0),
                }
            )
    return regressions


def _run_replay_mode(
    *,
    runtime: dict[str, Any],
    symbols: list[str],
    backtester: Backtester,
    top_n: int,
    min_trades: int,
    weekend_mode_only: bool,
    replay_closed_trades: list[dict[str, Any]],
    replay_day_key: str,
    verification_log_path: Path | None = None,
    window_spec: dict[str, Any] | None = None,
    reset_verification_log: bool = False,
) -> dict[str, Any]:
    feature_engineer = runtime["feature_engineer"]
    strategy_engines: dict[str, StrategyEngine] = runtime["strategy_engines"]
    settings = runtime["settings"]
    idea_lifecycle_config = settings.section("idea_lifecycle") if isinstance(settings.raw.get("idea_lifecycle"), dict) else {}
    report: dict[str, Any] = {
        "symbols": list(symbols),
        "weekend_mode_only": bool(weekend_mode_only),
        "window_id": str((window_spec or {}).get("window_id") or ""),
        "by_symbol": {},
    }
    all_rows: list[dict[str, Any]] = []
    verification_entries: list[dict[str, Any]] = []

    for configured_symbol in symbols:
        if weekend_mode_only and configured_symbol != "BTCUSD":
            report["by_symbol"][configured_symbol] = {"skipped": "weekend_mode_only"}
            continue
        frames = _load_frames(runtime, configured_symbol)
        if "M5" not in frames or "M15" not in frames or "H1" not in frames:
            report["by_symbol"][configured_symbol] = {"error": "insufficient_data"}
            continue

        features = feature_engineer.build(
            frames.get("M1"),
            frames["M5"],
            frames["M15"],
            frames["H1"],
            frames.get("H4"),
        )
        features = _windowed_features(features, window_spec)
        features = _market_open_features(features, configured_symbol)
        symbol_overrides = _symbol_backtest_overrides(configured_symbol)
        features = features.copy()
        features["instrument_point_size"] = float(symbol_overrides.get("instrument_point_size", 0.01))
        features["backtest_spread_points"] = float(symbol_overrides.get("spread_points", backtester.spread_points))
        if weekend_mode_only:
            weekend_mask = features["time"].apply(lambda value: bool(is_weekend_market_mode(_normalize_timestamp(value).to_pydatetime())))
            features = features.loc[weekend_mask].reset_index(drop=True)
            if features.empty or len(features) < 50:
                report["by_symbol"][configured_symbol] = {"error": "insufficient_weekend_data"}
                continue

        def _build_symbol_backtester(
            *,
            strict_plausibility: bool,
            whitelist_high_win_rate: bool,
        ) -> tuple[RouterStrategyEngineAdapter, Backtester]:
            adapter = RouterStrategyEngineAdapter(
                symbol_key=configured_symbol,
                strategy_engine=strategy_engines[configured_symbol],
                strategy_router=runtime["strategy_router"],
                session_profile=runtime["session_profile"],
                regime_detector=runtime["regime_detector"],
                max_spread_points=float(runtime["settings"].section("risk").get("max_spread_points", 60)),
                top_n=top_n,
                closed_trades=replay_closed_trades,
                current_day_key=replay_day_key,
                candidate_tier_config=runtime.get("candidate_tier_config", {}),
                orchestrator_config=runtime.get("bridge_orchestrator_config", {}),
                grid_scalper=runtime.get("grid_scalper"),
                exits_config=runtime["settings"].section("exits"),
                window_id=str((window_spec or {}).get("window_id") or ""),
                idea_lifecycle_config=idea_lifecycle_config,
            )
            symbol_backtester = replace(
                backtester,
                strategy_engine=adapter,
                spread_points=float(symbol_overrides.get("spread_points", backtester.spread_points)),
                slippage_points=float(symbol_overrides.get("slippage_points", backtester.slippage_points)),
                contract_size=float(symbol_overrides.get("contract_size", backtester.contract_size)),
                commission_per_lot=float(symbol_overrides.get("commission_per_lot", backtester.commission_per_lot)),
                be_trigger_r=float(symbol_overrides.get("be_trigger_r", backtester.be_trigger_r)),
                trail_start_r=float(symbol_overrides.get("trail_start_r", backtester.trail_start_r)),
                partial1_r=float(symbol_overrides.get("partial1_r", backtester.partial1_r)),
                partial1_fraction=float(symbol_overrides.get("partial1_fraction", backtester.partial1_fraction)),
                partial2_r=float(symbol_overrides.get("partial2_r", backtester.partial2_r)),
                partial2_fraction=float(symbol_overrides.get("partial2_fraction", backtester.partial2_fraction)),
                close_on_signal_flip=bool(symbol_overrides.get("close_on_signal_flip", backtester.close_on_signal_flip)),
                strict_plausibility=bool(strict_plausibility),
                whitelist_high_win_rate=bool(whitelist_high_win_rate),
            )
            return adapter, symbol_backtester

        adapter, symbol_backtester = _build_symbol_backtester(
            strict_plausibility=bool(backtester.strict_plausibility),
            whitelist_high_win_rate=bool(backtester.whitelist_high_win_rate),
        )
        plausibility_warning = ""
        try:
            metrics = symbol_backtester.run(features)
        except ValueError as exc:
            if "suspicious_backtest_win_rate" not in str(exc):
                report["by_symbol"][configured_symbol] = {
                    "error": f"backtest_failed:{type(exc).__name__}",
                    "message": str(exc),
                    "feature_rows": int(len(features)),
                    "window_id": str((window_spec or {}).get("window_id") or ""),
                }
                continue
            plausibility_warning = str(exc)
            adapter, relaxed_backtester = _build_symbol_backtester(
                strict_plausibility=False,
                whitelist_high_win_rate=True,
            )
            metrics = relaxed_backtester.run(features)
        except Exception as exc:
            report["by_symbol"][configured_symbol] = {
                "error": f"backtest_failed:{type(exc).__name__}",
                "message": str(exc),
                "feature_rows": int(len(features)),
                "window_id": str((window_spec or {}).get("window_id") or ""),
            }
            continue
        rows = _trade_rows(metrics)
        report["by_symbol"][configured_symbol] = {
            "trade_count": metrics.get("trade_count", 0),
            "win_rate": metrics.get("win_rate", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
            "expectancy_r": metrics.get("expectancy_r", 0.0),
            "net_r": metrics.get("net_r", 0.0),
            "max_drawdown_r": metrics.get("max_drawdown_r", 0.0),
            "max_consecutive_losses": metrics.get("max_consecutive_losses", 0),
            "events": metrics.get("events", {}),
            "feature_rows": int(len(features)),
            "plausibility_warning": str(plausibility_warning or ""),
            "verified_trade_count": int(adapter.verified_trade_count),
            "approved_reason_counts": dict(adapter.approved_reason_counts),
            "quality_tier_counts": dict(adapter.quality_tier_counts),
            "recycled_trade_count": int(adapter.recycled_trade_count),
        }
        all_rows.extend(rows)
        verification_entries.extend(_finalize_verification_entries(adapter.verification_entries, rows))

    if verification_log_path is not None:
        verification_log_path.parent.mkdir(parents=True, exist_ok=True)
        if reset_verification_log and verification_log_path.exists():
            verification_log_path.unlink()
        for entry in verification_entries:
            _append_candidate_verification_log(verification_log_path, entry)

    report["overall"] = _metrics(all_rows)
    report["by_pair"] = _group_report(all_rows, ("symbol",), min_trades=min_trades)
    report["by_pair_strategy"] = _group_report(all_rows, ("symbol", "strategy_key"), min_trades=min_trades)
    report["by_pair_strategy_session"] = _group_report(all_rows, ("symbol", "strategy_key", "session_name"), min_trades=min_trades)
    report["by_pair_strategy_session_regime"] = _group_report(
        all_rows,
        ("symbol", "strategy_key", "session_name", "regime_state"),
        min_trades=min_trades,
    )
    report["worst_buckets"] = list(report["by_pair_strategy_session_regime"][:20])
    report["exact_blocker_buckets"] = [
        item
        for item in report["by_pair_strategy_session_regime"][:20]
        if float(item.get("expectancy_r", 0.0) or 0.0) < 0.0
    ]
    report["by_pair_rr"] = _by_pair_rr(all_rows, min_trades=min_trades)
    report["btc_weekend_slice"] = _slice_report(all_rows, symbol_key="BTCUSD", weekend_only=True)
    report["xau_grid_pure_slice"] = _slice_report(all_rows, symbol_key="XAUUSD", strategy_key="XAUUSD_ADAPTIVE_M5_GRID")
    report.update(_verification_report(verification_entries))
    return report


def run_replay(
    *,
    symbols: list[str] | None,
    preset: str,
    top_n: int,
    min_trades: int,
    weekend_mode_only: bool = False,
    use_snapshot_path: str = "",
    refresh_snapshot: bool = False,
    xau_profile: str = "",
    proof_mode: str = "",
    xau_prime_session_suite: bool = False,
) -> dict[str, Any]:
    runtime = build_runtime(skip_mt5=True)
    runtime["regime_detector"].persist_history = False
    xau_proof_state = _apply_xau_profile(
        runtime,
        xau_profile=str(xau_profile or ""),
        proof_mode=str(proof_mode or ""),
    )
    settings = runtime["settings"]
    backtester: Backtester = runtime["backtester"]
    configured_symbols: list[str] = runtime["configured_symbols"]
    selected_symbols = [symbol for symbol in configured_symbols if symbol in symbols] if symbols else list(configured_symbols)

    backtest_settings = settings.section("backtest")
    preset_key = preset.strip().lower()
    if preset_key == "frictionless":
        backtester = replace(
            backtester,
            spread_points=0.0,
            slippage_points=0.0,
            commission_per_lot=0.0,
            latency_ms=0,
            strict_plausibility=False,
            whitelist_high_win_rate=True,
        )
    else:
        backtester = replace(
            backtester,
            spread_points=float(backtest_settings["default_spread_points"]),
            slippage_points=float(backtest_settings["default_slippage_points"]),
            commission_per_lot=float(backtest_settings["commission_per_lot"]),
            latency_ms=int(backtest_settings.get("default_latency_ms", 0)),
            strict_plausibility=bool(backtest_settings.get("strict_plausibility", True)),
            max_plausible_win_rate=float(backtest_settings.get("max_plausible_win_rate", 0.85)),
            min_trades_for_plausibility=int(backtest_settings.get("min_trades_for_plausibility", 200)),
            whitelist_high_win_rate=False,
        )

    journal = runtime["journal"]
    moving_baseline_stats = journal.stats(current_equity=100.0)
    moving_closed_trades = journal.closed_trades(250)
    verification_log_path = Path(runtime.get("candidate_verification_log_path") or settings.resolve_path_value("data/candidate_verification.log"))
    effective_use_snapshot_path = str(use_snapshot_path or "").strip()
    if (
        not effective_use_snapshot_path
        and bool(xau_prime_session_suite)
        and "XAUUSD" in {str(symbol).upper() for symbol in selected_symbols}
    ):
        configured_prime_snapshot = str(
            runtime.get("bridge_orchestrator_config", {}).get("xau_prime_session_snapshot_file", "") or ""
        ).strip()
        if configured_prime_snapshot:
            effective_use_snapshot_path = str(settings.resolve_path_value(configured_prime_snapshot))
    frozen_report: dict[str, Any]
    window_reports: dict[str, Any] = {}
    snapshot_suite: dict[str, Any] = {}
    base_prime_snapshot = _load_snapshot_metadata(effective_use_snapshot_path) if effective_use_snapshot_path else {}
    use_curated_xau_prime_suite = bool(
        xau_prime_session_suite
        and "XAUUSD" in {str(symbol).upper() for symbol in selected_symbols}
        and base_prime_snapshot
        and (base_prime_snapshot.get("window_start") or base_prime_snapshot.get("start"))
        and (base_prime_snapshot.get("window_end") or base_prime_snapshot.get("end"))
    )

    if weekend_mode_only:
        snapshot, snapshot_path = _resolve_snapshot(
            runtime,
            use_snapshot_path=str(effective_use_snapshot_path or ""),
            refresh_snapshot=bool(refresh_snapshot),
        )
        frozen_report = _run_replay_mode(
            runtime=runtime,
            symbols=selected_symbols,
            backtester=backtester,
            top_n=top_n,
            min_trades=min_trades,
            weekend_mode_only=bool(weekend_mode_only),
            replay_closed_trades=list(snapshot.get("closed_trades") or []),
            replay_day_key=str(snapshot.get("trading_day_key") or ""),
            verification_log_path=verification_log_path,
            reset_verification_log=True,
        )
        snapshot_suite["window_a"] = {
            "path": str(snapshot_path),
            "snapshot_timestamp": str(snapshot.get("snapshot_timestamp") or ""),
            "trading_day_key": str(snapshot.get("trading_day_key") or ""),
            "window_id": str(snapshot.get("window_id") or "window_a"),
        }
    elif use_curated_xau_prime_suite:
        base_window_spec = {
            "window_id": str(base_prime_snapshot.get("window_id") or "window_a"),
            "start": _maybe_timestamp(base_prime_snapshot.get("window_start") or base_prime_snapshot.get("start")),
            "end": _maybe_timestamp(base_prime_snapshot.get("window_end") or base_prime_snapshot.get("end")),
        }
        base_report = _run_replay_mode(
            runtime=runtime,
            symbols=selected_symbols,
            backtester=backtester,
            top_n=top_n,
            min_trades=min_trades,
            weekend_mode_only=False,
            replay_closed_trades=list(base_prime_snapshot.get("closed_trades") or []),
            replay_day_key=str(base_prime_snapshot.get("trading_day_key") or ""),
            verification_log_path=verification_log_path,
            window_spec=base_window_spec,
            reset_verification_log=True,
        )
        window_reports, snapshot_suite, window_activity_state = _xau_prime_session_window_reports(
            base_report,
            base_snapshot=base_prime_snapshot,
            base_snapshot_path=Path(effective_use_snapshot_path),
            xau_proof_state=xau_proof_state,
        )
        by_pair_payload = []
        xau_pair_payload = dict(base_report.get("xau_grid_pure_slice") or {})
        if xau_pair_payload:
            xau_pair_payload["symbol"] = "XAUUSD"
            by_pair_payload.append(xau_pair_payload)
        frozen_report = {
            "window_reports": window_reports,
            "snapshot_suite": snapshot_suite,
            "suite_mean": {
                "overall": dict(base_report.get("overall") or {}),
                "btc_weekend_slice": dict(base_report.get("btc_weekend_slice") or {}),
                "xau_grid_pure_slice": dict(base_report.get("xau_grid_pure_slice") or {}),
            },
            "suite_stddev": {
                "overall": {},
                "btc_weekend_slice": {},
                "xau_grid_pure_slice": {},
            },
            "suite_by_pair_mean": list(by_pair_payload),
            "suite_by_pair_stddev": {},
            "protected_regressions": _protected_regressions(runtime, list(by_pair_payload)),
            "overall": dict(base_report.get("overall") or {}),
            "by_pair": list(by_pair_payload),
            "weekend_btc_slice": dict(base_report.get("btc_weekend_slice") or {}),
            "xau_grid_pure_slice": dict(base_report.get("xau_grid_pure_slice") or {}),
            "window_activity_state": window_activity_state,
            "window_c_state": {},
            "xau_prime_session_base_replay": {
                "snapshot": {
                    "path": str(effective_use_snapshot_path),
                    "snapshot_timestamp": str(base_prime_snapshot.get("snapshot_timestamp") or ""),
                    "trading_day_key": str(base_prime_snapshot.get("trading_day_key") or ""),
                    "window_id": str(base_prime_snapshot.get("window_id") or ""),
                    "window_start": str(base_prime_snapshot.get("window_start") or ""),
                    "window_end": str(base_prime_snapshot.get("window_end") or ""),
                    "xau_proof_state": dict(xau_proof_state or base_prime_snapshot.get("xau_proof_state") or {}),
                },
                "overall": dict(base_report.get("overall") or {}),
                "xau_grid_pure_slice": dict(base_report.get("xau_grid_pure_slice") or {}),
            },
        }
    else:
        if bool(xau_prime_session_suite) and "XAUUSD" in {str(symbol).upper() for symbol in selected_symbols}:
            window_specs = _prime_session_window_specs(
                _load_snapshot_metadata(effective_use_snapshot_path) if effective_use_snapshot_path else {}
            )
        else:
            window_specs = _derive_window_specs(runtime, selected_symbols)
        if not window_specs:
            raise RuntimeError("Unable to derive replay windows from cached market data.")
        window_sequence: list[dict[str, Any]] = []
        for index, window_spec in enumerate(window_specs):
            snapshot, snapshot_path = _resolve_snapshot(
                runtime,
                use_snapshot_path=str(effective_use_snapshot_path or ""),
                refresh_snapshot=bool(refresh_snapshot),
                window_spec=window_spec,
            )
            report = _run_replay_mode(
                runtime=runtime,
                symbols=selected_symbols,
                backtester=backtester,
                top_n=top_n,
                min_trades=min_trades,
                weekend_mode_only=False,
                replay_closed_trades=list(snapshot.get("closed_trades") or []),
                replay_day_key=str(snapshot.get("trading_day_key") or ""),
                verification_log_path=verification_log_path,
                window_spec=window_spec,
                reset_verification_log=bool(index == 0),
            )
            report["snapshot"] = {
                "path": str(snapshot_path),
                "snapshot_timestamp": str(snapshot.get("snapshot_timestamp") or ""),
                "trading_day_key": str(snapshot.get("trading_day_key") or ""),
                "window_id": str(snapshot.get("window_id") or window_spec.get("window_id") or ""),
                "window_start": str(
                    snapshot.get("window_start")
                    or (window_spec.get("start").isoformat() if hasattr(window_spec.get("start"), "isoformat") else window_spec.get("start") or "")
                ),
                "window_end": str(
                    snapshot.get("window_end")
                    or (window_spec.get("end").isoformat() if hasattr(window_spec.get("end"), "isoformat") else window_spec.get("end") or "")
                ),
                "xau_proof_state": dict(snapshot.get("xau_proof_state") or xau_proof_state or {}),
            }
            window_reports[str(window_spec.get("window_id") or f"window_{index + 1}")] = report
            snapshot_suite[str(window_spec.get("window_id") or f"window_{index + 1}")] = dict(report.get("snapshot") or {})
            window_sequence.append(report)
        overall_mean, overall_stddev = _aggregate_overall_suite(window_sequence)
        by_pair_mean, by_pair_stddev = _aggregate_group_suite(window_sequence, group_key="symbol")
        btc_weekend_mean, btc_weekend_stddev = _aggregate_slice_suite(window_sequence, "btc_weekend_slice")
        xau_grid_mean, xau_grid_stddev = _aggregate_slice_suite(window_sequence, "xau_grid_pure_slice")
        frozen_report = {
            "window_reports": window_reports,
            "snapshot_suite": snapshot_suite,
            "suite_mean": {
                "overall": overall_mean,
                "btc_weekend_slice": btc_weekend_mean,
                "xau_grid_pure_slice": xau_grid_mean,
            },
            "suite_stddev": {
                "overall": overall_stddev,
                "btc_weekend_slice": btc_weekend_stddev,
                "xau_grid_pure_slice": xau_grid_stddev,
            },
            "suite_by_pair_mean": by_pair_mean,
            "suite_by_pair_stddev": by_pair_stddev,
            "protected_regressions": _protected_regressions(runtime, by_pair_mean),
            "overall": overall_mean,
            "by_pair": by_pair_mean,
            "weekend_btc_slice": btc_weekend_mean,
            "xau_grid_pure_slice": xau_grid_mean,
        }
        frozen_report["window_activity_state"] = {
            window_id: {
                "window_id": str(window_id),
                "active": bool(float((report.get("xau_grid_pure_slice") or {}).get("trades", 0.0) or 0.0) > 0.0),
                "reason": "" if float((report.get("xau_grid_pure_slice") or {}).get("trades", 0.0) or 0.0) > 0.0 else "no_xau_grid_trades_in_window",
                "xau_grid_trades": int(float((report.get("xau_grid_pure_slice") or {}).get("trades", 0.0) or 0.0)),
                "window_start": str(((report.get("snapshot") or {}).get("window_start") or "")),
                "window_end": str(((report.get("snapshot") or {}).get("window_end") or "")),
            }
            for window_id, report in window_reports.items()
        }
        frozen_report["window_c_state"] = dict(frozen_report["window_activity_state"].get("window_c") or {})

    moving_report = _run_replay_mode(
        runtime=runtime,
        symbols=selected_symbols,
        backtester=backtester,
        top_n=top_n,
        min_trades=min_trades,
        weekend_mode_only=bool(weekend_mode_only),
        replay_closed_trades=list(moving_closed_trades or []),
        replay_day_key=str(getattr(moving_baseline_stats, "trading_day_key", "") or ""),
        verification_log_path=None,
    )

    payload = dict(frozen_report)
    payload["preset"] = preset_key
    payload["symbols"] = list(selected_symbols)
    payload["weekend_mode_only"] = bool(weekend_mode_only)
    payload["xau_proof_state"] = dict(xau_proof_state or {})
    payload["proof_surface"] = {
        "mode": "local_replay",
        "xau_prime_session_suite": bool(xau_prime_session_suite),
        "weekend_mode_only": bool(weekend_mode_only),
        "snapshot_source": str(effective_use_snapshot_path or ""),
    }
    if weekend_mode_only:
        payload["snapshot"] = dict(snapshot_suite.get("window_a") or {})
    else:
        payload["snapshot"] = dict(next(iter(snapshot_suite.values())) if snapshot_suite else {})
        payload["snapshot_suite"] = snapshot_suite
        payload["window_activity_state"] = dict(frozen_report.get("window_activity_state") or {})
        payload["window_c_state"] = dict(frozen_report.get("window_c_state") or {})
    payload["frozen_replay_overall"] = dict(frozen_report.get("overall") or {})
    payload["moving_replay_overall"] = dict(moving_report.get("overall") or {})
    payload["moving_replay_context"] = {
        "trading_day_key": str(getattr(moving_baseline_stats, "trading_day_key", "") or ""),
        "closed_trade_sample_size": int(len(list(moving_closed_trades or []))),
    }
    payload["overall"] = dict(frozen_report.get("overall") or {})
    if weekend_mode_only:
        payload["weekend_btc_slice"] = dict(frozen_report)
    else:
        payload["weekend_btc_slice"] = dict(frozen_report.get("weekend_btc_slice") or {})
        payload["xau_grid_pure_slice"] = dict(frozen_report.get("xau_grid_pure_slice") or {})
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay cached candles through the live strategy router and score by pair/strategy/session/regime.")
    parser.add_argument("--symbols", default="", help="Comma-separated configured symbols to replay. Default: all configured symbols.")
    parser.add_argument("--preset", choices=("realistic", "frictionless"), default="realistic")
    parser.add_argument("--top-n", type=int, default=1, help="How many ranked router candidates per bar to feed into the backtester.")
    parser.add_argument("--min-trades", type=int, default=2)
    parser.add_argument("--weekend-mode-only", action="store_true", help="Replay only weekend-mode bars. Intended for BTC weekend proof.")
    parser.add_argument("--use-snapshot", default="", help="Use an existing frozen replay snapshot JSON path.")
    parser.add_argument("--refresh-snapshot", action="store_true", help="Refresh the replay snapshot before running.")
    parser.add_argument("--xau-profile", default="", help="Force the XAU grid profile for replay, e.g. checkpoint or density_branch.")
    parser.add_argument("--proof-mode", default="", help="Label the replay proof mode, defaults to the active XAU profile.")
    parser.add_argument("--xau-prime-session-suite", action="store_true", help="Replay XAU using fixed LONDON/OVERLAP/NEW_YORK session windows.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    symbols = [item.strip().upper() for item in str(args.symbols or "").split(",") if item.strip()]
    payload = run_replay(
        symbols=symbols or None,
        preset=args.preset,
        top_n=max(1, int(args.top_n)),
        min_trades=max(1, int(args.min_trades)),
        weekend_mode_only=bool(args.weekend_mode_only),
        use_snapshot_path=str(args.use_snapshot or ""),
        refresh_snapshot=bool(args.refresh_snapshot),
        xau_profile=str(args.xau_profile or ""),
        proof_mode=str(args.proof_mode or ""),
        xau_prime_session_suite=bool(args.xau_prime_session_suite),
    )
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print("Frozen replay overall:")
    print(json.dumps(payload.get("frozen_replay_overall", {}), indent=2, sort_keys=True))
    print("\nMoving replay overall:")
    print(json.dumps(payload.get("moving_replay_overall", {}), indent=2, sort_keys=True))
    print("\nWorst frozen pair+strategy+session+regime:")
    for item in payload.get("by_pair_strategy_session_regime", [])[:20]:
        print(
            f"{item['symbol']} | {item['strategy_key']} | {item['session_name']} | {item['regime_state']}: "
            f"trades={int(item['trades'])} win_rate={item['win_rate']:.3f} "
            f"pf={item['profit_factor']:.3f} expectancy={item['expectancy_r']:.3f} rr={item['winner_loss_ratio']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
