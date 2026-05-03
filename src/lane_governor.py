from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from src.utils import clamp, utc_now

UTC = timezone.utc
_WINDOW_DEFS: tuple[tuple[str, int], ...] = (
    ("1d", 1),
    ("3d", 3),
    ("7d", 7),
    ("30d", 30),
)
_LOSS_TAGS: tuple[str, ...] = (
    "bad_entry",
    "bad_regime",
    "bad_spread",
    "bad_timing",
    "bad_management",
    "bad_exit",
    "news_misread",
    "execution_miss",
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _safe_str(value: Any) -> str:
    return str(value or "").strip()


def _parse_time(row: dict[str, Any]) -> datetime | None:
    for key in ("closed_at", "timestamp_utc", "timestamp", "opened_at"):
        raw = _safe_str(row.get(key))
        if not raw:
            continue
        try:
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            continue
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt.astimezone(UTC)
    return None


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return float((ordered[mid - 1] + ordered[mid]) / 2.0)
    return float(ordered[mid])


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(item) for item in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pct = clamp(float(percentile), 0.0, 1.0)
    index = int(round((len(ordered) - 1) * pct))
    return float(ordered[index])


def _window_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    pnl_values = [_safe_float(row.get("pnl_r"), 0.0) for row in rows]
    wins = [value for value in pnl_values if value >= 0.0]
    losses = [abs(value) for value in pnl_values if value < 0.0]
    spread_values = [_safe_float(row.get("spread_points"), 0.0) for row in rows if row.get("spread_points") is not None]
    slippage_values = [
        _safe_float(
            row.get("slippage_points"),
            row.get("slippage"),
        )
        for row in rows
        if row.get("slippage_points") is not None or row.get("slippage") is not None
    ]
    mfe_values = [_safe_float(row.get("mfe_r"), 0.0) for row in rows if row.get("mfe_r") is not None]
    mae_values = [_safe_float(row.get("mae_r"), 0.0) for row in rows if row.get("mae_r") is not None]
    expectancy_r = _mean(pnl_values)
    win_rate = (len(wins) / len(pnl_values)) if pnl_values else 0.0
    spread_avg = _mean(spread_values)
    slippage_avg = _mean(slippage_values)
    spread_adjusted_edge = expectancy_r - min(0.30, spread_avg * 0.0025) - min(0.22, slippage_avg * 0.010)
    edge_score = clamp(
        0.50
        + (expectancy_r * 0.40)
        + ((win_rate - 0.50) * 0.55)
        - min(0.16, spread_avg * 0.0016)
        - min(0.12, slippage_avg * 0.009),
        0.0,
        1.0,
    )
    return {
        "trade_count": int(len(rows)),
        "win_rate": float(win_rate),
        "expectancy_r": float(expectancy_r),
        "pnl_r_total": float(sum(pnl_values)),
        "profit_factor": float((sum(wins) / sum(losses)) if losses else (float("inf") if wins else 0.0)),
        "winner_loss_ratio": float((_mean(wins) / _mean(losses)) if wins and losses else (_mean(wins) if wins else 0.0)),
        "mfe_median_r": float(_median(mfe_values)),
        "mae_median_r": float(_median(mae_values)),
        "spread_avg_points": float(spread_avg),
        "spread_p80_points": float(_percentile(spread_values, 0.8)),
        "slippage_avg_points": float(slippage_avg),
        "slippage_p80_points": float(_percentile(slippage_values, 0.8)),
        "spread_adjusted_edge": float(spread_adjusted_edge),
        "edge_score": float(edge_score),
    }


def classify_loss_attribution(row: dict[str, Any]) -> str:
    pnl_r = _safe_float(row.get("pnl_r"), 0.0)
    pnl_money = _safe_float(row.get("pnl_amount"), _safe_float(row.get("pnl_money"), 0.0))
    if pnl_r >= 0.0 and pnl_money >= 0.0:
        return ""
    news_state = _safe_str(row.get("news_state")).lower()
    reason_text = " ".join(
        part.lower()
        for part in (
            row.get("reason"),
            row.get("close_reason"),
            row.get("management_reason"),
            row.get("blocked_reason"),
            row.get("execution_reason"),
            row.get("last_error"),
        )
        if _safe_str(part)
    )
    spread_points = _safe_float(row.get("spread_points"), 0.0)
    slippage_points = _safe_float(row.get("slippage_points"), _safe_float(row.get("slippage"), 0.0))
    regime_fit = _safe_float(row.get("regime_fit"), 1.0)
    entry_timing = _safe_float(row.get("entry_timing_score"), 1.0)
    structure = _safe_float(row.get("structure_cleanliness_score"), 1.0)
    confluence = _safe_float(row.get("confluence_score"), 0.0)
    trade_age_minutes = _safe_float(row.get("age_minutes"), 0.0)
    mfe_r = _safe_float(row.get("mfe_r"), 0.0)
    execution_quality = _safe_float(row.get("execution_quality_score"), 0.7)

    if any(token in reason_text for token in ("timeout", "delivery", "reject", "invalid stops", "execution", "fill", "slippage")):
        return "execution_miss"
    if "news" in reason_text or any(token in news_state for token in ("block", "caution", "distortion", "armed")):
        return "news_misread"
    if spread_points >= 25.0 or "spread" in reason_text:
        return "bad_spread"
    if execution_quality > 0.0 and execution_quality < 0.42:
        return "execution_miss"
    if any(token in reason_text for token in ("trail", "profit_lock", "stall", "watchdog", "management")):
        return "bad_management"
    if any(token in reason_text for token in ("tp", "exit", "scratch", "close")) and mfe_r >= 0.35 and pnl_r < 0.0:
        return "bad_exit"
    if regime_fit > 0.0 and regime_fit < 0.45:
        return "bad_regime"
    if slippage_points >= 12.0:
        return "execution_miss"
    if (entry_timing > 0.0 and entry_timing < 0.45) or (structure > 0.0 and structure < 0.45) or (confluence > 0.0 and confluence < 3.6):
        return "bad_entry"
    if trade_age_minutes > 0.0 and trade_age_minutes <= 3.0 and mfe_r <= 0.15:
        return "bad_timing"
    return "bad_timing"


def build_loss_attribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {key: 0 for key in _LOSS_TAGS}
    total_losses = 0
    for row in rows:
        tag = classify_loss_attribution(row)
        if not tag:
            continue
        total_losses += 1
        counts[tag] = counts.get(tag, 0) + 1
    primary = max(counts.items(), key=lambda item: item[1])[0] if total_losses > 0 else ""
    normalized = {
        key: {
            "count": int(value),
            "share": float((value / total_losses) if total_losses else 0.0),
        }
        for key, value in counts.items()
    }
    return {
        "total_losses": int(total_losses),
        "primary_cause": primary,
        "counts": normalized,
    }


def build_walk_forward_scorecards(
    rows: list[dict[str, Any]],
    *,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    now = now_utc.astimezone(UTC) if isinstance(now_utc, datetime) else utc_now()
    dated_rows = [(row, _parse_time(row)) for row in rows]
    windows: dict[str, dict[str, Any]] = {}
    for window_name, days in _WINDOW_DEFS:
        cutoff = now - timedelta(days=days)
        scoped_rows = [dict(row) for row, timestamp in dated_rows if timestamp is None or timestamp >= cutoff]
        windows[window_name] = _window_metrics(scoped_rows)
    session_summary: dict[str, dict[str, Any]] = {}
    for session_name in sorted({str(row.get("session_name") or "").upper() for row in rows if _safe_str(row.get("session_name"))}):
        scoped_rows = [dict(row) for row in rows if _safe_str(row.get("session_name")).upper() == session_name]
        session_summary[session_name] = _window_metrics(scoped_rows)
    recent = windows["1d"]
    short = windows["3d"]
    medium = windows["7d"]
    long = windows["30d"]
    current_edge_score = clamp(
        (0.45 * float(recent.get("edge_score", 0.5)))
        + (0.30 * float(short.get("edge_score", 0.5)))
        + (0.15 * float(medium.get("edge_score", 0.5)))
        + (0.10 * float(long.get("edge_score", 0.5))),
        0.0,
        1.0,
    )
    return {
        "windows": windows,
        "session_summary": session_summary,
        "current_edge_score": float(current_edge_score),
        "recent_trade_count": int(recent.get("trade_count", 0)),
        "thirty_day_trade_count": int(long.get("trade_count", 0)),
    }


def resolve_lane_lifecycle(scorecards: dict[str, Any]) -> dict[str, Any]:
    windows = dict(scorecards.get("windows") or {})
    recent = dict(windows.get("1d") or {})
    short = dict(windows.get("3d") or {})
    medium = dict(windows.get("7d") or {})
    long = dict(windows.get("30d") or {})
    recent_count = _safe_int(recent.get("trade_count"), 0)
    short_count = _safe_int(short.get("trade_count"), 0)
    medium_count = _safe_int(medium.get("trade_count"), 0)
    long_count = _safe_int(long.get("trade_count"), 0)
    recent_wr = _safe_float(recent.get("win_rate"), 0.0)
    short_wr = _safe_float(short.get("win_rate"), 0.0)
    recent_exp = _safe_float(recent.get("expectancy_r"), 0.0)
    short_exp = _safe_float(short.get("expectancy_r"), 0.0)
    medium_exp = _safe_float(medium.get("expectancy_r"), 0.0)
    long_exp = _safe_float(long.get("expectancy_r"), 0.0)
    current_edge_score = clamp(_safe_float(scorecards.get("current_edge_score"), 0.5), 0.0, 1.0)
    recent_edge_broken = bool(recent_count >= 3 and (recent_exp <= -0.04 or recent_wr <= 0.42))
    short_edge_broken = bool(short_count >= 4 and (short_exp <= -0.02 or short_wr <= 0.46 or current_edge_score <= 0.46))
    proven_base = bool(medium_count >= 8 and medium_exp >= 0.01 and long_count >= 8 and long_exp >= -0.01)
    recovery_ready = bool(recent_count >= 3 and recent_wr >= 0.55 and recent_exp >= 0.01 and current_edge_score >= 0.52)

    state = "probation"
    reason = "insufficient_recent_sample"
    if (
        recent_count >= 5
        and recent_wr >= 0.68
        and recent_exp >= 0.10
        and short_count >= 6
        and short_wr >= 0.60
        and short_exp >= 0.04
        and current_edge_score >= 0.63
    ):
        state = "attack"
        reason = "recent_window_is_hot"
    elif (
        short_count >= 6
        and short_wr >= 0.58
        and short_exp >= 0.02
        and current_edge_score >= 0.57
        and (recent_count < 3 or recent_wr >= 0.50)
        and medium_exp >= -0.01
    ):
        state = "proven"
        reason = "recent_windows_are_stable"
    elif recent_edge_broken:
        if proven_base and not short_edge_broken:
            state = "degrade"
            reason = "recent_damage_against_positive_base"
        else:
            state = "shadow_only"
            reason = "recent_damage_without_proven_base"
    elif short_edge_broken:
        if proven_base:
            state = "degrade"
            reason = "short_window_edge_stressed"
        else:
            state = "shadow_only"
            reason = "short_window_edge_broken"
    elif recovery_ready and short_count >= 4 and short_exp >= 0.0:
        state = "proven"
        reason = "recent_recovery_confirming"
    elif medium_count >= 8 and medium_exp >= 0.03 and current_edge_score >= 0.52:
        state = "probation"
        reason = "historical_positive_recent_pending"
    elif long_count >= 4:
        state = "probation"
        reason = "awaiting_recent_confirmation"

    ramp_multiplier = {
        "probation": 0.82,
        "proven": 1.06,
        "attack": 1.24,
        "degrade": 0.52,
        "shadow_only": 0.20,
    }.get(state, 1.0)
    shadow_only = state == "shadow_only"
    live_allowed = state != "shadow_only"
    return {
        "state": state,
        "reason": reason,
        "shadow_only": bool(shadow_only),
        "live_allowed": bool(live_allowed),
        "ramp_multiplier": float(ramp_multiplier),
        "edge_score": float(current_edge_score),
        "recovery_ready": bool(recovery_ready),
        "recent_edge_broken": bool(recent_edge_broken),
        "short_edge_broken": bool(short_edge_broken),
    }


def build_shadow_challenger_pool(
    *,
    symbol_key: str,
    shadow_strategy_variants: list[dict[str, Any]],
    lifecycle_state: dict[str, Any] | None = None,
    current_session_name: str = "",
    limit: int = 5,
) -> dict[str, Any]:
    session_key = _safe_str(current_session_name).upper()
    candidates = [
        dict(item)
        for item in shadow_strategy_variants
        if _safe_str(item.get("symbol")).upper() == _safe_str(symbol_key).upper()
    ]
    candidates.sort(
        key=lambda item: (
            _safe_float(item.get("promotion_score"), 0.0)
            + _safe_float(item.get("slippage_adjusted_score"), 0.0)
            + (0.03 if _safe_str(item.get("session")).upper() == session_key and session_key else 0.0)
        ),
        reverse=True,
    )
    challengers = candidates[: max(1, int(limit))]
    top = dict(challengers[0]) if challengers else {}
    lifecycle = dict(lifecycle_state or {})
    promote_now = bool(
        top
        and (
            lifecycle.get("state") in {"degrade", "shadow_only"}
            or bool(top.get("promoted_candidate"))
        )
        and _safe_float(top.get("promotion_score"), 0.0) >= 0.60
    )
    return {
        "challengers": challengers,
        "top_challenger": top,
        "promote_now": bool(promote_now),
    }


def evaluate_execution_quality_gate(
    *,
    spread_points: float,
    typical_spread_points: float,
    stop_distance_points: float,
    slippage_quality_score: float = 0.65,
    execution_quality_score: float = 0.65,
    microstructure_alignment: float = 0.50,
    adverse_entry_risk: float = 0.30,
    lifecycle_state: str = "",
) -> dict[str, Any]:
    spread_ratio = max(0.0, float(spread_points)) / max(1.0, float(typical_spread_points))
    stop_sanity = clamp(float(stop_distance_points) / max(float(spread_points), 1.0), 0.0, 6.0) / 6.0
    quality_score = clamp(
        (0.28 * clamp(float(execution_quality_score), 0.0, 1.0))
        + (0.24 * clamp(float(slippage_quality_score), 0.0, 1.0))
        + (0.18 * clamp(float(microstructure_alignment), 0.0, 1.0))
        + (0.18 * stop_sanity)
        + (0.12 * (1.0 - clamp(float(adverse_entry_risk), 0.0, 1.0)))
        - min(0.22, max(0.0, spread_ratio - 1.0) * 0.18),
        0.0,
        1.0,
    )
    lifecycle = _safe_str(lifecycle_state).lower()
    block_floor = 0.42 if lifecycle in {"attack", "proven"} else 0.50 if lifecycle == "probation" else 0.58
    blocked = bool(quality_score < block_floor or spread_ratio >= 2.8 or adverse_entry_risk >= 0.88)
    state = "CLEAN" if quality_score >= 0.72 else "MIXED" if quality_score >= 0.50 else "ROUGH"
    return {
        "quality_score": float(quality_score),
        "state": state,
        "blocked": bool(blocked),
        "spread_ratio": float(spread_ratio),
        "stop_sanity": float(stop_sanity),
    }


def derive_management_state(
    *,
    previous_state: str,
    pnl_r: float,
    continuation_score: float,
    reversal_risk_score: float,
    decision_action: str,
) -> str:
    previous = _safe_str(previous_state).upper()
    if previous in {"EXIT", "FORCE_EXIT", "EXIT_READY"} or decision_action in {"FULL_EXIT", "CLOSE_POSITION", "CLOSE_PARTIAL", "PARTIAL_EXIT"}:
        return "EXIT"
    if pnl_r >= 0.95 and continuation_score >= 0.52:
        return "RUNNER"
    if pnl_r >= 0.18 or previous in {"PROTECTED", "RUNNER"}:
        return "PROTECTED"
    if pnl_r >= -0.10 or continuation_score >= 0.44 or previous == "ARMED":
        return "ARMED"
    if reversal_risk_score >= 0.80:
        return "EXIT"
    return "INIT"


def build_portable_funded_profile(
    *,
    generated_at: datetime | None = None,
    pair_directives: dict[str, dict[str, Any]],
    meeting_packet: dict[str, Any],
    local_summary: dict[str, Any],
    shadow_strategy_variants: list[dict[str, Any]],
) -> dict[str, Any]:
    generated = generated_at or utc_now()
    portable_directives = {}
    for symbol_key, directive in (pair_directives or {}).items():
        directive_payload = dict(directive or {})
        portable_directives[str(symbol_key)] = {
            "lane_state_machine": dict(directive_payload.get("lane_state_machine") or {}),
            "management_directives": dict(directive_payload.get("management_directives") or {}),
            "frequency_directives": dict(directive_payload.get("frequency_directives") or {}),
            "loss_attribution_summary": dict(directive_payload.get("loss_attribution_summary") or {}),
            "execution_quality_directives": dict(directive_payload.get("execution_quality_directives") or {}),
            "shadow_challenger_pool": dict(directive_payload.get("shadow_challenger_pool") or {}),
        }
    recent_shadow = [
        dict(item)
        for item in list(shadow_strategy_variants or [])[:20]
        if isinstance(item, dict)
    ]
    return {
        "generated_at": generated.isoformat(),
        "pair_directives": portable_directives,
        "profit_budget_state": dict(meeting_packet.get("profit_budget_state") or {}),
        "walk_forward_summary": dict(local_summary.get("pair_walk_forward_scorecards") or {}),
        "loss_attribution_summary": dict(local_summary.get("pair_loss_attribution") or {}),
        "shadow_strategy_variants": recent_shadow,
        "goal_state": dict(meeting_packet.get("goal_state") or {}),
    }
