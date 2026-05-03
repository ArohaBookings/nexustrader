from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math


@dataclass(frozen=True)
class LearningScalerConfig:
    enabled: bool = True
    min_real_trades_for_score: int = 10
    full_scale_min_trades: int = 20
    target_win_rate: float = 0.50
    target_expectancy_r: float = 0.05
    target_profit_factor: float = 1.20
    full_scale_expectancy_r: float = 0.10
    max_drawdown_r: float = 6.0
    positive_recent_delta_r: float = 0.01

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "LearningScalerConfig":
        data = dict(raw or {})
        return cls(
            enabled=bool(data.get("enabled", True)),
            min_real_trades_for_score=max(1, int(_number(data.get("min_real_trades_for_score"), 10))),
            full_scale_min_trades=max(1, int(_number(data.get("full_scale_min_trades"), 20))),
            target_win_rate=_bounded(_number(data.get("target_win_rate"), 0.50), 0.0, 1.0),
            target_expectancy_r=_number(data.get("target_expectancy_r"), 0.05),
            target_profit_factor=max(0.01, _number(data.get("target_profit_factor"), 1.20)),
            full_scale_expectancy_r=_number(data.get("full_scale_expectancy_r"), 0.10),
            max_drawdown_r=max(0.01, abs(_number(data.get("max_drawdown_r"), 6.0))),
            positive_recent_delta_r=_number(data.get("positive_recent_delta_r"), 0.01),
        )


def build_learning_scaler_scorecard(
    *,
    rollout_stats: Mapping[str, Any] | None = None,
    aggression_snapshot: Mapping[str, Any] | None = None,
    account_scaling: Mapping[str, Any] | None = None,
    raw_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = LearningScalerConfig.from_mapping(raw_config)
    rollout = dict(rollout_stats or {})
    aggression = dict(aggression_snapshot or {})
    account = dict(account_scaling or {})
    live = _live_evidence(rollout=rollout, aggression=aggression)
    overall = dict(rollout.get("overall") or {})
    last_20 = dict(rollout.get("last_20") or {})
    last_10 = dict(rollout.get("last_10") or {})

    recent_expectancy = _number(last_20.get("expectancy_r", last_10.get("expectancy_r")), live["expectancy_r"])
    recent_delta = recent_expectancy - live["expectancy_r"]
    equity = _number(account.get("equity", account.get("balance")), 0.0)
    blockers = [str(item) for item in _sequence(aggression.get("blockers")) if str(item).strip()]
    hard_blockers = [item for item in blockers if item not in {"telegram_aggression_unlock_required", "aggression_bucket_cap_reached"}]
    why_not_full = [str(item) for item in _sequence(aggression.get("why_not_full_aggression")) if str(item).strip()]

    if not config.enabled:
        status = "disabled"
    elif hard_blockers:
        status = "protected_by_hard_rails"
    elif "aggression_bucket_cap_reached" in blockers:
        status = "bucket_cap_reached_waiting_reset"
    elif not bool(aggression.get("owner_unlocked")):
        status = "ready_for_owner_base_unlock" if equity > 0 else "waiting_for_live_account"
    elif live["trade_count"] < config.min_real_trades_for_score:
        status = "collecting_live_evidence"
    elif not _meets_basic_edge(live, config):
        status = "learning_not_scaling"
    elif _meets_full_edge(live, config):
        status = "full_scaling_ready_inside_caps"
    else:
        status = "proven_scaling_ready"

    why_not_world_class = _why_not_world_class(
        config=config,
        live=live,
        blockers=blockers,
        hard_blockers=hard_blockers,
        recent_delta=recent_delta,
        equity=equity,
    )
    quick_learner_score = _quick_learner_score(
        config=config,
        live=live,
        recent_delta=recent_delta,
        hard_blockers=hard_blockers,
    )
    quick_scaler_score = _quick_scaler_score(
        config=config,
        live=live,
        aggression=aggression,
        hard_blockers=hard_blockers,
        equity=equity,
    )

    return {
        "enabled": bool(config.enabled),
        "status": status,
        "quick_learner_score": round(quick_learner_score, 1),
        "quick_scaler_score": round(quick_scaler_score, 1),
        "overall_score": round((quick_learner_score * 0.55) + (quick_scaler_score * 0.45), 1),
        "live_evidence": live,
        "recent_delta_expectancy_r": round(recent_delta, 4),
        "recent_expectancy_r": round(recent_expectancy, 4),
        "equity": round(equity, 2),
        "tier": str(aggression.get("tier") or "UNKNOWN"),
        "owner_unlocked": bool(aggression.get("owner_unlocked")),
        "bucket": {
            "cap": int(_number(aggression.get("cap"), 0.0)),
            "used": int(_number(aggression.get("used"), 0.0)),
            "remaining": int(_number(aggression.get("remaining"), 0.0)),
            "next_reset": str(aggression.get("next_reset") or ""),
        },
        "targets": {
            "min_real_trades_for_score": int(config.min_real_trades_for_score),
            "full_scale_min_trades": int(config.full_scale_min_trades),
            "target_win_rate": float(config.target_win_rate),
            "target_expectancy_r": float(config.target_expectancy_r),
            "target_profit_factor": float(config.target_profit_factor),
            "full_scale_expectancy_r": float(config.full_scale_expectancy_r),
            "max_drawdown_r": float(config.max_drawdown_r),
            "positive_recent_delta_r": float(config.positive_recent_delta_r),
        },
        "why_not_world_class": why_not_world_class,
        "why_not_quick_scaling": why_not_world_class + [item for item in why_not_full if item not in why_not_world_class],
        "next_safe_action": _next_safe_action(status=status, blockers=blockers, live=live, config=config),
        "proof_required": [
            "real_closed_trades_only",
            "positive_expectancy_after_costs",
            "drawdown_inside_limit",
            "no hard MT5, drawdown, funded, stale-data, spread, or kill rails",
        ],
        "claim": (
            "measured_scaling_ready" if status in {"proven_scaling_ready", "full_scaling_ready_inside_caps"} else
            "not_proven_world_class_yet"
        ),
    }


def _live_evidence(*, rollout: Mapping[str, Any], aggression: Mapping[str, Any]) -> dict[str, Any]:
    raw_live = aggression.get("live_evidence") if isinstance(aggression.get("live_evidence"), Mapping) else {}
    overall = rollout.get("overall") if isinstance(rollout.get("overall"), Mapping) else {}
    trade_count = int(_number(raw_live.get("trade_count", rollout.get("trade_count", overall.get("trades"))), 0.0))
    return {
        "trade_count": trade_count,
        "win_rate": _bounded(_number(raw_live.get("win_rate", overall.get("win_rate")), 0.0), 0.0, 1.0),
        "expectancy_r": _number(raw_live.get("expectancy_r", overall.get("expectancy_r")), 0.0),
        "profit_factor": _number(raw_live.get("profit_factor", overall.get("profit_factor")), 0.0),
        "max_drawdown_r": _number(raw_live.get("max_drawdown_r", overall.get("max_drawdown_r")), 0.0),
    }


def _meets_basic_edge(live: Mapping[str, Any], config: LearningScalerConfig) -> bool:
    return bool(
        int(live.get("trade_count") or 0) >= config.min_real_trades_for_score
        and float(live.get("win_rate") or 0.0) >= config.target_win_rate
        and float(live.get("expectancy_r") or 0.0) >= config.target_expectancy_r
        and float(live.get("profit_factor") or 0.0) >= config.target_profit_factor
        and abs(float(live.get("max_drawdown_r") or 0.0)) <= config.max_drawdown_r
    )


def _meets_full_edge(live: Mapping[str, Any], config: LearningScalerConfig) -> bool:
    return bool(
        _meets_basic_edge(live, config)
        and int(live.get("trade_count") or 0) >= config.full_scale_min_trades
        and float(live.get("expectancy_r") or 0.0) >= config.full_scale_expectancy_r
    )


def _quick_learner_score(
    *,
    config: LearningScalerConfig,
    live: Mapping[str, Any],
    recent_delta: float,
    hard_blockers: list[str],
) -> float:
    sample = _ratio(float(live.get("trade_count") or 0.0), float(config.min_real_trades_for_score)) * 20.0
    win = _ratio(float(live.get("win_rate") or 0.0), max(config.target_win_rate, 0.01)) * 18.0
    expectancy = _ratio(max(0.0, float(live.get("expectancy_r") or 0.0)), max(config.target_expectancy_r, 0.01)) * 24.0
    profit = _ratio(float(live.get("profit_factor") or 0.0), config.target_profit_factor) * 16.0
    dd = (1.0 - _ratio(abs(float(live.get("max_drawdown_r") or 0.0)), config.max_drawdown_r)) * 12.0
    trend = _ratio(max(0.0, recent_delta), max(config.positive_recent_delta_r, 0.001)) * 10.0
    penalty = min(25.0, 8.0 * len(hard_blockers))
    return _bounded(sample + win + expectancy + profit + dd + trend - penalty, 0.0, 100.0)


def _quick_scaler_score(
    *,
    config: LearningScalerConfig,
    live: Mapping[str, Any],
    aggression: Mapping[str, Any],
    hard_blockers: list[str],
    equity: float,
) -> float:
    tier = str(aggression.get("tier") or "UNKNOWN").upper()
    tier_points = {
        "DISABLED": 0.0,
        "BASE": 20.0,
        "PROVEN": 45.0,
        "FULL_BOOTSTRAP": 70.0,
        "FULL_GROWTH": 82.0,
        "FULL_GROWTH_HOT": 90.0,
    }.get(tier, 10.0)
    if not bool(aggression.get("owner_unlocked")):
        tier_points = min(tier_points, 15.0)
    sample = _ratio(float(live.get("trade_count") or 0.0), float(config.full_scale_min_trades)) * 18.0
    edge = _ratio(max(0.0, float(live.get("expectancy_r") or 0.0)), max(config.full_scale_expectancy_r, 0.01)) * 14.0
    capacity = _ratio(float(aggression.get("remaining") or 0.0), max(float(aggression.get("cap") or 0.0), 1.0)) * 8.0
    account_ready = 5.0 if equity > 0 else 0.0
    penalty = min(35.0, 10.0 * len(hard_blockers))
    return _bounded(tier_points + sample + edge + capacity + account_ready - penalty, 0.0, 100.0)


def _why_not_world_class(
    *,
    config: LearningScalerConfig,
    live: Mapping[str, Any],
    blockers: list[str],
    hard_blockers: list[str],
    recent_delta: float,
    equity: float,
) -> list[str]:
    reasons: list[str] = []
    if equity <= 0.0:
        reasons.append("missing_live_equity")
    if "telegram_aggression_unlock_required" in blockers:
        reasons.append("telegram_aggression_locked")
    if hard_blockers:
        reasons.extend([f"hard_rail_active:{item}" for item in hard_blockers[:5]])
    if "aggression_bucket_cap_reached" in blockers:
        reasons.append("current_2h_aggression_bucket_full")
    if int(live.get("trade_count") or 0) < config.min_real_trades_for_score:
        reasons.append("not_enough_real_closed_trades_for_learning_proof")
    if int(live.get("trade_count") or 0) < config.full_scale_min_trades:
        reasons.append("not_enough_real_closed_trades_for_full_scaling")
    if float(live.get("win_rate") or 0.0) < config.target_win_rate:
        reasons.append("win_rate_below_target")
    if float(live.get("expectancy_r") or 0.0) < config.target_expectancy_r:
        reasons.append("expectancy_below_learning_target")
    if float(live.get("expectancy_r") or 0.0) < config.full_scale_expectancy_r:
        reasons.append("expectancy_below_full_scaling_target")
    if float(live.get("profit_factor") or 0.0) < config.target_profit_factor:
        reasons.append("profit_factor_below_target")
    if abs(float(live.get("max_drawdown_r") or 0.0)) > config.max_drawdown_r:
        reasons.append("drawdown_too_deep_for_scaling")
    if recent_delta < config.positive_recent_delta_r:
        reasons.append("recent_expectancy_not_improving_fast_enough")
    return _dedupe(reasons)


def _next_safe_action(
    *,
    status: str,
    blockers: list[str],
    live: Mapping[str, Any],
    config: LearningScalerConfig,
) -> str:
    if status == "disabled":
        return "enable_learning_scaler_scorecard_if_runtime_visibility_is_required"
    if any(item not in {"telegram_aggression_unlock_required", "aggression_bucket_cap_reached"} for item in blockers):
        return "clear_hard_rail_before_any_scaling"
    if "telegram_aggression_unlock_required" in blockers:
        return "review_aggression_report_then_unlock_base_tier_from_telegram_if_you_accept_the_risk"
    if "aggression_bucket_cap_reached" in blockers:
        return "wait_for_next_2h_bucket_or_review_closed_trade_quality"
    if int(live.get("trade_count") or 0) < config.min_real_trades_for_score:
        return "collect_more_real_closed_trades_inside_current_cap"
    if not _meets_basic_edge(live, config):
        return "keep_base_or_proven_caps_and_let_loss_review_filter_weak_setups"
    if not _meets_full_edge(live, config):
        return "allow_proven_cap_only_and_continue_validation"
    return "full_tier_is_allowed_inside_caps_only_hard_rails_still_override"


def _sequence(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return []


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            output.append(text)
    return output


def _ratio(value: float, target: float) -> float:
    if target <= 0.0:
        return 1.0 if value > 0.0 else 0.0
    return _bounded(value / target, 0.0, 1.0)


def _bounded(value: float, low: float, high: float) -> float:
    if not math.isfinite(value):
        return low
    return max(low, min(high, value))


def _number(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default
