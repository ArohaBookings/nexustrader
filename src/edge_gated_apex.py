from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence


UTC = timezone.utc
PRIORITY_SYMBOLS = ("XAUUSD", "BTCUSD")
HARD_RAIL_TOKENS = (
    "drawdown",
    "daily_hard",
    "hard_stop",
    "loss_limit",
    "kill",
    "breach",
    "funded_buffer",
    "stale_mt5",
)
SOFT_REPAIR_TOKENS = (
    "stale",
    "api",
    "timeout",
    "disconnect",
    "gap",
    "data",
    "exchange",
    "bridge",
    "sync",
    "book",
    "latency",
)


def build_edge_gated_apex_policy(
    *,
    health: Mapping[str, Any],
    stats: Mapping[str, Any],
    symbols: Sequence[Mapping[str, Any]],
    institutional_apex: Mapping[str, Any],
    risk_config: Mapping[str, Any] | None = None,
    orchestrator_config: Mapping[str, Any] | None = None,
    xau_btc_trajectory_stats: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = _edge_config(orchestrator_config or {}, risk_config or {})
    apex = _record(institutional_apex)
    data_quality = _data_quality(apex)
    promotion_audit = _promotion_audit(apex, config)
    repair = _self_repair_overlay(apex)
    training = _training_bootstrap_status(
        health=health,
        data_quality=data_quality,
        repair=repair,
        promotion=promotion_audit,
        config=config,
    )
    trajectory = _trajectory_forecast(health, apex)
    opportunity_pipeline = _opportunity_pipeline(
        symbols=symbols,
        xau_btc_trajectory_stats=xau_btc_trajectory_stats or {},
        repair=repair,
        data_quality=data_quality,
        promotion=promotion_audit,
        training=training,
        config=config,
    )
    live_shadow = _live_shadow_gap(symbols, opportunity_pipeline)
    funded = _record(apex.get("funded_mission"))
    return {
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "scope": "btc_xau_first",
        "policy": "edge_gated_no_forced_live_frequency",
        "live_frequency_forced": False,
        "runtime_controls_only": True,
        "training_bootstrap_status": training,
        "data_quality": data_quality,
        "promotion_audit": promotion_audit,
        "self_repair": repair,
        "funded_mission": funded,
        "trajectory_forecast": trajectory,
        "xau_btc_opportunity_pipeline": opportunity_pipeline,
        "live_shadow_gap": live_shadow,
        "brain_summary": _brain_summary(
            apex=apex,
            health=health,
            data_quality=data_quality,
            promotion=promotion_audit,
            repair=repair,
            opportunity_pipeline=opportunity_pipeline,
        ),
        "operator_limits": [
            "No forced live frequency.",
            "No Telegram or dashboard direct trade placement.",
            "No AI-driven risk increases or code edits.",
            "Hard drawdown, funded, stale MT5, broker, and kill rails are never auto-repaired.",
            "BTCUSD and XAUUSD get shadow/candidate priority before broader pair expansion.",
        ],
    }


def _edge_config(orchestrator: Mapping[str, Any], risk_config: Mapping[str, Any]) -> dict[str, Any]:
    training = _record(orchestrator.get("training_bootstrap"))
    promotion = _record(orchestrator.get("edge_promotion"))
    frequency = _record(orchestrator.get("frequency_policy"))
    funded = _record(orchestrator.get("funded_mode")) or _record(risk_config.get("funded"))
    return {
        "training": {
            "max_years": int(_num(training.get("max_years"), 10.0)),
            "startup_shadow_minutes": int(_num(training.get("startup_shadow_minutes"), _num(training.get("startup_shadow_hours"), 1.0) * 60.0)),
            "min_data_quality": _clamp(_num(training.get("min_data_quality"), 0.55), 0.0, 1.0),
            "provider_priority": _seq(training.get("provider_priority"))
            or ["mt5", "polygon", "twelve_data", "finnhub", "binance", "bybit", "newsapi", "cryptopanic"],
        },
        "promotion": {
            "recent_window": int(_num(promotion.get("recent_window"), 200.0)),
            "validation_window": int(_num(promotion.get("validation_window"), 100.0)),
            "min_expectancy_gain": _num(promotion.get("min_expectancy_gain"), 0.03),
            "max_drawdown_degradation": _num(promotion.get("max_drawdown_degradation"), 0.0),
        },
        "frequency": {
            "priority_symbols": [str(item).upper() for item in (_seq(frequency.get("priority_symbols")) or PRIORITY_SYMBOLS)],
            "shadow_targets_10m": _record(frequency.get("shadow_targets_10m"))
            or {
                "XAUUSD": {"low": 6, "high": 8},
                "BTCUSD": {"low": 4, "high": 6},
            },
            "prime_sessions": _seq(frequency.get("prime_sessions")) or ["LONDON", "OVERLAP", "NEW_YORK"],
            "prime_session_multiplier": _num(frequency.get("prime_session_multiplier"), 1.25),
        },
        "funded": funded,
    }


def _data_quality(apex: Mapping[str, Any]) -> dict[str, Any]:
    fusion = _record(apex.get("data_fusion"))
    providers = [_record(item) for item in _seq(fusion.get("providers"))]
    native_ids = {"mt5", "hyperliquid", "binance", "bybit"}
    native_active = [item for item in providers if _txt(item.get("id")).lower() in native_ids and _txt(item.get("status")).lower() == "active"]
    proxy_active = [item for item in providers if _txt(item.get("id")).lower() not in native_ids and _txt(item.get("status")).lower() == "active"]
    degraded = [item for item in providers if _txt(item.get("status")).lower() == "degraded"]
    missing = [item for item in providers if _txt(item.get("status")).lower() == "missing"]
    consensus = _clamp(_num(fusion.get("consensus_score"), 0.0), 0.0, 1.0)
    score = _clamp(consensus + min(0.15, len(native_active) * 0.05) - min(0.20, len(degraded) * 0.03), 0.0, 1.0)
    if not providers:
        status = "unknown"
    elif native_active and score >= 0.55:
        status = "tradable_native_backed"
    elif proxy_active and score >= 0.45:
        status = "proxy_backed_research_only"
    else:
        status = "degraded"
    return {
        "status": status,
        "score": score,
        "consensus_score": consensus,
        "native_active_sources": [_txt(item.get("id")) for item in native_active],
        "proxy_active_sources": [_txt(item.get("id")) for item in proxy_active],
        "degraded_sources": [_txt(item.get("id")) for item in degraded],
        "missing_sources": [_txt(item.get("id")) for item in missing],
        "proxy_history_labeled": True,
        "native_truth_required_for_live_execution": True,
        "providers": providers,
    }


def _promotion_audit(apex: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    raw = _record(apex.get("anti_overfit"))
    promotion_config = _record(config.get("promotion"))
    recent_window = int(_num(promotion_config.get("recent_window"), 200.0))
    validation_window = int(_num(promotion_config.get("validation_window"), 100.0))
    min_gain = _num(promotion_config.get("min_expectancy_gain"), 0.03)
    max_dd_degradation = _num(promotion_config.get("max_drawdown_degradation"), 0.0)
    recent_sample = int(_num(raw.get("recent_sample"), 0.0))
    validation_sample = int(_num(raw.get("validation_sample"), 0.0))
    recent_delta = _num(raw.get("recent_delta"), 0.0)
    validation_delta = _num(raw.get("validation_delta"), 0.0)
    drawdown_degradation = _num(raw.get("drawdown_degradation"), 0.0)
    promotion_allowed = bool(
        recent_sample >= recent_window
        and validation_sample >= validation_window
        and recent_delta > min_gain
        and validation_delta > min_gain
        and drawdown_degradation <= max_dd_degradation
        and _num(raw.get("recent_expectancy"), 0.0) >= 0.0
        and _num(raw.get("validation_expectancy"), 0.0) >= 0.0
    )
    if promotion_allowed:
        reason = "edge_promotion_gate_cleared"
    elif recent_sample < recent_window or validation_sample < validation_window:
        reason = "insufficient_recent_or_validation_samples"
    elif recent_delta <= min_gain or validation_delta <= min_gain:
        reason = "expectancy_gain_below_3pct_gate"
    elif drawdown_degradation > max_dd_degradation:
        reason = "drawdown_degradation_not_allowed"
    else:
        reason = _txt(raw.get("reason"), "promotion_blocked")
    return {
        "promotion_allowed": promotion_allowed,
        "reason": reason,
        "recent_window": recent_window,
        "validation_window": validation_window,
        "min_expectancy_gain": min_gain,
        "max_drawdown_degradation": max_dd_degradation,
        "recent_sample": recent_sample,
        "validation_sample": validation_sample,
        "recent_expectancy": _num(raw.get("recent_expectancy"), 0.0),
        "validation_expectancy": _num(raw.get("validation_expectancy"), 0.0),
        "recent_delta": recent_delta,
        "validation_delta": validation_delta,
        "drawdown_degradation": drawdown_degradation,
        "raw_reason": _txt(raw.get("reason"), ""),
    }


def _self_repair_overlay(apex: Mapping[str, Any]) -> dict[str, Any]:
    raw = _record(apex.get("self_repair"))
    soft = [_record(item) for item in _seq(raw.get("soft_blockers"))]
    hard = [_record(item) for item in _seq(raw.get("hard_rails"))]
    soft_repairable = [item for item in soft if _soft_repairable(_txt(item.get("reason")))]
    actions = [_record(item) for item in _seq(raw.get("actions"))]
    if soft_repairable and not hard and not actions:
        actions.append({"action": "refresh_state", "reason": "soft stale/data blocker inside repair SLA", "allowed": True})
    status = "hard_rail_locked" if hard else "soft_repair_available" if soft_repairable else _txt(raw.get("status"), "clear")
    return {
        "status": status,
        "score": _num(raw.get("score"), 1.0 if not soft and not hard else 0.62),
        "sla_minutes": int(_num(raw.get("sla_minutes"), 5.0)),
        "soft_blockers": soft,
        "hard_rails": hard,
        "soft_repairable_count": len(soft_repairable),
        "hard_rails_locked": True,
        "recommended_bridge_action": "none" if hard else ("refresh_state" if soft_repairable else _txt(raw.get("recommended_bridge_action"), "none")),
        "actions": actions,
    }


def _training_bootstrap_status(
    *,
    health: Mapping[str, Any],
    data_quality: Mapping[str, Any],
    repair: Mapping[str, Any],
    promotion: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    training_config = _record(config.get("training"))
    brain = _record(health.get("learning_brain"))
    samples = int(_num(brain.get("trained_samples"), 0.0))
    seed_status = _txt(brain.get("last_market_history_seed_status"), "unknown")
    backfill_status = _txt(brain.get("last_market_history_backfill_status"), "unknown")
    warmup_active = bool(health.get("startup_warmup_active"))
    data_ready = _num(data_quality.get("score"), 0.0) >= _num(training_config.get("min_data_quality"), 0.55)
    hard_rails = bool(_seq(repair.get("hard_rails")))
    seed_ready = samples > 0 and seed_status in {"ok", "complete", "completed", "seeded"}
    expansion_allowed = bool(seed_ready and data_ready and not warmup_active and not hard_rails and promotion.get("promotion_allowed"))
    if hard_rails:
        status = "hard_rail_blocks_training_expansion"
    elif not seed_ready:
        status = "shadow_bootstrap_required"
    elif not data_ready:
        status = "data_quality_below_live_expansion_floor"
    elif warmup_active:
        status = "startup_shadow_observation_active"
    elif not bool(promotion.get("promotion_allowed")):
        status = "trained_observe_validate"
    else:
        status = "trained_edge_gate_cleared"
    return {
        "status": status,
        "shadow_first": True,
        "startup_shadow_minutes": int(_num(training_config.get("startup_shadow_minutes"), 60.0)),
        "max_history_years": int(_num(training_config.get("max_years"), 10.0)),
        "trained_samples": samples,
        "seed_status": seed_status,
        "backfill_status": backfill_status,
        "data_ready": data_ready,
        "live_risk_expansion_allowed": expansion_allowed,
        "provider_priority": list(_seq(training_config.get("provider_priority"))),
        "missing_provider_behavior": "degrade_to_bridge_or_shadow_only",
    }


def _trajectory_forecast(health: Mapping[str, Any], apex: Mapping[str, Any]) -> dict[str, Any]:
    brain = _record(health.get("learning_brain"))
    bundle = _record(brain.get("active_promotion_bundle"))
    goal = _record(bundle.get("goal_state"))
    if not goal:
        funded = _record(apex.get("funded_mission"))
        account = _record(funded.get("account"))
        equity = _num(account.get("equity"), _num(_record(apex.get("scaling")).get("equity"), 0.0))
        goal = {
            "current_equity": equity,
            "short_goal_equity": 100000.0,
            "short_goal_days": 240,
            "short_goal_on_track": False,
            "medium_goal_equity": 1000000.0,
            "medium_goal_days": 365,
            "medium_goal_on_track": False,
        }
    return {
        **goal,
        "forecast_type": "speculative_target_path",
        "risk_input": "not_used_for_sizing",
        "scaling_requires_edge_gate": True,
    }


def _opportunity_pipeline(
    *,
    symbols: Sequence[Mapping[str, Any]],
    xau_btc_trajectory_stats: Mapping[str, Any],
    repair: Mapping[str, Any],
    data_quality: Mapping[str, Any],
    promotion: Mapping[str, Any],
    training: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    frequency = _record(config.get("frequency"))
    targets = _record(frequency.get("shadow_targets_10m"))
    priority = [str(item).upper() for item in (_seq(frequency.get("priority_symbols")) or PRIORITY_SYMBOLS)]
    hard_rails = bool(_seq(repair.get("hard_rails")))
    rows: list[dict[str, Any]] = []
    for symbol in priority:
        card = _symbol_card(symbols, symbol)
        trajectory = _record(xau_btc_trajectory_stats.get(symbol))
        target = _record(targets.get(symbol))
        low_target = int(_num(_record(trajectory.get("soft_target_trades_last_10m")).get("low"), _num(target.get("low"), 4.0)))
        high_target = int(_num(_record(trajectory.get("soft_target_trades_last_10m")).get("high"), _num(target.get("high"), max(low_target, 6.0))))
        actual_candidates = int(_num(trajectory.get("actual_candidates_last_10m"), _num(card.get("candidate_attempts_last_15m"), 0.0)))
        actual_live = int(_num(trajectory.get("actual_trades_last_10m"), _num(card.get("delivered_actions_last_15m"), 0.0)))
        blocker = _txt(card.get("blocked_reason") or card.get("primary_block_reason"), "")
        quality = _clamp(_num(card.get("quality_score"), 0.0), 0.0, 1.0)
        approved = bool(card.get("approved")) and not blocker
        data_ok = _num(data_quality.get("score"), 0.0) >= 0.45
        has_native_or_proxy_feed = bool(_seq(data_quality.get("native_active_sources")) or _seq(data_quality.get("proxy_active_sources")))
        shadow_allowed = bool((data_ok or has_native_or_proxy_feed) and not hard_rails)
        live_expansion_allowed = bool(training.get("live_risk_expansion_allowed"))
        live_gate = "eligible_if_bridge_approves" if approved and not hard_rails else "blocked_by_edge_or_risk_gate"
        if blocker:
            action = "repair_or_shadow_validate_blocker"
        elif not live_expansion_allowed:
            action = "increase_shadow_sampling_not_live_risk"
        else:
            action = "allow_measured_live_expansion_inside_caps"
        rows.append(
            {
                "symbol": symbol,
                "priority": True,
                "shadow_target_10m": {"low": low_target, "high": high_target},
                "actual_candidates_last_10m": actual_candidates,
                "actual_live_trades_last_10m": actual_live,
                "candidate_debt_10m": max(0, low_target - actual_candidates),
                "live_trade_debt_10m": max(0, low_target - actual_live),
                "catchup_pressure": _clamp(_num(trajectory.get("catchup_pressure"), 0.0), 0.0, 1.0),
                "quality_score": quality,
                "approved_by_existing_bridge": approved,
                "shadow_burst_allowed": shadow_allowed,
                "live_expansion_allowed": live_expansion_allowed,
                "live_gate": live_gate,
                "blocker": blocker,
                "recommended_action": action,
                "forced_live_frequency": False,
            }
        )
    return {
        "policy": "shadow_priority_live_edge_gated",
        "priority_symbols": rows,
        "hard_rails_present": hard_rails,
        "promotion_allowed": bool(promotion.get("promotion_allowed")),
        "live_frequency_forced": False,
    }


def _live_shadow_gap(symbols: Sequence[Mapping[str, Any]], opportunity_pipeline: Mapping[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for card in symbols:
        symbol = _txt(card.get("symbol")).upper()
        if symbol not in PRIORITY_SYMBOLS:
            continue
        score = _clamp(_num(card.get("live_shadow_gap_risk_score"), _num(card.get("learning_live_shadow_gap_score"), 0.0)), 0.0, 1.0)
        rows.append(
            {
                "symbol": symbol,
                "gap_score": score,
                "status": "investigate_gap" if score >= 0.30 else "insufficient_or_aligned",
                "live_expectancy_r": _num(card.get("rolling_expectancy_by_pair"), 0.0),
                "shadow_expectancy_r": _num(card.get("shadow_expectancy_r"), 0.0),
                "action": "throttle_live_expand_shadow" if score >= 0.30 else "continue_observation",
            }
        )
    if not rows:
        rows = [
            {
                "symbol": row.get("symbol"),
                "gap_score": 0.0,
                "status": "insufficient_live_shadow_samples",
                "action": "collect_shadow_and_live_telemetry",
            }
            for row in _seq(opportunity_pipeline.get("priority_symbols"))
        ]
    worst = max((_num(item.get("gap_score"), 0.0) for item in rows), default=0.0)
    return {
        "status": "gap_throttle_required" if worst >= 0.30 else "collecting_or_aligned",
        "max_gap_score": worst,
        "priority_symbols": rows,
        "live_expansion_throttle": _clamp(1.0 - worst * 0.65, 0.35, 1.0),
    }


def _brain_summary(
    *,
    apex: Mapping[str, Any],
    health: Mapping[str, Any],
    data_quality: Mapping[str, Any],
    promotion: Mapping[str, Any],
    repair: Mapping[str, Any],
    opportunity_pipeline: Mapping[str, Any],
) -> dict[str, Any]:
    brain = _record(health.get("learning_brain"))
    priority = [_record(item) for item in _seq(opportunity_pipeline.get("priority_symbols"))]
    blocked = [item for item in priority if _txt(item.get("blocker"))]
    return {
        "readiness": _txt(apex.get("readiness"), "unknown"),
        "apex_summary": _txt(apex.get("summary"), ""),
        "trained_samples": int(_num(brain.get("trained_samples"), 0.0)),
        "pending_samples": int(_num(brain.get("pending_samples"), 0.0)),
        "data_quality_status": _txt(data_quality.get("status"), "unknown"),
        "promotion_reason": _txt(promotion.get("reason"), "unknown"),
        "repair_status": _txt(repair.get("status"), "unknown"),
        "priority_blocked_symbols": [_txt(item.get("symbol")) for item in blocked],
        "operator_message": (
            "BTCUSD/XAUUSD are prioritized for shadow opportunity capture; live expansion stays gated by validated edge and hard rails."
        ),
    }


def _symbol_card(symbols: Sequence[Mapping[str, Any]], symbol: str) -> dict[str, Any]:
    target = symbol.upper()
    for item in symbols:
        record = _record(item)
        if _txt(record.get("symbol")).upper() == target:
            return record
    return {"symbol": target}


def _hard_rail(reason: str) -> bool:
    value = reason.lower()
    return any(token in value for token in HARD_RAIL_TOKENS)


def _soft_repairable(reason: str) -> bool:
    value = reason.lower()
    return bool(value) and not _hard_rail(value) and any(token in value for token in SOFT_REPAIR_TOKENS)


def _record(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _seq(value: Any) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []


def _txt(value: Any, default: str = "") -> str:
    text = str(value or "").strip()
    return text if text else default


def _num(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    if parsed != parsed or parsed in (float("inf"), float("-inf")):
        return float(default)
    return parsed


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))
