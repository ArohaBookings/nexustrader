from __future__ import annotations

from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any
import pandas as pd

from src.ai_gate import AIGate
from src.aggression_runtime import (
    build_event_directive,
    build_execution_minute_profile,
    build_lead_lag_snapshot,
    build_microstructure_score,
)
from src.apex_learning_brain import ApexLearningBrain
from src.backtest import Backtester
from src.bridge_server import bridge_runtime_insights_state, runtime_entry_block_state
from src.bridge_stop_validation import (
    StopValidationInput,
    SymbolRule,
    load_symbol_rules,
    resolve_symbol_rule,
    validate_and_normalize_stops,
)
from src.config_loader import load_settings
from src.execution import ExecutionRequest, ExecutionService, TradeJournal, trading_day_key_for_timestamp
from src.feature_engineering import FeatureEngineer
from src.grid_scalper import XAUGridScalper
from src.logger import LoggerFactory
from src.market_data import MarketDataService
from src.monitor import DashboardState, KillSwitch, Monitor, SymbolStatus
from src.mt5_client import MT5Client, MT5Credentials, OrderResult
from src.news_engine import NewsEngine
from src.online_learning import OnlineLearningEngine
from src.performance_report import build_performance_report
from src.portfolio_manager import PortfolioManager
from src.position_manager import PositionManager
from src.regime_detector import RegimeDetector
from src.liquidity_map import evaluate_liquidity_map
from src.risk_engine import RiskEngine, RiskInputs, detect_funded_account_mode
from src.session_calendar import SYDNEY, describe_market_state, dominant_session_name, is_weekend_market_mode, market_open_tuple
from src.session_profile import SessionProfile
from src.symbol_universe import normalize_symbol_key as canonical_symbol_key, symbol_asset_class, symbol_family_defaults
from src.strategy_engine import SignalCandidate, StrategyEngine
from src.strategy_optimizer import StrategyOptimizer
from src.strategy_router import StrategyRouter
from src.strategies.trend_daytrade import resolve_strategy_key
from src.trade_idea_lifecycle import TradeIdeaLifecycle
from src.trade_execution_gate import validate_trade_executable
from src.trade_quality import (
    compression_strategy_bias,
    delta_proxy_score,
    evaluate_execution_quality,
    evaluate_trade_quality,
    infer_trade_lane,
    xau_grid_lane_for_session,
    normalize_strategy_family,
    pair_behavior_fit,
    quality_tier_from_scores,
    quality_tier_size_multiplier,
    quality_band_rank,
    runtime_regime_state,
    session_loosen_factor,
    session_adjusted_score,
    session_priority_context,
    session_priority_override_decision,
    strategy_allowed_regimes,
    strategy_health_state,
    strategy_management_template,
    strategy_regime_fit,
    strategy_recent_performance_score,
    strategy_selection_score,
    entry_timing_score,
    structure_cleanliness_score,
    winner_promotion_bonus,
)
from src.train import Trainer
from src.utils import clamp, deterministic_id, ensure_parent

UTC = timezone.utc
_BTC_WEEKEND_FORCE_EMIT_STATE: dict[str, datetime] = {}


def compute_session_state(settings, now: datetime) -> tuple[float, str, str]:
    for session in settings.sessions:
        if session.enabled and session.contains(now):
            return session.size_multiplier, "IN", session.name
    return 0.0, "OUT", "none"


def _friday_cutoff_hour(value: str) -> int:
    try:
        return int(str(value).split(":", 1)[0])
    except (TypeError, ValueError):
        return 20


def _is_dry_run_mode(mode: str) -> bool:
    return mode.upper() == "DRY_RUN"


def _normalize_symbol_key(value: str) -> str:
    return canonical_symbol_key(value)


def _btc_weekend_force_candidate(
    *,
    symbol: str,
    row: pd.Series,
    session_name: str,
    timestamp: datetime,
    emit_time: datetime | None = None,
    cadence_seconds: int = 60,
) -> SignalCandidate | None:
    symbol_key = _normalize_symbol_key(symbol)
    session_upper = str(session_name or "").upper()
    if symbol_key != "BTCUSD":
        return None
    if session_upper not in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}:
        return None
    current = pd.Timestamp(emit_time or datetime.now(timezone.utc))
    if current.tzinfo is None:
        current = current.tz_localize("UTC")
    else:
        current = current.tz_convert("UTC")
    current_dt = current.to_pydatetime()
    if not is_weekend_market_mode(current_dt):
        return None
    emit_key = f"{symbol_key}:{session_upper}"
    last_emit = _BTC_WEEKEND_FORCE_EMIT_STATE.get(emit_key)
    elapsed_since_emit = (current_dt - last_emit).total_seconds() if last_emit is not None else None
    if last_emit is not None and 0.0 <= float(elapsed_since_emit or 0.0) < max(30, int(cadence_seconds or 60)):
        return None

    close = float(row.get("m5_close", row.get("close", 0.0)) or 0.0)
    atr = max(float(row.get("m5_atr_14", row.get("m15_atr_14", 0.0)) or 0.0), 1e-6)
    if close <= 0.0 or atr <= 0.0:
        return None
    ema20 = float(row.get("m5_ema_20", close) or close)
    ema50 = float(row.get("m5_ema_50", close) or close)
    h1_ema20 = float(row.get("h1_ema_20", ema20) or ema20)
    h1_ema50 = float(row.get("h1_ema_50", ema50) or h1_ema20)
    momentum = float(row.get("m5_ret_1", 0.0) or 0.0)
    higher_momentum = float(row.get("m15_ret_1", momentum) or momentum)
    body_efficiency = clamp(abs(float(row.get("m5_body_efficiency", 0.0) or 0.0)), 0.0, 1.0)
    market_instability = clamp(float(row.get("market_instability_score", 0.0) or 0.0), 0.0, 1.0)
    range_position = clamp(float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5), 0.0, 1.0)
    spread_ratio = float(row.get("m5_spread_ratio_20", 1.0) or 1.0)
    atr_ratio = float(row.get("m5_atr_pct_of_avg", 1.0) or 1.0)
    rsi = clamp(float(row.get("m5_rsi_14", 50.0) or 50.0), 0.0, 100.0)
    upper_wick_ratio = clamp(float(row.get("m5_upper_wick_ratio", 0.0) or 0.0), 0.0, 1.0)
    lower_wick_ratio = clamp(float(row.get("m5_lower_wick_ratio", 0.0) or 0.0), 0.0, 1.0)
    short_bias = (ema20 - ema50) / atr
    higher_bias = (h1_ema20 - h1_ema50) / max(atr * 2.0, 1e-6)
    distance_to_ema = (close - ema20) / atr
    directional_score = short_bias + (0.65 * higher_bias) + (0.45 if momentum >= 0.0 else -0.45) + (0.25 if higher_momentum >= 0.0 else -0.25)
    if range_position >= 0.72:
        directional_score -= 0.18 + (0.10 if rsi >= 58.0 else 0.0)
    elif range_position <= 0.28:
        directional_score += 0.18 + (0.10 if rsi <= 42.0 else 0.0)
    if upper_wick_ratio >= 0.28 and range_position >= 0.60:
        directional_score -= 0.16 + min(0.10, max(0.0, upper_wick_ratio - lower_wick_ratio) * 0.40)
    if lower_wick_ratio >= 0.28 and range_position <= 0.40:
        directional_score += 0.16 + min(0.10, max(0.0, lower_wick_ratio - upper_wick_ratio) * 0.40)
    if distance_to_ema >= 0.95 and momentum <= 0.0:
        directional_score -= 0.14
    elif distance_to_ema <= -0.95 and momentum >= 0.0:
        directional_score += 0.14
    if abs(directional_score) < 0.08:
        if range_position >= 0.68 and (rsi >= 56.0 or upper_wick_ratio > lower_wick_ratio + 0.08):
            directional_score = -0.14
        elif range_position <= 0.32 and (rsi <= 44.0 or lower_wick_ratio > upper_wick_ratio + 0.08):
            directional_score = 0.14
        else:
            directional_score = 0.12 if distance_to_ema >= 0.0 else -0.12
    side = "BUY" if directional_score >= 0.0 else "SELL"
    setup = (
        "BTC_TOKYO_DRIFT_SCALP"
        if session_upper in {"SYDNEY", "TOKYO"}
        else "BTC_LONDON_IMPULSE_SCALP"
        if session_upper == "LONDON"
        else "BTC_NY_LIQUIDITY"
    )
    score_hint = 0.61 if session_upper in {"SYDNEY", "TOKYO"} else 0.65
    confluence = clamp(
        2.90
        + min(0.18, abs(short_bias) * 0.08)
        + min(0.16, abs(higher_bias) * 0.08)
        + min(0.18, body_efficiency * 0.24)
        + (0.08 if session_upper in {"SYDNEY", "TOKYO"} else 0.12),
        0.0,
        5.0,
    )
    _BTC_WEEKEND_FORCE_EMIT_STATE[emit_key] = current_dt
    return SignalCandidate(
        signal_id=deterministic_id(symbol_key, "btc-weekend-force", side, setup, current.isoformat()),
        setup=setup,
        side=side,
        score_hint=score_hint,
        reason="btc_weekend_force_emit",
        stop_atr=0.76,
        tp_r=1.16,
        strategy_family="SCALP",
        confluence_score=confluence,
        confluence_required=2.35,
        meta={
            "timeframe": "M15",
            "atr_field": "m5_atr_14",
            "allow_ai_approve_small": True,
            "approve_small_min_probability": 0.34,
            "approve_small_min_confluence": 2.35,
            "btc_strategy": "WEEKEND_FORCE_EMIT",
            "setup_family": "PRICE_ACTION",
            "btc_min_ai_confidence": 0.32,
            "proxyless_price_action_mode": True,
            "proxyless_weekend_heartbeat_mode": True,
            "btc_weekend_force_emit": True,
            "throughput_recovery_active": True,
            "quality_tier": "B",
            "router_rank_score": float(score_hint),
            "regime_fit": max(0.48, min(0.74, 0.50 + abs(float(higher_bias)) * 0.10)),
            "session_fit": 0.58 if session_upper in {"SYDNEY", "TOKYO"} else 0.66,
            "volatility_fit": 0.58 if atr_ratio <= 2.8 else 0.50,
            "pair_behavior_fit": 0.56 if session_upper in {"SYDNEY", "TOKYO"} else 0.60,
            "execution_quality_fit": 0.60 if spread_ratio <= 2.8 else 0.54,
            "entry_timing_score": 0.54 if body_efficiency < 0.18 else 0.60,
            "structure_cleanliness_score": 0.48 if market_instability >= 0.70 else 0.56,
            "strategy_recent_performance_seed": 0.58,
            "seasonality_edge_score": float(row.get("seasonality_edge_score", 0.5) or 0.5),
            "market_instability_score": float(market_instability),
            "range_position": float(range_position),
            "spread_ratio": float(spread_ratio),
            "atr_ratio": float(atr_ratio),
            "directional_score": float(directional_score),
            "rsi": float(rsi),
            "upper_wick_ratio": float(upper_wick_ratio),
            "lower_wick_ratio": float(lower_wick_ratio),
            "weekend_heartbeat_emit_session": session_upper,
        },
    )


_SUPER_AGGRESSIVE_NORMAL_SYMBOLS: set[str] = {
    "AUDJPY",
    "NZDJPY",
    "USDJPY",
    "AUDNZD",
    "EURUSD",
    "GBPUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "NAS100",
    "USOIL",
    "XAGUSD",
    "DOGUSD",
    "TRUMPUSD",
    "AAPL",
    "NVIDIA",
}
_SUPER_AGGRESSIVE_ASIA_HOME_SYMBOLS: set[str] = {"AUDJPY", "NZDJPY", "USDJPY", "AUDNZD"}


def _is_super_aggressive_normal_symbol(symbol_key: str) -> bool:
    return _normalize_symbol_key(symbol_key) in _SUPER_AGGRESSIVE_NORMAL_SYMBOLS


def _is_super_aggressive_home_session(symbol_key: str, session_name: str) -> bool:
    normalized_symbol = _normalize_symbol_key(symbol_key)
    session_key = str(session_name or "").strip().upper()
    if normalized_symbol in _SUPER_AGGRESSIVE_ASIA_HOME_SYMBOLS:
        return session_key in {"SYDNEY", "TOKYO"}
    return session_key in {"LONDON", "OVERLAP", "NEW_YORK"}


def _always_on_symbol_keys(system_config: dict) -> set[str]:
    raw = system_config.get("always_on_symbols", [])
    if not isinstance(raw, list):
        return set()
    return {_normalize_symbol_key(str(symbol)) for symbol in raw if str(symbol).strip()}


def _micro_cooldown_minutes(symbol: str, now: datetime, micro_config: dict[str, Any]) -> tuple[int, int]:
    loss_minutes = int(micro_config.get("cooldown_minutes_after_loss", 20) or 20)
    win_minutes = int(micro_config.get("cooldown_minutes_after_win", 5) or 5)
    if _normalize_symbol_key(symbol) == "BTCUSD" and is_weekend_market_mode(now):
        loss_minutes = min(loss_minutes, 1)
        win_minutes = min(win_minutes, 0)
    return max(0, loss_minutes), max(0, win_minutes)


def _is_always_on_symbol(configured_symbol: str, resolved_symbol: str, always_on: set[str]) -> bool:
    configured_key = _normalize_symbol_key(configured_symbol)
    resolved_key = _normalize_symbol_key(resolved_symbol)
    return configured_key in always_on or resolved_key in always_on


def _density_session_focus(symbol_key: str) -> list[str]:
    normalized_symbol = _normalize_symbol_key(symbol_key)
    asset_class = str(symbol_asset_class(normalized_symbol) or "").strip().lower()
    if normalized_symbol in {"XAUUSD", "BTCUSD"}:
        return ["SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"]
    if normalized_symbol in _SUPER_AGGRESSIVE_ASIA_HOME_SYMBOLS:
        return ["SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"]
    if asset_class in {"crypto", "forex", "commodity"}:
        return ["SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"]
    return ["LONDON", "OVERLAP", "NEW_YORK"]


def _augment_learning_policy_for_density(
    *,
    symbol_key: str,
    session_name: str,
    learning_policy: dict[str, Any] | None,
    current_scaling_state: dict[str, Any] | None,
) -> dict[str, Any]:
    normalized_symbol = _normalize_symbol_key(symbol_key)
    session_key = str(session_name or "").upper()
    payload = dict(learning_policy or {})
    pair_directive = dict(payload.get("pair_directive") or {})
    frequency_directives = dict(pair_directive.get("frequency_directives") or {})
    scaling_state = dict(current_scaling_state or {})
    asset_class = str(symbol_asset_class(normalized_symbol) or "").strip().lower()
    ramp_ready = bool(scaling_state.get("smart_scaling_ready", False))
    phase_name = str(scaling_state.get("current_phase", "PHASE_1") or "PHASE_1")

    base_aggression = 1.04
    base_soft_burst = 3
    density_entry_cap_bonus = 0
    density_compression_candidate_bonus = 0
    density_compression_multiplier = 1.0
    density_rank_bonus = 0.0
    density_size_bonus = 0.0
    density_activation_relax = 0.0
    proof_lane_ready = ramp_ready
    hot_hand_active = False
    profit_recycle_active = ramp_ready

    if normalized_symbol == "XAUUSD":
        base_aggression = 1.36
        base_soft_burst = (
            18 if session_key == "LONDON" else 16 if session_key in {"OVERLAP", "NEW_YORK"} else 5
        )
        density_entry_cap_bonus = 5
        density_compression_candidate_bonus = 4
        density_compression_multiplier = 1.28
        density_rank_bonus = 0.025
        density_size_bonus = 0.05
        density_activation_relax = 0.02
        proof_lane_ready = True
        hot_hand_active = True
    elif normalized_symbol in {"BTCUSD", "NAS100", "USDJPY", "AUDJPY", "NZDJPY", "AUDNZD", "XAGUSD"}:
        base_aggression = 1.14 if normalized_symbol == "BTCUSD" else 1.11
        base_soft_burst = (
            8 if session_key in {"LONDON", "OVERLAP", "NEW_YORK"} or normalized_symbol == "BTCUSD" else 6
        )
        density_entry_cap_bonus = 3
        density_compression_candidate_bonus = 2
        density_compression_multiplier = 1.18
        density_rank_bonus = 0.018
        density_size_bonus = 0.035
        density_activation_relax = 0.01
    elif normalized_symbol in {"EURUSD", "GBPUSD", "EURJPY", "GBPJPY"} or asset_class == "forex":
        base_aggression = 1.07
        base_soft_burst = 4 if session_key in {"LONDON", "OVERLAP", "NEW_YORK"} else 3
        density_entry_cap_bonus = 1
        density_compression_candidate_bonus = 1
        density_compression_multiplier = 1.10
        density_rank_bonus = 0.01
        density_size_bonus = 0.02
        density_activation_relax = 0.01
    elif normalized_symbol in {"USOIL", "AAPL", "NVIDIA"} or asset_class in {"equity", "index", "commodity"}:
        base_aggression = 1.08
        base_soft_burst = 4 if session_key in {"LONDON", "OVERLAP", "NEW_YORK"} else 2
        density_entry_cap_bonus = 1
        density_compression_candidate_bonus = 1
        density_compression_multiplier = 1.10
        density_rank_bonus = 0.01
        density_size_bonus = 0.02
        density_activation_relax = 0.01

    if phase_name in {"PHASE_2", "PHASE_3", "PHASE_4", "PHASE_5", "PHASE_6", "PHASE_7", "PHASE_8", "PHASE_9", "PHASE_10"}:
        hot_hand_active = hot_hand_active or normalized_symbol == "XAUUSD"

    if ramp_ready:
        lane_boost = max(1.0, float(scaling_state.get("smart_scaling_lane_boost", 1.0) or 1.0))
        base_aggression = min(1.60, base_aggression * lane_boost)
        base_soft_burst += int(scaling_state.get("smart_scaling_soft_burst_bonus", 2) or 2)
        density_entry_cap_bonus += int(scaling_state.get("smart_scaling_entry_cap_bonus", 1) or 1)
        density_compression_candidate_bonus += int(scaling_state.get("smart_scaling_compression_bonus", 1) or 1)
        density_compression_multiplier = max(
            density_compression_multiplier,
            float(scaling_state.get("smart_scaling_compression_multiplier", 1.0) or 1.0),
        )
        density_rank_bonus += float(scaling_state.get("smart_scaling_rank_bonus", 0.0) or 0.0)
        density_size_bonus += float(scaling_state.get("smart_scaling_size_bonus", 0.0) or 0.0)
        density_activation_relax = max(
            density_activation_relax,
            float(scaling_state.get("smart_scaling_activation_relax", 0.0) or 0.0),
        )
        proof_lane_ready = True
        hot_hand_active = True
        profit_recycle_active = True

    pair_directive["aggression_multiplier"] = max(
        float(pair_directive.get("aggression_multiplier", 1.0) or 1.0),
        float(base_aggression),
    )
    pair_directive["trade_horizon_bias"] = str(pair_directive.get("trade_horizon_bias") or "scalp")
    pair_directive["reentry_priority"] = max(
        float(pair_directive.get("reentry_priority", 0.0) or 0.0),
        0.85 if normalized_symbol == "XAUUSD" else 0.68 if ramp_ready else 0.52,
    )
    pair_directive["density_entry_cap_bonus"] = max(
        int(pair_directive.get("density_entry_cap_bonus", 0) or 0),
        int(density_entry_cap_bonus),
    )
    pair_directive["density_compression_candidate_bonus"] = max(
        int(pair_directive.get("density_compression_candidate_bonus", 0) or 0),
        int(density_compression_candidate_bonus),
    )
    pair_directive["density_compression_multiplier"] = max(
        float(pair_directive.get("density_compression_multiplier", 1.0) or 1.0),
        float(density_compression_multiplier),
    )
    pair_directive["density_rank_bonus"] = max(
        float(pair_directive.get("density_rank_bonus", 0.0) or 0.0),
        float(density_rank_bonus),
    )
    pair_directive["density_size_bonus"] = max(
        float(pair_directive.get("density_size_bonus", 0.0) or 0.0),
        float(density_size_bonus),
    )
    pair_directive["density_activation_relax"] = max(
        float(pair_directive.get("density_activation_relax", 0.0) or 0.0),
        float(density_activation_relax),
    )
    pair_directive["proof_lane_ready"] = bool(pair_directive.get("proof_lane_ready", False) or proof_lane_ready)
    pair_directive["hot_hand_active"] = bool(pair_directive.get("hot_hand_active", False) or hot_hand_active)
    pair_directive["profit_recycle_active"] = bool(
        pair_directive.get("profit_recycle_active", False) or profit_recycle_active
    )
    pair_directive["session_focus"] = _density_session_focus(normalized_symbol)

    frequency_directives["soft_burst_target_10m"] = max(
        int(frequency_directives.get("soft_burst_target_10m", 0) or 0),
        int(base_soft_burst),
    )
    frequency_directives["quota_boost_allowed"] = True
    frequency_directives["aggressive_reentry_enabled"] = True
    if normalized_symbol == "XAUUSD" or ramp_ready:
        frequency_directives["undertrade_fix_mode"] = True
    if ramp_ready:
        frequency_directives["idle_lane_recovery_active"] = True

    pair_directive["frequency_directives"] = frequency_directives
    payload["pair_directive"] = pair_directive
    return payload


def _symbol_entry_cap(
    symbol: str,
    candidates: list[SignalCandidate],
    *,
    default_cap: int,
    xau_grid_cap: int,
) -> int:
    resolved = max(1, int(default_cap))
    normalized_symbol = _normalize_symbol_key(symbol)
    if normalized_symbol == "XAUUSD":
        xau_grid_count = sum(1 for candidate in candidates if _is_xau_grid_setup(candidate.setup))
        if xau_grid_count >= 2:
            resolved = max(resolved, int(xau_grid_cap))
    density_caps = []
    for candidate in candidates:
        meta = candidate.meta if isinstance(candidate.meta, dict) else {}
        density_caps.append(max(0, int(meta.get("density_entry_cap", 0) or 0)))
    if density_caps:
        resolved = max(resolved, max(density_caps))
    return min(max(1, resolved), 32)


def _write_runtime_heartbeat(
    heartbeat_path: Path | None,
    *,
    now: datetime,
    mode: str,
    account_label: str,
    account_state: dict[str, Any],
    summary: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    if heartbeat_path is None:
        return
    payload: dict[str, Any] = {
        "updated_at": now.astimezone(UTC).isoformat(),
        "mode": str(mode).upper(),
        "account_label": str(account_label or ""),
        "equity": float(account_state.get("equity", 0.0) or 0.0),
        "balance": float(account_state.get("balance", 0.0) or 0.0),
        "free_margin": float(account_state.get("margin_free", account_state.get("free_margin", 0.0)) or 0.0),
        "loops": int(summary.get("loops", 0) or 0),
        "accepted": int(summary.get("accepted", 0) or 0),
        "rejected": int(summary.get("rejected", 0) or 0),
        "errors": int(summary.get("errors", 0) or 0),
    }
    if isinstance(extra, dict) and extra:
        payload.update(extra)
    try:
        ensure_parent(heartbeat_path)
        heartbeat_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    except Exception:
        return


def _is_xau_grid_setup(setup: str) -> bool:
    return str(setup).upper().startswith("XAUUSD_M5_GRID_SCALPER")


def _is_xau_higher_tf_candidate_setup(setup: str) -> bool:
    name = str(setup).upper()
    if name in {"SET_FORGET_H1_H4", "SCALE_IN_STEP", "SCALE_IN_PULLBACK"}:
        return True
    if name.startswith("XAUUSD_M1_MICRO_SCALPER"):
        return True
    if name.startswith("XAUUSD_M15_STRUCTURED"):
        return True
    if name.startswith("XAUUSD_ATR_EXPANSION_SCALPER"):
        return True
    if name in {"XAU_FAKEOUT_FADE", "XAU_BREAKOUT_RETEST"}:
        return True
    return False


def _is_account_wide_soft_kill(reason: str | None) -> bool:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return False
    return normalized in {"daily_circuit_breaker", "rolling_drawdown_kill", "invalid_lock_file"}


def _stale_soft_kill_should_clear(reason: str | None, now: datetime, system_config: dict[str, Any]) -> bool:
    normalized = str(reason or "").strip().lower()
    if normalized == "friday_flat_window":
        cutoff = _friday_cutoff_hour(system_config.get("force_flat_friday_gmt", "20:00"))
        return not (now.weekday() == 4 and now.hour >= cutoff)
    return False


def _soft_kill_recovery_note(
    reason: str | None,
    *,
    now: datetime,
    created_at: str | None = None,
    system_config: dict[str, Any],
    risk_config: dict[str, Any],
    micro_config: dict[str, Any] | None = None,
    account_equity: float | None = None,
    global_stats: Any,
) -> str | None:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return None
    if _stale_soft_kill_should_clear(reason, now, system_config):
        return "window_elapsed"
    if normalized == "invalid_lock_file":
        return "invalid_lock_file_cleared"
    if normalized == "rolling_drawdown_kill":
        current_drawdown = float(getattr(global_stats, "rolling_drawdown_pct", 0.0))
        max_drawdown_kill = float(risk_config.get("max_drawdown_kill", 0.0))
        if (
            isinstance(micro_config, dict)
            and bool(micro_config.get("bootstrap_enabled", False))
            and account_equity is not None
            and float(account_equity) <= float(micro_config.get("bootstrap_equity_threshold", 160.0))
        ):
            max_drawdown_kill = max(
                max_drawdown_kill,
                float(micro_config.get("bootstrap_drawdown_kill", max_drawdown_kill)),
            )
        if max_drawdown_kill <= 0.0 or current_drawdown < max_drawdown_kill:
            return f"rolling_drawdown_recovered:{current_drawdown:.6f}<{max_drawdown_kill:.6f}"
        return None
    if normalized == "absolute_drawdown_hard_stop":
        current_drawdown = float(getattr(global_stats, "absolute_drawdown_pct", 0.0))
        max_drawdown_hard_stop = float(risk_config.get("absolute_drawdown_hard_stop", 0.0))
        if (
            isinstance(micro_config, dict)
            and bool(micro_config.get("bootstrap_enabled", False))
            and account_equity is not None
            and float(account_equity) <= float(micro_config.get("bootstrap_equity_threshold", 160.0))
        ):
            max_drawdown_hard_stop = max(
                max_drawdown_hard_stop,
                float(micro_config.get("bootstrap_drawdown_kill", max_drawdown_hard_stop)),
            )
        if max_drawdown_hard_stop <= 0.0 or current_drawdown < max_drawdown_hard_stop:
            return f"absolute_drawdown_recovered:{current_drawdown:.6f}<{max_drawdown_hard_stop:.6f}"
        return None
    if normalized == "daily_circuit_breaker":
        current_daily_pnl = float(getattr(global_stats, "daily_pnl_pct", 0.0))
        daily_loss_limit = float(risk_config.get("circuit_breaker_daily_loss", 0.0))
        if daily_loss_limit <= 0.0 or current_daily_pnl > -daily_loss_limit:
            return f"daily_pnl_recovered:{current_daily_pnl:.6f}>{-daily_loss_limit:.6f}"
    if normalized == "hard_daily_dd":
        try:
            created_ts = datetime.fromisoformat(str(created_at or "").strip())
        except ValueError:
            created_ts = None
        if created_ts is not None:
            if created_ts.tzinfo is None:
                created_ts = created_ts.replace(tzinfo=UTC)
            created_day = created_ts.astimezone(SYDNEY).date()
            current_day = now.astimezone(SYDNEY).date()
            if current_day > created_day:
                return f"new_trading_day:{created_day.isoformat()}->{current_day.isoformat()}"
        hard_daily_dd_limit = float(risk_config.get("hard_daily_dd_pct", 0.0))
        current_daily_dd = float(getattr(global_stats, "daily_dd_pct_live", 0.0))
        if hard_daily_dd_limit <= 0.0 or current_daily_dd < hard_daily_dd_limit:
            return f"hard_daily_dd_recovered:{current_daily_dd:.6f}<{hard_daily_dd_limit:.6f}"
    return None


def _coerce_iso_datetime(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


def _non_losing_closes_since(
    closed_trades: list[dict[str, Any]],
    *,
    started_at: str | None,
) -> int:
    started_ts = _coerce_iso_datetime(started_at)
    if started_ts is None:
        return 0
    non_losses = 0
    for row in closed_trades:
        closed_at = (
            row.get("closed_at")
            or row.get("timestamp_utc")
            or row.get("timestamp")
        )
        closed_ts = _coerce_iso_datetime(closed_at)
        if closed_ts is None or closed_ts < started_ts:
            continue
        try:
            pnl_r = float(row.get("pnl_r", 0.0) or 0.0)
        except (TypeError, ValueError):
            pnl_r = 0.0
        if pnl_r >= 0.0:
            non_losses += 1
    return non_losses


def _recovery_mode_release_note(
    *,
    started_at: str | None,
    closed_trades: list[dict[str, Any]],
    learning_brain_status: dict[str, Any] | None = None,
) -> str | None:
    non_losses = _non_losing_closes_since(closed_trades, started_at=started_at)
    if non_losses >= 3:
        return f"recovery_non_loss_count:{non_losses}"
    brain_status = dict(learning_brain_status or {})
    last_cycle_at = _coerce_iso_datetime(brain_status.get("last_local_cycle_at"))
    started_ts = _coerce_iso_datetime(started_at)
    if (
        started_ts is not None
        and last_cycle_at is not None
        and last_cycle_at >= started_ts
        and not bool(brain_status.get("risk_reduction_active", False))
    ):
        return "learning_brain_promoted_normal"
    return None


def _effective_cooldown_trades_remaining(
    stats: Any,
    *,
    now: datetime,
    max_age_hours: float = 6.0,
) -> int:
    remaining = max(0, int(getattr(stats, "cooldown_trades_remaining", 0) or 0))
    if remaining <= 0:
        return 0
    close_times = list(getattr(stats, "today_closed_trade_times_raw", []) or [])
    last_close_ts: datetime | None = None
    for value in close_times:
        parsed = _coerce_iso_datetime(value)
        if parsed is None:
            continue
        if last_close_ts is None or parsed > last_close_ts:
            last_close_ts = parsed
    if last_close_ts is None:
        return 0
    if last_close_ts.astimezone(SYDNEY).date() < now.astimezone(SYDNEY).date():
        return 0
    max_age_seconds = max(300.0, float(max_age_hours) * 3600.0)
    if (now - last_close_ts).total_seconds() >= max_age_seconds:
        return 0
    return remaining


def _symbol_family_defaults(symbol: str) -> dict[str, float]:
    return symbol_family_defaults(symbol)


def _dedupe_symbols(symbols: list[str]) -> list[str]:
    output: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        key = _normalize_symbol_key(symbol)
        if not key or key in seen:
            continue
        seen.add(key)
        output.append(key)
    return output


def _expand_market_universe(
    *,
    settings: Any,
    mt5_client: MT5Client,
    configured_symbols: list[str],
    dry_run: bool,
    logger: Any | None,
) -> tuple[list[str], list[dict[str, Any]]]:
    system_config = settings.section("system")
    universe_config = system_config.get("market_universe", {}) if isinstance(system_config.get("market_universe"), dict) else {}
    mode = str(system_config.get("mode", "") or "").upper()
    live_startup_discovery_enabled = bool(universe_config.get("live_startup_discovery_enabled", False))
    if (
        dry_run
        or not bool(universe_config.get("enabled", False))
        or (mode == "LIVE" and not live_startup_discovery_enabled)
    ):
        if logger is not None and mode == "LIVE" and bool(universe_config.get("enabled", False)) and not live_startup_discovery_enabled:
            logger.info("market_universe_live_startup_discovery_skipped")
        return _dedupe_symbols(configured_symbols), []
    discovered = mt5_client.discover_symbol_universe(
        include_asset_classes={
            str(item).strip().lower()
            for item in list(universe_config.get("include_asset_classes", ["forex", "commodity", "crypto", "index", "equity"]))
            if str(item).strip()
        },
        max_per_class=dict(universe_config.get("max_symbols_per_class", {})) if isinstance(universe_config.get("max_symbols_per_class"), dict) else {},
        exclude_symbols=set(configured_symbols),
    )
    combined = list(configured_symbols)
    if bool(universe_config.get("trade_discovered_symbols", True)):
        combined.extend(str(item.get("symbol") or "") for item in discovered if str(item.get("symbol") or "").strip())
    combined = _dedupe_symbols(combined)
    snapshot_file = str(universe_config.get("snapshot_file", "") or "").strip()
    if snapshot_file:
        snapshot_path = settings.resolve_path_value(snapshot_file)
        ensure_parent(snapshot_path)
        snapshot_path.write_text(
            json.dumps(
                {
                    "generated_at": datetime.now(tz=UTC).isoformat(),
                    "configured_symbols": list(configured_symbols),
                    "active_symbols": list(combined),
                    "discovered_symbols": list(discovered),
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    if logger is not None and discovered:
        logger.info(
            f"market_universe_expanded configured={len(configured_symbols)} discovered={len(discovered)} active={len(combined)}"
        )
    return combined, discovered


def _warm_market_universe_history(
    *,
    settings: Any,
    market_data: MarketDataService,
    resolved_symbols: dict[str, str],
    dry_run: bool,
    logger: Any | None,
) -> dict[str, Any]:
    system_config = settings.section("system")
    universe_config = system_config.get("market_universe", {}) if isinstance(system_config.get("market_universe"), dict) else {}
    mode = str(system_config.get("mode", "") or "").upper()
    live_history_warmup_enabled = bool(universe_config.get("live_history_warmup_enabled", False))
    if dry_run or not bool(universe_config.get("history_warmup_enabled", False)):
        return {"enabled": False, "symbols": 0, "timeframes": 0, "warmed": 0, "failed": 0, "skipped_reason": "disabled"}
    if mode == "LIVE" and not live_history_warmup_enabled:
        if logger is not None:
            logger.info("market_universe_live_history_warmup_skipped")
        return {"enabled": False, "symbols": 0, "timeframes": 0, "warmed": 0, "failed": 0, "skipped_reason": "live_startup_skip"}
    timeframe_counts = dict(universe_config.get("history_warmup_timeframes", {})) if isinstance(universe_config.get("history_warmup_timeframes"), dict) else {}
    ordered_timeframes = [(str(timeframe).upper(), max(50, int(count))) for timeframe, count in timeframe_counts.items()]
    max_symbols = max(1, int(universe_config.get("history_warmup_max_symbols", len(resolved_symbols))))
    warmed = 0
    failed = 0
    for configured_symbol in list(resolved_symbols.keys())[:max_symbols]:
        resolved_symbol = resolved_symbols[configured_symbol]
        for timeframe, count in ordered_timeframes:
            cached = market_data.load_cached(resolved_symbol, timeframe)
            if cached is not None and not cached.empty and len(cached.index) >= int(count * 0.85):
                continue
            try:
                market_data.fetch(resolved_symbol, timeframe, count)
                warmed += 1
            except Exception:
                failed += 1
    summary = {
        "enabled": True,
        "symbols": min(len(resolved_symbols), max_symbols),
        "timeframes": len(ordered_timeframes),
        "warmed": int(warmed),
        "failed": int(failed),
    }
    if logger is not None and (warmed > 0 or failed > 0):
        logger.info(f"market_universe_history_warmup {summary}")
    return summary


def _is_positive_number(value: Any) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return float(default)
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _symbol_point_size(symbol_info: dict[str, Any] | None) -> float:
    info = symbol_info if isinstance(symbol_info, dict) else {}
    for field_name in ("point", "trade_tick_size", "tick_size"):
        value = _safe_float(info.get(field_name), 0.0)
        if value > 0.0:
            return float(value)
    digits = info.get("digits")
    try:
        digits_value = int(digits)
    except (TypeError, ValueError):
        return 0.0
    if digits_value < 0:
        return 0.0
    return float(10 ** (-digits_value))


def _normalize_runtime_spread_points(
    symbol_key: str,
    spread_points: float,
    *,
    symbol_info: dict[str, Any] | None = None,
    max_spread_points: float = 60.0,
) -> float:
    raw_spread = max(0.0, float(spread_points or 0.0))
    if raw_spread <= 0.0:
        return 0.0
    normalized_symbol = _normalize_symbol_key(symbol_key)
    asset_class = str(symbol_asset_class(normalized_symbol) or "").lower()
    point_size = _symbol_point_size(symbol_info)
    if asset_class not in {"crypto", "equity"} or point_size <= 0.0 or point_size >= 1.0:
        return raw_spread
    spread_price = raw_spread * point_size
    base_cap = max(1.0, float(max_spread_points or 0.0))
    raw_ratio = raw_spread / base_cap
    price_ratio = spread_price / base_cap
    if raw_ratio >= 2.0 and price_ratio <= 2.0:
        return float(spread_price)
    return raw_spread


def _merge_bridge_symbol_snapshot(
    symbol_info: dict[str, Any] | None,
    bridge_symbol_snapshot: dict[str, Any] | None,
    *,
    symbol: str,
) -> tuple[dict[str, Any], str]:
    merged = dict(symbol_info or {})
    source_parts: list[str] = ["mt5_client" if merged else "empty"]
    if isinstance(bridge_symbol_snapshot, dict):
        bridge_overrides = {
            "volume_min": bridge_symbol_snapshot.get("lot_min"),
            "volume_max": bridge_symbol_snapshot.get("lot_max"),
            "volume_step": bridge_symbol_snapshot.get("lot_step"),
            "trade_tick_size": bridge_symbol_snapshot.get("tick_size"),
            "trade_tick_value": bridge_symbol_snapshot.get("tick_value"),
            "trade_contract_size": bridge_symbol_snapshot.get("contract_size"),
            "point": bridge_symbol_snapshot.get("point"),
            "digits": bridge_symbol_snapshot.get("digits"),
            "margin_initial": bridge_symbol_snapshot.get("margin"),
            "stops_level": bridge_symbol_snapshot.get("stops_level"),
            "freeze_level": bridge_symbol_snapshot.get("freeze_level"),
            "bid": bridge_symbol_snapshot.get("bid"),
            "ask": bridge_symbol_snapshot.get("ask"),
            "last": bridge_symbol_snapshot.get("last"),
        }
        applied = False
        for field_name, field_value in bridge_overrides.items():
            if field_name == "digits":
                if field_value is None:
                    continue
                merged[field_name] = _safe_int(field_value, _safe_int(merged.get(field_name), 0))
                applied = True
                continue
            if _is_positive_number(field_value):
                merged[field_name] = float(field_value)
                applied = True
        if applied:
            source_parts.append("bridge_snapshot")
    fallback_defaults = _symbol_family_defaults(symbol)
    fallback_fields: list[str] = []
    for field_name in ("volume_min", "volume_step", "point", "trade_tick_size"):
        if not _is_positive_number(merged.get(field_name)):
            merged[field_name] = fallback_defaults[field_name]
            fallback_fields.append(field_name)
    for optional_name in ("volume_max", "trade_contract_size", "trade_tick_value"):
        if not _is_positive_number(merged.get(optional_name)):
            merged[optional_name] = fallback_defaults[optional_name]
            fallback_fields.append(optional_name)
    if _safe_float(merged.get("volume_max"), 0.0) < _safe_float(merged.get("volume_min"), 0.01):
        merged["volume_max"] = max(_safe_float(merged.get("volume_min"), 0.01), float(fallback_defaults["volume_max"]))
        fallback_fields.append("volume_max")
    if fallback_fields:
        source_parts.append("fallback")
    return merged, "+".join(dict.fromkeys(source_parts))


def _resolve_runtime_symbol_info(
    *,
    mt5_client: Any,
    bridge_queue: Any | None,
    symbol_info_cache: dict[str, dict[str, Any]],
    symbol: str,
    bridge_trade_mode: bool,
    bridge_context_account: str | None,
    bridge_context_magic: int | None,
) -> dict[str, Any]:
    cached = symbol_info_cache.get(symbol)
    if isinstance(cached, dict):
        return cached

    raw_symbol_info = mt5_client.get_symbol_info(symbol)
    bridge_symbol_snapshot = None
    if bridge_trade_mode and bridge_context_account and bridge_context_magic:
        bridge_symbol_snapshot = _resolve_bridge_symbol_snapshot(
            bridge_queue,
            account=bridge_context_account,
            symbol=symbol,
            magic=bridge_context_magic,
        )
    resolved_symbol_info, economics_source = _merge_bridge_symbol_snapshot(
        raw_symbol_info,
        bridge_symbol_snapshot,
        symbol=symbol,
    )
    resolved_symbol_info["economics_source"] = economics_source
    resolved_symbol_info["bridge_snapshot"] = bridge_symbol_snapshot or {}
    symbol_info_cache[symbol] = resolved_symbol_info
    return resolved_symbol_info


def _clean_snapshot_meta(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        cleaned[str(key)] = value
    return cleaned


def _symbol_rule_with_snapshot_overrides(rule: SymbolRule, snapshot: dict[str, Any] | None) -> SymbolRule:
    snapshot_meta = _clean_snapshot_meta(snapshot)
    if not snapshot_meta:
        return rule

    def _pos_float(key: str, fallback: float) -> float:
        value = snapshot_meta.get(key)
        try:
            number = float(value)
        except (TypeError, ValueError):
            return fallback
        return number if number > 0 else fallback

    def _pos_int(key: str, fallback: int) -> int:
        value = snapshot_meta.get(key)
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return fallback
        return number if number >= 0 else fallback

    return replace(
        rule,
        digits=_pos_int("digits", int(rule.digits)),
        tick_size=_pos_float("tick_size", float(rule.tick_size)),
        point=_pos_float("point", float(rule.point)),
        min_stop_points=_pos_int("stops_level", int(rule.min_stop_points)),
        freeze_points=_pos_int("freeze_level", int(rule.freeze_points)),
        typical_spread_points=_pos_int("spread_points", int(rule.typical_spread_points)),
        tick_value=_pos_float("tick_value", float(rule.tick_value)),
        contract_size=_pos_float("contract_size", float(rule.contract_size)),
    )


def _normalize_pre_risk_exit_geometry(
    *,
    symbol: str,
    side: str,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    spread_points: float,
    symbol_info: dict[str, Any],
    symbol_rules: dict[str, SymbolRule],
    safety_buffer_points: int,
) -> dict[str, Any]:
    rule = _symbol_rule_with_snapshot_overrides(
        resolve_symbol_rule(symbol, symbol_rules),
        {
            "digits": symbol_info.get("digits"),
            "tick_size": symbol_info.get("trade_tick_size"),
            "point": symbol_info.get("point"),
            "stops_level": symbol_info.get("stops_level"),
            "freeze_level": symbol_info.get("freeze_level"),
            "spread_points": spread_points,
            "tick_value": symbol_info.get("trade_tick_value"),
            "contract_size": symbol_info.get("trade_contract_size"),
        },
    )
    validation = validate_and_normalize_stops(
        StopValidationInput(
            symbol=symbol,
            side=side,
            entry_price=float(entry_price),
            sl=float(stop_price),
            tp=float(tp_price),
            spread_points=float(spread_points),
            live_bid=float(symbol_info.get("bid") or 0.0),
            live_ask=float(symbol_info.get("ask") or 0.0),
            safety_buffer_points=max(0, int(safety_buffer_points)),
            allow_tp_none=True,
            push_sl_if_too_close=True,
        ),
        rule,
    )
    min_required_points = float(
        validation.min_required_points
        if validation.min_required_points > 0
        else (float(rule.min_stop_points + rule.freeze_points) + max(float(spread_points), float(rule.typical_spread_points)) + max(0, int(safety_buffer_points)))
    )
    normalized_stop = float(validation.normalized_sl) if validation.valid and validation.normalized_sl is not None else float(stop_price)
    normalized_tp = (
        float(validation.normalized_tp)
        if validation.valid and validation.normalized_tp is not None
        else float(tp_price)
    )
    tick_size = max(_safe_float(rule.tick_size, 0.0), _safe_float(rule.point, 0.0), 1e-9)
    return {
        "rule": rule,
        "validation_reason": str(validation.reason),
        "validated": bool(validation.valid),
        "clamped": (
            abs(normalized_stop - float(stop_price)) > (tick_size * 0.5)
            or abs(normalized_tp - float(tp_price)) > (tick_size * 0.5)
        ),
        "stop_price": normalized_stop,
        "tp_price": normalized_tp,
        "stop_distance": abs(float(entry_price) - normalized_stop),
        "min_stop_distance_points": min_required_points,
        "validation_snapshot": validation.snapshot(entry_price=float(entry_price)),
    }


def _resolve_bridge_symbol_snapshot(
    bridge_queue: Any,
    *,
    account: str,
    magic: int,
    symbol: str,
) -> dict[str, Any] | None:
    if bridge_queue is None:
        return None
    snapshot = None
    if hasattr(bridge_queue, "get_account_snapshot"):
        try:
            snapshot = bridge_queue.get_account_snapshot(
                account=account,
                symbol=symbol,
                magic=magic,
            )
        except Exception:
            snapshot = None
    if snapshot is None and hasattr(bridge_queue, "latest_account_snapshot"):
        try:
            snapshot = bridge_queue.latest_account_snapshot(
                account=account,
                magic=magic,
                symbol=symbol,
            )
        except Exception:
            snapshot = None
    return snapshot if isinstance(snapshot, dict) else None


def _parse_iso_utc(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _snapshot_is_recent(snapshot: dict[str, Any] | None, now_ts: datetime, stale_after_seconds: float) -> bool:
    if not isinstance(snapshot, dict):
        return False
    updated_at = _parse_iso_utc(snapshot.get("updated_at"))
    if updated_at is None:
        return False
    return max(0.0, (now_ts - updated_at).total_seconds()) <= max(1.0, float(stale_after_seconds))


def _stale_broker_position_reason(
    *,
    position: dict[str, Any],
    account_snapshot: dict[str, Any] | None,
    symbol_snapshot: dict[str, Any] | None,
    now_ts: datetime,
    stale_after_seconds: float = 180.0,
    grace_seconds: float = 120.0,
) -> str | None:
    opened_at = _parse_iso_utc(position.get("opened_at"))
    if opened_at is not None:
        age_seconds = max(0.0, (now_ts - opened_at).total_seconds())
        if age_seconds < max(1.0, float(grace_seconds)):
            return None
    if _snapshot_is_recent(account_snapshot, now_ts, stale_after_seconds):
        total_open_positions = _safe_int((account_snapshot or {}).get("total_open_positions"), -1)
        if total_open_positions == 0:
            return "broker_account_flat"
    if _snapshot_is_recent(symbol_snapshot, now_ts, stale_after_seconds):
        symbol_open_count = _safe_int((symbol_snapshot or {}).get("open_count"), -1)
        if symbol_open_count == 0:
            return "broker_symbol_flat"
    return None


def _filter_broker_confirmed_positions(
    *,
    positions: list[dict[str, Any]],
    account_snapshot: dict[str, Any] | None,
    symbol_snapshots: dict[str, dict[str, Any] | None],
    now_ts: datetime,
    stale_after_seconds: float = 180.0,
    grace_seconds: float = 120.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    confirmed: list[dict[str, Any]] = []
    stale: list[dict[str, Any]] = []
    for position in positions:
        symbol_key = _normalize_symbol_key(str(position.get("symbol", "")))
        symbol_snapshot = symbol_snapshots.get(symbol_key)
        stale_reason = _stale_broker_position_reason(
            position=position,
            account_snapshot=account_snapshot,
            symbol_snapshot=symbol_snapshot,
            now_ts=now_ts,
            stale_after_seconds=stale_after_seconds,
            grace_seconds=grace_seconds,
        )
        if stale_reason:
            stale.append(
                {
                    "position": position,
                    "reason": stale_reason,
                    "symbol_snapshot": symbol_snapshot,
                }
            )
            continue
        confirmed.append(position)
    return confirmed, stale


def _effective_min_stop_distance_points(
    configured_floor_points: float,
    live_required_points: float,
) -> float:
    live_required = max(0.0, float(live_required_points))
    if live_required > 0:
        return live_required
    return max(0.0, float(configured_floor_points))


def _resolve_candidate_stop_distance(
    *,
    candidate: Any,
    atr_for_candidate: float,
    point_size: float,
    sl_multiplier: float,
) -> tuple[float, str]:
    meta = candidate.meta if isinstance(getattr(candidate, "meta", None), dict) else {}
    explicit_distance_price = _safe_float(meta.get("stop_distance_price"), 0.0)
    if explicit_distance_price > 0:
        return float(explicit_distance_price), "explicit_price"
    explicit_stop_points = _safe_float(meta.get("stop_points"), 0.0)
    if explicit_stop_points > 0 and point_size > 0:
        return float(explicit_stop_points) * float(point_size), "explicit_points"
    return max(0.0, float(atr_for_candidate) * float(candidate.stop_atr) * max(float(sl_multiplier), 0.0)), "atr"


def _effective_live_entry_price(
    *,
    side: str,
    tick: dict[str, Any] | None,
    symbol_info: dict[str, Any] | None,
) -> tuple[float, str]:
    price_key = "ask" if str(side).upper() == "BUY" else "bid"
    info = symbol_info if isinstance(symbol_info, dict) else {}
    tick_payload = tick if isinstance(tick, dict) else {}
    bridge_price = _safe_float(info.get(price_key), 0.0)
    tick_price = _safe_float(tick_payload.get(price_key), 0.0)
    if bridge_price > 0:
        source = "bridge_snapshot" if "bridge_snapshot" in str(info.get("economics_source", "")) else "symbol_info"
        return bridge_price, source
    if tick_price > 0:
        return tick_price, "tick"
    return 0.0, "missing"


def _prep_checks_complete(news_snapshot: dict[str, Any], router_diagnostics: dict[str, Any], market_open: bool) -> bool:
    return bool(market_open or news_snapshot or router_diagnostics)


def _effective_symbol_interval_seconds(
    *,
    symbol: str,
    session_name: str,
    fast_symbols: set[str],
    fast_sessions: set[str],
    fast_seconds: float,
    default_seconds: float,
) -> float:
    normalized_symbol = _normalize_symbol_key(symbol)
    if normalized_symbol in fast_symbols and session_name in fast_sessions:
        return max(0.5, fast_seconds)
    return max(0.5, default_seconds)


def _is_weekend_market_mode(now: datetime) -> bool:
    return bool(is_weekend_market_mode(now))


def _market_open_status(symbol_key: str, now_ts: datetime) -> tuple[bool, str]:
    return market_open_tuple(symbol_key, now_ts)


def _approve_small_session_allowed(symbol_key: str, session_name: str, now_ts: datetime) -> bool:
    normalized_symbol = _normalize_symbol_key(symbol_key)
    session_upper = str(session_name or "").upper()
    asset_class = symbol_asset_class(normalized_symbol)
    if session_upper in {"LONDON", "OVERLAP", "NEW_YORK"}:
        return True
    if asset_class == "crypto":
        return session_upper in {"TOKYO", "SYDNEY"} or _is_weekend_market_mode(now_ts)
    if asset_class in {"forex", "commodity"}:
        return session_upper in {"TOKYO", "SYDNEY"}
    return False


def _resolve_timeframe_route(
    symbol_key: str,
    requested_timeframe: str,
    bridge_orchestrator: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalized_symbol = _normalize_symbol_key(symbol_key)
    requested_tf = str(requested_timeframe or "M15").upper()
    orchestrator = bridge_orchestrator if isinstance(bridge_orchestrator, dict) else {}
    asset_class = symbol_asset_class(normalized_symbol)
    default_routes: dict[str, dict[str, Any]] = {
        "XAUUSD": {"accepted": ("M15", "M5", "M3", "M1"), "execution": "M3", "internal": ("M1", "M3", "M5", "M15")},
        "BTCUSD": {"accepted": ("M15", "M5", "M3", "M1"), "execution": "M3", "internal": ("M1", "M3", "M5", "M15")},
        "EURUSD": {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M5", "M15")},
        "GBPUSD": {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M5", "M15")},
        "USDJPY": {"accepted": ("M15", "M5", "M3", "M1"), "execution": "M3", "internal": ("M1", "M3", "M5", "M15")},
        "AUDJPY": {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M5", "M15")},
        "NZDJPY": {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M5", "M15")},
        "AUDNZD": {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M5", "M15")},
        "EURJPY": {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M5", "M15")},
        "GBPJPY": {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M5", "M15")},
        "NAS100": {"accepted": ("M15", "M5", "M3", "M1"), "execution": "M3", "internal": ("M1", "M3", "M5", "M15")},
        "USOIL": {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M5", "M15")},
    }
    generic_route = {"accepted": (requested_tf,), "execution": requested_tf, "internal": (requested_tf,)}
    if asset_class == "crypto":
        generic_route = {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M1", "M5", "M15")}
    elif asset_class == "commodity":
        generic_route = {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M5", "M15", "H1")}
    elif asset_class == "index":
        generic_route = {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M5", "M15")}
    elif asset_class == "equity":
        generic_route = {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M5", "M15", "H1")}
    elif asset_class == "forex":
        generic_route = {"accepted": ("M15", "M5"), "execution": "M15", "internal": ("M5", "M15")}
    route = dict(default_routes.get(normalized_symbol, generic_route))
    configured_routes = orchestrator.get("symbol_timeframe_map", {}) if isinstance(orchestrator.get("symbol_timeframe_map"), dict) else {}
    configured_symbol_route = configured_routes.get(normalized_symbol, {})
    if isinstance(configured_symbol_route, dict) and configured_symbol_route:
        route.update({key: value for key, value in configured_symbol_route.items() if value})
    preferred_tfs = orchestrator.get("preferred_execution_tf_by_symbol", {}) if isinstance(orchestrator.get("preferred_execution_tf_by_symbol"), dict) else {}
    execution_tf = str(preferred_tfs.get(normalized_symbol, route.get("execution", requested_tf))).upper()
    accepted = tuple(str(item).upper() for item in route.get("accepted", ()) if str(item).strip()) or (execution_tf,)
    internal = tuple(str(item).upper() for item in route.get("internal", ()) if str(item).strip())
    if not internal:
        configured_internal = orchestrator.get("internal_analysis_tfs_by_symbol", {}) if isinstance(orchestrator.get("internal_analysis_tfs_by_symbol"), dict) else {}
        internal = tuple(str(item).upper() for item in configured_internal.get(normalized_symbol, ()) if str(item).strip()) or (execution_tf,)
    fallback_enabled = bool(orchestrator.get("attached_tf_fallback_enabled", True))
    fallback_used = bool(fallback_enabled and requested_tf not in set(accepted))
    execution_timeframe_used = execution_tf if (fallback_enabled or requested_tf in set(accepted)) else requested_tf
    attachment_dependency_resolved = bool(
        fallback_enabled and (requested_tf != execution_timeframe_used or any(tf != requested_tf for tf in internal))
    )
    return {
        "requested_timeframe": requested_tf,
        "accepted_execution_tfs": list(accepted),
        "execution_timeframe_used": execution_timeframe_used,
        "internal_timeframes_used": list(dict.fromkeys(internal)),
        "attachment_dependency_resolved": attachment_dependency_resolved,
        "attached_tf_fallback_used": fallback_used,
    }


def _is_pre_exec_block_reason(reason: str) -> bool:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return False
    prefixes = (
        "risk_budget_exceeded",
        "trade_plan_risk_exceeded",
        "bootstrap_trade_risk_exceeds_cap",
        "bootstrap_total_risk_exceeds_cap",
        "margin_insufficient",
        "insufficient_margin",
        "min_lot_over_budget",
        "candidate_rejected_pre_exec",
        "delivery_budget_mismatch",
        "spread_disorder",
        "news_lock",
        "micro_survival",
        "invalid_stop_distance",
        "final_stop_distance_invalid",
        "final_volume_zero",
    )
    return normalized.startswith(prefixes)


def _state_from_reason(reason: str) -> str:
    normalized = str(reason or "").strip().lower()
    if not normalized:
        return "SCANNING"
    if normalized in {"market_closed", "weekend_closed", "weekend_closed_block"} or normalized.endswith("_closed"):
        return "CLOSED"
    if normalized in {"outside_configured_session", "session_ineligible", "xau_session_block", "friday_flat_window"}:
        return "SESSION_BLOCK"
    if normalized.startswith("omega_"):
        return "REGIME_BLOCK"
    if normalized in {"idea_cooldown_active", "idea_recheck_wait", "duplicate_or_active_signal"}:
        return "COOLDOWN"
    if _is_pre_exec_block_reason(normalized):
        return "PRECHECK_FAIL"
    if normalized in {"queued_for_ea", "xau_grid_probe"}:
        return "QUEUED_FOR_EA"
    if normalized in {"executed", "paper_sim_executed"}:
        return "DELIVERED"
    if normalized in {"no_base_candidate", "ready_recheck", "scanning"}:
        return "SCANNING"
    return "BLOCK"


def _market_state_fields(symbol_key: str, now_ts: datetime) -> dict[str, Any]:
    state = describe_market_state(symbol_key, now_ts)
    return {
        "market_open": bool(state.market_open),
        "market_open_status": str(state.market_open_status),
        "next_open_time_utc": str(state.next_open_time_utc or ""),
        "next_open_time_local": str(state.next_open_time_local or ""),
        "pre_open_window_active": bool(state.pre_open_window_active),
        "dst_mode_active": dict(state.dst_mode_active),
        "dominant_session_name": str(state.dominant_session_name),
    }


def _hourly_learning_review(
    *,
    now_ts: datetime,
    journal: TradeJournal,
    runtime_snapshot: dict[str, dict[str, Any]],
    account: str | None,
    magic: int | None,
) -> dict[str, dict[str, Any]]:
    symbols = [
        "BTCUSD",
        "XAUUSD",
        "XAGUSD",
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "AUDJPY",
        "NZDJPY",
        "AUDNZD",
        "EURJPY",
        "GBPJPY",
        "NAS100",
        "USOIL",
        "DOGUSD",
        "TRUMPUSD",
        "AAPL",
        "NVIDIA",
    ]
    dynamic_symbols = {
        _normalize_symbol_key(symbol_key)
        for symbol_key in runtime_snapshot.keys()
        if _normalize_symbol_key(symbol_key)
    }
    symbols = list(dict.fromkeys([*symbols, *sorted(dynamic_symbols)]))
    output: dict[str, dict[str, Any]] = {}
    for symbol_key in symbols:
        runtime = dict(runtime_snapshot.get(symbol_key, {}))
        review = journal.recent_review_summary(limit=25, symbol=symbol_key, account=account, magic=magic)
        perf = journal.summary_last(20, symbol=symbol_key)
        candidate_attempts = int(runtime.get("candidate_attempts_last_15m", 0))
        candidate_count = int(runtime.get("candidate_count_last_15m", 0))
        no_candidate = str(runtime.get("last_reject_reason", "")) == "no_base_candidate"
        sensitivity_multiplier = 1.0
        promoted: list[str] = []
        suppressed: list[str] = []
        missed_summary = "balanced"
        if candidate_attempts == 0 and no_candidate:
            sensitivity_multiplier = 1.12
            missed_summary = "candidate_starvation"
            promoted.append("PRICE_ACTION_FALLBACK")
        elif candidate_attempts > 0 and candidate_count == 0:
            sensitivity_multiplier = 1.06
            missed_summary = "attempted_without_candidate"
            promoted.append("STRUCTURE_RECHECK")
        elif float(perf.get("trades", 0.0)) >= 4.0 and float(perf.get("win_rate", 0.0)) < 0.40:
            sensitivity_multiplier = 0.94
            missed_summary = "protect_after_weak_recent_win_rate"
            suppressed.append("LOW_QUALITY_CONTINUATION")
        elif float(perf.get("trades", 0.0)) >= 4.0 and float(perf.get("win_rate", 0.0)) >= 0.60:
            sensitivity_multiplier = 1.05
            missed_summary = "promote_recent_winners"
            promoted.append("WINNING_SESSION_FAMILY")
        output[symbol_key] = {
            "reviewed_at": now_ts.isoformat(),
            "hourly_learning_summary": review,
            "pair_hourly_review": {
                "symbol": symbol_key,
                "recent_trades": float(perf.get("trades", 0.0)),
                "recent_win_rate": float(perf.get("win_rate", 0.0)),
                "recent_expectancy": float(perf.get("expectancy_r", 0.0)),
            },
            "recent_missed_opportunity_summary": missed_summary,
            "hourly_parameter_adjustments": {
                "candidate_sensitivity_mult": sensitivity_multiplier,
            },
            "setup_families_promoted": promoted,
            "setup_families_suppressed": suppressed,
        }
    return output


def _micro_position_caps(
    micro_config: dict[str, Any],
    mode: str,
    equity: float,
    base_total: int,
    base_per_symbol: int,
) -> tuple[int, int, bool]:
    micro_enabled = bool(micro_config.get("enabled", False)) and mode.upper() in {"DEMO", "PAPER", "LIVE"}
    if not micro_enabled:
        return base_total, base_per_symbol, False
    bootstrap_enabled = bool(micro_config.get("bootstrap_enabled", True))
    bootstrap_equity_threshold = float(micro_config.get("bootstrap_equity_threshold", 160.0))
    if bootstrap_enabled and equity <= bootstrap_equity_threshold:
        bootstrap_total_cap = max(1, int(micro_config.get("bootstrap_max_positions_total", 3)))
        bootstrap_symbol_cap = max(1, int(micro_config.get("bootstrap_max_positions_per_symbol", 2)))
        return min(base_total, bootstrap_total_cap), min(base_per_symbol, bootstrap_symbol_cap), True
    micro_total_cap = max(1, int(micro_config.get("max_positions_total_micro", 1)))
    micro_symbol_cap = max(1, int(micro_config.get("max_positions_per_symbol_micro", 1)))
    if not bool(micro_config.get("one_trade_at_a_time", True)):
        return min(base_total, micro_total_cap), min(base_per_symbol, micro_symbol_cap), True

    one_trade_until = float(micro_config.get("one_trade_until_equity", 50.0))
    two_trade_until = float(micro_config.get("two_trade_until_equity", 100.0))
    if equity <= one_trade_until:
        return min(base_total, micro_total_cap), min(base_per_symbol, micro_symbol_cap), True
    if equity < two_trade_until:
        return (
            min(base_total, max(micro_total_cap, int(micro_config.get("max_positions_total_mid", 2)))),
            min(base_per_symbol, max(micro_symbol_cap, int(micro_config.get("max_positions_per_symbol_mid", 1)))),
            True,
        )
    return base_total, base_per_symbol, True


def _micro_lot_cap(
    micro_config: dict[str, Any],
    mode: str,
    equity: float,
) -> float | None:
    micro_enabled = bool(micro_config.get("enabled", False)) and mode.upper() in {"DEMO", "PAPER", "LIVE"}
    if not micro_enabled:
        return None
    mid_equity = float(micro_config.get("lot_cap_mid_equity", 50.0))
    high_equity = float(micro_config.get("lot_cap_high_equity", 100.0))
    if equity < mid_equity:
        return float(micro_config.get("max_lot_cap_low", 0.01))
    if equity < high_equity:
        return float(micro_config.get("max_lot_cap_mid", 0.02))
    return float(micro_config.get("max_lot_cap_high", 0.05))


def _preserve_approved_broker_min_lot(
    *,
    normalized_volume: float,
    approved_volume: float,
    broker_min_lot: float,
    preserve_min_lot: bool,
) -> float:
    if not preserve_min_lot:
        return float(normalized_volume)
    if float(approved_volume) + 1e-9 < float(broker_min_lot):
        return float(normalized_volume)
    if float(normalized_volume) + 1e-9 >= float(broker_min_lot):
        return float(normalized_volume)
    return float(broker_min_lot)


def determine_trading_state(
    mode: str,
    system_config: dict,
    live_allowed: bool,
    verify_only: bool = False,
) -> tuple[bool, str]:
    normalized = mode.upper()
    if verify_only:
        return False, "verify_mode"
    if normalized == "DRY_RUN":
        return False, "dry_run_mode"
    if normalized in {"DEMO", "PAPER"}:
        return (bool(system_config.get("trading_enabled", False)), "trading_enabled" if bool(system_config.get("trading_enabled", False)) else "trading_disabled")
    if normalized == "LIVE":
        if not bool(system_config.get("trading_enabled", False)):
            message = "LIVE BLOCKED: system.trading_enabled=false"
            print(message)
            return False, "live_blocked_trading_disabled"
        if not bool(system_config.get("live_trading_enabled", False)):
            message = "LIVE BLOCKED: system.live_trading_enabled=false"
            print(message)
            return False, "live_blocked_config_flag"
        return True, "live_unlocked"
    return False, f"unsupported_mode:{normalized}"


def build_runtime(force_mt5: bool = False, skip_mt5: bool = False):
    settings = load_settings()
    logging_config = settings.section("logging") if isinstance(settings.raw.get("logging"), dict) else {}
    logger = LoggerFactory(
        log_file=settings.runtime_paths.logs_dir / "apex.log",
        rotate_max_bytes=int(logging_config.get("rotate_max_bytes", 10 * 1024 * 1024)),
        rotate_backup_count=int(logging_config.get("rotate_backup_count", 7)),
        retention_days=int(logging_config.get("retention_days", 365)),
    ).build()
    isolated_runtime_root: Path | None = None
    if skip_mt5:
        isolated_runtime_root = Path(settings.resolve_path_value("data/paper_sim"))
        isolated_runtime_root.mkdir(parents=True, exist_ok=True)
    journal_db_path = (
        isolated_runtime_root / "trades_db.sqlite"
        if isolated_runtime_root is not None
        else settings.path("data.trade_db")
    )
    journal = TradeJournal(journal_db_path)
    system_config = settings.section("system")
    mode = str(system_config["mode"]).upper()
    dry_run = (skip_mt5 or _is_dry_run_mode(mode)) and not force_mt5

    default_terminal_path = str(system_config.get("mt5_terminal_path", "") or "") or None
    symbol_mapping = system_config.get("symbol_mapping", {})
    mt5_client = MT5Client(
        credentials=MT5Credentials.from_env(default_terminal_path=default_terminal_path),
        journal_db=journal_db_path,
        logger=logger,
        disable_mt5=dry_run,
        symbol_mapping=symbol_mapping if isinstance(symbol_mapping, dict) else {},
    )

    configured_symbols = _dedupe_symbols(settings.symbols())
    configured_symbols, discovered_universe = _expand_market_universe(
        settings=settings,
        mt5_client=mt5_client,
        configured_symbols=configured_symbols,
        dry_run=dry_run,
        logger=logger,
    )
    if dry_run:
        resolved_symbols = {symbol: symbol for symbol in configured_symbols}
    else:
        try:
            resolved_symbols = mt5_client.resolve_symbols(configured_symbols)
        except Exception as exc:
            logger.warning(f"MT5 symbol resolution fallback: {exc}")
            resolved_symbols = {symbol: symbol for symbol in configured_symbols}

    primary_symbol = resolved_symbols[configured_symbols[0]]
    market_data = MarketDataService(mt5_client, settings.path("data.candles_cache_dir"))
    bridge_config = settings.raw.get("bridge", {}) if isinstance(settings.raw.get("bridge"), dict) else {}
    symbol_rules = load_symbol_rules(settings.resolve_path_value(str(bridge_config.get("symbol_rules_file", "config/symbol_rules.yaml"))))
    bridge_queue_path = (
        isolated_runtime_root / "bridge_actions.sqlite"
        if isolated_runtime_root is not None
        else Path(settings.resolve_path_value(str(bridge_config.get("queue_db", "data/bridge_actions.sqlite"))))
    )
    bridge_ttl = int(bridge_config.get("poll_ttl_seconds", 10))
    from src.bridge_server import BridgeActionQueue  # lazy import to avoid optional dependency load in tests

    bridge_queue = BridgeActionQueue(
        db_path=bridge_queue_path,
        ttl_seconds=bridge_ttl,
        logger=logger,
    )
    bridge_orchestrator_config = dict(bridge_config.get("orchestrator", {}) if isinstance(bridge_config.get("orchestrator"), dict) else {})
    for section_name in (
        "microstructure",
        "lead_lag",
        "event_playbooks",
        "aggression",
        "shadow_promotion",
        "execution_memory",
        "self_heal",
        "training_bootstrap",
        "institutional_features",
        "edge_promotion",
        "frequency_policy",
        "funded_mode",
    ):
        if section_name not in bridge_orchestrator_config and isinstance(settings.raw.get(section_name), dict):
            bridge_orchestrator_config[section_name] = dict(settings.raw.get(section_name) or {})
    strategy_optimizer_payload = bridge_orchestrator_config.get("strategy_optimizer", {}) if isinstance(bridge_orchestrator_config.get("strategy_optimizer"), dict) else {}
    strategy_optimizer = StrategyOptimizer(
        trade_history_path=(
            isolated_runtime_root / "trade_history.json"
            if isolated_runtime_root is not None
            else Path(settings.resolve_path_value(str(strategy_optimizer_payload.get("trade_history_path", "data/trade_history.json"))))
        ),
        metrics_path=(
            isolated_runtime_root / "strategy_metrics.json"
            if isolated_runtime_root is not None
            else Path(settings.resolve_path_value(str(strategy_optimizer_payload.get("metrics_path", "data/strategy_metrics.json"))))
        ),
        enabled=bool(strategy_optimizer_payload.get("enabled", True)),
        min_trades_per_strategy=max(5, int(strategy_optimizer_payload.get("min_trades_per_strategy", 50))),
        lookback_days=max(7, int(strategy_optimizer_payload.get("lookback_days", 90))),
        low_win_rate_threshold=clamp(float(strategy_optimizer_payload.get("low_win_rate_threshold", 0.45)), 0.1, 0.9),
        high_win_rate_threshold=clamp(float(strategy_optimizer_payload.get("high_win_rate_threshold", 0.60)), 0.1, 0.95),
        adjustment_step_pct=clamp(float(strategy_optimizer_payload.get("adjustment_step_pct", 0.10)), 0.01, 0.20),
        min_adjustment_multiplier=clamp(float(strategy_optimizer_payload.get("min_adjustment_multiplier", 0.75)), 0.50, 1.0),
        max_adjustment_multiplier=clamp(float(strategy_optimizer_payload.get("max_adjustment_multiplier", 1.25)), 1.0, 2.0),
        cooldown_minutes=max(1, int(strategy_optimizer_payload.get("cooldown_minutes", 30))),
        logger=logger,
    )
    strategy_optimizer.register_strategies(
        [
            "XAUUSD_M5_GRID",
            "XAUUSD_M1_MICRO",
            "XAUUSD_M15_STRUCTURED",
            "XAUUSD_NON_GRID",
            "EURUSD_LONDON_BREAKOUT",
            "GBPUSD_BREAKER_BLOCK",
            "USDJPY_SESSION_SCALP",
            "BTC_TREND_SCALP",
            "BTC_WEEKEND_CANARY",
            "NAS_SESSION_SCALPER",
            "OIL_INVENTORY_SCALPER",
            "FORCE_TEST_TRADE",
            *[f"{_normalize_symbol_key(symbol)}_MULTI" for symbol in configured_symbols],
        ]
    )
    online_learning = OnlineLearningEngine(
        data_path=(
            isolated_runtime_root / "trades.csv"
            if isolated_runtime_root is not None
            else Path(settings.resolve_path_value(str(bridge_config.get("trades_dataset", "data/trades.csv"))))
        ),
        model_path=(
            isolated_runtime_root / "online_model.pkl"
            if isolated_runtime_root is not None
            else Path(settings.resolve_path_value(str(bridge_config.get("online_model_path", "models/online_model.pkl"))))
        ),
        history_cache_dir=Path(settings.resolve_path_value(str(settings.section("ai").get("market_history_cache_dir", "data/candles_cache")))),
        min_retrain_trades=int(settings.section("ai").get("online_learning_min_samples", 50)),
        rolling_window_trades=int(settings.section("ai").get("online_window_trades", 1000)),
        maintenance_interval_hours=int(settings.section("ai").get("online_maintenance_interval_hours", 4)),
        promotion_min_delta=float(settings.section("ai").get("online_promotion_min_delta", 0.02)),
        promotion_min_samples=int(settings.section("ai").get("online_promotion_min_samples", 100)),
        market_history_seed_enabled=bool(settings.section("ai").get("market_history_seed_enabled", True)),
        market_history_seed_max_samples=int(settings.section("ai").get("market_history_seed_max_samples", 3000)),
        market_history_seed_forward_bars=int(settings.section("ai").get("market_history_seed_forward_bars", 8)),
        market_history_seed_min_rows_per_file=int(settings.section("ai").get("market_history_seed_min_rows_per_file", 240)),
        market_history_backfill_enabled=bool(settings.section("ai").get("market_history_backfill_enabled", True)),
        market_history_backfill_years=int(settings.section("ai").get("market_history_backfill_years", 10)),
        market_history_universe_symbols=tuple(configured_symbols),
        logger=logger,
    )
    history_warmup_summary = _warm_market_universe_history(
        settings=settings,
        market_data=market_data,
        resolved_symbols=resolved_symbols,
        dry_run=dry_run,
        logger=logger,
    )
    learning_data_dir = (
        isolated_runtime_root / "learning_brain"
        if isolated_runtime_root is not None
        else Path(
            settings.resolve_path_value(
                str(bridge_orchestrator_config.get("learning_brain_data_dir", Path(str(bridge_config.get("trades_dataset", "data/trades.csv"))).parent))
            )
        )
    )
    session_profile = SessionProfile.from_config(settings.section("sessions") if isinstance(settings.raw.get("sessions"), dict) else {})
    router_config = settings.section("strategy_router") if isinstance(settings.raw.get("strategy_router"), dict) else {}
    btc_router_config = router_config.get("btc", {}) if isinstance(router_config.get("btc"), dict) else {}
    candidate_tier_config = router_config.get("candidate_tiers", {}) if isinstance(router_config.get("candidate_tiers"), dict) else {}
    spread_profile_config = router_config.get("spread_profiles", {}) if isinstance(router_config.get("spread_profiles"), dict) else {}
    xau_engine_config = settings.section("xau_multi_engine") if isinstance(settings.raw.get("xau_multi_engine"), dict) else {}
    xau_m1_config = xau_engine_config.get("m1", {}) if isinstance(xau_engine_config.get("m1"), dict) else {}
    xau_m15_config = xau_engine_config.get("m15", {}) if isinstance(xau_engine_config.get("m15"), dict) else {}
    candidate_sensitivity = settings.section("candidate_sensitivity") if isinstance(settings.raw.get("candidate_sensitivity"), dict) else {}
    strategy_router = StrategyRouter(
        density_profiles={
            (
                "__default__"
                if str(symbol).strip() == "__default__"
                else (
                    f"asset:{str(symbol).split(':', 1)[1].strip().lower()}"
                    if str(symbol).strip().lower().startswith("asset:")
                    else StrategyRouter._normalize_symbol(str(symbol))
                )
            ): dict(profile)
            for symbol, profile in (
                candidate_tier_config.get("density_profiles", {})
                if isinstance(candidate_tier_config.get("density_profiles"), dict)
                else {}
            ).items()
            if str(symbol).strip() and isinstance(profile, dict)
        },
        max_spread_points=float(router_config.get("max_spread_points", settings.section("risk").get("max_spread_points", 60))),
        xau_active_sessions=tuple(
            str(item).upper()
            for item in xau_engine_config.get("active_sessions", ["TOKYO", "LONDON", "OVERLAP", "NEW_YORK"])
            if str(item).strip()
        ),
        xau_m1_enabled=bool(xau_m1_config.get("enabled", True)),
        xau_m15_enabled=bool(xau_m15_config.get("enabled", True)),
        xau_m1_min_impulse_ratio=float(xau_m1_config.get("min_impulse_ratio", 0.48)),
        xau_m1_min_volume_ratio=float(xau_m1_config.get("min_volume_ratio", 1.0)),
        xau_m1_confluence_floor=float(xau_m1_config.get("confluence_floor", 3.0)),
        xau_m15_confluence_floor=float(xau_m15_config.get("confluence_floor", 4.0)),
        xau_m15_breakout_atr_threshold=float(xau_m15_config.get("breakout_atr_threshold", 0.20)),
        forex_sensitivity=float(candidate_sensitivity.get("forex", 1.0)),
        nas_sensitivity=float(candidate_sensitivity.get("nas100", 1.0)),
        oil_sensitivity=float(candidate_sensitivity.get("usoil", 1.0)),
        btc_sensitivity=float(candidate_sensitivity.get("btc", 1.0)),
        btc_trade_sessions=tuple(
            str(item).upper()
            for item in btc_router_config.get("trade_sessions", ["TOKYO", "LONDON", "OVERLAP", "NEW_YORK"])
            if str(item).strip()
        ),
        btc_allowed_start_hour_utc=int(btc_router_config.get("allowed_start_hour_utc", 0)),
        btc_allowed_end_hour_utc=int(btc_router_config.get("allowed_end_hour_utc", 24)),
        btc_spread_cap_points=float(btc_router_config.get("spread_cap_points", 2500.0)),
        btc_min_ai_confidence=float(btc_router_config.get("min_ai_confidence", 0.56)),
        btc_volatility_pause_move_pct_30m=float(btc_router_config.get("volatility_pause_move_pct_30m", 0.015)),
        btc_funding_rate_min_abs=float(btc_router_config.get("funding_rate_min_abs", 0.001)),
        btc_liquidation_usd_threshold=float(btc_router_config.get("liquidation_usd_threshold", 10_000_000.0)),
        btc_whale_flow_threshold_btc=float(btc_router_config.get("whale_flow_threshold_btc", 5000.0)),
        btc_dxy_move_threshold=float(btc_router_config.get("dxy_move_threshold", 0.003)),
        btc_weekend_gap_min_pct=float(btc_router_config.get("weekend_gap_min_pct", 0.02)),
        b_tier_size_mult_min=float(candidate_tier_config.get("b_tier_size_mult_min", 0.70)),
        b_tier_size_mult_max=float(candidate_tier_config.get("b_tier_size_mult_max", 0.90)),
        high_liquidity_loosen_pct=float(candidate_tier_config.get("high_liquidity_loosen_pct", 0.30)),
        candidate_scarcity_loosen_pct=float(candidate_tier_config.get("candidate_scarcity_loosen_pct", 0.12)),
        symbol_spread_caps={
            StrategyRouter._normalize_symbol(str(symbol)): float(limit)
            for symbol, limit in spread_profile_config.items()
            if str(symbol).strip()
        },
        recycle_regime_boost=float(candidate_tier_config.get("recycle_regime_boost", 0.25)),
        recycle_max_per_family_per_day=int(candidate_tier_config.get("recycle_max_per_family_per_day", 1)),
        family_rotation_window_trades=int(candidate_tier_config.get("family_rotation_window_trades", 20)),
        family_rotation_share_threshold=float(candidate_tier_config.get("family_rotation_share_threshold", 0.60)),
        family_rotation_score_penalty=float(candidate_tier_config.get("family_rotation_score_penalty", 0.20)),
        compression_burst_candidates=int(candidate_tier_config.get("compression_burst_candidates", 3)),
        compression_burst_size_multiplier=float(candidate_tier_config.get("compression_burst_size_multiplier", 1.20)),
        xau_m5_burst_target=int(candidate_tier_config.get("xau_m5_burst_target", 8)),
        xau_prime_session_mult=float(candidate_tier_config.get("xau_prime_session_mult", 2.50)),
        transition_score_bonus=float(candidate_tier_config.get("transition_score_bonus", 0.35)),
        transition_size_multiplier=float(candidate_tier_config.get("transition_size_multiplier", 1.20)),
        btc_weekend_score_bonus=float(candidate_tier_config.get("btc_weekend_score_bonus", 0.50)),
        btc_weekend_size_boost=float(candidate_tier_config.get("btc_weekend_size_boost", 1.20)),
        btc_weekend_burst_target=int(candidate_tier_config.get("btc_weekend_burst_target", 40)),
        all_pairs_aggression=float(candidate_tier_config.get("all_pairs_aggression", 1.20)),
        btc_velocity_decay_trigger_trades_per_10_bars=float(
            candidate_tier_config.get("btc_velocity_decay_trigger_trades_per_10_bars", 1.50)
        ),
        xau_grid_compression_spacing_multiplier=float(
            candidate_tier_config.get("xau_grid_compression_spacing_multiplier", 1.30)
        ),
        xau_grid_expansion_burst_size_multiplier=float(
            candidate_tier_config.get("xau_grid_expansion_burst_size_multiplier", 1.25)
        ),
    )
    grid_scalper = XAUGridScalper.from_config(
        settings.section("xau_grid_scalper") if isinstance(settings.raw.get("xau_grid_scalper"), dict) else {},
        logger=logger,
    )
    feature_engineer = FeatureEngineer()
    regime_detector = RegimeDetector(settings.path("models.regime_classifier"), settings.path("data.regime_history_file"))
    strategy_config = settings.section("strategy")
    strategy_engines = {
        symbol: StrategyEngine(
            resolved,
            entry_method=str(strategy_config.get("entry_method", "BREAKOUT_VOLUME_ATR")),
            breakout_lookback=int(strategy_config.get("breakout_lookback", 20)),
            min_atr_pct=float(strategy_config.get("min_atr_pct", 0.75)),
            min_volume_ratio=float(strategy_config.get("min_volume_ratio", 1.0)),
            max_spread_points=float(strategy_config.get("max_spread_points", settings.section("risk").get("max_spread_points", 35))),
            allow_long=bool(strategy_config.get("allow_long", True)),
            allow_short=bool(strategy_config.get("allow_short", True)),
            scaling_enabled=bool(strategy_config.get("scaling_enabled", True)),
            scale_in_step_atr=float(strategy_config.get("scale_in_step_atr", 0.6)),
            pullback_add_enabled=bool(strategy_config.get("pullback_add_enabled", True)),
            pullback_ema_distance_atr=float(strategy_config.get("pullback_ema_distance_atr", 0.35)),
        )
        for symbol, resolved in resolved_symbols.items()
    }
    ai_gate = AIGate(
        scorer_path=settings.path("models.trade_scorer"),
        value_path=settings.path("models.trade_value_model"),
        risk_modulator_path=settings.path("models.risk_modulator"),
        schema_path=settings.path("models.feature_schema"),
        min_probabilities={
            "DEFAULT": float(settings.section("ai")["min_probability"]),
            "TREND_CONTINUATION": float(settings.section("ai")["min_probability_trend"]),
            "BREAKOUT_RETEST": float(settings.section("ai")["min_probability_breakout"]),
            "RANGE_REVERSAL": float(settings.section("ai")["min_probability_range"]),
            "EV": float(settings.section("ai")["min_probability_expected_value"]),
        },
        enabled=bool(settings.section("ai")["enabled"]),
        remote_enabled=bool(settings.section("ai").get("remote_enabled", True)),
        live_remote_enabled=bool(settings.section("ai").get("live_remote_ai_enabled", False)),
        openai_api_env=str(settings.section("ai").get("openai_api_env", "OPENAI_API_KEY")),
        openai_model=str(settings.section("ai").get("openai_model", "gpt-4o-mini")),
        openai_timeout_seconds=float(settings.section("ai").get("openai_timeout_seconds", 8.0)),
        openai_retry_once=bool(settings.section("ai").get("openai_retry_once", True)),
        remote_score_enabled=bool(settings.section("ai").get("remote_score_enabled", False)),
        logger=logger,
    )
    learning_brain = ApexLearningBrain(
        online_learning=online_learning,
        strategy_optimizer=strategy_optimizer,
        journal=journal,
        data_dir=learning_data_dir,
        logger=logger,
        offline_gpt_enabled=bool(settings.section("ai").get("offline_gpt_enabled", True)),
        openai_api_env=str(settings.section("ai").get("openai_api_env", "OPENAI_API_KEY")),
        openai_model=str(settings.section("ai").get("openai_model", "gpt-4o-mini")),
        openai_timeout_seconds=float(settings.section("ai").get("openai_timeout_seconds", 8.0)),
        openai_retry_once=bool(settings.section("ai").get("openai_retry_once", True)),
        last_trades_review_limit=int(settings.section("ai").get("offline_review_limit_trades", 200)),
        monte_carlo_min_realities=int(settings.section("ai").get("offline_mc_min_realities", 500)),
        monte_carlo_pass_floor=float(settings.section("ai").get("offline_mc_pass_floor", 0.88)),
        new_pair_observation_days=int(settings.section("ai").get("new_pair_observation_days", 1)),
        new_pair_observation_hours=int(settings.section("ai").get("new_pair_observation_hours", 4)),
        short_goal_equity=float(settings.section("ai").get("trajectory_short_target_equity", 100000.0)),
        medium_goal_equity=float(settings.section("ai").get("trajectory_medium_target_equity", 1000000.0)),
        shadow_default_variants=int(
            (
                bridge_orchestrator_config.get("shadow_promotion", {})
                if isinstance(bridge_orchestrator_config.get("shadow_promotion"), dict)
                else {}
            ).get("variants_per_active_lane", 50)
        ),
        shadow_hot_variants=int(
            (
                bridge_orchestrator_config.get("shadow_promotion", {})
                if isinstance(bridge_orchestrator_config.get("shadow_promotion"), dict)
                else {}
            ).get("variants_per_hot_lane", 64)
        ),
        shadow_promotion_threshold=float(
            (
                bridge_orchestrator_config.get("shadow_promotion", {})
                if isinstance(bridge_orchestrator_config.get("shadow_promotion"), dict)
                else {}
            ).get("promotion_threshold", 0.64)
        ),
    )
    news_config = settings.section("news")
    news_bias_config = news_config.get("bias", {}) if isinstance(news_config.get("bias"), dict) else {}
    unknown_block_names = {str(name) for name in news_config.get("unknown_block_session_names", [])}
    fallback_windows = [session for session in settings.sessions if session.name in unknown_block_names]
    news_engine = NewsEngine(
        cache_path=settings.path("data.news_cache_file"),
        provider=str(news_config["provider"]),
        api_base_url=str(news_config["api_base_url"]),
        api_key_env=str(news_config["api_key_env"]),
        cache_ttl_seconds=int(news_config["cache_ttl_seconds"]),
        block_high_impact=bool(news_config["block_high_impact"]),
        block_medium_impact=bool(news_config["block_medium_impact"]),
        block_window_minutes_before=int(news_config["block_window_minutes_before"]),
        block_window_minutes_after=int(news_config["block_window_minutes_after"]),
        fail_open=bool(news_config["fail_open"]),
        enabled=bool(news_config["enabled"]),
        api_key=str(news_config.get("api_key", "")),
        fallback_provider=str(news_config.get("fallback_provider", "")),
        fallback_api_base_url=str(news_config.get("fallback_api_base_url", "")),
        fallback_api_key_env=str(news_config.get("fallback_api_key_env", "")),
        fallback_api_key=str(news_config.get("fallback_api_key", "")),
        fallback_session_windows=fallback_windows,
        logger=logger,
        log_refresh_seconds=int(news_config.get("log_refresh_seconds", 900)),
        bias_enabled=bool(news_bias_config.get("enabled", True)),
        bias_lookback_minutes=int(news_bias_config.get("lookback_minutes", 240)),
        http_timeout_seconds=float(news_config.get("http_timeout_seconds", 6.0)),
        http_retries=int(news_config.get("http_retries", 2)),
        stale_cache_max_age_multiplier=int(news_config.get("stale_cache_max_age_multiplier", 12)),
        user_agent=str(news_config.get("user_agent", "Mozilla/5.0 (ApexBot News)")),
        supplemental_rss_feeds=[
            dict(item)
            for item in news_config.get("supplemental_rss_feeds", [])
            if isinstance(item, dict) and str(item.get("url") or "").strip()
        ],
        rss_headline_limit=int(news_config.get("rss_headline_limit", 6)),
        event_playbook_map=dict(
            (
                settings.raw.get("event_playbooks", {})
                if isinstance(settings.raw.get("event_playbooks"), dict)
                else {}
            ).get("playbook_map", {})
            if isinstance(
                (
                    settings.raw.get("event_playbooks", {})
                    if isinstance(settings.raw.get("event_playbooks"), dict)
                    else {}
                ).get("playbook_map"),
                dict,
            )
            else {}
        ),
    )
    risk_engine = RiskEngine()
    execution = ExecutionService(mt5_client, journal, logger)
    positions = PositionManager(
        allow_partial_closes=bool(settings.section("exits").get("allow_partial_closes", False)),
        partial_close_r=float(settings.section("exits")["partial_close_r"]),
        partial_close_fraction=float(settings.section("exits")["partial_close_fraction"]),
        trail_activation_r=float(settings.section("exits")["trail_activation_r"]),
        trail_atr_multiple=float(settings.section("exits")["trail_atr"]),
        time_stop_hours=int(settings.section("exits")["time_stop_hours"]),
        break_even_trigger_r=float(settings.section("exits").get("break_even_trigger_r", 0.7)),
        break_even_buffer_r=float(settings.section("exits").get("break_even_buffer_r", 0.05)),
        partial_close_r2=float(settings.section("exits").get("partial_close_r2", 2.0)),
        partial_close_fraction2=float(settings.section("exits").get("partial_close_fraction2", 0.3)),
        basket_take_profit_r=float(settings.section("exits").get("basket_take_profit_r", 3.0)),
    )
    portfolio = PortfolioManager(
        max_positions_total=min(
            int(settings.section("risk").get("max_positions_total", system_config.get("max_positions_total", system_config.get("max_positions", 10)))),
            int(settings.section("risk").get("max_positions_total_hard_cap", system_config.get("max_positions_total", 10))),
        ),
        max_positions_per_symbol=int(settings.section("risk").get("max_positions_per_symbol", system_config.get("max_positions_per_symbol", 2))),
        max_same_direction=int(settings.section("risk")["max_same_direction_positions"]),
        correlation_window_minutes=int(settings.section("risk")["correlation_window_minutes"]),
    )
    kill_switch = KillSwitch(settings.runtime_paths.state_dir / "kill_switch.lock")
    monitor = Monitor(
        logger,
        settings.resolve_path_value(str(settings.section("monitoring")["alert_log_file"])),
        bool(settings.section("monitoring")["print_dashboard"]),
    )
    backtester = Backtester(
        strategy_engine=strategy_engines[configured_symbols[0]],
        regime_detector=regime_detector,
        ai_gate=ai_gate,
        spread_points=float(settings.section("backtest")["default_spread_points"]),
        slippage_points=float(settings.section("backtest")["default_slippage_points"]),
        commission_per_lot=float(settings.section("backtest")["commission_per_lot"]),
        latency_ms=int(settings.section("backtest").get("default_latency_ms", 0)),
        max_positions_per_symbol=int(settings.section("risk").get("max_positions_per_symbol", system_config.get("max_positions_per_symbol", 10))),
        max_positions_total=min(
            int(settings.section("risk").get("max_positions_total", system_config.get("max_positions_total", system_config.get("max_positions", 20)))),
            int(settings.section("risk").get("max_positions_total_hard_cap", system_config.get("max_positions_total", 20))),
        ),
        be_trigger_r=float(settings.section("exits").get("break_even_trigger_r", 0.7)),
        be_buffer_r=float(settings.section("exits").get("break_even_buffer_r", 0.05)),
        trail_start_r=float(settings.section("exits").get("trail_activation_r", 1.0)),
        trail_atr_mult=float(settings.section("exits").get("trail_atr", 1.0)),
        partial1_r=float(settings.section("exits").get("partial_close_r", 1.0)),
        partial1_fraction=float(settings.section("exits").get("partial_close_fraction", 0.4)),
        partial2_r=float(settings.section("exits").get("partial_close_r2", 2.0)),
        partial2_fraction=float(settings.section("exits").get("partial_close_fraction2", 0.3)),
        basket_tp_r=float(settings.section("exits").get("basket_take_profit_r", 3.0)),
        use_fixed_lot=bool(settings.section("risk").get("use_fixed_lot", False)),
        fixed_lot=float(settings.section("risk").get("fixed_lot", 0.01)),
        risk_per_trade=float(settings.section("risk").get("risk_per_trade", 0.0025)),
        strict_plausibility=bool(settings.section("backtest").get("strict_plausibility", True)),
        max_plausible_win_rate=float(settings.section("backtest").get("max_plausible_win_rate", 0.85)),
        min_trades_for_plausibility=int(settings.section("backtest").get("min_trades_for_plausibility", 200)),
        whitelist_high_win_rate=False,
    )
    trainer = Trainer(
        market_data=market_data,
        feature_engineer=feature_engineer,
        strategy_engine=strategy_engines[configured_symbols[0]],
        regime_detector=regime_detector,
        model_paths={
            "trade_scorer": settings.path("models.trade_scorer"),
            "trade_value_model": settings.path("models.trade_value_model"),
            "risk_modulator": settings.path("models.risk_modulator"),
            "regime_classifier": settings.path("models.regime_classifier"),
            "feature_schema": settings.path("models.feature_schema"),
            "metadata": settings.path("models.metadata"),
        },
        train_ratio=float(settings.section("backtest")["train_ratio"]),
        validation_ratio=float(settings.section("backtest")["validation_ratio"]),
        test_ratio=float(settings.section("backtest")["test_ratio"]),
    )
    runtime = {
        "settings": settings,
        "logger": logger,
        "journal": journal,
        "mt5_client": mt5_client,
        "configured_symbols": configured_symbols,
        "resolved_symbols": resolved_symbols,
        "discovered_market_universe": discovered_universe,
        "history_warmup_summary": history_warmup_summary,
        "primary_symbol": primary_symbol,
        "market_data": market_data,
        "feature_engineer": feature_engineer,
        "session_profile": session_profile,
        "regime_detector": regime_detector,
        "strategy_engines": strategy_engines,
        "strategy_router": strategy_router,
        "grid_scalper": grid_scalper,
        "ai_gate": ai_gate,
        "news_engine": news_engine,
        "risk_engine": risk_engine,
        "execution": execution,
        "positions": positions,
        "portfolio": portfolio,
        "kill_switch": kill_switch,
        "monitor": monitor,
        "backtester": backtester,
        "trainer": trainer,
        "bridge_queue": bridge_queue,
        "strategy_optimizer": strategy_optimizer,
        "online_learning": online_learning,
        "learning_brain": learning_brain,
        "bridge_config": bridge_config,
        "bridge_orchestrator_config": bridge_orchestrator_config,
        "candidate_tier_config": candidate_tier_config,
        "candidate_verification_log_path": settings.resolve_path_value(
            str(bridge_orchestrator_config.get("candidate_verification_log_file", "data/candidate_verification.log"))
        ),
        "symbol_rules": symbol_rules,
        "dry_run": dry_run,
    }
    learning_brain.apply_promoted_params_to_runtime(runtime)
    return runtime


def _load_symbol_frames(market_data: MarketDataService, symbol: str, counts: dict[str, int], dry_run: bool):
    required_timeframes = ("M5", "M15", "H1")
    optional_timeframes = {"M1", "H4"}
    frames: dict[str, object] = {}
    optional_errors: dict[str, str] = {}
    alias_fallbacks: list[str] = []
    canonical_symbol = _normalize_symbol_key(symbol)

    def _frame_tail_age_seconds(frame: object) -> float | None:
        if not isinstance(frame, pd.DataFrame) or frame.empty or "time" not in frame.columns:
            return None
        try:
            tail_time = pd.Timestamp(frame.iloc[-1]["time"])
        except Exception:
            return None
        if pd.isna(tail_time):
            return None
        if tail_time.tzinfo is None:
            tail_time = tail_time.tz_localize("UTC")
        else:
            tail_time = tail_time.tz_convert("UTC")
        return max(0.0, (pd.Timestamp.now(tz=UTC) - tail_time).total_seconds())

    def _needs_alias_fallback(frame: object, timeframe: str, count: int) -> bool:
        if canonical_symbol == symbol:
            return False
        if frame is None:
            return True
        if not isinstance(frame, pd.DataFrame):
            return False
        if frame.empty:
            return True
        tail_age = _frame_tail_age_seconds(frame)
        if tail_age is not None and tail_age > (MarketDataService._max_live_bar_age_seconds(timeframe) * 2.0):
            return True
        minimum_rows = max(3, int(max(1, int(count)) * 0.5))
        return len(frame) < minimum_rows

    def _prefer_canonical_alias_frame(frame: object, timeframe: str, count: int, *, live_mode: bool) -> tuple[object, str | None]:
        if canonical_symbol == symbol or not _needs_alias_fallback(frame, timeframe, count):
            return frame, None
        try:
            canonical_frame = market_data.fetch(canonical_symbol, timeframe, count) if live_mode else market_data.load_cached(canonical_symbol, timeframe)
        except Exception:
            return frame, None
        if canonical_frame is None:
            return frame, None
        if isinstance(canonical_frame, pd.DataFrame) and canonical_frame.empty:
            return frame, None
        return canonical_frame, canonical_symbol

    if not dry_run:
        for timeframe, count in counts.items():
            try:
                frame = market_data.fetch(symbol, timeframe, count)
            except Exception as exc:
                if canonical_symbol != symbol:
                    try:
                        frame = market_data.fetch(canonical_symbol, timeframe, count)
                        frames[timeframe] = frame
                        alias_fallbacks.append(f"{timeframe}:{canonical_symbol}")
                        continue
                    except Exception:
                        pass
                if timeframe in optional_timeframes:
                    optional_errors[str(timeframe).upper()] = str(exc)
                    continue
                return None, f"market_data_unavailable:{symbol}:{timeframe}:{exc}"
            preferred_frame, alias_used = _prefer_canonical_alias_frame(frame, timeframe, count, live_mode=True)
            frames[timeframe] = preferred_frame
            if alias_used:
                alias_fallbacks.append(f"{timeframe}:{alias_used}")
        missing_required = [timeframe for timeframe in required_timeframes if timeframe not in frames]
        if missing_required:
            return None, f"market_data_missing_required:{symbol}:{','.join(missing_required)}"
        reason_parts: list[str] = []
        if alias_fallbacks:
            reason_parts.append(f"market_data_alias_fallback_used:{symbol}:{','.join(alias_fallbacks)}")
        if optional_errors:
            reason_parts.append(f"market_data_optional_missing:{symbol}:{optional_errors}")
        return frames, " | ".join(reason_parts) if reason_parts else None

    for timeframe in counts:
        cached = market_data.load_cached(symbol, timeframe)
        preferred_cached, alias_used = _prefer_canonical_alias_frame(cached, timeframe, int(counts[timeframe]), live_mode=False)
        cached = preferred_cached
        if alias_used:
            alias_fallbacks.append(f"{timeframe}:{alias_used}")
        if cached is None:
            if timeframe in optional_timeframes:
                optional_errors[str(timeframe).upper()] = "no cache yet"
                continue
            return None, "no cache yet"
        frames[timeframe] = cached
    missing_required = [timeframe for timeframe in required_timeframes if timeframe not in frames]
    if missing_required:
        return None, f"market_data_missing_required:{symbol}:{','.join(missing_required)}"
    reason_parts = []
    if alias_fallbacks:
        reason_parts.append(f"market_data_alias_fallback_used:{symbol}:{','.join(alias_fallbacks)}")
    if optional_errors:
        reason_parts.append(f"market_data_optional_missing:{symbol}:{optional_errors}")
    return frames, " | ".join(reason_parts) if reason_parts else None


def _timeframe_floor_utc(value: datetime, timeframe: str) -> datetime:
    current = value.astimezone(UTC)
    tf = str(timeframe).upper()
    if tf == "M1":
        return current.replace(second=0, microsecond=0)
    if tf == "M5":
        minute = (current.minute // 5) * 5
        return current.replace(minute=minute, second=0, microsecond=0)
    if tf == "M15":
        minute = (current.minute // 15) * 15
        return current.replace(minute=minute, second=0, microsecond=0)
    return current.replace(second=0, microsecond=0)


def _refresh_frame_with_bridge_quote(
    frame: pd.DataFrame | None,
    *,
    timeframe: str,
    bridge_symbol_snapshot: dict[str, Any] | None,
    now_utc: datetime,
) -> pd.DataFrame | None:
    if frame is None or frame.empty or "time" not in frame.columns or not isinstance(bridge_symbol_snapshot, dict):
        return frame
    bid = _safe_float(bridge_symbol_snapshot.get("bid"), 0.0)
    ask = _safe_float(bridge_symbol_snapshot.get("ask"), 0.0)
    price = _safe_float(bridge_symbol_snapshot.get("last"), 0.0)
    if price <= 0.0 and bid > 0.0 and ask > 0.0:
        price = (bid + ask) * 0.5
    if price <= 0.0:
        price = max(bid, ask)
    if price <= 0.0:
        return frame
    snapshot_ts = _parse_iso_utc(bridge_symbol_snapshot.get("updated_at")) or now_utc
    bucket_ts = _timeframe_floor_utc(snapshot_ts, timeframe)
    refreshed = frame.copy()
    last_time = pd.Timestamp(refreshed.iloc[-1]["time"])
    if last_time.tzinfo is None:
        last_time = last_time.tz_localize("UTC")
    else:
        last_time = last_time.tz_convert("UTC")
    spread_points = _safe_float(bridge_symbol_snapshot.get("spread_points"), 0.0)
    if bucket_ts <= last_time.to_pydatetime():
        index = refreshed.index[-1]
        open_price = _safe_float(refreshed.at[index, "open"], price)
        high_price = _safe_float(refreshed.at[index, "high"], open_price)
        low_price = _safe_float(refreshed.at[index, "low"], open_price)
        refreshed.at[index, "high"] = max(high_price, price, open_price)
        refreshed.at[index, "low"] = min(low_price, price, open_price)
        refreshed.at[index, "close"] = price
        if "spread" in refreshed.columns and spread_points > 0.0:
            refreshed.at[index, "spread"] = spread_points
        return refreshed
    previous_close = _safe_float(refreshed.iloc[-1].get("close"), price)
    template = refreshed.iloc[-1].copy()
    template["time"] = pd.Timestamp(bucket_ts)
    template["open"] = previous_close
    template["high"] = max(previous_close, price)
    template["low"] = min(previous_close, price)
    template["close"] = price
    if "spread" in refreshed.columns and spread_points > 0.0:
        template["spread"] = spread_points
    for volume_key in ("tick_volume", "real_volume"):
        if volume_key in refreshed.columns:
            template[volume_key] = 0
    return pd.concat([refreshed, pd.DataFrame([template])], ignore_index=True)


def _fallback_account_snapshot(internal_equity_estimate: float = 50.0) -> dict[str, float | None]:
    estimate = max(1.0, float(internal_equity_estimate))
    return {
        "login": None,
        "server": None,
        "leverage": None,
        "equity": estimate,
        "balance": estimate,
        "margin_free": estimate,
    }


def _apply_runtime_account_snapshot(
    account: dict[str, Any],
    *,
    account_from_mt5: bool,
    bridge_snapshot: dict[str, Any] | None,
    internal_equity_estimate: float,
) -> tuple[dict[str, Any], str, bool]:
    merged = dict(account)
    bridge_snapshot_active = isinstance(bridge_snapshot, dict) and bridge_snapshot.get("equity") is not None
    account_label = "MT5_FEED" if account_from_mt5 else "INTERNAL (NO MT5 EQUITY FEED)"
    if bridge_snapshot_active:
        numeric_field_map = {
            "balance": "balance",
            "equity": "equity",
            "free_margin": "margin_free",
            "margin": "margin",
            "margin_level": "margin_level",
            "leverage": "leverage",
        }
        for snapshot_key, account_key in numeric_field_map.items():
            value = bridge_snapshot.get(snapshot_key)
            if value is None:
                continue
            try:
                merged[account_key] = float(value)
            except (TypeError, ValueError):
                continue
        account_label = "LIVE_BRIDGE_FEED"
    if (not account_from_mt5) and (not bridge_snapshot_active):
        conservative_equity = max(1.0, float(internal_equity_estimate))
        for key in ("equity", "balance", "margin_free"):
            current = float(merged.get(key, conservative_equity))
            merged[key] = min(current, conservative_equity)
    return merged, account_label, bridge_snapshot_active


def _equity_band(equity: float, bootstrap_equity_threshold: float) -> str:
    if equity <= max(1.0, bootstrap_equity_threshold * 0.5):
        return "bootstrap_aggressive"
    if equity <= max(1.0, bootstrap_equity_threshold):
        return "bootstrap_balanced"
    if equity <= max(1.0, bootstrap_equity_threshold * 2.0):
        return "growth_standard"
    return "standard"


def _phase_state(equity: float, performance: dict[str, Any] | None) -> dict[str, Any]:
    overall = (performance or {}).get("overall", {}) if isinstance(performance, dict) else {}
    trades = int(float(overall.get("trades", 0.0) or 0.0))
    win_rate = float(overall.get("win_rate", 0.0) or 0.0)
    expectancy = float(overall.get("expectancy_r", 0.0) or 0.0)
    daily_green_streak = max(0, int((performance or {}).get("daily_green_streak", 0) or 0))
    smart_scaling_green_days_required = 2
    smart_scaling_trade_gate = 12
    smart_scaling_win_rate_gate = 0.52
    smart_scaling_expectancy_gate = 0.00
    smart_scaling_ready = bool(
        daily_green_streak >= smart_scaling_green_days_required
        and trades >= smart_scaling_trade_gate
        and win_rate >= smart_scaling_win_rate_gate
        and expectancy >= smart_scaling_expectancy_gate
    )
    smart_scaling_mode = "PROVEN_2DAY_RAMP" if smart_scaling_ready else "BASELINE_XAU_LEAD"
    smart_scaling_lane_boost = 1.18 if smart_scaling_ready else 1.0
    smart_scaling_soft_burst_bonus = 2 if smart_scaling_ready else 0
    smart_scaling_entry_cap_bonus = 1 if smart_scaling_ready else 0
    smart_scaling_compression_bonus = 1 if smart_scaling_ready else 0
    smart_scaling_compression_multiplier = 1.18 if smart_scaling_ready else 1.0
    smart_scaling_rank_bonus = 0.01 if smart_scaling_ready else 0.0
    smart_scaling_size_bonus = 0.02 if smart_scaling_ready else 0.0
    smart_scaling_activation_relax = 0.02 if smart_scaling_ready else 0.0

    phase = "PHASE_1"
    reason = "equity_below_growth_threshold"
    risk_pct = 0.01
    max_risk_pct = 0.01
    daily_trade_cap = 240
    stretch_daily_trade_target = 420
    overflow_daily_trade_cap = 720
    hourly_base_target = 36
    hourly_stretch_target = 60
    ai_threshold_mode = "moderate"
    next_requirements = {
        "phase": "PHASE_2",
        "min_equity": 100.0,
        "min_trades": 12,
        "min_win_rate": 0.50,
        "min_expectancy_r": 0.05,
    }
    next_equity_milestone = 100.0

    if equity >= 1_000_000.0 and trades >= 600 and win_rate >= 0.60 and expectancy >= 0.15:
        phase = "PHASE_10"
        reason = "institutional_scale_confirmed"
        daily_trade_cap = 760
        stretch_daily_trade_target = 1120
        overflow_daily_trade_cap = 1680
        hourly_base_target = 104
        hourly_stretch_target = 156
        ai_threshold_mode = "aggressive"
        next_requirements = {}
        next_equity_milestone = 2_500_000.0
    elif equity >= 500_000.0 and trades >= 500 and win_rate >= 0.60 and expectancy >= 0.15:
        phase = "PHASE_9"
        reason = "ultra_high_compounding_confirmed"
        daily_trade_cap = 680
        stretch_daily_trade_target = 1020
        overflow_daily_trade_cap = 1560
        hourly_base_target = 92
        hourly_stretch_target = 140
        ai_threshold_mode = "aggressive"
        next_requirements = {
            "phase": "PHASE_10",
            "min_equity": 1_000_000.0,
            "min_trades": 600,
            "min_win_rate": 0.60,
            "min_expectancy_r": 0.15,
        }
        next_equity_milestone = 1_000_000.0
    elif equity >= 250_000.0 and trades >= 400 and win_rate >= 0.59 and expectancy >= 0.15:
        phase = "PHASE_8"
        reason = "hyper_scale_confirmed"
        daily_trade_cap = 600
        stretch_daily_trade_target = 920
        overflow_daily_trade_cap = 1440
        hourly_base_target = 84
        hourly_stretch_target = 128
        ai_threshold_mode = "aggressive"
        next_requirements = {
            "phase": "PHASE_9",
            "min_equity": 500_000.0,
            "min_trades": 500,
            "min_win_rate": 0.60,
            "min_expectancy_r": 0.15,
        }
        next_equity_milestone = 500_000.0
    elif equity >= 100_000.0 and trades >= 300 and win_rate >= 0.58 and expectancy >= 0.14:
        phase = "PHASE_7"
        reason = "large_account_flow_confirmed"
        daily_trade_cap = 540
        stretch_daily_trade_target = 840
        overflow_daily_trade_cap = 1320
        hourly_base_target = 76
        hourly_stretch_target = 116
        ai_threshold_mode = "aggressive"
        next_requirements = {
            "phase": "PHASE_8",
            "min_equity": 250_000.0,
            "min_trades": 400,
            "min_win_rate": 0.59,
            "min_expectancy_r": 0.15,
        }
        next_equity_milestone = 250_000.0
    elif equity >= 25_000.0 and trades >= 200 and win_rate >= 0.58 and expectancy >= 0.14:
        phase = "PHASE_6"
        reason = "professional_compounding_confirmed"
        risk_pct = 0.02
        max_risk_pct = 0.025
        daily_trade_cap = 480
        stretch_daily_trade_target = 760
        overflow_daily_trade_cap = 1200
        hourly_base_target = 68
        hourly_stretch_target = 104
        ai_threshold_mode = "aggressive"
        next_requirements = {
            "phase": "PHASE_7",
            "min_equity": 100_000.0,
            "min_trades": 300,
            "min_win_rate": 0.58,
            "min_expectancy_r": 0.14,
        }
        next_equity_milestone = 100_000.0
    elif equity >= 5_000.0 and trades >= 120 and win_rate >= 0.57 and expectancy >= 0.12:
        phase = "PHASE_5"
        reason = "compound_scaling_confirmed"
        risk_pct = 0.0175
        max_risk_pct = 0.025
        daily_trade_cap = 420
        stretch_daily_trade_target = 680
        overflow_daily_trade_cap = 1080
        hourly_base_target = 60
        hourly_stretch_target = 92
        ai_threshold_mode = "aggressive"
        next_requirements = {
            "phase": "PHASE_6",
            "min_equity": 25_000.0,
            "min_trades": 200,
            "min_win_rate": 0.58,
            "min_expectancy_r": 0.14,
        }
        next_equity_milestone = 25_000.0
    elif equity >= 1_500.0 and trades >= 60 and win_rate >= 0.56 and expectancy >= 0.10:
        phase = "PHASE_4"
        reason = "small_account_scaling_confirmed"
        risk_pct = 0.02
        max_risk_pct = 0.025
        daily_trade_cap = 360
        stretch_daily_trade_target = 600
        overflow_daily_trade_cap = 960
        hourly_base_target = 52
        hourly_stretch_target = 82
        ai_threshold_mode = "aggressive"
        next_requirements = {
            "phase": "PHASE_5",
            "min_equity": 5_000.0,
            "min_trades": 120,
            "min_win_rate": 0.57,
            "min_expectancy_r": 0.12,
        }
        next_equity_milestone = 5_000.0
    elif equity >= 300.0 and trades >= 30 and win_rate >= 0.55 and expectancy >= 0.10:
        phase = "PHASE_3"
        reason = "equity_and_performance_confirmed"
        risk_pct = 0.02 if equity >= 500.0 else 0.015
        max_risk_pct = 0.025
        daily_trade_cap = 320
        stretch_daily_trade_target = 540
        overflow_daily_trade_cap = 900
        hourly_base_target = 48
        hourly_stretch_target = 76
        ai_threshold_mode = "aggressive"
        next_requirements = {
            "phase": "PHASE_4",
            "min_equity": 1_500.0,
            "min_trades": 60,
            "min_win_rate": 0.56,
            "min_expectancy_r": 0.10,
        }
        next_equity_milestone = 1_500.0
    elif equity >= 100.0 and trades >= 12 and win_rate >= 0.50 and expectancy >= 0.05:
        phase = "PHASE_2"
        reason = "edge_emerging"
        risk_pct = 0.01
        max_risk_pct = 0.025
        daily_trade_cap = 280
        stretch_daily_trade_target = 460
        overflow_daily_trade_cap = 760
        hourly_base_target = 40
        hourly_stretch_target = 64
        ai_threshold_mode = "moderate"
        next_requirements = {
            "phase": "PHASE_3",
            "min_equity": 300.0,
            "min_trades": 30,
            "min_win_rate": 0.55,
            "min_expectancy_r": 0.10,
        }
        next_equity_milestone = 300.0
    elif equity >= 100.0:
        reason = "equity_up_but_performance_unproven"

    if (
        equity >= 500.0
        and daily_green_streak >= 3
        and trades >= 30
        and win_rate >= 0.55
        and expectancy >= 0.08
    ):
        phase = "PHASE_3"
        reason = "equity_and_multi_day_streak_confirmed"
        risk_pct = max(risk_pct, 0.035)
        max_risk_pct = max(max_risk_pct, 0.04)
        daily_trade_cap = max(daily_trade_cap, 320)
        stretch_daily_trade_target = max(stretch_daily_trade_target, 520)
        overflow_daily_trade_cap = max(overflow_daily_trade_cap, 820)
        hourly_base_target = max(hourly_base_target, 44)
        hourly_stretch_target = max(hourly_stretch_target, 70)
        ai_threshold_mode = "aggressive"

    if smart_scaling_ready:
        daily_trade_cap = max(
            daily_trade_cap,
            max(daily_trade_cap + 4, int(round(daily_trade_cap * 1.35)), 300),
        )
        stretch_daily_trade_target = max(
            stretch_daily_trade_target,
            max(stretch_daily_trade_target + 8, int(round(stretch_daily_trade_target * 1.45)), 520),
        )
        overflow_daily_trade_cap = max(
            overflow_daily_trade_cap,
            max(overflow_daily_trade_cap + 12, int(round(overflow_daily_trade_cap * 1.55)), 840),
        )
        hourly_base_target = max(
            hourly_base_target,
            max(hourly_base_target + 2, int(round(hourly_base_target * 1.35)), 42),
        )
        hourly_stretch_target = max(
            hourly_stretch_target,
            max(hourly_stretch_target + 4, int(round(hourly_stretch_target * 1.40)), 72),
        )

    current_growth_bias = (
        "aggressive_compounding"
        if expectancy > 0.06 and win_rate >= 0.52
        else ("balanced_compounding" if expectancy >= -0.10 else "capital_rebuild")
    )
    current_compounding_state = (
        "validated_density_ramp"
        if smart_scaling_ready
        else ("stretch_eligible" if expectancy > 0.05 and win_rate >= 0.55 else "base_flow")
    )
    return {
        "current_phase": phase,
        "phase_reason": reason,
        "current_risk_pct": risk_pct,
        "current_max_risk_pct": max_risk_pct,
        "current_daily_trade_cap": daily_trade_cap,
        "base_daily_trade_target": daily_trade_cap,
        "stretch_daily_trade_target": stretch_daily_trade_target,
        "current_overflow_daily_trade_cap": overflow_daily_trade_cap,
        "hard_upper_limit": overflow_daily_trade_cap,
        "hourly_base_target": hourly_base_target,
        "hourly_stretch_target": hourly_stretch_target,
        "current_ai_threshold_mode": ai_threshold_mode,
        "scaling_mode": "quick_smart_scaler",
        "current_compounding_state": current_compounding_state,
        "current_growth_bias": current_growth_bias,
        "daily_green_streak": int(daily_green_streak),
        "smart_scaling_ready": bool(smart_scaling_ready),
        "smart_scaling_mode": str(smart_scaling_mode),
        "smart_scaling_green_days_required": int(smart_scaling_green_days_required),
        "smart_scaling_trade_gate": int(smart_scaling_trade_gate),
        "smart_scaling_win_rate_gate": float(smart_scaling_win_rate_gate),
        "smart_scaling_expectancy_gate": float(smart_scaling_expectancy_gate),
        "smart_scaling_lane_boost": float(smart_scaling_lane_boost),
        "smart_scaling_soft_burst_bonus": int(smart_scaling_soft_burst_bonus),
        "smart_scaling_entry_cap_bonus": int(smart_scaling_entry_cap_bonus),
        "smart_scaling_compression_bonus": int(smart_scaling_compression_bonus),
        "smart_scaling_compression_multiplier": float(smart_scaling_compression_multiplier),
        "smart_scaling_rank_bonus": float(smart_scaling_rank_bonus),
        "smart_scaling_size_bonus": float(smart_scaling_size_bonus),
        "smart_scaling_activation_relax": float(smart_scaling_activation_relax),
        "next_equity_milestone": next_equity_milestone,
        "projected_trade_capacity_today": stretch_daily_trade_target,
        "equity_phase_thresholds": {
            "PHASE_1_max": 99.99,
            "PHASE_2_min": 100.0,
            "PHASE_3_min": 300.0,
            "PHASE_4_min": 1500.0,
            "PHASE_5_min": 5000.0,
            "PHASE_6_min": 25000.0,
            "PHASE_7_min": 100000.0,
            "PHASE_8_min": 250000.0,
            "PHASE_9_min": 500000.0,
            "PHASE_10_min": 1000000.0,
        },
        "performance_phase_thresholds": {
            "PHASE_2": {"min_trades": 12, "min_win_rate": 0.50, "min_expectancy_r": 0.05},
            "PHASE_3": {"min_trades": 30, "min_win_rate": 0.55, "min_expectancy_r": 0.10},
            "PHASE_4": {"min_trades": 60, "min_win_rate": 0.56, "min_expectancy_r": 0.10},
            "PHASE_5": {"min_trades": 120, "min_win_rate": 0.57, "min_expectancy_r": 0.12},
            "PHASE_6": {"min_trades": 200, "min_win_rate": 0.58, "min_expectancy_r": 0.14},
            "PHASE_7": {"min_trades": 300, "min_win_rate": 0.58, "min_expectancy_r": 0.14},
            "PHASE_8": {"min_trades": 400, "min_win_rate": 0.59, "min_expectancy_r": 0.15},
            "PHASE_9": {"min_trades": 500, "min_win_rate": 0.60, "min_expectancy_r": 0.15},
            "PHASE_10": {"min_trades": 600, "min_win_rate": 0.60, "min_expectancy_r": 0.15},
        },
        "next_phase_requirements": next_requirements,
    }


def _empty_band_counts() -> dict[str, int]:
    return {"A+": 0, "A": 0, "A-": 0, "B+": 0, "B": 0, "C": 0}


def _record_band_count(counts: dict[str, int], band: str) -> None:
    normalized = str(band or "").strip().upper()
    if normalized in counts:
        counts[normalized] = int(counts.get(normalized, 0)) + 1


def _band_target_from_counts(counts: dict[str, int]) -> str:
    if int(counts.get("A+", 0)) > 0:
        return "A+"
    if int(counts.get("A", 0)) > 0 or int(counts.get("A-", 0)) > 0:
        return "A/A-"
    if int(counts.get("B+", 0)) > 0:
        return "B+"
    if int(counts.get("B", 0)) > 0:
        return "B"
    if int(counts.get("C", 0)) > 0:
        return "C"
    return "A+"


def _fallback_band_reason(current_band_target: str) -> str:
    target = str(current_band_target or "A+").upper()
    if target == "A/A-":
        return "no_A_plus_candidate"
    if target == "B+":
        return "no_A_plus_or_A_candidate"
    if target == "B":
        return "no_A_plus_A_or_B_plus_candidate"
    if target == "C":
        return "no_A_plus_A_B_plus_or_B_candidate"
    return ""


def _strategy_state_size_multiplier(strategy_state: str) -> float:
    return {
        "ATTACK": 1.05,
        "NORMAL": 1.00,
        "REDUCED": 0.90,
        "QUARANTINED": 0.75,
    }.get(str(strategy_state or "NORMAL").upper(), 1.00)


def _quality_tier_exit_profile(quality_tier: str, exits_config: dict[str, Any]) -> dict[str, Any]:
    tier = str(quality_tier or "B").upper()
    if tier == "A+":
        partials = [
            {"triggerR": 0.50, "closeFraction": 0.40, "reason": "tier_a_plus_lock_1"},
            {"triggerR": 1.20, "closeFraction": 0.30, "reason": "tier_a_plus_lock_2"},
        ]
        break_even_trigger_r = 0.50
        trail_activation_r = min(1.2, float(exits_config.get("trail_activation_r", 1.2)))
        trail_atr = max(0.85, float(exits_config.get("trail_atr", 1.0)))
        approved_rr_target = "2.4-2.8"
    elif tier == "A":
        partials = [
            {"triggerR": 0.50, "closeFraction": 0.35, "reason": "tier_a_lock_1"},
            {"triggerR": 1.00, "closeFraction": 0.30, "reason": "tier_a_lock_2"},
        ]
        break_even_trigger_r = 0.50
        trail_activation_r = float(exits_config.get("trail_activation_r", 1.0))
        trail_atr = float(exits_config.get("trail_atr", 1.0))
        approved_rr_target = "2.0-2.4"
    else:
        partials = [
            {"triggerR": 0.50, "closeFraction": 0.30, "reason": "tier_b_lock_1"},
            {"triggerR": 0.90, "closeFraction": 0.30, "reason": "tier_b_lock_2"},
        ]
        break_even_trigger_r = 0.50
        trail_activation_r = max(0.90, float(exits_config.get("trail_activation_r", 1.0)))
        trail_atr = max(0.90, float(exits_config.get("trail_atr", 1.0)))
        approved_rr_target = "1.7-2.0"
    return {
        "approved_rr_target": str(approved_rr_target),
        "breakeven_trigger_r": float(break_even_trigger_r),
        "trail_activation_r": float(trail_activation_r),
        "trail_atr": float(trail_atr),
        "be_buffer_r": 0.06 if tier in {"A+", "A"} else 0.05,
        "min_profit_protection_r": 0.12 if tier in {"A+", "A"} else 0.08,
        "trail_backoff_r": 0.55 if tier in {"A+", "A"} else 0.50,
        "trail_requires_partial1": False,
        "no_progress_bars": 0,
        "no_progress_mfe_r": 0.0,
        "early_invalidation_r": -1.0,
        "time_stop_bars": 0,
        "time_stop_max_r": 0.0,
        "partials": partials,
    }


def _exit_partials_enabled(exits_config: dict[str, Any]) -> bool:
    return bool(exits_config.get("allow_partial_closes", exits_config.get("enable_partial_closes", False)))


def _strategy_exit_profile(
    *,
    symbol_key: str,
    strategy_key: str,
    quality_tier: str,
    exits_config: dict[str, Any],
    streak_adjust_mode: str = "NEUTRAL",
    session_name: str = "",
    regime_state: str = "",
    weekend_mode: bool = False,
) -> dict[str, Any]:
    profile = dict(_quality_tier_exit_profile(quality_tier, exits_config))
    normalized_strategy_key = str(strategy_key or "").strip().upper()
    streak_mode = str(streak_adjust_mode or "NEUTRAL").upper()
    session_key = str(session_name or "").strip().upper()
    regime_key = runtime_regime_state(regime_state)
    if normalized_strategy_key == "XAUUSD_ADAPTIVE_M5_GRID":
        profile.update(
            {
                "approved_rr_target": "2.6-3.2",
                "breakeven_trigger_r": 0.78,
                "be_buffer_r": 0.18,
                "min_profit_protection_r": 0.22,
                "trail_activation_r": max(1.45, float(profile.get("trail_activation_r", 1.0) or 1.0)),
                "trail_atr": max(1.10, float(profile.get("trail_atr", 1.0) or 1.0)),
                "trail_backoff_r": 0.96,
                "basket_take_profit_r": 2.10,
                "no_progress_bars": 6,
                "no_progress_mfe_r": 0.18,
                "time_stop_bars": 12,
                "time_stop_max_r": 0.18,
                "partials": [
                    {"triggerR": 0.95, "closeFraction": 0.12, "reason": "xau_grid_lock_1"},
                    {"triggerR": 2.05, "closeFraction": 0.16, "reason": "xau_grid_lock_2"},
                ],
            }
        )
        if streak_mode == "WIN_STREAK":
            profile.update(
                {
                    "approved_rr_target": "3.0-3.6",
                    "breakeven_trigger_r": 0.96,
                    "be_buffer_r": 0.20,
                    "min_profit_protection_r": 0.28,
                    "trail_activation_r": max(1.72, float(profile.get("trail_activation_r", 1.45) or 1.45)),
                    "trail_atr": max(1.15, float(profile.get("trail_atr", 1.10) or 1.10)),
                    "trail_backoff_r": 1.08,
                    "basket_take_profit_r": 2.35,
                    "partials": [
                        {"triggerR": 1.05, "closeFraction": 0.10, "reason": "xau_grid_streak_lock_1"},
                        {"triggerR": 2.20, "closeFraction": 0.15, "reason": "xau_grid_streak_lock_2"},
                    ],
                }
            )
        elif streak_mode == "LOSS_STREAK":
            profile.update(
                {
                    "approved_rr_target": "1.9-2.4",
                    "breakeven_trigger_r": 0.58,
                    "be_buffer_r": 0.12,
                    "min_profit_protection_r": 0.16,
                    "trail_activation_r": max(1.00, min(1.20, float(profile.get("trail_activation_r", 1.35) or 1.35))),
                    "trail_atr": max(0.96, float(profile.get("trail_atr", 1.05) or 1.05)),
                    "trail_backoff_r": 0.64,
                    "basket_take_profit_r": 1.35,
                    "no_progress_bars": 5,
                    "no_progress_mfe_r": 0.12,
                    "time_stop_bars": 8,
                    "time_stop_max_r": 0.10,
                    "partials": [
                        {"triggerR": 0.65, "closeFraction": 0.20, "reason": "xau_grid_defensive_lock_1"},
                        {"triggerR": 1.25, "closeFraction": 0.20, "reason": "xau_grid_defensive_lock_2"},
                    ],
                }
            )
    elif normalized_strategy_key == "XAUUSD_LONDON_LIQUIDITY_SWEEP":
        profile.update(
            {
                "approved_rr_target": "2.1-2.7",
                "breakeven_trigger_r": 0.55,
                "be_buffer_r": 0.16,
                "min_profit_protection_r": 0.18,
                "trail_activation_r": max(1.10, float(profile.get("trail_activation_r", 1.0) or 1.0)),
                "trail_atr": max(0.90, float(profile.get("trail_atr", 1.0) or 1.0)),
                "trail_backoff_r": 0.62,
                "basket_take_profit_r": 1.45,
                "partials": [
                    {"triggerR": 0.55, "closeFraction": 0.28, "reason": "xau_sweep_lock_1"},
                    {"triggerR": 1.25, "closeFraction": 0.22, "reason": "xau_sweep_lock_2"},
                ],
            }
        )
        if session_key in {"LONDON", "NEW_YORK"} and regime_key == "MEAN_REVERSION":
            profile.update(
                {
                    "approved_rr_target": "1.7-2.2",
                    "breakeven_trigger_r": 0.42,
                    "be_buffer_r": 0.12,
                    "min_profit_protection_r": 0.14,
                    "trail_activation_r": max(0.88, float(profile.get("trail_activation_r", 1.10) or 1.10)),
                    "trail_atr": max(0.82, float(profile.get("trail_atr", 0.90) or 0.90)),
                    "trail_backoff_r": 0.50,
                    "basket_take_profit_r": 0.95,
                    "no_progress_bars": 5,
                    "no_progress_mfe_r": 0.10,
                    "time_stop_bars": 8,
                    "time_stop_max_r": 0.04,
                    "partials": [
                        {"triggerR": 0.42, "closeFraction": 0.32, "reason": "xau_sweep_meanrev_lock_1"},
                        {"triggerR": 0.90, "closeFraction": 0.22, "reason": "xau_sweep_meanrev_lock_2"},
                    ],
                }
            )
        if streak_mode == "WIN_STREAK":
            profile.update(
                {
                    "approved_rr_target": "2.2-2.8",
                    "trail_activation_r": max(1.05, float(profile.get("trail_activation_r", 0.95) or 0.95)),
                    "basket_take_profit_r": 1.40,
                }
            )
        elif streak_mode == "LOSS_STREAK":
            profile.update(
                {
                    "approved_rr_target": "1.7-2.1",
                    "breakeven_trigger_r": 0.35,
                    "trail_activation_r": max(0.82, min(0.92, float(profile.get("trail_activation_r", 0.95) or 0.95))),
                    "basket_take_profit_r": 1.05,
                    "partials": [
                        {"triggerR": 0.35, "closeFraction": 0.40, "reason": "xau_sweep_defensive_lock_1"},
                        {"triggerR": 0.90, "closeFraction": 0.25, "reason": "xau_sweep_defensive_lock_2"},
                    ],
                }
            )
    elif normalized_strategy_key in {"BTCUSD_PRICE_ACTION_CONTINUATION", "BTCUSD_VOLATILE_RETEST"}:
        profile.update(
            {
                "approved_rr_target": "2.0-2.4",
                "breakeven_trigger_r": 0.62,
                "be_buffer_r": 0.16,
                "min_profit_protection_r": 0.24,
                "trail_activation_r": max(1.25, float(profile.get("trail_activation_r", 1.0) or 1.0)),
                "trail_backoff_r": 0.80,
                "basket_take_profit_r": 1.55,
                "no_progress_bars": 6,
                "no_progress_mfe_r": 0.16,
                "early_invalidation_r": -0.42,
                "time_stop_bars": 14,
                "time_stop_max_r": 0.05,
                "partials": [
                    {"triggerR": 0.70, "closeFraction": 0.22, "reason": "btc_lock_1"},
                    {"triggerR": 1.55, "closeFraction": 0.18, "reason": "btc_lock_2"},
                ],
            }
        )
        if weekend_mode:
            profile.update(
                {
                    "approved_rr_target": "2.2-2.8",
                    "breakeven_trigger_r": 0.86,
                    "be_buffer_r": 0.18,
                    "min_profit_protection_r": 0.28,
                    "trail_activation_r": max(1.82, float(profile.get("trail_activation_r", 1.25) or 1.25)),
                    "trail_backoff_r": 0.96,
                    "basket_take_profit_r": 1.90,
                    "no_progress_bars": 7,
                    "no_progress_mfe_r": 0.20,
                    "partials": [
                        {"triggerR": 0.90, "closeFraction": 0.16, "reason": "btc_weekend_lock_1"},
                        {"triggerR": 1.85, "closeFraction": 0.14, "reason": "btc_weekend_lock_2"},
                    ],
                }
            )
        if not weekend_mode and session_key in {"SYDNEY", "TOKYO"} and regime_key in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
            profile.update(
                {
                    "approved_rr_target": "1.5-1.8",
                    "breakeven_trigger_r": 0.42,
                    "be_buffer_r": 0.10,
                    "min_profit_protection_r": 0.16,
                    "trail_activation_r": max(0.85, float(profile.get("trail_activation_r", 1.10) or 1.10)),
                    "trail_backoff_r": 0.55,
                    "basket_take_profit_r": 1.10,
                    "no_progress_bars": 4,
                    "no_progress_mfe_r": 0.14,
                    "early_invalidation_r": -0.28,
                    "time_stop_bars": 10,
                    "time_stop_max_r": 0.05,
                    "partials": [
                        {"triggerR": 0.50, "closeFraction": 0.28, "reason": "btc_asia_lock_1"},
                        {"triggerR": 1.10, "closeFraction": 0.22, "reason": "btc_asia_lock_2"},
                    ],
                }
            )
        elif not weekend_mode and session_key == "LONDON" and regime_key == "LOW_LIQUIDITY_CHOP":
            profile.update(
                {
                    "approved_rr_target": "1.6-1.9",
                    "breakeven_trigger_r": 0.45,
                    "be_buffer_r": 0.12,
                    "min_profit_protection_r": 0.18,
                    "trail_activation_r": max(0.92, float(profile.get("trail_activation_r", 1.10) or 1.10)),
                    "trail_backoff_r": 0.58,
                    "basket_take_profit_r": 1.05,
                    "time_stop_bars": 11,
                    "time_stop_max_r": 0.05,
                    "partials": [
                        {"triggerR": 0.45, "closeFraction": 0.30, "reason": "btc_london_lock_1"},
                        {"triggerR": 0.95, "closeFraction": 0.22, "reason": "btc_london_lock_2"},
                    ],
                }
            )
        if streak_mode == "WIN_STREAK":
            profile.update(
                {
                    "approved_rr_target": "2.4-3.0" if weekend_mode else "2.2-2.6",
                    "be_buffer_r": max(0.18 if weekend_mode else 0.15, float(profile.get("be_buffer_r", 0.14) or 0.14)),
                    "trail_activation_r": max(1.80 if weekend_mode else 1.35, float(profile.get("trail_activation_r", 1.25) or 1.25)),
                    "trail_backoff_r": max(1.00 if weekend_mode else 0.88, float(profile.get("trail_backoff_r", 0.80) or 0.80)),
                    "basket_take_profit_r": 2.15 if weekend_mode else 1.65,
                }
            )
        elif streak_mode == "LOSS_STREAK":
            profile.update(
                {
                    "approved_rr_target": "2.0-2.4" if weekend_mode else "1.5-1.7",
                    "breakeven_trigger_r": min(0.48 if weekend_mode else 0.40, float(profile.get("breakeven_trigger_r", 0.55) or 0.55)),
                    "be_buffer_r": min(0.12 if weekend_mode else 0.10, float(profile.get("be_buffer_r", 0.14) or 0.14)),
                    "min_profit_protection_r": min(0.18 if weekend_mode else 0.16, float(profile.get("min_profit_protection_r", 0.20) or 0.20)),
                    "basket_take_profit_r": 1.15 if weekend_mode else 0.95,
                    "partials": [
                        {"triggerR": 0.50 if weekend_mode else 0.40, "closeFraction": 0.34, "reason": "btc_defensive_lock_1"},
                        {"triggerR": 1.00 if weekend_mode else 0.80, "closeFraction": 0.24, "reason": "btc_defensive_lock_2"},
                    ],
                }
            )
    elif normalized_strategy_key == "USOIL_LONDON_TREND_EXPANSION":
        profile.update(
            {
                "approved_rr_target": "1.8-2.3",
                "breakeven_trigger_r": 0.62,
                "be_buffer_r": 0.16,
                "min_profit_protection_r": 0.22,
                "trail_activation_r": max(1.25, float(profile.get("trail_activation_r", 1.0) or 1.0)),
                "trail_backoff_r": 0.76,
                "no_progress_bars": 7,
                "no_progress_mfe_r": 0.18,
                "early_invalidation_r": -0.40,
                "time_stop_bars": 12,
                "time_stop_max_r": 0.08,
                "partials": [
                    {"triggerR": 0.65, "closeFraction": 0.25, "reason": "usoil_lock_1"},
                    {"triggerR": 1.45, "closeFraction": 0.20, "reason": "usoil_lock_2"},
                ],
            }
        )
        if regime_key in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"} or session_key == "SYDNEY":
            profile.update(
                {
                    "approved_rr_target": "1.4-1.8",
                    "breakeven_trigger_r": 0.50,
                    "be_buffer_r": 0.12,
                    "min_profit_protection_r": 0.16,
                    "trail_activation_r": max(1.00, float(profile.get("trail_activation_r", 1.25) or 1.25)),
                    "trail_backoff_r": 0.58,
                    "no_progress_bars": 5,
                    "no_progress_mfe_r": 0.12,
                    "early_invalidation_r": -0.28,
                    "time_stop_bars": 10,
                    "time_stop_max_r": 0.05,
                    "partials": [
                        {"triggerR": 0.50, "closeFraction": 0.30, "reason": "usoil_defensive_lock_1"},
                        {"triggerR": 1.00, "closeFraction": 0.22, "reason": "usoil_defensive_lock_2"},
                    ],
                }
            )
    elif normalized_strategy_key == "EURUSD_VWAP_PULLBACK":
        profile.update(
            {
                "approved_rr_target": "1.8-2.2",
                "breakeven_trigger_r": 0.60,
                "be_buffer_r": 0.14,
                "min_profit_protection_r": 0.18,
                "trail_activation_r": max(1.15, float(profile.get("trail_activation_r", 1.0) or 1.0)),
                "trail_backoff_r": 0.62,
                "no_progress_bars": 8,
                "no_progress_mfe_r": 0.10,
                "early_invalidation_r": -0.32,
                "time_stop_bars": 12,
                "time_stop_max_r": 0.08,
                "partials": [
                    {"triggerR": 0.60, "closeFraction": 0.25, "reason": "eurusd_vwap_lock_1"},
                    {"triggerR": 1.25, "closeFraction": 0.20, "reason": "eurusd_vwap_lock_2"},
                ],
            }
        )
    elif normalized_strategy_key == "NAS100_VWAP_TREND_STRATEGY":
        profile.update(
            {
                "approved_rr_target": "1.8-2.2",
                "breakeven_trigger_r": 0.60,
                "be_buffer_r": 0.16,
                "min_profit_protection_r": 0.20,
                "trail_activation_r": max(1.10, float(profile.get("trail_activation_r", 1.0) or 1.0)),
                "trail_backoff_r": 0.66,
                "no_progress_bars": 7,
                "no_progress_mfe_r": 0.16,
                "early_invalidation_r": -0.34,
                "time_stop_bars": 11,
                "time_stop_max_r": 0.08,
                "partials": [
                    {"triggerR": 0.60, "closeFraction": 0.22, "reason": "nas_vwap_lock_1"},
                    {"triggerR": 1.20, "closeFraction": 0.18, "reason": "nas_vwap_lock_2"},
                ],
            }
        )
    elif normalized_strategy_key in {"NAS100_LIQUIDITY_SWEEP_REVERSAL", "NAS100_OPENING_DRIVE_BREAKOUT"}:
        profile.update(
            {
                "approved_rr_target": "1.5-1.9",
                "breakeven_trigger_r": 0.55,
                "be_buffer_r": 0.14,
                "min_profit_protection_r": 0.18,
                "trail_activation_r": max(1.05, float(profile.get("trail_activation_r", 1.0) or 1.0)),
                "trail_backoff_r": 0.62,
                "no_progress_bars": 6,
                "no_progress_mfe_r": 0.14,
                "early_invalidation_r": -0.36,
                "time_stop_bars": 10,
                "time_stop_max_r": 0.06,
                "partials": [
                    {"triggerR": 0.55, "closeFraction": 0.25, "reason": "nas_lock_1"},
                    {"triggerR": 1.05, "closeFraction": 0.20, "reason": "nas_lock_2"},
                ],
            }
        )
    elif normalized_strategy_key == "GBPUSD_LONDON_EXPANSION_BREAKOUT":
        profile.update(
            {
                "approved_rr_target": "2.1-2.5",
                "breakeven_trigger_r": 0.55,
                "trail_activation_r": max(1.15, float(profile.get("trail_activation_r", 1.0) or 1.0)),
                "trail_atr": max(0.95, float(profile.get("trail_atr", 1.0) or 1.0)),
                "partials": [
                    {"triggerR": 0.60, "closeFraction": 0.30, "reason": "gbpusd_breakout_lock_1"},
                    {"triggerR": 1.30, "closeFraction": 0.25, "reason": "gbpusd_breakout_lock_2"},
                ],
            }
        )
        if session_key == "LONDON" and regime_key == "LOW_LIQUIDITY_CHOP":
            profile.update(
                {
                    "approved_rr_target": "1.6-2.0",
                    "breakeven_trigger_r": 0.42,
                    "be_buffer_r": 0.12,
                    "min_profit_protection_r": 0.14,
                    "trail_activation_r": max(0.92, min(1.02, float(profile.get("trail_activation_r", 1.15) or 1.15))),
                    "trail_atr": max(0.88, float(profile.get("trail_atr", 0.95) or 0.95)),
                    "trail_backoff_r": 0.50,
                    "basket_take_profit_r": 1.00,
                    "no_progress_bars": 6,
                    "no_progress_mfe_r": 0.10,
                    "time_stop_bars": 9,
                    "time_stop_max_r": 0.04,
                    "partials": [
                        {"triggerR": 0.45, "closeFraction": 0.35, "reason": "gbpusd_breakout_london_chop_lock_1"},
                        {"triggerR": 0.95, "closeFraction": 0.25, "reason": "gbpusd_breakout_london_chop_lock_2"},
                    ],
                }
            )
        if streak_mode == "WIN_STREAK":
            profile.update(
                {
                    "approved_rr_target": "2.3-2.8",
                    "trail_activation_r": max(1.30, float(profile.get("trail_activation_r", 1.15) or 1.15)),
                    "partials": [
                        {"triggerR": 0.65, "closeFraction": 0.30, "reason": "gbpusd_breakout_streak_lock_1"},
                        {"triggerR": 1.45, "closeFraction": 0.25, "reason": "gbpusd_breakout_streak_lock_2"},
                    ],
                }
            )
        elif streak_mode == "LOSS_STREAK":
            profile.update(
                {
                    "approved_rr_target": "1.8-2.2",
                    "breakeven_trigger_r": 0.45,
                    "trail_activation_r": max(1.00, min(1.10, float(profile.get("trail_activation_r", 1.15) or 1.15))),
                    "partials": [
                        {"triggerR": 0.45, "closeFraction": 0.35, "reason": "gbpusd_breakout_defensive_lock_1"},
                        {"triggerR": 1.00, "closeFraction": 0.25, "reason": "gbpusd_breakout_defensive_lock_2"},
                    ],
                }
            )
    elif normalized_strategy_key == "GBPUSD_TREND_PULLBACK_RIDE":
        profile.update(
            {
                "approved_rr_target": "1.8-2.2",
                "breakeven_trigger_r": 0.45,
                "trail_activation_r": max(0.95, min(1.05, float(profile.get("trail_activation_r", 1.0) or 1.0))),
                "trail_atr": max(0.90, float(profile.get("trail_atr", 1.0) or 1.0)),
                "partials": [
                    {"triggerR": 0.45, "closeFraction": 0.35, "reason": "gbpusd_pullback_lock_1"},
                    {"triggerR": 0.95, "closeFraction": 0.30, "reason": "gbpusd_pullback_lock_2"},
                ],
            }
        )
        if streak_mode == "WIN_STREAK":
            profile.update(
                {
                    "approved_rr_target": "2.0-2.4",
                    "trail_activation_r": max(1.05, float(profile.get("trail_activation_r", 0.95) or 0.95)),
                }
            )
        elif streak_mode == "LOSS_STREAK":
            profile.update(
                {
                    "approved_rr_target": "1.6-1.9",
                    "breakeven_trigger_r": 0.40,
                    "partials": [
                        {"triggerR": 0.40, "closeFraction": 0.40, "reason": "gbpusd_pullback_defensive_lock_1"},
                        {"triggerR": 0.80, "closeFraction": 0.30, "reason": "gbpusd_pullback_defensive_lock_2"},
                    ],
                }
            )
    elif normalized_strategy_key == "AUDJPY_LIQUIDITY_SWEEP_REVERSAL":
        profile.update(
            {
                "approved_rr_target": "1.3-1.7",
                "breakeven_trigger_r": 0.40,
                "be_buffer_r": 0.10,
                "min_profit_protection_r": 0.12,
                "trail_activation_r": max(0.80, float(profile.get("trail_activation_r", 1.0) or 1.0)),
                "trail_backoff_r": 0.46,
                "basket_take_profit_r": 0.95,
                "no_progress_bars": 5,
                "no_progress_mfe_r": 0.10,
                "early_invalidation_r": -0.24,
                "time_stop_bars": 8,
                "time_stop_max_r": 0.04,
                "partials": [
                    {"triggerR": 0.40, "closeFraction": 0.32, "reason": "audjpy_sweep_lock_1"},
                    {"triggerR": 0.85, "closeFraction": 0.24, "reason": "audjpy_sweep_lock_2"},
                ],
            }
        )
        if session_key == "OVERLAP" and regime_key == "MEAN_REVERSION":
            profile.update(
                {
                    "approved_rr_target": "1.5-1.9",
                    "breakeven_trigger_r": 0.42,
                    "be_buffer_r": 0.12,
                    "min_profit_protection_r": 0.14,
                    "trail_activation_r": max(0.86, float(profile.get("trail_activation_r", 0.80) or 0.80)),
                    "trail_backoff_r": 0.48,
                    "basket_take_profit_r": 1.02,
                    "no_progress_bars": 6,
                    "no_progress_mfe_r": 0.12,
                }
            )
        elif session_key == "SYDNEY" and regime_key == "MEAN_REVERSION":
            profile.update(
                {
                    "approved_rr_target": "1.2-1.5",
                    "breakeven_trigger_r": 0.34,
                    "basket_take_profit_r": 0.82,
                    "partials": [
                        {"triggerR": 0.36, "closeFraction": 0.34, "reason": "audjpy_sydney_sweep_lock_1"},
                        {"triggerR": 0.78, "closeFraction": 0.24, "reason": "audjpy_sydney_sweep_lock_2"},
                    ],
                }
            )
    elif normalized_strategy_key in {
        "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
        "AUDJPY_LONDON_CARRY_TREND",
        "AUDJPY_ATR_COMPRESSION_BREAKOUT",
        "NZDJPY_PULLBACK_CONTINUATION",
        "USDJPY_MACRO_TREND_RIDE",
        "USDJPY_LIQUIDITY_SWEEP_REVERSAL",
        "AUDNZD_RANGE_ROTATION",
    }:
        profile.update(
            {
                "approved_rr_target": "1.7-2.2" if "RANGE_ROTATION" not in normalized_strategy_key and "LIQUIDITY_SWEEP" not in normalized_strategy_key else "1.4-1.8",
                "breakeven_trigger_r": 0.58 if "RANGE_ROTATION" not in normalized_strategy_key else 0.50,
                "be_buffer_r": 0.15 if "RANGE_ROTATION" not in normalized_strategy_key else 0.10,
                "min_profit_protection_r": 0.18 if "RANGE_ROTATION" not in normalized_strategy_key else 0.14,
                "trail_activation_r": max(1.10, float(profile.get("trail_activation_r", 1.0) or 1.0)),
                "trail_backoff_r": 0.66 if "RANGE_ROTATION" not in normalized_strategy_key else 0.50,
                "no_progress_bars": 7 if "RANGE_ROTATION" not in normalized_strategy_key else 6,
                "no_progress_mfe_r": 0.12,
                "early_invalidation_r": -0.36 if "RANGE_ROTATION" not in normalized_strategy_key else -0.28,
                "time_stop_bars": 12 if "RANGE_ROTATION" not in normalized_strategy_key else 10,
                "time_stop_max_r": 0.06 if "RANGE_ROTATION" not in normalized_strategy_key else 0.08,
                "partials": [
                    {"triggerR": 0.60 if "RANGE_ROTATION" not in normalized_strategy_key else 0.50, "closeFraction": 0.25, "reason": "session_trend_lock_1"},
                    {"triggerR": 1.30 if "RANGE_ROTATION" not in normalized_strategy_key else 1.00, "closeFraction": 0.20, "reason": "session_trend_lock_2"},
                ],
            }
        )
    elif normalized_strategy_key == "NAS100_LIQUIDITY_SWEEP_REVERSAL":
        profile["approved_rr_target"] = "1.6-2.0"
    if (
        _is_super_aggressive_normal_symbol(symbol_key)
        and not normalized_strategy_key.startswith(("XAUUSD", "BTCUSD"))
    ):
        explicit_specialized_profile = normalized_strategy_key in {
            "USOIL_LONDON_TREND_EXPANSION",
            "EURUSD_VWAP_PULLBACK",
            "NAS100_VWAP_TREND_STRATEGY",
            "NAS100_LIQUIDITY_SWEEP_REVERSAL",
            "NAS100_OPENING_DRIVE_BREAKOUT",
            "GBPUSD_LONDON_EXPANSION_BREAKOUT",
            "GBPUSD_TREND_PULLBACK_RIDE",
            "AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
            "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
            "AUDJPY_LONDON_CARRY_TREND",
            "AUDJPY_ATR_COMPRESSION_BREAKOUT",
            "NZDJPY_PULLBACK_CONTINUATION",
            "USDJPY_MACRO_TREND_RIDE",
            "USDJPY_LIQUIDITY_SWEEP_REVERSAL",
            "AUDNZD_RANGE_ROTATION",
        }
        home_session = _is_super_aggressive_home_session(symbol_key, session_key)
        trend_like = any(
            token in normalized_strategy_key
            for token in ("BREAKOUT", "IMPULSE", "CONTINUATION", "TREND", "PULLBACK", "EXPANSION", "MOMENTUM")
        )
        reversal_like = any(
            token in normalized_strategy_key
            for token in ("SWEEP", "REVERSION", "RANGE", "ROTATION", "VWAP", "TRAP")
        )
        weak_context = bool(
            (symbol_key == "GBPUSD" and normalized_strategy_key == "GBPUSD_LONDON_EXPANSION_BREAKOUT" and session_key == "LONDON" and regime_key in {"RANGING", "LOW_LIQUIDITY_CHOP"})
            or (symbol_key == "NAS100" and normalized_strategy_key == "NAS100_OPENING_DRIVE_BREAKOUT" and regime_key != "BREAKOUT_EXPANSION")
            or (symbol_key == "USOIL" and normalized_strategy_key == "USOIL_LONDON_TREND_EXPANSION" and regime_key not in {"TRENDING", "BREAKOUT_EXPANSION"})
        )
        if not weak_context and not explicit_specialized_profile:
            profile.update(
                {
                    "approved_rr_target": (
                        "2.2-2.9"
                        if home_session and trend_like
                        else "1.8-2.4"
                        if home_session and reversal_like
                        else "1.9-2.5"
                        if trend_like
                        else "1.6-2.1"
                    ),
                    "breakeven_trigger_r": min(
                        0.48 if home_session else 0.54,
                        float(profile.get("breakeven_trigger_r", 0.60) or 0.60),
                    ),
                    "be_buffer_r": max(
                        0.12 if reversal_like else 0.14,
                        float(profile.get("be_buffer_r", 0.10) or 0.10),
                    ),
                    "min_profit_protection_r": max(
                        0.16 if reversal_like else 0.20,
                        float(profile.get("min_profit_protection_r", 0.12) or 0.12),
                    ),
                    "trail_activation_r": max(
                        1.05 if home_session else 1.12,
                        float(profile.get("trail_activation_r", 1.0) or 1.0),
                    ),
                    "trail_atr": max(
                        0.92 if home_session else 0.96,
                        float(profile.get("trail_atr", 1.0) or 1.0),
                    ),
                    "trail_backoff_r": max(
                        0.72 if trend_like else 0.58,
                        float(profile.get("trail_backoff_r", 0.50) or 0.50),
                    ),
                    "basket_take_profit_r": max(
                        1.65 if trend_like and home_session else 1.35 if trend_like else 1.10,
                        float(profile.get("basket_take_profit_r", 0.0) or 0.0),
                    ),
                    "no_progress_bars": max(
                        7 if trend_like else 6,
                        int(profile.get("no_progress_bars", 0) or 0),
                    ),
                    "no_progress_mfe_r": max(
                        0.14 if trend_like else 0.12,
                        float(profile.get("no_progress_mfe_r", 0.0) or 0.0),
                    ),
                    "time_stop_bars": max(
                        12 if trend_like else 10,
                        int(profile.get("time_stop_bars", 0) or 0),
                    ),
                    "partials": [
                        {
                            "triggerR": 0.48 if home_session else 0.52,
                            "closeFraction": 0.22,
                            "reason": "super_aggro_lock_1",
                        },
                        {
                            "triggerR": 1.05 if reversal_like else 1.20,
                            "closeFraction": 0.18,
                            "reason": "super_aggro_lock_2",
                        },
                    ],
                }
            )
            if streak_mode == "WIN_STREAK":
                profile.update(
                    {
                        "approved_rr_target": (
                            "2.5-3.2" if trend_like and home_session else "2.0-2.6"
                        ),
                        "trail_activation_r": max(
                            1.22 if trend_like else 1.10,
                            float(profile.get("trail_activation_r", 1.05) or 1.05),
                        ),
                        "trail_backoff_r": max(
                            0.82 if trend_like else 0.64,
                            float(profile.get("trail_backoff_r", 0.58) or 0.58),
                        ),
                        "basket_take_profit_r": max(
                            1.90 if trend_like and home_session else 1.55,
                            float(profile.get("basket_take_profit_r", 0.0) or 0.0),
                        ),
                    }
                )
            elif streak_mode == "LOSS_STREAK":
                profile.update(
                    {
                        "approved_rr_target": (
                            "1.7-2.1" if trend_like else "1.5-1.9"
                        ),
                        "breakeven_trigger_r": min(
                            0.42,
                            float(profile.get("breakeven_trigger_r", 0.48) or 0.48),
                        ),
                        "trail_activation_r": max(
                            0.96,
                            min(1.08, float(profile.get("trail_activation_r", 1.12) or 1.12)),
                        ),
                        "partials": [
                            {"triggerR": 0.42, "closeFraction": 0.28, "reason": "super_aggro_defensive_lock_1"},
                            {"triggerR": 0.90, "closeFraction": 0.20, "reason": "super_aggro_defensive_lock_2"},
                        ],
                    }
                )
    profile["streak_adjust_mode"] = str(streak_mode)
    if not _exit_partials_enabled(exits_config):
        profile["partials"] = []
        profile["trail_requires_partial1"] = False
    return profile


def _hydrate_candidate_strategy_meta(
    *,
    symbol_key: str,
    candidate: Any,
    session_name: str,
    row: Any | None = None,
    regime: Any | None = None,
    symbol_info: dict[str, Any] | None = None,
    max_spread_points: float = 60.0,
) -> dict[str, Any]:
    normalized_symbol = _normalize_symbol_key(symbol_key)
    candidate_meta = dict(getattr(candidate, "meta", {}) or {})
    setup_name = str(getattr(candidate, "setup", "") or "")
    strategy_family = normalize_strategy_family(
        str(candidate_meta.get("setup_family") or getattr(candidate, "strategy_family", "") or setup_name)
    )
    lane_name = str(
        candidate_meta.get("lane_name")
        or infer_trade_lane(
            symbol=normalized_symbol,
            setup=setup_name,
            setup_family=strategy_family,
            session_name=session_name,
        )
    )
    strategy_key = str(
        candidate_meta.get("strategy_key")
        or resolve_strategy_key(normalized_symbol, setup_name)
    ).strip()
    raw_regime_state = str(
        candidate_meta.get("regime_state")
        or candidate_meta.get("regime")
        or getattr(regime, "state_label", "")
        or getattr(regime, "label", "")
        or ""
    )
    regime_state = runtime_regime_state(raw_regime_state)
    timestamp_value = candidate_meta.get("time") or candidate_meta.get("timestamp") or getattr(row, "get", lambda *_args, **_kwargs: None)("time")
    weekend_mode = bool(is_weekend_market_mode(pd.Timestamp(timestamp_value).to_pydatetime())) if timestamp_value is not None else False
    priority = session_priority_context(
        symbol=normalized_symbol,
        lane_name=lane_name,
        session_name=session_name,
    )

    row_values = row if row is not None else {}

    def _row_float(key: str, default: float = 0.0) -> float:
        try:
            value = row_values.get(key, default)
        except AttributeError:
            value = default
        try:
            return float(value or default)
        except (TypeError, ValueError):
            return float(default)

    raw_spread_points = max(
        0.0,
        float(
            candidate_meta.get("spread_points")
            or _row_float("m5_spread")
            or _row_float("spread_points")
            or 0.0
        ),
    )
    spread_points = _normalize_runtime_spread_points(
        normalized_symbol,
        raw_spread_points,
        symbol_info=symbol_info,
        max_spread_points=max_spread_points,
    )
    volatility_state = str(
        candidate_meta.get("volatility_state")
        or getattr(regime, "details", {}).get("volatility_forecast_state", "BALANCED")
        or "BALANCED"
    ).upper()
    pressure_alignment = clamp(
        float(
            candidate_meta.get("pressure_alignment")
            or getattr(regime, "details", {}).get("pressure_proxy_score", 0.60)
            or 0.60
        ),
        0.0,
        1.0,
    )
    structure_score = clamp(
        max(
            float(candidate_meta.get("structure_score", 0.0) or 0.0),
            _row_float("m5_structure_score"),
            _row_float("m15_structure_score"),
            _row_float("m5_body_efficiency", _row_float("m5_candle_efficiency", 0.55)),
            clamp(float(getattr(candidate, "confluence_score", 0.0) or 0.0) / 5.0, 0.0, 1.0),
        ),
        0.0,
        1.0,
    )
    liquidity_score = clamp(
        float(
            candidate_meta.get("liquidity_score")
            or max(
                float(getattr(regime, "details", {}).get("absorption_signal", 0.0) or 0.0),
                1.0 - clamp(_row_float("m5_range_position_20", 0.5) - 0.5, -0.5, 0.5) ** 2,
            )
        ),
        0.0,
        1.0,
    )
    structure_clean = float(candidate_meta.get("structure_cleanliness_score", 0.0) or 0.0)
    if structure_clean <= 0.0:
        structure_clean = structure_cleanliness_score(
            spread_points=spread_points,
            spread_limit=float(max_spread_points),
            structure_score=structure_score,
            liquidity_score=liquidity_score,
            volatility_state=volatility_state,
            regime_state=regime_state,
            pressure_alignment=pressure_alignment,
        )
    atr_ratio = clamp(_row_float("m5_atr_pct_of_avg", 1.0), 0.0, 3.0)
    body_efficiency = _row_float("m5_body_efficiency", _row_float("m5_candle_efficiency", 0.55))
    range_position = _row_float("m15_range_position_20", _row_float("m5_range_position_20", 0.5))
    volume_ratio = _row_float("m5_volume_ratio_20", _row_float("m15_volume_ratio_20", 1.0))
    delta_proxy = float(candidate_meta.get("delta_proxy_score", 0.0) or 0.0)
    if delta_proxy == 0.0:
        delta_proxy = delta_proxy_score(
            side=str(getattr(candidate, "side", candidate_meta.get("side", "")) or ""),
            body_efficiency=body_efficiency,
            short_return=_row_float("m5_ret_1", _row_float("m15_ret_1", 0.0)),
            range_position=range_position,
            volume_ratio=volume_ratio,
            upper_wick_ratio=_row_float("m5_upper_wick_ratio"),
            lower_wick_ratio=_row_float("m5_lower_wick_ratio"),
        )
    compression_proxy_state = str(
        candidate_meta.get("compression_proxy_state")
        or getattr(regime, "details", {}).get("compression_proxy_state", "NEUTRAL")
        or "NEUTRAL"
    ).upper()
    compression_expansion_score = clamp(
        float(
            candidate_meta.get("compression_expansion_score")
            or getattr(regime, "details", {}).get("compression_expansion_score", 0.0)
            or 0.0
        ),
        0.0,
        1.0,
    )
    session_loosen = float(candidate_meta.get("session_loosen_factor", 0.0) or 0.0)
    if session_loosen <= 0.0:
        session_loosen = session_loosen_factor(
            session_name=session_name,
            symbol=normalized_symbol,
            weekend_mode=weekend_mode,
            candidate_scarcity=False,
        )
    market_instability = clamp(
        float(
            candidate_meta.get("market_instability_score")
            or _row_float("market_instability_score")
            or getattr(regime, "details", {}).get("market_instability_score", 0.0)
            or 0.0
        ),
        0.0,
        1.0,
    )
    feature_drift = clamp(
        float(
            candidate_meta.get("feature_drift_score")
            or _row_float("feature_drift_score")
            or getattr(regime, "details", {}).get("feature_drift_score", 0.0)
            or 0.0
        ),
        0.0,
        1.0,
    )
    multi_tf_alignment = clamp(
        float(
            candidate_meta.get("multi_tf_alignment_score")
            or _row_float("multi_tf_alignment_score")
            or getattr(regime, "details", {}).get("multi_tf_alignment_score", 0.5)
            or 0.5
        ),
        0.0,
        1.0,
    )
    seasonality_edge = clamp(
        float(
            candidate_meta.get("seasonality_edge_score")
            or _row_float("seasonality_edge_score")
            or getattr(regime, "details", {}).get("seasonality_edge_score", 0.5)
            or 0.5
        ),
        0.0,
        1.0,
    )
    fractal_persistence = clamp(
        float(
            candidate_meta.get("fractal_persistence_score")
            or _row_float("fractal_persistence_score")
            or getattr(regime, "details", {}).get("fractal_persistence_score", 0.5)
            or 0.5
        ),
        0.0,
        1.0,
    )
    trend_like_strategy = any(token in strategy_key for token in ("TREND", "CONTINUATION", "BREAKOUT", "IMPULSE", "EXPANSION", "MOMENTUM"))
    grid_like_strategy = "GRID" in strategy_key
    mean_reversion_like_strategy = any(token in strategy_key for token in ("SWEEP", "REVERSION", "RANGE", "FADE"))
    chase_penalty = 0.0
    if atr_ratio >= 1.60 and body_efficiency >= 0.82 and any(
        token in strategy_key for token in ("BREAKOUT", "IMPULSE", "EXPANSION")
    ):
        chase_penalty = 0.12
    if any(token in strategy_key for token in ("TREND_SCALP", "CONTINUATION", "BREAKOUT", "IMPULSE")):
        chase_penalty += max(0.0, 0.40 - body_efficiency) * 0.20
    instability_penalty = market_instability * (0.14 if trend_like_strategy else (0.10 if grid_like_strategy else 0.06))
    structure_bonus = (multi_tf_alignment * (0.08 if trend_like_strategy else 0.05)) + (fractal_persistence * (0.06 if trend_like_strategy or grid_like_strategy else 0.03))
    seasonality_bonus = seasonality_edge * (0.05 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.03)
    chase_penalty *= float(session_loosen)
    chase_penalty += instability_penalty
    entry_timing = float(candidate_meta.get("entry_timing_score", 0.0) or 0.0)
    if entry_timing <= 0.0:
        entry_timing = entry_timing_score(
            structure_cleanliness=structure_clean,
            probability=clamp(
                float(candidate_meta.get("probability", getattr(candidate, "score_hint", 0.0)) or 0.0),
                0.0,
                1.0,
            ),
            expected_value_r=float(candidate_meta.get("expected_value_r", getattr(candidate, "tp_r", 0.0)) or 0.0),
            spread_points=spread_points,
            spread_limit=float(max_spread_points),
            volatility_state=volatility_state,
            regime_state=regime_state,
            chase_penalty=chase_penalty,
            delta_proxy_score_value=delta_proxy,
        )
    entry_timing = clamp(
        float(entry_timing) + structure_bonus + seasonality_bonus - (feature_drift * 0.06),
        0.0,
        1.0,
    )
    regime_fit = float(candidate_meta.get("regime_fit", 0.0) or 0.0)
    if regime_fit <= 0.0:
        regime_fit = strategy_regime_fit(strategy_key, regime_state)
    regime_fit = clamp(
        float(regime_fit)
        + (0.06 * max(0.0, delta_proxy))
        + float(compression_strategy_bias(strategy_key, compression_proxy_state) * 0.50),
        0.0,
        1.0,
    )
    regime_fit = clamp(regime_fit + structure_bonus - instability_penalty, 0.0, 1.0)
    session_fit = float(candidate_meta.get("session_fit", 0.0) or 0.0)
    if session_fit <= 0.0:
        session_fit = clamp(float(priority.session_priority_multiplier) - 0.10, 0.0, 1.0)
    session_fit = clamp(float(session_fit) + seasonality_bonus - (0.03 * feature_drift), 0.0, 1.0)
    volatility_fit = float(candidate_meta.get("volatility_fit", 0.0) or 0.0)
    if volatility_fit <= 0.0:
        volatility_fit = StrategyRouter._volatility_fit(
            strategy_key,
            row_values,
            regime,
        ) if row is not None and regime is not None else 0.70
    pair_fit = float(candidate_meta.get("pair_behavior_fit", 0.0) or 0.0)
    if pair_fit <= 0.0:
        pair_fit = pair_behavior_fit(
            symbol=normalized_symbol,
            strategy_key=strategy_key,
            session_name=session_name,
            regime_state=regime_state,
            weekend_mode=weekend_mode,
        )
    pair_fit = clamp(
        float(pair_fit)
        + (0.04 * seasonality_edge)
        + (0.04 * fractal_persistence)
        - (0.04 * market_instability),
        0.0,
        1.0,
    )
    strategy_recent_seed = float(candidate_meta.get("strategy_recent_performance_seed", 0.0) or 0.0)
    if strategy_recent_seed <= 0.0:
        strategy_recent_seed = StrategyRouter._strategy_performance_seed(
            symbol=normalized_symbol,
            strategy_key=strategy_key,
            session_name=session_name,
        )
    execution_fit = float(candidate_meta.get("execution_quality_fit", 0.0) or 0.0)
    if execution_fit <= 0.0:
        execution_fit = clamp(1.0 - (spread_points / max(1.0, float(max_spread_points))), 0.0, 1.0)
    execution_fit = clamp(float(execution_fit) - (0.04 * feature_drift) - (0.03 * market_instability), 0.0, 1.0)
    router_rank_score = float(
        candidate_meta.get("router_rank_score", getattr(candidate, "score_hint", 0.0)) or 0.0
    )
    if router_rank_score <= 0.0:
        router_rank_score = strategy_selection_score(
            ev_estimate=clamp(float(getattr(candidate, "score_hint", 0.0) or 0.0), 0.0, 1.0),
            regime_fit=regime_fit,
            session_fit=session_fit,
            volatility_fit=volatility_fit,
            pair_behavior_fit_score=pair_fit,
            strategy_recent_performance=strategy_recent_seed,
            execution_quality_fit=execution_fit,
            entry_timing_score_value=entry_timing,
            structure_cleanliness_score_value=structure_clean,
            drawdown_penalty=0.0,
            false_break_penalty=max(0.0, 0.55 - structure_clean),
            chop_penalty=0.20 if regime_state == "LOW_LIQUIDITY_CHOP" and any(
                token in strategy_key for token in ("BREAKOUT", "IMPULSE", "EXPANSION")
            ) else 0.0,
        )
    router_rank_score = clamp(
        float(router_rank_score)
        + winner_promotion_bonus(
            symbol=normalized_symbol,
            strategy_key=strategy_key,
            regime_state=regime_state,
            session_name=session_name,
            weekend_mode=weekend_mode,
        )
        + (0.05 * max(0.0, delta_proxy))
        + float(compression_strategy_bias(strategy_key, compression_proxy_state)),
        0.0,
        1.0,
    )
    router_rank_score = clamp(
        float(router_rank_score)
        + structure_bonus
        + seasonality_bonus
        - instability_penalty
        - (0.04 * feature_drift),
        0.0,
        1.0,
    )
    throughput_recovery_active = bool(candidate_meta.get("throughput_recovery_active", False))
    quality_tier = str(candidate_meta.get("quality_tier") or "").upper()
    if not quality_tier:
        quality_tier = quality_tier_from_scores(
            structure_cleanliness=structure_clean,
            regime_fit=regime_fit,
            execution_quality_fit=execution_fit,
            high_liquidity=bool(session_name in {"LONDON", "OVERLAP", "NEW_YORK"} or (normalized_symbol == "BTCUSD" and weekend_mode)),
            throughput_recovery_active=throughput_recovery_active,
        )
    tier_size_multiplier = float(candidate_meta.get("tier_size_multiplier", 0.0) or 0.0)
    if tier_size_multiplier <= 0.0:
        tier_size_multiplier = quality_tier_size_multiplier(
            quality_tier=quality_tier,
            strategy_score=router_rank_score,
        )
    xau_grid_scaler_ready = bool(
        normalized_symbol == "XAUUSD"
        and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
        and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
        and regime_state in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP", "RANGING"}
        and compression_proxy_state == "EXPANSION_READY"
        and compression_expansion_score >= 0.49
        and delta_proxy >= 0.15
        and market_instability <= 0.40
        and feature_drift <= 0.35
    )
    if xau_grid_scaler_ready:
        session_loosen = max(session_loosen, 0.92)
        entry_timing = clamp(entry_timing + 0.06, 0.0, 1.0)
        regime_fit = clamp(regime_fit + 0.05, 0.0, 1.0)
        session_fit = clamp(session_fit + 0.05, 0.0, 1.0)
        execution_fit = clamp(execution_fit + 0.04, 0.0, 1.0)
        router_rank_score = clamp(router_rank_score + 0.08, 0.0, 1.0)
        if quality_tier == "C":
            quality_tier = "B"
        tier_size_multiplier = max(
            float(tier_size_multiplier),
            quality_tier_size_multiplier(
                quality_tier=quality_tier,
                strategy_score=router_rank_score,
            ),
        )

    hydrated = {
        **candidate_meta,
        "strategy_key": strategy_key,
        "strategy_pool": list(candidate_meta.get("strategy_pool") or StrategyRouter._strategy_pool_keys(normalized_symbol)),
        "lane_name": lane_name,
        "regime_state": regime_state,
        "allowed_regimes": list(candidate_meta.get("allowed_regimes") or strategy_allowed_regimes(strategy_key)),
        "management_template": str(candidate_meta.get("management_template") or strategy_management_template(strategy_key)),
        "regime_fit": float(regime_fit),
        "session_fit": float(session_fit),
        "volatility_fit": float(volatility_fit),
        "pair_behavior_fit": float(pair_fit),
        "strategy_recent_performance_seed": float(strategy_recent_seed),
        "execution_quality_fit": float(execution_fit),
        "spread_points_raw": float(raw_spread_points),
        "spread_points_effective": float(spread_points),
        "entry_timing_score": float(entry_timing),
        "structure_cleanliness_score": float(structure_clean),
        "router_rank_score": float(clamp(router_rank_score, 0.0, 1.0)),
        "quality_tier": str(quality_tier),
        "tier_size_multiplier": float(tier_size_multiplier),
        "delta_proxy_score": float(delta_proxy),
        "compression_proxy_state": str(compression_proxy_state),
        "compression_expansion_score": float(compression_expansion_score),
        "market_instability_score": float(market_instability),
        "feature_drift_score": float(feature_drift),
        "multi_tf_alignment_score": float(multi_tf_alignment),
        "seasonality_edge_score": float(seasonality_edge),
        "fractal_persistence_score": float(fractal_persistence),
        "transition_momentum": float(candidate_meta.get("transition_momentum", 0.0) or 0.0),
        "transition_momentum_size_multiplier": float(candidate_meta.get("transition_momentum_size_multiplier", 1.0) or 1.0),
        "compression_burst_size_multiplier": float(candidate_meta.get("compression_burst_size_multiplier", 1.0) or 1.0),
        "velocity_decay": float(candidate_meta.get("velocity_decay", 1.0) or 1.0),
        "velocity_decay_score_penalty": float(candidate_meta.get("velocity_decay_score_penalty", 0.0) or 0.0),
        "velocity_trades_per_10_bars": float(candidate_meta.get("velocity_trades_per_10_bars", 0.0) or 0.0),
        "correlation_penalty": float(candidate_meta.get("correlation_penalty", 0.0) or 0.0),
        "btc_weekend_mode": bool(candidate_meta.get("btc_weekend_mode", False)),
        "grid_spacing_multiplier_hint": float(candidate_meta.get("grid_spacing_multiplier_hint", 1.0) or 1.0),
        "session_loosen_factor": float(session_loosen),
        "throughput_recovery_active": bool(throughput_recovery_active),
        "session_priority_profile": str(priority.session_priority_profile),
        "session_native_pair": bool(priority.session_native_pair),
        "session_priority_multiplier": float(priority.session_priority_multiplier),
        "pair_priority_rank_in_session": int(priority.pair_priority_rank_in_session),
        "lane_budget_share": float(priority.lane_budget_share),
        "lane_session_priority": str(priority.lane_session_priority),
    }
    candidate.meta = hydrated
    return hydrated


def _candidate_strategy_pool_rankings(
    *,
    symbol_key: str,
    candidates: list[Any],
    session_name: str,
    row: Any | None = None,
    regime: Any | None = None,
    symbol_info: dict[str, Any] | None = None,
    max_spread_points: float = 60.0,
    closed_trades: list[dict[str, Any]] | None = None,
    current_day_key: str = "",
    candidate_tier_config: dict[str, Any] | None = None,
    orchestrator_config: dict[str, Any] | None = None,
) -> list[tuple[Any, dict[str, Any]]]:
    ranked: list[tuple[Any, dict[str, Any]]] = []
    normalized_symbol = _normalize_symbol_key(symbol_key)
    closed_trades = list(closed_trades or [])
    current_day_key = str(current_day_key or "")
    candidate_tier_config = dict(candidate_tier_config or {})
    orchestrator_config = dict(orchestrator_config or {})
    super_aggro_symbol = _is_super_aggressive_normal_symbol(normalized_symbol)
    super_aggro_home_session = _is_super_aggressive_home_session(normalized_symbol, session_name)
    b_tier_min = float(candidate_tier_config.get("b_tier_size_mult_min", 0.70))
    b_tier_max = float(candidate_tier_config.get("b_tier_size_mult_max", 0.90))
    candidate_scarcity_loosen_pct = float(candidate_tier_config.get("candidate_scarcity_loosen_pct", 0.12))
    btc_weekend_size_boost = float(candidate_tier_config.get("btc_weekend_size_boost", 1.20) or 1.20)
    btc_velocity_trigger = float(
        candidate_tier_config.get("btc_velocity_decay_trigger_trades_per_10_bars", 1.50) or 1.50
    )
    xau_grid_compression_spacing_multiplier = float(
        candidate_tier_config.get("xau_grid_compression_spacing_multiplier", 1.30) or 1.30
    )
    row_timestamp = None
    try:
        row_timestamp = row.get("time") if row is not None else None
    except Exception:
        row_timestamp = None
    weekend_mode = bool(is_weekend_market_mode(pd.Timestamp(row_timestamp).to_pydatetime())) if row_timestamp is not None else False
    high_liquidity_session = bool(session_name in {"LONDON", "OVERLAP", "NEW_YORK"} or (normalized_symbol == "BTCUSD" and weekend_mode))
    throughput_recovery_threshold = 3 if high_liquidity_session else 2
    if super_aggro_symbol:
        throughput_recovery_threshold = 4 if super_aggro_home_session else max(throughput_recovery_threshold, 3)
    throughput_recovery_active = bool(len(candidates) <= throughput_recovery_threshold)
    equity_momentum = _equity_momentum_throttle(closed_trades, orchestrator_config)
    throughput_loosen_pct = float(orchestrator_config.get("throughput_density_loosen_pct", candidate_scarcity_loosen_pct) or candidate_scarcity_loosen_pct)
    for candidate in candidates:
        candidate_meta = _hydrate_candidate_strategy_meta(
            symbol_key=normalized_symbol,
            candidate=candidate,
            session_name=session_name,
            row=row,
            regime=regime,
            symbol_info=symbol_info,
            max_spread_points=max_spread_points,
        )
        setup_name = str(getattr(candidate, "setup", "") or "")
        strategy_family = normalize_strategy_family(
            str(candidate_meta.get("setup_family") or getattr(candidate, "strategy_family", "") or setup_name)
        )
        recycle_session = bool(candidate_meta.get("recycle_session", False))
        recycle_origin_session = str(candidate_meta.get("recycle_origin_session") or "")
        recycle_boost_applied = float(candidate_meta.get("recycle_boost_applied", 0.0) or 0.0)
        lane_name = str(
            candidate_meta.get("lane_name")
            or infer_trade_lane(
                symbol=normalized_symbol,
                setup=setup_name,
                setup_family=strategy_family,
                session_name=session_name,
            )
        )
        priority = session_priority_context(
            symbol=normalized_symbol,
            lane_name=lane_name,
            session_name=session_name,
        )
        strategy_key = str(
            candidate_meta.get("strategy_key")
            or resolve_strategy_key(normalized_symbol, setup_name)
        ).strip()
        super_aggro_strategy = bool(
            super_aggro_symbol
            and any(
                token in strategy_key
                for token in (
                    "BREAKOUT",
                    "IMPULSE",
                    "CONTINUATION",
                    "TREND",
                    "PULLBACK",
                    "SWEEP",
                    "VWAP",
                    "ROTATION",
                    "TRAP",
                    "EXPANSION",
                    "MOMENTUM",
                    "RETEST",
                )
            )
        )
        xau_engine = str(candidate_meta.get("xau_engine") or "").upper()
        if (
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and xau_engine.startswith("GRID_DIRECTIONAL_MIRROR")
            and not bool(candidate_meta.get("mirror_emergency_fallback", False))
            and not bool(candidate_meta.get("mirror_live_enabled", False))
        ):
            continue
        regime_state = str(candidate_meta.get("regime_state") or candidate_meta.get("regime") or "").strip().upper()
        xau_grid_prime_session_candidate = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and not weekend_mode
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
        )
        xau_grid_density_first_active = bool(candidate_meta.get("quota_density_first_active", False))
        xau_grid_quota_debt_10m = max(0, int(candidate_meta.get("quota_debt_10m", 0) or 0))
        xau_grid_soft_penalty_score = float(candidate_meta.get("soft_quality_penalty_score", 0.0) or 0.0)
        xau_directional_throughput_recovery_blocked = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ATR_EXPANSION_SCALPER"
        )
        candidate_throughput_recovery_active = bool(
            throughput_recovery_active
            and not xau_grid_prime_session_candidate
            and not xau_directional_throughput_recovery_blocked
        )
        router_rank_score = clamp(
            float(candidate_meta.get("router_rank_score", getattr(candidate, "score_hint", 0.0)) or 0.0),
            0.0,
            1.0,
        )
        regime_fit = clamp(float(candidate_meta.get("regime_fit", 0.0) or 0.0), 0.0, 1.0)
        session_fit = clamp(float(candidate_meta.get("session_fit", 0.0) or 0.0), 0.0, 1.0)
        volatility_fit = clamp(float(candidate_meta.get("volatility_fit", 0.0) or 0.0), 0.0, 1.0)
        pair_behavior_fit = clamp(float(candidate_meta.get("pair_behavior_fit", 0.0) or 0.0), 0.0, 1.0)
        execution_quality_fit = clamp(float(candidate_meta.get("execution_quality_fit", 0.0) or 0.0), 0.0, 1.0)
        entry_timing_score = clamp(float(candidate_meta.get("entry_timing_score", 0.0) or 0.0), 0.0, 1.0)
        structure_cleanliness_score = clamp(
            float(candidate_meta.get("structure_cleanliness_score", 0.0) or 0.0),
            0.0,
            1.0,
        )
        quality_tier = str(candidate_meta.get("quality_tier") or "").upper()
        if not quality_tier:
            quality_tier = quality_tier_from_scores(
                structure_cleanliness=structure_cleanliness_score,
                regime_fit=regime_fit,
                execution_quality_fit=execution_quality_fit,
                high_liquidity=high_liquidity_session,
                throughput_recovery_active=candidate_throughput_recovery_active,
            )
        if (
            normalized_symbol == "BTCUSD"
            and weekend_mode
            and quality_tier == "C"
            and regime_state in {"MEAN_REVERSION", "RANGING", "LOW_LIQUIDITY_CHOP"}
        ):
            continue
        transition_momentum = float(candidate_meta.get("transition_momentum", 0.0) or 0.0)
        transition_momentum_size_multiplier = float(candidate_meta.get("transition_momentum_size_multiplier", 1.0) or 1.0)
        compression_burst_size_multiplier = float(candidate_meta.get("compression_burst_size_multiplier", 1.0) or 1.0)
        correlation_penalty = float(candidate_meta.get("correlation_penalty", 0.0) or 0.0)
        market_instability_score = clamp(float(candidate_meta.get("market_instability_score", 0.0) or 0.0), 0.0, 1.0)
        feature_drift_score = clamp(float(candidate_meta.get("feature_drift_score", 0.0) or 0.0), 0.0, 1.0)
        multi_tf_alignment_score = clamp(float(candidate_meta.get("multi_tf_alignment_score", 0.5) or 0.5), 0.0, 1.0)
        seasonality_edge_score = clamp(float(candidate_meta.get("seasonality_edge_score", 0.5) or 0.5), 0.0, 1.0)
        fractal_persistence_score = clamp(float(candidate_meta.get("fractal_persistence_score", 0.5) or 0.5), 0.0, 1.0)
        streak_adjust = _streak_adjustment_mode(
            closed_trades=closed_trades,
            symbol_key=normalized_symbol,
            strategy_key=strategy_key,
            orchestrator_config=orchestrator_config,
        )
        velocity_decay = _velocity_decay_profile(
            symbol_key=normalized_symbol,
            strategy_key=strategy_key,
            setup_family=strategy_family,
            closed_trades=closed_trades,
            row_timestamp=row_timestamp,
            weekend_mode=weekend_mode,
            trigger_trades_per_10_bars=float(candidate_tier_config.get("velocity_decay_trigger_trades_per_10_bars", 1.8) or 1.8),
            btc_weekend_trigger_trades_per_10_bars=float(btc_velocity_trigger),
            score_penalty=float(candidate_tier_config.get("velocity_decay_score_penalty", 0.15) or 0.15),
            base_multiplier=float(candidate_tier_config.get("velocity_decay_base_multiplier", 0.85) or 0.85),
        )
        if quality_tier == "B":
            b_tier_adjust_pct = float(equity_momentum.get("b_tier_adjust_pct", 0.0) or 0.0)
            if b_tier_adjust_pct != 0.0:
                adjustment = abs(b_tier_adjust_pct)
                direction = 1.0 if b_tier_adjust_pct > 0 else -1.0
                entry_timing_score = clamp(entry_timing_score + (direction * adjustment * 0.30), 0.0, 1.0)
                structure_cleanliness_score = clamp(structure_cleanliness_score + (direction * adjustment * 0.20), 0.0, 1.0)
                regime_fit = clamp(regime_fit + (direction * adjustment * 0.15), 0.0, 1.0)
        tier_size_multiplier = quality_tier_size_multiplier(
            quality_tier=quality_tier,
            strategy_score=router_rank_score,
            b_tier_min=b_tier_min,
            b_tier_max=b_tier_max,
        )
        if quality_tier == "A+" and str(equity_momentum.get("mode", "NEUTRAL")) == "HOT":
            tier_size_multiplier = clamp(
                float(tier_size_multiplier) * (1.0 + float(equity_momentum.get("a_plus_size_boost", 0.0) or 0.0)),
                0.50,
                1.10,
            )
        max_tier_size_cap = 1.35 if (normalized_symbol == "XAUUSD" and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID") else 1.20
        if normalized_symbol == "BTCUSD":
            max_tier_size_cap = max(max_tier_size_cap, 1.30 if weekend_mode else 1.24)
        elif super_aggro_symbol:
            max_tier_size_cap = max(max_tier_size_cap, 1.28 if normalized_symbol in {"NAS100", "USOIL"} else 1.24)
        confidence_multiplier = clamp(
            (
                float(velocity_decay.get("size_multiplier", 1.0) or 1.0)
                * (1.0 + min(0.15, max(0.0, float(transition_momentum)) * 0.15))
                * max(0.80, 1.0 - float(correlation_penalty))
                * max(0.82, 1.0 - (0.12 * market_instability_score) - (0.06 * feature_drift_score))
                * (1.0 + min(0.10, (0.06 * multi_tf_alignment_score) + (0.04 * seasonality_edge_score) + (0.04 * fractal_persistence_score)))
                * (
                    1.08
                    if str(equity_momentum.get("mode", "NEUTRAL")) == "HOT"
                    else (0.92 if str(equity_momentum.get("mode", "NEUTRAL")) == "COLD" else 1.0)
                )
            ),
            0.70,
            1.20,
        )
        tier_size_multiplier = clamp(
            float(tier_size_multiplier)
            * float(velocity_decay.get("size_multiplier", 1.0) or 1.0)
            * float(transition_momentum_size_multiplier)
            * float(compression_burst_size_multiplier),
            0.50,
            max_tier_size_cap,
        )
        btc_weekend_printer_lane = (
            normalized_symbol == "BTCUSD"
            and weekend_mode
            and strategy_key in {
                "BTCUSD_PRICE_ACTION_CONTINUATION",
                "BTCUSD_VOLATILE_RETEST",
                "BTCUSD_TREND_SCALP",
                "BTCUSD_RANGE_EXPANSION",
            }
        )
        if btc_weekend_printer_lane:
            tier_size_multiplier = clamp(float(tier_size_multiplier) * float(btc_weekend_size_boost), 0.50, max_tier_size_cap)
        strategy_recent_performance = clamp(
            float(candidate_meta.get("strategy_recent_performance_seed", 0.50) or 0.50),
            0.0,
            1.0,
        )
        strategy_state = str(candidate_meta.get("strategy_state") or "NORMAL").upper()
        strategy_bucket_metrics: dict[str, Any] = {}
        strategy_bucket_reason = ""
        if closed_trades:
            strategy_bucket_metrics = _pair_strategy_session_performance_state(
                symbol=normalized_symbol,
                strategy_key=strategy_key,
                session_name=session_name,
                regime_state=regime_state,
                session_native_pair=bool(priority.session_native_pair),
                closed_trades=closed_trades,
                current_day_key=current_day_key,
            )
            bucket_state = str(strategy_bucket_metrics.get("strategy_bucket_state") or strategy_state).upper()
            bucket_priority_multiplier = clamp(
                float(strategy_bucket_metrics.get("strategy_bucket_priority_multiplier", 1.0) or 1.0),
                0.72,
                1.12,
            )
            bucket_size_multiplier = clamp(
                float(strategy_bucket_metrics.get("strategy_bucket_size_multiplier", 1.0) or 1.0),
                0.80,
                1.08,
            )
            bucket_recent_performance = strategy_recent_performance_score(
                expectancy_r=float(strategy_bucket_metrics.get("strategy_bucket_recent_expectancy_r", 0.0) or 0.0),
                profit_factor=float(strategy_bucket_metrics.get("strategy_bucket_recent_profit_factor", 0.0) or 0.0),
                win_rate=float(strategy_bucket_metrics.get("strategy_bucket_recent_win_rate", 0.50) or 0.50),
                management_quality=float(strategy_bucket_metrics.get("strategy_bucket_management_quality", 0.50) or 0.50),
            )
            strategy_recent_performance = clamp(
                max(strategy_recent_performance * 0.45, bucket_recent_performance),
                0.0,
                1.0,
            )
            strategy_state = bucket_state
            strategy_bucket_reason = str(strategy_bucket_metrics.get("strategy_bucket_reason") or "")
            if bucket_state == "QUARANTINED":
                entry_timing_score = clamp(
                    entry_timing_score
                    - 0.08
                    - min(0.12, float(strategy_bucket_metrics.get("strategy_bucket_late_entry_rate", 0.0) or 0.0) * 0.20),
                    0.0,
                    1.0,
                )
                structure_cleanliness_score = clamp(
                    structure_cleanliness_score
                    - 0.08
                    - min(0.12, float(strategy_bucket_metrics.get("strategy_bucket_poor_structure_rate", 0.0) or 0.0) * 0.18),
                    0.0,
                    1.0,
                )
            elif bucket_state == "REDUCED":
                entry_timing_score = clamp(
                    entry_timing_score
                    - min(0.05, float(strategy_bucket_metrics.get("strategy_bucket_late_entry_rate", 0.0) or 0.0) * 0.08),
                    0.0,
                    1.0,
                )
                structure_cleanliness_score = clamp(
                    structure_cleanliness_score
                    - min(0.05, float(strategy_bucket_metrics.get("strategy_bucket_poor_structure_rate", 0.0) or 0.0) * 0.06),
                    0.0,
                    1.0,
                )
            execution_quality_fit = clamp(
                execution_quality_fit
                - min(
                    0.10,
                    (
                        float(strategy_bucket_metrics.get("strategy_bucket_fast_failure_rate", 0.0) or 0.0) * 0.08
                        + float(strategy_bucket_metrics.get("strategy_bucket_immediate_invalidation_rate", 0.0) or 0.0) * 0.08
                    ),
                ),
                0.0,
                1.0,
            )
            candidate_meta["strategy_state"] = str(bucket_state)
            candidate_meta["strategy_recent_performance"] = float(strategy_recent_performance)
            candidate_meta["strategy_bucket_metrics"] = dict(strategy_bucket_metrics)
            candidate_meta["size_multiplier"] = float(
                clamp(
                    float(candidate_meta.get("size_multiplier", 1.0) or 1.0) * bucket_size_multiplier * tier_size_multiplier,
                    0.50,
                    1.25,
                )
            )
        else:
            bucket_priority_multiplier = 1.0
        pre_selection_score = strategy_selection_score(
            ev_estimate=router_rank_score,
            regime_fit=regime_fit,
            session_fit=session_fit,
            volatility_fit=volatility_fit,
            pair_behavior_fit_score=pair_behavior_fit,
            strategy_recent_performance=strategy_recent_performance,
            execution_quality_fit=execution_quality_fit,
            entry_timing_score_value=entry_timing_score,
            structure_cleanliness_score_value=structure_cleanliness_score,
            drawdown_penalty=0.0,
            false_break_penalty=max(0.0, 0.55 - structure_cleanliness_score),
            chop_penalty=0.20 if regime_state == "LOW_LIQUIDITY_CHOP" and any(
                token in strategy_key for token in ("BREAKOUT", "IMPULSE", "EXPANSION")
            ) else 0.0,
        )
        pre_selection_score = clamp(
            float(pre_selection_score)
            + {
                "A+": 0.06,
                "A": 0.03,
                "B": 0.04 if high_liquidity_session or throughput_recovery_active else 0.0,
            }.get(quality_tier, 0.0),
            0.0,
            1.0,
        )
        pre_selection_score = clamp(
            float(pre_selection_score)
            - float(velocity_decay.get("score_penalty", 0.0) or 0.0)
            - float(correlation_penalty),
            0.0,
            1.0,
        )
        pre_selection_score = clamp(
            float(pre_selection_score)
            + (0.04 * multi_tf_alignment_score)
            + (0.03 * seasonality_edge_score)
            + (0.03 * fractal_persistence_score)
            - (0.08 * market_instability_score)
            - (0.04 * feature_drift_score),
            0.0,
            1.0,
        )
        if btc_weekend_printer_lane:
            pre_selection_score = clamp(pre_selection_score + 0.05, 0.0, 1.0)
        mc_win_rate = clamp(float(candidate_meta.get("mc_win_rate", 0.0) or 0.0), 0.0, 1.0)
        compression_signal_score = clamp(float(candidate_meta.get("compression_expansion_score", 0.0) or 0.0), 0.0, 1.0)
        super_aggro_weak_context = bool(
            (normalized_symbol == "GBPUSD" and strategy_key == "GBPUSD_LONDON_EXPANSION_BREAKOUT" and session_name == "LONDON" and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP"})
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_OPENING_DRIVE_BREAKOUT" and regime_state != "BREAKOUT_EXPANSION")
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_MOMENTUM_IMPULSE" and regime_state in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"})
            or (normalized_symbol == "USOIL" and strategy_key == "USOIL_LONDON_TREND_EXPANSION")
        )
        super_aggro_signal_quality = clamp(
            (0.28 * strategy_recent_performance)
            + (0.22 * multi_tf_alignment_score)
            + (0.20 * fractal_persistence_score)
            + (0.18 * execution_quality_fit)
            + (0.12 * max(compression_signal_score, mc_win_rate)),
            0.0,
            1.0,
        )
        super_aggro_attack_ready = bool(
            super_aggro_strategy
            and not super_aggro_weak_context
            and (
                (
                    super_aggro_home_session
                    and pre_selection_score >= 0.58
                    and regime_fit >= 0.54
                    and entry_timing_score >= 0.52
                    and structure_cleanliness_score >= 0.52
                    and super_aggro_signal_quality >= 0.56
                )
                or (
                    not super_aggro_home_session
                    and high_liquidity_session
                    and pre_selection_score >= 0.62
                    and regime_fit >= 0.58
                    and entry_timing_score >= 0.56
                    and structure_cleanliness_score >= 0.56
                    and super_aggro_signal_quality >= 0.60
                )
            )
        )
        if super_aggro_attack_ready:
            attack_bonus = 0.04 if super_aggro_home_session else 0.03
            attack_bonus += min(0.03, float(equity_momentum.get("super_aggro_score_boost", 0.0) or 0.0))
            attack_bonus += min(0.02, float(equity_momentum.get("trajectory_catchup_pressure", 0.0) or 0.0))
            if mc_win_rate >= 0.85:
                attack_bonus += 0.02
            pre_selection_score = clamp(pre_selection_score + attack_bonus, 0.0, 1.0)
            tier_size_multiplier = clamp(
                float(tier_size_multiplier)
                * (1.0 + min(0.12, float(equity_momentum.get("super_aggro_size_boost", 0.0) or 0.0)))
                * (1.06 if super_aggro_home_session else 1.03),
                0.50,
                max_tier_size_cap,
            )
            confidence_multiplier = clamp(
                float(confidence_multiplier)
                * (1.04 if super_aggro_home_session else 1.02),
                0.70,
                1.24,
            )
            candidate_meta["super_aggressive_pair_mode"] = True
            candidate_meta["trajectory_catchup_pressure"] = float(
                equity_momentum.get("trajectory_catchup_pressure", 0.0) or 0.0
            )
        family_rotation_penalty = _family_rotation_penalty(
            strategy_key=strategy_key,
            setup_family=strategy_family,
            closed_trades=closed_trades,
            candidate_tier_config=candidate_tier_config,
        )
        if recycle_session:
            pre_selection_score = clamp(pre_selection_score + 0.03, 0.0, 1.0)
        pre_selection_score = clamp(pre_selection_score * bucket_priority_multiplier, 0.0, 1.25)
        pre_session_adjusted_score = session_adjusted_score(
            base_score=float(pre_selection_score),
            session_priority_multiplier=float(priority.session_priority_multiplier),
            lane_strength_multiplier=1.0,
            quality_floor_edge=float(priority.quality_floor_edge),
        )
        btc_weekday_price_action_ready = bool(
            normalized_symbol == "BTCUSD"
            and not weekend_mode
            and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION"
            and (
                (
                    session_name == "OVERLAP"
                    and regime_state == "LOW_LIQUIDITY_CHOP"
                    and regime_fit >= 0.58
                    and entry_timing_score >= 0.56
                    and structure_cleanliness_score >= 0.56
                    and execution_quality_fit >= 0.56
                    and pair_behavior_fit >= 0.50
                    and strategy_recent_performance >= 0.56
                    and pre_session_adjusted_score >= 0.64
                )
                or (
                    session_name == "NEW_YORK"
                    and regime_state == "TRENDING"
                    and quality_tier in {"A", "A+"}
                    and regime_fit >= 0.72
                    and entry_timing_score >= 0.70
                    and structure_cleanliness_score >= 0.70
                    and execution_quality_fit >= 0.68
                    and pair_behavior_fit >= 0.60
                    and strategy_recent_performance >= 0.62
                    and pre_session_adjusted_score >= 0.74
                    and multi_tf_alignment_score >= 0.58
                    and market_instability_score <= 0.38
                    and feature_drift_score <= 0.30
                )
            )
        )
        btc_weekday_volatile_retest_ready = bool(
            normalized_symbol == "BTCUSD"
            and not weekend_mode
            and strategy_key == "BTCUSD_VOLATILE_RETEST"
            and (
                (
                    session_name == "LONDON"
                    and regime_state == "MEAN_REVERSION"
                    and regime_fit >= 0.58
                    and entry_timing_score >= 0.56
                    and structure_cleanliness_score >= 0.56
                    and execution_quality_fit >= 0.54
                    and pair_behavior_fit >= 0.50
                    and strategy_recent_performance >= 0.56
                    and pre_session_adjusted_score >= 0.62
                )
                or (
                    session_name == "NEW_YORK"
                    and regime_state in {"MEAN_REVERSION", "TRENDING"}
                    and regime_fit >= 0.62
                    and entry_timing_score >= 0.58
                    and structure_cleanliness_score >= 0.58
                    and execution_quality_fit >= 0.56
                    and pair_behavior_fit >= 0.52
                    and strategy_recent_performance >= 0.58
                    and pre_session_adjusted_score >= 0.66
                )
            )
        )
        btc_weekend_printer_ready = bool(
            normalized_symbol == "BTCUSD"
            and weekend_mode
            and strategy_key in {
                "BTCUSD_PRICE_ACTION_CONTINUATION",
                "BTCUSD_VOLATILE_RETEST",
                "BTCUSD_TREND_SCALP",
                "BTCUSD_RANGE_EXPANSION",
            }
            and (
                (
                    session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
                    and regime_fit >= 0.48
                    and entry_timing_score >= 0.48
                    and structure_cleanliness_score >= 0.48
                    and execution_quality_fit >= 0.50
                    and pair_behavior_fit >= 0.44
                    and strategy_recent_performance >= 0.48
                    and pre_session_adjusted_score >= 0.56
                )
                or (
                    session_name in {"SYDNEY", "TOKYO"}
                    and strategy_key in {
                        "BTCUSD_PRICE_ACTION_CONTINUATION",
                        "BTCUSD_VOLATILE_RETEST",
                        "BTCUSD_RANGE_EXPANSION",
                    }
                    and regime_state in {"TRENDING", "MEAN_REVERSION", "BREAKOUT_EXPANSION", "RANGING"}
                    and regime_fit >= 0.50
                    and entry_timing_score >= 0.48
                    and structure_cleanliness_score >= 0.48
                    and execution_quality_fit >= 0.48
                    and pair_behavior_fit >= 0.44
                    and strategy_recent_performance >= 0.48
                    and pre_session_adjusted_score >= 0.55
                )
                or (
                    session_name in {"SYDNEY", "TOKYO"}
                    and strategy_key == "BTCUSD_TREND_SCALP"
                    and regime_state in {"TRENDING", "BREAKOUT_EXPANSION"}
                    and regime_fit >= 0.54
                    and entry_timing_score >= 0.50
                    and structure_cleanliness_score >= 0.50
                    and execution_quality_fit >= 0.48
                    and pair_behavior_fit >= 0.45
                    and strategy_recent_performance >= 0.50
                    and pre_session_adjusted_score >= 0.56
                )
            )
        )
        absolute_exact_bucket = (
            (normalized_symbol == "AUDNZD" and strategy_key == "AUDNZD_STRUCTURE_BREAK_RETEST")
            or (normalized_symbol == "AUDJPY" and strategy_key == "AUDJPY_ATR_COMPRESSION_BREAKOUT" and session_name == "OVERLAP" and regime_state == "LOW_LIQUIDITY_CHOP")
            or (normalized_symbol == "AUDJPY" and strategy_key == "AUDJPY_ATR_COMPRESSION_BREAKOUT" and session_name == "NEW_YORK" and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "XAUUSD" and strategy_key == "XAUUSD_LONDON_LIQUIDITY_SWEEP" and session_name == "OVERLAP" and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "NZDJPY" and strategy_key == "NZDJPY_SESSION_RANGE_EXPANSION" and session_name in {"OVERLAP", "NEW_YORK"} and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"})
            or (normalized_symbol == "NZDJPY" and strategy_key == "NZDJPY_LIQUIDITY_TRAP_REVERSAL" and session_name == "NEW_YORK" and regime_state == "LOW_LIQUIDITY_CHOP")
            or (normalized_symbol == "EURJPY" and strategy_key == "EURJPY_MOMENTUM_IMPULSE" and session_name == "OVERLAP" and regime_state == "LOW_LIQUIDITY_CHOP")
            or (normalized_symbol == "EURJPY" and strategy_key == "EURJPY_LIQUIDITY_SWEEP_REVERSAL" and session_name == "OVERLAP" and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "EURJPY" and strategy_key == "EURJPY_SESSION_PULLBACK_CONTINUATION" and session_name == "LONDON" and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "EURUSD" and strategy_key == "EURUSD_LONDON_BREAKOUT" and session_name == "NEW_YORK" and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "EURUSD" and strategy_key == "EURUSD_VWAP_PULLBACK" and session_name == "NEW_YORK" and regime_state == "RANGING")
            or (
                normalized_symbol == "BTCUSD"
                and not weekend_mode
                and strategy_key in {"BTCUSD_PRICE_ACTION_CONTINUATION", "BTCUSD_TREND_SCALP", "BTCUSD_RANGE_EXPANSION"}
                and regime_state == "MEAN_REVERSION"
            )
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_TREND_SCALP" and session_name == "SYDNEY" and not weekend_mode)
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_TREND_SCALP" and session_name == "TOKYO" and regime_state in {"TRENDING", "RANGING"} and not weekend_mode)
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION" and session_name in {"SYDNEY", "TOKYO", "LONDON"} and not weekend_mode)
            or (
                normalized_symbol == "BTCUSD"
                and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION"
                and session_name == "OVERLAP"
                and regime_state == "TRENDING"
                and not weekend_mode
            )
            or (
                normalized_symbol == "BTCUSD"
                and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION"
                and session_name == "NEW_YORK"
                and regime_state == "TRENDING"
                and not weekend_mode
                and not btc_weekday_price_action_ready
            )
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_VOLATILE_RETEST" and session_name == "OVERLAP" and not weekend_mode)
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_VOLATILE_RETEST" and session_name == "SYDNEY" and regime_state == "MEAN_REVERSION" and not weekend_mode)
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_VOLATILE_RETEST" and session_name == "SYDNEY" and regime_state == "LOW_LIQUIDITY_CHOP" and not weekend_mode)
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_RANGE_EXPANSION" and session_name == "TOKYO" and regime_state == "MEAN_REVERSION" and not weekend_mode)
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_TREND_SCALP" and session_name == "TOKYO" and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"} and not weekend_mode)
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_TREND_SCALP" and session_name == "LONDON" and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"} and not weekend_mode)
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_VOLATILE_RETEST" and session_name == "TOKYO" and regime_state == "MEAN_REVERSION" and not weekend_mode)
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_VOLATILE_RETEST" and session_name == "TOKYO" and regime_state == "TRENDING" and not weekend_mode)
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_VOLATILE_RETEST" and session_name == "OVERLAP" and regime_state == "LOW_LIQUIDITY_CHOP" and not weekend_mode)
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION" and session_name == "NEW_YORK" and regime_state == "LOW_LIQUIDITY_CHOP" and not weekend_mode)
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_MOMENTUM_IMPULSE" and session_name == "OVERLAP" and regime_state == "TRENDING")
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_OPENING_DRIVE_BREAKOUT" and session_name in {"LONDON", "NEW_YORK"} and regime_state == "TRENDING")
            or (normalized_symbol == "GBPUSD" and strategy_key == "GBPUSD_LONDON_EXPANSION_BREAKOUT" and session_name == "LONDON" and regime_state == "RANGING")
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_OPENING_DRIVE_BREAKOUT" and session_name in {"LONDON", "NEW_YORK"} and regime_state == "RANGING")
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_LIQUIDITY_SWEEP_REVERSAL" and session_name in {"TOKYO", "NEW_YORK"} and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"})
            or (normalized_symbol == "USOIL" and strategy_key == "USOIL_INVENTORY_MOMENTUM" and session_name == "LONDON" and regime_state == "RANGING")
            or (normalized_symbol == "USOIL" and strategy_key == "USOIL_INVENTORY_MOMENTUM" and session_name == "OVERLAP" and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "USOIL" and strategy_key == "USOIL_LONDON_TREND_EXPANSION" and session_name == "LONDON" and regime_state == "RANGING")
            or (normalized_symbol == "USOIL" and strategy_key == "USOIL_LONDON_TREND_EXPANSION" and session_name == "LONDON" and regime_state == "TRENDING")
            or (normalized_symbol == "USOIL" and strategy_key == "USOIL_LONDON_TREND_EXPANSION" and session_name == "SYDNEY" and regime_state == "LOW_LIQUIDITY_CHOP")
        )
        if absolute_exact_bucket:
            continue
        xau_structured_directional_setup = bool(
            normalized_symbol == "XAUUSD"
            and (
                setup_name.startswith("XAUUSD_M1_MICRO_SCALPER")
                or setup_name.startswith("XAUUSD_M15_STRUCTURED")
                or setup_name == "XAUUSD_M15_FIX_FLOW"
                or setup_name == "XAU_BREAKOUT_RETEST"
                or setup_name == "XAU_FAKEOUT_FADE"
            )
        )
        xau_directional_breakout_setup = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_NY_MOMENTUM_BREAKOUT"
            and xau_structured_directional_setup
            and (
                setup_name.startswith("XAUUSD_M1_MICRO_SCALPER")
                or setup_name.startswith("XAUUSD_M15_STRUCTURED_BREAKOUT")
                or setup_name.startswith("XAUUSD_M15_STRUCTURED_PULLBACK")
                or setup_name.startswith("XAUUSD_M15_STRUCTURED_SWEEP_RETEST")
                or setup_name == "XAUUSD_M15_FIX_FLOW"
                or setup_name == "XAU_BREAKOUT_RETEST"
            )
        )
        xau_directional_structured_quality_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key != "XAUUSD_ATR_EXPANSION_SCALPER"
            and xau_structured_directional_setup
            and (
                quality_tier == "A"
                or (
                    quality_tier == "B"
                    and regime_fit >= 0.70
                    and entry_timing_score >= 0.68
                    and structure_cleanliness_score >= 0.70
                    and execution_quality_fit >= 0.68
                    and pair_behavior_fit >= 0.60
                    and strategy_recent_performance >= 0.58
                    and pre_session_adjusted_score >= 0.72
                )
            )
        )
        xau_directional_prime_attack_ready = bool(
            xau_directional_breakout_setup
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and regime_state == "TRENDING"
            and (
                quality_tier == "A"
                or (
                    quality_tier == "B"
                    and regime_fit >= 0.68
                    and entry_timing_score >= 0.66
                    and structure_cleanliness_score >= 0.68
                    and execution_quality_fit >= 0.66
                    and pair_behavior_fit >= 0.58
                    and strategy_recent_performance >= 0.56
                    and pre_session_adjusted_score >= 0.70
                    and multi_tf_alignment_score >= 0.56
                    and market_instability_score <= 0.48
                    and feature_drift_score <= 0.40
                )
            )
        )
        xau_directional_off_session_quality_ready = bool(
            normalized_symbol == "XAUUSD"
            and xau_directional_structured_quality_ready
            and (
                (
                    quality_tier == "A"
                    and regime_fit >= 0.74
                    and entry_timing_score >= 0.72
                    and structure_cleanliness_score >= 0.74
                    and execution_quality_fit >= 0.70
                    and pair_behavior_fit >= 0.62
                    and strategy_recent_performance >= 0.60
                    and pre_session_adjusted_score >= 0.78
                )
                or (
                    quality_tier == "B"
                    and regime_fit >= 0.74
                    and entry_timing_score >= 0.72
                    and structure_cleanliness_score >= 0.74
                    and execution_quality_fit >= 0.72
                    and pair_behavior_fit >= 0.64
                    and strategy_recent_performance >= 0.62
                    and pre_session_adjusted_score >= 0.80
                )
            )
        )
        xau_directional_off_session_attack_ready = bool(
            xau_directional_breakout_setup
            and session_name not in {"LONDON", "OVERLAP", "NEW_YORK"}
            and (
                (
                    quality_tier == "A"
                    and regime_fit >= 0.72
                    and entry_timing_score >= 0.70
                    and structure_cleanliness_score >= 0.72
                    and execution_quality_fit >= 0.68
                    and pair_behavior_fit >= 0.58
                    and strategy_recent_performance >= 0.58
                    and pre_session_adjusted_score >= 0.74
                    and multi_tf_alignment_score >= 0.58
                    and market_instability_score <= 0.42
                    and feature_drift_score <= 0.32
                )
                or (
                    quality_tier == "B"
                    and regime_fit >= 0.74
                    and entry_timing_score >= 0.72
                    and structure_cleanliness_score >= 0.74
                    and execution_quality_fit >= 0.70
                    and pair_behavior_fit >= 0.60
                    and strategy_recent_performance >= 0.60
                    and pre_session_adjusted_score >= 0.78
                    and multi_tf_alignment_score >= 0.60
                    and market_instability_score <= 0.38
                    and feature_drift_score <= 0.28
                )
            )
        )
        xau_directional_atr_off_session_quality_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ATR_EXPANSION_SCALPER"
            and session_name not in {"LONDON", "OVERLAP", "NEW_YORK"}
            and quality_tier == "A"
            and regime_fit >= 0.76
            and entry_timing_score >= 0.74
            and structure_cleanliness_score >= 0.76
            and execution_quality_fit >= 0.72
            and pair_behavior_fit >= 0.64
            and strategy_recent_performance >= 0.62
            and pre_session_adjusted_score >= 0.78
        )
        xau_directional_atr_quality_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ATR_EXPANSION_SCALPER"
            and quality_tier == "A"
            and regime_fit >= 0.76
            and entry_timing_score >= 0.74
            and structure_cleanliness_score >= 0.76
            and execution_quality_fit >= 0.72
            and pair_behavior_fit >= 0.64
            and strategy_recent_performance >= 0.62
            and pre_session_adjusted_score >= 0.80
        )
        xau_directional_breakout_ranking_bonus = 0.0
        if xau_directional_prime_attack_ready:
            xau_directional_breakout_ranking_bonus = 0.04 if session_name == "OVERLAP" else 0.03
            if london_session := bool(session_name == "LONDON"):
                xau_directional_breakout_ranking_bonus = 0.05
            if setup_name.startswith("XAUUSD_M15_STRUCTURED_BREAKOUT") or setup_name in {"XAU_BREAKOUT_RETEST", "XAUUSD_M15_FIX_FLOW"}:
                xau_directional_breakout_ranking_bonus += 0.01
            if not london_session and setup_name.startswith("XAUUSD_M1_MICRO_SCALPER"):
                xau_directional_breakout_ranking_bonus += 0.005
        elif xau_directional_off_session_attack_ready:
            xau_directional_breakout_ranking_bonus = 0.02
            if setup_name in {"XAU_BREAKOUT_RETEST", "XAUUSD_M15_FIX_FLOW"}:
                xau_directional_breakout_ranking_bonus += 0.01
        if xau_directional_breakout_ranking_bonus > 0.0:
            pre_session_adjusted_score = clamp(
                float(pre_session_adjusted_score) + float(xau_directional_breakout_ranking_bonus),
                0.0,
                1.0,
            )
        btc_directional_ranking_bonus = 0.0
        if btc_weekday_price_action_ready:
            btc_directional_ranking_bonus = 0.05 if session_name == "OVERLAP" else 0.03
        elif btc_weekday_volatile_retest_ready:
            btc_directional_ranking_bonus = 0.04 if session_name == "NEW_YORK" else 0.03
        elif btc_weekend_printer_ready:
            if session_name in {"LONDON", "OVERLAP", "NEW_YORK"}:
                btc_directional_ranking_bonus = (
                    0.10 if strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION"
                    else 0.09 if strategy_key == "BTCUSD_TREND_SCALP"
                    else 0.08
                )
            else:
                btc_directional_ranking_bonus = (
                    0.07 if strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION"
                    else 0.06 if strategy_key == "BTCUSD_VOLATILE_RETEST"
                    else 0.05
                )
        if btc_directional_ranking_bonus > 0.0:
            pre_session_adjusted_score = clamp(
                float(pre_session_adjusted_score) + float(btc_directional_ranking_bonus),
                0.0,
                1.0,
            )
        xau_grid_entry_profile = str(candidate_meta.get("grid_entry_profile") or "")
        xau_grid_density_override_profile = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and xau_engine == "GRID_NATIVE_SCALPER"
            and xau_grid_prime_session_candidate
            and xau_grid_density_first_active
            and xau_grid_quota_debt_10m >= (3 if session_name == "OVERLAP" else 2)
            and str(candidate_meta.get("grid_source_role") or "").upper() in {"NATIVE_PRIMARY", "NATIVE_ATTACK", "NATIVE_ADD_BURST"}
        )
        xau_grid_reclaim_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and xau_grid_entry_profile.startswith(
                (
                    "grid_liquidity_reclaim",
                    "grid_trend_reclaim",
                    "grid_m15_pullback_reclaim",
                    "grid_directional_flow",
                    "grid_breakout_reclaim",
                )
            )
            and float(candidate_meta.get("multi_tf_alignment_score", 0.5) or 0.5) >= 0.48
            and float(candidate_meta.get("fractal_persistence_score", 0.5) or 0.5) >= 0.46
            and float(candidate_meta.get("seasonality_edge_score", 0.5) or 0.5) >= 0.34
            and float(candidate_meta.get("market_instability_score", 0.0) or 0.0) <= 0.58
            and float(candidate_meta.get("feature_drift_score", 0.0) or 0.0) <= 0.48
        )
        xau_grid_expansion_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and xau_grid_entry_profile.startswith("grid_")
            and not xau_grid_entry_profile.startswith("grid_compression_expansion_follow")
            and (
                xau_grid_reclaim_ready
                or (
                    float(candidate_meta.get("compression_expansion_score", 0.0) or 0.0) >= 0.35
                    and float(candidate_meta.get("multi_tf_alignment_score", 0.5) or 0.5) >= 0.50
                    and float(candidate_meta.get("fractal_persistence_score", 0.5) or 0.5) >= 0.48
                    and float(candidate_meta.get("market_instability_score", 0.0) or 0.0) <= 0.58
                )
            )
        )
        xau_grid_ranging_reclaim_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and regime_state == "RANGING"
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and xau_grid_reclaim_ready
            and float(candidate_meta.get("multi_tf_alignment_score", 0.5) or 0.5) >= 0.55
            and float(candidate_meta.get("fractal_persistence_score", 0.5) or 0.5) >= 0.52
            and float(candidate_meta.get("market_instability_score", 0.0) or 0.0) <= 0.42
            and float(candidate_meta.get("feature_drift_score", 0.0) or 0.0) <= 0.28
        )
        xau_grid_entry_profile = str(candidate_meta.get("grid_entry_profile") or "")
        xau_grid_live_prime_profiles = {
            "grid_liquidity_reclaim_long",
            "grid_liquidity_reclaim_short",
            "grid_trend_reclaim_long",
            "grid_trend_reclaim_short",
            "grid_m15_pullback_reclaim_long",
            "grid_m15_pullback_reclaim_short",
            "grid_breakout_reclaim_long",
            "grid_breakout_reclaim_short",
            "grid_directional_flow_long",
            "grid_directional_flow_short",
            "grid_expansion_ready_scaler_long",
            "grid_expansion_ready_scaler_short",
            "grid_prime_session_momentum_long",
            "grid_prime_session_momentum_short",
        }
        xau_grid_fragile_prime_mean_reversion = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and regime_state == "MEAN_REVERSION"
            and session_name in {"OVERLAP", "NEW_YORK"}
            and not str(candidate_meta.get("xau_engine") or "").upper().startswith("GRID_DIRECTIONAL_MIRROR")
            and (
                xau_grid_entry_profile in {"grid_prime_session_momentum_long", "grid_prime_session_momentum_short"}
                or (
                    xau_grid_entry_profile in {"grid_directional_flow_long", "grid_directional_flow_short"}
                    and str(candidate_meta.get("compression_proxy_state") or "").upper() == "NEUTRAL"
                    and float(candidate_meta.get("compression_expansion_score", 0.0) or 0.0) < 0.70
                )
            )
            and str(candidate_meta.get("compression_proxy_state") or "").upper() == "NEUTRAL"
        )
        xau_grid_mirror_directional_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and str(candidate_meta.get("xau_engine") or "").upper().startswith("GRID_DIRECTIONAL_MIRROR")
            and xau_grid_entry_profile
            in {
                "grid_liquidity_reclaim_long",
                "grid_liquidity_reclaim_short",
                "grid_trend_reclaim_long",
                "grid_trend_reclaim_short",
                "grid_m15_pullback_reclaim_long",
                "grid_m15_pullback_reclaim_short",
                "grid_breakout_reclaim_long",
                "grid_breakout_reclaim_short",
                "grid_directional_flow_long",
                "grid_directional_flow_short",
            }
            and float(candidate_meta.get("multi_tf_alignment_score", 0.5) or 0.5) >= 0.44
            and float(candidate_meta.get("fractal_persistence_score", 0.5) or 0.5) >= 0.42
            and float(candidate_meta.get("seasonality_edge_score", 0.5) or 0.5) >= 0.28
            and float(candidate_meta.get("market_instability_score", 0.0) or 0.0) <= 0.64
            and float(candidate_meta.get("feature_drift_score", 0.0) or 0.0) <= 0.50
            and regime_fit >= 0.48
            and entry_timing_score >= 0.48
            and structure_cleanliness_score >= 0.50
            and execution_quality_fit >= 0.48
        )
        xau_grid_native_directional_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and str(candidate_meta.get("xau_engine") or "").upper() == "GRID_NATIVE_SCALPER"
            and not xau_grid_fragile_prime_mean_reversion
            and str(candidate_meta.get("grid_source_role") or "").upper()
            in {"NATIVE_PRIMARY", "NATIVE_ATTACK", "NATIVE_ADD_BURST"}
            and xau_grid_entry_profile
            in {
                "grid_liquidity_reclaim_long",
                "grid_liquidity_reclaim_short",
                "grid_trend_reclaim_long",
                "grid_trend_reclaim_short",
                "grid_m15_pullback_reclaim_long",
                "grid_m15_pullback_reclaim_short",
                "grid_breakout_reclaim_long",
                "grid_breakout_reclaim_short",
                "grid_directional_flow_long",
                "grid_directional_flow_short",
            }
            and float(candidate_meta.get("multi_tf_alignment_score", 0.5) or 0.5) >= 0.50
            and float(candidate_meta.get("fractal_persistence_score", 0.5) or 0.5) >= 0.48
            and float(candidate_meta.get("seasonality_edge_score", 0.5) or 0.5) >= 0.32
            and float(candidate_meta.get("market_instability_score", 0.0) or 0.0) <= 0.50
            and float(candidate_meta.get("feature_drift_score", 0.0) or 0.0) <= 0.34
            and regime_fit >= 0.54
            and entry_timing_score >= 0.52
            and structure_cleanliness_score >= 0.52
            and execution_quality_fit >= 0.52
        )
        xau_grid_native_prime_scaler_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and str(candidate_meta.get("xau_engine") or "").upper() == "GRID_NATIVE_SCALPER"
            and not xau_grid_fragile_prime_mean_reversion
            and str(candidate_meta.get("grid_source_role") or "").upper()
            in {"NATIVE_PRIMARY", "NATIVE_ATTACK", "NATIVE_ADD_BURST"}
            and xau_grid_entry_profile
            in {
                "grid_expansion_ready_scaler_long",
                "grid_expansion_ready_scaler_short",
                "grid_prime_session_momentum_long",
                "grid_prime_session_momentum_short",
            }
            and float(candidate_meta.get("multi_tf_alignment_score", 0.5) or 0.5) >= 0.44
            and float(candidate_meta.get("fractal_persistence_score", 0.5) or 0.5) >= 0.40
            and float(candidate_meta.get("seasonality_edge_score", 0.5) or 0.5) >= 0.26
            and float(candidate_meta.get("market_instability_score", 0.0) or 0.0) <= 0.58
            and float(candidate_meta.get("feature_drift_score", 0.0) or 0.0) <= 0.44
            and regime_fit >= 0.46
            and entry_timing_score >= 0.46
            and structure_cleanliness_score >= 0.48
            and execution_quality_fit >= 0.48
            and (
                session_name != "OVERLAP"
                or (
                    float(candidate_meta.get("multi_tf_alignment_score", 0.5) or 0.5) >= 0.52
                    and float(candidate_meta.get("fractal_persistence_score", 0.5) or 0.5) >= 0.48
                    and float(candidate_meta.get("seasonality_edge_score", 0.5) or 0.5) >= 0.30
                    and float(candidate_meta.get("market_instability_score", 0.0) or 0.0) <= 0.50
                    and float(candidate_meta.get("feature_drift_score", 0.0) or 0.0) <= 0.36
                    and regime_fit >= 0.52
                    and entry_timing_score >= 0.52
                    and structure_cleanliness_score >= 0.54
                    and execution_quality_fit >= 0.54
                )
            )
        )
        xau_grid_native_near_a_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and str(candidate_meta.get("xau_engine") or "").upper() == "GRID_NATIVE_SCALPER"
            and quality_tier == "B"
            and (xau_grid_native_directional_ready or xau_grid_native_prime_scaler_ready)
            and float(candidate_meta.get("mc_win_rate", 0.0) or 0.0) >= 0.84
            and float(candidate_meta.get("htf_alignment_score", 0.0) or 0.0) >= 0.60
            and structure_cleanliness_score >= 0.60
            and execution_quality_fit >= 0.58
            and router_rank_score >= 0.76
            and (
                session_name != "OVERLAP"
                or (
                    float(candidate_meta.get("mc_win_rate", 0.0) or 0.0) >= 0.86
                    and float(candidate_meta.get("htf_alignment_score", 0.0) or 0.0) >= 0.62
                    and structure_cleanliness_score >= 0.62
                    and execution_quality_fit >= 0.60
                    and router_rank_score >= 0.78
                )
            )
        )
        xau_grid_density_override_ready = bool(
            xau_grid_density_override_profile
            and not xau_grid_fragile_prime_mean_reversion
            and xau_grid_entry_profile in xau_grid_live_prime_profiles
            and float(candidate_meta.get("mc_win_rate", 0.0) or 0.0) >= (0.78 if session_name == "OVERLAP" else 0.74 if session_name == "NEW_YORK" else 0.72)
            and float(candidate_meta.get("multi_tf_alignment_score", 0.5) or 0.5) >= (0.50 if session_name == "OVERLAP" else 0.44)
            and float(candidate_meta.get("fractal_persistence_score", 0.5) or 0.5) >= (0.46 if session_name == "OVERLAP" else 0.40)
            and float(candidate_meta.get("seasonality_edge_score", 0.5) or 0.5) >= (0.28 if session_name == "OVERLAP" else 0.22)
            and float(candidate_meta.get("market_instability_score", 0.0) or 0.0) <= (0.54 if session_name == "OVERLAP" else 0.66 if session_name == "LONDON" else 0.60)
            and float(candidate_meta.get("feature_drift_score", 0.0) or 0.0) <= (0.40 if session_name == "OVERLAP" else 0.56 if session_name == "LONDON" else 0.50)
            and regime_fit >= (0.52 if session_name == "OVERLAP" else 0.46 if session_name == "NEW_YORK" else 0.44)
            and entry_timing_score >= (0.54 if session_name == "OVERLAP" else 0.46 if session_name == "NEW_YORK" else 0.44)
            and structure_cleanliness_score >= (0.54 if session_name == "OVERLAP" else 0.46 if session_name == "NEW_YORK" else 0.44)
            and execution_quality_fit >= (0.54 if session_name == "OVERLAP" else 0.46 if session_name == "NEW_YORK" else 0.44)
        )
        xau_grid_density_ranking_bonus = 0.0
        if xau_grid_density_override_ready:
            xau_grid_density_ranking_bonus = 0.06 if session_name == "LONDON" else 0.05 if session_name == "NEW_YORK" else 0.02
            xau_grid_density_ranking_bonus += min(0.03, 0.005 * float(xau_grid_quota_debt_10m))
            xau_grid_density_ranking_bonus = clamp(
                xau_grid_density_ranking_bonus - min(0.04, float(xau_grid_soft_penalty_score) * 0.50),
                0.0,
                0.10,
            )
            pre_session_adjusted_score = clamp(float(pre_session_adjusted_score) + float(xau_grid_density_ranking_bonus), 0.0, 1.0)
        xau_grid_stretch_reversion_live_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and not weekend_mode
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and xau_grid_entry_profile in {
                "grid_stretch_reversion_long",
                "grid_stretch_reversion_short",
                "grid_prime_stretch_reversion_long",
                "grid_prime_stretch_reversion_short",
            }
            and float(candidate_meta.get("multi_tf_alignment_score", 0.5) or 0.5) >= 0.70
            and float(candidate_meta.get("fractal_persistence_score", 0.5) or 0.5) >= 0.70
            and float(candidate_meta.get("seasonality_edge_score", 0.5) or 0.5) >= 0.30
            and float(candidate_meta.get("market_instability_score", 0.0) or 0.0) <= 0.18
            and float(candidate_meta.get("feature_drift_score", 0.0) or 0.0) <= 0.12
            and regime_fit >= 0.68
            and entry_timing_score >= 0.62
            and structure_cleanliness_score >= 0.72
            and execution_quality_fit >= 0.64
        )
        xau_grid_prime_session_live_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and not weekend_mode
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and (
                xau_grid_mirror_directional_ready
                or xau_grid_native_directional_ready
                or xau_grid_density_override_ready
                or (
                    str(candidate_meta.get("xau_engine") or "").upper().startswith("GRID_DIRECTIONAL_MIRROR")
                    and xau_grid_reclaim_ready
                )
                or xau_grid_reclaim_ready
                or xau_grid_ranging_reclaim_ready
                or xau_grid_expansion_ready
                or xau_grid_entry_profile in xau_grid_live_prime_profiles
            )
        )
        xau_grid_mean_reversion_live_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and regime_state == "MEAN_REVERSION"
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and not xau_grid_fragile_prime_mean_reversion
            and (
                xau_grid_mirror_directional_ready
                or
                xau_grid_native_directional_ready
                or
                xau_grid_density_override_ready
                or
                xau_grid_native_prime_scaler_ready
                or (
                    xau_grid_reclaim_ready
                    and float(candidate_meta.get("multi_tf_alignment_score", 0.5) or 0.5) >= 0.58
                    and float(candidate_meta.get("fractal_persistence_score", 0.5) or 0.5) >= 0.56
                    and float(candidate_meta.get("market_instability_score", 0.0) or 0.0) <= 0.38
                    and float(candidate_meta.get("feature_drift_score", 0.0) or 0.0) <= 0.24
                    and regime_fit >= 0.60
                    and entry_timing_score >= 0.60
                    and structure_cleanliness_score >= 0.58
                )
            )
        )
        xau_grid_low_liquidity_live_ready = bool(
            normalized_symbol == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and regime_state == "LOW_LIQUIDITY_CHOP"
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and (
                xau_grid_mirror_directional_ready
                or
                xau_grid_native_directional_ready
                or
                (xau_grid_density_override_ready and session_name != "OVERLAP")
                or
                xau_grid_native_prime_scaler_ready
                or (
                    xau_grid_reclaim_ready
                    and xau_grid_entry_profile
                    in {
                        "grid_liquidity_reclaim_long",
                        "grid_liquidity_reclaim_short",
                        "grid_trend_reclaim_long",
                        "grid_trend_reclaim_short",
                        "grid_m15_pullback_reclaim_long",
                        "grid_m15_pullback_reclaim_short",
                    }
                    and float(candidate_meta.get("multi_tf_alignment_score", 0.5) or 0.5) >= 0.60
                    and float(candidate_meta.get("fractal_persistence_score", 0.5) or 0.5) >= 0.58
                    and float(candidate_meta.get("market_instability_score", 0.0) or 0.0) <= 0.34
                    and float(candidate_meta.get("feature_drift_score", 0.0) or 0.0) <= 0.22
                    and regime_fit >= 0.64
                    and entry_timing_score >= 0.62
                    and structure_cleanliness_score >= 0.60
                )
            )
        )
        severe_exact_bucket = (
            (normalized_symbol == "EURUSD" and strategy_key == "EURUSD_LIQUIDITY_SWEEP" and session_name == "LONDON" and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "EURUSD" and strategy_key == "EURUSD_LIQUIDITY_SWEEP" and session_name in {"LONDON", "OVERLAP", "NEW_YORK"} and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP"})
            or (normalized_symbol == "EURUSD" and strategy_key == "EURUSD_VWAP_PULLBACK" and session_name == "LONDON" and regime_state == "LOW_LIQUIDITY_CHOP")
            or (normalized_symbol == "EURUSD" and strategy_key == "EURUSD_VWAP_PULLBACK" and session_name == "LONDON" and regime_state in {"RANGING", "MEAN_REVERSION"})
            or (normalized_symbol == "EURUSD" and strategy_key == "EURUSD_LONDON_BREAKOUT" and session_name in {"OVERLAP", "NEW_YORK"} and regime_state == "LOW_LIQUIDITY_CHOP")
            or (normalized_symbol == "EURUSD" and strategy_key == "EURUSD_VWAP_PULLBACK" and session_name == "OVERLAP" and regime_state == "LOW_LIQUIDITY_CHOP")
            or (normalized_symbol == "GBPJPY" and strategy_key == "GBPJPY_SESSION_PULLBACK_CONTINUATION" and session_name == "TOKYO" and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"})
            or (normalized_symbol == "GBPJPY" and strategy_key in {"GBPJPY_MOMENTUM_IMPULSE", "GBPJPY_SESSION_PULLBACK_CONTINUATION"} and session_name in {"SYDNEY", "LONDON", "NEW_YORK"} and regime_state == "LOW_LIQUIDITY_CHOP")
            or (normalized_symbol == "AUDNZD" and strategy_key == "AUDNZD_RANGE_ROTATION" and session_name == "TOKYO" and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "AUDJPY" and strategy_key == "AUDJPY_LIQUIDITY_SWEEP_REVERSAL" and session_name == "TOKYO" and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "AUDJPY" and strategy_key == "AUDJPY_LONDON_CARRY_TREND" and session_name == "LONDON" and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"})
            or (normalized_symbol == "AUDJPY" and strategy_key == "AUDJPY_LONDON_CARRY_TREND" and session_name == "NEW_YORK" and regime_state in {"MEAN_REVERSION", "RANGING"})
            or (normalized_symbol == "AUDJPY" and strategy_key == "AUDJPY_ATR_COMPRESSION_BREAKOUT" and session_name == "OVERLAP" and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "USDJPY" and strategy_key == "USDJPY_LIQUIDITY_SWEEP_REVERSAL" and session_name == "TOKYO" and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "USDJPY" and strategy_key == "USDJPY_MOMENTUM_IMPULSE" and session_name == "TOKYO" and regime_state == "LOW_LIQUIDITY_CHOP")
            or (normalized_symbol == "USDJPY" and strategy_key == "USDJPY_VWAP_TREND_CONTINUATION" and session_name == "LONDON" and regime_state == "LOW_LIQUIDITY_CHOP")
            or (normalized_symbol == "USDJPY" and strategy_key == "USDJPY_VWAP_TREND_CONTINUATION" and session_name == "OVERLAP" and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"})
            or (normalized_symbol == "NZDJPY" and strategy_key == "NZDJPY_LIQUIDITY_TRAP_REVERSAL" and session_name in {"TOKYO", "OVERLAP"} and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "NZDJPY" and strategy_key == "NZDJPY_SESSION_RANGE_EXPANSION" and session_name == "LONDON" and regime_state == "LOW_LIQUIDITY_CHOP")
            or (normalized_symbol == "GBPUSD" and strategy_key == "GBPUSD_TREND_PULLBACK_RIDE" and session_name == "LONDON" and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"})
            or (normalized_symbol == "GBPUSD" and strategy_key == "GBPUSD_TREND_PULLBACK_RIDE" and session_name == "OVERLAP" and regime_state == "LOW_LIQUIDITY_CHOP")
            or (normalized_symbol == "GBPUSD" and strategy_key == "GBPUSD_LONDON_EXPANSION_BREAKOUT" and session_name == "LONDON")
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_TREND_SCALP" and session_name == "TOKYO" and not weekend_mode and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"})
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_VOLATILE_RETEST" and ((session_name == "TOKYO" and regime_state == "MEAN_REVERSION" and not weekend_mode) or (session_name == "NEW_YORK" and regime_state == "LOW_LIQUIDITY_CHOP")))
            or (normalized_symbol == "BTCUSD" and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION" and session_name == "NEW_YORK" and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"} and not weekend_mode)
            or (normalized_symbol == "BTCUSD" and weekend_mode and strategy_key == "BTCUSD_RANGE_EXPANSION" and session_name in {"SYDNEY", "TOKYO"} and regime_state == "LOW_LIQUIDITY_CHOP")
            or (normalized_symbol == "BTCUSD" and weekend_mode and strategy_key == "BTCUSD_TREND_SCALP" and session_name == "SYDNEY" and regime_state == "MEAN_REVERSION" and execution_quality_fit < 0.46)
            or (normalized_symbol == "BTCUSD" and weekend_mode and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION" and session_name in {"TOKYO", "SYDNEY"} and regime_state == "LOW_LIQUIDITY_CHOP" and execution_quality_fit < 0.48)
            or (normalized_symbol == "XAUUSD" and weekend_mode and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID")
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_OPENING_DRIVE_BREAKOUT" and session_name in {"LONDON", "OVERLAP", "NEW_YORK"})
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_MOMENTUM_IMPULSE" and session_name == "TOKYO" and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_MOMENTUM_IMPULSE" and session_name in {"LONDON", "OVERLAP", "NEW_YORK"})
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_MOMENTUM_IMPULSE" and ((session_name == "TOKYO" and regime_state == "LOW_LIQUIDITY_CHOP") or (session_name == "SYDNEY" and regime_state == "LOW_LIQUIDITY_CHOP") or (session_name == "LONDON" and regime_state in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"}) or (session_name == "OVERLAP" and regime_state in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"}) or (session_name == "NEW_YORK" and regime_state in {"TRENDING", "MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"})))
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_LIQUIDITY_SWEEP_REVERSAL" and session_name in {"LONDON", "OVERLAP", "TOKYO"} and regime_state == "MEAN_REVERSION")
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_OPENING_DRIVE_BREAKOUT" and ((session_name == "LONDON" and regime_state == "LOW_LIQUIDITY_CHOP") or (session_name == "OVERLAP" and regime_state == "TRENDING")))
            or (normalized_symbol == "NAS100" and strategy_key == "NAS100_OPENING_DRIVE_BREAKOUT" and session_name == "TOKYO" and regime_state != "BREAKOUT_EXPANSION")
            or (normalized_symbol == "XAUUSD" and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID" and session_name not in {"LONDON", "OVERLAP", "NEW_YORK"})
            or (normalized_symbol == "XAUUSD" and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID" and str(candidate_meta.get("grid_entry_profile") or "").startswith("grid_compression_expansion_follow"))
            or (normalized_symbol == "XAUUSD" and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID" and session_name == "LONDON" and regime_state == "RANGING" and not xau_grid_prime_session_live_ready)
            or (normalized_symbol == "XAUUSD" and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID" and regime_state == "LOW_LIQUIDITY_CHOP" and not xau_grid_low_liquidity_live_ready)
            or (normalized_symbol == "XAUUSD" and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID" and session_name in {"OVERLAP", "NEW_YORK"} and regime_state == "RANGING" and not xau_grid_prime_session_live_ready)
            or (
                normalized_symbol == "XAUUSD"
                and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
                and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
                and quality_tier == "B"
                and not xau_engine.startswith("GRID_DIRECTIONAL_MIRROR")
                and not xau_grid_native_near_a_ready
                and not xau_grid_density_override_ready
            )
            or (normalized_symbol == "XAUUSD" and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID" and regime_state == "MEAN_REVERSION" and not xau_grid_mean_reversion_live_ready)
            or (normalized_symbol == "XAUUSD" and strategy_key == "XAUUSD_ATR_EXPANSION_SCALPER" and not xau_directional_atr_quality_ready and session_name in {"LONDON", "OVERLAP", "NEW_YORK"} and regime_state in {"TRENDING", "RANGING", "MEAN_REVERSION"})
            or (
                normalized_symbol == "XAUUSD"
                and strategy_key == "XAUUSD_NY_MOMENTUM_BREAKOUT"
                and not (xau_directional_structured_quality_ready or xau_directional_prime_attack_ready)
                and session_name in {"OVERLAP", "NEW_YORK"}
                and regime_state == "TRENDING"
            )
            or (
                normalized_symbol == "XAUUSD"
                and strategy_key == "XAUUSD_ATR_EXPANSION_SCALPER"
                and session_name not in {"LONDON", "OVERLAP", "NEW_YORK"}
                and not xau_directional_atr_off_session_quality_ready
            )
            or (
                normalized_symbol == "XAUUSD"
                and strategy_key == "XAUUSD_NY_MOMENTUM_BREAKOUT"
                and session_name not in {"OVERLAP", "NEW_YORK"}
                and not (xau_directional_off_session_quality_ready or xau_directional_off_session_attack_ready)
            )
            or (normalized_symbol == "USOIL" and strategy_key == "USOIL_INVENTORY_MOMENTUM" and session_name == "TOKYO" and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"})
            or (normalized_symbol == "USOIL" and strategy_key == "USOIL_LONDON_TREND_EXPANSION" and session_name in {"LONDON", "OVERLAP", "NEW_YORK"})
            or (normalized_symbol == "USOIL" and strategy_key == "USOIL_LONDON_TREND_EXPANSION" and ((session_name == "SYDNEY" and regime_state == "TRENDING") or (session_name == "OVERLAP" and regime_state in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"}) or (session_name == "NEW_YORK" and regime_state == "LOW_LIQUIDITY_CHOP")))
        )
        if severe_exact_bucket:
            allow_severe_bucket_override = (
                (
                    pre_session_adjusted_score >= 0.88
                    and regime_fit >= 0.86
                    and session_fit >= 0.74
                    and entry_timing_score >= 0.82
                    and structure_cleanliness_score >= 0.82
                    and execution_quality_fit >= 0.78
                    and pair_behavior_fit >= 0.64
                    and strategy_recent_performance >= 0.60
                )
                or (
                    xau_directional_breakout_setup
                    and pre_session_adjusted_score >= 0.82
                    and regime_fit >= 0.80
                    and session_fit >= 0.70
                    and entry_timing_score >= 0.76
                    and structure_cleanliness_score >= 0.78
                    and execution_quality_fit >= 0.72
                    and pair_behavior_fit >= 0.60
                    and strategy_recent_performance >= 0.58
                    and multi_tf_alignment_score >= 0.60
                    and market_instability_score <= 0.34
                    and feature_drift_score <= 0.28
                )
            )
            if not allow_severe_bucket_override:
                continue
        if (
            strategy_state == "QUARANTINED"
            and strategy_bucket_reason in {"targeted_exact_bucket_demoted", "targeted_ranging_bucket_fast_failures"}
        ):
            allow_quarantined_exploration = (
                normalized_symbol in {"XAUUSD", "BTCUSD", "NAS100"}
                and regime_fit >= 0.78
                and entry_timing_score >= 0.72
                and structure_cleanliness_score >= 0.72
                and execution_quality_fit >= 0.70
                and pair_behavior_fit >= 0.52
                and strategy_recent_performance >= 0.55
                and pre_session_adjusted_score >= 0.66
            )
            if not allow_quarantined_exploration:
                continue
        entry = {
            "strategy_key": strategy_key,
            "setup": setup_name,
            "lane_name": lane_name,
            "strategy_state": strategy_state,
            "router_rank_score": float(router_rank_score),
            "strategy_pool_rank_score": float(pre_session_adjusted_score),
            "strategy_score": float(pre_selection_score),
            "rank_score": float(pre_session_adjusted_score),
            "xau_directional_prime_attack_ready": bool(xau_directional_prime_attack_ready),
            "xau_directional_off_session_attack_ready": bool(xau_directional_off_session_attack_ready),
            "xau_directional_breakout_ranking_bonus": float(xau_directional_breakout_ranking_bonus),
            "btc_weekday_price_action_ready": bool(btc_weekday_price_action_ready),
            "btc_weekday_volatile_retest_ready": bool(btc_weekday_volatile_retest_ready),
            "btc_weekend_printer_ready": bool(btc_weekend_printer_ready),
            "btc_directional_ranking_bonus": float(btc_directional_ranking_bonus),
            "xau_grid_density_override_ready": bool(xau_grid_density_override_ready),
            "xau_grid_density_ranking_bonus": float(xau_grid_density_ranking_bonus),
            "regime_state": regime_state,
            "regime_fit": float(regime_fit),
            "session_fit": float(session_fit),
            "volatility_fit": float(volatility_fit),
            "pair_behavior_fit": float(pair_behavior_fit),
            "execution_quality_fit": float(execution_quality_fit),
            "entry_timing_score": float(entry_timing_score),
            "structure_cleanliness_score": float(structure_cleanliness_score),
            "quality_tier": str(quality_tier),
            "tier_size_multiplier": float(tier_size_multiplier),
            "delta_proxy_score": float(candidate_meta.get("delta_proxy_score", 0.0) or 0.0),
            "compression_proxy_state": str(candidate_meta.get("compression_proxy_state") or ""),
            "compression_expansion_score": float(candidate_meta.get("compression_expansion_score", 0.0) or 0.0),
            "transition_momentum": float(transition_momentum),
            "transition_momentum_size_multiplier": float(transition_momentum_size_multiplier),
            "compression_burst_size_multiplier": float(compression_burst_size_multiplier),
            "confidence_multiplier": float(confidence_multiplier),
            "market_instability_score": float(market_instability_score),
            "feature_drift_score": float(feature_drift_score),
            "multi_tf_alignment_score": float(multi_tf_alignment_score),
            "seasonality_edge_score": float(seasonality_edge_score),
            "fractal_persistence_score": float(fractal_persistence_score),
            "velocity_decay": float(velocity_decay.get("size_multiplier", 1.0) or 1.0),
            "velocity_decay_score_penalty": float(velocity_decay.get("score_penalty", 0.0) or 0.0),
            "velocity_trades_per_10_bars": float(velocity_decay.get("trades_per_10_bars", 0.0) or 0.0),
            "correlation_penalty": float(correlation_penalty),
            "btc_weekend_mode": bool(candidate_meta.get("btc_weekend_mode", False) or btc_weekend_printer_lane),
            "btc_velocity_decay": float(velocity_decay.get("size_multiplier", 1.0) or 1.0),
            "streak_adjust_mode": str(streak_adjust.get("mode", "NEUTRAL")),
            "manager_reason": str(streak_adjust.get("manager_reason", "streak_neutral")),
            "grid_spacing_multiplier_hint": float(
                xau_grid_compression_spacing_multiplier
                if (
                    normalized_symbol == "XAUUSD"
                    and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
                    and str(candidate_meta.get("compression_proxy_state") or "").upper() == "COMPRESSION"
                )
                else candidate_meta.get("grid_spacing_multiplier_hint", 1.0) or 1.0
            ),
            "session_loosen_factor": float(candidate_meta.get("session_loosen_factor", 1.0) or 1.0),
            "throughput_recovery_active": bool(candidate_throughput_recovery_active),
            "recycle_session": bool(recycle_session),
            "recycle_origin_session": str(recycle_origin_session),
            "recycle_boost_applied": float(recycle_boost_applied),
            "family_rotation_penalty": float(family_rotation_penalty),
            "equity_momentum_mode": str(equity_momentum.get("mode", "NEUTRAL")),
            "strategy_recent_performance": float(strategy_recent_performance),
            "session_priority_profile": str(priority.session_priority_profile),
            "lane_session_priority": str(priority.lane_session_priority),
            "session_native_pair": bool(priority.session_native_pair),
            "session_priority_multiplier": float(priority.session_priority_multiplier),
            "pair_priority_rank_in_session": int(priority.pair_priority_rank_in_session),
            "lane_budget_share": float(priority.lane_budget_share),
            "lane_available_capacity": float(candidate_meta.get("lane_available_capacity", 0.0) or 0.0),
            "lane_capacity_usage": float(candidate_meta.get("lane_capacity_usage", 0.0) or 0.0),
            "exceptional_override_used": bool(candidate_meta.get("exceptional_override_used", False)),
            "exceptional_override_reason": str(candidate_meta.get("exceptional_override_reason") or ""),
            "why_non_native_pair_won": str(candidate_meta.get("why_non_native_pair_won") or ""),
            "why_native_pair_lost_priority": str(candidate_meta.get("why_native_pair_lost_priority") or ""),
            "market_data_consensus_adjustment": float(candidate_meta.get("market_data_consensus_adjustment", 0.0) or 0.0),
            "why_strategy_promoted": str(strategy_bucket_metrics.get("why_strategy_promoted") or ""),
            "why_strategy_throttled": str(strategy_bucket_metrics.get("why_strategy_throttled") or ""),
            "strategy_bucket_reason": str(strategy_bucket_reason),
        }
        verified_reason_code, verified_reason_text = _verified_candidate_reason(entry)
        entry["verified_reason_code"] = str(verified_reason_code)
        entry["verified_reason_text"] = str(verified_reason_text)
        ranked.append((candidate, entry))
    ranked.sort(
        key=lambda item: (
            -float(item[1]["strategy_pool_rank_score"]),
            int(item[1]["pair_priority_rank_in_session"]),
            -float(item[1]["router_rank_score"]),
            str(item[1]["strategy_key"]),
        )
    )
    for index, (candidate, entry) in enumerate(ranked, start=1):
        entry["rank"] = int(index)
        candidate.meta["strategy_pool_rank"] = int(index)
        candidate.meta["strategy_pool_rank_score"] = float(entry["strategy_pool_rank_score"])
        candidate.meta["strategy_score"] = float(entry["strategy_score"])
        candidate.meta["rank_score"] = float(entry["rank_score"])
        candidate.meta["lane_name"] = str(entry["lane_name"])
        candidate.meta["xau_directional_prime_attack_ready"] = bool(entry.get("xau_directional_prime_attack_ready", False))
        candidate.meta["xau_directional_off_session_attack_ready"] = bool(entry.get("xau_directional_off_session_attack_ready", False))
        candidate.meta["xau_directional_breakout_ranking_bonus"] = float(entry.get("xau_directional_breakout_ranking_bonus", 0.0) or 0.0)
        candidate.meta["btc_weekday_price_action_ready"] = bool(entry.get("btc_weekday_price_action_ready", False))
        candidate.meta["btc_weekday_volatile_retest_ready"] = bool(entry.get("btc_weekday_volatile_retest_ready", False))
        candidate.meta["btc_weekend_printer_ready"] = bool(entry.get("btc_weekend_printer_ready", False))
        candidate.meta["btc_directional_ranking_bonus"] = float(entry.get("btc_directional_ranking_bonus", 0.0) or 0.0)
        candidate.meta["quality_tier"] = str(entry.get("quality_tier") or candidate.meta.get("quality_tier") or "")
        candidate.meta["tier_size_multiplier"] = float(entry.get("tier_size_multiplier", candidate.meta.get("tier_size_multiplier", 1.0)) or 1.0)
        candidate.meta["throughput_recovery_active"] = bool(entry.get("throughput_recovery_active", False))
        candidate.meta["recycle_session"] = bool(entry.get("recycle_session", False))
        candidate.meta["recycle_origin_session"] = str(entry.get("recycle_origin_session") or "")
        candidate.meta["recycle_boost_applied"] = float(entry.get("recycle_boost_applied", 0.0) or 0.0)
        candidate.meta["family_rotation_penalty"] = float(entry.get("family_rotation_penalty", 0.0) or 0.0)
        candidate.meta["equity_momentum_mode"] = str(entry.get("equity_momentum_mode") or "NEUTRAL")
        candidate.meta["transition_momentum"] = float(entry.get("transition_momentum", 0.0) or 0.0)
        candidate.meta["transition_momentum_size_multiplier"] = float(entry.get("transition_momentum_size_multiplier", 1.0) or 1.0)
        candidate.meta["compression_burst_size_multiplier"] = float(entry.get("compression_burst_size_multiplier", 1.0) or 1.0)
        candidate.meta["confidence_multiplier"] = float(entry.get("confidence_multiplier", 1.0) or 1.0)
        candidate.meta["velocity_decay"] = float(entry.get("velocity_decay", 1.0) or 1.0)
        candidate.meta["velocity_decay_score_penalty"] = float(entry.get("velocity_decay_score_penalty", 0.0) or 0.0)
        candidate.meta["velocity_trades_per_10_bars"] = float(entry.get("velocity_trades_per_10_bars", 0.0) or 0.0)
        candidate.meta["correlation_penalty"] = float(entry.get("correlation_penalty", 0.0) or 0.0)
        candidate.meta["verified_reason_code"] = str(entry.get("verified_reason_code") or "")
        candidate.meta["verified_reason_text"] = str(entry.get("verified_reason_text") or "")
    return ranked


def _strategy_pool_winner_reason(ranked_entries: list[dict[str, Any]]) -> str:
    if not ranked_entries:
        return ""
    winner = dict(ranked_entries[0] or {})
    if len(ranked_entries) == 1:
        return "only_strategy_candidate_available"
    runner_up = dict(ranked_entries[1] or {})
    winner_score = float(winner.get("strategy_pool_rank_score", 0.0) or 0.0)
    runner_score = float(runner_up.get("strategy_pool_rank_score", 0.0) or 0.0)
    delta = winner_score - runner_score
    if bool(winner.get("session_native_pair")) and not bool(runner_up.get("session_native_pair")) and delta < 0.06:
        return "session_native_priority_edge"
    if str(winner.get("strategy_state", "")).upper() == "ATTACK" and str(runner_up.get("strategy_state", "")).upper() != "ATTACK":
        return "attack_strategy_priority"
    if float(winner.get("regime_fit", 0.0) or 0.0) - float(runner_up.get("regime_fit", 0.0) or 0.0) >= 0.10:
        return "better_regime_fit"
    if float(winner.get("entry_timing_score", 0.0) or 0.0) - float(runner_up.get("entry_timing_score", 0.0) or 0.0) >= 0.10:
        return "better_entry_timing"
    if float(winner.get("structure_cleanliness_score", 0.0) or 0.0) - float(runner_up.get("structure_cleanliness_score", 0.0) or 0.0) >= 0.10:
        return "cleaner_structure"
    if float(winner.get("execution_quality_fit", 0.0) or 0.0) - float(runner_up.get("execution_quality_fit", 0.0) or 0.0) >= 0.10:
        return "better_execution_quality"
    if delta >= 0.10:
        return "clear_strategy_score_superiority"
    return "higher_combined_strategy_score"


def _verified_candidate_reason(entry: dict[str, Any]) -> tuple[str, str]:
    if bool(entry.get("recycle_session")):
        return "recycled_high_liquidity", "Recycled low-liquidity idea improved in a better session."
    if str(entry.get("compression_proxy_state") or "").upper() == "COMPRESSION":
        strategy_key = str(entry.get("strategy_key") or "").upper()
        if any(token in strategy_key for token in ("RETEST", "SWEEP", "ROTATION", "REVERSION", "VWAP")):
            return "compression_retest_edge", "Compression regime favored retest or sweep structure."
    if bool(entry.get("throughput_recovery_active")) and str(entry.get("quality_tier") or "").upper() == "B":
        return "throughput_recovery_b", "B-tier flow preserved while candidate density was low."
    if str(entry.get("quality_tier") or "").upper() == "A+":
        return "attack_quality_clean", "Attack-tier structure, timing, and execution quality aligned."
    return "combined_strategy_score", "Selected on combined strategy score, regime fit, and execution quality."


def _summarize_strategy_pool_rankings(
    *,
    symbol_key: str,
    ranked_entries: list[dict[str, Any]],
    preferred_strategy_key: str = "",
    session_name: str = "",
    regime_state: str = "",
) -> list[dict[str, Any]]:
    normalized_symbol = _normalize_symbol_key(symbol_key)
    ranked_by_key: dict[str, dict[str, Any]] = {}
    for entry in ranked_entries:
        entry_key = str(entry.get("strategy_key") or "").strip()
        if not entry_key:
            continue
        existing = ranked_by_key.get(entry_key)
        current_score = float(
            entry.get("strategy_score")
            or entry.get("strategy_pool_rank_score")
            or entry.get("rank_score")
            or 0.0
        )
        existing_score = float(
            (existing or {}).get("strategy_score")
            or (existing or {}).get("strategy_pool_rank_score")
            or (existing or {}).get("rank_score")
            or 0.0
        )
        if existing is None or current_score > existing_score:
            ranked_by_key[entry_key] = {
                "strategy_key": entry_key,
                "strategy_score": float(current_score),
                "rank_score": float(
                    entry.get("rank_score")
                    or entry.get("strategy_pool_rank_score")
                    or current_score
                ),
                "strategy_state": str(entry.get("strategy_state") or "NORMAL"),
                "regime_state": str(entry.get("regime_state") or "").strip().upper(),
                "session_native_pair": bool(entry.get("session_native_pair", False)),
                "session_priority_multiplier": float(entry.get("session_priority_multiplier", 1.0) or 1.0),
                "pair_priority_rank_in_session": int(entry.get("pair_priority_rank_in_session", 99) or 99),
                "regime_fit": float(entry.get("regime_fit", 0.0) or 0.0),
                "session_fit": float(entry.get("session_fit", 0.0) or 0.0),
                "volatility_fit": float(entry.get("volatility_fit", 0.0) or 0.0),
                "pair_behavior_fit": float(entry.get("pair_behavior_fit", 0.0) or 0.0),
                "execution_quality_fit": float(entry.get("execution_quality_fit", 0.0) or 0.0),
                "entry_timing_score": float(entry.get("entry_timing_score", 0.0) or 0.0),
                "structure_cleanliness_score": float(entry.get("structure_cleanliness_score", 0.0) or 0.0),
                "quality_tier": str(entry.get("quality_tier") or ""),
                "tier_size_multiplier": float(entry.get("tier_size_multiplier", 1.0) or 1.0),
                "delta_proxy_score": float(entry.get("delta_proxy_score", 0.0) or 0.0),
                "compression_proxy_state": str(entry.get("compression_proxy_state") or ""),
                "transition_momentum": float(entry.get("transition_momentum", 0.0) or 0.0),
                "transition_momentum_size_multiplier": float(entry.get("transition_momentum_size_multiplier", 1.0) or 1.0),
                "compression_burst_size_multiplier": float(entry.get("compression_burst_size_multiplier", 1.0) or 1.0),
                "velocity_decay": float(entry.get("velocity_decay", 1.0) or 1.0),
                "velocity_decay_score_penalty": float(entry.get("velocity_decay_score_penalty", 0.0) or 0.0),
                "velocity_trades_per_10_bars": float(entry.get("velocity_trades_per_10_bars", 0.0) or 0.0),
                "correlation_penalty": float(entry.get("correlation_penalty", 0.0) or 0.0),
                "session_loosen_factor": float(entry.get("session_loosen_factor", 1.0) or 1.0),
                "throughput_recovery_active": bool(entry.get("throughput_recovery_active", False)),
                "strategy_recent_performance": float(entry.get("strategy_recent_performance", 0.50) or 0.50),
                "lane_name": str(entry.get("lane_name") or ""),
                "session_priority_profile": str(entry.get("session_priority_profile") or ""),
                "lane_session_priority": str(entry.get("lane_session_priority") or ""),
                "lane_budget_share": float(entry.get("lane_budget_share", 0.0) or 0.0),
                "lane_available_capacity": float(entry.get("lane_available_capacity", 0.0) or 0.0),
                "lane_capacity_usage": float(entry.get("lane_capacity_usage", 0.0) or 0.0),
                "exceptional_override_used": bool(entry.get("exceptional_override_used", False)),
                "exceptional_override_reason": str(entry.get("exceptional_override_reason") or ""),
                "why_non_native_pair_won": str(entry.get("why_non_native_pair_won") or ""),
                "why_native_pair_lost_priority": str(entry.get("why_native_pair_lost_priority") or ""),
                "market_data_consensus_adjustment": float(
                    entry.get("market_data_consensus_adjustment", 0.0) or 0.0
                ),
            }
        elif existing is not None:
            existing_profile = str(existing.get("session_priority_profile") or "").upper()
            entry_profile = str(entry.get("session_priority_profile") or "").upper()
            if (not existing_profile or existing_profile == "GLOBAL") and entry_profile and entry_profile != "GLOBAL":
                existing["session_priority_profile"] = str(entry.get("session_priority_profile") or "")
            existing_lane_priority = str(existing.get("lane_session_priority") or "").upper()
            entry_lane_priority = str(entry.get("lane_session_priority") or "").upper()
            if (not existing_lane_priority or existing_lane_priority == "NEUTRAL") and entry_lane_priority and entry_lane_priority != "NEUTRAL":
                existing["lane_session_priority"] = str(entry.get("lane_session_priority") or "")
            if not bool(existing.get("session_native_pair", False)) and bool(entry.get("session_native_pair", False)):
                existing["session_native_pair"] = True
            existing["session_priority_multiplier"] = max(
                float(existing.get("session_priority_multiplier", 1.0) or 1.0),
                float(entry.get("session_priority_multiplier", 1.0) or 1.0),
            )
            existing["pair_priority_rank_in_session"] = min(
                int(existing.get("pair_priority_rank_in_session", 99) or 99),
                int(entry.get("pair_priority_rank_in_session", 99) or 99),
            )
            existing["lane_budget_share"] = max(
                float(existing.get("lane_budget_share", 0.0) or 0.0),
                float(entry.get("lane_budget_share", 0.0) or 0.0),
            )
            existing["lane_available_capacity"] = max(
                float(existing.get("lane_available_capacity", 0.0) or 0.0),
                float(entry.get("lane_available_capacity", 0.0) or 0.0),
            )
            existing["lane_capacity_usage"] = max(
                float(existing.get("lane_capacity_usage", 0.0) or 0.0),
                float(entry.get("lane_capacity_usage", 0.0) or 0.0),
            )
            existing["exceptional_override_used"] = bool(
                existing.get("exceptional_override_used", False) or entry.get("exceptional_override_used", False)
            )
            if not existing.get("exceptional_override_reason") and entry.get("exceptional_override_reason"):
                existing["exceptional_override_reason"] = str(entry.get("exceptional_override_reason") or "")
            if not existing.get("why_non_native_pair_won") and entry.get("why_non_native_pair_won"):
                existing["why_non_native_pair_won"] = str(entry.get("why_non_native_pair_won") or "")
            if not existing.get("why_native_pair_lost_priority") and entry.get("why_native_pair_lost_priority"):
                existing["why_native_pair_lost_priority"] = str(entry.get("why_native_pair_lost_priority") or "")
            existing["market_data_consensus_adjustment"] = max(
                float(existing.get("market_data_consensus_adjustment", 0.0) or 0.0),
                float(entry.get("market_data_consensus_adjustment", 0.0) or 0.0),
            )
    canonical_pool = list(StrategyRouter._strategy_pool_keys(normalized_symbol))
    preferred_key = str(preferred_strategy_key or "").strip()
    if preferred_key and preferred_key not in canonical_pool:
        canonical_pool.insert(0, preferred_key)
    ordered = sorted(
        ranked_by_key.values(),
        key=lambda item: (
            -float(item.get("strategy_score", 0.0) or 0.0),
            int(item.get("pair_priority_rank_in_session", 99) or 99),
            str(item.get("strategy_key") or ""),
        ),
    )
    if preferred_key and preferred_key in ranked_by_key:
        ordered.sort(
            key=lambda item: (
                0 if str(item.get("strategy_key") or "") == preferred_key else 1,
                -float(item.get("strategy_score", 0.0) or 0.0),
                int(item.get("pair_priority_rank_in_session", 99) or 99),
                str(item.get("strategy_key") or ""),
            ),
        )
    tail_score = float(ordered[-1].get("strategy_score", 0.0) or 0.0) if ordered else 0.0
    resolved_session_name = str(session_name or "").strip().upper()
    resolved_regime_state = runtime_regime_state(
        str(regime_state or (ordered[0].get("regime_state") if ordered else "") or "")
    )
    baseline_structure = clamp(
        float(ordered[0].get("structure_cleanliness_score", 0.62) or 0.62) if ordered else 0.62,
        0.35,
        0.95,
    )
    baseline_entry = clamp(
        float(ordered[0].get("entry_timing_score", 0.58) or 0.58) if ordered else 0.58,
        0.30,
        0.92,
    )
    baseline_execution = clamp(
        float(ordered[0].get("execution_quality_fit", 0.72) or 0.72) if ordered else 0.72,
        0.35,
        1.0,
    )
    for strategy_name in canonical_pool:
        if strategy_name in ranked_by_key:
            continue
        strategy_lane = infer_trade_lane(
            symbol=normalized_symbol,
            setup=strategy_name,
            setup_family="TREND",
            session_name=resolved_session_name,
        )
        priority = session_priority_context(
            symbol=normalized_symbol,
            lane_name=strategy_lane,
            session_name=resolved_session_name,
        )
        regime_fit_value = strategy_regime_fit(strategy_name, resolved_regime_state)
        session_fit_value = clamp(float(priority.session_priority_multiplier) - 0.10, 0.0, 1.0)
        pair_fit_value = pair_behavior_fit(
            symbol=normalized_symbol,
            strategy_key=strategy_name,
            session_name=resolved_session_name,
            regime_state=resolved_regime_state,
        )
        recent_seed_value = StrategyRouter._strategy_performance_seed(
            symbol=normalized_symbol,
            strategy_key=strategy_name,
            session_name=resolved_session_name,
        )
        false_break_penalty = max(0.0, 0.55 - baseline_structure)
        chop_penalty = (
            0.20
            if resolved_regime_state == "LOW_LIQUIDITY_CHOP"
            and any(token in str(strategy_name).upper() for token in ("BREAKOUT", "IMPULSE", "EXPANSION"))
            else 0.0
        )
        bench_strategy_score = strategy_selection_score(
            ev_estimate=max(0.0, tail_score - 0.04),
            regime_fit=float(regime_fit_value),
            session_fit=float(session_fit_value),
            volatility_fit=0.70,
            pair_behavior_fit_score=float(pair_fit_value),
            strategy_recent_performance=float(recent_seed_value),
            execution_quality_fit=float(baseline_execution),
            entry_timing_score_value=float(baseline_entry * 0.94),
            structure_cleanliness_score_value=float(baseline_structure * 0.94),
            drawdown_penalty=0.0,
            false_break_penalty=float(false_break_penalty),
            chop_penalty=float(chop_penalty),
        )
        bench_rank_score = session_adjusted_score(
            base_score=float(bench_strategy_score),
            session_priority_multiplier=float(priority.session_priority_multiplier),
            lane_strength_multiplier=1.0,
            quality_floor_edge=float(priority.quality_floor_edge),
        )
        fallback_score = max(
            0.0,
            min(
                float(bench_strategy_score),
                (tail_score - 0.01) if ordered else float(bench_strategy_score),
            ),
        )
        ordered.append(
            {
                "strategy_key": str(strategy_name),
                "strategy_score": float(fallback_score),
                "rank_score": float(bench_rank_score),
                "strategy_state": "NORMAL",
                "regime_state": str(resolved_regime_state),
                "session_native_pair": bool(priority.session_native_pair),
                "session_priority_multiplier": float(priority.session_priority_multiplier),
                "pair_priority_rank_in_session": int(priority.pair_priority_rank_in_session),
                "regime_fit": float(regime_fit_value),
                "session_fit": float(session_fit_value),
                "volatility_fit": 0.70,
                "pair_behavior_fit": float(pair_fit_value),
                "execution_quality_fit": float(baseline_execution),
                "entry_timing_score": float(baseline_entry * 0.94),
                "structure_cleanliness_score": float(baseline_structure * 0.94),
                "strategy_recent_performance": float(recent_seed_value),
                "lane_name": str(strategy_lane),
                "session_priority_profile": str(priority.session_priority_profile),
                "lane_session_priority": str(priority.lane_session_priority),
                "lane_budget_share": float(priority.lane_budget_share),
                "lane_available_capacity": 0.0,
                "lane_capacity_usage": 0.0,
                "exceptional_override_used": False,
                "exceptional_override_reason": "",
                "why_non_native_pair_won": "",
                "why_native_pair_lost_priority": "",
                "market_data_consensus_adjustment": 0.0,
            }
        )
    ordered.sort(
        key=lambda item: (
            0 if preferred_key and str(item.get("strategy_key") or "") == preferred_key else 1,
            -float(item.get("strategy_score", 0.0) or 0.0),
            int(item.get("pair_priority_rank_in_session", 99) or 99),
            str(item.get("strategy_key") or ""),
        ),
    )
    for index, entry in enumerate(ordered, start=1):
        entry["rank"] = int(index)
    return ordered


def _market_data_consensus_adjustment(consensus_state: str) -> float:
    state = str(consensus_state or "").strip().upper()
    if state == "MT5_PRIMARY":
        return 0.02
    if state == "EXTERNAL_MULTI_PROVIDER":
        return 0.015
    if state == "EXTERNAL_SINGLE_PROVIDER":
        return 0.0
    if state == "EXTERNAL_DIVERGENT":
        return -0.05
    if state == "UNAVAILABLE":
        return -0.03
    return 0.0


def _session_order_lane(symbol_key: str, session_name: str) -> str:
    normalized = _normalize_symbol_key(symbol_key)
    session_key = str(session_name or "").upper()
    if normalized == "XAUUSD" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
        return xau_grid_lane_for_session(session_key, category="ATTACK")
    return infer_trade_lane(symbol=normalized, setup_family="TREND", session_name=session_key)


def _session_runtime_defaults(
    *,
    symbol: str,
    session_name: str,
    current_scaling_state: dict[str, Any],
    setup: str = "",
    setup_family: str = "",
) -> dict[str, Any]:
    lane_name = infer_trade_lane(
        symbol=symbol,
        setup=str(setup or ""),
        setup_family=normalize_strategy_family(setup_family),
        session_name=session_name,
    )
    priority = session_priority_context(
        symbol=symbol,
        lane_name=str(lane_name),
        session_name=session_name,
    )
    projected_capacity = max(
        0,
        int(
            current_scaling_state.get(
                "projected_trade_capacity_today",
                current_scaling_state.get("stretch_daily_trade_target", 0),
            )
            or 0
        ),
    )
    lane_share = clamp(float(priority.lane_budget_share), 0.0, 1.0)
    lane_available_capacity = 0
    if lane_share > 0.0 and projected_capacity > 0:
        lane_available_capacity = max(1, int(round(projected_capacity * lane_share)))
        lane_available_capacity = min(projected_capacity, lane_available_capacity)
    return {
        "lane_name": str(lane_name),
        "session_priority_profile": str(priority.session_priority_profile),
        "lane_session_priority": str(priority.lane_session_priority),
        "session_native_pair": bool(priority.session_native_pair),
        "session_priority_multiplier": float(priority.session_priority_multiplier),
        "pair_priority_rank_in_session": int(priority.pair_priority_rank_in_session),
        "lane_budget_share": float(priority.lane_budget_share),
        "lane_available_capacity": int(lane_available_capacity),
    }


def _order_symbols_for_session(configured_symbols: list[str], symbol_contexts: dict[str, dict[str, Any]], session_name: str) -> list[str]:
    def _sort_key(symbol: str) -> tuple[int, float, int, str]:
        context = symbol_contexts.get(symbol) or {}
        resolved_symbol = str(context.get("resolved_symbol") or symbol)
        priority = session_priority_context(
            symbol=resolved_symbol,
            lane_name=_session_order_lane(resolved_symbol, session_name),
            session_name=session_name,
        )
        market_open = bool(context.get("market_open", True))
        return (
            0 if market_open else 1,
            int(priority.pair_priority_rank_in_session),
            -float(priority.session_priority_multiplier),
            str(resolved_symbol),
        )

    return sorted(list(configured_symbols), key=_sort_key)


def _session_native_leader_should_update(
    leader: dict[str, Any],
    *,
    band_rank: int,
    adjusted_score: float,
    probability: float,
) -> bool:
    if not leader:
        return True
    leader_band_rank = int(leader.get("band_rank", 0))
    leader_adjusted_score = float(leader.get("adjusted_score", 0.0))
    leader_probability = float(leader.get("probability", 0.0))
    if band_rank != leader_band_rank:
        return band_rank > leader_band_rank
    if abs(adjusted_score - leader_adjusted_score) >= 1e-6:
        return adjusted_score > leader_adjusted_score
    return probability > leader_probability


def _quality_cluster_score(counts: dict[str, int]) -> float:
    strong_count = (
        int(counts.get("A+", 0))
        + int(counts.get("A", 0))
        + int(counts.get("A-", 0))
        + int(counts.get("B+", 0))
    )
    weighted = (
        (int(counts.get("A+", 0)) * 1.00)
        + ((int(counts.get("A", 0)) + int(counts.get("A-", 0))) * 0.80)
        + (int(counts.get("B+", 0)) * 0.65)
        + (int(counts.get("B", 0)) * 0.35)
    )
    if strong_count <= 0:
        return 0.0
    return clamp((weighted / 3.0), 0.0, 1.0)


def _session_density_score(session_name: str) -> float:
    session_key = str(session_name or "").upper()
    if session_key == "OVERLAP":
        return 1.0
    if session_key in {"LONDON", "NEW_YORK"}:
        return 0.90
    if session_key in {"TOKYO", "SYDNEY"}:
        return 0.72
    return 0.55


def _lane_score_from_adjustment(summary: dict[str, Any], fallback_score: float) -> float:
    trades = int(float(summary.get("trades", 0.0) or 0.0))
    if trades <= 0:
        return clamp(float(fallback_score or 0.0), 0.0, 1.0)
    win_rate = float(summary.get("win_rate", 0.5) or 0.5)
    expectancy = float(summary.get("expectancy_r", 0.0) or 0.0)
    profit_factor = float(summary.get("profit_factor", 1.0) or 1.0)
    score = (
        0.50
        + ((win_rate - 0.50) * 0.60)
        + clamp(expectancy, -0.25, 0.50) * 0.40
        + clamp(profit_factor - 1.0, -0.40, 0.60) * 0.15
    )
    return clamp(score, 0.0, 1.0)


def _trade_pnl_r_value(trade: dict[str, Any]) -> float:
    pnl_r = float(trade.get("pnl_r") or 0.0)
    if abs(pnl_r) > 1e-9:
        return pnl_r
    pnl_amount = float(trade.get("pnl_amount") or 0.0)
    if pnl_amount > 0.0:
        return 0.5
    if pnl_amount < 0.0:
        return -0.5
    return 0.0


def _breakout_loss_flag(setup: str, pnl_amount: float) -> float:
    if float(pnl_amount) >= 0.0:
        return 0.0
    normalized = str(setup or "").upper()
    if any(token in normalized for token in ("BREAKOUT", "EXPANSION", "CONTINUATION", "RETEST")):
        return 1.0
    return 0.0


def _safe_json_dict(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw in (None, "", b""):
        return {}
    try:
        parsed = json.loads(str(raw))
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _trade_strategy_key(trade: dict[str, Any]) -> str:
    strategy_key = str(trade.get("strategy_key") or "").strip()
    if strategy_key:
        return strategy_key
    return str(
        resolve_strategy_key(
            _normalize_symbol_key(str(trade.get("symbol") or "")),
            str(trade.get("setup") or ""),
        )
    )


def _trade_session_name(trade: dict[str, Any]) -> str:
    session_name = str(trade.get("session_name") or "").strip().upper()
    if session_name:
        return session_name
    closed_at = _parse_iso_utc(trade.get("closed_at"))
    if closed_at is None:
        return ""
    return dominant_session_name(closed_at)


def _trade_regime_bucket(trade: dict[str, Any]) -> str:
    return runtime_regime_state(str(trade.get("regime") or ""))


def _trade_review_payload(trade: dict[str, Any]) -> dict[str, Any]:
    return _safe_json_dict(trade.get("post_trade_review_json"))


def _trade_management_payload(trade: dict[str, Any]) -> dict[str, Any]:
    return _safe_json_dict(trade.get("management_effect_json"))


def _trade_review_issue(trade: dict[str, Any]) -> str:
    review = _trade_review_payload(trade)
    return str(
        review.get("likely_issue")
        or review.get("failure_bucket")
        or review.get("pattern_issue")
        or review.get("review_reason")
        or ""
    ).strip().lower()


def _trade_management_reason(trade: dict[str, Any]) -> str:
    management = _trade_management_payload(trade)
    return str(
        management.get("reason")
        or management.get("protection_mode")
        or trade.get("exit_reason")
        or ""
    ).strip().lower()


def _trade_setup_family(trade: dict[str, Any]) -> str:
    explicit = normalize_strategy_family(
        str(
            trade.get("setup_family")
            or trade.get("strategy_family")
            or _trade_review_payload(trade).get("setup_family")
            or ""
        )
    )
    if explicit != "TREND" or str(trade.get("setup_family") or trade.get("strategy_family") or "").strip():
        return explicit
    return normalize_strategy_family(_trade_strategy_key(trade))


def _trade_lane_name(trade: dict[str, Any]) -> str:
    for payload in (
        trade,
        _safe_json_dict(trade.get("close_context_json")),
        _trade_review_payload(trade),
        _trade_management_payload(trade),
    ):
        lane_name = str(
            payload.get("lane_name")
            or payload.get("lane")
            or ""
        ).strip().upper()
        if lane_name:
            return lane_name
    return infer_trade_lane(
        symbol=str(trade.get("symbol") or ""),
        setup=str(trade.get("setup") or _trade_strategy_key(trade)),
        setup_family=_trade_setup_family(trade),
        session_name=_trade_session_name(trade),
    )


def _consecutive_lane_losses(closed_trades: list[dict[str, Any]], lane_name: str) -> int:
    lane_key = str(lane_name or "").strip().upper()
    if not lane_key:
        return 0
    losses = 0
    for trade in closed_trades:
        if _trade_lane_name(trade) != lane_key:
            continue
        pnl_r = float(trade.get("pnl_r", trade.get("pnl_amount", 0.0)) or 0.0)
        if pnl_r < 0.0:
            losses += 1
            continue
        break
    return losses


def _daily_green_streak(closed_trades: list[dict[str, Any]]) -> int:
    pnl_by_day: dict[str, float] = {}
    for trade in closed_trades:
        closed_at = _parse_iso_utc(trade.get("closed_at"))
        if closed_at is None:
            closed_at = _parse_iso_utc(trade.get("opened_at"))
        if closed_at is None:
            continue
        day_key = str(trading_day_key_for_timestamp(closed_at, tz=SYDNEY))
        pnl_by_day[day_key] = float(pnl_by_day.get(day_key, 0.0)) + float(
            trade.get("pnl_amount", trade.get("pnl_r", 0.0)) or 0.0
        )
    streak = 0
    for day_key in sorted(pnl_by_day.keys(), reverse=True):
        if float(pnl_by_day.get(day_key, 0.0)) > 0.0:
            streak += 1
            continue
        break
    return streak


def _equity_momentum_throttle(
    closed_trades: list[dict[str, Any]],
    orchestrator_config: dict[str, Any] | None = None,
) -> dict[str, float | str]:
    config = dict(orchestrator_config or {})
    sample = list(closed_trades[-50:])
    if not sample:
        return {
            "mode": "NEUTRAL",
            "expectancy_r": 0.0,
            "b_tier_adjust_pct": 0.0,
            "a_plus_size_boost": 0.0,
            "super_aggro_score_boost": 0.0,
            "super_aggro_size_boost": 0.0,
            "trajectory_catchup_pressure": 0.0,
        }
    expectancy_r = sum(float(item.get("r_multiple", 0.0) or 0.0) for item in sample) / max(len(sample), 1)
    hot_threshold = float(config.get("equity_momentum_hot_expectancy_r", 1.0))
    cold_threshold = float(config.get("equity_momentum_cold_expectancy_r", 0.6))
    if expectancy_r > hot_threshold:
        return {
            "mode": "HOT",
            "expectancy_r": float(expectancy_r),
            "b_tier_adjust_pct": float(config.get("equity_momentum_hot_b_tier_loosen_pct", 0.15) or 0.15),
            "a_plus_size_boost": float(config.get("equity_momentum_hot_a_plus_size_boost", 0.10) or 0.10),
            "super_aggro_score_boost": float(config.get("equity_momentum_hot_super_aggro_score_boost", 0.05) or 0.05),
            "super_aggro_size_boost": float(config.get("equity_momentum_hot_super_aggro_size_boost", 0.12) or 0.12),
            "trajectory_catchup_pressure": 0.0,
        }
    if expectancy_r < cold_threshold:
        return {
            "mode": "COLD",
            "expectancy_r": float(expectancy_r),
            "b_tier_adjust_pct": -abs(float(config.get("equity_momentum_cold_b_tier_tighten_pct", 0.10) or 0.10)),
            "a_plus_size_boost": 0.0,
            "super_aggro_score_boost": 0.0,
            "super_aggro_size_boost": 0.0,
            "trajectory_catchup_pressure": 0.0,
        }
    return {
        "mode": "NEUTRAL",
        "expectancy_r": float(expectancy_r),
        "b_tier_adjust_pct": 0.0,
        "a_plus_size_boost": 0.0,
        "super_aggro_score_boost": float(config.get("equity_momentum_neutral_super_aggro_score_boost", 0.02) or 0.02),
        "super_aggro_size_boost": float(config.get("equity_momentum_neutral_super_aggro_size_boost", 0.04) or 0.04),
        "trajectory_catchup_pressure": (
            float(config.get("equity_momentum_neutral_catchup_pressure", 0.08) or 0.08)
            if len(sample) >= 12 and expectancy_r >= max(0.10, cold_threshold)
            else 0.0
        ),
    }


def _velocity_decay_profile(
    *,
    symbol_key: str,
    strategy_key: str,
    setup_family: str,
    closed_trades: list[dict[str, Any]],
    row_timestamp: Any,
    trigger_trades_per_10_bars: float = 1.8,
    weekend_mode: bool = False,
    btc_weekend_trigger_trades_per_10_bars: float = 1.5,
    score_penalty: float = 0.15,
    base_multiplier: float = 0.85,
) -> dict[str, float]:
    if _normalize_symbol_key(symbol_key) != "BTCUSD":
        return {
            "trades_per_10_bars": 0.0,
            "score_penalty": 0.0,
            "size_multiplier": 1.0,
        }
    sample = list(closed_trades[-30:])
    family_key = normalize_strategy_family(str(setup_family or strategy_key))
    if len(sample) < 10:
        return {
            "trades_per_10_bars": 0.0,
            "score_penalty": 0.0,
            "size_multiplier": 1.0,
        }
    try:
        current_ts = pd.Timestamp(row_timestamp)
        current_ts = current_ts.tz_localize("UTC") if current_ts.tzinfo is None else current_ts.tz_convert("UTC")
    except Exception:
        return {
            "trades_per_10_bars": 0.0,
            "score_penalty": 0.0,
            "size_multiplier": 1.0,
        }
    family_trades = [trade for trade in sample if _trade_setup_family(trade) == family_key]
    if len(family_trades) < 3:
        return {
            "trades_per_10_bars": 0.0,
            "score_penalty": 0.0,
            "size_multiplier": 1.0,
        }
    timestamps: list[pd.Timestamp] = []
    for trade in family_trades:
        for key in ("closed_at", "opened_at", "timestamp_utc", "created_at"):
            raw = trade.get(key)
            if not raw:
                continue
            try:
                parsed = pd.Timestamp(raw)
                parsed = parsed.tz_localize("UTC") if parsed.tzinfo is None else parsed.tz_convert("UTC")
                timestamps.append(parsed)
                break
            except Exception:
                continue
    if len(timestamps) < 2:
        return {
            "trades_per_10_bars": 0.0,
            "score_penalty": 0.0,
            "size_multiplier": 1.0,
        }
    newest = max(max(timestamps), current_ts)
    oldest = min(timestamps)
    span_minutes = max((newest - oldest).total_seconds() / 60.0, 15.0)
    bars = max(span_minutes / 15.0, 1.0)
    trades_per_10_bars = (float(len(family_trades)) * 10.0) / bars
    trigger = max(
        0.1,
        float(btc_weekend_trigger_trades_per_10_bars if weekend_mode else trigger_trades_per_10_bars),
    )
    if trades_per_10_bars <= trigger:
        return {
            "trades_per_10_bars": float(trades_per_10_bars),
            "score_penalty": 0.0,
            "size_multiplier": 1.0,
        }
    excess = max(0.0, float(trades_per_10_bars) - trigger)
    return {
        "trades_per_10_bars": float(trades_per_10_bars),
        "score_penalty": max(0.0, float(score_penalty)),
        "size_multiplier": clamp(max(0.01, float(base_multiplier)) ** excess, 0.50, 1.0),
    }


def _streak_adjustment_mode(
    *,
    closed_trades: list[dict[str, Any]],
    symbol_key: str,
    strategy_key: str,
    orchestrator_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = dict(orchestrator_config or {})
    symbol_norm = _normalize_symbol_key(symbol_key)
    strategy_norm = str(strategy_key or "").strip().upper()
    recent = [
        trade
        for trade in reversed(list(closed_trades))
        if _normalize_symbol_key(str(trade.get("symbol") or "")) == symbol_norm
    ]
    if strategy_norm:
        strategy_recent = [
            trade
            for trade in recent
            if str(_trade_strategy_key(trade) or "").strip().upper() == strategy_norm
        ]
        if len(strategy_recent) >= 2:
            recent = strategy_recent
    if not recent:
        return {
            "mode": "NEUTRAL",
            "wins": 0,
            "losses": 0,
            "manager_reason": "streak_neutral",
        }
    win_threshold = max(1, int(config.get("streak_win_rr_threshold", 2)))
    loss_threshold = max(1, int(config.get("streak_loss_rr_threshold", 2)))
    if strategy_norm == "XAUUSD_ADAPTIVE_M5_GRID":
        win_threshold = max(1, int(config.get("xau_grid_win_streak_threshold", 3)))
        loss_threshold = max(1, int(config.get("xau_grid_loss_streak_threshold", 4)))
    wins = 0
    losses = 0
    for trade in recent:
        pnl_r = _trade_pnl_r_value(trade)
        if pnl_r >= 0.0:
            if losses > 0:
                break
            wins += 1
            continue
        if pnl_r < 0.0:
            if wins > 0:
                break
            losses += 1
            continue
        break
    if wins >= win_threshold:
        return {
            "mode": "WIN_STREAK",
            "wins": int(wins),
            "losses": 0,
            "manager_reason": f"streak_hold_bias:{wins}",
        }
    if losses >= loss_threshold:
        return {
            "mode": "LOSS_STREAK",
            "wins": 0,
            "losses": int(losses),
            "manager_reason": f"streak_protect_bias:{losses}",
        }
    return {
        "mode": "NEUTRAL",
        "wins": int(wins),
        "losses": int(losses),
        "manager_reason": "streak_neutral",
    }


def _family_rotation_penalty(
    *,
    strategy_key: str,
    setup_family: str,
    closed_trades: list[dict[str, Any]],
    candidate_tier_config: dict[str, Any] | None = None,
) -> float:
    config = dict(candidate_tier_config or {})
    window = max(1, int(config.get("family_rotation_window_trades", 20)))
    threshold = clamp(float(config.get("family_rotation_share_threshold", 0.60)), 0.05, 1.0)
    penalty = clamp(float(config.get("family_rotation_score_penalty", 0.20)), 0.0, 0.50)
    family_key = normalize_strategy_family(str(setup_family or strategy_key))
    if str(strategy_key or "").upper() == "XAUUSD_ADAPTIVE_M5_GRID":
        return 0.0
    recent = list(closed_trades[-window:])
    if len(recent) < min(5, window):
        return 0.0
    matching = sum(1 for trade in recent if _trade_setup_family(trade) == family_key)
    share = float(matching) / max(len(recent), 1)
    return float(penalty if share > threshold else 0.0)


def _append_candidate_verification_log(path_value: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path_value)
    ensure_parent(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _trade_mae_r(trade: dict[str, Any]) -> float:
    review = _trade_review_payload(trade)
    management = _trade_management_payload(trade)
    return _safe_float(
        review.get("mae_r", management.get("mae_r", trade.get("mae_r"))),
        0.0,
    )


def _trade_mfe_r(trade: dict[str, Any]) -> float:
    review = _trade_review_payload(trade)
    management = _trade_management_payload(trade)
    return _safe_float(
        review.get("mfe_r", management.get("mfe_r", trade.get("mfe_r"))),
        0.0,
    )


def _trade_duration_minutes(trade: dict[str, Any]) -> float:
    return _safe_float(trade.get("duration_minutes"), 0.0)


def _trade_actual_rr_achieved(trade: dict[str, Any]) -> float:
    review = _trade_review_payload(trade)
    return _safe_float(
        review.get("actual_rr_achieved", trade.get("r_multiple")),
        _safe_float(trade.get("r_multiple"), 0.0),
    )


def _contains_any(value: str, tokens: tuple[str, ...]) -> bool:
    lowered = str(value or "").lower()
    return any(token in lowered for token in tokens)


def _pair_session_metrics(trades: list[dict[str, Any]]) -> dict[str, float]:
    if not trades:
        return {
            "trades": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "win_rate": 0.5,
            "expectancy_r": 0.0,
            "profit_factor": 1.0,
            "false_break_rate": 0.0,
            "management_quality": 0.5,
        }
    wins = 0
    losses = 0
    r_values: list[float] = []
    profit_total = 0.0
    loss_total = 0.0
    breakout_losses = 0.0
    for trade in trades:
        pnl_amount = float(trade.get("pnl_amount") or 0.0)
        pnl_r = _trade_pnl_r_value(trade)
        r_values.append(pnl_r)
        if pnl_amount >= 0.0:
            wins += 1
            profit_total += max(0.0, pnl_amount)
        else:
            losses += 1
            loss_total += abs(pnl_amount)
            breakout_losses += _breakout_loss_flag(str(trade.get("setup") or ""), pnl_amount)
    trade_count = max(1, len(trades))
    profit_factor = profit_total / loss_total if loss_total > 1e-9 else (profit_total if profit_total > 0.0 else 1.0)
    expectancy_r = sum(r_values) / trade_count
    win_rate = wins / trade_count
    false_break_rate = breakout_losses / max(1.0, float(losses))
    avg_win_r = sum(value for value in r_values if value > 0.0) / max(1, wins)
    avg_loss_r = abs(sum(value for value in r_values if value < 0.0)) / max(1, losses)
    management_quality = clamp(
        0.50 + ((avg_win_r - avg_loss_r) * 0.18) + ((win_rate - 0.50) * 0.30),
        0.10,
        0.95,
    )
    return {
        "trades": float(len(trades)),
        "wins": float(wins),
        "losses": float(losses),
        "win_rate": float(win_rate),
        "expectancy_r": float(expectancy_r),
        "profit_factor": float(profit_factor),
        "false_break_rate": float(false_break_rate),
        "management_quality": float(management_quality),
    }


def _pair_strategy_session_metrics(trades: list[dict[str, Any]]) -> dict[str, float]:
    if not trades:
        return {
            "trades": 0.0,
            "wins": 0.0,
            "losses": 0.0,
            "win_rate": 0.5,
            "expectancy_r": 0.0,
            "profit_factor": 1.0,
            "false_break_rate": 0.0,
            "late_entry_rate": 0.0,
            "fast_failure_rate": 0.0,
            "immediate_invalidation_rate": 0.0,
            "poor_structure_rate": 0.0,
            "giveback_rate": 0.0,
            "avg_mae_r": 0.0,
            "avg_mfe_r": 0.0,
            "capture_efficiency": 0.5,
            "avg_win_r": 0.0,
            "avg_loss_r": 0.0,
            "realized_rr": 0.0,
            "over_scratch_rate": 0.0,
            "premature_protection_rate": 0.0,
            "stalled_winner_rate": 0.0,
            "management_quality": 0.5,
        }

    wins = 0
    losses = 0
    r_values: list[float] = []
    profit_total = 0.0
    loss_total = 0.0
    breakout_losses = 0.0
    late_entries = 0.0
    fast_failures = 0.0
    immediate_invalidations = 0.0
    poor_structures = 0.0
    givebacks = 0.0
    scratches = 0.0
    premature_protections = 0.0
    stalled_winners = 0.0
    mae_values: list[float] = []
    mfe_values: list[float] = []
    capture_values: list[float] = []

    for trade in trades:
        pnl_amount = _safe_float(trade.get("pnl_amount"), 0.0)
        pnl_r = _trade_pnl_r_value(trade)
        r_values.append(pnl_r)
        review_issue = _trade_review_issue(trade)
        management_reason = _trade_management_reason(trade)
        setup_name = str(trade.get("setup") or "")
        mfe_r = _trade_mfe_r(trade)
        mae_r = _trade_mae_r(trade)
        duration_minutes = _trade_duration_minutes(trade)
        mae_values.append(mae_r)
        mfe_values.append(mfe_r)
        if mfe_r > 0.0:
            capture_values.append(clamp(max(pnl_r, 0.0) / max(mfe_r, 1e-6), 0.0, 1.5))
        if abs(pnl_r) <= 0.10:
            scratches += 1.0

        if pnl_amount >= 0.0:
            wins += 1
            profit_total += max(0.0, pnl_amount)
            if (
                mfe_r >= 0.80
                and pnl_r <= 0.45
                and _contains_any(management_reason, ("profit_lock", "trail", "runner_trailing_update", "partial_profit_protect"))
            ):
                premature_protections += 1.0
            if mfe_r >= 1.00 and pnl_r <= 0.50:
                stalled_winners += 1.0
        else:
            losses += 1
            loss_total += abs(pnl_amount)
            breakout_losses += _breakout_loss_flag(setup_name, pnl_amount)
            if _contains_any(review_issue, ("late_entry", "no_followthrough", "chased_extension", "late breakout")):
                late_entries += 1.0
            if _contains_any(review_issue, ("false_break", "weak_retest", "failed breakout")):
                breakout_losses += 0.5
            if _contains_any(review_issue, ("poor_structure", "weak_retest", "poor_pullback", "messy")):
                poor_structures += 1.0
            if mfe_r <= 0.12:
                immediate_invalidations += 1.0
            if mfe_r <= 0.25 and duration_minutes <= 120.0:
                fast_failures += 1.0

        if (
            mfe_r >= 0.60
            and pnl_r <= 0.10
            and _contains_any(management_reason, ("profit_lock", "trail", "runner_trailing_update"))
        ):
            givebacks += 1.0

    trade_count = max(1, len(trades))
    loss_count = max(1, losses)
    profit_factor = profit_total / loss_total if loss_total > 1e-9 else (profit_total if profit_total > 0.0 else 1.0)
    expectancy_r = sum(r_values) / trade_count
    win_rate = wins / trade_count
    false_break_rate = breakout_losses / max(1.0, float(losses))
    late_entry_rate = late_entries / loss_count
    fast_failure_rate = fast_failures / loss_count
    immediate_invalidation_rate = immediate_invalidations / loss_count
    poor_structure_rate = poor_structures / loss_count
    giveback_rate = givebacks / max(1.0, float(trade_count))
    avg_mae_r = sum(mae_values) / trade_count if mae_values else 0.0
    avg_mfe_r = sum(mfe_values) / trade_count if mfe_values else 0.0
    capture_efficiency = sum(capture_values) / len(capture_values) if capture_values else 0.5
    avg_win_r = sum(value for value in r_values if value > 0.0) / max(1, wins)
    avg_loss_r = abs(sum(value for value in r_values if value < 0.0)) / max(1, losses)
    realized_rr = avg_win_r / avg_loss_r if avg_loss_r > 1e-6 else (avg_win_r if avg_win_r > 0.0 else 0.0)
    over_scratch_rate = scratches / max(1.0, float(trade_count))
    premature_protection_rate = premature_protections / max(1.0, float(wins))
    stalled_winner_rate = stalled_winners / max(1.0, float(wins))
    management_quality = clamp(
        0.48
        + ((win_rate - 0.50) * 0.28)
        + (clamp(expectancy_r, -0.25, 0.40) * 0.26)
        + ((capture_efficiency - 0.45) * 0.25)
        + ((clamp(realized_rr, 0.0, 1.5) - 0.70) * 0.16)
        - (giveback_rate * 0.18),
        0.05,
        0.95,
    )
    return {
        "trades": float(len(trades)),
        "wins": float(wins),
        "losses": float(losses),
        "win_rate": float(win_rate),
        "expectancy_r": float(expectancy_r),
        "profit_factor": float(profit_factor),
        "false_break_rate": float(clamp(false_break_rate, 0.0, 1.0)),
        "late_entry_rate": float(clamp(late_entry_rate, 0.0, 1.0)),
        "fast_failure_rate": float(clamp(fast_failure_rate, 0.0, 1.0)),
        "immediate_invalidation_rate": float(clamp(immediate_invalidation_rate, 0.0, 1.0)),
        "poor_structure_rate": float(clamp(poor_structure_rate, 0.0, 1.0)),
        "giveback_rate": float(clamp(giveback_rate, 0.0, 1.0)),
        "avg_mae_r": float(avg_mae_r),
        "avg_mfe_r": float(avg_mfe_r),
        "capture_efficiency": float(clamp(capture_efficiency, 0.0, 1.0)),
        "avg_win_r": float(avg_win_r),
        "avg_loss_r": float(avg_loss_r),
        "realized_rr": float(realized_rr),
        "over_scratch_rate": float(clamp(over_scratch_rate, 0.0, 1.0)),
        "premature_protection_rate": float(clamp(premature_protection_rate, 0.0, 1.0)),
        "stalled_winner_rate": float(clamp(stalled_winner_rate, 0.0, 1.0)),
        "management_quality": float(management_quality),
    }


def _pair_session_performance_state(
    *,
    symbol: str,
    session_name: str,
    session_native_pair: bool,
    closed_trades: list[dict[str, Any]],
    current_day_key: str,
) -> dict[str, Any]:
    symbol_key = _normalize_symbol_key(symbol)
    session_key = str(session_name or "").upper()
    symbol_trades = [trade for trade in closed_trades if _normalize_symbol_key(str(trade.get("symbol") or "")) == symbol_key][:30]
    pair_session_trades = [
        trade
        for trade in symbol_trades
        if str(trade.get("session_name") or "").upper() == session_key
    ][:12]
    today_session_trades = [
        trade
        for trade in pair_session_trades
        if trading_day_key_for_timestamp(trade.get("closed_at")) == current_day_key
    ]
    pair_metrics = _pair_session_metrics(symbol_trades)
    session_metrics = _pair_session_metrics(pair_session_trades if pair_session_trades else symbol_trades[:12])
    today_metrics = _pair_session_metrics(today_session_trades)

    state = "NORMAL"
    reason = "neutral_pair_flow"
    multiplier = 1.0
    why_promoted = ""
    why_throttled = ""

    session_trades = int(session_metrics["trades"])
    if session_trades >= 5 and (
        session_metrics["expectancy_r"] <= -0.18
        or session_metrics["profit_factor"] < 0.70
        or (session_metrics["win_rate"] < 0.38 and session_metrics["false_break_rate"] >= 0.60)
    ):
        state = "QUARANTINED"
        reason = "weak_pair_session_edge"
        multiplier = 0.74
        why_throttled = "weak_pair_session_edge"
    elif session_trades >= 4 and (
        session_metrics["expectancy_r"] < -0.02
        or session_metrics["profit_factor"] < 0.95
        or session_metrics["win_rate"] < 0.46
    ):
        state = "REDUCED"
        reason = "session_edge_soft_throttle"
        multiplier = 0.88
        why_throttled = "session_edge_soft_throttle"
    elif session_trades >= 4 and (
        session_metrics["expectancy_r"] >= 0.12
        and session_metrics["profit_factor"] >= 1.20
        and session_metrics["win_rate"] >= 0.55
    ):
        state = "ATTACK"
        reason = "session_pair_proving_edge"
        multiplier = 1.10
        why_promoted = "session_pair_proving_edge"

    if session_native_pair and int(today_metrics["wins"]) >= 2 and today_metrics["expectancy_r"] >= 0.0:
        state = "ATTACK"
        reason = "native_pair_proving_edge_today"
        multiplier = max(multiplier, 1.14)
        why_promoted = "native_pair_proving_edge_today"
        why_throttled = ""
    elif session_native_pair and int(today_metrics["losses"]) >= 2 and today_metrics["expectancy_r"] < 0.0:
        if state == "ATTACK":
            state = "NORMAL"
            reason = "native_pair_today_edge_cooling"
            multiplier = 1.0
        elif state == "NORMAL":
            state = "REDUCED"
            reason = "native_pair_today_edge_cooling"
            multiplier = min(multiplier, 0.92)
        why_promoted = ""
        why_throttled = "native_pair_today_edge_cooling"

    return {
        "pair_status": state,
        "pair_status_reason": reason,
        "pair_state_multiplier": float(clamp(multiplier, 0.70, 1.20)),
        "rolling_expectancy_by_pair": float(pair_metrics["expectancy_r"]),
        "rolling_pf_by_pair": float(pair_metrics["profit_factor"]),
        "rolling_expectancy_by_session": float(session_metrics["expectancy_r"]),
        "rolling_pf_by_session": float(session_metrics["profit_factor"]),
        "rolling_win_rate_by_pair": float(pair_metrics["win_rate"]),
        "rolling_win_rate_by_session": float(session_metrics["win_rate"]),
        "false_break_rate": float(session_metrics["false_break_rate"]),
        "management_quality_score": float(session_metrics["management_quality"]),
        "today_session_wins": int(today_metrics["wins"]),
        "today_session_losses": int(today_metrics["losses"]),
        "today_session_trades": int(today_metrics["trades"]),
        "why_pair_is_promoted": str(why_promoted),
        "why_pair_is_throttled": str(why_throttled),
    }


def _merge_strategy_health_states(*states: str) -> str:
    normalized = [str(state or "NORMAL").upper() for state in states if str(state or "").strip()]
    if "QUARANTINED" in normalized:
        return "QUARANTINED"
    if "REDUCED" in normalized:
        return "REDUCED"
    if "ATTACK" in normalized:
        return "ATTACK"
    return "NORMAL"


def _pair_strategy_session_performance_state(
    *,
    symbol: str,
    strategy_key: str,
    session_name: str,
    regime_state: str,
    session_native_pair: bool,
    closed_trades: list[dict[str, Any]],
    current_day_key: str,
) -> dict[str, Any]:
    symbol_key = _normalize_symbol_key(symbol)
    strategy_key_norm = str(strategy_key or "").strip().upper()
    session_key = str(session_name or "").strip().upper()
    regime_key = runtime_regime_state(regime_state)

    symbol_trades = [
        trade
        for trade in closed_trades
        if _normalize_symbol_key(str(trade.get("symbol") or "")) == symbol_key
    ][:80]
    strategy_trades = [
        trade
        for trade in symbol_trades
        if _trade_strategy_key(trade).strip().upper() == strategy_key_norm
    ][:36]
    strategy_session_trades = [
        trade
        for trade in strategy_trades
        if _trade_session_name(trade) == session_key
    ][:18]
    bucket_trades = [
        trade
        for trade in strategy_session_trades
        if _trade_regime_bucket(trade) == regime_key
    ][:12]
    sample_trades = (
        bucket_trades
        if len(bucket_trades) >= 3
        else strategy_session_trades
        if len(strategy_session_trades) >= 3
        else strategy_trades
    )
    today_sample_trades = [
        trade
        for trade in sample_trades
        if trading_day_key_for_timestamp(trade.get("closed_at")) == current_day_key
    ]

    strategy_metrics = _pair_strategy_session_metrics(strategy_trades[:18])
    session_metrics = _pair_strategy_session_metrics(strategy_session_trades if strategy_session_trades else strategy_trades[:12])
    bucket_metrics = _pair_strategy_session_metrics(sample_trades)
    today_metrics = _pair_strategy_session_metrics(today_sample_trades)
    avg_mae_r = float(bucket_metrics["avg_mae_r"])
    avg_mfe_r = float(bucket_metrics["avg_mfe_r"])
    capture_efficiency = float(bucket_metrics["capture_efficiency"])
    realized_rr = float(bucket_metrics["realized_rr"])
    premature_protection_rate = float(bucket_metrics["premature_protection_rate"])
    stalled_winner_rate = float(bucket_metrics["stalled_winner_rate"])
    giveback_rate = float(bucket_metrics["giveback_rate"])
    deep_mae_drag = avg_mae_r >= 0.68
    poor_mfe_drag = avg_mfe_r <= 0.28
    weak_capture_drag = capture_efficiency <= 0.34

    sample_size = int(bucket_metrics["trades"])
    state = "NORMAL"
    reason = "strategy_bucket_neutral"
    priority_multiplier = 1.0
    size_multiplier = 1.0
    why_promoted = ""
    why_throttled = ""
    strategy_tokens = strategy_key_norm
    targeted_ranging_bucket = (
        regime_key == "RANGING"
        and (
            (symbol_key in {"EURJPY", "GBPJPY"} and any(
                token in strategy_tokens
                for token in ("BREAKOUT", "IMPULSE", "CONTINUATION", "TREND", "RETEST", "MOMENTUM", "EXPANSION")
            ))
            or (symbol_key in {"AUDJPY", "NZDJPY", "USDJPY", "EURUSD", "GBPUSD"} and any(
                token in strategy_tokens
                for token in ("BREAKOUT", "IMPULSE", "CONTINUATION", "TREND", "RETEST", "MOMENTUM", "EXPANSION")
            ))
            or (symbol_key == "EURUSD" and "RANGE_FADE" in strategy_tokens)
            or (symbol_key == "BTCUSD" and any(
                token in strategy_tokens
                for token in ("TREND", "SCALP", "CONTINUATION", "IMPULSE")
            ))
        )
    )
    targeted_rotation_bucket = (
        symbol_key == "AUDNZD"
        and regime_key == "RANGING"
        and any(token in strategy_tokens for token in ("ROTATION", "REVERSION", "RANGE", "VWAP"))
    )
    targeted_exact_bucket = (
        (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_MOMENTUM_IMPULSE" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_MOMENTUM_IMPULSE" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_MOMENTUM_IMPULSE" and session_key == "SYDNEY" and regime_key == "RANGING")
        or (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_MOMENTUM_IMPULSE" and session_key == "LONDON" and regime_key == "RANGING")
        or (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_MACRO_TREND_RIDE" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_MACRO_TREND_RIDE" and session_key == "SYDNEY" and regime_key == "RANGING")
        or (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_MACRO_TREND_RIDE" and session_key == "SYDNEY" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_MACRO_TREND_RIDE" and session_key == "SYDNEY" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_MACRO_TREND_RIDE" and session_key == "OVERLAP" and regime_key == "RANGING")
        or (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_LIQUIDITY_SWEEP_REVERSAL" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_VWAP_TREND_CONTINUATION" and session_key == "LONDON" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_VWAP_TREND_CONTINUATION" and session_key == "OVERLAP" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "USDJPY" and strategy_key_norm == "USDJPY_VWAP_TREND_CONTINUATION" and session_key == "OVERLAP" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "EURJPY" and strategy_key_norm == "EURJPY_MOMENTUM_IMPULSE" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "EURJPY" and strategy_key_norm == "EURJPY_MOMENTUM_IMPULSE" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "EURJPY" and strategy_key_norm == "EURJPY_MOMENTUM_IMPULSE" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "EURJPY" and strategy_key_norm == "EURJPY_SESSION_PULLBACK_CONTINUATION" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "EURJPY" and strategy_key_norm == "EURJPY_SESSION_PULLBACK_CONTINUATION" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "EURJPY" and strategy_key_norm == "EURJPY_SESSION_PULLBACK_CONTINUATION" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "GBPJPY" and strategy_key_norm == "GBPJPY_MOMENTUM_IMPULSE" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "GBPJPY" and strategy_key_norm == "GBPJPY_MOMENTUM_IMPULSE" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "GBPJPY" and strategy_key_norm == "GBPJPY_MOMENTUM_IMPULSE" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "GBPJPY" and strategy_key_norm == "GBPJPY_SESSION_PULLBACK_CONTINUATION" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "GBPJPY" and strategy_key_norm == "GBPJPY_SESSION_PULLBACK_CONTINUATION" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "GBPJPY" and strategy_key_norm == "GBPJPY_SESSION_PULLBACK_CONTINUATION" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "AUDJPY" and strategy_key_norm == "AUDJPY_TOKYO_CONTINUATION_PULLBACK" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "AUDJPY" and strategy_key_norm == "AUDJPY_TOKYO_CONTINUATION_PULLBACK" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "AUDJPY" and strategy_key_norm == "AUDJPY_TOKYO_CONTINUATION_PULLBACK" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "AUDJPY" and strategy_key_norm == "AUDJPY_TOKYO_MOMENTUM_BREAKOUT" and session_key == "SYDNEY" and regime_key == "RANGING")
        or (symbol_key == "AUDJPY" and strategy_key_norm == "AUDJPY_TOKYO_MOMENTUM_BREAKOUT" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "AUDJPY" and strategy_key_norm == "AUDJPY_ATR_COMPRESSION_BREAKOUT" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "AUDJPY" and strategy_key_norm == "AUDJPY_ATR_COMPRESSION_BREAKOUT" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "EURUSD" and strategy_key_norm == "EURUSD_RANGE_FADE" and regime_key == "RANGING")
        or (symbol_key == "EURUSD" and strategy_key_norm == "EURUSD_LONDON_BREAKOUT" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "EURUSD" and strategy_key_norm == "EURUSD_LONDON_BREAKOUT" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "EURUSD" and strategy_key_norm == "EURUSD_LONDON_BREAKOUT" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "EURUSD" and strategy_key_norm == "EURUSD_LIQUIDITY_SWEEP" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "EURUSD" and strategy_key_norm == "EURUSD_LIQUIDITY_SWEEP" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "EURUSD" and strategy_key_norm == "EURUSD_LIQUIDITY_SWEEP" and session_key == "LONDON" and regime_key == "RANGING")
        or (symbol_key == "EURUSD" and strategy_key_norm == "EURUSD_LIQUIDITY_SWEEP" and session_key == "LONDON" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "EURUSD" and strategy_key_norm == "EURUSD_VWAP_PULLBACK" and session_key == "LONDON" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "EURUSD" and strategy_key_norm == "EURUSD_VWAP_PULLBACK" and session_key == "LONDON" and regime_key == "RANGING")
        or (symbol_key == "EURUSD" and strategy_key_norm == "EURUSD_VWAP_PULLBACK" and session_key == "LONDON" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "GBPUSD" and strategy_key_norm == "GBPUSD_TREND_PULLBACK_RIDE" and session_key == "LONDON" and regime_key == "RANGING")
        or (symbol_key == "GBPUSD" and strategy_key_norm == "GBPUSD_TREND_PULLBACK_RIDE" and session_key == "LONDON" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "GBPUSD" and strategy_key_norm == "GBPUSD_TREND_PULLBACK_RIDE" and session_key == "LONDON" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "GBPUSD" and strategy_key_norm == "GBPUSD_TREND_PULLBACK_RIDE" and session_key == "OVERLAP" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "NZDJPY" and strategy_key_norm == "NZDJPY_SESSION_RANGE_EXPANSION" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "NZDJPY" and strategy_key_norm == "NZDJPY_SESSION_RANGE_EXPANSION" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "NZDJPY" and strategy_key_norm == "NZDJPY_SESSION_RANGE_EXPANSION" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "NZDJPY" and strategy_key_norm == "NZDJPY_PULLBACK_CONTINUATION" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "NZDJPY" and strategy_key_norm == "NZDJPY_PULLBACK_CONTINUATION" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (
            symbol_key == "BTCUSD"
            and strategy_key_norm in {"BTCUSD_PRICE_ACTION_CONTINUATION", "BTCUSD_TREND_SCALP", "BTCUSD_RANGE_EXPANSION"}
            and regime_key == "MEAN_REVERSION"
        )
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_TREND_SCALP" and session_key == "LONDON" and regime_key == "TRENDING")
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_TREND_SCALP" and session_key == "TOKYO" and regime_key == "TRENDING")
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_TREND_SCALP" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_TREND_SCALP" and session_key == "SYDNEY" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_VOLATILE_RETEST" and session_key == "SYDNEY" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_VOLATILE_RETEST" and session_key == "SYDNEY" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_PRICE_ACTION_CONTINUATION" and session_key == "TOKYO" and regime_key == "TRENDING")
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_PRICE_ACTION_CONTINUATION" and session_key == "OVERLAP" and regime_key == "TRENDING")
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_PRICE_ACTION_CONTINUATION" and session_key == "NEW_YORK" and regime_key == "TRENDING")
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_VOLATILE_RETEST" and session_key == "TOKYO" and regime_key == "TRENDING")
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_RANGE_EXPANSION" and regime_key == "RANGING")
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_RANGE_EXPANSION" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "BTCUSD" and strategy_key_norm == "BTCUSD_RANGE_EXPANSION" and session_key == "OFF" and regime_key == "RANGING")
        or (symbol_key == "GBPUSD" and strategy_key_norm == "GBPUSD_LONDON_EXPANSION_BREAKOUT" and session_key == "LONDON" and regime_key == "RANGING")
        or (symbol_key == "GBPUSD" and strategy_key_norm == "GBPUSD_LONDON_EXPANSION_BREAKOUT" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "AUDNZD" and strategy_key_norm == "AUDNZD_VWAP_MEAN_REVERSION" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "AUDNZD" and strategy_key_norm == "AUDNZD_VWAP_MEAN_REVERSION" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "AUDNZD" and strategy_key_norm == "AUDNZD_STRUCTURE_BREAK_RETEST" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "AUDNZD" and strategy_key_norm == "AUDNZD_STRUCTURE_BREAK_RETEST" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "XAUUSD" and strategy_key_norm == "XAUUSD_ATR_EXPANSION_SCALPER" and session_key in {"SYDNEY", "TOKYO"} and regime_key == "RANGING")
        or (symbol_key == "XAUUSD" and strategy_key_norm == "XAUUSD_ATR_EXPANSION_SCALPER" and session_key == "LONDON" and regime_key in {"RANGING", "LOW_LIQUIDITY_CHOP", "TRENDING", "MEAN_REVERSION"})
        or (symbol_key == "XAUUSD" and strategy_key_norm == "XAUUSD_LONDON_LIQUIDITY_SWEEP" and ((session_key == "LONDON" and regime_key in {"RANGING", "LOW_LIQUIDITY_CHOP"}) or (session_key == "OVERLAP" and regime_key == "MEAN_REVERSION")))
        or (symbol_key == "XAUUSD" and strategy_key_norm == "XAUUSD_NY_MOMENTUM_BREAKOUT" and session_key == "LONDON" and regime_key in {"RANGING", "LOW_LIQUIDITY_CHOP", "TRENDING", "MEAN_REVERSION"})
        or (symbol_key == "NAS100" and strategy_key_norm == "NAS100_OPENING_DRIVE_BREAKOUT" and session_key == "LONDON" and regime_key == "RANGING")
        or (symbol_key == "NAS100" and strategy_key_norm == "NAS100_OPENING_DRIVE_BREAKOUT" and session_key == "NEW_YORK" and regime_key == "RANGING")
        or (symbol_key == "NAS100" and strategy_key_norm == "NAS100_LIQUIDITY_SWEEP_REVERSAL" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
        or (symbol_key == "NAS100" and strategy_key_norm == "NAS100_MOMENTUM_IMPULSE" and session_key == "TOKYO" and regime_key == "TRENDING")
        or (symbol_key == "USOIL" and strategy_key_norm == "USOIL_INVENTORY_MOMENTUM" and session_key == "LONDON" and regime_key == "RANGING")
        or (symbol_key == "USOIL" and strategy_key_norm == "USOIL_LONDON_TREND_EXPANSION" and session_key == "LONDON" and regime_key == "RANGING")
        or (symbol_key == "USOIL" and strategy_key_norm == "USOIL_LONDON_TREND_EXPANSION" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "USOIL" and strategy_key_norm == "USOIL_INVENTORY_MOMENTUM" and session_key == "TOKYO" and regime_key == "RANGING")
        or (symbol_key == "USOIL" and strategy_key_norm == "USOIL_INVENTORY_MOMENTUM" and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (symbol_key == "USOIL" and strategy_key_norm == "USOIL_INVENTORY_MOMENTUM" and session_key == "TOKYO" and regime_key == "MEAN_REVERSION")
    )
    asia_priority_exact_bucket = (
        symbol_key in {"AUDJPY", "NZDJPY"}
        and session_key in {"SYDNEY", "TOKYO"}
        and targeted_exact_bucket
    )
    london_priority_exact_bucket = (
        (
            symbol_key in {"USDJPY", "NAS100"}
            or (
                symbol_key == "GBPUSD"
                and strategy_key_norm == "GBPUSD_LONDON_EXPANSION_BREAKOUT"
            )
        )
        and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
        and targeted_exact_bucket
    )

    if targeted_exact_bucket and not (asia_priority_exact_bucket or london_priority_exact_bucket) and sample_size >= 2 and (
        bucket_metrics["expectancy_r"] <= -0.12
        or bucket_metrics["profit_factor"] < 0.82
        or bucket_metrics["late_entry_rate"] >= 0.26
        or bucket_metrics["false_break_rate"] >= 0.28
        or bucket_metrics["immediate_invalidation_rate"] >= 0.28
        or (deep_mae_drag and poor_mfe_drag)
        or (weak_capture_drag and giveback_rate >= 0.18)
    ):
        state = "QUARANTINED"
        reason = "targeted_exact_bucket_demoted"
        priority_multiplier = 0.74
        size_multiplier = 0.80
        why_throttled = "targeted_exact_bucket_demoted"
    elif targeted_exact_bucket and (asia_priority_exact_bucket or london_priority_exact_bucket) and sample_size >= 2 and (
        bucket_metrics["expectancy_r"] <= -0.22
        or bucket_metrics["profit_factor"] < 0.74
        or bucket_metrics["late_entry_rate"] >= 0.34
        or bucket_metrics["false_break_rate"] >= 0.36
        or bucket_metrics["immediate_invalidation_rate"] >= 0.36
        or (deep_mae_drag and poor_mfe_drag and weak_capture_drag)
        or (weak_capture_drag and giveback_rate >= 0.22)
    ):
        state = "REDUCED"
        reason = "targeted_growth_bucket_soft_protect"
        priority_multiplier = 0.86
        size_multiplier = 0.88
        why_throttled = "targeted_growth_bucket_soft_protect"
    elif targeted_ranging_bucket and sample_size >= 3 and (
        bucket_metrics["expectancy_r"] <= -0.16
        or bucket_metrics["profit_factor"] < 0.80
        or bucket_metrics["late_entry_rate"] >= 0.30
        or bucket_metrics["false_break_rate"] >= 0.40
        or bucket_metrics["immediate_invalidation_rate"] >= 0.40
        or (avg_mae_r >= 0.72 and avg_mfe_r <= 0.30)
        or (capture_efficiency <= 0.28 and giveback_rate >= 0.18)
    ):
        state = "QUARANTINED"
        reason = "targeted_ranging_bucket_fast_failures"
        priority_multiplier = 0.76
        size_multiplier = 0.80
        why_throttled = "targeted_ranging_bucket_fast_failures"
    elif sample_size >= 4 and (
        bucket_metrics["expectancy_r"] <= -0.18
        or bucket_metrics["profit_factor"] < 0.72
        or (
            bucket_metrics["late_entry_rate"] >= 0.45
            and bucket_metrics["fast_failure_rate"] >= 0.40
        )
        or bucket_metrics["immediate_invalidation_rate"] >= 0.55
    ):
        state = "QUARANTINED"
        reason = "strategy_bucket_repeating_fast_failures"
        priority_multiplier = 0.78
        size_multiplier = 0.82
        why_throttled = "strategy_bucket_repeating_fast_failures"
    elif targeted_ranging_bucket and sample_size >= 2 and (
        bucket_metrics["expectancy_r"] < -0.04
        or bucket_metrics["profit_factor"] < 0.92
        or bucket_metrics["late_entry_rate"] >= 0.20
        or bucket_metrics["false_break_rate"] >= 0.25
        or bucket_metrics["poor_structure_rate"] >= 0.30
        or (avg_mae_r >= 0.62 and avg_mfe_r <= 0.34)
        or capture_efficiency <= 0.36
        or (realized_rr <= 0.55 and premature_protection_rate >= 0.16)
    ):
        state = "REDUCED"
        reason = "targeted_ranging_bucket_soft_throttle"
        priority_multiplier = 0.88
        size_multiplier = 0.90
        why_throttled = "targeted_ranging_bucket_soft_throttle"
    elif sample_size >= 3 and (
        bucket_metrics["expectancy_r"] < -0.05
        or bucket_metrics["profit_factor"] < 0.95
        or bucket_metrics["false_break_rate"] >= 0.50
        or bucket_metrics["late_entry_rate"] >= 0.35
        or bucket_metrics["poor_structure_rate"] >= 0.45
    ):
        state = "REDUCED"
        reason = "strategy_bucket_soft_throttle"
        priority_multiplier = 0.90
        size_multiplier = 0.92
        why_throttled = "strategy_bucket_soft_throttle"
    elif sample_size >= 4 and (
        bucket_metrics["expectancy_r"] >= 0.08
        and bucket_metrics["profit_factor"] >= 1.15
        and bucket_metrics["win_rate"] >= 0.52
        and bucket_metrics["management_quality"] >= 0.52
        and bucket_metrics["capture_efficiency"] >= 0.45
        and realized_rr >= 0.72
        and premature_protection_rate <= 0.14
    ):
        state = "ATTACK"
        reason = "strategy_bucket_proving_edge"
        priority_multiplier = 1.08
        size_multiplier = 1.04
        why_promoted = "strategy_bucket_proving_edge"

    if (
        asia_priority_exact_bucket
        and sample_size >= 2
        and state != "QUARANTINED"
        and bucket_metrics["expectancy_r"] >= -0.04
        and bucket_metrics["profit_factor"] >= 0.90
        and capture_efficiency >= 0.10
        and immediate_invalidation_rate <= 0.28
    ):
        state = "NORMAL"
        reason = "asia_exact_bucket_preserved"
        priority_multiplier = max(priority_multiplier, 1.03)
        size_multiplier = max(size_multiplier, 1.00)
        if not why_promoted:
            why_promoted = "asia_exact_bucket_preserved"
        why_throttled = ""
    if (
        asia_priority_exact_bucket
        and sample_size >= 3
        and bucket_metrics["expectancy_r"] >= 0.06
        and bucket_metrics["profit_factor"] >= 1.05
        and capture_efficiency >= 0.16
        and premature_protection_rate <= 0.18
    ):
        state = "ATTACK"
        reason = "asia_exact_bucket_attack"
        priority_multiplier = max(priority_multiplier, 1.08)
        size_multiplier = max(size_multiplier, 1.03)
        if not why_promoted:
            why_promoted = "asia_exact_bucket_attack"
        why_throttled = ""
    if (
        london_priority_exact_bucket
        and sample_size >= 2
        and state != "QUARANTINED"
        and bucket_metrics["expectancy_r"] >= -0.03
        and bucket_metrics["profit_factor"] >= 0.94
        and capture_efficiency >= 0.12
        and immediate_invalidation_rate <= 0.30
    ):
        state = "NORMAL"
        reason = "london_exact_bucket_preserved"
        priority_multiplier = max(priority_multiplier, 1.03)
        size_multiplier = max(size_multiplier, 1.00)
        if not why_promoted:
            why_promoted = "london_exact_bucket_preserved"
        why_throttled = ""
    if (
        london_priority_exact_bucket
        and sample_size >= 3
        and bucket_metrics["expectancy_r"] >= 0.05
        and bucket_metrics["profit_factor"] >= 1.02
        and capture_efficiency >= 0.16
        and premature_protection_rate <= 0.20
    ):
        state = "ATTACK"
        reason = "london_exact_bucket_attack"
        priority_multiplier = max(priority_multiplier, 1.06)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "london_exact_bucket_attack"
        why_throttled = ""

    if session_native_pair and int(today_metrics["wins"]) >= 2 and today_metrics["expectancy_r"] >= 0.0:
        state = "ATTACK"
        if not why_promoted:
            reason = "native_strategy_bucket_proving_edge_today"
        priority_multiplier = max(priority_multiplier, 1.10)
        size_multiplier = max(size_multiplier, 1.04)
        if not why_promoted:
            why_promoted = "native_strategy_bucket_proving_edge_today"
        why_throttled = ""
    elif session_native_pair and int(today_metrics["losses"]) >= 2 and today_metrics["expectancy_r"] < 0.0:
        if state == "ATTACK":
            state = "NORMAL"
            reason = "native_strategy_bucket_cooling_today"
            priority_multiplier = 1.0
            size_multiplier = 1.0
        elif state == "NORMAL":
            state = "REDUCED"
            reason = "native_strategy_bucket_cooling_today"
            priority_multiplier = min(priority_multiplier, 0.94)
            size_multiplier = min(size_multiplier, 0.94)
        why_throttled = "native_strategy_bucket_cooling_today"
        why_promoted = ""

    if targeted_rotation_bucket and sample_size >= 3 and state != "QUARANTINED":
        if bucket_metrics["expectancy_r"] >= -0.02 and bucket_metrics["profit_factor"] >= 0.90:
            state = "NORMAL"
            reason = "audnzd_rotation_preserved"
            priority_multiplier = max(priority_multiplier, 1.0)
            size_multiplier = max(size_multiplier, 1.0)
            if not why_promoted:
                why_promoted = "audnzd_rotation_preserved"
            why_throttled = ""
    if (
        symbol_key == "AUDNZD"
        and strategy_key_norm == "AUDNZD_RANGE_ROTATION"
        and session_key in {"SYDNEY", "TOKYO"}
        and regime_key == "RANGING"
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.04
        and bucket_metrics["profit_factor"] >= 1.00
        and capture_efficiency >= 0.14
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "audnzd_rotation_attack"
        priority_multiplier = max(priority_multiplier, 1.08)
        size_multiplier = max(size_multiplier, 1.03)
        if not why_promoted:
            why_promoted = "audnzd_rotation_attack"
        why_throttled = ""
    if (
        symbol_key == "AUDJPY"
        and strategy_key_norm == "AUDJPY_TOKYO_MOMENTUM_BREAKOUT"
        and session_key == "TOKYO"
        and regime_key == "TRENDING"
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.12
        and bucket_metrics["profit_factor"] >= 1.08
        and capture_efficiency >= 0.28
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "audjpy_tokyo_breakout_attack"
        priority_multiplier = max(priority_multiplier, 1.08)
        size_multiplier = max(size_multiplier, 1.03)
        if not why_promoted:
            why_promoted = "audjpy_tokyo_breakout_attack"
        why_throttled = ""
    if (
        symbol_key == "AUDJPY"
        and strategy_key_norm == "AUDJPY_TOKYO_CONTINUATION_PULLBACK"
        and session_key == "TOKYO"
        and regime_key == "TRENDING"
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= -0.02
        and bucket_metrics["profit_factor"] >= 0.95
        and state != "QUARANTINED"
    ):
        state = "NORMAL"
        reason = "audjpy_trending_pullback_preserved"
        priority_multiplier = max(priority_multiplier, 1.02)
        size_multiplier = max(size_multiplier, 1.0)
        if not why_promoted:
            why_promoted = "audjpy_trending_pullback_preserved"
        why_throttled = ""
    if (
        symbol_key == "AUDJPY"
        and strategy_key_norm == "AUDJPY_TOKYO_CONTINUATION_PULLBACK"
        and session_key in {"SYDNEY", "TOKYO"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.04
        and bucket_metrics["profit_factor"] >= 1.02
        and capture_efficiency >= 0.18
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "audjpy_trending_pullback_attack"
        priority_multiplier = max(priority_multiplier, 1.06)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "audjpy_trending_pullback_attack"
        why_throttled = ""
    if (
        symbol_key == "NZDJPY"
        and strategy_key_norm == "NZDJPY_PULLBACK_CONTINUATION"
        and session_key == "TOKYO"
        and regime_key == "TRENDING"
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= -0.01
        and bucket_metrics["profit_factor"] >= 0.98
        and capture_efficiency >= 0.10
        and state != "QUARANTINED"
    ):
        state = "NORMAL"
        reason = "nzdjpy_trending_pullback_preserved"
        priority_multiplier = max(priority_multiplier, 1.02)
        size_multiplier = max(size_multiplier, 1.0)
        if not why_promoted:
            why_promoted = "nzdjpy_trending_pullback_preserved"
        why_throttled = ""
    if (
        symbol_key == "NZDJPY"
        and strategy_key_norm == "NZDJPY_PULLBACK_CONTINUATION"
        and session_key in {"SYDNEY", "TOKYO"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.04
        and bucket_metrics["profit_factor"] >= 1.02
        and capture_efficiency >= 0.18
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "nzdjpy_trending_pullback_attack"
        priority_multiplier = max(priority_multiplier, 1.06)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "nzdjpy_trending_pullback_attack"
        why_throttled = ""
    if (
        symbol_key == "NZDJPY"
        and strategy_key_norm == "NZDJPY_TOKYO_BREAKOUT"
        and session_key in {"SYDNEY", "TOKYO"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.04
        and bucket_metrics["profit_factor"] >= 1.02
        and capture_efficiency >= 0.18
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "nzdjpy_tokyo_breakout_attack"
        priority_multiplier = max(priority_multiplier, 1.07)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "nzdjpy_tokyo_breakout_attack"
        why_throttled = ""
    if (
        symbol_key == "USDJPY"
        and strategy_key_norm == "USDJPY_LIQUIDITY_SWEEP_REVERSAL"
        and session_key == "TOKYO"
        and regime_key == "RANGING"
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= -0.02
        and bucket_metrics["profit_factor"] >= 0.95
        and capture_efficiency >= 0.14
        and state != "QUARANTINED"
    ):
        state = "NORMAL"
        reason = "usdjpy_tokyo_sweep_preserved"
        priority_multiplier = max(priority_multiplier, 1.02)
        size_multiplier = max(size_multiplier, 1.0)
        if not why_promoted:
            why_promoted = "usdjpy_tokyo_sweep_preserved"
        why_throttled = ""
    if (
        symbol_key == "USDJPY"
        and strategy_key_norm == "USDJPY_MOMENTUM_IMPULSE"
        and session_key in {"SYDNEY", "TOKYO"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.02
        and bucket_metrics["profit_factor"] >= 0.98
        and capture_efficiency >= 0.14
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "usdjpy_asia_momentum_attack"
        priority_multiplier = max(priority_multiplier, 1.08)
        size_multiplier = max(size_multiplier, 1.03)
        if not why_promoted:
            why_promoted = "usdjpy_asia_momentum_attack"
        why_throttled = ""
    if (
        symbol_key == "USDJPY"
        and strategy_key_norm == "USDJPY_VWAP_TREND_CONTINUATION"
        and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION", "LOW_LIQUIDITY_CHOP"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.04
        and bucket_metrics["profit_factor"] >= 1.00
        and capture_efficiency >= 0.16
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "usdjpy_vwap_trend_attack"
        priority_multiplier = max(priority_multiplier, 1.07)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "usdjpy_vwap_trend_attack"
        why_throttled = ""
    if (
        symbol_key == "USDJPY"
        and strategy_key_norm == "USDJPY_MOMENTUM_IMPULSE"
        and session_key in {"OVERLAP", "NEW_YORK"}
        and regime_key == "LOW_LIQUIDITY_CHOP"
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.08
        and bucket_metrics["profit_factor"] >= 1.05
        and capture_efficiency >= 0.18
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "usdjpy_momentum_attack"
        priority_multiplier = max(priority_multiplier, 1.07)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "usdjpy_momentum_attack"
        why_throttled = ""
    if (
        symbol_key == "AUDJPY"
        and strategy_key_norm == "AUDJPY_LIQUIDITY_SWEEP_REVERSAL"
        and session_key in {"TOKYO", "SYDNEY"}
        and regime_key in {"RANGING", "MEAN_REVERSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.02
        and bucket_metrics["profit_factor"] >= 0.98
        and capture_efficiency >= 0.18
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "audjpy_asia_sweep_attack"
        priority_multiplier = max(priority_multiplier, 1.08)
        size_multiplier = max(size_multiplier, 1.03)
        if not why_promoted:
            why_promoted = "audjpy_asia_sweep_attack"
        why_throttled = ""
    if (
        symbol_key == "AUDJPY"
        and strategy_key_norm == "AUDJPY_TOKYO_MOMENTUM_BREAKOUT"
        and session_key in {"SYDNEY", "TOKYO"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.04
        and bucket_metrics["profit_factor"] >= 1.02
        and capture_efficiency >= 0.20
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "audjpy_tokyo_breakout_attack"
        priority_multiplier = max(priority_multiplier, 1.09)
        size_multiplier = max(size_multiplier, 1.03)
        if not why_promoted:
            why_promoted = "audjpy_tokyo_breakout_attack_plus"
        why_throttled = ""
    if (
        symbol_key == "AUDJPY"
        and strategy_key_norm == "AUDJPY_LIQUIDITY_SWEEP_REVERSAL"
        and session_key == "TOKYO"
        and regime_key == "RANGING"
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= -0.02
        and bucket_metrics["profit_factor"] >= 0.95
        and capture_efficiency >= 0.22
        and state != "QUARANTINED"
    ):
        state = "NORMAL"
        reason = "audjpy_tokyo_sweep_preserved"
        priority_multiplier = max(priority_multiplier, 1.02)
        size_multiplier = max(size_multiplier, 1.0)
        if not why_promoted:
            why_promoted = "audjpy_tokyo_sweep_preserved"
        why_throttled = ""
    if (
        symbol_key == "AUDJPY"
        and strategy_key_norm == "AUDJPY_LIQUIDITY_SWEEP_REVERSAL"
        and session_key == "OVERLAP"
        and regime_key == "MEAN_REVERSION"
        and sample_size >= 3
        and bucket_metrics["expectancy_r"] >= 0.05
        and bucket_metrics["profit_factor"] >= 1.05
        and capture_efficiency >= 0.14
    ):
        state = "ATTACK"
        reason = "audjpy_overlap_sweep_attack"
        priority_multiplier = max(priority_multiplier, 1.06)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "audjpy_overlap_sweep_attack"
        why_throttled = ""
    if (
        symbol_key == "AUDJPY"
        and strategy_key_norm == "AUDJPY_LIQUIDITY_SWEEP_REVERSAL"
        and session_key == "SYDNEY"
        and regime_key == "MEAN_REVERSION"
        and sample_size >= 2
        and state != "QUARANTINED"
        and (
            bucket_metrics["expectancy_r"] < 0.0
            or bucket_metrics["profit_factor"] < 0.90
            or avg_mfe_r < 0.25
        )
    ):
        state = "REDUCED"
        reason = "audjpy_sydney_sweep_throttle"
        priority_multiplier = min(priority_multiplier, 0.84)
        size_multiplier = min(size_multiplier, 0.86)
        why_throttled = "audjpy_sydney_sweep_throttle"
        why_promoted = ""
    if (
        symbol_key == "XAUUSD"
        and strategy_key_norm == "XAUUSD_LONDON_LIQUIDITY_SWEEP"
        and session_key in {"LONDON", "NEW_YORK"}
        and regime_key == "MEAN_REVERSION"
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.10
        and bucket_metrics["profit_factor"] >= 1.20
        and capture_efficiency >= 0.15
    ):
        state = "ATTACK"
        reason = "xau_mean_reversion_sweep_attack"
        priority_multiplier = max(priority_multiplier, 1.08)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "xau_mean_reversion_sweep_attack"
        why_throttled = ""
    if (
        symbol_key == "GBPUSD"
        and strategy_key_norm == "GBPUSD_LONDON_EXPANSION_BREAKOUT"
        and session_key == "LONDON"
        and regime_key == "LOW_LIQUIDITY_CHOP"
        and sample_size >= 4
        and (
            bucket_metrics["expectancy_r"] < 0.05
            or bucket_metrics["profit_factor"] < 1.05
            or capture_efficiency < 0.18
        )
    ):
        state = "REDUCED"
        reason = "gbpusd_london_breakout_london_chop_throttle"
        priority_multiplier = min(priority_multiplier, 0.84)
        size_multiplier = min(size_multiplier, 0.88)
        why_throttled = "gbpusd_london_breakout_london_chop_throttle"
        why_promoted = ""
    if (
        symbol_key == "GBPUSD"
        and strategy_key_norm == "GBPUSD_LONDON_EXPANSION_BREAKOUT"
        and session_key in {"OVERLAP", "NEW_YORK"}
        and regime_key == "LOW_LIQUIDITY_CHOP"
        and sample_size >= 3
        and bucket_metrics["expectancy_r"] >= 0.08
        and bucket_metrics["profit_factor"] >= 1.10
    ):
        state = "NORMAL"
        reason = "gbpusd_low_liquidity_breakout_preserved"
        priority_multiplier = max(priority_multiplier, 1.03)
        size_multiplier = max(size_multiplier, 1.0)
        if not why_promoted:
            why_promoted = "gbpusd_low_liquidity_breakout_preserved"
        why_throttled = ""
    if (
        symbol_key == "NZDJPY"
        and strategy_key_norm == "NZDJPY_TOKYO_BREAKOUT"
        and session_key in {"SYDNEY", "TOKYO"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.02
        and bucket_metrics["profit_factor"] >= 0.98
        and capture_efficiency >= 0.16
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "nzdjpy_tokyo_breakout_attack"
        priority_multiplier = max(priority_multiplier, 1.08)
        size_multiplier = max(size_multiplier, 1.03)
        if not why_promoted:
            why_promoted = "nzdjpy_tokyo_breakout_attack_plus"
        why_throttled = ""
    if (
        symbol_key == "NZDJPY"
        and strategy_key_norm == "NZDJPY_LIQUIDITY_TRAP_REVERSAL"
        and session_key == "SYDNEY"
        and regime_key in {"RANGING", "MEAN_REVERSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.05
        and bucket_metrics["profit_factor"] >= 1.05
        and capture_efficiency >= 0.18
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "nzdjpy_sydney_trap_attack"
        priority_multiplier = max(priority_multiplier, 1.07)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "nzdjpy_sydney_trap_attack"
        why_throttled = ""
    if (
        symbol_key == "NZDJPY"
        and strategy_key_norm == "NZDJPY_LIQUIDITY_TRAP_REVERSAL"
        and session_key == "TOKYO"
        and regime_key in {"RANGING", "MEAN_REVERSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= -0.02
        and bucket_metrics["profit_factor"] >= 0.95
        and capture_efficiency >= 0.20
        and state != "QUARANTINED"
    ):
        state = "NORMAL"
        reason = "nzdjpy_tokyo_trap_preserved"
        priority_multiplier = max(priority_multiplier, 1.02)
        size_multiplier = max(size_multiplier, 1.0)
        if not why_promoted:
            why_promoted = "nzdjpy_tokyo_trap_preserved"
        why_throttled = ""
    if (
        symbol_key == "AUDNZD"
        and strategy_key_norm == "AUDNZD_STRUCTURE_BREAK_RETEST"
        and session_key in {"SYDNEY", "TOKYO"}
        and regime_key == "RANGING"
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.04
        and bucket_metrics["profit_factor"] >= 1.05
        and capture_efficiency >= 0.42
        and state != "QUARANTINED"
    ):
        state = "NORMAL"
        reason = "audnzd_structure_break_preserved"
        priority_multiplier = max(priority_multiplier, 1.02)
        size_multiplier = max(size_multiplier, 1.0)
        if not why_promoted:
            why_promoted = "audnzd_structure_break_preserved"
        why_throttled = ""
    if (
        symbol_key == "AUDNZD"
        and strategy_key_norm == "AUDNZD_COMPRESSION_RELEASE"
        and session_key in {"SYDNEY", "TOKYO"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.04
        and bucket_metrics["profit_factor"] >= 1.02
        and capture_efficiency >= 0.16
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "audnzd_compression_release_attack"
        priority_multiplier = max(priority_multiplier, 1.07)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "audnzd_compression_release_attack"
        why_throttled = ""
    if (
        symbol_key == "AUDNZD"
        and strategy_key_norm == "AUDNZD_RANGE_ROTATION"
        and session_key in {"SYDNEY", "TOKYO"}
        and regime_key in {"RANGING", "MEAN_REVERSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.02
        and bucket_metrics["profit_factor"] >= 0.96
        and capture_efficiency >= 0.16
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "audnzd_rotation_attack"
        priority_multiplier = max(priority_multiplier, 1.07)
        size_multiplier = max(size_multiplier, 1.03)
        if not why_promoted:
            why_promoted = "audnzd_rotation_attack_plus"
        why_throttled = ""
    if (
        symbol_key == "AUDNZD"
        and strategy_key_norm == "AUDNZD_VWAP_MEAN_REVERSION"
        and session_key == "SYDNEY"
        and regime_key in {"RANGING", "MEAN_REVERSION"}
        and not targeted_rotation_bucket
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.0
        and bucket_metrics["profit_factor"] >= 0.95
        and capture_efficiency >= 0.14
        and state != "QUARANTINED"
    ):
        state = "NORMAL"
        reason = "audnzd_vwap_sydney_preserved"
        priority_multiplier = max(priority_multiplier, 1.02)
        size_multiplier = max(size_multiplier, 1.0)
        if not why_promoted:
            why_promoted = "audnzd_vwap_sydney_preserved"
        why_throttled = ""
    if (
        symbol_key == "EURUSD"
        and strategy_key_norm in {"EURUSD_LONDON_BREAKOUT", "EURUSD_VWAP_PULLBACK", "EURUSD_LIQUIDITY_SWEEP"}
        and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION", "MEAN_REVERSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.02
        and bucket_metrics["profit_factor"] >= 0.98
        and capture_efficiency >= 0.14
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "eurusd_major_session_attack"
        priority_multiplier = max(priority_multiplier, 1.06)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "eurusd_major_session_attack"
        why_throttled = ""
    if (
        symbol_key == "EURJPY"
        and strategy_key_norm == "EURJPY_MOMENTUM_IMPULSE"
        and session_key in {"LONDON", "OVERLAP"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.02
        and bucket_metrics["profit_factor"] >= 0.98
        and capture_efficiency >= 0.14
        and state != "QUARANTINED"
    ):
        state = "NORMAL"
        reason = "eurjpy_london_impulse_preserved"
        priority_multiplier = max(priority_multiplier, 1.03)
        size_multiplier = max(size_multiplier, 1.0)
        if not why_promoted:
            why_promoted = "eurjpy_london_impulse_preserved"
        why_throttled = ""
    if (
        symbol_key == "EURJPY"
        and strategy_key_norm in {"EURJPY_MOMENTUM_IMPULSE", "EURJPY_SESSION_PULLBACK_CONTINUATION"}
        and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.02
        and bucket_metrics["profit_factor"] >= 0.98
        and capture_efficiency >= 0.14
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "eurjpy_major_session_attack"
        priority_multiplier = max(priority_multiplier, 1.06)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "eurjpy_major_session_attack"
        why_throttled = ""
    if (
        symbol_key == "GBPJPY"
        and strategy_key_norm == "GBPJPY_MOMENTUM_IMPULSE"
        and session_key in {"LONDON", "OVERLAP"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.02
        and bucket_metrics["profit_factor"] >= 0.98
        and capture_efficiency >= 0.14
        and state != "QUARANTINED"
    ):
        state = "NORMAL"
        reason = "gbpjpy_london_impulse_preserved"
        priority_multiplier = max(priority_multiplier, 1.03)
        size_multiplier = max(size_multiplier, 1.0)
        if not why_promoted:
            why_promoted = "gbpjpy_london_impulse_preserved"
        why_throttled = ""
    if (
        symbol_key == "GBPJPY"
        and strategy_key_norm in {"GBPJPY_MOMENTUM_IMPULSE", "GBPJPY_SESSION_PULLBACK_CONTINUATION"}
        and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.02
        and bucket_metrics["profit_factor"] >= 0.98
        and capture_efficiency >= 0.14
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "gbpjpy_major_session_attack"
        priority_multiplier = max(priority_multiplier, 1.06)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "gbpjpy_major_session_attack"
        why_throttled = ""
    if (
        symbol_key == "USOIL"
        and strategy_key_norm == "USOIL_INVENTORY_MOMENTUM"
        and session_key in {"OVERLAP", "NEW_YORK"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.02
        and bucket_metrics["profit_factor"] >= 0.98
        and capture_efficiency >= 0.16
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "usoil_inventory_attack"
        priority_multiplier = max(priority_multiplier, 1.06)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "usoil_inventory_attack"
        why_throttled = ""
    if (
        symbol_key == "NAS100"
        and strategy_key_norm in {"NAS100_VWAP_TREND_STRATEGY", "NAS100_LIQUIDITY_SWEEP_REVERSAL"}
        and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
        and regime_key in {"TRENDING", "BREAKOUT_EXPANSION", "MEAN_REVERSION"}
        and sample_size >= 2
        and bucket_metrics["expectancy_r"] >= 0.02
        and bucket_metrics["profit_factor"] >= 0.98
        and capture_efficiency >= 0.14
        and state != "QUARANTINED"
    ):
        state = "ATTACK"
        reason = "nas100_major_session_attack"
        priority_multiplier = max(priority_multiplier, 1.05)
        size_multiplier = max(size_multiplier, 1.02)
        if not why_promoted:
            why_promoted = "nas100_major_session_attack"
        why_throttled = ""
    if (
        symbol_key == "BTCUSD"
        and strategy_key_norm == "BTCUSD_TREND_SCALP"
        and session_key == "OVERLAP"
        and regime_key == "TRENDING"
        and sample_size >= 4
        and bucket_metrics["expectancy_r"] >= 0.30
        and bucket_metrics["profit_factor"] >= 1.30
        and capture_efficiency >= 0.52
        and state != "QUARANTINED"
    ):
        state = "NORMAL"
        reason = "btc_overlap_trend_scalp_preserved"
        priority_multiplier = max(priority_multiplier, 1.02)
        size_multiplier = max(size_multiplier, 1.0)
        why_promoted = "btc_overlap_trend_scalp_preserved"
        why_throttled = ""
    density_recovery_override = False
    density_recovery_lane = bool(
        session_key in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}
        and (
            session_native_pair
            or symbol_key in {"XAUUSD", "USDJPY", "AUDJPY", "NZDJPY", "AUDNZD", "EURUSD", "GBPUSD", "BTCUSD", "NAS100", "EURJPY", "GBPJPY"}
        )
    )
    if (
        density_recovery_lane
        and state == "QUARANTINED"
        and sample_size <= 4
        and bucket_metrics["expectancy_r"] >= -0.10
        and bucket_metrics["profit_factor"] >= 0.82
        and bucket_metrics["late_entry_rate"] <= 0.34
        and bucket_metrics["immediate_invalidation_rate"] <= 0.34
        and bucket_metrics["fast_failure_rate"] <= 0.34
        and bucket_metrics["poor_structure_rate"] <= 0.34
        and capture_efficiency >= 0.09
        and (
            int(today_metrics["wins"]) >= 1
            or today_metrics["expectancy_r"] >= 0.0
            or strategy_metrics["profit_factor"] >= 1.00
            or session_metrics["profit_factor"] >= 1.00
        )
    ):
        density_recovery_override = True
        if (
            bucket_metrics["expectancy_r"] >= 0.0
            and bucket_metrics["profit_factor"] >= 1.00
            and int(today_metrics["wins"]) >= 2
        ):
            state = "NORMAL"
            reason = "density_recovery_lane_preserved"
            priority_multiplier = max(priority_multiplier, 0.98)
            size_multiplier = max(size_multiplier, 0.98)
        else:
            state = "REDUCED"
            reason = "density_recovery_lane_soft_guard"
            priority_multiplier = max(priority_multiplier, 0.92)
            size_multiplier = max(size_multiplier, 0.94)
        if not why_promoted:
            why_promoted = "density_recovery_lane_preserved"
        why_throttled = ""
    gbpusd_london_breakout_london_chop_block = bool(
        symbol_key == "GBPUSD"
        and strategy_key_norm == "GBPUSD_LONDON_EXPANSION_BREAKOUT"
        and session_key == "LONDON"
        and regime_key == "LOW_LIQUIDITY_CHOP"
        and sample_size >= 4
        and (
            bucket_metrics["expectancy_r"] < 0.05
            or bucket_metrics["profit_factor"] < 1.05
            or capture_efficiency < 0.18
        )
    )
    local_block_recommended = bool(
        not density_recovery_override
        and (
            state == "QUARANTINED"
            and sample_size >= (3 if targeted_ranging_bucket else 4)
            and (
                bucket_metrics["late_entry_rate"] >= 0.45
                or bucket_metrics["false_break_rate"] >= 0.55
                or bucket_metrics["immediate_invalidation_rate"] >= 0.45
            )
        )
        or (not density_recovery_override and gbpusd_london_breakout_london_chop_block)
    )
    local_hard_block_recommended = bool(
        (
            not density_recovery_override
            and (
            targeted_exact_bucket
            and (
                sample_size >= 3
                or (
                    sample_size >= 2
                    and bucket_metrics["expectancy_r"] <= -0.60
                    and bucket_metrics["profit_factor"] < 0.10
                )
            )
            and (
                bucket_metrics["expectancy_r"] <= -0.30
                or bucket_metrics["profit_factor"] < 0.25
                or (
                    bucket_metrics["late_entry_rate"] >= 0.30
                    and bucket_metrics["false_break_rate"] >= 0.30
                )
                or bucket_metrics["immediate_invalidation_rate"] >= 0.35
                or (
                    bucket_metrics["fast_failure_rate"] >= 0.40
                    and bucket_metrics["poor_structure_rate"] >= 0.30
                )
            )
            )
        )
        or (
            not density_recovery_override
            and gbpusd_london_breakout_london_chop_block
            and sample_size >= 6
            and (
                bucket_metrics["expectancy_r"] <= -0.40
                or bucket_metrics["profit_factor"] < 0.35
                or bucket_metrics["immediate_invalidation_rate"] >= 0.30
            )
        )
    )

    return {
        "strategy_bucket_state": state,
        "strategy_bucket_reason": reason,
        "strategy_bucket_priority_multiplier": float(clamp(priority_multiplier, 0.72, 1.12)),
        "strategy_bucket_size_multiplier": float(clamp(size_multiplier, 0.80, 1.08)),
        "strategy_bucket_sample_size": int(sample_size),
        "strategy_bucket_expectancy_r": float(bucket_metrics["expectancy_r"]),
        "strategy_bucket_profit_factor": float(bucket_metrics["profit_factor"]),
        "strategy_bucket_win_rate": float(bucket_metrics["win_rate"]),
        "strategy_bucket_false_break_rate": float(bucket_metrics["false_break_rate"]),
        "strategy_bucket_late_entry_rate": float(bucket_metrics["late_entry_rate"]),
        "strategy_bucket_fast_failure_rate": float(bucket_metrics["fast_failure_rate"]),
        "strategy_bucket_immediate_invalidation_rate": float(bucket_metrics["immediate_invalidation_rate"]),
        "strategy_bucket_poor_structure_rate": float(bucket_metrics["poor_structure_rate"]),
        "strategy_bucket_giveback_rate": float(bucket_metrics["giveback_rate"]),
        "strategy_bucket_capture_efficiency": float(bucket_metrics["capture_efficiency"]),
        "strategy_bucket_avg_win_r": float(bucket_metrics["avg_win_r"]),
        "strategy_bucket_avg_loss_r": float(bucket_metrics["avg_loss_r"]),
        "strategy_bucket_realized_rr": float(bucket_metrics["realized_rr"]),
        "strategy_bucket_over_scratch_rate": float(bucket_metrics["over_scratch_rate"]),
        "strategy_bucket_premature_protection_rate": float(bucket_metrics["premature_protection_rate"]),
        "strategy_bucket_stalled_winner_rate": float(bucket_metrics["stalled_winner_rate"]),
        "strategy_bucket_management_quality": float(bucket_metrics["management_quality"]),
        "strategy_bucket_avg_mae_r": float(bucket_metrics["avg_mae_r"]),
        "strategy_bucket_avg_mfe_r": float(bucket_metrics["avg_mfe_r"]),
        "strategy_bucket_today_wins": int(today_metrics["wins"]),
        "strategy_bucket_today_losses": int(today_metrics["losses"]),
        "strategy_bucket_today_trades": int(today_metrics["trades"]),
        "strategy_bucket_recent_expectancy_r": float(strategy_metrics["expectancy_r"]),
        "strategy_bucket_recent_profit_factor": float(strategy_metrics["profit_factor"]),
        "strategy_bucket_recent_win_rate": float(strategy_metrics["win_rate"]),
        "strategy_bucket_session_expectancy_r": float(session_metrics["expectancy_r"]),
        "strategy_bucket_session_profit_factor": float(session_metrics["profit_factor"]),
        "strategy_bucket_session_win_rate": float(session_metrics["win_rate"]),
        "strategy_bucket_should_block_b": bool(local_block_recommended),
        "strategy_bucket_should_block_all_bands": bool(local_hard_block_recommended),
        "why_strategy_promoted": str(why_promoted),
        "why_strategy_throttled": str(why_throttled),
    }


def _runtime_writable(path_value: Any) -> bool:
    try:
        target = str(path_value or "").strip()
        if not target:
            return False
        with open(os.path.join(target, ".apex_write_test"), "a", encoding="utf-8"):
            pass
        os.remove(os.path.join(target, ".apex_write_test"))
        return True
    except Exception:
        return False


def _detect_account_scaling_update(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    *,
    bootstrap_equity_threshold: float,
    material_change_pct: float = 0.05,
    min_absolute_change: float = 5.0,
) -> dict[str, Any]:
    balance = _safe_float(current.get("balance"), 0.0)
    equity = _safe_float(current.get("equity"), balance)
    free_margin = _safe_float(current.get("margin_free"), equity)
    prev_balance = _safe_float((previous or {}).get("balance"), balance)
    prev_equity = _safe_float((previous or {}).get("equity"), equity)
    prev_free_margin = _safe_float((previous or {}).get("margin_free"), free_margin)
    high_watermark = max(
        _safe_float((previous or {}).get("high_watermark_equity"), 0.0),
        equity,
    )

    def _material_change(current_value: float, previous_value: float) -> tuple[float, float, bool]:
        delta = float(current_value - previous_value)
        pct = float(delta / max(abs(previous_value), 1.0))
        material = abs(delta) >= max(float(min_absolute_change), abs(previous_value) * float(material_change_pct))
        return delta, pct, material

    balance_delta, balance_pct, balance_material = _material_change(balance, prev_balance)
    equity_delta, equity_pct, equity_material = _material_change(equity, prev_equity)
    margin_delta, margin_pct, margin_material = _material_change(free_margin, prev_free_margin)
    material_change_detected = bool(balance_material or equity_material or margin_material)
    account_increase_detected = bool(material_change_detected and (balance_delta > 0 or equity_delta > 0 or margin_delta > 0))
    account_decrease_detected = bool(material_change_detected and (balance_delta < 0 or equity_delta < 0 or margin_delta < 0))
    sizing_updated = bool(material_change_detected or previous is None)
    return {
        "balance": balance,
        "equity": equity,
        "free_margin": free_margin,
        "balance_delta": balance_delta,
        "equity_delta": equity_delta,
        "free_margin_delta": margin_delta,
        "balance_change_pct": balance_pct,
        "equity_change_pct": equity_pct,
        "free_margin_change_pct": margin_pct,
        "material_change_detected": material_change_detected,
        "account_increase_detected": account_increase_detected,
        "account_decrease_detected": account_decrease_detected,
        "sizing_updated": sizing_updated,
        "equity_band": _equity_band(equity, bootstrap_equity_threshold),
        "high_watermark_equity": high_watermark,
    }


def _paper_sim_outcome(
    request: ExecutionRequest,
    point: float,
    contract_size: float,
    commission_per_lot: float,
    slippage_points: int,
) -> tuple[float, float, float]:
    seed = int(hashlib.sha256(request.signal_id.encode("utf-8")).hexdigest()[:8], 16)
    probability = max(0.05, min(0.95, float(request.probability)))
    realized_win = (seed % 10_000) / 10_000.0 < probability
    risk_distance = max(abs(request.entry_price - request.stop_price), max(point * 10.0, 1e-6))
    tp_distance = max(abs(request.take_profit_price - request.entry_price), risk_distance)
    slip = max(0, slippage_points) * max(point, 1e-8)

    if request.side.upper() == "BUY":
        if realized_win:
            exit_price = request.entry_price + (tp_distance * 0.82) - slip
        else:
            exit_price = request.stop_price - slip
    else:
        if realized_win:
            exit_price = request.entry_price - (tp_distance * 0.82) + slip
        else:
            exit_price = request.stop_price + slip

    direction = 1.0 if request.side.upper() == "BUY" else -1.0
    pnl_r = ((exit_price - request.entry_price) * direction) / risk_distance
    pnl_money = (exit_price - request.entry_price) * direction * request.volume * max(contract_size, 1.0)
    pnl_money -= commission_per_lot * request.volume
    return pnl_money, pnl_r, exit_price


def _projected_trade_risk_usd(
    entry: float,
    stop: float,
    volume: float,
    contract_size: float,
    *,
    tick_size: float | None = None,
    tick_value: float | None = None,
) -> float:
    if entry <= 0 or stop <= 0 or volume <= 0:
        return 0.0
    stop_distance = abs(entry - stop)
    if _is_positive_number(tick_size) and _is_positive_number(tick_value):
        return (stop_distance / max(float(tick_size), 1e-9)) * float(tick_value) * volume
    if contract_size <= 0:
        return 0.0
    return stop_distance * volume * contract_size


def _runtime_bootstrap_tolerance_cap(
    *,
    symbol_key: str,
    per_trade_cap: float,
    account_equity: float,
    bootstrap_equity_threshold: float,
    broker_min_lot: float,
    setup: str = "",
    session_name: str = "",
    probability: float = 0.0,
    expected_value_r: float = 0.0,
    confluence_score: float = 0.0,
) -> float:
    base_cap = max(0.0, float(per_trade_cap))
    if base_cap <= 0.0:
        return 0.0
    threshold = max(1.0, float(bootstrap_equity_threshold))
    equity_ratio = max(0.0, float(account_equity) / threshold)
    multiplier = 1.10
    normalized_symbol = str(symbol_key or "").upper()
    normalized_setup = str(setup or "").upper()
    normalized_session = str(session_name or "").upper()
    if float(broker_min_lot) >= 1.0:
        multiplier = 1.15
    elif float(broker_min_lot) >= 0.1 or normalized_symbol.startswith(("NAS100", "USTEC", "US100")):
        multiplier = 1.45
        if equity_ratio >= 0.50:
            multiplier = 1.70
        elif equity_ratio >= 0.35:
            multiplier = 1.55
    elif equity_ratio >= 0.50:
        multiplier = 1.20
    strong_bootstrap_candidate = bool(
        float(probability or 0.0) >= 0.78
        and float(expected_value_r or 0.0) >= 0.85
        and float(confluence_score or 0.0) >= 4.0
    )
    if normalized_symbol.startswith(("XAUUSD", "GOLD")) and normalized_setup.startswith("XAUUSD_M5_GRID_SCALPER"):
        multiplier = max(multiplier, 1.26 if normalized_session in {"TOKYO", "SYDNEY"} else 1.32)
        if strong_bootstrap_candidate:
            multiplier = max(multiplier, 1.82 if normalized_session in {"LONDON", "OVERLAP", "NEW_YORK"} else 1.50)
    elif normalized_symbol.startswith(("USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "AUDNZD", "EURUSD", "GBPUSD", "XAGUSD")):
        if strong_bootstrap_candidate:
            multiplier = max(multiplier, 1.36 if equity_ratio < 0.35 else 1.28)
    elif normalized_symbol.startswith(("NAS100", "USTEC", "US100", "USOIL")):
        if strong_bootstrap_candidate:
            multiplier = max(multiplier, 1.82 if equity_ratio < 0.35 else 1.68)
    return base_cap * multiplier


def _runtime_bootstrap_trade_allowed(
    *,
    symbol_key: str,
    projected_loss_usd: float,
    volume: float,
    broker_min_lot: float,
    projected_open_risk_usd: float,
    per_trade_cap: float,
    total_exposure_cap: float,
    account_equity: float,
    bootstrap_equity_threshold: float,
    setup: str = "",
    session_name: str = "",
    probability: float = 0.0,
    expected_value_r: float = 0.0,
    confluence_score: float = 0.0,
) -> tuple[bool, float]:
    tolerance_cap = _runtime_bootstrap_tolerance_cap(
        symbol_key=symbol_key,
        per_trade_cap=per_trade_cap,
        account_equity=account_equity,
        bootstrap_equity_threshold=bootstrap_equity_threshold,
        broker_min_lot=broker_min_lot,
        setup=setup,
        session_name=session_name,
        probability=probability,
        expected_value_r=expected_value_r,
        confluence_score=confluence_score,
    )
    min_lot_only = float(volume) <= (float(broker_min_lot) + 1e-9)
    within_trade = float(projected_loss_usd) <= tolerance_cap
    within_total = (float(projected_open_risk_usd) + float(projected_loss_usd)) <= max(0.0, float(total_exposure_cap))
    return bool(min_lot_only and within_trade and within_total), float(tolerance_cap)


def _projected_open_risk_usd(
    open_positions: list[dict[str, Any]],
    mt5_client: MT5Client,
    symbol_info_cache: dict[str, dict[str, Any]],
    session_name: str | None = None,
) -> float:
    total = 0.0
    session_filter = str(session_name or "").upper()
    for position in open_positions:
        symbol = str(position.get("symbol", "")).upper()
        if not symbol:
            continue
        if session_filter:
            position_session = str(position.get("session_name") or "").upper()
            if position_session and position_session != session_filter:
                continue
        if symbol not in symbol_info_cache:
            try:
                symbol_info_cache[symbol] = mt5_client.get_symbol_info(symbol)
            except Exception:
                symbol_info_cache[symbol] = {"trade_contract_size": 1.0}
        info = symbol_info_cache.get(symbol, {})
        contract_size = float(info.get("trade_contract_size", 1.0))
        total += _projected_trade_risk_usd(
            entry=float(position.get("entry_price", 0.0)),
            stop=float(position.get("sl", 0.0)),
            volume=float(position.get("volume", 0.0)),
            contract_size=contract_size,
            tick_size=float(info.get("trade_tick_size")) if _is_positive_number(info.get("trade_tick_size")) else None,
            tick_value=float(info.get("trade_tick_value")) if _is_positive_number(info.get("trade_tick_value")) else None,
        )
    return total


def _projected_positions_risk_usd(
    positions: list[dict[str, Any]],
    contract_size: float,
    *,
    tick_size: float | None = None,
    tick_value: float | None = None,
) -> float:
    total = 0.0
    safe_contract = max(contract_size, 1.0)
    for position in positions:
        total += _projected_trade_risk_usd(
            entry=float(position.get("entry_price", 0.0)),
            stop=float(position.get("sl", 0.0)),
            volume=float(position.get("volume", 0.0)),
            contract_size=safe_contract,
            tick_size=tick_size,
            tick_value=tick_value,
        )
    return total


def _projected_cycle_risk_usd(
    positions: list[dict[str, Any]],
    contract_size: float,
    *,
    tick_size: float | None = None,
    tick_value: float | None = None,
    side: str | None = None,
) -> float:
    filtered: list[dict[str, Any]] = []
    for position in positions:
        if not _is_xau_grid_setup(str(position.get("setup", ""))):
            continue
        if side is not None and str(position.get("side", "")).upper() != str(side).upper():
            continue
        filtered.append(position)
    return _projected_positions_risk_usd(filtered, contract_size, tick_size=tick_size, tick_value=tick_value)


def run_bot(
    once: bool = False,
    max_loops: int | None = None,
    verify_only: bool = False,
    smoke_demo: bool = False,
    bridge_serve: bool = False,
    paper_sim: bool = False,
) -> dict[str, int]:
    runtime = build_runtime(skip_mt5=paper_sim)
    settings = runtime["settings"]
    logger = runtime["logger"]
    journal: TradeJournal = runtime["journal"]
    mt5_client: MT5Client = runtime["mt5_client"]
    configured_symbols: list[str] = runtime["configured_symbols"]
    resolved_symbols: dict[str, str] = runtime["resolved_symbols"]
    market_data: MarketDataService = runtime["market_data"]
    feature_engineer: FeatureEngineer = runtime["feature_engineer"]
    session_profile: SessionProfile = runtime["session_profile"]
    regime_detector: RegimeDetector = runtime["regime_detector"]
    strategy_engines: dict[str, StrategyEngine] = runtime["strategy_engines"]
    strategy_router: StrategyRouter = runtime["strategy_router"]
    grid_scalper: XAUGridScalper = runtime["grid_scalper"]
    ai_gate: AIGate = runtime["ai_gate"]
    news_engine: NewsEngine = runtime["news_engine"]
    risk_engine: RiskEngine = runtime["risk_engine"]
    execution: ExecutionService = runtime["execution"]
    positions: PositionManager = runtime["positions"]
    portfolio: PortfolioManager = runtime["portfolio"]
    kill_switch: KillSwitch = runtime["kill_switch"]
    monitor: Monitor = runtime["monitor"]
    bridge_queue = runtime["bridge_queue"]
    strategy_optimizer: StrategyOptimizer = runtime["strategy_optimizer"]
    online_learning: OnlineLearningEngine = runtime["online_learning"]
    learning_brain: ApexLearningBrain = runtime["learning_brain"]
    bridge_config: dict[str, Any] = runtime["bridge_config"]
    candidate_tier_config: dict[str, Any] = runtime.get("candidate_tier_config", {})
    symbol_rules: dict[str, SymbolRule] = runtime["symbol_rules"]
    dry_run = bool(runtime["dry_run"])
    if paper_sim:
        dry_run = False

    system_config = settings.section("system")
    risk_config = settings.section("risk")
    micro_config = settings.section("micro_account_mode") if isinstance(settings.raw.get("micro_account_mode"), dict) else {}
    ai_config = settings.section("ai")
    news_config = settings.section("news")
    strategy_config = settings.section("strategy")
    monitor_config = settings.section("monitoring")
    xau_grid_config = settings.section("xau_grid_scalper") if isinstance(settings.raw.get("xau_grid_scalper"), dict) else {}
    xau_grid_risk_config = xau_grid_config.get("risk", {}) if isinstance(xau_grid_config.get("risk"), dict) else {}
    risk_ramp_config = settings.section("risk_ramp") if isinstance(settings.raw.get("risk_ramp"), dict) else {}
    cadence_config = settings.section("evaluation_cadence") if isinstance(settings.raw.get("evaluation_cadence"), dict) else {}
    risk_scaling_config = settings.section("risk_scaling") if isinstance(settings.raw.get("risk_scaling"), dict) else {}
    idea_lifecycle_config = settings.section("idea_lifecycle") if isinstance(settings.raw.get("idea_lifecycle"), dict) else {}
    mode = str(system_config["mode"]).upper()
    interval = int(system_config["update_interval_seconds"])
    loop_sleep_seconds = min(1.0, float(cadence_config.get("loop_sleep_seconds", 1.0)))
    default_symbol_interval = min(1.0, float(cadence_config.get("default_symbol_seconds", 1.0)))
    fast_symbol_interval = min(1.0, float(cadence_config.get("fast_symbol_seconds", 1.0)))
    fast_symbols = {
        _normalize_symbol_key(str(item))
        for item in cadence_config.get("fast_symbols", ["XAUUSD"])
        if str(item).strip()
    } if isinstance(cadence_config.get("fast_symbols", ["XAUUSD"]), list) else {"XAUUSD"}
    fast_sessions = {
        str(item).upper()
        for item in cadence_config.get("fast_sessions", ["LONDON", "OVERLAP", "NEW_YORK"])
        if str(item).strip()
    } if isinstance(cadence_config.get("fast_sessions", ["LONDON", "OVERLAP", "NEW_YORK"]), list) else {"LONDON", "OVERLAP", "NEW_YORK"}
    risk_log_dedupe_seconds = int(cadence_config.get("risk_log_dedupe_seconds", 30))
    friday_cutoff = _friday_cutoff_hour(system_config.get("force_flat_friday_gmt", "20:00"))
    max_positions_total = int(risk_config.get("max_positions_total", system_config.get("max_positions_total", system_config.get("max_positions", 10))))
    max_positions_per_symbol = int(risk_config.get("max_positions_per_symbol", system_config.get("max_positions_per_symbol", 2)))
    max_positions_total = min(max_positions_total, int(risk_config.get("max_positions_total_hard_cap", max_positions_total)))
    max_entries_per_symbol_loop = int(strategy_config.get("max_entries_per_symbol_loop", 2))
    always_on_symbols = _always_on_symbol_keys(system_config)
    debug = bool(monitor_config.get("debug", False))
    loop_limit = 1 if once else max_loops
    bridge_trade_mode = bool(bridge_config.get("enabled", False)) and mode in {"DEMO", "PAPER", "LIVE"} and (not verify_only) and (not dry_run) and (not paper_sim)
    bridge_orchestrator = bridge_config.get("orchestrator", {}) if isinstance(bridge_config.get("orchestrator"), dict) else {}
    bridge_orchestrator_config = dict(bridge_orchestrator)
    for section_name in (
        "microstructure",
        "lead_lag",
        "event_playbooks",
        "aggression",
        "shadow_promotion",
        "execution_memory",
        "self_heal",
        "training_bootstrap",
        "institutional_features",
        "edge_promotion",
        "frequency_policy",
        "funded_mode",
    ):
        if section_name not in bridge_orchestrator_config and isinstance(settings.raw.get(section_name), dict):
            bridge_orchestrator_config[section_name] = dict(settings.raw.get(section_name) or {})
    xau_grid_max_entries_per_symbol_loop = max(
        max_entries_per_symbol_loop,
        int(bridge_orchestrator.get("xau_grid_max_entries_per_symbol_loop", max_entries_per_symbol_loop)),
    )
    bridge_min_lot_by_symbol = {
        _normalize_symbol_key(str(key)): float(value)
        for key, value in (bridge_orchestrator.get("min_lot_by_symbol", {}) if isinstance(bridge_orchestrator.get("min_lot_by_symbol"), dict) else {}).items()
        if _is_positive_number(value)
    }
    bridge_risk_defaults = {
        "base_risk_pct": float(bridge_orchestrator.get("base_risk_pct", 0.03)),
        "min_risk_pct": float(bridge_orchestrator.get("min_risk_pct", 0.015)),
        "max_risk_pct": float(bridge_orchestrator.get("max_risk_pct", 0.10)),
        "min_budget_usd": float(bridge_orchestrator.get("min_budget_usd", 0.5)),
        "max_budget_usd": float(bridge_orchestrator.get("max_budget_usd", 10.0)),
        "xau_grid_cycle_risk_pct": float(bridge_orchestrator.get("xau_grid_cycle_risk_pct", 0.10)),
    }
    bridge_safety_buffer_points = int(bridge_orchestrator.get("safety_buffer_points", 5))
    timeframe_map_config = system_config.get("timeframe_map", {}) if isinstance(system_config.get("timeframe_map"), dict) else {}
    requested_timeframe_default = str(timeframe_map_config.get("direction", "M15")).upper()
    summary = {"loops": 0, "signals": 0, "approved": 0, "rejected": 0, "orders_placed": 0, "errors": 0}
    rollout_started_at = datetime.now(tz=UTC)
    bridge_handle = None
    cooldown_log_state: dict[str, str] = {}
    block_reason_state: dict[str, str] = {}
    block_reason_display_state: dict[str, tuple[str, int]] = {}
    market_closed_log_state: dict[str, str] = {}
    session_block_log_state: dict[str, str] = {}
    last_trade_opened_at: datetime | None = None
    try:
        last_trade_opened_at = _latest_trade_marker()
    except Exception:
        last_trade_opened_at = datetime.now(tz=UTC)
    next_symbol_eval_at: dict[str, datetime] = {}
    cached_symbol_status: dict[str, SymbolStatus] = {}
    meta_fallback_next_log_at: dict[str, datetime] = {}
    last_risk_log_signature: dict[str, tuple[str, datetime]] = {}
    stage_log_state: dict[str, tuple[str, int, datetime]] = {}
    low_trade_warning_logged_at: datetime | None = None
    runtime_alert_state: dict[str, datetime] = {}
    symbol_activity_state: dict[str, dict[str, list[datetime]]] = {}
    symbol_runtime_metrics: dict[str, dict[str, Any]] = {}
    symbol_runtime_defaults: dict[str, Any] = {}
    account_scaling_state: dict[str, dict[str, Any]] = {}
    daily_governor_state_tracker: dict[str, dict[str, Any]] = {}
    hourly_learning_state: dict[str, dict[str, Any]] = {}
    last_hourly_learning_run_at: datetime | None = None
    router_base_sensitivity = {
        "BTCUSD": float(strategy_router.btc_sensitivity),
        "EURUSD": float(strategy_router.forex_sensitivity),
        "GBPUSD": float(strategy_router.forex_sensitivity),
        "USDJPY": float(strategy_router.forex_sensitivity),
        "NAS100": float(strategy_router.nas_sensitivity),
        "USOIL": float(strategy_router.oil_sensitivity),
        "XAUUSD": 1.0,
    }
    idea_lifecycle = TradeIdeaLifecycle(
        archive_minutes=int(idea_lifecycle_config.get("archive_minutes", 10)),
        max_rechecks_per_idea=int(idea_lifecycle_config.get("max_rechecks_per_idea", 5)),
        max_active_ideas_per_symbol=int(idea_lifecycle_config.get("max_active_ideas_per_symbol", 3)),
        recheck_seconds_default=int(idea_lifecycle_config.get("recheck_seconds_default", 60)),
        recheck_seconds_by_session={
            str(key).upper(): int(value)
            for key, value in (idea_lifecycle_config.get("recheck_seconds_by_session", {}) or {}).items()
        },
        cooldown_seconds_by_session={
            str(key).upper(): int(value)
            for key, value in (idea_lifecycle_config.get("cooldown_seconds_by_session", {}) or {}).items()
        },
    )
    reject_display_limit = max(1, int(idea_lifecycle_config.get("reject_display_limit", 2)))

    def log_debug(message: str) -> None:
        if debug:
            logger.info(message)

    def _emit_runtime_alert(
        alert_key: str,
        message: str,
        *,
        level: str = "warning",
        min_interval_seconds: float = 900.0,
        extra_fields: dict[str, Any] | None = None,
        now_ts: datetime | None = None,
    ) -> None:
        current = now_ts or datetime.now(tz=UTC)
        last_emitted_at = runtime_alert_state.get(str(alert_key))
        if last_emitted_at is not None and (current - last_emitted_at).total_seconds() < max(10.0, float(min_interval_seconds)):
            return
        payload = {"alert_key": str(alert_key), **dict(extra_fields or {})}
        log_method = getattr(logger, str(level).lower(), logger.warning)
        log_method(message, extra={"extra_fields": payload})
        runtime_alert_state[str(alert_key)] = current

    def log_stage(symbol: str, stage: str, detail: str, *, suppress_after: int = 0, now_ts: datetime | None = None) -> None:
        current = now_ts or datetime.now(tz=UTC)
        key = f"{symbol.upper()}::{stage.upper()}"
        signature = f"{stage.upper()}::{detail}"
        previous = stage_log_state.get(key)
        count = 1
        if previous is not None and previous[0] == signature:
            count = previous[1] + 1
        stage_log_state[key] = (signature, count, current)
        if suppress_after > 0 and count > suppress_after:
            return
        if stage.upper() == "SCANNING" and previous is not None and (current - previous[2]).total_seconds() < 20:
            return
        if previous is not None and previous[0] == signature and (current - previous[2]).total_seconds() < 20:
            return
        logger.info(
            "idea_stage",
            extra={
                "extra_fields": {
                    "symbol": symbol,
                    "stage": stage.upper(),
                    "detail": detail,
                }
            },
        )

    def _live_open_positions_for_context(
        *,
        account_id: str | None,
        magic_id: int | None,
        now_ts: datetime | None = None,
        reconcile_stale: bool = True,
    ) -> list[dict[str, Any]]:
        if not account_id or magic_id is None:
            return journal.get_open_positions(account=account_id, magic=magic_id)
        positions = journal.get_open_positions(account=account_id, magic=magic_id)
        if not positions or not bridge_trade_mode:
            return positions
        current = now_ts or datetime.now(tz=UTC)
        stale_after_seconds = max(60.0, float(bridge_orchestrator.get("reality_sync_stale_seconds", 120)))
        grace_seconds = max(90.0, stale_after_seconds)
        account_snapshot = None
        if hasattr(bridge_queue, "latest_account_snapshot"):
            try:
                account_snapshot = bridge_queue.latest_account_snapshot(account=account_id, magic=int(magic_id))
            except Exception:
                account_snapshot = None
        symbol_snapshots: dict[str, dict[str, Any] | None] = {}
        for position in positions:
            symbol_name = str(position.get("symbol") or "").strip()
            if not symbol_name:
                continue
            symbol_key = _normalize_symbol_key(symbol_name)
            if symbol_key not in symbol_snapshots:
                symbol_snapshots[symbol_key] = _resolve_bridge_symbol_snapshot(
                    bridge_queue,
                    account=str(account_id),
                    magic=int(magic_id),
                    symbol=symbol_name,
                )
        confirmed, stale = _filter_broker_confirmed_positions(
            positions=positions,
            account_snapshot=account_snapshot,
            symbol_snapshots=symbol_snapshots,
            now_ts=current,
            stale_after_seconds=stale_after_seconds,
            grace_seconds=grace_seconds,
        )
        if reconcile_stale and stale:
            equity_after_close = _safe_float((account_snapshot or {}).get("equity"), 0.0)
            for item in stale:
                position = item.get("position") if isinstance(item, dict) else {}
                signal_id = str((position or {}).get("signal_id") or "").strip()
                if not signal_id:
                    continue
                symbol_name = str((position or {}).get("symbol") or "").strip()
                reason = str(item.get("reason") or "broker_snapshot_reconcile")
                symbol_snapshot = item.get("symbol_snapshot") if isinstance(item.get("symbol_snapshot"), dict) else {}
                close_context = {
                    "reconcile_reason": reason,
                    "account": str(account_id),
                    "magic": int(magic_id),
                    "symbol": symbol_name,
                    "ticket": str((position or {}).get("ticket") or ""),
                    "account_total_open_positions": _safe_int((account_snapshot or {}).get("total_open_positions"), -1),
                    "symbol_open_count": _safe_int((symbol_snapshot or {}).get("open_count"), -1),
                    "account_snapshot_updated_at": str((account_snapshot or {}).get("updated_at") or ""),
                    "symbol_snapshot_updated_at": str((symbol_snapshot or {}).get("updated_at") or ""),
                }
                journal.reconcile_missing_close(
                    signal_id,
                    exit_reason="broker_snapshot_reconcile",
                    equity_after_close=equity_after_close if equity_after_close > 0 else None,
                    close_context_json=json.dumps(close_context, sort_keys=True),
                    post_trade_review_json=json.dumps({"likely_issue": "missing_close_report_reconciled"}, sort_keys=True),
                    adjustment_tags_json=json.dumps(["broker_snapshot_reconcile"], sort_keys=True),
                )
                journal.log_event(signal_id, "BROKER_SNAPSHOT_RECONCILE_CLOSE", close_context)
                logger.warning(
                    "journal_open_reconciled_from_broker_snapshot",
                    extra={
                        "extra_fields": {
                            "signal_id": signal_id,
                            "symbol": symbol_name,
                            "ticket": str((position or {}).get("ticket") or ""),
                            "reason": reason,
                            "account": str(account_id),
                            "magic": int(magic_id),
                        }
                    },
                )
        return confirmed

    def _latest_trade_marker(
        *,
        account_id: str | None = None,
        magic_id: int | None = None,
    ) -> datetime:
        markers: list[datetime] = []
        try:
            for position in _live_open_positions_for_context(account_id=account_id, magic_id=magic_id, reconcile_stale=False):
                opened_at_raw = str(position.get("opened_at") or "").strip()
                if not opened_at_raw:
                    continue
                opened_at = datetime.fromisoformat(opened_at_raw.replace("Z", "+00:00"))
                if opened_at.tzinfo is None:
                    opened_at = opened_at.replace(tzinfo=UTC)
                markers.append(opened_at.astimezone(UTC))
            last_closed_trade = journal.last_closed_trade(account=account_id, magic=magic_id)
            if isinstance(last_closed_trade, dict):
                closed_at_raw = str(last_closed_trade.get("closed_at") or "").strip()
                if closed_at_raw:
                    closed_at = datetime.fromisoformat(closed_at_raw.replace("Z", "+00:00"))
                    if closed_at.tzinfo is None:
                        closed_at = closed_at.replace(tzinfo=UTC)
                    markers.append(closed_at.astimezone(UTC))
        except Exception:
            markers = []
        return max(markers) if markers else datetime.now(tz=UTC)

    def _select_bridge_runtime_context() -> tuple[str | None, int | None, dict[str, Any] | None]:
        if (not bridge_trade_mode) or (not hasattr(bridge_queue, "recent_account_snapshots")):
            return None, None, None
        try:
            snapshots = list(
                bridge_queue.recent_account_snapshots(
                    limit=max(50, len(configured_symbols) * 20),
                )
            )
        except Exception:
            snapshots = []
        if not snapshots:
            return None, None, None
        configured_symbol_keys = {
            _normalize_symbol_key(resolved_symbols.get(symbol, symbol))
            for symbol in configured_symbols
        }
        grouped: dict[tuple[str, int], dict[str, Any]] = {}
        for snapshot in snapshots:
            account_id = str(snapshot.get("account") or "").strip()
            magic_id = _safe_int(snapshot.get("magic"), 0)
            if not account_id or magic_id <= 0:
                continue
            key = (account_id, magic_id)
            entry = grouped.setdefault(
                key,
                {
                    "symbols": set(),
                    "latest": snapshot,
                    "latest_updated_at": str(snapshot.get("updated_at") or ""),
                },
            )
            symbol_key = _normalize_symbol_key(str(snapshot.get("symbol_key") or snapshot.get("symbol") or ""))
            if symbol_key:
                entry["symbols"].add(symbol_key)
            updated_at = str(snapshot.get("updated_at") or "")
            if updated_at >= str(entry.get("latest_updated_at") or ""):
                entry["latest"] = snapshot
                entry["latest_updated_at"] = updated_at
        if not grouped:
            return None, None, None
        selected_account, selected_magic = max(
            grouped.keys(),
            key=lambda item: (
                len(grouped[item]["symbols"] & configured_symbol_keys),
                len(grouped[item]["symbols"]),
                str(grouped[item]["latest_updated_at"] or ""),
            ),
        )
        try:
            selected_snapshot = bridge_queue.latest_account_snapshot(
                account=selected_account,
                magic=selected_magic,
            )
        except Exception:
            selected_snapshot = None
        if not isinstance(selected_snapshot, dict):
            selected_snapshot = dict(grouped[(selected_account, selected_magic)]["latest"])
        return selected_account, int(selected_magic), selected_snapshot

    def _record_symbol_activity(symbol: str, activity: str, now_ts: datetime) -> None:
        symbol_key = _normalize_symbol_key(symbol)
        bucket = symbol_activity_state.setdefault(symbol_key, {}).setdefault(activity, [])
        bucket.append(now_ts)
        cutoff = now_ts - timedelta(minutes=20)
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)

    def _symbol_activity_count(symbol: str, activity: str, window_seconds: int, now_ts: datetime) -> int:
        symbol_key = _normalize_symbol_key(symbol)
        bucket = symbol_activity_state.setdefault(symbol_key, {}).setdefault(activity, [])
        cutoff = now_ts - timedelta(seconds=max(1, int(window_seconds)))
        while bucket and bucket[0] < cutoff:
            bucket.pop(0)
        return len(bucket)

    def _reason_key(reason: str) -> str:
        normalized = str(reason).strip() or "unknown_block"
        for prefix in (
            "grid_ai_deny",
            "final_gate_conservative_reject",
            "online_model_reject",
            "no_base_candidate",
            "idea_cooldown_active",
        ):
            if normalized.startswith(prefix):
                return prefix
        if ":" in normalized and len(normalized) > 40:
            return normalized.split(":", 1)[0]
        return normalized

    def _display_reason(symbol: str, reason: str) -> str:
        normalized_reason = str(reason).strip() or "unknown_block"
        reason_key = _reason_key(normalized_reason)
        if _state_from_reason(normalized_reason) in {"CLOSED", "SESSION_BLOCK", "PRECHECK_FAIL", "REGIME_BLOCK"}:
            block_reason_display_state[symbol] = (reason_key, 1)
            return normalized_reason
        display_previous = block_reason_display_state.get(symbol)
        display_count = 1
        if display_previous is not None and display_previous[0] == reason_key:
            display_count = display_previous[1] + 1
        block_reason_display_state[symbol] = (reason_key, display_count)
        return normalized_reason if display_count <= reject_display_limit else "ready_recheck"

    def _update_runtime_metrics(symbol: str, now_ts: datetime, **extra: Any) -> dict[str, Any]:
        symbol_key = _normalize_symbol_key(symbol)
        previous = dict(symbol_runtime_metrics.get(symbol_key, {}))
        snapshot = {
            **previous,
            **symbol_runtime_defaults,
            "scans_last_15m": _symbol_activity_count(symbol, "scans", 900, now_ts),
            "candidate_attempts_last_15m": _symbol_activity_count(symbol, "candidate_attempts", 900, now_ts),
            "candidate_count_last_15m": _symbol_activity_count(symbol, "candidate_count", 900, now_ts),
            "ai_reviews_last_15m": _symbol_activity_count(symbol, "ai_reviews", 900, now_ts),
            "actions_sent_last_15m": _symbol_activity_count(symbol, "actions_sent", 900, now_ts),
            "queued_for_ea_last_15m": _symbol_activity_count(symbol, "queued_for_ea", 900, now_ts),
            "pre_exec_rejects_last_15m": _symbol_activity_count(symbol, "pre_exec_rejects", 900, now_ts),
            "market_closed_rejects_last_15m": _symbol_activity_count(symbol, "market_closed_blocks", 900, now_ts),
            "session_ineligible_rejects_last_15m": _symbol_activity_count(symbol, "session_ineligible", 900, now_ts),
            "stale_archives_last_15m": _symbol_activity_count(symbol, "stale_archives", 900, now_ts),
            "min_lot_over_budget_last_15m": _symbol_activity_count(symbol, "min_lot_over_budget", 900, now_ts),
            "risk_cap_rejects_last_15m": _symbol_activity_count(symbol, "risk_cap_rejects", 900, now_ts),
            "margin_fail_last_15m": _symbol_activity_count(symbol, "margin_fail", 900, now_ts),
            "impossible_candidate_families_last_15m": _symbol_activity_count(symbol, "impossible_candidate_families", 900, now_ts),
        }
        snapshot.update(extra)
        resolved_strategy_key = str(snapshot.get("strategy_key") or snapshot.get("last_strategy_checked") or "").strip()
        resolved_strategy_pool = [
            str(item).strip()
            for item in list(snapshot.get("strategy_pool") or [])
            if str(item).strip()
        ]
        if resolved_strategy_key and resolved_strategy_key not in resolved_strategy_pool:
            resolved_strategy_pool.insert(0, resolved_strategy_key)
        if not resolved_strategy_pool and resolved_strategy_key:
            resolved_strategy_pool = [resolved_strategy_key]
        if not resolved_strategy_key and resolved_strategy_pool:
            resolved_strategy_key = str(resolved_strategy_pool[0])
        if resolved_strategy_key and not snapshot.get("last_strategy_checked"):
            snapshot["last_strategy_checked"] = str(resolved_strategy_key)
        existing_ranked_entries = [
            dict(item)
            for item in list(snapshot.get("strategy_pool_ranking") or [])
            if isinstance(item, dict)
        ]
        if resolved_strategy_key and (not existing_ranked_entries or not any(str(item.get("strategy_key") or "").strip() == resolved_strategy_key for item in existing_ranked_entries)):
            existing_ranked_entries.insert(
                0,
                {
                    "strategy_key": str(resolved_strategy_key),
                    "setup": str(snapshot.get("last_setup_family_considered") or snapshot.get("last_signal") or ""),
                    "lane_name": str(snapshot.get("lane_name") or ""),
                    "strategy_state": str(snapshot.get("strategy_state") or "NORMAL"),
                    "strategy_score": float(
                        snapshot.get("strategy_score")
                        or snapshot.get("session_adjusted_score")
                        or snapshot.get("trade_quality_score")
                        or 0.0
                    ),
                    "strategy_pool_rank_score": float(
                        snapshot.get("session_adjusted_score")
                        or snapshot.get("strategy_score")
                        or snapshot.get("trade_quality_score")
                        or 0.0
                    ),
                    "rank_score": float(
                        snapshot.get("session_adjusted_score")
                        or snapshot.get("strategy_score")
                        or snapshot.get("trade_quality_score")
                        or 0.0
                    ),
                    "regime_state": str(snapshot.get("regime_state") or ""),
                    "regime_fit": float(snapshot.get("regime_fit") or 0.0),
                    "session_fit": float(snapshot.get("session_fit") or 0.0),
                    "volatility_fit": float(snapshot.get("volatility_fit") or 0.0),
                    "pair_behavior_fit": float(snapshot.get("pair_behavior_fit") or 0.0),
                    "execution_quality_fit": float(snapshot.get("execution_quality_fit") or 0.0),
                    "entry_timing_score": float(snapshot.get("entry_timing_score") or 0.0),
                    "structure_cleanliness_score": float(snapshot.get("structure_cleanliness_score") or 0.0),
                    "strategy_recent_performance": float(snapshot.get("strategy_recent_performance") or 0.0),
                    "session_priority_profile": str(snapshot.get("session_priority_profile") or ""),
                    "lane_session_priority": str(snapshot.get("lane_session_priority") or ""),
                    "session_native_pair": bool(snapshot.get("session_native_pair", False)),
                    "session_priority_multiplier": float(snapshot.get("session_priority_multiplier") or 1.0),
                    "pair_priority_rank_in_session": int(snapshot.get("pair_priority_rank_in_session") or 99),
                    "lane_budget_share": float(snapshot.get("lane_budget_share") or 0.0),
                    "lane_available_capacity": float(snapshot.get("lane_available_capacity") or 0.0),
                    "lane_capacity_usage": float(snapshot.get("lane_capacity_usage") or 0.0),
                    "exceptional_override_used": bool(snapshot.get("exceptional_override_used", False)),
                    "exceptional_override_reason": str(snapshot.get("exceptional_override_reason") or ""),
                    "why_non_native_pair_won": str(snapshot.get("why_non_native_pair_won") or ""),
                    "why_native_pair_lost_priority": str(snapshot.get("why_native_pair_lost_priority") or ""),
                },
            )
        if resolved_strategy_key or resolved_strategy_pool or existing_ranked_entries:
            summarized_strategy_pool = _summarize_strategy_pool_rankings(
                symbol_key=symbol_key,
                ranked_entries=existing_ranked_entries,
                preferred_strategy_key=resolved_strategy_key,
            )
            resolved_strategy_pool = [
                str(item.get("strategy_key") or "").strip()
                for item in summarized_strategy_pool
                if str(item.get("strategy_key") or "").strip()
            ]
            if not resolved_strategy_key and resolved_strategy_pool:
                resolved_strategy_key = str(resolved_strategy_pool[0])
            snapshot["strategy_key"] = str(resolved_strategy_key)
            snapshot["strategy_pool"] = list(resolved_strategy_pool)
            snapshot["strategy_pool_ranking"] = list(summarized_strategy_pool)
            if resolved_strategy_key:
                snapshot["strategy_pool_winner"] = str(snapshot.get("strategy_pool_winner") or resolved_strategy_key)
            if snapshot.get("strategy_pool_winner") and not snapshot.get("winning_strategy_reason"):
                snapshot["winning_strategy_reason"] = "active_runtime_strategy"
        sticky_fields = (
            "pre_open_checks_complete",
            "pre_open_news_summary",
            "pre_open_risk_notes",
            "pre_open_setup_windows",
            "public_proxy_availability",
            "macro_event_bias",
            "session_bias_summary",
            "next_open_time_utc",
            "next_open_time_local",
            "dst_mode_active",
            "news_refresh_at",
            "next_macro_event",
            "event_risk_window_active",
            "post_news_trade_window_active",
            "news_bias_direction",
            "news_confidence",
            "news_data_quality",
            "news_headlines",
            "news_source_breakdown",
            "news_category_summary",
            "news_primary_category",
            "news_secondary_source_used",
            "news_rss_headlines",
            "news_rss_headline_count",
            "session_policy_current",
            "active_setup_windows",
            "setup_proxy_availability",
            "proxy_unavailable_fallback_used",
            "btc_reason_no_candidate",
            "btc_last_structure_state",
            "btc_last_volatility_state",
            "btc_last_spread_state",
            "strategy_key",
            "strategy_pool",
            "strategy_pool_ranking",
            "strategy_pool_winner",
            "winning_strategy_reason",
            "strategy_score",
            "strategy_state",
            "strategy_recent_performance",
            "quality_size_multiplier",
            "regime_fit",
            "session_fit",
            "volatility_fit",
            "pair_behavior_fit",
            "execution_quality_fit",
            "entry_timing_score",
            "structure_cleanliness_score",
            "candidate_family_counts",
            "last_setup_family_considered",
            "last_setup_policy_window",
            "last_strategy_checked",
            "xau_modes_enabled",
            "fix_window_active",
            "regime_state",
            "regime_confidence",
            "volatility_forecast_state",
            "trade_quality_score",
            "trade_quality_band",
            "trade_quality_components",
            "execution_quality_state",
            "execution_quality_score",
            "liquidity_alignment_score",
            "nearest_liquidity_above",
            "nearest_liquidity_below",
            "liquidity_sweep_detected",
            "pressure_proxy_score",
            "continuation_pressure",
            "exhaustion_signal",
            "absorption_signal",
            "adaptive_risk_state",
            "risk_modifiers",
            "overflow_band_active",
            "portfolio_bias",
            "exposure_cluster_detected",
            "daily_state",
            "daily_state_reason",
            "daily_realized_pnl",
            "trading_day_key",
            "timezone_used",
            "closed_trades_today",
            "allowed_by_daily_governor",
            "risk_multiplier_applied",
            "proof_exception_used",
            "proof_bucket_state",
            "proof_exception_reason",
            "xau_grid_override_allowed",
            "xau_grid_override_reason",
            "xau_grid_sub_budget",
            "xau_grid_risk_multiplier",
            "current_daily_trade_cap",
            "current_overflow_daily_trade_cap",
            "current_max_risk_pct",
            "current_phase_base_risk_pct",
            "current_phase_max_risk_pct",
            "base_daily_trade_target",
            "stretch_daily_trade_target",
            "hard_upper_limit",
            "hourly_base_target",
            "hourly_stretch_target",
            "projected_trade_capacity_today",
            "scaling_mode",
            "current_compounding_state",
            "current_growth_bias",
            "next_equity_milestone",
            "current_band_target",
            "fallback_band_active",
            "reason_higher_band_unavailable",
            "selected_trade_band",
            "best_A_plus_count",
            "best_A_count",
            "best_B_plus_count",
            "best_B_count",
            "best_C_count",
            "lane_name",
            "session_priority_profile",
            "lane_session_priority",
            "session_native_pair",
            "session_priority_multiplier",
            "pair_priority_rank_in_session",
            "lane_budget_share",
            "lane_available_capacity",
            "exceptional_override_used",
            "exceptional_override_reason",
            "why_non_native_pair_won",
            "why_native_pair_lost_priority",
            "pair_status",
            "pair_status_reason",
            "pair_state_multiplier",
            "rolling_expectancy_by_pair",
            "rolling_pf_by_pair",
            "rolling_expectancy_by_session",
            "rolling_pf_by_session",
            "rolling_win_rate_by_pair",
            "rolling_win_rate_by_session",
            "false_break_rate",
            "management_quality_score",
            "today_session_wins",
            "today_session_losses",
            "today_session_trades",
            "why_pair_is_promoted",
            "why_pair_is_throttled",
            "session_adjusted_score",
            "current_band_attempted",
            "what_would_make_it_pass",
            "lane_strength_multiplier",
            "lane_score",
            "session_density_score",
            "quality_cluster_score",
            "cluster_mode_active",
            "stretch_mode_active",
            "current_capacity_mode",
            "primary_block_reason",
            "secondary_block_reason",
            "lane_capacity_usage",
            "session_stop_state",
            "session_stop_reason",
            "session_entries_blocked",
            "session_realized_pnl",
            "session_realized_pnl_pct",
            "session_trade_count",
            "runtime_market_data_source",
            "runtime_market_data_consensus_state",
            "runtime_market_data_provider_diagnostics",
        )
        for field in sticky_fields:
            if field in extra:
                continue
            previous_value = previous.get(field)
            if previous_value in (None, "", [], {}):
                continue
            current_value = snapshot.get(field)
            default_value = symbol_runtime_defaults.get(field)
            if current_value in (None, "", [], {}) and current_value == default_value:
                snapshot[field] = previous_value
        symbol_runtime_metrics[symbol_key] = snapshot
        return snapshot

    def mark_block(symbol: str, reason: str, status: SymbolStatus | None = None, *, now_ts: datetime | None = None) -> None:
        summary["rejected"] += 1
        normalized_reason = str(reason).strip() or "unknown_block"
        current = now_ts or datetime.now(tz=UTC)
        previous = block_reason_state.get(symbol)
        if previous != normalized_reason:
            logger.info(
                "entry_blocked",
                extra={"extra_fields": {"symbol": symbol, "reason": normalized_reason}},
            )
            block_reason_state[symbol] = normalized_reason
        if _state_from_reason(normalized_reason) == "CLOSED":
            _record_symbol_activity(symbol, "market_closed_blocks", current)
        if _state_from_reason(normalized_reason) == "SESSION_BLOCK":
            _record_symbol_activity(symbol, "session_ineligible", current)
        if _is_pre_exec_block_reason(normalized_reason):
            _record_symbol_activity(symbol, "pre_exec_rejects", current)
        if normalized_reason.startswith(("min_lot_over_budget", "candidate_rejected_pre_exec:min_lot_over_budget")):
            _record_symbol_activity(symbol, "min_lot_over_budget", current)
            _record_symbol_activity(symbol, "impossible_candidate_families", current)
        if normalized_reason.startswith(
            (
                "risk_budget_exceeded",
                "trade_plan_risk_exceeded",
                "bootstrap_trade_risk_exceeds_cap",
                "bootstrap_total_risk_exceeds_cap",
                "micro_survival_trade_risk_exceeds_usd",
                "micro_survival_total_risk_exceeds_usd",
            )
        ):
            _record_symbol_activity(symbol, "risk_cap_rejects", current)
        if normalized_reason in {"margin_insufficient", "insufficient_margin"}:
            _record_symbol_activity(symbol, "margin_fail", current)
        if normalized_reason == "stale_idea_family_archived":
            _record_symbol_activity(symbol, "stale_archives", current)
        if status is not None:
            status.reason = _display_reason(symbol, normalized_reason)
            status.trading_allowed = False
            status.current_state = _state_from_reason(normalized_reason)
            status.last_block_reason = normalized_reason
            if _is_pre_exec_block_reason(normalized_reason):
                status.pre_exec_status = "FAIL"
                status.pre_exec_reason = normalized_reason
        journal.record_block(normalized_reason, symbol=symbol)

    if once:
        print("Resolved Symbols:", ", ".join(f"{symbol}->{resolved_symbols[symbol]}" for symbol in configured_symbols))

    bridge_explicitly_requested = bool(bridge_serve)
    bridge_enabled = bool(bridge_config.get("enabled", False))
    if bridge_explicitly_requested and not bridge_enabled:
        logger.warning("bridge.enabled is false in config; overriding because --bridge-serve was passed")

    if bridge_explicitly_requested:
        try:
            from src.bridge_server import start_bridge_background

            bridge_orchestrator_config = bridge_config.get("orchestrator", {}) if isinstance(bridge_config.get("orchestrator"), dict) else {}
            if isinstance(bridge_orchestrator_config, dict):
                if "news_mode" not in bridge_orchestrator_config:
                    bridge_orchestrator_config["news_mode"] = str(news_config.get("mode", "SAFE")).upper()
                if "nas_strategy" not in bridge_orchestrator_config and isinstance(settings.raw.get("nas_strategy"), dict):
                    bridge_orchestrator_config["nas_strategy"] = settings.raw.get("nas_strategy")
                if "oil_strategy" not in bridge_orchestrator_config and isinstance(settings.raw.get("oil_strategy"), dict):
                    bridge_orchestrator_config["oil_strategy"] = settings.raw.get("oil_strategy")
            dashboard_runtime_config = settings.raw.get("dashboard", {}) if isinstance(settings.raw.get("dashboard"), dict) else {}
            bind_host = str(bridge_config.get("host", "127.0.0.1"))
            bind_port = int(bridge_config.get("port", 8000))
            if bool(dashboard_runtime_config.get("public_enabled", False)):
                bind_host = str(dashboard_runtime_config.get("bind_host") or dashboard_runtime_config.get("host") or "0.0.0.0")
                bind_port = int(dashboard_runtime_config.get("bind_port") or dashboard_runtime_config.get("port") or bind_port)
            bridge_handle = start_bridge_background(
                host=bind_host,
                port=bind_port,
                queue=bridge_queue,
                journal=journal,
                online_learning=online_learning,
                learning_brain=learning_brain,
                session_name_resolver=session_profile.infer_name,
                ai_gate=ai_gate,
                logger=logger,
                auth_token=str(bridge_config.get("auth_token", "")),
                orchestrator_config=bridge_orchestrator_config,
                risk_config=settings.section("risk"),
                execution_config=settings.section("execution"),
                xau_grid_config=settings.section("xau_grid_scalper"),
                dashboard_config=settings.raw.get("dashboard", {}),
                strategy_optimizer=strategy_optimizer,
                symbol_rules_path=settings.resolve_path_value(str(bridge_config.get("symbol_rules_file", "config/symbol_rules.yaml"))),
                market_data_status_provider=market_data.status_snapshot,
                runtime_metrics_provider=lambda: {key: dict(value) for key, value in symbol_runtime_metrics.items()},
                telegram_config=settings.raw.get("telegram", {}) if isinstance(settings.raw.get("telegram"), dict) else {},
                aggression_config=settings.raw.get("aggression_controller", {}) if isinstance(settings.raw.get("aggression_controller"), dict) else {},
            )
        except Exception as exc:
            logger.warning(f"Bridge server start failed: {exc}")
            raise RuntimeError("Bridge server failed to start while --bridge-serve was requested") from exc

    if smoke_demo:
        if mode not in {"DEMO", "PAPER"}:
            print(f"SMOKE DEMO BLOCKED: mode must be DEMO or PAPER, got {mode}")
            return summary
        verification = mt5_client.verify_connection(configured_symbols)
        print("SMOKE VERIFY:", json.dumps(verification, indent=2, default=str, sort_keys=True))
        if not verification.get("ok", False):
            summary["errors"] += 1
            return summary

    try:
        heartbeat_value = str(os.getenv("APEX_RUNTIME_HEARTBEAT_FILE") or system_config.get("runtime_heartbeat_file") or "").strip()
        heartbeat_path = Path(heartbeat_value) if heartbeat_value else None
        while True:
            summary["loops"] += 1
            now = datetime.now(tz=UTC)
            local_now = datetime.now().astimezone()
            idea_lifecycle.archive_stale(now)
            kill_status = kill_switch.status()
            live_allowed = True
            trading_enabled, trading_reason = determine_trading_state(mode, system_config, live_allowed, verify_only=verify_only)

            try:
                account = mt5_client.get_account_info()
                account_from_mt5 = bool(getattr(mt5_client, "connected", False) and (not bool(getattr(mt5_client, "disable_mt5", False))))
            except Exception as exc:
                summary["errors"] += 1
                logger.warning(f"MT5 account snapshot unavailable: {exc}")
                account = _fallback_account_snapshot(float(micro_config.get("internal_equity_estimate", 50.0)))
                account_from_mt5 = False
            if account_from_mt5:
                runtime_alert_state.pop("mt5_runtime_disconnected", None)
            elif not (dry_run or verify_only):
                _emit_runtime_alert(
                    "mt5_runtime_disconnected",
                    "mt5_runtime_disconnected",
                    min_interval_seconds=300.0,
                    now_ts=now,
                    extra_fields={
                        "mode": mode,
                        "bridge_trade_mode": bool(bridge_trade_mode),
                        "paper_sim": bool(paper_sim),
                    },
                )

            bridge_context_account: str | None = None
            bridge_context_magic: int | None = None
            bridge_snapshot = None
            if bridge_trade_mode:
                bridge_context_account, bridge_context_magic, bridge_snapshot = _select_bridge_runtime_context()
            account, account_label, _bridge_snapshot_active = _apply_runtime_account_snapshot(
                account,
                account_from_mt5=account_from_mt5,
                bridge_snapshot=bridge_snapshot if bridge_trade_mode else None,
                internal_equity_estimate=float(micro_config.get("internal_equity_estimate", 50.0)),
            )
            account_scaling_key = f"{str(bridge_context_account or 'default').strip()}::{int(bridge_context_magic or 0)}"
            previous_scaling_state = account_scaling_state.get(account_scaling_key)
            current_scaling_state = _detect_account_scaling_update(
                previous_scaling_state,
                account,
                bootstrap_equity_threshold=float(micro_config.get("bootstrap_equity_threshold", 160.0)),
            )
            if current_scaling_state.get("material_change_detected"):
                logger.info(
                    "account_scaling_update",
                    extra={
                        "extra_fields": {
                            "account": str(bridge_context_account or ""),
                            "magic": int(bridge_context_magic or 0),
                            "balance": float(current_scaling_state["balance"]),
                            "equity": float(current_scaling_state["equity"]),
                            "free_margin": float(current_scaling_state["free_margin"]),
                            "balance_delta": float(current_scaling_state["balance_delta"]),
                            "equity_delta": float(current_scaling_state["equity_delta"]),
                            "equity_band": str(current_scaling_state["equity_band"]),
                            "account_increase_detected": bool(current_scaling_state["account_increase_detected"]),
                            "account_decrease_detected": bool(current_scaling_state["account_decrease_detected"]),
                        }
                    },
                )
            account_scaling_state[account_scaling_key] = {
                **current_scaling_state,
                "updated_at": now.isoformat(),
            }
            brain = runtime.get("learning_brain") if isinstance(runtime, dict) else None
            heartbeat_brain_status: dict[str, Any] = {}
            if brain is not None and hasattr(brain, "status_snapshot"):
                try:
                    raw_brain_status = brain.status_snapshot()
                    heartbeat_brain_status = {
                        "mode": str(raw_brain_status.get("mode") or ""),
                        "recovery_mode_active": bool(raw_brain_status.get("recovery_mode_active", False)),
                        "last_cycle_report": dict(raw_brain_status.get("last_cycle_report") or {}),
                        "last_self_heal_actions": list(raw_brain_status.get("self_heal_actions") or [])[:3],
                    }
                except Exception:
                    heartbeat_brain_status = {}
            _write_runtime_heartbeat(
                heartbeat_path,
                now=now,
                mode=mode,
                account_label=account_label,
                account_state=account,
                summary=summary,
                extra={
                    "bridge_trade_mode": bool(bridge_trade_mode),
                    "session_name": str(dominant_session_name(now)),
                    "kill_switch_reason": str(getattr(kill_status, "reason", "") or ""),
                    "entry_block_state": runtime_entry_block_state(now) if bridge_trade_mode else {"blocked": False, "reason": ""},
                    "brain_status": heartbeat_brain_status,
                    "bridge_runtime_insights": bridge_runtime_insights_state() if bridge_trade_mode else {},
                },
            )
            symbol_runtime_defaults = {
                "active_bridge_account": str(bridge_context_account or ""),
                "active_bridge_magic": int(bridge_context_magic or 0),
                "live_balance": float(current_scaling_state["balance"]),
                "live_equity": float(current_scaling_state["equity"]),
                "live_free_margin": float(current_scaling_state["free_margin"]),
                "account_increase_detected": bool(current_scaling_state["account_increase_detected"]),
                "account_decrease_detected": bool(current_scaling_state["account_decrease_detected"]),
                "material_account_change_detected": bool(current_scaling_state["material_change_detected"]),
                "sizing_updated": bool(current_scaling_state["sizing_updated"]),
                "sizing_updated_at": str(current_scaling_state.get("updated_at") or now.isoformat()),
                "equity_band": str(current_scaling_state["equity_band"]),
                "high_watermark_equity": float(current_scaling_state["high_watermark_equity"]),
                "balance_change_pct": float(current_scaling_state["balance_change_pct"]),
                "equity_change_pct": float(current_scaling_state["equity_change_pct"]),
                "free_margin_change_pct": float(current_scaling_state["free_margin_change_pct"]),
                "amount_increase": float(current_scaling_state.get("amount_increase", 0.0)),
                "prior_risk_budget": float(current_scaling_state.get("prior_risk_budget", 0.0)),
                "new_risk_budget": float(current_scaling_state.get("new_risk_budget", 0.0)),
                "sizing_updated_reason": str(current_scaling_state.get("sizing_updated_reason", "")),
                "account_label": account_label,
                "pre_open_checks_complete": False,
                "pre_open_news_summary": "",
                "pre_open_risk_notes": {},
                "pre_open_setup_windows": [],
                "public_proxy_availability": {},
                "macro_event_bias": {},
                "session_bias_summary": "",
                "next_open_time_utc": "",
                "next_open_time_local": "",
                "dst_mode_active": {},
                "news_refresh_at": "",
                "next_macro_event": None,
                "event_risk_window_active": False,
                "post_news_trade_window_active": False,
                "news_bias_direction": "neutral",
                "news_confidence": 0.0,
                "news_data_quality": "unknown",
                "daily_governor_started_at": "",
                "daily_governor_trigger_day_key": "",
                "daily_governor_timeout_hours": float(risk_config.get("daily_governor_timeout_hours", 4.0)),
                "daily_governor_timeout_elapsed": False,
                "daily_governor_force_release": False,
                "daily_governor_emergency_review_ready": False,
                "hourly_learning_summary": {},
                "pair_hourly_review": {},
                "recent_missed_opportunity_summary": "",
                "hourly_parameter_adjustments": {},
                "setup_families_promoted": [],
                "setup_families_suppressed": [],
                "proxy_unavailable_fallback_used": False,
                "btc_reason_no_candidate": "",
                "btc_last_structure_state": "",
                "btc_last_volatility_state": "",
                "btc_last_spread_state": "",
                "strategy_key": "",
                "strategy_pool": [],
                "strategy_pool_ranking": [],
                "strategy_pool_winner": "",
                "winning_strategy_reason": "",
                "strategy_score": 0.0,
                "strategy_state": "NORMAL",
                "strategy_recent_performance": 0.5,
                "quality_size_multiplier": 1.0,
                "regime_fit": 0.0,
                "session_fit": 0.0,
                "volatility_fit": 0.0,
                "pair_behavior_fit": 0.0,
                "execution_quality_fit": 0.0,
                "entry_timing_score": 0.0,
                "structure_cleanliness_score": 0.0,
                "proof_bucket_state": "neutral",
                "proof_exception_reason": "",
                "lane_adjustment_state": "neutral",
                "last_strategy_checked": "",
                "current_band_target": "A+",
                "fallback_band_active": False,
                "reason_higher_band_unavailable": "",
                "selected_trade_band": "",
                "best_A_plus_count": 0,
                "best_A_count": 0,
                "best_B_plus_count": 0,
                "best_B_count": 0,
                "best_C_count": 0,
                "lane_name": "",
                "session_priority_profile": "GLOBAL",
                "lane_session_priority": "NEUTRAL",
                "session_native_pair": False,
                "session_priority_multiplier": 1.0,
                "pair_priority_rank_in_session": 99,
                "lane_budget_share": 0.0,
                "lane_available_capacity": 0.0,
                "exceptional_override_used": False,
                "exceptional_override_reason": "",
                "why_non_native_pair_won": "",
                "why_native_pair_lost_priority": "",
                "pair_status": "NORMAL",
                "pair_status_reason": "neutral_pair_flow",
                "pair_state_multiplier": 1.0,
                "rolling_expectancy_by_pair": 0.0,
                "rolling_pf_by_pair": 1.0,
                "rolling_expectancy_by_session": 0.0,
                "rolling_pf_by_session": 1.0,
                "rolling_win_rate_by_pair": 0.5,
                "rolling_win_rate_by_session": 0.5,
                "false_break_rate": 0.0,
                "management_quality_score": 0.5,
                "today_session_wins": 0,
                "today_session_losses": 0,
                "today_session_trades": 0,
                "why_pair_is_promoted": "",
                "why_pair_is_throttled": "",
                "session_adjusted_score": 0.0,
                "current_band_attempted": "",
                "what_would_make_it_pass": "",
                "lane_strength_multiplier": 1.0,
                "lane_score": 0.0,
                "session_density_score": 0.0,
                "quality_cluster_score": 0.0,
                "cluster_mode_active": False,
                "stretch_mode_active": False,
                "current_capacity_mode": "BASE",
                "primary_block_reason": "",
                "secondary_block_reason": "",
                "lane_capacity_usage": 0.0,
                "session_stop_state": "",
                "session_stop_reason": "",
                "session_entries_blocked": False,
                "session_realized_pnl": 0.0,
                "session_realized_pnl_pct": 0.0,
                "session_trade_count": 0,
                "runtime_market_data_source": "",
                "runtime_market_data_consensus_state": "",
                "runtime_market_data_provider_diagnostics": {},
                "microstructure_ready": False,
                "microstructure_direction": "neutral",
                "microstructure_confidence": 0.0,
                "microstructure_pressure_score": 0.0,
                "microstructure_cumulative_delta_score": 0.0,
                "microstructure_depth_imbalance": 0.0,
                "microstructure_drift_score": 0.0,
                "microstructure_spread_stability": 0.0,
            }
            if bridge_trade_mode:
                try:
                    last_trade_opened_at = _latest_trade_marker(account_id=bridge_context_account, magic_id=bridge_context_magic)
                except Exception:
                    last_trade_opened_at = datetime.now(tz=UTC)
            account_equity = float(account["equity"])
            effective_max_positions_total, effective_max_positions_per_symbol, micro_active = _micro_position_caps(
                micro_config=micro_config,
                mode=mode,
                equity=account_equity,
                base_total=max_positions_total,
                base_per_symbol=max_positions_per_symbol,
            )
            if bridge_trade_mode and not (bridge_context_account and bridge_context_magic):
                global_stats = journal.neutral_stats(
                    current_equity=account_equity,
                    now_ts=now,
                )
            else:
                global_stats = journal.stats(
                    current_equity=account_equity,
                    account=bridge_context_account,
                    magic=bridge_context_magic,
                )
            phase_performance = build_performance_report(
                journal.closed_trades(100, account=bridge_context_account, magic=bridge_context_magic),
                session_name_resolver=session_profile.infer_name,
            )
            recent_closed_trades = journal.closed_trades(
                250,
                account=bridge_context_account,
                magic=bridge_context_magic,
            )
            phase_performance["daily_green_streak"] = _daily_green_streak(recent_closed_trades)
            hard_kill_ttl_hours = max(1.0, float(system_config.get("hard_kill_ttl_hours", 6.0)))
            hard_kill_sydney_reset_enabled = bool(system_config.get("hard_kill_sydney_reset_enabled", True))
            kill_status = kill_switch.status(
                now=now,
                equity=account_equity,
                current_session_key=str(global_stats.trading_day_key),
                hard_ttl_hours=hard_kill_ttl_hours,
                sydney_reset_enabled=hard_kill_sydney_reset_enabled,
            )
            if kill_status.auto_clear_reason and kill_status.recovery_mode == "RECOVERY_DEFENSIVE" and kill_status.level is None:
                logger.info(
                    "kill_auto_cleared_to_recovery",
                    extra={
                        "extra_fields": {
                            "reason": str(kill_status.reason or ""),
                            "auto_clear_reason": str(kill_status.auto_clear_reason or ""),
                            "trading_day_key": str(global_stats.trading_day_key),
                            "equity": float(account_equity),
                        }
                    },
                )
                kill_switch.enter_recovery(
                    reason=str(kill_status.reason or "hard_kill_recovery"),
                    now=now,
                    session_key=str(global_stats.trading_day_key),
                    last_equity=float(account_equity),
                    auto_clear_reason=str(kill_status.auto_clear_reason or ""),
                    recovery_mode="RECOVERY_DEFENSIVE",
                    recovery_wins_needed=3,
                )
                kill_status = kill_switch.status(
                    now=now,
                    equity=account_equity,
                    current_session_key=str(global_stats.trading_day_key),
                    hard_ttl_hours=hard_kill_ttl_hours,
                    sydney_reset_enabled=hard_kill_sydney_reset_enabled,
                )
            recovery_mode_active = bool(kill_status.recovery_mode and kill_status.level is None)
            if recovery_mode_active:
                recovery_wins_observed = _non_losing_closes_since(
                    recent_closed_trades,
                    started_at=kill_status.recovery_started_at,
                )
                kill_switch.update_recovery_progress(wins_observed=recovery_wins_observed)
                recovery_note = _recovery_mode_release_note(
                    started_at=kill_status.recovery_started_at,
                    closed_trades=recent_closed_trades,
                    learning_brain_status=learning_brain.status_snapshot(),
                )
                if recovery_note:
                    logger.info(
                        "kill_recovery_released",
                        extra={
                            "extra_fields": {
                                "reason": str(kill_status.reason or ""),
                                "recovery_note": recovery_note,
                                "wins_observed": int(recovery_wins_observed),
                                "trading_day_key": str(global_stats.trading_day_key),
                            }
                        },
                    )
                    kill_switch.clear()
                    kill_status = kill_switch.status(
                        now=now,
                        equity=account_equity,
                        current_session_key=str(global_stats.trading_day_key),
                        hard_ttl_hours=hard_kill_ttl_hours,
                        sydney_reset_enabled=hard_kill_sydney_reset_enabled,
                    )
                    recovery_mode_active = False
                else:
                    kill_status = kill_switch.status(
                        now=now,
                        equity=account_equity,
                        current_session_key=str(global_stats.trading_day_key),
                        hard_ttl_hours=hard_kill_ttl_hours,
                        sydney_reset_enabled=hard_kill_sydney_reset_enabled,
                    )
            phase_state = _phase_state(account_equity, phase_performance)
            current_scaling_state.update(phase_state)
            previous_phase = str((previous_scaling_state or {}).get("current_phase", "") or "")
            current_phase_name = str(current_scaling_state.get("current_phase", "PHASE_1"))
            if current_phase_name != previous_phase:
                logger.info(
                    "capital_phase_transition",
                    extra={
                        "extra_fields": {
                            "previous_phase": previous_phase,
                            "current_phase": current_phase_name,
                            "reason": str(current_scaling_state.get("phase_reason", "")),
                            "equity": float(account_equity),
                            "rolling_win_rate": float((phase_performance.get("overall", {}) or {}).get("win_rate", 0.0)),
                            "sample_size": int(float((phase_performance.get("overall", {}) or {}).get("trades", 0.0) or 0.0)),
                        }
                    },
                )
            previous_phase_risk_pct = float((previous_scaling_state or {}).get("current_risk_pct", risk_config.get("risk_per_trade", 0.0025)))
            current_scaling_state["amount_increase"] = max(
                0.0,
                float(current_scaling_state.get("balance_delta", 0.0)),
                float(current_scaling_state.get("equity_delta", 0.0)),
            )
            current_scaling_state["prior_risk_budget"] = max(
                0.0,
                float((previous_scaling_state or {}).get("equity", account_equity)) * previous_phase_risk_pct,
            )
            current_scaling_state["new_risk_budget"] = max(
                0.0,
                float(current_scaling_state.get("equity", account_equity))
                * float(current_scaling_state.get("current_risk_pct", risk_config.get("risk_per_trade", 0.0025))),
            )
            if previous_scaling_state is None:
                current_scaling_state["sizing_updated_reason"] = "initial_snapshot"
            elif bool(current_scaling_state.get("account_increase_detected")):
                current_scaling_state["sizing_updated_reason"] = "deposit_or_equity_increase_detected"
            elif bool(current_scaling_state.get("account_decrease_detected")):
                current_scaling_state["sizing_updated_reason"] = "drawdown_tighten"
            elif bool(current_scaling_state.get("sizing_updated")):
                current_scaling_state["sizing_updated_reason"] = "material_account_change"
            else:
                current_scaling_state["sizing_updated_reason"] = "unchanged"
            account_scaling_state[account_scaling_key] = {
                **current_scaling_state,
                "updated_at": now.isoformat(),
            }
            daily_state_name, daily_state_reason = risk_engine.resolve_daily_state_from_stats(
                global_stats,
                caution_threshold_pct=float(risk_config.get("daily_caution_threshold_pct", 0.02)),
                defensive_threshold_pct=float(risk_config.get("daily_defensive_threshold_pct", 0.035)),
                hard_stop_threshold_pct=float(risk_config.get("daily_hard_stop_threshold_pct", risk_config.get("hard_daily_dd_pct", 0.05))),
            )
            daily_governor_timeout_hours = max(0.0, float(risk_config.get("daily_governor_timeout_hours", 4.0)))
            current_trading_day_key_value = str(global_stats.trading_day_key or now.astimezone(SYDNEY).date().isoformat())
            governor_tracker = dict(daily_governor_state_tracker.get(account_scaling_key) or {})
            if (
                str(governor_tracker.get("trading_day_key") or "").strip()
                and str(governor_tracker.get("trading_day_key") or "").strip() != current_trading_day_key_value
            ):
                governor_tracker = {}
            governor_state_active = str(daily_state_name).upper() in {"DAILY_CAUTION", "DAILY_DEFENSIVE", "DAILY_HARD_STOP"}
            if governor_state_active:
                if (
                    str(governor_tracker.get("state") or "").upper() != str(daily_state_name).upper()
                    or str(governor_tracker.get("trading_day_key") or "") != current_trading_day_key_value
                ):
                    governor_tracker = {
                        "state": str(daily_state_name).upper(),
                        "state_reason": str(daily_state_reason),
                        "started_at": now.isoformat(),
                        "trading_day_key": current_trading_day_key_value,
                        "emergency_learning_requested": False,
                    }
                else:
                    governor_tracker["state_reason"] = str(daily_state_reason)
            else:
                governor_tracker = {}
            governor_timeout_elapsed = False
            if governor_state_active and governor_tracker and daily_governor_timeout_hours > 0.0:
                started_at_raw = str(governor_tracker.get("started_at") or "").strip()
                if started_at_raw:
                    try:
                        governor_started_at = datetime.fromisoformat(started_at_raw.replace("Z", "+00:00"))
                        if governor_started_at.tzinfo is None:
                            governor_started_at = governor_started_at.replace(tzinfo=UTC)
                        governor_timeout_elapsed = (now.astimezone(governor_started_at.tzinfo) - governor_started_at).total_seconds() >= (
                            daily_governor_timeout_hours * 3600.0
                        )
                    except Exception:
                        governor_timeout_elapsed = False
            daily_governor_force_release = False
            if governor_state_active and governor_timeout_elapsed:
                daily_governor_force_release = True
                if (
                    str(daily_state_name).upper() == "DAILY_HARD_STOP"
                    and float(global_stats.daily_pnl_pct or 0.0) < 0.0
                    and float(global_stats.daily_dd_pct_live or 0.0) > float(risk_config.get("daily_caution_threshold_pct", 0.03))
                    and not bool(recovery_mode_active)
                ):
                    daily_state_name = "DAILY_DEFENSIVE"
                else:
                    daily_state_name = "DAILY_NORMAL"
                daily_state_reason = f"{daily_state_reason}:governor_timeout_release"
            governor_retrain_due = bool(
                governor_state_active
                and not bool(governor_tracker.get("emergency_learning_requested", False))
            )
            if governor_state_active:
                governor_tracker["emergency_learning_requested"] = True
                daily_governor_state_tracker[account_scaling_key] = dict(governor_tracker)
            else:
                daily_governor_state_tracker.pop(account_scaling_key, None)
            self_heal_config = (
                bridge_orchestrator_config.get("self_heal", {})
                if isinstance(bridge_orchestrator_config.get("self_heal"), dict)
                else {}
            )
            self_heal_enabled = bool(self_heal_config.get("enabled", False))
            retrain_interval_hours = max(1.0, float(self_heal_config.get("retrain_interval_hours", 4.0)))
            undertrade_trigger_trades = max(1, int(self_heal_config.get("undertrade_trigger_trades_per_day", 30)))
            last_self_heal_retrain_at = _parse_iso_utc(current_scaling_state.get("last_self_heal_retrain_at"))
            retrain_due = bool(
                last_self_heal_retrain_at is None
                or (now - last_self_heal_retrain_at).total_seconds() >= (retrain_interval_hours * 3600.0)
            )
            undertrade_retrain_due = bool(
                self_heal_enabled
                and retrain_due
                and int(global_stats.trades_today or 0) < undertrade_trigger_trades
                and dominant_session_name(now) in {"LONDON", "OVERLAP", "NEW_YORK"}
            )
            if self_heal_enabled and (governor_retrain_due or undertrade_retrain_due):
                retrain_reason = "daily_governor_emergency_review" if governor_retrain_due else "undertrade_self_heal_retrain"
                try:
                    learning_brain.run_cycle(
                        now_utc=now,
                        session_name=dominant_session_name(now),
                        account_state={
                            "equity": float(account_equity),
                            "balance": float(account.get("balance", account_equity) or account_equity),
                        },
                        runtime_state={
                            "symbols": list(configured_symbols),
                            "current_session_name": dominant_session_name(now),
                            "recovery_mode_active": bool(recovery_mode_active),
                            "daily_state": str(daily_state_name),
                            "daily_state_reason": str(daily_state_reason),
                            "trades_today": int(global_stats.trades_today),
                            "trading_day_key": str(current_trading_day_key_value),
                        },
                        weekly_prep=False,
                        force_local_retrain=True,
                    )
                    learning_brain.apply_promoted_params_to_runtime(runtime)
                    current_scaling_state["last_self_heal_retrain_at"] = now.isoformat()
                    current_scaling_state["last_self_heal_retrain_reason"] = retrain_reason
                    account_scaling_state[account_scaling_key] = {
                        **current_scaling_state,
                        "updated_at": now.isoformat(),
                    }
                    logger.info(
                        "self_heal_retrain_triggered",
                        extra={
                            "extra_fields": {
                                "reason": retrain_reason,
                                "trades_today": int(global_stats.trades_today),
                                "daily_state": str(daily_state_name),
                                "daily_state_reason": str(daily_state_reason),
                                "trading_day_key": str(current_trading_day_key_value),
                            }
                        },
                    )
                except Exception as exc:
                    logger.warning(
                        "self_heal_retrain_failed",
                        extra={
                            "extra_fields": {
                                "reason": retrain_reason,
                                "error": str(exc),
                            }
                        },
                    )
            soft_dd_active = bool(daily_state_name in {"DAILY_CAUTION", "DAILY_DEFENSIVE"})
            hard_dd_active = bool(daily_state_name == "DAILY_HARD_STOP")
            ai_key_present = bool(os.getenv(str(ai_config.get("openai_api_env", "OPENAI_API_KEY")), "").strip())
            news_key_present = bool(
                str(news_config.get("provider", "")).strip().lower() in {"stub", "safe", "disabled"}
                or os.getenv(str(news_config.get("api_key_env", "")), "").strip()
                or str(news_config.get("api_key", "")).strip()
                or os.getenv(str(news_config.get("fallback_api_key_env", "")), "").strip()
                or str(news_config.get("fallback_api_key", "")).strip()
            )
            bridge_ready = bool(bridge_trade_mode and bridge_context_account and bridge_context_magic)
            symbol_runtime_defaults.update(
                {
                    "current_phase": str(current_scaling_state.get("current_phase", "PHASE_1")),
                    "phase_reason": str(current_scaling_state.get("phase_reason", "")),
                    "current_risk_pct": float(current_scaling_state.get("current_risk_pct", risk_config.get("risk_per_trade", 0.0025))),
                    "current_max_risk_pct": float(current_scaling_state.get("current_max_risk_pct", risk_config.get("hard_risk_cap", 0.05))),
                    "current_daily_trade_cap": int(current_scaling_state.get("current_daily_trade_cap", 2)),
                    "current_overflow_daily_trade_cap": int(current_scaling_state.get("current_overflow_daily_trade_cap", current_scaling_state.get("current_daily_trade_cap", 2))),
                    "current_ai_threshold_mode": str(current_scaling_state.get("current_ai_threshold_mode", "conservative")),
                    "base_daily_trade_target": int(current_scaling_state.get("base_daily_trade_target", current_scaling_state.get("current_daily_trade_cap", 0))),
                    "stretch_daily_trade_target": int(current_scaling_state.get("stretch_daily_trade_target", current_scaling_state.get("current_overflow_daily_trade_cap", 0))),
                    "hard_upper_limit": int(current_scaling_state.get("hard_upper_limit", current_scaling_state.get("current_overflow_daily_trade_cap", 0))),
                    "hourly_base_target": int(current_scaling_state.get("hourly_base_target", risk_config.get("max_trades_per_hour", 0))),
                    "hourly_stretch_target": int(current_scaling_state.get("hourly_stretch_target", risk_config.get("max_trades_per_hour", 0))),
                    "projected_trade_capacity_today": int(current_scaling_state.get("projected_trade_capacity_today", current_scaling_state.get("stretch_daily_trade_target", 0))),
                    "scaling_mode": str(current_scaling_state.get("scaling_mode", "quick_smart_scaler")),
                    "current_compounding_state": str(current_scaling_state.get("current_compounding_state", "base_flow")),
                    "current_growth_bias": str(current_scaling_state.get("current_growth_bias", "balanced_compounding")),
                    "daily_green_streak": int(current_scaling_state.get("daily_green_streak", 0)),
                    "next_equity_milestone": float(current_scaling_state.get("next_equity_milestone", 0.0)),
                    "next_phase_requirements": dict(current_scaling_state.get("next_phase_requirements", {})),
                    "equity_phase_thresholds": dict(current_scaling_state.get("equity_phase_thresholds", {})),
                    "performance_phase_thresholds": dict(current_scaling_state.get("performance_phase_thresholds", {})),
                    "soft_dd_active": soft_dd_active,
                    "hard_dd_active": hard_dd_active,
                    "daily_state": str(daily_state_name),
                    "daily_state_reason": str(daily_state_reason),
                    "daily_governor_started_at": str(governor_tracker.get("started_at") or ""),
                    "daily_governor_trigger_day_key": str(governor_tracker.get("trading_day_key") or ""),
                    "daily_governor_timeout_hours": float(daily_governor_timeout_hours),
                    "daily_governor_timeout_elapsed": bool(governor_timeout_elapsed),
                    "daily_governor_force_release": bool(daily_governor_force_release),
                    "daily_governor_emergency_review_ready": bool(governor_state_active),
                    "daily_realized_pnl": float(global_stats.daily_realized_pnl),
                    "trading_day_key": str(current_trading_day_key_value),
                    "timezone_used": str(global_stats.timezone_used),
                    "recovery_mode_active": bool(recovery_mode_active),
                    "recovery_mode": str(kill_status.recovery_mode or ""),
                    "kill_switch_level": str(kill_status.level or ""),
                    "kill_switch_reason": str(kill_status.reason or ""),
                    "kill_switch_auto_clear_reason": str(kill_status.auto_clear_reason or ""),
                    "kill_switch_recovery_wins_needed": int(kill_status.recovery_wins_needed),
                    "kill_switch_recovery_wins_observed": int(kill_status.recovery_wins_observed),
                    "last_self_heal_retrain_at": str(current_scaling_state.get("last_self_heal_retrain_at", "")),
                    "last_self_heal_retrain_reason": str(current_scaling_state.get("last_self_heal_retrain_reason", "")),
                    "closed_trades_today": int(global_stats.trades_today),
                    "day_start_equity": float(global_stats.day_start_equity),
                    "day_high_equity": float(global_stats.day_high_equity),
                    "daily_dd_pct_live": float(global_stats.daily_dd_pct_live),
                    "soft_dd_elite_mode_active": False,
                    "soft_dd_trade_count": int(global_stats.soft_dd_trade_count),
                    "last_soft_dd_trade_reason": "",
                    "last_soft_dd_rejection_reason": "",
                    "runtime_paths_resolved": settings.runtime_paths.snapshot(),
                    "logs_writable": _runtime_writable(settings.runtime_paths.logs_dir),
                    "cache_writable": _runtime_writable(settings.runtime_paths.cache_dir),
                    "config_loaded": True,
                    "mt5_reachable": bool(account_from_mt5),
                    "bridge_reachable": bridge_ready,
                    "account_connected": bool(account_from_mt5 or _bridge_snapshot_active),
                    "ai_key_present": ai_key_present,
                    "public_news_proxy_available": bool(news_key_present),
                    "startup_sequence_status": "ready" if bridge_ready else "awaiting_bridge_context",
                    "restart_safe_reconciliation_status": "broker_snapshot_reconciled",
                }
            )
            if kill_status.level in {"SOFT", "HARD"}:
                recovery_note = _soft_kill_recovery_note(
                    kill_status.reason,
                    now=now,
                    created_at=kill_status.created_at,
                    system_config=system_config,
                    risk_config=risk_config,
                    micro_config=micro_config,
                    account_equity=account_equity,
                    global_stats=global_stats,
                )
                if recovery_note:
                    logger.info(
                        "kill_auto_cleared",
                        extra={
                            "extra_fields": {
                                "level": kill_status.level,
                                "reason": kill_status.reason,
                                "recovery_note": recovery_note,
                                "daily_pnl_pct": float(global_stats.daily_pnl_pct),
                                "rolling_drawdown_pct": float(global_stats.rolling_drawdown_pct),
                                "absolute_drawdown_pct": float(global_stats.absolute_drawdown_pct),
                            }
                        },
                    )
                    kill_switch.clear()
                    kill_status = kill_switch.status()
            if last_hourly_learning_run_at is None or (now - last_hourly_learning_run_at).total_seconds() >= 3600.0:
                hourly_learning_state = _hourly_learning_review(
                    now_ts=now,
                    journal=journal,
                    runtime_snapshot=symbol_runtime_metrics,
                    account=bridge_context_account,
                    magic=bridge_context_magic,
                )
                last_hourly_learning_run_at = now
                adaptive_bias: dict[str, float] = {}
                for symbol_key, payload in hourly_learning_state.items():
                    adjustments = payload.get("hourly_parameter_adjustments", {})
                    adaptive_bias[symbol_key] = clamp(float(adjustments.get("candidate_sensitivity_mult", 1.0)), 0.85, 1.20)
                strategy_router.adaptive_sensitivity = adaptive_bias
            open_positions_journal = _live_open_positions_for_context(
                account_id=bridge_context_account,
                magic_id=bridge_context_magic,
                now_ts=now,
            )
            symbol_info_cache: dict[str, dict[str, Any]] = {}
            projected_open_risk_usd = _projected_open_risk_usd(open_positions_journal, mt5_client, symbol_info_cache)
            if dry_run or verify_only or bridge_trade_mode or paper_sim:
                mt5_positions = []
            else:
                try:
                    mt5_positions = mt5_client.positions()
                except Exception as exc:
                    summary["errors"] += 1
                    logger.warning(f"MT5 open positions unavailable: {exc}")
                    mt5_positions = []

            session_context = session_profile.classify(now)
            session_projected_open_risk_usd = _projected_open_risk_usd(
                open_positions_journal,
                mt5_client,
                symbol_info_cache,
                session_name=session_context.session_name,
            )
            global_session_multiplier = session_context.size_multiplier
            if _is_weekend_market_mode(now):
                session_status = "WEEKEND BTC ONLY"
                active_session_name = "WEEKEND BTC ONLY"
            else:
                session_status = "IN" if session_context.in_session else "OUT"
                active_session_name = session_context.session_name
            queued_counts = bridge_queue.counts_by_symbol(list(resolved_symbols.values())) if bridge_trade_mode else {}
            bridge_delivery_counts: dict[str, int] = {}
            if bridge_trade_mode and bridge_context_account and bridge_context_magic:
                if isinstance(bridge_snapshot, dict):
                    for configured_symbol in configured_symbols:
                        delivered_symbol_key = _normalize_symbol_key(resolved_symbols[configured_symbol])
                        try:
                            bridge_delivery_counts[delivered_symbol_key] = int(
                                bridge_queue.hourly_delivered_count(
                                    account=bridge_context_account,
                                    magic=bridge_context_magic,
                                    symbol_key=delivered_symbol_key,
                                    window_minutes=15,
                                )
                            )
                        except Exception:
                            bridge_delivery_counts[delivered_symbol_key] = 0
            if session_context.in_session and last_trade_opened_at is not None:
                no_trade_seconds = max(0.0, (now - last_trade_opened_at).total_seconds())
                if no_trade_seconds >= 3600:
                    if low_trade_warning_logged_at is None or (now - low_trade_warning_logged_at).total_seconds() >= 1800:
                        logger.warning(
                            "low_trade_frequency_detected",
                            extra={
                                "extra_fields": {
                                    "session": session_context.session_name,
                                    "minutes_without_trade": round(no_trade_seconds / 60.0, 1),
                                }
                            },
                        )
                        low_trade_warning_logged_at = now
                    _emit_runtime_alert(
                        "no_trade_over_60m",
                        "no_trade_frequency_over_60m",
                        min_interval_seconds=1800.0,
                        now_ts=now,
                        extra_fields={
                            "session": session_context.session_name,
                            "minutes_without_trade": round(no_trade_seconds / 60.0, 1),
                            "bridge_trade_mode": bool(bridge_trade_mode),
                        },
                    )
                else:
                    low_trade_warning_logged_at = None
                    runtime_alert_state.pop("no_trade_over_60m", None)

            if kill_status.level == "HARD":
                if trading_enabled and (not bridge_trade_mode):
                    closed = execution.hard_flatten(None, int(risk_config["max_slippage_points"]))
                    monitor.alert("HARD_KILL", "Hard kill switch active; flattened positions", closed_positions=closed)
                else:
                    monitor.alert("HARD_KILL", "Hard kill switch active; trading disabled", closed_positions=0)
                if once:
                    break
                time.sleep(max(0.25, loop_sleep_seconds))
                continue

            counts = {"M1": 1800, "M5": 900, "M15": 700, "H1": 700, "H4": 500}
            market_snapshot: dict[str, dict[str, float]] = {}
            symbol_statuses: list[SymbolStatus] = []
            symbol_contexts: dict[str, dict[str, Any]] = {}

            for configured_symbol in configured_symbols:
                resolved_symbol = resolved_symbols[configured_symbol]
                normalized_symbol = _normalize_symbol_key(resolved_symbol)
                symbol_always_on = _is_always_on_symbol(configured_symbol, resolved_symbol, always_on_symbols)
                symbol_session_multiplier = global_session_multiplier
                timeframe_route = _resolve_timeframe_route(
                    resolved_symbol,
                    requested_timeframe_default,
                    bridge_orchestrator,
                )
                market_state = _market_state_fields(resolved_symbol, now)
                market_open = bool(market_state["market_open"])
                market_status = str(market_state["market_open_status"])
                session_allowed_for_entries = bool(symbol_session_multiplier > 0)
                session_block_reason = "outside_configured_session"
                engine_name = f"{normalized_symbol}_ROUTER"
                if normalized_symbol == "XAUUSD":
                    engine_name = "XAU_MULTI"
                    xau_allowed_sessions = set(strategy_router.xau_active_sessions)
                    if getattr(grid_scalper, "enabled", False):
                        xau_allowed_sessions |= {str(item).upper() for item in getattr(grid_scalper, "allowed_sessions", ())}
                    session_allowed_for_entries = bool(session_context.session_name in xau_allowed_sessions)
                    session_block_reason = "xau_session_block"
                symbol_eval_interval = _effective_symbol_interval_seconds(
                    symbol=resolved_symbol,
                    session_name=session_context.session_name,
                    fast_symbols=fast_symbols,
                    fast_sessions=fast_sessions,
                    fast_seconds=fast_symbol_interval,
                    default_seconds=default_symbol_interval,
                )
                symbol_eval_key = resolved_symbol.upper()
                due_at = next_symbol_eval_at.get(symbol_eval_key)
                open_for_symbol = [position for position in open_positions_journal if str(position.get("symbol", "")).upper() == resolved_symbol.upper()]
                symbol_status = SymbolStatus(
                    symbol=resolved_symbol,
                    regime="UNKNOWN",
                    news="PENDING",
                    trading_allowed=False,
                    reason="initializing",
                    open_positions=len(open_for_symbol),
                    max_positions=effective_max_positions_per_symbol,
                    last_signal="NONE",
                    last_score=0.0,
                    queued_actions=queued_counts.get(resolved_symbol.upper(), 0),
                    ai_reason="pending",
                    current_state="OPEN_POSITION" if open_for_symbol else "SCANNING",
                    engine=engine_name,
                    market_open_status=market_status,
                    eligible_session="YES" if session_allowed_for_entries else "NO",
                    session_allowed="YES" if (market_open and session_allowed_for_entries) else "NO",
                    requested_timeframe=str(timeframe_route.get("requested_timeframe", requested_timeframe_default)),
                    execution_timeframe_used=str(timeframe_route.get("execution_timeframe_used", requested_timeframe_default)),
                    internal_timeframes_used=list(timeframe_route.get("internal_timeframes_used", [])),
                    attachment_dependency_resolved=bool(timeframe_route.get("attachment_dependency_resolved", False)),
                    delivered_actions=bridge_delivery_counts.get(normalized_symbol, 0),
                )
                if not market_open and not symbol_always_on:
                    router_diagnostics: dict[str, Any] = {}
                    news_snapshot: dict[str, Any] = {}
                    runtime_market_data_mode = ""
                    runtime_market_data_source = ""
                    runtime_market_data_consensus_state = ""
                    runtime_market_data_ready = False
                    runtime_market_data_error = ""
                    try:
                        news_snapshot = news_engine.status_snapshot(configured_symbol, now)
                        bridge_symbol_snapshot = None
                        if bridge_trade_mode and bridge_context_account and bridge_context_magic:
                            bridge_symbol_snapshot = _resolve_bridge_symbol_snapshot(
                                bridge_queue,
                                account=bridge_context_account,
                                symbol=resolved_symbol,
                                magic=bridge_context_magic,
                            )
                        frames, _ = _load_symbol_frames(market_data, resolved_symbol, counts, dry_run)
                        if frames is not None and bridge_symbol_snapshot is not None:
                            for timeframe_key in ("M1", "M5", "M15"):
                                if timeframe_key in frames:
                                    frames[timeframe_key] = _refresh_frame_with_bridge_quote(
                                        frames[timeframe_key],
                                        timeframe=timeframe_key,
                                        bridge_symbol_snapshot=bridge_symbol_snapshot,
                                        now_utc=now,
                                    )
                        market_data_status = market_data.status_for_symbol(resolved_symbol)
                        runtime_market_data_mode = str(market_data_status.get("runtime_market_data_mode", ""))
                        runtime_market_data_source = str(market_data_status.get("runtime_market_data_source", ""))
                        runtime_market_data_consensus_state = str(
                            market_data_status.get("runtime_market_data_consensus_state", "")
                        )
                        runtime_market_data_ready = bool(market_data_status.get("runtime_market_data_ready"))
                        runtime_market_data_error = str(market_data_status.get("runtime_market_data_error", ""))
                        if frames is not None:
                            features = feature_engineer.build(
                                frames.get("M1"),
                                frames["M5"],
                                frames["M15"],
                                frames["H1"],
                                frames.get("H4"),
                            )
                            if not features.empty:
                                row = features.iloc[-1]
                                regime = regime_detector.classify(row)
                                current_regime_state = str(getattr(regime, "state_label", regime.label) or regime.label).upper()
                                previous_regime_state = str(symbol_runtime_metrics.get(normalized_symbol, {}).get("regime_state", ""))
                                if current_regime_state and current_regime_state != previous_regime_state:
                                    logger.info(
                                        "regime_transition",
                                        extra={
                                            "extra_fields": {
                                                "symbol": resolved_symbol,
                                                "regime_state": current_regime_state,
                                                "regime_confidence": float(getattr(regime, "state_confidence", 0.0) or regime.details.get("regime_state_confidence", 0.0)),
                                                "previous_state": previous_regime_state,
                                            }
                                        },
                                    )
                                learning_policy: dict[str, Any] = {}
                                brain = runtime.get("learning_brain")
                                if brain is not None and hasattr(brain, "live_policy_snapshot"):
                                    try:
                                        learning_policy = brain.live_policy_snapshot(symbol=resolved_symbol)
                                    except Exception:
                                        learning_policy = {}
                                learning_policy = _augment_learning_policy_for_density(
                                    symbol_key=resolved_symbol,
                                    session_name=session_context.session_name,
                                    learning_policy=learning_policy,
                                    current_scaling_state=current_scaling_state,
                                )
                                strategy_router.apply_learning_policy(learning_policy)
                                if normalized_symbol == "XAUUSD" and getattr(grid_scalper, "enabled", False):
                                    grid_scalper.apply_learning_policy(learning_policy)
                                router_diagnostics = strategy_router.diagnostics(
                                    symbol=resolved_symbol,
                                    row=row,
                                    regime=regime,
                                    session=session_context,
                                    timestamp=now,
                                )
                    except Exception:
                        router_diagnostics = {}
                        news_snapshot = {}
                    previous_market_status = market_closed_log_state.get(symbol_eval_key)
                    if previous_market_status != market_status:
                        logger.info(
                            "MARKET_CLOSED_BLOCK",
                            extra={
                                "extra_fields": {
                                    "symbol": resolved_symbol,
                                    "status": market_status,
                                    "session": session_context.session_name,
                                }
                            },
                        )
                        market_closed_log_state[symbol_eval_key] = market_status
                    _record_symbol_activity(resolved_symbol, "market_closed_blocks", now)
                    symbol_status.current_state = "CLOSED"
                    symbol_status.reason = market_status.lower()
                    symbol_status.last_block_reason = market_status.lower()
                    symbol_status.news = "CLOSED"
                    _update_runtime_metrics(
                        resolved_symbol,
                        now,
                        market_open_status=market_status,
                        next_open_time_utc=str(market_state.get("next_open_time_utc", "")),
                        next_open_time_local=str(market_state.get("next_open_time_local", "")),
                        dst_mode_active=dict(market_state.get("dst_mode_active", {})),
                        pre_open_checks_complete=_prep_checks_complete(news_snapshot, router_diagnostics, market_open),
                        pre_open_news_summary=str(news_snapshot.get("pre_open_news_summary", "")),
                        pre_open_risk_notes=dict(news_snapshot.get("pre_open_risk_notes", {})),
                        pre_open_setup_windows=list(router_diagnostics.get("active_setup_windows", [])),
                        public_proxy_availability=dict(news_snapshot.get("public_proxy_availability", {})),
                        macro_event_bias=dict(news_snapshot.get("macro_event_bias", {})),
                        session_bias_summary=str(news_snapshot.get("session_bias_summary", "")),
                        news_refresh_at=str(news_snapshot.get("news_refresh_at", "")),
                        next_macro_event=news_snapshot.get("next_macro_event"),
                        event_risk_window_active=bool(news_snapshot.get("event_risk_window_active", False)),
                        post_news_trade_window_active=bool(news_snapshot.get("post_news_trade_window_active", False)),
                        news_bias_direction=str(news_snapshot.get("news_bias_direction", "neutral")),
                        news_confidence=float(news_snapshot.get("news_confidence", 0.0) or 0.0),
                        news_data_quality=str(news_snapshot.get("news_data_quality", "unknown")),
                        news_headlines=list(news_snapshot.get("news_headlines", [])),
                        news_source_breakdown=dict(news_snapshot.get("news_source_breakdown", {})),
                        news_category_summary=dict(news_snapshot.get("news_category_summary", {})),
                        news_primary_category=str(news_snapshot.get("news_primary_category", "general_macro")),
                        news_secondary_source_used=bool(news_snapshot.get("news_secondary_source_used", False)),
                        news_rss_headlines=list(news_snapshot.get("news_rss_headlines", [])),
                        news_rss_headline_count=int(news_snapshot.get("news_rss_headline_count", 0) or 0),
                        requested_timeframe=symbol_status.requested_timeframe,
                        execution_timeframe_used=symbol_status.execution_timeframe_used,
                        internal_timeframes_used=list(symbol_status.internal_timeframes_used),
                        attachment_dependency_resolved=bool(symbol_status.attachment_dependency_resolved),
                        delivered_actions_last_15m=int(symbol_status.delivered_actions),
                        session_policy_current=str(router_diagnostics.get("session_policy_current", "")),
                        active_setup_windows=list(router_diagnostics.get("active_setup_windows", [])),
                        setup_proxy_availability=dict(router_diagnostics.get("setup_proxy_availability", {})),
                        weekend_vs_weekday_btc_mode=router_diagnostics.get("weekend_vs_weekday_btc_mode"),
                        funding_proxy_available=router_diagnostics.get("funding_proxy_available"),
                        liquidation_proxy_available=router_diagnostics.get("liquidation_proxy_available"),
                        whale_flow_proxy_available=router_diagnostics.get("whale_flow_proxy_available"),
                        dxy_proxy_available=router_diagnostics.get("dxy_proxy_available"),
                        weekend_gap_proxy_available=router_diagnostics.get("weekend_gap_proxy_available"),
                        xau_modes_enabled=router_diagnostics.get("xau_modes_enabled"),
                        fix_window_active=router_diagnostics.get("fix_window_active"),
                        proxy_unavailable_fallback_used=bool(router_diagnostics.get("proxy_unavailable_fallback_used", False)),
                        btc_reason_no_candidate=str(router_diagnostics.get("btc_reason_no_candidate", "")),
                        btc_last_structure_state=str(router_diagnostics.get("btc_last_structure_state", "")),
                        btc_last_volatility_state=str(router_diagnostics.get("btc_last_volatility_state", "")),
                        btc_last_spread_state=str(router_diagnostics.get("btc_last_spread_state", "")),
                        runtime_market_data_mode=runtime_market_data_mode,
                        runtime_market_data_source=runtime_market_data_source,
                        runtime_market_data_consensus_state=runtime_market_data_consensus_state,
                        runtime_market_data_ready=runtime_market_data_ready,
                        runtime_market_data_error=runtime_market_data_error,
                        hourly_learning_summary=dict(hourly_learning_state.get(normalized_symbol, {}).get("hourly_learning_summary", {})),
                        pair_hourly_review=dict(hourly_learning_state.get(normalized_symbol, {}).get("pair_hourly_review", {})),
                        recent_missed_opportunity_summary=str(hourly_learning_state.get(normalized_symbol, {}).get("recent_missed_opportunity_summary", "")),
                        hourly_parameter_adjustments=dict(hourly_learning_state.get(normalized_symbol, {}).get("hourly_parameter_adjustments", {})),
                        setup_families_promoted=list(hourly_learning_state.get(normalized_symbol, {}).get("setup_families_promoted", [])),
                        setup_families_suppressed=list(hourly_learning_state.get(normalized_symbol, {}).get("setup_families_suppressed", [])),
                    )
                    symbol_statuses.append(symbol_status)
                    cached_symbol_status[symbol_eval_key] = symbol_status
                    continue
                market_closed_log_state.pop(symbol_eval_key, None)
                if due_at and now < due_at:
                    cached = cached_symbol_status.get(symbol_eval_key)
                    if cached is not None:
                        symbol_status = replace(
                            cached,
                            open_positions=len(open_for_symbol),
                            max_positions=effective_max_positions_per_symbol,
                            queued_actions=queued_counts.get(resolved_symbol.upper(), 0),
                            delivered_actions=bridge_delivery_counts.get(normalized_symbol, cached.delivered_actions),
                        )
                    else:
                        symbol_status.reason = "cadence_wait"
                    symbol_statuses.append(symbol_status)
                    continue
                next_symbol_eval_at[symbol_eval_key] = now + timedelta(seconds=symbol_eval_interval)
                try:
                    news_snapshot = news_engine.status_snapshot(configured_symbol, now)
                    news_decision = news_snapshot["decision"]
                    bridge_symbol_snapshot = None
                    if bridge_trade_mode and bridge_context_account and bridge_context_magic:
                        bridge_symbol_snapshot = _resolve_bridge_symbol_snapshot(
                            bridge_queue,
                            account=bridge_context_account,
                            symbol=resolved_symbol,
                            magic=bridge_context_magic,
                        )
                    frames, load_reason = _load_symbol_frames(market_data, resolved_symbol, counts, dry_run)
                    if frames is not None and bridge_symbol_snapshot is not None:
                        for timeframe_key in ("M1", "M5", "M15"):
                            if timeframe_key in frames:
                                frames[timeframe_key] = _refresh_frame_with_bridge_quote(
                                    frames[timeframe_key],
                                    timeframe=timeframe_key,
                                    bridge_symbol_snapshot=bridge_symbol_snapshot,
                                    now_utc=now,
                                )
                    market_data_status = market_data.status_for_symbol(resolved_symbol)
                    symbol_status.runtime_market_data_mode = str(market_data_status.get("runtime_market_data_mode", ""))
                    symbol_status.runtime_market_data_source = str(market_data_status.get("runtime_market_data_source", ""))
                    runtime_market_data_consensus_state = str(
                        market_data_status.get("runtime_market_data_consensus_state", "")
                    )
                    symbol_status.runtime_market_data_ready = "YES" if bool(market_data_status.get("runtime_market_data_ready")) else "NO"
                    symbol_status.runtime_market_data_error = str(market_data_status.get("runtime_market_data_error", ""))
                    if frames is None:
                        symbol_status.reason = load_reason or symbol_status.runtime_market_data_error or "no cache yet"
                        symbol_status.current_state = "BLOCK"
                        symbol_status.last_block_reason = symbol_status.reason
                        _update_runtime_metrics(
                            resolved_symbol,
                            now,
                            runtime_market_data_mode=symbol_status.runtime_market_data_mode,
                            runtime_market_data_source=symbol_status.runtime_market_data_source,
                            runtime_market_data_consensus_state=runtime_market_data_consensus_state,
                            runtime_market_data_ready=bool(market_data_status.get("runtime_market_data_ready")),
                            runtime_market_data_error=symbol_status.runtime_market_data_error,
                        )
                        symbol_statuses.append(symbol_status)
                        cached_symbol_status[symbol_eval_key] = symbol_status
                        continue
                    features = feature_engineer.build(frames.get("M1"), frames["M5"], frames["M15"], frames["H1"], frames.get("H4"))
                    if features.empty:
                        symbol_status.reason = "no_features_available"
                        symbol_status.current_state = "BLOCK"
                        symbol_status.last_block_reason = "no_features_available"
                        symbol_statuses.append(symbol_status)
                        cached_symbol_status[symbol_eval_key] = symbol_status
                        continue
                    row = features.iloc[-1]
                    regime = regime_detector.classify(row)
                    current_regime_state = str(getattr(regime, "state_label", regime.label) or regime.label).upper()
                    previous_regime_state = str(symbol_runtime_metrics.get(normalized_symbol, {}).get("regime_state", ""))
                    if current_regime_state and current_regime_state != previous_regime_state:
                        logger.info(
                            "regime_transition",
                            extra={
                                "extra_fields": {
                                    "symbol": resolved_symbol,
                                    "regime_state": current_regime_state,
                                    "regime_confidence": float(getattr(regime, "state_confidence", 0.0) or regime.details.get("regime_state_confidence", 0.0)),
                                    "previous_state": previous_regime_state,
                                }
                            },
                        )
                    learning_policy: dict[str, Any] = {}
                    brain = runtime.get("learning_brain")
                    if brain is not None and hasattr(brain, "live_policy_snapshot"):
                        try:
                            learning_policy = brain.live_policy_snapshot(symbol=resolved_symbol)
                        except Exception:
                            learning_policy = {}
                    learning_policy = _augment_learning_policy_for_density(
                        symbol_key=resolved_symbol,
                        session_name=session_context.session_name,
                        learning_policy=learning_policy,
                        current_scaling_state=current_scaling_state,
                    )
                    strategy_router.apply_learning_policy(learning_policy)
                    if normalized_symbol == "XAUUSD" and getattr(grid_scalper, "enabled", False):
                        grid_scalper.apply_learning_policy(learning_policy)
                    router_diagnostics = strategy_router.diagnostics(
                        symbol=resolved_symbol,
                        row=row,
                        regime=regime,
                        session=session_context,
                        timestamp=now,
                    )
                    row = row.copy()
                    microstructure_config = (
                        bridge_orchestrator_config.get("microstructure", {})
                        if isinstance(bridge_orchestrator_config.get("microstructure"), dict)
                        else {}
                    )
                    lead_lag_config = (
                        bridge_orchestrator_config.get("lead_lag", {})
                        if isinstance(bridge_orchestrator_config.get("lead_lag"), dict)
                        else {}
                    )
                    event_playbook_config = (
                        bridge_orchestrator_config.get("event_playbooks", {})
                        if isinstance(bridge_orchestrator_config.get("event_playbooks"), dict)
                        else {}
                    )
                    execution_memory_config = (
                        bridge_orchestrator_config.get("execution_memory", {})
                        if isinstance(bridge_orchestrator_config.get("execution_memory"), dict)
                        else {}
                    )
                    microstructure_symbols = {
                        _normalize_symbol_key(item)
                        for item in microstructure_config.get("symbols", ["XAUUSD", "BTCUSD"])
                        if str(item).strip()
                    }
                    microstructure_enabled = bool(microstructure_config.get("enabled", True))
                    microstructure_supported = microstructure_enabled and normalized_symbol in microstructure_symbols
                    microstructure: dict[str, Any] = {
                        "ready": False,
                        "direction": "neutral",
                        "confidence": 0.0,
                        "composite_score": 0.5,
                        "directional_bias": 0.0,
                        "alignment_score": 0.0,
                        "pressure_score": 0.0,
                        "cumulative_delta_score": 0.0,
                        "depth_imbalance": 0.0,
                        "dom_imbalance": 0.0,
                        "drift_score": 0.0,
                        "spread_stability": 0.0,
                        "sweep_velocity": 0.0,
                        "absorption_score": 0.0,
                        "iceberg_score": 0.0,
                        "spread_shock_score": 0.0,
                        "quote_pull_stack_score": 0.0,
                        "tick_count": 0,
                        "book_levels": 0,
                    }
                    if dry_run or verify_only or paper_sim:
                        reference_price = float(row["m5_close"])
                        tick = {"bid": reference_price, "ask": reference_price, "time": int(now.timestamp())}
                    else:
                        try:
                            tick = mt5_client.get_tick(resolved_symbol)
                            reference_price = float((float(tick["bid"]) + float(tick["ask"])) / 2)
                        except Exception as exc:
                            reference_price = float(row["m5_close"])
                            tick = {"bid": reference_price, "ask": reference_price, "time": int(now.timestamp())}
                            _emit_runtime_alert(
                                f"tick_fallback::{resolved_symbol}",
                                "runtime_tick_fetch_fallback",
                                min_interval_seconds=600.0,
                                now_ts=now,
                                extra_fields={
                                    "symbol": resolved_symbol,
                                    "reason": str(exc),
                                    "bridge_trade_mode": bool(bridge_trade_mode),
                                },
                            )
                        if account_from_mt5 and market_open and microstructure_supported:
                            try:
                                fetched_microstructure = mt5_client.get_microstructure_snapshot(resolved_symbol, tick_count=96)
                                if isinstance(fetched_microstructure, dict):
                                    microstructure.update(fetched_microstructure)
                            except Exception as exc:
                                _emit_runtime_alert(
                                    f"microstructure_unavailable::{resolved_symbol}",
                                    "runtime_microstructure_unavailable",
                                    min_interval_seconds=1200.0,
                                    now_ts=now,
                                    extra_fields={"symbol": resolved_symbol, "reason": str(exc)},
                                )
                    microstructure_score = build_microstructure_score(microstructure, symbol=resolved_symbol)
                    lead_lag_snapshot = build_lead_lag_snapshot(
                        symbol=resolved_symbol,
                        context={
                            "dxy_ret_1": _safe_float(row.get("dxy_ret_1"), 0.0),
                            "dxy_ret_5": _safe_float(row.get("dxy_ret_5"), 0.0),
                            "us10y_ret_5": _safe_float(row.get("us10y_ret_5"), _safe_float(row.get("yield_proxy_ret_5"), 0.0)),
                            "yield_proxy_ret_5": _safe_float(row.get("yield_proxy_ret_5"), 0.0),
                            "nas100_ret_5": _safe_float(row.get("nas100_ret_5"), _safe_float(row.get("nas_ret_5"), 0.0)),
                            "nas_ret_5": _safe_float(row.get("nas_ret_5"), 0.0),
                            "usoil_ret_5": _safe_float(row.get("usoil_ret_5"), _safe_float(row.get("oil_ret_5"), 0.0)),
                            "oil_ret_5": _safe_float(row.get("oil_ret_5"), 0.0),
                            "eurusd_ret_5": _safe_float(row.get("eurusd_ret_5"), 0.0),
                            "usdjpy_ret_5": _safe_float(row.get("usdjpy_ret_5"), 0.0),
                            "usd_liquidity_score": _safe_float(router_diagnostics.get("usd_liquidity_score"), 0.0),
                            "weekend_volatility_score": _safe_float(router_diagnostics.get("weekend_volatility_score"), 0.0),
                            "btc_weekend_gap_score": _safe_float(router_diagnostics.get("btc_weekend_gap_score"), 0.0),
                        },
                        weights_config=(
                            lead_lag_config.get("weights_by_symbol", {})
                            if isinstance(lead_lag_config.get("weights_by_symbol"), dict)
                            else {}
                        ),
                    )
                    event_directive = build_event_directive(
                        symbol=resolved_symbol,
                        news_snapshot=news_snapshot,
                        lead_lag=lead_lag_snapshot,
                        microstructure=microstructure_score,
                        playbook_map=(
                            event_playbook_config.get("playbook_map", {})
                            if isinstance(event_playbook_config.get("playbook_map"), dict)
                            else {}
                        ),
                    )
                    execution_memory_profile = build_execution_minute_profile(
                        now_utc=now,
                        runtime={
                            "session_name": session_context.session_name,
                            "spread_quality_score": _safe_float(
                                symbol_runtime_metrics.get(normalized_symbol, {}).get("spread_quality_score"),
                                0.70,
                            ),
                            "execution_quality_score": _safe_float(
                                symbol_runtime_metrics.get(normalized_symbol, {}).get("execution_quality_score"),
                                0.70,
                            ),
                            "slippage_quality_score": _safe_float(
                                symbol_runtime_metrics.get(normalized_symbol, {}).get("slippage_quality_score"),
                                0.70,
                            ),
                        },
                        management_feedback=dict(symbol_runtime_metrics.get(normalized_symbol, {})),
                    )
                    row["microstructure_ready"] = bool(microstructure_score.ready)
                    row["microstructure_direction"] = str(microstructure_score.direction)
                    row["microstructure_confidence"] = float(microstructure_score.confidence)
                    row["microstructure_composite_score"] = float(microstructure_score.composite_score)
                    row["microstructure_directional_bias"] = float(microstructure_score.directional_bias)
                    row["microstructure_alignment_score"] = float(microstructure_score.alignment_score)
                    row["microstructure_pressure_score"] = float(microstructure_score.pressure_score)
                    row["microstructure_cumulative_delta_score"] = float(microstructure_score.cumulative_delta_score)
                    row["microstructure_depth_imbalance"] = float(microstructure_score.depth_imbalance)
                    row["microstructure_dom_imbalance"] = float(microstructure_score.dom_imbalance)
                    row["microstructure_drift_score"] = float(microstructure_score.drift_score)
                    row["microstructure_spread_stability"] = float(microstructure_score.spread_stability)
                    row["microstructure_sweep_velocity"] = float(microstructure_score.sweep_velocity)
                    row["microstructure_absorption_score"] = float(microstructure_score.absorption_score)
                    row["microstructure_iceberg_score"] = float(microstructure_score.iceberg_score)
                    row["microstructure_spread_shock_score"] = float(microstructure_score.spread_shock_score)
                    row["microstructure_quote_pull_stack_score"] = float(microstructure_score.quote_pull_stack_score)
                    row["lead_lag_direction"] = str(lead_lag_snapshot.direction)
                    row["lead_lag_confidence"] = float(lead_lag_snapshot.confidence)
                    row["lead_lag_alignment_score"] = float(lead_lag_snapshot.alignment_score)
                    row["lead_lag_disagreement_penalty"] = float(lead_lag_snapshot.disagreement_penalty)
                    row["event_playbook"] = str(event_directive.playbook)
                    row["event_base_class"] = str(event_directive.base_class)
                    row["event_sub_class"] = str(event_directive.sub_class)
                    row["event_pre_position_allowed"] = bool(event_directive.pre_position_allowed)
                    row["execution_minute_quality_score"] = float(execution_memory_profile.quality_score)
                    row["execution_minute_size_multiplier"] = float(execution_memory_profile.size_multiplier)
                    row["execution_minute_state"] = str(execution_memory_profile.state)

                    market_snapshot[resolved_symbol] = {
                        "price": reference_price,
                        "bid": float(tick["bid"]),
                        "ask": float(tick["ask"]),
                        "atr": float(row["m5_atr_14"]),
                    }
                    symbol_contexts[configured_symbol] = {
                        "resolved_symbol": resolved_symbol,
                        "symbol_info": dict(
                            _resolve_runtime_symbol_info(
                                mt5_client=mt5_client,
                                bridge_queue=bridge_queue,
                                symbol_info_cache=symbol_info_cache,
                                symbol=resolved_symbol,
                                bridge_trade_mode=bridge_trade_mode,
                                bridge_context_account=bridge_context_account,
                                bridge_context_magic=bridge_context_magic,
                            )
                        ),
                        "features": features,
                        "row": row,
                        "regime": regime,
                        "tick": tick,
                        "news_decision": news_decision,
                        "session_multiplier": symbol_session_multiplier,
                        "weekend_trading_allowed": symbol_always_on,
                        "market_open": market_open,
                        "market_status": market_status,
                        "timeframe_route": timeframe_route,
                        "session_allowed_for_new_entries": session_allowed_for_entries,
                        "session_block_reason": session_block_reason,
                        "router_diagnostics": router_diagnostics,
                        "symbol_status": symbol_status,
                        "news_snapshot": news_snapshot,
                        "market_state": market_state,
                        "microstructure": dict(microstructure),
                        "microstructure_score": microstructure_score.as_dict(),
                        "lead_lag_snapshot": lead_lag_snapshot.as_dict(),
                        "event_directive": event_directive.as_dict(),
                        "execution_minute_profile": execution_memory_profile.as_dict(),
                    }
                    symbol_status.regime = regime.label
                    news_state = str(getattr(news_decision, "state", "NEWS_SAFE" if news_decision.safe else "NEWS_BLOCKED")).upper()
                    if news_state == "NEWS_CAUTION":
                        symbol_status.news = "CAUTION"
                    elif news_decision.safe:
                        symbol_status.news = "SAFE"
                    else:
                        symbol_status.news = f"BLOCKED({news_decision.source})"
                    symbol_status.current_state = "SCANNING"
                    if not news_decision.safe:
                        symbol_status.current_state = "PRECHECK_FAIL"
                        symbol_status.reason = news_decision.reason
                        symbol_status.last_block_reason = news_decision.reason
                    elif not session_allowed_for_entries:
                        symbol_status.current_state = "SESSION_BLOCK"
                        symbol_status.reason = session_block_reason
                        symbol_status.last_block_reason = session_block_reason
                    else:
                        symbol_status.reason = "scanning"
                    _update_runtime_metrics(
                        resolved_symbol,
                        now,
                        market_open_status=market_status,
                        eligible_session=bool(session_allowed_for_entries),
                        session_allowed=bool(market_open and session_allowed_for_entries),
                        requested_timeframe=symbol_status.requested_timeframe,
                        execution_timeframe_used=symbol_status.execution_timeframe_used,
                        internal_timeframes_used=list(symbol_status.internal_timeframes_used),
                        attachment_dependency_resolved=bool(symbol_status.attachment_dependency_resolved),
                        delivered_actions_last_15m=int(symbol_status.delivered_actions),
                        next_open_time_utc=str(market_state.get("next_open_time_utc", "")),
                        next_open_time_local=str(market_state.get("next_open_time_local", "")),
                        dst_mode_active=dict(market_state.get("dst_mode_active", {})),
                        pre_open_checks_complete=_prep_checks_complete(news_snapshot, router_diagnostics, market_open),
                        pre_open_news_summary=str(news_snapshot.get("pre_open_news_summary", "")),
                        pre_open_risk_notes=dict(news_snapshot.get("pre_open_risk_notes", {})),
                        pre_open_setup_windows=list(router_diagnostics.get("active_setup_windows", [])),
                        public_proxy_availability=dict(news_snapshot.get("public_proxy_availability", {})),
                        macro_event_bias=dict(news_snapshot.get("macro_event_bias", {})),
                        session_bias_summary=str(news_snapshot.get("session_bias_summary", "")),
                        news_refresh_at=str(news_snapshot.get("news_refresh_at", "")),
                        news_state=str(news_snapshot.get("news_state", news_state)),
                        news_source_used=str(news_snapshot.get("news_source_used", "")),
                        news_fallback_used=bool(news_snapshot.get("news_fallback_used", False)),
                        news_decision_confidence=float(news_snapshot.get("news_decision_confidence", 0.0) or 0.0),
                        news_decision_reason=str(news_snapshot.get("news_decision_reason", "")),
                        news_next_event_time=str(news_snapshot.get("news_next_event_time", "")),
                        news_next_event_impact=str(news_snapshot.get("news_next_event_impact", "")),
                        next_macro_event=news_snapshot.get("next_macro_event"),
                        event_risk_window_active=bool(news_snapshot.get("event_risk_window_active", False)),
                        post_news_trade_window_active=bool(news_snapshot.get("post_news_trade_window_active", False)),
                        news_bias_direction=str(news_snapshot.get("news_bias_direction", "neutral")),
                        news_confidence=float(news_snapshot.get("news_confidence", 0.0) or 0.0),
                        news_data_quality=str(news_snapshot.get("news_data_quality", "unknown")),
                        news_headlines=list(news_snapshot.get("news_headlines", [])),
                        news_source_breakdown=dict(news_snapshot.get("news_source_breakdown", {})),
                        news_category_summary=dict(news_snapshot.get("news_category_summary", {})),
                        news_primary_category=str(news_snapshot.get("news_primary_category", "general_macro")),
                        news_secondary_source_used=bool(news_snapshot.get("news_secondary_source_used", False)),
                        news_rss_headlines=list(news_snapshot.get("news_rss_headlines", [])),
                        news_rss_headline_count=int(news_snapshot.get("news_rss_headline_count", 0) or 0),
                        session_policy_current=str(router_diagnostics.get("session_policy_current", "")),
                        active_setup_windows=list(router_diagnostics.get("active_setup_windows", [])),
                        setup_proxy_availability=dict(router_diagnostics.get("setup_proxy_availability", {})),
                        weekend_vs_weekday_btc_mode=router_diagnostics.get("weekend_vs_weekday_btc_mode"),
                        funding_proxy_available=router_diagnostics.get("funding_proxy_available"),
                        liquidation_proxy_available=router_diagnostics.get("liquidation_proxy_available"),
                        whale_flow_proxy_available=router_diagnostics.get("whale_flow_proxy_available"),
                        dxy_proxy_available=router_diagnostics.get("dxy_proxy_available"),
                        weekend_gap_proxy_available=router_diagnostics.get("weekend_gap_proxy_available"),
                        xau_modes_enabled=router_diagnostics.get("xau_modes_enabled"),
                        fix_window_active=router_diagnostics.get("fix_window_active"),
                        proxy_unavailable_fallback_used=bool(router_diagnostics.get("proxy_unavailable_fallback_used", False)),
                        btc_reason_no_candidate=str(router_diagnostics.get("btc_reason_no_candidate", "")),
                        btc_last_structure_state=str(router_diagnostics.get("btc_last_structure_state", "")),
                        btc_last_volatility_state=str(router_diagnostics.get("btc_last_volatility_state", "")),
                        btc_last_spread_state=str(router_diagnostics.get("btc_last_spread_state", "")),
                        runtime_market_data_mode=symbol_status.runtime_market_data_mode,
                        runtime_market_data_source=symbol_status.runtime_market_data_source,
                        runtime_market_data_ready=bool(market_data_status.get("runtime_market_data_ready")),
                        runtime_market_data_error=symbol_status.runtime_market_data_error,
                        microstructure_ready=bool(microstructure_score.ready),
                        microstructure_direction=str(microstructure_score.direction),
                        microstructure_confidence=float(microstructure_score.confidence),
                        microstructure_composite_score=float(microstructure_score.composite_score),
                        microstructure_directional_bias=float(microstructure_score.directional_bias),
                        microstructure_alignment_score=float(microstructure_score.alignment_score),
                        microstructure_pressure_score=float(microstructure_score.pressure_score),
                        microstructure_cumulative_delta_score=float(microstructure_score.cumulative_delta_score),
                        microstructure_depth_imbalance=float(microstructure_score.depth_imbalance),
                        microstructure_dom_imbalance=float(microstructure_score.dom_imbalance),
                        microstructure_drift_score=float(microstructure_score.drift_score),
                        microstructure_spread_stability=float(microstructure_score.spread_stability),
                        microstructure_sweep_velocity=float(microstructure_score.sweep_velocity),
                        microstructure_absorption_score=float(microstructure_score.absorption_score),
                        microstructure_iceberg_score=float(microstructure_score.iceberg_score),
                        microstructure_spread_shock_score=float(microstructure_score.spread_shock_score),
                        microstructure_quote_pull_stack_score=float(microstructure_score.quote_pull_stack_score),
                        lead_lag_snapshot=lead_lag_snapshot.as_dict(),
                        lead_lag_direction=str(lead_lag_snapshot.direction),
                        lead_lag_confidence=float(lead_lag_snapshot.confidence),
                        lead_lag_alignment_score=float(lead_lag_snapshot.alignment_score),
                        lead_lag_disagreement_penalty=float(lead_lag_snapshot.disagreement_penalty),
                        event_directive=event_directive.as_dict(),
                        event_playbook=str(event_directive.playbook),
                        event_base_class=str(event_directive.base_class),
                        event_sub_class=str(event_directive.sub_class),
                        event_pre_position_allowed=bool(event_directive.pre_position_allowed),
                        execution_minute_profile=execution_memory_profile.as_dict(),
                        execution_minute_quality_score=float(execution_memory_profile.quality_score),
                        execution_minute_size_multiplier=float(execution_memory_profile.size_multiplier),
                        execution_minute_state=str(execution_memory_profile.state),
                        hourly_learning_summary=dict(hourly_learning_state.get(normalized_symbol, {}).get("hourly_learning_summary", {})),
                        pair_hourly_review=dict(hourly_learning_state.get(normalized_symbol, {}).get("pair_hourly_review", {})),
                        recent_missed_opportunity_summary=str(hourly_learning_state.get(normalized_symbol, {}).get("recent_missed_opportunity_summary", "")),
                        hourly_parameter_adjustments=dict(hourly_learning_state.get(normalized_symbol, {}).get("hourly_parameter_adjustments", {})),
                        setup_families_promoted=list(hourly_learning_state.get(normalized_symbol, {}).get("setup_families_promoted", [])),
                        setup_families_suppressed=list(hourly_learning_state.get(normalized_symbol, {}).get("setup_families_suppressed", [])),
                    )
                    news_data_quality = str(news_snapshot.get("news_data_quality", "unknown")).lower()
                    if news_data_quality in {"degraded", "fallback", "unknown"}:
                        _emit_runtime_alert(
                            f"advisory_degraded::{resolved_symbol}",
                            "advisory_context_degraded",
                            min_interval_seconds=1800.0,
                            now_ts=now,
                            extra_fields={
                                "symbol": resolved_symbol,
                                "news_data_quality": news_data_quality,
                                "news_source_used": str(news_snapshot.get("news_source_used", "")),
                            },
                        )
                except Exception as exc:
                    summary["errors"] += 1
                    symbol_status.reason = f"data_error:{exc}"
                    symbol_status.current_state = "BLOCK"
                symbol_statuses.append(symbol_status)
                cached_symbol_status[symbol_eval_key] = symbol_status

            if trading_enabled and not verify_only and (not bridge_trade_mode) and (not paper_sim):
                try:
                    positions.sync(open_positions_journal, mt5_positions)
                    advice_by_ticket: dict[int, dict[str, Any]] = {}
                    for journal_position in open_positions_journal:
                        ticket_value = journal_position.get("ticket")
                        if ticket_value is None:
                            continue
                        try:
                            ticket = int(ticket_value)
                        except (TypeError, ValueError):
                            continue
                        symbol = str(journal_position.get("symbol", "")).upper()
                        context = next(
                            (
                                item
                                for item in symbol_contexts.values()
                                if str(item.get("resolved_symbol", "")).upper() == symbol
                            ),
                            None,
                        )
                        if context is None:
                            continue
                        news_for_symbol = context.get("news_decision")
                        news_summary = {
                            "safe": bool(getattr(news_for_symbol, "safe", True)),
                            "reason": str(getattr(news_for_symbol, "reason", "unknown")),
                        }
                        learning_policy: dict[str, Any] = {}
                        if "learning_brain" in runtime:
                            brain = runtime.get("learning_brain")
                            if brain is not None and hasattr(brain, "live_policy_snapshot"):
                                try:
                                    learning_policy = brain.live_policy_snapshot(
                                        symbol=symbol,
                                        setup=str(journal_position.get("setup", "")),
                                    )
                                except Exception:
                                    learning_policy = {}
                        learning_policy = _augment_learning_policy_for_density(
                            symbol_key=str(context.get("resolved_symbol") or symbol),
                            session_name=session_context.session_name,
                            learning_policy=learning_policy,
                            current_scaling_state=current_scaling_state,
                        )
                        advice = ai_gate.management_advice(
                            position=journal_position,
                            features=context["row"],
                            regime=str(context["regime"].label),
                            session_name=session_context.session_name,
                            news_summary=news_summary,
                            learning_policy=learning_policy,
                        )
                        advice_by_ticket[ticket] = advice

                    managed_actions = positions.plan_actions(market_snapshot, now, advice_by_ticket=advice_by_ticket)
                    managed_actions.extend(positions.plan_basket_actions(market_snapshot))
                    for action in managed_actions:
                        position = next((item for item in mt5_positions if int(item.get("ticket", 0)) == action.ticket), None)
                        if action.action == "PARTIAL_CLOSE":
                            if position and action.volume and execution.partial_close(position, action.signal_id, action.volume, int(risk_config["max_slippage_points"])):
                                positions.mark_partial(action.ticket)
                                log_debug(f"PARTIAL_CLOSE {action.signal_id} ticket={action.ticket} volume={action.volume}")
                        elif action.action == "MOVE_SL":
                            if action.new_sl is not None and execution.move_protection(action.ticket, action.signal_id, sl=action.new_sl):
                                positions.update_sl(action.ticket, action.new_sl)
                                log_debug(f"MOVE_SL {action.signal_id} ticket={action.ticket} sl={action.new_sl}")
                        elif action.action == "TIME_STOP" and position:
                            if mt5_client.close_position(position, int(risk_config["max_slippage_points"])):
                                journal.log_event(action.signal_id, "TIME_STOP_CLOSE", {"ticket": action.ticket})
                                log_debug(f"TIME_STOP_CLOSE {action.signal_id} ticket={action.ticket}")
                        elif action.action == "CLOSE_NOW" and position:
                            if mt5_client.close_position(position, int(risk_config["max_slippage_points"])):
                                journal.log_event(action.signal_id, "AI_CLOSE_NOW", {"ticket": action.ticket, "reason": action.reason})
                                log_debug(f"AI_CLOSE_NOW {action.signal_id} ticket={action.ticket}")
                        elif action.action == "CLOSE_ALL_SYMBOL" and action.symbol:
                            for mt5_position in [item for item in mt5_positions if str(item.get("symbol", "")).upper() == action.symbol.upper()]:
                                if mt5_client.close_position(mt5_position, int(risk_config["max_slippage_points"])):
                                    journal.log_event(action.signal_id, "BASKET_CLOSE", {"symbol": action.symbol, "ticket": int(mt5_position.get("ticket", 0))})
                                    log_debug(f"BASKET_CLOSE symbol={action.symbol}")
                except Exception as exc:
                    summary["errors"] += 1
                    logger.warning(f"Position management degraded: {exc}")

            open_positions_journal = _live_open_positions_for_context(
                account_id=bridge_context_account,
                magic_id=bridge_context_magic,
                now_ts=now,
            )
            loop_band_counts = _empty_band_counts()
            loop_lane_candidate_counts: dict[str, int] = {}
            ordered_configured_symbols = _order_symbols_for_session(
                configured_symbols,
                symbol_contexts,
                session_context.session_name,
            )
            session_native_leader: dict[str, Any] = {}
            account_wide_soft_kill = kill_status.level == "SOFT" and _is_account_wide_soft_kill(kill_status.reason)
            if not account_wide_soft_kill:
                for configured_symbol in ordered_configured_symbols:
                    context = symbol_contexts.get(configured_symbol)
                    if not context:
                        continue
                    resolved_symbol = str(context["resolved_symbol"])
                    features = context["features"]
                    row = context["row"]
                    regime = context["regime"]
                    tick = context["tick"]
                    news_decision = context["news_decision"]
                    symbol_session_multiplier = float(context["session_multiplier"])
                    weekend_trading_allowed = bool(context["weekend_trading_allowed"])
                    market_open = bool(context.get("market_open", True))
                    market_status = str(context.get("market_status", "OPEN"))
                    session_allowed_for_new_entries = bool(context.get("session_allowed_for_new_entries", True))
                    session_block_reason = str(context.get("session_block_reason", "outside_configured_session"))
                    timeframe_route = context.get("timeframe_route", {}) if isinstance(context.get("timeframe_route"), dict) else {}
                    router_diagnostics = context.get("router_diagnostics", {}) if isinstance(context.get("router_diagnostics"), dict) else {}
                    symbol_status = context["symbol_status"]

                    if not market_open:
                        symbol_status.current_state = "CLOSED"
                        symbol_status.reason = market_status.lower()
                        symbol_status.last_block_reason = market_status.lower()
                        _update_runtime_metrics(
                            resolved_symbol,
                            now,
                            market_open_status=market_status,
                            requested_timeframe=str(timeframe_route.get("requested_timeframe", symbol_status.requested_timeframe)),
                            execution_timeframe_used=str(timeframe_route.get("execution_timeframe_used", symbol_status.execution_timeframe_used)),
                            internal_timeframes_used=list(timeframe_route.get("internal_timeframes_used", symbol_status.internal_timeframes_used)),
                            attachment_dependency_resolved=bool(timeframe_route.get("attachment_dependency_resolved", symbol_status.attachment_dependency_resolved)),
                            delivered_actions_last_15m=int(bridge_delivery_counts.get(_normalize_symbol_key(resolved_symbol), symbol_status.delivered_actions)),
                        )
                        continue

                    if symbol_session_multiplier <= 0:
                        _record_symbol_activity(resolved_symbol, "session_ineligible", now)
                        symbol_status.current_state = "SESSION_BLOCK"
                        symbol_status.reason = "outside_configured_session"
                        symbol_status.last_block_reason = "outside_configured_session"
                        _update_runtime_metrics(resolved_symbol, now)
                        continue

                    if not session_allowed_for_new_entries:
                        previous_session_block = session_block_log_state.get(resolved_symbol.upper())
                        if previous_session_block != session_block_reason:
                            logger.info(
                                "SESSION_BLOCK",
                                extra={
                                    "extra_fields": {
                                        "symbol": resolved_symbol,
                                        "reason": session_block_reason,
                                        "session": session_context.session_name,
                                    }
                                },
                            )
                            session_block_log_state[resolved_symbol.upper()] = session_block_reason
                        _record_symbol_activity(resolved_symbol, "session_ineligible", now)
                        symbol_status.current_state = "SESSION_BLOCK"
                        symbol_status.reason = session_block_reason
                        symbol_status.last_block_reason = session_block_reason
                        _update_runtime_metrics(
                            resolved_symbol,
                            now,
                            market_open_status=market_status,
                            eligible_session=False,
                            session_allowed=False,
                            delivered_actions_last_15m=int(bridge_delivery_counts.get(_normalize_symbol_key(resolved_symbol), symbol_status.delivered_actions)),
                        )
                        continue
                    session_block_log_state.pop(resolved_symbol.upper(), None)

                    symbol_open_positions = [position for position in open_positions_journal if str(position["symbol"]).upper() == resolved_symbol.upper()]
                    grid_open_positions = [position for position in symbol_open_positions if _is_xau_grid_setup(str(position.get("setup", "")))]
                    normalized_symbol = _normalize_symbol_key(resolved_symbol)
                    grid_decision = None
                    if normalized_symbol == "XAUUSD" and grid_scalper.enabled:
                        symbol_info = _resolve_runtime_symbol_info(
                            mt5_client=mt5_client,
                            bridge_queue=bridge_queue,
                            symbol_info_cache=symbol_info_cache,
                            symbol=resolved_symbol,
                            bridge_trade_mode=bridge_trade_mode,
                            bridge_context_account=bridge_context_account,
                            bridge_context_magic=bridge_context_magic,
                        )
                        grid_decision = grid_scalper.evaluate(
                            symbol=resolved_symbol,
                            features=features,
                            row=row,
                            open_positions=grid_open_positions,
                            session_name=session_context.session_name,
                            news_safe=bool(news_decision.safe),
                            now_utc=now,
                            spread_points=float(row["m5_spread"]),
                            contract_size=float(symbol_info.get("trade_contract_size", 100.0)),
                            approver=ai_gate.approve_grid_cycle,
                        )
                        if grid_decision.close_cycle and grid_open_positions:
                            if bridge_trade_mode:
                                close_queued = 0
                                for grid_position in grid_open_positions:
                                    ticket_value = grid_position.get("ticket")
                                    if ticket_value in {None, "", 0, "0"}:
                                        continue
                                    close_signal_id = deterministic_id(
                                        resolved_symbol,
                                        "grid",
                                        "close",
                                        str(ticket_value),
                                        now.isoformat(),
                                        grid_decision.close_reason,
                                    )
                                    if bridge_queue.enqueue(
                                        {
                                            "signal_id": close_signal_id,
                                            "action_type": "CLOSE_POSITION",
                                            "ticket": str(ticket_value),
                                            "symbol": resolved_symbol,
                                            "target_ticket": str(ticket_value),
                                            "target_account": str(bridge_context_account or ""),
                                            "target_magic": int(bridge_context_magic or 0) if bridge_context_magic is not None else None,
                                            "side": str(grid_position.get("side", grid_decision.cycle_side or "BUY")).upper(),
                                            "lot": float(grid_position.get("volume", 0.01)),
                                            "sl": max(float(row["m5_close"]) - float(row["m5_atr_14"]), 0.0001),
                                            "tp": max(float(row["m5_close"]) + float(row["m5_atr_14"]), 0.0001),
                                            "max_slippage_points": int(risk_config["max_slippage_points"]),
                                            "reason": grid_decision.close_reason,
                                            "ai_summary": {"rationale": grid_decision.close_reason, "ai_mode": grid_decision.ai_mode},
                                            "mode": mode,
                                            "setup": "XAUUSD_M5_GRID_SCALPER_CLOSE",
                                            "regime": regime.label,
                                            "probability": 1.0,
                                            "expected_value_r": 0.0,
                                            "news_status": news_decision.reason,
                                            "final_decision_json": json.dumps({"close_cycle": True, "reason": grid_decision.close_reason}, sort_keys=True),
                                            "entry_price": float(row["m5_close"]),
                                            "timeframe": "M5",
                                        }
                                    ):
                                        close_queued += 1
                                if close_queued == 0 and len(grid_open_positions) == len(symbol_open_positions):
                                    close_signal_id = deterministic_id(resolved_symbol, "grid", "close_all", now.isoformat(), grid_decision.close_reason)
                                    if bridge_queue.enqueue(
                                        {
                                            "signal_id": close_signal_id,
                                            "action_type": "CLOSE_ALL",
                                            "symbol": resolved_symbol,
                                            "target_ticket": "",
                                            "target_account": str(bridge_context_account or ""),
                                            "target_magic": int(bridge_context_magic or 0) if bridge_context_magic is not None else None,
                                            "side": (grid_decision.cycle_side or "BUY"),
                                            "lot": 0.01,
                                            "sl": max(float(row["m5_close"]) - float(row["m5_atr_14"]), 0.0001),
                                            "tp": max(float(row["m5_close"]) + float(row["m5_atr_14"]), 0.0001),
                                            "max_slippage_points": int(risk_config["max_slippage_points"]),
                                            "reason": grid_decision.close_reason,
                                            "ai_summary": {"rationale": grid_decision.close_reason, "ai_mode": grid_decision.ai_mode},
                                            "mode": mode,
                                            "setup": "XAUUSD_M5_GRID_SCALPER_CLOSE",
                                            "regime": regime.label,
                                            "probability": 1.0,
                                            "expected_value_r": 0.0,
                                            "news_status": news_decision.reason,
                                            "final_decision_json": json.dumps({"close_cycle": True, "reason": grid_decision.close_reason}, sort_keys=True),
                                            "entry_price": float(row["m5_close"]),
                                            "timeframe": "M5",
                                        }
                                    ):
                                        close_queued = 1
                                symbol_status.reason = f"grid_close_queued:{grid_decision.close_reason}:{close_queued}"
                                symbol_status.trading_allowed = False
                                symbol_status.current_state = "QUEUED_FOR_EA" if close_queued > 0 else "BLOCK"
                                if close_queued > 0:
                                    _record_symbol_activity(resolved_symbol, "queued_for_ea", now)
                            elif paper_sim:
                                current_price = float(row["m5_close"])
                                for position in list(grid_open_positions):
                                    side = str(position.get("side", "BUY")).upper()
                                    direction = 1.0 if side == "BUY" else -1.0
                                    entry = float(position.get("entry_price", current_price))
                                    volume = float(position.get("volume", 0.0))
                                    contract_size = float(symbol_info.get("trade_contract_size", 100.0))
                                    pnl_amount = (current_price - entry) * direction * volume * contract_size
                                    risk = max(abs(entry - float(position.get("sl", entry))), 1e-6)
                                    pnl_r = ((current_price - entry) * direction) / risk
                                    journal.mark_closed(
                                        str(position["signal_id"]),
                                        pnl_amount=pnl_amount,
                                        pnl_r=pnl_r,
                                        equity_after_close=float(account["equity"]) + pnl_amount,
                                    )
                                    journal.log_event(
                                        str(position["signal_id"]),
                                        "GRID_CLOSE_ALL",
                                        {"reason": grid_decision.close_reason, "price": current_price},
                                    )
                                open_positions_journal = _live_open_positions_for_context(
                                    account_id=bridge_context_account,
                                    magic_id=bridge_context_magic,
                                    now_ts=now,
                                )
                                symbol_open_positions = [position for position in open_positions_journal if str(position["symbol"]).upper() == resolved_symbol.upper()]
                                grid_open_positions = [position for position in symbol_open_positions if _is_xau_grid_setup(str(position.get("setup", "")))]
                                symbol_status.reason = f"grid_closed:{grid_decision.close_reason}"
                                symbol_status.current_state = "MANAGING_POSITION"
                            else:
                                closed_count = 0
                                for position in list(grid_open_positions):
                                    ticket_value = position.get("ticket")
                                    if ticket_value in {None, "", 0, "0"}:
                                        continue
                                    try:
                                        target_ticket = int(ticket_value)
                                    except (TypeError, ValueError):
                                        continue
                                    mt5_position = next((item for item in mt5_positions if int(item.get("ticket", 0)) == target_ticket), None)
                                    if mt5_position and mt5_client.close_position(mt5_position, int(risk_config["max_slippage_points"])):
                                        closed_count += 1
                                if closed_count == 0 and len(grid_open_positions) == len(symbol_open_positions):
                                    closed_count = execution.hard_flatten(resolved_symbol, int(risk_config["max_slippage_points"]))
                                symbol_status.reason = f"grid_flatten:{grid_decision.close_reason}:{closed_count}"
                                symbol_status.trading_allowed = False
                                symbol_status.current_state = "MANAGING_POSITION"
                            if not grid_decision.candidates:
                                continue
                    router_candidates = strategy_router.generate(
                        symbol=resolved_symbol,
                        features=features,
                        regime=regime,
                        session=session_context,
                        strategy_engine=strategy_engines[configured_symbol],
                        open_positions=symbol_open_positions,
                        max_positions_per_symbol=effective_max_positions_per_symbol,
                        current_time=now,
                    )
                    xau_overlay_seed_candidates: list[SignalCandidate] = []
                    xau_native_live_only = False
                    if normalized_symbol == "XAUUSD" and grid_scalper.enabled:
                        xau_overlay_seed_candidates = [
                            candidate for candidate in router_candidates if _is_xau_higher_tf_candidate_setup(candidate.setup)
                        ]
                        router_candidates = list(xau_overlay_seed_candidates)
                        xau_native_live_only = bool(getattr(grid_scalper, "native_live_only", True))
                        if xau_native_live_only:
                            router_candidates = []
                    raw_grid_candidates: list[SignalCandidate] = []
                    usable_grid_candidates: list[SignalCandidate] = []
                    grid_filtered_profiles: list[str] = []
                    if grid_decision is not None:
                        raw_grid_candidates = list(grid_decision.candidates or [])
                        for candidate in raw_grid_candidates:
                            entry_profile = str(((candidate.meta or {}).get("grid_entry_profile") or "")).strip()
                            if entry_profile.startswith("grid_stretch_reversion"):
                                if entry_profile:
                                    grid_filtered_profiles.append(entry_profile)
                                continue
                            if grid_scalper.native_candidate_is_unsafe_extreme_bucket(
                                candidate=candidate,
                                row=row,
                                session_name=session_context.session_name,
                                regime_state=str(regime.label),
                            ):
                                grid_filtered_profiles.append("grid_native_extreme_bucket")
                                continue
                            usable_grid_candidates.append(candidate)
                    raw_grid_candidate_count = int(len(raw_grid_candidates))
                    usable_grid_candidate_count = int(len(usable_grid_candidates))
                    dropped_grid_candidate_count = max(0, raw_grid_candidate_count - usable_grid_candidate_count)
                    grid_filter_reason = ""
                    if dropped_grid_candidate_count > 0:
                        if "grid_native_extreme_bucket" in grid_filtered_profiles:
                            grid_filter_reason = "grid_native_extreme_bucket_filtered"
                        elif dropped_grid_candidate_count == raw_grid_candidate_count and grid_filtered_profiles:
                            grid_filter_reason = "grid_stretch_reversion_bucket_filtered"
                        else:
                            grid_filter_reason = "grid_candidate_bucket_filtered"
                    grid_overlay_candidates: list[SignalCandidate] = []
                    grid_overlay_source_ids: set[str] = set()
                    overlay_candidate_pool = xau_overlay_seed_candidates if normalized_symbol == "XAUUSD" else router_candidates
                    overlay_confluence_floor = 3.55 if session_context.session_name in {"SYDNEY", "TOKYO"} else 3.70
                    overlay_score_floor = 0.66 if session_context.session_name in {"SYDNEY", "TOKYO"} else 0.70
                    mirror_extension_side_counts: Counter[str] = Counter(
                        str(candidate.side).upper()
                        for candidate in overlay_candidate_pool
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
                        and float(getattr(candidate, "confluence_score", 0.0) or 0.0) >= overlay_confluence_floor
                        and float(getattr(candidate, "score_hint", 0.0) or 0.0) >= overlay_score_floor
                    )
                    mirror_overlay_capacity = max(
                        2,
                        int(bridge_orchestrator_config.get("xau_grid_max_entries_per_symbol_loop", 6) or 6),
                    )
                    native_grid_starved = bool(
                        normalized_symbol == "XAUUSD"
                        and xau_native_live_only
                        and not usable_grid_candidates
                        and grid_decision is not None
                        and str(getattr(grid_decision, "deny_reason", "") or "") in {
                            "",
                            "grid_no_reclaim_quality",
                            "grid_no_reclaim_or_exhaustion",
                            "grid_asia_probe_no_directional_trigger",
                            "grid_prime_session_quality_gate",
                            "grid_asia_probe_mc_floor",
                            "grid_no_stretch",
                        }
                    )
                    if (
                        normalized_symbol == "XAUUSD"
                        and grid_scalper.enabled
                        and bool(getattr(grid_scalper, "mirror_overlay_enabled", False))
                        and (not xau_native_live_only or native_grid_starved)
                        and len(grid_open_positions) < mirror_overlay_capacity
                        and session_context.session_name in set(grid_scalper.allowed_sessions)
                        and not usable_grid_candidates
                        and (
                            not grid_open_positions
                            or bool(mirror_extension_side_counts and max(mirror_extension_side_counts.values()) >= 2)
                        )
                    ):
                        preferred_grid_source = sorted(
                            (
                                candidate for candidate in overlay_candidate_pool
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
                            mirror_meta = dict(getattr(source_candidate, "meta", {}) or {})
                            support_candidates: list[SignalCandidate] = []
                            for support_candidate in preferred_grid_source[1:]:
                                if len(support_candidates) >= 2:
                                    break
                                if str(support_candidate.side).upper() != primary_side:
                                    continue
                                support_setup = str(support_candidate.setup).upper()
                                if support_setup == source_setup:
                                    continue
                                if float(getattr(support_candidate, "confluence_score", 0.0) or 0.0) < overlay_confluence_floor:
                                    continue
                                if float(getattr(support_candidate, "score_hint", 0.0) or 0.0) < (0.60 if session_context.session_name in {"SYDNEY", "TOKYO"} else 0.62):
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
                            session_profile = grid_scalper._session_profile(
                                session_name=session_context.session_name,
                                now_utc=now,
                                atr_ratio=max(float(row["m5_atr_14"]) / max(float(row.get("m5_atr_avg_20", row["m5_atr_14"]) or row["m5_atr_14"]), 1e-6), 0.0),
                                spread_points=float(row["m5_spread"]),
                            )
                            mirror_grid_mode = "ATTACK_GRID" if session_profile == "AGGRESSIVE" else "NORMAL_GRID"
                            mirror_step_points = grid_scalper._step_points(
                                atr=max(float(row["m5_atr_14"]), 1e-6),
                                multiplier=grid_scalper._grid_spacing_multiplier(mirror_grid_mode),
                            )
                            mirror_stop_atr = max(0.85, float(getattr(source_candidate, "stop_atr", 0.95) or 0.95))
                            alignment_score = clamp(
                                float(mirror_meta.get("multi_tf_alignment_score") or row.get("multi_tf_alignment_score") or 0.5),
                                0.0,
                                1.0,
                            )
                            seasonality_score = clamp(
                                float(mirror_meta.get("seasonality_edge_score") or row.get("seasonality_edge_score") or 0.5),
                                0.0,
                                1.0,
                            )
                            fractal_score = clamp(
                                float(
                                    mirror_meta.get("fractal_persistence_score")
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
                                    mirror_meta.get("market_instability_score")
                                    or row.get("market_instability_score")
                                    or row.get("feature_drift_score")
                                    or 0.0
                                ),
                                0.0,
                                1.0,
                            )
                            feature_drift_score = clamp(
                                float(mirror_meta.get("feature_drift_score") or row.get("feature_drift_score") or 0.0),
                                0.0,
                                1.0,
                            )
                            compression_expansion_score = clamp(
                                float(mirror_meta.get("compression_expansion_score") or row.get("compression_expansion_score") or 0.0),
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
                                    mirror_meta.get("trend_efficiency_score")
                                    or row.get("trend_efficiency_score")
                                    or row.get("m5_trend_efficiency")
                                    or 0.5
                                ),
                                0.0,
                                1.0,
                            )
                            spread_score = clamp(
                                1.0 - (float(row.get("m5_spread", 0.0) or 0.0) / max(float(grid_scalper.spread_max_points), 1.0)),
                                0.0,
                                1.0,
                            )
                            mirror_session_fit = clamp(
                                max(
                                    float(mirror_meta.get("session_fit", 0.0) or 0.0),
                                    0.88 if session_profile == "AGGRESSIVE" else (0.76 if session_profile == "MODERATE" else 0.54),
                                ),
                                0.0,
                                1.0,
                            )
                            mirror_volatility_fit = clamp(
                                max(
                                    float(mirror_meta.get("volatility_fit", 0.0) or 0.0),
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
                                    float(mirror_meta.get("pair_behavior_fit", 0.0) or 0.0),
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
                                    float(mirror_meta.get("execution_quality_fit", 0.0) or 0.0),
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
                                    float(mirror_meta.get("entry_timing_score", 0.0) or 0.0),
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
                                    float(mirror_meta.get("structure_cleanliness_score", 0.0) or 0.0),
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
                                    float(mirror_meta.get("regime_fit", 0.0) or 0.0),
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
                                    float(mirror_meta.get("router_rank_score", getattr(source_candidate, "score_hint", 0.0)) or 0.0),
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
                            mirror_quality_tier = quality_tier_from_scores(
                                structure_cleanliness=mirror_structure_cleanliness_score,
                                regime_fit=mirror_regime_fit,
                                execution_quality_fit=mirror_execution_quality_fit,
                                high_liquidity=bool(session_context.session_name in {"LONDON", "OVERLAP", "NEW_YORK"}),
                                throughput_recovery_active=False,
                            )
                            if (
                                mirror_quality_tier == "B"
                                and mirror_regime_fit >= 0.60
                                and mirror_entry_timing_score >= 0.58
                                and mirror_structure_cleanliness_score >= 0.58
                                and mirror_execution_quality_fit >= 0.56
                            ):
                                mirror_quality_tier = "A"
                            mirror_cycle_id = deterministic_id(
                                resolved_symbol,
                                "xau-grid-mirror-cycle",
                                source_candidate.side,
                                source_candidate.setup,
                                row["time"],
                            )
                            prime_recovery_active = bool(
                                session_context.session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
                                and (
                                    grid_scalper._last_signal_emitted_at is None
                                    or (now - grid_scalper._last_signal_emitted_at).total_seconds()
                                    >= (int(grid_scalper.prime_recovery_idle_minutes) * 60)
                                )
                            )
                            mirror_confluence = max(
                                3.2,
                                float(getattr(source_candidate, "confluence_score", 0.0) or 0.0),
                            )
                            score_bonus = 0.05
                            if source_setup.startswith(("XAUUSD_M15_FIX_FLOW", "XAU_BREAKOUT_RETEST", "XAUUSD_M15_STRUCTURED_BREAKOUT", "XAUUSD_ATR_EXPANSION_SCALPER")):
                                score_bonus = 0.08
                            burst_count = grid_scalper._burst_start_count(
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
                                grid_max_levels=int(grid_scalper.max_levels),
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
                            overlap_session = session_context.session_name == "OVERLAP"
                            london_session = session_context.session_name == "LONDON"
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
                                continue
                            grid_overlay_source_ids.add(str(source_candidate.signal_id))
                            for support_candidate in support_candidates:
                                grid_overlay_source_ids.add(str(support_candidate.signal_id))
                            base_score_hint = clamp(
                                float(getattr(source_candidate, "score_hint", 0.0) or 0.0)
                                + score_bonus
                                + (0.01 * support_source_count),
                                0.60,
                                0.92,
                            )
                            mirror_meta.update(
                                {
                                    "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                                    "setup_family": "GRID",
                                    "grid_cycle": True,
                                    "grid_action": "START",
                                    "grid_cycle_id": str(mirror_cycle_id),
                                    "grid_burst_size": int(burst_count),
                                    "grid_max_levels": int(grid_scalper.max_levels),
                                    "grid_probe": False,
                                    "grid_source_setup": str(source_candidate.setup),
                                    "grid_support_source_count": int(support_source_count),
                                    "grid_support_source_setups": list(support_source_setups),
                                    "session_profile": str(session_profile),
                                    "grid_entry_profile": str(mirror_profile),
                                    "grid_mode": str(mirror_grid_mode),
                                    "grid_volatility_multiplier": float(grid_scalper._grid_spacing_multiplier(mirror_grid_mode)),
                                    "grid_stop_atr_k": float(mirror_stop_atr),
                                    "xau_engine": "GRID_DIRECTIONAL_MIRROR",
                                    "grid_native_starved_overlay": bool(native_grid_starved),
                                    "mirror_live_enabled": True,
                                    "compression_proxy_state": str(
                                        mirror_meta.get("compression_proxy_state")
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
                                    "strategy_recent_performance_seed": float(max(0.58, float(mirror_meta.get("strategy_recent_performance_seed", 0.0) or 0.0))),
                                    "prime_session_recovery_active": bool(prime_recovery_active),
                                }
                            )
                            for leg_index in range(1, int(burst_count) + 1):
                                leg_step_points = float(mirror_step_points)
                                if leg_index > 1 and session_profile == "AGGRESSIVE":
                                    leg_step_points *= 0.96 if mirror_grid_mode == "ATTACK_GRID" else 0.98
                                leg_stop_points = float(grid_scalper._entry_stop_points(step_points=leg_step_points, probe_candidate=False))
                                leg_lot = max(
                                    0.01,
                                    float(grid_scalper._lot_for_level(leg_index))
                                    * (grid_scalper.attack_grid_lot_multiplier if mirror_grid_mode == "ATTACK_GRID" else 1.0),
                                )
                                if leg_index > 1:
                                    leg_lot *= 1.0 if mirror_grid_mode == "ATTACK_GRID" else 0.95
                                leg_meta = dict(mirror_meta)
                                leg_meta.update(
                                    {
                                        "grid_level": int(leg_index),
                                        "grid_lot": float(leg_lot),
                                        "grid_step_atr_k": float(grid_scalper.step_atr_k),
                                        "grid_step_points": float(leg_step_points),
                                        "chosen_spacing_points": float(leg_step_points),
                                        "stop_points": float(leg_stop_points),
                                        "grid_burst_index": int(leg_index),
                                    }
                                )
                                grid_overlay_candidates.append(
                                    SignalCandidate(
                                        signal_id=deterministic_id(
                                            resolved_symbol,
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
                                        confluence_required=max(
                                            3.0,
                                            float(getattr(source_candidate, "confluence_required", 0.0) or 0.0),
                                        ),
                                        meta=leg_meta,
                                    )
                                )
                            support_cycle_room = max(
                                0,
                                int(xau_grid_max_entries_per_symbol_loop) - int(burst_count),
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
                                    resolved_symbol,
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
                                            "grid_max_levels": int(grid_scalper.max_levels),
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
                                            "grid_volatility_multiplier": float(grid_scalper._grid_spacing_multiplier(mirror_grid_mode)),
                                            "grid_stop_atr_k": float(support_stop_atr),
                                            "xau_engine": "GRID_DIRECTIONAL_MIRROR_SECONDARY_PRIMARY" if support_cycle_from_primary else "GRID_DIRECTIONAL_MIRROR_SUPPORT",
                                            "grid_native_starved_overlay": bool(native_grid_starved),
                                            "compression_proxy_state": str(
                                                support_meta.get("compression_proxy_state")
                                                or mirror_meta.get("compression_proxy_state")
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
                                            grid_scalper._entry_stop_points(
                                                step_points=support_step_points,
                                                probe_candidate=False,
                                            )
                                        )
                                        support_lot = max(
                                            0.01,
                                            float(grid_scalper._lot_for_level(support_leg_index))
                                            * (
                                                grid_scalper.attack_grid_lot_multiplier
                                                if mirror_grid_mode == "ATTACK_GRID"
                                                else 1.0
                                            ),
                                        )
                                        support_leg_meta = dict(support_meta)
                                        support_leg_meta.update(
                                            {
                                                "grid_level": int(support_leg_index),
                                                "grid_lot": float(support_lot),
                                                "grid_step_atr_k": float(grid_scalper.step_atr_k),
                                                "grid_step_points": float(support_step_points),
                                                "chosen_spacing_points": float(support_step_points),
                                                "stop_points": float(support_stop_points),
                                                "grid_burst_index": int(support_leg_index),
                                            }
                                        )
                                        grid_overlay_candidates.append(
                                            SignalCandidate(
                                                signal_id=deterministic_id(
                                                    resolved_symbol,
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
                                grid_scalper._last_signal_emitted_at = now
                    if grid_overlay_source_ids:
                        router_candidates = [
                            candidate
                            for candidate in router_candidates
                            if str(candidate.signal_id) not in grid_overlay_source_ids
                        ]
                    if grid_overlay_candidates:
                        router_candidates = []
                    xau_native_grid_priority = bool(
                        normalized_symbol == "XAUUSD"
                        and bool(usable_grid_candidates)
                        and session_context.session_name in set(grid_scalper.allowed_sessions)
                        and bool(getattr(grid_scalper, "native_live_only", True))
                    )
                    if xau_native_grid_priority:
                        candidates = list(usable_grid_candidates)
                    elif grid_decision is not None:
                        candidates = list(usable_grid_candidates) + grid_overlay_candidates + router_candidates
                    else:
                        candidates = grid_overlay_candidates + router_candidates
                    if normalized_symbol == "BTCUSD" and not candidates:
                        btc_force_candidate = _btc_weekend_force_candidate(
                            symbol=resolved_symbol,
                            row=row,
                            session_name=session_context.session_name,
                            timestamp=row["time"],
                            emit_time=now,
                        )
                        if btc_force_candidate is not None:
                            candidates = [btc_force_candidate]
                    _record_symbol_activity(resolved_symbol, "candidate_attempts", now)
                    candidate_family_counts: dict[str, int] = {}
                    for candidate in candidates:
                        family_key = (
                            str(candidate.meta.get("setup_family") or candidate.meta.get("btc_strategy") or candidate.meta.get("xau_engine") or candidate.strategy_family or candidate.setup)
                            .strip()
                            .upper()
                        )
                        if not family_key:
                            family_key = str(candidate.setup).upper()
                        candidate_family_counts[family_key] = int(candidate_family_counts.get(family_key, 0)) + 1
                    active_setup_windows = list(router_diagnostics.get("active_setup_windows", []))
                    considered_setup_label = ",".join(candidate_family_counts.keys()) if candidate_family_counts else ",".join(active_setup_windows)
                    for _ in candidates:
                        _record_symbol_activity(resolved_symbol, "candidate_count", now)
                    _update_runtime_metrics(
                        resolved_symbol,
                        now,
                        last_reject_reason="",
                        candidate_family_counts=candidate_family_counts,
                        last_setup_family_considered=considered_setup_label,
                        pre_open_checks_complete=_prep_checks_complete(news_snapshot, router_diagnostics, market_open),
                        pre_open_news_summary=str(news_snapshot.get("pre_open_news_summary", "")),
                        pre_open_risk_notes=dict(news_snapshot.get("pre_open_risk_notes", {})),
                        pre_open_setup_windows=active_setup_windows,
                        public_proxy_availability=dict(news_snapshot.get("public_proxy_availability", {})),
                        macro_event_bias=dict(news_snapshot.get("macro_event_bias", {})),
                        session_bias_summary=str(news_snapshot.get("session_bias_summary", "")),
                        next_open_time_utc=str(market_state.get("next_open_time_utc", "")),
                        next_open_time_local=str(market_state.get("next_open_time_local", "")),
                        dst_mode_active=dict(market_state.get("dst_mode_active", {})),
                        news_refresh_at=str(news_snapshot.get("news_refresh_at", "")),
                        next_macro_event=news_snapshot.get("next_macro_event"),
                        event_risk_window_active=bool(news_snapshot.get("event_risk_window_active", False)),
                        post_news_trade_window_active=bool(news_snapshot.get("post_news_trade_window_active", False)),
                        news_bias_direction=str(news_snapshot.get("news_bias_direction", "neutral")),
                        news_confidence=float(news_snapshot.get("news_confidence", 0.0) or 0.0),
                        news_data_quality=str(news_snapshot.get("news_data_quality", "unknown")),
                        news_headlines=list(news_snapshot.get("news_headlines", [])),
                        news_source_breakdown=dict(news_snapshot.get("news_source_breakdown", {})),
                        news_category_summary=dict(news_snapshot.get("news_category_summary", {})),
                        news_primary_category=str(news_snapshot.get("news_primary_category", "general_macro")),
                        news_secondary_source_used=bool(news_snapshot.get("news_secondary_source_used", False)),
                        news_rss_headlines=list(news_snapshot.get("news_rss_headlines", [])),
                        news_rss_headline_count=int(news_snapshot.get("news_rss_headline_count", 0) or 0),
                        session_policy_current=str(router_diagnostics.get("session_policy_current", "")),
                        active_setup_windows=active_setup_windows,
                        setup_proxy_availability=dict(router_diagnostics.get("setup_proxy_availability", {})),
                        weekend_vs_weekday_btc_mode=router_diagnostics.get("weekend_vs_weekday_btc_mode"),
                        proxy_unavailable_fallback_used=bool(router_diagnostics.get("proxy_unavailable_fallback_used", False)),
                        btc_reason_no_candidate=str(router_diagnostics.get("btc_reason_no_candidate", "")),
                        btc_last_structure_state=str(router_diagnostics.get("btc_last_structure_state", "")),
                        btc_last_volatility_state=str(router_diagnostics.get("btc_last_volatility_state", "")),
                        btc_last_spread_state=str(router_diagnostics.get("btc_last_spread_state", "")),
                        xau_grid_raw_candidate_count=raw_grid_candidate_count if normalized_symbol == "XAUUSD" else 0,
                        xau_grid_usable_candidate_count=usable_grid_candidate_count if normalized_symbol == "XAUUSD" else 0,
                        xau_grid_dropped_candidate_count=dropped_grid_candidate_count if normalized_symbol == "XAUUSD" else 0,
                        xau_grid_filtered_profiles=list(grid_filtered_profiles) if normalized_symbol == "XAUUSD" else [],
                        xau_grid_last_filter_reason=str(grid_filter_reason) if normalized_symbol == "XAUUSD" else "",
                        xau_grid_last_deny_reason=str(getattr(grid_decision, "deny_reason", "") or "") if normalized_symbol == "XAUUSD" else "",
                        xau_grid_last_soft_penalty_reason=str(getattr(grid_decision, "soft_penalty_reason", "") or "") if normalized_symbol == "XAUUSD" else "",
                        xau_grid_last_soft_penalty_score=float(getattr(grid_decision, "soft_penalty_score", 0.0) or 0.0) if normalized_symbol == "XAUUSD" else 0.0,
                        xau_grid_last_entry_profile=str(getattr(grid_decision, "entry_profile", "") or "") if normalized_symbol == "XAUUSD" else "",
                        xau_grid_last_session_profile=str(getattr(grid_decision, "session_profile", "") or "") if normalized_symbol == "XAUUSD" else "",
                        xau_grid_last_density_relief_active=bool(getattr(grid_decision, "density_relief_active", False)) if normalized_symbol == "XAUUSD" else False,
                        xau_grid_last_mc_floor=float(getattr(grid_decision, "mc_floor", 0.0) or 0.0) if normalized_symbol == "XAUUSD" else 0.0,
                        xau_grid_last_mc_win_rate=float(getattr(grid_decision, "mc_win_rate", 0.0) or 0.0) if normalized_symbol == "XAUUSD" else 0.0,
                        xau_grid_quota_target_10m=int(getattr(grid_decision, "quota_target_10m", 0) or 0) if normalized_symbol == "XAUUSD" else 0,
                        xau_grid_quota_approved_last_10m=int(getattr(grid_decision, "quota_approved_last_10m", 0) or 0) if normalized_symbol == "XAUUSD" else 0,
                        xau_grid_quota_debt_10m=int(getattr(grid_decision, "quota_debt_10m", 0) or 0) if normalized_symbol == "XAUUSD" else 0,
                        xau_grid_quota_density_first_active=bool(getattr(grid_decision, "quota_density_first_active", False)) if normalized_symbol == "XAUUSD" else False,
                        xau_grid_quota_state=str(getattr(grid_decision, "quota_state", "") or "") if normalized_symbol == "XAUUSD" else "",
                        xau_grid_last_quota_window_id=str(getattr(grid_decision, "quota_window_id", "") or "") if normalized_symbol == "XAUUSD" else "",
                        xau_grid_active_profile=str(getattr(grid_scalper, "active_profile", "default")) if normalized_symbol == "XAUUSD" else "",
                        xau_grid_proof_mode=str(getattr(grid_scalper, "proof_mode", "default")) if normalized_symbol == "XAUUSD" else "",
                        xau_grid_checkpoint_artifact=str(getattr(grid_scalper, "checkpoint_artifact", "") or "") if normalized_symbol == "XAUUSD" else "",
                        xau_grid_density_branch_artifact=str(getattr(grid_scalper, "density_branch_artifact", "") or "") if normalized_symbol == "XAUUSD" else "",
                        xau_grid_hard_vs_soft_gate_mode=(
                            "DENSITY_FIRST" if bool(getattr(grid_scalper, "density_first_mode", False)) else "CHECKPOINT_QUALITY"
                        ) if normalized_symbol == "XAUUSD" else "",
                        xau_grid_last_candidate_stage=(
                            "native_candidates_ready"
                            if normalized_symbol == "XAUUSD" and usable_grid_candidate_count > 0
                            else (
                                "native_candidates_filtered"
                                if normalized_symbol == "XAUUSD" and raw_grid_candidate_count > 0 and usable_grid_candidate_count == 0
                                else (
                                    "native_soft_penalty"
                                    if normalized_symbol == "XAUUSD" and str(getattr(grid_decision, "soft_penalty_reason", "") or "")
                                    else (
                                    "native_denied"
                                    if normalized_symbol == "XAUUSD" and str(getattr(grid_decision, "deny_reason", "") or "")
                                    else ""
                                    )
                                )
                            )
                        ),
                    )
                    if normalized_symbol == "BTCUSD":
                        logger.info(
                            "BTC_CANDIDATE_FAMILY_CONSIDERED",
                            extra={
                                "extra_fields": {
                                    "symbol": resolved_symbol,
                                    "families": dict(candidate_family_counts),
                                    "fallback_used": bool(router_diagnostics.get("proxy_unavailable_fallback_used", False)),
                                    "reason_no_candidate": str(router_diagnostics.get("btc_reason_no_candidate", "")),
                                }
                            },
                        )
                    if normalized_symbol == "XAUUSD" and grid_scalper.enabled:
                        grid_open = [position for position in symbol_open_positions if _is_xau_grid_setup(str(position.get("setup", "")))]
                        symbol_status.grid_cycle_state = "ACTIVE" if grid_open else "IDLE"
                        symbol_status.grid_leg = f"{len(grid_open)}/{max(1, int(grid_scalper.max_levels))}"
                        symbol_status.engine = "XAUUSD_M5_GRID" if (grid_open or usable_grid_candidates) else "XAU_MULTI"
                        if grid_open:
                            symbol_status.grid_last_entry = f"{float(grid_open[-1].get('entry_price', 0.0)):.2f}"
                    _record_symbol_activity(resolved_symbol, "scans", now)
                    symbol_status.current_state = "SCANNING" if not symbol_open_positions else "OPEN_POSITION"
                    log_stage(resolved_symbol, "SCANNING", "evaluating_setups", now_ts=now)
                    session_defaults = _session_runtime_defaults(
                        symbol=resolved_symbol,
                        session_name=session_context.session_name,
                        current_scaling_state=current_scaling_state,
                        setup=str(considered_setup_label or ""),
                        setup_family=str(considered_setup_label or "TREND"),
                    )
                    if not candidates:
                        recent_candidate_flow = (
                            _symbol_activity_count(resolved_symbol, "candidate_count", 900, now) > 0
                            or _symbol_activity_count(resolved_symbol, "candidate_attempts", 900, now) > 0
                        )
                        if normalized_symbol == "BTCUSD":
                            logger.info(
                                "BTC_PROXY_UNAVAILABLE_FALLBACK_USED",
                                extra={
                                    "extra_fields": {
                                        "symbol": resolved_symbol,
                                        "fallback_used": bool(router_diagnostics.get("proxy_unavailable_fallback_used", False)),
                                        "structure_state": str(router_diagnostics.get("btc_last_structure_state", "")),
                                        "volatility_state": str(router_diagnostics.get("btc_last_volatility_state", "")),
                                        "spread_state": str(router_diagnostics.get("btc_last_spread_state", "")),
                                    }
                                },
                            )
                        soft_xau_retry_reason = bool(
                            normalized_symbol == "XAUUSD"
                            and recent_candidate_flow
                            and str(getattr(grid_decision, "deny_reason", "") or "")
                            in {
                                "grid_no_reclaim_quality",
                                "grid_prime_session_quality_gate",
                                "grid_no_reclaim_or_exhaustion",
                                "grid_no_stretch",
                            }
                        )
                        if grid_decision is not None and grid_decision.deny_reason and not soft_xau_retry_reason:
                            symbol_status.reason = _display_reason(resolved_symbol, grid_decision.deny_reason)
                        elif recent_candidate_flow:
                            symbol_status.reason = _display_reason(resolved_symbol, "ready_recheck")
                        else:
                            symbol_status.reason = _display_reason(resolved_symbol, "no_base_candidate")
                        symbol_status.current_state = _state_from_reason(symbol_status.reason)
                        log_stage(resolved_symbol, "SCANNING", symbol_status.reason, now_ts=now)
                        _update_runtime_metrics(
                            resolved_symbol,
                            now,
                            market_open_status=market_status,
                            candidate_family_counts=candidate_family_counts,
                            last_setup_family_considered=considered_setup_label,
                            pre_open_checks_complete=_prep_checks_complete(news_snapshot, router_diagnostics, market_open),
                            pre_open_news_summary=str(news_snapshot.get("pre_open_news_summary", "")),
                            pre_open_risk_notes=dict(news_snapshot.get("pre_open_risk_notes", {})),
                            pre_open_setup_windows=active_setup_windows,
                            public_proxy_availability=dict(news_snapshot.get("public_proxy_availability", {})),
                            macro_event_bias=dict(news_snapshot.get("macro_event_bias", {})),
                            session_bias_summary=str(news_snapshot.get("session_bias_summary", "")),
                            next_open_time_utc=str(market_state.get("next_open_time_utc", "")),
                            next_open_time_local=str(market_state.get("next_open_time_local", "")),
                            dst_mode_active=dict(market_state.get("dst_mode_active", {})),
                            news_refresh_at=str(news_snapshot.get("news_refresh_at", "")),
                            next_macro_event=news_snapshot.get("next_macro_event"),
                            event_risk_window_active=bool(news_snapshot.get("event_risk_window_active", False)),
                            post_news_trade_window_active=bool(news_snapshot.get("post_news_trade_window_active", False)),
                            news_bias_direction=str(news_snapshot.get("news_bias_direction", "neutral")),
                            news_confidence=float(news_snapshot.get("news_confidence", 0.0) or 0.0),
                            news_data_quality=str(news_snapshot.get("news_data_quality", "unknown")),
                            news_headlines=list(news_snapshot.get("news_headlines", [])),
                            news_source_breakdown=dict(news_snapshot.get("news_source_breakdown", {})),
                            news_category_summary=dict(news_snapshot.get("news_category_summary", {})),
                            news_primary_category=str(news_snapshot.get("news_primary_category", "general_macro")),
                            news_secondary_source_used=bool(news_snapshot.get("news_secondary_source_used", False)),
                            news_rss_headlines=list(news_snapshot.get("news_rss_headlines", [])),
                            news_rss_headline_count=int(news_snapshot.get("news_rss_headline_count", 0) or 0),
                            session_policy_current=str(router_diagnostics.get("session_policy_current", "")),
                            active_setup_windows=active_setup_windows,
                            setup_proxy_availability=dict(router_diagnostics.get("setup_proxy_availability", {})),
                            weekend_vs_weekday_btc_mode=router_diagnostics.get("weekend_vs_weekday_btc_mode"),
                            lane_name=str(session_defaults.get("lane_name", "")),
                            session_priority_profile=str(session_defaults.get("session_priority_profile", "GLOBAL")),
                            lane_session_priority=str(session_defaults.get("lane_session_priority", "NEUTRAL")),
                            session_native_pair=bool(session_defaults.get("session_native_pair", False)),
                            session_priority_multiplier=float(session_defaults.get("session_priority_multiplier", 1.0)),
                            pair_priority_rank_in_session=int(session_defaults.get("pair_priority_rank_in_session", 99)),
                            lane_budget_share=float(session_defaults.get("lane_budget_share", 0.0)),
                            lane_available_capacity=float(session_defaults.get("lane_available_capacity", 0.0)),
                            proxy_unavailable_fallback_used=bool(router_diagnostics.get("proxy_unavailable_fallback_used", False)),
                            btc_reason_no_candidate=str(router_diagnostics.get("btc_reason_no_candidate", "")),
                            btc_last_structure_state=str(router_diagnostics.get("btc_last_structure_state", "")),
                            btc_last_volatility_state=str(router_diagnostics.get("btc_last_volatility_state", "")),
                            btc_last_spread_state=str(router_diagnostics.get("btc_last_spread_state", "")),
                            last_reject_reason=(
                                (
                                    "ready_recheck"
                                    if soft_xau_retry_reason
                                    else str(getattr(grid_decision, "deny_reason", "") or "no_base_candidate")
                                )
                                if normalized_symbol == "XAUUSD"
                                else ("ready_recheck" if recent_candidate_flow else "no_base_candidate")
                            ),
                            xau_grid_raw_candidate_count=raw_grid_candidate_count if normalized_symbol == "XAUUSD" else 0,
                            xau_grid_usable_candidate_count=usable_grid_candidate_count if normalized_symbol == "XAUUSD" else 0,
                            xau_grid_dropped_candidate_count=dropped_grid_candidate_count if normalized_symbol == "XAUUSD" else 0,
                            xau_grid_filtered_profiles=list(grid_filtered_profiles) if normalized_symbol == "XAUUSD" else [],
                            xau_grid_last_filter_reason=str(grid_filter_reason) if normalized_symbol == "XAUUSD" else "",
                            xau_grid_last_deny_reason=str(getattr(grid_decision, "deny_reason", "") or "") if normalized_symbol == "XAUUSD" else "",
                            xau_grid_last_soft_penalty_reason=str(getattr(grid_decision, "soft_penalty_reason", "") or "") if normalized_symbol == "XAUUSD" else "",
                            xau_grid_last_soft_penalty_score=float(getattr(grid_decision, "soft_penalty_score", 0.0) or 0.0) if normalized_symbol == "XAUUSD" else 0.0,
                            xau_grid_last_entry_profile=str(getattr(grid_decision, "entry_profile", "") or "") if normalized_symbol == "XAUUSD" else "",
                            xau_grid_last_session_profile=str(getattr(grid_decision, "session_profile", "") or "") if normalized_symbol == "XAUUSD" else "",
                            xau_grid_last_density_relief_active=bool(getattr(grid_decision, "density_relief_active", False)) if normalized_symbol == "XAUUSD" else False,
                            xau_grid_last_mc_floor=float(getattr(grid_decision, "mc_floor", 0.0) or 0.0) if normalized_symbol == "XAUUSD" else 0.0,
                            xau_grid_last_mc_win_rate=float(getattr(grid_decision, "mc_win_rate", 0.0) or 0.0) if normalized_symbol == "XAUUSD" else 0.0,
                            xau_grid_quota_target_10m=int(getattr(grid_decision, "quota_target_10m", 0) or 0) if normalized_symbol == "XAUUSD" else 0,
                            xau_grid_quota_approved_last_10m=int(getattr(grid_decision, "quota_approved_last_10m", 0) or 0) if normalized_symbol == "XAUUSD" else 0,
                            xau_grid_quota_debt_10m=int(getattr(grid_decision, "quota_debt_10m", 0) or 0) if normalized_symbol == "XAUUSD" else 0,
                            xau_grid_quota_density_first_active=bool(getattr(grid_decision, "quota_density_first_active", False)) if normalized_symbol == "XAUUSD" else False,
                            xau_grid_quota_state=str(getattr(grid_decision, "quota_state", "") or "") if normalized_symbol == "XAUUSD" else "",
                            xau_grid_last_quota_window_id=str(getattr(grid_decision, "quota_window_id", "") or "") if normalized_symbol == "XAUUSD" else "",
                            xau_grid_active_profile=str(getattr(grid_scalper, "active_profile", "default")) if normalized_symbol == "XAUUSD" else "",
                            xau_grid_proof_mode=str(getattr(grid_scalper, "proof_mode", "default")) if normalized_symbol == "XAUUSD" else "",
                            xau_grid_checkpoint_artifact=str(getattr(grid_scalper, "checkpoint_artifact", "") or "") if normalized_symbol == "XAUUSD" else "",
                            xau_grid_density_branch_artifact=str(getattr(grid_scalper, "density_branch_artifact", "") or "") if normalized_symbol == "XAUUSD" else "",
                            xau_grid_hard_vs_soft_gate_mode=(
                                "DENSITY_FIRST" if bool(getattr(grid_scalper, "density_first_mode", False)) else "CHECKPOINT_QUALITY"
                            ) if normalized_symbol == "XAUUSD" else "",
                            xau_grid_last_candidate_stage=(
                                "native_denied"
                                if normalized_symbol == "XAUUSD" and str(getattr(grid_decision, "deny_reason", "") or "")
                                else (
                                    "native_soft_penalty"
                                    if normalized_symbol == "XAUUSD" and str(getattr(grid_decision, "soft_penalty_reason", "") or "")
                                    else (
                                    "native_candidates_filtered"
                                    if normalized_symbol == "XAUUSD" and raw_grid_candidate_count > 0 and usable_grid_candidate_count == 0
                                    else "candidate_search"
                                    )
                                )
                            ),
                            delivered_actions_last_15m=int(bridge_delivery_counts.get(_normalize_symbol_key(resolved_symbol), symbol_status.delivered_actions)),
                        )
                        continue

                    summary["signals"] += len(candidates)
                    symbol_info = _resolve_runtime_symbol_info(
                        mt5_client=mt5_client,
                        bridge_queue=bridge_queue,
                        symbol_info_cache=symbol_info_cache,
                        symbol=resolved_symbol,
                        bridge_trade_mode=bridge_trade_mode,
                        bridge_context_account=bridge_context_account,
                        bridge_context_magic=bridge_context_magic,
                    )
                    ranked_strategy_pool = _candidate_strategy_pool_rankings(
                        symbol_key=resolved_symbol,
                        candidates=candidates,
                        session_name=session_context.session_name,
                        row=row,
                        regime=regime,
                        symbol_info=symbol_info,
                        max_spread_points=float(risk_config["max_spread_points"]),
                        closed_trades=recent_closed_trades,
                        current_day_key=str(global_stats.trading_day_key),
                        candidate_tier_config=candidate_tier_config,
                        orchestrator_config=bridge_orchestrator_config,
                    )
                    if ranked_strategy_pool:
                        candidates = [candidate for candidate, _ in ranked_strategy_pool]
                    raw_strategy_pool_ranking = [dict(entry) for _, entry in ranked_strategy_pool]
                    strategy_pool_runtime_map = {
                        str(entry.get("strategy_key") or ""): dict(entry)
                        for entry in raw_strategy_pool_ranking
                        if str(entry.get("strategy_key") or "").strip()
                    }
                    symbol_status.last_signal = candidates[0].setup
                    best_candidate = candidates[0]
                    winning_strategy_key = str(raw_strategy_pool_ranking[0].get("strategy_key", "")) if raw_strategy_pool_ranking else ""
                    strategy_pool_ranking = _summarize_strategy_pool_rankings(
                        symbol_key=resolved_symbol,
                        ranked_entries=raw_strategy_pool_ranking,
                        preferred_strategy_key=winning_strategy_key,
                        session_name=str(session_context.session_name),
                        regime_state=str(regime.state_label or regime.label),
                    )
                    winning_strategy_reason = _strategy_pool_winner_reason(strategy_pool_ranking)
                    resolved_strategy_pool = [
                        str(item.get("strategy_key") or "").strip()
                        for item in strategy_pool_ranking
                        if str(item.get("strategy_key") or "").strip()
                    ]
                    best_candidate.meta["strategy_pool"] = list(resolved_strategy_pool)
                    best_candidate.meta["strategy_pool_ranking"] = list(strategy_pool_ranking)
                    best_candidate.meta["strategy_pool_winner"] = str(winning_strategy_key)
                    best_candidate.meta["winning_strategy_reason"] = str(winning_strategy_reason)
                    best_candidate_family = (
                        str(
                            best_candidate.meta.get("setup_family")
                            or best_candidate.meta.get("btc_strategy")
                            or best_candidate.meta.get("xau_engine")
                            or best_candidate.strategy_family
                            or best_candidate.setup
                        )
                        .strip()
                        .upper()
                    )
                    if normalized_symbol == "XAUUSD" and _is_xau_grid_setup(best_candidate.setup):
                        grid_level = int(best_candidate.meta.get("grid_level", 1))
                        grid_max = int(best_candidate.meta.get("grid_max_levels", max(1, int(grid_scalper.max_levels))))
                        symbol_status.grid_cycle_state = "ACTIVE"
                        symbol_status.grid_leg = f"{max(1, grid_level)}/{max(1, grid_max)}"
                        symbol_status.grid_cycle_id = str(best_candidate.meta.get("grid_cycle_id", best_candidate.signal_id))
                        symbol_status.grid_last_entry = f"{float(row['m5_close']):.2f}"
                        symbol_status.engine = "XAUUSD_M5_GRID"
                    elif normalized_symbol == "XAUUSD":
                        symbol_status.engine = str(best_candidate.setup)
                    log_stage(
                        resolved_symbol,
                        "CANDIDATE_FOUND",
                        f"{best_candidate.setup} score={float(best_candidate.score_hint):.2f}",
                        now_ts=now,
                    )
                    if normalized_symbol == "BTCUSD":
                        logger.info(
                            "BTC_CANDIDATE_CREATED",
                            extra={
                                "extra_fields": {
                                    "symbol": resolved_symbol,
                                    "setup": best_candidate.setup,
                                    "family": best_candidate_family,
                                    "probability_hint": float(best_candidate.score_hint),
                                    "confluence": float(best_candidate.confluence_score),
                                }
                            },
                        )
                    symbol_status.current_state = "CANDIDATE_FOUND"
                    _update_runtime_metrics(
                        resolved_symbol,
                        now,
                        last_setup_family_considered=best_candidate_family,
                        candidate_family_counts=candidate_family_counts,
                        lane_name=str(
                            _session_runtime_defaults(
                                symbol=resolved_symbol,
                                session_name=session_context.session_name,
                                current_scaling_state=current_scaling_state,
                                setup=str(best_candidate.setup),
                                setup_family=str(best_candidate.strategy_family),
                            ).get("lane_name", "")
                        ),
                        pre_open_checks_complete=_prep_checks_complete(news_snapshot, router_diagnostics, market_open),
                        pre_open_news_summary=str(news_snapshot.get("pre_open_news_summary", "")),
                        pre_open_risk_notes=dict(news_snapshot.get("pre_open_risk_notes", {})),
                        pre_open_setup_windows=active_setup_windows,
                        public_proxy_availability=dict(news_snapshot.get("public_proxy_availability", {})),
                        macro_event_bias=dict(news_snapshot.get("macro_event_bias", {})),
                        session_bias_summary=str(news_snapshot.get("session_bias_summary", "")),
                        next_open_time_utc=str(market_state.get("next_open_time_utc", "")),
                        next_open_time_local=str(market_state.get("next_open_time_local", "")),
                        dst_mode_active=dict(market_state.get("dst_mode_active", {})),
                        news_refresh_at=str(news_snapshot.get("news_refresh_at", "")),
                        next_macro_event=news_snapshot.get("next_macro_event"),
                        event_risk_window_active=bool(news_snapshot.get("event_risk_window_active", False)),
                        post_news_trade_window_active=bool(news_snapshot.get("post_news_trade_window_active", False)),
                        news_bias_direction=str(news_snapshot.get("news_bias_direction", "neutral")),
                        news_confidence=float(news_snapshot.get("news_confidence", 0.0) or 0.0),
                        news_data_quality=str(news_snapshot.get("news_data_quality", "unknown")),
                        news_headlines=list(news_snapshot.get("news_headlines", [])),
                        news_source_breakdown=dict(news_snapshot.get("news_source_breakdown", {})),
                        news_category_summary=dict(news_snapshot.get("news_category_summary", {})),
                        news_primary_category=str(news_snapshot.get("news_primary_category", "general_macro")),
                        news_secondary_source_used=bool(news_snapshot.get("news_secondary_source_used", False)),
                        news_rss_headlines=list(news_snapshot.get("news_rss_headlines", [])),
                        news_rss_headline_count=int(news_snapshot.get("news_rss_headline_count", 0) or 0),
                        last_setup_policy_window=str(best_candidate.meta.get("policy_window", "")),
                        last_btc_setup_family_considered=best_candidate_family if normalized_symbol == "BTCUSD" else None,
                        market_open_status=market_status,
                        strategy_key=str(
                            winning_strategy_key
                            or best_candidate.meta.get("strategy_key")
                            or resolve_strategy_key(normalized_symbol, str(best_candidate.setup))
                        ),
                        strategy_pool=list(resolved_strategy_pool),
                        strategy_pool_ranking=strategy_pool_ranking,
                        strategy_pool_winner=str(winning_strategy_key),
                        winning_strategy_reason=str(winning_strategy_reason),
                        strategy_score=float(
                            best_candidate.meta.get(
                                "strategy_pool_rank_score",
                                best_candidate.meta.get("router_rank_score", best_candidate.score_hint or 0.0),
                            )
                            or 0.0
                        ),
                        strategy_state=str(best_candidate.meta.get("strategy_state") or "NORMAL"),
                        strategy_recent_performance=float(best_candidate.meta.get("strategy_recent_performance_seed", 0.50) or 0.50),
                        management_template=str(best_candidate.meta.get("management_template") or ""),
                        regime_fit=float(best_candidate.meta.get("regime_fit", 0.0) or 0.0),
                        session_fit=float(best_candidate.meta.get("session_fit", 0.0) or 0.0),
                        volatility_fit=float(best_candidate.meta.get("volatility_fit", 0.0) or 0.0),
                        pair_behavior_fit=float(best_candidate.meta.get("pair_behavior_fit", 0.0) or 0.0),
                        execution_quality_fit=float(best_candidate.meta.get("execution_quality_fit", 0.0) or 0.0),
                        entry_timing_score=float(best_candidate.meta.get("entry_timing_score", 0.0) or 0.0),
                        structure_cleanliness_score=float(best_candidate.meta.get("structure_cleanliness_score", 0.0) or 0.0),
                        xau_grid_mode=str(best_candidate.meta.get("grid_mode") or ""),
                        delivered_actions_last_15m=int(bridge_delivery_counts.get(_normalize_symbol_key(resolved_symbol), symbol_status.delivered_actions)),
                    )
                    symbol_stats = journal.stats(
                        current_equity=float(account["equity"]),
                        symbol=resolved_symbol,
                        account=bridge_context_account,
                        magic=bridge_context_magic,
                    )
                    recent_trades = journal.recent_executions(
                        resolved_symbol,
                        60,
                        account=bridge_context_account,
                        magic=bridge_context_magic,
                    )
                    xau_grid_candidate_available = bool(
                        normalized_symbol == "XAUUSD"
                        and any(_is_xau_grid_setup(candidate.setup) for candidate in candidates)
                    )
                    entries_this_symbol = 0
                    if micro_active and not xau_grid_candidate_available:
                        cooldown_after_loss_minutes, cooldown_after_win_minutes = _micro_cooldown_minutes(
                            resolved_symbol,
                            now,
                            micro_config,
                        )
                        cooldown_reason = journal.cooldown_block_reason(
                            now=now,
                            symbol=resolved_symbol,
                            cooldown_after_loss_minutes=cooldown_after_loss_minutes,
                            cooldown_after_win_minutes=cooldown_after_win_minutes,
                            account=bridge_context_account,
                            magic=bridge_context_magic,
                        )
                        previous = cooldown_log_state.get(resolved_symbol, "")
                        if cooldown_reason:
                            if previous != cooldown_reason:
                                logger.info(
                                    "micro_cooldown_block",
                                    extra={
                                        "extra_fields": {
                                            "symbol": resolved_symbol,
                                            "reason": cooldown_reason,
                                        }
                                    },
                                )
                            cooldown_log_state[resolved_symbol] = cooldown_reason
                            mark_block(resolved_symbol, cooldown_reason, symbol_status)
                            continue
                        if previous:
                            logger.info(
                                "micro_cooldown_clear",
                                extra={"extra_fields": {"symbol": resolved_symbol}},
                            )
                        cooldown_log_state.pop(resolved_symbol, None)

                    symbol_entry_cap = _symbol_entry_cap(
                        resolved_symbol,
                        candidates,
                        default_cap=max_entries_per_symbol_loop,
                        xau_grid_cap=xau_grid_max_entries_per_symbol_loop,
                    )

                    for candidate in candidates:
                        if entries_this_symbol >= symbol_entry_cap:
                            break
                        if len(open_positions_journal) >= effective_max_positions_total:
                            mark_block(resolved_symbol, "max_positions_total_reached", symbol_status)
                            break
                        if len(symbol_open_positions) >= effective_max_positions_per_symbol:
                            mark_block(resolved_symbol, "max_positions_per_symbol_reached", symbol_status)
                            break
                        adaptive_feedback = journal.adaptive_feedback(
                            resolved_symbol,
                            candidate.setup,
                            account=bridge_context_account,
                            magic=bridge_context_magic,
                        )
                        open_positions_view = [
                            {
                                "symbol": position["symbol"],
                                "side": position["side"],
                                "opened_at": datetime.fromisoformat(position["opened_at"]),
                            }
                            for position in open_positions_journal
                        ]
                        portfolio_decision = portfolio.assess_new_position(open_positions_view, resolved_symbol, candidate.side)
                        if not portfolio_decision.allowed:
                            mark_block(resolved_symbol, portfolio_decision.reason, symbol_status)
                            continue

                        symbol_info = _resolve_runtime_symbol_info(
                            mt5_client=mt5_client,
                            bridge_queue=bridge_queue,
                            symbol_info_cache=symbol_info_cache,
                            symbol=resolved_symbol,
                            bridge_trade_mode=bridge_trade_mode,
                            bridge_context_account=bridge_context_account,
                            bridge_context_magic=bridge_context_magic,
                        )
                        entry_price, entry_price_source = _effective_live_entry_price(
                            side=candidate.side,
                            tick=tick,
                            symbol_info=symbol_info,
                        )
                        if entry_price <= 0:
                            mark_block(resolved_symbol, "missing_live_entry_price", symbol_status)
                            continue
                        preliminary_ai = ai_gate.evaluate(candidate, row, regime.label, symbol_stats.consecutive_losses)
                        symbol_status.last_score = preliminary_ai.probability
                        idea_structure = TradeIdeaLifecycle.build_structure_snapshot(
                            row=row,
                            confluence_score=float(candidate.confluence_score),
                            entry_price=entry_price,
                            atr=float(row.get("m5_atr_14", 0.0)),
                        )
                        idea = idea_lifecycle.upsert(
                            symbol=resolved_symbol,
                            setup_type=str(candidate.setup),
                            side=str(candidate.side),
                            confidence=float(preliminary_ai.probability),
                            confluence_score=float(candidate.confluence_score),
                            entry_price=entry_price,
                            atr=float(row.get("m5_atr_14", 0.0)),
                            now=now,
                            structure=idea_structure,
                        )
                        if _is_xau_grid_setup(str(candidate.setup)):
                            allow_eval, eval_reason = True, "grid_fast_recycle"
                        else:
                            allow_eval, eval_reason = idea_lifecycle.can_evaluate(
                                idea=idea,
                                now=now,
                                session_name=session_context.session_name,
                                structure=idea_structure,
                            )
                        if not allow_eval:
                            if eval_reason == "cooldown_active":
                                log_stage(
                                    resolved_symbol,
                                    "COOLDOWN",
                                    f"{candidate.setup} cooldown_active",
                                    suppress_after=reject_display_limit,
                                    now_ts=now,
                                )
                                mark_block(resolved_symbol, "idea_cooldown_active", symbol_status)
                            elif eval_reason == "recheck_wait":
                                log_stage(resolved_symbol, "RECHECKING", f"{candidate.setup} waiting_interval", now_ts=now)
                                mark_block(resolved_symbol, "idea_recheck_wait", symbol_status)
                            continue
                        if eval_reason == "reactivated":
                            log_stage(resolved_symbol, "RECHECKING", f"{candidate.setup} structure_improved", now_ts=now)
                        idea_lifecycle.mark_evaluated(idea=idea, now=now, structure=idea_structure)

                        candidate_timeframe = str(candidate.meta.get("timeframe", "M5")).upper()
                        atr_field = str(candidate.meta.get("atr_field", "m5_atr_14"))
                        atr_for_candidate = max(float(row.get(atr_field, row.get("m5_atr_14", 0.0))), 1e-6)
                        if "fallback" in str(symbol_info.get("economics_source", "")).lower():
                            next_allowed_log = meta_fallback_next_log_at.get(resolved_symbol.upper())
                            if next_allowed_log is None or now >= next_allowed_log:
                                logger.warning(
                                    "meta_fallback_used",
                                    extra={
                                        "extra_fields": {
                                            "symbol": resolved_symbol,
                                            "economics_source": str(symbol_info.get("economics_source", "")),
                                        }
                                    },
                                )
                                meta_fallback_next_log_at[resolved_symbol.upper()] = now + timedelta(hours=1)
                        provisional_stop_distance, stop_geometry_source = _resolve_candidate_stop_distance(
                            candidate=candidate,
                            atr_for_candidate=atr_for_candidate,
                            point_size=float(symbol_info.get("point", 0.0) or 0.0),
                            sl_multiplier=preliminary_ai.sl_multiplier,
                        )
                        if provisional_stop_distance <= 0:
                            mark_block(resolved_symbol, "invalid_stop_distance", symbol_status)
                            continue

                        if candidate.side == "BUY":
                            stop_price = entry_price - provisional_stop_distance
                            tp_price = entry_price + (provisional_stop_distance * preliminary_ai.tp_r)
                        else:
                            stop_price = entry_price + provisional_stop_distance
                            tp_price = entry_price - (provisional_stop_distance * preliminary_ai.tp_r)
                        is_grid_candidate = bool(candidate.meta.get("grid_cycle", False))
                        grid_max_levels = int(candidate.meta.get("grid_max_levels", 0)) if is_grid_candidate else 0
                        if is_grid_candidate and grid_max_levels > 0 and len(symbol_open_positions) >= grid_max_levels:
                            mark_block(resolved_symbol, "grid_max_levels_reached", symbol_status)
                            continue
                        grid_lot_hint = float(candidate.meta.get("grid_lot", 0.0)) if is_grid_candidate else 0.0
                        grid_use_fixed_lot = is_grid_candidate and grid_lot_hint > 0.0
                        effective_use_fixed_lot = grid_use_fixed_lot or bool(risk_config.get("use_fixed_lot", False))
                        effective_fixed_lot = grid_lot_hint if grid_use_fixed_lot else float(risk_config.get("fixed_lot", 0.01))
                        raw_spread_points = float(row["m5_spread"])
                        spread_points = _normalize_runtime_spread_points(
                            resolved_symbol,
                            raw_spread_points,
                            symbol_info=symbol_info,
                            max_spread_points=float(risk_config["max_spread_points"]),
                        )
                        pre_risk_stop_geometry = _normalize_pre_risk_exit_geometry(
                            symbol=resolved_symbol,
                            side=candidate.side,
                            entry_price=entry_price,
                            stop_price=stop_price,
                            tp_price=tp_price,
                            spread_points=raw_spread_points,
                            symbol_info=symbol_info,
                            symbol_rules=symbol_rules,
                            safety_buffer_points=bridge_safety_buffer_points,
                        )
                        stop_price = float(pre_risk_stop_geometry["stop_price"])
                        tp_price = float(pre_risk_stop_geometry["tp_price"])
                        provisional_stop_distance = max(float(pre_risk_stop_geometry["stop_distance"]), 0.0)
                        spread_ratio = spread_points / max(float(risk_config["max_spread_points"]), 1.0)
                        tokyo_sydney_spread_guard = session_context.session_name in {"TOKYO", "SYDNEY"} and spread_ratio > 0.70
                        session_offset = float((ai_config.get("session_threshold_offsets", {}) or {}).get(session_context.session_name, session_context.ai_threshold_offset))
                        adaptive_floor_offset = 0.0
                        no_trade_minutes_for_floor = 0.0
                        if last_trade_opened_at is not None:
                            no_trade_minutes_for_floor = max(0.0, (now - last_trade_opened_at).total_seconds() / 60.0)
                        if no_trade_minutes_for_floor >= 60:
                            adaptive_floor_offset = -0.05
                        elif no_trade_minutes_for_floor >= 30:
                            adaptive_floor_offset = -0.03
                        session_offset += adaptive_floor_offset
                        rule_confluence_required = float(candidate.confluence_required or max(1.0, candidate.confluence_score))
                        rule_confluence_required = max(1.0, rule_confluence_required + float(session_context.confluence_delta))
                        if tokyo_sydney_spread_guard:
                            boosted_confluence = rule_confluence_required + 1.0
                            if float(candidate.confluence_score) >= boosted_confluence:
                                rule_confluence_required = boosted_confluence
                            else:
                                session_offset += 0.02
                        regime_state = str(getattr(regime, "state_label", regime.label) or regime.label).upper()
                        regime_confidence = clamp(
                            float(
                                getattr(regime, "state_confidence", None)
                                or getattr(regime, "confidence", 0.0)
                                or regime.details.get("regime_state_confidence", 0.0)
                            ),
                            0.0,
                            1.0,
                        )
                        volatility_state = str(regime.details.get("volatility_forecast_state", "BALANCED") or "BALANCED").upper()
                        if not volatility_state:
                            volatility_state = "BALANCED"
                        pressure_continuation = clamp(float(regime.details.get("continuation_pressure", 0.5) or 0.5), 0.0, 1.0)
                        pressure_exhaustion = clamp(float(regime.details.get("exhaustion_signal", 0.0) or 0.0), 0.0, 1.0)
                        pressure_absorption = clamp(float(regime.details.get("absorption_signal", 0.0) or 0.0), 0.0, 1.0)
                        pressure_alignment = clamp(
                            pressure_continuation - (0.35 * pressure_exhaustion) + (0.15 * pressure_absorption),
                            0.0,
                            1.0,
                        )
                        liquidity_decision = evaluate_liquidity_map(
                            symbol_key=resolved_symbol,
                            side=candidate.side,
                            entry_price=entry_price,
                            point_size=max(float(symbol_info["point"]), 1e-9),
                            context=dict(row),
                            now_utc=now,
                        )
                        recent_win_rate = clamp(
                            float(adaptive_feedback.get("weighted_win_rate", 0.5) or 0.5),
                            0.0,
                            1.0,
                        )
                        recent_expectancy_r = float(adaptive_feedback.get("weighted_avg_r", 0.0) or 0.0)
                        performance_score = clamp(
                            0.50
                            + ((recent_win_rate - 0.50) * 0.60)
                            + clamp(recent_expectancy_r, -0.50, 1.00) * 0.20,
                            0.20,
                            0.95,
                        )
                        typical_spread_points = _normalize_runtime_spread_points(
                            resolved_symbol,
                            max(
                                1.0,
                                float(
                                    row.get("m5_spread_avg_20")
                                    or row.get("m15_spread_avg_20")
                                    or row.get("m5_spread")
                                    or raw_spread_points
                                ),
                            ),
                            symbol_info=symbol_info,
                            max_spread_points=float(risk_config["max_spread_points"]),
                        )
                        stale_idea_rate = clamp(
                            float(symbol_runtime_metrics.get(_normalize_symbol_key(resolved_symbol), {}).get("stale_archives_last_15m", 0) or 0.0) / 10.0,
                            0.0,
                            1.0,
                        )
                        execution_quality = evaluate_execution_quality(
                            spread_points=spread_points,
                            typical_spread_points=typical_spread_points,
                            stale_idea_rate=stale_idea_rate,
                            bridge_latency_ms=float(symbol_runtime_metrics.get(_normalize_symbol_key(resolved_symbol), {}).get("runtime_market_data_latency_ms", 0.0) or 0.0),
                        )
                        session_quality_score = 0.95 if session_context.session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.78 if session_context.session_name in {"TOKYO", "SYDNEY"} else 0.65
                        if tokyo_sydney_spread_guard:
                            session_quality_score *= 0.90
                        structure_score = clamp(
                            max(float(candidate.confluence_score) / max(rule_confluence_required, 1.0), preliminary_ai.probability),
                            0.0,
                            1.0,
                        )
                        trade_quality = evaluate_trade_quality(
                            symbol=resolved_symbol,
                            session_name=session_context.session_name,
                            setup_family=normalize_strategy_family(candidate.strategy_family),
                            regime_state=regime_state,
                            regime_confidence=regime_confidence,
                            spread_points=spread_points,
                            spread_limit=float(risk_config["max_spread_points"]),
                            volatility_state=volatility_state,
                            liquidity_score=float(liquidity_decision.score),
                            news_state=str(getattr(news_decision, "state", "NEWS_SAFE")),
                            news_confidence=float(getattr(news_decision, "decision_confidence", 1.0)),
                            structure_score=structure_score,
                            execution_feasibility=1.0 if bool(pre_risk_stop_geometry.get("valid", True)) else 0.0,
                            expected_value_r=float(preliminary_ai.expected_value_r),
                            probability=float(preliminary_ai.probability),
                            performance_score=performance_score,
                            execution_quality_score=float(execution_quality.score),
                            pressure_alignment=pressure_alignment,
                        )
                        trade_lane = infer_trade_lane(
                            symbol=resolved_symbol,
                            setup=str(candidate.setup),
                            setup_family=normalize_strategy_family(candidate.strategy_family),
                            session_name=session_context.session_name,
                        )
                        daily_state_name, _ = risk_engine.resolve_daily_state_from_stats(
                            global_stats,
                            caution_threshold_pct=float(risk_config.get("daily_caution_threshold_pct", 0.02)),
                            defensive_threshold_pct=float(risk_config.get("daily_defensive_threshold_pct", 0.035)),
                            hard_stop_threshold_pct=float(risk_config.get("daily_hard_stop_threshold_pct", risk_config.get("hard_daily_dd_pct", 0.05))),
                        )
                        strategy_key = str(
                            candidate.meta.get("strategy_key")
                            or resolve_strategy_key(_normalize_symbol_key(resolved_symbol), str(candidate.setup))
                        )
                        strategy_score = float(candidate.meta.get("router_rank_score", candidate.score_hint or 0.0) or 0.0)
                        quality_tier = str(candidate.meta.get("quality_tier") or "B").upper()
                        tier_size_multiplier = float(candidate.meta.get("tier_size_multiplier", 1.0) or 1.0)
                        delta_proxy_score_value = float(candidate.meta.get("delta_proxy_score", 0.0) or 0.0)
                        compression_proxy_state = str(candidate.meta.get("compression_proxy_state") or "")
                        session_loosen_factor_value = float(candidate.meta.get("session_loosen_factor", 1.0) or 1.0)
                        throughput_recovery_active = bool(candidate.meta.get("throughput_recovery_active", False))
                        btc_weekend_force_emit = bool(
                            normalized_symbol == "BTCUSD" and bool(candidate.meta.get("btc_weekend_force_emit", False))
                        )
                        strategy_state = str(candidate.meta.get("strategy_state", "NORMAL") or "NORMAL")
                        strategy_pool = list(candidate.meta.get("strategy_pool") or [])
                        strategy_regime_fit = float(candidate.meta.get("regime_fit", 0.0) or 0.0)
                        strategy_session_fit = float(candidate.meta.get("session_fit", 0.0) or 0.0)
                        strategy_volatility_fit = float(candidate.meta.get("volatility_fit", 0.0) or 0.0)
                        strategy_pair_behavior_fit = float(candidate.meta.get("pair_behavior_fit", 0.0) or 0.0)
                        strategy_execution_fit = float(candidate.meta.get("execution_quality_fit", execution_quality.score) or 0.0)
                        strategy_recent_performance_seed = float(
                            candidate.meta.get("strategy_recent_performance_seed", 0.50) or 0.50
                        )
                        entry_timing_score_value = float(candidate.meta.get("entry_timing_score", 0.0) or 0.0)
                        structure_cleanliness_score_value = float(candidate.meta.get("structure_cleanliness_score", 0.0) or 0.0)
                        if btc_weekend_force_emit:
                            throughput_recovery_active = True
                            if quality_tier == "C":
                                quality_tier = "B"
                            if quality_band_rank(trade_quality.band) < quality_band_rank("B"):
                                trade_quality = replace(
                                    trade_quality,
                                    score=max(float(trade_quality.score), 0.61),
                                    band="B",
                                    legacy_band="B",
                                    elite=False,
                                    acceptable=True,
                                    should_skip=False,
                                    overflow_eligible=True,
                                    size_multiplier=max(float(trade_quality.size_multiplier), 0.80),
                                )
                            strategy_state = "ATTACK" if strategy_state == "NORMAL" else strategy_state
                            strategy_regime_fit = max(float(strategy_regime_fit), 0.50)
                            strategy_session_fit = max(float(strategy_session_fit), 0.56)
                            strategy_volatility_fit = max(float(strategy_volatility_fit), 0.54)
                            strategy_pair_behavior_fit = max(float(strategy_pair_behavior_fit), 0.54)
                            strategy_execution_fit = max(float(strategy_execution_fit), 0.56)
                            strategy_recent_performance_seed = max(float(strategy_recent_performance_seed), 0.56)
                            entry_timing_score_value = max(float(entry_timing_score_value), 0.52)
                            structure_cleanliness_score_value = max(float(structure_cleanliness_score_value), 0.48)
                        proof_bucket = strategy_optimizer.bucket_proof_state(
                            symbol=resolved_symbol,
                            strategy=strategy_key,
                            lane=str(trade_lane),
                            regime=str(regime_state),
                            session=str(session_context.session_name),
                            direction=str(candidate.side),
                            quality_band=str(trade_quality.band),
                        )
                        bucket_adjustment = strategy_optimizer.bucket_adjustment(
                            symbol=resolved_symbol,
                            strategy=strategy_key,
                            lane=str(trade_lane),
                            regime=str(regime_state),
                            session=str(session_context.session_name),
                            direction=str(candidate.side),
                            quality_band=str(trade_quality.band),
                            min_samples=5,
                        )
                        proof_bucket_state = str(proof_bucket.get("state", "neutral"))
                        lane_adjustment_state = str(bucket_adjustment.get("state", "neutral"))
                        lane_strength_multiplier = float(bucket_adjustment.get("multiplier", 1.0) or 1.0)
                        lane_score = _lane_score_from_adjustment(
                            dict(bucket_adjustment.get("summary") or {}),
                            float(trade_quality.score),
                        )
                        session_priority = session_priority_context(
                            symbol=resolved_symbol,
                            lane_name=str(trade_lane),
                            session_name=session_context.session_name,
                        )
                        pair_state = _pair_session_performance_state(
                            symbol=resolved_symbol,
                            session_name=session_context.session_name,
                            session_native_pair=bool(session_priority.session_native_pair),
                            closed_trades=recent_closed_trades,
                            current_day_key=str(global_stats.trading_day_key),
                        )
                        strategy_bucket_state = _pair_strategy_session_performance_state(
                            symbol=resolved_symbol,
                            strategy_key=strategy_key,
                            session_name=session_context.session_name,
                            regime_state=str(regime_state),
                            session_native_pair=bool(session_priority.session_native_pair),
                            closed_trades=recent_closed_trades,
                            current_day_key=str(global_stats.trading_day_key),
                        )
                        optimizer_bucket_summary = dict(bucket_adjustment.get("summary") or proof_bucket.get("summary") or {})
                        strategy_trade_count = max(
                            int(optimizer_bucket_summary.get("trades", 0) or 0),
                            int(strategy_bucket_state.get("strategy_bucket_sample_size", 0) or 0),
                        )
                        strategy_recent_performance = float(strategy_recent_performance_seed)
                        if strategy_trade_count > 0:
                            optimizer_recent_performance = strategy_recent_performance_score(
                                win_rate=float(optimizer_bucket_summary.get("win_rate", 0.5) or 0.5),
                                profit_factor=float(optimizer_bucket_summary.get("profit_factor", 1.0) or 1.0),
                                expectancy_r=float(optimizer_bucket_summary.get("expectancy_r", 0.0) or 0.0),
                                management_quality=float(pair_state["management_quality_score"]),
                            )
                            bucket_recent_performance = strategy_recent_performance_score(
                                win_rate=float(strategy_bucket_state.get("strategy_bucket_win_rate", 0.5) or 0.5),
                                profit_factor=float(strategy_bucket_state.get("strategy_bucket_profit_factor", 1.0) or 1.0),
                                expectancy_r=float(strategy_bucket_state.get("strategy_bucket_expectancy_r", 0.0) or 0.0),
                                management_quality=float(strategy_bucket_state.get("strategy_bucket_management_quality", pair_state["management_quality_score"])),
                            )
                            strategy_recent_performance = clamp(
                                (0.55 * float(optimizer_recent_performance))
                                + (0.45 * float(bucket_recent_performance)),
                                0.0,
                                1.0,
                            )
                        optimizer_strategy_state = strategy_health_state(
                            win_rate=float(optimizer_bucket_summary.get("win_rate", 0.5) or 0.5),
                            profit_factor=float(optimizer_bucket_summary.get("profit_factor", 1.0) or 1.0),
                            expectancy_r=float(optimizer_bucket_summary.get("expectancy_r", 0.0) or 0.0),
                            management_quality=float(pair_state["management_quality_score"]),
                            sample_size=int(strategy_trade_count),
                        )
                        strategy_state = _merge_strategy_health_states(
                            optimizer_strategy_state,
                            str(strategy_bucket_state.get("strategy_bucket_state", "NORMAL") or "NORMAL"),
                        )
                        strategy_bucket_avg_mae_r = float(
                            strategy_bucket_state.get("strategy_bucket_avg_mae_r", 0.0) or 0.0
                        )
                        strategy_bucket_avg_mfe_r = float(
                            strategy_bucket_state.get("strategy_bucket_avg_mfe_r", 0.0) or 0.0
                        )
                        strategy_bucket_capture_efficiency = float(
                            strategy_bucket_state.get("strategy_bucket_capture_efficiency", 0.5) or 0.5
                        )
                        strategy_bucket_giveback_rate = float(
                            strategy_bucket_state.get("strategy_bucket_giveback_rate", 0.0) or 0.0
                        )
                        mae_drag = clamp(
                            max(0.0, strategy_bucket_avg_mae_r - 0.55) / 0.55,
                            0.0,
                            1.0,
                        )
                        mfe_drag = clamp(
                            max(0.0, 0.40 - strategy_bucket_avg_mfe_r) / 0.40,
                            0.0,
                            1.0,
                        )
                        capture_drag = clamp(
                            max(0.0, 0.42 - strategy_bucket_capture_efficiency) / 0.42,
                            0.0,
                            1.0,
                        )
                        giveback_drag = clamp(strategy_bucket_giveback_rate / 0.30, 0.0, 1.0)
                        entry_timing_score_value = clamp(
                            float(entry_timing_score_value)
                            - (0.18 * float(strategy_bucket_state.get("strategy_bucket_late_entry_rate", 0.0) or 0.0))
                            - (0.10 * float(strategy_bucket_state.get("strategy_bucket_fast_failure_rate", 0.0) or 0.0))
                            - (0.08 * float(strategy_bucket_state.get("strategy_bucket_immediate_invalidation_rate", 0.0) or 0.0))
                            - (0.10 * mae_drag)
                            - (0.06 * mfe_drag)
                            - (0.06 * capture_drag)
                            + (0.03 if strategy_state == "ATTACK" else 0.0),
                            0.0,
                            1.0,
                        )
                        structure_cleanliness_score_value = clamp(
                            float(structure_cleanliness_score_value)
                            - (0.18 * float(strategy_bucket_state.get("strategy_bucket_poor_structure_rate", 0.0) or 0.0))
                            - (0.10 * float(strategy_bucket_state.get("strategy_bucket_false_break_rate", 0.0) or 0.0))
                            - (0.08 * mae_drag)
                            - (0.08 * capture_drag)
                            - (0.05 * giveback_drag)
                            + (0.02 if strategy_state == "ATTACK" else 0.0),
                            0.0,
                            1.0,
                        )
                        strategy_state_multiplier = {
                            "ATTACK": 1.06,
                            "NORMAL": 1.00,
                            "REDUCED": 0.92,
                            "QUARANTINED": 0.82,
                        }.get(strategy_state, 1.00)
                        strategy_state_size_multiplier = (
                            _strategy_state_size_multiplier(strategy_state)
                            * float(strategy_bucket_state.get("strategy_bucket_size_multiplier", 1.0) or 1.0)
                        )
                        symbol_runtime_market_data_consensus_state = str(
                            getattr(symbol_status, "runtime_market_data_consensus_state", "")
                            or runtime_market_data_consensus_state
                            or ""
                        )
                        market_data_consensus_state = str(symbol_runtime_market_data_consensus_state)
                        market_data_consensus_adjustment = _market_data_consensus_adjustment(
                            market_data_consensus_state
                        )
                        strategy_management_quality = clamp(
                            (0.65 * float(pair_state["management_quality_score"]))
                            + (0.35 * float(strategy_bucket_state.get("strategy_bucket_management_quality", pair_state["management_quality_score"]) or pair_state["management_quality_score"])),
                            0.0,
                            1.0,
                        )
                        strategy_false_break_penalty = clamp(
                            max(0.0, float(pair_state["false_break_rate"]))
                            + max(0.0, 0.55 - float(structure_cleanliness_score_value)) * 0.50
                            + (0.35 * float(strategy_bucket_state.get("strategy_bucket_false_break_rate", 0.0) or 0.0))
                            + (0.18 * float(strategy_bucket_state.get("strategy_bucket_late_entry_rate", 0.0) or 0.0)),
                            0.0,
                            1.0,
                        )
                        strategy_false_break_penalty = clamp(
                            float(strategy_false_break_penalty)
                            + (0.08 * mfe_drag)
                            + (0.10 * capture_drag),
                            0.0,
                            1.0,
                        )
                        strategy_drawdown_penalty = clamp(
                            max(0.0, -float(pair_state["rolling_expectancy_by_pair"])) * 0.75
                            + max(0.0, -float(pair_state["rolling_expectancy_by_session"])) * 0.45
                            + max(0.0, 1.0 - float(pair_state["rolling_pf_by_pair"])) * 0.18
                            + max(0.0, -float(strategy_bucket_state.get("strategy_bucket_expectancy_r", 0.0) or 0.0)) * 0.55
                            + max(0.0, 1.0 - float(strategy_bucket_state.get("strategy_bucket_profit_factor", 1.0) or 1.0)) * 0.12
                            + (0.12 * float(strategy_bucket_state.get("strategy_bucket_fast_failure_rate", 0.0) or 0.0))
                            + (0.14 * float(strategy_bucket_state.get("strategy_bucket_immediate_invalidation_rate", 0.0) or 0.0))
                            + (0.10 * float(strategy_bucket_state.get("strategy_bucket_giveback_rate", 0.0) or 0.0))
                            + (0.10 * mae_drag)
                            + (0.08 * giveback_drag)
                            + (0.12 if strategy_state == "REDUCED" else 0.22 if strategy_state == "QUARANTINED" else 0.0),
                            0.0,
                            1.0,
                        )
                        strategy_chop_penalty = 0.0
                        if str(regime_state).upper() == "LOW_LIQUIDITY_CHOP" and any(
                            token in strategy_key for token in ("BREAKOUT", "IMPULSE", "EXPANSION")
                        ):
                            strategy_chop_penalty = 0.24
                        strategy_ev_estimate = clamp(
                            (0.32 * float(trade_quality.score))
                            + (0.28 * clamp(float(preliminary_ai.probability), 0.0, 1.0))
                            + (0.24 * clamp((float(preliminary_ai.expected_value_r) + 0.10) / 1.50, 0.0, 1.0))
                            + (0.16 * clamp(float(liquidity_decision.score), 0.0, 1.0)),
                            0.0,
                            1.0,
                        )
                        strategy_recent_performance_effective = clamp(
                            (0.78 * float(strategy_recent_performance))
                            + (0.22 * float(strategy_management_quality)),
                            0.0,
                            1.0,
                        )
                        strategy_score = strategy_selection_score(
                            ev_estimate=float(strategy_ev_estimate),
                            regime_fit=float(strategy_regime_fit),
                            session_fit=float(strategy_session_fit),
                            volatility_fit=float(strategy_volatility_fit),
                            pair_behavior_fit_score=float(strategy_pair_behavior_fit),
                            strategy_recent_performance=float(strategy_recent_performance_effective),
                            execution_quality_fit=float(strategy_execution_fit),
                            entry_timing_score_value=float(entry_timing_score_value),
                            structure_cleanliness_score_value=float(structure_cleanliness_score_value),
                            drawdown_penalty=float(strategy_drawdown_penalty),
                            false_break_penalty=float(strategy_false_break_penalty),
                            chop_penalty=float(strategy_chop_penalty),
                        )
                        strategy_score = clamp(
                            float(strategy_score)
                            + {
                                "ATTACK": 0.03,
                                "NORMAL": 0.0,
                                "REDUCED": -0.03,
                                "QUARANTINED": -0.08,
                            }.get(strategy_state, 0.0),
                            0.0,
                            1.0,
                        )
                        strategy_score = clamp(
                            float(strategy_score)
                            + {
                                "A+": 0.04,
                                "A": 0.02,
                                "B": 0.03 if throughput_recovery_active else 0.0,
                            }.get(quality_tier, 0.0),
                            0.0,
                            1.0,
                        )
                        strategy_score = clamp(
                            float(strategy_score) + float(market_data_consensus_adjustment),
                            0.0,
                            1.0,
                        )
                        xau_grid_density_override_ready = bool(candidate.meta.get("xau_grid_density_override_ready", False))
                        xau_grid_density_ranking_bonus = float(candidate.meta.get("xau_grid_density_ranking_bonus", 0.0) or 0.0)
                        if (
                            normalized_symbol == "XAUUSD"
                            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
                            and xau_grid_density_override_ready
                            and xau_grid_density_ranking_bonus > 0.0
                        ):
                            strategy_score = clamp(float(strategy_score) + float(xau_grid_density_ranking_bonus), 0.0, 1.0)
                        lane_strength_multiplier *= (
                            float(pair_state["pair_state_multiplier"])
                            * float(strategy_state_multiplier)
                            * float(strategy_bucket_state.get("strategy_bucket_priority_multiplier", 1.0) or 1.0)
                        )
                        lane_strength_multiplier = clamp(lane_strength_multiplier, 0.70, 1.35)
                        session_adjusted = session_adjusted_score(
                            base_score=float(clamp((0.62 * float(trade_quality.score)) + (0.38 * float(strategy_score)), 0.0, 1.0)),
                            session_priority_multiplier=float(session_priority.session_priority_multiplier),
                            lane_strength_multiplier=float(lane_strength_multiplier),
                            quality_floor_edge=float(session_priority.quality_floor_edge),
                        )
                        session_adjusted = clamp(
                            float(session_adjusted) + float(market_data_consensus_adjustment),
                            0.0,
                            1.0,
                        )
                        lane_score = clamp(float(lane_score) * float(pair_state["pair_state_multiplier"]), 0.0, 1.0)
                        candidate.meta["strategy_key"] = str(strategy_key)
                        candidate.meta["strategy_pool"] = list(strategy_pool)
                        candidate.meta["strategy_state"] = str(strategy_state)
                        candidate.meta["strategy_state_reason"] = str(
                            strategy_bucket_state.get("strategy_bucket_reason", "")
                            or optimizer_strategy_state.lower()
                        )
                        candidate.meta["strategy_score"] = float(strategy_score)
                        candidate.meta["rank_score"] = float(strategy_score)
                        candidate.meta["regime_state"] = str(regime_state)
                        candidate.meta["regime_fit"] = float(strategy_regime_fit)
                        candidate.meta["session_fit"] = float(strategy_session_fit)
                        candidate.meta["volatility_fit"] = float(strategy_volatility_fit)
                        candidate.meta["pair_behavior_fit"] = float(strategy_pair_behavior_fit)
                        candidate.meta["execution_quality_fit"] = float(strategy_execution_fit)
                        candidate.meta["entry_timing_score"] = float(entry_timing_score_value)
                        candidate.meta["structure_cleanliness_score"] = float(structure_cleanliness_score_value)
                        candidate.meta["strategy_recent_performance"] = float(strategy_recent_performance)
                        candidate.meta["strategy_bucket_state"] = str(strategy_bucket_state.get("strategy_bucket_state", "NORMAL"))
                        candidate.meta["strategy_bucket_reason"] = str(strategy_bucket_state.get("strategy_bucket_reason", ""))
                        candidate.meta["strategy_bucket_expectancy_r"] = float(strategy_bucket_state.get("strategy_bucket_expectancy_r", 0.0) or 0.0)
                        candidate.meta["strategy_bucket_profit_factor"] = float(strategy_bucket_state.get("strategy_bucket_profit_factor", 1.0) or 1.0)
                        candidate.meta["strategy_bucket_false_break_rate"] = float(strategy_bucket_state.get("strategy_bucket_false_break_rate", 0.0) or 0.0)
                        candidate.meta["strategy_bucket_late_entry_rate"] = float(strategy_bucket_state.get("strategy_bucket_late_entry_rate", 0.0) or 0.0)
                        candidate.meta["strategy_bucket_fast_failure_rate"] = float(strategy_bucket_state.get("strategy_bucket_fast_failure_rate", 0.0) or 0.0)
                        candidate.meta["strategy_bucket_capture_efficiency"] = float(strategy_bucket_state.get("strategy_bucket_capture_efficiency", 0.5) or 0.5)
                        candidate.meta["strategy_bucket_management_quality"] = float(strategy_bucket_state.get("strategy_bucket_management_quality", 0.5) or 0.5)
                        candidate.meta["lane_name"] = str(trade_lane)
                        candidate.meta["session_native_pair"] = bool(session_priority.session_native_pair)
                        candidate.meta["session_priority_multiplier"] = float(session_priority.session_priority_multiplier)
                        candidate.meta["pair_priority_rank_in_session"] = int(session_priority.pair_priority_rank_in_session)
                        candidate.meta["lane_budget_share"] = float(session_priority.lane_budget_share)
                        candidate.meta["lane_available_capacity"] = float(
                            candidate.meta.get("lane_available_capacity")
                            or session_priority.lane_budget_share * max(
                                0.0,
                                float(
                                    current_scaling_state.get(
                                        "projected_trade_capacity_today",
                                        current_scaling_state.get("stretch_daily_trade_target", 0),
                                    )
                                    or 0.0
                                ),
                            )
                        )
                        candidate.meta["lane_strength_multiplier"] = float(lane_strength_multiplier)
                        candidate.meta["strategy_pool_rank_score"] = float(session_adjusted)
                        candidate.meta["market_data_consensus_adjustment"] = float(market_data_consensus_adjustment)
                        candidate.meta["runtime_market_data_consensus_state"] = str(market_data_consensus_state)
                        candidate.meta["runtime_market_data_source"] = str(
                            getattr(symbol_status, "runtime_market_data_source", "")
                            or runtime_market_data_source
                        )
                        strategy_pool_runtime_map[strategy_key] = {
                            **dict(strategy_pool_runtime_map.get(strategy_key, {})),
                            "strategy_key": str(strategy_key),
                            "setup": str(candidate.setup),
                            "lane_name": str(trade_lane),
                            "strategy_state": str(strategy_state),
                            "strategy_state_reason": str(
                                strategy_bucket_state.get("strategy_bucket_reason", "")
                                or optimizer_strategy_state.lower()
                            ),
                            "strategy_score": float(strategy_score),
                            "strategy_pool_rank_score": float(session_adjusted),
                            "rank_score": float(session_adjusted),
                            "regime_state": str(regime_state),
                            "regime_fit": float(strategy_regime_fit),
                            "session_fit": float(strategy_session_fit),
                            "volatility_fit": float(strategy_volatility_fit),
                            "pair_behavior_fit": float(strategy_pair_behavior_fit),
                            "execution_quality_fit": float(strategy_execution_fit),
                            "entry_timing_score": float(entry_timing_score_value),
                            "structure_cleanliness_score": float(structure_cleanliness_score_value),
                            "strategy_recent_performance": float(strategy_recent_performance),
                            "strategy_bucket_state": str(strategy_bucket_state.get("strategy_bucket_state", "NORMAL")),
                            "strategy_bucket_reason": str(strategy_bucket_state.get("strategy_bucket_reason", "")),
                            "strategy_bucket_expectancy_r": float(strategy_bucket_state.get("strategy_bucket_expectancy_r", 0.0) or 0.0),
                            "strategy_bucket_profit_factor": float(strategy_bucket_state.get("strategy_bucket_profit_factor", 1.0) or 1.0),
                            "strategy_bucket_false_break_rate": float(strategy_bucket_state.get("strategy_bucket_false_break_rate", 0.0) or 0.0),
                            "strategy_bucket_late_entry_rate": float(strategy_bucket_state.get("strategy_bucket_late_entry_rate", 0.0) or 0.0),
                            "strategy_bucket_fast_failure_rate": float(strategy_bucket_state.get("strategy_bucket_fast_failure_rate", 0.0) or 0.0),
                            "session_native_pair": bool(session_priority.session_native_pair),
                            "session_priority_multiplier": float(session_priority.session_priority_multiplier),
                            "pair_priority_rank_in_session": int(session_priority.pair_priority_rank_in_session),
                            "market_data_consensus_adjustment": float(market_data_consensus_adjustment),
                            "runtime_market_data_consensus_state": str(market_data_consensus_state),
                            "runtime_market_data_source": str(
                                getattr(symbol_status, "runtime_market_data_source", "")
                                or runtime_market_data_source
                            ),
                        }
                        strategy_pool_ranking = _summarize_strategy_pool_rankings(
                            symbol_key=resolved_symbol,
                            ranked_entries=list(strategy_pool_runtime_map.values()),
                            preferred_strategy_key=strategy_key,
                            session_name=str(session_context.session_name),
                            regime_state=str(regime_state),
                        )
                        winning_strategy_reason = _strategy_pool_winner_reason(strategy_pool_ranking)
                        resolved_strategy_pool = [
                            str(item.get("strategy_key") or "").strip()
                            for item in strategy_pool_ranking
                            if str(item.get("strategy_key") or "").strip()
                        ]
                        winning_strategy_key = str(strategy_pool_ranking[0].get("strategy_key", "")) if strategy_pool_ranking else str(strategy_key)
                        candidate.meta["strategy_pool"] = list(resolved_strategy_pool)
                        candidate.meta["strategy_pool_ranking"] = list(strategy_pool_ranking)
                        candidate.meta["strategy_pool_winner"] = str(winning_strategy_key)
                        candidate.meta["winning_strategy_reason"] = str(winning_strategy_reason)
                        _record_band_count(loop_band_counts, trade_quality.band)
                        loop_lane_candidate_counts[str(trade_lane)] = int(loop_lane_candidate_counts.get(str(trade_lane), 0)) + 1
                        current_band_target = _band_target_from_counts(loop_band_counts)
                        fallback_band_active = current_band_target != "A+"
                        reason_higher_band_unavailable = _fallback_band_reason(current_band_target)
                        quality_cluster_score = _quality_cluster_score(loop_band_counts)
                        session_density_score = _session_density_score(session_context.session_name)
                        cluster_mode_active = bool(
                            quality_cluster_score >= 0.65
                            or (
                                (
                                    int(loop_band_counts.get("A+", 0))
                                    + int(loop_band_counts.get("A", 0))
                                    + int(loop_band_counts.get("A-", 0))
                                    + int(loop_band_counts.get("B+", 0))
                                ) >= 2
                                and session_density_score >= 0.72
                            )
                        )
                        best_a_count = int(loop_band_counts.get("A", 0)) + int(loop_band_counts.get("A-", 0))
                        quality_floor_default = float(risk_scaling_config.get("trade_quality_floor", 0.58))
                        quality_floor_exception = float(risk_scaling_config.get("trade_quality_exception_floor", 0.60))
                        quality_exception_probability = float(risk_scaling_config.get("trade_quality_exception_probability_floor", 0.60))
                        quality_exception_ev = float(risk_scaling_config.get("trade_quality_exception_ev_floor", 0.20))
                        quality_exception_confluence = float(risk_scaling_config.get("trade_quality_exception_confluence_floor", 3.0))
                        band_rank = quality_band_rank(trade_quality.band)
                        current_band_attempted = str(trade_quality.band)
                        effective_quality_floor_default = max(
                            0.50,
                            quality_floor_default - float(session_priority.quality_floor_edge),
                        )
                        proof_exception_allowed = False
                        proof_exception_reason = ""
                        qualified_b_allowed = (
                            trade_quality.score >= effective_quality_floor_default
                            and float(preliminary_ai.probability) >= max(0.52, quality_exception_probability - 0.08)
                            and float(preliminary_ai.expected_value_r) >= max(0.10, quality_exception_ev - 0.10)
                            and float(candidate.confluence_score) >= max(2.5, min(quality_exception_confluence, rule_confluence_required))
                            and bool(liquidity_decision.allowed)
                            and str(getattr(news_decision, "state", "NEWS_SAFE")).upper() in {"NEWS_SAFE", "NEWS_CAUTION"}
                            and float(execution_quality.score) >= 0.55
                        )
                        qualified_c_allowed = (
                            proof_bucket_state in {"trusted", "elite-proof"}
                            and trade_quality.score >= 0.48
                            and float(preliminary_ai.probability) >= max(0.68, quality_exception_probability)
                            and float(preliminary_ai.expected_value_r) >= max(0.30, quality_exception_ev)
                            and float(candidate.confluence_score) >= max(quality_exception_confluence + 0.5, rule_confluence_required + 0.5)
                            and bool(liquidity_decision.allowed)
                            and str(getattr(news_decision, "state", "NEWS_SAFE")).upper() in {"NEWS_SAFE", "NEWS_CAUTION"}
                            and float(execution_quality.score) >= 0.70
                        )
                        if band_rank == quality_band_rank("C") and qualified_c_allowed:
                            proof_exception_allowed = True
                            proof_exception_reason = "proof_backed_c"
                        elif band_rank == quality_band_rank("B") and proof_bucket_state in {"trusted", "elite-proof"} and daily_state_name in {"DAILY_CAUTION", "DAILY_DEFENSIVE"}:
                            proof_exception_allowed = True
                            proof_exception_reason = "proof_backed_b"
                        if btc_weekend_force_emit:
                            proof_exception_allowed = True
                            proof_exception_reason = "btc_weekend_force_emit"
                            qualified_b_allowed = True
                            qualified_c_allowed = True
                        reduced_b_recovery_allowed = bool(
                            band_rank == quality_band_rank("B")
                            and str(strategy_state).upper() == "REDUCED"
                            and not bool(strategy_bucket_state.get("strategy_bucket_should_block_all_bands", False))
                            and float(entry_timing_score_value) >= 0.74
                            and float(structure_cleanliness_score_value) >= 0.72
                            and float(strategy_regime_fit) >= 0.68
                            and float(strategy_execution_fit) >= 0.68
                            and float(strategy_recent_performance) >= 0.46
                        )
                        throughput_b_recovery_allowed = bool(
                            band_rank == quality_band_rank("B")
                            and quality_tier == "B"
                            and throughput_recovery_active
                            and not (
                                normalized_symbol == "XAUUSD"
                                and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
                                and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
                            )
                            and not bool(strategy_bucket_state.get("strategy_bucket_should_block_all_bands", False))
                            and float(entry_timing_score_value) >= 0.62
                            and float(structure_cleanliness_score_value) >= 0.58
                            and float(strategy_regime_fit) >= 0.54
                            and float(strategy_execution_fit) >= 0.58
                        )
                        btc_weekend_heartbeat_recovery_allowed = bool(
                            normalized_symbol == "BTCUSD"
                            and bool(candidate.meta.get("proxyless_weekend_heartbeat_mode", False))
                            and throughput_recovery_active
                            and not bool(strategy_bucket_state.get("strategy_bucket_should_block_all_bands", False))
                            and float(entry_timing_score_value) >= 0.48
                            and float(structure_cleanliness_score_value) >= 0.44
                            and float(strategy_execution_fit) >= 0.50
                            and float(preliminary_ai.probability) >= max(0.54, float(min_probability_floor) - 0.02)
                        )
                        strategy_bucket_b_block = bool(
                            band_rank == quality_band_rank("B")
                            and bool(strategy_bucket_state.get("strategy_bucket_should_block_b", False))
                            and not proof_exception_allowed
                            and not reduced_b_recovery_allowed
                            and not throughput_b_recovery_allowed
                            and not btc_weekend_heartbeat_recovery_allowed
                            and not (
                                bool(session_priority.session_native_pair)
                                and str(strategy_state).upper() == "ATTACK"
                            )
                        )
                        strategy_bucket_all_band_block = bool(
                            strategy_bucket_state.get("strategy_bucket_should_block_all_bands", False)
                            and not btc_weekend_force_emit
                        )
                        if band_rank == 0 and not proof_exception_allowed:
                            mark_block(resolved_symbol, "trade_quality_below_floor", symbol_status)
                            _update_runtime_metrics(
                                resolved_symbol,
                                now,
                                trade_quality_score=float(trade_quality.score),
                                trade_quality_band=str(trade_quality.band),
                                lane_name=str(trade_lane),
                                session_priority_profile=str(session_priority.session_priority_profile),
                                lane_session_priority=str(session_priority.lane_session_priority),
                                session_native_pair=bool(session_priority.session_native_pair),
                                session_priority_multiplier=float(session_priority.session_priority_multiplier),
                                pair_priority_rank_in_session=int(session_priority.pair_priority_rank_in_session),
                                lane_budget_share=float(session_priority.lane_budget_share),
                                strategy_key=str(strategy_key),
                                strategy_pool=list(resolved_strategy_pool),
                                strategy_pool_ranking=list(strategy_pool_ranking),
                                strategy_pool_winner=str(winning_strategy_key),
                                winning_strategy_reason=str(winning_strategy_reason),
                                strategy_score=float(strategy_score),
                                strategy_state=str(strategy_state),
                                strategy_recent_performance=float(strategy_recent_performance),
                                regime_fit=float(strategy_regime_fit),
                                session_fit=float(strategy_session_fit),
                                volatility_fit=float(strategy_volatility_fit),
                                pair_behavior_fit=float(strategy_pair_behavior_fit),
                                execution_quality_fit=float(strategy_execution_fit),
                                entry_timing_score=float(entry_timing_score_value),
                                structure_cleanliness_score=float(structure_cleanliness_score_value),
                                session_adjusted_score=float(session_adjusted),
                                selected_trade_band=str(trade_quality.band),
                                current_band_attempted=str(current_band_attempted),
                                current_band_target=str(current_band_target),
                                fallback_band_active=bool(fallback_band_active),
                                reason_higher_band_unavailable=str(reason_higher_band_unavailable),
                                best_A_plus_count=int(loop_band_counts.get("A+", 0)),
                                best_A_count=int(best_a_count),
                                best_B_plus_count=int(loop_band_counts.get("B+", 0)),
                                best_B_count=int(loop_band_counts.get("B", 0)),
                                best_C_count=int(loop_band_counts.get("C", 0)),
                                trade_quality_components=dict(trade_quality.components),
                                regime_state=str(regime_state),
                                regime_confidence=float(regime_confidence),
                                execution_quality_state=str(execution_quality.state),
                                execution_quality_score=float(execution_quality.score),
                                liquidity_alignment_score=float(liquidity_decision.score),
                                nearest_liquidity_above=float(liquidity_decision.nearest_liquidity_above),
                                nearest_liquidity_below=float(liquidity_decision.nearest_liquidity_below),
                                liquidity_sweep_detected=bool(liquidity_decision.liquidity_sweep_detected),
                                volatility_forecast_state=str(volatility_state),
                                pressure_proxy_score=float(regime.details.get("pressure_proxy_score", 0.0) or 0.0),
                                continuation_pressure=float(pressure_continuation),
                                exhaustion_signal=float(pressure_exhaustion),
                                absorption_signal=float(pressure_absorption),
                                last_decision_type="QUALITY_GATE",
                                last_decision_reason="trade_quality_below_floor",
                                primary_block_reason="trade_quality_below_floor",
                                secondary_block_reason="",
                                proof_bucket_state=str(proof_bucket_state),
                                proof_exception_used=False,
                                proof_exception_reason=str(proof_exception_reason),
                                exceptional_override_used=False,
                                exceptional_override_reason="",
                                why_non_native_pair_won="",
                                why_native_pair_lost_priority="",
                                pair_status=str(pair_state["pair_status"]),
                                pair_status_reason=str(pair_state["pair_status_reason"]),
                                pair_state_multiplier=float(pair_state["pair_state_multiplier"]),
                                rolling_expectancy_by_pair=float(pair_state["rolling_expectancy_by_pair"]),
                                rolling_pf_by_pair=float(pair_state["rolling_pf_by_pair"]),
                                rolling_expectancy_by_session=float(pair_state["rolling_expectancy_by_session"]),
                                rolling_pf_by_session=float(pair_state["rolling_pf_by_session"]),
                                rolling_win_rate_by_pair=float(pair_state["rolling_win_rate_by_pair"]),
                                rolling_win_rate_by_session=float(pair_state["rolling_win_rate_by_session"]),
                                false_break_rate=float(pair_state["false_break_rate"]),
                                management_quality_score=float(pair_state["management_quality_score"]),
                                today_session_wins=int(pair_state["today_session_wins"]),
                                today_session_losses=int(pair_state["today_session_losses"]),
                                today_session_trades=int(pair_state["today_session_trades"]),
                                why_pair_is_promoted=str(pair_state["why_pair_is_promoted"]),
                                why_pair_is_throttled=str(pair_state["why_pair_is_throttled"]),
                                lane_strength_multiplier=float(lane_strength_multiplier),
                                lane_score=float(lane_score),
                                session_density_score=float(session_density_score),
                                quality_cluster_score=float(quality_cluster_score),
                                cluster_mode_active=bool(cluster_mode_active),
                                stretch_mode_active=bool(cluster_mode_active),
                                current_capacity_mode="SCOUTING",
                                lane_adjustment_state=str(lane_adjustment_state),
                                daily_state=str(daily_state_name),
                            )
                            continue
                        if strategy_bucket_all_band_block:
                            mark_block(resolved_symbol, "strategy_bucket_all_band_block", symbol_status)
                            _update_runtime_metrics(
                                resolved_symbol,
                                now,
                                trade_quality_score=float(trade_quality.score),
                                trade_quality_band=str(trade_quality.band),
                                lane_name=str(trade_lane),
                                session_priority_profile=str(session_priority.session_priority_profile),
                                lane_session_priority=str(session_priority.lane_session_priority),
                                session_native_pair=bool(session_priority.session_native_pair),
                                session_priority_multiplier=float(session_priority.session_priority_multiplier),
                                pair_priority_rank_in_session=int(session_priority.pair_priority_rank_in_session),
                                lane_budget_share=float(session_priority.lane_budget_share),
                                strategy_key=str(strategy_key),
                                strategy_pool=list(resolved_strategy_pool),
                                strategy_pool_ranking=list(strategy_pool_ranking),
                                strategy_pool_winner=str(winning_strategy_key),
                                winning_strategy_reason=str(winning_strategy_reason),
                                strategy_score=float(strategy_score),
                                strategy_state=str(strategy_state),
                                strategy_recent_performance=float(strategy_recent_performance),
                                regime_fit=float(strategy_regime_fit),
                                session_fit=float(strategy_session_fit),
                                volatility_fit=float(strategy_volatility_fit),
                                pair_behavior_fit=float(strategy_pair_behavior_fit),
                                execution_quality_fit=float(strategy_execution_fit),
                                entry_timing_score=float(entry_timing_score_value),
                                structure_cleanliness_score=float(structure_cleanliness_score_value),
                                session_adjusted_score=float(session_adjusted),
                                selected_trade_band=str(trade_quality.band),
                                current_band_attempted=str(current_band_attempted),
                                current_band_target=str(current_band_target),
                                fallback_band_active=bool(fallback_band_active),
                                reason_higher_band_unavailable=str(reason_higher_band_unavailable),
                                best_A_plus_count=int(loop_band_counts.get("A+", 0)),
                                best_A_count=int(best_a_count),
                                best_B_plus_count=int(loop_band_counts.get("B+", 0)),
                                best_B_count=int(loop_band_counts.get("B", 0)),
                                best_C_count=int(loop_band_counts.get("C", 0)),
                                trade_quality_components=dict(trade_quality.components),
                                last_decision_type="QUALITY_GATE",
                                last_decision_reason="strategy_bucket_all_band_block",
                                primary_block_reason="strategy_bucket_all_band_block",
                                secondary_block_reason=str(strategy_bucket_state.get("strategy_bucket_reason", "") or ""),
                                what_would_make_it_pass="repair local pair+strategy+session+regime edge before allowing any band",
                                proof_bucket_state=str(proof_bucket_state),
                                proof_exception_used=False,
                                proof_exception_reason=str(proof_exception_reason),
                                exceptional_override_used=False,
                                exceptional_override_reason="",
                                why_non_native_pair_won="",
                                why_native_pair_lost_priority="",
                                pair_status=str(pair_state["pair_status"]),
                                pair_status_reason=str(pair_state["pair_status_reason"]),
                                pair_state_multiplier=float(pair_state["pair_state_multiplier"]),
                                rolling_expectancy_by_pair=float(pair_state["rolling_expectancy_by_pair"]),
                                rolling_pf_by_pair=float(pair_state["rolling_pf_by_pair"]),
                                rolling_expectancy_by_session=float(pair_state["rolling_expectancy_by_session"]),
                                rolling_pf_by_session=float(pair_state["rolling_pf_by_session"]),
                                rolling_win_rate_by_pair=float(pair_state["rolling_win_rate_by_pair"]),
                                rolling_win_rate_by_session=float(pair_state["rolling_win_rate_by_session"]),
                                false_break_rate=float(pair_state["false_break_rate"]),
                                management_quality_score=float(pair_state["management_quality_score"]),
                                today_session_wins=int(pair_state["today_session_wins"]),
                                today_session_losses=int(pair_state["today_session_losses"]),
                                today_session_trades=int(pair_state["today_session_trades"]),
                                why_pair_is_promoted=str(pair_state["why_pair_is_promoted"]),
                                why_pair_is_throttled=str(pair_state["why_pair_is_throttled"]),
                                lane_strength_multiplier=float(lane_strength_multiplier),
                                lane_score=float(lane_score),
                                session_density_score=float(session_density_score),
                                quality_cluster_score=float(quality_cluster_score),
                                cluster_mode_active=bool(cluster_mode_active),
                                stretch_mode_active=bool(cluster_mode_active),
                                current_capacity_mode="SCOUTING",
                                lane_adjustment_state=str(lane_adjustment_state),
                                daily_state=str(daily_state_name),
                            )
                            continue
                        if strategy_bucket_b_block:
                            mark_block(resolved_symbol, "strategy_bucket_b_block", symbol_status)
                            _update_runtime_metrics(
                                resolved_symbol,
                                now,
                                trade_quality_score=float(trade_quality.score),
                                trade_quality_band=str(trade_quality.band),
                                lane_name=str(trade_lane),
                                session_priority_profile=str(session_priority.session_priority_profile),
                                lane_session_priority=str(session_priority.lane_session_priority),
                                session_native_pair=bool(session_priority.session_native_pair),
                                session_priority_multiplier=float(session_priority.session_priority_multiplier),
                                pair_priority_rank_in_session=int(session_priority.pair_priority_rank_in_session),
                                lane_budget_share=float(session_priority.lane_budget_share),
                                strategy_key=str(strategy_key),
                                strategy_pool=list(resolved_strategy_pool),
                                strategy_pool_ranking=list(strategy_pool_ranking),
                                strategy_pool_winner=str(winning_strategy_key),
                                winning_strategy_reason=str(winning_strategy_reason),
                                strategy_score=float(strategy_score),
                                strategy_state=str(strategy_state),
                                strategy_recent_performance=float(strategy_recent_performance),
                                regime_fit=float(strategy_regime_fit),
                                session_fit=float(strategy_session_fit),
                                volatility_fit=float(strategy_volatility_fit),
                                pair_behavior_fit=float(strategy_pair_behavior_fit),
                                execution_quality_fit=float(strategy_execution_fit),
                                entry_timing_score=float(entry_timing_score_value),
                                structure_cleanliness_score=float(structure_cleanliness_score_value),
                                session_adjusted_score=float(session_adjusted),
                                selected_trade_band=str(trade_quality.band),
                                current_band_attempted=str(current_band_attempted),
                                current_band_target=str(current_band_target),
                                fallback_band_active=bool(fallback_band_active),
                                reason_higher_band_unavailable=str(reason_higher_band_unavailable),
                                best_A_plus_count=int(loop_band_counts.get("A+", 0)),
                                best_A_count=int(best_a_count),
                                best_B_plus_count=int(loop_band_counts.get("B+", 0)),
                                best_B_count=int(loop_band_counts.get("B", 0)),
                                best_C_count=int(loop_band_counts.get("C", 0)),
                                trade_quality_components=dict(trade_quality.components),
                                last_decision_type="QUALITY_GATE",
                                last_decision_reason="strategy_bucket_b_block",
                                primary_block_reason="strategy_bucket_b_block",
                                secondary_block_reason=str(strategy_bucket_state.get("strategy_bucket_reason", "") or ""),
                                what_would_make_it_pass="improve local pair+strategy+session expectancy or cut late-entry/false-break behavior",
                                proof_bucket_state=str(proof_bucket_state),
                                proof_exception_used=False,
                                proof_exception_reason=str(proof_exception_reason),
                                exceptional_override_used=False,
                                exceptional_override_reason="",
                                why_non_native_pair_won="",
                                why_native_pair_lost_priority="",
                                pair_status=str(pair_state["pair_status"]),
                                pair_status_reason=str(pair_state["pair_status_reason"]),
                                pair_state_multiplier=float(pair_state["pair_state_multiplier"]),
                                rolling_expectancy_by_pair=float(pair_state["rolling_expectancy_by_pair"]),
                                rolling_pf_by_pair=float(pair_state["rolling_pf_by_pair"]),
                                rolling_expectancy_by_session=float(pair_state["rolling_expectancy_by_session"]),
                                rolling_pf_by_session=float(pair_state["rolling_pf_by_session"]),
                                rolling_win_rate_by_pair=float(pair_state["rolling_win_rate_by_pair"]),
                                rolling_win_rate_by_session=float(pair_state["rolling_win_rate_by_session"]),
                                false_break_rate=float(pair_state["false_break_rate"]),
                                management_quality_score=float(pair_state["management_quality_score"]),
                                today_session_wins=int(pair_state["today_session_wins"]),
                                today_session_losses=int(pair_state["today_session_losses"]),
                                today_session_trades=int(pair_state["today_session_trades"]),
                                why_pair_is_promoted=str(pair_state["why_pair_is_promoted"]),
                                why_pair_is_throttled=str(pair_state["why_pair_is_throttled"]),
                                lane_strength_multiplier=float(lane_strength_multiplier),
                                lane_score=float(lane_score),
                                session_density_score=float(session_density_score),
                                quality_cluster_score=float(quality_cluster_score),
                                cluster_mode_active=bool(cluster_mode_active),
                                stretch_mode_active=bool(cluster_mode_active),
                                current_capacity_mode="SCOUTING",
                                lane_adjustment_state=str(lane_adjustment_state),
                                daily_state=str(daily_state_name),
                            )
                            continue
                        if band_rank == quality_band_rank("B") and not qualified_b_allowed:
                            mark_block(resolved_symbol, "trade_quality_b_unqualified", symbol_status)
                            _update_runtime_metrics(
                                resolved_symbol,
                                now,
                                trade_quality_score=float(trade_quality.score),
                                trade_quality_band=str(trade_quality.band),
                                lane_name=str(trade_lane),
                                session_priority_profile=str(session_priority.session_priority_profile),
                                lane_session_priority=str(session_priority.lane_session_priority),
                                session_native_pair=bool(session_priority.session_native_pair),
                                session_priority_multiplier=float(session_priority.session_priority_multiplier),
                                pair_priority_rank_in_session=int(session_priority.pair_priority_rank_in_session),
                                lane_budget_share=float(session_priority.lane_budget_share),
                                strategy_key=str(strategy_key),
                                strategy_pool=list(resolved_strategy_pool),
                                strategy_pool_ranking=list(strategy_pool_ranking),
                                strategy_pool_winner=str(winning_strategy_key),
                                winning_strategy_reason=str(winning_strategy_reason),
                                strategy_score=float(strategy_score),
                                strategy_state=str(strategy_state),
                                strategy_recent_performance=float(strategy_recent_performance),
                                regime_fit=float(strategy_regime_fit),
                                session_fit=float(strategy_session_fit),
                                volatility_fit=float(strategy_volatility_fit),
                                pair_behavior_fit=float(strategy_pair_behavior_fit),
                                execution_quality_fit=float(strategy_execution_fit),
                                entry_timing_score=float(entry_timing_score_value),
                                structure_cleanliness_score=float(structure_cleanliness_score_value),
                                session_adjusted_score=float(session_adjusted),
                                selected_trade_band=str(trade_quality.band),
                                current_band_attempted=str(current_band_attempted),
                                current_band_target=str(current_band_target),
                                fallback_band_active=bool(fallback_band_active),
                                reason_higher_band_unavailable=str(reason_higher_band_unavailable),
                                best_A_plus_count=int(loop_band_counts.get("A+", 0)),
                                best_A_count=int(best_a_count),
                                best_B_plus_count=int(loop_band_counts.get("B+", 0)),
                                best_B_count=int(loop_band_counts.get("B", 0)),
                                best_C_count=int(loop_band_counts.get("C", 0)),
                                trade_quality_components=dict(trade_quality.components),
                                last_decision_type="QUALITY_GATE",
                                last_decision_reason="trade_quality_b_unqualified",
                                primary_block_reason="trade_quality_b_unqualified",
                                secondary_block_reason="",
                                proof_bucket_state=str(proof_bucket_state),
                                proof_exception_used=False,
                                proof_exception_reason=str(proof_exception_reason),
                                exceptional_override_used=False,
                                exceptional_override_reason="",
                                why_non_native_pair_won="",
                                why_native_pair_lost_priority="",
                                pair_status=str(pair_state["pair_status"]),
                                pair_status_reason=str(pair_state["pair_status_reason"]),
                                pair_state_multiplier=float(pair_state["pair_state_multiplier"]),
                                rolling_expectancy_by_pair=float(pair_state["rolling_expectancy_by_pair"]),
                                rolling_pf_by_pair=float(pair_state["rolling_pf_by_pair"]),
                                rolling_expectancy_by_session=float(pair_state["rolling_expectancy_by_session"]),
                                rolling_pf_by_session=float(pair_state["rolling_pf_by_session"]),
                                rolling_win_rate_by_pair=float(pair_state["rolling_win_rate_by_pair"]),
                                rolling_win_rate_by_session=float(pair_state["rolling_win_rate_by_session"]),
                                false_break_rate=float(pair_state["false_break_rate"]),
                                management_quality_score=float(pair_state["management_quality_score"]),
                                today_session_wins=int(pair_state["today_session_wins"]),
                                today_session_losses=int(pair_state["today_session_losses"]),
                                today_session_trades=int(pair_state["today_session_trades"]),
                                why_pair_is_promoted=str(pair_state["why_pair_is_promoted"]),
                                why_pair_is_throttled=str(pair_state["why_pair_is_throttled"]),
                                lane_strength_multiplier=float(lane_strength_multiplier),
                                lane_score=float(lane_score),
                                session_density_score=float(session_density_score),
                                quality_cluster_score=float(quality_cluster_score),
                                cluster_mode_active=bool(cluster_mode_active),
                                stretch_mode_active=bool(cluster_mode_active),
                                current_capacity_mode="SCOUTING",
                                lane_adjustment_state=str(lane_adjustment_state),
                                daily_state=str(daily_state_name),
                            )
                            continue
                        if band_rank == quality_band_rank("C") and not proof_exception_allowed:
                            mark_block(resolved_symbol, "trade_quality_needs_exception", symbol_status)
                            _update_runtime_metrics(
                                resolved_symbol,
                                now,
                                trade_quality_score=float(trade_quality.score),
                                trade_quality_band=str(trade_quality.band),
                                lane_name=str(trade_lane),
                                session_priority_profile=str(session_priority.session_priority_profile),
                                lane_session_priority=str(session_priority.lane_session_priority),
                                session_native_pair=bool(session_priority.session_native_pair),
                                session_priority_multiplier=float(session_priority.session_priority_multiplier),
                                pair_priority_rank_in_session=int(session_priority.pair_priority_rank_in_session),
                                lane_budget_share=float(session_priority.lane_budget_share),
                                strategy_key=str(strategy_key),
                                strategy_pool=list(resolved_strategy_pool),
                                strategy_pool_ranking=list(strategy_pool_ranking),
                                strategy_pool_winner=str(winning_strategy_key),
                                winning_strategy_reason=str(winning_strategy_reason),
                                strategy_score=float(strategy_score),
                                strategy_state=str(strategy_state),
                                strategy_recent_performance=float(strategy_recent_performance),
                                regime_fit=float(strategy_regime_fit),
                                session_fit=float(strategy_session_fit),
                                volatility_fit=float(strategy_volatility_fit),
                                pair_behavior_fit=float(strategy_pair_behavior_fit),
                                execution_quality_fit=float(strategy_execution_fit),
                                entry_timing_score=float(entry_timing_score_value),
                                structure_cleanliness_score=float(structure_cleanliness_score_value),
                                session_adjusted_score=float(session_adjusted),
                                selected_trade_band=str(trade_quality.band),
                                current_band_attempted=str(current_band_attempted),
                                current_band_target=str(current_band_target),
                                fallback_band_active=bool(fallback_band_active),
                                reason_higher_band_unavailable=str(reason_higher_band_unavailable),
                                best_A_plus_count=int(loop_band_counts.get("A+", 0)),
                                best_A_count=int(best_a_count),
                                best_B_plus_count=int(loop_band_counts.get("B+", 0)),
                                best_B_count=int(loop_band_counts.get("B", 0)),
                                best_C_count=int(loop_band_counts.get("C", 0)),
                                trade_quality_components=dict(trade_quality.components),
                                last_decision_type="QUALITY_GATE",
                                last_decision_reason="trade_quality_needs_exception",
                                primary_block_reason="trade_quality_needs_exception",
                                secondary_block_reason="",
                                proof_bucket_state=str(proof_bucket_state),
                                proof_exception_used=False,
                                proof_exception_reason=str(proof_exception_reason),
                                exceptional_override_used=False,
                                exceptional_override_reason="",
                                why_non_native_pair_won="",
                                why_native_pair_lost_priority="",
                                pair_status=str(pair_state["pair_status"]),
                                pair_status_reason=str(pair_state["pair_status_reason"]),
                                pair_state_multiplier=float(pair_state["pair_state_multiplier"]),
                                rolling_expectancy_by_pair=float(pair_state["rolling_expectancy_by_pair"]),
                                rolling_pf_by_pair=float(pair_state["rolling_pf_by_pair"]),
                                rolling_expectancy_by_session=float(pair_state["rolling_expectancy_by_session"]),
                                rolling_pf_by_session=float(pair_state["rolling_pf_by_session"]),
                                rolling_win_rate_by_pair=float(pair_state["rolling_win_rate_by_pair"]),
                                rolling_win_rate_by_session=float(pair_state["rolling_win_rate_by_session"]),
                                false_break_rate=float(pair_state["false_break_rate"]),
                                management_quality_score=float(pair_state["management_quality_score"]),
                                today_session_wins=int(pair_state["today_session_wins"]),
                                today_session_losses=int(pair_state["today_session_losses"]),
                                today_session_trades=int(pair_state["today_session_trades"]),
                                why_pair_is_promoted=str(pair_state["why_pair_is_promoted"]),
                                why_pair_is_throttled=str(pair_state["why_pair_is_throttled"]),
                                lane_strength_multiplier=float(lane_strength_multiplier),
                                lane_score=float(lane_score),
                                session_density_score=float(session_density_score),
                                quality_cluster_score=float(quality_cluster_score),
                                cluster_mode_active=bool(cluster_mode_active),
                                stretch_mode_active=bool(cluster_mode_active),
                                current_capacity_mode="SCOUTING",
                                lane_adjustment_state=str(lane_adjustment_state),
                                daily_state=str(daily_state_name),
                            )
                            continue

                        session_override = session_priority_override_decision(
                            session_name=session_context.session_name,
                            candidate_symbol=resolved_symbol,
                            candidate_band=str(trade_quality.band),
                            candidate_adjusted_score=float(session_adjusted),
                            candidate_probability=float(preliminary_ai.probability),
                            candidate_native=bool(session_priority.session_native_pair),
                            candidate_lane_priority=str(session_priority.lane_session_priority),
                            candidate_override_delta=float(session_priority.native_override_delta),
                            candidate_override_band_delta=float(session_priority.native_override_band_delta),
                            best_native_symbol=str(session_native_leader.get("symbol", "")),
                            best_native_band=str(session_native_leader.get("band", "")),
                            best_native_adjusted_score=float(session_native_leader.get("adjusted_score", 0.0)),
                            best_native_probability=float(session_native_leader.get("probability", 0.0)),
                            throughput_recovery_active=bool(throughput_recovery_active),
                            trajectory_catchup_pressure=float(candidate.meta.get("trajectory_catchup_pressure", 0.0) or 0.0),
                        )
                        paper_sim_session_priority_override = bool(
                            paper_sim
                            and resolved_symbol == "BTCUSD"
                            and str(trade_quality.band) in {"A+", "A", "A-"}
                            and float(preliminary_ai.probability) >= 0.80
                        )
                        if not session_override.allowed and not paper_sim_session_priority_override:
                            mark_block(resolved_symbol, "session_native_priority_block", symbol_status)
                            _update_runtime_metrics(
                                resolved_symbol,
                                now,
                                trade_quality_score=float(trade_quality.score),
                                trade_quality_band=str(trade_quality.band),
                                lane_name=str(trade_lane),
                                session_priority_profile=str(session_priority.session_priority_profile),
                                lane_session_priority=str(session_priority.lane_session_priority),
                                session_native_pair=bool(session_priority.session_native_pair),
                                session_priority_multiplier=float(session_priority.session_priority_multiplier),
                                pair_priority_rank_in_session=int(session_priority.pair_priority_rank_in_session),
                                lane_budget_share=float(session_priority.lane_budget_share),
                                session_adjusted_score=float(session_adjusted),
                                selected_trade_band=str(trade_quality.band),
                                current_band_attempted=str(current_band_attempted),
                                current_band_target=str(current_band_target),
                                fallback_band_active=bool(fallback_band_active),
                                reason_higher_band_unavailable=str(reason_higher_band_unavailable),
                                best_A_plus_count=int(loop_band_counts.get("A+", 0)),
                                best_A_count=int(best_a_count),
                                best_B_plus_count=int(loop_band_counts.get("B+", 0)),
                                best_B_count=int(loop_band_counts.get("B", 0)),
                                best_C_count=int(loop_band_counts.get("C", 0)),
                                trade_quality_components=dict(trade_quality.components),
                                last_decision_type="SESSION_PRIORITY",
                                last_decision_reason="session_native_priority_block",
                                primary_block_reason="session_native_priority_block",
                                secondary_block_reason=str(session_override.why_native_pair_lost_priority),
                                what_would_make_it_pass="be materially stronger than the best session-native candidate",
                                proof_bucket_state=str(proof_bucket_state),
                                proof_exception_used=False,
                                proof_exception_reason=str(proof_exception_reason),
                                exceptional_override_used=False,
                                exceptional_override_reason="",
                                why_non_native_pair_won="",
                                why_native_pair_lost_priority=str(session_override.why_native_pair_lost_priority),
                                pair_status=str(pair_state["pair_status"]),
                                pair_status_reason=str(pair_state["pair_status_reason"]),
                                pair_state_multiplier=float(pair_state["pair_state_multiplier"]),
                                rolling_expectancy_by_pair=float(pair_state["rolling_expectancy_by_pair"]),
                                rolling_pf_by_pair=float(pair_state["rolling_pf_by_pair"]),
                                rolling_expectancy_by_session=float(pair_state["rolling_expectancy_by_session"]),
                                rolling_pf_by_session=float(pair_state["rolling_pf_by_session"]),
                                rolling_win_rate_by_pair=float(pair_state["rolling_win_rate_by_pair"]),
                                rolling_win_rate_by_session=float(pair_state["rolling_win_rate_by_session"]),
                                false_break_rate=float(pair_state["false_break_rate"]),
                                management_quality_score=float(pair_state["management_quality_score"]),
                                today_session_wins=int(pair_state["today_session_wins"]),
                                today_session_losses=int(pair_state["today_session_losses"]),
                                today_session_trades=int(pair_state["today_session_trades"]),
                                why_pair_is_promoted=str(pair_state["why_pair_is_promoted"]),
                                why_pair_is_throttled=str(pair_state["why_pair_is_throttled"]),
                                lane_strength_multiplier=float(lane_strength_multiplier),
                                lane_score=float(lane_score),
                                session_density_score=float(session_density_score),
                                quality_cluster_score=float(quality_cluster_score),
                                cluster_mode_active=bool(cluster_mode_active),
                                stretch_mode_active=bool(cluster_mode_active),
                                current_capacity_mode="SCOUTING",
                                lane_adjustment_state=str(lane_adjustment_state),
                                daily_state=str(daily_state_name),
                            )
                            continue
                        if bool(session_priority.session_native_pair) and _session_native_leader_should_update(
                            session_native_leader,
                            band_rank=int(band_rank),
                            adjusted_score=float(session_adjusted),
                            probability=float(preliminary_ai.probability),
                        ):
                            session_native_leader = {
                                "symbol": str(resolved_symbol),
                                "band": str(trade_quality.band),
                                "band_rank": int(band_rank),
                                "adjusted_score": float(session_adjusted),
                                "probability": float(preliminary_ai.probability),
                            }

                        requested_risk_pct = float(current_scaling_state.get("current_risk_pct", risk_config["risk_per_trade"]))
                        strategy_risk_cap: float | None = None
                        skip_micro_risk_clamp = False
                        max_loss_usd_floor = 0.0
                        non_grid_hard_cap = float(risk_scaling_config.get("max_effective_risk_pct_non_grid", 0.02))
                        if _is_xau_grid_setup(candidate.setup):
                            cycle_risk_default = float(xau_grid_risk_config.get("cycle_risk_pct_default", requested_risk_pct))
                            cycle_risk_min = float(xau_grid_risk_config.get("cycle_risk_pct_min", cycle_risk_default))
                            cycle_risk_max = float(
                                min(
                                    xau_grid_risk_config.get("cycle_risk_pct_max", cycle_risk_default),
                                    risk_scaling_config.get("max_effective_risk_pct_xau_grid", xau_grid_risk_config.get("cycle_risk_pct_max", cycle_risk_default)),
                                )
                            )
                            requested_risk_pct = max(cycle_risk_min, min(cycle_risk_default, cycle_risk_max))
                            strategy_risk_cap = cycle_risk_max
                            skip_micro_risk_clamp = True
                            max_loss_usd_floor = float(xau_grid_risk_config.get("max_loss_usd_floor", 0.0))
                        else:
                            strategy_risk_cap = non_grid_hard_cap

                        no_trade_boost_config: dict[str, Any] = {}
                        if isinstance(risk_ramp_config, dict) and risk_ramp_config:
                            no_trade_boost_config = risk_ramp_config
                        else:
                            candidate_boost = risk_config.get("no_trade_boost", {})
                            if isinstance(candidate_boost, dict):
                                no_trade_boost_config = candidate_boost
                        no_trade_elapsed_minutes = 0.0
                        if last_trade_opened_at is not None:
                            no_trade_elapsed_minutes = max(0.0, (now - last_trade_opened_at).total_seconds() / 60.0)
                        atr_spike_active = (
                            float(row["m5_atr_avg_20"]) > 0
                            and float(row["m5_atr_14"]) > (float(row["m5_atr_avg_20"]) * float(risk_config["atr_spike_multiple"]))
                        )
                        daily_loss_locked = (
                            float(global_stats.daily_dd_pct_live) >= float(risk_config.get("soft_daily_dd_pct", 0.03))
                            or float(global_stats.daily_pnl_pct) <= -float(risk_config["circuit_breaker_daily_loss"])
                        )
                        idle_minutes_for_scaling = float(risk_scaling_config.get("idle_minutes_for_boost", 45))
                        require_daily_non_negative = bool(risk_scaling_config.get("require_daily_non_negative", True))
                        require_no_open_positions = bool(risk_scaling_config.get("require_no_open_positions", True))
                        daily_scaling_ok = float(global_stats.daily_pnl_pct) >= 0.0 if require_daily_non_negative else True
                        no_open_scaling_ok = len(open_positions_journal) == 0 if require_no_open_positions else True
                        no_trade_boost_eligible = (
                            bool(risk_scaling_config.get("enabled", True))
                            and bool(news_decision.safe)
                            and str(getattr(news_decision, "state", "NEWS_SAFE")).upper() != "NEWS_CAUTION"
                            and (kill_status.level is None)
                            and (not daily_loss_locked)
                            and (not atr_spike_active)
                            and spread_points <= float(risk_config["max_spread_points"])
                            and daily_scaling_ok
                            and no_open_scaling_ok
                            and no_trade_elapsed_minutes >= idle_minutes_for_scaling
                        )

                        symbol_tick_value = symbol_info.get("trade_tick_value")
                        if symbol_tick_value is None:
                            symbol_tick_value = symbol_info.get("trade_tick_value_profit")
                        if symbol_tick_value is None:
                            symbol_tick_value = symbol_info.get("trade_tick_value_loss")

                        projected_cycle_risk = 0.0
                        if _is_xau_grid_setup(candidate.setup):
                            projected_cycle_risk = _projected_cycle_risk_usd(
                                positions=symbol_open_positions,
                                contract_size=float(symbol_info["trade_contract_size"]),
                                side=candidate.side,
                            )

                        risk_stats = replace(
                            symbol_stats,
                            trades_today=max(int(symbol_stats.trades_today), int(global_stats.trades_today)),
                            daily_pnl_pct=float(global_stats.daily_pnl_pct),
                            day_start_equity=float(global_stats.day_start_equity),
                            day_high_equity=float(global_stats.day_high_equity),
                            daily_dd_pct_live=float(global_stats.daily_dd_pct_live),
                            rolling_drawdown_pct=float(global_stats.rolling_drawdown_pct),
                            absolute_drawdown_pct=float(global_stats.absolute_drawdown_pct),
                            soft_dd_trade_count=int(global_stats.soft_dd_trade_count),
                            consecutive_losses=max(int(symbol_stats.consecutive_losses), int(global_stats.consecutive_losses)),
                            cooldown_trades_remaining=max(
                                _effective_cooldown_trades_remaining(symbol_stats, now=now),
                                _effective_cooldown_trades_remaining(global_stats, now=now),
                            ),
                            closed_trades_total=max(int(symbol_stats.closed_trades_total), int(global_stats.closed_trades_total)),
                            winning_streak=max(int(symbol_stats.winning_streak), int(global_stats.winning_streak)),
                        )
                        learning_policy: dict[str, Any] = {}
                        if "learning_brain" in runtime:
                            brain = runtime.get("learning_brain")
                            if brain is not None and hasattr(brain, "live_policy_snapshot"):
                                try:
                                    learning_policy = brain.live_policy_snapshot(
                                        symbol=resolved_symbol,
                                        setup=str(candidate.setup),
                                    )
                                except Exception:
                                    learning_policy = {}
                        learning_policy = _augment_learning_policy_for_density(
                            symbol_key=resolved_symbol,
                            session_name=session_context.session_name,
                            learning_policy=learning_policy,
                            current_scaling_state=current_scaling_state,
                        )
                        pair_directive = dict(learning_policy.get("pair_directive") or {})
                        setup_hour_directive = dict(learning_policy.get("setup_hour_directive") or {})
                        learning_bundle = dict(learning_policy.get("bundle") or {})
                        learning_bundle_local_summary = dict(learning_bundle.get("local_summary") or {})
                        expectancy_directive = dict(setup_hour_directive or pair_directive)
                        current_sydney_hour = int(now.astimezone(SYDNEY).hour)
                        strong_hours_sydney = {
                            int(item)
                            for item in expectancy_directive.get("strong_hours_sydney", [])
                            if str(item).strip().lstrip("-").isdigit()
                        }
                        weak_hours_sydney = {
                            int(item)
                            for item in expectancy_directive.get("weak_hours_sydney", [])
                            if str(item).strip().lstrip("-").isdigit()
                        }
                        lane_expectancy_score = clamp(
                            float(expectancy_directive.get("hour_expectancy_score", pair_directive.get("hour_expectancy_score", lane_score)) or lane_score),
                            0.0,
                            1.0,
                        )
                        lane_expectancy_multiplier = clamp(
                            float(expectancy_directive.get("lane_expectancy_multiplier", pair_directive.get("lane_expectancy_multiplier", 1.0)) or 1.0),
                            0.80,
                            1.35,
                        )
                        if current_sydney_hour in strong_hours_sydney:
                            lane_expectancy_score = clamp(lane_expectancy_score + 0.06, 0.0, 1.0)
                            lane_expectancy_multiplier = clamp(lane_expectancy_multiplier + 0.05, 0.80, 1.35)
                        elif current_sydney_hour in weak_hours_sydney:
                            lane_expectancy_score = clamp(lane_expectancy_score - 0.08, 0.0, 1.0)
                            lane_expectancy_multiplier = clamp(lane_expectancy_multiplier - 0.06, 0.80, 1.35)
                        hot_hand_active = bool(expectancy_directive.get("hot_hand_active", pair_directive.get("hot_hand_active", False)))
                        hot_hand_score = clamp(
                            float(expectancy_directive.get("hot_hand_score", pair_directive.get("hot_hand_score", 0.0)) or 0.0),
                            0.0,
                            1.0,
                        )
                        session_bankroll_bias = clamp(
                            float(expectancy_directive.get("session_bankroll_bias", pair_directive.get("session_bankroll_bias", 1.0)) or 1.0),
                            0.85,
                            1.35,
                        )
                        profit_recycle_active = bool(
                            learning_bundle_local_summary.get("profit_recycle_active", pair_directive.get("profit_recycle_active", False))
                        )
                        profit_recycle_boost = clamp(
                            float(
                                learning_bundle_local_summary.get(
                                    "profit_recycle_boost",
                                    pair_directive.get("profit_recycle_boost", 0.0),
                                )
                                or 0.0
                            ),
                            0.0,
                            0.25,
                        )
                        close_winners_score = clamp(
                            float(
                                expectancy_directive.get(
                                    "close_winners_score",
                                    pair_directive.get("close_winners_score", 0.5),
                                )
                                or 0.5
                            ),
                            0.0,
                            1.0,
                        )
                        recovery_mode_active_runtime = bool(symbol_runtime_defaults.get("recovery_mode_active", False))
                        recovery_risk_multiplier = clamp(
                            float(risk_config.get("recovery_risk_multiplier", system_config.get("recovery_risk_multiplier", 0.60)) or 0.60),
                            0.25,
                            1.0,
                        )
                        phase_risk_pct = float(current_scaling_state.get("current_risk_pct", risk_config["risk_per_trade"]))
                        effective_requested_risk_pct = max(float(requested_risk_pct), phase_risk_pct)
                        effective_hard_risk_cap = max(float(risk_config["hard_risk_cap"]), phase_risk_pct)
                        if recovery_mode_active_runtime:
                            phase_risk_pct *= recovery_risk_multiplier
                            effective_requested_risk_pct = max(0.0001, effective_requested_risk_pct * recovery_risk_multiplier)
                            effective_hard_risk_cap = max(
                                effective_requested_risk_pct,
                                effective_hard_risk_cap * recovery_risk_multiplier,
                            )
                        funded_config = dict(risk_config.get("funded") or {})
                        funded_auto_detected = detect_funded_account_mode(
                            *(account.get(key) for key in ("name", "server", "company", "broker"))
                        )
                        funded_mode_active = bool(funded_config.get("enabled", funded_auto_detected))
                        aggression_config = (
                            bridge_orchestrator_config.get("aggression", {})
                            if isinstance(bridge_orchestrator_config.get("aggression"), dict)
                            else {}
                        )
                        lead_lag_config = (
                            bridge_orchestrator_config.get("lead_lag", {})
                            if isinstance(bridge_orchestrator_config.get("lead_lag"), dict)
                            else {}
                        )
                        event_playbook_config = (
                            bridge_orchestrator_config.get("event_playbooks", {})
                            if isinstance(bridge_orchestrator_config.get("event_playbooks"), dict)
                            else {}
                        )
                        candidate_microstructure_score = build_microstructure_score(
                            context.get("microstructure") if isinstance(context.get("microstructure"), dict) else {},
                            symbol=resolved_symbol,
                            side=str(candidate.side),
                        )
                        candidate_lead_lag_snapshot = build_lead_lag_snapshot(
                            symbol=resolved_symbol,
                            side=str(candidate.side),
                            context={
                                "dxy_ret_1": _safe_float(row.get("dxy_ret_1"), 0.0),
                                "dxy_ret_5": _safe_float(row.get("dxy_ret_5"), 0.0),
                                "us10y_ret_5": _safe_float(row.get("us10y_ret_5"), _safe_float(row.get("yield_proxy_ret_5"), 0.0)),
                                "yield_proxy_ret_5": _safe_float(row.get("yield_proxy_ret_5"), 0.0),
                                "nas100_ret_5": _safe_float(row.get("nas100_ret_5"), _safe_float(row.get("nas_ret_5"), 0.0)),
                                "nas_ret_5": _safe_float(row.get("nas_ret_5"), 0.0),
                                "usoil_ret_5": _safe_float(row.get("usoil_ret_5"), _safe_float(row.get("oil_ret_5"), 0.0)),
                                "oil_ret_5": _safe_float(row.get("oil_ret_5"), 0.0),
                                "btc_ret_5": _safe_float(row.get("btc_ret_5"), 0.0),
                                "btc_ret_1": _safe_float(row.get("btc_ret_1"), 0.0),
                                "eurusd_ret_5": _safe_float(row.get("eurusd_ret_5"), 0.0),
                                "usdjpy_ret_5": _safe_float(row.get("usdjpy_ret_5"), 0.0),
                                "usd_liquidity_score": _safe_float(router_diagnostics.get("usd_liquidity_score"), 0.0),
                                "risk_sentiment_score": _safe_float(router_diagnostics.get("risk_sentiment_score"), 0.0),
                                "weekend_volatility_score": _safe_float(router_diagnostics.get("weekend_volatility_score"), 0.0),
                                "btc_weekend_gap_score": _safe_float(router_diagnostics.get("btc_weekend_gap_score"), 0.0),
                            },
                            weights_config=(
                                lead_lag_config.get("weights_by_symbol", {})
                                if isinstance(lead_lag_config.get("weights_by_symbol"), dict)
                                else {}
                            ),
                        )
                        candidate_event_directive = build_event_directive(
                            symbol=resolved_symbol,
                            news_snapshot=context.get("news_snapshot") if isinstance(context.get("news_snapshot"), dict) else news_snapshot,
                            lead_lag=candidate_lead_lag_snapshot,
                            microstructure=candidate_microstructure_score,
                            playbook_map=(
                                event_playbook_config.get("playbook_map", {})
                                if isinstance(event_playbook_config.get("playbook_map"), dict)
                                else {}
                            ),
                        )
                        candidate_execution_profile = dict(context.get("execution_minute_profile") or {})
                        candidate.meta["microstructure_score"] = candidate_microstructure_score.as_dict()
                        candidate.meta["lead_lag_snapshot"] = candidate_lead_lag_snapshot.as_dict()
                        candidate.meta["event_directive"] = candidate_event_directive.as_dict()
                        candidate.meta["event_playbook"] = str(candidate_event_directive.playbook)
                        candidate.meta["execution_minute_profile"] = dict(candidate_execution_profile)
                        candidate_lane_loss_streak = _consecutive_lane_losses(recent_closed_trades, trade_lane)
                        candidate_spread_atr_points = _normalize_runtime_spread_points(
                            resolved_symbol,
                            max(
                                0.0,
                                float(
                                    row.get("m5_spread_atr_14")
                                    or row.get("m15_spread_atr_14")
                                    or row.get("m5_spread_mean_14")
                                    or row.get("m15_spread_mean_14")
                                    or 0.0
                                ),
                            ),
                            symbol_info=symbol_info,
                            max_spread_points=float(risk_config["max_spread_points"]),
                        )
                        candidate.meta["lane_consecutive_losses"] = int(candidate_lane_loss_streak)
                        candidate.meta["daily_green_streak"] = int(current_scaling_state.get("daily_green_streak", 0))
                        risk_decision = risk_engine.evaluate(
                            RiskInputs(
                                symbol=resolved_symbol,
                                mode=mode,
                                live_enabled=bool(system_config.get("live_trading_enabled", False)),
                                live_allowed=live_allowed,
                                current_time=now,
                                spread_points=spread_points,
                                entry_price=entry_price,
                                stop_price=stop_price,
                                tp_price=tp_price,
                                equity=float(account["equity"]),
                                account_balance=float(account["balance"]),
                                margin_free=float(account["margin_free"]),
                                open_positions=len(open_positions_journal),
                                open_positions_symbol=len(symbol_open_positions),
                                same_direction_positions=sum(
                                    1
                                    for position in symbol_open_positions
                                    if str(position["side"]).upper() == candidate.side
                                ),
                                session_multiplier=symbol_session_multiplier,
                                symbol_point=float(symbol_info["point"]),
                                contract_size=float(symbol_info["trade_contract_size"]),
                                volume_min=max(float(symbol_info["volume_min"]), float(micro_config.get("min_lot", 0.0)) if micro_active else float(symbol_info["volume_min"])),
                                volume_max=float(symbol_info["volume_max"]),
                                volume_step=float(symbol_info["volume_step"]),
                                requested_risk_pct=effective_requested_risk_pct,
                                hard_risk_cap=effective_hard_risk_cap,
                                max_positions=effective_max_positions_total,
                                max_positions_per_symbol=effective_max_positions_per_symbol,
                                max_daily_loss=float(risk_config["max_daily_loss"]),
                                circuit_breaker_daily_loss=float(risk_config["circuit_breaker_daily_loss"]),
                                max_drawdown_kill=float(risk_config["max_drawdown_kill"]),
                                absolute_drawdown_hard_stop=float(risk_config["absolute_drawdown_hard_stop"]),
                                max_spread_points=float(risk_config["max_spread_points"]),
                                atr_current=float(row["m5_atr_14"]),
                                atr_average=float(row["m5_atr_avg_20"]),
                                atr_spike_multiple=float(risk_config["atr_spike_multiple"]),
                                volatility_pause_minutes=int(risk_config["volatility_shock_pause_minutes"]),
                                regime=regime.label,
                                ai_probability=preliminary_ai.probability,
                                ai_size_multiplier=preliminary_ai.size_multiplier,
                                portfolio_size_multiplier=portfolio_decision.size_multiplier,
                                recent_trades_last_hour=recent_trades,
                                max_trades_per_hour=int(current_scaling_state.get("hourly_base_target", risk_config["max_trades_per_hour"])),
                                use_kelly=bool(risk_config["use_kelly"]),
                                kelly_fraction=float(risk_config["kelly_fraction"]),
                                use_fixed_lot=effective_use_fixed_lot,
                                fixed_lot=effective_fixed_lot,
                                stats=risk_stats,
                                friday_cutoff_hour=friday_cutoff,
                                weekend_trading_allowed=weekend_trading_allowed,
                                micro_enabled=bool(micro_config.get("enabled", False)),
                                micro_min_trades=int(micro_config.get("min_trades_for_normal_risk", 50)),
                                micro_risk_pct_floor=float(micro_config.get("risk_pct_floor", 0.001)),
                                micro_risk_pct_ceiling=float(micro_config.get("risk_pct_ceiling", 0.0025)),
                                micro_daily_loss_pct=float(micro_config.get("daily_loss_pct_limit", 0.01)),
                                first_trade_protection_trades=int(micro_config.get("first_trade_protection_trades", 2)),
                                first_trade_size_factor=float(micro_config.get("first_trade_size_factor", 0.5)),
                                first_trade_max_sl_atr=float(micro_config.get("first_trade_max_sl_atr", 1.4)),
                                anti_martingale_enabled=bool((risk_config.get("anti_martingale", {}) or {}).get("enabled", False)),
                                anti_martingale_step=float((risk_config.get("anti_martingale", {}) or {}).get("step", 0.05)),
                                anti_martingale_cap=float((risk_config.get("anti_martingale", {}) or {}).get("cap", 1.25)),
                                min_stop_distance_points=max(
                                    0.0,
                                    _effective_min_stop_distance_points(
                                        float(micro_config.get("min_stop_distance_points", risk_config.get("min_stop_distance_points", 0.0))),
                                        float(pre_risk_stop_geometry.get("min_stop_distance_points", 0.0)),
                                    ),
                                ),
                                projected_open_risk_usd=float(projected_open_risk_usd),
                                projected_cycle_risk_usd=float(projected_cycle_risk),
                                micro_max_loss_usd=float(micro_config.get("micro_max_loss_usd", 2.5)),
                                micro_total_risk_usd=float(micro_config.get("micro_total_risk_usd", 5.0)),
                                setup=str(candidate.setup),
                                candidate_stop_atr=float(candidate.stop_atr),
                                symbol_digits=int(symbol_info.get("digits")) if symbol_info.get("digits") is not None else None,
                                symbol_tick_size=float(symbol_info.get("trade_tick_size")) if symbol_info.get("trade_tick_size") is not None else None,
                                symbol_tick_value=float(symbol_tick_value) if symbol_tick_value is not None else None,
                                margin_per_lot=float(symbol_info.get("margin_initial")) if symbol_info.get("margin_initial") is not None else None,
                                account_leverage=float(account.get("leverage")) if account.get("leverage") is not None else None,
                                strategy_risk_cap=strategy_risk_cap,
                                skip_micro_risk_clamp=skip_micro_risk_clamp,
                                max_loss_usd_floor=max_loss_usd_floor,
                                no_trade_boost_enabled=bool(no_trade_boost_config.get("enabled", False)),
                                no_trade_boost_eligible=no_trade_boost_eligible,
                                no_trade_boost_elapsed_minutes=no_trade_elapsed_minutes,
                                no_trade_boost_after_minutes=int(no_trade_boost_config.get("after_minutes", no_trade_boost_config.get("no_trade_minutes", idle_minutes_for_scaling))),
                                no_trade_boost_interval_minutes=int(no_trade_boost_config.get("interval_minutes", 15)),
                                no_trade_boost_step_pct=float(no_trade_boost_config.get("step_pct", 0.02)),
                                no_trade_boost_max_pct=float(
                                    min(
                                        no_trade_boost_config.get("max_risk_pct", no_trade_boost_config.get("max_pct", 0.10)),
                                        float(
                                            min(
                                                xau_grid_risk_config.get("cycle_risk_pct_max", 0.10),
                                                risk_scaling_config.get("max_effective_risk_pct_xau_grid", 0.10),
                                            )
                                        ) if _is_xau_grid_setup(candidate.setup) else non_grid_hard_cap,
                                    )
                                ),
                                account_currency=str(micro_config.get("account_currency", "NZD")).upper(),
                                economics_source=str(symbol_info.get("economics_source", "mt5_client")),
                                confluence_score=float(candidate.confluence_score),
                                expected_value_r=float(preliminary_ai.expected_value_r),
                                max_trades_per_day=int(current_scaling_state.get("current_daily_trade_cap", 2)),
                                overflow_max_trades_per_day=int(current_scaling_state.get("current_overflow_daily_trade_cap", current_scaling_state.get("current_daily_trade_cap", 2))),
                                bootstrap_enabled=bool(micro_config.get("bootstrap_enabled", True)),
                                bootstrap_equity_threshold=float(micro_config.get("bootstrap_equity_threshold", 160.0)),
                                bootstrap_per_trade_hard_cap=float(micro_config.get("bootstrap_per_trade_hard_cap", micro_config.get("micro_max_loss_usd", 2.5))),
                                bootstrap_total_exposure_cap=float(micro_config.get("bootstrap_total_exposure_cap", micro_config.get("micro_total_risk_usd", 5.0))),
                                bootstrap_min_risk_amount=float(micro_config.get("bootstrap_min_risk_amount", 1.0)),
                                bootstrap_min_lot_risk_multiplier=float(micro_config.get("bootstrap_min_lot_risk_multiplier", 6.0)),
                                bootstrap_drawdown_kill=float(micro_config.get("bootstrap_drawdown_kill", risk_config.get("max_drawdown_kill", 0.05))),
                                bootstrap_first_trade_max_sl_atr=float(micro_config.get("bootstrap_first_trade_max_sl_atr", 3.0)),
                                soft_daily_dd_pct=float(risk_config.get("soft_daily_dd_pct", 0.03)),
                                hard_daily_dd_pct=float(risk_config.get("hard_daily_dd_pct", 0.05)),
                                soft_dd_probability_floor=float(risk_config.get("soft_dd_probability_floor", 0.72)),
                                soft_dd_expected_value_floor=float(risk_config.get("soft_dd_expected_value_floor", 0.35)),
                                soft_dd_confluence_floor=float(risk_config.get("soft_dd_confluence_floor", 3.8)),
                                current_phase=str(current_scaling_state.get("current_phase", "PHASE_1")),
                                current_ai_threshold_mode=str(current_scaling_state.get("current_ai_threshold_mode", "conservative")),
                                current_base_risk_pct=float(current_scaling_state.get("current_risk_pct", phase_risk_pct)),
                                current_max_risk_pct=float(current_scaling_state.get("current_max_risk_pct", max(float(risk_config["hard_risk_cap"]), phase_risk_pct))),
                                lane_consecutive_losses=int(candidate_lane_loss_streak),
                                same_lane_loss_caution_streak=int(aggression_config.get("same_lane_loss_caution_streak", 3)),
                                low_equity_safety_enabled=bool(aggression_config.get("low_equity_safety_enabled", True)),
                                low_equity_max_equity=float(aggression_config.get("low_equity_max_equity", 300.0)),
                                low_equity_risk_floor_pct=float(aggression_config.get("low_equity_risk_floor_pct", 0.01)),
                                low_equity_risk_ceiling_pct=float(aggression_config.get("low_equity_risk_ceiling_pct", 0.02)),
                                low_equity_monte_carlo_floor=float(aggression_config.get("low_equity_monte_carlo_floor", 0.88)),
                                candidate_monte_carlo_win_rate=float(candidate.meta.get("mc_win_rate", 0.0) or 0.0),
                                spread_atr_reference_points=float(candidate_spread_atr_points),
                                low_equity_spread_atr_cap=float(aggression_config.get("low_equity_spread_atr_cap", 1.2)),
                                trade_quality_score=float(trade_quality.score),
                                trade_quality_band=str(trade_quality.band),
                                trade_quality_detail=str(trade_quality.band),
                                quality_size_multiplier=float(trade_quality.size_multiplier)
                                * float(strategy_state_size_multiplier)
                                * float(tier_size_multiplier),
                                regime_confidence=float(regime_confidence),
                                execution_quality_score=float(execution_quality.score),
                                execution_quality_state=str(execution_quality.state),
                                spread_quality_score=float(trade_quality.components.get("spread_quality", 0.0)),
                                session_quality_score=float(session_quality_score),
                                execution_minute_quality_score=float(candidate_execution_profile.get("quality_score", row.get("execution_minute_quality_score", 0.70)) or 0.70),
                                execution_minute_size_multiplier=float(candidate_execution_profile.get("size_multiplier", row.get("execution_minute_size_multiplier", 1.0)) or 1.0),
                                execution_minute_state=str(candidate_execution_profile.get("state", row.get("execution_minute_state", "MIXED")) or "MIXED"),
                                recent_expectancy_r=float(recent_expectancy_r),
                                recent_win_rate=float(recent_win_rate),
                                total_open_risk_pct=float(projected_open_risk_usd) / max(float(account["equity"]), 1e-9),
                                correlation_multiplier=float(portfolio_decision.size_multiplier),
                                correlation_cluster=str(portfolio_decision.exposure_cluster_detected or ""),
                                news_state=str(getattr(news_decision, "state", "NEWS_SAFE")),
                                news_confidence=float(getattr(news_decision, "decision_confidence", 1.0)),
                                microstructure_alignment_score=float(candidate_microstructure_score.alignment_score),
                                microstructure_confidence=float(candidate_microstructure_score.confidence),
                                microstructure_composite_score=float(candidate_microstructure_score.composite_score),
                                lead_lag_alignment_score=float(candidate_lead_lag_snapshot.alignment_score),
                                lead_lag_confidence=float(candidate_lead_lag_snapshot.confidence),
                                lead_lag_disagreement_penalty=float(candidate_lead_lag_snapshot.disagreement_penalty),
                                event_playbook=str(candidate_event_directive.playbook),
                                event_base_class=str(candidate_event_directive.base_class),
                                event_pre_position_allowed=bool(candidate_event_directive.pre_position_allowed),
                                degraded_mode_active=bool(str(execution_quality.state).upper() == "DEGRADED"),
                                daily_caution_threshold_pct=float(
                                    risk_config.get(
                                        "daily_caution_threshold_pct",
                                        risk_config.get("soft_daily_dd_pct", 0.02),
                                    )
                                ),
                                daily_defensive_threshold_pct=float(
                                    risk_config.get(
                                        "daily_defensive_threshold_pct",
                                        risk_config.get("soft_daily_dd_pct", 0.035),
                                    )
                                ),
                                daily_hard_stop_threshold_pct=float(
                                    risk_config.get(
                                        "daily_hard_stop_threshold_pct",
                                        risk_config.get("hard_daily_dd_pct", 0.05),
                                    )
                                ),
                                daily_governor_timeout_hours=float(symbol_runtime_defaults.get("daily_governor_timeout_hours", risk_config.get("daily_governor_timeout_hours", 4.0))),
                                daily_governor_started_at=str(symbol_runtime_defaults.get("daily_governor_started_at", "")),
                                daily_governor_trigger_day_key=str(symbol_runtime_defaults.get("daily_governor_trigger_day_key", "")),
                                daily_governor_force_release=bool(symbol_runtime_defaults.get("daily_governor_force_release", False)),
                                daily_governor_emergency_review_ready=bool(symbol_runtime_defaults.get("daily_governor_emergency_review_ready", False)),
                                daily_normal_quality_floor=float(risk_scaling_config.get("trade_quality_floor", 0.58)),
                                daily_caution_quality_floor=float(risk_config.get("daily_caution_quality_floor", 0.70)),
                                daily_defensive_quality_floor=float(risk_config.get("daily_defensive_quality_floor", 0.85)),
                                daily_normal_risk_multiplier=float(risk_config.get("daily_normal_risk_multiplier", 1.0)),
                                daily_caution_risk_multiplier=float(risk_config.get("daily_caution_risk_multiplier", 0.75)),
                                daily_defensive_risk_multiplier=float(risk_config.get("daily_defensive_risk_multiplier", 0.45)),
                                daily_hard_stop_risk_multiplier=float(risk_config.get("daily_hard_stop_risk_multiplier", 0.0)),
                                strategy_family=str(strategy_key),
                                lane_name=str(trade_lane),
                                proof_bucket_state=str(proof_bucket_state),
                                proof_exception_allowed=bool(proof_exception_allowed),
                                proof_exception_reason=str(proof_exception_reason),
                                xau_grid_sub_budget_pct=float(xau_grid_risk_config.get("daily_sub_budget_pct", 0.35)),
                                xau_grid_cycle_quality=float(candidate.meta.get("grid_cycle_quality", candidate.meta.get("cycle_quality", 0.0)) or 0.0),
                                xau_grid_caution_cycle_quality=float(xau_grid_risk_config.get("caution_cycle_quality", 0.85)),
                                xau_grid_defensive_cycle_quality=float(xau_grid_risk_config.get("defensive_cycle_quality", 0.92)),
                                xau_grid_caution_risk_multiplier=float(xau_grid_risk_config.get("caution_risk_multiplier", 0.90)),
                                xau_grid_defensive_risk_multiplier=float(xau_grid_risk_config.get("defensive_risk_multiplier", 0.65)),
                                stretch_max_trades_per_day=int(current_scaling_state.get("stretch_daily_trade_target", current_scaling_state.get("current_overflow_daily_trade_cap", current_scaling_state.get("current_daily_trade_cap", 2)))),
                                hard_upper_limit=int(current_scaling_state.get("hard_upper_limit", current_scaling_state.get("current_overflow_daily_trade_cap", current_scaling_state.get("current_daily_trade_cap", 2)))),
                                stretch_max_trades_per_hour=int(current_scaling_state.get("hourly_stretch_target", current_scaling_state.get("hourly_base_target", risk_config["max_trades_per_hour"]))),
                                cluster_mode_active=bool(cluster_mode_active),
                                quality_cluster_score=float(quality_cluster_score),
                                lane_strength_multiplier=float(lane_strength_multiplier),
                                lane_score=float(lane_score),
                                lane_expectancy_multiplier=float(lane_expectancy_multiplier),
                                lane_expectancy_score=float(lane_expectancy_score),
                                session_density_score=float(session_density_score),
                                winning_streak_mode_active=bool(int(global_stats.winning_streak or 0) >= 2),
                                current_capacity_mode="STRETCH" if cluster_mode_active else "BASE",
                                session_priority_profile=str(session_priority.session_priority_profile),
                                lane_session_priority=str(session_priority.lane_session_priority),
                                session_native_pair=bool(session_priority.session_native_pair),
                                session_priority_multiplier=float(session_priority.session_priority_multiplier),
                                pair_priority_rank_in_session=int(session_priority.pair_priority_rank_in_session),
                                lane_budget_share=float(session_priority.lane_budget_share),
                                lane_available_capacity=0.0,
                                hot_hand_active=bool(hot_hand_active),
                                hot_hand_score=float(hot_hand_score),
                                session_bankroll_bias=float(session_bankroll_bias),
                                profit_recycle_active=bool(profit_recycle_active),
                                profit_recycle_boost=float(profit_recycle_boost),
                                close_winners_score=float(close_winners_score),
                                soft_trade_budget_enabled=bool(aggression_config.get("soft_trade_budgets_enabled", True)),
                                aggression_profile=str(
                                    aggression_config.get(
                                        "funded_profile",
                                        aggression_config.get("profile", "BOUNDED_AGGRO"),
                                    )
                                    if funded_mode_active
                                    else aggression_config.get("profile", "BOUNDED_AGGRO")
                                ),
                                aggression_lane_multiplier=float(
                                    aggression_config.get(
                                        "funded_lane_multiplier",
                                        aggression_config.get("lane_multiplier", 1.0),
                                    )
                                    if funded_mode_active
                                    else aggression_config.get("lane_multiplier", 1.0)
                                ),
                                hot_lane_concurrency_bonus=int(
                                    aggression_config.get(
                                        "hot_lane_concurrency_bonus",
                                        1 if hot_hand_active else 0,
                                    )
                                ),
                                exceptional_override_used=bool(session_override.exceptional_override_used),
                                exceptional_override_reason=str(session_override.exceptional_override_reason),
                                recovery_mode_active=bool(recovery_mode_active_runtime),
                                funded_account_mode=bool(funded_mode_active),
                                funded_phase=str(funded_config.get("phase", "evaluation")),
                                funded_daily_loss_limit_pct=float(funded_config.get("daily_loss_limit_pct", 0.05)),
                                funded_overall_drawdown_limit_pct=float(funded_config.get("overall_drawdown_limit_pct", 0.10)),
                                funded_profit_target_pct=float(funded_config.get("profit_target_pct", 0.08)),
                                funded_remaining_target_pct=float(
                                    funded_config.get("remaining_target_pct", funded_config.get("profit_target_pct", 0.08))
                                ),
                                funded_guard_buffer_pct=float(funded_config.get("guard_buffer_pct", 0.02)),
                            )
                        )
                        risk_diagnostics = risk_decision.diagnostics or {}
                        estimated_loss_usd = float(
                            risk_diagnostics.get(
                                "estimated_loss_usd",
                                _projected_trade_risk_usd(
                                    entry=entry_price,
                                    stop=stop_price,
                                    volume=max(risk_decision.volume, effective_fixed_lot),
                                    contract_size=float(symbol_info["trade_contract_size"]),
                                    tick_size=float(symbol_info.get("trade_tick_size")) if _is_positive_number(symbol_info.get("trade_tick_size")) else None,
                                    tick_value=float(symbol_tick_value) if symbol_tick_value is not None else None,
                                ),
                            )
                        )
                        runtime_effective_risk_cap = float(
                            risk_diagnostics.get(
                                "effective_risk_cap",
                                micro_config.get("micro_max_loss_usd", 2.5),
                            )
                        )
                        runtime_total_exposure_cap = float(
                            risk_diagnostics.get(
                                "total_exposure_cap",
                                micro_config.get("micro_total_risk_usd", 5.0),
                            )
                        )
                        base_risk_pct = float(risk_diagnostics.get("base_risk_pct", requested_risk_pct))
                        effective_risk_pct = float(risk_diagnostics.get("effective_risk_pct", risk_decision.risk_pct))
                        budget_usd = float(risk_diagnostics.get("budget_usd", float(account["equity"]) * effective_risk_pct))
                        current_day_start_equity = float(
                            risk_diagnostics.get("day_start_equity", symbol_runtime_defaults.get("day_start_equity", global_stats.day_start_equity))
                        )
                        current_day_high_equity = float(
                            risk_diagnostics.get("day_high_equity", symbol_runtime_defaults.get("day_high_equity", global_stats.day_high_equity))
                        )
                        current_daily_dd_pct_live = float(
                            risk_diagnostics.get("daily_dd_pct_live", symbol_runtime_defaults.get("daily_dd_pct_live", global_stats.daily_dd_pct_live))
                        )
                        current_daily_state_runtime = str(
                            risk_diagnostics.get("daily_state")
                            or symbol_runtime_defaults.get("daily_state")
                            or daily_state_name
                        )
                        current_daily_state_reason_runtime = str(
                            risk_diagnostics.get("daily_state_reason")
                            or symbol_runtime_defaults.get("daily_state_reason")
                            or daily_state_reason
                        )
                        current_daily_realized_pnl = float(
                            risk_diagnostics.get("daily_realized_pnl", symbol_runtime_defaults.get("daily_realized_pnl", global_stats.daily_realized_pnl))
                        )
                        current_trading_day_key_runtime = str(
                            risk_diagnostics.get("trading_day_key")
                            or symbol_runtime_defaults.get("trading_day_key")
                            or global_stats.trading_day_key
                        )
                        current_timezone_used_runtime = str(
                            risk_diagnostics.get("timezone_used")
                            or symbol_runtime_defaults.get("timezone_used")
                            or global_stats.timezone_used
                            or "Australia/Sydney"
                        )
                        current_closed_trades_today_runtime = int(
                            risk_diagnostics.get("closed_trades_today", symbol_runtime_defaults.get("closed_trades_today", global_stats.trades_today))
                        )
                        current_soft_dd_trade_count_runtime = int(
                            risk_diagnostics.get("soft_dd_trade_count", symbol_runtime_defaults.get("soft_dd_trade_count", global_stats.soft_dd_trade_count))
                        )
                        current_allowed_by_daily_governor_runtime = bool(risk_diagnostics.get("allowed_by_daily_governor", True))
                        current_risk_multiplier_applied_runtime = float(risk_diagnostics.get("risk_multiplier_applied", 1.0))
                        _update_runtime_metrics(
                            resolved_symbol,
                            now,
                            market_open_status=market_status,
                            requested_timeframe=str(timeframe_route.get("requested_timeframe", symbol_status.requested_timeframe)),
                            execution_timeframe_used=str(timeframe_route.get("execution_timeframe_used", symbol_status.execution_timeframe_used)),
                            internal_timeframes_used=list(timeframe_route.get("internal_timeframes_used", symbol_status.internal_timeframes_used)),
                            attachment_dependency_resolved=bool(timeframe_route.get("attachment_dependency_resolved", symbol_status.attachment_dependency_resolved)),
                            last_candidate_signal_id=str(candidate.signal_id),
                            last_candidate_score=float(preliminary_ai.probability),
                            last_candidate_probability=float(preliminary_ai.probability),
                            last_candidate_confluence=float(candidate.confluence_score),
                            last_candidate_expected_value_r=float(preliminary_ai.expected_value_r),
                            last_decision_type="RISK_GATE",
                            last_decision_reason=str(risk_decision.reason),
                            economics_source=str(symbol_info.get("economics_source", "mt5_client")),
                            entry_price_source=str(entry_price_source),
                            risk_mode=str(risk_diagnostics.get("risk_mode", "standard")),
                            bootstrap_enabled=bool(
                                micro_active
                                and bool(micro_config.get("bootstrap_enabled", True))
                                and float(account_equity) <= float(micro_config.get("bootstrap_equity_threshold", 160.0))
                            ),
                            regime_state=str(regime_state),
                            regime_confidence=float(regime_confidence),
                            volatility_forecast_state=str(volatility_state),
                            trade_quality_score=float(trade_quality.score),
                            trade_quality_band=str(trade_quality.band),
                            lane_name=str(trade_lane),
                            selected_trade_band=str(trade_quality.band),
                            current_band_target=str(current_band_target),
                            fallback_band_active=bool(fallback_band_active),
                            reason_higher_band_unavailable=str(reason_higher_band_unavailable),
                            best_A_plus_count=int(loop_band_counts.get("A+", 0)),
                            best_A_count=int(best_a_count),
                            best_B_plus_count=int(loop_band_counts.get("B+", 0)),
                            best_B_count=int(loop_band_counts.get("B", 0)),
                            best_C_count=int(loop_band_counts.get("C", 0)),
                                quality_size_multiplier=float(trade_quality.size_multiplier)
                                * float(strategy_state_size_multiplier)
                                * float(tier_size_multiplier),
                            trade_quality_components=dict(trade_quality.components),
                            execution_quality_state=str(execution_quality.state),
                            execution_quality_score=float(execution_quality.score),
                            spread_quality_score=float(trade_quality.components.get("spread_quality", 0.0)),
                            session_quality_score=float(session_quality_score),
                            liquidity_alignment_score=float(liquidity_decision.score),
                            nearest_liquidity_above=float(liquidity_decision.nearest_liquidity_above),
                            nearest_liquidity_below=float(liquidity_decision.nearest_liquidity_below),
                            liquidity_sweep_detected=bool(liquidity_decision.liquidity_sweep_detected),
                            pressure_proxy_score=float(regime.details.get("pressure_proxy_score", 0.0) or 0.0),
                            continuation_pressure=float(pressure_continuation),
                            exhaustion_signal=float(pressure_exhaustion),
                            absorption_signal=float(pressure_absorption),
                            correlation_adjustment=float(portfolio_decision.size_multiplier),
                            exposure_cluster_detected=str(portfolio_decision.exposure_cluster_detected or ""),
                            portfolio_bias=str(portfolio_decision.portfolio_bias or ""),
                            scaling_mode=str(current_scaling_state.get("scaling_mode", "quick_smart_scaler")),
                            current_compounding_state=str(current_scaling_state.get("current_compounding_state", "base_flow")),
                            current_growth_bias=str(current_scaling_state.get("current_growth_bias", "balanced_compounding")),
                            base_daily_trade_target=int(current_scaling_state.get("base_daily_trade_target", current_scaling_state.get("current_daily_trade_cap", 0))),
                            stretch_daily_trade_target=int(current_scaling_state.get("stretch_daily_trade_target", current_scaling_state.get("current_overflow_daily_trade_cap", 0))),
                            hard_upper_limit=int(current_scaling_state.get("hard_upper_limit", current_scaling_state.get("current_overflow_daily_trade_cap", 0))),
                            hourly_base_target=int(current_scaling_state.get("hourly_base_target", risk_config.get("max_trades_per_hour", 0))),
                            hourly_stretch_target=int(current_scaling_state.get("hourly_stretch_target", risk_config.get("max_trades_per_hour", 0))),
                            next_equity_milestone=float(current_scaling_state.get("next_equity_milestone", 0.0)),
                            projected_trade_capacity_today=int(current_scaling_state.get("projected_trade_capacity_today", current_scaling_state.get("current_daily_trade_cap", 0))),
                            stretch_mode_active=bool(risk_diagnostics.get("stretch_mode_active", cluster_mode_active)),
                            cluster_mode_active=bool(risk_diagnostics.get("cluster_mode_active", cluster_mode_active)),
                            current_capacity_mode=str(risk_diagnostics.get("current_capacity_mode", "STRETCH" if cluster_mode_active else "BASE")),
                            quality_cluster_score=float(risk_diagnostics.get("quality_cluster_score", quality_cluster_score)),
                            lane_strength_multiplier=float(risk_diagnostics.get("lane_strength_multiplier", lane_strength_multiplier)),
                            lane_score=float(risk_diagnostics.get("lane_score", lane_score)),
                            session_density_score=float(risk_diagnostics.get("session_density_score", session_density_score)),
                            effective_risk_model=str(risk_diagnostics.get("effective_risk_model", "")),
                            modeled_loss_account_ccy=float(estimated_loss_usd),
                            estimated_loss_usd=float(estimated_loss_usd),
                            min_lot_loss_usd=float(risk_diagnostics.get("min_lot_loss_usd", 0.0)),
                            required_margin=float(risk_diagnostics.get("required_margin", 0.0)),
                            margin_free=float(account.get("margin_free", 0.0) or 0.0),
                            effective_risk_cap_usd=float(runtime_effective_risk_cap),
                            risk_cap_usd=float(runtime_effective_risk_cap),
                            total_exposure_cap=float(runtime_total_exposure_cap),
                            computed_lot=float(risk_decision.volume),
                            min_broker_lot=float(symbol_info.get("volume_min", 0.01)),
                            chosen_stop_distance=float(provisional_stop_distance),
                            stop_geometry_source=str(stop_geometry_source),
                            last_precheck_reason=str(pre_risk_stop_geometry.get("validation_reason", "")),
                            soft_dd_active=bool(risk_diagnostics.get("soft_dd_active", symbol_runtime_defaults.get("soft_dd_active", False))),
                            hard_dd_active=bool(risk_diagnostics.get("hard_dd_active", symbol_runtime_defaults.get("hard_dd_active", False))),
                            day_start_equity=float(current_day_start_equity),
                            day_high_equity=float(current_day_high_equity),
                            daily_dd_pct_live=float(current_daily_dd_pct_live),
                            soft_dd_elite_mode_active=bool(risk_diagnostics.get("soft_dd_elite_mode_active", False)),
                            soft_dd_trade_count=int(current_soft_dd_trade_count_runtime),
                            last_soft_dd_trade_reason=str(risk_diagnostics.get("last_soft_dd_trade_reason", "")),
                            adaptive_risk_state=str(risk_diagnostics.get("adaptive_risk_state", "")),
                            daily_state=str(current_daily_state_runtime),
                            daily_state_reason=str(current_daily_state_reason_runtime),
                            daily_realized_pnl=float(current_daily_realized_pnl),
                            trading_day_key=str(current_trading_day_key_runtime),
                            timezone_used=str(current_timezone_used_runtime),
                            closed_trades_today=int(current_closed_trades_today_runtime),
                            allowed_by_daily_governor=bool(current_allowed_by_daily_governor_runtime),
                            risk_multiplier_applied=float(current_risk_multiplier_applied_runtime),
                            proof_exception_used=bool(risk_diagnostics.get("proof_exception_used", False)),
                            proof_bucket_state=str(proof_bucket_state),
                            proof_exception_reason=str(proof_exception_reason),
                            session_priority_profile=str(session_priority.session_priority_profile),
                            lane_session_priority=str(session_priority.lane_session_priority),
                            session_native_pair=bool(session_priority.session_native_pair),
                            session_priority_multiplier=float(session_priority.session_priority_multiplier),
                            pair_priority_rank_in_session=int(session_priority.pair_priority_rank_in_session),
                            lane_budget_share=float(session_priority.lane_budget_share),
                            lane_available_capacity=float(risk_diagnostics.get("lane_available_capacity", 0.0)),
                            lane_capacity_usage=float(risk_diagnostics.get("session_trade_count", 0) or 0.0)
                            / max(float(risk_diagnostics.get("lane_available_capacity", 0.0) or 0.0), 1.0),
                            session_adjusted_score=float(session_adjusted),
                            exceptional_override_used=bool(session_override.exceptional_override_used),
                            exceptional_override_reason=str(session_override.exceptional_override_reason),
                            why_non_native_pair_won=str(session_override.why_non_native_pair_won),
                            why_native_pair_lost_priority=str(session_override.why_native_pair_lost_priority),
                            pair_status=str(pair_state["pair_status"]),
                            pair_status_reason=str(pair_state["pair_status_reason"]),
                            pair_state_multiplier=float(pair_state["pair_state_multiplier"]),
                            rolling_expectancy_by_pair=float(pair_state["rolling_expectancy_by_pair"]),
                            rolling_pf_by_pair=float(pair_state["rolling_pf_by_pair"]),
                            rolling_expectancy_by_session=float(pair_state["rolling_expectancy_by_session"]),
                            rolling_pf_by_session=float(pair_state["rolling_pf_by_session"]),
                            rolling_win_rate_by_pair=float(pair_state["rolling_win_rate_by_pair"]),
                            rolling_win_rate_by_session=float(pair_state["rolling_win_rate_by_session"]),
                            false_break_rate=float(pair_state["false_break_rate"]),
                            management_quality_score=float(pair_state["management_quality_score"]),
                            today_session_wins=int(pair_state["today_session_wins"]),
                            today_session_losses=int(pair_state["today_session_losses"]),
                            today_session_trades=int(pair_state["today_session_trades"]),
                            why_pair_is_promoted=str(pair_state["why_pair_is_promoted"]),
                            why_pair_is_throttled=str(pair_state["why_pair_is_throttled"]),
                            current_band_attempted=str(current_band_attempted),
                            what_would_make_it_pass=str(session_override.why_native_pair_lost_priority)
                            if (not session_priority.session_native_pair and bool(session_native_leader))
                            else "",
                            primary_block_reason="",
                            secondary_block_reason=str(risk_diagnostics.get("last_precheck_reason", "")),
                            xau_grid_override_allowed=bool(risk_diagnostics.get("xau_grid_override_allowed", False)),
                            xau_grid_override_reason=str(risk_diagnostics.get("xau_grid_override_reason", "")),
                            xau_grid_sub_budget=float(risk_diagnostics.get("xau_grid_sub_budget", 0.0)),
                            xau_grid_risk_multiplier=float(risk_diagnostics.get("xau_grid_risk_multiplier", 0.0)),
                            base_risk_pct=float(risk_diagnostics.get("base_risk_pct", base_risk_pct)),
                            adjusted_risk_pct=float(risk_diagnostics.get("adjusted_risk_pct", effective_risk_pct)),
                            risk_modifiers=dict(risk_diagnostics.get("risk_modifiers", {})),
                            total_open_risk_pct=float(risk_diagnostics.get("total_open_risk_pct", float(projected_open_risk_usd) / max(float(account["equity"]), 1e-9))),
                            overflow_band_active=bool(risk_diagnostics.get("overflow_band_active", False)),
                            current_phase=str(current_scaling_state.get("current_phase", "PHASE_1")),
                            current_daily_trade_cap=int(current_scaling_state.get("current_daily_trade_cap", 2)),
                            current_overflow_daily_trade_cap=int(current_scaling_state.get("current_overflow_daily_trade_cap", current_scaling_state.get("current_daily_trade_cap", 2))),
                            current_phase_base_risk_pct=float(current_scaling_state.get("current_risk_pct", phase_risk_pct)),
                            current_phase_max_risk_pct=float(current_scaling_state.get("current_max_risk_pct", effective_hard_risk_cap)),
                            session_stop_state=str(risk_diagnostics.get("session_stop_state", "")),
                            session_stop_reason=str(risk_diagnostics.get("session_stop_reason", "")),
                            session_entries_blocked=bool(risk_diagnostics.get("session_entries_blocked", False)),
                            session_realized_pnl=float(risk_diagnostics.get("session_realized_pnl", 0.0)),
                            session_realized_pnl_pct=float(risk_diagnostics.get("session_realized_pnl_pct", 0.0)),
                            session_trade_count=int(risk_diagnostics.get("session_trade_count", 0)),
                        )
                        risk_signature = "|".join(
                            [
                                str(candidate.setup),
                                str(resolved_symbol),
                                str(risk_decision.approved),
                                str(risk_decision.reason),
                                f"{risk_decision.volume:.4f}",
                                f"{effective_risk_pct:.6f}",
                            ]
                        )
                        risk_signature_key = f"{resolved_symbol.upper()}::{candidate.setup}"
                        should_log_risk = True
                        previous_risk_log = last_risk_log_signature.get(risk_signature_key)
                        if previous_risk_log is not None:
                            previous_signature, previous_time = previous_risk_log
                            if previous_signature == risk_signature and (now - previous_time).total_seconds() < max(1, risk_log_dedupe_seconds):
                                should_log_risk = False
                        if should_log_risk:
                            logger.info(
                                "risk_gate_decision",
                                extra={
                                    "extra_fields": {
                                        "setup_id": candidate.setup,
                                        "symbol": resolved_symbol,
                                        "approved": risk_decision.approved,
                                        "reason": risk_decision.reason,
                                        "lot": risk_decision.volume,
                                        "sl_distance_price": abs(entry_price - stop_price),
                                        "estimated_loss_usd": estimated_loss_usd,
                                        "equity": float(account["equity"]),
                                        "base_risk_pct": base_risk_pct,
                                        "effective_risk_pct": effective_risk_pct,
                                        "trade_quality_score": float(trade_quality.score),
                                        "trade_quality_band": str(trade_quality.band),
                                        "selected_trade_band": str(trade_quality.band),
                                        "current_band_target": str(current_band_target),
                                        "fallback_band_active": bool(fallback_band_active),
                                        "reason_higher_band_unavailable": str(reason_higher_band_unavailable),
                                        "daily_state": str(current_daily_state_runtime),
                                        "daily_state_reason": str(current_daily_state_reason_runtime),
                                        "daily_realized_pnl": float(current_daily_realized_pnl),
                                        "daily_pnl_pct": float(risk_diagnostics.get("daily_pnl_pct", global_stats.daily_pnl_pct)),
                                        "daily_dd_pct_live": float(current_daily_dd_pct_live),
                                        "allowed_by_daily_governor": bool(current_allowed_by_daily_governor_runtime),
                                        "risk_multiplier_applied": float(current_risk_multiplier_applied_runtime),
                                        "proof_exception_used": bool(risk_diagnostics.get("proof_exception_used", False)),
                                        "xau_grid_override_used": bool(risk_diagnostics.get("xau_grid_override_allowed", False)),
                                        "session_priority_profile": str(session_priority.session_priority_profile),
                                        "lane_session_priority": str(session_priority.lane_session_priority),
                                        "session_native_pair": bool(session_priority.session_native_pair),
                                        "session_priority_multiplier": float(session_priority.session_priority_multiplier),
                                        "pair_priority_rank_in_session": int(session_priority.pair_priority_rank_in_session),
                                        "lane_budget_share": float(session_priority.lane_budget_share),
                                        "lane_available_capacity": float(risk_diagnostics.get("lane_available_capacity", 0.0)),
                                        "exceptional_override_used": bool(session_override.exceptional_override_used),
                                        "exceptional_override_reason": str(session_override.exceptional_override_reason),
                                        "strategy_family": str(strategy_key),
                                        "lane_name": str(trade_lane),
                                        "regime_state": str(regime_state),
                                        "regime_confidence": float(regime_confidence),
                                        "execution_quality_state": str(execution_quality.state),
                                        "budget_usd": budget_usd,
                                        "risk_boost_active": bool(risk_diagnostics.get("risk_boost_active", False)),
                                        "economics_source": str(symbol_info.get("economics_source", "mt5_client")),
                                        "risk_mode": str(risk_diagnostics.get("risk_mode", "standard")),
                                        "effective_risk_model": str(risk_diagnostics.get("effective_risk_model", "")),
                                        "stop_geometry_source": str(stop_geometry_source),
                                    }
                                },
                            )
                            last_risk_log_signature[risk_signature_key] = (risk_signature, now)

                        if risk_decision.kill:
                            if _is_account_wide_soft_kill(risk_decision.reason) or str(risk_decision.kill).upper() == "HARD":
                                kill_switch.activate(
                                    risk_decision.kill,
                                    risk_decision.reason,
                                    now=now,
                                    session_key=str(global_stats.trading_day_key),
                                    last_equity=float(account_equity),
                                    hard_ttl_hours=hard_kill_ttl_hours,
                                    recovery_mode="RECOVERY_DEFENSIVE",
                                    recovery_wins_needed=3,
                                )
                                monitor.alert(f"{risk_decision.kill}_KILL", "Kill switch activated", reason=risk_decision.reason, symbol=resolved_symbol)
                            mark_block(resolved_symbol, risk_decision.reason, symbol_status)
                            if _is_account_wide_soft_kill(risk_decision.reason) or str(risk_decision.kill).upper() == "HARD":
                                break
                            continue
                        if not risk_decision.approved:
                            if str(risk_decision.reason).startswith("daily_"):
                                _update_runtime_metrics(
                                    resolved_symbol,
                                    now,
                                    last_soft_dd_rejection_reason=str(
                                        (risk_decision.diagnostics or {}).get("last_soft_dd_rejection_reason", risk_decision.reason)
                                    ),
                                )
                            if risk_decision.reason == "first_trade_sl_too_wide":
                                details = risk_decision.diagnostics or {}
                                logger.info(
                                    "first_trade_sl_too_wide",
                                    extra={
                                        "extra_fields": {
                                            "symbol": str(details.get("symbol", resolved_symbol)),
                                            "sl_distance_price": details.get("sl_distance_price"),
                                            "sl_distance_points": details.get("sl_distance_points"),
                                            "max_allowed_points": details.get("max_allowed_points"),
                                            "point_size": details.get("point_size"),
                                            "point_source": details.get("point_source"),
                                            "digits": details.get("digits"),
                                            "tick_size": details.get("tick_size"),
                                        }
                                    },
                                )
                            mark_block(resolved_symbol, risk_decision.reason, symbol_status)
                            continue

                        if risk_decision.extra_confluence_required:
                            rule_confluence_required += 1.0
                        if (
                            session_context.session_name in {"LONDON", "OVERLAP"}
                            and bool(regime.details.get("trend_flag", regime.label == "TRENDING"))
                            and spread_points <= (float(risk_config["max_spread_points"]) * 0.75)
                        ):
                            session_offset -= 0.01
                        entry_block_state = runtime_entry_block_state(now) if bridge_trade_mode else {
                            "blocked": False,
                            "reason": "",
                            "entries_blocked_until": "",
                        }
                        if bool(entry_block_state.get("blocked")):
                            block_reason = str(entry_block_state.get("reason") or "startup_warmup")
                            _update_runtime_metrics(
                                resolved_symbol,
                                now,
                                last_decision_type="ENTRY_BLOCK",
                                last_decision_reason=block_reason,
                                startup_warmup_active=bool(block_reason == "startup_warmup"),
                                startup_entries_blocked_until=str(entry_block_state.get("entries_blocked_until") or ""),
                            )
                            log_stage(
                                resolved_symbol,
                                "ENTRY_BLOCKED",
                                block_reason,
                                now_ts=now,
                            )
                            mark_block(resolved_symbol, block_reason, symbol_status)
                            continue

                        risk_summary = {
                            "approved": risk_decision.approved,
                            "reason": risk_decision.reason,
                            "consecutive_losses": symbol_stats.consecutive_losses,
                            "rolling_drawdown_pct": global_stats.rolling_drawdown_pct,
                            "spread_points": spread_points,
                            "max_spread_points": float(risk_config["max_spread_points"]),
                            "max_slippage_points": int(risk_config["max_slippage_points"]),
                            "portfolio_size_multiplier": portfolio_decision.size_multiplier,
                            "requires_ai_override": portfolio_decision.requires_ai_override,
                            "adaptive_samples": adaptive_feedback["samples"],
                            "adaptive_win_rate": adaptive_feedback["weighted_win_rate"],
                            "adaptive_avg_r": adaptive_feedback["weighted_avg_r"],
                            "adaptive_recent_loss_streak": adaptive_feedback["recent_loss_streak"],
                            "session_name": session_context.session_name,
                            "session_probability_offset": session_offset,
                            "rule_confluence_score": float(candidate.confluence_score),
                            "rule_confluence_required": rule_confluence_required,
                            "trade_quality_score": float(trade_quality.score),
                            "trade_quality_band": str(trade_quality.band),
                            "regime_state": str(regime_state),
                            "regime_confidence": float(regime_confidence),
                            "execution_quality_state": str(execution_quality.state),
                            "execution_quality_score": float(execution_quality.score),
                            "liquidity_alignment_score": float(liquidity_decision.score),
                            "strong_trend_regime": bool(regime.details.get("trend_flag", regime.label == "TRENDING")),
                            "extra_confluence_required": bool(risk_decision.extra_confluence_required),
                            "micro_mode": micro_active,
                            "tokyo_sydney_spread_guard": tokyo_sydney_spread_guard,
                            "is_grid_candidate": is_grid_candidate,
                            "adaptive_min_probability_floor": float(ai_config.get("adaptive_min_probability_floor", 0.50)),
                            "economics_source": str(symbol_info.get("economics_source", "mt5_client")),
                            "risk_mode": str(risk_diagnostics.get("risk_mode", "standard")),
                            "effective_risk_model": str(risk_diagnostics.get("effective_risk_model", "")),
                            "microstructure_ready": bool(candidate_microstructure_score.ready),
                            "microstructure_direction": str(candidate_microstructure_score.direction),
                            "microstructure_confidence": float(candidate_microstructure_score.confidence),
                            "microstructure_composite_score": float(candidate_microstructure_score.composite_score),
                            "microstructure_alignment_score": float(candidate_microstructure_score.alignment_score),
                            "microstructure_pressure_score": float(candidate_microstructure_score.pressure_score),
                            "microstructure_cumulative_delta_score": float(candidate_microstructure_score.cumulative_delta_score),
                            "microstructure_depth_imbalance": float(candidate_microstructure_score.depth_imbalance),
                            "microstructure_dom_imbalance": float(candidate_microstructure_score.dom_imbalance),
                            "microstructure_drift_score": float(candidate_microstructure_score.drift_score),
                            "microstructure_spread_stability": float(candidate_microstructure_score.spread_stability),
                            "microstructure_sweep_velocity": float(candidate_microstructure_score.sweep_velocity),
                            "microstructure_absorption_score": float(candidate_microstructure_score.absorption_score),
                            "microstructure_iceberg_score": float(candidate_microstructure_score.iceberg_score),
                            "lead_lag_direction": str(candidate_lead_lag_snapshot.direction),
                            "lead_lag_confidence": float(candidate_lead_lag_snapshot.confidence),
                            "lead_lag_alignment_score": float(candidate_lead_lag_snapshot.alignment_score),
                            "lead_lag_disagreement_penalty": float(candidate_lead_lag_snapshot.disagreement_penalty),
                            "event_playbook": str(candidate_event_directive.playbook),
                            "event_base_class": str(candidate_event_directive.base_class),
                            "event_pre_position_allowed": bool(candidate_event_directive.pre_position_allowed),
                            "execution_minute_profile": dict(candidate_execution_profile),
                            "execution_minute_quality_score": float(candidate_execution_profile.get("quality_score", 0.70) or 0.70),
                            "execution_minute_size_multiplier": float(candidate_execution_profile.get("size_multiplier", 1.0) or 1.0),
                        }
                        news_override = is_grid_candidate and bool(candidate.meta.get("news_override", False))
                        news_safe_for_gate = bool(news_decision.safe) or news_override
                        news_summary = {
                            "safe": news_safe_for_gate,
                            "reason": ("grid_news_override" if news_override else news_decision.reason),
                            "state": "NEWS_SAFE" if news_override else str(getattr(news_decision, "state", "NEWS_SAFE")),
                            "source": news_decision.source,
                            "source_used": str(getattr(news_decision, "source_used", news_decision.source)),
                            "fallback_used": bool(getattr(news_decision, "fallback_used", False)),
                            "decision_confidence": float(getattr(news_decision, "decision_confidence", 1.0)),
                            "next_safe_time": news_decision.next_safe_time.isoformat() if news_decision.next_safe_time else None,
                            "bias_direction": news_decision.bias_direction,
                            "bias_confidence": news_decision.bias_confidence,
                            "bias_reason": news_decision.bias_reason,
                            "source_confidence": news_decision.source_confidence,
                            "authenticity_risk": news_decision.authenticity_risk,
                            "sentiment_extreme": news_decision.sentiment_extreme,
                            "crowding_bias": news_decision.crowding_bias,
                            "trade_quality_score": float(trade_quality.score),
                            "event_playbook": str(candidate_event_directive.playbook),
                            "event_base_class": str(candidate_event_directive.base_class),
                            "event_pre_position_allowed": bool(candidate_event_directive.pre_position_allowed),
                            "caution_probability_floor": 0.68 if str(getattr(news_decision, "state", "NEWS_SAFE")).upper() == "NEWS_CAUTION" else 0.0,
                            "caution_confluence_floor": 0.70 if str(getattr(news_decision, "state", "NEWS_SAFE")).upper() == "NEWS_CAUTION" else 0.0,
                        }
                        log_stage(
                            resolved_symbol,
                            "AI_REVIEW",
                            f"{candidate.setup} p={float(preliminary_ai.probability):.2f}",
                            now_ts=now,
                        )
                        symbol_status.current_state = "AI_REVIEW"
                        _record_symbol_activity(resolved_symbol, "ai_reviews", now)
                        ai_started_at = time.perf_counter()
                        approved, final_decision = ai_gate.approve_order(
                            candidate=candidate,
                            features=row,
                            regime=regime.label,
                            risk_summary=risk_summary,
                            news_summary=news_summary,
                            open_positions=open_positions_journal,
                            account_state=account,
                            precomputed_gate=preliminary_ai,
                        )
                        ai_latency_ms = max(0.0, (time.perf_counter() - ai_started_at) * 1000.0)
                        last_ai_called_at = datetime.now(tz=UTC).isoformat()
                        _update_runtime_metrics(
                            resolved_symbol,
                            now,
                            market_open_status=market_status,
                            delivered_actions_last_15m=int(bridge_delivery_counts.get(_normalize_symbol_key(resolved_symbol), symbol_status.delivered_actions)),
                            last_ai_called_at=last_ai_called_at,
                            last_ai_mode=str(final_decision.get("ai_mode", "local_fallback")),
                            last_ai_latency_ms=round(ai_latency_ms, 2),
                        )
                        symbol_status.last_score = float(final_decision.get("probability", symbol_status.last_score))
                        symbol_status.ai_reason = str(final_decision.get("rationale", ""))
                        symbol_status.last_ai_mode = str(final_decision.get("ai_mode", "local_fallback"))
                        logger.info(
                            "ai_entry_call",
                            extra={
                                "extra_fields": {
                                    "symbol": resolved_symbol,
                                    "setup_id": candidate.setup,
                                    "signal_id": candidate.signal_id,
                                    "ai_mode": str(final_decision.get("ai_mode", "local_fallback")),
                                    "latency_ms": round(ai_latency_ms, 2),
                                    "approved": bool(approved),
                                    "probability": float(final_decision.get("probability", preliminary_ai.probability)),
                                }
                            },
                        )
                        journal.log_event(
                            candidate.signal_id,
                            "AI_DECISION",
                            {
                                "approved": bool(approved),
                                "ai_mode": str(final_decision.get("ai_mode", "local_fallback")),
                                "rationale": str(final_decision.get("rationale", "")),
                            },
                        )
                        logger.info(
                            "ai_final_decision",
                            extra={
                                "extra_fields": {
                                    "signal_id": candidate.signal_id,
                                    "symbol": resolved_symbol,
                                    "approved": bool(approved),
                                    "ai_mode": str(final_decision.get("ai_mode", "local_fallback")),
                                    "reason": str(final_decision.get("rationale", "")),
                                }
                            },
                        )

                        allow_approve_small = bool(candidate.meta.get("allow_ai_approve_small", _is_xau_grid_setup(candidate.setup)))
                        approve_small_probability = float(
                            candidate.meta.get(
                                "approve_small_min_probability",
                                0.50 if _is_xau_grid_setup(candidate.setup) else (0.48 if str(candidate.setup).upper().startswith("XAUUSD_M1_") else 0.0),
                            )
                        )
                        approve_small_confluence = float(candidate.meta.get("approve_small_min_confluence", 3.0))
                        ai_probability = float(final_decision.get("probability", preliminary_ai.probability))
                        ai_rationale = str(final_decision.get("rationale", ""))
                        approve_small_session_allowed = _approve_small_session_allowed(
                            resolved_symbol,
                            session_context.session_name,
                            now,
                        )
                        risk_mode_name = str(risk_summary.get("risk_mode", "")).lower()
                        if risk_mode_name.startswith("bootstrap"):
                            approve_small_probability = max(0.48, approve_small_probability - 0.02)
                            approve_small_confluence = max(2.8, approve_small_confluence - 0.10)
                        if bool(candidate.meta.get("btc_weekend_force_emit", False)):
                            approve_small_probability = min(approve_small_probability, 0.34)
                            approve_small_confluence = min(approve_small_confluence, 2.35)
                        if (
                            (not approved)
                            and allow_approve_small
                            and approve_small_session_allowed
                            and ai_probability >= approve_small_probability
                            and float(candidate.confluence_score) >= approve_small_confluence
                            and any(token in ai_rationale.lower() for token in ("reject", "pass", "neutral", "conservative", "mixed"))
                        ):
                            approved = True
                            final_decision["ai_verdict_class"] = "APPROVE_SMALL"
                            final_decision["size_multiplier"] = clamp(
                                float(final_decision.get("size_multiplier", 0.35) or 0.35),
                                0.15,
                                0.45,
                            )
                            final_decision["recommended_sl_atr_mult"] = max(
                                0.8,
                                float(final_decision.get("recommended_sl_atr_mult", candidate.stop_atr)),
                            )
                            final_decision["recommended_tp_r"] = max(
                                1.1,
                                float(final_decision.get("recommended_tp_r", candidate.tp_r)),
                            )
                            final_decision["rationale"] = f"approve_small_override:{ai_rationale or 'neutral'}"
                            symbol_status.ai_reason = str(final_decision["rationale"])
                            logger.info(
                                "ai_approve_small_override",
                                extra={
                                    "extra_fields": {
                                        "symbol": resolved_symbol,
                                        "setup_id": candidate.setup,
                                        "signal_id": candidate.signal_id,
                                        "probability": ai_probability,
                                        "confluence_score": float(candidate.confluence_score),
                                    }
                                },
                            )

                        online_score = online_learning.predict_score(
                            {
                                "symbol": resolved_symbol,
                                "side": candidate.side,
                                "entry": entry_price,
                                "sl": stop_price,
                                "tp": tp_price,
                                "lot": risk_decision.volume,
                                "ai_probability": float(final_decision.get("probability", preliminary_ai.probability)),
                                "spread_points": _normalize_runtime_spread_points(
                                    resolved_symbol,
                                    float(row["m5_spread"]),
                                    symbol_info=symbol_info,
                                    max_spread_points=float(risk_config["max_spread_points"]),
                                ),
                                "news_state": news_decision.reason,
                                "session_state": session_status,
                                "session_name": session_context.session_name,
                                "regime": regime.label,
                                "regime_state": str(regime_state),
                                "setup": candidate.setup,
                                "strategy_key": str(strategy_key),
                                "lane_name": str(trade_lane),
                                "management_template": str(candidate.meta.get("management_template") or ""),
                                "strategy_state": str(strategy_state),
                                "regime_fit": float(strategy_regime_fit),
                                "session_fit": float(strategy_session_fit),
                                "volatility_fit": float(strategy_volatility_fit),
                                "pair_behavior_fit": float(strategy_pair_behavior_fit),
                                "execution_quality_fit": float(strategy_execution_fit),
                                "entry_timing_score": float(entry_timing_score_value),
                                "structure_cleanliness_score": float(structure_cleanliness_score_value),
                                "strategy_recent_performance": float(strategy_recent_performance),
                                "market_data_consensus_state": str(
                                    symbol_status.runtime_market_data_consensus_state
                                    or runtime_market_data_consensus_state
                                ),
                                "mc_win_rate": float(candidate.meta.get("mc_win_rate", 0.0) or 0.0),
                                "multi_tf_alignment_score": float(candidate.meta.get("multi_tf_alignment_score", 0.5) or 0.5),
                                "fractal_persistence_score": float(candidate.meta.get("fractal_persistence_score", 0.5) or 0.5),
                                "compression_expansion_score": float(candidate.meta.get("compression_expansion_score", 0.0) or 0.0),
                                "dxy_support_score": float(candidate.meta.get("dxy_support_score", 0.5) or 0.5),
                                "aggressive_pair_mode": 1.0 if _is_super_aggressive_normal_symbol(resolved_symbol) else 0.0,
                                "trajectory_catchup_pressure": float(candidate.meta.get("trajectory_catchup_pressure", 0.0) or 0.0),
                            }
                        )
                        final_decision["online_score"] = online_score
                        if online_score < float(ai_config.get("online_min_probability", 0.40)):
                            approved = False
                            final_decision["rationale"] = f"online_model_reject:{online_score:.3f}"
                            symbol_status.ai_reason = str(final_decision["rationale"])
                        if not approved:
                            if resolved_symbol == "BTCUSD":
                                logger.info(
                                    "BTC_AI_VETO",
                                    extra={
                                        "extra_fields": {
                                            "symbol": resolved_symbol,
                                            "setup": candidate.setup,
                                            "signal_id": candidate.signal_id,
                                            "reason": str(final_decision.get("rationale", "final_gate_rejected")),
                                            "probability": float(final_decision.get("probability", preliminary_ai.probability)),
                                        }
                                    },
                                )
                            idea_lifecycle.reject(
                                idea=idea,
                                reason=str(final_decision.get("rationale", "final_gate_rejected")),
                                now=now,
                                session_name=session_context.session_name,
                                structure=idea_structure,
                            )
                            log_stage(
                                resolved_symbol,
                                "AI_REJECTED",
                                str(final_decision.get("rationale", "final_gate_rejected")),
                                suppress_after=reject_display_limit,
                                now_ts=now,
                            )
                            mark_block(resolved_symbol, str(final_decision.get("rationale", "final_gate_rejected")), symbol_status)
                            continue

                        log_stage(
                            resolved_symbol,
                            "AI_APPROVED",
                            f"{candidate.setup} p={float(final_decision.get('probability', preliminary_ai.probability)):.2f}",
                            now_ts=now,
                        )
                        symbol_status.current_state = "AI_APPROVED"
                        summary["approved"] += 1
                        if not trading_enabled:
                            symbol_status.trading_allowed = False
                            symbol_status.reason = trading_reason
                            symbol_status.current_state = "BLOCK"
                            break
                        if dry_run:
                            symbol_status.trading_allowed = False
                            symbol_status.reason = "dry_run_preview_only"
                            symbol_status.current_state = "PREVIEW"
                            break

                        final_stop_distance, final_stop_source = _resolve_candidate_stop_distance(
                            candidate=candidate,
                            atr_for_candidate=atr_for_candidate,
                            point_size=float(symbol_info.get("point", 0.0) or 0.0),
                            sl_multiplier=float(final_decision["recommended_sl_atr_mult"]),
                        )
                        if final_stop_distance <= 0:
                            mark_block(resolved_symbol, "final_stop_distance_invalid", symbol_status)
                            continue
                        streak_adjust = _streak_adjustment_mode(
                            closed_trades=recent_closed_trades,
                            symbol_key=resolved_symbol,
                            strategy_key=str(strategy_key),
                            orchestrator_config=bridge_orchestrator_config,
                        )
                        exit_profile = _strategy_exit_profile(
                            symbol_key=resolved_symbol,
                            strategy_key=str(strategy_key),
                            quality_tier=quality_tier,
                            exits_config=settings.section("exits"),
                            streak_adjust_mode=str(streak_adjust.get("mode", "NEUTRAL")),
                            session_name=str(session_context.session_name),
                            regime_state=str(regime_state),
                            weekend_mode=bool(is_weekend_market_mode(now)),
                        )
                        rr_target_band = str(exit_profile.get("approved_rr_target", "2.0-2.3") or "2.0-2.3")
                        rr_min_str, _, rr_max_str = rr_target_band.partition("-")
                        rr_min = max(0.5, float(rr_min_str or 2.0))
                        rr_max = max(rr_min, float(rr_max_str or rr_min))
                        approved_tp_r = clamp(float(final_decision["recommended_tp_r"]), rr_min, rr_max)
                        if candidate.side == "BUY":
                            final_stop = entry_price - final_stop_distance
                            final_tp = entry_price + (final_stop_distance * float(approved_tp_r))
                        else:
                            final_stop = entry_price + final_stop_distance
                            final_tp = entry_price - (final_stop_distance * float(approved_tp_r))
                        final_stop_geometry = _normalize_pre_risk_exit_geometry(
                            symbol=resolved_symbol,
                            side=candidate.side,
                            entry_price=entry_price,
                            stop_price=final_stop,
                            tp_price=final_tp,
                            spread_points=spread_points,
                            symbol_info=symbol_info,
                            symbol_rules=symbol_rules,
                            safety_buffer_points=bridge_safety_buffer_points,
                        )
                        final_stop = float(final_stop_geometry["stop_price"])
                        final_tp = float(final_stop_geometry["tp_price"])
                        final_stop_distance = max(float(final_stop_geometry["stop_distance"]), 0.0)
                        if final_stop_distance <= 0:
                            mark_block(resolved_symbol, "final_stop_geometry_invalid", symbol_status)
                            continue

                        volume = risk_decision.volume
                        volume *= float(tier_size_multiplier)
                        volume *= float(final_decision.get("size_multiplier", 1.0))
                        volume *= provisional_stop_distance / final_stop_distance
                        if news_override:
                            volume *= max(0.1, min(1.0, float(grid_scalper.news_override_size_multiplier)))
                        normalized_volume = mt5_client.normalize_volume(resolved_symbol, volume)
                        configured_max_lot = float(risk_config.get("max_lot_per_trade", normalized_volume))
                        broker_min_lot = float(symbol_info.get("volume_min", 0.01) or 0.01)
                        effective_risk_cap = float(runtime_effective_risk_cap)
                        total_exposure_cap = float(runtime_total_exposure_cap)
                        bootstrap_active = bool(
                            micro_active
                            and bool(micro_config.get("bootstrap_enabled", True))
                            and float(account_equity) <= float(micro_config.get("bootstrap_equity_threshold", 160.0))
                        )
                        preserve_approved_min_lot = (
                            float(risk_decision.volume) + 1e-9 >= broker_min_lot
                            and str(risk_decision.reason or "").startswith("approved_bootstrap_min_lot")
                        )
                        micro_lot_cap = _micro_lot_cap(micro_config, mode, account_equity)
                        if micro_lot_cap is not None:
                            configured_max_lot = min(configured_max_lot, micro_lot_cap)
                            if normalized_volume + 1e-9 >= broker_min_lot:
                                configured_max_lot = max(configured_max_lot, broker_min_lot)
                        normalized_volume = min(normalized_volume, configured_max_lot)
                        if normalized_volume + 1e-9 < broker_min_lot and volume + 1e-9 >= broker_min_lot:
                            normalized_volume = broker_min_lot
                        normalized_volume = _preserve_approved_broker_min_lot(
                            normalized_volume=normalized_volume,
                            approved_volume=float(risk_decision.volume),
                            broker_min_lot=float(broker_min_lot),
                            preserve_min_lot=preserve_approved_min_lot,
                        )
                        normalized_volume = mt5_client.normalize_volume(resolved_symbol, normalized_volume)
                        if bootstrap_active and normalized_volume > 0.0 and normalized_volume + 1e-9 < broker_min_lot:
                            min_lot_projected_loss = _projected_trade_risk_usd(
                                entry=entry_price,
                                stop=final_stop,
                                volume=float(broker_min_lot),
                                contract_size=float(symbol_info.get("trade_contract_size", 1.0)),
                                tick_size=float(symbol_info.get("trade_tick_size")) if _is_positive_number(symbol_info.get("trade_tick_size")) else None,
                                tick_value=float(symbol_tick_value) if symbol_tick_value is not None else None,
                            )
                            min_lot_tolerance_cap = _runtime_bootstrap_tolerance_cap(
                                symbol_key=resolved_symbol,
                                per_trade_cap=float(effective_risk_cap),
                                account_equity=float(account_equity),
                                bootstrap_equity_threshold=float(micro_config.get("bootstrap_equity_threshold", 160.0)),
                                broker_min_lot=float(broker_min_lot),
                                setup=str(candidate.setup or ""),
                                session_name=str(session_context.session_name),
                                probability=float(final_decision.get("probability", preliminary_ai.probability)),
                                expected_value_r=float(preliminary_ai.expected_value_r),
                                confluence_score=float(candidate.confluence_score),
                            )
                            if (
                                min_lot_projected_loss <= min_lot_tolerance_cap
                                and (float(projected_open_risk_usd) + float(min_lot_projected_loss)) <= float(total_exposure_cap)
                            ):
                                normalized_volume = float(broker_min_lot)
                                normalized_volume = mt5_client.normalize_volume(resolved_symbol, normalized_volume)
                        if normalized_volume <= 0:
                            mark_block(resolved_symbol, "final_volume_zero", symbol_status)
                            continue

                        final_projected_loss = _projected_trade_risk_usd(
                            entry=entry_price,
                            stop=final_stop,
                            volume=normalized_volume,
                            contract_size=float(symbol_info.get("trade_contract_size", 1.0)),
                            tick_size=float(symbol_info.get("trade_tick_size")) if _is_positive_number(symbol_info.get("trade_tick_size")) else None,
                            tick_value=float(symbol_tick_value) if symbol_tick_value is not None else None,
                        )
                        runtime_tolerance_applied = False
                        runtime_tolerance_cap = 0.0
                        if micro_active and final_projected_loss > effective_risk_cap:
                            if bootstrap_active:
                                runtime_tolerance_applied, runtime_tolerance_cap = _runtime_bootstrap_trade_allowed(
                                    symbol_key=resolved_symbol,
                                    projected_loss_usd=float(final_projected_loss),
                                    volume=float(normalized_volume),
                                    broker_min_lot=float(broker_min_lot),
                                    projected_open_risk_usd=float(projected_open_risk_usd),
                                    per_trade_cap=float(effective_risk_cap),
                                    total_exposure_cap=float(total_exposure_cap),
                                    account_equity=float(account_equity),
                                    bootstrap_equity_threshold=float(micro_config.get("bootstrap_equity_threshold", 160.0)),
                                    setup=str(candidate.setup or ""),
                                    session_name=str(session_context.session_name),
                                    probability=float(final_decision.get("probability", preliminary_ai.probability)),
                                    expected_value_r=float(preliminary_ai.expected_value_r),
                                    confluence_score=float(candidate.confluence_score),
                                )
                            if not runtime_tolerance_applied:
                                mark_block(
                                    resolved_symbol,
                                    "bootstrap_trade_risk_exceeds_cap" if bootstrap_active else "micro_survival_trade_risk_exceeds_usd",
                                    symbol_status,
                                )
                                continue
                        if micro_active and (projected_open_risk_usd + final_projected_loss) > total_exposure_cap:
                            if not runtime_tolerance_applied:
                                mark_block(
                                    resolved_symbol,
                                    "bootstrap_total_risk_exceeds_cap" if bootstrap_active else "micro_survival_total_risk_exceeds_usd",
                                    symbol_status,
                                )
                                continue
                        session_basket_cap = max(0.0, float(account_equity) * 0.05)
                        if (float(session_projected_open_risk_usd) + float(final_projected_loss)) > session_basket_cap:
                            mark_block(
                                resolved_symbol,
                                "session_basket_dd_projection_exceeds_cap",
                                symbol_status,
                            )
                            continue

                        slippage_points = int(final_decision.get("max_slippage_points") or int(risk_config["max_slippage_points"]))
                        slippage_points = min(slippage_points, int(risk_config["max_slippage_points"]))
                        ai_summary_payload = {
                            "probability": float(final_decision.get("probability", preliminary_ai.probability)),
                            "expected_value_r": float(preliminary_ai.expected_value_r),
                            "rationale": str(final_decision.get("rationale", "")),
                            "online_score": float(online_score),
                            "ai_mode": str(final_decision.get("ai_mode", "local_fallback")),
                            "verdict_class": str(final_decision.get("ai_verdict_class", "")),
                        }
                        broker_snapshot_payload = (
                            dict(symbol_info.get("bridge_snapshot", {}))
                            if isinstance(symbol_info.get("bridge_snapshot"), dict)
                            else {}
                        )
                        entry_context_payload = {
                            "strategy_family": candidate.strategy_family,
                            "strategy_key": str(strategy_key),
                            "strategy_state": str(strategy_state),
                            "strategy_score": float(strategy_score),
                            "strategy_recent_performance": float(strategy_recent_performance),
                            "strategy_pool": list(resolved_strategy_pool),
                            "strategy_pool_ranking": list(strategy_pool_ranking),
                            "strategy_pool_winner": str(winning_strategy_key),
                            "winning_strategy_reason": str(winning_strategy_reason),
                            "entry_kind": candidate.entry_kind,
                            "lane_name": str(trade_lane),
                            "management_template": str(candidate.meta.get("management_template") or ""),
                            "timeframe": candidate_timeframe,
                            "session_name": session_context.session_name,
                            "market_status": market_status,
                            "regime_state": str(regime_state),
                            "regime_fit": float(strategy_regime_fit),
                            "session_fit": float(strategy_session_fit),
                            "volatility_fit": float(strategy_volatility_fit),
                            "pair_behavior_fit": float(strategy_pair_behavior_fit),
                            "execution_quality_fit": float(strategy_execution_fit),
                            "entry_timing_score": float(entry_timing_score_value),
                            "structure_cleanliness_score": float(structure_cleanliness_score_value),
                            "market_data_source": str(symbol_status.runtime_market_data_source or runtime_market_data_source),
                            "market_data_consensus_state": str(
                                symbol_status.runtime_market_data_consensus_state
                                or runtime_market_data_consensus_state
                            ),
                            "economics_source": str(symbol_info.get("economics_source", "mt5_client")),
                            "risk_mode": str(risk_diagnostics.get("risk_mode", "standard")),
                            "effective_risk_model": str(risk_diagnostics.get("effective_risk_model", "")),
                            "microstructure_score": candidate_microstructure_score.as_dict(),
                            "lead_lag_snapshot": candidate_lead_lag_snapshot.as_dict(),
                            "event_directive": candidate_event_directive.as_dict(),
                            "execution_minute_profile": dict(candidate_execution_profile),
                            "runtime_bootstrap_tolerance_applied": bool(runtime_tolerance_applied),
                            "runtime_bootstrap_tolerance_cap_usd": float(runtime_tolerance_cap),
                            "confluence_score": float(candidate.confluence_score),
                            "rule_confluence_required": float(rule_confluence_required),
                            "stop_geometry_source": str(stop_geometry_source),
                            "final_stop_geometry_source": str(final_stop_source),
                            "xau_grid_mode": str(candidate.meta.get("grid_mode") or ""),
                            "approved_rr_target": str(exit_profile.get("approved_rr_target", "")),
                            "transition_momentum": float(candidate.meta.get("transition_momentum", 0.0) or 0.0),
                            "transition_momentum_size_multiplier": float(candidate.meta.get("transition_momentum_size_multiplier", 1.0) or 1.0),
                            "compression_burst_size_multiplier": float(candidate.meta.get("compression_burst_size_multiplier", 1.0) or 1.0),
                            "velocity_decay": float(candidate.meta.get("velocity_decay", 1.0) or 1.0),
                            "velocity_decay_score_penalty": float(candidate.meta.get("velocity_decay_score_penalty", 0.0) or 0.0),
                            "velocity_trades_per_10_bars": float(candidate.meta.get("velocity_trades_per_10_bars", 0.0) or 0.0),
                            "correlation_penalty": float(candidate.meta.get("correlation_penalty", 0.0) or 0.0),
                        }
                        final_decision["partial_close_enabled"] = False
                        request = ExecutionRequest(
                            signal_id=candidate.signal_id,
                            symbol=resolved_symbol,
                            side=candidate.side,
                            volume=normalized_volume,
                            entry_price=entry_price,
                            stop_price=final_stop,
                            take_profit_price=final_tp,
                            mode=mode,
                            setup=candidate.setup,
                            regime=regime.label,
                            probability=float(final_decision["probability"]),
                            expected_value_r=preliminary_ai.expected_value_r,
                            slippage_points=slippage_points,
                            trailing_enabled=bool(final_decision["trailing_enabled"]),
                            partial_close_enabled=False,
                            news_status=news_decision.reason,
                            final_decision_json=json.dumps(final_decision, sort_keys=True),
                            trading_enabled=trading_enabled,
                            account=str(bridge_context_account or ""),
                            magic=int(bridge_context_magic or 0),
                            timeframe=candidate_timeframe,
                            proof_trade=False,
                            entry_reason=str(final_decision.get("rationale", candidate.setup)),
                            ai_summary_json=json.dumps(ai_summary_payload, sort_keys=True),
                            broker_snapshot_json=json.dumps(broker_snapshot_payload, sort_keys=True),
                            entry_context_json=json.dumps(entry_context_payload, sort_keys=True),
                            account_currency=str(micro_config.get("account_currency", "NZD")).upper(),
                            entry_spread_points=float(spread_points),
                            session_name=str(session_context.session_name),
                            strategy_key=str(strategy_key),
                        )

                        if bridge_trade_mode:
                            bridge_symbol_key = _normalize_symbol_key(request.symbol)
                            bridge_min_lot = max(
                                float(symbol_info.get("volume_min", 0.01)),
                                float(bridge_min_lot_by_symbol.get(bridge_symbol_key, symbol_info.get("volume_min", 0.01))),
                            )
                            bridge_risk_pct = clamp(
                                float(bridge_risk_defaults["base_risk_pct"])
                                * clamp(float(account["equity"]) / 50.0 if float(account["equity"]) > 0 else 1.0, 0.5, 4.0),
                                float(bridge_risk_defaults["min_risk_pct"]),
                                float(bridge_risk_defaults["max_risk_pct"]),
                            )
                            if _is_xau_grid_setup(candidate.setup):
                                bridge_risk_pct = max(
                                    bridge_risk_pct,
                                    min(
                                        float(bridge_risk_defaults["xau_grid_cycle_risk_pct"]),
                                        float(bridge_risk_defaults["max_risk_pct"]),
                                    ),
                                )
                            bridge_budget_usd = max(
                                float(budget_usd),
                                clamp(
                                    float(account["equity"]) * float(bridge_risk_pct),
                                    float(bridge_risk_defaults["min_budget_usd"]),
                                    float(bridge_risk_defaults["max_budget_usd"]),
                                ),
                            )
                            stop_distance = abs(float(request.entry_price) - float(request.stop_price))
                            margin_initial = symbol_info.get("margin_initial")
                            margin_per_lot = float(margin_initial) if _is_positive_number(margin_initial) else None
                            log_stage(
                                resolved_symbol,
                                "EXECUTION_CHECK",
                                f"{candidate.setup} lot={request.volume:.2f}",
                                now_ts=now,
                            )
                            symbol_status.current_state = "EXECUTION_CHECK"
                            executable_check = validate_trade_executable(
                                account_equity=float(account["equity"]),
                                symbol=request.symbol,
                                lot=float(request.volume),
                                stop_distance=float(stop_distance),
                                contract_size=float(symbol_info.get("trade_contract_size", 1.0)),
                                tick_size=float(symbol_info.get("trade_tick_size", symbol_info.get("point", 0.0001))),
                                tick_value=float(symbol_info.get("trade_tick_value", 1.0)),
                                min_lot=float(bridge_min_lot),
                                margin_free=float(account.get("margin_free", 0.0)) if _is_positive_number(account.get("margin_free")) else None,
                                margin_per_lot=margin_per_lot,
                                risk_budget_usd=float(budget_usd),
                                trade_plan_risk_cap_usd=float(effective_risk_cap),
                                projected_open_risk_usd=float(projected_open_risk_usd),
                                max_total_risk_usd=float(total_exposure_cap),
                                micro_account_equity_threshold=float(micro_config.get("bootstrap_equity_threshold", 160.0)),
                                micro_min_risk_usd=float(micro_config.get("bootstrap_min_risk_amount", 1.0)),
                                micro_risk_pct=max(float(requested_risk_pct), float(micro_config.get("risk_pct_floor", 0.001))),
                                micro_min_lot_risk_multiplier=float(micro_config.get("bootstrap_min_lot_risk_multiplier", 6.0)),
                            )
                            if not executable_check.executable:
                                symbol_status.pre_exec_status = "FAIL"
                                symbol_status.pre_exec_reason = str(executable_check.reason)
                                _update_runtime_metrics(
                                    resolved_symbol,
                                    now,
                                    pre_exec_pass=False,
                                    pre_exec_fail_reason=str(executable_check.reason),
                                    broker_min_lot_tolerance_active=bool(executable_check.min_lot_tolerance_applied),
                                    estimated_loss_usd=float(executable_check.estimated_loss_usd),
                                    min_lot_loss_usd=float(executable_check.min_lot_loss_usd),
                                    effective_risk_cap_usd=float(effective_risk_cap),
                                    risk_cap_usd=float(effective_risk_cap),
                                    required_margin=float(executable_check.required_margin),
                                    margin_free=float(executable_check.margin_free or 0.0),
                                    last_decision_type="EXECUTION_CHECK",
                                    last_decision_reason=str(executable_check.reason),
                                )
                                mark_block(resolved_symbol, executable_check.reason, symbol_status)
                                continue
                            symbol_status.pre_exec_status = "PASS"
                            symbol_status.pre_exec_reason = ""
                            _update_runtime_metrics(
                                resolved_symbol,
                                now,
                                pre_exec_pass=True,
                                pre_exec_fail_reason="",
                                broker_min_lot_tolerance_active=bool(executable_check.min_lot_tolerance_applied),
                                estimated_loss_usd=float(executable_check.estimated_loss_usd),
                                min_lot_loss_usd=float(executable_check.min_lot_loss_usd),
                                effective_risk_cap_usd=float(effective_risk_cap),
                                risk_cap_usd=float(effective_risk_cap),
                                required_margin=float(executable_check.required_margin),
                                margin_free=float(executable_check.margin_free or 0.0),
                                last_decision_type="EXECUTION_CHECK",
                                last_decision_reason=str(executable_check.reason),
                            )
                            point_size = max(float(symbol_info.get("point", 0.0001)), 1e-9)
                            spread_reference_points = max(
                                1.0,
                                _normalize_runtime_spread_points(
                                    resolved_symbol,
                                    float(
                                        row.get("m5_spread_mean_14")
                                        or row.get("m5_spread_avg_20")
                                        or row.get("m15_spread_mean_14")
                                        or row.get("m15_spread_avg_20")
                                        or row.get("m5_spread")
                                        or 0.0
                                    ),
                                    symbol_info=symbol_info,
                                    max_spread_points=float(risk_config["max_spread_points"]),
                                ),
                            )
                            spread_atr_points = max(
                                0.0,
                                _normalize_runtime_spread_points(
                                    resolved_symbol,
                                    float(
                                        row.get("m5_spread_atr_14")
                                        or row.get("m15_spread_atr_14")
                                        or 0.0
                                    ),
                                    symbol_info=symbol_info,
                                    max_spread_points=float(risk_config["max_spread_points"]),
                                ),
                            )
                            stop_distance_points = abs(float(request.entry_price) - float(request.stop_price)) / point_size
                            spread_penalty_r = _normalize_runtime_spread_points(
                                resolved_symbol,
                                float(row["m5_spread"]),
                                symbol_info=symbol_info,
                                max_spread_points=float(risk_config["max_spread_points"]),
                            ) / max(float(stop_distance_points), 1.0)
                            net_expected_value_r = float(request.expected_value_r) - float(spread_penalty_r)
                            grid_step_points_hint = 0.0
                            if is_grid_candidate:
                                grid_step_atr_k = float(candidate.meta.get("grid_step_atr_k", 0.0))
                                if grid_step_atr_k > 0:
                                    spacing_multiplier_hint = float(candidate.meta.get("grid_spacing_multiplier_hint", 1.0) or 1.0)
                                    grid_step_points_hint = max(
                                        1.0,
                                        ((float(row["m5_atr_14"]) * grid_step_atr_k) / point_size) * spacing_multiplier_hint,
                                    )
                            manager_reason = str(
                                candidate.meta.get("manager_reason")
                                or streak_adjust.get("manager_reason")
                                or "manager_neutral"
                            )
                            action = {
                                "signal_id": request.signal_id,
                                "action_type": "OPEN_MARKET",
                                "symbol": request.symbol,
                                "side": request.side,
                                "lot": request.volume,
                                "sl": request.stop_price,
                                "tp": request.take_profit_price,
                                "max_slippage_points": int(bridge_config.get("max_slippage_points", request.slippage_points)),
                                "expiry_utc": (datetime.now(tz=UTC) + timedelta(seconds=max(3, int(bridge_config.get("poll_ttl_seconds", 10))))).isoformat(),
                                "reason": request.news_status or "queued",
                                "ai_summary": {
                                    "probability": request.probability,
                                    "expected_value_r": request.expected_value_r,
                                    "rationale": final_decision.get("rationale", ""),
                                    "online_score": online_score,
                                    "ai_mode": final_decision.get("ai_mode", "local_fallback"),
                                },
                                "trailing": {
                                    "enabled": request.trailing_enabled,
                                    "activationR": float(exit_profile["trail_activation_r"]),
                                    "atrMult": float(exit_profile["trail_atr"]),
                                },
                                "breakeven": {
                                    "enabled": bool(request.trailing_enabled or float(exit_profile["breakeven_trigger_r"]) > 0.0),
                                    "triggerPoints": 0,
                                    "triggerR": float(exit_profile["breakeven_trigger_r"]),
                                },
                                "partials": [],
                                "mode": request.mode,
                                "setup": request.setup,
                                "strategy_family": candidate.strategy_family,
                                "strategy_key": str(strategy_key),
                                "strategy_pool": list(strategy_pool),
                                "strategy_pool_ranking": list(strategy_pool_ranking),
                                "strategy_pool_winner": str(winning_strategy_key or strategy_key),
                                "winning_strategy_reason": str(winning_strategy_reason),
                                "strategy_score": float(strategy_score),
                                "strategy_state": str(strategy_state),
                                "strategy_recent_performance": float(strategy_recent_performance),
                                "entry_kind": candidate.entry_kind,
                                "regime": request.regime,
                                "regime_state": str(regime_state),
                                "regime_confidence": float(regime_confidence),
                                "regime_fit": float(strategy_regime_fit),
                                "probability": request.probability,
                                "expected_value_r": request.expected_value_r,
                                "net_expected_value_r": float(net_expected_value_r),
                                "spread_penalty_r": float(spread_penalty_r),
                                "spread_reference_points": float(spread_reference_points),
                                "spread_atr_points": float(spread_atr_points),
                                "approved_rr_target": str(exit_profile.get("approved_rr_target", "")),
                                "dynamic_rr_band": str(exit_profile.get("approved_rr_target", "")),
                                "trade_quality_score": float(trade_quality.score),
                                "trade_quality_band": str(trade_quality.band),
                                "trade_quality_components": dict(trade_quality.components),
                                "microstructure_score": candidate_microstructure_score.as_dict(),
                                "lead_lag_snapshot": candidate_lead_lag_snapshot.as_dict(),
                                "event_directive": candidate_event_directive.as_dict(),
                                "execution_minute_profile": dict(candidate_execution_profile),
                                "execution_quality_state": str(execution_quality.state),
                                "execution_quality_score": float(execution_quality.score),
                                "execution_quality_fit": float(strategy_execution_fit),
                                "session_fit": float(strategy_session_fit),
                                "volatility_fit": float(strategy_volatility_fit),
                                "pair_behavior_fit": float(strategy_pair_behavior_fit),
                                "entry_timing_score": float(entry_timing_score_value),
                                "structure_cleanliness_score": float(structure_cleanliness_score_value),
                                "quality_tier": str(quality_tier),
                                "tier_size_multiplier": float(tier_size_multiplier),
                                "delta_proxy_score": float(delta_proxy_score_value),
                                "compression_proxy_state": str(compression_proxy_state),
                                "compression_expansion_score": float(candidate.meta.get("compression_expansion_score", 0.0) or 0.0),
                                "session_loosen_factor": float(session_loosen_factor_value),
                                "throughput_recovery_active": bool(throughput_recovery_active),
                                "recycle_session": bool(candidate.meta.get("recycle_session", False)),
                                "recycle_origin_session": str(candidate.meta.get("recycle_origin_session") or ""),
                                "recycle_boost_applied": float(candidate.meta.get("recycle_boost_applied", 0.0) or 0.0),
                                "family_rotation_penalty": float(candidate.meta.get("family_rotation_penalty", 0.0) or 0.0),
                                "equity_momentum_mode": str(candidate.meta.get("equity_momentum_mode") or "NEUTRAL"),
                                "verified_reason_code": str(candidate.meta.get("verified_reason_code") or ""),
                                "verified_reason_text": str(candidate.meta.get("verified_reason_text") or ""),
                                "pressure_proxy_score": float(regime.details.get("pressure_proxy_score", 0.0) or 0.0),
                                "continuation_pressure": float(pressure_continuation),
                                "exhaustion_signal": float(pressure_exhaustion),
                                "absorption_signal": float(pressure_absorption),
                                "volatility_forecast_state": str(volatility_state),
                                "liquidity_alignment_score": float(liquidity_decision.score),
                                "nearest_liquidity_above": float(liquidity_decision.nearest_liquidity_above),
                                "nearest_liquidity_below": float(liquidity_decision.nearest_liquidity_below),
                                "liquidity_sweep_detected": bool(liquidity_decision.liquidity_sweep_detected),
                                "news_status": request.news_status,
                                "final_decision_json": request.final_decision_json,
                                "entry_price": request.entry_price,
                                "timeframe": candidate_timeframe,
                                "confluence_score": float(candidate.confluence_score),
                                "action_score": float(trade_quality.score),
                                "risk_cost": float(estimated_loss_usd) / max(float(runtime_effective_risk_cap), 1e-6),
                                "grid_cycle": bool(candidate.meta.get("grid_cycle", False)),
                                "grid_action": str(candidate.meta.get("grid_action", "")),
                                "grid_level": int(candidate.meta.get("grid_level", 0)) if is_grid_candidate else 0,
                                "grid_max_levels": int(candidate.meta.get("grid_max_levels", 0)) if is_grid_candidate else 0,
                                "grid_step_points_hint": float(grid_step_points_hint),
                                "grid_cycle_id": str(candidate.meta.get("grid_cycle_id", "")),
                                "estimated_loss_usd": float(executable_check.estimated_loss_usd),
                                "execution_check_reason": str(executable_check.reason),
                                "economics_source": str(symbol_info.get("economics_source", "mt5_client")),
                                "risk_mode": str(risk_diagnostics.get("risk_mode", "standard")),
                                "effective_risk_model": str(risk_diagnostics.get("effective_risk_model", "")),
                                "effective_risk_cap_usd": float(effective_risk_cap),
                                "total_exposure_cap": float(total_exposure_cap),
                                "session_stop_state": str(risk_diagnostics.get("session_stop_state", "")),
                                "session_stop_reason": str(risk_diagnostics.get("session_stop_reason", "")),
                                "account_currency": str(micro_config.get("account_currency", "NZD")).upper(),
                                "target_account": str(bridge_context_account or ""),
                                "target_magic": int(bridge_context_magic or 0) if bridge_context_magic is not None else None,
                                "market_data_source": str(symbol_status.runtime_market_data_source or runtime_market_data_source),
                                "market_data_consensus_state": str(
                                    symbol_status.runtime_market_data_consensus_state
                                    or runtime_market_data_consensus_state
                                ),
                                "broker_snapshot": broker_snapshot_payload,
                                "entry_context": entry_context_payload,
                                "context_json": {
                                    "symbol": str(request.symbol),
                                    "side": str(request.side),
                                    "setup": str(request.setup or candidate.setup),
                                    "session_name": str(session_context.session_name),
                                    "spread_reference_points": float(spread_reference_points),
                                    "spread_atr_points": float(spread_atr_points),
                                    "spread_penalty_r": float(spread_penalty_r),
                                    "net_expected_value_r": float(net_expected_value_r),
                                    "regime_state": str(regime_state),
                                    "regime_confidence": float(regime_confidence),
                                    "trade_quality_score": float(trade_quality.score),
                                    "trade_quality_band": str(trade_quality.band),
                                    "trade_quality_components": dict(trade_quality.components),
                                    "microstructure_score": candidate_microstructure_score.as_dict(),
                                    "lead_lag_snapshot": candidate_lead_lag_snapshot.as_dict(),
                                    "event_directive": candidate_event_directive.as_dict(),
                                    "execution_minute_profile": dict(candidate_execution_profile),
                                    "selected_trade_band": str(trade_quality.band),
                                    "current_band_target": str(current_band_target),
                                    "fallback_band_active": bool(fallback_band_active),
                                    "reason_higher_band_unavailable": str(reason_higher_band_unavailable),
                                    "best_A_plus_count": int(loop_band_counts.get("A+", 0)),
                                    "best_A_count": int(best_a_count),
                                    "best_B_plus_count": int(loop_band_counts.get("B+", 0)),
                                    "best_B_count": int(loop_band_counts.get("B", 0)),
                                    "best_C_count": int(loop_band_counts.get("C", 0)),
                                    "execution_quality_state": str(execution_quality.state),
                                    "execution_quality_score": float(execution_quality.score),
                                    "news_confidence": float(getattr(news_decision, "decision_confidence", 1.0)),
                                    "daily_state": str(current_daily_state_runtime),
                                    "daily_state_reason": str(current_daily_state_reason_runtime),
                                    "proof_bucket_state": str(proof_bucket_state),
                                    "proof_exception_reason": str(proof_exception_reason),
                                    "session_priority_profile": str(session_priority.session_priority_profile),
                                    "lane_session_priority": str(session_priority.lane_session_priority),
                                    "session_native_pair": bool(session_priority.session_native_pair),
                                    "session_priority_multiplier": float(session_priority.session_priority_multiplier),
                                    "pair_priority_rank_in_session": int(session_priority.pair_priority_rank_in_session),
                                    "lane_budget_share": float(session_priority.lane_budget_share),
                                    "lane_available_capacity": float(risk_diagnostics.get("lane_available_capacity", 0.0)),
                                    "session_adjusted_score": float(session_adjusted),
                                    "exceptional_override_used": bool(session_override.exceptional_override_used),
                                    "exceptional_override_reason": str(session_override.exceptional_override_reason),
                                    "why_non_native_pair_won": str(session_override.why_non_native_pair_won),
                                    "why_native_pair_lost_priority": str(session_override.why_native_pair_lost_priority),
                                    "primary_block_reason": "",
                                    "secondary_block_reason": str(risk_diagnostics.get("last_precheck_reason", "")),
                                    "current_band_attempted": str(current_band_attempted),
                                    "what_would_make_it_pass": str(session_override.why_native_pair_lost_priority)
                                    if (not session_priority.session_native_pair and bool(session_native_leader))
                                    else "",
                                    "strategy_family": str(strategy_key),
                                    "strategy_key": str(strategy_key),
                                    "strategy_pool": list(strategy_pool),
                                    "strategy_pool_ranking": list(strategy_pool_ranking),
                                    "strategy_pool_winner": str(winning_strategy_key or strategy_key),
                                    "winning_strategy_reason": str(winning_strategy_reason),
                                    "strategy_score": float(strategy_score),
                                    "strategy_state": str(strategy_state),
                                    "strategy_recent_performance": float(strategy_recent_performance),
                                    "regime_fit": float(strategy_regime_fit),
                                    "session_fit": float(strategy_session_fit),
                                    "volatility_fit": float(strategy_volatility_fit),
                                    "pair_behavior_fit": float(strategy_pair_behavior_fit),
                                    "execution_quality_fit": float(strategy_execution_fit),
                                    "entry_timing_score": float(entry_timing_score_value),
                                    "structure_cleanliness_score": float(structure_cleanliness_score_value),
                                    "quality_tier": str(quality_tier),
                                    "tier_size_multiplier": float(tier_size_multiplier),
                                    "delta_proxy_score": float(delta_proxy_score_value),
                                    "compression_proxy_state": str(compression_proxy_state),
                                    "compression_expansion_score": float(candidate.meta.get("compression_expansion_score", 0.0) or 0.0),
                                    "session_loosen_factor": float(session_loosen_factor_value),
                                    "throughput_recovery_active": bool(throughput_recovery_active),
                                    "recycle_session": bool(candidate.meta.get("recycle_session", False)),
                                    "recycle_origin_session": str(candidate.meta.get("recycle_origin_session") or ""),
                                    "recycle_boost_applied": float(candidate.meta.get("recycle_boost_applied", 0.0) or 0.0),
                                    "family_rotation_penalty": float(candidate.meta.get("family_rotation_penalty", 0.0) or 0.0),
                                    "equity_momentum_mode": str(candidate.meta.get("equity_momentum_mode") or "NEUTRAL"),
                                    "transition_momentum": float(candidate.meta.get("transition_momentum", 0.0) or 0.0),
                                    "transition_momentum_size_multiplier": float(candidate.meta.get("transition_momentum_size_multiplier", 1.0) or 1.0),
                                    "compression_burst_size_multiplier": float(candidate.meta.get("compression_burst_size_multiplier", 1.0) or 1.0),
                                "velocity_decay": float(candidate.meta.get("velocity_decay", 1.0) or 1.0),
                                "velocity_decay_score_penalty": float(candidate.meta.get("velocity_decay_score_penalty", 0.0) or 0.0),
                                "velocity_trades_per_10_bars": float(candidate.meta.get("velocity_trades_per_10_bars", 0.0) or 0.0),
                                "btc_weekend_mode": bool(candidate.meta.get("btc_weekend_mode", False)),
                                "btc_velocity_decay": float(candidate.meta.get("btc_velocity_decay", 1.0) or 1.0),
                                "streak_adjust_mode": str(streak_adjust.get("mode", candidate.meta.get("streak_adjust_mode", "NEUTRAL"))),
                                "manager_reason": str(manager_reason),
                                "correlation_penalty": float(candidate.meta.get("correlation_penalty", 0.0) or 0.0),
                                "verified_reason_code": str(candidate.meta.get("verified_reason_code") or ""),
                                "verified_reason_text": str(candidate.meta.get("verified_reason_text") or ""),
                                    "approved_rr_target": str(exit_profile.get("approved_rr_target", "")),
                                    "dynamic_rr_band": str(exit_profile.get("approved_rr_target", "")),
                                    "actual_rr_achieved": None,
                                    "btc_weekend_mode": bool(candidate.meta.get("btc_weekend_mode", False)),
                                    "btc_velocity_decay": float(candidate.meta.get("btc_velocity_decay", 1.0) or 1.0),
                                    "streak_adjust_mode": str(streak_adjust.get("mode", candidate.meta.get("streak_adjust_mode", "NEUTRAL"))),
                                    "manager_reason": str(manager_reason),
                                    "market_data_source": str(symbol_status.runtime_market_data_source or runtime_market_data_source),
                                    "market_data_consensus_state": str(
                                        symbol_status.runtime_market_data_consensus_state
                                        or runtime_market_data_consensus_state
                                    ),
                                    "current_phase": str(current_scaling_state.get("current_phase", "PHASE_1")),
                                    "daily_green_streak": int(current_scaling_state.get("daily_green_streak", 0)),
                                    "current_risk_pct": float(current_scaling_state.get("current_risk_pct", 0.0) or 0.0),
                                    "current_max_risk_pct": float(current_scaling_state.get("current_max_risk_pct", 0.0) or 0.0),
                                    "current_ai_threshold_mode": str(current_scaling_state.get("current_ai_threshold_mode", "conservative")),
                                    "base_daily_trade_target": int(current_scaling_state.get("base_daily_trade_target", current_scaling_state.get("current_daily_trade_cap", 0))),
                                    "stretch_daily_trade_target": int(current_scaling_state.get("stretch_daily_trade_target", current_scaling_state.get("current_overflow_daily_trade_cap", 0))),
                                    "hard_upper_limit": int(current_scaling_state.get("hard_upper_limit", current_scaling_state.get("current_overflow_daily_trade_cap", 0))),
                                    "hourly_base_target": int(current_scaling_state.get("hourly_base_target", risk_config.get("max_trades_per_hour", 0))),
                                    "hourly_stretch_target": int(current_scaling_state.get("hourly_stretch_target", risk_config.get("max_trades_per_hour", 0))),
                                    "projected_trade_capacity_today": int(risk_diagnostics.get("projected_trade_capacity_today", current_scaling_state.get("projected_trade_capacity_today", 0))),
                                    "stretch_mode_active": bool(risk_diagnostics.get("stretch_mode_active", cluster_mode_active)),
                                    "cluster_mode_active": bool(risk_diagnostics.get("cluster_mode_active", cluster_mode_active)),
                                    "current_capacity_mode": str(risk_diagnostics.get("current_capacity_mode", "STRETCH" if cluster_mode_active else "BASE")),
                                    "quality_cluster_score": float(risk_diagnostics.get("quality_cluster_score", quality_cluster_score)),
                                    "lane_strength_multiplier": float(risk_diagnostics.get("lane_strength_multiplier", lane_strength_multiplier)),
                                    "lane_score": float(risk_diagnostics.get("lane_score", lane_score)),
                                    "session_density_score": float(risk_diagnostics.get("session_density_score", session_density_score)),
                                    "volatility_forecast_state": str(volatility_state),
                                    "pressure_proxy_score": float(regime.details.get("pressure_proxy_score", 0.0) or 0.0),
                                    "continuation_pressure": float(pressure_continuation),
                                    "exhaustion_signal": float(pressure_exhaustion),
                                    "absorption_signal": float(pressure_absorption),
                                    "liquidity_alignment_score": float(liquidity_decision.score),
                                    "nearest_liquidity_above": float(liquidity_decision.nearest_liquidity_above),
                                    "nearest_liquidity_below": float(liquidity_decision.nearest_liquidity_below),
                                    "liquidity_sweep_detected": bool(liquidity_decision.liquidity_sweep_detected),
                                    "portfolio_bias": str(portfolio_decision.portfolio_bias or ""),
                                    "exposure_cluster_detected": str(portfolio_decision.exposure_cluster_detected or ""),
                                    "correlation_adjustment": float(portfolio_decision.size_multiplier),
                                    "adaptive_risk_state": str(risk_diagnostics.get("adaptive_risk_state", "")),
                                    "risk_modifiers": dict(risk_diagnostics.get("risk_modifiers", {})),
                                    "overflow_band_active": bool(risk_diagnostics.get("overflow_band_active", False)),
                                    "xau_grid_cycle_quality": float(candidate.meta.get("grid_cycle_quality", trade_quality.score) if isinstance(candidate.meta, dict) else float(trade_quality.score)),
                                },
                            }
                            enqueue_result = (
                                bridge_queue.enqueue_with_result(action)
                                if hasattr(bridge_queue, "enqueue_with_result")
                                else None
                            )
                            enqueued = bool(enqueue_result.accepted) if enqueue_result is not None else bool(bridge_queue.enqueue(action))
                            block_reason = str(enqueue_result.reason_code) if enqueue_result is not None else "queue_rejected"
                            symbol_status.trading_allowed = enqueued
                            symbol_status.reason = "queued_for_ea" if enqueued else block_reason
                            if enqueued:
                                summary["orders_placed"] += 1
                                entries_this_symbol += 1
                                projected_open_risk_usd += final_projected_loss
                                session_projected_open_risk_usd += final_projected_loss
                                _append_candidate_verification_log(
                                    runtime["candidate_verification_log_path"],
                                    {
                                        "timestamp": now.isoformat(),
                                        "symbol": str(request.symbol),
                                        "session": str(session_context.session_name),
                                        "regime": str(regime_state),
                                        "strategy_key": str(strategy_key),
                                        "setup_family": str(candidate.strategy_family),
                                        "quality_tier": str(quality_tier),
                                        "tier_size_multiplier": float(tier_size_multiplier),
                                        "recycle_session": bool(candidate.meta.get("recycle_session", False)),
                                        "recycle_origin_session": str(candidate.meta.get("recycle_origin_session") or ""),
                                        "family_rotation_penalty": float(candidate.meta.get("family_rotation_penalty", 0.0) or 0.0),
                                        "delta_proxy_score": float(delta_proxy_score_value),
                                        "compression_state": str(compression_proxy_state),
                                        "compression_expansion_score": float(candidate.meta.get("compression_expansion_score", 0.0) or 0.0),
                                        "transition_momentum": float(candidate.meta.get("transition_momentum", 0.0) or 0.0),
                                        "velocity_decay": float(candidate.meta.get("velocity_decay", 1.0) or 1.0),
                                        "velocity_decay_score_penalty": float(candidate.meta.get("velocity_decay_score_penalty", 0.0) or 0.0),
                                        "velocity_trades_per_10_bars": float(candidate.meta.get("velocity_trades_per_10_bars", 0.0) or 0.0),
                                        "velocity": float(candidate.meta.get("velocity_trades_per_10_bars", 0.0) or 0.0),
                                        "correlation_penalty": float(candidate.meta.get("correlation_penalty", 0.0) or 0.0),
                                        "btc_weekend_mode": bool(candidate.meta.get("btc_weekend_mode", False)),
                                        "btc_velocity_decay": float(candidate.meta.get("btc_velocity_decay", 1.0) or 1.0),
                                        "session_loosen_factor": float(session_loosen_factor_value),
                                        "equity_momentum_mode": str(candidate.meta.get("equity_momentum_mode") or "NEUTRAL"),
                                        "streak_adjust_mode": str(streak_adjust.get("mode", candidate.meta.get("streak_adjust_mode", "NEUTRAL"))),
                                        "manager_reason": str(manager_reason),
                                        "final_score": float(strategy_score),
                                        "final_score_reason": str(winning_strategy_reason),
                                        "verified_reason_code": str(candidate.meta.get("verified_reason_code") or ""),
                                        "verified_reason_text": str(candidate.meta.get("verified_reason_text") or ""),
                                        "approved_rr_target": str(exit_profile.get("approved_rr_target", "")),
                                        "actual_rr_achieved": None,
                                    },
                                )
                                _record_symbol_activity(resolved_symbol, "actions_sent", now)
                                _record_symbol_activity(resolved_symbol, "queued_for_ea", now)
                                idea_lifecycle.mark_delivery_pending(
                                    idea=idea,
                                    now=now,
                                    retry_after_seconds=30,
                                )
                                queued_counts[resolved_symbol.upper()] = queued_counts.get(resolved_symbol.upper(), 0) + 1
                                symbol_status.queued_actions = queued_counts.get(resolved_symbol.upper(), 0)
                                symbol_status.current_state = "QUEUED_FOR_EA"
                                if is_grid_candidate:
                                    symbol_status.grid_cycle_state = "ACTIVE"
                                    symbol_status.grid_leg = (
                                        f"{int(candidate.meta.get('grid_level', 1))}/"
                                        f"{max(1, int(candidate.meta.get('grid_max_levels', grid_scalper.max_levels)))}"
                                    )
                                    symbol_status.grid_cycle_id = str(candidate.meta.get("grid_cycle_id", candidate.signal_id))
                                    symbol_status.grid_last_entry = f"{float(request.entry_price):.2f}"
                                log_stage(
                                    resolved_symbol,
                                    "QUEUED_FOR_EA",
                                    f"{candidate.setup} {candidate.side} queued",
                                    now_ts=now,
                                )
                                logger.info(
                                    "TRADE_QUEUED_FOR_EA",
                                    extra={
                                        "extra_fields": {
                                            "symbol": resolved_symbol,
                                            "setup_id": candidate.setup,
                                            "signal_id": candidate.signal_id,
                                            "strategy": candidate.strategy_family,
                                            "lot": float(request.volume),
                                            "risk_usd": float(executable_check.estimated_loss_usd),
                                            "cycle_id": str(candidate.meta.get("grid_cycle_id", "")),
                                        }
                                    },
                                )
                                if is_grid_candidate and str(candidate.meta.get("grid_action", "")).upper() == "START":
                                    logger.info(
                                        "GRID_CYCLE_START",
                                        extra={
                                            "extra_fields": {
                                                "symbol": resolved_symbol,
                                                "cycle_id": str(candidate.meta.get("grid_cycle_id", candidate.signal_id)),
                                                "probability": float(request.probability),
                                                "confluence": float(candidate.confluence_score),
                                                "spread": float(spread_points),
                                                "session": str(session_context.session_name),
                                            }
                                        },
                                    )
                                    symbol_status.reason = "xau_grid_probe"
                                    symbol_status.last_signal = candidate.setup
                                    symbol_status.ai_reason = f"grid_cycle={candidate.meta.get('grid_cycle_id', candidate.signal_id)}"
                                _update_runtime_metrics(
                                    resolved_symbol,
                                    now,
                                    market_open_status=market_status,
                                    delivered_actions_last_15m=int(bridge_delivery_counts.get(_normalize_symbol_key(resolved_symbol), symbol_status.delivered_actions)),
                                )
                                log_debug(f"ENTRY_QUEUED {resolved_symbol} {candidate.side} {candidate.setup} vol={normalized_volume}")
                            else:
                                mark_block(resolved_symbol, block_reason, symbol_status)
                        elif paper_sim:
                            simulated = OrderResult(
                                accepted=True,
                                order_id=f"paper-{request.signal_id}",
                                reason="paper_simulated",
                                raw={"retcode": "PAPER_SIM"},
                            )
                            journal.record_execution(request, simulated, equity=float(account["equity"]))
                            pnl_amount, pnl_r, exit_price = _paper_sim_outcome(
                                request=request,
                                point=float(symbol_info.get("point", 0.0001)),
                                contract_size=float(symbol_info.get("trade_contract_size", 100.0)),
                                commission_per_lot=float(settings.section("backtest").get("commission_per_lot", 7.0)),
                                slippage_points=slippage_points,
                            )
                            equity_after_close = float(account["equity"]) + pnl_amount
                            journal.mark_closed(
                                request.signal_id,
                                pnl_amount=pnl_amount,
                                pnl_r=pnl_r,
                                equity_after_close=equity_after_close,
                            )
                            journal.log_event(
                                request.signal_id,
                                "PAPER_SIM_CLOSE",
                                {"exit_price": exit_price, "pnl_r": pnl_r, "pnl_amount": pnl_amount},
                            )
                            online_learning.on_trade_close(
                                {
                                    "timestamp_utc": now.isoformat(),
                                    "signal_id": request.signal_id,
                                    "symbol": request.symbol,
                                    "timeframe": candidate_timeframe,
                                    "features_used_hash": hashlib.sha256(request.final_decision_json.encode("utf-8")).hexdigest()[:24],
                                    "side": request.side,
                                    "entry": request.entry_price,
                                    "sl": request.stop_price,
                                    "tp": request.take_profit_price,
                                    "exit": exit_price,
                                    "pnl_r": pnl_r,
                                    "pnl_money": pnl_amount,
                                    "news_state": request.news_status,
                                    "session_state": session_status,
                                    "ai_decision": request.final_decision_json,
                                    "ai_probability": request.probability,
                                    "spread_points": float(row["m5_spread"]),
                                    "lot": request.volume,
                                    "regime": request.regime,
                                    "regime_state": str(candidate.meta.get("regime_state") or request.regime),
                                    "setup": request.setup,
                                    "strategy_key": str(request.strategy_key or candidate.meta.get("strategy_key") or ""),
                                    "lane_name": str(candidate.meta.get("lane_name") or ""),
                                    "management_template": str(candidate.meta.get("management_template") or ""),
                                    "strategy_state": str(candidate.meta.get("strategy_state") or "NORMAL"),
                                    "regime_fit": float(candidate.meta.get("regime_fit") or 0.0),
                                    "session_fit": float(candidate.meta.get("session_fit") or 0.0),
                                    "volatility_fit": float(candidate.meta.get("volatility_fit") or 0.0),
                                    "pair_behavior_fit": float(candidate.meta.get("pair_behavior_fit") or 0.0),
                                    "execution_quality_fit": float(candidate.meta.get("execution_quality_fit") or 0.0),
                                    "entry_timing_score": float(candidate.meta.get("entry_timing_score") or 0.0),
                                    "structure_cleanliness_score": float(candidate.meta.get("structure_cleanliness_score") or 0.0),
                                    "strategy_recent_performance": float(candidate.meta.get("strategy_recent_performance") or 0.0),
                                    "session_name": str(request.session_name or session_context.session_name),
                                    "market_data_source": str(symbol_status.runtime_market_data_source or runtime_market_data_source),
                                    "market_data_consensus_state": str(
                                        symbol_status.runtime_market_data_consensus_state
                                        or runtime_market_data_consensus_state
                                    ),
                                    "mc_win_rate": float(candidate.meta.get("mc_win_rate", 0.0) or 0.0),
                                    "ga_generation_id": float(candidate.meta.get("ga_generation_id", 0.0) or 0.0),
                                    "multi_tf_alignment_score": float(candidate.meta.get("multi_tf_alignment_score", 0.5) or 0.5),
                                    "fractal_persistence_score": float(candidate.meta.get("fractal_persistence_score", 0.5) or 0.5),
                                    "compression_expansion_score": float(candidate.meta.get("compression_expansion_score", 0.0) or 0.0),
                                    "dxy_support_score": float(candidate.meta.get("dxy_support_score", 0.5) or 0.5),
                                    "aggressive_pair_mode": 1.0 if _is_super_aggressive_normal_symbol(request.symbol) else 0.0,
                                    "trajectory_catchup_pressure": float(candidate.meta.get("trajectory_catchup_pressure", 0.0) or 0.0),
                                }
                            )
                            summary["orders_placed"] += 1
                            entries_this_symbol += 1
                            projected_open_risk_usd += final_projected_loss
                            session_projected_open_risk_usd += final_projected_loss
                            _append_candidate_verification_log(
                                runtime["candidate_verification_log_path"],
                                {
                                    "timestamp": now.isoformat(),
                                    "symbol": str(request.symbol),
                                    "session": str(session_context.session_name),
                                    "regime": str(regime_state),
                                    "strategy_key": str(strategy_key),
                                    "setup_family": str(candidate.strategy_family),
                                    "quality_tier": str(quality_tier),
                                    "tier_size_multiplier": float(tier_size_multiplier),
                                    "recycle_session": bool(candidate.meta.get("recycle_session", False)),
                                    "recycle_origin_session": str(candidate.meta.get("recycle_origin_session") or ""),
                                    "family_rotation_penalty": float(candidate.meta.get("family_rotation_penalty", 0.0) or 0.0),
                                    "delta_proxy_score": float(delta_proxy_score_value),
                                    "compression_state": str(compression_proxy_state),
                                    "compression_expansion_score": float(candidate.meta.get("compression_expansion_score", 0.0) or 0.0),
                                    "transition_momentum": float(candidate.meta.get("transition_momentum", 0.0) or 0.0),
                                    "velocity_decay": float(candidate.meta.get("velocity_decay", 1.0) or 1.0),
                                    "velocity_decay_score_penalty": float(candidate.meta.get("velocity_decay_score_penalty", 0.0) or 0.0),
                                    "velocity_trades_per_10_bars": float(candidate.meta.get("velocity_trades_per_10_bars", 0.0) or 0.0),
                                    "correlation_penalty": float(candidate.meta.get("correlation_penalty", 0.0) or 0.0),
                                    "session_loosen_factor": float(session_loosen_factor_value),
                                    "equity_momentum_mode": str(candidate.meta.get("equity_momentum_mode") or "NEUTRAL"),
                                    "final_score": float(strategy_score),
                                    "final_score_reason": "paper_simulated_execution",
                                    "verified_reason_code": str(candidate.meta.get("verified_reason_code") or ""),
                                    "verified_reason_text": str(candidate.meta.get("verified_reason_text") or ""),
                                    "approved_rr_target": str(exit_profile.get("approved_rr_target", "")),
                                    "actual_rr_achieved": None,
                                },
                            )
                            last_trade_opened_at = now
                            _record_symbol_activity(resolved_symbol, "actions_sent", now)
                            idea_lifecycle.mark_trade_sent(idea=idea, now=now)
                            symbol_status.current_state = "DELIVERED"
                            log_stage(
                                resolved_symbol,
                                "TRADE_SENT",
                                f"{candidate.setup} {candidate.side} paper",
                                now_ts=now,
                            )
                            symbol_status.trading_allowed = True
                            symbol_status.reason = "paper_sim_executed"
                            open_positions_journal = journal.get_open_positions()
                        else:
                            receipt = execution.place(request, equity=float(account["equity"]))
                            symbol_status.trading_allowed = receipt.accepted
                            symbol_status.reason = "executed" if receipt.accepted else receipt.reason
                            if receipt.accepted:
                                summary["orders_placed"] += 1
                                entries_this_symbol += 1
                                projected_open_risk_usd += final_projected_loss
                                session_projected_open_risk_usd += final_projected_loss
                                _append_candidate_verification_log(
                                    runtime["candidate_verification_log_path"],
                                    {
                                        "timestamp": now.isoformat(),
                                        "symbol": str(request.symbol),
                                        "session": str(session_context.session_name),
                                        "regime": str(regime_state),
                                        "strategy_key": str(strategy_key),
                                        "setup_family": str(candidate.strategy_family),
                                        "quality_tier": str(quality_tier),
                                        "tier_size_multiplier": float(tier_size_multiplier),
                                        "recycle_session": bool(candidate.meta.get("recycle_session", False)),
                                        "recycle_origin_session": str(candidate.meta.get("recycle_origin_session") or ""),
                                        "family_rotation_penalty": float(candidate.meta.get("family_rotation_penalty", 0.0) or 0.0),
                                        "delta_proxy_score": float(delta_proxy_score_value),
                                        "compression_state": str(compression_proxy_state),
                                        "compression_expansion_score": float(candidate.meta.get("compression_expansion_score", 0.0) or 0.0),
                                        "transition_momentum": float(candidate.meta.get("transition_momentum", 0.0) or 0.0),
                                        "velocity_decay": float(candidate.meta.get("velocity_decay", 1.0) or 1.0),
                                        "velocity_decay_score_penalty": float(candidate.meta.get("velocity_decay_score_penalty", 0.0) or 0.0),
                                        "velocity_trades_per_10_bars": float(candidate.meta.get("velocity_trades_per_10_bars", 0.0) or 0.0),
                                        "correlation_penalty": float(candidate.meta.get("correlation_penalty", 0.0) or 0.0),
                                        "session_loosen_factor": float(session_loosen_factor_value),
                                        "equity_momentum_mode": str(candidate.meta.get("equity_momentum_mode") or "NEUTRAL"),
                                        "final_score": float(strategy_score),
                                        "final_score_reason": "live_execution_accepted",
                                        "verified_reason_code": str(candidate.meta.get("verified_reason_code") or ""),
                                        "verified_reason_text": str(candidate.meta.get("verified_reason_text") or ""),
                                        "approved_rr_target": str(exit_profile.get("approved_rr_target", "")),
                                        "actual_rr_achieved": None,
                                    },
                                )
                                last_trade_opened_at = now
                                _record_symbol_activity(resolved_symbol, "actions_sent", now)
                                idea_lifecycle.mark_trade_sent(idea=idea, now=now)
                                symbol_status.current_state = "DELIVERED"
                                log_stage(
                                    resolved_symbol,
                                    "TRADE_SENT",
                                    f"{candidate.setup} {candidate.side} live",
                                    now_ts=now,
                                )
                                open_positions_journal = _live_open_positions_for_context(
                                    account_id=bridge_context_account,
                                    magic_id=bridge_context_magic,
                                    now_ts=now,
                                )
                            else:
                                mark_block(resolved_symbol, receipt.reason, symbol_status)
            else:
                for symbol_status in symbol_statuses:
                    symbol_status.reason = f"soft_kill_manage_only:{kill_status.reason or 'unknown'}"
                    symbol_status.trading_allowed = False
                    symbol_status.current_state = "BLOCK"

            if global_stats.consecutive_losses >= 3:
                monitor.alert("LOSS_STREAK", "Three or more consecutive losses detected", streak=global_stats.consecutive_losses)
            for configured_symbol in configured_symbols:
                context = symbol_contexts.get(configured_symbol)
                if not context:
                    continue
                row = context["row"]
                effective_spread_points = _normalize_runtime_spread_points(
                    str(context["resolved_symbol"]),
                    float(row["m5_spread"]),
                    symbol_info=context.get("symbol_info") if isinstance(context.get("symbol_info"), dict) else None,
                    max_spread_points=float(risk_config["max_spread_points"]),
                )
                if effective_spread_points >= float(risk_config["spread_disorder_points"]):
                    monitor.alert(
                        "SPREAD_DISORDER",
                        "Spread exceeds disorder threshold",
                        symbol=str(context["resolved_symbol"]),
                        spread=float(effective_spread_points),
                    )

            for symbol_status in symbol_statuses:
                metrics = symbol_runtime_metrics.get(_normalize_symbol_key(symbol_status.symbol), {})
                if not metrics:
                    continue
                symbol_status.delivered_actions = int(metrics.get("delivered_actions_last_15m", symbol_status.delivered_actions))
                symbol_status.stale_archives = int(metrics.get("stale_archives_last_15m", symbol_status.stale_archives))
                symbol_status.market_closed_blocks = int(metrics.get("market_closed_rejects_last_15m", symbol_status.market_closed_blocks))
                symbol_status.last_ai_mode = str(metrics.get("last_ai_mode", symbol_status.last_ai_mode or "fallback"))
                if (
                    int(symbol_status.queued_actions) > 0
                    and symbol_status.current_state not in {"DELIVERED", "OPEN_POSITION", "MANAGING_POSITION", "CLOSED"}
                ):
                    symbol_status.current_state = "QUEUED_FOR_EA"
                if not symbol_status.current_state:
                    symbol_status.current_state = _state_from_reason(symbol_status.reason)
                symbol_status.ai_reason = (
                    f"mode={symbol_status.last_ai_mode or 'none'} "
                    f"last_ai={metrics.get('last_ai_called_at', '') or 'none'} "
                    f"cand15={int(metrics.get('candidate_attempts_last_15m', 0))} "
                    f"ai15={int(metrics.get('ai_reviews_last_15m', 0))} "
                    f"q15={int(metrics.get('queued_for_ea_last_15m', 0))} "
                    f"del15={int(metrics.get('delivered_actions_last_15m', 0))} "
                    f"closed15={int(metrics.get('market_closed_rejects_last_15m', 0))}"
                )

            last100 = journal.summary_last(100)
            last200 = journal.summary_last(200)
            rollout_summary = journal.summary_scope(500, closed_after=rollout_started_at)
            monitor.dashboard(
                DashboardState(
                    mode=mode,
                    current_utc=now,
                    current_local=local_now,
                    session_status=session_status,
                    active_session_name=active_session_name,
                    equity=float(account["equity"]),
                    daily_pnl_pct=global_stats.daily_pnl_pct,
                    drawdown_pct=global_stats.rolling_drawdown_pct,
                    total_open_positions=len(open_positions_journal),
                    max_total_positions=effective_max_positions_total,
                    ai_active=bool(ai_config["enabled"]),
                    win_rate_last_100=float(last100["win_rate"]),
                    win_rate_last_200=float(last200["win_rate"]),
                    queued_actions_total=sum(queued_counts.values()) if bridge_trade_mode else 0,
                    symbols=symbol_statuses,
                    next_check_seconds=max(1, int(round(loop_sleep_seconds))),
                    rollout_win_rate=float(rollout_summary["win_rate"]),
                    rollout_trade_count=int(rollout_summary["trades"]),
                    account_label=account_label,
                )
            )

            if once:
                break
            if loop_limit is not None and summary["loops"] >= loop_limit:
                break
            time.sleep(max(1.0, loop_sleep_seconds))
    finally:
        if bridge_handle is not None:
            bridge_handle.stop()

    return summary


def run_verify() -> int:
    settings = load_settings()
    logging_config = settings.section("logging") if isinstance(settings.raw.get("logging"), dict) else {}
    logger = LoggerFactory(
        log_file=settings.runtime_paths.logs_dir / "apex.log",
        rotate_max_bytes=int(logging_config.get("rotate_max_bytes", 10 * 1024 * 1024)),
        rotate_backup_count=int(logging_config.get("rotate_backup_count", 7)),
        retention_days=int(logging_config.get("retention_days", 365)),
    ).build()
    system_config = settings.section("system")
    configured_symbols = settings.symbols()
    default_terminal_path = str(system_config.get("mt5_terminal_path", "") or "") or None
    mt5_client = MT5Client(
        credentials=MT5Credentials.from_env(default_terminal_path=default_terminal_path),
        journal_db=settings.path("data.trade_db"),
        logger=logger,
        disable_mt5=False,
    )

    print("Configured Symbols:", ", ".join(configured_symbols))
    verification = mt5_client.verify_connection(configured_symbols)
    print(json.dumps(verification, indent=2, default=str, sort_keys=True))
    if mt5_client.connected:
        mt5_client.shutdown()
    return 0 if verification.get("ok", False) else 1


def run_bridge_only() -> int:
    runtime = build_runtime(skip_mt5=True)
    bridge_config: dict[str, Any] = runtime["bridge_config"]
    if not bool(bridge_config.get("enabled", False)):
        print("bridge.enabled is false in config/settings.yaml; overriding because --bridge-only was passed")
    try:
        from src.bridge_server import run_bridge_forever
    except Exception as exc:
        print(f"Unable to start bridge server: {exc}")
        return 1

    dashboard_runtime_config = runtime["settings"].raw.get("dashboard", {}) if isinstance(runtime["settings"].raw.get("dashboard"), dict) else {}
    bind_host = str(bridge_config.get("host", "127.0.0.1"))
    bind_port = int(bridge_config.get("port", 8000))
    if bool(dashboard_runtime_config.get("public_enabled", False)):
        bind_host = str(dashboard_runtime_config.get("bind_host") or dashboard_runtime_config.get("host") or "0.0.0.0")
        bind_port = int(dashboard_runtime_config.get("bind_port") or dashboard_runtime_config.get("port") or bind_port)

    run_bridge_forever(
        host=bind_host,
        port=bind_port,
        queue=runtime["bridge_queue"],
        journal=runtime["journal"],
        online_learning=runtime["online_learning"],
        learning_brain=runtime["learning_brain"],
        session_name_resolver=runtime["session_profile"].infer_name,
        ai_gate=runtime["ai_gate"],
        logger=runtime["logger"],
        auth_token=str(bridge_config.get("auth_token", "")),
        orchestrator_config={
            **(bridge_config.get("orchestrator", {}) if isinstance(bridge_config.get("orchestrator"), dict) else {}),
            **{
                key: dict(runtime["settings"].raw.get(key) or {})
                for key in (
                    "microstructure",
                    "lead_lag",
                    "event_playbooks",
                    "aggression",
                    "shadow_promotion",
                    "execution_memory",
                    "self_heal",
                )
                if key
                not in (bridge_config.get("orchestrator", {}) if isinstance(bridge_config.get("orchestrator"), dict) else {})
                and isinstance(runtime["settings"].raw.get(key), dict)
            },
            **(
                {"news_mode": str(runtime["settings"].section("news").get("mode", "SAFE")).upper()}
                if "news_mode"
                not in (bridge_config.get("orchestrator", {}) if isinstance(bridge_config.get("orchestrator"), dict) else {})
                else {}
            ),
        },
        execution_config=runtime["settings"].section("execution"),
        risk_config=runtime["settings"].section("risk"),
        xau_grid_config=runtime["settings"].section("xau_grid_scalper"),
        dashboard_config=runtime["settings"].raw.get("dashboard", {}),
        strategy_optimizer=runtime["strategy_optimizer"],
        symbol_rules_path=runtime["settings"].resolve_path_value(str(bridge_config.get("symbol_rules_file", "config/symbol_rules.yaml"))),
        market_data_status_provider=runtime["market_data"].status_snapshot,
        runtime_metrics_provider=lambda: {},
        telegram_config=runtime["settings"].raw.get("telegram", {}) if isinstance(runtime["settings"].raw.get("telegram"), dict) else {},
        aggression_config=runtime["settings"].raw.get("aggression_controller", {}) if isinstance(runtime["settings"].raw.get("aggression_controller"), dict) else {},
    )
    return 0


def run_eval_last(limit: int) -> int:
    runtime = build_runtime(skip_mt5=True)
    journal: TradeJournal = runtime["journal"]
    online_learning: OnlineLearningEngine = runtime["online_learning"]
    session_profile: SessionProfile = runtime["session_profile"]
    closed = journal.closed_trades(max(1, limit))
    summary = {
        "journal": journal.summary_last(max(1, limit)),
        "performance": build_performance_report(closed, session_name_resolver=session_profile.infer_name),
        "online_model": online_learning.eval_last(max(1, limit)),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def run_report(limit: int = 500) -> int:
    runtime = build_runtime(skip_mt5=True)
    settings = runtime["settings"]
    journal: TradeJournal = runtime["journal"]
    session_profile: SessionProfile = runtime["session_profile"]
    fee_estimate = float(settings.section("risk").get("fee_per_lot_estimate", settings.section("backtest").get("commission_per_lot", 7.0)))
    payload = {
        "report_limit": int(limit),
        "last_100": build_performance_report(journal.closed_trades(100), session_name_resolver=session_profile.infer_name),
        "last_200": build_performance_report(journal.closed_trades(200), session_name_resolver=session_profile.infer_name),
        "last_n": build_performance_report(journal.closed_trades(max(1, int(limit))), session_name_resolver=session_profile.infer_name),
        "open_positions": len(journal.get_open_positions()),
        "micro_survivability_last_20": journal.micro_survivability_summary(limit=20, fee_per_lot_estimate=fee_estimate),
        "blocked_counts_total": journal.blocked_counts(),
        "blocked_counts_last_24h": journal.blocked_counts(since_hours=24),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def run_ai_test() -> int:
    runtime = build_runtime(skip_mt5=True)
    ai_gate: AIGate = runtime["ai_gate"]
    passed, reason = ai_gate.ai_test()
    health = ai_gate.health()
    status = "PASS" if passed else "FAIL"
    print(f"AI TEST {status}: {reason}")
    print(json.dumps(health, indent=2, sort_keys=True))
    return 0 if passed else 1


def run_training() -> None:
    runtime = build_runtime()
    trainer: Trainer = runtime["trainer"]
    primary_symbol = runtime["primary_symbol"]
    metrics = trainer.run(primary_symbol)
    print(metrics)


def run_backtest(preset: str = "realistic") -> None:
    runtime = build_runtime()
    settings = runtime["settings"]
    market_data: MarketDataService = runtime["market_data"]
    feature_engineer: FeatureEngineer = runtime["feature_engineer"]
    backtester: Backtester = runtime["backtester"]
    strategy_engines: dict[str, StrategyEngine] = runtime["strategy_engines"]
    configured_symbols: list[str] = runtime["configured_symbols"]
    resolved_symbols: dict[str, str] = runtime["resolved_symbols"]
    backtest_settings = settings.section("backtest")

    preset_key = preset.strip().lower()
    if preset_key == "frictionless":
        backtester.spread_points = 0.0
        backtester.slippage_points = 0.0
        backtester.commission_per_lot = 0.0
        backtester.latency_ms = 0
        backtester.strict_plausibility = False
        backtester.whitelist_high_win_rate = True
    else:
        backtester.spread_points = float(backtest_settings["default_spread_points"])
        backtester.slippage_points = float(backtest_settings["default_slippage_points"])
        backtester.commission_per_lot = float(backtest_settings["commission_per_lot"])
        backtester.latency_ms = int(backtest_settings.get("default_latency_ms", 0))
        backtester.strict_plausibility = bool(backtest_settings.get("strict_plausibility", True))
        backtester.max_plausible_win_rate = float(backtest_settings.get("max_plausible_win_rate", 0.85))
        backtester.min_trades_for_plausibility = int(backtest_settings.get("min_trades_for_plausibility", 200))
        backtester.whitelist_high_win_rate = False

    symbols_to_test = configured_symbols[:3]
    counts = {"M1": 12000, "M5": 20000, "M15": 10000, "H1": 4000, "H4": 2000}
    report: dict[str, dict[str, object]] = {}
    report["_preset"] = {
        "name": preset_key,
        "spread_points": backtester.spread_points,
        "slippage_points": backtester.slippage_points,
        "commission_per_lot": backtester.commission_per_lot,
        "latency_ms": backtester.latency_ms,
        "strict_plausibility": backtester.strict_plausibility,
    }

    for configured_symbol in symbols_to_test:
        resolved_symbol = resolved_symbols[configured_symbol]
        frames: dict[str, object] = {}
        for timeframe, count in counts.items():
            cached = market_data.load_cached(resolved_symbol, timeframe)
            if cached is not None and len(cached) > 100:
                frames[timeframe] = cached
            elif timeframe in {"M1", "H4"}:
                continue
            else:
                frames[timeframe] = market_data.fetch(resolved_symbol, timeframe, count)

        if "M5" not in frames or "M15" not in frames or "H1" not in frames:
            report[resolved_symbol] = {"error": "insufficient_data"}
            continue

        features = feature_engineer.build(
            frames.get("M1"),
            frames["M5"],
            frames["M15"],
            frames["H1"],
            frames.get("H4"),
        )
        backtester.strategy_engine = strategy_engines[configured_symbol]
        metrics = backtester.run(features)
        report[resolved_symbol] = {
            "trade_count": metrics.get("trade_count"),
            "win_rate": metrics.get("win_rate"),
            "profit_factor": metrics.get("profit_factor"),
            "expectancy_r": metrics.get("expectancy_r"),
            "net_r": metrics.get("net_r"),
            "max_drawdown_r": metrics.get("max_drawdown_r"),
            "sharpe": metrics.get("sharpe"),
            "max_consecutive_losses": metrics.get("max_consecutive_losses"),
            "events": metrics.get("events", {}),
        }
    print(json.dumps(report, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX autonomous MT5 multi-symbol trader")
    parser.add_argument("--once", action="store_true", help="Run one loop then exit")
    parser.add_argument("--verify", action="store_true", help="Verify MT5 connectivity and symbol resolution, then exit")
    parser.add_argument("--bridge-serve", action="store_true", help="Run bridge API server in background while strategy loop runs")
    parser.add_argument("--bridge-only", action="store_true", help="Run bridge API server only and do not run strategy loop")
    parser.add_argument("--paper-sim", action="store_true", help="Run strategy loop with internal simulated fills (no MT5 orders)")
    parser.add_argument("--smoke-demo", action="store_true", help="Run a bounded DEMO/PAPER smoke loop that may place demo orders if all gates approve")
    parser.add_argument("--smoke-loops", type=int, default=20, help="Loop count for --smoke-demo")
    parser.add_argument("--eval-last", type=int, metavar="N", help="Print win-rate/expectancy/DD metrics for the last N closed trades")
    parser.add_argument("--report", action="store_true", help="Print grouped performance report (overall/per-symbol/per-session/per-setup)")
    parser.add_argument("--ai-test", action="store_true", help="Run OpenAI connectivity self-test and exit")
    parser.add_argument("--train", action="store_true", help="Run training instead of trading")
    parser.add_argument("--backtest", action="store_true", help="Run a backtest report instead of trading")
    parser.add_argument("--preset", choices=["realistic", "frictionless"], default="realistic", help="Backtest execution preset")
    args = parser.parse_args()

    if args.bridge_only:
        raise SystemExit(run_bridge_only())
    if args.verify:
        raise SystemExit(run_verify())
    if args.eval_last is not None:
        raise SystemExit(run_eval_last(args.eval_last))
    if args.report:
        raise SystemExit(run_report())
    if args.ai_test:
        raise SystemExit(run_ai_test())
    if args.train:
        run_training()
        return
    if args.backtest:
        run_backtest(preset=args.preset)
        return
    if args.smoke_demo:
        summary = run_bot(once=False, max_loops=max(1, args.smoke_loops), smoke_demo=True, bridge_serve=args.bridge_serve)
        print("SMOKE SUMMARY:", json.dumps(summary, sort_keys=True))
        return
    run_bot(once=args.once, bridge_serve=args.bridge_serve, paper_sim=args.paper_sim)


if __name__ == "__main__":
    main()
