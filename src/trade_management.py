from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from src.bridge_stop_validation import SymbolRule
from src.lane_governor import derive_management_state
from src.utils import clamp


def _normalize_price(value: float, tick_size: float, digits: int) -> float:
    if tick_size <= 0:
        return round(float(value), max(0, int(digits)))
    ticks = round(float(value) / tick_size)
    normalized = ticks * tick_size
    return round(normalized, max(0, int(digits)))


def _almost_same(left: float | None, right: float | None, tick_size: float) -> bool:
    if left is None and right is None:
        return True
    if left is None or right is None:
        return False
    tolerance = max(abs(float(tick_size)) / 2.0, 1e-9)
    return abs(float(left) - float(right)) <= tolerance


def _directional_feature(value: Any, reference_price: float, side: str) -> float:
    try:
        raw = float(value)
    except (TypeError, ValueError):
        return 0.0
    normalized = raw
    if reference_price > 0 and abs(raw) > max(reference_price * 0.02, 5.0):
        normalized = raw / reference_price
    direction = 1.0 if side == "BUY" else -1.0
    return normalized * direction


_SUPER_AGGRESSIVE_SWING_SYMBOLS: set[str] = {
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
}

_ASIA_ATTACK_SYMBOLS: set[str] = {
    "AUDJPY",
    "NZDJPY",
    "USDJPY",
    "AUDNZD",
}

_WESTERN_ATTACK_SYMBOLS: set[str] = {
    "EURUSD",
    "GBPUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "NAS100",
    "USOIL",
}


def _dxy_support_score(symbol_key: str, runtime: dict[str, Any], side: str) -> float:
    explicit = runtime.get("dxy_support_score")
    if explicit is not None:
        return clamp(float(explicit or 0.5), 0.0, 1.0)
    ret_1 = _directional_feature(runtime.get("dxy_ret_1"), 1.0, side)
    ret_5 = _directional_feature(runtime.get("dxy_ret_5"), 1.0, side)
    dxy_move = 0.7 * ret_1 + 0.3 * ret_5
    if symbol_key in {"EURUSD", "GBPUSD", "EURGBP", "NAS100", "USOIL"}:
        dxy_move *= -1.0
    elif symbol_key in {"USDJPY"}:
        dxy_move *= 1.0
    else:
        return 0.5
    return clamp(0.5 + (dxy_move * 180.0), 0.0, 1.0)


def _watchlist_matches(symbol_key: str, setup_name: str, watch_items: list[str]) -> bool:
    setup_text = str(setup_name or "").upper()
    symbol_text = str(symbol_key or "").upper()
    for item in watch_items:
        token = str(item or "").upper().strip()
        if not token:
            continue
        if symbol_text and symbol_text in token:
            return True
        if setup_text and setup_text in token:
            return True
        if setup_text and any(part and part in token for part in setup_text.split("_")):
            return True
    return False


def _setup_type(setup_name: str) -> str:
    setup_text = str(setup_name or "").upper()
    if "GRID" in setup_text:
        return "grid_manage"
    if any(token in setup_text for token in ("DAY", "SET_FORGET", "H1", "H4", "SWING")):
        return "daytrade"
    return "scalp"


def _session_attack_bonus(symbol_key: str, session_name: str) -> float:
    session_key = str(session_name or "").upper()
    if symbol_key in _ASIA_ATTACK_SYMBOLS:
        if session_key in {"SYDNEY", "TOKYO"}:
            return 0.10
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            return 0.03
        return 0.0
    if symbol_key in _WESTERN_ATTACK_SYMBOLS:
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            return 0.10
        if session_key in {"SYDNEY", "TOKYO"}:
            return 0.02
        return 0.0
    return 0.04 if session_key in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.0


def _would_loosen_stop(*, side: str, current_sl: float, proposed_sl: float | None) -> bool:
    if proposed_sl is None or current_sl <= 0.0:
        return False
    if str(side or "").upper() == "BUY":
        return float(proposed_sl) + 1e-9 < float(current_sl)
    if str(side or "").upper() == "SELL":
        return float(proposed_sl) - 1e-9 > float(current_sl)
    return False


def _finalize_management_decision(
    decision: RetracementManagementDecision,
    *,
    payload: "RetracementManagementInput",
    side: str,
    protected_lock_enabled: bool = False,
) -> RetracementManagementDecision:
    management_state = derive_management_state(
        previous_state=str(payload.last_trade_state or ""),
        pnl_r=float(payload.pnl_r),
        continuation_score=float(decision.continuation_score),
        reversal_risk_score=float(decision.reversal_risk_score),
        decision_action=str(decision.management_action or ""),
    )
    previous_state = str(payload.last_trade_state or "").upper()
    hard_exit_reason = str(decision.reason or "") in {
        "hard_loss_cut",
        "time_stop_exit",
        "scratch_exit",
        "early_exit_thesis_invalidated",
        "reversal_risk_materially_high",
        "stall_exit",
        "xau_grid_fast_fail",
        "xau_grid_weak_launch_fail",
        "breakout_fast_fail",
    }
    if (
        protected_lock_enabled
        and previous_state in {"PROTECTED", "RUNNER", "EXIT"}
        and str(decision.management_action or "").upper() in {"TRAIL_STOP", "TIGHTEN_STOP", "EXTEND_TP"}
        and _would_loosen_stop(side=side, current_sl=float(payload.sl or 0.0), proposed_sl=decision.tighten_to_price)
        and not hard_exit_reason
    ):
        decision = replace(
            decision,
            management_action="HOLD",
            tighten_to_price=float(payload.sl or 0.0),
            reason="protected_state_no_loosen_hold",
        )
        management_state = derive_management_state(
            previous_state=previous_state,
            pnl_r=float(payload.pnl_r),
            continuation_score=float(decision.continuation_score),
            reversal_risk_score=float(decision.reversal_risk_score),
            decision_action=str(decision.management_action or ""),
        )
    details = {
        **dict(decision.details or {}),
        "management_state": management_state,
        "previous_management_state": previous_state or "INIT",
        "protected_no_loosen_enabled": bool(protected_lock_enabled),
    }
    return replace(decision, details=details)


@dataclass(frozen=True)
class BrokerSafeModifyInput:
    symbol: str
    side: str
    current_bid: float
    current_ask: float
    desired_sl: float = 0.0
    desired_tp: float = 0.0
    current_sl: float = 0.0
    current_tp: float = 0.0
    safety_buffer_points: int = 0
    allow_tp_none: bool = True


@dataclass(frozen=True)
class BrokerSafeModifyResult:
    valid: bool
    actionable: bool
    symbol_key: str
    normalized_sl: float | None
    normalized_tp: float | None
    reason: str
    clamp_reasons: tuple[str, ...] = ()
    bid: float = 0.0
    ask: float = 0.0
    reference_price: float = 0.0
    min_stop_distance_points: float = 0.0
    freeze_distance_points: float = 0.0
    effective_distance_points: float = 0.0


@dataclass(frozen=True)
class RetracementManagementInput:
    symbol: str
    side: str
    entry_price: float
    current_price: float
    sl: float
    tp: float
    pnl_r: float
    age_minutes: float
    spread_points: float
    typical_spread_points: float
    volume: float
    min_lot: float
    point_size: float = 0.0
    ai_decision: str = ""
    ai_confidence: float = 0.0
    runtime_features: dict[str, Any] = field(default_factory=dict)
    time_stop_minutes: int = 120
    setup: str = ""
    strategy_family: str = ""
    timeframe: str = ""
    session_name: str = ""
    bars_in_trade: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0
    last_trade_state: str = "INIT"
    spread_recovery_partial_taken: bool = False
    runner_partial_taken: bool = False
    event_risk_active: bool = False
    enable_stall_exit: bool = True
    enable_adaptive_tp: bool = True
    enable_dynamic_trail: bool = True
    learning_brain_bundle: dict[str, Any] = field(default_factory=dict)
    weekly_reentry_watchlist: list[str] = field(default_factory=list)
    proven_reentry_queue: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetracementManagementDecision:
    management_action: str
    retracement_exit_score: float
    continuation_score: float
    reversal_risk_score: float
    protection_mode: str
    tighten_to_price: float | None = None
    updated_tp_price: float | None = None
    close_fraction: float = 0.0
    trade_state: str = "INIT"
    trade_quality: str = "ACCEPTABLE"
    stall_detected: bool = False
    profit_lock_r: float = 0.0
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)


_MANAGEMENT_PROFILES: dict[str, dict[str, float]] = {
    "TREND": {
        "min_profit_r": 0.35,
        "min_age_minutes": 4.0,
        "protect_confirm_r": 0.40,
        "runner_start_r": 0.95,
        "trail_backoff_r": 0.55,
        "stall_min_profit_r": 0.45,
        "stall_bars": 6.0,
        "time_stop_factor": 1.0,
        "tp_extend_progress": 0.76,
        "tp_extend_r": 0.35,
        "scratch_loss_r": 0.24,
    },
    "RANGE/REVERSION": {
        "min_profit_r": 0.30,
        "min_age_minutes": 3.0,
        "protect_confirm_r": 0.38,
        "runner_start_r": 0.78,
        "trail_backoff_r": 0.38,
        "stall_min_profit_r": 0.34,
        "stall_bars": 4.0,
        "time_stop_factor": 0.8,
        "tp_extend_progress": 0.88,
        "tp_extend_r": 0.18,
        "scratch_loss_r": 0.18,
    },
    "GRID": {
        "min_profit_r": 0.30,
        "min_age_minutes": 2.0,
        "protect_confirm_r": 0.40,
        "runner_start_r": 0.80,
        "trail_backoff_r": 0.32,
        "stall_min_profit_r": 0.35,
        "stall_bars": 3.0,
        "time_stop_factor": 0.65,
        "tp_extend_progress": 0.90,
        "tp_extend_r": 0.12,
        "scratch_loss_r": 0.18,
    },
    "CRYPTO MOMENTUM": {
        "min_profit_r": 0.35,
        "min_age_minutes": 2.0,
        "protect_confirm_r": 0.50,
        "runner_start_r": 1.20,
        "trail_backoff_r": 0.85,
        "stall_min_profit_r": 0.50,
        "stall_bars": 5.0,
        "time_stop_factor": 1.15,
        "tp_extend_progress": 0.80,
        "tp_extend_r": 0.45,
        "scratch_loss_r": 0.35,
    },
}


def _infer_strategy_family(payload: RetracementManagementInput, symbol_key: str) -> str:
    family = str(payload.strategy_family or "").strip().upper()
    if family in _MANAGEMENT_PROFILES:
        return family
    setup_text = str(payload.setup or "").upper()
    classifier_text = f"{family} {setup_text}".strip()
    if "GRID" in classifier_text:
        return "GRID"
    if symbol_key == "BTCUSD" or any(token in classifier_text for token in ("BTC", "CRYPTO")):
        return "CRYPTO MOMENTUM"
    if any(token in classifier_text for token in ("RANGE", "REVERSION", "FADE", "MEAN", "VWAP")):
        return "RANGE/REVERSION"
    return "TREND"


def _profile_overrides(
    payload: RetracementManagementInput,
    *,
    symbol_key: str,
    family: str,
) -> dict[str, float]:
    setup_text = f"{str(payload.strategy_family or '').upper()} {str(payload.setup or '').upper()}".strip()
    runtime = dict(payload.runtime_features or {})
    session_key = str(payload.session_name or "").upper()
    regime_key = str(runtime.get("regime_state") or runtime.get("regime") or "").upper()
    lane_name = str(runtime.get("lane_name") or runtime.get("entry_lane_name") or "").upper()
    execution_tf = str(payload.timeframe or runtime.get("execution_timeframe_used") or runtime.get("timeframe") or "").upper()
    overrides: dict[str, float] = {}
    if symbol_key == "XAUUSD" and "GRID" in setup_text:
        overrides.update(
            {
                "min_profit_r": 0.22,
                "protect_confirm_r": 0.28,
                "runner_start_r": 0.72,
                "trail_backoff_r": 0.24,
                "stall_min_profit_r": 0.24,
                "time_stop_factor": 0.45,
                "tp_extend_progress": 0.72,
                "tp_extend_r": 0.32,
                "scratch_loss_r": 0.14,
            }
        )
    elif symbol_key == "XAUUSD" and "LONDON_LIQUIDITY_SWEEP" in setup_text:
        overrides.update(
            {
                "min_profit_r": 0.50,
                "protect_confirm_r": 0.58,
                "runner_start_r": 1.05,
                "trail_backoff_r": 0.58,
                "stall_min_profit_r": 0.56,
                "tp_extend_progress": 0.84,
                "tp_extend_r": 0.22,
                "scratch_loss_r": 0.22,
            }
        )
        if session_key in {"LONDON", "NEW_YORK"} and regime_key == "MEAN_REVERSION":
            overrides.update(
                {
                    "min_profit_r": 0.38,
                    "protect_confirm_r": 0.46,
                    "runner_start_r": 0.88,
                    "trail_backoff_r": 0.50,
                    "stall_min_profit_r": 0.42,
                    "stall_bars": 4.0,
                    "tp_extend_progress": 0.90,
                    "tp_extend_r": 0.14,
                    "scratch_loss_r": 0.18,
                }
            )
    elif symbol_key == "GBPUSD" and "LONDON_EXPANSION_BREAKOUT" in setup_text:
        overrides.update(
            {
                "min_profit_r": 0.48,
                "protect_confirm_r": 0.56,
                "runner_start_r": 1.12,
                "trail_backoff_r": 0.62,
                "stall_min_profit_r": 0.54,
                "tp_extend_progress": 0.84,
                "tp_extend_r": 0.25,
                "scratch_loss_r": 0.22,
            }
        )
        if session_key == "LONDON" and regime_key == "LOW_LIQUIDITY_CHOP":
            overrides.update(
                {
                    "min_profit_r": 0.40,
                    "protect_confirm_r": 0.46,
                    "runner_start_r": 0.92,
                    "trail_backoff_r": 0.50,
                    "stall_min_profit_r": 0.42,
                    "stall_bars": 4.0,
                    "tp_extend_progress": 0.88,
                    "tp_extend_r": 0.16,
                    "scratch_loss_r": 0.18,
                }
            )
    elif symbol_key == "BTCUSD" and any(token in setup_text for token in ("PRICE_ACTION_CONTINUATION", "VOLATILE_RETEST")):
        overrides.update(
            {
                "min_profit_r": 0.48,
                "protect_confirm_r": 0.62,
                "runner_start_r": 1.35,
                "trail_backoff_r": 0.92,
                "stall_min_profit_r": 0.72,
                "tp_extend_progress": 0.82,
                "tp_extend_r": 0.38,
                "scratch_loss_r": 0.30,
            }
        )
    elif symbol_key == "USOIL" and "TREND_EXPANSION" in setup_text:
        overrides.update(
            {
                "min_profit_r": 0.45,
                "protect_confirm_r": 0.60,
                "runner_start_r": 1.22,
                "trail_backoff_r": 0.74,
                "stall_min_profit_r": 0.60,
                "tp_extend_progress": 0.80,
                "tp_extend_r": 0.24,
                "scratch_loss_r": 0.24,
            }
        )
    elif symbol_key == "EURUSD" and "VWAP_PULLBACK" in setup_text:
        overrides.update(
            {
                "min_profit_r": 0.40,
                "protect_confirm_r": 0.55,
                "runner_start_r": 1.12,
                "trail_backoff_r": 0.58,
                "stall_min_profit_r": 0.52,
                "tp_extend_progress": 0.86,
                "tp_extend_r": 0.20,
            }
        )
    elif symbol_key == "AUDJPY" and "LIQUIDITY_SWEEP_REVERSAL" in setup_text:
        overrides.update(
            {
                "min_profit_r": 0.36,
                "protect_confirm_r": 0.42,
                "runner_start_r": 0.82,
                "trail_backoff_r": 0.46,
                "stall_min_profit_r": 0.40,
                "stall_bars": 4.0,
                "tp_extend_progress": 0.90,
                "tp_extend_r": 0.14,
                "scratch_loss_r": 0.16,
            }
        )
        if session_key == "OVERLAP" and regime_key == "MEAN_REVERSION":
            overrides.update(
                {
                    "min_profit_r": 0.38,
                    "protect_confirm_r": 0.46,
                    "runner_start_r": 0.84,
                    "trail_backoff_r": 0.45,
                    "stall_min_profit_r": 0.42,
                    "stall_bars": 5.0,
                }
            )
        elif session_key == "SYDNEY" and regime_key == "MEAN_REVERSION":
            overrides.update(
                {
                    "min_profit_r": 0.30,
                    "protect_confirm_r": 0.36,
                    "runner_start_r": 0.70,
                    "trail_backoff_r": 0.40,
                    "stall_bars": 3.0,
                    "scratch_loss_r": 0.14,
                }
            )
        elif session_key == "TOKYO":
            overrides.update(
                {
                    "min_profit_r": 0.28,
                    "protect_confirm_r": 0.34,
                    "runner_start_r": 0.66,
                    "trail_backoff_r": 0.36,
                    "stall_min_profit_r": 0.30,
                    "stall_bars": 3.0,
                    "time_stop_factor": 0.38,
                    "tp_extend_progress": 0.82,
                    "tp_extend_r": 0.12,
                    "scratch_loss_r": 0.12,
                }
            )
    elif symbol_key == "NZDJPY" and "LIQUIDITY_TRAP_REVERSAL" in setup_text:
        overrides.update(
            {
                "min_profit_r": 0.30,
                "protect_confirm_r": 0.36,
                "runner_start_r": 0.68,
                "trail_backoff_r": 0.38,
                "stall_min_profit_r": 0.32,
                "stall_bars": 3.0,
                "time_stop_factor": 0.40,
                "tp_extend_progress": 0.84,
                "tp_extend_r": 0.12,
                "scratch_loss_r": 0.12,
            }
        )
    elif symbol_key in {"AUDJPY", "NZDJPY", "USDJPY"} and any(
        token in setup_text for token in ("CONTINUATION", "TREND", "BREAKOUT", "PULLBACK", "CARRY")
    ):
        overrides.update(
            {
                "min_profit_r": 0.42,
                "protect_confirm_r": 0.58,
                "runner_start_r": 1.18,
                "trail_backoff_r": 0.68,
                "stall_min_profit_r": 0.58,
                "tp_extend_progress": 0.82,
                "tp_extend_r": 0.24,
                "scratch_loss_r": 0.22,
            }
        )
    elif symbol_key == "AUDNZD" and any(token in setup_text for token in ("ROTATION", "REVERSION", "RANGE")):
        overrides.update(
            {
                "min_profit_r": 0.36,
                "protect_confirm_r": 0.46,
                "runner_start_r": 0.98,
                "trail_backoff_r": 0.48,
                "stall_min_profit_r": 0.42,
            }
        )
    elif symbol_key == "AUDNZD" and "COMPRESSION_RELEASE" in setup_text:
        overrides.update(
            {
                "min_profit_r": 0.28,
                "protect_confirm_r": 0.34,
                "runner_start_r": 0.66,
                "trail_backoff_r": 0.34,
                "stall_min_profit_r": 0.30,
                "stall_bars": 3.0,
                "time_stop_factor": 0.42,
                "tp_extend_progress": 0.82,
                "tp_extend_r": 0.12,
                "scratch_loss_r": 0.11,
            }
        )
    elif symbol_key == "NAS100" and any(token in setup_text for token in ("SWEEP", "OPENING_DRIVE")):
        overrides.update(
            {
                "min_profit_r": 0.42,
                "protect_confirm_r": 0.54,
                "runner_start_r": 1.08,
                "trail_backoff_r": 0.60,
                "stall_min_profit_r": 0.50,
                "tp_extend_progress": 0.78,
                "tp_extend_r": 0.20,
                "scratch_loss_r": 0.22,
            }
        )
    elif family == "TREND" and symbol_key not in {"GBPUSD"} and any(
        token in setup_text for token in ("BREAKOUT", "IMPULSE", "CONTINUATION", "TREND", "PULLBACK")
    ):
        overrides.update(
            {
                "protect_confirm_r": max(overrides.get("protect_confirm_r", 0.0), 0.54),
                "runner_start_r": max(overrides.get("runner_start_r", 0.0), 1.10),
                "trail_backoff_r": max(overrides.get("trail_backoff_r", 0.0), 0.64),
            }
        )
    fast_execution_mode = execution_tf in {"M1", "M3"} or str(runtime.get("fast_execution_profile") or "").upper() == "M3_ATTACK"
    asia_fast_lane = symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD"} and session_key in {"SYDNEY", "TOKYO"}
    fast_lane_symbol = symbol_key in {"XAUUSD", "NAS100", "BTCUSD", "USDJPY"} or asia_fast_lane
    fast_lane_active = fast_execution_mode or lane_name.startswith("XAU_") or asia_fast_lane
    if fast_lane_symbol and fast_lane_active:
        if asia_fast_lane:
            fast_profile = {
                "min_profit_r": 0.22,
                "protect_confirm_r": 0.26,
                "runner_start_r": 0.60,
                "trail_backoff_r": 0.26,
                "stall_min_profit_r": 0.22,
                "time_stop_factor": 0.34,
                "stall_bars": 2.0,
                "tp_extend_progress": 0.68,
                "tp_extend_r": 0.14,
                "scratch_loss_r": 0.10,
            }
        else:
            fast_profile = {
                "min_profit_r": 0.20 if symbol_key == "XAUUSD" else 0.24,
                "protect_confirm_r": 0.24 if symbol_key == "XAUUSD" else 0.28,
                "runner_start_r": 0.58 if symbol_key == "XAUUSD" else 0.72,
                "trail_backoff_r": 0.20 if symbol_key == "XAUUSD" else 0.26,
                "stall_min_profit_r": 0.20 if symbol_key == "XAUUSD" else 0.24,
                "time_stop_factor": 0.28 if symbol_key == "XAUUSD" else 0.36,
                "stall_bars": 2.0 if symbol_key == "XAUUSD" else 3.0,
                "tp_extend_progress": 0.66 if symbol_key == "XAUUSD" else 0.70,
                "tp_extend_r": 0.18,
                "scratch_loss_r": 0.10 if symbol_key == "XAUUSD" else 0.14,
            }
        if lane_name.endswith("_REENTRY"):
            fast_profile.update(
                {
                    "runner_start_r": 0.52 if symbol_key == "XAUUSD" else (0.58 if asia_fast_lane else 0.66),
                    "time_stop_factor": 0.24 if symbol_key == "XAUUSD" else (0.30 if asia_fast_lane else 0.32),
                    "scratch_loss_r": 0.09 if symbol_key == "XAUUSD" else (0.09 if asia_fast_lane else 0.12),
                }
            )
        elif lane_name.endswith("_BREAKOUT"):
            fast_profile.update(
                {
                    "runner_start_r": 0.64 if symbol_key == "XAUUSD" else (0.66 if asia_fast_lane else 0.78),
                    "tp_extend_progress": 0.62 if symbol_key == "XAUUSD" else (0.66 if asia_fast_lane else 0.68),
                }
            )
        elif lane_name.endswith("_RECLAIM"):
            fast_profile.update(
                {
                    "protect_confirm_r": 0.22 if symbol_key == "XAUUSD" else (0.24 if asia_fast_lane else 0.26),
                    "runner_start_r": 0.56 if symbol_key == "XAUUSD" else (0.60 if asia_fast_lane else 0.70),
                }
            )
        for key, value in fast_profile.items():
            current = overrides.get(key)
            if key == "stall_bars":
                overrides[key] = min(float(current), float(value)) if current is not None else float(value)
            elif key == "tp_extend_r":
                overrides[key] = min(float(current), float(value)) if current is not None else float(value)
            else:
                overrides[key] = min(float(current), float(value)) if current is not None else float(value)
    return overrides


def _timeframe_minutes(timeframe: str) -> int:
    text = str(timeframe or "").strip().upper()
    if text.startswith("M") and text[1:].isdigit():
        return max(1, int(text[1:]))
    if text.startswith("H") and text[1:].isdigit():
        return max(1, int(text[1:]) * 60)
    return 5


def _protect_price(entry_price: float, risk_distance: float, side: str, lock_r: float) -> float:
    if side == "BUY":
        return float(entry_price) + (risk_distance * max(0.0, float(lock_r)))
    return float(entry_price) - (risk_distance * max(0.0, float(lock_r)))


def _rewrite_no_partial_exit(
    decision: RetracementManagementDecision,
    *,
    payload: RetracementManagementInput,
    risk_distance: float,
    side: str,
    details: dict[str, Any],
) -> RetracementManagementDecision:
    reason_text = str(decision.reason or "").strip()
    rewritten_reason = f"{reason_text}_no_partial_support" if reason_text else "no_partial_support"
    rewritten_details = {
        **dict(details),
        "original_management_action": str(decision.management_action or ""),
        "original_protection_mode": str(decision.protection_mode or ""),
        "partial_close_rewritten": True,
    }
    full_exit_reasons = {
        "stall_exit",
        "profit_decent_continuation_weak",
        "trend_partial_capture_on_weakening_continuation",
        "trend_partial_capture_on_degrading_extension",
    }
    if reason_text in full_exit_reasons or float(decision.continuation_score) < 0.38:
        return replace(
            decision,
            management_action="FULL_EXIT",
            protection_mode="no_partial_full_exit",
            trade_state="FORCE_EXIT",
            trade_quality=decision.trade_quality if decision.trade_quality != "ACCEPTABLE" else "DEGRADING",
            close_fraction=0.0,
            reason=rewritten_reason,
            details=rewritten_details,
        )
    profit_lock_r = max(
        float(decision.profit_lock_r),
        max(0.06, float(payload.pnl_r) - (0.18 if float(payload.pnl_r) < 0.70 else 0.26)),
    )
    tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
    return replace(
        decision,
        management_action="TIGHTEN_STOP",
        protection_mode="no_partial_profit_lock",
        tighten_to_price=tighten_to,
        close_fraction=0.0,
        trade_state="PROTECTED" if float(payload.pnl_r) < 0.90 else "RUNNER",
        profit_lock_r=profit_lock_r,
        reason=rewritten_reason,
        details=rewritten_details,
    )


def _quality_bucket(continuation_score: float, reversal_risk_score: float, stall_detected: bool) -> str:
    if reversal_risk_score >= 0.82 or continuation_score <= 0.26:
        return "DEAD"
    if stall_detected or reversal_risk_score >= 0.62 or continuation_score <= 0.44:
        return "DEGRADING"
    if continuation_score >= 0.68 and reversal_risk_score <= 0.42:
        return "STRONG"
    return "ACCEPTABLE"


def _swing_continuation_score(
    payload: RetracementManagementInput,
    *,
    side: str,
    runtime: dict[str, Any],
    continuation_score: float,
) -> float:
    symbol_key = str(payload.symbol or "").upper()
    reference_price = max(float(payload.current_price or 0.0), float(payload.entry_price or 0.0), 1e-9)
    directional_values = [
        _directional_feature(runtime.get("h1_ret_1"), reference_price, side),
        _directional_feature(runtime.get("h1_ret_3"), reference_price, side),
        _directional_feature(runtime.get("h4_ret_1"), reference_price, side),
        _directional_feature(runtime.get("h1_momentum_3"), reference_price, side),
        _directional_feature(runtime.get("h1_slope"), reference_price, side),
        _directional_feature(runtime.get("h4_slope"), reference_price, side),
    ]
    directional_values = [value for value in directional_values if value != 0.0]
    directional_score = clamp(0.5 + (((sum(directional_values) / len(directional_values)) if directional_values else 0.0) * 220.0), 0.0, 1.0)
    multi_tf_alignment = clamp(float(runtime.get("multi_tf_alignment_score", 0.5) or 0.5), 0.0, 1.0)
    h1_efficiency = clamp(float(runtime.get("h1_trend_efficiency_16", runtime.get("h1_trend_efficiency_32", 0.5)) or 0.5), 0.0, 1.0)
    h4_efficiency = clamp(float(runtime.get("h4_trend_efficiency_16", runtime.get("h4_trend_efficiency_32", 0.5)) or 0.5), 0.0, 1.0)
    structure_cleanliness = clamp(
        float(
            runtime.get("structure_cleanliness_score")
            or runtime.get("trend_efficiency_score")
            or runtime.get("pair_behavior_fit")
            or 0.5
        ),
        0.0,
        1.0,
    )
    dxy_support = _dxy_support_score(symbol_key, runtime, side)
    fractal_persistence = clamp(float(runtime.get("fractal_persistence_score", 0.5) or 0.5), 0.0, 1.0)
    compression_expansion = clamp(
        float(runtime.get("compression_expansion_score", 1.0 - float(runtime.get("m5_atr_pct_of_avg", 1.0) or 1.0)) or 0.0),
        0.0,
        1.0,
    )
    mc_win_rate = clamp(float(runtime.get("mc_win_rate", 0.5) or 0.5), 0.0, 1.0)
    transition_momentum = clamp(float(runtime.get("transition_momentum", 0.0) or 0.0), 0.0, 1.0)
    super_aggro_symbol = symbol_key in _SUPER_AGGRESSIVE_SWING_SYMBOLS
    return clamp(
        (0.24 * continuation_score)
        + (0.20 * multi_tf_alignment)
        + (0.14 * directional_score)
        + (0.10 * h1_efficiency)
        + (0.06 * h4_efficiency)
        + (0.04 * structure_cleanliness)
        + (0.08 * fractal_persistence)
        + (0.06 * compression_expansion)
        + (0.04 * transition_momentum)
        + (
            (0.06 * dxy_support) + (0.08 * mc_win_rate)
            if super_aggro_symbol
            else (0.04 * dxy_support) + (0.04 * mc_win_rate)
        ),
        0.0,
        1.0,
    )


def build_local_trade_plan(context: dict[str, Any]) -> dict[str, Any]:
    symbol_key = str(context.get("symbol", "")).upper()
    setup_name = str(context.get("setup", "")).upper()
    side = str(context.get("side", "BUY")).upper()
    session_name = str(context.get("session", "")).upper()
    setup_type = _setup_type(setup_name)
    probability = clamp(float(context.get("probability", 0.5) or 0.5), 0.0, 1.0)
    expected_value_r = float(context.get("expected_value_r", 0.0) or 0.0)
    spread_points = max(0.0, float(context.get("spread_points", 0.0) or 0.0))
    point_size = max(1e-9, float(context.get("point_size", 0.0001) or 0.0001))
    min_stop_points = max(1.0, float(context.get("min_stop_points", 10.0) or 10.0))
    atr_price = max(point_size, float(context.get("atr_price", point_size * 40.0) or (point_size * 40.0)))
    atr_points = max(1.0, atr_price / point_size)
    learning_bundle = dict(context.get("learning_brain_bundle") or {})
    pair_directive = dict((learning_bundle.get("pair_directives") or {}).get(symbol_key) or {})
    meeting_packet = dict(learning_bundle.get("meeting_packet") or {})
    hour_expectancy_matrix = dict(meeting_packet.get("hour_expectancy_matrix") or {})
    setup_hour_directive = dict(hour_expectancy_matrix.get(f"{symbol_key}:{setup_name}") or {})
    weekly_watchlist = [str(item) for item in context.get("weekly_reentry_watchlist", []) if str(item).strip()]
    goal_state = dict(learning_bundle.get("goal_state") or {})
    risk_reduction_active = bool(learning_bundle.get("risk_reduction_active", False))
    aggressive_goal_chasing = bool(goal_state.get("lagging_short_goal")) and not risk_reduction_active
    aggression_multiplier = clamp(float(pair_directive.get("aggression_multiplier", 1.0) or 1.0), 0.75, 1.50)
    min_confluence_override = max(0.0, float(pair_directive.get("min_confluence_override", 0.0) or 0.0))
    trade_horizon_bias = str(pair_directive.get("trade_horizon_bias") or "").lower()
    reentry_priority = clamp(float(pair_directive.get("reentry_priority", 0.0) or 0.0), 0.0, 1.0)
    slippage_regime = str(pair_directive.get("slippage_regime") or "").lower()
    shadow_experiment_active = bool(pair_directive.get("shadow_experiment_active", False))
    opportunity_capture_gap_r = max(0.0, float(pair_directive.get("opportunity_capture_gap_r", 0.0) or 0.0))
    management_quality_score = clamp(float(pair_directive.get("management_quality_score", 0.0) or 0.0), 0.0, 1.0)
    hour_expectancy_score = clamp(
        float(setup_hour_directive.get("hour_expectancy_score", pair_directive.get("hour_expectancy_score", 0.5)) or 0.5),
        0.0,
        1.0,
    )
    lane_expectancy_multiplier = clamp(
        float(setup_hour_directive.get("lane_expectancy_multiplier", pair_directive.get("lane_expectancy_multiplier", 1.0)) or 1.0),
        0.80,
        1.35,
    )
    lane_mfe_median_r = max(
        0.0,
        float(setup_hour_directive.get("lane_mfe_median_r", pair_directive.get("lane_mfe_median_r", 0.0)) or 0.0),
    )
    lane_mae_median_r = max(
        0.0,
        float(setup_hour_directive.get("lane_mae_median_r", pair_directive.get("lane_mae_median_r", 0.0)) or 0.0),
    )
    lane_capture_efficiency = clamp(
        float(setup_hour_directive.get("lane_capture_efficiency", pair_directive.get("lane_capture_efficiency", 0.5)) or 0.5),
        0.0,
        1.0,
    )
    management_directives = dict(pair_directive.get("management_directives") or {})
    confluence_score = max(0.0, float(context.get("confluence_score", 0.0) or 0.0))

    mc_win_rate = clamp(
        float(
            context.get("mc_win_rate", learning_bundle.get("monte_carlo_pass_floor", 0.82))
            or learning_bundle.get("monte_carlo_pass_floor", 0.82)
        ),
        0.0,
        1.0,
    )
    fractal_persistence = clamp(float(context.get("fractal_persistence_score", 0.5) or 0.5), 0.0, 1.0)
    multi_tf_alignment = clamp(float(context.get("multi_tf_alignment_score", 0.5) or 0.5), 0.0, 1.0)
    dxy_support = clamp(float(context.get("dxy_support_score", 0.5) or 0.5), 0.0, 1.0)
    compression_expansion = clamp(float(context.get("compression_expansion_score", 0.5) or 0.5), 0.0, 1.0)
    trajectory_catchup_pressure = clamp(
        float(
            context.get(
                "trajectory_catchup_pressure",
                learning_bundle.get("quota_catchup_pressure", 0.0),
            )
            or learning_bundle.get("quota_catchup_pressure", 0.0)
        ),
        0.0,
        1.0,
    )
    promoted_patterns = {
        str(item).upper()
        for item in learning_bundle.get("promoted_patterns", [])
        if str(item).strip()
    }
    weak_pair_focus = {
        str(item).upper()
        for item in learning_bundle.get("weak_pair_focus", [])
        if str(item).strip()
    }
    watchlist_hit = _watchlist_matches(symbol_key, setup_name, weekly_watchlist)
    promoted_pattern = setup_name in promoted_patterns or symbol_key in promoted_patterns
    weak_pair = symbol_key in weak_pair_focus
    session_bonus = _session_attack_bonus(symbol_key, session_name)
    spread_penalty = clamp(spread_points / max(min_stop_points * 1.5, 1.0), 0.0, 1.0)
    quality_score = clamp(
        (0.28 * probability)
        + (0.16 * clamp(expected_value_r / 1.5, -1.0, 1.0))
        + (0.16 * mc_win_rate)
        + (0.10 * clamp(confluence_score / 5.0, 0.0, 1.0))
        + (0.12 * multi_tf_alignment)
        + (0.10 * fractal_persistence)
        + (0.08 * dxy_support)
        + (0.06 * compression_expansion)
        + session_bonus
        + (0.05 if promoted_pattern else 0.0)
        + (0.04 if watchlist_hit else 0.0)
        + min(0.05, max(0.0, aggression_multiplier - 1.0) * 0.08)
        + min(0.04, max(0.0, lane_expectancy_multiplier - 1.0) * 0.12)
        + (0.03 * reentry_priority if watchlist_hit else 0.0)
        + (0.04 * max(0.0, hour_expectancy_score - 0.5))
        + (0.03 * max(0.0, lane_capture_efficiency - 0.5))
        - (0.10 if weak_pair else 0.0)
        - (0.12 * spread_penalty),
        0.0,
        1.0,
    )

    mc_floor = max(
        0.78 if trajectory_catchup_pressure >= 0.60 else 0.82,
        float(learning_bundle.get("monte_carlo_pass_floor", 0.82) or 0.82)
        - (0.02 if trajectory_catchup_pressure >= 0.75 and promoted_pattern else 0.0),
    )
    min_probability = max(0.46, float(context.get("min_probability", 0.56) or 0.56))
    probability_floor = clamp(
        min_probability
        - (0.03 if promoted_pattern else 0.0)
        - (0.02 if watchlist_hit else 0.0)
        - (0.02 if trajectory_catchup_pressure >= 0.65 and session_bonus > 0.0 else 0.0)
        - (0.02 if aggression_multiplier >= 1.10 and (promoted_pattern or watchlist_hit) else 0.0)
        - (0.01 if aggressive_goal_chasing and promoted_pattern else 0.0)
        + (0.03 if weak_pair else 0.0),
        0.42,
        0.82,
    )
    if risk_reduction_active:
        probability_floor = clamp(probability_floor + 0.03, 0.42, 0.86)
        mc_floor = clamp(mc_floor + 0.01, 0.5, 0.95)
    if setup_hour_directive:
        if hour_expectancy_score <= 0.42:
            probability_floor = clamp(probability_floor + 0.02, 0.42, 0.90)
            min_confluence_override = max(min_confluence_override, 3.8)
        elif hour_expectancy_score >= 0.68:
            probability_floor = clamp(probability_floor - 0.01, 0.42, 0.90)
    if min_confluence_override > 0.0 and confluence_score < min_confluence_override and not promoted_pattern:
        probability_floor = clamp(probability_floor + 0.04, 0.42, 0.88)
    ev_floor = -0.05 if trajectory_catchup_pressure >= 0.80 and promoted_pattern else 0.0
    decision = "TAKE" if (quality_score >= probability_floor and mc_win_rate >= mc_floor and expected_value_r >= ev_floor) else "PASS"

    base_rr = 1.35 if setup_type == "scalp" else 2.60 if setup_type == "daytrade" else 1.20
    if trade_horizon_bias == "daytrade" and setup_type == "scalp":
        base_rr += 0.10
    elif trade_horizon_bias == "swing":
        base_rr += 0.20
    if slippage_regime == "clean":
        base_rr += 0.05
    elif slippage_regime == "rough":
        base_rr -= 0.08
    if shadow_experiment_active and not weak_pair:
        base_rr += 0.04
    rr_bonus = (
        (0.30 if setup_type == "scalp" else 0.45 if setup_type == "daytrade" else 0.10)
        if quality_score >= 0.78 and mc_win_rate >= 0.85 and multi_tf_alignment >= 0.62
        else 0.0
    )
    if opportunity_capture_gap_r >= 0.60 and management_quality_score <= 0.55:
        rr_bonus += 0.08
    runner_lane = bool(lane_mfe_median_r >= 1.75 and lane_capture_efficiency >= 0.58 and lane_expectancy_multiplier >= 1.02)
    stall_lane = bool(
        (lane_mfe_median_r > 0.0 and lane_mfe_median_r <= 1.05)
        or (lane_mfe_median_r > 0.0 and lane_capture_efficiency <= 0.48)
    )
    if runner_lane:
        rr_bonus += 0.12 if setup_type in {"scalp", "daytrade"} else 0.06
    elif stall_lane:
        rr_bonus -= 0.08
    rr_target = clamp(base_rr + rr_bonus, 1.15, 3.20)
    stop_mult = (
        0.95 if setup_type == "scalp" else 1.20 if setup_type == "daytrade" else 0.85
    )
    if trajectory_catchup_pressure >= 0.60 and promoted_pattern:
        stop_mult = max(0.80, stop_mult - 0.08)
    sl_points = max(min_stop_points, atr_points * stop_mult)
    tp_points = max(min_stop_points, sl_points * rr_target)

    elite_daytrade_attack = (
        setup_type == "daytrade"
        and probability >= max(probability_floor, 0.68)
        and expected_value_r >= max(ev_floor, 0.30)
        and mc_win_rate >= max(mc_floor - 0.02, 0.80)
        and quality_score >= 0.62
        and multi_tf_alignment >= 0.48
        and fractal_persistence >= 0.45
    )
    if elite_daytrade_attack or (
        quality_score >= 0.70 and mc_win_rate >= 0.86 and (promoted_pattern or session_bonus >= 0.08 or watchlist_hit)
    ):
        risk_tier = "HIGH"
    elif quality_score >= 0.70:
        risk_tier = "NORMAL"
    else:
        risk_tier = "LOW"
    if risk_reduction_active and risk_tier == "HIGH":
        risk_tier = "NORMAL"

    move_to_be_r = max(0.5, min(0.9, 0.50 + (spread_penalty * 0.15)))
    trail_after_r = 0.50 if setup_type == "scalp" else 0.85 if setup_type == "daytrade" else 0.45
    trail_value = 0.90 if setup_type == "scalp" else 1.20 if setup_type == "daytrade" else 0.82
    time_stop_minutes = 55 if setup_type == "scalp" else 240 if setup_type == "daytrade" else 90
    if slippage_regime == "clean" and (shadow_experiment_active or aggressive_goal_chasing):
        trail_after_r = max(0.40, trail_after_r - 0.05)
        trail_value = max(0.75, trail_value - 0.04)
    elif slippage_regime == "rough":
        move_to_be_r = max(0.35, move_to_be_r - 0.08)
        time_stop_minutes = int(max(25, time_stop_minutes * 0.85))
    if trade_horizon_bias == "swing":
        time_stop_minutes = max(time_stop_minutes, 360)
    elif trade_horizon_bias == "daytrade" and setup_type == "scalp":
        time_stop_minutes = max(time_stop_minutes, 120)
    if session_bonus > 0.08 and setup_type == "scalp":
        time_stop_minutes = 40
    if runner_lane:
        move_to_be_r = min(0.95, move_to_be_r + 0.05)
        trail_after_r = min(1.10, trail_after_r + 0.05)
        trail_value = min(1.35, trail_value + 0.06)
        time_stop_minutes = int(max(time_stop_minutes, time_stop_minutes * 1.20))
    elif stall_lane:
        move_to_be_r = max(0.35, move_to_be_r - 0.08)
        trail_after_r = max(0.35, trail_after_r - 0.05)
        trail_value = max(0.72, trail_value - 0.06)
        time_stop_minutes = int(max(25, time_stop_minutes * 0.80))
    if management_directives:
        move_to_be_r = min(
            move_to_be_r,
            clamp(float(management_directives.get("early_protect_r", move_to_be_r) or move_to_be_r), 0.15, 0.95),
        )
        trail_after_r = min(
            trail_after_r,
            clamp(float(management_directives.get("early_protect_r", trail_after_r) or trail_after_r) + 0.06, 0.20, 1.20),
        )
        trail_backoff_r = clamp(float(management_directives.get("trail_backoff_r", 0.36) or 0.36), 0.18, 0.90)
        trail_value = clamp(0.50 + trail_backoff_r, 0.68, 1.40)
        if float(management_directives.get("stall_exit_bias", 0.0) or 0.0) >= 0.55:
            time_stop_minutes = int(max(20, time_stop_minutes * 0.80))
        if float(management_directives.get("tp_extension_bias", 0.0) or 0.0) >= 0.25:
            rr_target = clamp(rr_target + 0.08, 1.15, 3.30)

    return {
        "decision": decision,
        "setup_type": setup_type,
        "side": side,
        "sl_points": float(sl_points),
        "tp_points": float(tp_points),
        "rr_target": float(rr_target),
        "confidence": float(clamp(max(probability, quality_score), 0.0, 1.0)),
        "expected_value_r": float(max(expected_value_r, 0.0) if decision == "TAKE" else expected_value_r),
        "risk_tier": risk_tier,
        "management_plan": {
            "move_sl_to_be_at_r": float(move_to_be_r),
            "trail_after_r": float(trail_after_r),
            "trail_method": "atr",
            "trail_value": float(trail_value),
            "take_partial_at_r": 0.0,
            "time_stop_minutes": int(time_stop_minutes),
            "early_exit_rules": "local_brain_stall_news_invalidation",
            "management_directives": dict(management_directives),
        },
        "notes": "local_brain_trade_plan",
    }


def build_local_management_plan(context: dict[str, Any]) -> dict[str, Any]:
    symbol_key = str(context.get("symbol", "")).upper()
    setup_name = str(context.get("setup", "")).upper()
    session_name = str(context.get("session", "")).upper()
    setup_type = _setup_type(setup_name)
    pnl_r = float(context.get("pnl_r", 0.0) or 0.0)
    age_minutes = max(0.0, float(context.get("age_minutes", 0.0) or 0.0))
    spread_points = max(0.0, float(context.get("spread_points", 0.0) or 0.0))
    typical_spread_points = max(1.0, float(context.get("typical_spread_points", spread_points or 1.0) or 1.0))
    runtime = dict(context.get("runtime_features") or {})
    learning_bundle = dict(context.get("learning_brain_bundle") or {})
    pair_directive = dict((learning_bundle.get("pair_directives") or {}).get(symbol_key) or {})
    management_directives = dict(pair_directive.get("management_directives") or {})
    risk_reduction_active = bool(learning_bundle.get("risk_reduction_active", False))
    slippage_regime = str(pair_directive.get("slippage_regime") or "").lower()
    opportunity_capture_gap_r = max(0.0, float(pair_directive.get("opportunity_capture_gap_r", 0.0) or 0.0))
    shadow_experiment_active = bool(pair_directive.get("shadow_experiment_active", False))
    hour_expectancy_score = clamp(float(pair_directive.get("hour_expectancy_score", 0.5) or 0.5), 0.0, 1.0)
    lane_expectancy_multiplier = clamp(float(pair_directive.get("lane_expectancy_multiplier", 1.0) or 1.0), 0.80, 1.35)
    lane_mfe_median_r = max(0.0, float(pair_directive.get("lane_mfe_median_r", 0.0) or 0.0))
    lane_capture_efficiency = clamp(float(pair_directive.get("lane_capture_efficiency", 0.5) or 0.5), 0.0, 1.0)

    mc_win_rate = clamp(float(runtime.get("mc_win_rate", 0.5) or 0.5), 0.0, 1.0)
    fractal_persistence = clamp(float(runtime.get("fractal_persistence_score", 0.5) or 0.5), 0.0, 1.0)
    multi_tf_alignment = clamp(float(runtime.get("multi_tf_alignment_score", 0.5) or 0.5), 0.0, 1.0)
    dxy_support = clamp(float(runtime.get("dxy_support_score", _dxy_support_score(symbol_key, runtime, str(context.get("side", "BUY")))) or 0.5), 0.0, 1.0)
    compression_expansion = clamp(float(runtime.get("compression_expansion_score", 0.5) or 0.5), 0.0, 1.0)
    trajectory_catchup_pressure = clamp(float(learning_bundle.get("quota_catchup_pressure", 0.0) or 0.0), 0.0, 1.0)
    spread_drag = clamp(spread_points / typical_spread_points, 0.0, 3.0)

    confidence = clamp(
        (0.22 * clamp(max(pnl_r, 0.0) / 1.5, 0.0, 1.0))
        + (0.20 * mc_win_rate)
        + (0.16 * multi_tf_alignment)
        + (0.12 * fractal_persistence)
        + (0.10 * dxy_support)
        + (0.08 * compression_expansion)
        + _session_attack_bonus(symbol_key, session_name)
        - (0.10 * min(1.0, spread_drag / 2.0)),
        0.0,
        1.0,
    )

    decision = "HOLD"
    if pnl_r <= -1.20:
        decision = "CLOSE"
    elif pnl_r >= 0.45 or confidence >= 0.62:
        decision = "MODIFY"

    move_to_be_r = 0.5 if setup_type == "scalp" else 0.8
    if spread_drag >= 1.4:
        move_to_be_r = max(0.35, move_to_be_r - 0.10)
    if slippage_regime == "clean" and (shadow_experiment_active or opportunity_capture_gap_r >= 0.50):
        move_to_be_r = min(move_to_be_r, 0.42 if setup_type == "scalp" else 0.72)
    if trajectory_catchup_pressure >= 0.70 and pnl_r >= 0.35:
        decision = "MODIFY"
    if risk_reduction_active and pnl_r >= 0.25:
        decision = "MODIFY"
    runner_lane = bool(lane_mfe_median_r >= 1.75 and lane_capture_efficiency >= 0.58 and lane_expectancy_multiplier >= 1.02)
    stall_lane = bool(
        (lane_mfe_median_r > 0.0 and lane_mfe_median_r <= 1.05)
        or (lane_mfe_median_r > 0.0 and lane_capture_efficiency <= 0.48)
        or (hour_expectancy_score <= 0.42 and lane_mfe_median_r > 0.0)
    )
    if stall_lane and pnl_r >= 0.25:
        decision = "MODIFY"
    elif runner_lane and pnl_r >= 0.65:
        decision = "HOLD" if confidence >= 0.58 else decision

    trail_after_r = 0.45 if setup_type == "scalp" and slippage_regime == "clean" else 0.5 if setup_type == "scalp" else 0.85 if slippage_regime == "clean" else 0.9
    trail_value = 0.82 if setup_type == "scalp" and slippage_regime == "clean" else 0.9 if setup_type == "scalp" else 1.05 if slippage_regime == "clean" else 1.15
    time_stop_minutes = int(45 if setup_type == "scalp" and opportunity_capture_gap_r >= 0.50 else 50 if setup_type == "scalp" else 240 if setup_type == "daytrade" else 90)
    if runner_lane:
        move_to_be_r = min(0.95, move_to_be_r + 0.05)
        trail_after_r = min(1.10, trail_after_r + 0.05)
        trail_value = min(1.35, trail_value + 0.06)
        time_stop_minutes = int(max(time_stop_minutes, time_stop_minutes * 1.15))
    elif stall_lane:
        move_to_be_r = max(0.30, move_to_be_r - 0.08)
        trail_after_r = max(0.35, trail_after_r - 0.05)
        trail_value = max(0.72, trail_value - 0.06)
        time_stop_minutes = int(max(25, time_stop_minutes * 0.75))
    if management_directives:
        move_to_be_r = min(
            move_to_be_r,
            clamp(float(management_directives.get("early_protect_r", move_to_be_r) or move_to_be_r), 0.15, 0.95),
        )
        trail_after_r = min(
            trail_after_r,
            clamp(float(management_directives.get("early_protect_r", trail_after_r) or trail_after_r) + 0.05, 0.20, 1.20),
        )
        trail_backoff_r = clamp(float(management_directives.get("trail_backoff_r", 0.36) or 0.36), 0.18, 0.90)
        trail_value = clamp(0.50 + trail_backoff_r, 0.68, 1.40)
        if float(management_directives.get("stall_exit_bias", 0.0) or 0.0) >= 0.55:
            time_stop_minutes = int(max(20, time_stop_minutes * 0.80))

    return {
        "decision": decision,
        "confidence": float(confidence),
        "management_plan": {
            "move_sl_to_be_at_r": float(move_to_be_r),
            "trail_after_r": float(trail_after_r),
            "trail_method": "atr",
            "trail_value": float(trail_value),
            "take_partial_at_r": 0.0,
            "time_stop_minutes": int(time_stop_minutes),
            "early_exit_rules": "local_brain_stall_news_invalidation",
            "management_directives": dict(management_directives),
        },
        "notes": "local_brain_management_plan",
    }


def validate_broker_safe_modify(
    payload: BrokerSafeModifyInput,
    rule: SymbolRule,
) -> BrokerSafeModifyResult:
    side = str(payload.side or "").upper()
    symbol_key = str(rule.symbol or payload.symbol or "").upper()
    if side not in {"BUY", "SELL"}:
        return BrokerSafeModifyResult(
            valid=False,
            actionable=False,
            symbol_key=symbol_key,
            normalized_sl=None,
            normalized_tp=None,
            reason="invalid_due_to_side_geometry",
        )
    bid = float(payload.current_bid or 0.0)
    ask = float(payload.current_ask or 0.0)
    if bid <= 0.0 or ask <= 0.0:
        return BrokerSafeModifyResult(
            valid=False,
            actionable=False,
            symbol_key=symbol_key,
            normalized_sl=None,
            normalized_tp=None,
            reason="invalid_due_to_missing_live_price",
            bid=bid,
            ask=ask,
        )

    point = max(float(rule.point), 1e-9)
    tick_size = max(float(rule.tick_size), point)
    digits = int(rule.digits)
    buffer_points = max(0.0, float(payload.safety_buffer_points))
    min_stop_points = max(0.0, float(rule.min_stop_points))
    freeze_points = max(0.0, float(rule.freeze_points))
    effective_points = max(min_stop_points, freeze_points) + buffer_points
    min_gap_price = max(tick_size, (min_stop_points + buffer_points) * point)
    freeze_gap_price = max(0.0, (freeze_points + buffer_points) * point)
    effective_gap_price = max(tick_size, effective_points * point)
    reference_price = bid if side == "BUY" else ask

    current_sl = _normalize_price(float(payload.current_sl or 0.0), tick_size, digits) if float(payload.current_sl or 0.0) > 0 else None
    current_tp = _normalize_price(float(payload.current_tp or 0.0), tick_size, digits) if float(payload.current_tp or 0.0) > 0 else None

    wants_sl = float(payload.desired_sl or 0.0) > 0.0
    wants_tp = float(payload.desired_tp or 0.0) > 0.0
    if not wants_sl and not wants_tp:
        return BrokerSafeModifyResult(
            valid=True,
            actionable=False,
            symbol_key=symbol_key,
            normalized_sl=current_sl,
            normalized_tp=current_tp,
            reason="no_desired_change",
            bid=bid,
            ask=ask,
            reference_price=reference_price,
            min_stop_distance_points=min_stop_points,
            freeze_distance_points=freeze_points,
            effective_distance_points=effective_points,
        )

    clamp_reasons: list[str] = []

    def _normalize_sl(desired: float) -> float:
        candidate = _normalize_price(desired, tick_size, digits)
        if candidate != round(float(desired), digits):
            clamp_reasons.append("invalid_due_to_tick_rounding")
        if side == "BUY":
            if candidate >= reference_price:
                clamp_reasons.append("invalid_due_to_side_geometry")
            if (reference_price - candidate) < min_gap_price:
                clamp_reasons.append("invalid_due_to_stop_level")
            if freeze_gap_price > 0 and (reference_price - candidate) < freeze_gap_price:
                clamp_reasons.append("invalid_due_to_freeze_level")
            candidate = min(candidate, reference_price - effective_gap_price)
        else:
            if candidate <= reference_price:
                clamp_reasons.append("invalid_due_to_side_geometry")
            if (candidate - reference_price) < min_gap_price:
                clamp_reasons.append("invalid_due_to_stop_level")
            if freeze_gap_price > 0 and (candidate - reference_price) < freeze_gap_price:
                clamp_reasons.append("invalid_due_to_freeze_level")
            candidate = max(candidate, reference_price + effective_gap_price)
        candidate = _normalize_price(candidate, tick_size, digits)
        if side == "BUY" and candidate >= reference_price:
            raise ValueError("invalid_due_to_side_geometry")
        if side == "SELL" and candidate <= reference_price:
            raise ValueError("invalid_due_to_side_geometry")
        return candidate

    def _normalize_tp(desired: float) -> float:
        candidate = _normalize_price(desired, tick_size, digits)
        if candidate != round(float(desired), digits):
            clamp_reasons.append("invalid_due_to_tick_rounding")
        if side == "BUY":
            if candidate <= reference_price:
                clamp_reasons.append("invalid_due_to_side_geometry")
            if (candidate - reference_price) < min_gap_price:
                clamp_reasons.append("invalid_due_to_stop_level")
            if freeze_gap_price > 0 and (candidate - reference_price) < freeze_gap_price:
                clamp_reasons.append("invalid_due_to_freeze_level")
            candidate = max(candidate, reference_price + effective_gap_price)
        else:
            if candidate >= reference_price:
                clamp_reasons.append("invalid_due_to_side_geometry")
            if (reference_price - candidate) < min_gap_price:
                clamp_reasons.append("invalid_due_to_stop_level")
            if freeze_gap_price > 0 and (reference_price - candidate) < freeze_gap_price:
                clamp_reasons.append("invalid_due_to_freeze_level")
            candidate = min(candidate, reference_price - effective_gap_price)
        candidate = _normalize_price(candidate, tick_size, digits)
        if side == "BUY" and candidate <= reference_price:
            raise ValueError("invalid_due_to_side_geometry")
        if side == "SELL" and candidate >= reference_price:
            raise ValueError("invalid_due_to_side_geometry")
        return candidate

    try:
        normalized_sl = _normalize_sl(float(payload.desired_sl)) if wants_sl else current_sl
    except ValueError as exc:
        return BrokerSafeModifyResult(
            valid=False,
            actionable=False,
            symbol_key=symbol_key,
            normalized_sl=None,
            normalized_tp=None,
            reason=str(exc),
            clamp_reasons=tuple(dict.fromkeys(clamp_reasons)),
            bid=bid,
            ask=ask,
            reference_price=reference_price,
            min_stop_distance_points=min_stop_points,
            freeze_distance_points=freeze_points,
            effective_distance_points=effective_points,
        )

    try:
        if wants_tp:
            normalized_tp = _normalize_tp(float(payload.desired_tp))
        elif current_tp is not None:
            normalized_tp = current_tp
        elif payload.allow_tp_none:
            normalized_tp = None
        else:
            normalized_tp = None
    except ValueError as exc:
        return BrokerSafeModifyResult(
            valid=False,
            actionable=False,
            symbol_key=symbol_key,
            normalized_sl=normalized_sl,
            normalized_tp=None,
            reason=str(exc),
            clamp_reasons=tuple(dict.fromkeys(clamp_reasons)),
            bid=bid,
            ask=ask,
            reference_price=reference_price,
            min_stop_distance_points=min_stop_points,
            freeze_distance_points=freeze_points,
            effective_distance_points=effective_points,
        )

    if normalized_sl is None and normalized_tp is None:
        return BrokerSafeModifyResult(
            valid=True,
            actionable=False,
            symbol_key=symbol_key,
            normalized_sl=None,
            normalized_tp=None,
            reason="no_desired_change",
            clamp_reasons=tuple(dict.fromkeys(clamp_reasons)),
            bid=bid,
            ask=ask,
            reference_price=reference_price,
            min_stop_distance_points=min_stop_points,
            freeze_distance_points=freeze_points,
            effective_distance_points=effective_points,
        )

    if _almost_same(normalized_sl, current_sl, tick_size) and _almost_same(normalized_tp, current_tp, tick_size):
        return BrokerSafeModifyResult(
            valid=True,
            actionable=False,
            symbol_key=symbol_key,
            normalized_sl=normalized_sl,
            normalized_tp=normalized_tp,
            reason="no_effective_change",
            clamp_reasons=tuple(dict.fromkeys(clamp_reasons)),
            bid=bid,
            ask=ask,
            reference_price=reference_price,
            min_stop_distance_points=min_stop_points,
            freeze_distance_points=freeze_points,
            effective_distance_points=effective_points,
        )

    return BrokerSafeModifyResult(
        valid=True,
        actionable=True,
        symbol_key=symbol_key,
        normalized_sl=normalized_sl,
        normalized_tp=normalized_tp,
        reason="validated_clamped" if clamp_reasons else "validated",
        clamp_reasons=tuple(dict.fromkeys(clamp_reasons)),
        bid=bid,
        ask=ask,
        reference_price=reference_price,
        min_stop_distance_points=min_stop_points,
        freeze_distance_points=freeze_points,
        effective_distance_points=effective_points,
    )


def evaluate_profitable_trade_management(
    payload: RetracementManagementInput,
) -> RetracementManagementDecision:
    symbol_key = str(payload.symbol or "").upper()
    side = str(payload.side or "").upper()
    if side not in {"BUY", "SELL"}:
        return RetracementManagementDecision(
            management_action="HOLD",
            retracement_exit_score=0.0,
            continuation_score=0.0,
            reversal_risk_score=0.0,
            protection_mode="inactive",
            trade_state="INIT",
            trade_quality="DEAD",
            reason="invalid_side",
        )
    if float(payload.entry_price) <= 0 or float(payload.current_price) <= 0:
        return RetracementManagementDecision(
            management_action="HOLD",
            retracement_exit_score=0.0,
            continuation_score=0.0,
            reversal_risk_score=0.0,
            protection_mode="inactive",
            trade_state="INIT",
            trade_quality="DEAD",
            reason="invalid_price_context",
        )

    family = _infer_strategy_family(payload, symbol_key)
    profile = dict(_MANAGEMENT_PROFILES.get(family, _MANAGEMENT_PROFILES["TREND"]))
    profile.update(_profile_overrides(payload, symbol_key=symbol_key, family=family))
    tolerant_symbol = symbol_key in {"BTCUSD", "XAUUSD"}
    min_profit_r = float(profile["min_profit_r"])
    min_age_minutes = float(profile["min_age_minutes"])
    pnl_r = float(payload.pnl_r)
    timeframe_minutes = _timeframe_minutes(payload.timeframe)
    bars_in_trade = int(payload.bars_in_trade) if int(payload.bars_in_trade or 0) > 0 else int(max(1.0, float(payload.age_minutes) / max(1, timeframe_minutes)))

    runtime = dict(payload.runtime_features or {})
    target_distance = abs(float(payload.tp) - float(payload.entry_price)) if float(payload.tp) > 0 else 0.0
    remaining_distance = abs(float(payload.tp) - float(payload.current_price)) if float(payload.tp) > 0 else 0.0
    tp_progress = clamp(1.0 - (remaining_distance / max(target_distance, 1e-9)), 0.0, 1.0) if target_distance > 0 else clamp(pnl_r / 2.0, 0.0, 1.0)
    typical_spread = max(float(payload.typical_spread_points or 0.0), 1.0)
    spread_ratio = clamp(float(payload.spread_points) / typical_spread, 0.0, 3.0)
    time_stop = max(15.0, float(payload.time_stop_minutes or 120))
    age_factor = clamp(float(payload.age_minutes) / time_stop, 0.0, 1.0)

    directional_values = [
        _directional_feature(runtime.get("m1_ret_1"), payload.current_price, side),
        _directional_feature(runtime.get("m1_ret_3"), payload.current_price, side),
        _directional_feature(runtime.get("m5_ret_1"), payload.current_price, side),
        _directional_feature(runtime.get("m5_ret_3"), payload.current_price, side),
        _directional_feature(runtime.get("m1_momentum_1"), payload.current_price, side),
        _directional_feature(runtime.get("m5_momentum_3"), payload.current_price, side),
        _directional_feature(runtime.get("m5_macd_hist_slope"), payload.current_price, side),
        _directional_feature(runtime.get("m5_slope"), payload.current_price, side),
    ]
    directional_values = [value for value in directional_values if value != 0.0]
    directional_momentum = sum(directional_values) / len(directional_values) if directional_values else 0.0
    momentum_score = clamp(0.5 + (directional_momentum * 250.0), 0.0, 1.0)
    if not directional_values:
        momentum_score = 0.5

    compression_proxy = clamp(
        1.0 - float(runtime.get("m5_atr_pct_of_avg", runtime.get("atr_ratio", 1.0))),
        0.0,
        1.0,
    )
    local_macd_slope = float(runtime.get("m5_macd_hist_slope", 0.0) or 0.0)
    local_price_slope = float(runtime.get("m5_slope", 0.0) or 0.0)
    reversal_candle_proxy = clamp(
        max(
            float(runtime.get("m5_pinbar_bear" if side == "BUY" else "m5_pinbar_bull", 0.0)),
            float(runtime.get("m5_engulf_bear" if side == "BUY" else "m5_engulf_bull", 0.0)),
        ),
        0.0,
        1.0,
    )
    wick_rejection_proxy = clamp(
        float(runtime.get("m5_upper_wick_ratio" if side == "BUY" else "m5_lower_wick_ratio", 0.0)),
        0.0,
        1.0,
    )
    range_position = clamp(float(runtime.get("m5_range_position_20", 0.5)), 0.0, 1.0)
    structural_progress = range_position if side == "BUY" else (1.0 - range_position)
    ai_decision = str(payload.ai_decision or "").upper()
    ai_confidence = clamp(float(payload.ai_confidence or 0.0), 0.0, 1.0)
    ai_close_bias = ai_confidence if ai_decision == "CLOSE" else (ai_confidence * 0.5 if ai_decision == "MODIFY" else 0.0)
    ai_continue_bias = ai_confidence if ai_decision == "HOLD" else (ai_confidence * 0.4 if ai_decision == "MODIFY" else 0.0)

    protection_urge = clamp((pnl_r - min_profit_r) / (1.35 if tolerant_symbol else 1.15), 0.0, 1.0)
    continuation_score = clamp(
        (0.26 * tp_progress)
        + (0.24 * momentum_score)
        + (0.12 * structural_progress)
        + (0.12 * (1.0 - min(spread_ratio / 1.5, 1.0)))
        + (0.10 * (1.0 - reversal_candle_proxy))
        + (0.08 * (1.0 - wick_rejection_proxy))
        + (0.08 * ai_continue_bias),
        0.0,
        1.0,
    )
    reversal_risk_score = clamp(
        (0.16 * protection_urge)
        + (0.14 * age_factor)
        + (0.16 * min(spread_ratio / 1.5, 1.0))
        + (0.16 * (1.0 - momentum_score))
        + (0.12 * reversal_candle_proxy)
        + (0.08 * wick_rejection_proxy)
        + (0.06 * compression_proxy)
        + (0.05 * (1.0 - structural_progress))
        + (0.07 * ai_close_bias),
        0.0,
        1.0,
    )
    retracement_exit_score = clamp(
        (0.55 * reversal_risk_score) + (0.30 * protection_urge) - (0.35 * continuation_score),
        0.0,
        1.0,
    )
    swing_continuation_score = _swing_continuation_score(
        payload,
        side=side,
        runtime=runtime,
        continuation_score=continuation_score,
    )
    stall_bar_threshold = max(1.0, float(profile["stall_bars"]))
    stall_detected = bool(
        payload.enable_stall_exit
        and pnl_r >= float(profile["stall_min_profit_r"])
        and bars_in_trade >= stall_bar_threshold
        and tp_progress >= 0.18
        and continuation_score < 0.54
        and (
            compression_proxy >= 0.52
            or reversal_candle_proxy >= 0.40
            or wick_rejection_proxy >= 0.45
            or structural_progress < 0.56
        )
    )
    trade_quality = _quality_bucket(continuation_score, reversal_risk_score, stall_detected)

    risk_distance = abs(float(payload.entry_price) - float(payload.sl))
    if risk_distance <= 0.0:
        risk_distance = max(float(payload.current_price) * 0.0015, 1e-6)
    spread_value = max(float(payload.spread_points or 0.0), 0.0)
    point_size = max(float(payload.point_size or 0.0), 1e-9)
    spread_recovery_r = (spread_value * point_size) / max(risk_distance, 1e-9)
    spread_recovery_partial_trigger_r = spread_recovery_r + 0.30
    spread_recovery_trail_unlock_r = spread_recovery_r + 0.50
    protect_confirm_r = float(profile["protect_confirm_r"])
    runner_start_r = float(profile["runner_start_r"])
    trail_backoff_r = float(profile["trail_backoff_r"])
    protect_confirm_r = max(protect_confirm_r, spread_recovery_trail_unlock_r)
    runner_start_r = max(runner_start_r, spread_recovery_trail_unlock_r)
    trade_state = "INIT"
    if pnl_r < protect_confirm_r:
        trade_state = "PROVING"
    elif pnl_r < runner_start_r:
        trade_state = "PROTECTED"
    else:
        trade_state = "RUNNER"
    if trade_quality == "DEGRADING":
        trade_state = "EXIT_READY"
    if trade_quality == "DEAD":
        trade_state = "FORCE_EXIT"

    details = {
        "family": family,
        "tp_progress": tp_progress,
        "momentum_score": momentum_score,
        "spread_ratio": spread_ratio,
        "compression_proxy": compression_proxy,
        "reversal_candle_proxy": reversal_candle_proxy,
        "wick_rejection_proxy": wick_rejection_proxy,
        "structural_progress": structural_progress,
        "bars_in_trade": bars_in_trade,
        "mfe_r": float(payload.mfe_r),
        "mae_r": float(payload.mae_r),
        "spread_recovery_r": spread_recovery_r,
        "spread_recovery_partial_trigger_r": spread_recovery_partial_trigger_r,
        "spread_recovery_trail_unlock_r": spread_recovery_trail_unlock_r,
        "spread_recovery_partial_taken": bool(payload.spread_recovery_partial_taken),
        "swing_continuation_score": swing_continuation_score,
    }
    super_aggro_symbol = symbol_key in _SUPER_AGGRESSIVE_SWING_SYMBOLS
    dxy_support = _dxy_support_score(symbol_key, runtime, side)
    fractal_persistence = clamp(float(runtime.get("fractal_persistence_score", 0.5) or 0.5), 0.0, 1.0)
    compression_expansion = clamp(float(runtime.get("compression_expansion_score", compression_proxy) or compression_proxy), 0.0, 1.0)
    mc_win_rate = clamp(float(runtime.get("mc_win_rate", 0.5) or 0.5), 0.0, 1.0)
    transition_momentum = clamp(float(runtime.get("transition_momentum", 0.0) or 0.0), 0.0, 1.0)
    details.update(
        {
            "dxy_support_score": dxy_support,
            "fractal_persistence_score": fractal_persistence,
            "compression_expansion_score": compression_expansion,
            "mc_win_rate": mc_win_rate,
            "transition_momentum": transition_momentum,
        }
    )

    if pnl_r <= -1.0:
        return RetracementManagementDecision(
            management_action="FULL_EXIT",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="hard_loss_cut",
            trade_state="FORCE_EXIT",
            trade_quality="DEAD",
            stall_detected=False,
            reason="hard_loss_cut",
            details=details,
        )

    if pnl_r < min_profit_r and float(payload.age_minutes) < min_age_minutes:
        return RetracementManagementDecision(
            management_action="HOLD",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="inactive",
            trade_state=trade_state,
            trade_quality=trade_quality,
            stall_detected=False,
            reason="profit_threshold_not_met",
            details={**details, "min_profit_r": min_profit_r, "min_age_minutes": min_age_minutes},
        )

    time_stop_minutes = max(5.0, float(payload.time_stop_minutes or 120)) * float(profile["time_stop_factor"])
    if float(payload.age_minutes) >= time_stop_minutes and pnl_r <= 0.35:
        reason = "time_stop_exit"
        if pnl_r <= 0.12:
            reason = "scratch_exit"
        return RetracementManagementDecision(
            management_action="FULL_EXIT",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="time_stop",
            trade_state="FORCE_EXIT",
            trade_quality=trade_quality,
            stall_detected=stall_detected,
            reason=reason,
            details=details,
        )

    invalidation_score = clamp(
        (0.45 * reversal_risk_score)
        + (0.20 * (1.0 - continuation_score))
        + (0.15 * wick_rejection_proxy)
        + (0.10 * (1.0 - structural_progress))
        + (0.10 * ai_close_bias),
        0.0,
        1.0,
    )
    if pnl_r <= max(0.15, -float(profile["scratch_loss_r"])) and float(payload.age_minutes) >= min_age_minutes and invalidation_score >= 0.78:
        return RetracementManagementDecision(
            management_action="FULL_EXIT",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="thesis_invalidation",
            trade_state="FORCE_EXIT",
            trade_quality="DEAD",
            stall_detected=False,
            reason="early_exit_thesis_invalidated",
            details={**details, "invalidation_score": invalidation_score},
        )

    close_threshold = 0.76 if tolerant_symbol else 0.74
    if retracement_exit_score >= close_threshold and continuation_score < 0.40 and reversal_risk_score >= 0.85:
        return RetracementManagementDecision(
            management_action="FULL_EXIT",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="full_profit_protect",
            trade_state="FORCE_EXIT",
            trade_quality="DEAD",
            stall_detected=stall_detected,
            reason="reversal_risk_materially_high",
            details=details,
        )

    if (
        not bool(payload.spread_recovery_partial_taken)
        and float(payload.volume) >= max(float(payload.min_lot) * 2.0, float(payload.min_lot) + 1e-9)
        and pnl_r >= spread_recovery_partial_trigger_r
        and pnl_r < 0.80
    ):
        return RetracementManagementDecision(
            management_action="PARTIAL_EXIT",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="spread_recovery_partial",
            close_fraction=0.45,
            trade_state="PROTECTED" if pnl_r < runner_start_r else "RUNNER",
            trade_quality=trade_quality,
            stall_detected=stall_detected,
            reason="spread_recovery_partial",
            details=details,
        )

    if stall_detected and continuation_score < 0.50:
        if float(payload.volume) >= max(float(payload.min_lot) * 2.0, float(payload.min_lot) + 1e-9) and 0.85 <= pnl_r < 1.6 and tp_progress < 0.90:
            return RetracementManagementDecision(
                management_action="PARTIAL_EXIT",
                retracement_exit_score=retracement_exit_score,
                continuation_score=continuation_score,
                reversal_risk_score=reversal_risk_score,
                protection_mode="stall_partial_exit",
                close_fraction=0.5,
                trade_state="EXIT_READY",
                trade_quality=trade_quality,
                stall_detected=True,
                reason="stall_exit",
                details=details,
            )
        return RetracementManagementDecision(
            management_action="FULL_EXIT",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="full_profit_protect" if pnl_r >= 1.5 else "stall_full_exit",
            trade_state="FORCE_EXIT",
            trade_quality=trade_quality,
            stall_detected=True,
            reason="stall_exit",
            details=details,
        )

    if (
        float(payload.volume) >= max(float(payload.min_lot) * 2.0, float(payload.min_lot) + 1e-9)
        and pnl_r >= 1.0
        and retracement_exit_score >= 0.68
        and continuation_score < 0.52
    ):
        return RetracementManagementDecision(
            management_action="PARTIAL_EXIT",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="partial_profit_protect",
            close_fraction=0.5,
            trade_state="EXIT_READY",
            trade_quality=trade_quality,
            stall_detected=stall_detected,
            reason="profit_decent_continuation_weak",
            details=details,
        )

    spread_drag = clamp(
        float(payload.spread_points) / max(float(payload.typical_spread_points or 1.0), 1.0),
        1.0,
        3.0,
    )
    trend_like_key = f"{str(payload.strategy_family or '').upper()} {str(payload.setup or '').upper()}".strip()
    fx_trend_like = bool(
        family == "TREND"
        and symbol_key not in {"BTCUSD", "XAUUSD", "NAS100", "USOIL"}
        and any(
            token in trend_like_key
            for token in ("BREAKOUT", "IMPULSE", "CONTINUATION", "TREND", "PULLBACK")
        )
    )
    if (
        family in {"TREND", "CRYPTO MOMENTUM"}
        and float(payload.volume) >= max(float(payload.min_lot) * 2.0, float(payload.min_lot) + 1e-9)
        and pnl_r >= 0.92
        and spread_drag >= 1.25
        and continuation_score < 0.50
        and retracement_exit_score >= 0.24
    ):
        return RetracementManagementDecision(
            management_action="PARTIAL_EXIT",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="trend_partial_capture",
            close_fraction=0.5,
            trade_state="EXIT_READY",
            trade_quality=trade_quality,
            stall_detected=stall_detected,
            reason="trend_partial_capture_on_weakening_continuation",
            details=details,
        )
    if (
        fx_trend_like
        and float(payload.volume) >= max(float(payload.min_lot) * 2.0, float(payload.min_lot) + 1e-9)
        and pnl_r >= 0.78
        and tp_progress >= 0.42
        and (
            continuation_score < 0.44
            or (
                spread_drag >= 1.25
                and wick_rejection_proxy >= 0.34
                and (
                    (side == "BUY" and local_macd_slope < 0.0 and local_price_slope < 0.0)
                    or (side == "SELL" and local_macd_slope > 0.0 and local_price_slope > 0.0)
                )
            )
        )
        and (
            retracement_exit_score >= 0.22
            or reversal_risk_score >= 0.28
        )
    ):
        return RetracementManagementDecision(
            management_action="PARTIAL_EXIT",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="trend_partial_capture",
            close_fraction=0.5,
            trade_state="EXIT_READY",
            trade_quality=trade_quality,
            stall_detected=stall_detected,
            reason="trend_partial_capture_on_degrading_extension",
            details=details,
        )

    net_positive_lock_floor = 0.0
    if family == "GRID":
        net_positive_lock_floor = max(0.12, 0.07 * spread_drag)
    elif family in {"TREND", "CRYPTO MOMENTUM"}:
        net_positive_lock_floor = max(0.26, 0.14 * spread_drag)
        if fx_trend_like:
            net_positive_lock_floor = max(net_positive_lock_floor, 0.34, 0.18 * spread_drag)
    else:
        net_positive_lock_floor = max(0.12, 0.08 * spread_drag)

    profit_lock_r = 0.0
    if pnl_r >= 1.5:
        profit_lock_r = 0.68 if family in {"TREND", "CRYPTO MOMENTUM"} else 0.40
    elif pnl_r >= 1.0:
        profit_lock_r = 0.35 if family in {"TREND", "CRYPTO MOMENTUM"} else 0.22
    elif pnl_r >= protect_confirm_r and continuation_score >= 0.52:
        if family == "GRID":
            profit_lock_r = 0.12
        elif family in {"TREND", "CRYPTO MOMENTUM"}:
            profit_lock_r = max(0.18, 0.10 * spread_drag)
        else:
            profit_lock_r = 0.12
    if profit_lock_r > 0.0:
        profit_lock_r = max(profit_lock_r, net_positive_lock_floor)
    if fx_trend_like and profit_lock_r > 0.0 and spread_drag >= 1.15:
        profit_lock_r = max(profit_lock_r, 0.42, 0.20 * spread_drag)

    swing_family = family in {"TREND", "CRYPTO MOMENTUM"} or bool(
        symbol_key == "XAUUSD" and "GRID" not in str(payload.setup or "").upper()
    )
    swing_candidate = bool(
        not payload.event_risk_active
        and swing_family
        and pnl_r >= max(runner_start_r, 1.05 if family == "GRID" else 0.95)
        and tp_progress >= 0.58
        and continuation_score >= (0.68 if symbol_key == "XAUUSD" else 0.64)
        and swing_continuation_score >= (0.74 if family == "CRYPTO MOMENTUM" else 0.70)
        and reversal_risk_score <= 0.40
        and not stall_detected
    )
    super_aggro_swing_candidate = bool(
        super_aggro_symbol
        and not payload.event_risk_active
        and pnl_r >= max(0.88, runner_start_r - 0.10)
        and tp_progress >= 0.46
        and continuation_score >= 0.60
        and swing_continuation_score >= 0.66
        and fractal_persistence >= 0.62
        and compression_expansion >= 0.54
        and mc_win_rate >= 0.85
        and momentum_score >= 0.58
        and transition_momentum >= 0.10
        and dxy_support >= 0.50
        and reversal_risk_score <= 0.42
        and not stall_detected
    )
    details["super_aggro_swing_candidate"] = super_aggro_swing_candidate
    if super_aggro_swing_candidate:
        swing_candidate = True
    swing_extension_r = 0.0
    swing_trail_backoff_r = trail_backoff_r
    if swing_candidate:
        swing_extension_r = (
            0.55
            if family == "CRYPTO MOMENTUM"
            else 0.60
            if symbol_key in {"NAS100", "USOIL"} and super_aggro_swing_candidate
            else 0.46
            if super_aggro_swing_candidate
            else 0.42
            if symbol_key == "XAUUSD"
            else 0.34
        )
        swing_trail_backoff_r = max(
            trail_backoff_r + (0.28 if family == "CRYPTO MOMENTUM" else 0.26 if super_aggro_swing_candidate else 0.18),
            0.70 if family == "CRYPTO MOMENTUM" else 0.76 if super_aggro_swing_candidate else 0.56,
        )

    if payload.enable_adaptive_tp and swing_candidate:
        extended_tp = float(payload.tp) + (risk_distance * swing_extension_r) if side == "BUY" else float(payload.tp) - (risk_distance * swing_extension_r)
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, max(profit_lock_r, max(0.0, pnl_r - swing_trail_backoff_r)))
        return RetracementManagementDecision(
            management_action="EXTEND_TP",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="swing_runner_extension",
            tighten_to_price=tighten_to,
            updated_tp_price=extended_tp,
            trade_state="SWING",
            trade_quality="STRONG",
            stall_detected=False,
            profit_lock_r=max(profit_lock_r, max(0.0, pnl_r - swing_trail_backoff_r)),
            reason="swing_continuation_extend_tp",
            details={**details, "swing_extension_r": swing_extension_r, "swing_trail_backoff_r": swing_trail_backoff_r},
        )

    if payload.enable_adaptive_tp and pnl_r >= 1.0 and tp_progress >= float(profile["tp_extend_progress"]) and continuation_score >= 0.74 and reversal_risk_score <= 0.42:
        extended_tp = float(payload.tp) + (risk_distance * float(profile["tp_extend_r"])) if side == "BUY" else float(payload.tp) - (risk_distance * float(profile["tp_extend_r"]))
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, max(profit_lock_r, max(0.0, pnl_r - trail_backoff_r)))
        return RetracementManagementDecision(
            management_action="EXTEND_TP",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="runner_extension",
            tighten_to_price=tighten_to,
            updated_tp_price=extended_tp,
            trade_state="RUNNER",
            trade_quality=trade_quality,
            stall_detected=False,
            profit_lock_r=max(profit_lock_r, max(0.0, pnl_r - trail_backoff_r)),
            reason="continuation_strengthening_extend_tp",
            details=details,
        )

    if payload.enable_dynamic_trail and swing_candidate:
        swing_lock_r = max(profit_lock_r, max(0.0, pnl_r - swing_trail_backoff_r))
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, swing_lock_r)
        return RetracementManagementDecision(
            management_action="TRAIL_STOP",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="swing_runner_trail",
            tighten_to_price=tighten_to,
            trade_state="SWING",
            trade_quality="STRONG",
            stall_detected=False,
            profit_lock_r=swing_lock_r,
            reason="swing_runner_trailing_update",
            details={**details, "swing_trail_backoff_r": swing_trail_backoff_r},
        )

    if payload.enable_dynamic_trail and pnl_r >= runner_start_r and continuation_score >= (0.56 if family in {"TREND", "CRYPTO MOMENTUM"} else 0.60) and not stall_detected:
        trail_lock_r = max(profit_lock_r, max(0.0, pnl_r - trail_backoff_r))
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, trail_lock_r)
        return RetracementManagementDecision(
            management_action="TRAIL_STOP",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="runner_trail",
            tighten_to_price=tighten_to,
            trade_state="RUNNER",
            trade_quality=trade_quality,
            stall_detected=False,
            profit_lock_r=trail_lock_r,
            reason="runner_trailing_update",
            details=details,
        )

    profit_lock_confirmation_continuation = 0.46
    profit_lock_confirmation_buffer = (
        max(0.18, 0.12 * spread_drag)
        if family in {"TREND", "CRYPTO MOMENTUM"}
        else max(0.10, 0.06 * spread_drag)
    )
    if fx_trend_like:
        profit_lock_confirmation_continuation = 0.62
        profit_lock_confirmation_buffer = max(profit_lock_confirmation_buffer, 0.40, 0.22 * spread_drag)
        if pnl_r < 0.90 and continuation_score < 0.60:
            profit_lock_confirmation_buffer = max(profit_lock_confirmation_buffer, 0.48)
    elif family == "GRID":
        profit_lock_confirmation_buffer = max(profit_lock_confirmation_buffer, 0.16, 0.10 * spread_drag)

    if (
        profit_lock_r > 0.0
        and pnl_r >= min_profit_r
        and continuation_score >= profit_lock_confirmation_continuation
        and (pnl_r - profit_lock_r) >= profit_lock_confirmation_buffer
    ):
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
        return RetracementManagementDecision(
            management_action="TIGHTEN_STOP",
            retracement_exit_score=retracement_exit_score,
            continuation_score=continuation_score,
            reversal_risk_score=reversal_risk_score,
            protection_mode="profit_lock",
            tighten_to_price=tighten_to,
            trade_state=trade_state if trade_state != "INIT" else "PROTECTED",
            trade_quality=trade_quality,
            stall_detected=stall_detected,
            profit_lock_r=profit_lock_r,
            reason="profit_lock_confirmation",
            details=details,
        )

    return RetracementManagementDecision(
        management_action="HOLD",
        retracement_exit_score=retracement_exit_score,
        continuation_score=continuation_score,
        reversal_risk_score=reversal_risk_score,
        protection_mode="hold",
        trade_state="SWING" if swing_candidate else trade_state,
        trade_quality="STRONG" if swing_candidate else trade_quality,
        stall_detected=stall_detected,
        profit_lock_r=profit_lock_r,
        reason=(
            "profit_threshold_not_met"
            if pnl_r < min_profit_r
            else ("swing_continuation_hold" if swing_candidate else "continuation_still_acceptable")
        ),
        details={
            **details,
            "swing_candidate": bool(swing_candidate),
            "swing_trail_backoff_r": swing_trail_backoff_r if swing_candidate else 0.0,
        },
    )


def smart_manage_trade(
    payload: RetracementManagementInput,
) -> RetracementManagementDecision:
    base_decision = evaluate_profitable_trade_management(payload)
    runtime = dict(payload.runtime_features or {})
    learning_bundle = dict(payload.learning_brain_bundle or {})
    symbol_key = str(payload.symbol or "").upper()
    pair_directive = dict((learning_bundle.get("pair_directives") or {}).get(symbol_key) or {})
    side = str(payload.side or "").upper()
    risk_distance = abs(float(payload.entry_price) - float(payload.sl))
    if risk_distance <= 0.0:
        risk_distance = max(float(payload.current_price or payload.entry_price or 1.0) * 0.0015, 1e-6)

    mc_win_rate = clamp(float(runtime.get("mc_win_rate", 0.5) or 0.5), 0.0, 1.0)
    account_equity = max(
        0.0,
        float(
            runtime.get(
                "account_equity",
                runtime.get("equity", runtime.get("current_equity", 0.0)),
            )
            or 0.0
        ),
    )
    microstructure_score = clamp(
        float(
            runtime.get(
                "microstructure_composite_score",
                runtime.get("microstructure_score", runtime.get("microstructure_alignment_score", base_decision.continuation_score)),
            )
            or base_decision.continuation_score
        ),
        0.0,
        1.0,
    )
    fractal_persistence = clamp(float(runtime.get("fractal_persistence_score", 0.5) or 0.5), 0.0, 1.0)
    dxy_support = clamp(
        float(runtime.get("dxy_support_score", _dxy_support_score(symbol_key, runtime, side)) or 0.5),
        0.0,
        1.0,
    )
    trajectory_catchup_pressure = clamp(float(learning_bundle.get("quota_catchup_pressure", 0.0) or 0.0), 0.0, 1.0)
    risk_reduction_active = bool(learning_bundle.get("risk_reduction_active", False))
    aggression_multiplier = clamp(float(pair_directive.get("aggression_multiplier", 1.0) or 1.0), 0.75, 1.50)
    trade_horizon_bias = str(pair_directive.get("trade_horizon_bias") or "").lower()
    partials_supported = bool(runtime.get("partial_closes_supported", runtime.get("allow_partial_closes", True)))
    management_directives = dict(pair_directive.get("management_directives") or {})
    early_protect_r = clamp(float(management_directives.get("early_protect_r", 0.25) or 0.25), 0.15, 0.60)
    trail_backoff_r = clamp(float(management_directives.get("trail_backoff_r", 0.34) or 0.34), 0.18, 0.90)
    tp_extension_bias = clamp(float(management_directives.get("tp_extension_bias", 0.16) or 0.16), 0.0, 0.50)
    reentry_bias = clamp(float(management_directives.get("reentry_bias", pair_directive.get("reentry_priority", 0.0)) or 0.0), 0.0, 1.0)
    stall_exit_bias = clamp(float(management_directives.get("stall_exit_bias", 0.25) or 0.25), 0.0, 1.0)
    protected_no_loosen = bool(management_directives.get("no_loosen_after_protected", False))
    runner_lane = bool(management_directives.get("runner_lane", False))
    stall_lane = bool(management_directives.get("stall_lane", False))
    xau_grid_active = bool(symbol_key == "XAUUSD" and "GRID" in str(payload.setup or "").upper())
    btc_fast_lock_active = bool(symbol_key == "BTCUSD")
    session_key = str(payload.session_name or runtime.get("session_name", "") or "").upper()
    xau_grid_prime_session = bool(xau_grid_active and session_key in {"LONDON", "OVERLAP", "NEW_YORK"})
    xau_grid_asia_session = bool(xau_grid_active and session_key in {"TOKYO", "SYDNEY"})
    watchdog_force = bool(runtime.get("management_watchdog_force", False))
    watchdog_reason = str(runtime.get("management_watchdog_reason") or "").strip()
    watchdog_stale_seconds = max(0.0, float(runtime.get("management_watchdog_stale_seconds") or 0.0))
    event_directive = dict(runtime.get("event_directive") or {})
    execution_minute_profile = dict(runtime.get("execution_minute_profile") or {})
    event_playbook = str(event_directive.get("playbook") or runtime.get("event_playbook") or "").lower()
    execution_minute_quality = clamp(
        float(execution_minute_profile.get("quality_score", runtime.get("execution_minute_quality_score", 0.70)) or 0.70),
        0.0,
        1.0,
    )
    execution_minute_size_multiplier = clamp(
        float(execution_minute_profile.get("size_multiplier", runtime.get("execution_minute_size_multiplier", 1.0)) or 1.0),
        0.70,
        1.35,
    )
    execution_minute_state = str(execution_minute_profile.get("state") or runtime.get("execution_minute_state") or "MIXED").upper()
    if event_playbook == "wait_then_retest":
        early_protect_r = max(early_protect_r, 0.22)
        stall_exit_bias = clamp(stall_exit_bias + 0.08, 0.0, 1.0)
    elif event_playbook in {"breakout", "risk_on_follow"}:
        early_protect_r = min(early_protect_r, 0.18)
        tp_extension_bias = clamp(tp_extension_bias + 0.06, 0.0, 0.50)
        reentry_bias = clamp(reentry_bias + 0.06, 0.0, 1.0)
    elif event_playbook == "swing_hold":
        trail_backoff_r = clamp(max(trail_backoff_r, 0.38), 0.18, 0.90)
        tp_extension_bias = clamp(tp_extension_bias + 0.08, 0.0, 0.50)
    if execution_minute_quality >= 0.78 and execution_minute_state == "CLEAN":
        early_protect_r = clamp(early_protect_r * 0.92, 0.12, 0.60)
        tp_extension_bias = clamp(tp_extension_bias + 0.04, 0.0, 0.50)
    elif execution_minute_quality <= 0.45 or execution_minute_state == "ROUGH":
        early_protect_r = clamp(max(early_protect_r, 0.24), 0.12, 0.60)
        trail_backoff_r = clamp(max(trail_backoff_r, 0.40), 0.18, 0.90)
        stall_exit_bias = clamp(stall_exit_bias + 0.10, 0.0, 1.0)
    if xau_grid_active:
        partials_supported = False
        if xau_grid_prime_session:
            early_protect_r = min(early_protect_r, 0.10)
            trail_backoff_r = clamp(min(trail_backoff_r, 0.16), 0.10, 0.90)
            tp_extension_bias = clamp(tp_extension_bias + 0.04, 0.0, 0.50)
            reentry_bias = clamp(reentry_bias + 0.10, 0.0, 1.0)
        elif xau_grid_asia_session:
            early_protect_r = min(early_protect_r, 0.14)
            trail_backoff_r = clamp(max(0.18, min(trail_backoff_r, 0.22)), 0.10, 0.90)
            tp_extension_bias = clamp(min(tp_extension_bias, 0.18), 0.0, 0.50)
            reentry_bias = clamp(min(reentry_bias, 0.55), 0.0, 1.0)
            stall_exit_bias = clamp(stall_exit_bias + 0.10, 0.0, 1.0)
        else:
            early_protect_r = min(early_protect_r, 0.12)
            trail_backoff_r = clamp(min(trail_backoff_r, 0.18), 0.10, 0.90)
            tp_extension_bias = clamp(tp_extension_bias + 0.05, 0.0, 0.50)
            reentry_bias = clamp(reentry_bias + 0.08, 0.0, 1.0)
        if 0.0 < account_equity < 300.0:
            early_protect_r = min(early_protect_r, 0.10)
            trail_backoff_r = clamp(min(trail_backoff_r, 0.14), 0.08, 0.90)
            tp_extension_bias = clamp(min(tp_extension_bias, 0.18), 0.0, 0.50)
        if float(payload.pnl_r) >= 0.18:
            stall_exit_bias = clamp(stall_exit_bias + 0.10, 0.0, 1.0)
    elif btc_fast_lock_active:
        partials_supported = False
        early_protect_r = min(early_protect_r, 0.08)
        trail_backoff_r = clamp(min(trail_backoff_r, 0.12), 0.08, 0.90)
        tp_extension_bias = clamp(min(tp_extension_bias, 0.22), 0.0, 0.50)
        if 0.0 < account_equity < 300.0:
            early_protect_r = min(early_protect_r, 0.06)
            trail_backoff_r = clamp(min(trail_backoff_r, 0.10), 0.06, 0.90)
            stall_exit_bias = clamp(stall_exit_bias + 0.08, 0.0, 1.0)
    spread_recovery_partial_trigger_r = float(base_decision.details.get("spread_recovery_partial_trigger_r", 0.0) or 0.0)
    spread_recovery_trail_unlock_r = float(base_decision.details.get("spread_recovery_trail_unlock_r", 0.0) or 0.0)
    spread_recovery_r = float(base_decision.details.get("spread_recovery_r", 0.0) or 0.0)
    btc_profit_lock_trigger_r = max(0.02, min(0.06, spread_recovery_r + 0.01))
    btc_profit_trail_trigger_r = max(0.06, min(0.12, spread_recovery_r + 0.03))
    xau_profit_lock_trigger_r = 0.08 if xau_grid_prime_session else 0.10
    xau_profit_trail_trigger_r = 0.16 if xau_grid_prime_session else 0.20
    watchlist_hit = _watchlist_matches(
        symbol_key,
        payload.setup,
        list(payload.weekly_reentry_watchlist or []) + list(payload.proven_reentry_queue or []),
    )
    local_details = {
        **dict(base_decision.details),
        "local_only_management": True,
        "learning_quota_catchup_pressure": float(trajectory_catchup_pressure),
        "watchlist_reentry_match": bool(watchlist_hit),
        "risk_reduction_active": bool(risk_reduction_active),
        "pair_aggression_multiplier": float(aggression_multiplier),
        "trade_horizon_bias": trade_horizon_bias,
        "management_directives": dict(management_directives),
        "watchdog_force": bool(watchdog_force),
        "watchdog_reason": str(watchdog_reason),
        "watchdog_stale_seconds": float(watchdog_stale_seconds),
        "event_playbook": str(event_playbook),
        "execution_minute_quality_score": float(execution_minute_quality),
        "execution_minute_size_multiplier": float(execution_minute_size_multiplier),
        "execution_minute_state": str(execution_minute_state),
        "microstructure_score": float(microstructure_score),
        "account_equity": float(account_equity),
        "partial_closes_supported": bool(partials_supported),
        "protected_no_loosen_enabled": bool(protected_no_loosen),
        "btc_profit_lock_trigger_r": float(btc_profit_lock_trigger_r),
        "btc_profit_trail_trigger_r": float(btc_profit_trail_trigger_r),
        "xau_profit_lock_trigger_r": float(xau_profit_lock_trigger_r),
        "xau_profit_trail_trigger_r": float(xau_profit_trail_trigger_r),
    }
    def finalize(decision: RetracementManagementDecision) -> RetracementManagementDecision:
        return _finalize_management_decision(
            decision,
            payload=payload,
            side=side,
            protected_lock_enabled=protected_no_loosen,
        )

    if not partials_supported and base_decision.management_action in {"PARTIAL_EXIT", "CLOSE_PARTIAL"}:
        base_decision = _rewrite_no_partial_exit(
            base_decision,
            payload=payload,
            risk_distance=risk_distance,
            side=side,
            details=local_details,
        )

    if (
        btc_fast_lock_active
        and base_decision.management_action == "HOLD"
        and float(payload.age_minutes) >= 0.80
        and float(payload.pnl_r) <= -0.12
        and float(payload.mfe_r) <= 0.10
        and (
            base_decision.continuation_score <= 0.44
            or base_decision.reversal_risk_score >= 0.50
            or bool(base_decision.stall_detected)
        )
    ):
        return finalize(replace(
            base_decision,
            management_action="FULL_EXIT",
            protection_mode="btc_fast_fail",
            trade_state="FORCE_EXIT",
            trade_quality="DEAD",
            close_fraction=0.0,
            reason="btc_fast_fail",
            details=local_details,
        ))

    if (
        btc_fast_lock_active
        and base_decision.management_action in {"HOLD", "TIGHTEN_STOP", "TRAIL_STOP"}
        and float(payload.mfe_r) >= max(0.10, btc_profit_lock_trigger_r + 0.03)
        and float(payload.pnl_r) <= max(-0.02, float(payload.mfe_r) - 0.08)
        and (
            bool(base_decision.stall_detected)
            or base_decision.continuation_score <= 0.48
            or base_decision.reversal_risk_score >= 0.50
        )
    ):
        return finalize(replace(
            base_decision,
            management_action="FULL_EXIT",
            protection_mode="btc_profit_giveback_exit",
            trade_state="FORCE_EXIT",
            trade_quality="WEAK",
            close_fraction=0.0,
            reason="btc_profit_giveback_exit",
            details=local_details,
        ))

    if (
        btc_fast_lock_active
        and base_decision.management_action in {"HOLD", "TIGHTEN_STOP", "TRAIL_STOP"}
        and btc_profit_lock_trigger_r <= float(payload.pnl_r) <= btc_profit_trail_trigger_r
        and (
            bool(base_decision.stall_detected)
            or base_decision.continuation_score <= 0.44
            or base_decision.reversal_risk_score >= 0.56
        )
    ):
        return finalize(replace(
            base_decision,
            management_action="FULL_EXIT",
            protection_mode="btc_micro_profit_recycle",
            trade_state="FORCE_EXIT",
            trade_quality="WEAK",
            close_fraction=0.0,
            reason="btc_micro_profit_recycle",
            details=local_details,
        ))

    if (
        btc_fast_lock_active
        and base_decision.management_action == "HOLD"
        and float(payload.pnl_r) >= btc_profit_lock_trigger_r
        and float(payload.pnl_r) < btc_profit_trail_trigger_r
        and not watchdog_force
    ):
        profit_lock_r = max(
            float(base_decision.profit_lock_r),
            min(max(spread_recovery_r + 0.01, float(payload.pnl_r) - 0.01), 0.10),
        )
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
        return finalize(replace(
            base_decision,
            management_action="TIGHTEN_STOP",
            protection_mode="btc_be_profit_lock",
            tighten_to_price=tighten_to,
            trade_state="PROTECTED",
            profit_lock_r=profit_lock_r,
            reason="btc_be_profit_lock",
            details=local_details,
        ))

    if (
        btc_fast_lock_active
        and base_decision.management_action in {"HOLD", "TIGHTEN_STOP"}
        and float(payload.pnl_r) >= btc_profit_trail_trigger_r
        and not watchdog_force
    ):
        ratchet_steps = max(1, int((max(0.0, float(payload.pnl_r) - 0.08)) // 0.15) + 1)
        profit_lock_r = max(
            float(base_decision.profit_lock_r),
            min(
                max(0.06, 0.06 + ((ratchet_steps - 1) * 0.12)),
                max(0.06, float(payload.pnl_r) - 0.04),
            ),
        )
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
        return finalize(replace(
            base_decision,
            management_action="TRAIL_STOP",
            protection_mode="btc_ratchet_trail",
            tighten_to_price=tighten_to,
            trade_state="RUNNER" if (float(payload.pnl_r) >= 0.60 and mc_win_rate >= 0.85 and microstructure_score >= 0.54) else "PROTECTED",
            profit_lock_r=profit_lock_r,
            reason="btc_ratchet_trail",
            details={**local_details, "btc_ratchet_steps": int(ratchet_steps), "trail_backoff_r": float(trail_backoff_r)},
        ))

    if (
        xau_grid_active
        and base_decision.management_action == "HOLD"
        and float(payload.pnl_r) <= (-0.18 if xau_grid_prime_session else -0.22)
        and (
            base_decision.trade_state in {"EXIT_READY", "FORCE_EXIT"}
            or base_decision.continuation_score <= 0.46
            or base_decision.reversal_risk_score >= 0.52
        )
    ):
        return finalize(replace(
            base_decision,
            management_action="FULL_EXIT",
            protection_mode="xau_grid_fast_fail",
            trade_state="FORCE_EXIT",
            trade_quality="DEAD",
            close_fraction=0.0,
            reason="xau_grid_fast_fail",
            details=local_details,
        ))

    if (
        xau_grid_active
        and base_decision.management_action in {"HOLD", "TIGHTEN_STOP", "TRAIL_STOP"}
        and float(payload.mfe_r) >= (0.12 if xau_grid_prime_session else 0.14)
        and float(payload.pnl_r) <= max(-0.02, float(payload.mfe_r) - (0.08 if xau_grid_prime_session else 0.10))
        and (
            bool(base_decision.stall_detected)
            or base_decision.continuation_score <= 0.46
            or base_decision.reversal_risk_score >= 0.50
        )
    ):
        return finalize(replace(
            base_decision,
            management_action="FULL_EXIT",
            protection_mode="xau_grid_profit_giveback_exit",
            trade_state="FORCE_EXIT",
            trade_quality="WEAK",
            close_fraction=0.0,
            reason="xau_grid_profit_giveback_exit",
            details=local_details,
        ))

    if (
        xau_grid_active
        and base_decision.management_action == "HOLD"
        and float(payload.age_minutes) >= (0.85 if xau_grid_prime_session else 1.20)
        and float(payload.pnl_r) <= (-0.08 if xau_grid_prime_session else -0.10)
        and float(payload.mfe_r) <= (0.12 if xau_grid_prime_session else 0.10)
        and (
            base_decision.continuation_score <= 0.42
            or base_decision.reversal_risk_score >= 0.50
            or bool(base_decision.stall_detected)
        )
    ):
        return finalize(replace(
            base_decision,
            management_action="FULL_EXIT",
            protection_mode="xau_grid_weak_launch_fail",
            trade_state="FORCE_EXIT",
            trade_quality="DEAD",
            close_fraction=0.0,
            reason="xau_grid_weak_launch_fail",
            details=local_details,
        ))

    if (
        xau_grid_active
        and xau_grid_prime_session
        and base_decision.management_action == "HOLD"
        and float(payload.age_minutes) >= 0.70
        and float(payload.pnl_r) <= 0.04
        and (
            bool(base_decision.stall_detected)
            or base_decision.continuation_score <= 0.40
            or base_decision.reversal_risk_score >= 0.54
        )
    ):
        return finalize(replace(
            base_decision,
            management_action="FULL_EXIT",
            protection_mode="xau_grid_prime_recycle_exit",
            trade_state="FORCE_EXIT",
            trade_quality="WEAK",
            close_fraction=0.0,
            reason="xau_grid_prime_recycle_exit",
            details=local_details,
        ))

    if (
        xau_grid_active
        and base_decision.management_action in {"HOLD", "TIGHTEN_STOP", "TRAIL_STOP"}
        and 0.10 <= float(payload.pnl_r) <= 0.24
        and (
            bool(base_decision.stall_detected)
            or base_decision.continuation_score <= 0.38
            or base_decision.reversal_risk_score >= 0.58
        )
    ):
        return finalize(replace(
            base_decision,
            management_action="FULL_EXIT",
            protection_mode="xau_grid_micro_profit_recycle",
            trade_state="FORCE_EXIT",
            trade_quality="WEAK",
            close_fraction=0.0,
            reason="xau_grid_micro_profit_recycle",
            details=local_details,
        ))

    if (
        symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD"}
        and "BREAKOUT" in str(payload.setup or "").upper()
        and base_decision.management_action == "HOLD"
        and float(payload.age_minutes) >= 1.0
        and float(payload.pnl_r) <= -0.22
        and (
            base_decision.trade_state in {"EXIT_READY", "FORCE_EXIT"}
            or base_decision.continuation_score <= 0.48
            or base_decision.reversal_risk_score >= 0.48
            or float(payload.mfe_r) <= 0.16
        )
    ):
        return finalize(replace(
            base_decision,
            management_action="FULL_EXIT",
            protection_mode="breakout_fast_fail",
            trade_state="FORCE_EXIT",
            trade_quality="DEAD",
            close_fraction=0.0,
            reason="breakout_fast_fail",
            details=local_details,
        ))

    if (
        not xau_grid_active
        and (
        not bool(payload.spread_recovery_partial_taken)
        and float(payload.volume) >= max(float(payload.min_lot) * 2.0, float(payload.min_lot) + 1e-9)
        and float(payload.pnl_r) >= spread_recovery_partial_trigger_r
        and float(payload.pnl_r) < 0.95
        )
    ):
        partial_decision = replace(
            base_decision,
            management_action="PARTIAL_EXIT",
            protection_mode="spread_recovery_partial",
            close_fraction=0.40,
            trade_state="PROTECTED" if float(payload.pnl_r) < 1.0 else "RUNNER",
            reason="spread_recovery_partial",
            details=local_details,
        )
        if partials_supported:
            return finalize(partial_decision)
        return finalize(_rewrite_no_partial_exit(
            partial_decision,
            payload=payload,
            risk_distance=risk_distance,
            side=side,
            details=local_details,
        ))

    if (
        not xau_grid_active
        and (
        not bool(payload.runner_partial_taken)
        and float(payload.volume) >= max(float(payload.min_lot) * 2.0, float(payload.min_lot) + 1e-9)
        and float(payload.pnl_r) >= 1.0
        and base_decision.continuation_score >= 0.50
        and base_decision.management_action in {"HOLD", "TIGHTEN_STOP", "TRAIL_STOP", "EXTEND_TP"}
        )
    ):
        partial_decision = replace(
            base_decision,
            management_action="PARTIAL_EXIT",
            protection_mode="runner_scale_out",
            close_fraction=0.30,
            trade_state="RUNNER",
            reason="runner_scale_out_partial",
            details=local_details,
        )
        if partials_supported:
            return finalize(partial_decision)
        return finalize(_rewrite_no_partial_exit(
            partial_decision,
            payload=payload,
            risk_distance=risk_distance,
            side=side,
            details=local_details,
        ))

    if (
        float(payload.pnl_r) >= max(0.55, spread_recovery_trail_unlock_r)
        and base_decision.management_action == "HOLD"
        and base_decision.continuation_score >= 0.48
    ):
        profit_lock_r = max(float(base_decision.profit_lock_r), 0.30)
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
        return finalize(replace(
            base_decision,
            management_action="TIGHTEN_STOP",
            protection_mode="spread_recovery_be_buffer",
            tighten_to_price=tighten_to,
            trade_state="PROTECTED",
            profit_lock_r=profit_lock_r,
            reason="spread_recovery_be_buffer",
            details=local_details,
        ))

    if xau_grid_active:
        if (
            base_decision.management_action == "HOLD"
            and float(payload.pnl_r) >= xau_profit_lock_trigger_r
            and float(payload.pnl_r) < xau_profit_trail_trigger_r
            and not watchdog_force
        ):
            profit_lock_r = max(
                float(base_decision.profit_lock_r),
                min(max(0.04, float(payload.pnl_r) - 0.015), 0.10),
            )
            tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
            return finalize(replace(
                base_decision,
                management_action="TIGHTEN_STOP",
                protection_mode="xau_grid_be_profit_lock",
                tighten_to_price=tighten_to,
                trade_state="PROTECTED",
                profit_lock_r=profit_lock_r,
                reason="xau_grid_be_profit_lock",
                details=local_details,
            ))

        if (
            base_decision.management_action in {"HOLD", "TIGHTEN_STOP"}
            and float(payload.pnl_r) >= xau_profit_trail_trigger_r
            and not bool(base_decision.stall_detected)
            and not watchdog_force
            and (
                not xau_grid_prime_session
                or float(payload.pnl_r) >= xau_profit_trail_trigger_r
                or (
                    microstructure_score >= 0.56
                    and base_decision.continuation_score >= 0.54
                )
            )
        ):
            ratchet_steps = max(1, int((max(0.0, float(payload.pnl_r) - xau_profit_lock_trigger_r)) // 0.16) + 1)
            profit_lock_r = max(
                float(base_decision.profit_lock_r),
                min(
                    max(0.04, 0.04 + ((ratchet_steps - 1) * 0.14)),
                    max(0.04, float(payload.pnl_r) - 0.03),
                ),
            )
            tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
            return finalize(replace(
                base_decision,
                management_action="TRAIL_STOP",
                protection_mode="xau_grid_ratchet_trail",
                tighten_to_price=tighten_to,
                trade_state="RUNNER" if float(payload.pnl_r) >= 0.60 else "PROTECTED",
                profit_lock_r=profit_lock_r,
                reason="xau_grid_ratchet_trail",
                details={**local_details, "xau_grid_ratchet_steps": int(ratchet_steps), "trail_backoff_r": float(trail_backoff_r)},
            ))

        if (
            base_decision.management_action in {"HOLD", "TIGHTEN_STOP"}
            and float(payload.pnl_r) >= (0.18 if xau_grid_prime_session else 0.24)
            and float(payload.age_minutes) >= (0.08 if xau_grid_prime_session else 0.24)
            and float(payload.mfe_r) >= (0.20 if xau_grid_prime_session else 0.30)
            and base_decision.continuation_score >= (0.48 if xau_grid_prime_session else 0.50)
            and base_decision.reversal_risk_score <= (0.44 if xau_grid_prime_session else 0.42)
            and not bool(base_decision.stall_detected)
        ):
            profit_lock_r = max(
                float(base_decision.profit_lock_r),
                max(0.10, float(payload.pnl_r) - (0.10 if xau_grid_prime_session else 0.12)),
            )
            tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
            return finalize(replace(
                base_decision,
                management_action="TRAIL_STOP",
                protection_mode="xau_grid_capture_trail",
                tighten_to_price=tighten_to,
                trade_state="RUNNER" if float(payload.pnl_r) >= 0.70 else "PROTECTED",
                profit_lock_r=profit_lock_r,
                reason="xau_grid_capture_trail",
                details={**local_details, "trail_backoff_r": float(min(trail_backoff_r, 0.12 if xau_grid_prime_session else 0.16))},
            ))

        if (
            payload.enable_adaptive_tp
            and xau_grid_prime_session
            and base_decision.management_action in {"HOLD", "TIGHTEN_STOP", "TRAIL_STOP"}
            and float(payload.pnl_r) >= 0.50
            and mc_win_rate >= (0.88 if 0.0 < account_equity < 300.0 else 0.85)
            and microstructure_score >= 0.62
            and base_decision.continuation_score >= 0.62
            and base_decision.reversal_risk_score <= 0.34
        ):
            extension_r = clamp(0.16 + tp_extension_bias, 0.12, 0.34)
            updated_tp = (
                float(payload.tp) + (risk_distance * extension_r)
                if side == "BUY"
                else float(payload.tp) - (risk_distance * extension_r)
            )
            profit_lock_r = max(float(base_decision.profit_lock_r), max(0.18, float(payload.pnl_r) - 0.24))
            tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
            return finalize(replace(
                base_decision,
                management_action="EXTEND_TP",
                protection_mode="xau_grid_capture_extension",
                updated_tp_price=updated_tp,
                tighten_to_price=tighten_to,
                trade_state="RUNNER",
                trade_quality="STRONG",
                profit_lock_r=profit_lock_r,
                reason="xau_grid_capture_extension",
                details={**local_details, "xau_grid_extension_r": float(extension_r), "microstructure_score": float(microstructure_score)},
            ))

        if (
            bool(base_decision.stall_detected)
            and float(payload.pnl_r) >= 0.10
            and float(payload.age_minutes) >= (1.2 if xau_grid_prime_session else 2.0)
            and base_decision.management_action == "HOLD"
        ):
            profit_lock_r = max(float(base_decision.profit_lock_r), max(0.04, float(payload.pnl_r) - 0.08))
            tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
            return finalize(replace(
                base_decision,
                management_action="TIGHTEN_STOP",
                protection_mode="xau_grid_stall_profit_lock",
                tighten_to_price=tighten_to,
                trade_state="PROTECTED",
                profit_lock_r=profit_lock_r,
                reason="xau_grid_stall_profit_lock",
                details=local_details,
            ))

    if float(payload.pnl_r) >= 1.0 and base_decision.management_action == "HOLD":
        profit_lock_r = max(float(base_decision.profit_lock_r), 0.35)
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
        return finalize(replace(
            base_decision,
            management_action="TIGHTEN_STOP",
            protection_mode="profit_guard_default",
            tighten_to_price=tighten_to,
            trade_state="PROTECTED",
            profit_lock_r=profit_lock_r,
            reason="profit_guard_default",
            details=local_details,
        ))

    risk_reduction_trigger_r = 0.34 if symbol_key in {"XAUUSD", "BTCUSD", "NAS100", "USDJPY"} else 0.25
    risk_reduction_min_age = 4.0 if symbol_key in {"XAUUSD", "BTCUSD", "NAS100", "USDJPY"} else 0.0
    if (
        risk_reduction_active
        and float(payload.pnl_r) >= risk_reduction_trigger_r
        and float(payload.age_minutes) >= risk_reduction_min_age
        and base_decision.management_action == "HOLD"
    ):
        profit_lock_r = max(float(base_decision.profit_lock_r), 0.20)
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
        return finalize(replace(
            base_decision,
            management_action="TIGHTEN_STOP",
            protection_mode="learning_brain_risk_reduction",
            tighten_to_price=tighten_to,
            trade_state="PROTECTED",
            profit_lock_r=profit_lock_r,
            reason="learning_brain_risk_reduction",
            details=local_details,
        ))

    if (
        watchdog_force
        and float(payload.pnl_r) >= 0.15
        and base_decision.management_action == "HOLD"
    ):
        profit_lock_r = max(
            float(base_decision.profit_lock_r),
            max(0.05, float(payload.pnl_r) - (0.12 if symbol_key == "XAUUSD" else 0.18)),
        )
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
        return finalize(replace(
            base_decision,
            management_action="TIGHTEN_STOP",
            protection_mode="watchdog_profit_protect",
            tighten_to_price=tighten_to,
            trade_state="PROTECTED" if float(payload.pnl_r) < 0.80 else "RUNNER",
            profit_lock_r=profit_lock_r,
            reason=watchdog_reason or "watchdog_profit_protect",
            details=local_details,
        ))

    if (
        float(payload.pnl_r) >= early_protect_r
        and base_decision.management_action == "HOLD"
        and base_decision.continuation_score >= 0.42
        and not (
            xau_grid_prime_session
            and not watchdog_force
            and float(payload.pnl_r) < 0.40
            and base_decision.continuation_score >= 0.56
            and microstructure_score >= 0.58
        )
    ):
        profit_lock_r = max(
            float(base_decision.profit_lock_r),
            max(0.06, float(payload.pnl_r) - max(0.18, trail_backoff_r * 0.55)),
        )
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
        return finalize(replace(
            base_decision,
            management_action="TIGHTEN_STOP",
            protection_mode="ultra_ratchet_early_protect",
            tighten_to_price=tighten_to,
            trade_state="PROTECTED" if float(payload.pnl_r) < 1.0 else "RUNNER",
            profit_lock_r=profit_lock_r,
            reason="ultra_ratchet_early_protect",
            details=local_details,
        ))

    if (
        float(payload.pnl_r) >= max(early_protect_r + 0.10, 0.32)
        and base_decision.management_action in {"HOLD", "TIGHTEN_STOP"}
        and base_decision.continuation_score >= (0.44 if symbol_key == "XAUUSD" else 0.48)
        and not bool(base_decision.stall_detected)
    ):
        profit_lock_r = max(
            float(base_decision.profit_lock_r),
            max(0.10, float(payload.pnl_r) - trail_backoff_r),
        )
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
        return finalize(replace(
            base_decision,
            management_action="TRAIL_STOP",
            protection_mode="ultra_ratchet_trail",
            tighten_to_price=tighten_to,
            trade_state="RUNNER" if float(payload.pnl_r) >= 0.85 else "PROTECTED",
            profit_lock_r=profit_lock_r,
            reason="ultra_ratchet_trail",
            details={**local_details, "trail_backoff_r": trail_backoff_r},
        ))

    if (
        payload.enable_adaptive_tp
        and base_decision.management_action in {"HOLD", "TIGHTEN_STOP", "TRAIL_STOP"}
        and mc_win_rate >= max(0.82, 0.88 - (tp_extension_bias * 0.20))
        and fractal_persistence >= 0.62
        and dxy_support >= 0.55
        and base_decision.continuation_score >= (0.58 if aggression_multiplier >= 1.10 else 0.60)
        and base_decision.reversal_risk_score <= 0.40
    ):
        swing_extension_r = 0.60 if symbol_key in {"NAS100", "USOIL"} else 0.52 if symbol_key in _SUPER_AGGRESSIVE_SWING_SYMBOLS else 0.42
        if trade_horizon_bias == "swing":
            swing_extension_r += 0.10
        elif trade_horizon_bias == "daytrade":
            swing_extension_r += 0.05
        swing_extension_r += tp_extension_bias
        updated_tp = (
            float(payload.tp) + (risk_distance * swing_extension_r)
            if side == "BUY"
            else float(payload.tp) - (risk_distance * swing_extension_r)
        )
        profit_lock_r = max(float(base_decision.profit_lock_r), max(0.30, float(payload.pnl_r) - 0.72))
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
        return finalize(replace(
            base_decision,
            management_action="EXTEND_TP",
            protection_mode="mc_fractal_swing_extension",
            updated_tp_price=updated_tp,
            tighten_to_price=tighten_to,
            trade_state="SWING",
            trade_quality="STRONG",
            profit_lock_r=profit_lock_r,
            reason="mc_fractal_swing_extension",
            details={**local_details, "swing_extension_r": swing_extension_r},
        ))

    if (
        stall_lane
        and float(payload.pnl_r) >= 0.10
        and float(payload.age_minutes) >= 12.0
        and base_decision.continuation_score <= 0.36
        and stall_exit_bias >= 0.55
        and base_decision.management_action == "HOLD"
    ):
        profit_lock_r = max(float(base_decision.profit_lock_r), max(0.04, float(payload.pnl_r) - 0.10))
        tighten_to = _protect_price(float(payload.entry_price), risk_distance, side, profit_lock_r)
        return finalize(replace(
            base_decision,
            management_action="TIGHTEN_STOP",
            protection_mode="stall_exit_bias_lock",
            tighten_to_price=tighten_to,
            trade_state="PROTECTED",
            profit_lock_r=profit_lock_r,
            reason="stall_exit_bias_lock",
            details=local_details,
        ))

    if watchlist_hit:
        return finalize(replace(
            base_decision,
            details={
                **local_details,
                "watchlist_reentry_ready": bool(mc_win_rate >= max(0.80, 0.88 - (reentry_bias * 0.10)) and fractal_persistence >= 0.60),
            },
        ))

    return finalize(replace(base_decision, details=local_details))
