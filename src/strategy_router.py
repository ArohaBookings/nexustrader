from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.execution import trading_day_key_for_timestamp
from src.regime_detector import RegimeClassification
from src.session_calendar import is_weekend_market_mode
from src.session_profile import SessionContext
from src.symbol_universe import symbol_asset_class
from src.strategies.trend_daytrade import resolve_strategy_key
from src.strategy_engine import SignalCandidate, StrategyEngine
from src.trade_quality import (
    compression_strategy_bias,
    delta_proxy_score,
    infer_trade_lane,
    is_xau_grid_lane,
    normalize_strategy_family,
    pair_behavior_fit,
    quality_tier_from_scores,
    quality_tier_size_multiplier,
    runtime_regime_state,
    session_loosen_factor,
    session_priority_context,
    strategy_allowed_regimes,
    strategy_management_template,
    strategy_regime_fit,
    strategy_selection_score,
    structure_cleanliness_score,
    entry_timing_score,
    winner_promotion_bonus,
    xau_attack_lane_category,
    xau_grid_lane_for_session,
)
from src.utils import clamp, deterministic_id


@dataclass
class StrategyRouter:
    max_spread_points: float = 60.0
    xau_active_sessions: tuple[str, ...] = ("SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK")
    xau_m1_enabled: bool = True
    xau_m15_enabled: bool = True
    xau_m1_min_impulse_ratio: float = 0.48
    xau_m1_min_volume_ratio: float = 1.00
    xau_m1_confluence_floor: float = 3.0
    xau_m15_confluence_floor: float = 4.0
    xau_m15_breakout_atr_threshold: float = 0.20
    forex_sensitivity: float = 1.0
    nas_sensitivity: float = 1.0
    oil_sensitivity: float = 1.0
    btc_sensitivity: float = 1.0
    btc_trade_sessions: tuple[str, ...] = ("TOKYO", "LONDON", "OVERLAP", "NEW_YORK")
    btc_allowed_start_hour_utc: int = 0
    btc_allowed_end_hour_utc: int = 24
    btc_spread_cap_points: float = 2500.0
    btc_min_ai_confidence: float = 0.56
    btc_volatility_pause_move_pct_30m: float = 0.015
    btc_funding_rate_min_abs: float = 0.001
    btc_liquidation_usd_threshold: float = 10_000_000.0
    btc_whale_flow_threshold_btc: float = 5000.0
    btc_dxy_move_threshold: float = 0.003
    btc_weekend_gap_min_pct: float = 0.02
    b_tier_size_mult_min: float = 0.70
    b_tier_size_mult_max: float = 0.90
    high_liquidity_loosen_pct: float = 0.30
    candidate_scarcity_loosen_pct: float = 0.12
    recycle_regime_boost: float = 0.25
    recycle_max_per_family_per_day: int = 1
    family_rotation_window_trades: int = 20
    family_rotation_share_threshold: float = 0.60
    family_rotation_score_penalty: float = 0.20
    compression_burst_candidates: int = 3
    compression_burst_size_multiplier: float = 1.20
    xau_m5_burst_target: int = 8
    xau_prime_session_mult: float = 2.50
    transition_score_bonus: float = 0.35
    transition_size_multiplier: float = 1.20
    btc_weekend_score_bonus: float = 0.50
    btc_weekend_size_boost: float = 1.20
    btc_weekend_burst_target: int = 40
    all_pairs_aggression: float = 1.20
    btc_velocity_decay_trigger_trades_per_10_bars: float = 1.50
    xau_grid_compression_spacing_multiplier: float = 1.30
    xau_grid_expansion_burst_size_multiplier: float = 1.25
    adaptive_sensitivity: dict[str, float] = field(default_factory=dict)
    density_profiles: dict[str, dict[str, Any]] = field(default_factory=dict)
    symbol_spread_caps: dict[str, float] = field(default_factory=dict)
    recycle_queue: dict[str, dict[str, Any]] = field(default_factory=dict)
    compression_burst_state: dict[str, dict[str, Any]] = field(default_factory=dict)
    learning_brain_bundle: dict[str, Any] = field(default_factory=dict)
    learning_brain_projection: dict[str, float] = field(default_factory=dict)
    learning_symbol: str = ""
    learning_pair_directive: dict[str, Any] = field(default_factory=dict)
    learning_reentry_watchlist: tuple[str, ...] = field(default_factory=tuple)
    learning_weekly_trade_ideas: tuple[str, ...] = field(default_factory=tuple)
    btc_heartbeat_cadence_seconds: int = 60
    btc_heartbeat_emit_state: dict[str, datetime] = field(default_factory=dict)

    @staticmethod
    def _fast_execution_lane_active(symbol: str, session_name: str, weekend_mode: bool = False) -> bool:
        symbol_key = str(symbol or "").upper()
        session_key = str(session_name or "").upper()
        if symbol_key == "XAUUSD":
            return session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
        if symbol_key == "NAS100":
            return session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
        if symbol_key == "BTCUSD":
            return bool(weekend_mode) or session_key in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}
        if symbol_key == "USDJPY":
            return session_key in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP"}
        return False

    @staticmethod
    def _resolve_xau_attack_lane(session_name: str, candidate: SignalCandidate) -> str:
        candidate_meta = dict(candidate.meta or {})
        entry_profile = str(candidate_meta.get("grid_entry_profile") or "").strip().lower()
        reentry_tag = str(candidate_meta.get("reentry_source_tag") or "").strip().lower()
        category = "ATTACK"
        if (
            reentry_tag in {"prime_recovery_repeat", "follow_through_reentry"}
            or bool(candidate_meta.get("prime_session_recovery_active", False))
            or bool(candidate_meta.get("quota_reclaim_rescue_selected", False))
        ):
            category = "REENTRY"
        elif entry_profile.startswith(("grid_liquidity_reclaim", "grid_trend_reclaim", "grid_m15_pullback_reclaim")):
            category = "RECLAIM"
        elif entry_profile.startswith(("grid_breakout_reclaim", "grid_expansion_ready_scaler")) or any(
            token in entry_profile for token in ("breakout", "expansion")
        ):
            category = "BREAKOUT"
        return xau_grid_lane_for_session(session_name, category=category)

    def apply_learning_policy(self, policy: dict[str, Any] | None) -> None:
        payload = dict(policy or {})
        self.learning_symbol = self._normalize_symbol(str(payload.get("symbol") or ""))
        self.learning_brain_bundle = dict(payload.get("bundle") or {})
        self.learning_brain_projection = dict(payload.get("trajectory_projection") or {})
        self.learning_pair_directive = dict(payload.get("pair_directive") or {})
        self.learning_reentry_watchlist = tuple(
            str(item).upper()
            for item in payload.get("reentry_watchlist", [])
            if str(item).strip()
        )
        self.learning_weekly_trade_ideas = tuple(
            str(item).upper()
            for item in payload.get("weekly_trade_ideas", [])
            if str(item).strip()
        )

    @staticmethod
    def _hour_fraction(timestamp) -> float:
        current = StrategyRouter._utc_timestamp(timestamp)
        return float(current.hour) + (float(current.minute) / 60.0) + (float(current.second) / 3600.0)

    @staticmethod
    def _learning_native_sessions(symbol: str) -> set[str]:
        symbol_key = StrategyRouter._normalize_symbol(symbol)
        if symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD", "USDJPY", "EURJPY", "GBPJPY"}:
            return {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}
        if symbol_key in {"EURUSD", "GBPUSD", "EURGBP", "NAS100", "USOIL"}:
            return {"LONDON", "OVERLAP", "NEW_YORK"}
        if symbol_key == "XAUUSD":
            return {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}
        if symbol_key == "BTCUSD":
            return {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}
        return {"LONDON", "OVERLAP", "NEW_YORK"}

    def _learning_pattern_state(
        self,
        *,
        symbol: str,
        strategy_key: str = "",
        session_name: str = "",
    ) -> dict[str, Any]:
        bundle = dict(self.learning_brain_bundle or {})
        symbol_key = self._normalize_symbol(symbol)
        strategy_upper = str(strategy_key or "").strip().upper()
        promoted_patterns = {
            str(item).strip().upper()
            for item in bundle.get("promoted_patterns", [])
            if str(item).strip()
        }
        weak_pair_focus = {
            self._normalize_symbol(item)
            for item in bundle.get("weak_pair_focus", [])
            if str(item).strip()
        }
        shadow_pair_focus = {
            self._normalize_symbol(item)
            for item in bundle.get("shadow_pair_focus", [])
            if str(item).strip()
        }
        session_key = str(session_name or "").upper()
        watchlist_items = set(self.learning_reentry_watchlist) | set(self.learning_weekly_trade_ideas)
        watchlist_match = any(
            symbol_key and (symbol_key in item or (strategy_upper and strategy_upper in item))
            for item in watchlist_items
        )
        pair_directive = dict(self.learning_pair_directive or {})
        frequency_directives = dict(pair_directive.get("frequency_directives") or {})
        trajectory_catchup_pressure = clamp(float(bundle.get("quota_catchup_pressure", 0.0) or 0.0), 0.0, 1.0)
        trajectory_catchup_pressure = max(
            trajectory_catchup_pressure,
            clamp(float(frequency_directives.get("catchup_pressure", 0.0) or 0.0), 0.0, 1.0),
        )
        native_session = session_key in self._learning_native_sessions(symbol_key)
        directive_sessions = {
            str(item).upper()
            for item in pair_directive.get("session_focus", [])
            if str(item).strip()
        }
        if directive_sessions and session_key in directive_sessions:
            native_session = True
        promoted_pattern = bool(strategy_upper and strategy_upper in promoted_patterns)
        weak_focus = bool(symbol_key and (symbol_key in weak_pair_focus or symbol_key in shadow_pair_focus))
        aggression_multiplier = clamp(float(pair_directive.get("aggression_multiplier", 1.0) or 1.0), 0.75, 1.80)
        shadow_experiment_active = bool(pair_directive.get("shadow_experiment_active", False))
        slippage_regime = str(pair_directive.get("slippage_regime") or "").lower()
        soft_burst_target_10m = max(1, int(frequency_directives.get("soft_burst_target_10m", 0) or 0))
        quota_boost_allowed = bool(frequency_directives.get("quota_boost_allowed", False))
        aggressive_reentry_enabled = bool(frequency_directives.get("aggressive_reentry_enabled", False))
        undertrade_fix_mode = bool(frequency_directives.get("undertrade_fix_mode", False))
        gpt_hybrid_advisory = dict(pair_directive.get("gpt_hybrid_advisory") or {})
        gpt_hybrid_sessions = {
            str(item).upper()
            for item in gpt_hybrid_advisory.get("session_focus", [])
            if str(item).strip()
        }
        gpt_hybrid_conviction = clamp(float(gpt_hybrid_advisory.get("conviction", 0.0) or 0.0), 0.0, 1.0)
        gpt_hybrid_aggression_delta = clamp(float(gpt_hybrid_advisory.get("aggression_delta", 0.0) or 0.0), -0.10, 0.25)
        gpt_hybrid_threshold_delta = clamp(float(gpt_hybrid_advisory.get("threshold_delta", 0.0) or 0.0), -0.08, 0.05)
        gpt_hybrid_active = bool(gpt_hybrid_advisory.get("enabled", False))
        gpt_hybrid_session_match = bool(gpt_hybrid_active and (not gpt_hybrid_sessions or session_key in gpt_hybrid_sessions))
        base_native_aggression = clamp(float(self.all_pairs_aggression or 1.0), 1.0, 1.40)
        asia_native_pair = symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD", "USDJPY", "EURJPY", "GBPJPY"}
        xau_prime_session = symbol_key == "XAUUSD" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
        xau_asia_grid = symbol_key == "XAUUSD" and session_key in {"TOKYO", "SYDNEY"} and "GRID" in strategy_upper
        btc_priority_window = symbol_key == "BTCUSD" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
        throughput_recovery_active = bool(
            trajectory_catchup_pressure >= 0.60 and (promoted_pattern or weak_focus or watchlist_match)
        )
        if not throughput_recovery_active and trajectory_catchup_pressure >= 0.82 and (
            native_session or not session_key
        ):
            throughput_recovery_active = True
        rank_bonus = 0.0
        if promoted_pattern:
            rank_bonus += 0.05
        if watchlist_match:
            rank_bonus += 0.03
        if weak_focus and trajectory_catchup_pressure >= 0.50:
            rank_bonus += 0.02
        if native_session and trajectory_catchup_pressure >= 0.70:
            rank_bonus += 0.02
        if asia_native_pair and native_session and session_key in {"SYDNEY", "TOKYO"}:
            rank_bonus += 0.03
        if throughput_recovery_active and (native_session or not str(session_name or "").strip()) and trajectory_catchup_pressure >= 0.82:
            rank_bonus += 0.02
        if quota_boost_allowed and (native_session or watchlist_match or promoted_pattern):
            rank_bonus += min(0.03, 0.01 + (trajectory_catchup_pressure * 0.03))
        size_bonus = 0.0
        if promoted_pattern and native_session:
            size_bonus += 0.06
        if watchlist_match:
            size_bonus += 0.03
        if weak_focus and trajectory_catchup_pressure >= 0.75:
            size_bonus += 0.02
        if throughput_recovery_active and (native_session or not str(session_name or "").strip()) and trajectory_catchup_pressure >= 0.82:
            size_bonus += 0.02
        if quota_boost_allowed and (native_session or promoted_pattern):
            size_bonus += min(0.04, 0.01 + (trajectory_catchup_pressure * 0.04))
        if aggression_multiplier > 1.0 and (promoted_pattern or native_session or watchlist_match):
            rank_bonus += min(0.04, (aggression_multiplier - 1.0) * 0.10)
            size_bonus += min(0.05, (aggression_multiplier - 1.0) * 0.12)
        if asia_native_pair and native_session and session_key in {"SYDNEY", "TOKYO"}:
            rank_bonus += 0.02
            size_bonus += 0.03
            if undertrade_fix_mode or soft_burst_target_10m >= 5:
                rank_bonus += 0.02
                size_bonus += 0.03
        if xau_prime_session:
            rank_bonus += min(0.05, max(0.0, float(self.xau_prime_session_mult) - 1.0) * 0.04)
            if throughput_recovery_active or promoted_pattern or watchlist_match:
                size_bonus += min(0.06, max(0.0, float(self.xau_prime_session_mult) - 1.0) * 0.05)
            if soft_burst_target_10m >= 8:
                rank_bonus += 0.02
                size_bonus += 0.02
        elif xau_asia_grid:
            rank_bonus = max(0.0, rank_bonus - 0.01)
            size_bonus = max(0.0, size_bonus - 0.01)
        elif symbol_key == "XAUUSD" and session_key in {"TOKYO", "SYDNEY"}:
            rank_bonus += 0.01 if soft_burst_target_10m >= 2 else 0.0
        elif native_session and symbol_key != "XAUUSD" and base_native_aggression > 1.0:
            rank_bonus += min(0.03, (base_native_aggression - 1.0) * 0.12)
            if promoted_pattern or watchlist_match or throughput_recovery_active:
                size_bonus += min(0.04, (base_native_aggression - 1.0) * 0.14)
        if btc_priority_window and (promoted_pattern or watchlist_match or throughput_recovery_active):
            weekend_pressure = max(0.0, float(self.btc_weekend_burst_target) - 20.0)
            rank_bonus += min(0.04, weekend_pressure / 400.0)
            size_bonus += min(0.05, weekend_pressure / 350.0)
            if soft_burst_target_10m >= 6:
                rank_bonus += 0.02
                size_bonus += 0.02
        if shadow_experiment_active and (native_session or promoted_pattern or watchlist_match):
            rank_bonus += 0.02
            size_bonus += 0.02
        if aggressive_reentry_enabled and (native_session or throughput_recovery_active):
            rank_bonus += 0.01
        if undertrade_fix_mode and (native_session or symbol_key in {"XAUUSD", "BTCUSD"}):
            rank_bonus += 0.02
            size_bonus += 0.02
        if slippage_regime == "clean" and (native_session or watchlist_match):
            rank_bonus += 0.01
        elif slippage_regime == "rough":
            rank_bonus = max(0.0, rank_bonus - 0.02)
        if gpt_hybrid_session_match and gpt_hybrid_conviction >= 0.45:
            rank_bonus += min(0.03, 0.01 + (gpt_hybrid_conviction * 0.03) + max(0.0, gpt_hybrid_aggression_delta * 0.05))
            if native_session or watchlist_match or throughput_recovery_active:
                size_bonus += min(0.04, 0.01 + (gpt_hybrid_conviction * 0.04) + max(0.0, gpt_hybrid_aggression_delta * 0.06))
        return {
            "symbol_focus": bool(symbol_key and symbol_key == self.learning_symbol),
            "promoted_pattern": bool(promoted_pattern),
            "weak_focus": bool(weak_focus),
            "watchlist_match": bool(watchlist_match),
            "shadow_focus": bool(weak_focus and not promoted_pattern),
            "native_session": bool(native_session),
            "xau_prime_session": bool(xau_prime_session),
            "btc_priority_window": bool(btc_priority_window),
            "throughput_recovery_active": bool(throughput_recovery_active),
            "trajectory_catchup_pressure": float(trajectory_catchup_pressure),
            "trade_horizon_bias": str(pair_directive.get("trade_horizon_bias") or ""),
            "aggression_multiplier": float(aggression_multiplier),
            "shadow_experiment_active": bool(shadow_experiment_active),
            "slippage_regime": str(slippage_regime),
            "min_confluence_override": float(pair_directive.get("min_confluence_override", 0.0) or 0.0),
            "reentry_priority": float(pair_directive.get("reentry_priority", 0.0) or 0.0),
            "frequency_catchup_pressure": float(trajectory_catchup_pressure),
            "soft_burst_target_10m": int(soft_burst_target_10m),
            "quota_boost_allowed": bool(quota_boost_allowed),
            "aggressive_reentry_enabled": bool(aggressive_reentry_enabled),
            "undertrade_fix_mode": bool(undertrade_fix_mode),
            "gpt_hybrid_active": bool(gpt_hybrid_active),
            "gpt_hybrid_session_match": bool(gpt_hybrid_session_match),
            "gpt_hybrid_conviction": float(gpt_hybrid_conviction),
            "gpt_hybrid_setup_bias": str(gpt_hybrid_advisory.get("setup_bias") or ""),
            "gpt_hybrid_direction_bias": str(gpt_hybrid_advisory.get("direction_bias") or ""),
            "gpt_hybrid_threshold_delta": float(gpt_hybrid_threshold_delta),
            "gpt_hybrid_reason": str(gpt_hybrid_advisory.get("reason") or ""),
            "rank_bonus": clamp(float(rank_bonus), 0.0, 0.16),
            "size_bonus": clamp(float(size_bonus), 0.0, 0.16),
        }

    def _density_profile_state(
        self,
        *,
        symbol: str,
        session_name: str = "",
        weekend_mode: bool = False,
    ) -> dict[str, Any]:
        symbol_key = self._normalize_symbol(symbol)
        asset_class = str(symbol_asset_class(symbol_key) or "").strip().lower()
        default_profile = dict(self.density_profiles.get("__default__", {}) or {})
        asset_profile = dict(self.density_profiles.get(f"asset:{asset_class}", {}) or {}) if asset_class else {}
        symbol_profile = dict(self.density_profiles.get(symbol_key, {}) or {})
        merged = {**default_profile, **asset_profile, **symbol_profile}
        pair_directive = dict(self.learning_pair_directive or {})
        frequency_directives = dict(pair_directive.get("frequency_directives") or {})
        session_key = str(session_name or "").upper()
        sessions = {
            str(item).upper()
            for item in merged.get("sessions", [])
            if str(item).strip()
        }
        weekend_sessions = {
            str(item).upper()
            for item in merged.get("weekend_sessions", [])
            if str(item).strip()
        }
        active_sessions = weekend_sessions if weekend_mode and weekend_sessions else sessions
        active = bool(merged) and (not active_sessions or session_key in active_sessions)
        rank_bonus = clamp(
            float(merged.get("rank_bonus", 0.0) or 0.0)
            + float(pair_directive.get("density_rank_bonus", 0.0) or 0.0),
            0.0,
            0.08,
        )
        size_bonus = clamp(
            float(merged.get("size_bonus", 0.0) or 0.0)
            + float(pair_directive.get("density_size_bonus", 0.0) or 0.0),
            0.0,
            0.12,
        )
        activation_score = clamp(
            float(merged.get("activation_score", 0.58) or 0.58)
            - float(pair_directive.get("density_activation_relax", 0.0) or 0.0),
            0.45,
            0.80,
        )
        entry_cap = max(0, int(merged.get("entry_cap", 0) or 0)) + max(
            0,
            int(pair_directive.get("density_entry_cap_bonus", 0) or 0),
        )
        compression_candidates = max(0, int(merged.get("compression_candidates", 0) or 0))
        compression_candidates += max(
            0,
            int(pair_directive.get("density_compression_candidate_bonus", 0) or 0),
        )
        compression_candidates = max(
            compression_candidates,
            max(0, int(frequency_directives.get("soft_burst_target_10m", 0) or 0)),
        )
        compression_multiplier = clamp(
            max(
                float(merged.get("compression_multiplier", 1.0) or 1.0),
                float(pair_directive.get("density_compression_multiplier", 1.0) or 1.0),
            ),
            1.0,
            1.60,
        )
        return {
            "active": bool(active),
            "session_match": bool(active),
            "rank_bonus": float(rank_bonus),
            "size_bonus": float(size_bonus),
            "activation_score": float(activation_score),
            "entry_cap": max(0, min(36, int(entry_cap))),
            "compression_candidates": max(0, min(24, int(compression_candidates))),
            "compression_multiplier": float(compression_multiplier),
        }

    @staticmethod
    def _in_utc_window(timestamp, start_hour: float, end_hour: float) -> bool:
        hour = StrategyRouter._hour_fraction(timestamp)
        return float(start_hour) <= hour < float(end_hour)

    @staticmethod
    def _is_last_friday_of_month(timestamp) -> bool:
        current = StrategyRouter._utc_timestamp(timestamp)
        if int(current.weekday()) != 4:
            return False
        return int((current + pd.Timedelta(days=7)).month) != int(current.month)

    @staticmethod
    def _feature_available(row: pd.Series, *keys: str) -> bool:
        for key in keys:
            if key in row.index:
                value = row.get(key)
                if value is not None and not pd.isna(value):
                    return True
        return False

    @staticmethod
    def _strategy_pool_keys(symbol: str) -> list[str]:
        normalized = StrategyRouter._normalize_symbol(symbol)
        mapping = {
            "AUDJPY": [
                "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
                "AUDJPY_SYDNEY_RANGE_BREAK",
                "AUDJPY_LONDON_CARRY_TREND",
                "AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
                "AUDJPY_ATR_COMPRESSION_BREAKOUT",
            ],
            "NZDJPY": [
                "NZDJPY_TOKYO_BREAKOUT",
                "NZDJPY_PULLBACK_CONTINUATION",
                "NZDJPY_LIQUIDITY_TRAP_REVERSAL",
                "NZDJPY_SESSION_RANGE_EXPANSION",
            ],
            "AUDNZD": [
                "AUDNZD_RANGE_ROTATION",
                "AUDNZD_COMPRESSION_RELEASE",
                "AUDNZD_VWAP_MEAN_REVERSION",
                "AUDNZD_STRUCTURE_BREAK_RETEST",
            ],
            "USDJPY": [
                "USDJPY_MOMENTUM_IMPULSE",
                "USDJPY_VWAP_TREND_CONTINUATION",
                "USDJPY_LIQUIDITY_SWEEP_REVERSAL",
                "USDJPY_MACRO_TREND_RIDE",
            ],
            "EURJPY": [
                "EURJPY_MOMENTUM_IMPULSE",
                "EURJPY_SESSION_PULLBACK_CONTINUATION",
                "EURJPY_LIQUIDITY_SWEEP_REVERSAL",
                "EURJPY_RANGE_FADE",
            ],
            "GBPJPY": [
                "GBPJPY_MOMENTUM_IMPULSE",
                "GBPJPY_SESSION_PULLBACK_CONTINUATION",
                "GBPJPY_LIQUIDITY_SWEEP_REVERSAL",
                "GBPJPY_RANGE_FADE",
            ],
            "EURUSD": [
                "EURUSD_LONDON_BREAKOUT",
                "EURUSD_VWAP_PULLBACK",
                "EURUSD_LIQUIDITY_SWEEP",
                "EURUSD_RANGE_FADE",
            ],
            "GBPUSD": [
                "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                "GBPUSD_TREND_PULLBACK_RIDE",
                "GBPUSD_STOP_HUNT_REVERSAL",
                "GBPUSD_ATR_EXPANSION_SCALPER",
            ],
            "XAUUSD": [
                "XAUUSD_ADAPTIVE_M5_GRID",
                "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                "XAUUSD_NY_MOMENTUM_BREAKOUT",
                "XAUUSD_VWAP_REVERSION",
                "XAUUSD_ATR_EXPANSION_SCALPER",
            ],
            "NAS100": [
                "NAS100_OPENING_DRIVE_BREAKOUT",
                "NAS100_VWAP_TREND_STRATEGY",
                "NAS100_LIQUIDITY_SWEEP_REVERSAL",
                "NAS100_MOMENTUM_IMPULSE",
            ],
            "USOIL": [
                "USOIL_INVENTORY_MOMENTUM",
                "USOIL_LONDON_TREND_EXPANSION",
                "USOIL_VWAP_REVERSION",
                "USOIL_BREAKOUT_RETEST",
            ],
            "BTCUSD": [
                "BTCUSD_TREND_SCALP",
                "BTCUSD_RANGE_EXPANSION",
                "BTCUSD_PRICE_ACTION_CONTINUATION",
                "BTCUSD_VOLATILE_RETEST",
            ],
        }
        return list(mapping.get(normalized, [f"{normalized}_MULTI"]))

    @staticmethod
    def _volatility_fit(strategy_key: str, row: pd.Series, regime: RegimeClassification) -> float:
        strategy_key = str(strategy_key or "").upper()
        atr_ratio = clamp(float(row.get("m5_atr_pct_of_avg", 1.0) or 1.0), 0.0, 3.0)
        compression = float(getattr(regime, "details", {}).get("compression_score", 0.0) or 0.0)
        volatility_state = str(getattr(regime, "details", {}).get("volatility_forecast_state", "") or "").upper()
        if "GRID" in strategy_key:
            if volatility_state in {"BALANCED", "EXPANSION_IMMINENT"}:
                return 0.88
            if volatility_state == "COMPRESSION":
                return 0.74
            if volatility_state == "SPIKE":
                return 0.28
            return clamp(0.55 + (compression * 0.20), 0.20, 0.90)
        if any(token in strategy_key for token in ("BREAKOUT", "IMPULSE", "EXPANSION", "CONTINUATION")):
            if volatility_state in {"EXPANSION_IMMINENT", "BALANCED"}:
                return 0.90
            if volatility_state == "COMPRESSION":
                return 0.82
            if volatility_state == "SPIKE":
                return 0.55
            return clamp(0.50 + (atr_ratio * 0.12), 0.25, 0.90)
        if any(token in strategy_key for token in ("VWAP", "RANGE", "REVERSION", "ROTATION", "SWEEP")):
            if volatility_state == "COMPRESSION":
                return 0.88
            if volatility_state == "BALANCED":
                return 0.84
            if volatility_state == "SPIKE":
                return 0.25
            return clamp(0.60 + (compression * 0.18), 0.25, 0.90)
        return 0.70

    @staticmethod
    def _strategy_performance_seed(
        *,
        symbol: str,
        strategy_key: str,
        session_name: str,
    ) -> float:
        priority = session_priority_context(
            symbol=symbol,
            lane_name=infer_trade_lane(symbol=symbol, setup=strategy_key, setup_family="TREND", session_name=session_name),
            session_name=session_name,
        )
        native_bonus = 0.10 if priority.session_native_pair else 0.0
        secondary_bonus = 0.05 if str(priority.lane_session_priority).upper() == "SECONDARY" else 0.0
        return clamp(0.52 + native_bonus + secondary_bonus + ((float(priority.session_priority_multiplier) - 1.0) * 0.40), 0.30, 0.95)

    def _enrich_candidate(
        self,
        *,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        candidate: SignalCandidate,
    ) -> SignalCandidate:
        normalized = self._normalize_symbol(symbol)
        strategy_key = resolve_strategy_key(normalized, str(candidate.setup))
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        raw_regime_label = str(getattr(regime, "state_label", regime.label) or regime.label).strip().upper()
        regime_label_state = runtime_regime_state(str(getattr(regime, "label", regime_state) or regime_state))
        session_name = str(session.session_name).upper()
        timestamp = row.get("time", pd.Timestamp.now(tz="UTC"))
        weekend_mode = self._is_weekend_market_mode(timestamp)
        density_profile = self._density_profile_state(
            symbol=normalized,
            session_name=session_name,
            weekend_mode=weekend_mode,
        )
        rejection_reason = ""
        session_drift_lane = bool((candidate.meta or {}).get("session_drift_lane", False))
        drift_lane_override = bool(session_drift_lane and raw_regime_label == "LOW_LIQUIDITY_DRIFT")
        if (
            normalized in {"EURJPY", "GBPJPY"}
            and strategy_key in {"EURJPY_SESSION_PULLBACK_CONTINUATION", "GBPJPY_SESSION_PULLBACK_CONTINUATION"}
            and session_name == "TOKYO"
            and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
        ):
            rejection_reason = "tokyo_jpy_pullback_bad_regime"
        elif (
            normalized in {"EURJPY", "GBPJPY"}
            and strategy_key in {"EURJPY_MOMENTUM_IMPULSE", "GBPJPY_MOMENTUM_IMPULSE"}
            and session_name == "TOKYO"
            and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
        ):
            rejection_reason = "tokyo_jpy_impulse_bad_regime"
        elif (
            normalized == "BTCUSD"
            and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION"
            and session_name == "TOKYO"
            and regime_state == "RANGING"
        ):
            rejection_reason = "btc_tokyo_price_action_ranging"
        elif (
            normalized == "BTCUSD"
            and weekend_mode
            and strategy_key in {"BTCUSD_RANGE_EXPANSION", "BTCUSD_VOLATILE_RETEST"}
            and session_name in {"SYDNEY", "TOKYO"}
            and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
        ):
            rejection_reason = "btc_weekend_asia_breakout_bad_regime"
        elif (
            normalized == "BTCUSD"
            and strategy_key == "BTCUSD_TREND_SCALP"
            and session_name == "TOKYO"
            and not weekend_mode
            and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
        ):
            rejection_reason = "btc_tokyo_trend_scalp_bad_liquidity"
        elif (
            normalized == "BTCUSD"
            and weekend_mode
            and strategy_key == "BTCUSD_TREND_SCALP"
            and session_name == "SYDNEY"
            and regime_state == "MEAN_REVERSION"
        ):
            rejection_reason = "btc_weekend_sydney_trend_scalp_mean_reversion"
        elif (
            normalized == "BTCUSD"
            and weekend_mode
            and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION"
            and session_name == "LONDON"
            and regime_state == "MEAN_REVERSION"
        ):
            rejection_reason = "btc_weekend_london_price_action_mean_reversion"
        elif (
            normalized == "BTCUSD"
            and weekend_mode
            and strategy_key == "BTCUSD_RANGE_EXPANSION"
            and session_name == "LONDON"
            and regime_state == "MEAN_REVERSION"
        ):
            rejection_reason = "btc_weekend_london_range_expansion_mean_reversion"
        elif (
            normalized == "NAS100"
            and strategy_key == "NAS100_VWAP_TREND_STRATEGY"
            and session_name == "TOKYO"
            and regime_state == "MEAN_REVERSION"
        ):
            rejection_reason = "nas_vwap_trend_tokyo_mean_reversion"
        elif (
            normalized == "NAS100"
            and strategy_key == "NAS100_MOMENTUM_IMPULSE"
            and session_name == "TOKYO"
            and regime_state == "MEAN_REVERSION"
        ):
            rejection_reason = "nas_momentum_impulse_tokyo_bad_regime"
        elif (
            normalized == "NAS100"
            and strategy_key == "NAS100_MOMENTUM_IMPULSE"
            and (
                (session_name == "TOKYO" and regime_state in {"TRENDING", "LOW_LIQUIDITY_CHOP"})
                or (session_name == "SYDNEY" and regime_state == "LOW_LIQUIDITY_CHOP")
                or (session_name == "LONDON" and regime_state in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"})
                or (session_name == "OVERLAP" and regime_state in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"})
                or (session_name == "NEW_YORK" and regime_state in {"TRENDING", "MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"})
            )
        ):
            rejection_reason = "nas_momentum_impulse_nonexpansion_leak"
        elif (
            normalized == "NAS100"
            and strategy_key == "NAS100_OPENING_DRIVE_BREAKOUT"
            and session_name == "TOKYO"
            and regime_state != "BREAKOUT_EXPANSION"
        ):
            rejection_reason = "nas_opening_drive_tokyo_non_expansion"
        elif (
            normalized == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and session_name not in {"LONDON", "OVERLAP", "NEW_YORK"}
        ):
            # Keep XAU grid off-session blocks, but let the dedicated XAU live logic
            # decide prime-session quality instead of killing nearly every bucket here.
            rejection_reason = "xau_grid_off_session"
        elif (
            normalized == "USOIL"
            and strategy_key == "USOIL_INVENTORY_MOMENTUM"
            and session_name == "TOKYO"
            and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
        ):
            rejection_reason = "usoil_inventory_tokyo_bad_regime"
        elif (
            normalized == "USOIL"
            and strategy_key == "USOIL_LONDON_TREND_EXPANSION"
            and (
                (session_name == "SYDNEY" and regime_state == "TRENDING")
                or (session_name == "OVERLAP" and regime_state in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"})
                or (session_name == "NEW_YORK" and regime_state == "LOW_LIQUIDITY_CHOP")
            )
        ):
            rejection_reason = "usoil_trend_expansion_bad_bucket"
        elif (
            normalized == "EURUSD"
            and strategy_key == "EURUSD_LONDON_BREAKOUT"
            and session_name in {"OVERLAP", "NEW_YORK"}
            and regime_state == "LOW_LIQUIDITY_CHOP"
            and not drift_lane_override
        ):
            rejection_reason = "eurusd_breakout_overlap_ny_low_liquidity"
        elif (
            normalized == "EURUSD"
            and strategy_key == "EURUSD_VWAP_PULLBACK"
            and session_name == "OVERLAP"
            and regime_state == "LOW_LIQUIDITY_CHOP"
            and not drift_lane_override
        ):
            rejection_reason = "eurusd_vwap_overlap_low_liquidity"
        elif (
            normalized == "EURJPY"
            and strategy_key == "EURJPY_SESSION_PULLBACK_CONTINUATION"
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and regime_state == "LOW_LIQUIDITY_CHOP"
        ):
            rejection_reason = "eurjpy_pullback_low_liquidity"
        elif (
            normalized == "GBPJPY"
            and strategy_key in {"GBPJPY_MOMENTUM_IMPULSE", "GBPJPY_SESSION_PULLBACK_CONTINUATION"}
            and session_name in {"SYDNEY", "LONDON", "NEW_YORK"}
            and regime_state == "LOW_LIQUIDITY_CHOP"
        ):
            rejection_reason = "gbpjpy_low_liquidity_trend_leak"
        elif (
            normalized == "NZDJPY"
            and strategy_key == "NZDJPY_LIQUIDITY_TRAP_REVERSAL"
            and session_name == "OVERLAP"
            and regime_state == "MEAN_REVERSION"
        ):
            rejection_reason = "nzdjpy_trap_mean_reversion_leak"
        elif (
            normalized == "NZDJPY"
            and strategy_key == "NZDJPY_SESSION_RANGE_EXPANSION"
            and session_name == "LONDON"
            and regime_state == "LOW_LIQUIDITY_CHOP"
        ):
            rejection_reason = "nzdjpy_range_expansion_london_chop"
        elif (
            normalized == "AUDJPY"
            and strategy_key == "AUDJPY_LONDON_CARRY_TREND"
            and session_name == "NEW_YORK"
            and regime_state in {"MEAN_REVERSION", "RANGING"}
        ):
            rejection_reason = "audjpy_carry_trend_ny_bad_regime"
        elif (
            normalized == "AUDJPY"
            and strategy_key == "AUDJPY_ATR_COMPRESSION_BREAKOUT"
            and session_name == "OVERLAP"
            and regime_state == "MEAN_REVERSION"
        ):
            rejection_reason = "audjpy_overlap_compression_mean_reversion"
        elif (
            normalized == "BTCUSD"
            and strategy_key == "BTCUSD_VOLATILE_RETEST"
            and (
                (session_name == "TOKYO" and regime_state == "MEAN_REVERSION" and not weekend_mode)
                or (session_name == "NEW_YORK" and regime_state == "LOW_LIQUIDITY_CHOP")
            )
        ):
            rejection_reason = "btc_retest_bad_bucket"
        elif (
            normalized == "BTCUSD"
            and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION"
            and session_name == "NEW_YORK"
            and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
            and not weekend_mode
        ):
            rejection_reason = "btc_price_action_ny_bad_bucket"
        elif (
            normalized == "GBPUSD"
            and strategy_key == "GBPUSD_TREND_PULLBACK_RIDE"
            and (
                (
                    session_name == "LONDON"
                    and ({regime_state, regime_label_state} & {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"})
                )
                or (
                    session_name == "OVERLAP"
                    and ({regime_state, regime_label_state} & {"LOW_LIQUIDITY_CHOP"})
                )
            )
            and not drift_lane_override
        ):
            rejection_reason = "gbpusd_london_trend_pullback_bad_regime"
        trade_lane = infer_trade_lane(
            symbol=normalized,
            setup=str(candidate.setup),
            setup_family=str(candidate.strategy_family),
            session_name=session_name,
        )
        if normalized == "XAUUSD" and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID":
            trade_lane = self._resolve_xau_attack_lane(session_name, candidate)
        priority = session_priority_context(
            symbol=normalized,
            lane_name=trade_lane,
            session_name=session_name,
        )
        learning_state = self._learning_pattern_state(
            symbol=normalized,
            strategy_key=strategy_key,
            session_name=session_name,
        )
        pressure_alignment = float(getattr(regime, "details", {}).get("pressure_proxy_score", 0.60) or 0.60)
        btc_weekend_drift_lane = bool(candidate.meta.get("btc_weekend_drift_lane", False))
        scoring_regime_state = regime_state
        if normalized == "BTCUSD" and weekend_mode and btc_weekend_drift_lane and raw_regime_label == "LOW_LIQUIDITY_DRIFT":
            scoring_regime_state = "TRENDING"
        if normalized in {"GBPUSD", "EURUSD", "USDJPY", "AUDJPY", "NZDJPY", "AUDNZD"} and drift_lane_override:
            scoring_regime_state = "TRENDING"
        structure_score = clamp(
            max(
                float(row.get("m5_structure_score", 0.0) or 0.0),
                float(row.get("m15_structure_score", 0.0) or 0.0),
                float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)) or 0.55),
            ),
            0.0,
            1.0,
        )
        liquidity_score = clamp(
            max(
                float(getattr(regime, "details", {}).get("absorption_signal", 0.0) or 0.0),
                1.0 - clamp(float(row.get("m5_range_position_20", 0.5) or 0.5) - 0.5, -0.5, 0.5) ** 2,
            ),
            0.0,
            1.0,
        )
        spread_points = float(row.get("m5_spread", 0.0) or 0.0)
        spread_reference_limit = float(self._spread_reference_limit(normalized))
        execution_quality_fit = clamp(1.0 - (spread_points / max(1.0, spread_reference_limit)), 0.0, 1.0)
        structure_clean = structure_cleanliness_score(
            spread_points=spread_points,
            spread_limit=spread_reference_limit,
            structure_score=structure_score,
            liquidity_score=liquidity_score,
            volatility_state=str(getattr(regime, "details", {}).get("volatility_forecast_state", "") or ""),
            regime_state=regime_state,
            pressure_alignment=pressure_alignment,
        )
        if normalized == "BTCUSD" and weekend_mode and btc_weekend_drift_lane:
            structure_clean = clamp(
                max(structure_clean, float(candidate.meta.get("structure_cleanliness_floor", 0.60) or 0.60)),
                0.0,
                1.0,
            )
        if normalized in {"GBPUSD", "EURUSD", "USDJPY", "AUDJPY", "NZDJPY", "AUDNZD"} and drift_lane_override:
            structure_clean = clamp(
                max(structure_clean, float(candidate.meta.get("structure_cleanliness_floor", 0.56) or 0.56)),
                0.0,
                1.0,
            )
        chase_penalty = 0.0
        body_efficiency = self._body_efficiency_value(row)
        atr_ratio = float(row.get("m5_atr_pct_of_avg", 1.0) or 1.0)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        volume_ratio = float(row.get("m5_volume_ratio_20", row.get("m15_volume_ratio_20", 1.0)) or 1.0)
        close = float(row.get("m15_close", row.get("m5_close", 0.0)) or 0.0)
        ema20 = float(row.get("m15_ema_20", row.get("m5_ema_20", close)) or close)
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0)) or 0.0), 1e-6)
        compression_proxy_state = str(getattr(regime, "details", {}).get("compression_proxy_state", "NEUTRAL") or "NEUTRAL").upper()
        compression_expansion_score = clamp(float(getattr(regime, "details", {}).get("compression_expansion_score", 0.0) or 0.0), 0.0, 1.0)
        delta_proxy = delta_proxy_score(
            side=str(candidate.side),
            body_efficiency=body_efficiency,
            short_return=float(row.get("m5_ret_1", row.get("m15_ret_1", 0.0)) or 0.0),
            range_position=range_position,
            volume_ratio=volume_ratio,
            upper_wick_ratio=float(row.get("m5_upper_wick_ratio", 0.0) or 0.0),
            lower_wick_ratio=float(row.get("m5_lower_wick_ratio", 0.0) or 0.0),
        )
        session_loosen = session_loosen_factor(
            session_name=session_name,
            symbol=normalized,
            weekend_mode=weekend_mode,
            candidate_scarcity=False,
        )
        if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} or (normalized == "BTCUSD" and weekend_mode):
            session_loosen = min(session_loosen, clamp(1.0 - float(self.high_liquidity_loosen_pct), 0.60, 1.0))
        if atr_ratio >= 1.60 and body_efficiency >= 0.82 and any(token in strategy_key for token in ("BREAKOUT", "IMPULSE", "EXPANSION")):
            chase_penalty = 0.12
        if any(token in strategy_key for token in ("BREAKOUT", "IMPULSE", "EXPANSION", "CONTINUATION")):
            if (str(candidate.side).upper() == "BUY" and range_position >= 0.92) or (
                str(candidate.side).upper() == "SELL" and range_position <= 0.08
            ):
                chase_penalty += 0.08
            if abs(close - ema20) > (atr * 0.90):
                chase_penalty += 0.05
        if any(token in strategy_key for token in ("TREND_SCALP", "CONTINUATION", "BREAKOUT", "IMPULSE")):
            chase_penalty += max(0.0, 0.40 - body_efficiency) * 0.20
        chase_penalty *= float(session_loosen)
        entry_timing = entry_timing_score(
            structure_cleanliness=structure_clean,
            probability=clamp(float(candidate.score_hint or 0.0), 0.0, 1.0),
            expected_value_r=float(candidate.tp_r or 0.0),
            spread_points=spread_points,
            spread_limit=spread_reference_limit,
            volatility_state=str(getattr(regime, "details", {}).get("volatility_forecast_state", "") or ""),
            regime_state=scoring_regime_state,
            chase_penalty=chase_penalty,
            delta_proxy_score_value=delta_proxy,
        )
        if normalized == "BTCUSD" and weekend_mode and btc_weekend_drift_lane:
            entry_timing = clamp(
                max(entry_timing, float(candidate.meta.get("entry_timing_floor", 0.62) or 0.62)),
                0.0,
                1.0,
            )
        if normalized in {"GBPUSD", "EURUSD", "USDJPY", "AUDJPY", "NZDJPY", "AUDNZD"} and drift_lane_override:
            entry_timing = clamp(
                max(entry_timing, float(candidate.meta.get("entry_timing_floor", 0.58) or 0.58)),
                0.0,
                1.0,
            )
        regime_fit = strategy_regime_fit(strategy_key, scoring_regime_state)
        regime_fit = clamp(
            float(regime_fit)
            + (0.06 * max(0.0, delta_proxy))
            + float(compression_strategy_bias(strategy_key, compression_proxy_state) * 0.50),
            0.0,
            1.0,
        )
        session_fit = clamp(float(priority.session_priority_multiplier) - 0.10, 0.0, 1.0)
        volatility_fit = self._volatility_fit(strategy_key, row, regime)
        pair_fit = pair_behavior_fit(
            symbol=normalized,
            strategy_key=strategy_key,
            session_name=str(session.session_name),
            regime_state=scoring_regime_state,
            weekend_mode=weekend_mode,
        )
        session_fit = clamp(
            float(session_fit)
            + (0.04 if bool(learning_state.get("watchlist_match", False)) else 0.0)
            + (0.03 if bool(learning_state.get("native_session", False) and learning_state.get("promoted_pattern", False)) else 0.0),
            0.0,
            1.0,
        )
        pair_fit = clamp(
            float(pair_fit)
            + (0.05 if bool(learning_state.get("promoted_pattern", False)) else 0.0)
            + (0.03 if bool(learning_state.get("watchlist_match", False)) else 0.0),
            0.0,
            1.0,
        )
        perf_seed = self._strategy_performance_seed(
            symbol=normalized,
            strategy_key=strategy_key,
            session_name=str(session.session_name),
        )
        perf_seed = clamp(
            float(perf_seed)
            + (0.04 if bool(learning_state.get("promoted_pattern", False)) else 0.0)
            + (0.02 if bool(learning_state.get("watchlist_match", False)) else 0.0),
            0.0,
            1.0,
        )
        score = strategy_selection_score(
            ev_estimate=clamp(float(candidate.score_hint or 0.0), 0.0, 1.0),
            regime_fit=regime_fit,
            session_fit=session_fit,
            volatility_fit=volatility_fit,
            pair_behavior_fit_score=pair_fit,
            strategy_recent_performance=perf_seed,
            execution_quality_fit=execution_quality_fit,
            entry_timing_score_value=entry_timing,
            structure_cleanliness_score_value=structure_clean,
            drawdown_penalty=0.0,
            false_break_penalty=max(0.0, 0.55 - structure_clean),
            chop_penalty=0.20 if scoring_regime_state == "LOW_LIQUIDITY_CHOP" and "BREAKOUT" in strategy_key else 0.0,
        )
        score = clamp(
            float(score)
            + winner_promotion_bonus(
                symbol=normalized,
                strategy_key=strategy_key,
                regime_state=scoring_regime_state,
                session_name=session_name,
                weekend_mode=weekend_mode,
            )
            + (0.05 * max(0.0, delta_proxy))
            + float(compression_strategy_bias(strategy_key, compression_proxy_state)),
            0.0,
            1.0,
        )
        score = clamp(float(score) + float(learning_state.get("rank_bonus", 0.0) or 0.0), 0.0, 1.0)
        if normalized == "BTCUSD" and weekend_mode and strategy_key in {"BTCUSD_PRICE_ACTION_CONTINUATION", "BTCUSD_VOLATILE_RETEST"}:
            score = clamp(float(score) + float(self.btc_weekend_score_bonus), 0.0, 1.0)
        if (
            normalized == "XAUUSD"
            and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
            and bool(learning_state.get("xau_prime_session", False))
            and not rejection_reason
        ):
            xau_attack_category = xau_attack_lane_category(trade_lane)
            prime_burst_bonus = min(0.12, max(0.0, float(self.xau_prime_session_mult) - 1.0) * 0.05)
            if bool(learning_state.get("throughput_recovery_active", False)):
                prime_burst_bonus += 0.02
            if compression_proxy_state in {"COMPRESSION", "EXPANSION_READY"}:
                prime_burst_bonus += 0.02
            if xau_attack_category == "BREAKOUT":
                prime_burst_bonus += 0.03 if compression_expansion_score >= 0.34 else 0.02
            elif xau_attack_category == "RECLAIM":
                prime_burst_bonus += 0.025
            elif xau_attack_category == "REENTRY":
                prime_burst_bonus += 0.02
            else:
                prime_burst_bonus += 0.015
            score = clamp(float(score) + prime_burst_bonus, 0.0, 1.0)
            min_confluence = max(4.2, float(learning_state.get("min_confluence_override", 0.0) or 0.0))
            if regime_fit < 0.82 or float(candidate.confluence_score or 0.0) < min_confluence or structure_clean < 0.60:
                rejection_reason = "xau_prime_quality_floor"
            elif compression_expansion_score >= 0.36 or compression_proxy_state == "EXPANSION_READY":
                candidate.tp_r = max(float(candidate.tp_r or 0.0), 1.80)
                candidate.meta = {
                    **dict(candidate.meta or {}),
                    "tp_extension_active": True,
                }
        if (
            normalized == "BTCUSD"
            and weekend_mode
            and strategy_key in {"BTCUSD_PRICE_ACTION_CONTINUATION", "BTCUSD_VOLATILE_RETEST", "BTCUSD_RANGE_EXPANSION", "BTC_FUNDING_ARB", "BTC_WEEKEND_GAP_FADE"}
            and not rejection_reason
        ):
            weekend_bonus = min(0.10, max(0.0, float(self.btc_weekend_burst_target) - 20.0) / 200.0)
            if session_name in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}:
                score = clamp(float(score) + weekend_bonus, 0.0, 1.0)
        if normalized == "BTCUSD" and not weekend_mode and session_name in {"SYDNEY", "TOKYO"} and not rejection_reason:
            score = clamp(float(score) - 0.06, 0.0, 1.0)
        if normalized != "XAUUSD" and bool(learning_state.get("native_session", False)) and not rejection_reason:
            native_bonus = min(0.06, max(0.0, float(self.all_pairs_aggression) - 1.0) * 0.20)
            if normalized in {"USOIL", "NAS100"} and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}:
                native_bonus += 0.02
            elif normalized in {"AUDJPY", "NZDJPY", "AUDNZD", "USDJPY", "EURJPY", "GBPJPY"} and session_name in {"SYDNEY", "TOKYO"}:
                native_bonus += 0.02
            if score >= 0.55 or bool(learning_state.get("throughput_recovery_active", False)):
                score = clamp(float(score) + native_bonus, 0.0, 1.0)
        fast_execution_active = self._fast_execution_lane_active(normalized, session_name, weekend_mode=weekend_mode)
        fast_lane_cap_bonus = 0
        fast_lane_rank_bonus = 0.0
        fast_lane_size_boost = 1.0
        if fast_execution_active and not rejection_reason:
            if execution_quality_fit >= 0.66 and structure_clean >= 0.56:
                fast_lane_rank_bonus += 0.02
            if score >= 0.58 or bool(learning_state.get("throughput_recovery_active", False)):
                fast_lane_cap_bonus += 2
                if normalized == "XAUUSD":
                    fast_lane_cap_bonus += 4
                    if is_xau_grid_lane(trade_lane):
                        fast_lane_cap_bonus += 2
                elif normalized in {"BTCUSD", "NAS100"}:
                    fast_lane_cap_bonus += 1
                fast_lane_size_boost *= 1.04
                if session_name in {"LONDON", "OVERLAP", "NEW_YORK"}:
                    fast_lane_rank_bonus += 0.01
            score = clamp(float(score) + fast_lane_rank_bonus, 0.0, 1.0)
        density_profile_ready = bool(
            density_profile.get("active", False)
            and not rejection_reason
            and (
                score >= float(density_profile.get("activation_score", 0.58) or 0.58)
                or bool(learning_state.get("throughput_recovery_active", False))
                or bool(learning_state.get("native_session", False))
            )
        )
        density_profile_rank_bonus = 0.0
        if density_profile_ready:
            density_profile_rank_bonus = float(density_profile.get("rank_bonus", 0.0) or 0.0)
            if bool(learning_state.get("throughput_recovery_active", False)):
                density_profile_rank_bonus = min(0.08, density_profile_rank_bonus + 0.01)
            score = clamp(float(score) + density_profile_rank_bonus, 0.0, 1.0)
        if bool(learning_state.get("shadow_experiment_active", False)) and not rejection_reason:
            if score >= 0.58 or bool(learning_state.get("watchlist_match", False)):
                score = clamp(float(score) + 0.02, 0.0, 1.0)
        high_liquidity = session_name in {"LONDON", "OVERLAP", "NEW_YORK"} or (normalized == "BTCUSD" and weekend_mode)
        quality_tier = quality_tier_from_scores(
            structure_cleanliness=structure_clean,
            regime_fit=regime_fit,
            execution_quality_fit=execution_quality_fit,
            high_liquidity=high_liquidity,
            throughput_recovery_active=bool(learning_state.get("throughput_recovery_active", False)),
        )
        tier_size_multiplier = quality_tier_size_multiplier(
            quality_tier=quality_tier,
            strategy_score=score,
            b_tier_min=float(self.b_tier_size_mult_min),
            b_tier_max=float(self.b_tier_size_mult_max),
        )
        tier_size_multiplier = clamp(
            float(tier_size_multiplier) * (1.0 + float(learning_state.get("size_bonus", 0.0) or 0.0)),
            0.55,
            1.50,
        )
        if (
            normalized == "BTCUSD"
            and weekend_mode
            and strategy_key in {"BTCUSD_PRICE_ACTION_CONTINUATION", "BTCUSD_VOLATILE_RETEST", "BTCUSD_RANGE_EXPANSION", "BTC_FUNDING_ARB", "BTC_WEEKEND_GAP_FADE"}
        ):
            tier_size_multiplier = clamp(
                float(tier_size_multiplier) * max(1.0, float(self.btc_weekend_size_boost)),
                0.55,
                1.70,
            )
        elif normalized == "BTCUSD" and not weekend_mode and session_name in {"SYDNEY", "TOKYO"}:
            tier_size_multiplier = clamp(float(tier_size_multiplier) * 0.92, 0.55, 1.50)
        if density_profile_ready:
            tier_size_multiplier = clamp(
                float(tier_size_multiplier)
                * (1.0 + float(density_profile.get("size_bonus", 0.0) or 0.0))
                * fast_lane_size_boost,
                0.55,
                1.70,
            )
        density_entry_cap = (
            int(density_profile.get("entry_cap", 0) or 0) + int(fast_lane_cap_bonus)
            if density_profile_ready and quality_tier in {"A+", "A", "B"}
            else 0
        )
        allowed_regimes = strategy_allowed_regimes(strategy_key)
        candidate.score_hint = max(float(candidate.score_hint or 0.0), score)
        candidate.meta = {
            **dict(candidate.meta or {}),
            "symbol": str(normalized),
            "symbol_key": str(normalized),
            "session_name": str(session_name),
            "strategy_key": strategy_key,
            "strategy_pool": self._strategy_pool_keys(normalized),
            "lane_name": trade_lane,
            "attack_lane_active": bool(fast_execution_active),
            "fast_execution_profile": "M3_ATTACK" if fast_execution_active else "STANDARD",
            "xau_attack_category": xau_attack_lane_category(trade_lane) if is_xau_grid_lane(trade_lane) else "",
            "regime_state": scoring_regime_state,
            "raw_regime_state": regime_state,
            "allowed_regimes": list(allowed_regimes),
            "management_template": strategy_management_template(strategy_key),
            "regime_fit": float(regime_fit),
            "session_fit": float(session_fit),
            "volatility_fit": float(volatility_fit),
            "pair_behavior_fit": float(pair_fit),
            "strategy_recent_performance_seed": float(perf_seed),
            "execution_quality_fit": execution_quality_fit,
            "spread_reference_limit": float(spread_reference_limit),
            "entry_timing_score": float(entry_timing),
            "structure_cleanliness_score": float(structure_clean),
            "router_rank_score": float(score),
            "quality_tier": str(quality_tier),
            "tier_size_multiplier": float(tier_size_multiplier),
            "density_profile_active": bool(density_profile_ready),
            "density_profile_rank_bonus": float(density_profile_rank_bonus),
            "density_entry_cap": int(density_entry_cap),
            "density_profile_compression_candidates": int(density_profile.get("compression_candidates", 0) or 0),
            "delta_proxy_score": float(delta_proxy),
            "compression_proxy_state": str(compression_proxy_state),
            "compression_expansion_score": float(compression_expansion_score),
            "session_loosen_factor": float(session_loosen),
            "throughput_recovery_active": bool(learning_state.get("throughput_recovery_active", False)),
            "session_priority_profile": str(priority.session_priority_profile),
            "session_native_pair": bool(priority.session_native_pair),
            "session_priority_multiplier": float(priority.session_priority_multiplier),
            "pair_priority_rank_in_session": int(priority.pair_priority_rank_in_session),
            "lane_budget_share": float(priority.lane_budget_share),
            "router_reject": bool(rejection_reason),
            "router_reject_reason": str(rejection_reason),
            "recycle_session": False,
            "recycle_origin_session": "",
            "recycle_boost_applied": 0.0,
            "family_rotation_penalty": 0.0,
            "equity_momentum_mode": "NEUTRAL",
            "btc_weekend_mode": bool(normalized == "BTCUSD" and weekend_mode),
            "btc_weekend_drift_lane": bool(btc_weekend_drift_lane),
            "session_drift_lane": bool(session_drift_lane),
            "verified_reason_code": "",
            "verified_reason_text": "",
            "learning_brain_symbol_focus": bool(learning_state.get("symbol_focus", False)),
            "learning_brain_promoted_pattern": bool(learning_state.get("promoted_pattern", False)),
            "learning_brain_watchlist_match": bool(learning_state.get("watchlist_match", False)),
            "learning_brain_weak_focus": bool(learning_state.get("weak_focus", False)),
            "learning_brain_shadow_focus": bool(learning_state.get("shadow_focus", False)),
            "learning_trajectory_catchup_pressure": float(learning_state.get("trajectory_catchup_pressure", 0.0)),
            "frequency_catchup_pressure": float(learning_state.get("frequency_catchup_pressure", 0.0)),
            "soft_burst_target_10m": int(learning_state.get("soft_burst_target_10m", 0) or 0),
            "quota_boost_allowed": bool(learning_state.get("quota_boost_allowed", False)),
            "aggressive_reentry_enabled": bool(learning_state.get("aggressive_reentry_enabled", False)),
            "undertrade_fix_mode": bool(learning_state.get("undertrade_fix_mode", False)),
            "gpt_hybrid_active": bool(learning_state.get("gpt_hybrid_active", False)),
            "gpt_hybrid_session_match": bool(learning_state.get("gpt_hybrid_session_match", False)),
            "gpt_hybrid_conviction": float(learning_state.get("gpt_hybrid_conviction", 0.0) or 0.0),
            "gpt_hybrid_setup_bias": str(learning_state.get("gpt_hybrid_setup_bias", "")),
            "gpt_hybrid_direction_bias": str(learning_state.get("gpt_hybrid_direction_bias", "")),
            "gpt_hybrid_threshold_delta": float(learning_state.get("gpt_hybrid_threshold_delta", 0.0) or 0.0),
            "gpt_hybrid_reason": str(learning_state.get("gpt_hybrid_reason", "")),
            "learning_rank_bonus": float(learning_state.get("rank_bonus", 0.0)),
            "learning_size_bonus": float(learning_state.get("size_bonus", 0.0)),
        }
        return candidate

    def _symbol_sensitivity(self, symbol: str, base: float) -> float:
        normalized = self._normalize_symbol(symbol)
        adaptive = float(self.adaptive_sensitivity.get(normalized, 1.0))
        learning_state = self._learning_pattern_state(symbol=normalized)
        learning_boost = 1.0
        if bool(learning_state.get("weak_focus", False)):
            learning_boost += 0.04
        if bool(learning_state.get("promoted_pattern", False)):
            learning_boost += 0.05
        if bool(learning_state.get("throughput_recovery_active", False)):
            recovery_pressure = float(learning_state.get("trajectory_catchup_pressure", 0.0) or 0.0)
            recovery_cap = 0.12 if bool(learning_state.get("native_session", False)) else 0.09
            recovery_bonus = (recovery_pressure * 0.08) + (0.02 if bool(learning_state.get("native_session", False)) else 0.0)
            learning_boost += min(recovery_cap, recovery_bonus)
        return clamp(float(base) * adaptive * learning_boost, 0.80, 1.40)

    @staticmethod
    def _ratio_or_neutral(value: object, *, neutral: float = 1.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float(neutral)
        if pd.isna(numeric) or numeric <= 0.0:
            return float(neutral)
        return numeric

    @staticmethod
    def _body_efficiency_value(
        row: pd.Series,
        *,
        primary: str = "m5",
        fallback: str = "m15",
        default: float = 0.55,
    ) -> float:
        for prefix in (primary, fallback):
            for suffix in ("body_efficiency", "candle_efficiency"):
                value = row.get(f"{prefix}_{suffix}")
                if value is not None and not pd.isna(value):
                    return float(value)
        return float(default)

    def _spread_reference_limit(self, symbol: str) -> float:
        symbol_key = self._normalize_symbol(symbol)
        configured_limit = float(self.symbol_spread_caps.get(symbol_key, 0.0) or 0.0)
        if configured_limit > 0.0:
            return max(1.0, configured_limit)
        if symbol_key == "BTCUSD":
            return max(float(self.btc_spread_cap_points), float(self.max_spread_points), 100.0)
        default_limits = {
            "USDJPY": 32.0,
            "AUDJPY": 42.0,
            "NZDJPY": 40.0,
            "AUDNZD": 30.0,
            "EURJPY": 44.0,
            "GBPJPY": 52.0,
            "EURGBP": 24.0,
            "NAS100": 95.0,
            "USOIL": 22.0,
            "XAGUSD": 42.0,
            "DOGUSD": 260.0,
            "TRUMPUSD": 320.0,
            "AAPL": 18.0,
            "NVIDIA": 22.0,
        }
        if symbol_key in default_limits:
            return max(float(self.max_spread_points), float(default_limits[symbol_key]))
        asset_class = symbol_asset_class(symbol_key)
        if asset_class == "crypto":
            return max(float(self.max_spread_points), 180.0)
        if asset_class == "equity":
            return max(float(self.max_spread_points), 18.0)
        if asset_class == "commodity":
            return max(float(self.max_spread_points), 40.0)
        if asset_class == "index":
            return max(float(self.max_spread_points), 95.0)
        return max(float(self.max_spread_points), float(default_limits.get(symbol_key, self.max_spread_points)))

    def _spread_limit(self, symbol: str, *, multiplier: float = 1.0) -> float:
        return max(1.0, self._spread_reference_limit(symbol) * float(multiplier))

    def _drift_continuation_ready(
        self,
        *,
        symbol: str,
        row: pd.Series,
        session_name: str,
        raw_regime_label: str,
    ) -> bool:
        symbol_key = self._normalize_symbol(symbol)
        session_key = str(session_name or "").upper()
        if str(raw_regime_label or "").upper() != "LOW_LIQUIDITY_DRIFT":
            return False
        allowed_sessions: dict[str, set[str]] = {
            "GBPUSD": {"LONDON", "OVERLAP", "NEW_YORK"},
            "EURUSD": {"LONDON", "OVERLAP", "NEW_YORK"},
            "USDJPY": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "AUDJPY": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "NZDJPY": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "AUDNZD": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "NAS100": {"LONDON", "OVERLAP", "NEW_YORK"},
        }
        if session_key not in allowed_sessions.get(symbol_key, set()):
            return False
        close = float(row.get("m15_close", row.get("m5_close", 0.0)) or 0.0)
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0)) or 0.0), 1e-6)
        spread = float(row.get("m5_spread", 0.0) or 0.0)
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        body_efficiency = self._body_efficiency_value(row)
        atr_ratio = float(row.get("m15_atr_pct_of_avg", row.get("m5_atr_pct_of_avg", 1.0)) or 1.0)
        alignment_score = float(row.get("multi_tf_alignment_score", 0.5) or 0.5)
        seasonality_score = float(row.get("seasonality_edge_score", 0.5) or 0.5)
        instability_score = float(row.get("market_instability_score", row.get("feature_drift_score", 0.0)) or 0.0)
        feature_drift = float(row.get("feature_drift_score", 0.0) or 0.0)
        fractal_score = float(
            row.get(
                "fractal_persistence_score",
                row.get("hurst_persistence_score", row.get("m15_hurst_proxy_64", row.get("m5_hurst_proxy_64", 0.5))),
            )
            or 0.5
        )
        ema20 = float(row.get("m15_ema_20", row.get("m5_ema_20", close)) or close)
        ema50 = float(row.get("m15_ema_50", row.get("m5_ema_50", close)) or close)
        h1_ema_50 = float(row.get("h1_ema_50", ema50) or ema50)
        h1_ema_200 = float(row.get("h1_ema_200", h1_ema_50) or h1_ema_50)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        ret_1 = float(row.get("m15_ret_1", row.get("m5_ret_1", 0.0)) or 0.0)
        ret_3 = float(row.get("m15_ret_3", row.get("m5_ret_3", 0.0)) or 0.0)
        directional_bias = (
            (ema20 >= ema50 and h1_ema_50 >= h1_ema_200 and range_position >= 0.54 and (ret_1 >= 0.0 or ret_3 >= 0.0))
            or (ema20 <= ema50 and h1_ema_50 <= h1_ema_200 and range_position <= 0.46 and (ret_1 <= 0.0 or ret_3 <= 0.0))
        )
        asia_body_efficiency = max(
            body_efficiency,
            float(row.get("m15_body_efficiency", row.get("m15_candle_efficiency", 0.0)) or 0.0),
            float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.0)) or 0.0),
        )
        trend_gap_ratio = abs(ema20 - ema50) / max(atr, 1e-6)
        asia_extreme_range_buy = range_position >= 0.86 and ret_1 > 0.0 and ret_3 > 0.0 and close >= ema20 and h1_ema_50 >= h1_ema_200
        asia_extreme_range_sell = range_position <= 0.14 and ret_1 < 0.0 and ret_3 < 0.0 and close <= ema20 and h1_ema_50 <= h1_ema_200
        asia_micro_bias = bool(
            symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD"}
            and session_key in {"SYDNEY", "TOKYO"}
            and trend_gap_ratio <= 0.22
            and (asia_extreme_range_buy or asia_extreme_range_sell)
        )
        directional_bias = bool(directional_bias or asia_micro_bias)
        profiles: dict[str, dict[str, float]] = {
            "GBPUSD": {"min_align": 0.84, "min_fractal": 0.80, "min_body": 0.38, "min_vol": 0.88, "min_seasonality": 0.18, "max_instability": 0.18, "max_feature_drift": 0.22, "max_atr": 1.35},
            "EURUSD": {"min_align": 0.82, "min_fractal": 0.80, "min_body": 0.38, "min_vol": 0.88, "min_seasonality": 0.18, "max_instability": 0.18, "max_feature_drift": 0.22, "max_atr": 1.35},
            "USDJPY": {"min_align": 0.78, "min_fractal": 0.74, "min_body": 0.38, "min_vol": 0.88, "min_seasonality": 0.16, "max_instability": 0.20, "max_feature_drift": 0.24, "max_atr": 1.45},
            "AUDJPY": {"min_align": 0.76, "min_fractal": 0.72, "min_body": 0.38, "min_vol": 0.86, "min_seasonality": 0.18, "max_instability": 0.22, "max_feature_drift": 0.24, "max_atr": 1.45},
            "NZDJPY": {"min_align": 0.76, "min_fractal": 0.72, "min_body": 0.38, "min_vol": 0.88, "min_seasonality": 0.18, "max_instability": 0.22, "max_feature_drift": 0.24, "max_atr": 1.45},
            "AUDNZD": {"min_align": 0.74, "min_fractal": 0.70, "min_body": 0.38, "min_vol": 0.82, "min_seasonality": 0.18, "max_instability": 0.24, "max_feature_drift": 0.26, "max_atr": 1.30},
        }
        profile = profiles.get(symbol_key)
        if profile is None:
            return False
        if spread > self._spread_limit(symbol_key, multiplier=(1.06 if session_key in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.96)):
            return False
        asia_fast_track_profiles: dict[str, dict[str, float]] = {
            "AUDJPY": {
                "min_align": 0.24,
                "min_fractal": 0.42,
                "min_body": 0.42,
                "min_vol": 0.96,
                "min_seasonality": 0.30,
                "max_instability": 0.34,
                "max_feature_drift": 0.14,
                "max_atr": 1.55,
            },
            "NZDJPY": {
                "min_align": 0.72,
                "min_fractal": 0.42,
                "min_body": 0.38,
                "min_vol": 0.96,
                "min_seasonality": 0.30,
                "max_instability": 0.30,
                "max_feature_drift": 0.16,
                "max_atr": 1.50,
            },
            "AUDNZD": {
                "min_align": 0.72,
                "min_fractal": 0.24,
                "min_body": 0.40,
                "min_vol": 0.96,
                "min_seasonality": 0.30,
                "max_instability": 0.30,
                "max_feature_drift": 0.16,
                "max_atr": 1.55,
            },
        }
        asia_fast_track_profile = asia_fast_track_profiles.get(symbol_key)
        if (
            asia_fast_track_profile is not None
            and session_key in {"SYDNEY", "TOKYO"}
            and directional_bias
        ):
            if (
                atr_ratio >= 0.88
                and atr_ratio <= float(asia_fast_track_profile["max_atr"])
                and alignment_score >= float(asia_fast_track_profile["min_align"])
                and fractal_score >= float(asia_fast_track_profile["min_fractal"])
                and seasonality_score >= float(asia_fast_track_profile["min_seasonality"])
                and instability_score <= float(asia_fast_track_profile["max_instability"])
                and feature_drift <= float(asia_fast_track_profile["max_feature_drift"])
                and volume_ratio >= float(asia_fast_track_profile["min_vol"])
                and asia_body_efficiency >= float(asia_fast_track_profile["min_body"])
                and (
                    (range_position >= 0.86 and ret_1 > 0.0 and ret_3 > 0.0)
                    or (range_position <= 0.14 and ret_1 < 0.0 and ret_3 < 0.0)
                )
            ):
                return True
        if atr_ratio < 0.70 or atr_ratio > float(profile["max_atr"]):
            return False
        if alignment_score < float(profile["min_align"]) or fractal_score < float(profile["min_fractal"]):
            return False
        if seasonality_score < float(profile["min_seasonality"]):
            return False
        if instability_score > float(profile["max_instability"]) or feature_drift > float(profile["max_feature_drift"]):
            return False
        if body_efficiency < float(profile["min_body"]) or volume_ratio < float(profile["min_vol"]):
            return False
        if close <= 0.0 or not directional_bias:
            return False
        return True

    @staticmethod
    def _bullish_flag(row: pd.Series, *, primary: str = "m5", fallback: str = "m15") -> bool:
        if int(row.get(f"{primary}_bullish", row.get(f"{fallback}_bullish", 0)) or 0) == 1:
            return True
        return float(row.get(f"{primary}_ret_1", row.get(f"{fallback}_ret_1", 0.0)) or 0.0) > 0.0

    @staticmethod
    def _bearish_flag(row: pd.Series, *, primary: str = "m5", fallback: str = "m15") -> bool:
        if int(row.get(f"{primary}_bearish", row.get(f"{fallback}_bearish", 0)) or 0) == 1:
            return True
        return float(row.get(f"{primary}_ret_1", row.get(f"{fallback}_ret_1", 0.0)) or 0.0) < 0.0

    def diagnostics(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> dict[str, object]:
        normalized = self._normalize_symbol(symbol)
        weekend_mode = self._is_weekend_market_mode(timestamp)
        session_name = str(session.session_name).upper()
        diagnostics: dict[str, object] = {
            "session_policy_current": "DEFAULT",
            "weekend_mode": bool(weekend_mode),
            "active_setup_windows": [],
            "setup_proxy_availability": {},
            "spread_reference_limit": float(self._spread_reference_limit(normalized)),
        }
        learning_state = self._learning_pattern_state(symbol=normalized, session_name=session_name)
        diagnostics.update(
            {
                "learning_symbol_focus": bool(learning_state.get("symbol_focus", False)),
                "learning_weak_pair_focus": bool(learning_state.get("weak_focus", False)),
                "learning_watchlist_match": bool(learning_state.get("watchlist_match", False)),
                "learning_throughput_recovery_active": bool(learning_state.get("throughput_recovery_active", False)),
                "learning_trajectory_catchup_pressure": float(learning_state.get("trajectory_catchup_pressure", 0.0)),
            }
        )
        if normalized == "BTCUSD":
            sensitivity = self._symbol_sensitivity(symbol, self.btc_sensitivity)
            funding_available = self._feature_available(row, "btc_funding_rate_8h", "funding_rate_8h")
            liquidation_available = self._feature_available(row, "btc_liquidations_5m_usd", "liquidation_5m_usd")
            whale_available = self._feature_available(row, "btc_whale_flow_btc_1h", "btc_exchange_flow_btc_1h")
            dxy_available = self._feature_available(row, "dxy_ret_5", "dxy_ret_1")
            gap_available = self._feature_available(row, "btc_weekend_gap_pct", "weekend_gap_pct")
            spread = float(row.get("m5_spread", 0.0))
            spread_ratio = float(row.get("m5_spread_ratio_20", 1.0))
            atr_ratio = float(row.get("m5_atr_pct_of_avg", 1.0))
            close = float(row.get("m5_close", 0.0))
            prev_high = float(row.get("m15_rolling_high_prev_20", row.get("m5_rolling_high_prev_20", close)))
            prev_low = float(row.get("m15_rolling_low_prev_20", row.get("m5_rolling_low_prev_20", close)))
            breakout_up = close > prev_high
            breakout_down = close < prev_low
            range_position = float(row.get("m15_range_position_20", 0.5))
            fallback_used = not any((funding_available, liquidation_available, whale_available, dxy_available, gap_available))
            active_windows: list[str] = []
            if self._btc_session_window_allowed(timestamp):
                active_windows.append("BTC_NY_LIQUIDITY")
            if self._btc_funding_entry_window(timestamp):
                active_windows.append("BTC_FUNDING_WINDOW")
            if weekend_mode:
                active_windows.append("BTC_WEEKEND_MODE")
            if str(session_name).upper() == "SYDNEY":
                active_windows.append("BTC_SYDNEY_DRIFT")
            if str(session_name).upper() == "TOKYO":
                active_windows.append("BTC_TOKYO_DRIFT")
            if str(session_name).upper() == "LONDON":
                active_windows.append("BTC_LONDON_IMPULSE")
            if breakout_up or breakout_down:
                active_windows.append("BTC_RANGE_EXPANSION")
            if fallback_used:
                active_windows.append("BTC_PRICE_ACTION_FALLBACK")
            diagnostics.update(
                {
                    "session_policy_current": "BTC_WEEKEND_PRIORITY" if weekend_mode else "BTC_WEEKDAY_SELECTIVE",
                    "weekend_vs_weekday_btc_mode": "WEEKEND" if weekend_mode else "WEEKDAY",
                    "btc_session_window_allowed": bool(self._btc_session_window_allowed(timestamp)),
                    "proxy_unavailable_fallback_used": bool(fallback_used),
                    "btc_last_structure_state": (
                        "BREAKOUT_UP"
                        if breakout_up
                        else "BREAKOUT_DOWN"
                        if breakout_down
                        else "RANGE_EDGE"
                        if range_position <= 0.25 or range_position >= 0.75
                        else "MID_RANGE"
                    ),
                    "btc_last_volatility_state": (
                        "VOLATILE"
                        if atr_ratio >= (2.2 / max(sensitivity, 0.8))
                        else "COMPRESSED"
                        if atr_ratio <= 0.85
                        else "NORMAL"
                    ),
                    "btc_last_spread_state": (
                        "TOO_WIDE"
                        if spread > max(self.btc_spread_cap_points, 100.0)
                        else "ELEVATED"
                        if spread_ratio >= 1.5
                        else "SANE"
                    ),
                    "btc_reason_no_candidate": "proxy_unavailable_price_action_fallback" if fallback_used else "awaiting_setup_confirmation",
                    "funding_proxy_available": bool(funding_available),
                    "liquidation_proxy_available": bool(liquidation_available),
                    "whale_flow_proxy_available": bool(whale_available),
                    "dxy_proxy_available": bool(dxy_available),
                    "weekend_gap_proxy_available": bool(gap_available),
                    "active_setup_windows": active_windows,
                    "setup_proxy_availability": {
                        "funding": bool(funding_available),
                        "liquidation": bool(liquidation_available),
                        "whale_flow": bool(whale_available),
                        "dxy_lag": bool(dxy_available),
                        "weekend_gap": bool(gap_available),
                    },
                }
            )
        elif normalized == "XAUUSD":
            dxy_available = self._feature_available(row, "dxy_ret_5", "dxy_ret_1")
            yield_available = self._feature_available(row, "us10y_ret_5", "us10y_ret_1", "yield_ret_5")
            diagnostics.update(
                {
                    "session_policy_current": "XAU_WEEKDAY_PRIORITY",
                    "xau_modes_enabled": {
                        "M5_GRID": True,
                        "M15_STRUCTURED": bool(self.xau_m15_enabled),
                        "M1_MICRO": bool(self.xau_m1_enabled),
                    },
                    "fix_window_active": bool(self._in_utc_window(timestamp, 14.75, 15.25)),
                    "btc_reason_no_candidate": "",
                    "dxy_proxy_available": bool(dxy_available),
                    "yield_proxy_available": bool(yield_available),
                    "active_setup_windows": [
                        name
                        for name, enabled in (
                            ("XAU_TOKYO_SESSION", session_name == "TOKYO"),
                            ("XAU_LONDON_FIX", self._in_utc_window(timestamp, 14.75, 15.25)),
                            ("XAU_LONDON_SESSION", session_name == "LONDON"),
                            ("XAU_OVERLAP_SESSION", session_name == "OVERLAP"),
                            ("XAU_NEW_YORK_SESSION", session_name == "NEW_YORK"),
                        )
                        if enabled
                    ],
                    "setup_proxy_availability": {
                        "dxy_alignment": bool(dxy_available),
                        "yield_alignment": bool(yield_available),
                    },
                }
            )
        elif normalized == "EURUSD":
            policy_name = "EURUSD_ASIA_SCALP" if session_name in {"SYDNEY", "TOKYO"} else "EURUSD_LONDON_EVENT"
            diagnostics.update(
                {
                    "session_policy_current": policy_name,
                    "active_setup_windows": (
                        ["EURUSD_LONDON_FIX"]
                        if self._in_utc_window(timestamp, 10.75, 11.10)
                        else ([f"EURUSD_{session_name}_SCALP"] if session_name in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"} else [])
                    ),
                    "setup_proxy_availability": {"dxy_alignment": self._feature_available(row, "dxy_ret_5", "dxy_ret_1")},
                }
            )
        elif normalized == "GBPUSD":
            policy_name = "GBPUSD_ASIA_SCALP" if session_name in {"SYDNEY", "TOKYO"} else "GBPUSD_FLOW_FADE"
            diagnostics.update(
                {
                    "session_policy_current": policy_name,
                    "active_setup_windows": (
                        ["GBPUSD_MONTH_END_FLOW"]
                        if self._is_last_friday_of_month(timestamp) and self._in_utc_window(timestamp, 11.0, 12.0)
                        else ([f"GBPUSD_{session_name}_SCALP"] if session_name in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"} else [])
                    ),
                    "setup_proxy_availability": {"month_end_window": True},
                }
            )
        elif normalized == "USDJPY":
            policy_name = "USDJPY_TOKYO_MICRO" if session_name in {"SYDNEY", "TOKYO"} else "USDJPY_GLOBAL_DAYTRADE"
            diagnostics.update(
                {
                    "session_policy_current": policy_name,
                    "active_setup_windows": (
                        ["USDJPY_TOKYO_LUNCH_FADE"]
                        if self._in_utc_window(timestamp, 3.0, 4.0)
                        else ([f"USDJPY_{session_name}_DAYTRADE"] if session_name in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"} else [])
                    ),
                    "setup_proxy_availability": {"intervention_headline": self._feature_available(row, "usdjpy_intervention_headline_score", "jpy_intervention_public_score")},
                }
            )
        elif normalized == "AUDJPY":
            diagnostics.update(
                {
                    "session_policy_current": "AUDJPY_ASIA_PRIORITY",
                    "active_setup_windows": (
                        [f"AUDJPY_{session_name}_MOMENTUM"]
                        if session_name in {"SYDNEY", "TOKYO"}
                        else ([f"AUDJPY_{session_name}_DAYTRADE"] if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else [])
                    ),
                    "setup_proxy_availability": {
                        "session_range_compression": self._feature_available(row, "m15_range_position_20", "m15_rolling_high_prev_20"),
                        "volume_expansion": self._feature_available(row, "m15_volume_ratio_20", "m5_volume_ratio_20"),
                    },
                }
            )
        elif normalized == "NZDJPY":
            diagnostics.update(
                {
                    "session_policy_current": "NZDJPY_ASIA_PRIORITY",
                    "active_setup_windows": (
                        [f"NZDJPY_{session_name}_MOMENTUM"]
                        if session_name in {"SYDNEY", "TOKYO"}
                        else ([f"NZDJPY_{session_name}_DAYTRADE"] if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else [])
                    ),
                    "setup_proxy_availability": {
                        "session_range_compression": self._feature_available(row, "m15_range_position_20", "m15_rolling_high_prev_20"),
                        "volume_expansion": self._feature_available(row, "m15_volume_ratio_20", "m5_volume_ratio_20"),
                    },
                }
            )
        elif normalized == "AUDNZD":
            diagnostics.update(
                {
                    "session_policy_current": "AUDNZD_ASIA_STRUCTURED",
                    "active_setup_windows": (
                        [f"AUDNZD_{session_name}_ROTATION"]
                        if session_name in {"SYDNEY", "TOKYO"}
                        else ([f"AUDNZD_{session_name}_DAYTRADE"] if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else [])
                    ),
                    "setup_proxy_availability": {
                        "structure_rotation": self._feature_available(row, "m15_range_position_20", "m15_rolling_high_prev_20"),
                        "volume_expansion": self._feature_available(row, "m15_volume_ratio_20", "m5_volume_ratio_20"),
                    },
                }
            )
        elif normalized in {"EURJPY", "GBPJPY"}:
            diagnostics.update(
                {
                    "session_policy_current": f"{normalized}_JPY_CROSS",
                    "active_setup_windows": (
                        [f"{normalized}_{session_name}_SESSION"]
                        if session_name in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}
                        else []
                    ),
                    "setup_proxy_availability": {
                        "jpy_cross_structure": self._feature_available(row, "m15_range_position_20", "m15_rolling_high_prev_20"),
                    },
                }
            )
        elif normalized == "NAS100":
            policy_name = "NAS100_SESSION_DAYTRADE"
            diagnostics.update(
                {
                    "session_policy_current": policy_name,
                    "active_setup_windows": (
                        ["NAS_PREMARKET"]
                        if self._in_utc_window(timestamp, 13.0, 13.5)
                        else ([f"NAS100_{session_name}_SCALP"] if session_name in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"} else [])
                    ),
                    "setup_proxy_availability": {
                        "futures_volume": self._feature_available(row, "nas_futures_volume_ratio_20", "nq_volume_ratio_20"),
                        "futures_direction": self._feature_available(row, "nas_futures_ret_15", "nq_ret_15"),
                    },
                }
            )
        elif normalized == "USOIL":
            current = self._utc_timestamp(timestamp)
            inventory_window = (
                (int(current.weekday()) == 1 and self._in_utc_window(timestamp, 20.5, 23.0))
                or (int(current.weekday()) == 2 and self._in_utc_window(timestamp, 14.0, 16.0))
            )
            diagnostics.update(
                {
                    "session_policy_current": "USOIL_EVENT_DRIVEN" if inventory_window else "USOIL_SESSION_DAYTRADE",
                    "active_setup_windows": (
                        ["USOIL_INVENTORY_EVENT"]
                        if inventory_window
                        else ([f"USOIL_{session_name}_SCALP"] if session_name in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"} else [])
                    ),
                    "setup_proxy_availability": {"inventory_surprise": self._feature_available(row, "oil_inventory_surprise_m", "usoil_inventory_surprise_m")},
                }
            )
        return diagnostics

    @staticmethod
    def _is_weekend_market_mode(timestamp) -> bool:
        return bool(is_weekend_market_mode(pd.Timestamp(timestamp).to_pydatetime()))

    @staticmethod
    def _utc_timestamp(timestamp) -> pd.Timestamp:
        current = pd.Timestamp(timestamp)
        if current.tzinfo is None:
            return current.tz_localize(timezone.utc)
        return current.tz_convert(timezone.utc)

    def _btc_session_window_allowed(self, timestamp) -> bool:
        current = self._utc_timestamp(timestamp)
        hour = current.hour + (current.minute / 60.0) + (current.second / 3600.0)
        return float(self.btc_allowed_start_hour_utc) <= hour < float(self.btc_allowed_end_hour_utc)

    def _btc_funding_entry_window(self, timestamp) -> bool:
        current = self._utc_timestamp(timestamp)
        snapshot_hours = {0, 8, 16}
        next_hour = (current.hour + 1) % 24
        return next_hour in snapshot_hours and current.minute >= 45

    def _btc_heartbeat_fallback(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        normalized = self._normalize_symbol(symbol)
        session_name = str(session.session_name).upper()
        weekend_mode = self._is_weekend_market_mode(timestamp)
        if normalized != "BTCUSD" or not weekend_mode:
            return []
        if session_name not in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}:
            return []
        current = self._utc_timestamp(datetime.now(timezone.utc))
        emit_key = f"{normalized}:{session_name}"
        last_emit = self.btc_heartbeat_emit_state.get(emit_key)
        if last_emit is not None and (current.to_pydatetime() - last_emit) < timedelta(seconds=max(30, int(self.btc_heartbeat_cadence_seconds or 120))):
            return []

        atr = max(float(row.get("m5_atr_14", row.get("m15_atr_14", 0.0)) or 0.0), 1e-6)
        close = float(row.get("m5_close", row.get("close", 0.0)) or 0.0)
        ema20 = float(row.get("m5_ema_20", close) or close)
        ema50 = float(row.get("m5_ema_50", close) or close)
        h1_ema20 = float(row.get("h1_ema_20", ema20) or ema20)
        h1_ema50 = float(row.get("h1_ema_50", ema50) or ema50)
        range_position = clamp(float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5), 0.0, 1.0)
        atr_ratio = float(row.get("m5_atr_pct_of_avg", 1.0) or 1.0)
        spread_points = float(row.get("m5_spread", row.get("spread_points", 0.0)) or 0.0)
        point_size = max(
            float(row.get("point", 0.0) or 0.0),
            float(row.get("trade_tick_size", row.get("tick_size", 0.0)) or 0.0),
        )
        if point_size <= 0.0:
            digits = int(row.get("digits", 0) or 0)
            if digits > 0:
                point_size = 10.0 ** (-digits)
        if point_size <= 0.0 and symbol_asset_class(symbol) == "crypto":
            point_size = 0.01
        spread = spread_points * point_size if point_size > 0.0 else spread_points
        stored_spread_ratio = float(row.get("m5_spread_ratio_20", 1.0) or 1.0)
        computed_spread_ratio = spread / max(atr, 1e-6)
        spread_ratio = computed_spread_ratio if point_size > 0.0 or stored_spread_ratio > 5.0 else stored_spread_ratio
        market_instability = clamp(float(row.get("market_instability_score", 0.0) or 0.0), 0.0, 1.0)
        seasonality_edge = clamp(float(row.get("seasonality_edge_score", 0.5) or 0.5), 0.0, 1.0)
        body_efficiency = clamp(abs(float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.0)) or 0.0)), 0.0, 1.0)
        momentum = float(row.get("m5_ret_1", 0.0) or 0.0)
        higher_momentum = float(row.get("m15_ret_1", momentum) or momentum)
        rsi = clamp(float(row.get("m5_rsi_14", 50.0) or 50.0), 0.0, 100.0)
        upper_wick_ratio = clamp(float(row.get("m5_upper_wick_ratio", 0.0) or 0.0), 0.0, 1.0)
        lower_wick_ratio = clamp(float(row.get("m5_lower_wick_ratio", 0.0) or 0.0), 0.0, 1.0)
        move_30m_pct = abs(
            float(
                row.get(
                    "quote_move_30m_pct",
                    row.get("m5_ret_5", row.get("m15_ret_2", row.get("m15_ret_1", 0.0))),
                )
                or 0.0
            )
        )
        if close <= 0.0 or atr <= 0.0:
            return []
        raw_spread_cap_points = max(float(self.btc_spread_cap_points), self.max_spread_points * 34.0, 100.0)
        raw_spread_cap = raw_spread_cap_points * point_size if point_size > 0.0 else raw_spread_cap_points
        if (
            spread_ratio > 3.25
            or atr_ratio > 3.60
            or (spread > raw_spread_cap and spread_ratio >= 1.50)
            or move_30m_pct >= max(0.018, float(self.btc_volatility_pause_move_pct_30m) * 1.20)
        ):
            return []

        short_bias = (ema20 - ema50) / atr
        higher_bias = (h1_ema20 - h1_ema50) / max(atr * 2.0, 1e-6)
        distance_to_ema = (close - ema20) / atr

        buy_score = 0.0
        sell_score = 0.0
        if ema20 >= ema50:
            buy_score += 1.0
        if ema20 <= ema50:
            sell_score += 1.0
        if momentum >= 0.0:
            buy_score += 0.8
        if momentum <= 0.0:
            sell_score += 0.8
        if higher_momentum >= 0.0:
            buy_score += 0.4
        if higher_momentum <= 0.0:
            sell_score += 0.4
        if close >= ema20:
            buy_score += 0.5
        if close <= ema20:
            sell_score += 0.5
        if range_position <= 0.42:
            buy_score += 0.35
        if range_position >= 0.58:
            sell_score += 0.35
        if range_position <= 0.28 and rsi <= 42.0:
            buy_score += 0.35
        if range_position >= 0.72 and rsi >= 58.0:
            sell_score += 0.35
        if lower_wick_ratio >= 0.28 and range_position <= 0.40:
            buy_score += 0.30 + min(0.12, max(0.0, lower_wick_ratio - upper_wick_ratio) * 0.40)
        if upper_wick_ratio >= 0.28 and range_position >= 0.60:
            sell_score += 0.30 + min(0.12, max(0.0, upper_wick_ratio - lower_wick_ratio) * 0.40)
        if body_efficiency >= 0.32:
            if momentum >= 0.0:
                buy_score += 0.25
            if momentum <= 0.0:
                sell_score += 0.25
        if higher_bias > 0.15:
            buy_score += 0.15
        elif higher_bias < -0.15:
            sell_score += 0.15
        if distance_to_ema >= 0.95 and momentum <= 0.0:
            sell_score += 0.22
        elif distance_to_ema <= -0.95 and momentum >= 0.0:
            buy_score += 0.22

        if buy_score == sell_score:
            if range_position >= 0.65 and upper_wick_ratio > lower_wick_ratio:
                sell_score += 0.12
            elif range_position <= 0.35 and lower_wick_ratio > upper_wick_ratio:
                buy_score += 0.12
            elif distance_to_ema >= 0.0:
                buy_score += 0.1
            else:
                sell_score += 0.1

        side = "BUY" if buy_score >= sell_score else "SELL"
        directional_score = buy_score - sell_score
        setup = (
            "BTC_TOKYO_DRIFT_SCALP"
            if session_name in {"SYDNEY", "TOKYO"}
            else "BTC_LONDON_IMPULSE_SCALP"
            if session_name == "LONDON"
            else "BTC_NY_LIQUIDITY"
        )
        confluence = clamp(
            2.82
            + min(0.16, abs(short_bias) * 0.08)
            + min(0.14, abs(higher_bias) * 0.06)
            + min(0.14, body_efficiency * 0.22)
            + (0.10 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.04),
            0.0,
            5.0,
        )
        score_hint = 0.60 if session_name in {"SYDNEY", "TOKYO"} else 0.64
        reason = (
            "BTC weekend heartbeat fallback buy on executable short-term drift"
            if side == "BUY"
            else "BTC weekend heartbeat fallback sell on executable short-term drift"
        )
        candidate = SignalCandidate(
            signal_id=deterministic_id(symbol, "router", "btc-heartbeat-fallback", side, current),
            setup=setup,
            side=side,
            score_hint=score_hint,
            reason=reason,
            stop_atr=0.78,
            tp_r=1.18,
            strategy_family="SCALP",
            confluence_score=confluence,
            confluence_required=2.45,
            meta={
                "timeframe": "M15",
                "atr_field": "m5_atr_14",
                "allow_ai_approve_small": True,
                "approve_small_min_probability": 0.36,
                "approve_small_min_confluence": 2.40,
                "btc_strategy": "WEEKEND_HEARTBEAT_FALLBACK",
                "setup_family": "PRICE_ACTION",
                "btc_min_ai_confidence": 0.34,
                "proxyless_price_action_mode": True,
                "proxyless_weekend_heartbeat_mode": True,
                "btc_weekend_force_emit": True,
                "throughput_recovery_active": True,
                "weekend_heartbeat_emit_session": session_name,
                "weekend_heartbeat_short_bias": float(short_bias),
                "weekend_heartbeat_higher_bias": float(higher_bias),
                "weekend_heartbeat_distance_to_ema": float(distance_to_ema),
                "weekend_heartbeat_directional_score": float(directional_score),
                "weekend_heartbeat_rsi": float(rsi),
                "weekend_heartbeat_upper_wick_ratio": float(upper_wick_ratio),
                "weekend_heartbeat_lower_wick_ratio": float(lower_wick_ratio),
                "quality_tier": "B",
                "router_rank_score": float(score_hint),
                "regime_fit": max(0.48, min(0.72, 0.48 + abs(float(higher_bias)) * 0.12)),
                "session_fit": 0.58 if session_name in {"SYDNEY", "TOKYO"} else 0.64,
                "volatility_fit": 0.56 if atr_ratio <= 2.8 else 0.50,
                "pair_behavior_fit": 0.54 if session_name in {"SYDNEY", "TOKYO"} else 0.58,
                "execution_quality_fit": 0.58 if spread_ratio <= 2.8 else 0.52,
                "entry_timing_score": 0.52 if body_efficiency < 0.18 else 0.58,
                "structure_cleanliness_score": 0.48 if market_instability >= 0.70 else 0.54,
                "strategy_recent_performance_seed": 0.56,
                "seasonality_edge_score": float(seasonality_edge),
                "market_instability_score": float(market_instability),
                "quote_move_30m_pct": float(move_30m_pct),
            },
        )
        self.btc_heartbeat_emit_state[emit_key] = current.to_pydatetime()
        return [candidate]

    @staticmethod
    def _candidate_family(candidate: SignalCandidate) -> str:
        meta = dict(candidate.meta or {})
        return normalize_strategy_family(
            str(meta.get("setup_family") or candidate.strategy_family or candidate.setup or "TREND")
        )

    @staticmethod
    def _candidate_strategy_key(candidate: SignalCandidate, symbol: str) -> str:
        meta = dict(candidate.meta or {})
        return str(meta.get("strategy_key") or resolve_strategy_key(symbol, str(candidate.setup or ""))).strip().upper()

    def _compression_transition_context(self, features: pd.DataFrame) -> dict[str, float | bool | str]:
        if features.empty or len(features.index) < 2:
            return {
                "active": False,
                "current_state": "NEUTRAL",
                "score_bonus": 0.0,
                "size_multiplier": 1.0,
            }

        def _state_from_row(hist_row: pd.Series) -> str:
            atr_ratio = clamp(float(hist_row.get("m5_atr_pct_of_avg", 1.0) or 1.0), 0.0, 3.0)
            spread_ratio = clamp(float(hist_row.get("m5_spread_ratio_20", 1.0) or 1.0), 0.0, 3.0)
            range_position = clamp(float(hist_row.get("m15_range_position_20", hist_row.get("m5_range_position_20", 0.5)) or 0.5), 0.0, 1.0)
            m5_ret_1 = float(hist_row.get("m5_ret_1", 0.0) or 0.0)
            compression_score = clamp(1.15 - atr_ratio, 0.0, 1.0)
            spread_compression = clamp((1.20 - spread_ratio) / 0.30, 0.0, 1.0)
            range_tightness = clamp(1.0 - abs(range_position - 0.50) * 2.0, 0.0, 1.0)
            expansion_score = clamp(
                (0.45 * compression_score)
                + (0.25 * spread_compression)
                + (0.15 * range_tightness)
                + (0.15 * max(0.0, 1.0 - abs(m5_ret_1) * 900.0)),
                0.0,
                1.0,
            )
            if atr_ratio <= 0.92 and spread_ratio <= 1.20 and abs(m5_ret_1) <= 0.0012 and 0.35 <= range_position <= 0.65:
                return "COMPRESSION"
            if expansion_score >= 0.42 and ((range_position >= 0.70 and m5_ret_1 > 0.0) or (range_position <= 0.30 and m5_ret_1 < 0.0)):
                return "EXPANSION_READY"
            return "NEUTRAL"

        recent = features.tail(6)
        states = [_state_from_row(recent.iloc[index]) for index in range(len(recent.index))]
        current_state = str(states[-1] or "NEUTRAL")
        previous_states = list(states[:-1][-5:])
        transition_active = current_state == "EXPANSION_READY" and any(state == "COMPRESSION" for state in previous_states)
        return {
            "active": bool(transition_active),
            "current_state": str(current_state),
            "score_bonus": float(self.transition_score_bonus) if transition_active else 0.0,
            "size_multiplier": float(self.transition_size_multiplier) if transition_active else 1.0,
        }

    def _apply_same_bar_substitution(
        self,
        *,
        candidates: list[SignalCandidate],
        symbol: str,
        session_name: str,
        regime_state: str,
    ) -> list[SignalCandidate]:
        normalized = self._normalize_symbol(symbol)
        session_key = str(session_name or "").upper()
        regime_key = runtime_regime_state(regime_state)
        if not candidates:
            return candidates
        strategy_keys = {self._candidate_strategy_key(candidate, normalized) for candidate in candidates}
        blocked: set[str] = set()
        if (
            normalized == "NAS100"
            and session_key == "TOKYO"
            and regime_key == "TRENDING"
            and "NAS100_LIQUIDITY_SWEEP_REVERSAL" in strategy_keys
            and "NAS100_MOMENTUM_IMPULSE" in strategy_keys
        ):
            blocked.add("NAS100_MOMENTUM_IMPULSE")
        if (
            normalized == "EURJPY"
            and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
            and "EURJPY_MOMENTUM_IMPULSE" in strategy_keys
            and "EURJPY_SESSION_PULLBACK_CONTINUATION" in strategy_keys
        ):
            blocked.add("EURJPY_SESSION_PULLBACK_CONTINUATION")
        if (
            normalized == "EURUSD"
            and session_key in {"OVERLAP", "NEW_YORK"}
            and regime_key == "LOW_LIQUIDITY_CHOP"
            and "EURUSD_VWAP_PULLBACK" in strategy_keys
            and "EURUSD_LONDON_BREAKOUT" in strategy_keys
        ):
            blocked.add("EURUSD_LONDON_BREAKOUT")
        if (
            normalized == "GBPJPY"
            and regime_key == "LOW_LIQUIDITY_CHOP"
            and "GBPJPY_LIQUIDITY_SWEEP_REVERSAL" in strategy_keys
        ):
            blocked.update({"GBPJPY_MOMENTUM_IMPULSE", "GBPJPY_SESSION_PULLBACK_CONTINUATION"})
        if (
            normalized == "BTCUSD"
            and session_key == "TOKYO"
            and "BTCUSD_PRICE_ACTION_CONTINUATION" in strategy_keys
            and "BTCUSD_TREND_SCALP" in strategy_keys
        ):
            blocked.add("BTCUSD_TREND_SCALP")
        if not blocked:
            return candidates
        return [
            candidate
            for candidate in candidates
            if self._candidate_strategy_key(candidate, normalized) not in blocked
        ]

    def _recycle_key(self, *, timestamp: Any, symbol: str, family: str, side: str) -> str:
        day_key = str(trading_day_key_for_timestamp(pd.Timestamp(timestamp).to_pydatetime()))
        return f"{day_key}|{self._normalize_symbol(symbol)}|{normalize_strategy_family(family)}|{str(side or '').upper()}"

    def _cleanup_recycle_queue(self, timestamp: Any) -> None:
        if not self.recycle_queue:
            return
        active_day = str(trading_day_key_for_timestamp(pd.Timestamp(timestamp).to_pydatetime()))
        stale_keys = [key for key in self.recycle_queue if not str(key).startswith(f"{active_day}|")]
        for key in stale_keys:
            self.recycle_queue.pop(key, None)

    def _eligible_recycle_session(self, *, symbol: str, session_name: str, weekend_mode: bool) -> bool:
        session_key = str(session_name or "").upper()
        symbol_key = self._normalize_symbol(symbol)
        if symbol_key == "BTCUSD" and weekend_mode:
            return True
        return session_key in {"LONDON", "OVERLAP", "NEW_YORK"}

    @staticmethod
    def _recycle_eligible_reject_reason(reason: str) -> bool:
        normalized = str(reason or "").strip().lower()
        return normalized in {
            "btc_tokyo_price_action_ranging",
            "nas_vwap_trend_tokyo_mean_reversion",
        }

    def _queue_recyclable_candidate(
        self,
        *,
        candidate: SignalCandidate,
        symbol: str,
        session_name: str,
        timestamp: Any,
    ) -> None:
        family = self._candidate_family(candidate)
        key = self._recycle_key(timestamp=timestamp, symbol=symbol, family=family, side=candidate.side)
        if key in self.recycle_queue:
            return
        payload = replace(candidate, meta=dict(candidate.meta or {}))
        payload.meta["recycle_origin_session"] = str(session_name)
        payload.meta["recycle_session"] = False
        payload.meta["recycle_boost_applied"] = 0.0
        self.recycle_queue[key] = {
            "candidate": payload,
            "created_at": str(pd.Timestamp(timestamp)),
            "session_name": str(session_name),
        }

    def _pull_recycled_candidates(
        self,
        *,
        symbol: str,
        session_name: str,
        timestamp: Any,
        weekend_mode: bool,
    ) -> list[SignalCandidate]:
        self._cleanup_recycle_queue(timestamp)
        if not self._eligible_recycle_session(symbol=symbol, session_name=session_name, weekend_mode=weekend_mode):
            return []
        symbol_key = self._normalize_symbol(symbol)
        active_day = str(trading_day_key_for_timestamp(pd.Timestamp(timestamp).to_pydatetime()))
        recycled: list[SignalCandidate] = []
        for key, payload in list(self.recycle_queue.items()):
            if not key.startswith(f"{active_day}|{symbol_key}|"):
                continue
            candidate = replace(payload["candidate"], meta=dict(payload["candidate"].meta or {}))
            base_regime_fit = float(candidate.meta.get("regime_fit", 0.0) or 0.0)
            candidate.meta["recycle_session"] = True
            candidate.meta["recycle_origin_session"] = str(payload.get("session_name") or "")
            candidate.meta["recycle_boost_applied"] = float(self.recycle_regime_boost)
            candidate.meta["regime_fit"] = clamp(base_regime_fit + float(self.recycle_regime_boost), 0.0, 1.0)
            candidate.meta["router_rank_score"] = clamp(
                float(candidate.meta.get("router_rank_score", candidate.score_hint or 0.0) or 0.0) + 0.03,
                0.0,
                1.0,
            )
            recycled.append(candidate)
            self.recycle_queue.pop(key, None)
        return recycled

    def _prime_compression_burst(
        self,
        *,
        symbol: str,
        session_name: str,
        regime: RegimeClassification,
        weekend_mode: bool,
    ) -> None:
        symbol_key = self._normalize_symbol(symbol)
        state_key = f"{symbol_key}|{str(session_name or '').upper()}"
        compression_state = str(getattr(regime, "details", {}).get("compression_state", getattr(regime, "details", {}).get("compression_proxy_state", "NEUTRAL")) or "NEUTRAL").upper()
        learning_state = self._learning_pattern_state(symbol=symbol_key, session_name=session_name)
        density_profile = self._density_profile_state(symbol=symbol_key, session_name=session_name, weekend_mode=weekend_mode)
        prime_xau_session = symbol_key == "XAUUSD" and str(session_name or "").upper() in {"LONDON", "OVERLAP", "NEW_YORK"}
        if compression_state == "EXPANSION_READY" or (prime_xau_session and compression_state == "COMPRESSION"):
            multiplier = float(self.compression_burst_size_multiplier)
            if symbol_key == "XAUUSD":
                multiplier = max(multiplier, float(self.xau_grid_expansion_burst_size_multiplier))
                if prime_xau_session:
                    multiplier = max(multiplier, 1.0 + (max(0.0, float(self.xau_prime_session_mult) - 1.0) * 0.20))
            if bool(learning_state.get("throughput_recovery_active", False)):
                multiplier = max(
                    multiplier,
                    1.0
                    + (0.05 * float(learning_state.get("trajectory_catchup_pressure", 0.0) or 0.0))
                    + (0.05 if bool(learning_state.get("promoted_pattern", False)) else 0.0),
                )
            if bool(density_profile.get("active", False)):
                multiplier = max(multiplier, float(density_profile.get("compression_multiplier", 1.0) or 1.0))
            self.compression_burst_state[state_key] = {
                "remaining": max(
                    0,
                    int(self.compression_burst_candidates)
                    + (1 if bool(learning_state.get("throughput_recovery_active", False)) else 0)
                    + (1 if bool(density_profile.get("active", False)) else 0),
                ),
                "multiplier": float(multiplier),
            }
            if bool(density_profile.get("active", False)):
                self.compression_burst_state[state_key]["remaining"] = max(
                    int(self.compression_burst_state[state_key]["remaining"]),
                    int(density_profile.get("compression_candidates", 0) or 0),
                )
            if prime_xau_session:
                self.compression_burst_state[state_key]["remaining"] = max(
                    int(self.compression_burst_state[state_key]["remaining"]),
                    int(self.xau_m5_burst_target),
                )
        elif state_key not in self.compression_burst_state:
            self.compression_burst_state[state_key] = {"remaining": 0, "multiplier": 1.0}

    def _apply_compression_burst(
        self,
        *,
        candidates: list[SignalCandidate],
        symbol: str,
        session_name: str,
    ) -> list[SignalCandidate]:
        state_key = f"{self._normalize_symbol(symbol)}|{str(session_name or '').upper()}"
        state = dict(self.compression_burst_state.get(state_key) or {})
        remaining = max(0, int(state.get("remaining", 0) or 0))
        multiplier = max(1.0, float(state.get("multiplier", 1.0) or 1.0))
        if remaining <= 0:
            return candidates
        output: list[SignalCandidate] = []
        for candidate in candidates:
            if remaining > 0 and not bool((candidate.meta or {}).get("router_reject")):
                candidate.meta["compression_burst_size_multiplier"] = float(multiplier)
                remaining -= 1
            output.append(candidate)
        self.compression_burst_state[state_key] = {"remaining": remaining, "multiplier": multiplier}
        return output

    def generate(
        self,
        symbol: str,
        features: pd.DataFrame,
        regime: RegimeClassification,
        session: SessionContext,
        strategy_engine: StrategyEngine,
        open_positions: list[dict] | None,
        max_positions_per_symbol: int,
        current_time=None,
    ) -> list[SignalCandidate]:
        if features.empty:
            return []
        row = features.iloc[-1]
        timestamp = current_time if current_time is not None else self._resolve_timestamp(features, row)
        normalized = self._normalize_symbol(symbol)
        open_positions = open_positions or []
        session_name = str(session.session_name).upper()
        weekend_mode = self._is_weekend_market_mode(timestamp)
        self._cleanup_recycle_queue(timestamp)
        self._prime_compression_burst(
            symbol=normalized,
            session_name=session_name,
            regime=regime,
            weekend_mode=weekend_mode,
        )
        transition_context = self._compression_transition_context(features)

        base = strategy_engine.generate(
            features=features,
            regime=regime.label,
            open_positions=open_positions,
            max_positions_per_symbol=max_positions_per_symbol,
        )
        routed: list[SignalCandidate] = []
        if normalized in {"EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "AUDNZD"}:
            routed.extend(self._forex_trend(symbol, row, regime, session, timestamp))
            routed.extend(self._forex_range(symbol, row, regime, session, timestamp))
            routed.extend(self._forex_breakout_retest(symbol, row, regime, session, timestamp))
            routed.extend(self._liquidity_sweep_reclaim(symbol, row, regime, session, timestamp))
            routed.extend(self._session_pullback_continuation(symbol, row, regime, session, timestamp))
            routed.extend(self._session_momentum_boost(symbol, row, regime, session, timestamp))
            routed.extend(self._session_drift_scalp(symbol, row, regime, session, timestamp))
            if normalized == "EURUSD":
                routed.extend(self._eurusd_fix_flow(symbol, row, regime, session, timestamp))
            if normalized == "GBPUSD":
                routed.extend(self._gbpusd_month_end_flow(symbol, row, regime, session, timestamp))
            if normalized == "USDJPY":
                routed.extend(self._usdjpy_public_intervention(symbol, row, regime, session, timestamp))
                routed.extend(self._usdjpy_tokyo_lunch_fade(symbol, row, regime, session, timestamp))
            if normalized in {"AUDJPY", "NZDJPY"}:
                routed.extend(self._asia_momentum_breakout(symbol, row, regime, session, timestamp))
                routed.extend(self._asia_continuation_pullback(symbol, row, regime, session, timestamp))
                routed.extend(self._asia_rotation_reclaim(symbol, row, regime, session, timestamp))
                routed.extend(self._asia_drift_continuation(symbol, row, regime, session, timestamp))
            if normalized == "AUDJPY":
                routed.extend(self._audjpy_sydney_range_break(symbol, row, regime, session, timestamp))
                routed.extend(self._audjpy_london_carry_continuation(symbol, row, regime, session, timestamp))
            if normalized == "NZDJPY":
                routed.extend(self._nzdjpy_sydney_breakout_retest(symbol, row, regime, session, timestamp))
            if normalized == "AUDNZD":
                routed.extend(self._audnzd_rotation_breakout(symbol, row, regime, session, timestamp))
                routed.extend(self._audnzd_rotation_pullback(symbol, row, regime, session, timestamp))
                routed.extend(self._audnzd_range_rejection(symbol, row, regime, session, timestamp))
                routed.extend(self._audnzd_compression_release(symbol, row, regime, session, timestamp))
                routed.extend(self._asia_drift_continuation(symbol, row, regime, session, timestamp))
        elif normalized == "XAUUSD":
            routed.extend(self._xau_fix_flow(symbol, row, regime, session, timestamp))
            routed.extend(self._xau_fakeout(symbol, row, regime, session, timestamp))
            routed.extend(self._xau_m1_micro_scalper(symbol, row, regime, session, timestamp))
            routed.extend(self._xau_m15_structured(symbol, row, regime, session, timestamp))
        elif normalized == "NAS100":
            routed.extend(self._nas_premarket_futures(symbol, row, regime, session, timestamp))
            routed.extend(self._nas_session_scalper(symbol, row, regime, session, timestamp))
            routed.extend(self._liquidity_sweep_reclaim(symbol, row, regime, session, timestamp))
            routed.extend(self._session_pullback_continuation(symbol, row, regime, session, timestamp))
            routed.extend(self._session_momentum_boost(symbol, row, regime, session, timestamp))
        elif normalized == "USOIL":
            routed.extend(self._oil_inventory_event(symbol, row, regime, session, timestamp))
            routed.extend(self._oil_inventory_scalper(symbol, row, regime, session, timestamp))
            routed.extend(self._liquidity_sweep_reclaim(symbol, row, regime, session, timestamp))
            routed.extend(self._session_pullback_continuation(symbol, row, regime, session, timestamp))
            routed.extend(self._session_momentum_boost(symbol, row, regime, session, timestamp))
        elif normalized == "BTCUSD":
            routed.extend(self._btc(symbol, row, regime, session, timestamp))

        filtered_base = self._filter_base(base, normalized, regime, session, row)
        recovery_candidates: list[SignalCandidate] = []
        if not routed and not filtered_base:
            recovery_candidates = self._live_candidate_recovery(
                symbol=normalized,
                row=row,
                regime=regime,
                session=session,
                timestamp=timestamp,
            )
        heartbeat_candidates: list[SignalCandidate] = []
        if normalized == "BTCUSD" and weekend_mode:
            heartbeat_candidates = self._btc_heartbeat_fallback(
                symbol=normalized,
                row=row,
                regime=regime,
                session=session,
                timestamp=timestamp,
            )
        ranked_candidates = [
            self._enrich_candidate(
                symbol=normalized,
                row=row,
                regime=regime,
                session=session,
                candidate=candidate,
            )
            for candidate in (routed + filtered_base + recovery_candidates + heartbeat_candidates)
        ]
        if bool(transition_context.get("active")):
            for candidate in ranked_candidates:
                strategy_key = self._candidate_strategy_key(candidate, normalized)
                transition_eligible = (
                    (normalized == "BTCUSD" and strategy_key in {"BTCUSD_PRICE_ACTION_CONTINUATION", "BTCUSD_VOLATILE_RETEST"})
                    or (normalized == "NAS100" and strategy_key == "NAS100_LIQUIDITY_SWEEP_REVERSAL")
                    or (normalized == "XAUUSD" and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID")
                )
                candidate.meta["transition_momentum"] = float(transition_context.get("score_bonus", 0.0) or 0.0) if transition_eligible else 0.0
                candidate.meta["transition_momentum_size_multiplier"] = float(transition_context.get("size_multiplier", 1.0) or 1.0) if transition_eligible else 1.0
                candidate.meta["correlation_penalty"] = float(candidate.meta.get("correlation_penalty", 0.0) or 0.0)
                if transition_eligible:
                    candidate.meta["router_rank_score"] = clamp(
                        float(candidate.meta.get("router_rank_score", candidate.score_hint or 0.0) or 0.0)
                        + float(transition_context.get("score_bonus", 0.0) or 0.0),
                        0.0,
                        1.0,
                    )
        ranked_candidates = self._apply_same_bar_substitution(
            candidates=ranked_candidates,
            symbol=normalized,
            session_name=session_name,
            regime_state=str(getattr(regime, "state_label", regime.label) or regime.label),
        )
        recyclable_sessions = {"SYDNEY", "TOKYO"}
        for candidate in ranked_candidates:
            meta = dict(candidate.meta or {})
            if (
                session_name in recyclable_sessions
                and str(meta.get("quality_tier") or "").upper() == "B"
                and bool(meta.get("router_reject"))
                and self._recycle_eligible_reject_reason(str(meta.get("router_reject_reason") or ""))
                and not bool(meta.get("recycle_session"))
            ):
                self._queue_recyclable_candidate(
                    candidate=candidate,
                    symbol=normalized,
                    session_name=session_name,
                    timestamp=timestamp,
                )
        ranked_candidates = [candidate for candidate in ranked_candidates if not bool((candidate.meta or {}).get("router_reject"))]
        if not ranked_candidates and not recovery_candidates:
            late_recovery_candidates = self._live_candidate_recovery(
                symbol=normalized,
                row=row,
                regime=regime,
                session=session,
                timestamp=timestamp,
            )
            if late_recovery_candidates:
                ranked_candidates = [
                    self._enrich_candidate(
                        symbol=normalized,
                        row=row,
                        regime=regime,
                        session=session,
                        candidate=candidate,
                    )
                    for candidate in late_recovery_candidates
                ]
                ranked_candidates = [
                    candidate for candidate in ranked_candidates if not bool((candidate.meta or {}).get("router_reject"))
                ]
        ranked_candidates.extend(
            self._pull_recycled_candidates(
                symbol=normalized,
                session_name=session_name,
                timestamp=timestamp,
                weekend_mode=weekend_mode,
            )
        )
        ranked_candidates = self._apply_compression_burst(
            candidates=ranked_candidates,
            symbol=normalized,
            session_name=session_name,
        )
        return self._unique(
            sorted(
                ranked_candidates,
                key=lambda item: float((item.meta or {}).get("router_rank_score", item.score_hint or 0.0)),
                reverse=True,
            )
        )

    def _forex_trend(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        sensitivity = self._symbol_sensitivity(symbol, self.forex_sensitivity)
        symbol_key = self._normalize_symbol(symbol)
        session_name = str(session.session_name).upper()
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        raw_regime_label = str(getattr(regime, "state_label", regime.label) or regime.label).strip().upper()
        if "TREND" not in session.allowed_strategies:
            return []
        if not bool(regime.details.get("trend_flag", regime.label == "TRENDING")):
            return []
        if symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD"}:
            return []
        if symbol_key in {"EURUSD", "GBPUSD"} and session_name in {"SYDNEY", "TOKYO"}:
            return []
        if symbol_key == "USDJPY" and session_name in {"SYDNEY", "TOKYO"} and regime_state not in {"TRENDING", "BREAKOUT_EXPANSION"}:
            return []
        atr = max(float(row.get("m5_atr_14", 0.0)), 1e-6)
        close = float(row.get("m5_close", 0.0))
        ema20 = float(row.get("m5_ema_20", close))
        ema50 = float(row.get("m5_ema_50", close))
        h1_up = float(row.get("h1_ema_50", 0.0)) > float(row.get("h1_ema_200", 0.0))
        h1_down = float(row.get("h1_ema_50", 0.0)) < float(row.get("h1_ema_200", 0.0))
        pullback = abs(close - ema20) <= (atr * (0.95 + (0.15 * max(0.0, sensitivity - 1.0))))
        spread_ok = float(row.get("m5_spread", 0.0)) <= self._spread_reference_limit(symbol_key)
        volume_ratio = self._ratio_or_neutral(row.get("m5_volume_ratio_20", row.get("m15_volume_ratio_20", 1.0)))
        body_efficiency = float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)) or 0.55)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        near_resistance = (float(row.get("m15_rolling_high_20", close + atr)) - close) <= (atr * 0.20)
        near_support = (close - float(row.get("m15_rolling_low_20", close - atr))) <= (atr * 0.20)
        momentum_up = float(row.get("m5_macd_hist_slope", 0.0)) > 0 and int(row.get("m5_bullish", 0)) == 1
        momentum_down = float(row.get("m5_macd_hist_slope", 0.0)) < 0 and int(row.get("m5_bearish", 0)) == 1
        if symbol_key == "USDJPY" and session_name in {"SYDNEY", "TOKYO"}:
            volume_floor = 1.14 if session_name == "SYDNEY" else 1.16
            body_floor = 0.62 if session_name == "SYDNEY" else 0.64
            if volume_ratio < volume_floor or body_efficiency < body_floor:
                return []
            if range_position >= (0.82 if session_name == "SYDNEY" else 0.84) or range_position <= (0.18 if session_name == "SYDNEY" else 0.16):
                return []
        output: list[SignalCandidate] = []

        if spread_ok and pullback and h1_up and ema20 >= ema50 and momentum_up and (not near_resistance or sensitivity > 1.0):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "forex-trend", "BUY", timestamp),
                    setup="FOREX_TREND_PULLBACK",
                    side="BUY",
                    score_hint=0.60 + min(0.07, max(0.0, sensitivity - 1.0) * 0.10),
                    reason="H1 trend + M5 pullback continuation",
                    stop_atr=1.2,
                    tp_r=1.8,
                    strategy_family="TREND",
                    confluence_score=3.8,
                    confluence_required=4.0,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.52,
                        "approve_small_min_confluence": 3.4,
                    },
                )
            )
        if spread_ok and pullback and h1_down and ema20 <= ema50 and momentum_down and (not near_support or sensitivity > 1.0):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "forex-trend", "SELL", timestamp),
                    setup="FOREX_TREND_PULLBACK",
                    side="SELL",
                    score_hint=0.60 + min(0.07, max(0.0, sensitivity - 1.0) * 0.10),
                    reason="H1 trend + M5 pullback continuation",
                    stop_atr=1.2,
                    tp_r=1.8,
                    strategy_family="TREND",
                    confluence_score=3.8,
                    confluence_required=4.0,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.52,
                        "approve_small_min_confluence": 3.4,
                    },
                )
            )
        return output

    def _eurusd_fix_flow(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if not self._in_utc_window(timestamp, 10.75, 11.10):
            return []
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_reference_limit(symbol):
            return []
        dxy_move = float(row.get("dxy_ret_5", row.get("dxy_ret_1", 0.0)))
        if abs(dxy_move) <= 1e-6:
            return []
        vol_ratio = self._ratio_or_neutral(row.get("m5_volume_ratio_20", 1.0))
        if vol_ratio < 0.95:
            return []
        side = "BUY" if dxy_move < 0 else "SELL"
        return [
            SignalCandidate(
                signal_id=deterministic_id(symbol, "router", "eur-fix", side, timestamp),
                setup="EURUSD_FIX_FLOW",
                side=side,
                score_hint=0.63,
                reason="EURUSD London fix flow with DXY alignment",
                stop_atr=0.95,
                tp_r=1.8,
                strategy_family="TREND",
                confluence_score=3.9,
                confluence_required=3.8,
                meta={
                    "timeframe": "M15",
                    "atr_field": "m15_atr_14",
                    "policy_window": "LONDON_FIX",
                    "setup_family": "FLOW",
                },
            )
        ]

    def _gbpusd_month_end_flow(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if not (self._is_last_friday_of_month(timestamp) and self._in_utc_window(timestamp, 11.0, 12.0)):
            return []
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_reference_limit(symbol):
            return []
        monthly_trend = float(row.get("h4_ema_50", row.get("m15_close", 0.0))) - float(row.get("h4_ema_200", row.get("m15_close", 0.0)))
        if abs(monthly_trend) <= 1e-9:
            return []
        side = "BUY" if monthly_trend > 0 else "SELL"
        return [
            SignalCandidate(
                signal_id=deterministic_id(symbol, "router", "gbp-month-end", side, timestamp),
                setup="GBPUSD_MONTH_END_FLOW",
                side=side,
                score_hint=0.64,
                reason="GBP month-end flow aligned with higher-timeframe trend",
                stop_atr=1.0,
                tp_r=1.9,
                strategy_family="TREND",
                confluence_score=4.0,
                confluence_required=3.9,
                meta={
                    "timeframe": "M15",
                    "atr_field": "m15_atr_14",
                    "policy_window": "MONTH_END_FLOW",
                    "setup_family": "FLOW",
                },
            )
        ]

    def _usdjpy_public_intervention(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        intervention_score = float(row.get("usdjpy_intervention_headline_score", row.get("jpy_intervention_public_score", 0.0)))
        if intervention_score < 0.7:
            return []
        side = "SELL" if float(row.get("m15_ret_1", row.get("m5_ret_1", 0.0))) > 0 else "BUY"
        return [
            SignalCandidate(
                signal_id=deterministic_id(symbol, "router", "usdjpy-intervention", side, timestamp),
                setup="USDJPY_PUBLIC_INTERVENTION",
                side=side,
                score_hint=0.69,
                reason="USDJPY public intervention headline response",
                stop_atr=1.0,
                tp_r=2.0,
                strategy_family="FAKEOUT",
                confluence_score=4.4,
                confluence_required=4.1,
                meta={
                    "timeframe": "M15",
                    "atr_field": "m15_atr_14",
                    "policy_window": "PUBLIC_INTERVENTION",
                    "setup_family": "HEADLINE",
                },
            )
        ]

    def _usdjpy_tokyo_lunch_fade(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if not self._in_utc_window(timestamp, 3.0, 4.0):
            return []
        if bool(regime.details.get("trend_flag", regime.label == "TRENDING")):
            return []
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_reference_limit(symbol):
            return []
        range_pos = float(row.get("m15_range_position_20", 0.5))
        output: list[SignalCandidate] = []
        if range_pos >= 0.72 and int(row.get("m5_pinbar_bear", 0)) == 1:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "usdjpy-lunch-fade", "SELL", timestamp),
                    setup="USDJPY_TOKYO_LUNCH_FADE",
                    side="SELL",
                    score_hint=0.60,
                    reason="USDJPY Tokyo lunch fade from range ceiling",
                    stop_atr=0.85,
                    tp_r=1.4,
                    strategy_family="RANGE",
                    confluence_score=3.5,
                    confluence_required=3.4,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "policy_window": "TOKYO_LUNCH", "setup_family": "RANGE_FADE"},
                )
            )
        if range_pos <= 0.28 and int(row.get("m5_pinbar_bull", 0)) == 1:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "usdjpy-lunch-fade", "BUY", timestamp),
                    setup="USDJPY_TOKYO_LUNCH_FADE",
                    side="BUY",
                    score_hint=0.60,
                    reason="USDJPY Tokyo lunch fade from range floor",
                    stop_atr=0.85,
                    tp_r=1.4,
                    strategy_family="RANGE",
                    confluence_score=3.5,
                    confluence_required=3.4,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "policy_window": "TOKYO_LUNCH", "setup_family": "RANGE_FADE"},
                )
            )
        return output

    def _xau_fix_flow(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if not self._in_utc_window(timestamp, 14.75, 15.25):
            return []
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_reference_limit(symbol):
            return []
        dxy_move = float(row.get("dxy_ret_5", row.get("dxy_ret_1", 0.0)))
        yield_move = float(row.get("us10y_ret_5", row.get("us10y_ret_1", row.get("yield_ret_5", 0.0))))
        if abs(dxy_move) <= 1e-6 or abs(yield_move) <= 1e-6:
            return []
        vol_ratio = self._ratio_or_neutral(row.get("m5_volume_ratio_20", 1.0))
        if vol_ratio < 1.0:
            return []
        output: list[SignalCandidate] = []
        if dxy_move < 0 and yield_move < 0 and int(row.get("m5_bullish", 0)) == 1:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "xau-fix", "flow", "BUY", timestamp),
                    setup="XAUUSD_M15_FIX_FLOW",
                    side="BUY",
                    score_hint=0.68,
                    reason="Gold fix flow aligned with weaker DXY and yields",
                    stop_atr=1.0,
                    tp_r=1.8,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=4.2,
                    confluence_required=4.0,
                    meta={"xau_engine": "M15_STRUCTURED", "timeframe": "M15", "atr_field": "m15_atr_14", "policy_window": "LONDON_FIX", "setup_family": "FIX_FLOW"},
                )
            )
        if dxy_move > 0 and yield_move > 0 and int(row.get("m5_bearish", 0)) == 1:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "xau-fix", "flow", "SELL", timestamp),
                    setup="XAUUSD_M15_FIX_FLOW",
                    side="SELL",
                    score_hint=0.68,
                    reason="Gold fix flow aligned with stronger DXY and yields",
                    stop_atr=1.0,
                    tp_r=1.8,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=4.2,
                    confluence_required=4.0,
                    meta={"xau_engine": "M15_STRUCTURED", "timeframe": "M15", "atr_field": "m15_atr_14", "policy_window": "LONDON_FIX", "setup_family": "FIX_FLOW"},
                )
            )
        return output

    def _forex_range(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        symbol_key = self._normalize_symbol(symbol)
        if symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD", "USDJPY"}:
            return []
        if "RANGE" not in session.allowed_strategies:
            return []
        range_flag = bool(regime.details.get("range_flag", 0.0)) or str(regime.label).upper() == "RANGING"
        if not range_flag:
            return []
        if bool(regime.details.get("trend_flag", regime.label == "TRENDING")):
            return []
        session_name = str(session.session_name).upper()
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        prev_high = float(row.get("m15_rolling_high_prev_20", close + atr))
        prev_low = float(row.get("m15_rolling_low_prev_20", close - atr))
        range_pos = float(row.get("m15_range_position_20", 0.5))
        rsi = float(row.get("m5_rsi_14", 50.0))
        spread_ok = float(row.get("m5_spread", 0.0)) <= self._spread_limit(symbol_key, multiplier=0.90)
        volume_ratio = self._ratio_or_neutral(row.get("m5_volume_ratio_20", 1.0))
        volume_ok = volume_ratio >= 0.92
        body_efficiency = float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)) or 0.55)
        atr_ratio = float(row.get("m15_atr_pct_of_avg", row.get("m5_atr_pct_of_avg", 1.0)) or 1.0)
        lower_wick = float(row.get("m5_lower_wick_ratio", 0.0) or 0.0)
        upper_wick = float(row.get("m5_upper_wick_ratio", 0.0) or 0.0)
        bull_rejection = int(row.get("m5_pinbar_bull", 0)) == 1 or int(row.get("m5_engulf_bull", 0)) == 1
        bear_rejection = int(row.get("m5_pinbar_bear", 0)) == 1 or int(row.get("m5_engulf_bear", 0)) == 1
        soft_bull_edge = (
            range_pos <= 0.14
            and rsi <= 38
            and ((close - prev_low) <= (atr * 0.45))
        )
        soft_bear_edge = (
            range_pos >= 0.86
            and rsi >= 62
            and ((prev_high - close) <= (atr * 0.45))
        )
        output: list[SignalCandidate] = []
        strict_range_pair = symbol_key in {"EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY"}
        major_pair = symbol_key in {"EURUSD", "GBPUSD"}
        audnzd_asia_pair = symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"}
        asia_secondary_jpy_pair = symbol_key in {"EURJPY", "GBPJPY"} and session_name in {"SYDNEY", "TOKYO"}
        if major_pair and session_name in {"SYDNEY", "TOKYO"}:
            return []
        if asia_secondary_jpy_pair and runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label)) in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
            return []
        if strict_range_pair:
            volume_ok = volume_ratio >= (1.14 if major_pair else 1.10 if asia_secondary_jpy_pair else 1.04)
            if not (bull_rejection or bear_rejection):
                return []
            if major_pair and (body_efficiency > 0.54 or atr_ratio > 0.92 or volume_ratio > 1.30):
                return []
            if asia_secondary_jpy_pair and (body_efficiency > 0.58 or atr_ratio > 1.00 or volume_ratio > 1.34):
                return []
            if major_pair and abs(close - (prev_low if range_pos <= 0.50 else prev_high)) > (atr * 0.22):
                return []
            if asia_secondary_jpy_pair and abs(close - (prev_low if range_pos <= 0.50 else prev_high)) > (atr * 0.20):
                return []
            if major_pair:
                if bull_rejection and lower_wick < 0.24:
                    return []
                if bear_rejection and upper_wick < 0.24:
                    return []
            if asia_secondary_jpy_pair:
                if bull_rejection and lower_wick < 0.22:
                    return []
                if bear_rejection and upper_wick < 0.22:
                    return []
        tighter_major_range_session = major_pair and session_name in {"LONDON", "OVERLAP", "NEW_YORK", "OFF"}
        if tighter_major_range_session:
            if volume_ratio > 1.18 or body_efficiency > 0.50 or atr_ratio > 0.88:
                return []
            edge_anchor = prev_low if range_pos <= 0.50 else prev_high
            if abs(close - edge_anchor) > (atr * 0.18):
                return []
            if bull_rejection and lower_wick < 0.30:
                return []
            if bear_rejection and upper_wick < 0.30:
                return []
        if audnzd_asia_pair:
            volume_ok = volume_ratio >= 1.02
            if not (bull_rejection or bear_rejection):
                return []
            if body_efficiency > 0.64:
                return []

        if (
            spread_ok
            and volume_ok
            and body_efficiency <= (0.62 if major_pair else 0.68 if strict_range_pair else 0.64 if audnzd_asia_pair else 0.78)
            and range_pos <= (0.12 if major_pair else 0.16 if strict_range_pair else 0.18 if audnzd_asia_pair else 0.22)
            and rsi <= (36 if major_pair else 38 if strict_range_pair else 40 if audnzd_asia_pair else 42)
            and (
                bull_rejection
                or (
                    not strict_range_pair
                    and not audnzd_asia_pair
                    and soft_bull_edge
                    and session_name in {"SYDNEY", "TOKYO"}
                )
            )
            and (not audnzd_asia_pair or lower_wick >= 0.20)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "forex-range", "BUY", timestamp),
                    setup="FOREX_RANGE_REVERSION",
                    side="BUY",
                    score_hint=0.62 if major_pair and bull_rejection else 0.59 if asia_secondary_jpy_pair and bull_rejection else 0.60 if strict_range_pair and bull_rejection else 0.60 if bull_rejection else 0.56,
                    reason="Range floor rejection with RSI support" if bull_rejection else "Range floor pressure with stretched RSI",
                    stop_atr=0.95 if major_pair else 1.0 if strict_range_pair else 0.9,
                    tp_r=1.7 if major_pair else 1.75 if strict_range_pair else 1.6,
                    strategy_family="RANGE",
                    confluence_score=4.0 if major_pair and bull_rejection else 4.0 if asia_secondary_jpy_pair and bull_rejection else 3.9 if strict_range_pair and bull_rejection else 3.8 if bull_rejection else 3.3,
                    confluence_required=4.1 if major_pair else 4.1 if asia_secondary_jpy_pair else 4.0 if strict_range_pair else 4.0 if bull_rejection else 3.4,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.54 if major_pair else 0.55 if strict_range_pair else 0.52,
                        "approve_small_min_confluence": 3.8 if major_pair else 3.9 if strict_range_pair else 3.4,
                    },
                )
            )
        if (
            spread_ok
            and volume_ok
            and body_efficiency <= (0.62 if major_pair else 0.68 if strict_range_pair else 0.64 if audnzd_asia_pair else 0.78)
            and range_pos >= (0.88 if major_pair else 0.84 if strict_range_pair else 0.82 if audnzd_asia_pair else 0.78)
            and rsi >= (64 if major_pair else 62 if strict_range_pair else 60 if audnzd_asia_pair else 58)
            and (
                bear_rejection
                or (
                    not strict_range_pair
                    and not audnzd_asia_pair
                    and soft_bear_edge
                    and session_name in {"SYDNEY", "TOKYO"}
                )
            )
            and (not audnzd_asia_pair or upper_wick >= 0.20)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "forex-range", "SELL", timestamp),
                    setup="FOREX_RANGE_REVERSION",
                    side="SELL",
                    score_hint=0.62 if major_pair and bear_rejection else 0.59 if asia_secondary_jpy_pair and bear_rejection else 0.60 if strict_range_pair and bear_rejection else 0.60 if bear_rejection else 0.56,
                    reason="Range cap rejection with RSI support" if bear_rejection else "Range cap pressure with stretched RSI",
                    stop_atr=0.95 if major_pair else 1.0 if strict_range_pair else 0.9,
                    tp_r=1.7 if major_pair else 1.75 if strict_range_pair else 1.6,
                    strategy_family="RANGE",
                    confluence_score=4.0 if major_pair and bear_rejection else 4.0 if asia_secondary_jpy_pair and bear_rejection else 3.9 if strict_range_pair and bear_rejection else 3.8 if bear_rejection else 3.3,
                    confluence_required=4.1 if major_pair else 4.1 if asia_secondary_jpy_pair else 4.0 if strict_range_pair else 4.0 if bear_rejection else 3.4,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.54 if major_pair else 0.55 if strict_range_pair else 0.52,
                        "approve_small_min_confluence": 3.8 if major_pair else 3.9 if strict_range_pair else 3.4,
                    },
                )
            )
        return output

    def _forex_breakout_retest(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        symbol_key = self._normalize_symbol(symbol)
        session_name = str(session.session_name).upper()
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        raw_regime_label = str(getattr(regime, "state_label", regime.label) or regime.label).strip().upper()
        sensitivity = self._symbol_sensitivity(symbol, self.forex_sensitivity)
        drift_trend_ready = self._drift_continuation_ready(
            symbol=symbol_key,
            row=row,
            session_name=session_name,
            raw_regime_label=raw_regime_label,
        )
        if "TREND" not in session.allowed_strategies:
            return []
        if symbol_key in {"EURUSD", "GBPUSD"} and session_name in {"SYDNEY", "TOKYO"}:
            return []
        if symbol_key in {"AUDJPY", "NZDJPY"} and session_name in {"SYDNEY", "TOKYO"} and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"} and not drift_trend_ready:
            return []
        if symbol_key == "USDJPY" and session_name in {"SYDNEY", "TOKYO"} and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"} and not drift_trend_ready:
            return []
        if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"} and regime_state not in {"TRENDING", "BREAKOUT_EXPANSION"} and not drift_trend_ready:
            return []
        spread_ok = float(row.get("m5_spread", 0.0)) <= self._spread_reference_limit(symbol_key)
        if not spread_ok:
            return []
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        high = float(row.get("m15_high", close))
        low = float(row.get("m15_low", close))
        prev_high = float(row.get("m15_rolling_high_prev_20", close + atr))
        prev_low = float(row.get("m15_rolling_low_prev_20", close - atr))
        vol_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        bullish = self._bullish_flag(row)
        bearish = self._bearish_flag(row)
        upper_wick = float(row.get("m5_upper_wick_ratio", 0.0) or 0.0)
        lower_wick = float(row.get("m5_lower_wick_ratio", 0.0) or 0.0)
        ret_3 = abs(float(row.get("m15_ret_3", row.get("m5_ret_3", 0.0)) or 0.0))
        body_efficiency = float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)) or 0.55)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        trend_flag = bool(regime.details.get("trend_flag", regime.label == "TRENDING")) or drift_trend_ready
        if not trend_flag:
            return []

        retest_tolerance = atr * (0.30 + (0.12 * max(0.0, sensitivity - 1.0)))
        vol_floor = 0.95 - (0.15 * max(0.0, sensitivity - 1.0))
        if symbol_key in {"AUDJPY", "NZDJPY"} and session_name == "TOKYO":
            vol_floor = max(vol_floor, 0.96 if symbol_key == "AUDJPY" else 0.98)
        if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"}:
            vol_floor = max(vol_floor, 1.04)
        if symbol_key in {"AUDJPY", "NZDJPY"} and session_name in {"SYDNEY", "TOKYO"}:
            vol_floor = max(vol_floor, 0.96 if symbol_key == "AUDJPY" else 0.98)
            retest_tolerance = min(retest_tolerance, atr * (0.26 if symbol_key == "AUDJPY" else 0.22))
        if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"}:
            vol_floor = max(vol_floor, 1.04)
            retest_tolerance = min(retest_tolerance, atr * 0.20)
        if body_efficiency < (0.48 if drift_trend_ready else 0.58 if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"} else 0.54 if symbol_key in {"AUDJPY", "NZDJPY"} and session_name == "TOKYO" else 0.52):
            return []
        if symbol_key in {"AUDJPY", "NZDJPY"} and session_name in {"SYDNEY", "TOKYO"}:
            min_body = 0.54 if symbol_key == "AUDJPY" else 0.56
            if body_efficiency < min_body or ret_3 > (atr * (0.32 if symbol_key == "AUDJPY" else 0.28)):
                return []
        if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"}:
            if body_efficiency < 0.58 or ret_3 > (atr * 0.28):
                return []
        output: list[SignalCandidate] = []
        drift_meta = {
            "session_drift_lane": True,
            "structure_cleanliness_floor": 0.56,
            "entry_timing_floor": 0.58,
            "setup_family": "DRIFT_CONTINUATION",
        } if drift_trend_ready else {}
        breakout_up = close > prev_high and vol_ratio >= vol_floor and bullish and range_position >= (0.56 if symbol_key == "AUDNZD" else 0.52)
        breakout_up = breakout_up and (
            (close - prev_high) <= (
                atr * (
                    0.24 if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"}
                    else 0.32 if symbol_key == "AUDJPY" and session_name in {"SYDNEY", "TOKYO"}
                    else 0.28 if symbol_key == "NZDJPY" and session_name in {"SYDNEY", "TOKYO"}
                    else 0.42
                )
            )
        )
        retest_up = (low <= (prev_high + retest_tolerance)) and close >= prev_high
        if symbol_key in {"AUDJPY", "NZDJPY"} and session_name in {"SYDNEY", "TOKYO"} and upper_wick > (0.20 if symbol_key == "AUDJPY" else 0.18):
            breakout_up = False
        if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"} and upper_wick > 0.18:
            breakout_up = False
        if breakout_up and retest_up:
            asia_breakout = symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD"} and session_name in {"SYDNEY", "TOKYO"}
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "forex-breakout", "BUY", timestamp),
                    setup="FOREX_BREAKOUT_RETEST",
                    side="BUY",
                    score_hint=0.62 if asia_breakout else 0.61,
                    reason="M15 breakout retest with trend continuation",
                    stop_atr=1.0,
                    tp_r=1.95 if asia_breakout else 1.9,
                    strategy_family="TREND",
                    confluence_score=3.95 if asia_breakout else 3.9,
                    confluence_required=3.75 if asia_breakout else 4.0,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.50 if asia_breakout else 0.52,
                        "approve_small_min_confluence": 3.2 if asia_breakout else 3.4,
                        **drift_meta,
                    },
                )
            )
        breakout_down = close < prev_low and vol_ratio >= vol_floor and bearish and range_position <= (0.44 if symbol_key == "AUDNZD" else 0.48)
        breakout_down = breakout_down and (
            (prev_low - close) <= (
                atr * (
                    0.24 if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"}
                    else 0.32 if symbol_key == "AUDJPY" and session_name in {"SYDNEY", "TOKYO"}
                    else 0.28 if symbol_key == "NZDJPY" and session_name in {"SYDNEY", "TOKYO"}
                    else 0.42
                )
            )
        )
        retest_down = (high >= (prev_low - retest_tolerance)) and close <= prev_low
        if symbol_key in {"AUDJPY", "NZDJPY"} and session_name in {"SYDNEY", "TOKYO"} and lower_wick > (0.20 if symbol_key == "AUDJPY" else 0.18):
            breakout_down = False
        if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"} and lower_wick > 0.18:
            breakout_down = False
        if breakout_down and retest_down:
            asia_breakout = symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD"} and session_name in {"SYDNEY", "TOKYO"}
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "forex-breakout", "SELL", timestamp),
                    setup="FOREX_BREAKOUT_RETEST",
                    side="SELL",
                    score_hint=0.62 if asia_breakout else 0.61,
                    reason="M15 downside breakout retest with trend continuation",
                    stop_atr=1.0,
                    tp_r=1.95 if asia_breakout else 1.9,
                    strategy_family="TREND",
                    confluence_score=3.95 if asia_breakout else 3.9,
                    confluence_required=3.75 if asia_breakout else 4.0,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.50 if asia_breakout else 0.52,
                        "approve_small_min_confluence": 3.2 if asia_breakout else 3.4,
                        **drift_meta,
                    },
                )
            )
        return output

    def _xau_fakeout(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        session_name = str(session.session_name).upper()
        if "FAKEOUT" not in session.allowed_strategies:
            return []
        if session_name not in {"LONDON", "OVERLAP", "NEW_YORK"}:
            return []
        atr = max(float(row.get("m5_atr_14", 0.0)), 1e-6)
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_reference_limit(symbol):
            return []
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        volume_ratio = self._ratio_or_neutral(row.get("m5_volume_ratio_20", row.get("m15_volume_ratio_20", 1.0)))
        body_efficiency = float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)) or 0.55)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        upper_wick = float(row.get("m5_upper_wick_ratio", 0.0) or 0.0)
        lower_wick = float(row.get("m5_lower_wick_ratio", 0.0) or 0.0)
        if volume_ratio < 0.94 or body_efficiency < 0.54:
            return []

        close = float(row.get("m5_close", 0.0))
        high = float(row.get("m5_high", close))
        low = float(row.get("m5_low", close))
        prev_high = float(row.get("m5_rolling_high_prev_20", close + atr))
        prev_low = float(row.get("m5_rolling_low_prev_20", close - atr))
        fakeout_up = high > prev_high and close < prev_high
        fakeout_down = low < prev_low and close > prev_low
        clean_breakout_up = close > prev_high and float(row.get("m5_ret_1", 0.0)) > 0 and int(row.get("m5_bullish", 0)) == 1
        clean_breakout_down = close < prev_low and float(row.get("m5_ret_1", 0.0)) < 0 and int(row.get("m5_bearish", 0)) == 1
        fakeout_up_extension = max(0.0, high - prev_high) / atr
        fakeout_down_extension = max(0.0, prev_low - low) / atr
        breakout_up_extension = max(0.0, close - prev_high) / atr
        breakout_down_extension = max(0.0, prev_low - close) / atr
        trend_flag = bool(regime.details.get("trend_flag", regime.label == "TRENDING"))
        output: list[SignalCandidate] = []

        if (
            fakeout_up
            and regime_state in {"RANGING", "MEAN_REVERSION"}
            and volume_ratio >= 0.98
            and body_efficiency >= 0.56
            and upper_wick >= 0.20
            and range_position >= 0.70
            and 0.08 <= fakeout_up_extension <= 1.10
            and (int(row.get("m5_pinbar_bear", 0)) == 1 or int(row.get("m5_engulf_bear", 0)) == 1)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "xau-fakeout", "SELL", timestamp),
                    setup="XAU_FAKEOUT_FADE",
                    side="SELL",
                    score_hint=0.64,
                    reason="Gold fakeout above structure with rejection",
                    stop_atr=1.8,
                    tp_r=2.2,
                    strategy_family="FAKEOUT",
                    confluence_score=5.0,
                    confluence_required=5.0,
                )
            )
        if (
            fakeout_down
            and regime_state in {"RANGING", "MEAN_REVERSION"}
            and volume_ratio >= 0.98
            and body_efficiency >= 0.56
            and lower_wick >= 0.20
            and range_position <= 0.30
            and 0.08 <= fakeout_down_extension <= 1.10
            and (int(row.get("m5_pinbar_bull", 0)) == 1 or int(row.get("m5_engulf_bull", 0)) == 1)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "xau-fakeout", "BUY", timestamp),
                    setup="XAU_FAKEOUT_FADE",
                    side="BUY",
                    score_hint=0.64,
                    reason="Gold fakeout below structure with rejection",
                    stop_atr=1.8,
                    tp_r=2.2,
                    strategy_family="FAKEOUT",
                    confluence_score=5.0,
                    confluence_required=5.0,
                )
            )
        if (
            clean_breakout_up
            and trend_flag
            and regime_state in {"TRENDING", "BREAKOUT_EXPANSION"}
            and volume_ratio >= 1.08
            and body_efficiency >= 0.62
            and range_position >= 0.56
            and 0.10 <= breakout_up_extension <= 1.25
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "xau-breakout", "BUY", timestamp),
                    setup="XAU_BREAKOUT_RETEST",
                    side="BUY",
                    score_hint=0.64,
                    reason="Gold clean breakout continuation",
                    stop_atr=1.6,
                    tp_r=2.0,
                    strategy_family="TREND",
                    confluence_score=4.0,
                    confluence_required=5.0,
                )
            )
        if (
            clean_breakout_down
            and trend_flag
            and regime_state in {"TRENDING", "BREAKOUT_EXPANSION"}
            and volume_ratio >= 1.08
            and body_efficiency >= 0.62
            and range_position <= 0.44
            and 0.10 <= breakout_down_extension <= 1.25
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "xau-breakout", "SELL", timestamp),
                    setup="XAU_BREAKOUT_RETEST",
                    side="SELL",
                    score_hint=0.64,
                    reason="Gold clean breakout continuation",
                    stop_atr=1.6,
                    tp_r=2.0,
                    strategy_family="TREND",
                    confluence_score=4.0,
                    confluence_required=5.0,
                )
            )
        return output

    def _xau_m1_micro_scalper(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if not self.xau_m1_enabled:
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}:
            return []
        prime_session = session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
        spread_points = float(row.get("m5_spread", 0.0))
        spread_cap = self.max_spread_points * (0.92 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.78)
        if spread_points > spread_cap:
            return []
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        compression_state = str(getattr(regime, "details", {}).get("compression_proxy_state", row.get("compression_proxy_state", "NEUTRAL")) or "NEUTRAL").upper()
        compression_expansion_score = clamp(
            self._ratio_or_neutral(getattr(regime, "details", {}).get("compression_expansion_score", row.get("compression_expansion_score", 0.0)), neutral=0.0),
            0.0,
            1.0,
        )
        alignment_score = clamp(
            self._ratio_or_neutral(getattr(regime, "details", {}).get("multi_tf_alignment_score", row.get("multi_tf_alignment_score", 0.5)), neutral=0.5),
            0.0,
            1.0,
        )
        fractal_score = clamp(
            self._ratio_or_neutral(getattr(regime, "details", {}).get("fractal_persistence_score", row.get("fractal_persistence_score", 0.5)), neutral=0.5),
            0.0,
            1.0,
        )
        seasonality_score = clamp(
            self._ratio_or_neutral(getattr(regime, "details", {}).get("seasonality_edge_score", row.get("seasonality_edge_score", 0.5)), neutral=0.5),
            0.0,
            1.0,
        )
        instability_score = clamp(
            self._ratio_or_neutral(getattr(regime, "details", {}).get("market_instability_score", row.get("market_instability_score", row.get("feature_drift_score", 0.0))), neutral=0.0),
            0.0,
            1.0,
        )
        structure_ready = bool(
            alignment_score >= (0.58 if session_name in {"SYDNEY", "TOKYO"} else 0.46 if prime_session else 0.50)
            and fractal_score >= (0.54 if session_name in {"SYDNEY", "TOKYO"} else 0.44 if prime_session else 0.48)
            and seasonality_score >= (0.40 if session_name in {"SYDNEY", "TOKYO"} else 0.30 if prime_session else 0.34)
            and instability_score <= (0.44 if session_name in {"SYDNEY", "TOKYO"} else 0.58 if prime_session else 0.55)
        )
        expansion_ready = compression_state == "EXPANSION_READY" or compression_expansion_score >= 0.36
        if regime_state not in {"BREAKOUT_EXPANSION", "TRENDING"} and not (structure_ready and expansion_ready):
            return []

        m1_atr = max(float(row.get("m1_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        m1_body = float(row.get("m1_body", 0.0))
        m1_impulse = abs(m1_body) / m1_atr
        m1_volume_ratio = self._ratio_or_neutral(row.get("m1_volume_ratio_20", 1.0))
        m1_body_efficiency = float(row.get("m1_body_efficiency", row.get("m1_candle_efficiency", 0.55)) or 0.55)
        m15_volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        if m1_impulse < self.xau_m1_min_impulse_ratio or m1_volume_ratio < self.xau_m1_min_volume_ratio:
            return []
        if (
            m1_body_efficiency < (0.28 if session_name in {"SYDNEY", "TOKYO"} else 0.30 if prime_session else 0.32)
            or m1_impulse > (1.75 if session_name in {"SYDNEY", "TOKYO"} else 1.85 if prime_session else 1.75)
            or m15_volume_ratio < (0.84 if session_name in {"SYDNEY", "TOKYO"} else 0.84 if prime_session else 0.88)
        ):
            return []

        m1_momentum = float(row.get("m1_momentum_3", 0.0))
        m5_slope = float(row.get("m5_macd_hist_slope", 0.0))
        bullish = int(row.get("m1_bullish", 0)) == 1
        bearish = int(row.get("m1_bearish", 0)) == 1
        trend_flag = bool(regime.details.get("trend_flag", regime.label == "TRENDING"))
        if not trend_flag and not structure_ready:
            return []
        base_confluence = (
            2.0
            + min(1.3, max(0.0, m1_impulse - self.xau_m1_min_impulse_ratio) * 2.0)
            + min(0.8, max(0.0, m1_volume_ratio - self.xau_m1_min_volume_ratio) * 1.5)
            + (0.4 if trend_flag else 0.2)
            + (0.25 if structure_ready else 0.0)
            + (0.25 if expansion_ready else 0.0)
            + (0.20 if spread_points <= (self.max_spread_points * 0.8) else 0.0)
            + (0.10 if prime_session and structure_ready else 0.0)
        )
        confluence = clamp(base_confluence, 0.0, 5.0)
        if confluence < self.xau_m1_confluence_floor:
            return []

        output: list[SignalCandidate] = []
        score_hint = clamp(0.60 + (0.06 * (confluence - self.xau_m1_confluence_floor)) + (0.04 if structure_ready else 0.0), 0.56, 0.84)
        if bullish and m1_momentum > 0 and m5_slope >= -0.004:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "xau-m1", "micro", "BUY", timestamp),
                    setup="XAUUSD_M1_MICRO_SCALPER",
                    side="BUY",
                    score_hint=score_hint,
                    reason="M1 momentum spike with volume expansion",
                    stop_atr=0.80,
                    tp_r=1.45,
                    entry_kind="SCALP",
                    strategy_family="SCALP",
                    confluence_score=confluence,
                    confluence_required=self.xau_m1_confluence_floor,
                    meta={
                        "xau_engine": "M1_MICRO_SCALPER",
                        "timeframe": "M1",
                        "atr_field": "m1_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.48,
                        "approve_small_min_confluence": 3.0,
                        "compression_proxy_state": compression_state,
                        "compression_expansion_score": compression_expansion_score,
                        "multi_tf_alignment_score": alignment_score,
                        "seasonality_edge_score": seasonality_score,
                        "fractal_persistence_score": fractal_score,
                        "market_instability_score": instability_score,
                    },
                )
            )
        if bearish and m1_momentum < 0 and m5_slope <= 0.004:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "xau-m1", "micro", "SELL", timestamp),
                    setup="XAUUSD_M1_MICRO_SCALPER",
                    side="SELL",
                    score_hint=score_hint,
                    reason="M1 downside momentum spike with volume expansion",
                    stop_atr=0.80,
                    tp_r=1.45,
                    entry_kind="SCALP",
                    strategy_family="SCALP",
                    confluence_score=confluence,
                    confluence_required=self.xau_m1_confluence_floor,
                    meta={
                        "xau_engine": "M1_MICRO_SCALPER",
                        "timeframe": "M1",
                        "atr_field": "m1_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.48,
                        "approve_small_min_confluence": 3.0,
                        "compression_proxy_state": compression_state,
                        "compression_expansion_score": compression_expansion_score,
                        "multi_tf_alignment_score": alignment_score,
                        "seasonality_edge_score": seasonality_score,
                        "fractal_persistence_score": fractal_score,
                        "market_instability_score": instability_score,
                    },
                )
            )
        return output

    def _xau_m15_structured(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if not self.xau_m15_enabled:
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}:
            return []
        prime_session = session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
        spread_points = float(row.get("m5_spread", 0.0))
        spread_cap = self.max_spread_points * (0.88 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.78)
        if spread_points > spread_cap:
            return []

        m15_atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        high = float(row.get("m15_high", close))
        low = float(row.get("m15_low", close))
        prev_high = float(row.get("m15_rolling_high_prev_20", close + m15_atr))
        prev_low = float(row.get("m15_rolling_low_prev_20", close - m15_atr))
        bullish = int(row.get("m15_bullish", row.get("m5_bullish", 0))) == 1
        bearish = int(row.get("m15_bearish", row.get("m5_bearish", 0))) == 1
        vol_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        body_efficiency = float(row.get("m15_body_efficiency", row.get("m15_candle_efficiency", row.get("m5_body_efficiency", 0.55))) or 0.55)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        lower_wick = float(row.get("m15_lower_wick_ratio", row.get("m5_lower_wick_ratio", 0.0)) or 0.0)
        upper_wick = float(row.get("m15_upper_wick_ratio", row.get("m5_upper_wick_ratio", 0.0)) or 0.0)
        breakout_threshold = self.xau_m15_breakout_atr_threshold * m15_atr
        trend_flag = bool(regime.details.get("trend_flag", regime.label == "TRENDING"))
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        compression_state = str(getattr(regime, "details", {}).get("compression_proxy_state", row.get("compression_proxy_state", "NEUTRAL")) or "NEUTRAL").upper()
        compression_expansion_score = clamp(
            self._ratio_or_neutral(getattr(regime, "details", {}).get("compression_expansion_score", row.get("compression_expansion_score", 0.0)), neutral=0.0),
            0.0,
            1.0,
        )
        alignment_score = clamp(
            self._ratio_or_neutral(getattr(regime, "details", {}).get("multi_tf_alignment_score", row.get("multi_tf_alignment_score", 0.5)), neutral=0.5),
            0.0,
            1.0,
        )
        fractal_score = clamp(
            self._ratio_or_neutral(getattr(regime, "details", {}).get("fractal_persistence_score", row.get("fractal_persistence_score", 0.5)), neutral=0.5),
            0.0,
            1.0,
        )
        seasonality_score = clamp(
            self._ratio_or_neutral(getattr(regime, "details", {}).get("seasonality_edge_score", row.get("seasonality_edge_score", 0.5)), neutral=0.5),
            0.0,
            1.0,
        )
        instability_score = clamp(
            self._ratio_or_neutral(getattr(regime, "details", {}).get("market_instability_score", row.get("market_instability_score", row.get("feature_drift_score", 0.0))), neutral=0.0),
            0.0,
            1.0,
        )
        structure_ready = bool(
            alignment_score >= (0.60 if session_name in {"SYDNEY", "TOKYO"} else 0.48 if prime_session else 0.52)
            and fractal_score >= (0.56 if session_name in {"SYDNEY", "TOKYO"} else 0.46 if prime_session else 0.50)
            and seasonality_score >= (0.40 if session_name in {"SYDNEY", "TOKYO"} else 0.30 if prime_session else 0.34)
            and instability_score <= (0.42 if session_name in {"SYDNEY", "TOKYO"} else 0.56 if prime_session else 0.52)
        )
        expansion_ready = compression_state == "EXPANSION_READY" or compression_expansion_score >= 0.36

        output: list[SignalCandidate] = []
        breakout_up = close > prev_high and (close - prev_high) >= breakout_threshold and bullish
        retest_up = low <= (prev_high + breakout_threshold)
        breakout_up_extension = max(0.0, close - prev_high) / m15_atr
        breakout_down_extension = max(0.0, prev_low - close) / m15_atr
        if (
            breakout_up
            and retest_up
            and (trend_flag or structure_ready)
            and (regime_state in {"TRENDING", "BREAKOUT_EXPANSION"} or (structure_ready and expansion_ready))
            and vol_ratio >= (0.92 if session_name in {"SYDNEY", "TOKYO"} else 0.94 if prime_session else 0.98)
            and body_efficiency >= (0.28 if session_name in {"SYDNEY", "TOKYO"} else 0.28 if prime_session else 0.32)
            and range_position >= 0.58
            and (0.06 if prime_session else 0.08) <= breakout_up_extension <= 1.60
        ):
            confluence = clamp(3.8 + min(1.2, (close - prev_high) / max(m15_atr, 1e-6)) + (0.4 if vol_ratio >= 1.0 else 0.0) + (0.20 if structure_ready else 0.0), 0.0, 5.0)
            if confluence >= self.xau_m15_confluence_floor:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "xau-m15", "breakout", "BUY", timestamp),
                        setup="XAUUSD_M15_STRUCTURED_BREAKOUT",
                        side="BUY",
                        score_hint=0.66,
                        reason="M15 breakout and retest continuation",
                        stop_atr=1.05,
                        tp_r=2.4,
                        entry_kind="DAYTRADE",
                        strategy_family="TREND",
                        confluence_score=confluence,
                        confluence_required=self.xau_m15_confluence_floor,
                        meta={
                            "xau_engine": "M15_STRUCTURED",
                            "timeframe": "M15",
                            "atr_field": "m15_atr_14",
                            "compression_proxy_state": compression_state,
                            "compression_expansion_score": compression_expansion_score,
                            "multi_tf_alignment_score": alignment_score,
                            "seasonality_edge_score": seasonality_score,
                            "fractal_persistence_score": fractal_score,
                            "market_instability_score": instability_score,
                        },
                    )
                )
        breakout_down = close < prev_low and (prev_low - close) >= breakout_threshold and bearish
        retest_down = high >= (prev_low - breakout_threshold)
        if (
            breakout_down
            and retest_down
            and (trend_flag or structure_ready)
            and (regime_state in {"TRENDING", "BREAKOUT_EXPANSION"} or (structure_ready and expansion_ready))
            and vol_ratio >= (0.92 if session_name in {"SYDNEY", "TOKYO"} else 0.94 if prime_session else 0.98)
            and body_efficiency >= (0.28 if session_name in {"SYDNEY", "TOKYO"} else 0.28 if prime_session else 0.32)
            and range_position <= 0.42
            and (0.06 if prime_session else 0.08) <= breakout_down_extension <= 1.60
        ):
            confluence = clamp(3.8 + min(1.2, (prev_low - close) / max(m15_atr, 1e-6)) + (0.4 if vol_ratio >= 1.0 else 0.0) + (0.20 if structure_ready else 0.0), 0.0, 5.0)
            if confluence >= self.xau_m15_confluence_floor:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "xau-m15", "breakout", "SELL", timestamp),
                        setup="XAUUSD_M15_STRUCTURED_BREAKOUT",
                        side="SELL",
                        score_hint=0.66,
                        reason="M15 downside breakout and retest continuation",
                        stop_atr=1.05,
                        tp_r=2.4,
                        entry_kind="DAYTRADE",
                        strategy_family="TREND",
                        confluence_score=confluence,
                        confluence_required=self.xau_m15_confluence_floor,
                        meta={
                            "xau_engine": "M15_STRUCTURED",
                            "timeframe": "M15",
                            "atr_field": "m15_atr_14",
                            "compression_proxy_state": compression_state,
                            "compression_expansion_score": compression_expansion_score,
                            "multi_tf_alignment_score": alignment_score,
                            "seasonality_edge_score": seasonality_score,
                            "fractal_persistence_score": fractal_score,
                            "market_instability_score": instability_score,
                        },
                    )
                )

        sweep_buy = low < prev_low and close > prev_low and (int(row.get("m15_pinbar_bull", 0)) == 1 or int(row.get("m15_engulf_bull", 0)) == 1)
        sweep_sell = high > prev_high and close < prev_high and (int(row.get("m15_pinbar_bear", 0)) == 1 or int(row.get("m15_engulf_bear", 0)) == 1)
        if (
            sweep_buy
            and regime_state in {"RANGING", "MEAN_REVERSION", "BREAKOUT_EXPANSION"}
            and vol_ratio >= (0.88 if prime_session else 0.92)
            and body_efficiency >= (0.28 if prime_session else 0.30)
            and lower_wick >= (0.14 if prime_session else 0.16)
            and range_position <= 0.34
        ):
            confluence = clamp(4.0 + (0.5 if vol_ratio >= 1.0 else 0.0) + (0.15 if structure_ready else 0.0), 0.0, 5.0)
            if confluence >= self.xau_m15_confluence_floor:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "xau-m15", "sweep", "BUY", timestamp),
                        setup="XAUUSD_M15_STRUCTURED_SWEEP_RETEST",
                        side="BUY",
                        score_hint=0.66,
                        reason="M15 liquidity sweep and reclaim long",
                        stop_atr=1.0,
                        tp_r=2.1,
                        entry_kind="DAYTRADE",
                        strategy_family="FAKEOUT",
                        confluence_score=confluence,
                        confluence_required=self.xau_m15_confluence_floor,
                        meta={"xau_engine": "M15_STRUCTURED", "timeframe": "M15", "atr_field": "m15_atr_14"},
                    )
                )
        if (
            sweep_sell
            and regime_state in {"RANGING", "MEAN_REVERSION", "BREAKOUT_EXPANSION"}
            and vol_ratio >= (0.88 if prime_session else 0.92)
            and body_efficiency >= (0.28 if prime_session else 0.30)
            and upper_wick >= (0.14 if prime_session else 0.16)
            and range_position >= 0.66
        ):
            confluence = clamp(4.0 + (0.5 if vol_ratio >= 1.0 else 0.0) + (0.15 if structure_ready else 0.0), 0.0, 5.0)
            if confluence >= self.xau_m15_confluence_floor:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "xau-m15", "sweep", "SELL", timestamp),
                        setup="XAUUSD_M15_STRUCTURED_SWEEP_RETEST",
                        side="SELL",
                        score_hint=0.66,
                        reason="M15 liquidity sweep and rejection short",
                        stop_atr=1.0,
                        tp_r=2.1,
                        entry_kind="DAYTRADE",
                        strategy_family="FAKEOUT",
                        confluence_score=confluence,
                        confluence_required=self.xau_m15_confluence_floor,
                        meta={"xau_engine": "M15_STRUCTURED", "timeframe": "M15", "atr_field": "m15_atr_14"},
                    )
                )
        if not output and (trend_flag or structure_ready) and (regime_state in {"TRENDING", "BREAKOUT_EXPANSION"} or (structure_ready and expansion_ready)) and vol_ratio >= 0.86:
            m15_ema20 = float(row.get("m15_ema_20", close))
            m15_ema50 = float(row.get("m15_ema_50", row.get("m5_ema_50", close)))
            m15_range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)))
            pullback_ratio = abs(close - m15_ema20) / max(m15_atr, 1e-6)
            near_value = abs(close - m15_ema20) <= (m15_atr * (0.52 if prime_session else 0.48))
            pullback_ok = 0.08 <= pullback_ratio <= (0.60 if prime_session else 0.54)
            if near_value and pullback_ok and body_efficiency >= (0.30 if prime_session else 0.34) and m15_ema20 >= m15_ema50 and m15_range_position <= 0.62 and bullish:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "xau-m15", "pullback", "BUY", timestamp),
                        setup="XAUUSD_M15_STRUCTURED_PULLBACK",
                        side="BUY",
                        score_hint=0.60,
                        reason="M15 gold pullback continuation with executable structure",
                        stop_atr=0.95,
                        tp_r=2.0,
                        entry_kind="DAYTRADE",
                        strategy_family="TREND",
                        confluence_score=max(3.4, self.xau_m15_confluence_floor - 0.2),
                        confluence_required=max(3.4, self.xau_m15_confluence_floor - 0.4),
                        meta={"xau_engine": "M15_STRUCTURED", "timeframe": "M15", "atr_field": "m15_atr_14"},
                    )
                )
            if near_value and pullback_ok and body_efficiency >= (0.30 if prime_session else 0.34) and m15_ema20 <= m15_ema50 and m15_range_position >= 0.38 and bearish:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "xau-m15", "pullback", "SELL", timestamp),
                        setup="XAUUSD_M15_STRUCTURED_PULLBACK",
                        side="SELL",
                        score_hint=0.60,
                        reason="M15 gold downside pullback continuation with executable structure",
                        stop_atr=0.95,
                        tp_r=2.0,
                        entry_kind="DAYTRADE",
                        strategy_family="TREND",
                        confluence_score=max(3.4, self.xau_m15_confluence_floor - 0.2),
                        confluence_required=max(3.4, self.xau_m15_confluence_floor - 0.4),
                        meta={"xau_engine": "M15_STRUCTURED", "timeframe": "M15", "atr_field": "m15_atr_14"},
                    )
                )
        return output

    def _nas_session_scalper(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        symbol_sensitivity = self._symbol_sensitivity(symbol, self.nas_sensitivity)
        session_name = str(session.session_name).upper()
        tokyo_session = session_name == "TOKYO"
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        spread = float(row.get("m5_spread", 0.0))
        spread_limit = self._spread_limit("NAS100", multiplier=(1.15 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.95))
        if spread > spread_limit:
            return []
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        prev_high = float(row.get("m15_rolling_high_prev_20", close + atr))
        prev_low = float(row.get("m15_rolling_low_prev_20", close - atr))
        ema20 = float(row.get("m5_ema_20", close))
        ema50 = float(row.get("m5_ema_50", close))
        vol_ratio = self._ratio_or_neutral(row.get("m5_volume_ratio_20", 1.0))
        sensitivity = max(0.8, symbol_sensitivity)
        vol_floor = 0.95 - (0.12 * max(0.0, sensitivity - 1.0))
        trend_flag = bool(regime.details.get("trend_flag", regime.label == "TRENDING"))
        output: list[SignalCandidate] = []
        if trend_flag and close > prev_high and vol_ratio >= vol_floor and ema20 >= ema50 and int(row.get("m5_bullish", 0)) == 1:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "nas-orb", "BUY", timestamp),
                    setup="NAS_SESSION_SCALPER_ORB",
                    side="BUY",
                    score_hint=0.62,
                    reason="NAS session breakout continuation",
                    stop_atr=1.1,
                    tp_r=1.8,
                    strategy_family="TREND",
                    confluence_score=3.8,
                    confluence_required=3.8,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14"},
                )
            )
        if trend_flag and close < prev_low and vol_ratio >= vol_floor and ema20 <= ema50 and int(row.get("m5_bearish", 0)) == 1:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "nas-orb", "SELL", timestamp),
                    setup="NAS_SESSION_SCALPER_ORB",
                    side="SELL",
                    score_hint=0.62,
                    reason="NAS downside session breakout continuation",
                    stop_atr=1.1,
                    tp_r=1.8,
                    strategy_family="TREND",
                    confluence_score=3.8,
                    confluence_required=3.8,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14"},
                )
            )
        range_pos = float(row.get("m15_range_position_20", 0.5))
        lower_wick = float(row.get("m5_lower_wick_ratio", 0.0) or 0.0)
        upper_wick = float(row.get("m5_upper_wick_ratio", 0.0) or 0.0)
        bull_reclaim = (
            int(row.get("m5_pinbar_bull", 0)) == 1
            or int(row.get("m5_engulf_bull", 0)) == 1
            or (tokyo_session and int(row.get("m5_bullish", 0)) == 1 and close > prev_low and lower_wick >= 0.14)
        )
        bear_reclaim = (
            int(row.get("m5_pinbar_bear", 0)) == 1
            or int(row.get("m5_engulf_bear", 0)) == 1
            or (tokyo_session and int(row.get("m5_bearish", 0)) == 1 and close < prev_high and upper_wick >= 0.14)
        )
        if tokyo_session and regime_state in {"TRENDING", "MEAN_REVERSION"}:
            mr_vol_floor = max(vol_floor, 0.82)
            lower_range_floor = 0.34
            upper_range_floor = 0.66
        elif tokyo_session:
            mr_vol_floor = max(vol_floor, 0.90)
            lower_range_floor = 0.30
            upper_range_floor = 0.70
        else:
            mr_vol_floor = vol_floor
            lower_range_floor = 0.25
            upper_range_floor = 0.75
        if (not trend_flag or tokyo_session) and vol_ratio >= mr_vol_floor:
            if range_pos <= lower_range_floor and bull_reclaim:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "nas-vwap-mr", "BUY", timestamp),
                        setup="NAS_SESSION_SCALPER_VWAP_MR",
                        side="BUY",
                        score_hint=0.60 if tokyo_session else 0.59,
                        reason="NAS mean reversion at range floor",
                        stop_atr=0.9,
                        tp_r=1.5,
                        strategy_family="RANGE",
                        confluence_score=3.6 if tokyo_session else 3.5,
                        confluence_required=3.5,
                        meta={"timeframe": "M15", "atr_field": "m15_atr_14"},
                    )
                )
            if range_pos >= upper_range_floor and bear_reclaim:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "nas-vwap-mr", "SELL", timestamp),
                        setup="NAS_SESSION_SCALPER_VWAP_MR",
                        side="SELL",
                        score_hint=0.60 if tokyo_session else 0.59,
                        reason="NAS mean reversion at range ceiling",
                        stop_atr=0.9,
                        tp_r=1.5,
                        strategy_family="RANGE",
                        confluence_score=3.6 if tokyo_session else 3.5,
                        confluence_required=3.5,
                        meta={"timeframe": "M15", "atr_field": "m15_atr_14"},
                    )
                )
        return output

    def _nas_premarket_futures(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if not self._in_utc_window(timestamp, 13.0, 13.5):
            return []
        futures_vol = self._ratio_or_neutral(row.get("nas_futures_volume_ratio_20", row.get("nq_volume_ratio_20", 0.0)))
        futures_ret = float(row.get("nas_futures_ret_15", row.get("nq_ret_15", 0.0)))
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_limit("NAS100", multiplier=1.15):
            return []
        if futures_vol < 1.5 or abs(futures_ret) < 0.001:
            return []
        side = "BUY" if futures_ret > 0 else "SELL"
        return [
            SignalCandidate(
                signal_id=deterministic_id(symbol, "router", "nas-premarket", side, timestamp),
                setup="NAS_PREMARKET_VOLUME_CONFIRM",
                side=side,
                score_hint=0.66,
                reason="NASDAQ premarket futures volume and direction confirm cash-open bias",
                stop_atr=0.95,
                tp_r=1.7,
                strategy_family="TREND",
                confluence_score=4.0,
                confluence_required=3.9,
                meta={"timeframe": "M15", "atr_field": "m15_atr_14", "policy_window": "PREMARKET", "setup_family": "FUTURES_CONFIRM"},
            )
        ]

    def _oil_inventory_scalper(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        sensitivity = self._symbol_sensitivity(symbol, self.oil_sensitivity)
        spread = float(row.get("m5_spread", 0.0))
        spread_limit = self._spread_limit("USOIL", multiplier=(1.20 if str(session.session_name).upper() in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.90))
        if spread > spread_limit:
            return []
        atr_ratio = float(row.get("m5_atr_pct_of_avg", 1.0))
        if atr_ratio >= (2.4 + max(0.0, (1.0 - sensitivity))):
            return []
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        prev_high = float(row.get("m15_rolling_high_prev_20", close + atr))
        prev_low = float(row.get("m15_rolling_low_prev_20", close - atr))
        vol_ratio = self._ratio_or_neutral(row.get("m5_volume_ratio_20", 1.0))
        vol_floor = 0.92 - (0.10 * max(0.0, sensitivity - 1.0))
        trend_flag = bool(regime.details.get("trend_flag", regime.label == "TRENDING"))
        session_name = str(session.session_name).upper()
        body_efficiency = float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)) or 0.55)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        impulse = abs(float(row.get("m5_ret_1", 0.0))) / max(atr, 1e-6)
        if session_name == "TOKYO" and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
            vol_floor = max(vol_floor, 1.00)
        output: list[SignalCandidate] = []
        if session_name == "TOKYO":
            if regime_state not in {"TRENDING", "BREAKOUT_EXPANSION"}:
                return []
            breakout_extension = max(close - prev_high, prev_low - close, 0.0) / max(atr, 1e-6)
            if body_efficiency < 0.66 or impulse < 0.09 or impulse > 0.24 or breakout_extension > 0.65:
                return []
            vol_floor = max(vol_floor, 1.08)
        if (
            trend_flag
            and close > prev_high
            and vol_ratio >= vol_floor
            and int(row.get("m5_bullish", 0)) == 1
            and not (session_name == "TOKYO" and range_position > 0.86)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "oil-breakout", "BUY", timestamp),
                    setup="OIL_INVENTORY_SCALPER_BREAKOUT",
                    side="BUY",
                    score_hint=0.61,
                    reason="Oil breakout continuation with session momentum",
                    stop_atr=1.2,
                    tp_r=1.9,
                    strategy_family="TREND",
                    confluence_score=3.7,
                    confluence_required=3.7,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14"},
                )
            )
        if (
            trend_flag
            and close < prev_low
            and vol_ratio >= vol_floor
            and int(row.get("m5_bearish", 0)) == 1
            and not (session_name == "TOKYO" and range_position < 0.14)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "oil-breakout", "SELL", timestamp),
                    setup="OIL_INVENTORY_SCALPER_BREAKOUT",
                    side="SELL",
                    score_hint=0.61,
                    reason="Oil downside breakout continuation with session momentum",
                    stop_atr=1.2,
                    tp_r=1.9,
                    strategy_family="TREND",
                    confluence_score=3.7,
                    confluence_required=3.7,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14"},
                )
            )
        if (
            not trend_flag
            and vol_ratio >= vol_floor
            and regime_state not in {"LOW_LIQUIDITY_CHOP", "NEWS_VOLATILE"}
            and session_name not in {"TOKYO"}
        ):
            range_pos = float(row.get("m15_range_position_20", 0.5))
            if range_pos <= 0.28 and int(row.get("m5_pinbar_bull", 0)) == 1:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "oil-pullback", "BUY", timestamp),
                        setup="OIL_INVENTORY_SCALPER_PULLBACK",
                        side="BUY",
                        score_hint=0.58,
                        reason="Oil range pullback long with rejection wick",
                        stop_atr=1.0,
                        tp_r=1.6,
                        strategy_family="RANGE",
                        confluence_score=3.4,
                        confluence_required=3.4,
                        meta={"timeframe": "M15", "atr_field": "m15_atr_14"},
                    )
                )
            if range_pos >= 0.72 and int(row.get("m5_pinbar_bear", 0)) == 1:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "oil-pullback", "SELL", timestamp),
                        setup="OIL_INVENTORY_SCALPER_PULLBACK",
                        side="SELL",
                        score_hint=0.58,
                        reason="Oil range pullback short with rejection wick",
                        stop_atr=1.0,
                        tp_r=1.6,
                        strategy_family="RANGE",
                        confluence_score=3.4,
                        confluence_required=3.4,
                        meta={"timeframe": "M15", "atr_field": "m15_atr_14"},
                    )
                )
        return output

    def _oil_inventory_event(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        current = self._utc_timestamp(timestamp)
        inventory_window = (
            (int(current.weekday()) == 1 and self._in_utc_window(timestamp, 20.5, 23.0))
            or (int(current.weekday()) == 2 and self._in_utc_window(timestamp, 14.0, 16.0))
        )
        if not inventory_window:
            return []
        surprise = float(row.get("oil_inventory_surprise_m", row.get("usoil_inventory_surprise_m", 0.0)))
        spread = float(row.get("m5_spread", 0.0))
        if abs(surprise) < 2.0 or spread > self._spread_limit("USOIL", multiplier=1.20):
            return []
        side = "BUY" if surprise < 0 else "SELL"
        return [
            SignalCandidate(
                signal_id=deterministic_id(symbol, "router", "oil-inventory-event", side, timestamp),
                setup="USOIL_INVENTORY_EVENT",
                side=side,
                score_hint=0.67,
                reason="USOIL inventory surprise directional event window",
                stop_atr=1.1,
                tp_r=2.0,
                strategy_family="TREND",
                confluence_score=4.1,
                confluence_required=4.0,
                meta={"timeframe": "M15", "atr_field": "m15_atr_14", "policy_window": "INVENTORY_EVENT", "setup_family": "EVENT_DRIVEN"},
            )
        ]

    def _liquidity_sweep_reclaim(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        symbol_key = self._normalize_symbol(symbol)
        if symbol_key not in {"EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "AUDNZD", "NAS100", "USOIL"}:
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}:
            return []
        if session_name == "SYDNEY" and symbol_key not in {"USDJPY", "AUDJPY", "NZDJPY", "AUDNZD"}:
            return []
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        if symbol_key in {"EURUSD", "GBPUSD"} and session_name in {"SYDNEY", "TOKYO"}:
            return []
        if symbol_key == "USOIL" and session_name == "TOKYO" and regime_state == "LOW_LIQUIDITY_CHOP":
            return []
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        high = float(row.get("m15_high", close))
        low = float(row.get("m15_low", close))
        prev_high = float(row.get("m15_rolling_high_prev_20", close + atr))
        prev_low = float(row.get("m15_rolling_low_prev_20", close - atr))
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_limit(symbol_key, multiplier=1.15):
            return []
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        body_efficiency = self._body_efficiency_value(row)
        lower_wick = float(row.get("m5_lower_wick_ratio", 0.0) or 0.0)
        upper_wick = float(row.get("m5_upper_wick_ratio", 0.0) or 0.0)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        if symbol_key in {"EURUSD", "GBPUSD"} and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}:
            min_volume = 1.08 if regime_state in {"RANGING", "MEAN_REVERSION"} else 1.02
            min_body = 0.62 if regime_state in {"RANGING", "MEAN_REVERSION"} else 0.58
            if volume_ratio < min_volume or body_efficiency < min_body:
                return []
        if volume_ratio < (
            0.68 if symbol_key == "NAS100" and session_name == "TOKYO" and regime_state in {"TRENDING", "MEAN_REVERSION"}
            else 0.72 if symbol_key == "NAS100" and session_name == "TOKYO"
            else 0.80
        ):
            return []
        if body_efficiency < (
            0.40 if symbol_key == "NAS100" and session_name == "TOKYO" and regime_state in {"TRENDING", "MEAN_REVERSION"}
            else 0.44 if symbol_key == "NAS100" and session_name == "TOKYO"
            else 0.50
        ):
            return []
        asia_sweep_pair = symbol_key in {"USDJPY", "AUDJPY", "NZDJPY", "AUDNZD", "EURJPY", "GBPJPY"} and session_name in {"SYDNEY", "TOKYO"}
        if symbol_key in {"USDJPY", "AUDJPY", "NZDJPY"} and session_name in {"SYDNEY", "TOKYO"}:
            min_volume = 0.94 if regime_state == "LOW_LIQUIDITY_CHOP" else 0.88 if regime_state in {"RANGING", "MEAN_REVERSION"} else 0.92
            min_body = 0.50 if regime_state == "LOW_LIQUIDITY_CHOP" else 0.48 if regime_state in {"RANGING", "MEAN_REVERSION"} else 0.50
            if volume_ratio < min_volume or body_efficiency < min_body:
                return []
        if symbol_key in {"EURJPY", "GBPJPY"} and session_name in {"SYDNEY", "TOKYO"}:
            if regime_state == "LOW_LIQUIDITY_CHOP":
                return []
            min_volume = 1.02 if regime_state == "MEAN_REVERSION" else 0.98 if regime_state == "RANGING" else 1.00
            min_body = 0.56 if regime_state == "MEAN_REVERSION" else 0.54 if regime_state == "RANGING" else 0.56
            if volume_ratio < min_volume or body_efficiency < min_body:
                return []
        if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"}:
            min_volume = 0.96 if regime_state == "LOW_LIQUIDITY_CHOP" else 0.92 if regime_state in {"RANGING", "MEAN_REVERSION"} else 0.96
            min_body = 0.54 if regime_state == "LOW_LIQUIDITY_CHOP" else 0.52 if regime_state in {"RANGING", "MEAN_REVERSION"} else 0.54
            if volume_ratio < min_volume or body_efficiency < min_body:
                return []
        if symbol_key == "USOIL" and session_name == "TOKYO" and body_efficiency < 0.56:
            return []
        output: list[SignalCandidate] = []
        if (
            low < prev_low
            and close > prev_low
            and (
                int(row.get("m5_pinbar_bull", 0)) == 1
                or int(row.get("m15_engulf_bull", 0)) == 1
                or (
                    symbol_key == "NAS100"
                    and session_name == "TOKYO"
                    and int(row.get("m5_bullish", 0)) == 1
                    and lower_wick >= 0.14
                )
            )
            and lower_wick >= (0.18 if symbol_key in {"AUDJPY", "NZDJPY", "USDJPY", "AUDNZD"} else 0.18 if symbol_key in {"EURJPY", "GBPJPY"} and session_name in {"SYDNEY", "TOKYO"} else 0.12 if symbol_key == "NAS100" and session_name == "TOKYO" and regime_state in {"TRENDING", "MEAN_REVERSION"} else 0.14 if symbol_key == "NAS100" and session_name == "TOKYO" else 0.12)
            and range_position <= (0.28 if symbol_key in {"EURJPY", "GBPJPY"} and session_name in {"SYDNEY", "TOKYO"} else 0.42 if symbol_key == "NAS100" and session_name == "TOKYO" and regime_state in {"TRENDING", "MEAN_REVERSION"} else 0.38 if symbol_key == "NAS100" and session_name == "TOKYO" else 0.32)
            and (not asia_sweep_pair or regime_state != "LOW_LIQUIDITY_CHOP" or range_position <= 0.24)
            and (
                symbol_key not in {"EURUSD", "GBPUSD"}
                or session_name not in {"LONDON", "OVERLAP", "NEW_YORK"}
                or (lower_wick >= 0.18 and range_position <= 0.26)
            )
        ):
            asia_reclaim_score = 0.61 if symbol_key in {"USDJPY", "AUDJPY", "NZDJPY"} and session_name in {"SYDNEY", "TOKYO"} else 0.58 if symbol_key in {"EURJPY", "GBPJPY"} and session_name in {"SYDNEY", "TOKYO"} else 0.60 if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"} else 0.57
            asia_reclaim_confluence = 3.35 if symbol_key in {"USDJPY", "AUDJPY", "NZDJPY"} and session_name in {"SYDNEY", "TOKYO"} else 3.30 if symbol_key in {"EURJPY", "GBPJPY"} and session_name in {"SYDNEY", "TOKYO"} else 3.25 if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"} else 3.2
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "sweep-reclaim", "BUY", timestamp),
                    setup=f"{symbol_key}_SWEEP_RECLAIM",
                    side="BUY",
                    score_hint=asia_reclaim_score,
                    reason="Liquidity sweep below structure with reclaim",
                    stop_atr=0.95,
                    tp_r=1.60 if session_name in {"SYDNEY", "TOKYO"} and symbol_key in {"USDJPY", "AUDJPY", "NZDJPY", "AUDNZD"} else 1.55,
                    strategy_family="FAKEOUT",
                    confluence_score=asia_reclaim_confluence,
                    confluence_required=3.0 if asia_reclaim_score <= 0.60 else 3.05,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.50 if asia_reclaim_score <= 0.60 else 0.51,
                        "approve_small_min_confluence": 3.0 if asia_reclaim_score <= 0.60 else 3.05,
                        "setup_family": "SWEEP_RECLAIM",
                    },
                )
            )
        if (
            high > prev_high
            and close < prev_high
            and (
                int(row.get("m5_pinbar_bear", 0)) == 1
                or int(row.get("m15_engulf_bear", 0)) == 1
                or (
                    symbol_key == "NAS100"
                    and session_name == "TOKYO"
                    and int(row.get("m5_bearish", 0)) == 1
                    and upper_wick >= 0.14
                )
            )
            and upper_wick >= (0.18 if symbol_key in {"AUDJPY", "NZDJPY", "USDJPY", "AUDNZD"} else 0.18 if symbol_key in {"EURJPY", "GBPJPY"} and session_name in {"SYDNEY", "TOKYO"} else 0.12 if symbol_key == "NAS100" and session_name == "TOKYO" and regime_state in {"TRENDING", "MEAN_REVERSION"} else 0.14 if symbol_key == "NAS100" and session_name == "TOKYO" else 0.12)
            and range_position >= (0.72 if symbol_key in {"EURJPY", "GBPJPY"} and session_name in {"SYDNEY", "TOKYO"} else 0.58 if symbol_key == "NAS100" and session_name == "TOKYO" and regime_state in {"TRENDING", "MEAN_REVERSION"} else 0.62 if symbol_key == "NAS100" and session_name == "TOKYO" else 0.68)
            and (not asia_sweep_pair or regime_state != "LOW_LIQUIDITY_CHOP" or range_position >= 0.76)
            and (
                symbol_key not in {"EURUSD", "GBPUSD"}
                or session_name not in {"LONDON", "OVERLAP", "NEW_YORK"}
                or (upper_wick >= 0.18 and range_position >= 0.74)
            )
        ):
            asia_reclaim_score = 0.61 if symbol_key in {"USDJPY", "AUDJPY", "NZDJPY"} and session_name in {"SYDNEY", "TOKYO"} else 0.58 if symbol_key in {"EURJPY", "GBPJPY"} and session_name in {"SYDNEY", "TOKYO"} else 0.60 if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"} else 0.57
            asia_reclaim_confluence = 3.35 if symbol_key in {"USDJPY", "AUDJPY", "NZDJPY"} and session_name in {"SYDNEY", "TOKYO"} else 3.30 if symbol_key in {"EURJPY", "GBPJPY"} and session_name in {"SYDNEY", "TOKYO"} else 3.25 if symbol_key == "AUDNZD" and session_name in {"SYDNEY", "TOKYO"} else 3.2
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "sweep-reclaim", "SELL", timestamp),
                    setup=f"{symbol_key}_SWEEP_RECLAIM",
                    side="SELL",
                    score_hint=asia_reclaim_score,
                    reason="Liquidity sweep above structure with rejection",
                    stop_atr=0.95,
                    tp_r=1.60 if session_name in {"SYDNEY", "TOKYO"} and symbol_key in {"USDJPY", "AUDJPY", "NZDJPY", "AUDNZD"} else 1.55,
                    strategy_family="FAKEOUT",
                    confluence_score=asia_reclaim_confluence,
                    confluence_required=3.0 if asia_reclaim_score <= 0.60 else 3.05,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.50 if asia_reclaim_score <= 0.60 else 0.51,
                        "approve_small_min_confluence": 3.0 if asia_reclaim_score <= 0.60 else 3.05,
                        "setup_family": "SWEEP_RECLAIM",
                    },
                )
            )
        return output

    def _session_pullback_continuation(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        symbol_key = self._normalize_symbol(symbol)
        if symbol_key not in {"EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY", "NAS100", "USOIL"}:
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}:
            return []
        if "TREND" not in session.allowed_strategies:
            return []
        if session_name == "TOKYO" and symbol_key in {"EURUSD", "GBPUSD"}:
            return []

        spread = float(row.get("m5_spread", 0.0))
        spread_cap = self._spread_limit(symbol_key, multiplier=(1.10 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.95))
        if spread > spread_cap:
            return []

        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        ema20 = float(row.get("m5_ema_20", close))
        ema50 = float(row.get("m5_ema_50", close))
        h1_ema_50 = float(row.get("h1_ema_50", row.get("m15_ema_20", close)))
        h1_ema_200 = float(row.get("h1_ema_200", row.get("h1_ema_50", h1_ema_50)))
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        distance_to_ema = abs(close - ema20)
        pullback_ratio = distance_to_ema / max(atr, 1e-6)
        pullback_ok = (atr * 0.06) <= distance_to_ema <= (
            atr * (0.76 if symbol_key in {"USDJPY", "EURJPY", "GBPJPY"} else 0.70)
        )
        if not pullback_ok:
            return []
        body_efficiency = self._body_efficiency_value(row)
        if body_efficiency < 0.42:
            return []
        raw_regime_label = str(getattr(regime, "state_label", regime.label) or regime.label).strip().upper()
        drift_trend_ready = self._drift_continuation_ready(
            symbol=symbol_key,
            row=row,
            session_name=session_name,
            raw_regime_label=raw_regime_label,
        )

        if session_name == "TOKYO" and symbol_key == "USDJPY":
            if volume_ratio < (0.88 if drift_trend_ready else 1.04) or distance_to_ema > (atr * 0.28) or body_efficiency < (0.42 if drift_trend_ready else 0.54):
                return []
        elif session_name == "TOKYO" and symbol_key in {"EURJPY", "GBPJPY"}:
            return []
        elif symbol_key in {"EURJPY", "GBPJPY"} and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}:
            if volume_ratio < (0.88 if drift_trend_ready else 0.94) or body_efficiency < (0.40 if drift_trend_ready else 0.48) or pullback_ratio > 0.54:
                return []
        elif symbol_key == "GBPUSD" and (pullback_ratio > 0.42 or volume_ratio < (0.88 if drift_trend_ready else 1.00) or body_efficiency < (0.40 if drift_trend_ready else 0.50)):
            return []
        elif symbol_key == "EURUSD" and session_name in {"LONDON", "OVERLAP", "NEW_YORK"} and volume_ratio < (0.88 if drift_trend_ready else 0.94):
            return []

        trend_flag = bool(regime.details.get("trend_flag", regime.label == "TRENDING"))
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        if symbol_key == "USOIL" and session_name == "TOKYO" and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
            return []
        if symbol_key == "NAS100":
            if session_name not in {"LONDON", "OVERLAP", "NEW_YORK"}:
                return []
            if regime_state != "BREAKOUT_EXPANSION":
                return []
            if volume_ratio < 1.02 or body_efficiency < 0.56:
                return []
        if (not trend_flag and not drift_trend_ready) or (regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"} and not drift_trend_ready):
            return []
        bullish_bias = ema20 >= ema50 and h1_ema_50 >= h1_ema_200 and close >= (ema20 - (atr * 0.18))
        bearish_bias = ema20 <= ema50 and h1_ema_50 <= h1_ema_200 and close <= (ema20 + (atr * 0.18))
        range_position = float(row.get("m15_range_position_20", 0.5))
        bullish_ok = bullish_bias and range_position <= (0.78 if session_name == "TOKYO" and symbol_key == "USDJPY" else 0.84)
        bearish_ok = bearish_bias and range_position >= (0.22 if session_name == "TOKYO" and symbol_key == "USDJPY" else 0.16)
        ret_1 = float(row.get("m15_ret_1", row.get("m5_ret_1", 0.0)) or 0.0)
        ret_3 = float(row.get("m15_ret_3", row.get("m5_ret_3", 0.0)) or 0.0)
        lower_wick = float(row.get("m5_lower_wick_ratio", 0.0) or 0.0)
        upper_wick = float(row.get("m5_upper_wick_ratio", 0.0) or 0.0)
        ret_fields_present = any(
            key in row.index and row.get(key) is not None and not pd.isna(row.get(key))
            for key in ("m15_ret_1", "m5_ret_1", "m15_ret_3", "m5_ret_3")
        )
        clean_reclaim_bias = body_efficiency >= ((0.42 if session_name == "TOKYO" and symbol_key == "USDJPY" else 0.40) if drift_trend_ready else (0.52 if session_name == "TOKYO" and symbol_key == "USDJPY" else 0.54)) and volume_ratio >= ((0.88 if session_name == "TOKYO" and symbol_key == "USDJPY" else 0.88) if drift_trend_ready else (1.00 if session_name == "TOKYO" and symbol_key == "USDJPY" else (1.00 if session_name == "TOKYO" else 0.94)))
        has_reclaim_impulse = abs(ret_1) >= (atr * (0.04 if session_name == "TOKYO" and symbol_key == "USDJPY" else 0.04)) or abs(ret_3) >= (atr * (0.08 if session_name == "TOKYO" and symbol_key == "USDJPY" else 0.08))
        if ret_fields_present:
            if not has_reclaim_impulse:
                return []
        elif not clean_reclaim_bias:
            return []
        if session_name == "TOKYO" and symbol_key == "USDJPY":
            if ret_1 > 0 and lower_wick < 0.16:
                return []
            if ret_1 < 0 and upper_wick < 0.16:
                return []
            if abs(ret_3) >= (atr * 0.26):
                return []
        base_confluence = (
            2.9
            + (0.30 if trend_flag else 0.10)
            + min(0.55, max(0.0, volume_ratio - 0.85))
            + (0.25 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.10)
        )
        if session_name == "TOKYO" and symbol_key in {"EURUSD", "GBPUSD"}:
            base_confluence -= 0.18
        elif session_name == "TOKYO" and symbol_key == "USDJPY":
            base_confluence -= 0.12
        confluence = clamp(base_confluence, 0.0, 5.0)
        output: list[SignalCandidate] = []
        drift_meta = {
            "session_drift_lane": True,
            "structure_cleanliness_floor": 0.56,
            "entry_timing_floor": 0.58,
            "setup_family": "DRIFT_CONTINUATION",
        } if drift_trend_ready else {}
        score_hint = clamp(0.56 + (0.04 * (confluence - 3.0)), 0.53, 0.74)
        if session_name == "TOKYO" and symbol_key in {"EURUSD", "GBPUSD"}:
            score_hint = clamp(score_hint - 0.04, 0.50, 0.68)
        elif session_name == "TOKYO" and symbol_key == "USDJPY":
            score_hint = clamp(score_hint - 0.02, 0.51, 0.71)
        bullish_reclaim = ret_1 > 0 or (not ret_fields_present and clean_reclaim_bias and bullish_ok)
        bearish_reclaim = ret_1 < 0 or (not ret_fields_present and clean_reclaim_bias and bearish_ok)
        nas_meta = {
            "strategy_key": "NAS100_VWAP_TREND_STRATEGY",
            "setup_family": "VWAP_PULLBACK",
        } if symbol_key == "NAS100" else {}
        if bullish_ok and bullish_reclaim:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "session-pullback", "BUY", timestamp),
                    setup="NAS100_VWAP_PULLBACK" if symbol_key == "NAS100" else f"{symbol_key}_SESSION_PULLBACK",
                    side="BUY",
                    score_hint=score_hint,
                    reason="Session pullback continuation with higher-timeframe alignment",
                    stop_atr=0.88 if symbol_key == "NAS100" else 0.95,
                    tp_r=1.95 if symbol_key == "NAS100" else (1.8 if symbol_key in {"EURUSD", "GBPUSD"} else 1.7),
                    entry_kind="DAYTRADE" if symbol_key == "NAS100" else "SCALP",
                    strategy_family="TREND",
                    confluence_score=confluence,
                    confluence_required=3.0,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.50,
                        "approve_small_min_confluence": 3.0,
                        "setup_family": "PULLBACK",
                        **nas_meta,
                        **drift_meta,
                    },
                )
            )
        if bearish_ok and bearish_reclaim:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "session-pullback", "SELL", timestamp),
                    setup="NAS100_VWAP_PULLBACK" if symbol_key == "NAS100" else f"{symbol_key}_SESSION_PULLBACK",
                    side="SELL",
                    score_hint=score_hint,
                    reason="Session downside pullback continuation with higher-timeframe alignment",
                    stop_atr=0.88 if symbol_key == "NAS100" else 0.95,
                    tp_r=1.95 if symbol_key == "NAS100" else (1.8 if symbol_key in {"EURUSD", "GBPUSD"} else 1.7),
                    entry_kind="DAYTRADE" if symbol_key == "NAS100" else "SCALP",
                    strategy_family="TREND",
                    confluence_score=confluence,
                    confluence_required=3.0,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.50,
                        "approve_small_min_confluence": 3.0,
                        "setup_family": "PULLBACK",
                        **nas_meta,
                        **drift_meta,
                    },
                )
            )
        return output

    def _btc(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        sensitivity = self._symbol_sensitivity(symbol, self.btc_sensitivity)
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        raw_regime_label = str(getattr(regime, "state_label", regime.label) or regime.label).strip().upper()
        atr_ratio = float(row.get("m5_atr_pct_of_avg", 1.0))
        spread_points = float(row.get("m5_spread", row.get("spread_points", 0.0)) or 0.0)
        close = float(row.get("m5_close", 0.0))
        atr = max(float(row.get("m5_atr_14", row.get("m15_atr_14", 0.0))), 1e-6)
        point_size = max(
            float(row.get("point", 0.0) or 0.0),
            float(row.get("trade_tick_size", row.get("tick_size", 0.0)) or 0.0),
        )
        if point_size <= 0.0:
            digits = int(row.get("digits", 0) or 0)
            if digits > 0:
                point_size = 10.0 ** (-digits)
        if point_size <= 0.0 and symbol_asset_class(symbol) == "crypto":
            point_size = 0.01
        spread = spread_points * point_size if point_size > 0.0 else spread_points
        stored_spread_ratio = float(row.get("m5_spread_ratio_20", 1.0) or 1.0)
        computed_spread_ratio = spread / max(atr, 1e-6)
        spread_ratio = computed_spread_ratio if point_size > 0.0 or stored_spread_ratio > 5.0 else stored_spread_ratio
        weekend_mode = self._is_weekend_market_mode(timestamp)
        raw_spread_cap_points = max(float(self.btc_spread_cap_points), self.max_spread_points * (34.0 if weekend_mode else 20.0), 100.0)
        raw_spread_cap = raw_spread_cap_points * point_size if point_size > 0.0 else raw_spread_cap_points
        session_window_allowed = self._btc_session_window_allowed(timestamp)
        if spread_ratio >= (3.25 + max(0.0, 1.0 - sensitivity)):
            return []
        if atr_ratio >= (3.4 + max(0.0, 1.0 - sensitivity)) and spread_ratio >= 2.0:
            return []
        if spread > raw_spread_cap and spread_ratio >= 1.5:
            return []
        move_30m = max(
            abs(float(row.get("m5_ret_5", 0.0))),
            abs(float(row.get("m15_ret_1", 0.0))),
        )
        if move_30m >= float(self.btc_volatility_pause_move_pct_30m):
            return []
        output: list[SignalCandidate] = []
        ema20 = float(row.get("m5_ema_20", 0.0))
        ema50 = float(row.get("m5_ema_50", 0.0))
        prev_high = float(row.get("m15_rolling_high_prev_20", row.get("m5_rolling_high_prev_20", close)))
        prev_low = float(row.get("m15_rolling_low_prev_20", row.get("m5_rolling_low_prev_20", close)))
        vol_ratio = self._ratio_or_neutral(row.get("m5_volume_ratio_20", 1.0))
        body_efficiency = float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)) or 0.55)
        momentum = float(row.get("m5_ret_1", 0.0))
        bullish = int(row.get("m5_bullish", 0)) == 1 or momentum > 0
        bearish = int(row.get("m5_bearish", 0)) == 1 or momentum < 0
        breakout_up = close > prev_high and vol_ratio >= 0.92
        breakout_down = close < prev_low and vol_ratio >= 0.92
        trend_flag = bool(regime.details.get("trend_flag", regime.label == "TRENDING"))
        session_name = str(session.session_name).upper()
        btc_trend_allowed = "TREND" in session.allowed_strategies or session_name in set(self.btc_trade_sessions) or weekend_mode
        base_confluence = 3.35 + min(0.9, max(0.0, vol_ratio - 0.80)) + (0.25 if trend_flag else 0.0)
        if weekend_mode:
            base_confluence += 0.10
        base_probability = max(0.54, float(self.btc_min_ai_confidence) - 0.03)
        approve_small_probability = max(0.50, float(self.btc_min_ai_confidence) - 0.10)
        approve_small_confluence = 2.9
        range_position = float(row.get("m15_range_position_20", 0.5))
        rsi = float(row.get("m5_rsi_14", 50.0))
        trend_gap_ratio = abs(ema20 - ema50) / max(abs(close), 1e-6)
        compression = atr_ratio <= 0.88 and spread_ratio <= 1.35
        impulse_floor = max(0.0007, 0.00062 / max(sensitivity, 0.8))
        impulse_ready = abs(momentum) >= impulse_floor
        impulse = abs(momentum) / max(atr / max(abs(close), 1e-6), 1e-6)
        bullish_reclaim = low_reclaim = close > prev_low and int(row.get("m5_pinbar_bull", 0)) == 1
        bearish_reject = high_reject = close < prev_high and int(row.get("m5_pinbar_bear", 0)) == 1
        alignment_score = float(row.get("multi_tf_alignment_score", 0.5) or 0.5)
        fractal_score = float(
            row.get(
                "fractal_persistence_score",
                row.get("hurst_persistence_score", row.get("m15_hurst_proxy_64", 0.5)),
            )
            or 0.5
        )
        instability_score = float(row.get("market_instability_score", row.get("feature_drift_score", 0.0)) or 0.0)
        seasonality_score = float(row.get("seasonality_edge_score", 0.5) or 0.5)
        drift_move_30m = max(abs(float(row.get("m5_ret_5", 0.0) or 0.0)), abs(float(row.get("m15_ret_1", 0.0) or 0.0)))
        weekend_asia_bad_regime = bool(
            weekend_mode
            and session_name in {"SYDNEY", "TOKYO"}
            and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
        )
        weekend_exceptional_trend = bool(
            trend_flag
            and alignment_score >= 0.62
            and fractal_score >= 0.58
            and body_efficiency >= 0.62
            and instability_score <= 0.42
            and seasonality_score >= 0.44
            and impulse_ready
        )
        allow_weekend_asia_trend_candidates = bool(not weekend_asia_bad_regime or weekend_exceptional_trend)
        weekend_drift_trend_ready = bool(
            weekend_mode
            and raw_regime_label == "LOW_LIQUIDITY_DRIFT"
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and spread_ratio <= 1.12
            and vol_ratio >= 0.92
            and alignment_score >= 0.92
            and fractal_score >= 0.90
            and instability_score <= 0.12
            and seasonality_score >= 0.50
            and body_efficiency >= 0.38
            and trend_gap_ratio >= 0.00035
            and drift_move_30m >= 0.00014
        )
        trend_context_ready = bool(trend_flag or weekend_drift_trend_ready)
        impulse_context_ready = bool(impulse_ready or weekend_drift_trend_ready)
        funding_available = self._feature_available(row, "btc_funding_rate_8h", "funding_rate_8h")
        liquidation_available = self._feature_available(row, "btc_liquidations_5m_usd", "liquidation_5m_usd")
        whale_available = self._feature_available(row, "btc_whale_flow_btc_1h", "btc_exchange_flow_btc_1h")
        dxy_available = self._feature_available(row, "dxy_ret_5", "dxy_ret_1")
        gap_available = self._feature_available(row, "btc_weekend_gap_pct", "weekend_gap_pct")
        proxyless_price_action_mode = not any(
            (funding_available, liquidation_available, whale_available, dxy_available, gap_available)
        )

        funding_rate = float(row.get("btc_funding_rate_8h", row.get("funding_rate_8h", 0.0)))
        if self._btc_funding_entry_window(timestamp) and abs(funding_rate) >= float(self.btc_funding_rate_min_abs):
            fade_side = "SELL" if funding_rate > 0 else "BUY"
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "btc-funding", fade_side, timestamp),
                    setup="BTC_FUNDING_ARB",
                    side=fade_side,
                    score_hint=max(0.66, float(self.btc_min_ai_confidence) + 0.01),
                    reason="BTC funding arb against crowded positioning",
                    stop_atr=1.15,
                    tp_r=1.6,
                    strategy_family="TREND",
                    confluence_score=4.2,
                    confluence_required=4.0,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "btc_strategy": "FUNDING_ARB",
                        "btc_min_ai_confidence": float(self.btc_min_ai_confidence),
                    },
                )
            )

        liquidation_notional = float(row.get("btc_liquidations_5m_usd", row.get("liquidation_5m_usd", 0.0)))
        if liquidation_notional >= float(self.btc_liquidation_usd_threshold) and vol_ratio >= 1.4:
            if int(row.get("m5_pinbar_bull", 0)) == 1 or int(row.get("m5_engulf_bull", 0)) == 1:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-liq", "BUY", timestamp),
                        setup="BTC_LIQUIDATION_FADE",
                        side="BUY",
                        score_hint=0.68,
                        reason="BTC liquidation cascade fade long",
                        stop_atr=0.95,
                        tp_r=1.5,
                        strategy_family="FAKEOUT",
                        confluence_score=4.1,
                        confluence_required=4.0,
                        meta={"timeframe": "M15", "atr_field": "m5_atr_14", "btc_strategy": "LIQUIDATION_FADE", "btc_min_ai_confidence": float(self.btc_min_ai_confidence)},
                    )
                )
            if int(row.get("m5_pinbar_bear", 0)) == 1 or int(row.get("m5_engulf_bear", 0)) == 1:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-liq", "SELL", timestamp),
                        setup="BTC_LIQUIDATION_FADE",
                        side="SELL",
                        score_hint=0.68,
                        reason="BTC short liquidation fade short",
                        stop_atr=0.95,
                        tp_r=1.5,
                        strategy_family="FAKEOUT",
                        confluence_score=4.1,
                        confluence_required=4.0,
                        meta={"timeframe": "M15", "atr_field": "m5_atr_14", "btc_strategy": "LIQUIDATION_FADE", "btc_min_ai_confidence": float(self.btc_min_ai_confidence)},
                    )
                )

        weekend_gap_pct = float(row.get("btc_weekend_gap_pct", row.get("weekend_gap_pct", 0.0)))
        if weekend_mode and abs(weekend_gap_pct) >= float(self.btc_weekend_gap_min_pct):
            gap_side = "SELL" if weekend_gap_pct > 0 else "BUY"
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "btc-weekend-gap", gap_side, timestamp),
                    setup="BTC_WEEKEND_GAP_FADE",
                    side=gap_side,
                    score_hint=0.70,
                    reason="BTC weekend gap fade toward Friday close",
                    stop_atr=1.05,
                    tp_r=1.8,
                    strategy_family="FAKEOUT",
                    confluence_score=4.4,
                    confluence_required=4.1,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "btc_strategy": "WEEKEND_GAP_FADE", "btc_min_ai_confidence": float(self.btc_min_ai_confidence)},
                )
            )

        if (
            weekend_mode
            and spread_ratio <= 1.6
            and vol_ratio >= 1.05
            and allow_weekend_asia_trend_candidates
            and trend_context_ready
            and alignment_score >= 0.54
            and fractal_score >= 0.52
            and instability_score <= 0.68
            and seasonality_score >= 0.40
            and body_efficiency >= 0.58
            and impulse_context_ready
        ):
            if breakout_up and ema20 >= ema50:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-weekend-breakout", "BUY", timestamp),
                        setup="BTC_WEEKEND_BREAKOUT_RECLAIM",
                        side="BUY",
                        score_hint=0.67,
                        reason="BTC weekend breakout reclaim with supportive momentum",
                        stop_atr=0.85,
                        tp_r=1.6,
                        strategy_family="TREND",
                        confluence_score=4.0,
                        confluence_required=3.9,
                        meta={"timeframe": "M15", "atr_field": "m5_atr_14", "btc_strategy": "WEEKEND_BREAKOUT_RECLAIM", "btc_min_ai_confidence": float(self.btc_min_ai_confidence)},
                    )
                )
            if breakout_down and ema20 <= ema50:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-weekend-breakout", "SELL", timestamp),
                        setup="BTC_WEEKEND_BREAKOUT_RECLAIM",
                        side="SELL",
                        score_hint=0.67,
                        reason="BTC weekend downside breakout reclaim with supportive momentum",
                        stop_atr=0.85,
                        tp_r=1.6,
                        strategy_family="TREND",
                        confluence_score=4.0,
                        confluence_required=3.9,
                        meta={"timeframe": "M15", "atr_field": "m5_atr_14", "btc_strategy": "WEEKEND_BREAKOUT_RECLAIM", "btc_min_ai_confidence": float(self.btc_min_ai_confidence)},
                    )
                )

        if weekend_drift_trend_ready:
            if ema20 >= ema50 and momentum >= 0.0 and close >= ema20 and range_position >= 0.55:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-weekend-drift", "BUY", timestamp),
                        setup="BTC_MOMENTUM_CONTINUATION",
                        side="BUY",
                        score_hint=clamp(base_probability + 0.10, 0.64, 0.82),
                        reason="BTC weekend drift continuation with aligned multi-TF structure",
                        stop_atr=1.20,
                        tp_r=1.8,
                        strategy_family="TREND",
                        confluence_score=clamp(4.0 + min(0.4, max(0.0, vol_ratio - 0.90)) + (0.20 * alignment_score), 3.9, 4.8),
                        confluence_required=3.9,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": 3.7,
                            "btc_strategy": "WEEKEND_DRIFT_CONTINUATION",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.54,
                            "btc_weekend_drift_lane": True,
                            "structure_cleanliness_floor": 0.60,
                            "entry_timing_floor": 0.62,
                        },
                    )
                )
            if ema20 <= ema50 and momentum <= 0.0 and close <= ema20 and range_position <= 0.45:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-weekend-drift", "SELL", timestamp),
                        setup="BTC_MOMENTUM_CONTINUATION",
                        side="SELL",
                        score_hint=clamp(base_probability + 0.10, 0.64, 0.82),
                        reason="BTC weekend downside drift continuation with aligned multi-TF structure",
                        stop_atr=1.20,
                        tp_r=1.8,
                        strategy_family="TREND",
                        confluence_score=clamp(4.0 + min(0.4, max(0.0, vol_ratio - 0.90)) + (0.20 * alignment_score), 3.9, 4.8),
                        confluence_required=3.9,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": 3.7,
                            "btc_strategy": "WEEKEND_DRIFT_CONTINUATION",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.54,
                            "btc_weekend_drift_lane": True,
                            "structure_cleanliness_floor": 0.60,
                            "entry_timing_floor": 0.62,
                        },
                    )
                )

        if (
            breakout_up
            and vol_ratio >= (0.98 - (0.08 * max(0.0, sensitivity - 1.0)))
            and range_position <= 0.82
            and (not weekend_asia_bad_regime or weekend_exceptional_trend)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "btc-range-expansion", "BUY", timestamp),
                    setup="BTC_RANGE_EXPANSION",
                    side="BUY",
                    score_hint=clamp(base_probability + 0.01, 0.55, 0.74),
                    reason="BTC range expansion breakout with executable spread",
                    stop_atr=0.85,
                    tp_r=1.45,
                    strategy_family="TREND",
                    confluence_score=clamp(base_confluence, 3.0, 4.6),
                    confluence_required=3.2,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m5_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": approve_small_probability,
                        "approve_small_min_confluence": approve_small_confluence,
                        "btc_strategy": "RANGE_EXPANSION",
                        "setup_family": "PRICE_ACTION",
                        "btc_min_ai_confidence": 0.50,
                    },
                )
            )
        if (
            breakout_down
            and vol_ratio >= (0.98 - (0.08 * max(0.0, sensitivity - 1.0)))
            and range_position >= 0.18
            and (not weekend_asia_bad_regime or weekend_exceptional_trend)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "btc-range-expansion", "SELL", timestamp),
                    setup="BTC_RANGE_EXPANSION",
                    side="SELL",
                    score_hint=clamp(base_probability + 0.01, 0.55, 0.74),
                    reason="BTC downside range expansion breakout with executable spread",
                    stop_atr=0.85,
                    tp_r=1.45,
                    strategy_family="TREND",
                    confluence_score=clamp(base_confluence, 3.0, 4.6),
                    confluence_required=3.2,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m5_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": approve_small_probability,
                        "approve_small_min_confluence": approve_small_confluence,
                        "btc_strategy": "RANGE_EXPANSION",
                        "setup_family": "PRICE_ACTION",
                        "btc_min_ai_confidence": 0.50,
                    },
                )
            )

        near_ema = abs(close - ema20) <= (atr * 1.08)
        h1_ema20 = float(row.get("h1_ema_20", ema20))
        h1_ema50 = float(row.get("h1_ema_50", ema50))
        rsi = float(row.get("m5_rsi_14", 50.0))
        if (
            btc_trend_allowed
            and trend_context_ready
            and near_ema
            and vol_ratio >= 0.68
            and (not weekend_mode or session_name not in {"SYDNEY", "TOKYO"} or allow_weekend_asia_trend_candidates)
        ):
            if ema20 >= ema50 and h1_ema20 >= h1_ema50 and close >= (ema20 - (atr * 0.55)) and rsi <= 69:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-momentum-continuation", "BUY", timestamp),
                        setup="BTC_MOMENTUM_CONTINUATION",
                        side="BUY",
                        score_hint=clamp(base_probability, 0.55, 0.74),
                        reason="BTC trend pullback continuation with executable spread",
                        stop_atr=0.90,
                        tp_r=1.55,
                        strategy_family="TREND",
                        confluence_score=clamp(base_confluence + 0.10, 3.1, 4.7),
                        confluence_required=3.1,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": approve_small_confluence,
                            "btc_strategy": "MOMENTUM_CONTINUATION",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )
            if ema20 <= ema50 and h1_ema20 <= h1_ema50 and close <= (ema20 + (atr * 0.55)) and rsi >= 31:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-momentum-continuation", "SELL", timestamp),
                        setup="BTC_MOMENTUM_CONTINUATION",
                        side="SELL",
                        score_hint=clamp(base_probability, 0.55, 0.74),
                        reason="BTC downside trend pullback continuation with executable spread",
                        stop_atr=0.90,
                        tp_r=1.55,
                        strategy_family="TREND",
                        confluence_score=clamp(base_confluence + 0.10, 3.1, 4.7),
                        confluence_required=3.1,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": approve_small_confluence,
                            "btc_strategy": "MOMENTUM_CONTINUATION",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )

        if (
            ((range_position <= 0.22 and bullish_reclaim) or (int(row.get("m5_engulf_bull", 0)) == 1 and range_position <= 0.30))
            and (not weekend_mode or session_name not in {"SYDNEY", "TOKYO"} or allow_weekend_asia_trend_candidates)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "btc-liquidity-sweep", "BUY", timestamp),
                    setup="BTC_LIQUIDITY_SWEEP_FADE",
                    side="BUY",
                    score_hint=clamp(base_probability, 0.54, 0.73),
                    reason="BTC sweep below local range with reclaim",
                    stop_atr=0.90,
                    tp_r=1.40,
                    strategy_family="FAKEOUT",
                    confluence_score=clamp(3.2 + min(0.8, vol_ratio - 0.8), 3.0, 4.4),
                    confluence_required=3.0,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m5_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": approve_small_probability,
                        "approve_small_min_confluence": approve_small_confluence,
                        "btc_strategy": "LIQUIDITY_SWEEP_FADE",
                        "setup_family": "PRICE_ACTION",
                        "btc_min_ai_confidence": 0.50,
                    },
                )
            )
        if (
            ((range_position >= 0.78 and bearish_reject) or (int(row.get("m5_engulf_bear", 0)) == 1 and range_position >= 0.70))
            and (not weekend_mode or session_name not in {"SYDNEY", "TOKYO"} or allow_weekend_asia_trend_candidates)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "btc-liquidity-sweep", "SELL", timestamp),
                    setup="BTC_LIQUIDITY_SWEEP_FADE",
                    side="SELL",
                    score_hint=clamp(base_probability, 0.54, 0.73),
                    reason="BTC sweep above local range with rejection",
                    stop_atr=0.90,
                    tp_r=1.40,
                    strategy_family="FAKEOUT",
                    confluence_score=clamp(3.2 + min(0.8, vol_ratio - 0.8), 3.0, 4.4),
                    confluence_required=3.0,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m5_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": approve_small_probability,
                        "approve_small_min_confluence": approve_small_confluence,
                        "btc_strategy": "LIQUIDITY_SWEEP_FADE",
                        "setup_family": "PRICE_ACTION",
                        "btc_min_ai_confidence": 0.50,
                    },
                )
            )

        if (
            trend_flag
            and trend_gap_ratio >= 0.0010
            and abs(close - ema20) <= (atr * 1.15)
            and vol_ratio >= 0.66
            and body_efficiency >= 0.46
            and (not weekend_mode or session_name not in {"SYDNEY", "TOKYO"} or allow_weekend_asia_trend_candidates)
        ):
            if ema20 >= ema50 and low_reclaim and range_position <= 0.52:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-price-action-reclaim", "BUY", timestamp),
                        setup="BTC_PRICE_ACTION_RECLAIM",
                        side="BUY",
                        score_hint=clamp(base_probability, 0.54, 0.73),
                        reason="BTC price-action reclaim aligned with local trend support",
                        stop_atr=0.85,
                        tp_r=1.55,
                        strategy_family="TREND",
                        confluence_score=clamp(base_confluence + 0.05, 3.1, 4.6),
                        confluence_required=3.0,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": approve_small_confluence,
                            "btc_strategy": "PRICE_ACTION_RECLAIM",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )
            if ema20 <= ema50 and high_reject and range_position >= 0.48:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-price-action-reclaim", "SELL", timestamp),
                        setup="BTC_PRICE_ACTION_RECLAIM",
                        side="SELL",
                        score_hint=clamp(base_probability, 0.54, 0.73),
                        reason="BTC price-action rejection aligned with local trend pressure",
                        stop_atr=0.85,
                        tp_r=1.55,
                        strategy_family="TREND",
                        confluence_score=clamp(base_confluence + 0.05, 3.1, 4.6),
                        confluence_required=3.0,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": approve_small_confluence,
                            "btc_strategy": "PRICE_ACTION_RECLAIM",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )
        

        if (
            trend_context_ready
            and impulse_context_ready
            and vol_ratio >= 0.82
            and ema20 >= ema50
            and bullish
            and (not weekend_mode or session_name not in {"SYDNEY", "TOKYO"} or allow_weekend_asia_trend_candidates)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "btc-momentum-cont", "BUY", timestamp),
                    setup="BTC_MOMENTUM_CONTINUATION",
                    side="BUY",
                    score_hint=clamp(base_probability + 0.02, 0.56, 0.76),
                    reason="BTC momentum continuation on executable impulse",
                    stop_atr=0.90,
                    tp_r=1.55,
                    strategy_family="TREND",
                    confluence_score=clamp(base_confluence + 0.15, 3.1, 4.8),
                    confluence_required=3.2,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m5_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": approve_small_probability,
                        "approve_small_min_confluence": approve_small_confluence,
                        "btc_strategy": "MOMENTUM_CONTINUATION",
                        "setup_family": "PRICE_ACTION",
                        "btc_min_ai_confidence": 0.50,
                    },
                )
            )
        if (
            trend_context_ready
            and impulse_context_ready
            and vol_ratio >= 0.82
            and ema20 <= ema50
            and bearish
            and (not weekend_mode or session_name not in {"SYDNEY", "TOKYO"} or allow_weekend_asia_trend_candidates)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "btc-momentum-cont", "SELL", timestamp),
                    setup="BTC_MOMENTUM_CONTINUATION",
                    side="SELL",
                    score_hint=clamp(base_probability + 0.02, 0.56, 0.76),
                    reason="BTC downside momentum continuation on executable impulse",
                    stop_atr=0.90,
                    tp_r=1.55,
                    strategy_family="TREND",
                    confluence_score=clamp(base_confluence + 0.15, 3.1, 4.8),
                    confluence_required=3.2,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m5_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": approve_small_probability,
                        "approve_small_min_confluence": approve_small_confluence,
                        "btc_strategy": "MOMENTUM_CONTINUATION",
                        "setup_family": "PRICE_ACTION",
                        "btc_min_ai_confidence": 0.50,
                    },
                )
            )

        if (
            compression
            and vol_ratio >= 0.74
            and session_name in {"TOKYO", "LONDON", "OVERLAP", "NEW_YORK", "SYDNEY"}
            and (not weekend_mode or session_name not in {"SYDNEY", "TOKYO"} or allow_weekend_asia_trend_candidates)
        ):
            if bullish and ema20 >= ema50:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-compression-break", "BUY", timestamp),
                        setup="BTC_WEEKEND_BREAKOUT" if weekend_mode else "BTC_TOKYO_DRIFT_SCALP" if session_name == "TOKYO" else "BTC_LONDON_IMPULSE_SCALP" if session_name == "LONDON" else "BTC_NY_LIQUIDITY",
                        side="BUY",
                        score_hint=clamp(base_probability - 0.01, 0.53, 0.72),
                        reason="BTC compression to expansion directional scalp",
                        stop_atr=0.80,
                        tp_r=1.35,
                        strategy_family="SCALP",
                        confluence_score=3.1,
                        confluence_required=3.0,
                        meta={
                            "timeframe": "M5",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": 3.0,
                            "btc_strategy": "SESSION_EXPANSION",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )
            if bearish and ema20 <= ema50:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-compression-break", "SELL", timestamp),
                        setup="BTC_WEEKEND_BREAKOUT" if weekend_mode else "BTC_TOKYO_DRIFT_SCALP" if session_name == "TOKYO" else "BTC_LONDON_IMPULSE_SCALP" if session_name == "LONDON" else "BTC_NY_LIQUIDITY",
                        side="SELL",
                        score_hint=clamp(base_probability - 0.01, 0.53, 0.72),
                        reason="BTC compression to expansion directional scalp",
                        stop_atr=0.80,
                        tp_r=1.35,
                        strategy_family="SCALP",
                        confluence_score=3.1,
                        confluence_required=3.0,
                        meta={
                            "timeframe": "M5",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": 3.0,
                            "btc_strategy": "SESSION_EXPANSION",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )

        if (
            ((rsi <= 38 and range_position <= 0.30 and int(row.get("m5_pinbar_bull", 0)) == 1) or low_reclaim)
            and (not weekend_mode or session_name not in {"SYDNEY", "TOKYO"} or allow_weekend_asia_trend_candidates)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "btc-exhaustion", "BUY", timestamp),
                    setup="BTC_EXHAUSTION_REVERSION",
                    side="BUY",
                    score_hint=0.55,
                    reason="BTC exhaustion fade with reclaim support",
                    stop_atr=0.85,
                    tp_r=1.30,
                    strategy_family="RANGE",
                    confluence_score=3.0,
                    confluence_required=2.9,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m5_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.50,
                        "approve_small_min_confluence": 2.9,
                        "btc_strategy": "EXHAUSTION_REVERSION",
                        "setup_family": "PRICE_ACTION",
                        "btc_min_ai_confidence": 0.50,
                    },
                )
            )
        if (
            ((rsi >= 62 and range_position >= 0.70 and int(row.get("m5_pinbar_bear", 0)) == 1) or high_reject)
            and (not weekend_mode or session_name not in {"SYDNEY", "TOKYO"} or allow_weekend_asia_trend_candidates)
        ):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "btc-exhaustion", "SELL", timestamp),
                    setup="BTC_EXHAUSTION_REVERSION",
                    side="SELL",
                    score_hint=0.55,
                    reason="BTC exhaustion fade with rejection overhead",
                    stop_atr=0.85,
                    tp_r=1.30,
                    strategy_family="RANGE",
                    confluence_score=3.0,
                    confluence_required=2.9,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m5_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.50,
                        "approve_small_min_confluence": 2.9,
                        "btc_strategy": "EXHAUSTION_REVERSION",
                        "setup_family": "PRICE_ACTION",
                        "btc_min_ai_confidence": 0.50,
                    },
                )
            )

        whale_flow_btc = float(row.get("btc_whale_flow_btc_1h", row.get("btc_exchange_flow_btc_1h", 0.0)))
        if abs(whale_flow_btc) >= float(self.btc_whale_flow_threshold_btc) and vol_ratio >= 1.0:
            flow_side = "BUY" if whale_flow_btc < 0 else "SELL"
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "btc-whale", flow_side, timestamp),
                    setup="BTC_WHALE_FLOW_BREAKOUT",
                    side=flow_side,
                    score_hint=0.67,
                    reason="BTC whale exchange flow confirmation",
                    stop_atr=1.0,
                    tp_r=1.8,
                    strategy_family="TREND",
                    confluence_score=4.0,
                    confluence_required=3.9,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "btc_strategy": "WHALE_FLOW", "btc_min_ai_confidence": float(self.btc_min_ai_confidence)},
                )
            )

        dxy_move = float(row.get("dxy_ret_5", row.get("dxy_ret_1", 0.0)))
        btc_short_move = float(row.get("m5_ret_1", 0.0))
        if abs(dxy_move) >= float(self.btc_dxy_move_threshold) and abs(btc_short_move) < abs(dxy_move):
            lag_side = "BUY" if dxy_move < 0 else "SELL"
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "btc-dxy-lag", lag_side, timestamp),
                    setup="BTC_DXY_LAG_ARB",
                    side=lag_side,
                    score_hint=0.66,
                    reason="BTC lag arb against DXY impulse",
                    stop_atr=0.8,
                    tp_r=1.5,
                    strategy_family="TREND",
                    confluence_score=4.0,
                    confluence_required=3.9,
                    meta={"timeframe": "M5", "atr_field": "m5_atr_14", "btc_strategy": "DXY_LAG_ARB", "btc_min_ai_confidence": float(self.btc_min_ai_confidence)},
                )
            )

        if (
            btc_trend_allowed
            and (session_window_allowed or weekend_mode or session_name in {"TOKYO", "LONDON"})
            and (not weekend_mode or session_name not in {"SYDNEY", "TOKYO"} or allow_weekend_asia_trend_candidates)
        ):
            trend_gap_ratio = abs(ema20 - ema50) / max(abs(close), 1e-6)
            btc_body_efficiency = float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)) or 0.55)
            london_weekday_scalp = session_name == "LONDON" and not weekend_mode
            tokyo_scalp = session_name == "TOKYO" and not weekend_mode
            trend_scalp_ready = (
                vol_ratio >= (1.18 if weekend_mode else 1.20 if london_weekday_scalp else 1.04)
                and btc_body_efficiency >= (0.70 if london_weekday_scalp else 0.60)
                and spread_ratio <= (1.12 if london_weekday_scalp else 1.35)
                and (0.18 if london_weekday_scalp else 0.10) <= impulse <= 0.72
                and trend_gap_ratio >= (0.0016 if london_weekday_scalp else 0.0012)
            )
            if tokyo_scalp:
                trend_scalp_ready = (
                    trend_scalp_ready
                    and regime_state == "BREAKOUT_EXPANSION"
                    and vol_ratio >= 1.16
                    and btc_body_efficiency >= 0.66
                    and spread_ratio <= 1.22
                    and 0.12 <= impulse <= 0.42
                    and trend_gap_ratio >= 0.00145
                )
            breakout_extension_cap = 0.04
            if session_name in {"OVERLAP", "NEW_YORK"}:
                breakout_extension_cap = 0.14
            elif london_weekday_scalp:
                breakout_extension_cap = 0.10
            elif tokyo_scalp:
                breakout_extension_cap = 0.05
            if (
                trend_scalp_ready
                and ema20 > ema50
                and close >= ema20
                and breakout_up
                and close <= (prev_high + (atr * breakout_extension_cap))
                and not (tokyo_scalp and range_position < 0.58)
            ):
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-trend", "BUY", timestamp),
                        setup="BTC_TREND_SCALP",
                        side="BUY",
                        score_hint=clamp(base_probability + 0.01, 0.55, 0.75),
                        reason="BTC momentum continuation during active liquidity window",
                        stop_atr=0.90 if weekend_mode else 1.00,
                        tp_r=1.5 if weekend_mode else 1.7,
                        strategy_family="TREND",
                        confluence_score=clamp(base_confluence, 3.0, 4.8),
                        confluence_required=3.2,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": approve_small_confluence,
                            "btc_strategy": "TREND_SCALP",
                            "btc_min_ai_confidence": float(self.btc_min_ai_confidence),
                            "weekend_mode": weekend_mode,
                            "spread_ratio": spread_ratio,
                            "btc_session_window_allowed": bool(session_window_allowed),
                        },
                    )
                )
            if (
                trend_scalp_ready
                and ema20 < ema50
                and close <= ema20
                and breakout_down
                and close >= (prev_low - (atr * breakout_extension_cap))
                and not (tokyo_scalp and range_position > 0.42)
            ):
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-trend", "SELL", timestamp),
                        setup="BTC_TREND_SCALP",
                        side="SELL",
                        score_hint=clamp(base_probability + 0.01, 0.55, 0.75),
                        reason="BTC momentum continuation during active liquidity window",
                        stop_atr=0.90 if weekend_mode else 1.00,
                        tp_r=1.5 if weekend_mode else 1.7,
                        strategy_family="TREND",
                        confluence_score=clamp(base_confluence, 3.0, 4.8),
                        confluence_required=3.2,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": approve_small_confluence,
                            "btc_strategy": "TREND_SCALP",
                            "btc_min_ai_confidence": float(self.btc_min_ai_confidence),
                            "weekend_mode": weekend_mode,
                            "spread_ratio": spread_ratio,
                            "btc_session_window_allowed": bool(session_window_allowed),
                        },
                    )
                )

        near_ema = abs(close - ema20) / max(abs(close), 1e-6) <= 0.0082
        if (
            btc_trend_allowed
            and near_ema
            and trend_gap_ratio >= 0.0010
            and session_name in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}
            and (not weekend_mode or session_name not in {"SYDNEY", "TOKYO"} or allow_weekend_asia_trend_candidates)
        ):
            if ema20 > ema50 and close >= ema20 * 0.998:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-trend-pullback", "BUY", timestamp),
                        setup="BTC_MOMENTUM_CONTINUATION",
                        side="BUY",
                        score_hint=clamp(base_probability - 0.01, 0.53, 0.71),
                        reason="BTC trend pullback continuation with neutral volume fallback",
                        stop_atr=0.95,
                        tp_r=1.45,
                        strategy_family="TREND",
                        confluence_score=clamp(base_confluence - 0.10, 3.0, 4.4),
                        confluence_required=3.0,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": approve_small_confluence,
                            "btc_strategy": "TREND_PULLBACK",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )
            if ema20 < ema50 and close <= ema20 * 1.002:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-trend-pullback", "SELL", timestamp),
                        setup="BTC_MOMENTUM_CONTINUATION",
                        side="SELL",
                        score_hint=clamp(base_probability - 0.01, 0.53, 0.71),
                        reason="BTC downside trend pullback continuation with neutral volume fallback",
                        stop_atr=0.95,
                        tp_r=1.45,
                        strategy_family="TREND",
                        confluence_score=clamp(base_confluence - 0.10, 3.0, 4.4),
                        confluence_required=3.0,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": approve_small_confluence,
                            "btc_strategy": "TREND_PULLBACK",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )
        failed_break_up = close <= prev_high and close >= (prev_high - (max(float(row.get("m5_atr_14", 0.0)), 1e-6) * 0.18)) and ema20 >= ema50
        failed_break_down = close >= prev_low and close <= (prev_low + (max(float(row.get("m5_atr_14", 0.0)), 1e-6) * 0.18)) and ema20 <= ema50
        if (
            btc_trend_allowed
            and session_name in {"TOKYO", "LONDON", "OVERLAP", "NEW_YORK", "SYDNEY"}
            and spread_ratio <= 1.55
            and (not weekend_mode or session_name not in {"SYDNEY", "TOKYO"} or allow_weekend_asia_trend_candidates)
        ):
            if failed_break_up and bullish:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-break-retest", "BUY", timestamp),
                        setup="BTC_NY_LIQUIDITY" if session_name in {"OVERLAP", "NEW_YORK"} else "BTC_TOKYO_DRIFT_SCALP" if session_name == "TOKYO" else "BTC_LONDON_IMPULSE_SCALP",
                        side="BUY",
                        score_hint=clamp(base_probability - 0.01, 0.54, 0.72),
                        reason="BTC failed breakdown and reclaim continuation",
                        stop_atr=0.88,
                        tp_r=1.45,
                        strategy_family="TREND",
                        confluence_score=clamp(base_confluence - 0.05, 3.0, 4.4),
                        confluence_required=3.0,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": approve_small_confluence,
                            "btc_strategy": "FAILED_BREAK_RECLAIM",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )
            if failed_break_down and bearish:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-break-retest", "SELL", timestamp),
                        setup="BTC_NY_LIQUIDITY" if session_name in {"OVERLAP", "NEW_YORK"} else "BTC_TOKYO_DRIFT_SCALP" if session_name == "TOKYO" else "BTC_LONDON_IMPULSE_SCALP",
                        side="SELL",
                        score_hint=clamp(base_probability - 0.01, 0.54, 0.72),
                        reason="BTC failed breakout and rejection continuation",
                        stop_atr=0.88,
                        tp_r=1.45,
                        strategy_family="TREND",
                        confluence_score=clamp(base_confluence - 0.05, 3.0, 4.4),
                        confluence_required=3.0,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": approve_small_confluence,
                            "btc_strategy": "FAILED_BREAK_RECLAIM",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )
        midpoint_pullback = 0.30 <= range_position <= 0.70 and abs(close - ema20) <= (atr * 1.25)
        if (not output) and btc_trend_allowed and midpoint_pullback and spread_ratio <= 1.55 and vol_ratio >= 0.72:
            if ema20 > ema50 and h1_ema20 > h1_ema50 and close >= ema20 * 0.996:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-midrange-fallback", "BUY", timestamp),
                        setup="BTC_TOKYO_DRIFT_SCALP" if session_name in {"SYDNEY", "TOKYO"} else "BTC_LONDON_IMPULSE_SCALP" if session_name == "LONDON" else "BTC_NY_LIQUIDITY",
                        side="BUY",
                        score_hint=0.54,
                        reason="BTC price-action fallback buy from mid-range pullback with aligned trend",
                        stop_atr=0.88,
                        tp_r=1.35,
                        strategy_family="TREND",
                        confluence_score=3.0,
                        confluence_required=2.9,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": approve_small_confluence,
                            "btc_strategy": "PRICE_ACTION_FALLBACK",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )
            if ema20 < ema50 and h1_ema20 < h1_ema50 and close <= ema20 * 1.004:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-midrange-fallback", "SELL", timestamp),
                        setup="BTC_TOKYO_DRIFT_SCALP" if session_name in {"SYDNEY", "TOKYO"} else "BTC_LONDON_IMPULSE_SCALP" if session_name == "LONDON" else "BTC_NY_LIQUIDITY",
                        side="SELL",
                        score_hint=0.54,
                        reason="BTC price-action fallback sell from mid-range pullback with aligned trend",
                        stop_atr=0.88,
                        tp_r=1.35,
                        strategy_family="TREND",
                        confluence_score=3.0,
                        confluence_required=2.9,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": approve_small_confluence,
                            "btc_strategy": "PRICE_ACTION_FALLBACK",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                )
            )
        if (
            (not output)
            and proxyless_price_action_mode
            and session_name in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}
            and spread_ratio <= (1.82 if session_name in {"SYDNEY", "TOKYO"} else 1.72)
            and 0.58 <= atr_ratio <= 2.75
            and instability_score <= 0.44
            and seasonality_score >= 0.32
        ):
            proxyless_setup = (
                "BTC_PRICE_ACTION_CONTINUATION"
                if session_name in {"SYDNEY", "TOKYO", "LONDON"}
                else "BTC_NY_LIQUIDITY"
            )
            proxyless_buy = bool(
                ema20 >= ema50
                and h1_ema20 >= h1_ema50
                and close >= (ema20 - (atr * 0.62))
                and range_position <= 0.72
                and (bullish or momentum >= -0.00055 or low_reclaim or rsi <= 58)
                and (vol_ratio >= 0.56 or body_efficiency >= 0.42 or alignment_score >= 0.60)
            )
            proxyless_sell = bool(
                ema20 <= ema50
                and h1_ema20 <= h1_ema50
                and close <= (ema20 + (atr * 0.62))
                and range_position >= 0.28
                and (bearish or momentum <= 0.00055 or high_reject or rsi >= 42)
                and (vol_ratio >= 0.56 or body_efficiency >= 0.42 or alignment_score >= 0.60)
            )
            proxyless_confluence = clamp(
                2.96
                + min(0.22, max(0.0, vol_ratio - 0.56))
                + min(0.18, max(0.0, body_efficiency - 0.40))
                + min(0.18, max(0.0, alignment_score - 0.56)),
                0.0,
                5.0,
            )
            if proxyless_buy:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-proxyless-price-action", "BUY", timestamp),
                        setup=proxyless_setup,
                        side="BUY",
                        score_hint=0.55,
                        reason="BTC proxyless price-action continuation with aligned short-term structure",
                        stop_atr=0.90,
                        tp_r=1.38,
                        strategy_family="TREND",
                        confluence_score=proxyless_confluence,
                        confluence_required=2.9,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": 2.9,
                            "btc_strategy": "PRICE_ACTION_FALLBACK",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.49,
                            "proxyless_price_action_mode": True,
                        },
                    )
                )
            if proxyless_sell:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-proxyless-price-action", "SELL", timestamp),
                        setup=proxyless_setup,
                        side="SELL",
                        score_hint=0.55,
                        reason="BTC proxyless downside continuation with aligned short-term structure",
                        stop_atr=0.90,
                        tp_r=1.38,
                        strategy_family="TREND",
                        confluence_score=proxyless_confluence,
                        confluence_required=2.9,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": 2.9,
                            "btc_strategy": "PRICE_ACTION_FALLBACK",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.49,
                            "proxyless_price_action_mode": True,
                        },
                    )
                )
        trend_bias = ema20 - ema50
        session_range_edge_allowed = session_name in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}
        if (not output) and session_range_edge_allowed and spread_ratio <= 1.75 and 0.65 <= atr_ratio <= 2.55:
            fallback_setup = (
                "BTC_TOKYO_DRIFT_SCALP"
                if session_name in {"SYDNEY", "TOKYO"}
                else "BTC_LONDON_IMPULSE_SCALP"
                if session_name == "LONDON"
                else "BTC_NY_LIQUIDITY"
            )
            if (
                range_position <= 0.34
                and trend_bias >= -(atr * 0.22)
                and close >= ema20 * 0.992
                and (bullish or momentum >= -0.00035 or low_reclaim or rsi <= 45)
            ):
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-range-edge-fallback", "BUY", timestamp),
                        setup=fallback_setup,
                        side="BUY",
                        score_hint=0.56,
                        reason="BTC range-edge fallback buy with aligned short-term structure",
                        stop_atr=0.86,
                        tp_r=1.30,
                        strategy_family="SCALP",
                        confluence_score=3.05,
                        confluence_required=2.9,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": 2.9,
                            "btc_strategy": "RANGE_EDGE_FALLBACK",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )
            if (
                range_position >= 0.66
                and trend_bias <= (atr * 0.22)
                and close <= ema20 * 1.008
                and (bearish or momentum <= 0.00035 or high_reject or rsi >= 55)
            ):
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-range-edge-fallback", "SELL", timestamp),
                        setup=fallback_setup,
                        side="SELL",
                        score_hint=0.56,
                        reason="BTC range-edge fallback sell with aligned short-term structure",
                        stop_atr=0.86,
                        tp_r=1.30,
                        strategy_family="SCALP",
                        confluence_score=3.05,
                        confluence_required=2.9,
                        meta={
                            "timeframe": "M15",
                            "atr_field": "m5_atr_14",
                            "allow_ai_approve_small": True,
                            "approve_small_min_probability": approve_small_probability,
                            "approve_small_min_confluence": 2.9,
                            "btc_strategy": "RANGE_EDGE_FALLBACK",
                            "setup_family": "PRICE_ACTION",
                            "btc_min_ai_confidence": 0.50,
                        },
                    )
                )
        if (
            (not output)
            and proxyless_price_action_mode
            and weekend_mode
            and session_name in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}
            and spread_ratio <= (2.10 if session_name in {"SYDNEY", "TOKYO"} else 1.92)
            and 0.50 <= atr_ratio <= 2.95
            and instability_score <= 0.80
            and seasonality_score >= 0.18
            and 0.26 <= range_position <= 0.74
        ):
            spray_setup = (
                "BTC_TOKYO_DRIFT_SCALP"
                if session_name in {"SYDNEY", "TOKYO"}
                else "BTC_LONDON_IMPULSE_SCALP"
                if session_name == "LONDON"
                else "BTC_NY_LIQUIDITY"
            )
            spray_buy = bool(
                ema20 >= ema50
                and close >= (ema20 - (atr * 0.95))
                and (
                    bullish
                    or momentum >= -(impulse_floor * 0.40)
                    or close >= ema20
                    or range_position <= 0.64
                    or vol_ratio >= 0.78
                )
            )
            spray_sell = bool(
                ema20 <= ema50
                and close <= (ema20 + (atr * 0.95))
                and (
                    bearish
                    or momentum <= (impulse_floor * 0.40)
                    or close <= ema20
                    or range_position >= 0.36
                    or vol_ratio >= 0.78
                )
            )
            spray_confluence = clamp(
                2.98
                + min(0.20, max(0.0, vol_ratio - 0.55))
                + min(0.18, max(0.0, body_efficiency - 0.38))
                + min(0.16, max(0.0, alignment_score - 0.52))
                + (0.10 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.04),
                0.0,
                5.0,
            )
            spray_meta = {
                "timeframe": "M15",
                "atr_field": "m5_atr_14",
                "allow_ai_approve_small": True,
                "approve_small_min_probability": max(0.49, approve_small_probability - 0.01),
                "approve_small_min_confluence": 2.8,
                "btc_strategy": "WEEKEND_PROXYLESS_SPRAY",
                "setup_family": "PRICE_ACTION",
                "btc_min_ai_confidence": 0.48,
                "proxyless_price_action_mode": True,
                "proxyless_weekend_spray_mode": True,
            }
            if spray_buy:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-weekend-proxyless-spray", "BUY", timestamp),
                        setup=spray_setup,
                        side="BUY",
                        score_hint=0.56 if session_name in {"SYDNEY", "TOKYO"} else 0.58,
                        reason="BTC weekend proxyless spray buy with executable local structure",
                        stop_atr=0.82,
                        tp_r=1.28,
                        strategy_family="SCALP",
                        confluence_score=spray_confluence,
                        confluence_required=2.8,
                        meta=dict(spray_meta),
                    )
                )
            if spray_sell:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "btc-weekend-proxyless-spray", "SELL", timestamp),
                        setup=spray_setup,
                        side="SELL",
                        score_hint=0.56 if session_name in {"SYDNEY", "TOKYO"} else 0.58,
                        reason="BTC weekend proxyless spray sell with executable local structure",
                        stop_atr=0.82,
                        tp_r=1.28,
                        strategy_family="SCALP",
                        confluence_score=spray_confluence,
                        confluence_required=2.8,
                        meta=dict(spray_meta),
                    )
                )
        return output


    def _asia_momentum_breakout(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        symbol_key = self._normalize_symbol(symbol)
        if symbol_key not in {"AUDJPY", "NZDJPY"}:
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"SYDNEY", "TOKYO"}:
            return []
        if "TREND" not in session.allowed_strategies:
            return []

        learning_state = self._learning_pattern_state(symbol=symbol_key, session_name=session_name)
        throughput_recovery = bool(learning_state.get("throughput_recovery_active", False))
        catchup_pressure = clamp(float(learning_state.get("trajectory_catchup_pressure", 0.0) or 0.0), 0.0, 1.0)
        spread = float(row.get("m5_spread", 0.0))
        spread_multiplier = 0.92 + (0.06 if throughput_recovery else 0.0) + (0.02 if catchup_pressure >= 0.85 else 0.0)
        if spread > self._spread_limit(symbol_key, multiplier=spread_multiplier):
            return []
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        prev_high = float(row.get("m15_rolling_high_prev_20", close + atr))
        prev_low = float(row.get("m15_rolling_low_prev_20", close - atr))
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        trend_flag = bool(regime.details.get("trend_flag", regime.label == "TRENDING"))
        if not trend_flag:
            return []

        breakout_margin = (0.12 if symbol_key == "AUDJPY" else 0.15) - (0.02 if throughput_recovery else 0.0)
        volume_floor = (0.96 if symbol_key == "AUDJPY" else 1.00) - (0.06 if throughput_recovery else 0.0)
        ret_1 = float(row.get("m15_ret_1", row.get("m5_ret_1", 0.0)))
        bullish = int(row.get("m5_bullish", 0)) == 1
        bearish = int(row.get("m5_bearish", 0)) == 1
        impulse_up = close > prev_high and (close - prev_high) >= (atr * breakout_margin) and ret_1 > 0
        impulse_down = close < prev_low and (prev_low - close) >= (atr * breakout_margin) and ret_1 < 0
        if volume_ratio < volume_floor:
            return []
        body_efficiency = float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)) or 0.55)
        upper_wick = float(row.get("m5_upper_wick_ratio", 0.0) or 0.0)
        lower_wick = float(row.get("m5_lower_wick_ratio", 0.0) or 0.0)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        atr_ratio = float(row.get("m15_atr_pct_of_avg", row.get("m5_atr_pct_of_avg", 1.0)) or 1.0)
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        if regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION", "RANGING"}:
            return []
        min_body_efficiency = (0.56 if symbol_key == "AUDJPY" else 0.60) - (0.04 if throughput_recovery else 0.0)
        if body_efficiency < min_body_efficiency:
            return []
        min_atr_ratio = (0.84 if symbol_key == "AUDJPY" else 0.88) - (0.06 if throughput_recovery else 0.0)
        if atr_ratio < min_atr_ratio:
            return []
        max_extension = (0.56 if symbol_key == "AUDJPY" else 0.50) + (0.06 if throughput_recovery else 0.0)
        if impulse_up and (close - prev_high) >= (atr * max_extension):
            return []
        if impulse_down and (prev_low - close) >= (atr * max_extension):
            return []
        wick_limit = (0.28 if symbol_key == "AUDJPY" else 0.24) + (0.04 if throughput_recovery else 0.0)
        if impulse_up and upper_wick > wick_limit:
            return []
        if impulse_down and lower_wick > wick_limit:
            return []
        ret_3 = abs(float(row.get("m15_ret_3", row.get("m5_ret_3", 0.0)) or 0.0))
        if ret_3 > (atr * ((0.32 if symbol_key == "AUDJPY" else 0.28) + (0.03 if throughput_recovery else 0.0))):
            return []

        confluence = clamp(3.85 + min(0.7, max(0.0, volume_ratio - volume_floor)) + (0.30 if session_name == "TOKYO" else 0.18), 0.0, 5.0)
        score_hint = 0.69 if symbol_key == "AUDJPY" else 0.67
        confluence_required = 3.8 - (0.18 if throughput_recovery else 0.0)
        output: list[SignalCandidate] = []
        if impulse_up and bullish and range_position >= 0.60:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "asia-momentum-breakout", "BUY", timestamp),
                    setup=f"{symbol_key}_TOKYO_MOMENTUM_BREAKOUT",
                    side="BUY",
                    score_hint=score_hint,
                    reason=f"{symbol_key} Asia breakout impulse continuation",
                    stop_atr=0.95,
                    tp_r=1.9,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=confluence,
                    confluence_required=confluence_required,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "TOKYO_BREAKOUT"},
                )
            )
        if impulse_down and bearish and range_position <= 0.40:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "asia-momentum-breakout", "SELL", timestamp),
                    setup=f"{symbol_key}_TOKYO_MOMENTUM_BREAKOUT",
                    side="SELL",
                    score_hint=score_hint,
                    reason=f"{symbol_key} Asia downside breakout impulse continuation",
                    stop_atr=0.95,
                    tp_r=1.9,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=confluence,
                    confluence_required=confluence_required,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "TOKYO_BREAKOUT"},
                )
            )
        return output

    def _asia_continuation_pullback(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        symbol_key = self._normalize_symbol(symbol)
        if symbol_key not in {"AUDJPY", "NZDJPY"}:
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"SYDNEY", "TOKYO"}:
            return []
        if "TREND" not in session.allowed_strategies:
            return []

        learning_state = self._learning_pattern_state(symbol=symbol_key, session_name=session_name)
        throughput_recovery = bool(learning_state.get("throughput_recovery_active", False))
        catchup_pressure = clamp(float(learning_state.get("trajectory_catchup_pressure", 0.0) or 0.0), 0.0, 1.0)
        spread = float(row.get("m5_spread", 0.0))
        spread_multiplier = 0.90 + (0.08 if throughput_recovery else 0.0) + (0.02 if catchup_pressure >= 0.85 else 0.0)
        if spread > self._spread_limit(symbol_key, multiplier=spread_multiplier):
            return []
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        ema20 = float(row.get("m15_ema_20", row.get("m5_ema_20", close)))
        ema50 = float(row.get("m15_ema_50", row.get("m5_ema_50", close)))
        h1_ema20 = float(row.get("h1_ema_20", ema20))
        h1_ema50 = float(row.get("h1_ema_50", ema50))
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        body_efficiency = self._body_efficiency_value(row)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        lower_wick = float(row.get("m5_lower_wick_ratio", 0.0) or 0.0)
        upper_wick = float(row.get("m5_upper_wick_ratio", 0.0) or 0.0)
        atr_ratio = float(row.get("m15_atr_pct_of_avg", row.get("m5_atr_pct_of_avg", 1.0)) or 1.0)
        trend_flag = bool(regime.details.get("trend_flag", regime.label == "TRENDING"))
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        if regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
            return []
        min_volume = (0.94 if symbol_key == "AUDJPY" else 0.98) - (0.08 if throughput_recovery else 0.0)
        max_distance_mult = (0.34 if symbol_key == "AUDJPY" else 0.30) + (0.08 if throughput_recovery else 0.0)
        min_body_efficiency = (0.54 if symbol_key == "AUDJPY" else 0.58) - (0.05 if throughput_recovery else 0.0)
        if not trend_flag or volume_ratio < min_volume or body_efficiency < min_body_efficiency:
            return []
        min_atr_ratio = (0.84 if symbol_key == "AUDJPY" else 0.88) - (0.06 if throughput_recovery else 0.0)
        if atr_ratio < min_atr_ratio:
            return []
        distance_to_ema = abs(close - ema20)
        pullback_ratio = distance_to_ema / max(atr, 1e-6)
        if pullback_ratio < 0.12 or distance_to_ema > (atr * max_distance_mult):
            return []
        reclaim_ret = float(row.get("m15_ret_1", row.get("m5_ret_1", 0.0)) or 0.0)
        reclaim_ret3 = float(row.get("m15_ret_3", row.get("m5_ret_3", 0.0)) or 0.0)
        ret_present = any(
            key in row.index and row.get(key) is not None and not pd.isna(row.get(key))
            for key in ("m15_ret_1", "m5_ret_1", "m15_ret_3", "m5_ret_3")
        )
        clean_reclaim_bias = body_efficiency >= min_body_efficiency and volume_ratio >= min_volume
        if ret_present:
            if abs(reclaim_ret) < (atr * (0.03 if symbol_key == "AUDJPY" else 0.04)):
                return []
            if abs(reclaim_ret3) >= (atr * ((0.28 if symbol_key == "AUDJPY" else 0.25) + (0.02 if throughput_recovery else 0.0))):
                return []
            if abs(reclaim_ret) > (atr * ((0.22 if symbol_key == "AUDJPY" else 0.20) + (0.02 if throughput_recovery else 0.0))):
                return []
            if reclaim_ret != 0.0 and reclaim_ret3 != 0.0 and np.sign(reclaim_ret) != np.sign(reclaim_ret3):
                return []
        elif not clean_reclaim_bias:
            return []
        if reclaim_ret > 0 and lower_wick < (0.15 if symbol_key == "AUDJPY" else 0.17):
            return []
        if reclaim_ret < 0 and upper_wick < (0.15 if symbol_key == "AUDJPY" else 0.17):
            return []
        bullish_candle = int(row.get("m5_bullish", row.get("m15_bullish", 0))) == 1
        bearish_candle = int(row.get("m5_bearish", row.get("m15_bearish", 0))) == 1
        bullish_confirmation = bullish_candle or (not ret_present and clean_reclaim_bias and body_efficiency >= (min_body_efficiency + 0.01))
        bearish_confirmation = bearish_candle or (not ret_present and clean_reclaim_bias and body_efficiency >= (min_body_efficiency + 0.01))
        trend_gap_ratio = abs(ema20 - ema50) / max(atr, 1e-6)
        h1_gap_ratio = abs(h1_ema20 - h1_ema50) / max(atr, 1e-6)
        if trend_gap_ratio < 0.12 or h1_gap_ratio < 0.12:
            return []

        bullish = (
            ema20 >= ema50
            and h1_ema20 >= h1_ema50
            and close >= ema20
            and close >= (ema20 - (atr * 0.04))
            and range_position <= (0.64 if symbol_key == "AUDJPY" else 0.62)
            and bullish_confirmation
            and (reclaim_ret > 0 or (not ret_present and clean_reclaim_bias))
        )
        bearish = (
            ema20 <= ema50
            and h1_ema20 <= h1_ema50
            and close <= ema20
            and close <= (ema20 + (atr * 0.04))
            and range_position >= (0.36 if symbol_key == "AUDJPY" else 0.38)
            and bearish_confirmation
            and (reclaim_ret < 0 or (not ret_present and clean_reclaim_bias))
        )
        confluence = clamp(
            3.75
            + min(0.55, max(0.0, volume_ratio - 0.98))
            + (0.12 if body_efficiency >= 0.66 else 0.0),
            0.0,
            5.0,
        )
        score_hint = 0.66 if symbol_key == "AUDJPY" else 0.64
        confluence_required = 3.5 - (0.15 if throughput_recovery else 0.0)
        output: list[SignalCandidate] = []
        if bullish:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "asia-pullback", "BUY", timestamp),
                    setup=f"{symbol_key}_TOKYO_CONTINUATION_PULLBACK",
                    side="BUY",
                    score_hint=score_hint,
                    reason=f"{symbol_key} Asia pullback continuation",
                    stop_atr=0.88,
                    tp_r=1.8,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=confluence,
                    confluence_required=confluence_required,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "TOKYO_PULLBACK"},
                )
            )
        if bearish:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "asia-pullback", "SELL", timestamp),
                    setup=f"{symbol_key}_TOKYO_CONTINUATION_PULLBACK",
                    side="SELL",
                    score_hint=score_hint,
                    reason=f"{symbol_key} Asia downside pullback continuation",
                    stop_atr=0.88,
                    tp_r=1.8,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=confluence,
                    confluence_required=confluence_required,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "TOKYO_PULLBACK"},
                )
            )
        return output

    def _asia_rotation_reclaim(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        symbol_key = self._normalize_symbol(symbol)
        if symbol_key not in {"AUDJPY", "NZDJPY"}:
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"SYDNEY", "TOKYO"}:
            return []
        spread = float(row.get("m5_spread", 0.0))
        spread_cap = self._spread_limit(symbol_key, multiplier=(0.92 if symbol_key == "AUDJPY" else 0.88))
        if spread > spread_cap:
            return []
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        if regime_state == "NEWS_VOLATILE":
            return []
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        high = float(row.get("m15_high", row.get("m5_high", close)))
        low = float(row.get("m15_low", row.get("m5_low", close)))
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        atr_ratio = float(row.get("m15_atr_pct_of_avg", row.get("m5_atr_pct_of_avg", 1.0)))
        if atr_ratio < 0.66 or atr_ratio > 1.72:
            return []
        prev_high = float(row.get("m15_rolling_high_prev_20", close + atr))
        prev_low = float(row.get("m15_rolling_low_prev_20", close - atr))
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        body_efficiency = max(
            self._body_efficiency_value(row),
            float(row.get("m15_body_efficiency", row.get("m15_candle_efficiency", 0.0)) or 0.0),
        )
        rsi = float(row.get("m5_rsi_14", 50.0))
        bullish_candle = int(row.get("m5_bullish", row.get("m15_bullish", 0))) == 1
        bearish_candle = int(row.get("m5_bearish", row.get("m15_bearish", 0))) == 1
        bull_rejection = bullish_candle or int(row.get("m5_pinbar_bull", 0)) == 1 or int(row.get("m5_engulf_bull", 0)) == 1
        bear_rejection = bearish_candle or int(row.get("m5_pinbar_bear", 0)) == 1 or int(row.get("m5_engulf_bear", 0)) == 1
        extreme_rejection = (
            (bull_rejection and range_position <= (0.22 if symbol_key == "AUDJPY" else 0.18))
            or (bear_rejection and range_position >= (0.78 if symbol_key == "AUDJPY" else 0.82))
        )
        volume_floor = (0.76 if symbol_key == "AUDJPY" else 0.82) if regime_state == "MEAN_REVERSION" and extreme_rejection else (0.80 if symbol_key == "AUDJPY" else 0.86)
        body_floor = (0.44 if symbol_key == "AUDJPY" else 0.48) if regime_state == "MEAN_REVERSION" and extreme_rejection else (0.50 if symbol_key == "AUDJPY" else 0.54)
        if volume_ratio < volume_floor or body_efficiency < body_floor:
            return []
        if session_name in {"SYDNEY", "TOKYO"} and regime_state == "MEAN_REVERSION":
            mean_reversion_volume_floor = (0.92 if symbol_key == "AUDJPY" else 0.96) if extreme_rejection else (0.94 if symbol_key == "AUDJPY" else 0.98)
            mean_reversion_body_floor = (0.44 if symbol_key == "AUDJPY" else 0.48) if extreme_rejection else (0.62 if symbol_key == "AUDJPY" else 0.65)
            if volume_ratio < mean_reversion_volume_floor or body_efficiency < mean_reversion_body_floor:
                return []
        if session_name in {"SYDNEY", "TOKYO"} and regime_state == "LOW_LIQUIDITY_CHOP":
            if volume_ratio < (1.00 if symbol_key == "AUDJPY" else 1.04) or body_efficiency < (0.66 if symbol_key == "AUDJPY" else 0.69):
                return []

        sweep_reclaim_buy = (
            (low < prev_low and close > (prev_low - (atr * 0.02)) and bull_rejection)
            or (
                range_position <= (0.24 if symbol_key == "AUDJPY" else 0.20)
                and close >= (prev_low + (atr * 0.04))
                and bull_rejection
                and rsi <= (49 if symbol_key == "AUDJPY" else 47)
            )
        )
        sweep_reclaim_sell = (
            (high > prev_high and close < (prev_high + (atr * 0.02)) and bear_rejection)
            or (
                range_position >= (0.76 if symbol_key == "AUDJPY" else 0.80)
                and close <= (prev_high - (atr * 0.04))
                and bear_rejection
                and rsi >= (51 if symbol_key == "AUDJPY" else 53)
            )
        )
        if session_name in {"SYDNEY", "TOKYO"} and regime_state == "LOW_LIQUIDITY_CHOP":
            if sweep_reclaim_buy and range_position > (0.20 if symbol_key == "AUDJPY" else 0.18):
                sweep_reclaim_buy = False
            if sweep_reclaim_sell and range_position < (0.80 if symbol_key == "AUDJPY" else 0.82):
                sweep_reclaim_sell = False
        confluence = clamp(
            3.45
            + min(0.45, max(0.0, volume_ratio - volume_floor))
            + (0.15 if body_efficiency >= (body_floor + 0.08) else 0.0)
            + (0.12 if session_name in {"SYDNEY", "TOKYO"} else 0.05),
            0.0,
            5.0,
        )
        score_hint = 0.65 if symbol_key == "AUDJPY" else 0.63
        output: list[SignalCandidate] = []
        if sweep_reclaim_buy:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "asia-rotation-reclaim", "BUY", timestamp),
                    setup=f"{symbol_key}_SWEEP_RECLAIM",
                    side="BUY",
                    score_hint=score_hint,
                    reason=f"{symbol_key} Asia range sweep reclaim with structured reversal context",
                    stop_atr=0.82 if symbol_key == "AUDJPY" else 0.86,
                    tp_r=1.60 if symbol_key == "AUDJPY" else 1.55,
                    entry_kind="SCALP",
                    strategy_family="FAKEOUT",
                    confluence_score=confluence,
                    confluence_required=3.3,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "ASIA_RECLAIM"},
                )
            )
        if sweep_reclaim_sell:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "asia-rotation-reclaim", "SELL", timestamp),
                    setup=f"{symbol_key}_SWEEP_RECLAIM",
                    side="SELL",
                    score_hint=score_hint,
                    reason=f"{symbol_key} Asia range sweep rejection with structured reversal context",
                    stop_atr=0.82 if symbol_key == "AUDJPY" else 0.86,
                    tp_r=1.60 if symbol_key == "AUDJPY" else 1.55,
                    entry_kind="SCALP",
                    strategy_family="FAKEOUT",
                    confluence_score=confluence,
                    confluence_required=3.3,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "ASIA_RECLAIM"},
                )
            )
        return output

    def _asia_drift_continuation(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        symbol_key = self._normalize_symbol(symbol)
        if symbol_key not in {"AUDJPY", "NZDJPY", "AUDNZD"}:
            return []
        session_name = str(session.session_name).upper()
        raw_regime_label = str(getattr(regime, "state_label", regime.label) or regime.label).strip().upper()
        learning_state = self._learning_pattern_state(symbol=symbol_key, session_name=session_name)
        throughput_recovery = bool(learning_state.get("throughput_recovery_active", False))
        if not self._drift_continuation_ready(
            symbol=symbol_key,
            row=row,
            session_name=session_name,
            raw_regime_label=raw_regime_label,
        ):
            return []
        close = float(row.get("m15_close", row.get("m5_close", 0.0)) or 0.0)
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0)) or 0.0), 1e-6)
        ema20 = float(row.get("m15_ema_20", row.get("m5_ema_20", close)) or close)
        ema50 = float(row.get("m15_ema_50", row.get("m5_ema_50", close)) or close)
        h1_ema_50 = float(row.get("h1_ema_50", ema50) or ema50)
        h1_ema_200 = float(row.get("h1_ema_200", h1_ema_50) or h1_ema_50)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        body_efficiency = max(
            self._body_efficiency_value(row),
            float(row.get("m15_body_efficiency", row.get("m15_candle_efficiency", 0.0)) or 0.0),
        )
        ret_1 = float(row.get("m15_ret_1", row.get("m5_ret_1", 0.0)) or 0.0)
        ret_3 = float(row.get("m15_ret_3", row.get("m5_ret_3", 0.0)) or 0.0)
        near_ema = abs(close - ema20) <= (
            atr * ((0.30 if symbol_key in {"AUDJPY", "NZDJPY"} else 0.28) + (0.06 if throughput_recovery else 0.0))
        )
        bullish = ema20 >= ema50 and h1_ema_50 >= h1_ema_200 and range_position >= 0.52 and ret_1 >= 0.0
        bearish = ema20 <= ema50 and h1_ema_50 <= h1_ema_200 and range_position <= 0.48 and ret_1 <= 0.0
        trend_gap_ratio = abs(ema20 - ema50) / max(atr, 1e-6)
        bullish = bool(
            bullish
            or (
                symbol_key == "AUDNZD"
                and h1_ema_50 >= h1_ema_200
                and range_position >= 0.80
                and ret_1 > 0.0
                and ret_3 > 0.0
                and close >= ema20
                and trend_gap_ratio <= 0.30
            )
        )
        bearish = bool(
            bearish
            or (
                symbol_key == "AUDNZD"
                and h1_ema_50 <= h1_ema_200
                and range_position <= 0.20
                and ret_1 < 0.0
                and ret_3 < 0.0
                and close <= ema20
                and trend_gap_ratio <= 0.30
            )
        )
        if not (bullish or bearish):
            return []
        if symbol_key == "AUDJPY":
            momentum_breakout_ready = bool(
                not near_ema
                and trend_gap_ratio >= 0.22
                and volume_ratio >= 1.02
                and body_efficiency >= 0.48
                and abs(ret_3) >= 0.014
                and ((bullish and range_position >= 0.66) or (bearish and range_position <= 0.34))
            )
            setup = "AUDJPY_ASIA_MOMENTUM_BREAKOUT" if momentum_breakout_ready else "AUDJPY_TOKYO_CONTINUATION_PULLBACK"
            score_hint = 0.65
            tp_r = 1.85
        elif symbol_key == "NZDJPY":
            momentum_breakout_ready = bool(
                not near_ema
                and trend_gap_ratio >= 0.20
                and volume_ratio >= 0.98
                and body_efficiency >= 0.46
                and abs(ret_3) >= 0.013
                and ((bullish and range_position >= 0.66) or (bearish and range_position <= 0.34))
            )
            setup = "NZDJPY_ASIA_MOMENTUM_BREAKOUT" if momentum_breakout_ready else "NZDJPY_TOKYO_CONTINUATION_PULLBACK"
            score_hint = 0.64
            tp_r = 1.90
        else:
            momentum_breakout_ready = bool(
                not near_ema
                and trend_gap_ratio >= 0.16
                and volume_ratio >= 0.92
                and body_efficiency >= 0.42
                and abs(ret_3) >= 0.010
                and ((bullish and range_position >= 0.64) or (bearish and range_position <= 0.36))
            )
            setup = "AUDNZD_COMPRESSION_RELEASE" if momentum_breakout_ready else "AUDNZD_ROTATION_PULLBACK"
            score_hint = 0.63
            tp_r = 1.95
        drift_meta = {
            "timeframe": "M15",
            "atr_field": "m15_atr_14",
            "session_drift_lane": True,
            "structure_cleanliness_floor": 0.54,
            "entry_timing_floor": 0.56,
            "setup_family": "DRIFT_CONTINUATION",
        }
        confluence_required = (2.98 if symbol_key == "AUDNZD" else 2.92) - (0.12 if throughput_recovery else 0.0)
        confluence = clamp(
            3.45
            + min(0.45, max(0.0, volume_ratio - 0.82))
            + min(0.25, max(0.0, body_efficiency - 0.40)),
            0.0,
            5.0,
        )
        output: list[SignalCandidate] = []
        if bullish:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "asia-drift", "BUY", timestamp),
                    setup=setup,
                    side="BUY",
                    score_hint=score_hint,
                    reason=f"{symbol_key} drift continuation with aligned session structure",
                    stop_atr=0.90,
                    tp_r=tp_r,
                    entry_kind="SCALP" if session_name in {"SYDNEY", "TOKYO"} else "DAYTRADE",
                    strategy_family="TREND",
                    confluence_score=confluence,
                    confluence_required=confluence_required,
                    meta=dict(drift_meta),
                )
            )
        if bearish:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "asia-drift", "SELL", timestamp),
                    setup=setup,
                    side="SELL",
                    score_hint=score_hint,
                    reason=f"{symbol_key} downside drift continuation with aligned session structure",
                    stop_atr=0.90,
                    tp_r=tp_r,
                    entry_kind="SCALP" if session_name in {"SYDNEY", "TOKYO"} else "DAYTRADE",
                    strategy_family="TREND",
                    confluence_score=confluence,
                    confluence_required=confluence_required,
                    meta=dict(drift_meta),
                )
            )
        return output

    def _session_drift_scalp(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        symbol_key = self._normalize_symbol(symbol)
        session_name = str(session.session_name).upper()
        allowed_sessions: dict[str, set[str]] = {
            "GBPUSD": {"LONDON", "OVERLAP", "NEW_YORK"},
            "EURUSD": {"LONDON", "OVERLAP", "NEW_YORK"},
            "USDJPY": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "AUDJPY": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "NZDJPY": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "AUDNZD": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
        }
        if session_name not in allowed_sessions.get(symbol_key, set()):
            return []

        raw_regime_label = str(getattr(regime, "state_label", regime.label) or regime.label).strip().upper()
        if not self._drift_continuation_ready(
            symbol=symbol_key,
            row=row,
            session_name=session_name,
            raw_regime_label=raw_regime_label,
        ):
            return []

        close = float(row.get("m15_close", row.get("m5_close", 0.0)) or 0.0)
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0)) or 0.0), 1e-6)
        spread = float(row.get("m5_spread", 0.0) or 0.0)
        if spread > self._spread_limit(symbol_key, multiplier=(1.05 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.95)):
            return []

        ema20 = float(row.get("m15_ema_20", row.get("m5_ema_20", close)) or close)
        ema50 = float(row.get("m15_ema_50", row.get("m5_ema_50", close)) or close)
        h1_ema_50 = float(row.get("h1_ema_50", ema50) or ema50)
        h1_ema_200 = float(row.get("h1_ema_200", h1_ema_50) or h1_ema_50)
        volume_ratio = self._ratio_or_neutral(row.get("m5_volume_ratio_20", row.get("m15_volume_ratio_20", 1.0)))
        body_efficiency = float(row.get("m15_body_efficiency", row.get("m15_candle_efficiency", self._body_efficiency_value(row))) or self._body_efficiency_value(row))
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        alignment_score = float(row.get("multi_tf_alignment_score", 0.5) or 0.5)
        fractal_score = float(row.get("fractal_persistence_score", 0.5) or 0.5)
        seasonality_score = float(row.get("seasonality_edge_score", 0.5) or 0.5)
        instability_score = float(row.get("market_instability_score", row.get("feature_drift_score", 0.0)) or 0.0)
        distance_to_ema_ratio = abs(close - ema20) / max(atr, 1e-6)
        trend_gap_ratio = abs(ema20 - ema50) / max(atr, 1e-6)
        ret_1 = float(row.get("m15_ret_1", row.get("m5_ret_1", 0.0)) or 0.0)
        ret_3 = float(row.get("m15_ret_3", row.get("m5_ret_3", 0.0)) or 0.0)
        bullish_candle = self._bullish_flag(row, primary="m15", fallback="m5")
        bearish_candle = self._bearish_flag(row, primary="m15", fallback="m5")

        if body_efficiency < 0.38 or volume_ratio < 0.82:
            return []
        if symbol_key == "NAS100" and (
            volume_ratio < 0.92
            or body_efficiency < 0.48
            or seasonality_score < 0.22
            or alignment_score < 0.56
            or fractal_score < 0.54
            or instability_score > 0.18
        ):
            return []
        max_distance_ratio = 4.25 if symbol_key in {"GBPUSD", "EURUSD", "USDJPY"} else 1.35
        if distance_to_ema_ratio > max_distance_ratio or trend_gap_ratio < 0.04:
            return []
        if seasonality_score < 0.16 or alignment_score < 0.74 or fractal_score < 0.70:
            return []
        if instability_score > 0.24:
            return []

        bullish_bias = ema20 >= ema50 and h1_ema_50 >= h1_ema_200 and range_position >= 0.50 and (ret_1 >= 0.0 or ret_3 >= 0.0 or bullish_candle)
        bearish_bias = ema20 <= ema50 and h1_ema_50 <= h1_ema_200 and range_position <= 0.50 and (ret_1 <= 0.0 or ret_3 <= 0.0 or bearish_candle)
        if not bullish_bias and not bearish_bias:
            return []

        session_boost = 0.22 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.10
        confluence = clamp(
            3.05
            + session_boost
            + min(0.45, max(0.0, volume_ratio - 0.82))
            + min(0.35, max(0.0, body_efficiency - 0.38))
            + min(0.25, max(0.0, alignment_score - 0.74))
            + min(0.20, max(0.0, fractal_score - 0.70)),
            0.0,
            5.0,
        )
        score_hint = clamp(0.57 + (0.04 * (confluence - 3.0)), 0.55, 0.76)
        tp_r = 1.90 if symbol_key == "NAS100" else (1.85 if symbol_key in {"GBPUSD", "EURUSD", "USDJPY"} else 1.75)
        stop_atr = 0.86 if symbol_key == "NAS100" else (0.88 if symbol_key in {"GBPUSD", "EURUSD", "USDJPY"} else 0.92)
        setup_name = (
            "AUDNZD_ROTATION_PULLBACK"
            if symbol_key == "AUDNZD"
            else "NAS100_VWAP_PULLBACK"
            if symbol_key == "NAS100"
            else f"{symbol_key}_SESSION_PULLBACK"
        )
        drift_meta = {
            "timeframe": "M5",
            "atr_field": "m5_atr_14",
            "allow_ai_approve_small": True,
            "approve_small_min_probability": 0.49,
            "approve_small_min_confluence": 3.0 if symbol_key == "USDJPY" else 3.1,
            "session_drift_lane": True,
            "structure_cleanliness_floor": 0.56,
            "entry_timing_floor": 0.58,
            "setup_family": "DRIFT_CONTINUATION",
        }
        if symbol_key == "NAS100":
            drift_meta["strategy_key"] = "NAS100_VWAP_TREND_STRATEGY"
        output: list[SignalCandidate] = []
        if bullish_bias:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "session-drift-scalp", "BUY", timestamp),
                    setup=setup_name,
                    side="BUY",
                    score_hint=score_hint,
                    reason="Low-liquidity drift continuation with aligned structure",
                    stop_atr=stop_atr,
                    tp_r=tp_r,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=confluence,
                    confluence_required=2.95 if symbol_key == "USDJPY" else 3.0,
                    meta=dict(drift_meta),
                )
            )
        if bearish_bias:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "session-drift-scalp", "SELL", timestamp),
                    setup=setup_name,
                    side="SELL",
                    score_hint=score_hint,
                    reason="Low-liquidity drift continuation with aligned structure",
                    stop_atr=stop_atr,
                    tp_r=tp_r,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=confluence,
                    confluence_required=2.95 if symbol_key == "USDJPY" else 3.0,
                    meta=dict(drift_meta),
                )
            )
        return output

    def _audjpy_sydney_range_break(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if self._normalize_symbol(symbol) != "AUDJPY":
            return []
        session_name = str(session.session_name).upper()
        if session_name != "SYDNEY":
            return []
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_limit("AUDJPY", multiplier=0.88):
            return []
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        atr_ratio = float(row.get("m15_atr_pct_of_avg", row.get("m5_atr_pct_of_avg", 1.0)))
        prev_high = float(row.get("m15_rolling_high_prev_20", close + atr))
        prev_low = float(row.get("m15_rolling_low_prev_20", close - atr))
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        if atr_ratio > 1.38 or volume_ratio < 0.80:
            return []
        bullish = self._bullish_flag(row)
        bearish = self._bearish_flag(row)
        output: list[SignalCandidate] = []
        if close > prev_high and (close - prev_high) >= (atr * 0.10) and bullish:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "audjpy-sydney-range-break", "BUY", timestamp),
                    setup="AUDJPY_SYDNEY_RANGE_BREAK",
                    side="BUY",
                    score_hint=0.64,
                    reason="AUDJPY Sydney compression break with continuation bias",
                    stop_atr=0.90,
                    tp_r=1.9,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=4.0,
                    confluence_required=3.8,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "TOKYO_BREAKOUT"},
                )
            )
        if close < prev_low and (prev_low - close) >= (atr * 0.10) and bearish:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "audjpy-sydney-range-break", "SELL", timestamp),
                    setup="AUDJPY_SYDNEY_RANGE_BREAK",
                    side="SELL",
                    score_hint=0.64,
                    reason="AUDJPY Sydney downside compression break with continuation bias",
                    stop_atr=0.90,
                    tp_r=1.9,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=4.0,
                    confluence_required=3.8,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "TOKYO_BREAKOUT"},
                )
            )
        return output

    def _audjpy_london_carry_continuation(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if self._normalize_symbol(symbol) != "AUDJPY":
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"LONDON", "OVERLAP", "NEW_YORK"}:
            return []
        if "TREND" not in session.allowed_strategies:
            return []
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_limit("AUDJPY", multiplier=0.96):
            return []
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        ema20 = float(row.get("m15_ema_20", row.get("m5_ema_20", close)))
        ema50 = float(row.get("m15_ema_50", row.get("m5_ema_50", close)))
        h1_ema_50 = float(row.get("h1_ema_50", ema50))
        h1_ema_200 = float(row.get("h1_ema_200", h1_ema_50))
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        if volume_ratio < 0.92 or abs(close - ema20) > (atr * 0.52):
            return []
        bullish = ema20 >= ema50 and h1_ema_50 >= h1_ema_200 and close >= (ema20 - (atr * 0.10))
        bearish = ema20 <= ema50 and h1_ema_50 <= h1_ema_200 and close <= (ema20 + (atr * 0.10))
        output: list[SignalCandidate] = []
        if bullish:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "audjpy-london-carry", "BUY", timestamp),
                    setup="AUDJPY_LONDON_CARRY_CONTINUATION",
                    side="BUY",
                    score_hint=0.61,
                    reason="AUDJPY London carry continuation after orderly pullback",
                    stop_atr=0.90,
                    tp_r=1.9,
                    entry_kind="DAYTRADE",
                    strategy_family="TREND",
                    confluence_score=3.8,
                    confluence_required=3.6,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "PULLBACK"},
                )
            )
        if bearish:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "audjpy-london-carry", "SELL", timestamp),
                    setup="AUDJPY_LONDON_CARRY_CONTINUATION",
                    side="SELL",
                    score_hint=0.61,
                    reason="AUDJPY London downside carry continuation after orderly pullback",
                    stop_atr=0.90,
                    tp_r=1.9,
                    entry_kind="DAYTRADE",
                    strategy_family="TREND",
                    confluence_score=3.8,
                    confluence_required=3.6,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "PULLBACK"},
                )
            )
        return output

    def _nzdjpy_sydney_breakout_retest(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if self._normalize_symbol(symbol) != "NZDJPY":
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"SYDNEY", "TOKYO"}:
            return []
        if "TREND" not in session.allowed_strategies:
            return []
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_limit("NZDJPY", multiplier=0.90):
            return []
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        if regime_state not in {"TRENDING", "BREAKOUT_EXPANSION", "RANGING"}:
            return []
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        atr_ratio = float(row.get("m15_atr_pct_of_avg", row.get("m5_atr_pct_of_avg", 1.0)))
        prev_high = float(row.get("m15_rolling_high_prev_20", close + atr))
        prev_low = float(row.get("m15_rolling_low_prev_20", close - atr))
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        body_efficiency = self._body_efficiency_value(row)
        ret_1 = abs(float(row.get("m5_ret_1", 0.0)))
        relative_impulse = ret_1 / max(atr / max(abs(close), 1e-6), 1e-9)
        if atr_ratio < 0.78 or atr_ratio > 1.78 or volume_ratio < 0.88 or body_efficiency < 0.48:
            return []
        low = float(row.get("m15_low", row.get("m5_low", close)))
        high = float(row.get("m15_high", row.get("m5_high", close)))
        if relative_impulse < 0.018 or relative_impulse > 0.28:
            return []
        range_pos = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        bullish_candle = int(row.get("m5_bullish", row.get("m15_bullish", 0))) == 1
        bearish_candle = int(row.get("m5_bearish", row.get("m15_bearish", 0))) == 1
        bullish = (
            bullish_candle
            and
            close >= prev_high
            and abs(close - prev_high) <= (atr * 0.10)
            and low <= (prev_high + (atr * 0.10))
            and range_pos >= 0.64
        )
        bearish = (
            bearish_candle
            and
            close <= prev_low
            and abs(close - prev_low) <= (atr * 0.10)
            and high >= (prev_low - (atr * 0.10))
            and range_pos <= 0.36
        )
        output: list[SignalCandidate] = []
        if bullish:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "nzdjpy-sydney-retest", "BUY", timestamp),
                    setup="NZDJPY_SYDNEY_BREAKOUT_RETEST",
                    side="BUY",
                    score_hint=0.63,
                    reason="NZDJPY Sydney breakout retest continuation",
                    stop_atr=0.92,
                    tp_r=2.0,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=4.0,
                    confluence_required=3.8,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "TOKYO_BREAKOUT"},
                )
            )
        if bearish:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "nzdjpy-sydney-retest", "SELL", timestamp),
                    setup="NZDJPY_SYDNEY_BREAKOUT_RETEST",
                    side="SELL",
                    score_hint=0.63,
                    reason="NZDJPY Sydney downside breakout retest continuation",
                    stop_atr=0.92,
                    tp_r=2.0,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=4.0,
                    confluence_required=3.8,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "TOKYO_BREAKOUT"},
                )
            )
        return output

    def _audnzd_rotation_breakout(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if self._normalize_symbol(symbol) != "AUDNZD":
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"SYDNEY", "TOKYO"}:
            return []
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        if regime_state not in {"TRENDING", "BREAKOUT_EXPANSION", "RANGING"}:
            return []

        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_limit("AUDNZD", multiplier=0.88):
            return []
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        prev_high = float(row.get("m15_rolling_high_prev_20", close + atr))
        prev_low = float(row.get("m15_rolling_low_prev_20", close - atr))
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        range_position = float(row.get("m15_range_position_20", 0.5))
        bullish = self._bullish_flag(row)
        bearish = self._bearish_flag(row)
        if volume_ratio < 0.78:
            return []

        output: list[SignalCandidate] = []
        if close > prev_high and (close - prev_high) >= (atr * 0.10) and bullish and range_position >= 0.40:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "audnzd-rotation-breakout", "BUY", timestamp),
                    setup="AUDNZD_ASIA_ROTATION_BREAKOUT",
                    side="BUY",
                    score_hint=0.64,
                    reason="AUDNZD structured Asia breakout continuation",
                    stop_atr=0.95,
                    tp_r=2.1,
                    entry_kind="DAYTRADE",
                    strategy_family="TREND",
                    confluence_score=4.1,
                    confluence_required=3.9,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "ASIA_ROTATION"},
                )
            )
        if close < prev_low and (prev_low - close) >= (atr * 0.10) and bearish and range_position <= 0.60:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "audnzd-rotation-breakout", "SELL", timestamp),
                    setup="AUDNZD_ASIA_ROTATION_BREAKOUT",
                    side="SELL",
                    score_hint=0.64,
                    reason="AUDNZD structured Asia downside breakout continuation",
                    stop_atr=0.95,
                    tp_r=2.1,
                    entry_kind="DAYTRADE",
                    strategy_family="TREND",
                    confluence_score=4.1,
                    confluence_required=3.9,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "ASIA_ROTATION"},
                )
            )
        return output

    def _audnzd_rotation_pullback(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if self._normalize_symbol(symbol) != "AUDNZD":
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"SYDNEY", "TOKYO"}:
            return []
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        if "TREND" not in session.allowed_strategies and regime_state not in {"RANGING"}:
            return []
        if regime_state not in {"TRENDING", "BREAKOUT_EXPANSION", "RANGING"}:
            return []

        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_limit("AUDNZD", multiplier=0.86):
            return []
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        ema20 = float(row.get("m15_ema_20", row.get("m5_ema_20", close)))
        ema50 = float(row.get("m15_ema_50", row.get("m5_ema_50", close)))
        h1_ema_50 = float(row.get("h1_ema_50", row.get("m15_ema_20", ema20)))
        h1_ema_200 = float(row.get("h1_ema_200", row.get("h1_ema_50", h1_ema_50)))
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        range_position = float(row.get("m15_range_position_20", 0.5))
        body_efficiency = self._body_efficiency_value(row)
        reclaim_ret = float(row.get("m15_ret_1", row.get("m5_ret_1", 0.0)) or 0.0)
        if volume_ratio < 0.84 or body_efficiency < 0.48 or abs(close - ema20) < (atr * 0.03) or abs(close - ema20) > (atr * 0.48):
            return []
        if reclaim_ret == 0.0 and not (body_efficiency >= 0.60 and volume_ratio >= 0.98):
            return []

        output: list[SignalCandidate] = []
        if ema20 >= ema50 and h1_ema_50 >= h1_ema_200 and range_position <= 0.64 and (reclaim_ret > 0 or (reclaim_ret == 0.0 and body_efficiency >= 0.60)):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "audnzd-rotation-pullback", "BUY", timestamp),
                    setup="AUDNZD_ASIA_ROTATION_PULLBACK",
                    side="BUY",
                    score_hint=0.62,
                    reason="AUDNZD selective Asia structured pullback",
                    stop_atr=0.85,
                    tp_r=2.0,
                    entry_kind="DAYTRADE",
                    strategy_family="TREND",
                    confluence_score=3.9,
                    confluence_required=3.6,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "ASIA_ROTATION"},
                )
            )
        if ema20 <= ema50 and h1_ema_50 <= h1_ema_200 and range_position >= 0.36 and (reclaim_ret < 0 or (reclaim_ret == 0.0 and body_efficiency >= 0.60)):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "audnzd-rotation-pullback", "SELL", timestamp),
                    setup="AUDNZD_ASIA_ROTATION_PULLBACK",
                    side="SELL",
                    score_hint=0.62,
                    reason="AUDNZD selective Asia structured downside pullback",
                    stop_atr=0.85,
                    tp_r=2.0,
                    entry_kind="DAYTRADE",
                    strategy_family="TREND",
                    confluence_score=3.9,
                    confluence_required=3.6,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "ASIA_ROTATION"},
                )
            )
        return output

    def _audnzd_range_rejection(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if self._normalize_symbol(symbol) != "AUDNZD":
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"SYDNEY", "TOKYO"}:
            return []
        if "RANGE" not in session.allowed_strategies and regime.label not in {"RANGING", "MEAN_REVERSION"}:
            return []
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        if regime_state not in {"RANGING", "MEAN_REVERSION", "TRENDING"}:
            return []
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_limit("AUDNZD", multiplier=0.88):
            return []
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        range_position = float(row.get("m15_range_position_20", 0.5))
        atr_ratio = float(row.get("m15_atr_pct_of_avg", row.get("m5_atr_pct_of_avg", 1.0)) or 1.0)
        bullish_candle = int(row.get("m5_bullish", row.get("m15_bullish", 0)) or 0) == 1
        bearish_candle = int(row.get("m5_bearish", row.get("m15_bearish", 0)) or 0) == 1
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        body_efficiency = max(
            self._body_efficiency_value(row),
            float(row.get("m15_body_efficiency", row.get("m15_candle_efficiency", 0.0)) or 0.0),
        )
        lower_wick = float(row.get("m5_lower_wick_ratio", 0.0) or 0.0)
        upper_wick = float(row.get("m5_upper_wick_ratio", 0.0) or 0.0)
        synthetic_bull_rejection = range_position <= 0.12 and lower_wick >= 0.30 and bullish_candle
        synthetic_bear_rejection = range_position >= 0.88 and upper_wick >= 0.30 and bearish_candle
        pinbar_bull = int(row.get("m5_pinbar_bull", 0)) == 1 or int(row.get("m5_engulf_bull", 0)) == 1 or synthetic_bull_rejection
        pinbar_bear = int(row.get("m5_pinbar_bear", 0)) == 1 or int(row.get("m5_engulf_bear", 0)) == 1 or synthetic_bear_rejection
        volume_low = 0.86 if regime_state in {"RANGING", "TRENDING"} else 0.84
        volume_high = 1.30 if regime_state in {"RANGING", "TRENDING"} else 1.34
        synthetic_extreme_rejection = synthetic_bull_rejection or synthetic_bear_rejection
        body_floor = 0.10 if synthetic_extreme_rejection else 0.48 if regime_state in {"RANGING", "TRENDING"} else 0.46
        atr_cap = 1.55 if synthetic_extreme_rejection else 1.24 if regime_state == "RANGING" else 1.26 if regime_state == "TRENDING" else 1.30
        if volume_ratio < volume_low or volume_ratio > volume_high or body_efficiency < body_floor or atr_ratio > atr_cap:
            return []
        output: list[SignalCandidate] = []
        lower_wick_floor = 0.28 if regime_state in {"RANGING", "TRENDING"} else 0.24
        upper_wick_floor = 0.28 if regime_state in {"RANGING", "TRENDING"} else 0.24
        if range_position <= 0.28 and pinbar_bull and lower_wick >= lower_wick_floor and body_efficiency <= 0.78:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "audnzd-range-reject", "BUY", timestamp),
                    setup="AUDNZD_RANGE_REJECTION",
                    side="BUY",
                    score_hint=0.57,
                    reason="AUDNZD selective Asia range rejection long",
                    stop_atr=0.80,
                    tp_r=1.8,
                    entry_kind="DAYTRADE",
                    strategy_family="RANGE",
                    confluence_score=3.7,
                    confluence_required=3.6,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "RANGE_REJECTION"},
                )
            )
        if range_position >= 0.72 and pinbar_bear and upper_wick >= upper_wick_floor and body_efficiency <= 0.78:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "audnzd-range-reject", "SELL", timestamp),
                    setup="AUDNZD_RANGE_REJECTION",
                    side="SELL",
                    score_hint=0.57,
                    reason="AUDNZD selective Asia range rejection short",
                    stop_atr=0.80,
                    tp_r=1.8,
                    entry_kind="DAYTRADE",
                    strategy_family="RANGE",
                    confluence_score=3.7,
                    confluence_required=3.6,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "RANGE_REJECTION"},
                )
            )
        return output

    def _audnzd_compression_release(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        if self._normalize_symbol(symbol) != "AUDNZD":
            return []
        session_name = str(session.session_name).upper()
        if session_name not in {"SYDNEY", "TOKYO"}:
            return []
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        if regime_state not in {"BREAKOUT_EXPANSION", "TRENDING", "RANGING"}:
            return []
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_limit("AUDNZD", multiplier=0.84):
            return []
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0))), 1e-6)
        atr_ratio = float(row.get("m15_atr_pct_of_avg", row.get("m5_atr_pct_of_avg", 1.0)))
        close = float(row.get("m15_close", row.get("m5_close", 0.0)))
        prev_high = float(row.get("m15_rolling_high_prev_20", close + atr))
        prev_low = float(row.get("m15_rolling_low_prev_20", close - atr))
        volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)))
        if atr_ratio > 1.28 or volume_ratio < 0.78:
            return []
        output: list[SignalCandidate] = []
        if close > prev_high and (close - prev_high) >= (atr * 0.08):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "audnzd-compression", "BUY", timestamp),
                    setup="AUDNZD_COMPRESSION_RELEASE",
                    side="BUY",
                    score_hint=0.61,
                    reason="AUDNZD Asia compression release breakout",
                    stop_atr=0.90,
                    tp_r=2.0,
                    entry_kind="DAYTRADE",
                    strategy_family="TREND",
                    confluence_score=3.9,
                    confluence_required=3.8,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "ASIA_ROTATION"},
                )
            )
        if close < prev_low and (prev_low - close) >= (atr * 0.08):
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "audnzd-compression", "SELL", timestamp),
                    setup="AUDNZD_COMPRESSION_RELEASE",
                    side="SELL",
                    score_hint=0.61,
                    reason="AUDNZD Asia downside compression release breakout",
                    stop_atr=0.90,
                    tp_r=2.0,
                    entry_kind="DAYTRADE",
                    strategy_family="TREND",
                    confluence_score=3.9,
                    confluence_required=3.8,
                    meta={"timeframe": "M15", "atr_field": "m15_atr_14", "setup_family": "ASIA_ROTATION"},
                )
            )
        return output

    def _session_momentum_boost(
        self,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        symbol_key = self._normalize_symbol(symbol)
        if symbol_key not in {"EURUSD", "GBPUSD", "USDJPY", "NAS100", "USOIL"}:
            return []
        session_name = str(session.session_name).upper()
        weekend_mode = self._is_weekend_market_mode(timestamp)
        if session_name not in {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}:
            return []

        if session_name in {"SYDNEY", "TOKYO"} and symbol_key in {"EURUSD", "GBPUSD"}:
            return []
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        raw_regime_label = str(getattr(regime, "state_label", regime.label) or regime.label).strip().upper()
        drift_trend_ready = self._drift_continuation_ready(
            symbol=symbol_key,
            row=row,
            session_name=session_name,
            raw_regime_label=raw_regime_label,
        )
        if symbol_key == "USOIL" and session_name == "TOKYO" and regime_state == "LOW_LIQUIDITY_CHOP":
            return []

        spread = float(row.get("m5_spread", 0.0))
        spread_cap = self._spread_limit(symbol_key, multiplier=(
            1.08 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.90 if session_name == "TOKYO" else 0.82
        ))
        if spread > spread_cap:
            return []

        close = float(row.get("m5_close", 0.0))
        atr = max(float(row.get("m5_atr_14", 0.0)), 1e-6)
        m5_ret_1 = abs(float(row.get("m5_ret_1", 0.0)))
        impulse = (m5_ret_1 / max(atr / max(abs(close), 1e-6), 1e-9))
        volume_ratio = self._ratio_or_neutral(row.get("m5_volume_ratio_20", 1.0))
        body_efficiency = float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)) or 0.55)
        ema20 = float(row.get("m5_ema_20", close))
        distance_to_ema = abs(close - ema20)
        is_index_or_oil = symbol_key in {"NAS100", "USOIL"}
        range_pos_raw = row.get("m15_range_position_20", row.get("m5_range_position_20", None))
        range_pos = None if range_pos_raw is None else float(range_pos_raw)
        min_volume = 0.80 if is_index_or_oil else (0.86 if drift_trend_ready else 0.95)
        if volume_ratio < min_volume:
            return []
        if impulse < 0.05 and int(row.get("m5_bullish", 0)) == 0 and int(row.get("m5_bearish", 0)) == 0:
            return []
        if not is_index_or_oil:
            if body_efficiency < (0.40 if drift_trend_ready else 0.58):
                return []
            if distance_to_ema > (atr * 0.95):
                return []
            impulse_cap = 0.40 if drift_trend_ready and symbol_key == "USDJPY" else 0.48 if drift_trend_ready else 0.32 if session_name in {"SYDNEY", "TOKYO"} and symbol_key == "USDJPY" else 0.42
            if impulse > impulse_cap:
                return []
            trend_gap_ratio = abs(float(row.get("m5_ema_20", close)) - float(row.get("m5_ema_50", close))) / max(atr, 1e-6)
            if trend_gap_ratio < (0.12 if drift_trend_ready else 0.35):
                return []
            if symbol_key == "GBPUSD":
                if session_name not in {"LONDON", "OVERLAP", "NEW_YORK"}:
                    return []
                if volume_ratio < (0.90 if drift_trend_ready else 0.98) or body_efficiency < (0.46 if drift_trend_ready else 0.58) or distance_to_ema > (atr * 0.72) or impulse < (0.03 if drift_trend_ready else 0.05) or impulse > (0.38 if drift_trend_ready else 0.30):
                    return []
            if symbol_key == "EURUSD" and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}:
                if volume_ratio < (0.84 if drift_trend_ready else 0.88) or body_efficiency < (0.40 if drift_trend_ready else 0.50) or impulse < (0.02 if drift_trend_ready else 0.04) or impulse > (0.46 if drift_trend_ready else 0.38):
                    return []
            if symbol_key == "USDJPY":
                if (
                    volume_ratio < (0.84 if drift_trend_ready else 0.88)
                    or body_efficiency < (0.40 if drift_trend_ready else 0.50)
                    or distance_to_ema < (atr * 0.02)
                    or distance_to_ema > (atr * 0.90)
                    or impulse < (0.02 if drift_trend_ready else 0.04)
                    or impulse > (0.44 if drift_trend_ready else 0.36)
                    or trend_gap_ratio < (0.10 if drift_trend_ready else 0.24)
                ):
                    return []

        trend_bias = float(row.get("m5_ema_20", close)) - float(row.get("m5_ema_50", close))
        explicit_bullish = int(row.get("m5_bullish", 0)) == 1
        explicit_bearish = int(row.get("m5_bearish", 0)) == 1
        bullish = explicit_bullish if not is_index_or_oil else explicit_bullish or trend_bias > 0
        bearish = explicit_bearish if not is_index_or_oil else explicit_bearish or trend_bias < 0
        if not is_index_or_oil and not (bullish or bearish):
            return []
        if range_pos is not None:
            if symbol_key == "USDJPY":
                if bullish and not (0.46 <= range_pos <= 0.96):
                    return []
                if bearish and not (0.04 <= range_pos <= 0.54):
                    return []
            elif symbol_key in {"EURUSD", "GBPUSD"}:
                if bullish and range_pos < 0.50:
                    return []
                if bearish and range_pos > 0.50:
                    return []
        trend_flag = bool(regime.details.get("trend_flag", regime.label == "TRENDING"))
        if not trend_flag and not is_index_or_oil and not drift_trend_ready:
            return []
        if symbol_key == "NAS100":
            if session_name not in {"LONDON", "OVERLAP", "NEW_YORK"}:
                return []
            if regime_state != "BREAKOUT_EXPANSION" or volume_ratio < 1.02 or body_efficiency < 0.56 or impulse < 0.08 or impulse > 0.24:
                return []
        if symbol_key == "USOIL" and session_name == "TOKYO":
            if regime_state in {"RANGING", "MEAN_REVERSION"}:
                if body_efficiency < 0.72 or volume_ratio < 1.16 or impulse < 0.10 or impulse > 0.22:
                    return []
            elif body_efficiency < 0.68 or volume_ratio < 1.08 or impulse < 0.10 or impulse > 0.26:
                return []
        if symbol_key == "USOIL" and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}:
            if regime_state != "BREAKOUT_EXPANSION" or volume_ratio < 1.04 or body_efficiency < 0.60 or impulse < 0.08 or impulse > 0.24:
                return []
        session_boost = 0.35 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.12 if session_name == "TOKYO" else 0.06
        base_confluence = (
            2.7
            + min(0.8, max(0.0, impulse - 0.06))
            + min(0.8, max(0.0, volume_ratio - 0.85))
            + session_boost
            + (0.25 if trend_flag else 0.0)
        )
        if session_name in {"SYDNEY", "TOKYO"} and symbol_key == "USDJPY":
            base_confluence -= 0.12
        confluence = clamp(base_confluence, 0.0, 5.0)
        if confluence < 2.6:
            return []

        score_hint = clamp(0.56 + (0.05 * (confluence - 2.6)), 0.53, 0.77)
        setup = f"{symbol_key}_SESSION_MOMENTUM"
        output: list[SignalCandidate] = []
        drift_meta = {
            "session_drift_lane": True,
            "structure_cleanliness_floor": 0.56,
            "entry_timing_floor": 0.58,
            "setup_family": "DRIFT_CONTINUATION",
        } if drift_trend_ready else {}
        if bullish:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "session-momentum", "BUY", timestamp),
                    setup=setup,
                    side="BUY",
                    score_hint=score_hint,
                    reason="Session momentum continuation",
                    stop_atr=0.95,
                    tp_r=1.75,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=confluence,
                    confluence_required=3.0,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.50,
                        "approve_small_min_confluence": 3.0,
                        **drift_meta,
                    },
                )
            )
        if bearish:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "session-momentum", "SELL", timestamp),
                    setup=setup,
                    side="SELL",
                    score_hint=score_hint,
                    reason="Session momentum continuation",
                    stop_atr=0.95,
                    tp_r=1.75,
                    entry_kind="SCALP",
                    strategy_family="TREND",
                    confluence_score=confluence,
                    confluence_required=3.0,
                    meta={
                        "timeframe": "M15",
                        "atr_field": "m15_atr_14",
                        "allow_ai_approve_small": True,
                        "approve_small_min_probability": 0.50,
                        "approve_small_min_confluence": 3.0,
                        **drift_meta,
                    },
                )
            )
        return output

    def _live_candidate_recovery(
        self,
        *,
        symbol: str,
        row: pd.Series,
        regime: RegimeClassification,
        session: SessionContext,
        timestamp,
    ) -> list[SignalCandidate]:
        symbol_key = self._normalize_symbol(symbol)
        if symbol_key not in {"EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "AUDNZD", "NAS100", "USOIL", "BTCUSD", "XAUUSD", "XAGUSD"}:
            return []
        session_name = str(session.session_name).upper()
        allowed_sessions: dict[str, set[str]] = {
            "EURUSD": {"LONDON", "OVERLAP", "NEW_YORK"},
            "GBPUSD": {"LONDON", "OVERLAP", "NEW_YORK"},
            "USDJPY": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "EURJPY": {"LONDON", "OVERLAP", "NEW_YORK"},
            "GBPJPY": {"LONDON", "OVERLAP", "NEW_YORK"},
            "AUDJPY": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "NZDJPY": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "AUDNZD": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "NAS100": {"LONDON", "OVERLAP", "NEW_YORK"},
            "USOIL": {"LONDON", "OVERLAP", "NEW_YORK", "TOKYO"},
            "BTCUSD": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "XAUUSD": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
            "XAGUSD": {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
        }
        if session_name not in allowed_sessions.get(symbol_key, set()):
            return []

        spread = float(row.get("m5_spread", 0.0) or 0.0)
        spread_limit = self._spread_limit(
            symbol_key,
            multiplier=1.08 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.96,
        )
        if spread > spread_limit:
            return []

        close = float(row.get("m15_close", row.get("m5_close", 0.0)) or 0.0)
        atr = max(float(row.get("m15_atr_14", row.get("m5_atr_14", 0.0)) or 0.0), 1e-6)
        if close <= 0.0:
            return []
        m5_ema20 = float(row.get("m5_ema_20", close) or close)
        m5_ema50 = float(row.get("m5_ema_50", m5_ema20) or m5_ema20)
        m15_ema20 = float(row.get("m15_ema_20", m5_ema20) or m5_ema20)
        m15_ema50 = float(row.get("m15_ema_50", m15_ema20) or m15_ema20)
        h1_ema50 = float(row.get("h1_ema_50", m15_ema50) or m15_ema50)
        h1_ema200 = float(row.get("h1_ema_200", h1_ema50) or h1_ema50)
        m5_volume_ratio = self._ratio_or_neutral(row.get("m5_volume_ratio_20", 1.0))
        m15_volume_ratio = self._ratio_or_neutral(row.get("m15_volume_ratio_20", m5_volume_ratio))
        m5_tick_volume = float(row.get("m5_tick_volume", 0.0) or 0.0)
        m15_tick_volume = float(row.get("m15_tick_volume", 0.0) or 0.0)
        sparse_tick_volume = m5_tick_volume <= 0.0 and m15_tick_volume <= 0.0
        effective_volume = max(
            m5_volume_ratio,
            m15_volume_ratio * (0.90 if sparse_tick_volume else 0.96),
        )
        effective_body = max(
            self._body_efficiency_value(row),
            float(row.get("m15_body_efficiency", row.get("m15_candle_efficiency", 0.0)) or 0.0),
        )
        alignment_score = float(row.get("multi_tf_alignment_score", 0.5) or 0.5)
        fractal_score = float(
            row.get(
                "fractal_persistence_score",
                row.get("hurst_persistence_score", row.get("m15_hurst_proxy_64", row.get("m5_hurst_proxy_64", 0.5))),
            )
            or 0.5
        )
        seasonality_score = float(row.get("seasonality_edge_score", 0.5) or 0.5)
        instability_score = float(row.get("market_instability_score", row.get("feature_drift_score", 0.0)) or 0.0)
        feature_drift = float(row.get("feature_drift_score", 0.0) or 0.0)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        ret_1 = float(row.get("m15_ret_1", row.get("m5_ret_1", 0.0)) or 0.0)
        ret_3 = float(row.get("m15_ret_3", row.get("m5_ret_3", 0.0)) or 0.0)
        distance_to_ema_ratio = abs(close - m15_ema20) / max(atr, 1e-6)
        trend_gap_ratio = abs(m15_ema20 - m15_ema50) / max(atr, 1e-6)
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        raw_regime_label = str(getattr(regime, "state_label", regime.label) or regime.label).strip().upper()
        learning_state = self._learning_pattern_state(symbol=symbol_key, session_name=session_name)
        throughput_recovery = bool(learning_state.get("throughput_recovery_active", False))

        if seasonality_score < 0.18:
            return []
        if instability_score > 0.46 or feature_drift > 0.34:
            return []

        min_alignment = (
            0.42
            if symbol_key in {"EURUSD", "GBPUSD", "EURJPY", "GBPJPY"}
            else 0.38
            if symbol_key in {"NAS100", "USOIL"}
            else 0.26
            if symbol_key == "BTCUSD"
            else 0.24
            if symbol_key in {"XAUUSD", "XAGUSD"}
            else 0.34
        )
        min_alignment -= 0.04 if throughput_recovery else 0.0
        min_fractal = 0.22 if symbol_key in {"EURUSD", "GBPUSD", "EURJPY", "GBPJPY"} else 0.16 if symbol_key in {"BTCUSD", "XAUUSD", "XAGUSD"} else 0.18
        min_fractal -= 0.04 if throughput_recovery else 0.0
        min_body = (
            0.42
            if symbol_key in {"EURUSD", "GBPUSD"}
            else 0.32
            if symbol_key in {"NAS100", "USOIL"}
            else 0.18
            if symbol_key == "BTCUSD"
            else 0.12
            if symbol_key in {"XAUUSD", "XAGUSD"}
            else 0.36
        )
        min_body -= 0.04 if throughput_recovery else 0.0
        min_volume = (
            0.86
            if symbol_key in {"EURUSD", "GBPUSD"}
            else 0.52
            if symbol_key in {"NAS100", "USOIL"}
            else 0.34
            if symbol_key == "BTCUSD"
            else 0.28
            if symbol_key in {"XAUUSD", "XAGUSD"}
            else 0.82
        )
        min_volume -= 0.06 if throughput_recovery else 0.0
        if symbol_key == "AUDNZD":
            min_body = min(min_body, 0.28)
        if (
            symbol_key in {"USDJPY", "AUDJPY", "NZDJPY"}
            and sparse_tick_volume
            and alignment_score >= 0.68
            and effective_volume >= max(0.72, min_volume - 0.10)
        ):
            min_body = min(min_body, 0.12)
        if symbol_key in {"USDJPY", "AUDJPY", "NZDJPY"} and sparse_tick_volume and (range_position <= 0.24 or range_position >= 0.76):
            min_alignment = min(min_alignment, 0.24)
            min_body = min(min_body, 0.28)

        bullish_bias = bool(
            close >= m5_ema20
            and m5_ema20 >= m5_ema50
            and m15_ema20 >= m15_ema50
            and (h1_ema50 >= h1_ema200 or alignment_score >= 0.70 or range_position >= 0.78 or ret_3 > 0.0)
        )
        bearish_bias = bool(
            close <= m5_ema20
            and m5_ema20 <= m5_ema50
            and m15_ema20 <= m15_ema50
            and (h1_ema50 <= h1_ema200 or alignment_score >= 0.70 or range_position <= 0.22 or ret_3 < 0.0)
        )
        soft_bullish_bias = bool(
            close >= m15_ema20
            and m15_ema20 >= m15_ema50
            and ret_1 >= -0.0004
            and range_position >= 0.34
        )
        soft_bearish_bias = bool(
            close <= m15_ema20
            and m15_ema20 <= m15_ema50
            and ret_1 <= 0.0004
            and range_position <= 0.66
        )

        recovery_candidates: list[SignalCandidate] = []
        trend_gap_floor = (
            0.010
            if symbol_key == "BTCUSD"
            else 0.008
            if symbol_key in {"XAUUSD", "XAGUSD"}
            else 0.018
            if symbol_key in {"USDJPY", "AUDJPY", "NZDJPY"}
            else 0.016
            if symbol_key == "AUDNZD"
            else 0.025
            if symbol_key in {"NAS100", "USOIL"}
            else 0.035
            if symbol_key in {"EURUSD", "GBPUSD", "EURJPY", "GBPJPY"}
            else 0.05
        )
        trend_ready = bool(
            effective_volume >= min_volume
            and effective_body >= min_body
            and alignment_score >= min_alignment
            and fractal_score >= min_fractal
            and trend_gap_ratio >= trend_gap_floor
            and (bullish_bias or bearish_bias)
        )
        if trend_ready and symbol_key not in {"NAS100", "USOIL"}:
            if bullish_bias and range_position >= 0.92 and distance_to_ema_ratio >= 0.90:
                trend_ready = False
            if bearish_bias and range_position <= 0.08 and distance_to_ema_ratio >= 0.90:
                trend_ready = False
        if trend_ready:
            momentum_ready = bool(
                range_position <= 0.24
                or range_position >= 0.76
                or abs(ret_1) >= 0.00008
                or distance_to_ema_ratio >= 0.26
            )
            pullback_ready = bool(0.04 <= distance_to_ema_ratio <= 0.72 and 0.16 <= range_position <= 0.84)
            if symbol_key == "AUDJPY":
                setup = "AUDJPY_ASIA_CONTINUATION_PULLBACK" if pullback_ready and not momentum_ready else "AUDJPY_ASIA_MOMENTUM_BREAKOUT"
                tp_r = 1.85
            elif symbol_key == "NZDJPY":
                setup = "NZDJPY_ASIA_CONTINUATION_PULLBACK" if pullback_ready and not momentum_ready else "NZDJPY_ASIA_MOMENTUM_BREAKOUT"
                tp_r = 1.90
            elif symbol_key == "AUDNZD":
                setup = "AUDNZD_ROTATION_PULLBACK" if pullback_ready and not momentum_ready else "AUDNZD_COMPRESSION_RELEASE"
                tp_r = 1.90
            elif symbol_key == "BTCUSD":
                setup = "BTCUSD_PRICE_ACTION_CONTINUATION" if momentum_ready else "BTCUSD_RANGE_EXPANSION"
                tp_r = 2.00
            elif symbol_key == "XAUUSD":
                setup = "XAUUSD_M1_MICRO_SCALPER" if momentum_ready else "XAU_BREAKOUT_RETEST"
                tp_r = 1.78
            elif symbol_key == "USDJPY":
                setup = "USDJPY_SESSION_PULLBACK" if pullback_ready and not momentum_ready else "USDJPY_SESSION_MOMENTUM"
                tp_r = 1.78
            elif symbol_key in {"EURJPY", "GBPJPY"}:
                setup = f"{symbol_key}_SESSION_PULLBACK" if pullback_ready and not momentum_ready else f"{symbol_key}_SESSION_MOMENTUM"
                tp_r = 1.76
            elif symbol_key == "NAS100":
                setup = "NAS100_VWAP_PULLBACK"
                tp_r = 1.85
            elif symbol_key == "USOIL":
                setup = "USOIL_BREAKOUT_RETEST"
                tp_r = 1.82
            else:
                setup = f"{symbol_key}_SESSION_PULLBACK" if pullback_ready and not momentum_ready else f"{symbol_key}_SESSION_MOMENTUM"
                tp_r = 1.72
            confluence = clamp(
                3.02
                + min(0.36, max(0.0, effective_volume - min_volume))
                + min(0.24, max(0.0, effective_body - min_body))
                + min(0.24, max(0.0, alignment_score - min_alignment))
                + min(0.18, max(0.0, fractal_score - min_fractal)),
                0.0,
                5.0,
            )
            score_hint = clamp(0.55 + (0.04 * (confluence - 3.0)), 0.53, 0.74)
            recovery_meta = {
                "timeframe": "M15",
                "atr_field": "m15_atr_14",
                "allow_ai_approve_small": True,
                "approve_small_min_probability": 0.49,
                "approve_small_min_confluence": 3.0,
                "setup_family": "LIVE_RECOVERY_TREND",
                "fallback_live_recovery": True,
                "live_recovery_mode": "trend",
                "structure_cleanliness_floor": 0.52,
                "entry_timing_floor": 0.54,
            }
            if symbol_key == "NAS100":
                recovery_meta["strategy_key"] = "NAS100_VWAP_TREND_STRATEGY" if "VWAP" in setup else "NAS100_MOMENTUM_IMPULSE"
            elif symbol_key == "BTCUSD":
                recovery_meta["strategy_key"] = "BTCUSD_PRICE_ACTION_CONTINUATION"
                recovery_meta["setup_family"] = "LIVE_RECOVERY_BTC"
            elif symbol_key == "XAUUSD":
                recovery_meta["setup_family"] = "LIVE_RECOVERY_XAU"
            if bullish_bias:
                recovery_candidates.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "live-recovery", "BUY", timestamp),
                        setup=setup,
                        side="BUY",
                        score_hint=score_hint,
                        reason="Local live recovery promoted a clean continuation candidate",
                        stop_atr=0.88 if symbol_key == "XAUUSD" else 0.90 if symbol_key in {"NAS100", "USOIL"} else 0.92 if symbol_key == "BTCUSD" else 0.94,
                        tp_r=tp_r,
                        entry_kind="DAYTRADE" if symbol_key in {"NAS100", "USOIL"} else "SCALP",
                        strategy_family="TREND",
                        confluence_score=confluence,
                        confluence_required=3.0,
                        meta=dict(recovery_meta),
                    )
                )
            if bearish_bias:
                recovery_candidates.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "live-recovery", "SELL", timestamp),
                        setup=setup,
                        side="SELL",
                        score_hint=score_hint,
                        reason="Local live recovery promoted a clean continuation candidate",
                        stop_atr=0.88 if symbol_key == "XAUUSD" else 0.90 if symbol_key in {"NAS100", "USOIL"} else 0.92 if symbol_key == "BTCUSD" else 0.94,
                        tp_r=tp_r,
                        entry_kind="DAYTRADE" if symbol_key in {"NAS100", "USOIL"} else "SCALP",
                        strategy_family="TREND",
                        confluence_score=confluence,
                        confluence_required=3.0,
                        meta=dict(recovery_meta),
                    )
                )
        if recovery_candidates:
            return recovery_candidates

        london_major = symbol_key in {"EURUSD", "GBPUSD"}
        thin_fx_or_major = symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD", "USDJPY", "EURUSD", "GBPUSD"}
        moderate_recovery_ready = bool(
            symbol_key in {"XAUUSD", "XAGUSD", "BTCUSD", "USDJPY", "AUDJPY", "NZDJPY", "AUDNZD", "EURUSD", "GBPUSD", "NAS100"}
            and effective_volume >= max(0.20, min_volume - (0.10 if symbol_key in {"XAUUSD", "XAGUSD", "BTCUSD"} else 0.08))
            and effective_body >= max(
                0.05 if thin_fx_or_major else 0.06,
                min_body - (0.12 if symbol_key in {"XAUUSD", "XAGUSD", "BTCUSD"} else 0.12 if thin_fx_or_major else 0.10),
            )
            and alignment_score >= max(
                0.06 if thin_fx_or_major else 0.08,
                min_alignment - (0.12 if symbol_key in {"XAUUSD", "XAGUSD", "BTCUSD"} else 0.14 if thin_fx_or_major else 0.10),
            )
            and fractal_score >= max(0.06, min_fractal - 0.10)
            and trend_gap_ratio >= max(0.0038 if thin_fx_or_major else 0.0045, trend_gap_floor * (0.45 if thin_fx_or_major else 0.55))
            and distance_to_ema_ratio <= (1.30 if symbol_key in {"XAUUSD", "XAGUSD", "BTCUSD"} else 1.26 if thin_fx_or_major else 1.18)
            and instability_score <= (0.64 if thin_fx_or_major else 0.58)
            and feature_drift <= (0.46 if thin_fx_or_major else 0.40)
            and 0.10 <= range_position <= 0.90
            and (
                bullish_bias
                or bearish_bias
                or (
                    thin_fx_or_major
                    and (soft_bullish_bias or soft_bearish_bias)
                )
            )
        )
        if moderate_recovery_ready:
            pullback_bias = bool(0.18 <= range_position <= 0.82 and distance_to_ema_ratio <= 0.72)
            breakout_bias = bool(
                trend_gap_ratio >= max(0.010, trend_gap_floor * 0.85)
                and abs(ret_3) >= (0.0009 if symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD", "USDJPY"} else 0.0012)
            )
            if symbol_key == "XAUUSD":
                setup = "XAUUSD_M1_MICRO_SCALPER" if pullback_bias or not breakout_bias else "XAU_BREAKOUT_RETEST"
                tp_r = 1.72
                stop_atr = 0.84
                recovery_family = "LIVE_RECOVERY_XAU"
            elif symbol_key == "XAGUSD":
                setup = "XAGUSD_SESSION_PULLBACK" if pullback_bias or not breakout_bias else "XAGUSD_BREAKOUT_RETEST"
                tp_r = 1.74
                stop_atr = 0.88
                recovery_family = "LIVE_RECOVERY_METALS"
            elif symbol_key == "BTCUSD":
                setup = "BTC_MOMENTUM_CONTINUATION" if pullback_bias or not breakout_bias else "BTC_RANGE_EXPANSION"
                tp_r = 1.76
                stop_atr = 0.90
                recovery_family = "LIVE_RECOVERY_BTC"
            elif symbol_key == "NAS100":
                setup = "NAS100_VWAP_PULLBACK" if pullback_bias or not breakout_bias else "NAS100_OPENING_DRIVE_BREAKOUT"
                tp_r = 1.82
                stop_atr = 0.86
                recovery_family = "LIVE_RECOVERY_INDEX"
            elif symbol_key == "AUDJPY":
                setup = "AUDJPY_TOKYO_CONTINUATION_PULLBACK"
                tp_r = 1.72
                stop_atr = 0.90
                recovery_family = "LIVE_RECOVERY_ASIA_ROTATION"
            elif symbol_key == "NZDJPY":
                setup = "NZDJPY_SWEEP_RECLAIM" if range_position <= 0.30 or range_position >= 0.70 else "NZDJPY_TOKYO_CONTINUATION_PULLBACK"
                tp_r = 1.74
                stop_atr = 0.92
                recovery_family = "LIVE_RECOVERY_ASIA_ROTATION"
            elif symbol_key == "AUDNZD":
                setup = "AUDNZD_ASIA_ROTATION_PULLBACK"
                tp_r = 1.78
                stop_atr = 0.94
                recovery_family = "LIVE_RECOVERY_ASIA_ROTATION"
            elif symbol_key == "EURUSD":
                setup = "EURUSD_LONDON_BREAKOUT"
                tp_r = 1.62
                stop_atr = 0.90
                recovery_family = "LIVE_RECOVERY_LONDON_MAJOR"
            elif symbol_key == "GBPUSD":
                setup = "GBPUSD_LONDON_EXPANSION_BREAKOUT"
                tp_r = 1.66
                stop_atr = 0.92
                recovery_family = "LIVE_RECOVERY_LONDON_MAJOR"
            else:
                setup = "USDJPY_SESSION_PULLBACK"
                tp_r = 1.70
                stop_atr = 0.92
                recovery_family = "LIVE_RECOVERY_ASIA_ROTATION"
            confluence = clamp(
                2.92
                + min(0.30, max(0.0, effective_volume - max(0.20, min_volume - 0.10)))
                + min(0.22, max(0.0, effective_body - max(0.06, min_body - 0.12)))
                + min(0.18, max(0.0, alignment_score - max(0.08, min_alignment - 0.12)))
                + min(0.12, max(0.0, fractal_score - max(0.06, min_fractal - 0.10))),
                0.0,
                5.0,
            )
            score_hint = clamp(0.54 + (0.04 * (confluence - 2.9)), 0.53, 0.72)
            recovery_meta = {
                "timeframe": "M15",
                "atr_field": "m15_atr_14",
                "allow_ai_approve_small": True,
                "approve_small_min_probability": 0.49,
                "approve_small_min_confluence": 2.85,
                "setup_family": recovery_family,
                "fallback_live_recovery": True,
                "live_recovery_mode": "moderate_rotation",
                "structure_cleanliness_floor": 0.48,
                "entry_timing_floor": 0.50,
            }
            if symbol_key == "NAS100":
                recovery_meta["strategy_key"] = "NAS100_VWAP_TREND_STRATEGY" if "VWAP" in setup else "NAS100_OPENING_DRIVE_BREAKOUT"
            output: list[SignalCandidate] = []
            if bullish_bias:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "live-moderate-recovery", "BUY", timestamp),
                        setup=setup,
                        side="BUY",
                        score_hint=score_hint,
                        reason="Local live recovery promoted a moderate-quality continuation",
                        stop_atr=stop_atr,
                        tp_r=tp_r,
                        entry_kind="SCALP",
                        strategy_family="TREND",
                        confluence_score=confluence,
                        confluence_required=2.85,
                        meta=dict(recovery_meta),
                    )
                )
            if bearish_bias:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "live-moderate-recovery", "SELL", timestamp),
                        setup=setup,
                        side="SELL",
                        score_hint=score_hint,
                        reason="Local live recovery promoted a moderate-quality continuation",
                        stop_atr=stop_atr,
                        tp_r=tp_r,
                        entry_kind="SCALP",
                        strategy_family="TREND",
                        confluence_score=confluence,
                        confluence_required=2.85,
                        meta=dict(recovery_meta),
                    )
                )
            if output:
                return output

        range_rotation_recovery_ready = bool(
            symbol_key in {"BTCUSD", "USDJPY", "AUDJPY", "NZDJPY", "AUDNZD", "EURUSD", "GBPUSD"}
            and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION", "NEWS_VOLATILE"}
            and effective_volume >= max(0.14, min_volume - (0.20 if london_major else 0.18))
            and effective_body >= max(0.04, min_body - (0.20 if london_major else 0.18))
            and alignment_score >= max(0.04, min_alignment - (0.20 if london_major else 0.18))
            and fractal_score >= max(0.04, min_fractal - 0.12)
            and distance_to_ema_ratio <= (0.94 if london_major else 0.88)
            and instability_score <= (0.70 if london_major else 0.66)
            and feature_drift <= (0.52 if london_major else 0.48)
            and 0.16 <= range_position <= 0.84
            and (bullish_bias or bearish_bias or soft_bullish_bias or soft_bearish_bias)
        )
        if range_rotation_recovery_ready:
            long_bias = bool(bullish_bias or soft_bullish_bias)
            short_bias = bool(bearish_bias or soft_bearish_bias)
            if symbol_key == "BTCUSD":
                setup = "BTC_MOMENTUM_CONTINUATION"
                tp_r = 1.66
                stop_atr = 0.88
                recovery_family = "LIVE_RECOVERY_BTC"
            elif symbol_key == "USDJPY":
                setup = "USDJPY_SESSION_PULLBACK"
                tp_r = 1.64
                stop_atr = 0.90
                recovery_family = "LIVE_RECOVERY_ASIA_ROTATION"
            elif symbol_key == "AUDJPY":
                setup = "AUDJPY_TOKYO_CONTINUATION_PULLBACK"
                tp_r = 1.66
                stop_atr = 0.90
                recovery_family = "LIVE_RECOVERY_ASIA_ROTATION"
            elif symbol_key == "NZDJPY":
                setup = "NZDJPY_TOKYO_CONTINUATION_PULLBACK"
                tp_r = 1.68
                stop_atr = 0.92
                recovery_family = "LIVE_RECOVERY_ASIA_ROTATION"
            elif symbol_key == "EURUSD":
                setup = "EURUSD_LONDON_BREAKOUT"
                tp_r = 1.58
                stop_atr = 0.88
                recovery_family = "LIVE_RECOVERY_LONDON_MAJOR"
            elif symbol_key == "GBPUSD":
                setup = "GBPUSD_LONDON_EXPANSION_BREAKOUT"
                tp_r = 1.62
                stop_atr = 0.90
                recovery_family = "LIVE_RECOVERY_LONDON_MAJOR"
            else:
                setup = "AUDNZD_ASIA_ROTATION_PULLBACK"
                tp_r = 1.70
                stop_atr = 0.94
                recovery_family = "LIVE_RECOVERY_ASIA_ROTATION"
            confluence = clamp(
                2.82
                + min(0.28, max(0.0, effective_volume - max(0.14, min_volume - 0.18)))
                + min(0.18, max(0.0, effective_body - max(0.04, min_body - 0.18)))
                + min(0.16, max(0.0, alignment_score - max(0.04, min_alignment - 0.18)))
                + min(0.10, max(0.0, fractal_score - max(0.04, min_fractal - 0.12))),
                0.0,
                5.0,
            )
            score_hint = clamp(0.53 + (0.04 * (confluence - 2.8)), 0.52, 0.70)
            recovery_meta = {
                "timeframe": "M15",
                "atr_field": "m15_atr_14",
                "allow_ai_approve_small": True,
                "approve_small_min_probability": 0.48,
                "approve_small_min_confluence": 2.75,
                "setup_family": recovery_family,
                "fallback_live_recovery": True,
                "live_recovery_mode": "range_rotation",
                "structure_cleanliness_floor": 0.44,
                "entry_timing_floor": 0.46,
            }
            output: list[SignalCandidate] = []
            if long_bias:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "live-range-recovery", "BUY", timestamp),
                        setup=setup,
                        side="BUY",
                        score_hint=score_hint,
                        reason="Local live recovery promoted a range-rotation continuation candidate",
                        stop_atr=stop_atr,
                        tp_r=tp_r,
                        entry_kind="SCALP",
                        strategy_family="TREND",
                        confluence_score=confluence,
                        confluence_required=2.75,
                        meta=dict(recovery_meta),
                    )
                )
            if short_bias:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "live-range-recovery", "SELL", timestamp),
                        setup=setup,
                        side="SELL",
                        score_hint=score_hint,
                        reason="Local live recovery promoted a range-rotation continuation candidate",
                        stop_atr=stop_atr,
                        tp_r=tp_r,
                        entry_kind="SCALP",
                        strategy_family="TREND",
                        confluence_score=confluence,
                        confluence_required=2.75,
                        meta=dict(recovery_meta),
                    )
                )
            if output:
                return output

        london_major_recovery_ready = bool(
            symbol_key in {"EURUSD", "GBPUSD"}
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
            and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION", "NEWS_VOLATILE"}
            and effective_volume >= max(0.10, min_volume - 0.24)
            and effective_body >= max(0.03, min_body - 0.22)
            and alignment_score >= max(0.02, min_alignment - 0.22)
            and fractal_score >= max(0.02, min_fractal - 0.14)
            and distance_to_ema_ratio <= 1.02
            and instability_score <= 0.74
            and feature_drift <= 0.56
            and 0.20 <= range_position <= 0.80
            and (bullish_bias or bearish_bias or soft_bullish_bias or soft_bearish_bias)
        )
        if london_major_recovery_ready:
            long_bias = bool(bullish_bias or soft_bullish_bias)
            short_bias = bool(bearish_bias or soft_bearish_bias)
            setup = "EURUSD_LONDON_BREAKOUT" if symbol_key == "EURUSD" else "GBPUSD_LONDON_EXPANSION_BREAKOUT"
            tp_r = 1.56 if symbol_key == "EURUSD" else 1.60
            stop_atr = 0.88 if symbol_key == "EURUSD" else 0.90
            confluence = clamp(
                2.76
                + min(0.24, max(0.0, effective_volume - max(0.10, min_volume - 0.24)))
                + min(0.16, max(0.0, effective_body - max(0.03, min_body - 0.22)))
                + min(0.14, max(0.0, alignment_score - max(0.02, min_alignment - 0.22)))
                + min(0.10, max(0.0, fractal_score - max(0.02, min_fractal - 0.14))),
                0.0,
                5.0,
            )
            score_hint = clamp(0.52 + (0.04 * (confluence - 2.7)), 0.51, 0.68)
            recovery_meta = {
                "timeframe": "M15",
                "atr_field": "m15_atr_14",
                "allow_ai_approve_small": True,
                "approve_small_min_probability": 0.47,
                "approve_small_min_confluence": 2.7,
                "setup_family": "LIVE_RECOVERY_LONDON_MAJOR",
                "fallback_live_recovery": True,
                "live_recovery_mode": "london_major_rotation",
                "structure_cleanliness_floor": 0.42,
                "entry_timing_floor": 0.44,
            }
            output: list[SignalCandidate] = []
            if long_bias:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "live-london-major-recovery", "BUY", timestamp),
                        setup=setup,
                        side="BUY",
                        score_hint=score_hint,
                        reason="Local live recovery promoted a London-major rotation candidate",
                        stop_atr=stop_atr,
                        tp_r=tp_r,
                        entry_kind="SCALP",
                        strategy_family="TREND",
                        confluence_score=confluence,
                        confluence_required=2.7,
                        meta=dict(recovery_meta),
                    )
                )
            if short_bias:
                output.append(
                    SignalCandidate(
                        signal_id=deterministic_id(symbol, "router", "live-london-major-recovery", "SELL", timestamp),
                        setup=setup,
                        side="SELL",
                        score_hint=score_hint,
                        reason="Local live recovery promoted a London-major rotation candidate",
                        stop_atr=stop_atr,
                        tp_r=tp_r,
                        entry_kind="SCALP",
                        strategy_family="TREND",
                        confluence_score=confluence,
                        confluence_required=2.7,
                        meta=dict(recovery_meta),
                    )
                )
            if output:
                return output

        oil_trend_retest_ready = bool(
            symbol_key == "USOIL"
            and effective_volume >= 0.50
            and effective_body >= 0.48
            and alignment_score >= 0.48
            and fractal_score >= 0.18
            and instability_score <= 0.46
            and feature_drift <= 0.26
        )
        if oil_trend_retest_ready:
            oil_long = bool(
                h1_ema50 >= h1_ema200
                and m5_ema20 >= m5_ema50
                and ret_3 <= 0.0
                and 0.24 <= range_position <= 0.58
            )
            oil_short = bool(
                h1_ema50 <= h1_ema200
                and m5_ema20 <= m5_ema50
                and ret_3 >= 0.0
                and 0.42 <= range_position <= 0.76
            )
            if oil_long or oil_short:
                recovery_meta = {
                    "timeframe": "M15",
                    "atr_field": "m15_atr_14",
                    "allow_ai_approve_small": True,
                    "approve_small_min_probability": 0.49,
                    "approve_small_min_confluence": 3.0,
                    "setup_family": "LIVE_RECOVERY_TREND_RETEST",
                    "fallback_live_recovery": True,
                    "live_recovery_mode": "trend_retest",
                    "structure_cleanliness_floor": 0.52,
                    "entry_timing_floor": 0.54,
                }
                confluence = clamp(
                    3.04
                    + min(0.34, max(0.0, effective_volume - 0.50))
                    + min(0.26, max(0.0, effective_body - 0.48))
                    + min(0.18, max(0.0, alignment_score - 0.48))
                    + min(0.14, max(0.0, fractal_score - 0.18)),
                    0.0,
                    5.0,
                )
                score_hint = clamp(0.56 + (0.04 * (confluence - 3.0)), 0.54, 0.76)
                output: list[SignalCandidate] = []
                if oil_long:
                    output.append(
                        SignalCandidate(
                            signal_id=deterministic_id(symbol, "router", "live-trend-retest", "BUY", timestamp),
                            setup="USOIL_BREAKOUT_RETEST",
                            side="BUY",
                            score_hint=score_hint,
                            reason="Local live recovery promoted a clean oil trend retest",
                            stop_atr=0.88,
                            tp_r=1.86,
                            entry_kind="DAYTRADE",
                            strategy_family="TREND",
                            confluence_score=confluence,
                            confluence_required=3.0,
                            meta=dict(recovery_meta),
                        )
                    )
                if oil_short:
                    output.append(
                        SignalCandidate(
                            signal_id=deterministic_id(symbol, "router", "live-trend-retest", "SELL", timestamp),
                            setup="USOIL_BREAKOUT_RETEST",
                            side="SELL",
                            score_hint=score_hint,
                            reason="Local live recovery promoted a clean oil trend retest",
                            stop_atr=0.88,
                            tp_r=1.86,
                            entry_kind="DAYTRADE",
                            strategy_family="TREND",
                            confluence_score=confluence,
                            confluence_required=3.0,
                            meta=dict(recovery_meta),
                        )
                    )
                if output:
                    return output

        jpy_cross_pullback_ready = bool(
            symbol_key in {"EURJPY", "GBPJPY"}
            and sparse_tick_volume
            and effective_volume >= 0.80
            and effective_body >= 0.44
            and fractal_score >= 0.24
            and instability_score <= 0.40
            and feature_drift <= 0.24
            and 0.32 <= range_position <= 0.70
        )
        if jpy_cross_pullback_ready:
            cross_long = bool(
                close >= m15_ema20
                and m15_ema20 >= m15_ema50
                and ret_1 >= 0.0004
            )
            cross_short = bool(
                close <= m15_ema20
                and m15_ema20 <= m15_ema50
                and ret_1 <= -0.0004
            )
            if cross_long or cross_short:
                recovery_meta = {
                    "timeframe": "M15",
                    "atr_field": "m15_atr_14",
                    "allow_ai_approve_small": True,
                    "approve_small_min_probability": 0.49,
                    "approve_small_min_confluence": 3.0,
                    "setup_family": "LIVE_RECOVERY_JPY_CROSS_PULLBACK",
                    "fallback_live_recovery": True,
                    "live_recovery_mode": "cross_pullback",
                    "structure_cleanliness_floor": 0.52,
                    "entry_timing_floor": 0.54,
                }
                confluence = clamp(
                    3.02
                    + min(0.34, max(0.0, effective_volume - 0.80))
                    + min(0.22, max(0.0, effective_body - 0.44))
                    + min(0.16, max(0.0, fractal_score - 0.24)),
                    0.0,
                    5.0,
                )
                score_hint = clamp(0.56 + (0.04 * (confluence - 3.0)), 0.54, 0.74)
                output: list[SignalCandidate] = []
                if cross_long:
                    output.append(
                        SignalCandidate(
                            signal_id=deterministic_id(symbol, "router", "live-cross-pullback", "BUY", timestamp),
                            setup=f"{symbol_key}_SESSION_PULLBACK",
                            side="BUY",
                            score_hint=score_hint,
                            reason="Local live recovery promoted a clean JPY cross pullback",
                            stop_atr=0.92,
                            tp_r=1.78,
                            entry_kind="SCALP",
                            strategy_family="TREND",
                            confluence_score=confluence,
                            confluence_required=3.0,
                            meta=dict(recovery_meta),
                        )
                    )
                if cross_short:
                    output.append(
                        SignalCandidate(
                            signal_id=deterministic_id(symbol, "router", "live-cross-pullback", "SELL", timestamp),
                            setup=f"{symbol_key}_SESSION_PULLBACK",
                            side="SELL",
                            score_hint=score_hint,
                            reason="Local live recovery promoted a clean JPY cross pullback",
                            stop_atr=0.92,
                            tp_r=1.78,
                            entry_kind="SCALP",
                            strategy_family="TREND",
                            confluence_score=confluence,
                            confluence_required=3.0,
                            meta=dict(recovery_meta),
                        )
                    )
                if output:
                    return output

        mean_reversion_ready = bool(
            symbol_key in {"EURUSD", "GBPUSD", "AUDJPY", "NZDJPY", "AUDNZD", "USOIL"}
            and regime_state in {"RANGING", "MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"}
            and effective_volume >= (0.84 if symbol_key in {"EURUSD", "GBPUSD"} else 0.74 if symbol_key in {"AUDJPY", "NZDJPY"} else 0.46)
            and effective_body >= (0.46 if symbol_key in {"EURUSD", "GBPUSD"} else 0.30 if symbol_key in {"AUDJPY", "NZDJPY"} else 0.40)
            and instability_score <= 0.50
            and float(row.get("m15_atr_pct_of_avg", row.get("m5_atr_pct_of_avg", 1.0)) or 1.0) <= (1.18 if symbol_key in {"AUDJPY", "NZDJPY"} else 1.22 if symbol_key == "AUDNZD" else 1.35)
            and effective_volume <= (1.22 if symbol_key in {"AUDJPY", "NZDJPY"} else 1.30 if symbol_key == "AUDNZD" else 1.45)
        )
        if not mean_reversion_ready:
            return []
        lower_reversion_edge = 0.28 if sparse_tick_volume and symbol_key in {"EURUSD", "GBPUSD"} else 0.24
        upper_reversion_edge = 0.72 if sparse_tick_volume and symbol_key in {"EURUSD", "GBPUSD"} else 0.76
        bullish_reversion = bool(
            range_position <= lower_reversion_edge
            and (
                ret_1 >= 0.0
                or ret_3 >= 0.0
                or close >= m5_ema20
                or (symbol_key == "USOIL" and h1_ema50 >= h1_ema200 and ret_3 <= -0.008)
            )
        )
        bearish_reversion = bool(
            range_position >= upper_reversion_edge
            and (
                ret_1 <= 0.0
                or ret_3 <= 0.0
                or close <= m5_ema20
                or (symbol_key in {"EURUSD", "GBPUSD"} and h1_ema50 <= h1_ema200)
            )
        )
        if not bullish_reversion and not bearish_reversion:
            return []
        if symbol_key == "AUDNZD":
            setup = "AUDNZD_RANGE_REJECTION"
        elif symbol_key in {"AUDJPY", "NZDJPY"}:
            setup = f"{symbol_key}_SWEEP_RECLAIM"
        elif symbol_key == "USOIL":
            setup = "USOIL_VWAP_REVERSION"
        else:
            setup = f"{symbol_key}_RANGE_REVERSION"
        confluence = clamp(
            2.96
            + min(0.30, max(0.0, effective_volume - (0.84 if symbol_key in {"EURUSD", "GBPUSD"} else 0.46)))
            + min(0.22, max(0.0, effective_body - (0.46 if symbol_key in {"EURUSD", "GBPUSD"} else 0.40)))
            + min(0.16, max(0.0, abs(0.5 - range_position))),
            0.0,
            5.0,
        )
        score_hint = clamp(0.54 + (0.04 * (confluence - 2.9)), 0.52, 0.70)
        reversion_meta = {
            "timeframe": "M15",
            "atr_field": "m15_atr_14",
            "allow_ai_approve_small": True,
            "approve_small_min_probability": 0.49,
            "approve_small_min_confluence": 2.95,
            "setup_family": "LIVE_RECOVERY_MEAN_REVERSION",
            "fallback_live_recovery": True,
            "live_recovery_mode": "mean_reversion",
            "structure_cleanliness_floor": 0.50,
            "entry_timing_floor": 0.52,
        }
        output: list[SignalCandidate] = []
        if bullish_reversion:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "live-range-recovery", "BUY", timestamp),
                    setup=setup,
                    side="BUY",
                    score_hint=score_hint,
                    reason="Local live recovery promoted a clean range rejection",
                    stop_atr=0.92,
                    tp_r=1.64 if symbol_key in {"AUDJPY", "NZDJPY"} else 1.62 if symbol_key in {"EURUSD", "GBPUSD"} else 1.70,
                    entry_kind="SCALP" if symbol_key != "USOIL" else "DAYTRADE",
                    strategy_family="FAKEOUT",
                    confluence_score=confluence,
                    confluence_required=2.95,
                    meta=dict(reversion_meta),
                )
            )
        if bearish_reversion:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "router", "live-range-recovery", "SELL", timestamp),
                    setup=setup,
                    side="SELL",
                    score_hint=score_hint,
                    reason="Local live recovery promoted a clean range rejection",
                    stop_atr=0.92,
                    tp_r=1.64 if symbol_key in {"AUDJPY", "NZDJPY"} else 1.62 if symbol_key in {"EURUSD", "GBPUSD"} else 1.70,
                    entry_kind="SCALP" if symbol_key != "USOIL" else "DAYTRADE",
                    strategy_family="FAKEOUT",
                    confluence_score=confluence,
                    confluence_required=2.95,
                    meta=dict(reversion_meta),
                )
            )
        return output

    def _filter_base(
        self,
        candidates: list[SignalCandidate],
        symbol: str,
        regime: RegimeClassification,
        session: SessionContext,
        row: pd.Series,
    ) -> list[SignalCandidate]:
        symbol_key = self._normalize_symbol(symbol)
        spread = float(row.get("m5_spread", 0.0))
        if spread > self._spread_reference_limit(symbol_key):
            return []
        session_name = str(session.session_name).upper()
        timestamp = row.get("time", pd.Timestamp.now(tz="UTC"))
        weekend_mode = self._is_weekend_market_mode(timestamp)
        regime_state = runtime_regime_state(str(getattr(regime, "state_label", regime.label) or regime.label))
        atr = max(float(row.get("m5_atr_14", row.get("m15_atr_14", 0.0)) or 0.0), 1e-6)
        close = float(row.get("m5_close", row.get("m15_close", 0.0)) or 0.0)
        ema20 = float(row.get("m5_ema_20", close) or close)
        distance_to_ema = abs(close - ema20)
        volume_ratio = self._ratio_or_neutral(row.get("m5_volume_ratio_20", row.get("m15_volume_ratio_20", 1.0)))
        body_efficiency = float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)) or 0.55)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        trend_flag = bool(regime.details.get("trend_flag", regime.label == "TRENDING"))
        if symbol_key in {"AUDJPY", "NZDJPY", "AUDNZD", "XAUUSD"}:
            return []
        output: list[SignalCandidate] = []
        for candidate in candidates:
            family = candidate.strategy_family.upper()
            setup_name = str(candidate.setup).upper()
            strategy_key = resolve_strategy_key(symbol_key, setup_name)
            if family == "GENERIC":
                if setup_name.find("RANGE") >= 0:
                    family = "RANGE"
                elif setup_name.find("FAKEOUT") >= 0:
                    family = "FAKEOUT"
                else:
                    family = "TREND"
            if family == "TREND" and "TREND" not in session.allowed_strategies:
                continue
            if family == "RANGE" and "RANGE" not in session.allowed_strategies:
                continue
            if family == "FAKEOUT" and "FAKEOUT" not in session.allowed_strategies:
                continue
            if (
                symbol_key in {"EURJPY", "GBPJPY"}
                and strategy_key in {"EURJPY_SESSION_PULLBACK_CONTINUATION", "GBPJPY_SESSION_PULLBACK_CONTINUATION"}
                and session_name == "TOKYO"
                and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
            ):
                continue
            if (
                symbol_key in {"EURJPY", "GBPJPY"}
                and strategy_key in {"EURJPY_MOMENTUM_IMPULSE", "GBPJPY_MOMENTUM_IMPULSE"}
                and session_name == "TOKYO"
                and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION"
                and session_name == "TOKYO"
                and regime_state == "RANGING"
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and not weekend_mode
                and strategy_key in {"BTCUSD_PRICE_ACTION_CONTINUATION", "BTCUSD_TREND_SCALP", "BTCUSD_RANGE_EXPANSION"}
                and regime_state == "MEAN_REVERSION"
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and weekend_mode
                and strategy_key in {"BTCUSD_RANGE_EXPANSION", "BTCUSD_VOLATILE_RETEST"}
                and session_name in {"SYDNEY", "TOKYO"}
                and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and weekend_mode
                and strategy_key == "BTCUSD_TREND_SCALP"
                and session_name == "SYDNEY"
                and regime_state == "MEAN_REVERSION"
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and weekend_mode
                and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION"
                and session_name == "LONDON"
                and regime_state == "MEAN_REVERSION"
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and weekend_mode
                and strategy_key == "BTCUSD_RANGE_EXPANSION"
                and session_name == "LONDON"
                and regime_state == "MEAN_REVERSION"
            ):
                continue
            if (
                symbol_key == "NAS100"
                and strategy_key == "NAS100_VWAP_TREND_STRATEGY"
                and session_name == "TOKYO"
                and regime_state == "MEAN_REVERSION"
            ):
                continue
            if (
                symbol_key == "NAS100"
                and strategy_key == "NAS100_MOMENTUM_IMPULSE"
                and session_name == "TOKYO"
                and regime_state in {"MEAN_REVERSION", "TRENDING"}
            ):
                continue
            if (
                symbol_key == "NAS100"
                and strategy_key == "NAS100_MOMENTUM_IMPULSE"
                and (
                    (session_name == "TOKYO" and regime_state == "LOW_LIQUIDITY_CHOP")
                    or (session_name == "SYDNEY" and regime_state == "LOW_LIQUIDITY_CHOP")
                    or (session_name == "LONDON" and regime_state in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"})
                    or (session_name == "OVERLAP" and regime_state in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"})
                    or (session_name == "NEW_YORK" and regime_state in {"TRENDING", "MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"})
                )
            ):
                continue
            if (
                symbol_key == "NAS100"
                and strategy_key == "NAS100_OPENING_DRIVE_BREAKOUT"
                and session_name == "TOKYO"
                and regime_state != "BREAKOUT_EXPANSION"
            ):
                continue
            if (
                symbol_key == "XAUUSD"
                and strategy_key == "XAUUSD_ADAPTIVE_M5_GRID"
                and (
                    (session_name == "LONDON" and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"})
                    or (session_name in {"OVERLAP", "NEW_YORK"} and regime_state in {"TRENDING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION", "RANGING"})
                )
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and strategy_key == "BTCUSD_VOLATILE_RETEST"
                and session_name == "TOKYO"
                and regime_state == "MEAN_REVERSION"
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and strategy_key == "BTCUSD_TREND_SCALP"
                and session_name == "TOKYO"
                and regime_state == "TRENDING"
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and strategy_key == "BTCUSD_TREND_SCALP"
                and session_name == "LONDON"
                and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and strategy_key == "BTCUSD_RANGE_EXPANSION"
                and session_name == "TOKYO"
                and regime_state == "MEAN_REVERSION"
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and strategy_key == "BTCUSD_VOLATILE_RETEST"
                and session_name == "OVERLAP"
                and regime_state == "LOW_LIQUIDITY_CHOP"
            ):
                continue
            if (
                symbol_key == "USOIL"
                and strategy_key == "USOIL_INVENTORY_MOMENTUM"
                and session_name == "TOKYO"
                and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
            ):
                continue
            if (
                symbol_key == "USOIL"
                and strategy_key == "USOIL_INVENTORY_MOMENTUM"
                and session_name == "OVERLAP"
                and regime_state == "MEAN_REVERSION"
            ):
                continue
            if (
                symbol_key == "USOIL"
                and strategy_key == "USOIL_LONDON_TREND_EXPANSION"
                and (
                    (session_name == "SYDNEY" and regime_state == "TRENDING")
                    or (session_name == "OVERLAP" and regime_state in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"})
                    or (session_name == "NEW_YORK" and regime_state == "LOW_LIQUIDITY_CHOP")
                )
            ):
                continue
            if (
                symbol_key == "EURUSD"
                and strategy_key == "EURUSD_LONDON_BREAKOUT"
                and session_name in {"OVERLAP", "NEW_YORK"}
                and regime_state == "LOW_LIQUIDITY_CHOP"
            ):
                continue
            if (
                symbol_key == "EURUSD"
                and strategy_key == "EURUSD_LONDON_BREAKOUT"
                and session_name == "NEW_YORK"
                and regime_state == "MEAN_REVERSION"
            ):
                continue
            if (
                symbol_key == "EURUSD"
                and strategy_key == "EURUSD_VWAP_PULLBACK"
                and session_name == "OVERLAP"
                and regime_state == "LOW_LIQUIDITY_CHOP"
            ):
                continue
            if (
                symbol_key == "EURUSD"
                and strategy_key == "EURUSD_VWAP_PULLBACK"
                and session_name == "NEW_YORK"
                and regime_state == "RANGING"
            ):
                continue
            if (
                symbol_key == "EURJPY"
                and strategy_key == "EURJPY_SESSION_PULLBACK_CONTINUATION"
                and session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
                and regime_state == "LOW_LIQUIDITY_CHOP"
            ):
                continue
            if (
                symbol_key == "EURJPY"
                and strategy_key == "EURJPY_SESSION_PULLBACK_CONTINUATION"
                and session_name == "LONDON"
                and regime_state == "MEAN_REVERSION"
            ):
                continue
            if (
                symbol_key == "EURJPY"
                and strategy_key in {"EURJPY_MOMENTUM_IMPULSE", "EURJPY_LIQUIDITY_SWEEP_REVERSAL"}
                and session_name == "OVERLAP"
                and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
            ):
                continue
            if (
                symbol_key == "GBPJPY"
                and strategy_key in {"GBPJPY_MOMENTUM_IMPULSE", "GBPJPY_SESSION_PULLBACK_CONTINUATION"}
                and session_name in {"SYDNEY", "LONDON", "NEW_YORK"}
                and regime_state == "LOW_LIQUIDITY_CHOP"
            ):
                continue
            if (
                symbol_key == "NZDJPY"
                and strategy_key == "NZDJPY_LIQUIDITY_TRAP_REVERSAL"
                and session_name == "OVERLAP"
                and regime_state == "MEAN_REVERSION"
            ):
                continue
            if (
                symbol_key == "NZDJPY"
                and strategy_key == "NZDJPY_SESSION_RANGE_EXPANSION"
                and session_name == "LONDON"
                and regime_state == "LOW_LIQUIDITY_CHOP"
            ):
                continue
            if (
                symbol_key == "NZDJPY"
                and strategy_key == "NZDJPY_SESSION_RANGE_EXPANSION"
                and session_name in {"OVERLAP", "NEW_YORK"}
                and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
            ):
                continue
            if (
                symbol_key == "NZDJPY"
                and strategy_key == "NZDJPY_LIQUIDITY_TRAP_REVERSAL"
                and session_name == "NEW_YORK"
                and regime_state == "LOW_LIQUIDITY_CHOP"
            ):
                continue
            if (
                symbol_key == "AUDNZD"
                and strategy_key == "AUDNZD_STRUCTURE_BREAK_RETEST"
                and (
                    (session_name == "LONDON" and regime_state == "MEAN_REVERSION")
                    or (session_name == "OVERLAP" and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"})
                    or (session_name == "NEW_YORK" and regime_state == "LOW_LIQUIDITY_CHOP")
                )
            ):
                continue
            if (
                symbol_key == "AUDJPY"
                and strategy_key == "AUDJPY_LONDON_CARRY_TREND"
                and session_name == "NEW_YORK"
                and regime_state in {"MEAN_REVERSION", "RANGING"}
            ):
                continue
            if (
                symbol_key == "AUDJPY"
                and strategy_key == "AUDJPY_ATR_COMPRESSION_BREAKOUT"
                and session_name == "OVERLAP"
                and regime_state == "MEAN_REVERSION"
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and strategy_key == "BTCUSD_VOLATILE_RETEST"
                and (
                    (session_name == "TOKYO" and regime_state == "MEAN_REVERSION")
                    or (session_name == "NEW_YORK" and regime_state == "LOW_LIQUIDITY_CHOP")
                )
            ):
                continue
            if (
                symbol_key == "BTCUSD"
                and strategy_key == "BTCUSD_PRICE_ACTION_CONTINUATION"
                and session_name == "NEW_YORK"
                and regime_state in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}
            ):
                continue
            if (
                symbol_key == "GBPUSD"
                and strategy_key == "GBPUSD_TREND_PULLBACK_RIDE"
                and (
                    (session_name == "LONDON" and regime_state in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"})
                    or (session_name == "OVERLAP" and regime_state == "LOW_LIQUIDITY_CHOP")
                )
            ):
                continue
            if symbol == "BTCUSD" and float(row.get("m5_atr_pct_of_avg", 1.0)) >= 2.7:
                continue
            if symbol == "BTCUSD" and not str(candidate.setup).upper().startswith("BTC_"):
                continue
            if setup_name in {"M1_M5_BREAKOUT", "M1_M5_EMA_IMPULSE"}:
                if symbol_key in {"EURUSD", "GBPUSD", "USDJPY"}:
                    continue
                if regime_state not in {"TRENDING", "BREAKOUT_EXPANSION"}:
                    continue
                if session_name in {"SYDNEY", "TOKYO"} and symbol_key in {"EURUSD", "GBPUSD", "USDJPY"}:
                    continue
                if volume_ratio < 1.08 or body_efficiency < 0.60 or distance_to_ema > (atr * 0.55):
                    continue
                if (str(candidate.side).upper() == "BUY" and range_position >= 0.86) or (
                    str(candidate.side).upper() == "SELL" and range_position <= 0.14
                ):
                    continue
            if setup_name == "SET_FORGET_H1_H4":
                if session_name not in {"LONDON", "OVERLAP", "NEW_YORK"}:
                    continue
                if symbol_key not in {"USDJPY", "EURUSD"}:
                    continue
                if symbol_key == "USDJPY":
                    if session_name != "OVERLAP":
                        continue
                    if not trend_flag or volume_ratio < 1.12 or body_efficiency < 0.62:
                        continue
                elif not trend_flag or volume_ratio < 1.08 or body_efficiency < 0.60:
                    continue
            output.append(candidate)
        return output

    @staticmethod
    def _candidate_unique_key(candidate: SignalCandidate) -> tuple[str, ...]:
        meta = candidate.meta if isinstance(candidate.meta, dict) else {}
        if bool(meta.get("grid_cycle")):
            cycle_id = str(meta.get("grid_cycle_id") or "")
            grid_level = str(meta.get("grid_level") or meta.get("grid_burst_index") or "")
            signal_id = str(candidate.signal_id or "")
            return ("GRID", str(candidate.setup), str(candidate.side), cycle_id, grid_level, signal_id)
        return (str(candidate.setup), str(candidate.side))

    @staticmethod
    def _unique(candidates: list[SignalCandidate]) -> list[SignalCandidate]:
        unique: list[SignalCandidate] = []
        seen: set[tuple[str, ...]] = set()
        for candidate in candidates:
            key = StrategyRouter._candidate_unique_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    @staticmethod
    def _normalize_symbol(value: str) -> str:
        compact = "".join(char for char in value.upper() if char.isalnum())
        if compact.startswith("XAUUSD") or compact.startswith("GOLD"):
            return "XAUUSD"
        if compact.startswith("BTCUSD") or compact.startswith("BTCUSDT") or compact.startswith("XBTUSD"):
            return "BTCUSD"
        if compact.startswith(("NAS100", "US100", "NASDAQ", "USTEC", "NAS", "NQ")):
            return "NAS100"
        if compact.startswith(("USOIL", "XTIUSD", "OILUSD", "WTI", "CL", "OIL", "USO")):
            return "USOIL"
        if compact.startswith("EURUSD"):
            return "EURUSD"
        if compact.startswith("GBPUSD"):
            return "GBPUSD"
        if compact.startswith("USDJPY"):
            return "USDJPY"
        if compact.startswith("EURJPY"):
            return "EURJPY"
        if compact.startswith("GBPJPY"):
            return "GBPJPY"
        if compact.startswith("AUDJPY"):
            return "AUDJPY"
        if compact.startswith("NZDJPY"):
            return "NZDJPY"
        if compact.startswith("AUDNZD"):
            return "AUDNZD"
        return compact

    @staticmethod
    def _resolve_timestamp(features: pd.DataFrame, row: pd.Series):
        value = row.get("time")
        if pd.notna(value):
            return value
        value = row.get("timestamp")
        if pd.notna(value):
            return value
        if len(features.index) > 0:
            idx_value = features.index[-1]
            if pd.notna(idx_value):
                return idx_value
        return pd.Timestamp.now(tz="UTC")
