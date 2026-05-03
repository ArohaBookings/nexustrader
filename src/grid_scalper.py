from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Callable

import pandas as pd

from src.strategy_engine import SignalCandidate
from src.trade_quality import quality_tier_from_scores
from src.utils import clamp, deterministic_id


UTC = timezone.utc


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(default)
    if pd.isna(numeric):
        return float(default)
    return float(numeric)


@dataclass
class GridScalperDecision:
    candidates: list[SignalCandidate] = field(default_factory=list)
    close_cycle: bool = False
    close_reason: str = ""
    deny_reason: str = ""
    cycle_side: str | None = None
    cycle_levels: int = 0
    ai_mode: str = "local_fallback"
    soft_penalty_reason: str = ""
    soft_penalty_score: float = 0.0
    quota_target_10m: int = 0
    quota_approved_last_10m: int = 0
    quota_debt_10m: int = 0
    quota_density_first_active: bool = False
    quota_state: str = ""
    quota_window_id: str = ""
    mc_floor: float = 0.0
    mc_win_rate: float = 0.0
    entry_profile: str = ""
    session_profile: str = ""
    density_relief_active: bool = False


@dataclass
class XAUGridScalper:
    enabled: bool = False
    symbol: str = "XAUUSD"
    timeframe: str = "M5"
    allowed_sessions: tuple[str, ...] = ("SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK")
    active_profile: str = "default"
    proof_mode: str = "default"
    checkpoint_artifact: str = ""
    density_branch_artifact: str = ""
    session_density_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)
    native_live_only: bool = True
    mirror_overlay_enabled: bool = False
    prime_burst_entries: int = 8
    moderate_burst_entries: int = 4
    aggressive_add_burst_entries: int = 6
    quota_target_actions_per_window: int = 24
    quota_min_actions_per_window: int = 16
    quota_catchup_burst_cap: int = 16
    density_first_mode: bool = True
    density_soft_penalty_max: float = 0.12
    monte_carlo_win_rate_floor: float = 0.80
    burst_window_minutes: int = 5
    prime_recovery_idle_minutes: int = 60
    prime_recovery_threshold_relax: float = 0.04
    prime_recovery_score_boost: float = 1.5
    ema_stretch_k: float = 0.75
    step_atr_k: float = 0.18
    step_points_min: float = 10.0
    step_points_max: float = 22.0
    point_size: float = 0.01
    stop_atr_k: float = 2.5
    entry_stop_atr_k: float = 2.5
    add_stop_atr_k: float = 2.5
    entry_stop_step_multiplier: float = 6.0
    add_stop_step_multiplier: float = 6.5
    entry_stop_points_min: float = 170.0
    entry_stop_points_max: float = 220.0
    add_stop_points_min: float = 190.0
    add_stop_points_max: float = 240.0
    max_levels: int = 6
    max_open_positions_symbol: int = 10
    max_open_cycles: int = 1
    base_lot: float = 0.01
    allow_mild_scale: bool = False
    probe_leg_enabled: bool = True
    probe_stretch_factor: float = 0.70
    probe_min_confluence: float = 2.8
    lot_schedule: tuple[float, ...] = ()
    profit_target_usd: float = 0.18
    micro_take_enabled: bool = True
    micro_take_usd: float = 0.06
    density_micro_scaler_enabled: bool = True
    density_micro_scaler_mc_floor_relax: float = 0.03
    prime_directional_stop_points_boost: float = 1.10
    density_micro_stop_points_boost: float = 1.06
    prime_directional_tp_r_bonus: float = 0.22
    density_micro_tp_r: float = 1.90
    max_cycle_minutes: int = 8
    time_exit_bounce_atr: float = 0.2
    time_exit_loss_usd_cap: float = 1.0
    spread_max_points: float = 50.0
    add_spread_max_points: float = 45.0
    atr_spike_threshold: float = 2.0
    volatility_pause_minutes: int = 20
    cooldown_after_stop_minutes: int = 10
    loss_streak_threshold: int = 6
    entry_deceleration_lookback: int = 4
    entry_deceleration_factor: float = 0.9
    rsi_period: int = 7
    rsi_overbought: float = 68.0
    rsi_oversold: float = 32.0
    news_block_new_cycles: bool = True
    news_block_adds: bool = True
    allow_news_high_confluence: bool = True
    news_override_min_probability: float = 0.72
    news_override_min_confluence: float = 4.0
    news_override_size_multiplier: float = 0.5
    asia_probe_enabled: bool = True
    asia_probe_sessions: tuple[str, ...] = ("TOKYO", "SYDNEY")
    asia_probe_spread_max_points: float = 18.0
    asia_probe_atr_ratio_max: float = 0.95
    asia_probe_spread_atr_ratio_cap: float = 1.20
    asia_probe_mc_floor: float = 0.88
    asia_probe_grid_max_levels: int = 2
    asia_probe_grid_lot_multiplier: float = 0.40
    london_aggressive_hours_utc: tuple[int, int] = (7, 12)
    ny_aggressive_hours_utc: tuple[int, int] = (12, 17)
    flatten_spread_points: float = 40.0
    no_progress_minutes: int = 1
    reclaim_volume_ratio_min: float = 1.0
    delta_pressure_threshold: float = 0.60
    attack_grid_spacing_multiplier: float = 0.78
    defensive_grid_spacing_multiplier: float = 1.15
    attack_grid_lot_multiplier: float = 1.10
    defensive_grid_lot_multiplier: float = 0.85
    spread_spacing_multiplier: float = 3.0
    spread_spacing_buffer_points: float = 5.0
    logger: Any | None = None
    learning_brain_bundle: dict[str, Any] = field(default_factory=dict)
    learning_brain_projection: dict[str, float] = field(default_factory=dict)
    learning_symbol_focus: bool = False
    learning_pair_directive: dict[str, Any] = field(default_factory=dict)
    learning_reentry_watchlist: tuple[str, ...] = field(default_factory=tuple)
    learning_weekly_trade_ideas: tuple[str, ...] = field(default_factory=tuple)
    _cooldown_until: datetime | None = field(default=None, init=False, repr=False)
    _volatility_pause_until: datetime | None = field(default=None, init=False, repr=False)
    _loss_streak: int = field(default=0, init=False, repr=False)
    _last_signal_emitted_at: datetime | None = field(default=None, init=False, repr=False)
    _quota_action_log: list[tuple[datetime, str, int]] = field(default_factory=list, init=False, repr=False)

    @staticmethod
    def _normalize_profile_name(value: Any, default: str = "default") -> str:
        normalized = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
        return normalized or str(default)

    @classmethod
    def _resolve_profile_config(cls, config: dict[str, Any]) -> dict[str, Any]:
        resolved = dict(config or {})
        profiles_raw = resolved.get("profiles", {})
        profiles = profiles_raw if isinstance(profiles_raw, dict) else {}
        default_profile = "checkpoint" if "checkpoint" in profiles else "default"
        active_profile = cls._normalize_profile_name(resolved.get("active_profile"), default=default_profile)
        selected_profile = profiles.get(active_profile)
        if isinstance(selected_profile, dict):
            for key, value in selected_profile.items():
                resolved[key] = value
        resolved["active_profile"] = active_profile
        resolved["proof_mode"] = cls._normalize_profile_name(resolved.get("proof_mode"), default=active_profile)
        return resolved

    @classmethod
    def from_config(cls, config: dict[str, Any], logger: Any | None = None) -> "XAUGridScalper":
        resolved_config = cls._resolve_profile_config(config)
        lot_schedule_raw = resolved_config.get("lot_schedule", [])
        if isinstance(lot_schedule_raw, list):
            lot_schedule = tuple(max(0.0, float(item)) for item in lot_schedule_raw if float(item) > 0)
        else:
            lot_schedule = ()
        sessions_raw = resolved_config.get("allowed_sessions", ["SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"])
        sessions = tuple(str(item).upper() for item in sessions_raw) if isinstance(sessions_raw, list) else ("SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK")
        asia_sessions_raw = resolved_config.get("asia_probe_sessions", ["TOKYO", "SYDNEY"])
        asia_sessions = tuple(str(item).upper() for item in asia_sessions_raw) if isinstance(asia_sessions_raw, list) else ("TOKYO", "SYDNEY")
        session_density_overrides_raw = resolved_config.get("session_density_overrides", {})
        session_density_overrides = {}
        if isinstance(session_density_overrides_raw, dict):
            for session_name, override_payload in session_density_overrides_raw.items():
                if not isinstance(override_payload, dict):
                    continue
                session_density_overrides[str(session_name).upper()] = dict(override_payload)
        return cls(
            enabled=bool(resolved_config.get("enabled", False)),
            symbol=str(resolved_config.get("symbol", "XAUUSD")).upper(),
            timeframe=str(resolved_config.get("timeframe", "M5")).upper(),
            allowed_sessions=sessions,
            active_profile=str(resolved_config.get("active_profile", "default")),
            proof_mode=str(resolved_config.get("proof_mode", resolved_config.get("active_profile", "default"))),
            checkpoint_artifact=str(resolved_config.get("checkpoint_artifact", "")),
            density_branch_artifact=str(resolved_config.get("density_branch_artifact", "")),
            session_density_overrides=session_density_overrides,
            native_live_only=bool(resolved_config.get("native_live_only", True)),
            mirror_overlay_enabled=bool(resolved_config.get("mirror_overlay_enabled", False)),
            prime_burst_entries=max(1, int(resolved_config.get("prime_burst_entries", 8))),
            moderate_burst_entries=max(1, int(resolved_config.get("moderate_burst_entries", 4))),
            aggressive_add_burst_entries=max(1, int(resolved_config.get("aggressive_add_burst_entries", 6))),
            quota_target_actions_per_window=max(1, int(resolved_config.get("quota_target_actions_per_window", 24))),
            quota_min_actions_per_window=max(1, int(resolved_config.get("quota_min_actions_per_window", 16))),
            quota_catchup_burst_cap=max(1, int(resolved_config.get("quota_catchup_burst_cap", resolved_config.get("prime_burst_entries", 16)))),
            density_first_mode=bool(resolved_config.get("density_first_mode", True)),
            density_soft_penalty_max=clamp(float(resolved_config.get("density_soft_penalty_max", 0.12)), 0.0, 0.25),
            monte_carlo_win_rate_floor=clamp(float(resolved_config.get("monte_carlo_win_rate_floor", 0.80)), 0.60, 0.95),
            burst_window_minutes=max(5, int(resolved_config.get("burst_window_minutes", 5))),
            prime_recovery_idle_minutes=max(5, int(resolved_config.get("prime_recovery_idle_minutes", 60))),
            prime_recovery_threshold_relax=clamp(float(resolved_config.get("prime_recovery_threshold_relax", 0.04)), 0.0, 0.10),
            prime_recovery_score_boost=clamp(float(resolved_config.get("prime_recovery_score_boost", 1.5)), 1.0, 2.0),
            ema_stretch_k=float(resolved_config.get("ema_stretch_k", 0.75)),
            step_atr_k=float(resolved_config.get("step_atr_k", 0.18)),
            step_points_min=float(resolved_config.get("step_points_min", 10.0)),
            step_points_max=float(resolved_config.get("step_points_max", 22.0)),
            point_size=max(1e-6, float(resolved_config.get("point_size", 0.01))),
            stop_atr_k=float(resolved_config.get("stop_atr_k", 2.5)),
            entry_stop_atr_k=float(resolved_config.get("entry_stop_atr_k", resolved_config.get("stop_atr_k", 2.5))),
            add_stop_atr_k=float(resolved_config.get("add_stop_atr_k", resolved_config.get("entry_stop_atr_k", resolved_config.get("stop_atr_k", 2.5)))),
            entry_stop_step_multiplier=float(resolved_config.get("entry_stop_step_multiplier", 6.0)),
            add_stop_step_multiplier=float(resolved_config.get("add_stop_step_multiplier", 6.5)),
            entry_stop_points_min=float(resolved_config.get("entry_stop_points_min", 170.0)),
            entry_stop_points_max=float(resolved_config.get("entry_stop_points_max", 220.0)),
            add_stop_points_min=float(resolved_config.get("add_stop_points_min", 190.0)),
            add_stop_points_max=float(resolved_config.get("add_stop_points_max", 240.0)),
            max_levels=max(1, int(resolved_config.get("max_levels", 6))),
            max_open_positions_symbol=max(1, int(resolved_config.get("max_open_positions_symbol", 10))),
            max_open_cycles=max(1, int(resolved_config.get("max_open_cycles", 1))),
            base_lot=float(resolved_config.get("base_lot", 0.01)),
            allow_mild_scale=bool(resolved_config.get("allow_mild_scale", False)),
            probe_leg_enabled=bool(resolved_config.get("probe_leg_enabled", True)),
            probe_stretch_factor=clamp(float(resolved_config.get("probe_stretch_factor", 0.70)), 0.50, 0.95),
            probe_min_confluence=clamp(float(resolved_config.get("probe_min_confluence", 2.8)), 1.5, 4.0),
            lot_schedule=lot_schedule,
            profit_target_usd=float(resolved_config.get("profit_target_usd", 0.18)),
            micro_take_enabled=bool(resolved_config.get("micro_take_enabled", True)),
            micro_take_usd=float(resolved_config.get("micro_take_usd", 0.06)),
            density_micro_scaler_enabled=bool(resolved_config.get("density_micro_scaler_enabled", True)),
            density_micro_scaler_mc_floor_relax=clamp(float(resolved_config.get("density_micro_scaler_mc_floor_relax", 0.03)), 0.0, 0.08),
            prime_directional_stop_points_boost=clamp(float(resolved_config.get("prime_directional_stop_points_boost", 1.10)), 1.0, 1.25),
            density_micro_stop_points_boost=clamp(float(resolved_config.get("density_micro_stop_points_boost", 1.06)), 1.0, 1.18),
            prime_directional_tp_r_bonus=clamp(float(resolved_config.get("prime_directional_tp_r_bonus", 0.22)), 0.0, 0.45),
            density_micro_tp_r=clamp(float(resolved_config.get("density_micro_tp_r", 1.90)), 1.5, 2.4),
            max_cycle_minutes=max(3, int(resolved_config.get("max_cycle_minutes", 8))),
            time_exit_bounce_atr=float(resolved_config.get("time_exit_bounce_atr", 0.2)),
            time_exit_loss_usd_cap=float(resolved_config.get("time_exit_loss_usd_cap", 1.0)),
            spread_max_points=float(resolved_config.get("spread_max_points", 50.0)),
            add_spread_max_points=float(resolved_config.get("add_spread_max_points", 45.0)),
            atr_spike_threshold=float(resolved_config.get("atr_spike_threshold", 2.0)),
            volatility_pause_minutes=max(1, int(resolved_config.get("volatility_pause_minutes", 20))),
            cooldown_after_stop_minutes=max(1, int(resolved_config.get("cooldown_after_stop_minutes", 10))),
            loss_streak_threshold=max(1, int(resolved_config.get("loss_streak_threshold", resolved_config.get("xau_grid_loss_streak_threshold", 6)))),
            entry_deceleration_lookback=max(3, int(resolved_config.get("entry_deceleration_lookback", 4))),
            entry_deceleration_factor=float(resolved_config.get("entry_deceleration_factor", 0.9)),
            rsi_period=max(3, int(resolved_config.get("rsi_period", 7))),
            rsi_overbought=float(resolved_config.get("rsi_overbought", 68.0)),
            rsi_oversold=float(resolved_config.get("rsi_oversold", 32.0)),
            news_block_new_cycles=bool(resolved_config.get("news_block_new_cycles", True)),
            news_block_adds=bool(resolved_config.get("news_block_adds", True)),
            allow_news_high_confluence=bool(resolved_config.get("allow_news_high_confluence", True)),
            news_override_min_probability=float(resolved_config.get("news_override_min_probability", 0.72)),
            news_override_min_confluence=float(resolved_config.get("news_override_min_confluence", 4.0)),
            news_override_size_multiplier=float(resolved_config.get("news_override_size_multiplier", 0.5)),
            asia_probe_enabled=bool(resolved_config.get("asia_probe_enabled", True)),
            asia_probe_sessions=asia_sessions,
            asia_probe_spread_max_points=float(resolved_config.get("asia_probe_spread_max_points", 18.0)),
            asia_probe_atr_ratio_max=float(resolved_config.get("asia_probe_atr_ratio_max", 0.95)),
            asia_probe_spread_atr_ratio_cap=float(resolved_config.get("asia_probe_spread_atr_ratio_cap", 1.20)),
            asia_probe_mc_floor=clamp(float(resolved_config.get("asia_probe_mc_floor", 0.88)), 0.80, 0.95),
            asia_probe_grid_max_levels=max(1, int(resolved_config.get("asia_probe_grid_max_levels", 2))),
            asia_probe_grid_lot_multiplier=clamp(float(resolved_config.get("asia_probe_grid_lot_multiplier", 0.40)), 0.25, 0.50),
            london_aggressive_hours_utc=tuple(resolved_config.get("london_aggressive_hours_utc", [7, 12])),
            ny_aggressive_hours_utc=tuple(resolved_config.get("ny_aggressive_hours_utc", [12, 17])),
            flatten_spread_points=float(resolved_config.get("flatten_spread_points", 40.0)),
            no_progress_minutes=max(1, int(resolved_config.get("no_progress_minutes", 1))),
            reclaim_volume_ratio_min=float(resolved_config.get("reclaim_volume_ratio_min", 1.0)),
            delta_pressure_threshold=clamp(float(resolved_config.get("delta_pressure_threshold", 0.60)), 0.5, 0.95),
            attack_grid_spacing_multiplier=float(resolved_config.get("attack_grid_spacing_multiplier", 0.78)),
            defensive_grid_spacing_multiplier=float(resolved_config.get("defensive_grid_spacing_multiplier", 1.15)),
            attack_grid_lot_multiplier=float(resolved_config.get("attack_grid_lot_multiplier", 1.10)),
            defensive_grid_lot_multiplier=float(resolved_config.get("defensive_grid_lot_multiplier", 0.85)),
            spread_spacing_multiplier=max(0.0, float(resolved_config.get("spread_spacing_multiplier", 3.0))),
            spread_spacing_buffer_points=max(0.0, float(resolved_config.get("spread_spacing_buffer_points", 5.0))),
            logger=logger,
        )

    def profile_state(self) -> dict[str, Any]:
        return {
            "active_profile": str(self.active_profile or "default"),
            "proof_mode": str(self.proof_mode or self.active_profile or "default"),
            "checkpoint_artifact": str(self.checkpoint_artifact or ""),
            "density_branch_artifact": str(self.density_branch_artifact or ""),
            "density_first_mode": bool(self.density_first_mode),
            "quota_target_actions_per_window": int(self.quota_target_actions_per_window),
            "quota_min_actions_per_window": int(self.quota_min_actions_per_window),
        }

    def apply_learning_policy(self, policy: dict[str, Any] | None) -> None:
        payload = dict(policy or {})
        self.learning_brain_bundle = dict(payload.get("bundle") or {})
        self.learning_brain_projection = dict(payload.get("trajectory_projection") or {})
        self.learning_symbol_focus = bool(payload.get("symbol_is_weak_focus", False))
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

    def _learning_policy_state(self, *, session_name: str, strategy_key: str = "XAUUSD_ADAPTIVE_M5_GRID") -> dict[str, Any]:
        bundle = dict(self.learning_brain_bundle or {})
        session_key = str(session_name or "").upper()
        asia_probe_session = session_key in set(self.asia_probe_sessions)
        promoted_patterns = {
            str(item).strip().upper()
            for item in bundle.get("promoted_patterns", [])
            if str(item).strip()
        }
        watchlist_items = set(self.learning_reentry_watchlist) | set(self.learning_weekly_trade_ideas)
        strategy_upper = str(strategy_key or "").strip().upper()
        watchlist_match = any("XAU" in item or strategy_upper in item for item in watchlist_items)
        promoted_pattern = bool(strategy_upper and strategy_upper in promoted_patterns)
        trajectory_catchup_pressure = clamp(float(bundle.get("quota_catchup_pressure", 0.0) or 0.0), 0.0, 1.0)
        pair_directive = dict(self.learning_pair_directive or {})
        frequency_directive = dict(pair_directive.get("frequency_directives") or {})
        aggression_multiplier = clamp(float(pair_directive.get("aggression_multiplier", 1.0) or 1.0), 0.75, 1.50)
        hot_hand_active = bool(pair_directive.get("hot_hand_active", False))
        profit_recycle_active = bool(pair_directive.get("profit_recycle_active", False))
        proof_lane_ready = bool(pair_directive.get("proof_lane_ready", False))
        idle_lane_recovery_active = bool(frequency_directive.get("idle_lane_recovery_active", False))
        density_scaling_active = bool(self.density_first_mode and str(self.active_profile or "").lower() != "checkpoint")
        checkpoint_recovery_active = bool(
            not density_scaling_active
            and str(self.active_profile or "").lower() == "checkpoint"
            and trajectory_catchup_pressure >= 0.85
            and (promoted_pattern or watchlist_match or self.learning_symbol_focus)
            and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
        )
        quota_bonus = (
            1
            if density_scaling_active
            and trajectory_catchup_pressure >= 0.65
            and (promoted_pattern or watchlist_match or self.learning_symbol_focus)
            else 0
        )
        if aggression_multiplier >= 1.15 and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            quota_bonus += 1
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and (hot_hand_active or profit_recycle_active or proof_lane_ready):
            quota_bonus += 1
        add_bonus = 1 if density_scaling_active and quota_bonus > 0 and session_key in {"LONDON", "OVERLAP", "NEW_YORK"} else 0
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and (idle_lane_recovery_active or aggression_multiplier >= 1.22):
            add_bonus += 1
        mc_floor_relax = (
            0.02
            if density_scaling_active and promoted_pattern and trajectory_catchup_pressure >= 0.75
            else (
                0.01
                if density_scaling_active and watchlist_match and trajectory_catchup_pressure >= 0.70
                else (0.01 if checkpoint_recovery_active and trajectory_catchup_pressure >= 0.90 else 0.0)
            )
        )
        size_bonus = 0.0
        if promoted_pattern:
            size_bonus += 0.05
        if watchlist_match:
            size_bonus += 0.03
        if aggression_multiplier > 1.0:
            size_bonus += min(0.04, (aggression_multiplier - 1.0) * 0.10)
        if asia_probe_session:
            checkpoint_recovery_active = False
            quota_bonus = 0
            add_bonus = 0
            mc_floor_relax = 0.0
            aggression_multiplier = min(aggression_multiplier, 1.0)
            size_bonus = min(size_bonus, 0.02)
        return {
            "promoted_pattern": bool(promoted_pattern),
            "watchlist_match": bool(watchlist_match),
            "trajectory_catchup_pressure": float(trajectory_catchup_pressure),
            "density_scaling_active": bool(density_scaling_active),
            "checkpoint_recovery_active": bool(checkpoint_recovery_active),
            "checkpoint_recovery_relax": 0.06 if checkpoint_recovery_active else 0.0,
            "quota_bonus": int(quota_bonus),
            "add_bonus": int(add_bonus),
            "mc_floor_relax": float(mc_floor_relax),
            "aggression_multiplier": float(aggression_multiplier),
            "size_bonus": clamp(float(size_bonus), 0.0, 0.10),
            "soft_burst_target_10m": int(frequency_directive.get("soft_burst_target_10m", 0) or 0),
            "hot_hand_active": bool(hot_hand_active),
            "profit_recycle_active": bool(profit_recycle_active),
            "proof_lane_ready": bool(proof_lane_ready),
            "idle_lane_recovery_active": bool(idle_lane_recovery_active),
        }

    def _session_density_config(self, *, session_name: str) -> dict[str, Any]:
        session_key = str(session_name or "").upper()
        override = dict(self.session_density_overrides.get(session_key) or {})
        learning_state = self._learning_policy_state(session_name=session_key)
        quota_bonus = int(learning_state.get("quota_bonus", 0))
        add_bonus = int(learning_state.get("add_bonus", 0))
        soft_burst_target = max(0, int(learning_state.get("soft_burst_target_10m", 0) or 0))
        if session_key in set(self.asia_probe_sessions):
            quota_bonus = 0
            add_bonus = 0
        resolved = {
            "quota_target_actions_per_window": max(
                1,
                int(override.get("quota_target_actions_per_window", self.quota_target_actions_per_window)) + quota_bonus,
            ),
            "quota_min_actions_per_window": max(
                1,
                min(
                    int(override.get("quota_target_actions_per_window", self.quota_target_actions_per_window)) + quota_bonus,
                    int(override.get("quota_min_actions_per_window", self.quota_min_actions_per_window)) + min(1, quota_bonus),
                ),
            ),
            "quota_catchup_burst_cap": max(
                1,
                int(override.get("quota_catchup_burst_cap", self.quota_catchup_burst_cap)) + quota_bonus,
            ),
            "prime_burst_entries": max(
                1,
                int(override.get("prime_burst_entries", self.prime_burst_entries)) + quota_bonus,
            ),
            "aggressive_add_burst_entries": max(
                1,
                int(override.get("aggressive_add_burst_entries", self.aggressive_add_burst_entries)) + add_bonus,
            ),
            "density_soft_penalty_max": clamp(
                float(override.get("density_soft_penalty_max", self.density_soft_penalty_max)),
                0.0,
                0.25,
            ),
        }
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and soft_burst_target > 0:
            resolved["quota_target_actions_per_window"] = max(
                int(resolved["quota_target_actions_per_window"]),
                min(soft_burst_target, int(self.quota_catchup_burst_cap) + 2),
            )
            resolved["quota_catchup_burst_cap"] = max(
                int(resolved["quota_catchup_burst_cap"]),
                min(max(soft_burst_target + 1, 6), int(self.max_levels) + 4),
            )
            resolved["prime_burst_entries"] = max(
                int(resolved["prime_burst_entries"]),
                min(max(soft_burst_target, int(self.prime_burst_entries)), int(self.max_levels) + 3),
            )
            resolved["aggressive_add_burst_entries"] = max(
                int(resolved["aggressive_add_burst_entries"]),
                min(max(soft_burst_target - 2, int(self.aggressive_add_burst_entries)), int(self.max_levels) + 1),
            )
        return resolved

    def evaluate(
        self,
        *,
        symbol: str,
        features: pd.DataFrame,
        row: pd.Series,
        open_positions: list[dict[str, Any]],
        session_name: str,
        news_safe: bool,
        now_utc: datetime,
        spread_points: float,
        contract_size: float,
        approver: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    ) -> GridScalperDecision:
        decision = GridScalperDecision()
        normalized_symbol = self._normalize_symbol(symbol)
        if (not self.enabled) or normalized_symbol != self._normalize_symbol(self.symbol):
            return decision

        if self._cooldown_until and now_utc < self._cooldown_until:
            decision.deny_reason = f"grid_cooldown_until_{self._cooldown_until.isoformat()}"
            return decision
        if self._volatility_pause_until and now_utc < self._volatility_pause_until:
            decision.deny_reason = f"grid_volatility_pause_until_{self._volatility_pause_until.isoformat()}"
            return decision

        atr = max(float(row.get("m5_atr_14", 0.0)), 1e-6)
        atr_avg = max(float(row.get("m5_atr_avg_20", 0.0)), 1e-6)
        atr_points = atr / max(self.point_size, 1e-6)
        close = float(row.get("m5_close", 0.0))
        ema20 = float(row.get("m5_ema_20", close))
        ema50 = float(row.get("m5_ema_50", close))
        cycle_positions = [position for position in open_positions if self._normalize_symbol(position.get("symbol", "")) == normalized_symbol]
        valid_cycle_positions = [
            position
            for position in cycle_positions
            if float(position.get("entry_price", 0.0) or 0.0) > 0.0
            and str(position.get("side", "")).upper() in {"BUY", "SELL"}
        ]
        if cycle_positions and not valid_cycle_positions:
            cycle_positions = []
        elif valid_cycle_positions:
            cycle_positions = valid_cycle_positions
        decision.cycle_levels = len(cycle_positions)

        atr_ratio = atr / atr_avg if atr_avg > 0 else 1.0
        session_profile = self._session_profile(
            session_name=session_name,
            now_utc=now_utc,
            atr_ratio=atr_ratio,
            spread_points=spread_points,
        )
        if session_profile == "ASIA_PROBE":
            asia_spread_limit = min(
                float(self.asia_probe_spread_max_points),
                float(atr_points * self.asia_probe_spread_atr_ratio_cap),
            )
            if spread_points > asia_spread_limit:
                decision.deny_reason = "grid_asia_probe_spread_guard"
                return decision
        add_block_reason: str | None = None
        if atr_ratio >= self.atr_spike_threshold:
            self._volatility_pause_until = now_utc + timedelta(minutes=self.volatility_pause_minutes)
            add_block_reason = "grid_add_atr_spike_pause"
        if spread_points > self.spread_max_points:
            add_block_reason = add_block_reason or "grid_spread_too_wide"

        if cycle_positions:
            return self._evaluate_existing_cycle(
                cycle_positions=cycle_positions,
                row=row,
                features=features,
                close=close,
                ema20=ema20,
                ema50=ema50,
                atr=atr,
                atr_ratio=atr_ratio,
                spread_points=spread_points,
                contract_size=contract_size,
                news_safe=news_safe,
                now_utc=now_utc,
                session_name=session_name,
                session_profile=session_profile,
                approver=approver,
                add_block_reason=add_block_reason,
            )
        if session_profile == "DISABLED":
            decision.deny_reason = "grid_session_block"
            return decision
        if add_block_reason:
            decision.deny_reason = add_block_reason
            return decision
        return self._evaluate_new_cycle(
            symbol=symbol,
            row=row,
            features=features,
            close=close,
            ema50=ema50,
            atr=atr,
            atr_ratio=atr_ratio,
            spread_points=spread_points,
            news_safe=news_safe,
            now_utc=now_utc,
            session_name=session_name,
            session_profile=session_profile,
            approver=approver,
        )

    def _record_cycle_result(self, now_utc: datetime, pnl_usd: float, *, hard_loss: bool = False) -> None:
        if float(pnl_usd) > 0.0:
            self._loss_streak = 0
            self._cooldown_until = None
            return
        if hard_loss:
            self._loss_streak += 1
        elif float(pnl_usd) < 0.0:
            self._loss_streak += 1
        else:
            self._loss_streak = max(0, self._loss_streak - 1)
        if self._loss_streak >= max(1, int(self.loss_streak_threshold)):
            self._cooldown_until = now_utc + timedelta(minutes=self.cooldown_after_stop_minutes)

    def _is_prime_density_session(self, *, session_name: str, session_profile: str) -> bool:
        session_key = str(session_name or "").upper()
        profile_key = str(session_profile or "").upper()
        if (
            profile_key == "ASIA_PROBE"
            and self.asia_probe_enabled
            and session_key in set(self.asia_probe_sessions)
            and session_key in set(self.allowed_sessions)
        ):
            return True
        return bool(
            session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
            and profile_key == "AGGRESSIVE"
            and session_key in set(self.allowed_sessions)
        )

    def _session_grid_level_cap(self, *, session_name: str, session_profile: str) -> int:
        session_key = str(session_name or "").upper()
        profile_key = str(session_profile or "").upper()
        if profile_key == "ASIA_PROBE" or session_key in set(self.asia_probe_sessions):
            return max(1, min(int(self.max_levels), int(self.asia_probe_grid_max_levels), 2))
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and profile_key == "AGGRESSIVE":
            return max(1, min(int(self.max_levels), 6))
        if profile_key == "MODERATE":
            return max(1, min(int(self.max_levels), 4))
        return max(1, min(int(self.max_levels), 5))

    def _prune_quota_action_log(self, *, now_utc: datetime) -> None:
        cutoff = now_utc.astimezone(UTC) - timedelta(minutes=max(30, int(self.burst_window_minutes) * 6))
        self._quota_action_log = [
            (ts, session, count)
            for ts, session, count in self._quota_action_log
            if ts >= cutoff
        ]

    def _quota_state(self, *, now_utc: datetime, session_name: str, session_profile: str) -> dict[str, Any]:
        session_config = self._session_density_config(session_name=session_name)
        target = max(1, int(session_config.get("quota_target_actions_per_window", self.quota_target_actions_per_window)))
        minimum = max(1, int(session_config.get("quota_min_actions_per_window", self.quota_min_actions_per_window)))
        window_id = self._burst_window_id(now_utc=now_utc, session_name=session_name)
        if not self._is_prime_density_session(session_name=session_name, session_profile=session_profile):
            return {
                "window_id": str(window_id),
                "target": int(target),
                "minimum": int(minimum),
                "approved": 0,
                "quota_debt": 0,
                "density_first_active": False,
                "state": "INACTIVE",
            }
        self._prune_quota_action_log(now_utc=now_utc)
        cutoff = now_utc.astimezone(UTC) - timedelta(minutes=max(1, int(self.burst_window_minutes)))
        approved = sum(
            int(count)
            for ts, session, count in self._quota_action_log
            if ts >= cutoff and str(session).upper() == str(session_name or "").upper()
        )
        quota_debt = max(0, target - approved)
        density_first_active = bool(self.density_first_mode and quota_debt > 0)
        state = "MET" if quota_debt <= 0 else ("CATCHUP" if density_first_active else "QUALITY_LOCK")
        return {
            "window_id": str(window_id),
            "target": int(target),
            "minimum": int(minimum),
            "approved": int(approved),
            "quota_debt": int(quota_debt),
            "density_first_active": bool(density_first_active),
            "state": str(state),
        }

    def _record_quota_actions(self, *, now_utc: datetime, session_name: str, session_profile: str, count: int) -> None:
        if count <= 0 or not self._is_prime_density_session(session_name=session_name, session_profile=session_profile):
            return
        self._prune_quota_action_log(now_utc=now_utc)
        self._quota_action_log.append(
            (
                now_utc.astimezone(UTC),
                str(session_name or "").upper(),
                max(0, int(count)),
            )
        )

    def _quota_quality_penalty(
        self,
        *,
        gate_reason: str,
        session_name: str,
        quota_debt: int,
        support_sources: int,
    ) -> float:
        if not str(gate_reason or "").strip():
            return 0.0
        session_config = self._session_density_config(session_name=session_name)
        base = 0.03 + (0.01 * max(0, int(quota_debt)))
        if str(session_name or "").upper() == "NEW_YORK":
            base += 0.02
        if int(support_sources) <= 0:
            base += 0.01
        return clamp(base, 0.0, float(session_config.get("density_soft_penalty_max", self.density_soft_penalty_max)))

    def _quota_floor_burst_count(
        self,
        *,
        burst_count: int,
        quota_state: dict[str, Any],
        session_name: str,
        session_profile: str,
        grid_mode: str,
        entry_profile: str,
        support_sources: int,
        grid_max_levels: int,
    ) -> int:
        promoted = int(burst_count)
        if not bool(quota_state.get("density_first_active")):
            return promoted
        if not self._is_prime_density_session(session_name=session_name, session_profile=session_profile):
            return promoted
        strong_profile = str(entry_profile or "").startswith(
            (
                "grid_liquidity_reclaim",
                "grid_trend_reclaim",
                "grid_m15_pullback_reclaim",
                "grid_breakout_reclaim",
                "grid_directional_flow",
                "grid_expansion_ready_scaler",
                "grid_prime_session_momentum",
            )
        )
        if not strong_profile:
            return promoted
        debt = max(0, int(quota_state.get("quota_debt", 0)))
        minimum = max(1, int(quota_state.get("minimum", self.quota_min_actions_per_window)))
        target = max(minimum, int(quota_state.get("target", self.quota_target_actions_per_window)))
        session_upper = str(session_name or "").upper()
        session_config = self._session_density_config(session_name=session_upper)
        if session_upper == "OVERLAP":
            minimum = max(3, minimum - 1)
            target = max(minimum, min(target, minimum + 1))
        elif session_upper == "NEW_YORK":
            minimum = max(minimum, 4)
        elif session_upper == "LONDON":
            minimum = max(minimum, 5)
        catchup_cap = max(
            1,
            min(
                int(grid_max_levels),
                int(session_config.get("quota_catchup_burst_cap", self.quota_catchup_burst_cap)),
                int(session_config.get("prime_burst_entries", self.prime_burst_entries)),
            ),
        )
        if debt >= 1:
            promoted = max(promoted, min(int(grid_max_levels), minimum))
        if debt >= 2 and str(grid_mode or "").upper() == "ATTACK_GRID":
            promoted = max(promoted, min(int(grid_max_levels), max(minimum, target)))
        if (
            debt >= (4 if session_upper == "OVERLAP" else 3)
            and str(grid_mode or "").upper() == "ATTACK_GRID"
            and int(support_sources) >= (2 if session_upper == "OVERLAP" else 1)
        ):
            promoted = max(promoted, min(int(grid_max_levels), catchup_cap))
        return max(1, min(int(grid_max_levels), promoted))

    def _apply_density_first_monte_carlo_burst_cap(
        self,
        *,
        burst_count: int,
        monte_carlo_win_rate: float,
        mc_floor: float,
        quota_state: dict[str, Any],
        session_name: str,
        session_profile: str,
    ) -> tuple[int, str, float]:
        current = max(1, int(burst_count))
        if str(session_profile or "").upper() != "AGGRESSIVE" or float(monte_carlo_win_rate) >= float(mc_floor):
            return current, "", 0.0
        if not (
            bool(quota_state.get("density_first_active"))
            and self._is_prime_density_session(session_name=session_name, session_profile=session_profile)
        ):
            return max(1, min(current, 2)), "", 0.0
        session_upper = str(session_name or "").upper()
        session_config = self._session_density_config(session_name=session_upper)
        quota_debt = max(0, int(quota_state.get("quota_debt", 0)))
        if session_upper == "LONDON":
            soft_floor = 5 if quota_debt >= 5 else 4
            soft_cap = int(session_config.get("prime_burst_entries", self.prime_burst_entries))
            penalty = 0.02
        elif session_upper == "NEW_YORK":
            soft_floor = 4 if quota_debt >= 4 else 3
            soft_cap = min(int(session_config.get("prime_burst_entries", self.prime_burst_entries)), 5)
            penalty = 0.03
        else:
            soft_floor = 3
            soft_cap = min(int(session_config.get("prime_burst_entries", self.prime_burst_entries)), 4)
            penalty = 0.04
        adjusted = max(soft_floor, min(current, soft_cap))
        return adjusted, "grid_monte_carlo_soft_cap", clamp(
            penalty,
            0.0,
            float(session_config.get("density_soft_penalty_max", self.density_soft_penalty_max)),
        )

    def _quota_floor_add_count(
        self,
        *,
        add_count: int,
        quota_state: dict[str, Any],
        session_name: str,
        session_profile: str,
        grid_mode: str,
        follow_through_add_ready: bool,
        support_sources: int,
        remaining_levels: int,
    ) -> int:
        promoted = max(1, int(add_count))
        if remaining_levels <= 0 or not follow_through_add_ready:
            return max(0, min(int(remaining_levels), promoted))
        if not bool(quota_state.get("density_first_active")):
            return max(1, min(int(remaining_levels), promoted))
        if not self._is_prime_density_session(session_name=session_name, session_profile=session_profile):
            return max(1, min(int(remaining_levels), promoted))
        debt = max(0, int(quota_state.get("quota_debt", 0)))
        session_upper = str(session_name or "").upper()
        session_config = self._session_density_config(session_name=session_upper)
        if session_upper == "LONDON":
            minimum = 3 if debt >= 3 else 2
            target = min(
                int(session_config.get("aggressive_add_burst_entries", self.aggressive_add_burst_entries)),
                4 if debt >= 5 and str(grid_mode or "").upper() == "ATTACK_GRID" else 3,
            )
        elif session_upper == "NEW_YORK":
            minimum = 2
            target = min(
                int(session_config.get("aggressive_add_burst_entries", self.aggressive_add_burst_entries)),
                3 if debt >= 4 and str(grid_mode or "").upper() == "ATTACK_GRID" else 2,
            )
        else:
            minimum = 2 if debt >= 3 else 1
            target = min(
                int(session_config.get("aggressive_add_burst_entries", self.aggressive_add_burst_entries)),
                3 if debt >= 5 and int(support_sources) >= 1 and str(grid_mode or "").upper() == "ATTACK_GRID" else 2,
            )
        promoted = max(promoted, minimum)
        if debt >= 2:
            promoted = max(promoted, target)
        return max(1, min(int(remaining_levels), promoted))

    def _apply_density_first_monte_carlo_add_cap(
        self,
        *,
        add_count: int,
        monte_carlo_win_rate: float,
        mc_floor: float,
        quota_state: dict[str, Any],
        session_name: str,
        session_profile: str,
        remaining_levels: int,
    ) -> tuple[int, str, float]:
        current = max(1, min(int(remaining_levels), int(add_count)))
        if str(session_profile or "").upper() != "AGGRESSIVE" or float(monte_carlo_win_rate) >= float(mc_floor):
            return current, "", 0.0
        if not (
            bool(quota_state.get("density_first_active"))
            and self._is_prime_density_session(session_name=session_name, session_profile=session_profile)
        ):
            return max(1, min(current, 2)), "", 0.0
        session_upper = str(session_name or "").upper()
        session_config = self._session_density_config(session_name=session_upper)
        quota_debt = max(0, int(quota_state.get("quota_debt", 0)))
        if session_upper == "LONDON":
            soft_floor = 3 if quota_debt >= 4 else 2
            soft_cap = int(session_config.get("aggressive_add_burst_entries", self.aggressive_add_burst_entries))
            penalty = 0.02
        elif session_upper == "NEW_YORK":
            soft_floor = 2
            soft_cap = min(int(session_config.get("aggressive_add_burst_entries", self.aggressive_add_burst_entries)), 3)
            penalty = 0.03
        else:
            soft_floor = 2 if quota_debt >= 4 else 1
            soft_cap = min(int(session_config.get("aggressive_add_burst_entries", self.aggressive_add_burst_entries)), 2)
            penalty = 0.04
        adjusted = max(soft_floor, min(current, soft_cap))
        return max(1, min(int(remaining_levels), adjusted)), "grid_add_monte_carlo_soft_cap", clamp(
            penalty,
            0.0,
            float(session_config.get("density_soft_penalty_max", self.density_soft_penalty_max)),
        )

    def _evaluate_new_cycle(
        self,
        *,
        symbol: str,
        row: pd.Series,
        features: pd.DataFrame,
        close: float,
        ema50: float,
        atr: float,
        atr_ratio: float,
        spread_points: float,
        news_safe: bool,
        now_utc: datetime,
        session_name: str,
        session_profile: str,
        approver: Callable[[dict[str, Any]], dict[str, Any]] | None,
    ) -> GridScalperDecision:
        decision = GridScalperDecision()
        high = float(row.get("m5_high", close))
        low = float(row.get("m5_low", close))
        open_price = float(row.get("m5_open", close - float(row.get("m5_body", 0.0))))
        ema20 = float(row.get("m5_ema_20", close))
        stretch = (close - ema50) / max(atr, 1e-6)
        abs_stretch = abs(stretch)
        volume_ratio = max(0.0, _finite_float(row.get("m5_volume_ratio_20", 1.0), 1.0))
        m15_volume_ratio = max(0.0, _finite_float(row.get("m15_volume_ratio_20", volume_ratio), volume_ratio))
        body_efficiency = clamp(
            abs(_finite_float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.40)), 0.40)),
            0.0,
            1.0,
        )
        compression_state = str(
            row.get("compression_proxy_state", row.get("compression_state", "NEUTRAL")) or "NEUTRAL"
        ).upper()
        compression_expansion_score = clamp(_finite_float(row.get("compression_expansion_score", 0.0), 0.0), 0.0, 1.0)
        alignment_score = clamp(_finite_float(row.get("multi_tf_alignment_score", 0.5), 0.5), 0.0, 1.0)
        fractal_score = clamp(
            _finite_float(
                row.get(
                    "fractal_persistence_score",
                    row.get("hurst_persistence_score", row.get("m5_hurst_proxy_64", 0.5)),
                ),
                0.5,
            ),
            0.0,
            1.0,
        )
        seasonality_score = clamp(_finite_float(row.get("seasonality_edge_score", 0.5), 0.5), 0.0, 1.0)
        instability_score = clamp(
            _finite_float(row.get("market_instability_score", row.get("feature_drift_score", 0.0)), 0.0),
            0.0,
            1.0,
        )
        feature_drift_score = clamp(_finite_float(row.get("feature_drift_score", 0.0), 0.0), 0.0, 1.0)
        trend_efficiency = clamp(
            _finite_float(row.get("m5_trend_efficiency_16", row.get("m5_trend_efficiency_32", 0.5)), 0.5),
            0.0,
            1.0,
        )
        if (
            trend_efficiency <= 0.02
            and alignment_score >= 0.80
            and fractal_score >= 0.80
            and body_efficiency >= 0.28
        ):
            trend_efficiency = clamp(
                0.32 + (0.18 * clamp(abs(close - ema20) / max(atr, 1e-6), 0.0, 1.0)),
                0.0,
                1.0,
            )
        range_position = clamp(
            _finite_float(row.get("m5_range_position_20", row.get("m15_range_position_20", 0.5)), 0.5),
            0.0,
            1.0,
        )
        prev_high = _finite_float(row.get("m15_rolling_high_prev_20", row.get("m5_rolling_high_prev_20", high)), high)
        prev_low = _finite_float(row.get("m15_rolling_low_prev_20", row.get("m5_rolling_low_prev_20", low)), low)
        m15_atr = max(_finite_float(row.get("m15_atr_14", atr), atr), 1e-6)
        m15_ema20 = _finite_float(row.get("m15_ema_20", close), close)
        m15_ema50 = _finite_float(row.get("m15_ema_50", row.get("m5_ema_50", close)), close)
        h1_ema50 = _finite_float(row.get("h1_ema_50", m15_ema50), m15_ema50)
        h1_ema200 = _finite_float(row.get("h1_ema_200", h1_ema50), h1_ema50)
        m15_range_position = clamp(_finite_float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)), 0.5), 0.0, 1.0)
        m15_bullish = int(row.get("m15_bullish", row.get("m5_bullish", 0))) == 1
        m15_bearish = int(row.get("m15_bearish", row.get("m5_bearish", 0))) == 1
        # Some live feeds expose m1_momentum_3 as a raw price delta. On XAU that
        # can be several whole dollars and would incorrectly trip the tiny
        # directional thresholds below. Normalize oversized raw moves back into a
        # short-term return-like signal before using them for gating.
        m1_momentum_raw = _finite_float(row.get("m1_momentum_3", row.get("m5_ret_1", 0.0)), 0.0)
        m1_momentum = float(m1_momentum_raw)
        if abs(m1_momentum_raw) > 0.20 and abs(close) >= 100.0:
            m1_momentum = m1_momentum_raw / max(abs(close), 1.0)
        m1_momentum_atr = m1_momentum_raw / max(atr, 1e-6)
        m15_momentum = _finite_float(row.get("m15_ret_1", row.get("m5_ret_5", 0.0)), 0.0)
        prev_close = close
        if len(features) >= 2 and "m5_close" in features:
            try:
                prev_close = _finite_float(features["m5_close"].iloc[-2], close)
            except Exception:
                prev_close = close
        asian_high, asian_low = self._asian_range(features, now_utc=now_utc)
        sweep_side, sweep_reason = self._detect_sweep_reclaim(
            high=high,
            low=low,
            close=close,
            asian_high=asian_high,
            asian_low=asian_low,
        )
        structure = self._structure_bias(features=features, row=row, close=close)
        deceleration = self._deceleration(features)
        rsi = _finite_float(self._rsi(features["m5_close"], self.rsi_period), 50.0)
        neutral_rsi = 40.0 <= rsi <= 60.0
        extreme = rsi >= self.rsi_overbought or rsi <= self.rsi_oversold
        strong_stretch = abs_stretch >= (self.ema_stretch_k * 1.2)
        active_session = session_profile in {"AGGRESSIVE", "MODERATE"}
        session_upper = str(session_name or "").upper()
        prime_session_force_mode = bool(
            session_upper in {"LONDON", "OVERLAP", "NEW_YORK"}
            and session_profile in {"AGGRESSIVE", "MODERATE"}
        )
        prime_recovery_active = bool(
            prime_session_force_mode
            and (
                self._last_signal_emitted_at is None
                or (now_utc - self._last_signal_emitted_at).total_seconds() >= (self.prime_recovery_idle_minutes * 60)
            )
        )
        quota_state = self._quota_state(now_utc=now_utc, session_name=session_name, session_profile=session_profile)
        decision.quota_target_10m = int(quota_state.get("target", 0))
        decision.quota_approved_last_10m = int(quota_state.get("approved", 0))
        decision.quota_debt_10m = int(quota_state.get("quota_debt", 0))
        decision.quota_density_first_active = bool(quota_state.get("density_first_active", False))
        decision.quota_state = str(quota_state.get("state", ""))
        decision.quota_window_id = str(quota_state.get("window_id", ""))
        quota_debt = max(0, int(quota_state.get("quota_debt", 0)))
        learning_state = self._learning_policy_state(session_name=session_name)
        checkpoint_recovery_active = bool(learning_state.get("checkpoint_recovery_active", False))
        idle_lane_recovery_active = bool(learning_state.get("idle_lane_recovery_active", False))
        trajectory_catchup_pressure = float(learning_state.get("trajectory_catchup_pressure", 0.0) or 0.0)
        quota_reclaim_rescue_active = bool(
            prime_session_force_mode
            and session_profile == "AGGRESSIVE"
            and (
                (
                    bool(quota_state.get("density_first_active", False))
                    and quota_debt >= (3 if session_upper == "OVERLAP" else 2)
                )
                or checkpoint_recovery_active
                or (
                    prime_recovery_active
                    and (
                        idle_lane_recovery_active
                        or trajectory_catchup_pressure >= 0.70
                    )
                )
            )
        )
        recovery_relax = clamp(
            (self.prime_recovery_threshold_relax if prime_recovery_active else 0.0)
            + float(learning_state.get("checkpoint_recovery_relax", 0.0) or 0.0),
            0.0,
            0.12,
        )
        reclaim_alignment_floor = max(0.30, 0.42 if prime_session_force_mode else 0.46 if active_session else 0.50) - recovery_relax
        reclaim_fractal_floor = max(0.28, 0.40 if prime_session_force_mode else 0.44 if active_session else 0.50) - recovery_relax
        reclaim_seasonality_floor = max(0.16, 0.25 if prime_session_force_mode else 0.30 if active_session else 0.35) - (recovery_relax * 0.75)
        reclaim_trend_efficiency_floor = max(0.20, 0.30 if prime_session_force_mode else 0.34 if active_session else 0.42) - recovery_relax
        reclaim_instability_ceiling = min(0.82, 0.70 if prime_session_force_mode else 0.62 if active_session else 0.55) + recovery_relax
        reclaim_drift_ceiling = min(0.72, 0.62 if prime_session_force_mode else 0.56 if active_session else 0.48) + recovery_relax
        reclaim_body_floor = max(0.22, 0.28 if prime_session_force_mode else 0.32 if active_session else 0.36) - (recovery_relax * 0.60)
        reclaim_volume_floor = max(
            0.74 if prime_session_force_mode else 0.82 if active_session else 0.88,
            self.reclaim_volume_ratio_min * (0.80 if prime_session_force_mode else 0.88 if active_session else 0.93),
        ) - (0.08 if prime_recovery_active else 0.0)
        pressure_floor = max(self.delta_pressure_threshold, 0.54 if active_session else 0.57)
        pressure = self._delta_pressure(
            open_price=open_price,
            close=close,
            high=high,
            low=low,
            volume_ratio=volume_ratio,
        )
        compression_ready = (
            compression_state in {"COMPRESSION", "EXPANSION_READY"}
            or compression_expansion_score >= (0.28 if session_profile == "AGGRESSIVE" else 0.32 if session_profile == "MODERATE" else 0.34)
            or (alignment_score >= 0.72 and trend_efficiency >= 0.70 and instability_score <= 0.10)
            or (
                prime_session_force_mode
                and alignment_score >= 0.80
                and fractal_score >= 0.80
                and body_efficiency >= 0.28
                and instability_score <= 0.12
                and 0.12 <= range_position <= 0.94
                and 0.12 <= m15_range_position <= 0.94
            )
        )
        follow_through_ready = bool(
            alignment_score >= reclaim_alignment_floor
            and fractal_score >= reclaim_fractal_floor
            and seasonality_score >= reclaim_seasonality_floor
            and trend_efficiency >= reclaim_trend_efficiency_floor
            and instability_score <= reclaim_instability_ceiling
            and feature_drift_score <= reclaim_drift_ceiling
            and volume_ratio >= reclaim_volume_floor
            and body_efficiency >= reclaim_body_floor
        )
        bullish_pressure = pressure >= pressure_floor
        bearish_pressure = pressure <= (1.0 - pressure_floor)
        prime_bullish_pressure = pressure >= max(0.52, pressure_floor - 0.04)
        prime_bearish_pressure = pressure <= min(0.48, (1.0 - pressure_floor) + 0.04)
        prime_edge_bullish_pressure = bool(
            prime_bullish_pressure
            or (m15_momentum >= 0.0015 and range_position >= 0.88)
            or (m15_range_position >= 0.96 and close >= ema20)
        )
        prime_edge_bearish_pressure = bool(
            prime_bearish_pressure
            or (m15_momentum <= -0.0015 and range_position <= 0.12)
            or (m15_range_position <= 0.04 and close <= ema20)
        )
        checkpoint_directional_bias_buy = bool(
            checkpoint_recovery_active
            and close >= ema20
            and ema20 >= ema50
            and body_efficiency >= 0.28
            and range_position >= 0.62
            and m15_range_position >= 0.84
            and seasonality_score >= 0.55
            and instability_score <= 0.82
            and feature_drift_score <= 0.76
            and m15_momentum >= -0.0025
        )
        checkpoint_directional_bias_sell = bool(
            checkpoint_recovery_active
            and close <= ema20
            and ema20 <= ema50
            and body_efficiency >= 0.28
            and range_position <= 0.38
            and m15_range_position <= 0.16
            and seasonality_score >= 0.55
            and instability_score <= 0.82
            and feature_drift_score <= 0.76
            and m15_momentum <= 0.0025
        )
        checkpoint_volume_ready = bool(
            volume_ratio >= 0.48
            or (
                checkpoint_recovery_active
                and volume_ratio >= 0.12
                and body_efficiency >= 0.18
                and seasonality_score >= 0.60
                and instability_score <= 0.50
            )
        )
        checkpoint_m1_buy_ok = bool(
            m1_momentum >= -0.014
            or (checkpoint_recovery_active and m1_momentum_atr >= -0.70)
        )
        checkpoint_m1_sell_ok = bool(
            m1_momentum <= 0.014
            or (checkpoint_recovery_active and m1_momentum_atr <= 0.70)
        )
        bullish_structure = bool(structure.get("buy_side", False))
        bearish_structure = bool(structure.get("sell_side", False))
        bullish_breakout = bool(
            close >= max(ema50, ema20)
            and ema20 >= ema50
            and close > prev_high
            and (close - prev_high) >= (atr * 0.10)
            and bullish_pressure
            and m1_momentum >= 0.0
            and m15_momentum >= -1e-6
        )
        bearish_breakout = bool(
            close <= min(ema50, ema20)
            and ema20 <= ema50
            and close < prev_low
            and (prev_low - close) >= (atr * 0.10)
            and bearish_pressure
            and m1_momentum <= 0.0
            and m15_momentum <= 1e-6
        )
        bullish_breakout_reclaim = bool(
            bullish_breakout
            and low <= (ema20 + (atr * 0.08))
            and 0.52 <= range_position <= 0.86
            and alignment_score >= 0.60
            and fractal_score >= 0.56
            and trend_efficiency >= 0.50
            and instability_score <= 0.26
            and feature_drift_score <= 0.24
        )
        bearish_breakout_reclaim = bool(
            bearish_breakout
            and high >= (ema20 - (atr * 0.08))
            and 0.14 <= range_position <= 0.48
            and alignment_score >= 0.60
            and fractal_score >= 0.56
            and trend_efficiency >= 0.50
            and instability_score <= 0.26
            and feature_drift_score <= 0.24
        )
        m15_pullback_ratio = abs(close - m15_ema20) / max(m15_atr, 1e-6)
        m15_near_value = abs(close - m15_ema20) <= (m15_atr * 0.48)
        m15_pullback_ok = 0.08 <= m15_pullback_ratio <= 0.60
        bullish_m15_pullback = bool(
            m15_near_value
            and m15_pullback_ok
            and body_efficiency >= 0.34
            and m15_ema20 >= m15_ema50
            and m15_range_position <= 0.64
            and m15_bullish
            and bullish_pressure
            and alignment_score >= 0.48
            and fractal_score >= 0.46
            and instability_score <= 0.58
            and seasonality_score >= 0.34
        )
        bearish_m15_pullback = bool(
            m15_near_value
            and m15_pullback_ok
            and body_efficiency >= 0.34
            and m15_ema20 <= m15_ema50
            and m15_range_position >= 0.36
            and m15_bearish
            and bearish_pressure
            and alignment_score >= 0.48
            and fractal_score >= 0.46
            and instability_score <= 0.58
            and seasonality_score >= 0.34
        )
        bullish_expansion_ready_scaler = bool(
            active_session
            and compression_state == "EXPANSION_READY"
            and compression_expansion_score >= 0.46
            and alignment_score >= 0.50
            and fractal_score >= 0.48
            and seasonality_score >= 0.32
            and trend_efficiency >= 0.36
            and instability_score <= 0.58
            and feature_drift_score <= 0.48
            and volume_ratio >= 0.80
            and body_efficiency >= 0.30
            and close >= ema20
            and bullish_pressure
            and m1_momentum >= -0.005
            and m15_momentum >= -0.008
            and 0.40 <= range_position <= 0.90
        )
        bearish_expansion_ready_scaler = bool(
            active_session
            and compression_state == "EXPANSION_READY"
            and compression_expansion_score >= 0.46
            and alignment_score >= 0.50
            and fractal_score >= 0.48
            and seasonality_score >= 0.32
            and trend_efficiency >= 0.36
            and instability_score <= 0.58
            and feature_drift_score <= 0.48
            and volume_ratio >= 0.80
            and body_efficiency >= 0.30
            and close <= ema20
            and bearish_pressure
            and m1_momentum <= 0.005
            and m15_momentum <= 0.008
            and 0.10 <= range_position <= 0.60
        )
        reclaim_buy = bool(
            close > ema20
            and low <= (ema20 + (atr * (0.14 if active_session else 0.10)))
            and bullish_pressure
            and range_position >= (0.48 if active_session else 0.52)
        )
        reclaim_sell = bool(
            close < ema20
            and high >= (ema20 - (atr * (0.14 if active_session else 0.10)))
            and bearish_pressure
            and range_position <= (0.52 if active_session else 0.48)
        )
        bullish_flow_bias = bool(
            close >= ema20
            and ema20 >= ema50
            and low <= (ema20 + (atr * (0.20 if prime_session_force_mode else 0.16 if active_session else 0.12)))
            and bullish_pressure
            and alignment_score >= (0.38 if prime_session_force_mode else 0.44 if active_session else 0.48)
            and fractal_score >= (0.36 if prime_session_force_mode else 0.42 if active_session else 0.46)
            and trend_efficiency >= (0.30 if prime_session_force_mode else 0.34 if active_session else 0.40)
            and instability_score <= (0.70 if prime_session_force_mode else 0.64 if active_session else 0.58)
            and seasonality_score >= (0.26 if prime_session_force_mode else 0.30 if active_session else 0.34)
            and body_efficiency >= (0.28 if prime_session_force_mode else 0.32 if active_session else 0.36)
            and m1_momentum >= (-0.006 if prime_session_force_mode else -0.004 if active_session else -0.002)
            and m15_momentum >= (-0.010 if prime_session_force_mode else -0.006 if active_session else -0.004)
            and (0.34 if prime_session_force_mode else 0.42 if active_session else 0.46) <= range_position <= (0.94 if prime_session_force_mode else 0.97)
            and m15_range_position <= (0.94 if prime_session_force_mode else 0.94)
        )
        bearish_flow_bias = bool(
            close <= ema20
            and ema20 <= ema50
            and high >= (ema20 - (atr * (0.20 if prime_session_force_mode else 0.16 if active_session else 0.12)))
            and bearish_pressure
            and alignment_score >= (0.38 if prime_session_force_mode else 0.44 if active_session else 0.48)
            and fractal_score >= (0.36 if prime_session_force_mode else 0.42 if active_session else 0.46)
            and trend_efficiency >= (0.30 if prime_session_force_mode else 0.34 if active_session else 0.40)
            and instability_score <= (0.70 if prime_session_force_mode else 0.64 if active_session else 0.58)
            and seasonality_score >= (0.26 if prime_session_force_mode else 0.30 if active_session else 0.34)
            and body_efficiency >= (0.28 if prime_session_force_mode else 0.32 if active_session else 0.36)
            and m1_momentum <= (0.006 if prime_session_force_mode else 0.004 if active_session else 0.002)
            and m15_momentum <= (0.010 if prime_session_force_mode else 0.006 if active_session else 0.004)
            and (0.06 if prime_session_force_mode else 0.03) <= range_position <= (0.66 if prime_session_force_mode else 0.58 if active_session else 0.54)
            and m15_range_position >= (0.06 if prime_session_force_mode else 0.06)
        )
        native_mirror_ready = bool(
            active_session
            and compression_ready
            and volume_ratio >= (0.76 if prime_session_force_mode else 0.88)
            and body_efficiency >= (0.26 if prime_session_force_mode else 0.30)
            and alignment_score >= (0.40 if prime_session_force_mode else 0.44)
            and fractal_score >= (0.38 if prime_session_force_mode else 0.42)
            and seasonality_score >= (0.26 if prime_session_force_mode else 0.30)
            and instability_score <= (0.68 if prime_session_force_mode else 0.62)
            and feature_drift_score <= (0.60 if prime_session_force_mode else 0.56)
        )
        native_directional_mirror_buy = bool(
            native_mirror_ready
            and close >= ema20
            and ema20 >= ema50
            and m15_ema20 >= m15_ema50
            and bullish_pressure
            and m1_momentum >= (-0.006 if prime_session_force_mode else -0.004)
            and m15_momentum >= (-0.010 if prime_session_force_mode else -0.008)
            and range_position >= (0.34 if prime_session_force_mode else 0.40)
            and m15_range_position <= (0.94 if prime_session_force_mode else 0.76)
        )
        native_directional_mirror_sell = bool(
            native_mirror_ready
            and close <= ema20
            and ema20 <= ema50
            and m15_ema20 <= m15_ema50
            and bearish_pressure
            and m1_momentum <= (0.006 if prime_session_force_mode else 0.004)
            and m15_momentum <= (0.010 if prime_session_force_mode else 0.008)
            and range_position <= (0.66 if prime_session_force_mode else 0.60)
            and m15_range_position >= (0.06 if prime_session_force_mode else 0.24)
        )
        native_pullback_mirror_buy = bool(
            native_mirror_ready
            and m15_near_value
            and m15_pullback_ratio <= (0.82 if prime_session_force_mode else 0.72)
            and close >= ema20
            and m15_ema20 >= m15_ema50
            and bullish_pressure
            and (0.20 if prime_session_force_mode else 0.24) <= m15_range_position <= (0.86 if prime_session_force_mode else 0.74)
        )
        native_pullback_mirror_sell = bool(
            native_mirror_ready
            and m15_near_value
            and m15_pullback_ratio <= (0.82 if prime_session_force_mode else 0.72)
            and close <= ema20
            and m15_ema20 <= m15_ema50
            and bearish_pressure
            and (0.14 if prime_session_force_mode else 0.26) <= m15_range_position <= (0.86 if prime_session_force_mode else 0.76)
        )
        prime_directional_rescue_buy = bool(
            prime_session_force_mode
            and (
                compression_ready
                or follow_through_ready
                or strong_stretch
                or quota_reclaim_rescue_active
                or idle_lane_recovery_active
                or checkpoint_recovery_active
            )
            and close >= ema20
            and (ema20 >= ema50 or trend_efficiency >= 0.20)
            and prime_bullish_pressure
            and volume_ratio >= (0.54 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.60)
            and body_efficiency >= (0.12 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.18)
            and alignment_score >= (0.20 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.26)
            and fractal_score >= (0.18 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.24)
            and trend_efficiency >= (0.12 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.18)
            and instability_score <= 0.84
            and feature_drift_score <= 0.76
            and m1_momentum >= -0.012
            and m15_momentum >= -0.016
            and (range_position >= 0.22 or close >= (prev_high - (atr * 0.28)))
            and m15_range_position <= 0.94
        )
        prime_directional_rescue_sell = bool(
            prime_session_force_mode
            and (
                compression_ready
                or follow_through_ready
                or strong_stretch
                or quota_reclaim_rescue_active
                or idle_lane_recovery_active
                or checkpoint_recovery_active
            )
            and close <= ema20
            and (ema20 <= ema50 or trend_efficiency >= 0.20)
            and prime_bearish_pressure
            and volume_ratio >= (0.54 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.60)
            and body_efficiency >= (0.12 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.18)
            and alignment_score >= (0.20 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.26)
            and fractal_score >= (0.18 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.24)
            and trend_efficiency >= (0.12 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.18)
            and instability_score <= 0.84
            and feature_drift_score <= 0.76
            and m1_momentum <= 0.012
            and m15_momentum <= 0.016
            and (range_position <= 0.78 or close <= (prev_low + (atr * 0.28)))
            and m15_range_position >= 0.06
        )
        # Prime-session forcing logic for live XAU grid. When the normal reclaim
        # stack is close but not fully green, allow a tighter momentum/reclaim
        # start instead of sitting idle through London or overlap/NY.
        prime_momentum_fallback_buy = bool(
            prime_session_force_mode
            and (
                compression_ready
                or follow_through_ready
                or strong_stretch
                or quota_reclaim_rescue_active
                or idle_lane_recovery_active
                or checkpoint_recovery_active
            )
            and close > prev_close
            and close >= ema20
            and low <= (ema20 + (atr * 0.20))
            and prime_bullish_pressure
            and body_efficiency >= (0.12 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.18)
            and volume_ratio >= (0.54 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.62)
            and alignment_score >= (0.22 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.28)
            and fractal_score >= (0.18 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.24)
            and instability_score <= (0.86 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.80)
            and feature_drift_score <= (0.80 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.74)
            and (range_position >= 0.24 or close >= (prev_high - (atr * 0.18)))
        )
        prime_momentum_fallback_sell = bool(
            prime_session_force_mode
            and (
                compression_ready
                or follow_through_ready
                or strong_stretch
                or quota_reclaim_rescue_active
                or idle_lane_recovery_active
                or checkpoint_recovery_active
            )
            and close < prev_close
            and close <= ema20
            and high >= (ema20 - (atr * 0.20))
            and prime_bearish_pressure
            and body_efficiency >= (0.12 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.18)
            and volume_ratio >= (0.54 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.62)
            and alignment_score >= (0.22 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.28)
            and fractal_score >= (0.18 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.24)
            and instability_score <= (0.86 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.80)
            and feature_drift_score <= (0.80 if (quota_reclaim_rescue_active or idle_lane_recovery_active or checkpoint_recovery_active) else 0.74)
            and (range_position <= 0.76 or close <= (prev_low + (atr * 0.18)))
        )
        prime_session_scaler_buy = bool(
            prime_session_force_mode
            and (
                compression_ready
                or strong_stretch
                or deceleration
                or (alignment_score >= (0.16 if quota_reclaim_rescue_active else 0.22) and body_efficiency >= (0.10 if quota_reclaim_rescue_active else 0.14))
                or checkpoint_directional_bias_buy
                or quota_reclaim_rescue_active
            )
            and (close >= ema20 or close >= ema50)
            and (ema20 >= ema50 or trend_efficiency >= (0.08 if quota_reclaim_rescue_active else 0.14))
            and (prime_bullish_pressure or checkpoint_directional_bias_buy)
            and checkpoint_volume_ready
            and body_efficiency >= (0.08 if quota_reclaim_rescue_active else 0.12)
            and alignment_score >= (0.12 if quota_reclaim_rescue_active else 0.18)
            and fractal_score >= (0.10 if quota_reclaim_rescue_active else 0.16)
            and trend_efficiency >= (0.08 if quota_reclaim_rescue_active else 0.12)
            and instability_score <= (0.92 if quota_reclaim_rescue_active else 0.88)
            and feature_drift_score <= (0.86 if quota_reclaim_rescue_active else 0.82)
            and checkpoint_m1_buy_ok
            and m15_momentum >= (-0.028 if quota_reclaim_rescue_active else -0.020)
            and range_position >= (0.10 if quota_reclaim_rescue_active else 0.16)
            and m15_range_position <= (0.99 if quota_reclaim_rescue_active else (0.98 if checkpoint_recovery_active else 0.94))
            and not (
                quota_reclaim_rescue_active
                and range_position >= 0.94
                and m15_range_position >= 0.94
                and m1_momentum >= 0.004
                and m15_momentum >= 0.006
            )
        )
        prime_session_scaler_sell = bool(
            prime_session_force_mode
            and (
                compression_ready
                or strong_stretch
                or deceleration
                or (alignment_score >= (0.16 if quota_reclaim_rescue_active else 0.22) and body_efficiency >= (0.10 if quota_reclaim_rescue_active else 0.14))
                or checkpoint_directional_bias_sell
                or quota_reclaim_rescue_active
            )
            and (close <= ema20 or close <= ema50)
            and (ema20 <= ema50 or trend_efficiency >= (0.08 if quota_reclaim_rescue_active else 0.14))
            and (prime_bearish_pressure or checkpoint_directional_bias_sell)
            and checkpoint_volume_ready
            and body_efficiency >= (0.08 if quota_reclaim_rescue_active else 0.12)
            and alignment_score >= (0.12 if quota_reclaim_rescue_active else 0.18)
            and fractal_score >= (0.10 if quota_reclaim_rescue_active else 0.16)
            and trend_efficiency >= (0.08 if quota_reclaim_rescue_active else 0.12)
            and instability_score <= (0.92 if quota_reclaim_rescue_active else 0.88)
            and feature_drift_score <= (0.86 if quota_reclaim_rescue_active else 0.82)
            and checkpoint_m1_sell_ok
            and m15_momentum <= (0.028 if quota_reclaim_rescue_active else 0.020)
            and range_position <= (0.90 if quota_reclaim_rescue_active else 0.84)
            and m15_range_position >= (0.01 if quota_reclaim_rescue_active else (0.02 if checkpoint_recovery_active else 0.06))
            and not (
                quota_reclaim_rescue_active
                and range_position <= 0.06
                and m15_range_position <= 0.06
                and m1_momentum <= -0.004
                and m15_momentum <= -0.006
            )
        )
        prime_extreme_htf_buy = bool(
            h1_ema50 >= h1_ema200
            and max(volume_ratio, m15_volume_ratio) >= 0.80
            and body_efficiency >= 0.32
            and m15_range_position >= 0.92
        )
        prime_extreme_htf_sell = bool(
            h1_ema50 <= h1_ema200
            and max(volume_ratio, m15_volume_ratio) >= 0.80
            and body_efficiency >= 0.32
            and m15_range_position <= 0.08
        )
        prime_extreme_alignment_floor_buy = 0.22 if prime_extreme_htf_buy else 0.42
        prime_extreme_alignment_floor_sell = 0.22 if prime_extreme_htf_sell else 0.42
        prime_extreme_continuation_buy = bool(
            prime_session_force_mode
            and (strong_stretch or follow_through_ready or compression_state in {"EXPANSION_READY", "NEUTRAL"})
            and close >= ema20
            and ema20 >= ema50
            and m15_ema20 >= m15_ema50
            and prime_edge_bullish_pressure
            and max(volume_ratio, m15_volume_ratio * 0.74) >= 0.60
            and body_efficiency >= 0.24
            and alignment_score >= prime_extreme_alignment_floor_buy
            and fractal_score >= 0.28
            and seasonality_score >= 0.22
            and instability_score <= 0.44
            and feature_drift_score <= 0.34
            and m1_momentum >= -0.016
            and m15_momentum >= -0.020
            and range_position >= 0.18
            and m15_range_position >= 0.92
            and low <= (ema20 + (atr * 0.20))
            and close >= (prev_high - (atr * 0.22))
        )
        prime_extreme_continuation_sell = bool(
            prime_session_force_mode
            and (strong_stretch or follow_through_ready or compression_state in {"EXPANSION_READY", "NEUTRAL"})
            and close <= ema20
            and ema20 <= ema50
            and m15_ema20 <= m15_ema50
            and prime_edge_bearish_pressure
            and max(volume_ratio, m15_volume_ratio * 0.74) >= 0.60
            and body_efficiency >= 0.24
            and alignment_score >= prime_extreme_alignment_floor_sell
            and fractal_score >= 0.28
            and seasonality_score >= 0.22
            and instability_score <= 0.44
            and feature_drift_score <= 0.34
            and m1_momentum <= 0.016
            and m15_momentum <= 0.020
            and range_position <= 0.82
            and m15_range_position <= 0.08
            and close <= (prev_low + (atr * 0.22))
        )
        quota_rescue_relax = 0.08 if checkpoint_recovery_active else 0.0
        quota_rescue_volume_floor = max(
            0.34,
            (0.44 if session_upper == "LONDON" else 0.50 if session_upper == "NEW_YORK" else 0.56) - quota_rescue_relax,
        )
        quota_rescue_body_floor = max(
            0.06,
            (0.08 if session_upper == "LONDON" else 0.12 if session_upper == "NEW_YORK" else 0.16) - (quota_rescue_relax * 0.50),
        )
        quota_rescue_alignment_floor = max(
            0.10,
            (0.14 if session_upper == "LONDON" else 0.20 if session_upper == "NEW_YORK" else 0.26) - quota_rescue_relax,
        )
        quota_rescue_fractal_floor = max(
            0.10,
            (0.12 if session_upper == "LONDON" else 0.18 if session_upper == "NEW_YORK" else 0.24) - quota_rescue_relax,
        )
        quota_rescue_trend_floor = max(
            0.06,
            (0.08 if session_upper == "LONDON" else 0.12 if session_upper == "NEW_YORK" else 0.18) - (quota_rescue_relax * 0.50),
        )
        quota_rescue_instability_ceiling = min(
            0.98,
            (0.92 if session_upper == "LONDON" else 0.86 if session_upper == "NEW_YORK" else 0.76) + quota_rescue_relax,
        )
        quota_rescue_drift_ceiling = min(
            0.94,
            (0.86 if session_upper == "LONDON" else 0.78 if session_upper == "NEW_YORK" else 0.70) + quota_rescue_relax,
        )
        quota_rescue_range_floor = 0.12 if session_upper == "LONDON" else 0.18 if session_upper == "NEW_YORK" else 0.26
        quota_rescue_range_ceiling = 0.88 if session_upper == "LONDON" else 0.82 if session_upper == "NEW_YORK" else 0.74
        quota_rescue_m1_floor = -0.022 if session_upper == "LONDON" else -0.016 if session_upper == "NEW_YORK" else -0.012
        quota_rescue_m15_floor = -0.028 if session_upper == "LONDON" else -0.020 if session_upper == "NEW_YORK" else -0.016
        quota_rescue_pullback_atr = 0.28 if session_upper == "LONDON" else 0.24 if session_upper == "NEW_YORK" else 0.20
        quota_reclaim_rescue_buy = bool(
            quota_reclaim_rescue_active
            and (compression_ready or follow_through_ready or strong_stretch or deceleration)
            and (close >= ema20 or (session_upper == "LONDON" and close >= ema50))
            and (ema20 >= ema50 or trend_efficiency >= quota_rescue_trend_floor)
            and (reclaim_buy or low <= (ema20 + (atr * quota_rescue_pullback_atr)))
            and prime_bullish_pressure
            and volume_ratio >= quota_rescue_volume_floor
            and body_efficiency >= quota_rescue_body_floor
            and alignment_score >= quota_rescue_alignment_floor
            and fractal_score >= quota_rescue_fractal_floor
            and trend_efficiency >= quota_rescue_trend_floor
            and instability_score <= quota_rescue_instability_ceiling
            and feature_drift_score <= quota_rescue_drift_ceiling
            and m1_momentum >= quota_rescue_m1_floor
            and m15_momentum >= quota_rescue_m15_floor
            and quota_rescue_range_floor <= range_position <= 0.96
            and m15_range_position <= 0.96
        )
        quota_reclaim_rescue_sell = bool(
            quota_reclaim_rescue_active
            and (compression_ready or follow_through_ready or strong_stretch or deceleration)
            and (close <= ema20 or (session_upper == "LONDON" and close <= ema50))
            and (ema20 <= ema50 or trend_efficiency >= quota_rescue_trend_floor)
            and (reclaim_sell or high >= (ema20 - (atr * quota_rescue_pullback_atr)))
            and prime_bearish_pressure
            and volume_ratio >= quota_rescue_volume_floor
            and body_efficiency >= quota_rescue_body_floor
            and alignment_score >= quota_rescue_alignment_floor
            and fractal_score >= quota_rescue_fractal_floor
            and trend_efficiency >= quota_rescue_trend_floor
            and instability_score <= quota_rescue_instability_ceiling
            and feature_drift_score <= quota_rescue_drift_ceiling
            and m1_momentum <= abs(quota_rescue_m1_floor)
            and m15_momentum <= abs(quota_rescue_m15_floor)
            and 0.04 <= range_position <= quota_rescue_range_ceiling
            and m15_range_position >= 0.04
        )
        london_open_recovery_active = bool(
            session_upper == "LONDON"
            and session_profile in {"MODERATE", "AGGRESSIVE"}
            and (
                quota_debt > 0
                or prime_recovery_active
                or idle_lane_recovery_active
                or checkpoint_recovery_active
                or trajectory_catchup_pressure >= 0.18
            )
        )
        london_open_recovery_buy = bool(
            london_open_recovery_active
            and sweep_side != "SELL"
            and close >= min(ema20, ema50)
            and (
                prime_bullish_pressure
                or bullish_pressure
                or pressure >= 0.50
                or m1_momentum >= -0.006
            )
            and max(volume_ratio, m15_volume_ratio * 0.80) >= 0.26
            and body_efficiency >= 0.03
            and alignment_score >= 0.02
            and fractal_score >= 0.02
            and trend_efficiency >= 0.03
            and instability_score <= 0.92
            and feature_drift_score <= 0.82
            and 0.10 <= range_position <= 0.96
            and 0.08 <= m15_range_position <= 0.98
            and m15_momentum >= -0.012
            and not (strong_stretch and bearish_pressure and range_position >= 0.90)
        )
        london_open_recovery_sell = bool(
            london_open_recovery_active
            and sweep_side != "BUY"
            and close <= max(ema20, ema50)
            and (
                prime_bearish_pressure
                or bearish_pressure
                or pressure <= 0.50
                or m1_momentum <= 0.006
            )
            and max(volume_ratio, m15_volume_ratio * 0.80) >= 0.26
            and body_efficiency >= 0.03
            and alignment_score >= 0.02
            and fractal_score >= 0.02
            and trend_efficiency >= 0.03
            and instability_score <= 0.92
            and feature_drift_score <= 0.82
            and 0.04 <= range_position <= 0.90
            and 0.02 <= m15_range_position <= 0.92
            and m15_momentum <= 0.012
            and not (strong_stretch and bullish_pressure and range_position <= 0.10)
        )
        stretch_reversion_buy = bool(
            session_profile != "ASIA_PROBE"
            and stretch <= -(self.ema_stretch_k * (0.85 if prime_session_force_mode else 1.0))
            and deceleration
            and bullish_pressure
            and close >= open_price
            and (neutral_rsi or rsi <= (self.rsi_oversold + 8.0))
            and volume_ratio >= 0.78
            and body_efficiency >= 0.24
            and instability_score <= 0.74
            and feature_drift_score <= 0.70
            and range_position <= 0.62
        )
        stretch_reversion_sell = bool(
            session_profile != "ASIA_PROBE"
            and stretch >= (self.ema_stretch_k * (0.85 if prime_session_force_mode else 1.0))
            and deceleration
            and bearish_pressure
            and close <= open_price
            and (neutral_rsi or rsi >= (self.rsi_overbought - 8.0))
            and volume_ratio >= 0.78
            and body_efficiency >= 0.24
            and instability_score <= 0.74
            and feature_drift_score <= 0.70
            and range_position >= 0.38
        )
        prime_stretch_reversal_buy = bool(
            prime_session_force_mode
            and stretch <= -(self.ema_stretch_k * 1.0)
            and close >= open_price
            and bullish_pressure
            and body_efficiency >= 0.32
            and volume_ratio >= 0.88
            and alignment_score >= 0.40
            and fractal_score >= 0.32
            and instability_score <= 0.52
            and feature_drift_score <= 0.28
            and range_position <= 0.24
            and low <= (prev_low + (atr * 0.28))
            and m15_momentum >= -0.004
        )
        prime_stretch_reversal_sell = bool(
            prime_session_force_mode
            and stretch >= (self.ema_stretch_k * 1.0)
            and close <= open_price
            and bearish_pressure
            and body_efficiency >= 0.32
            and volume_ratio >= 0.88
            and alignment_score >= 0.40
            and fractal_score >= 0.32
            and instability_score <= 0.52
            and feature_drift_score <= 0.28
            and range_position >= 0.76
            and high >= (prev_high - (atr * 0.28))
            and m15_momentum <= 0.004
        )
        asia_relief_bias = bool(
            session_profile == "ASIA_PROBE"
            and session_upper in set(self.asia_probe_sessions)
            and (
                bool(quota_state.get("density_first_active", False))
                or quota_debt > 0
                or prime_recovery_active
                or idle_lane_recovery_active
                or trajectory_catchup_pressure >= 0.32
            )
        )
        prime_exhaustion_probe_buy = bool(
            prime_session_force_mode
            and stretch <= -(self.ema_stretch_k * 1.35)
            and rsi <= (self.rsi_oversold + 4.0)
            and range_position <= 0.26
            and max(volume_ratio, m15_volume_ratio * 0.90) >= 0.34
            and body_efficiency >= 0.08
            and alignment_score >= 0.22
            and fractal_score >= 0.48
            and instability_score <= 0.56
            and feature_drift_score <= 0.52
            and m15_momentum >= -0.015
            and (
                bullish_pressure
                or deceleration
                or close >= (low + max((high - low) * 0.22, atr * 0.04))
            )
            and (
                low <= (prev_low + (atr * 0.40))
                or clamp(_finite_float(row.get("predicted_liquidity_hunt_score", 0.0), 0.0), 0.0, 1.0) >= 0.85
            )
        )
        prime_exhaustion_probe_sell = bool(
            prime_session_force_mode
            and stretch >= (self.ema_stretch_k * 1.35)
            and rsi >= (self.rsi_overbought - 4.0)
            and range_position >= 0.74
            and max(volume_ratio, m15_volume_ratio * 0.90) >= 0.34
            and body_efficiency >= 0.08
            and alignment_score >= 0.22
            and fractal_score >= 0.48
            and instability_score <= 0.56
            and feature_drift_score <= 0.52
            and m15_momentum <= 0.015
            and (
                bearish_pressure
                or deceleration
                or close <= (high - max((high - low) * 0.22, atr * 0.04))
            )
            and (
                high >= (prev_high - (atr * 0.40))
                or clamp(_finite_float(row.get("predicted_liquidity_hunt_score", 0.0), 0.0), 0.0, 1.0) >= 0.85
            )
        )
        asia_probe_directional_buy = bool(
            session_profile == "ASIA_PROBE"
            and (
                sweep_side == "BUY"
                or (
                    sweep_side is None
                    and compression_ready
                    and (
                        bullish_pressure
                        or prime_bullish_pressure
                        or deceleration
                        or strong_stretch
                        or (
                            asia_relief_bias
                            and close >= ema20
                            and pressure >= 0.50
                            and m15_momentum >= -0.004
                        )
                    )
                )
            )
            and close >= ema20
            and (ema20 >= ema50 or trend_efficiency >= max(0.22, reclaim_trend_efficiency_floor - 0.14))
            and volume_ratio >= max(0.56 if asia_relief_bias else 0.62, reclaim_volume_floor - (0.24 if asia_relief_bias else 0.18))
            and body_efficiency >= max(0.12 if asia_relief_bias else 0.16, reclaim_body_floor - (0.22 if asia_relief_bias else 0.16))
            and alignment_score >= max(0.14 if asia_relief_bias else 0.20, reclaim_alignment_floor - (0.24 if asia_relief_bias else 0.18))
            and fractal_score >= max(0.12 if asia_relief_bias else 0.18, reclaim_fractal_floor - (0.26 if asia_relief_bias else 0.20))
            and trend_efficiency >= max(0.16 if asia_relief_bias else 0.22, reclaim_trend_efficiency_floor - (0.20 if asia_relief_bias else 0.14))
            and instability_score <= min(0.90 if asia_relief_bias else 0.86, reclaim_instability_ceiling + (0.18 if asia_relief_bias else 0.14))
            and feature_drift_score <= min(0.86 if asia_relief_bias else 0.82, reclaim_drift_ceiling + (0.22 if asia_relief_bias else 0.18))
            and range_position >= 0.18
            and m15_range_position >= 0.14
            and m1_momentum >= (-0.030 if asia_relief_bias else -0.020)
            and m15_momentum >= (-0.036 if asia_relief_bias else -0.028)
        )
        asia_probe_directional_sell = bool(
            session_profile == "ASIA_PROBE"
            and (
                sweep_side == "SELL"
                or (
                    sweep_side is None
                    and compression_ready
                    and (
                        bearish_pressure
                        or prime_bearish_pressure
                        or deceleration
                        or strong_stretch
                        or (
                            asia_relief_bias
                            and close <= ema20
                            and pressure <= 0.50
                            and m15_momentum <= 0.004
                        )
                    )
                )
            )
            and close <= ema20
            and (ema20 <= ema50 or trend_efficiency >= max(0.22, reclaim_trend_efficiency_floor - 0.14))
            and volume_ratio >= max(0.56 if asia_relief_bias else 0.62, reclaim_volume_floor - (0.24 if asia_relief_bias else 0.18))
            and body_efficiency >= max(0.12 if asia_relief_bias else 0.16, reclaim_body_floor - (0.22 if asia_relief_bias else 0.16))
            and alignment_score >= max(0.14 if asia_relief_bias else 0.20, reclaim_alignment_floor - (0.24 if asia_relief_bias else 0.18))
            and fractal_score >= max(0.12 if asia_relief_bias else 0.18, reclaim_fractal_floor - (0.26 if asia_relief_bias else 0.20))
            and trend_efficiency >= max(0.16 if asia_relief_bias else 0.22, reclaim_trend_efficiency_floor - (0.20 if asia_relief_bias else 0.14))
            and instability_score <= min(0.90 if asia_relief_bias else 0.86, reclaim_instability_ceiling + (0.18 if asia_relief_bias else 0.14))
            and feature_drift_score <= min(0.86 if asia_relief_bias else 0.82, reclaim_drift_ceiling + (0.22 if asia_relief_bias else 0.18))
            and range_position <= 0.82
            and m15_range_position <= 0.86
            and m1_momentum <= (0.030 if asia_relief_bias else 0.020)
            and m15_momentum <= (0.036 if asia_relief_bias else 0.028)
        )
        asia_probe_micro_scaler_buy = bool(
            session_profile == "ASIA_PROBE"
            and sweep_side != "SELL"
            and (
                bullish_pressure
                or pressure >= (0.50 if asia_relief_bias else 0.54)
                or m1_momentum >= (0.000 if asia_relief_bias else 0.008)
                or m15_momentum >= (0.000 if asia_relief_bias else 0.010)
                or (asia_relief_bias and close >= ema20 and m15_range_position >= 0.12)
            )
            and close >= min(ema20, ema50)
            and volume_ratio >= max(0.46 if asia_relief_bias else 0.54, reclaim_volume_floor - (0.32 if asia_relief_bias else 0.24))
            and body_efficiency >= max(0.06 if asia_relief_bias else 0.10, reclaim_body_floor - (0.28 if asia_relief_bias else 0.22))
            and alignment_score >= max(0.08 if asia_relief_bias else 0.12, reclaim_alignment_floor - (0.30 if asia_relief_bias else 0.24))
            and fractal_score >= max(0.06 if asia_relief_bias else 0.10, reclaim_fractal_floor - (0.32 if asia_relief_bias else 0.26))
            and trend_efficiency >= max(0.10 if asia_relief_bias else 0.16, reclaim_trend_efficiency_floor - (0.26 if asia_relief_bias else 0.20))
            and instability_score <= min(0.94 if asia_relief_bias else 0.90, reclaim_instability_ceiling + (0.22 if asia_relief_bias else 0.18))
            and feature_drift_score <= min(0.90 if asia_relief_bias else 0.86, reclaim_drift_ceiling + (0.26 if asia_relief_bias else 0.22))
            and range_position >= (0.04 if asia_relief_bias else 0.08)
            and m15_range_position >= (0.04 if asia_relief_bias else 0.08)
            and not strong_stretch
        )
        asia_probe_micro_scaler_sell = bool(
            session_profile == "ASIA_PROBE"
            and sweep_side != "BUY"
            and (
                bearish_pressure
                or pressure <= (0.50 if asia_relief_bias else 0.46)
                or m1_momentum <= (0.000 if asia_relief_bias else -0.008)
                or m15_momentum <= (0.000 if asia_relief_bias else -0.010)
                or (asia_relief_bias and close <= ema20 and m15_range_position <= 0.88)
            )
            and close <= max(ema20, ema50)
            and volume_ratio >= max(0.46 if asia_relief_bias else 0.54, reclaim_volume_floor - (0.32 if asia_relief_bias else 0.24))
            and body_efficiency >= max(0.06 if asia_relief_bias else 0.10, reclaim_body_floor - (0.28 if asia_relief_bias else 0.22))
            and alignment_score >= max(0.08 if asia_relief_bias else 0.12, reclaim_alignment_floor - (0.30 if asia_relief_bias else 0.24))
            and fractal_score >= max(0.06 if asia_relief_bias else 0.10, reclaim_fractal_floor - (0.32 if asia_relief_bias else 0.26))
            and trend_efficiency >= max(0.10 if asia_relief_bias else 0.16, reclaim_trend_efficiency_floor - (0.26 if asia_relief_bias else 0.20))
            and instability_score <= min(0.94 if asia_relief_bias else 0.90, reclaim_instability_ceiling + (0.22 if asia_relief_bias else 0.18))
            and feature_drift_score <= min(0.90 if asia_relief_bias else 0.86, reclaim_drift_ceiling + (0.26 if asia_relief_bias else 0.22))
            and range_position <= (0.96 if asia_relief_bias else 0.92)
            and m15_range_position <= (0.96 if asia_relief_bias else 0.92)
            and not strong_stretch
        )
        asia_probe_continuation_buy = bool(
            session_profile == "ASIA_PROBE"
            and asia_relief_bias
            and sweep_side != "SELL"
            and close >= max(ema20, ema50)
            and (m15_ema20 >= m15_ema50 or trend_efficiency >= 0.24)
            and range_position >= 0.82
            and m15_range_position >= 0.80
            and (
                bullish_pressure
                or pressure >= 0.54
                or m1_momentum >= 0.004
                or m15_momentum >= 0.002
            )
            and max(volume_ratio, m15_volume_ratio * 0.60) >= 0.12
            and body_efficiency >= 0.16
            and alignment_score >= 0.34
            and fractal_score >= 0.18
            and trend_efficiency >= 0.08
            and instability_score <= 0.78
            and feature_drift_score <= 0.34
            and not (bearish_pressure and m1_momentum <= 0.0 and range_position >= 0.98)
        )
        asia_probe_continuation_sell = bool(
            session_profile == "ASIA_PROBE"
            and asia_relief_bias
            and sweep_side != "BUY"
            and close <= min(ema20, ema50)
            and (m15_ema20 <= m15_ema50 or trend_efficiency >= 0.24)
            and range_position <= 0.18
            and m15_range_position <= 0.20
            and (
                bearish_pressure
                or pressure <= 0.46
                or m1_momentum <= -0.004
                or m15_momentum <= -0.002
            )
            and max(volume_ratio, m15_volume_ratio * 0.60) >= 0.12
            and body_efficiency >= 0.16
            and alignment_score >= 0.34
            and fractal_score >= 0.18
            and trend_efficiency >= 0.08
            and instability_score <= 0.78
            and feature_drift_score <= 0.34
            and not (bullish_pressure and m1_momentum >= 0.0 and range_position <= 0.02)
        )
        asia_probe_recovery_buy = bool(
            session_profile == "ASIA_PROBE"
            and asia_relief_bias
            and sweep_side != "SELL"
            and close >= ema20
            and (
                bullish_pressure
                or pressure >= 0.49
                or m1_momentum >= -0.004
                or m15_momentum >= -0.006
            )
            and volume_ratio >= 0.38
            and body_efficiency >= 0.04
            and alignment_score >= 0.04
            and fractal_score >= 0.04
            and trend_efficiency >= 0.04
            and instability_score <= 0.92
            and feature_drift_score <= 0.88
            and 0.12 <= range_position <= 0.94
            and 0.08 <= m15_range_position <= 0.96
            and not (strong_stretch and bearish_pressure and range_position >= 0.86)
        )
        asia_probe_recovery_sell = bool(
            session_profile == "ASIA_PROBE"
            and asia_relief_bias
            and sweep_side != "BUY"
            and close <= ema20
            and (
                bearish_pressure
                or pressure <= 0.51
                or m1_momentum <= 0.004
                or m15_momentum <= 0.006
            )
            and volume_ratio >= 0.38
            and body_efficiency >= 0.04
            and alignment_score >= 0.04
            and fractal_score >= 0.04
            and trend_efficiency >= 0.04
            and instability_score <= 0.92
            and feature_drift_score <= 0.88
            and 0.06 <= range_position <= 0.88
            and 0.04 <= m15_range_position <= 0.92
            and not (strong_stretch and bullish_pressure and range_position <= 0.14)
        )
        asia_density_relief_active = bool(
            self.density_micro_scaler_enabled
            and session_profile == "ASIA_PROBE"
            and session_upper in set(self.asia_probe_sessions)
            and float(spread_points) <= max(18.0, float(self.spread_max_points) * 0.42)
            and (
                bool(quota_state.get("density_first_active", False))
                or quota_debt > 0
                or prime_recovery_active
                or idle_lane_recovery_active
                or trajectory_catchup_pressure >= 0.32
            )
            and feature_drift_score <= min(0.82, reclaim_drift_ceiling + 0.18)
            and instability_score <= min(0.84, reclaim_instability_ceiling + 0.12)
        )
        density_micro_scaler_active = bool(
            self.density_micro_scaler_enabled
            and (
                (
                    session_profile != "ASIA_PROBE"
                    and (active_session or session_upper in {"LONDON", "OVERLAP", "NEW_YORK"})
                    and (
                        bool(quota_state.get("density_first_active", False))
                        or quota_debt > 0
                        or prime_recovery_active
                        or idle_lane_recovery_active
                        or trajectory_catchup_pressure >= 0.55
                    )
                )
                or asia_density_relief_active
            )
        )
        density_micro_scaler_buy = bool(
            density_micro_scaler_active
            and sweep_side != "SELL"
            and (
                bullish_pressure
                or prime_bullish_pressure
                or pressure >= 0.52
                or m1_momentum >= 0.004
                or m15_momentum >= 0.005
            )
            and close >= (min(ema20, ema50) - (atr * (0.10 if asia_density_relief_active else 0.08)))
            and max(volume_ratio, m15_volume_ratio * 0.90) >= max(0.34 if asia_density_relief_active else 0.38, reclaim_volume_floor - (0.40 if asia_density_relief_active else 0.34))
            and body_efficiency >= max(0.04 if asia_density_relief_active else 0.05, reclaim_body_floor - (0.34 if asia_density_relief_active else 0.30))
            and alignment_score >= max(0.04 if asia_density_relief_active else 0.06, reclaim_alignment_floor - (0.40 if asia_density_relief_active else 0.36))
            and fractal_score >= max(0.04 if asia_density_relief_active else 0.05, reclaim_fractal_floor - (0.40 if asia_density_relief_active else 0.36))
            and trend_efficiency >= max(0.04 if asia_density_relief_active else 0.05, reclaim_trend_efficiency_floor - (0.30 if asia_density_relief_active else 0.26))
            and instability_score <= min(0.95 if asia_density_relief_active else 0.94, reclaim_instability_ceiling + (0.20 if asia_density_relief_active else 0.18))
            and feature_drift_score <= min(0.91 if asia_density_relief_active else 0.90, reclaim_drift_ceiling + (0.22 if asia_density_relief_active else 0.20))
            and 0.08 <= range_position <= 0.92
            and 0.04 <= m15_range_position <= 0.94
            and not (strong_stretch and bearish_pressure and range_position >= 0.88)
        )
        density_micro_scaler_sell = bool(
            density_micro_scaler_active
            and sweep_side != "BUY"
            and (
                bearish_pressure
                or prime_bearish_pressure
                or pressure <= 0.48
                or m1_momentum <= -0.004
                or m15_momentum <= -0.005
            )
            and close <= (max(ema20, ema50) + (atr * (0.10 if asia_density_relief_active else 0.08)))
            and max(volume_ratio, m15_volume_ratio * 0.90) >= max(0.34 if asia_density_relief_active else 0.38, reclaim_volume_floor - (0.40 if asia_density_relief_active else 0.34))
            and body_efficiency >= max(0.04 if asia_density_relief_active else 0.05, reclaim_body_floor - (0.34 if asia_density_relief_active else 0.30))
            and alignment_score >= max(0.04 if asia_density_relief_active else 0.06, reclaim_alignment_floor - (0.40 if asia_density_relief_active else 0.36))
            and fractal_score >= max(0.04 if asia_density_relief_active else 0.05, reclaim_fractal_floor - (0.40 if asia_density_relief_active else 0.36))
            and trend_efficiency >= max(0.04 if asia_density_relief_active else 0.05, reclaim_trend_efficiency_floor - (0.30 if asia_density_relief_active else 0.26))
            and instability_score <= min(0.95 if asia_density_relief_active else 0.94, reclaim_instability_ceiling + (0.20 if asia_density_relief_active else 0.18))
            and feature_drift_score <= min(0.91 if asia_density_relief_active else 0.90, reclaim_drift_ceiling + (0.22 if asia_density_relief_active else 0.20))
            and 0.08 <= range_position <= 0.92
            and 0.06 <= m15_range_position <= 0.96
            and not (strong_stretch and bullish_pressure and range_position <= 0.12)
        )
        asia_probe_density_relief_buy = bool(
            session_profile == "ASIA_PROBE"
            and asia_density_relief_active
            and sweep_side != "SELL"
            and close >= min(ema20, ema50)
            and (
                bullish_pressure
                or pressure >= 0.50
                or m1_momentum >= -0.003
                or m15_momentum >= -0.005
            )
            and max(volume_ratio, m15_volume_ratio * 0.85) >= 0.28
            and body_efficiency >= 0.03
            and alignment_score >= 0.02
            and fractal_score >= 0.02
            and trend_efficiency >= 0.03
            and instability_score <= 0.94
            and feature_drift_score <= 0.84
            and 0.10 <= range_position <= 0.94
            and 0.08 <= m15_range_position <= 0.96
            and not (strong_stretch and bearish_pressure and range_position >= 0.90)
        )
        asia_probe_density_relief_sell = bool(
            session_profile == "ASIA_PROBE"
            and asia_density_relief_active
            and sweep_side != "BUY"
            and close <= max(ema20, ema50)
            and (
                bearish_pressure
                or pressure <= 0.50
                or m1_momentum <= 0.003
                or m15_momentum <= 0.005
            )
            and max(volume_ratio, m15_volume_ratio * 0.85) >= 0.28
            and body_efficiency >= 0.03
            and alignment_score >= 0.02
            and fractal_score >= 0.02
            and trend_efficiency >= 0.03
            and instability_score <= 0.94
            and feature_drift_score <= 0.84
            and 0.06 <= range_position <= 0.90
            and 0.04 <= m15_range_position <= 0.92
            and not (strong_stretch and bullish_pressure and range_position <= 0.10)
        )
        asia_rotation_rescue_buy = bool(
            asia_density_relief_active
            and session_profile in {"ASIA_PROBE", "MODERATE", "AGGRESSIVE"}
            and sweep_side != "SELL"
            and close >= (ema20 - (atr * 0.18))
            and (
                ema20 >= ema50
                or bullish_pressure
                or pressure >= 0.48
                or m1_momentum >= -0.002
                or m15_momentum >= -0.004
            )
            and max(volume_ratio, m15_volume_ratio * 0.84) >= 0.22
            and body_efficiency >= 0.02
            and alignment_score >= 0.02
            and fractal_score >= 0.02
            and trend_efficiency >= 0.02
            and instability_score <= 0.90
            and feature_drift_score <= 0.80
            and 0.18 <= range_position <= 0.86
            and 0.14 <= m15_range_position <= 0.88
        )
        asia_rotation_rescue_sell = bool(
            asia_density_relief_active
            and session_profile in {"ASIA_PROBE", "MODERATE", "AGGRESSIVE"}
            and sweep_side != "BUY"
            and close <= (ema20 + (atr * 0.18))
            and (
                ema20 <= ema50
                or bearish_pressure
                or pressure <= 0.52
                or m1_momentum <= 0.002
                or m15_momentum <= 0.004
            )
            and max(volume_ratio, m15_volume_ratio * 0.84) >= 0.22
            and body_efficiency >= 0.02
            and alignment_score >= 0.02
            and fractal_score >= 0.02
            and trend_efficiency >= 0.02
            and instability_score <= 0.90
            and feature_drift_score <= 0.80
            and 0.14 <= range_position <= 0.82
            and 0.12 <= m15_range_position <= 0.86
        )
        stretch_reversion_profile = self._stretch_reversion_profile
        side = ""
        entry_profile = ""
        quota_reclaim_rescue_selected = False
        if follow_through_ready and sweep_side == "BUY" and (bullish_structure or reclaim_buy):
            side = "BUY"
            entry_profile = "grid_liquidity_reclaim_long"
        elif follow_through_ready and sweep_side == "SELL" and (bearish_structure or reclaim_sell):
            side = "SELL"
            entry_profile = "grid_liquidity_reclaim_short"
        elif compression_ready and follow_through_ready and reclaim_buy and body_efficiency >= 0.36:
            side = "BUY"
            entry_profile = "grid_trend_reclaim_long"
        elif compression_ready and follow_through_ready and reclaim_sell and body_efficiency >= 0.36:
            side = "SELL"
            entry_profile = "grid_trend_reclaim_short"
        elif bullish_m15_pullback:
            side = "BUY"
            entry_profile = "grid_m15_pullback_reclaim_long"
        elif bearish_m15_pullback:
            side = "SELL"
            entry_profile = "grid_m15_pullback_reclaim_short"
        elif compression_ready and follow_through_ready and bullish_breakout_reclaim:
            side = "BUY"
            entry_profile = "grid_breakout_reclaim_long"
        elif compression_ready and follow_through_ready and bearish_breakout_reclaim:
            side = "SELL"
            entry_profile = "grid_breakout_reclaim_short"
        elif asia_probe_directional_buy:
            side = "BUY"
            entry_profile = "grid_asia_probe_directional_long"
        elif asia_probe_directional_sell:
            side = "SELL"
            entry_profile = "grid_asia_probe_directional_short"
        elif asia_probe_micro_scaler_buy:
            side = "BUY"
            entry_profile = "grid_asia_probe_micro_scaler_long"
        elif asia_probe_micro_scaler_sell:
            side = "SELL"
            entry_profile = "grid_asia_probe_micro_scaler_short"
        elif asia_probe_continuation_buy:
            side = "BUY"
            entry_profile = "grid_asia_probe_continuation_long"
        elif asia_probe_continuation_sell:
            side = "SELL"
            entry_profile = "grid_asia_probe_continuation_short"
        elif asia_probe_recovery_buy:
            side = "BUY"
            entry_profile = "grid_asia_probe_recovery_long"
        elif asia_probe_recovery_sell:
            side = "SELL"
            entry_profile = "grid_asia_probe_recovery_short"
        elif native_pullback_mirror_buy:
            side = "BUY"
            entry_profile = "grid_m15_pullback_reclaim_long"
        elif native_pullback_mirror_sell:
            side = "SELL"
            entry_profile = "grid_m15_pullback_reclaim_short"
        elif native_directional_mirror_buy:
            side = "BUY"
            entry_profile = "grid_directional_flow_long"
        elif native_directional_mirror_sell:
            side = "SELL"
            entry_profile = "grid_directional_flow_short"
        elif bullish_expansion_ready_scaler:
            side = "BUY"
            entry_profile = "grid_expansion_ready_scaler_long"
        elif bearish_expansion_ready_scaler:
            side = "SELL"
            entry_profile = "grid_expansion_ready_scaler_short"
        elif bullish_flow_bias:
            side = "BUY"
            entry_profile = "grid_directional_flow_long"
        elif bearish_flow_bias:
            side = "SELL"
            entry_profile = "grid_directional_flow_short"
        elif prime_directional_rescue_buy:
            side = "BUY"
            entry_profile = "grid_directional_flow_long"
        elif prime_directional_rescue_sell:
            side = "SELL"
            entry_profile = "grid_directional_flow_short"
        elif prime_momentum_fallback_buy:
            side = "BUY"
            entry_profile = "grid_prime_session_momentum_long"
        elif prime_momentum_fallback_sell:
            side = "SELL"
            entry_profile = "grid_prime_session_momentum_short"
        elif quota_reclaim_rescue_buy:
            side = "BUY"
            entry_profile = "grid_prime_session_momentum_long"
            quota_reclaim_rescue_selected = True
        elif quota_reclaim_rescue_sell:
            side = "SELL"
            entry_profile = "grid_prime_session_momentum_short"
            quota_reclaim_rescue_selected = True
        elif london_open_recovery_buy:
            side = "BUY"
            entry_profile = "grid_prime_session_momentum_long"
        elif london_open_recovery_sell:
            side = "SELL"
            entry_profile = "grid_prime_session_momentum_short"
        elif prime_session_scaler_buy:
            side = "BUY"
            entry_profile = "grid_prime_session_momentum_long"
        elif prime_session_scaler_sell:
            side = "SELL"
            entry_profile = "grid_prime_session_momentum_short"
        elif prime_extreme_continuation_buy:
            side = "BUY"
            entry_profile = "grid_prime_session_momentum_long"
        elif prime_extreme_continuation_sell:
            side = "SELL"
            entry_profile = "grid_prime_session_momentum_short"
        elif prime_exhaustion_probe_buy:
            side = "BUY"
            entry_profile = "grid_prime_stretch_reversion_long"
        elif prime_exhaustion_probe_sell:
            side = "SELL"
            entry_profile = "grid_prime_stretch_reversion_short"
        elif prime_stretch_reversal_buy:
            side = "BUY"
            entry_profile = "grid_prime_stretch_reversion_long"
        elif prime_stretch_reversal_sell:
            side = "SELL"
            entry_profile = "grid_prime_stretch_reversion_short"
        elif density_micro_scaler_buy:
            side = "BUY"
            entry_profile = "grid_density_micro_scaler_long"
        elif density_micro_scaler_sell:
            side = "SELL"
            entry_profile = "grid_density_micro_scaler_short"
        elif (
            session_profile == "AGGRESSIVE"
            and density_micro_scaler_active
            and sweep_side != "SELL"
            and close >= (min(ema20, ema50) - (atr * 0.14))
            and max(volume_ratio, m15_volume_ratio * 0.85) >= 0.25
            and body_efficiency >= 0.03
            and alignment_score >= 0.02
            and fractal_score >= 0.02
            and trend_efficiency >= 0.02
            and instability_score <= 0.96
            and feature_drift_score <= 0.90
            and pressure >= 0.50
            and 0.06 <= range_position <= 0.96
            and 0.04 <= m15_range_position <= 0.96
        ):
            side = "BUY"
            entry_profile = "grid_density_micro_scaler_long"
        elif (
            session_profile == "AGGRESSIVE"
            and density_micro_scaler_active
            and sweep_side != "BUY"
            and close <= (max(ema20, ema50) + (atr * 0.14))
            and max(volume_ratio, m15_volume_ratio * 0.85) >= 0.25
            and body_efficiency >= 0.03
            and alignment_score >= 0.02
            and fractal_score >= 0.02
            and trend_efficiency >= 0.02
            and instability_score <= 0.96
            and feature_drift_score <= 0.90
            and pressure <= 0.50
            and 0.04 <= range_position <= 0.94
            and 0.04 <= m15_range_position <= 0.96
        ):
            side = "SELL"
            entry_profile = "grid_density_micro_scaler_short"
        elif (
            session_profile == "AGGRESSIVE"
            and str(session_name or "").upper() in {"LONDON", "OVERLAP", "NEW_YORK"}
            and spread_points <= min(float(self.spread_max_points), float((atr / max(self.point_size, 1e-6))) * 1.20)
            and close >= (min(ema20, ema50) - (atr * 0.10))
            and max(volume_ratio, m15_volume_ratio * 0.82) >= 0.16
            and body_efficiency >= 0.01
            and trend_efficiency >= 0.01
            and instability_score <= 0.98
            and feature_drift_score <= 0.94
            and pressure >= 0.48
            and 0.02 <= range_position <= 0.98
            and 0.02 <= m15_range_position <= 0.98
        ):
            side = "BUY"
            entry_profile = "grid_prime_session_momentum_long"
        elif (
            session_profile == "AGGRESSIVE"
            and str(session_name or "").upper() in {"LONDON", "OVERLAP", "NEW_YORK"}
            and spread_points <= min(float(self.spread_max_points), float((atr / max(self.point_size, 1e-6))) * 1.20)
            and close <= (max(ema20, ema50) + (atr * 0.10))
            and max(volume_ratio, m15_volume_ratio * 0.82) >= 0.16
            and body_efficiency >= 0.01
            and trend_efficiency >= 0.01
            and instability_score <= 0.98
            and feature_drift_score <= 0.94
            and pressure <= 0.52
            and 0.02 <= range_position <= 0.98
            and 0.02 <= m15_range_position <= 0.98
        ):
            side = "SELL"
            entry_profile = "grid_prime_session_momentum_short"
        elif asia_probe_density_relief_buy:
            side = "BUY"
            entry_profile = "grid_asia_probe_recovery_long"
        elif asia_probe_density_relief_sell:
            side = "SELL"
            entry_profile = "grid_asia_probe_recovery_short"
        elif asia_rotation_rescue_buy:
            side = "BUY"
            entry_profile = "grid_directional_flow_long"
        elif asia_rotation_rescue_sell:
            side = "SELL"
            entry_profile = "grid_directional_flow_short"
        elif stretch_reversion_buy:
            side = "BUY"
            entry_profile = "grid_stretch_reversion_long"
        elif stretch_reversion_sell:
            decision.deny_reason = "grid_stretch_reversion_short_blocked"
            return decision
        if not side:
            decision.entry_profile = str(entry_profile)
            decision.session_profile = str(session_profile)
            decision.density_relief_active = bool(asia_density_relief_active)
            if session_profile == "ASIA_PROBE":
                decision.deny_reason = "grid_asia_probe_no_directional_trigger"
            else:
                decision.deny_reason = "grid_no_reclaim_quality"
            return decision
        pressure_ok = bullish_pressure if side == "BUY" else bearish_pressure
        structure_ok = bullish_structure if side == "BUY" else bearish_structure
        pressure_edge = pressure if side == "BUY" else (1.0 - pressure)
        effective_mc_floor = clamp(
            float(self.monte_carlo_win_rate_floor) - float(learning_state.get("mc_floor_relax", 0.0) or 0.0),
            0.78,
            0.95,
        )
        if session_profile == "ASIA_PROBE":
            effective_mc_floor = self._effective_asia_probe_mc_floor(
                base_floor=float(effective_mc_floor),
                asia_probe_mc_floor=float(self.asia_probe_mc_floor),
                entry_profile=str(entry_profile),
                asia_density_relief_active=bool(asia_density_relief_active),
                pressure_edge=float(pressure_edge),
                trend_efficiency=float(trend_efficiency),
                alignment_score=float(alignment_score),
                body_efficiency=float(body_efficiency),
                volume_ratio=float(volume_ratio),
            )
        elif entry_profile.startswith("grid_density_micro_scaler"):
            effective_mc_floor = max(0.78, float(effective_mc_floor) - float(self.density_micro_scaler_mc_floor_relax))
        elif entry_profile.startswith("grid_prime_session_momentum"):
            prime_relax = 0.06 + (0.02 if prime_recovery_active else 0.0)
            effective_mc_floor = max(0.72, float(effective_mc_floor) - float(prime_relax))
        elif entry_profile.startswith("grid_directional_flow") and session_profile in {"AGGRESSIVE", "MODERATE"}:
            directional_relax = 0.04 + (0.01 if prime_recovery_active else 0.0)
            effective_mc_floor = max(0.74, float(effective_mc_floor) - float(directional_relax))
        decision.entry_profile = str(entry_profile)
        decision.session_profile = str(session_profile)
        decision.density_relief_active = bool(asia_density_relief_active)
        decision.mc_floor = float(effective_mc_floor)
        htf_alignment_score = self._htf_alignment_score(
            side=side,
            row=row,
            close=close,
            ema20=ema20,
            ema50=ema50,
        )
        dxy_tailwind_score = self._dxy_tailwind_score(side=side, row=row)
        monte_carlo_win_rate = self._monte_carlo_win_rate(
            side=side,
            session_profile=session_profile,
            entry_profile=entry_profile,
            alignment_score=alignment_score,
            fractal_score=fractal_score,
            seasonality_score=seasonality_score,
            trend_efficiency=trend_efficiency,
            body_efficiency=body_efficiency,
            pressure_edge=pressure_edge,
            compression_expansion_score=compression_expansion_score,
            instability_score=instability_score,
            feature_drift_score=feature_drift_score,
            htf_alignment_score=htf_alignment_score,
            dxy_tailwind_score=dxy_tailwind_score,
        )
        decision.mc_win_rate = float(monte_carlo_win_rate)
        if session_profile == "ASIA_PROBE" and monte_carlo_win_rate < float(effective_mc_floor):
            decision.deny_reason = "grid_asia_probe_mc_floor"
            return decision
        session_grid_level_cap = self._session_grid_level_cap(
            session_name=session_name,
            session_profile=session_profile,
        )
        support_sources = self._support_sources(
            htf_alignment_score=htf_alignment_score,
            dxy_tailwind_score=dxy_tailwind_score,
            monte_carlo_win_rate=monte_carlo_win_rate,
            mc_floor=float(effective_mc_floor),
        )
        probe_candidate = bool(self.probe_leg_enabled) and (
            entry_profile.startswith("grid_trend_reclaim")
            or entry_profile.startswith("grid_m15_pullback_reclaim")
            or entry_profile.startswith("grid_breakout_reclaim")
            or entry_profile.startswith("grid_asia_probe_micro_scaler")
            or entry_profile.startswith("grid_asia_probe_continuation")
            or entry_profile.startswith("grid_asia_probe_recovery")
            or entry_profile.startswith("grid_density_micro_scaler")
            or (
                compression_ready
                and follow_through_ready
                and abs_stretch >= (self.ema_stretch_k * self.probe_stretch_factor)
                and (structure_ok or pressure_ok)
            )
        )
        prime_momentum_rescue = bool(
            session_profile in {"AGGRESSIVE", "MODERATE"}
            and entry_profile.startswith(("grid_prime_session_momentum", "grid_directional_flow"))
            and monte_carlo_win_rate >= max(0.66, float(effective_mc_floor) - 0.14)
            and htf_alignment_score >= 0.46
            and (pressure_edge >= 0.46 or alignment_score >= 0.42)
            and instability_score <= 0.42
            and feature_drift_score <= 0.28
        )
        grid_mode = self._grid_mode(
            session_profile=session_profile,
            atr_ratio=atr_ratio,
            stretch=stretch,
            structure_ok=structure_ok,
            pressure_ok=pressure_ok,
            sweep_side=sweep_side,
        )
        if (
            session_profile == "ASIA_PROBE"
            and sweep_side is None
            and not entry_profile.startswith(
                (
                    "grid_asia_probe_directional",
                    "grid_asia_probe_micro_scaler",
                    "grid_asia_probe_continuation",
                    "grid_asia_probe_recovery",
                )
            )
        ):
            decision.deny_reason = "grid_asia_probe_requires_sweep"
            return decision
        dxy_conflict = self._dxy_blocks_side(side=side, row=row)
        if dxy_conflict and not (
            prime_session_force_mode
            and entry_profile.startswith(
                (
                    "grid_liquidity_reclaim",
                    "grid_trend_reclaim",
                    "grid_m15_pullback_reclaim",
                    "grid_breakout_reclaim",
                    "grid_directional_flow",
                    "grid_prime_session_momentum",
                    "grid_density_micro_scaler",
                )
            )
            and (structure_ok or pressure_ok or sweep_side is not None or strong_stretch)
        ):
            decision.deny_reason = "grid_dxy_block"
            return decision
        if not (structure_ok or sweep_side is not None or strong_stretch or probe_candidate or compression_ready or prime_momentum_rescue):
            decision.deny_reason = "grid_no_stretch"
            return decision
        if not (
            follow_through_ready
            or deceleration
            or extreme
            or strong_stretch
            or structure_ok
            or sweep_side is not None
            or (probe_candidate and neutral_rsi)
            or (prime_momentum_rescue and pressure_edge >= 0.44)
        ):
            decision.deny_reason = "grid_no_reclaim_or_exhaustion"
            return decision

        confluence_raw = self._confluence_score(
            abs_stretch=abs_stretch,
            deceleration=deceleration,
            extreme=extreme,
            spread_points=spread_points,
        )
        confluence = confluence_raw
        confluence += 1.0 if sweep_side is not None else 0.0
        confluence += 0.8 if structure_ok else 0.0
        confluence += 0.6 if pressure_ok else 0.0
        confluence += 0.7 if compression_ready else 0.0
        confluence += 0.5 if follow_through_ready else 0.0
        confluence += 0.3 if alignment_score >= 0.60 else 0.0
        confluence += 0.2 if fractal_score >= 0.58 else 0.0
        confluence += 0.25 if htf_alignment_score >= 0.62 else 0.0
        confluence += 0.18 if dxy_tailwind_score >= 0.55 else 0.0
        confluence += 0.20 if monte_carlo_win_rate >= float(effective_mc_floor) else 0.0
        confluence += 0.16 if entry_profile.startswith("grid_asia_probe_micro_scaler") else 0.0
        confluence += 0.18 if entry_profile.startswith("grid_asia_probe_continuation") else 0.0
        confluence += 0.18 if entry_profile.startswith("grid_asia_probe_recovery") else 0.0
        confluence += 0.14 if entry_profile.startswith("grid_density_micro_scaler") else 0.0
        confluence += 0.4 if (neutral_rsi and session_profile in {"AGGRESSIVE", "MODERATE"} and (structure_ok or sweep_side is not None)) else 0.0
        confluence += 0.25 if session_profile == "AGGRESSIVE" else 0.0
        confluence += 0.35 if (
            prime_session_force_mode
            and entry_profile.startswith(
                (
                    "grid_liquidity_reclaim",
                    "grid_trend_reclaim",
                    "grid_m15_pullback_reclaim",
                    "grid_breakout_reclaim",
                    "grid_directional_flow",
                    "grid_prime_session_momentum",
                    "grid_density_micro_scaler",
                )
            )
        ) else 0.0
        confluence -= 0.18 if dxy_conflict else 0.0
        confluence -= min(0.6, instability_score * 0.6)
        confluence = clamp(confluence, 0.0, 5.0)
        regime_multiplier = self._adaptive_regime_multiplier(
            abs_stretch=abs_stretch,
            atr_ratio=atr_ratio,
            row=row,
        )
        ai_override_allowed = False
        ai_tweaks: dict[str, Any] = {"approve": True, "reason": "local"}
        if approver:
            ai_tweaks = approver(
                {
                    "mode": "NEW_CYCLE",
                    "symbol": symbol,
                    "timeframe": self.timeframe,
                    "side": side,
                    "spread_points": spread_points,
                    "atr": atr,
                    "ema_dev_atr": stretch,
                    "rsi": rsi,
                    "confluence": confluence,
                    "session_profile": session_profile,
                    "asian_high": asian_high,
                    "asian_low": asian_low,
                    "sweep_reason": sweep_reason,
                    "order_block": bool(structure.get("order_block")),
                    "fair_value_gap": bool(structure.get("fair_value_gap")),
                    "delta_pressure": pressure,
                    "regime_multiplier": regime_multiplier,
                    "max_levels": self.max_levels,
                    "step_atr_k": self.step_atr_k,
                    "news_safe": news_safe,
                    "compression_state": compression_state,
                    "compression_expansion_score": compression_expansion_score,
                    "multi_tf_alignment_score": alignment_score,
                    "fractal_persistence_score": fractal_score,
                    "seasonality_edge_score": seasonality_score,
                    "market_instability_score": instability_score,
                    "feature_drift_score": feature_drift_score,
                    "timestamp": now_utc.isoformat(),
                }
            )
            if not bool(ai_tweaks.get("approve", False)):
                ai_conf = float(ai_tweaks.get("confidence", 0.0))
                if not (ai_conf >= 0.50 and confluence >= 3.0):
                    decision.deny_reason = f"grid_ai_deny:{ai_tweaks.get('reason', 'denied')}"
                    decision.ai_mode = str(ai_tweaks.get("ai_mode", "local_fallback"))
                    return decision
                ai_tweaks = dict(ai_tweaks)
                ai_tweaks["approve"] = True
                ai_tweaks["reason"] = "approve_small_probe_override"
                ai_tweaks["lot_multiplier"] = min(1.0, float(ai_tweaks.get("lot_multiplier", 1.0)) * 0.75)
            decision.ai_mode = str(ai_tweaks.get("ai_mode", "local_fallback"))
            ai_conf = float(ai_tweaks.get("confidence", 0.0))
            ai_override_allowed = ai_conf >= self.news_override_min_probability and confluence >= self.news_override_min_confluence

        if (not news_safe) and self.news_block_new_cycles and not (self.allow_news_high_confluence and ai_override_allowed):
            decision.deny_reason = "grid_news_block_new_cycle"
            return decision

        lot = self._lot_for_level(1)
        lot *= clamp(float(ai_tweaks.get("lot_multiplier", 1.0)), 0.1, 1.0)
        if session_profile == "ASIA_PROBE":
            lot *= float(self.asia_probe_grid_lot_multiplier)
        else:
            lot *= self._grid_lot_multiplier(grid_mode)
        lot *= 1.0 + float(learning_state.get("size_bonus", 0.0) or 0.0)
        if (not news_safe) and (self.allow_news_high_confluence and ai_override_allowed):
            lot *= self.news_override_size_multiplier
        step_multiplier = (
            float(ai_tweaks.get("step_multiplier", 1.0))
            * regime_multiplier
            * self._grid_spacing_multiplier(grid_mode)
        )
        if session_profile == "AGGRESSIVE" and entry_profile.startswith(
            (
                "grid_liquidity_reclaim",
                "grid_trend_reclaim",
                "grid_m15_pullback_reclaim",
                "grid_breakout_reclaim",
                "grid_directional_flow",
                "grid_prime_session_momentum",
                "grid_density_micro_scaler",
                "grid_expansion_ready_scaler",
            )
        ):
            step_multiplier *= 0.70
        elif session_profile == "MODERATE" and entry_profile.startswith(
            ("grid_directional_flow", "grid_prime_session_momentum", "grid_density_micro_scaler")
        ):
            step_multiplier *= 0.82
        elif session_profile == "ASIA_PROBE":
            step_multiplier *= 1.08
        step_multiplier = clamp(step_multiplier, 0.45, 2.50)
        step_points = self._apply_spread_spacing_floor(
            step_points=self._step_points(atr=atr, multiplier=step_multiplier),
            spread_points=spread_points,
        )
        atr_points = atr / max(self.point_size, 1e-6)
        if spread_points > min(float(self.spread_max_points), float(atr_points) * 1.20):
            decision.deny_reason = "grid_spread_atr_guard"
            return decision
        stop_points = self._entry_stop_points(step_points=step_points, probe_candidate=probe_candidate)
        if spread_points >= self.flatten_spread_points:
            decision.deny_reason = "grid_spread_disorder"
            return decision

        if entry_profile:
            reason = entry_profile
        elif sweep_side is not None:
            reason = sweep_reason
        elif structure_ok and bool(structure.get("fair_value_gap")):
            reason = "grid_fvg_reclaim"
        elif structure_ok and bool(structure.get("order_block")):
            reason = "grid_order_block_reject"
        elif probe_candidate:
            reason = "grid_cycle_start_probe"
        else:
            reason = "grid_cycle_start_mean_reversion"

        pressure_edge = pressure if side == "BUY" else (1.0 - pressure)
        volume_score = clamp(volume_ratio / 1.25, 0.0, 1.0)
        spread_score = clamp(1.0 - (spread_points / max(self.spread_max_points, 1.0)), 0.0, 1.0)
        session_fit = clamp(
            (
                0.84
                if session_profile == "AGGRESSIVE"
                else (0.74 if session_profile == "MODERATE" else (0.46 if session_profile == "ASIA_PROBE" else 0.0))
            )
            + (0.04 if prime_session_force_mode else 0.0),
            0.0,
            1.0,
        )
        volatility_fit = clamp(
            0.46
            + (0.24 * clamp(1.0 - abs(atr_ratio - 1.0), 0.0, 1.0))
            + (0.10 * float(compression_ready))
            + (0.10 * spread_score)
            + (0.10 * clamp(1.0 - instability_score, 0.0, 1.0)),
            0.0,
            1.0,
        )
        pair_behavior_fit = clamp(
            0.34
            + (0.24 * alignment_score)
            + (0.18 * fractal_score)
            + (0.14 * seasonality_score)
            + (0.10 * trend_efficiency),
            0.0,
            1.0,
        )
        execution_quality_fit = clamp(
            0.34
            + (0.24 * spread_score)
            + (0.14 * body_efficiency)
            + (0.12 * volume_score)
            + (0.10 * clamp(1.0 - instability_score, 0.0, 1.0))
            + (0.06 * clamp(1.0 - feature_drift_score, 0.0, 1.0)),
            0.0,
            1.0,
        )
        structure_cleanliness_score = clamp(
            0.16
            + (0.28 * alignment_score)
            + (0.22 * fractal_score)
            + (0.12 * trend_efficiency)
            + (0.10 * body_efficiency)
            + (0.08 * compression_expansion_score)
            + (0.08 * float(compression_ready))
            + (0.06 * float(structure_ok or sweep_side is not None))
            - (0.08 * instability_score),
            0.0,
            1.0,
        )
        entry_timing_score = clamp(
            0.20
            + (0.18 * body_efficiency)
            + (0.14 * volume_score)
            + (0.14 * pressure_edge)
            + (0.12 * compression_expansion_score)
            + (0.10 * float(compression_ready))
            + (0.08 * float(follow_through_ready or deceleration))
            + (0.06 * clamp(1.0 - instability_score, 0.0, 1.0))
            + (0.04 * clamp(1.0 - feature_drift_score, 0.0, 1.0)),
            0.0,
            1.0,
        )
        regime_fit = clamp(
            0.18
            + (0.18 * alignment_score)
            + (0.10 * htf_alignment_score)
            + (0.14 * fractal_score)
            + (0.10 * float(compression_ready))
            + (0.08 * float(follow_through_ready))
            + (0.06 * trend_efficiency)
            + (0.06 * float(structure_ok or sweep_side is not None))
            + (0.08 * float(entry_profile.startswith(("grid_liquidity_reclaim", "grid_trend_reclaim", "grid_m15_pullback_reclaim", "grid_breakout_reclaim", "grid_directional_flow", "grid_prime_session_momentum"))))
            + (
                0.14
                if (
                    stretch_reversion_profile(entry_profile)
                    and str(row.get("regime_state", row.get("regime", "")) or "").upper()
                    in {"MEAN_REVERSION", "LOW_LIQUIDITY_DRIFT", "LOW_LIQUIDITY_CHOP", "RANGING"}
                )
                else 0.0
            )
            - (0.10 * instability_score)
            - (0.06 * feature_drift_score),
            0.0,
            1.0,
        )
        router_rank_score = clamp(
            max(
                float(
                    0.58
                    + min(0.12, max(0.0, alignment_score - 0.50) * 0.30)
                    + min(0.08, max(0.0, compression_expansion_score - 0.35) * 0.20)
                    + min(0.08, max(0.0, body_efficiency - 0.50) * 0.20)
                ),
                0.58
                + (0.08 * alignment_score)
                + (0.06 * fractal_score)
                + (0.05 * float(prime_session_force_mode))
                + (0.04 * float(stretch_reversion_profile(entry_profile)))
                + (0.03 * volume_score)
                - (0.04 * instability_score),
            ),
            0.58,
            0.92,
        )
        quality_tier = quality_tier_from_scores(
            structure_cleanliness=structure_cleanliness_score,
            regime_fit=regime_fit,
            execution_quality_fit=execution_quality_fit,
            high_liquidity=bool(session_name in {"LONDON", "OVERLAP", "NEW_YORK"}),
            throughput_recovery_active=False,
        )
        if quality_tier == "B" and monte_carlo_win_rate >= 0.84 and htf_alignment_score >= 0.66:
            quality_tier = "A"
        if (
            quality_tier == "B"
            and prime_session_force_mode
            and stretch_reversion_profile(entry_profile)
            and structure_cleanliness_score >= 0.74
            and regime_fit >= 0.68
            and execution_quality_fit >= 0.64
        ):
            quality_tier = "A"
        session_quality_gate = self._prime_session_native_quality_gate(
            session_name=session_name,
            entry_profile=entry_profile,
            quality_tier=quality_tier,
            monte_carlo_win_rate=monte_carlo_win_rate,
            mc_floor=float(effective_mc_floor),
            htf_alignment_score=htf_alignment_score,
            structure_cleanliness_score=structure_cleanliness_score,
            execution_quality_fit=execution_quality_fit,
            router_rank_score=router_rank_score,
            support_sources=support_sources,
        )
        soft_penalty_reason = ""
        soft_penalty_score = 0.0
        prime_spray_soft_gate = bool(
            session_profile == "AGGRESSIVE"
            and str(session_name or "").upper() in {"LONDON", "OVERLAP", "NEW_YORK"}
            and entry_profile.startswith(
                (
                    "grid_density_micro_scaler",
                    "grid_prime_session_momentum",
                    "grid_directional_flow",
                )
            )
            and monte_carlo_win_rate >= max(float(effective_mc_floor) - 0.04, 0.72)
        )
        if session_quality_gate:
            if bool(quota_state.get("density_first_active")) or prime_spray_soft_gate:
                soft_penalty_reason = str(session_quality_gate)
                soft_penalty_score = self._quota_quality_penalty(
                    gate_reason=session_quality_gate,
                    session_name=session_name,
                    quota_debt=int(quota_state.get("quota_debt", 0)),
                    support_sources=support_sources,
                )
                if prime_spray_soft_gate and not bool(quota_state.get("density_first_active")):
                    soft_penalty_score = clamp(float(soft_penalty_score) * 0.60, 0.0, float(self.density_soft_penalty_max))
                decision.soft_penalty_reason = str(soft_penalty_reason)
                decision.soft_penalty_score = float(soft_penalty_score)
            else:
                decision.deny_reason = str(session_quality_gate)
                return decision

        score_hint = clamp(
            0.58
            + min(0.12, max(0.0, alignment_score - 0.50) * 0.30)
            + min(0.08, max(0.0, compression_expansion_score - 0.35) * 0.20)
            + min(0.08, max(0.0, body_efficiency - 0.50) * 0.20),
            0.54,
            0.86,
        )
        score_hint = clamp(
            score_hint * (self.prime_recovery_score_boost if prime_recovery_active else 1.0),
            0.54,
            0.94,
        )
        score_hint = clamp(score_hint - float(soft_penalty_score), 0.46, 0.94)
        grid_max_levels = min(
            session_grid_level_cap,
            max(1, int(ai_tweaks.get("max_levels", session_grid_level_cap))),
        )
        burst_count = self._burst_start_count(
            session_profile=session_profile,
            grid_mode=grid_mode,
            entry_profile=entry_profile,
            confluence=confluence,
            alignment_score=alignment_score,
            fractal_score=fractal_score,
            trend_efficiency=trend_efficiency,
            body_efficiency=body_efficiency,
            compression_expansion_score=compression_expansion_score,
            instability_score=instability_score,
            prime_recovery_active=prime_recovery_active,
            support_sources=support_sources,
            grid_max_levels=grid_max_levels,
        )
        burst_count = self._london_native_burst_floor(
            burst_count=burst_count,
            session_name=session_name,
            session_profile=session_profile,
            grid_mode=grid_mode,
            entry_profile=entry_profile,
            quality_tier=quality_tier,
            confluence=confluence,
            monte_carlo_win_rate=monte_carlo_win_rate,
            mc_floor=float(effective_mc_floor),
            htf_alignment_score=htf_alignment_score,
            structure_cleanliness_score=structure_cleanliness_score,
            execution_quality_fit=execution_quality_fit,
            router_rank_score=router_rank_score,
            support_sources=support_sources,
            grid_max_levels=grid_max_levels,
        )
        burst_count = self._quota_floor_burst_count(
            burst_count=burst_count,
            quota_state=quota_state,
            session_name=session_name,
            session_profile=session_profile,
            grid_mode=grid_mode,
            entry_profile=entry_profile,
            support_sources=support_sources,
            grid_max_levels=grid_max_levels,
        )
        burst_count, mc_soft_penalty_reason, mc_soft_penalty_score = self._apply_density_first_monte_carlo_burst_cap(
            burst_count=burst_count,
            monte_carlo_win_rate=monte_carlo_win_rate,
            mc_floor=float(effective_mc_floor),
            quota_state=quota_state,
            session_name=session_name,
            session_profile=session_profile,
        )
        if mc_soft_penalty_reason:
            soft_penalty_reason = (
                f"{soft_penalty_reason}|{mc_soft_penalty_reason}" if soft_penalty_reason else str(mc_soft_penalty_reason)
            )
            soft_penalty_score = clamp(
                float(soft_penalty_score) + float(mc_soft_penalty_score),
                0.0,
                float(self.density_soft_penalty_max),
            )
            score_hint = clamp(score_hint - float(mc_soft_penalty_score), 0.46, 0.94)
        native_burst_window_id = self._burst_window_id(now_utc=now_utc, session_name=session_name)
        cycle_id = deterministic_id(symbol, "grid", "cycle", side, str(row["time"]), entry_profile)
        base_meta = {
            "grid_cycle": True,
            "grid_action": "START",
            "grid_cycle_id": str(cycle_id),
            "grid_burst_size": int(burst_count),
            "native_burst_window_id": str(native_burst_window_id),
            "news_override": (not news_safe) and ai_override_allowed,
            "ai_mode": decision.ai_mode,
            "grid_probe": bool(probe_candidate),
            "session_profile": session_profile,
            "asian_high": asian_high,
            "asian_low": asian_low,
            "sweep_reason": sweep_reason,
            "order_block": bool(structure.get("order_block")),
            "fair_value_gap": bool(structure.get("fair_value_gap")),
            "delta_pressure": float(pressure),
            "regime_multiplier": float(regime_multiplier),
            "grid_entry_profile": str(entry_profile),
            "compression_proxy_state": str(compression_state),
            "compression_expansion_score": float(compression_expansion_score),
            "multi_tf_alignment_score": float(alignment_score),
            "seasonality_edge_score": float(seasonality_score),
            "fractal_persistence_score": float(fractal_score),
            "market_instability_score": float(instability_score),
            "feature_drift_score": float(feature_drift_score),
            "trend_efficiency_score": float(trend_efficiency),
            "htf_alignment_score": float(htf_alignment_score),
            "dxy_tailwind_score": float(dxy_tailwind_score),
            "mc_win_rate": float(monte_carlo_win_rate),
            "ga_generation_id": int(_finite_float(row.get("ga_generation_id", 0.0), 0.0)),
            "reentry_source_tag": "prime_recovery_repeat" if prime_recovery_active else "fresh_native_cycle",
            "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
            "xau_engine": "GRID_NATIVE_SCALPER",
            "grid_source_role": "NATIVE_ATTACK" if grid_mode == "ATTACK_GRID" else "NATIVE_PRIMARY",
            "regime_fit": float(regime_fit),
            "session_fit": float(session_fit),
            "volatility_fit": float(volatility_fit),
            "pair_behavior_fit": float(pair_behavior_fit),
            "execution_quality_fit": float(execution_quality_fit),
            "entry_timing_score": float(entry_timing_score),
            "structure_cleanliness_score": float(structure_cleanliness_score),
            "router_rank_score": float(router_rank_score),
            "quality_tier": str(quality_tier),
            "hard_vs_soft_gate_mode": "DENSITY_FIRST" if bool(quota_state.get("density_first_active")) else "QUALITY_FIRST",
            "quota_target_10m": int(quota_state.get("target", 0)),
            "quota_approved_last_10m": int(quota_state.get("approved", 0)),
            "quota_debt_10m": int(quota_state.get("quota_debt", 0)),
            "quota_density_first_active": bool(quota_state.get("density_first_active", False)),
            "quota_state": str(quota_state.get("state", "")),
            "soft_quality_penalty_reason": str(soft_penalty_reason),
            "soft_quality_penalty_score": float(soft_penalty_score),
            "tier_size_multiplier": 1.0,
            "strategy_recent_performance_seed": 0.66 if prime_session_force_mode else 0.58,
            "allow_ai_approve_small": True,
            "approve_small_min_probability": 0.50,
            "approve_small_min_confluence": 3.1,
            "grid_mode": grid_mode,
            "grid_volatility_multiplier": float(self._grid_spacing_multiplier(grid_mode)),
            "grid_stop_atr_k": float(
                self._entry_stop_atr_for_profile(
                    session_profile=session_profile,
                    probe_candidate=probe_candidate,
                    atr_ratio=atr_ratio,
                    entry_profile=entry_profile,
                )
            ),
            "prime_session_recovery_active": bool(prime_recovery_active),
            "support_sources": int(support_sources),
            "quota_reclaim_rescue_selected": bool(quota_reclaim_rescue_selected),
            "quota_reclaim_rescue_active": bool(quota_reclaim_rescue_active),
            "quota_reclaim_rescue_debt": int(quota_debt),
            "learning_brain_promoted_pattern": bool(learning_state.get("promoted_pattern", False)),
            "learning_brain_watchlist_match": bool(learning_state.get("watchlist_match", False)),
            "learning_trajectory_catchup_pressure": float(learning_state.get("trajectory_catchup_pressure", 0.0)),
            "learning_mc_floor_relax": float(learning_state.get("mc_floor_relax", 0.0)),
        }
        start_candidates: list[SignalCandidate] = []
        for leg_index in range(1, burst_count + 1):
            leg_step_points = float(step_points)
            if leg_index > 1 and session_profile == "AGGRESSIVE":
                leg_step_points *= 0.96 if grid_mode == "ATTACK_GRID" else 0.98
            leg_step_points = self._apply_spread_spacing_floor(step_points=leg_step_points, spread_points=spread_points)
            leg_stop_points = self._entry_stop_points_for_profile(
                step_points=leg_step_points,
                probe_candidate=probe_candidate,
                entry_profile=entry_profile,
                session_profile=session_profile,
            )
            leg_stop_atr = self._entry_stop_atr_for_profile(
                session_profile=session_profile,
                probe_candidate=probe_candidate,
                atr_ratio=atr_ratio,
                entry_profile=entry_profile,
            )
            leg_tp_r = self._entry_tp_r(
                entry_profile=entry_profile,
                grid_mode=grid_mode,
                session_profile=session_profile,
            )
            leg_lot = max(0.0, self._lot_for_level(leg_index))
            if leg_index > 1:
                leg_lot *= 1.0 if grid_mode == "ATTACK_GRID" else 0.95
            if session_profile == "ASIA_PROBE":
                leg_lot *= float(self.asia_probe_grid_lot_multiplier)
            if monte_carlo_win_rate < float(effective_mc_floor):
                leg_lot *= 0.80
            if soft_penalty_score > 0.0:
                leg_lot *= clamp(1.0 - (soft_penalty_score * 1.5), 0.65, 1.0)
            leg_meta = dict(base_meta)
            leg_meta.update(
                {
                    "grid_level": int(leg_index),
                    "grid_lot": float(leg_lot),
                    "grid_step_atr_k": clamp(step_multiplier * self.step_atr_k, self.step_atr_k, self.step_atr_k * 2.0),
                    "grid_step_points": float(leg_step_points),
                    "grid_max_levels": int(grid_max_levels),
                    "chosen_spacing_points": float(leg_step_points),
                    "stop_points": float(leg_stop_points),
                    "grid_burst_index": int(leg_index),
                }
            )
            start_candidates.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "grid", "start", side, row["time"], cycle_id, leg_index),
                    setup="XAUUSD_M5_GRID_SCALPER_START",
                    side=side,
                    score_hint=clamp(score_hint - ((leg_index - 1) * 0.01), 0.54, 0.94),
                    reason=reason,
                    stop_atr=leg_stop_atr,
                    tp_r=leg_tp_r,
                    entry_kind="GRID_PROBE" if probe_candidate else "GRID_START",
                    strategy_family="GRID",
                    confluence_score=confluence,
                    confluence_required=self.probe_min_confluence if probe_candidate else 3.2,
                    meta=leg_meta,
                )
            )
        decision.soft_penalty_reason = str(soft_penalty_reason)
        decision.soft_penalty_score = float(soft_penalty_score)
        self._record_quota_actions(
            now_utc=now_utc,
            session_name=session_name,
            session_profile=session_profile,
            count=len(start_candidates),
        )
        self._last_signal_emitted_at = now_utc
        decision.candidates = start_candidates
        return decision

    def _evaluate_existing_cycle(
        self,
        *,
        cycle_positions: list[dict[str, Any]],
        row: pd.Series,
        features: pd.DataFrame,
        close: float,
        ema20: float,
        ema50: float,
        atr: float,
        atr_ratio: float,
        spread_points: float,
        contract_size: float,
        news_safe: bool,
        now_utc: datetime,
        session_name: str,
        session_profile: str,
        approver: Callable[[dict[str, Any]], dict[str, Any]] | None,
        add_block_reason: str | None = None,
    ) -> GridScalperDecision:
        decision = GridScalperDecision(cycle_levels=len(cycle_positions))
        sides = {str(position.get("side", "")).upper() for position in cycle_positions}
        if len(sides) != 1:
            decision.close_cycle = True
            decision.close_reason = "grid_mixed_sides_flatten"
            return decision

        side = sides.pop()
        decision.cycle_side = side
        if len(cycle_positions) > self.max_open_positions_symbol:
            decision.close_cycle = True
            decision.close_reason = "grid_symbol_position_cap_exceeded"
            return decision

        avg_entry = self._avg_entry(cycle_positions)
        newest_entry = float(cycle_positions[-1].get("entry_price", avg_entry))
        pnl_usd = self._cycle_pnl_usd(cycle_positions, close, contract_size)
        open_seconds = self._open_seconds(cycle_positions, now_utc)
        open_minutes = self._open_minutes(cycle_positions, now_utc)
        if spread_points >= self.flatten_spread_points:
            decision.close_cycle = True
            decision.close_reason = "grid_spread_disorder_flat"
            return decision
        hard_stop_hit = (
            (side == "BUY" and close <= (avg_entry - (self.stop_atr_k * atr)))
            or (side == "SELL" and close >= (avg_entry + (self.stop_atr_k * atr)))
        )
        if hard_stop_hit:
            decision.close_cycle = True
            decision.close_reason = "grid_cycle_hard_stop"
            self._record_cycle_result(now_utc, pnl_usd, hard_loss=True)
            return decision

        mean_revert_hit = (
            (side == "BUY" and close >= ema20)
            or (side == "SELL" and close <= ema20)
            or abs(close - ema50) <= (atr * 0.15)
        )
        compression_state = str(
            row.get("compression_proxy_state", row.get("compression_state", "NEUTRAL")) or "NEUTRAL"
        ).upper()
        compression_expansion_score = clamp(_finite_float(row.get("compression_expansion_score", 0.0), 0.0), 0.0, 1.0)
        alignment_score = clamp(_finite_float(row.get("multi_tf_alignment_score", 0.5), 0.5), 0.0, 1.0)
        fractal_score = clamp(
            _finite_float(
                row.get(
                    "fractal_persistence_score",
                    row.get("hurst_persistence_score", row.get("m5_hurst_proxy_64", 0.5)),
                ),
                0.5,
            ),
            0.0,
            1.0,
        )
        seasonality_score = clamp(_finite_float(row.get("seasonality_edge_score", 0.5), 0.5), 0.0, 1.0)
        instability_score = clamp(
            _finite_float(row.get("market_instability_score", row.get("feature_drift_score", 0.0)), 0.0),
            0.0,
            1.0,
        )
        feature_drift_score = clamp(_finite_float(row.get("feature_drift_score", 0.0), 0.0), 0.0, 1.0)
        trend_efficiency = clamp(
            _finite_float(row.get("m5_trend_efficiency_16", row.get("m5_trend_efficiency_32", 0.5)), 0.5),
            0.0,
            1.0,
        )
        body_efficiency = clamp(
            abs(_finite_float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.40)), 0.40)),
            0.0,
            1.0,
        )
        open_price = float(row.get("m5_open", close - float(row.get("m5_body", 0.0))))
        high = float(row.get("m5_high", close))
        low = float(row.get("m5_low", close))
        pressure = self._delta_pressure(
            open_price=open_price,
            close=close,
            high=high,
            low=low,
            volume_ratio=float(row.get("m5_volume_ratio_20", 1.0)),
        )
        pressure_ok = pressure >= self.delta_pressure_threshold if side == "BUY" else pressure <= (1.0 - self.delta_pressure_threshold)
        pressure_edge = pressure if side == "BUY" else (1.0 - pressure)
        htf_alignment_score = self._htf_alignment_score(
            side=side,
            row=row,
            close=close,
            ema20=ema20,
            ema50=ema50,
        )
        dxy_tailwind_score = self._dxy_tailwind_score(side=side, row=row)
        monte_carlo_win_rate = self._monte_carlo_win_rate(
            side=side,
            session_profile=session_profile,
            entry_profile="grid_add_burst_follow_through",
            alignment_score=alignment_score,
            fractal_score=fractal_score,
            seasonality_score=seasonality_score,
            trend_efficiency=trend_efficiency,
            body_efficiency=body_efficiency,
            pressure_edge=pressure_edge,
            compression_expansion_score=compression_expansion_score,
            instability_score=instability_score,
            feature_drift_score=feature_drift_score,
            htf_alignment_score=htf_alignment_score,
            dxy_tailwind_score=dxy_tailwind_score,
        )
        learning_state = self._learning_policy_state(session_name=session_name)
        effective_mc_floor = clamp(
            float(self.monte_carlo_win_rate_floor) - float(learning_state.get("mc_floor_relax", 0.0) or 0.0),
            0.78,
            0.95,
        )
        support_sources = self._support_sources(
            htf_alignment_score=htf_alignment_score,
            dxy_tailwind_score=dxy_tailwind_score,
            monte_carlo_win_rate=monte_carlo_win_rate,
            mc_floor=float(effective_mc_floor),
        )
        directional_follow_through = bool(
            (compression_state in {"COMPRESSION", "EXPANSION_READY"} or alignment_score >= 0.72)
            and compression_expansion_score >= 0.0
            and alignment_score >= 0.50
            and fractal_score >= 0.50
            and seasonality_score >= 0.35
            and trend_efficiency >= 0.42
            and instability_score <= 0.55
            and feature_drift_score <= 0.48
            and pressure_ok
            and body_efficiency >= 0.36
            and ((side == "BUY" and close >= ema20 and ema20 >= ema50) or (side == "SELL" and close <= ema20 and ema20 <= ema50))
        )
        realized_target_usd = self.profit_target_usd * (1.15 if directional_follow_through else 0.70)
        if pnl_usd >= realized_target_usd:
            decision.close_cycle = True
            decision.close_reason = "grid_profit_target_hit"
            self._record_cycle_result(now_utc, pnl_usd)
            return decision
        micro_take_profile = self._micro_take_profile(session_name=session_name, session_profile=session_profile)
        quick_green_threshold = max(0.05, self.micro_take_usd * float(micro_take_profile["quick_green_multiplier"]))
        if (
            self.micro_take_enabled
            and len(cycle_positions) <= 1
            and pnl_usd >= quick_green_threshold
            and open_minutes >= int(micro_take_profile["min_open_minutes"])
            and not directional_follow_through
            and (mean_revert_hit or abs(close - ema20) <= (atr * 0.30))
        ):
            decision.close_cycle = True
            decision.close_reason = "grid_quick_green_scalp"
            self._record_cycle_result(now_utc, pnl_usd)
            return decision
        if (
            self.micro_take_enabled
            and mean_revert_hit
            and pnl_usd >= max(quick_green_threshold, self.micro_take_usd * float(micro_take_profile["mean_revert_multiplier"]))
            and not directional_follow_through
        ):
            decision.close_cycle = True
            decision.close_reason = "grid_micro_take_mean_revert"
            self._record_cycle_result(now_utc, pnl_usd)
            return decision
        if (
            open_minutes >= self.no_progress_minutes
            and len(cycle_positions) <= 1
            and abs(pnl_usd) <= max(0.05, self.micro_take_usd * 0.25)
            and not directional_follow_through
        ):
            decision.close_cycle = True
            decision.close_reason = "grid_no_progress_time_exit"
            self._record_cycle_result(now_utc, pnl_usd)
            return decision
        if open_minutes >= self.max_cycle_minutes:
            time_bounce = abs(close - ema20) <= (self.time_exit_bounce_atr * atr)
            if pnl_usd >= 0 or (time_bounce and pnl_usd >= -abs(self.time_exit_loss_usd_cap)):
                decision.close_cycle = True
                decision.close_reason = "grid_time_exit"
                self._record_cycle_result(now_utc, pnl_usd)
                return decision

        if add_block_reason:
            decision.deny_reason = add_block_reason
            return decision

        session_grid_level_cap = self._session_grid_level_cap(
            session_name=session_name,
            session_profile=session_profile,
        )
        if len(cycle_positions) >= session_grid_level_cap:
            decision.deny_reason = "grid_max_levels_reached"
            return decision
        if spread_points > self.add_spread_max_points:
            decision.deny_reason = "grid_add_spread_block"
            return decision
        if spread_points > min(float(self.add_spread_max_points), float((atr / max(self.point_size, 1e-6))) * 1.20):
            decision.deny_reason = "grid_add_spread_atr_guard"
            return decision

        adverse_move = (newest_entry - close) if side == "BUY" else (close - newest_entry)
        favorable_move = (close - newest_entry) if side == "BUY" else (newest_entry - close)
        structure = self._structure_bias(features=features, row=row, close=close)
        confluence = self._confluence_score(
            abs_stretch=abs((close - ema50) / max(atr, 1e-6)),
            deceleration=self._deceleration(features),
            extreme=(self._rsi(features["m5_close"], self.rsi_period) >= self.rsi_overbought
                     or self._rsi(features["m5_close"], self.rsi_period) <= self.rsi_oversold),
            spread_points=spread_points,
        )
        if bool(structure.get("buy_side" if side == "BUY" else "sell_side", False)):
            confluence = clamp(confluence + 0.8, 0.0, 5.0)
        if pressure_ok:
            confluence = clamp(confluence + 0.5, 0.0, 5.0)
        grid_mode = self._grid_mode(
            session_profile=session_profile,
            atr_ratio=atr_ratio,
            stretch=(close - ema50) / max(atr, 1e-6),
            structure_ok=bool(structure.get("buy_side" if side == "BUY" else "sell_side", False)),
            pressure_ok=pressure_ok,
            sweep_side=None,
        )
        ai_override_allowed = False
        ai_tweaks: dict[str, Any] = {"approve": True, "reason": "local"}
        if approver:
            ai_tweaks = approver(
                {
                    "mode": "ADD_LEVEL",
                    "symbol": cycle_positions[0].get("symbol", "XAUUSD"),
                    "timeframe": self.timeframe,
                    "side": side,
                    "spread_points": spread_points,
                    "atr": atr,
                    "adverse_move_atr": adverse_move / max(atr, 1e-6),
                    "confluence": confluence,
                    "current_levels": len(cycle_positions),
                    "max_levels": self.max_levels,
                    "step_atr_k": self.step_atr_k,
                    "news_safe": news_safe,
                    "timestamp": now_utc.isoformat(),
                }
            )
            if not bool(ai_tweaks.get("approve", False)):
                decision.deny_reason = f"grid_ai_deny:{ai_tweaks.get('reason', 'denied')}"
                decision.ai_mode = str(ai_tweaks.get("ai_mode", "local_fallback"))
                return decision
            decision.ai_mode = str(ai_tweaks.get("ai_mode", "local_fallback"))
            ai_conf = float(ai_tweaks.get("confidence", 0.0))
            ai_override_allowed = ai_conf >= self.news_override_min_probability and confluence >= self.news_override_min_confluence

        step_multiplier = float(ai_tweaks.get("step_multiplier", 1.0)) * self._grid_spacing_multiplier(grid_mode)
        if session_profile == "AGGRESSIVE":
            step_multiplier *= 0.72
        elif session_profile == "MODERATE":
            step_multiplier *= 0.84
        elif session_profile == "ASIA_PROBE":
            step_multiplier *= 1.08
        step_multiplier = clamp(step_multiplier, 0.45, 2.50)
        step_points = self._apply_spread_spacing_floor(
            step_points=self._step_points(atr=atr, multiplier=step_multiplier),
            spread_points=spread_points,
        )
        add_stop_atr = self._add_stop_atr(atr_ratio=(atr / max(float(row.get("m5_atr_avg_20", atr)), 1e-6)))
        add_stop_points = self._add_stop_points(step_points=step_points)
        step_needed = step_points * self.point_size
        effective_step_needed = step_needed
        if directional_follow_through and session_profile == "AGGRESSIVE":
            effective_step_needed *= 0.70
        elif directional_follow_through and session_profile == "MODERATE":
            effective_step_needed *= 0.82
        burst_step_needed = step_needed
        if session_profile == "AGGRESSIVE":
            burst_step_needed *= 0.45
        elif session_profile == "MODERATE":
            burst_step_needed *= 0.60
        else:
            burst_step_needed *= 0.75
        aggressive_prime_add_ready = bool(
            session_profile == "AGGRESSIVE"
            and directional_follow_through
            and len(cycle_positions) < min(session_grid_level_cap, 6)
            and favorable_move >= (burst_step_needed * 0.55)
            and pressure_ok
            and body_efficiency >= 0.20
            and compression_expansion_score >= 0.10
            and trend_efficiency >= 0.22
            and instability_score <= 0.76
        )
        aggressive_prime_pyramid_ready = bool(
            session_profile == "AGGRESSIVE"
            and len(cycle_positions) < min(session_grid_level_cap, 6)
            and favorable_move >= (burst_step_needed * 0.20)
            and pressure_ok
            and body_efficiency >= 0.10
            and compression_expansion_score >= 0.04
            and trend_efficiency >= 0.10
            and instability_score <= 0.86
        )
        aggressive_prime_micro_add_ready = bool(
            session_profile == "AGGRESSIVE"
            and len(cycle_positions) < min(session_grid_level_cap, 6)
            and favorable_move >= max(self.point_size * 6.0, burst_step_needed * 0.06)
            and pressure_ok
            and body_efficiency >= 0.04
            and compression_expansion_score >= 0.0
            and trend_efficiency >= 0.05
            and instability_score <= 0.92
            and monte_carlo_win_rate >= max(0.66, float(effective_mc_floor) - 0.18)
        )
        follow_through_add_ready = bool(
            directional_follow_through
            and session_profile in {"AGGRESSIVE", "MODERATE"}
            and len(cycle_positions) < min(session_grid_level_cap, 6)
            and favorable_move >= (burst_step_needed * (0.80 if monte_carlo_win_rate >= float(effective_mc_floor) and htf_alignment_score >= 0.60 else 1.0))
            and pressure_ok
            and body_efficiency >= (0.38 if monte_carlo_win_rate >= float(effective_mc_floor) else 0.42)
            and compression_expansion_score >= (0.18 if support_sources >= 2 else 0.25)
            and trend_efficiency >= (0.42 if support_sources >= 1 else 0.48)
        )
        if aggressive_prime_add_ready:
            follow_through_add_ready = True
        if aggressive_prime_pyramid_ready:
            follow_through_add_ready = True
        if aggressive_prime_micro_add_ready:
            follow_through_add_ready = True
        direction_intact = bool(
            pressure_ok and ((side == "BUY" and close >= ema50) or (side == "SELL" and close <= ema50))
        )
        if (
            session_profile == "AGGRESSIVE"
            and len(cycle_positions) <= 2
            and open_seconds >= 12.0
            and pnl_usd <= max(0.08, self.micro_take_usd * 0.80)
            and favorable_move <= max(self.point_size * 8.0, burst_step_needed * 0.12)
            and not follow_through_add_ready
            and (
                not direction_intact
                or not directional_follow_through
                or not bool(structure.get("buy_side" if side == "BUY" else "sell_side", False))
            )
        ):
            decision.close_cycle = True
            decision.close_reason = "grid_prime_rearm_no_follow_through"
            self._record_cycle_result(now_utc, pnl_usd)
            return decision
        if not direction_intact and pnl_usd <= max(-abs(self.time_exit_loss_usd_cap), -0.10) and open_seconds >= 25.0:
            decision.close_cycle = True
            decision.close_reason = "grid_direction_lost_abort"
            self._record_cycle_result(now_utc, pnl_usd, hard_loss=True)
            return decision
        if (
            grid_mode == "DEFENSIVE_GRID"
            and len(cycle_positions) >= 2
            and adverse_move >= (step_needed * 1.35)
            and not bool(structure.get("buy_side" if side == "BUY" else "sell_side", False))
            and not pressure_ok
            and pnl_usd <= max(-abs(self.time_exit_loss_usd_cap), -0.10)
        ):
            decision.close_cycle = True
            decision.close_reason = "grid_hostile_environment_abort"
            self._record_cycle_result(now_utc, pnl_usd, hard_loss=True)
            return decision
        if adverse_move < effective_step_needed and not follow_through_add_ready:
            decision.deny_reason = "grid_step_not_reached"
            return decision
        if not follow_through_add_ready:
            decision.deny_reason = "grid_add_requires_follow_through"
            return decision
        if (
            len(cycle_positions) <= 2
            and not directional_follow_through
            and not bool(structure.get("buy_side" if side == "BUY" else "sell_side", False))
            and not pressure_ok
        ):
            decision.deny_reason = "grid_add_no_structure"
            return decision
        if not directional_follow_through and grid_mode == "ATTACK_GRID":
            decision.deny_reason = "grid_attack_follow_through_missing"
            return decision

        if (not news_safe) and self.news_block_adds and not (self.allow_news_high_confluence and ai_override_allowed):
            decision.deny_reason = "grid_news_block_add"
            return decision

        grid_max_levels = min(
            session_grid_level_cap,
            max(1, int(ai_tweaks.get("max_levels", session_grid_level_cap))),
        )
        quota_state = self._quota_state(now_utc=now_utc, session_name=session_name, session_profile=session_profile)
        decision.quota_target_10m = int(quota_state.get("target", 0))
        decision.quota_approved_last_10m = int(quota_state.get("approved", 0))
        decision.quota_debt_10m = int(quota_state.get("quota_debt", 0))
        decision.quota_density_first_active = bool(quota_state.get("density_first_active", False))
        decision.quota_state = str(quota_state.get("state", ""))
        decision.quota_window_id = str(quota_state.get("window_id", ""))
        remaining_levels = max(0, grid_max_levels - len(cycle_positions))
        burst_add_count = self._burst_add_count(
            session_profile=session_profile,
            grid_mode=grid_mode,
            favorable_move=favorable_move,
            burst_step_needed=burst_step_needed,
            follow_through_add_ready=follow_through_add_ready,
            alignment_score=alignment_score,
            fractal_score=fractal_score,
            trend_efficiency=trend_efficiency,
            body_efficiency=body_efficiency,
            instability_score=instability_score,
            remaining_levels=remaining_levels,
        )
        burst_add_count = self._quota_floor_add_count(
            add_count=burst_add_count,
            quota_state=quota_state,
            session_name=session_name,
            session_profile=session_profile,
            grid_mode=grid_mode,
            follow_through_add_ready=follow_through_add_ready,
            support_sources=support_sources,
            remaining_levels=remaining_levels,
        )
        add_soft_penalty_reason = ""
        add_soft_penalty_score = 0.0
        burst_add_count, add_mc_soft_penalty_reason, add_mc_soft_penalty_score = self._apply_density_first_monte_carlo_add_cap(
            add_count=burst_add_count,
            monte_carlo_win_rate=monte_carlo_win_rate,
            mc_floor=float(effective_mc_floor),
            quota_state=quota_state,
            session_name=session_name,
            session_profile=session_profile,
            remaining_levels=remaining_levels,
        )
        if add_mc_soft_penalty_reason:
            add_soft_penalty_reason = str(add_mc_soft_penalty_reason)
            add_soft_penalty_score = float(add_mc_soft_penalty_score)
            decision.soft_penalty_reason = str(add_soft_penalty_reason)
            decision.soft_penalty_score = float(add_soft_penalty_score)
        cycle_id = self._cycle_id_from_positions(cycle_positions)
        symbol = str(cycle_positions[0].get("symbol", "XAUUSD"))
        add_reason = "grid_add_trend_burst"
        native_burst_window_id = self._burst_window_id(now_utc=now_utc, session_name=session_name)
        add_candidates: list[SignalCandidate] = []
        for leg_offset in range(1, burst_add_count + 1):
            level = len(cycle_positions) + leg_offset
            leg_step_points = float(step_points) * (0.94 if leg_offset > 1 and session_profile == "AGGRESSIVE" else 1.0)
            leg_step_points = self._apply_spread_spacing_floor(step_points=leg_step_points, spread_points=spread_points)
            leg_stop_points = self._add_stop_points(step_points=leg_step_points)
            lot = self._lot_for_level(level)
            lot *= clamp(float(ai_tweaks.get("lot_multiplier", 1.0)), 0.1, 1.0)
            if session_profile == "ASIA_PROBE":
                lot *= float(self.asia_probe_grid_lot_multiplier)
            else:
                lot *= self._grid_lot_multiplier(grid_mode)
            lot *= 1.0 + float(learning_state.get("size_bonus", 0.0) or 0.0)
            if follow_through_add_ready:
                lot *= 1.25 if session_profile == "AGGRESSIVE" else 1.15
            if leg_offset > 1:
                lot *= 0.95
            if (not news_safe) and (self.allow_news_high_confluence and ai_override_allowed):
                lot *= self.news_override_size_multiplier
            if monte_carlo_win_rate < float(effective_mc_floor):
                lot *= 0.82
            if add_soft_penalty_score > 0.0:
                lot *= clamp(1.0 - (add_soft_penalty_score * 1.5), 0.65, 1.0)
            add_candidates.append(
                SignalCandidate(
                    signal_id=deterministic_id(symbol, "grid", "add", side, row["time"], cycle_id, level),
                    setup="XAUUSD_M5_GRID_SCALPER_ADD",
                    side=side,
                    score_hint=clamp(
                        (
                            0.62 + min(0.18, (favorable_move / max(burst_step_needed, 1e-6) - 1.0) * 0.06)
                            if follow_through_add_ready
                            else 0.55 + min(0.2, (adverse_move / max(effective_step_needed, 1e-6) - 1.0) * 0.08)
                        )
                        - ((leg_offset - 1) * 0.01),
                        0.55,
                        0.92,
                    ),
                    reason=add_reason,
                    stop_atr=add_stop_atr,
                    tp_r=2.35 if follow_through_add_ready else (2.1 if directional_follow_through else 1.8),
                    entry_kind="GRID_ADD",
                    strategy_family="GRID",
                    confluence_score=confluence,
                    confluence_required=3.0,
                    meta={
                        "grid_cycle": True,
                        "grid_action": "ADD",
                        "grid_cycle_id": str(cycle_id),
                        "grid_level": int(level),
                        "grid_lot": max(0.0, lot),
                        "grid_step_atr_k": clamp(step_multiplier * self.step_atr_k, self.step_atr_k, self.step_atr_k * 2.0),
                        "grid_step_points": float(leg_step_points),
                        "grid_max_levels": int(grid_max_levels),
                        "native_burst_window_id": str(native_burst_window_id),
                        "news_override": (not news_safe) and ai_override_allowed,
                        "ai_mode": decision.ai_mode,
                        "order_block": bool(structure.get("order_block")),
                        "fair_value_gap": bool(structure.get("fair_value_gap")),
                        "delta_pressure": float(pressure),
                        "grid_entry_profile": (
                            "grid_add_burst_follow_through"
                            if follow_through_add_ready
                            else ("grid_add_follow_through" if directional_follow_through else "grid_add_defensive")
                        ),
                        "compression_proxy_state": str(compression_state),
                        "compression_expansion_score": float(compression_expansion_score),
                        "multi_tf_alignment_score": float(alignment_score),
                        "seasonality_edge_score": float(seasonality_score),
                        "fractal_persistence_score": float(fractal_score),
                        "market_instability_score": float(instability_score),
                        "feature_drift_score": float(feature_drift_score),
                        "trend_efficiency_score": float(trend_efficiency),
                        "htf_alignment_score": float(htf_alignment_score),
                        "dxy_tailwind_score": float(dxy_tailwind_score),
                        "mc_win_rate": float(monte_carlo_win_rate),
                        "ga_generation_id": int(_finite_float(row.get("ga_generation_id", 0.0), 0.0)),
                        "reentry_source_tag": "follow_through_reentry" if favorable_move > 0.0 else "native_cycle_add",
                        "grid_source_role": "NATIVE_ADD_BURST",
                        "xau_engine": "GRID_NATIVE_SCALPER",
                        "grid_mode": grid_mode,
                        "grid_volatility_multiplier": float(self._grid_spacing_multiplier(grid_mode)),
                        "grid_follow_through_add": bool(follow_through_add_ready),
                        "chosen_spacing_points": float(leg_step_points),
                        "grid_stop_atr_k": float(add_stop_atr),
                        "stop_points": float(leg_stop_points),
                        "grid_burst_size": int(burst_add_count),
                        "grid_burst_index": int(leg_offset),
                        "support_sources": int(support_sources),
                        "quota_target_10m": int(quota_state.get("target", 0)),
                        "quota_approved_last_10m": int(quota_state.get("approved", 0)),
                        "quota_debt_10m": int(quota_state.get("quota_debt", 0)),
                        "quota_density_first_active": bool(quota_state.get("density_first_active", False)),
                        "quota_state": str(quota_state.get("state", "")),
                        "soft_quality_penalty_reason": str(add_soft_penalty_reason),
                        "soft_quality_penalty_score": float(add_soft_penalty_score),
                        "learning_brain_promoted_pattern": bool(learning_state.get("promoted_pattern", False)),
                        "learning_brain_watchlist_match": bool(learning_state.get("watchlist_match", False)),
                        "learning_trajectory_catchup_pressure": float(learning_state.get("trajectory_catchup_pressure", 0.0)),
                        "learning_mc_floor_relax": float(learning_state.get("mc_floor_relax", 0.0)),
                    },
                )
            )
        decision.candidates = add_candidates
        self._record_quota_actions(
            now_utc=now_utc,
            session_name=session_name,
            session_profile=session_profile,
            count=len(add_candidates),
        )
        return decision

    def _burst_window_id(self, *, now_utc: datetime, session_name: str) -> str:
        bucket = (int(now_utc.minute) // max(1, int(self.burst_window_minutes))) * max(1, int(self.burst_window_minutes))
        return f"{now_utc.astimezone(UTC).strftime('%Y%m%dT%H')}:{bucket:02d}:{str(session_name or '').upper()}"

    def _htf_alignment_score(
        self,
        *,
        side: str,
        row: pd.Series,
        close: float,
        ema20: float,
        ema50: float,
    ) -> float:
        side_key = str(side or "").upper()
        scores: list[float] = []
        frames = ("m15", "h1", "h4", "d1")
        for prefix in frames:
            frame_close = _finite_float(row.get(f"{prefix}_close", close), close)
            frame_ema20 = _finite_float(row.get(f"{prefix}_ema_20", ema20), ema20)
            frame_ema50 = _finite_float(row.get(f"{prefix}_ema_50", ema50), ema50)
            range_position = clamp(_finite_float(row.get(f"{prefix}_range_position_20", row.get("m15_range_position_20", 0.5)), 0.5), 0.0, 1.0)
            if side_key == "BUY":
                frame_score = (
                    (0.40 if frame_close >= frame_ema20 else 0.0)
                    + (0.35 if frame_ema20 >= frame_ema50 else 0.0)
                    + (0.25 if range_position >= 0.34 else 0.0)
                )
            else:
                frame_score = (
                    (0.40 if frame_close <= frame_ema20 else 0.0)
                    + (0.35 if frame_ema20 <= frame_ema50 else 0.0)
                    + (0.25 if range_position <= 0.66 else 0.0)
                )
            scores.append(frame_score)
        if not scores:
            return 0.5
        return clamp(sum(scores) / len(scores), 0.0, 1.0)

    def _dxy_tailwind_score(self, *, side: str, row: pd.Series) -> float:
        side_key = str(side or "").upper()
        dxy_fast = _finite_float(row.get("dxy_ret_15", row.get("dxy_ret_5", 0.0)), 0.0)
        dxy_slow = _finite_float(row.get("dxy_ret_60", row.get("dxy_ret_15", 0.0)), 0.0)
        directional_edge = ((-dxy_fast) * 110.0) + ((-dxy_slow) * 160.0) if side_key == "BUY" else (dxy_fast * 110.0) + (dxy_slow * 160.0)
        return clamp(0.5 + directional_edge, 0.0, 1.0)

    def _monte_carlo_win_rate(
        self,
        *,
        side: str,
        session_profile: str,
        entry_profile: str,
        alignment_score: float,
        fractal_score: float,
        seasonality_score: float,
        trend_efficiency: float,
        body_efficiency: float,
        pressure_edge: float,
        compression_expansion_score: float,
        instability_score: float,
        feature_drift_score: float,
        htf_alignment_score: float,
        dxy_tailwind_score: float,
    ) -> float:
        strong_profile = entry_profile.startswith(
            (
                "grid_liquidity_reclaim",
                "grid_trend_reclaim",
                "grid_m15_pullback_reclaim",
                "grid_breakout_reclaim",
                "grid_directional_flow",
                "grid_asia_probe_directional",
                "grid_asia_probe_micro_scaler",
                "grid_asia_probe_continuation",
                "grid_asia_probe_recovery",
                "grid_density_micro_scaler",
                "grid_expansion_ready_scaler",
                "grid_prime_session_momentum",
            )
        )
        estimate = (
            0.44
            + (0.11 * alignment_score)
            + (0.09 * fractal_score)
            + (0.07 * seasonality_score)
            + (0.07 * trend_efficiency)
            + (0.06 * body_efficiency)
            + (0.06 * pressure_edge)
            + (0.05 * compression_expansion_score)
            + (0.09 * htf_alignment_score)
            + (0.07 * dxy_tailwind_score)
            + (0.04 if session_profile == "AGGRESSIVE" else (0.02 if session_profile == "MODERATE" else 0.0))
            + (0.03 if strong_profile else 0.0)
            - (0.08 * instability_score)
            - (0.06 * feature_drift_score)
        )
        return clamp(estimate, 0.50, 0.97)

    @staticmethod
    def _support_sources(*, htf_alignment_score: float, dxy_tailwind_score: float, monte_carlo_win_rate: float, mc_floor: float) -> int:
        return int(htf_alignment_score >= 0.62) + int(dxy_tailwind_score >= 0.55) + int(monte_carlo_win_rate >= max(mc_floor, 0.84))

    @staticmethod
    def _effective_asia_probe_mc_floor(
        *,
        base_floor: float,
        asia_probe_mc_floor: float,
        entry_profile: str,
        asia_density_relief_active: bool,
        pressure_edge: float,
        trend_efficiency: float,
        alignment_score: float,
        body_efficiency: float,
        volume_ratio: float,
    ) -> float:
        effective_floor = max(float(base_floor), float(asia_probe_mc_floor))
        relief = 0.0
        if entry_profile.startswith("grid_density_micro_scaler"):
            relief += 0.05
        elif entry_profile.startswith("grid_asia_probe_continuation"):
            relief += 0.08
        elif entry_profile.startswith("grid_asia_probe_recovery"):
            relief += 0.10
        elif entry_profile.startswith("grid_asia_probe_micro_scaler"):
            relief += 0.02
        elif entry_profile.startswith("grid_asia_probe_directional"):
            relief += 0.01
        elif asia_density_relief_active and entry_profile.startswith("grid_directional_flow"):
            relief += 0.13
        if asia_density_relief_active:
            relief += 0.01
            if (
                float(trend_efficiency) >= 0.18
                and float(alignment_score) >= 0.16
                and float(body_efficiency) >= 0.08
            ):
                relief += 0.02
        if float(pressure_edge) >= 0.58 and float(trend_efficiency) >= 0.23 and float(alignment_score) >= 0.20:
            relief += 0.02
        if float(body_efficiency) >= 0.18 and float(volume_ratio) >= 0.58:
            relief += 0.01
        if (
            asia_density_relief_active
            and (
                entry_profile.startswith("grid_density_micro_scaler")
                or entry_profile.startswith("grid_asia_probe_continuation")
                or entry_profile.startswith("grid_asia_probe_recovery")
            )
            and float(pressure_edge) >= 0.56
            and float(trend_efficiency) >= 0.18
            and float(alignment_score) >= 0.16
        ):
            relief += 0.02
        minimum_floor = (
            0.70
            if (
                entry_profile.startswith("grid_density_micro_scaler")
                or entry_profile.startswith("grid_asia_probe_continuation")
                or entry_profile.startswith("grid_asia_probe_recovery")
                or (asia_density_relief_active and entry_profile.startswith("grid_directional_flow"))
            )
            else 0.76
        )
        return clamp(effective_floor - relief, minimum_floor, 0.95)

    def native_candidate_is_unsafe_extreme_bucket(
        self,
        *,
        candidate: SignalCandidate,
        row: pd.Series,
        session_name: str,
        regime_state: str = "",
    ) -> bool:
        meta = dict(candidate.meta or {})
        if str(meta.get("xau_engine") or "").upper() != "GRID_NATIVE_SCALPER":
            return False
        entry_profile = str(meta.get("grid_entry_profile") or "")
        if entry_profile not in {
            "grid_directional_flow_long",
            "grid_directional_flow_short",
            "grid_prime_session_momentum_long",
            "grid_prime_session_momentum_short",
        }:
            return False
        if str(session_name or "").upper() not in {"OVERLAP", "NEW_YORK"}:
            return False
        if str(regime_state or meta.get("regime_state") or "").upper() not in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP", "RANGING"}:
            return False

        side = str(getattr(candidate, "side", "") or "").upper()
        m5_range_position = _finite_float(row.get("m5_range_position_20", 0.5), 0.5)
        m15_range_position = _finite_float(row.get("m15_range_position_20", 0.5), 0.5)
        body_efficiency = max(
            _finite_float(row.get("m5_body_efficiency", 0.0), 0.0),
            _finite_float(row.get("body_efficiency", 0.0), 0.0),
        )
        trend_efficiency = max(
            _finite_float(meta.get("trend_efficiency_score", 0.0), 0.0),
            _finite_float(row.get("m5_trend_efficiency_16", 0.0), 0.0),
            _finite_float(row.get("m5_trend_efficiency_32", 0.0), 0.0),
        )
        alignment_score = max(
            _finite_float(meta.get("multi_tf_alignment_score", 0.0), 0.0),
            _finite_float(row.get("multi_tf_alignment_score", 0.0), 0.0),
        )
        fractal_score = max(
            _finite_float(meta.get("fractal_persistence_score", 0.0), 0.0),
            _finite_float(row.get("fractal_persistence_score", 0.0), 0.0),
        )
        instability_score = min(
            _finite_float(meta.get("market_instability_score", 1.0), 1.0),
            _finite_float(row.get("market_instability_score", 1.0), 1.0),
        )
        feature_drift_score = min(
            _finite_float(meta.get("feature_drift_score", 1.0), 1.0),
            _finite_float(row.get("feature_drift_score", 1.0), 1.0),
        )
        range_extreme = (
            (side == "BUY" and m5_range_position >= 0.94)
            or (side == "SELL" and m5_range_position <= 0.06)
        )
        return bool(
            range_extreme
            and 0.35 <= m15_range_position <= 0.65
            and body_efficiency >= 0.34
            and trend_efficiency >= 0.92
            and alignment_score >= 0.50
            and fractal_score >= 0.60
            and instability_score <= 0.10
            and feature_drift_score <= 0.16
        )

    def _grid_mode(
        self,
        *,
        session_profile: str,
        atr_ratio: float,
        stretch: float,
        structure_ok: bool,
        pressure_ok: bool,
        sweep_side: str | None,
    ) -> str:
        if session_profile == "ASIA_PROBE":
            return "DEFENSIVE_GRID"
        if (
            session_profile == "AGGRESSIVE"
            and (
                (structure_ok or pressure_ok or sweep_side is not None)
                or abs(stretch) >= (self.ema_stretch_k * 0.75)
            )
            and (
                atr_ratio >= 0.75
                or abs(stretch) >= (self.ema_stretch_k * 0.85)
            )
        ):
            return "ATTACK_GRID"
        if atr_ratio >= (self.atr_spike_threshold * 0.85):
            return "DEFENSIVE_GRID"
        if not structure_ok and not pressure_ok and sweep_side is None:
            return "DEFENSIVE_GRID"
        return "NORMAL_GRID"

    def _grid_spacing_multiplier(self, grid_mode: str) -> float:
        if grid_mode == "ATTACK_GRID":
            return clamp(self.attack_grid_spacing_multiplier, 0.75, 1.0)
        if grid_mode == "DEFENSIVE_GRID":
            return clamp(self.defensive_grid_spacing_multiplier, 1.0, 1.5)
        return 1.0

    def _grid_lot_multiplier(self, grid_mode: str) -> float:
        if grid_mode == "ATTACK_GRID":
            return clamp(self.attack_grid_lot_multiplier, 1.0, 1.25)
        if grid_mode == "DEFENSIVE_GRID":
            return clamp(self.defensive_grid_lot_multiplier, 0.5, 1.0)
        return 1.0

    def _session_profile(
        self,
        *,
        session_name: str,
        now_utc: datetime,
        atr_ratio: float,
        spread_points: float,
    ) -> str:
        session = str(session_name or "").upper()
        hour = now_utc.astimezone(UTC).hour
        if self.asia_probe_enabled and session in set(self.asia_probe_sessions) and session in self.allowed_sessions:
            if spread_points <= self.asia_probe_spread_max_points and atr_ratio <= self.asia_probe_atr_ratio_max:
                return "ASIA_PROBE"
            return "DISABLED"
        if session in self.allowed_sessions:
            # Prime-session forcing logic: keep the full London and overlap/NY
            # windows aggressive so the live XAU grid does not go idle.
            prime_xau_window = (
                (session == "LONDON" and 8 <= hour < 12)
                or (session in {"OVERLAP", "NEW_YORK"} and 13 <= hour < 17)
            )
            if prime_xau_window or self._hour_in_window(hour, self.london_aggressive_hours_utc) or self._hour_in_window(hour, self.ny_aggressive_hours_utc):
                return "AGGRESSIVE"
            return "MODERATE"
        if (
            self.asia_probe_enabled
            and session in set(self.asia_probe_sessions)
            and spread_points <= self.asia_probe_spread_max_points
            and atr_ratio <= self.asia_probe_atr_ratio_max
        ):
            return "ASIA_PROBE"
        return "DISABLED"

    @staticmethod
    def _hour_in_window(hour: int, window: tuple[int, int]) -> bool:
        if len(window) < 2:
            return False
        start = int(window[0])
        end = int(window[1])
        return start <= hour < end

    def _asian_range(self, features: pd.DataFrame, *, now_utc: datetime) -> tuple[float, float]:
        if features.empty or "time" not in features:
            return 0.0, 0.0
        times = pd.to_datetime(features["time"], utc=True, errors="coerce")
        if times.isna().all():
            return 0.0, 0.0
        date_floor = now_utc.astimezone(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        asia_end = date_floor + timedelta(hours=8)
        mask = (times >= date_floor) & (times < asia_end)
        window = features.loc[mask]
        if window.empty:
            window = features.tail(96)
        highs = window["m5_high"] if "m5_high" in window else window["m5_close"]
        lows = window["m5_low"] if "m5_low" in window else window["m5_close"]
        try:
            return float(highs.max()), float(lows.min())
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _detect_sweep_reclaim(
        *,
        high: float,
        low: float,
        close: float,
        asian_high: float,
        asian_low: float,
    ) -> tuple[str | None, str]:
        if asian_low > 0 and low < asian_low and close > asian_low:
            return "BUY", "grid_liquidity_sweep_reclaim_long"
        if asian_high > 0 and high > asian_high and close < asian_high:
            return "SELL", "grid_liquidity_sweep_reclaim_short"
        return None, ""

    def _structure_bias(self, *, features: pd.DataFrame, row: pd.Series, close: float) -> dict[str, Any]:
        bullish_ob = bool(int(row.get("m5_pinbar_bull", 0)) == 1 or int(row.get("m5_engulf_bull", 0)) == 1)
        bearish_ob = bool(int(row.get("m5_pinbar_bear", 0)) == 1 or int(row.get("m5_engulf_bear", 0)) == 1)
        bullish_fvg = False
        bearish_fvg = False
        if len(features) >= 3:
            recent = features.tail(3).reset_index(drop=True)
            first_high = float(recent.iloc[0].get("m5_high", recent.iloc[0].get("m5_close", close)))
            first_low = float(recent.iloc[0].get("m5_low", recent.iloc[0].get("m5_close", close)))
            last_high = float(recent.iloc[-1].get("m5_high", recent.iloc[-1].get("m5_close", close)))
            last_low = float(recent.iloc[-1].get("m5_low", recent.iloc[-1].get("m5_close", close)))
            bullish_fvg = last_low > first_high
            bearish_fvg = last_high < first_low
        return {
            "buy_side": bullish_ob or bullish_fvg,
            "sell_side": bearish_ob or bearish_fvg,
            "order_block": bullish_ob or bearish_ob,
            "fair_value_gap": bullish_fvg or bearish_fvg,
        }

    @staticmethod
    def _delta_pressure(
        *,
        open_price: float,
        close: float,
        high: float,
        low: float,
        volume_ratio: float,
    ) -> float:
        total_range = max(high - low, 1e-6)
        close_pos = (close - low) / total_range
        pressure = 0.50
        pressure += 0.12 if close >= open_price else -0.12
        pressure += clamp((close_pos - 0.50) * 0.50, -0.15, 0.15)
        pressure += clamp((float(volume_ratio) - 1.0) * 0.10, -0.10, 0.10)
        return clamp(pressure, 0.0, 1.0)

    @staticmethod
    def _dxy_blocks_side(*, side: str, row: pd.Series) -> bool:
        dxy_ret = float(row.get("dxy_ret_60", row.get("dxy_ret_15", 0.0)))
        if side == "BUY" and dxy_ret > 0.0025:
            return True
        if side == "SELL" and dxy_ret < -0.0025:
            return True
        return False

    def _adaptive_regime_multiplier(self, *, abs_stretch: float, atr_ratio: float, row: pd.Series) -> float:
        trend_bias = abs(float(row.get("m5_trend_bias", 0.0)))
        if atr_ratio >= 1.8:
            return 1.8
        if abs_stretch >= 1.4 or trend_bias >= 20.0:
            return 1.5
        if abs_stretch >= 1.0 or atr_ratio >= 1.2:
            return 1.2
        if atr_ratio <= 0.9:
            return 0.85
        return 1.0

    def _lot_for_level(self, level: int) -> float:
        level_index = max(1, int(level))
        if self.lot_schedule and level_index <= len(self.lot_schedule):
            return max(self.base_lot, float(self.lot_schedule[level_index - 1]))
        if not self.allow_mild_scale:
            return self.base_lot
        # Mild anti-martingale-like scaling; no geometric blow-up.
        mild = self.base_lot + (0.01 * ((level_index - 1) // 2))
        return max(self.base_lot, mild)

    def _entry_stop_atr(self, *, session_profile: str, probe_candidate: bool, atr_ratio: float) -> float:
        base = max(0.25, float(self.entry_stop_atr_k))
        if probe_candidate:
            base *= 0.95
        if str(session_profile).upper() == "ASIA_PROBE":
            base *= 0.96
        if float(atr_ratio) >= 1.6:
            base *= 1.08
        elif float(atr_ratio) <= 0.9:
            base *= 0.96
        return clamp(base, 0.25, max(0.25, float(self.stop_atr_k)))

    def _entry_stop_atr_for_profile(
        self,
        *,
        session_profile: str,
        probe_candidate: bool,
        atr_ratio: float,
        entry_profile: str,
    ) -> float:
        base = self._entry_stop_atr(
            session_profile=session_profile,
            probe_candidate=probe_candidate,
            atr_ratio=atr_ratio,
        )
        profile_key = str(entry_profile or "")
        session_key = str(session_profile or "").upper()
        if profile_key.startswith(("grid_directional_flow", "grid_prime_session_momentum", "grid_expansion_ready_scaler", "grid_breakout_reclaim")):
            boost = self.prime_directional_stop_points_boost if session_key in {"AGGRESSIVE", "MODERATE"} else 1.04
            return clamp(base * float(boost), 0.25, max(0.25, float(self.stop_atr_k)))
        if profile_key.startswith("grid_asia_probe_continuation"):
            return clamp(base * max(1.04, float(self.density_micro_stop_points_boost) * 0.98), 0.25, max(0.25, float(self.stop_atr_k)))
        if profile_key.startswith("grid_asia_probe_recovery"):
            return clamp(base * max(1.02, float(self.density_micro_stop_points_boost) * 0.96), 0.25, max(0.25, float(self.stop_atr_k)))
        if profile_key.startswith("grid_density_micro_scaler"):
            return clamp(base * float(self.density_micro_stop_points_boost), 0.25, max(0.25, float(self.stop_atr_k)))
        return base

    def _entry_tp_r(
        self,
        *,
        entry_profile: str,
        grid_mode: str,
        session_profile: str,
    ) -> float:
        profile_key = str(entry_profile or "")
        session_key = str(session_profile or "").upper()
        attack_mode = str(grid_mode or "").upper() == "ATTACK_GRID"
        prime_bonus = float(self.prime_directional_tp_r_bonus) if session_key in {"AGGRESSIVE", "MODERATE"} else 0.12
        if profile_key.startswith("grid_breakout_reclaim"):
            return 2.45 if attack_mode else 2.20
        if profile_key.startswith(("grid_directional_flow", "grid_prime_session_momentum", "grid_expansion_ready_scaler")):
            return clamp(2.0 + prime_bonus, 2.0, 2.6)
        if profile_key.startswith("grid_density_micro_scaler"):
            return clamp(float(self.density_micro_tp_r), 1.5, 2.4)
        if profile_key.startswith("grid_asia_probe_continuation"):
            return 1.95
        if profile_key.startswith("grid_asia_probe_recovery"):
            return 1.85
        if profile_key.startswith("grid_asia_probe_micro_scaler"):
            return 1.75
        return 2.0

    def _add_stop_atr(self, *, atr_ratio: float) -> float:
        base = max(0.25, float(self.add_stop_atr_k))
        if float(atr_ratio) >= 1.6:
            base *= 1.08
        elif float(atr_ratio) <= 0.9:
            base *= 0.97
        return clamp(base, 0.25, max(0.25, float(self.stop_atr_k)))

    @staticmethod
    def _normalize_symbol(value: Any) -> str:
        normalized = "".join(char for char in str(value).upper() if char.isalnum())
        if normalized.startswith("GOLD") or normalized.startswith("XAUUSD"):
            return "XAUUSD"
        return normalized

    @staticmethod
    def _rsi(series: pd.Series, period: int) -> float:
        if series.empty:
            return 50.0
        delta = series.diff()
        gains = delta.clip(lower=0.0)
        losses = -delta.clip(upper=0.0)
        avg_gain = gains.ewm(alpha=1 / max(1, period), min_periods=max(1, period), adjust=False).mean()
        avg_loss = losses.ewm(alpha=1 / max(1, period), min_periods=max(1, period), adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, pd.NA)
        values = 100 - (100 / (1 + rs))
        try:
            return float(values.iloc[-1])
        except Exception:
            return 50.0

    def _deceleration(self, features: pd.DataFrame) -> bool:
        lookback = max(3, self.entry_deceleration_lookback)
        if len(features) < lookback:
            return False
        bodies = features["m5_body"].tail(lookback).abs().reset_index(drop=True)
        last_body = float(bodies.iloc[-1])
        prior = float(bodies.iloc[:-1].mean()) if len(bodies) > 1 else last_body
        if prior <= 0:
            return False
        return last_body <= (prior * self.entry_deceleration_factor)

    @staticmethod
    def _avg_entry(positions: list[dict[str, Any]]) -> float:
        weighted = 0.0
        volume_total = 0.0
        for position in positions:
            entry = float(position.get("entry_price", 0.0))
            volume = abs(float(position.get("volume", 0.0)))
            weighted += entry * volume
            volume_total += volume
        if volume_total <= 0:
            return float(positions[-1].get("entry_price", 0.0))
        return weighted / volume_total

    @staticmethod
    def _cycle_pnl_usd(positions: list[dict[str, Any]], price: float, contract_size: float) -> float:
        total = 0.0
        for position in positions:
            side = str(position.get("side", "")).upper()
            entry = float(position.get("entry_price", 0.0))
            volume = abs(float(position.get("volume", 0.0)))
            direction = 1.0 if side == "BUY" else -1.0
            total += (price - entry) * direction * volume * max(contract_size, 1.0)
        return total

    @staticmethod
    def _open_seconds(positions: list[dict[str, Any]], now_utc: datetime) -> float:
        opened_values: list[datetime] = []
        for position in positions:
            raw = str(position.get("opened_at", ""))
            if not raw:
                continue
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            opened_values.append(parsed.astimezone(UTC))
        if not opened_values:
            return 0.0
        oldest = min(opened_values)
        now = now_utc.astimezone(UTC) if now_utc.tzinfo is not None else now_utc.replace(tzinfo=UTC)
        return max(0.0, (now - oldest).total_seconds())

    @staticmethod
    def _open_minutes(positions: list[dict[str, Any]], now_utc: datetime) -> int:
        opened_values: list[datetime] = []
        for position in positions:
            raw = str(position.get("opened_at", ""))
            if not raw:
                continue
            try:
                parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            except ValueError:
                continue
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=UTC)
            opened_values.append(parsed.astimezone(UTC))
        if not opened_values:
            return 0
        oldest = min(opened_values)
        now = now_utc.astimezone(UTC) if now_utc.tzinfo is not None else now_utc.replace(tzinfo=UTC)
        return int(max(0.0, (now - oldest).total_seconds() / 60.0))

    def _confluence_score(
        self,
        *,
        abs_stretch: float,
        deceleration: bool,
        extreme: bool,
        spread_points: float,
    ) -> float:
        score = 0.0
        if abs_stretch >= self.ema_stretch_k:
            score += 1.0
        if abs_stretch >= (self.ema_stretch_k * 1.25):
            score += 1.0
        if deceleration:
            score += 1.0
        if extreme:
            score += 1.0
        if spread_points <= (self.spread_max_points * 0.75):
            score += 1.0
        return clamp(score, 0.0, 5.0)

    @staticmethod
    def _stretch_reversion_profile(value: str) -> bool:
        return str(value or "").startswith(("grid_stretch_reversion", "grid_prime_stretch_reversion"))

    def _burst_start_count(
        self,
        *,
        session_profile: str,
        grid_mode: str,
        entry_profile: str,
        confluence: float,
        alignment_score: float,
        fractal_score: float,
        trend_efficiency: float,
        body_efficiency: float,
        compression_expansion_score: float,
        instability_score: float,
        prime_recovery_active: bool,
        support_sources: int = 0,
        grid_max_levels: int,
    ) -> int:
        support_sources = max(0, int(support_sources))
        support_burst_allowance = min(2, support_sources)
        if str(session_profile or "").upper() == "AGGRESSIVE":
            burst_cap = max(1, int(grid_max_levels))
        elif str(session_profile or "").upper() == "MODERATE":
            burst_cap = max(
                1,
                min(int(grid_max_levels), (max(1, int(self.max_open_cycles)) * 3) + 1 + support_burst_allowance),
            )
        else:
            burst_cap = max(
                1,
                min(int(grid_max_levels), (max(1, int(self.max_open_cycles)) * 2) + support_burst_allowance),
            )
        if session_profile == "ASIA_PROBE":
            return 1
        if session_profile == "MODERATE":
            if (
                confluence >= 3.4
                and alignment_score >= 0.46
                and fractal_score >= 0.44
                and trend_efficiency >= 0.34
                and instability_score <= 0.58
                and not self._stretch_reversion_profile(entry_profile)
            ):
                return min(burst_cap, max(2, int(self.moderate_burst_entries)))
            return 1
        burst_count = 1
        strong_profile = entry_profile.startswith(
            (
                "grid_liquidity_reclaim",
                "grid_trend_reclaim",
                "grid_m15_pullback_reclaim",
                "grid_breakout_reclaim",
                "grid_directional_flow",
                "grid_expansion_ready_scaler",
                "grid_prime_session_momentum",
            )
        )
        if (
            strong_profile
            and confluence >= 3.2
            and alignment_score >= 0.40
            and fractal_score >= 0.38
            and trend_efficiency >= 0.30
            and body_efficiency >= 0.22
            and instability_score <= 0.70
        ):
            burst_count = 2
        if (
            grid_mode == "ATTACK_GRID"
            and strong_profile
            and confluence >= 3.10
            and alignment_score >= 0.34
            and fractal_score >= 0.30
            and trend_efficiency >= 0.22
            and body_efficiency >= 0.14
            and compression_expansion_score >= 0.06
            and instability_score <= 0.76
        ):
            burst_count = 3
        if (
            grid_mode == "ATTACK_GRID"
            and strong_profile
            and confluence >= 3.9
            and alignment_score >= 0.50
            and fractal_score >= 0.46
            and trend_efficiency >= 0.38
            and body_efficiency >= 0.28
            and compression_expansion_score >= 0.24
            and instability_score <= 0.54
        ):
            burst_count = 4
        if (
            support_sources >= 1
            and grid_mode == "ATTACK_GRID"
            and strong_profile
            and confluence >= 3.8
            and alignment_score >= 0.48
            and fractal_score >= 0.44
            and trend_efficiency >= 0.38
            and body_efficiency >= 0.26
            and compression_expansion_score >= 0.22
            and instability_score <= 0.52
        ):
            burst_count = max(burst_count, 5)
        if (
            support_sources >= 2
            and grid_mode == "ATTACK_GRID"
            and strong_profile
            and confluence >= 4.2
            and alignment_score >= 0.52
            and fractal_score >= 0.48
            and trend_efficiency >= 0.40
            and body_efficiency >= 0.28
            and compression_expansion_score >= 0.26
            and instability_score <= 0.46
        ):
            burst_count = max(burst_count, 6)
        if (
            support_sources >= 2
            and grid_mode == "ATTACK_GRID"
            and strong_profile
            and confluence >= 4.35
            and alignment_score >= 0.56
            and fractal_score >= 0.52
            and trend_efficiency >= 0.44
            and body_efficiency >= 0.30
            and compression_expansion_score >= 0.30
            and instability_score <= 0.42
        ):
            burst_count = max(burst_count, 7)
        if (
            support_sources >= 3
            and grid_mode == "ATTACK_GRID"
            and strong_profile
            and confluence >= 4.5
            and alignment_score >= 0.60
            and fractal_score >= 0.54
            and trend_efficiency >= 0.48
            and body_efficiency >= 0.32
            and compression_expansion_score >= 0.32
            and instability_score <= 0.38
        ):
            burst_count = max(burst_count, 8)
        if prime_recovery_active and strong_profile:
            burst_count = max(
                burst_count,
                min(4 + min(2, support_sources), max(int(self.prime_burst_entries), burst_cap)),
            )
        prime_burst_cap = max(int(self.prime_burst_entries), 6 + support_burst_allowance)
        return max(1, min(burst_cap, prime_burst_cap, burst_count))

    def _burst_add_count(
        self,
        *,
        session_profile: str,
        grid_mode: str,
        favorable_move: float,
        burst_step_needed: float,
        follow_through_add_ready: bool,
        alignment_score: float,
        fractal_score: float,
        trend_efficiency: float,
        body_efficiency: float,
        instability_score: float,
        remaining_levels: int,
    ) -> int:
        if remaining_levels <= 0:
            return 0
        if str(session_profile or "").upper() == "ASIA_PROBE":
            return 0
        burst_count = 1
        if (
            session_profile == "AGGRESSIVE"
            and follow_through_add_ready
            and remaining_levels >= 2
            and favorable_move >= (burst_step_needed * 0.95)
            and alignment_score >= 0.40
            and fractal_score >= 0.34
            and trend_efficiency >= 0.28
            and body_efficiency >= 0.24
            and instability_score <= 0.68
        ):
            burst_count = min(max(2, int(self.aggressive_add_burst_entries) - 1), remaining_levels)
        if (
            session_profile == "AGGRESSIVE"
            and grid_mode == "ATTACK_GRID"
            and follow_through_add_ready
            and remaining_levels >= 2
            and favorable_move >= (burst_step_needed * 1.20)
            and alignment_score >= 0.50
            and fractal_score >= 0.48
            and trend_efficiency >= 0.44
            and body_efficiency >= 0.40
            and instability_score <= 0.48
        ):
            burst_count = min(int(self.aggressive_add_burst_entries), remaining_levels)
        if (
            session_profile == "AGGRESSIVE"
            and grid_mode == "ATTACK_GRID"
            and follow_through_add_ready
            and remaining_levels >= 3
            and favorable_move >= (burst_step_needed * 1.45)
            and alignment_score >= 0.56
            and fractal_score >= 0.54
            and trend_efficiency >= 0.48
            and body_efficiency >= 0.44
            and instability_score <= 0.42
        ):
            burst_count = min(max(burst_count, int(self.aggressive_add_burst_entries)), remaining_levels)
        return max(1, min(remaining_levels, burst_count))

    @staticmethod
    def _prime_session_native_quality_gate(
        *,
        session_name: str,
        entry_profile: str,
        quality_tier: str,
        monte_carlo_win_rate: float,
        mc_floor: float,
        htf_alignment_score: float,
        structure_cleanliness_score: float,
        execution_quality_fit: float,
        router_rank_score: float,
        support_sources: int,
    ) -> str:
        session = str(session_name or "").upper()
        profile = str(entry_profile or "")
        tier = str(quality_tier or "").upper()
        if session not in {"OVERLAP", "NEW_YORK"}:
            return ""
        directional_profile = profile.startswith(("grid_directional_flow", "grid_prime_session_momentum"))
        reclaim_profile = profile.startswith(
            (
                "grid_liquidity_reclaim",
                "grid_trend_reclaim",
                "grid_m15_pullback_reclaim",
                "grid_breakout_reclaim",
            )
        )
        if directional_profile:
            required_support_sources = 2 if session == "NEW_YORK" else 1
            required_mc = max(float(mc_floor) + (0.01 if session == "NEW_YORK" else 0.0), 0.81 if session == "NEW_YORK" else 0.80)
            required_htf = 0.58 if session == "NEW_YORK" else 0.57
            required_structure = 0.58 if session == "NEW_YORK" else 0.57
            required_execution = 0.56 if session == "NEW_YORK" else 0.56
            required_router_rank = 0.76 if session == "NEW_YORK" else 0.75
            near_a_override = (
                session == "NEW_YORK"
                and tier == "B"
                and monte_carlo_win_rate >= max(float(mc_floor) + (0.03 if session == "NEW_YORK" else 0.02), 0.84 if session == "NEW_YORK" else 0.82)
                and htf_alignment_score >= (0.62 if session == "NEW_YORK" else 0.58)
                and structure_cleanliness_score >= (0.62 if session == "NEW_YORK" else 0.58)
                and execution_quality_fit >= (0.60 if session == "NEW_YORK" else 0.58)
                and router_rank_score >= (0.79 if session == "NEW_YORK" else 0.76)
            )
            if tier != "A" and int(support_sources) < required_support_sources and not near_a_override:
                return "grid_prime_session_quality_gate"
            if monte_carlo_win_rate < required_mc:
                return "grid_prime_session_quality_gate"
            if htf_alignment_score < required_htf:
                return "grid_prime_session_quality_gate"
            if structure_cleanliness_score < required_structure or execution_quality_fit < required_execution:
                return "grid_prime_session_quality_gate"
            if router_rank_score < required_router_rank:
                return "grid_prime_session_quality_gate"
        elif reclaim_profile and tier == "C" and session == "NEW_YORK":
            if (
                monte_carlo_win_rate < max(float(mc_floor), 0.82)
                or htf_alignment_score < 0.58
                or structure_cleanliness_score < 0.58
                or execution_quality_fit < 0.56
            ):
                return "grid_prime_session_quality_gate"
        return ""

    @staticmethod
    def _london_native_burst_floor(
        *,
        burst_count: int,
        session_name: str,
        session_profile: str,
        grid_mode: str,
        entry_profile: str,
        quality_tier: str,
        confluence: float,
        monte_carlo_win_rate: float,
        mc_floor: float,
        htf_alignment_score: float,
        structure_cleanliness_score: float,
        execution_quality_fit: float,
        router_rank_score: float,
        support_sources: int,
        grid_max_levels: int,
    ) -> int:
        session = str(session_name or "").upper()
        profile = str(entry_profile or "")
        if session != "LONDON" or str(session_profile or "").upper() != "AGGRESSIVE":
            return int(burst_count)
        tier = str(quality_tier or "").upper()
        near_a_tier = (
            tier == "B"
            and monte_carlo_win_rate >= max(float(mc_floor) + 0.03, 0.85)
            and htf_alignment_score >= 0.64
            and structure_cleanliness_score >= 0.64
            and execution_quality_fit >= 0.60
            and router_rank_score >= 0.78
            and confluence >= 3.75
            and int(support_sources) >= 1
        )
        if tier != "A" and not near_a_tier:
            return int(burst_count)
        strong_profile = profile.startswith(
            (
                "grid_liquidity_reclaim",
                "grid_trend_reclaim",
                "grid_m15_pullback_reclaim",
                "grid_breakout_reclaim",
                "grid_directional_flow",
                "grid_prime_session_momentum",
            )
        )
        if not strong_profile:
            return int(burst_count)
        promoted = int(burst_count)
        if (
            monte_carlo_win_rate >= max(float(mc_floor), 0.82)
            and htf_alignment_score >= 0.58
            and structure_cleanliness_score >= 0.58
            and execution_quality_fit >= 0.56
            and confluence >= 3.20
        ):
            promoted = max(promoted, 2 if int(support_sources) < 2 else 3)
        if near_a_tier:
            promoted = max(promoted, 4 if profile.startswith(("grid_directional_flow", "grid_prime_session_momentum")) else 3)
        if (
            str(grid_mode or "").upper() == "ATTACK_GRID"
            and monte_carlo_win_rate >= max(float(mc_floor) + 0.03, 0.85)
            and htf_alignment_score >= 0.62
            and structure_cleanliness_score >= 0.64
            and execution_quality_fit >= 0.60
            and router_rank_score >= 0.78
            and confluence >= 3.55
            and int(support_sources) >= 1
        ):
            promoted = max(promoted, 5 if profile.startswith(("grid_directional_flow", "grid_prime_session_momentum")) else 4)
        if (
            profile.startswith(("grid_liquidity_reclaim", "grid_breakout_reclaim", "grid_trend_reclaim"))
            and str(grid_mode or "").upper() == "ATTACK_GRID"
            and monte_carlo_win_rate >= max(float(mc_floor) + 0.05, 0.87)
            and htf_alignment_score >= 0.68
            and structure_cleanliness_score >= 0.70
            and execution_quality_fit >= 0.64
            and router_rank_score >= 0.82
            and confluence >= 3.90
            and int(support_sources) >= 2
        ):
            promoted = max(promoted, 6)
        return max(1, min(int(grid_max_levels), int(promoted)))

    @staticmethod
    def _micro_take_profile(*, session_name: str, session_profile: str) -> dict[str, float]:
        session = str(session_name or "").upper()
        profile = str(session_profile or "").upper()
        if session == "NEW_YORK" and profile == "AGGRESSIVE":
            return {
                "quick_green_multiplier": 0.65,
                "mean_revert_multiplier": 0.85,
                "min_open_minutes": 2.0,
            }
        if session in {"SYDNEY", "TOKYO"} and profile == "ASIA_PROBE":
            return {
                "quick_green_multiplier": 0.28,
                "mean_revert_multiplier": 0.38,
                "min_open_minutes": 0.0,
            }
        return {
            "quick_green_multiplier": 0.40,
            "mean_revert_multiplier": 0.55,
            "min_open_minutes": 1.0,
        }

    def _cycle_id_from_positions(self, positions: list[dict[str, Any]]) -> str:
        for position in positions:
            cycle_id = str(position.get("grid_cycle_id") or "").strip()
            if cycle_id:
                return cycle_id
        first_signal = str((positions[0] if positions else {}).get("signal_id") or "").strip()
        if first_signal:
            return first_signal.split("::", 1)[0]
        first_opened = str((positions[0] if positions else {}).get("opened_at") or "").strip()
        return deterministic_id(self.symbol, "grid", "cycle", first_opened or "open")

    def _step_points(self, *, atr: float, multiplier: float = 1.0) -> float:
        atr_points = (max(0.0, float(atr)) * max(1.0, float(multiplier)) * max(0.0, self.step_atr_k)) / max(self.point_size, 1e-6)
        low = min(float(self.step_points_min), float(self.step_points_max))
        high = max(float(self.step_points_min), float(self.step_points_max))
        return clamp(atr_points, low, high)

    def _spread_spacing_floor(self, *, spread_points: float) -> float:
        return max(0.0, float(spread_points)) * max(0.0, float(self.spread_spacing_multiplier)) + max(
            0.0,
            float(self.spread_spacing_buffer_points),
        )

    def _apply_spread_spacing_floor(self, *, step_points: float, spread_points: float) -> float:
        return max(float(step_points), self._spread_spacing_floor(spread_points=spread_points))

    def _entry_stop_points(self, *, step_points: float, probe_candidate: bool) -> float:
        multiplier = float(self.entry_stop_step_multiplier) * (0.92 if probe_candidate else 1.0)
        low = min(float(self.entry_stop_points_min), float(self.entry_stop_points_max))
        high = max(float(self.entry_stop_points_min), float(self.entry_stop_points_max))
        return clamp(max(0.0, float(step_points)) * max(0.0, multiplier), low, high)

    def _entry_stop_points_for_profile(
        self,
        *,
        step_points: float,
        probe_candidate: bool,
        entry_profile: str,
        session_profile: str,
    ) -> float:
        base = self._entry_stop_points(step_points=step_points, probe_candidate=probe_candidate)
        profile_key = str(entry_profile or "")
        session_key = str(session_profile or "").upper()
        low = min(float(self.entry_stop_points_min), float(self.entry_stop_points_max))
        high = max(float(self.entry_stop_points_min), float(self.entry_stop_points_max))
        if profile_key.startswith(("grid_directional_flow", "grid_prime_session_momentum", "grid_expansion_ready_scaler", "grid_breakout_reclaim")):
            boosted_high = high * (1.18 if session_key in {"AGGRESSIVE", "MODERATE"} else 1.10)
            return clamp(base * float(self.prime_directional_stop_points_boost), low, boosted_high)
        if profile_key.startswith("grid_asia_probe_continuation"):
            return clamp(base * max(1.04, float(self.density_micro_stop_points_boost) * 0.98), low, high * 1.10)
        if profile_key.startswith("grid_asia_probe_recovery"):
            return clamp(base * max(1.02, float(self.density_micro_stop_points_boost) * 0.96), low, high * 1.08)
        if profile_key.startswith("grid_density_micro_scaler"):
            return clamp(base * float(self.density_micro_stop_points_boost), low, high * 1.10)
        return base

    def _add_stop_points(self, *, step_points: float) -> float:
        low = min(float(self.add_stop_points_min), float(self.add_stop_points_max))
        high = max(float(self.add_stop_points_min), float(self.add_stop_points_max))
        return clamp(max(0.0, float(step_points)) * max(0.0, float(self.add_stop_step_multiplier)), low, high)
