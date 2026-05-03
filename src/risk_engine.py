from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import re
from typing import Any

from src.session_calendar import NEW_YORK, SYDNEY, dominant_session_name, is_weekend_market_mode
from src.trade_quality import is_xau_grid_lane
from src.utils import clamp


_FUNDED_PROVIDER_PATTERNS: tuple[str, ...] = (
    r"\bftmo\b",
    r"\bthe\s*5ers\b",
    r"\b5ers\b",
    r"\bmyfundedfx\b",
    r"\bfundednext\b",
    r"\bfunding\s*pips\b",
    r"\bfundingpips\b",
    r"\bfunderpro\b",
    r"\btopstepx?\b",
    r"\bblueberry\s+funded\b",
    r"\bbespoke\s+funding\b",
    r"\bbrightfunded\b",
    r"\balpha\s+capital\b",
    r"\bskilled\s+funder\b",
    r"\bgoat\s+funded\b",
    r"\bfunded\s+trading\s+plus\b",
    r"\bthe\s+trading\s+pit\b",
)
_FUNDED_PHASE_PATTERNS: tuple[str, ...] = (
    r"\bchallenge\b",
    r"\bevaluation\b",
    r"\bverification\b",
    r"\bphase\s*[12]\b",
    r"\binstant\s+funding\b",
    r"\bfunded\s+account\b",
    r"\bprop\s+firm\b",
    r"\bprop\s+challenge\b",
)


def detect_funded_account_mode(*parts: Any) -> bool:
    descriptor = " ".join(str(part or "") for part in parts).strip().lower()
    if not descriptor:
        return False
    provider_hits = sum(1 for pattern in _FUNDED_PROVIDER_PATTERNS if re.search(pattern, descriptor))
    phase_hits = sum(1 for pattern in _FUNDED_PHASE_PATTERNS if re.search(pattern, descriptor))
    generic_funded = bool(re.search(r"\bfunded\b", descriptor))
    generic_prop = bool(re.search(r"\bprop\b", descriptor))
    if provider_hits >= 1:
        return True
    if phase_hits >= 2 and (generic_funded or generic_prop):
        return True
    if phase_hits >= 3:
        return True
    return False


def is_funded_challenge_phase(phase: str) -> bool:
    value = str(phase or "").strip().lower()
    if not value:
        return True
    return any(token in value for token in ("challenge", "evaluation", "verification", "phase 1", "phase 2", "phase_1", "phase_2", "eval"))


@dataclass
class TradeStats:
    win_rate: float = 0.5
    avg_win_r: float = 1.0
    avg_loss_r: float = 1.0
    consecutive_losses: int = 0
    winning_streak: int = 0
    cooldown_trades_remaining: int = 0
    closed_trades_total: int = 0
    trades_today: int = 0
    daily_pnl_pct: float = 0.0
    day_start_equity: float = 0.0
    day_high_equity: float = 0.0
    daily_realized_pnl: float = 0.0
    daily_dd_pct_live: float = 0.0
    rolling_drawdown_pct: float = 0.0
    absolute_drawdown_pct: float = 0.0
    soft_dd_trade_count: int = 0
    trading_day_key: str = ""
    timezone_used: str = "Australia/Sydney"
    previous_trading_day_key: str = ""
    today_closed_trade_ids: list[str] = field(default_factory=list)
    today_closed_trade_count: int = 0
    today_closed_trade_times_raw: list[str] = field(default_factory=list)
    today_closed_trade_times_sydney: list[str] = field(default_factory=list)
    today_closed_trade_details: list[dict[str, Any]] = field(default_factory=list)
    day_bucket_source: str = "close_time_sydney"
    reset_triggered_at: str = ""
    reset_reason: str = ""
    day_start_equity_source: str = "state:day_start_equity"
    day_high_equity_source: str = "state:day_high_equity"


@dataclass
class RiskInputs:
    symbol: str
    mode: str
    live_enabled: bool
    live_allowed: bool
    current_time: datetime
    spread_points: float
    entry_price: float
    stop_price: float
    tp_price: float
    equity: float
    account_balance: float
    margin_free: float
    open_positions: int
    open_positions_symbol: int
    same_direction_positions: int
    session_multiplier: float
    symbol_point: float
    contract_size: float
    volume_min: float
    volume_max: float
    volume_step: float
    requested_risk_pct: float
    hard_risk_cap: float
    max_positions: int
    max_positions_per_symbol: int
    max_daily_loss: float
    circuit_breaker_daily_loss: float
    max_drawdown_kill: float
    absolute_drawdown_hard_stop: float
    max_spread_points: float
    atr_current: float
    atr_average: float
    atr_spike_multiple: float
    volatility_pause_minutes: int
    regime: str
    ai_probability: float
    ai_size_multiplier: float
    portfolio_size_multiplier: float
    recent_trades_last_hour: int
    max_trades_per_hour: int
    use_kelly: bool
    kelly_fraction: float
    use_fixed_lot: bool
    fixed_lot: float
    stats: TradeStats
    friday_cutoff_hour: int = 20
    weekend_trading_allowed: bool = False
    micro_enabled: bool = False
    micro_min_trades: int = 50
    micro_risk_pct_floor: float = 0.001
    micro_risk_pct_ceiling: float = 0.0025
    micro_daily_loss_pct: float = 0.01
    first_trade_protection_trades: int = 2
    first_trade_size_factor: float = 0.5
    first_trade_max_sl_atr: float = 1.4
    anti_martingale_enabled: bool = False
    anti_martingale_step: float = 0.05
    anti_martingale_cap: float = 1.25
    min_stop_distance_points: float = 0.0
    projected_open_risk_usd: float = 0.0
    projected_cycle_risk_usd: float = 0.0
    micro_max_loss_usd: float = 2.5
    micro_total_risk_usd: float = 5.0
    setup: str = ""
    candidate_stop_atr: float = 0.0
    symbol_digits: int | None = None
    symbol_tick_size: float | None = None
    symbol_tick_value: float | None = None
    margin_per_lot: float | None = None
    account_leverage: float | None = None
    margin_safety_fraction: float = 0.25
    strategy_risk_cap: float | None = None
    skip_micro_risk_clamp: bool = False
    max_loss_usd_floor: float = 0.0
    no_trade_boost_enabled: bool = False
    no_trade_boost_eligible: bool = False
    no_trade_boost_elapsed_minutes: float = 0.0
    no_trade_boost_after_minutes: int = 60
    no_trade_boost_interval_minutes: int = 15
    no_trade_boost_step_pct: float = 0.02
    no_trade_boost_max_pct: float = 0.10
    account_currency: str = "USD"
    economics_source: str = "fallback"
    confluence_score: float = 0.0
    expected_value_r: float = 0.0
    max_trades_per_day: int = 999
    overflow_max_trades_per_day: int = 0
    bootstrap_enabled: bool = False
    bootstrap_equity_threshold: float = 100.0
    bootstrap_per_trade_hard_cap: float = 2.5
    bootstrap_total_exposure_cap: float = 5.0
    bootstrap_min_risk_amount: float = 0.5
    bootstrap_min_lot_risk_multiplier: float = 4.0
    bootstrap_drawdown_kill: float | None = None
    bootstrap_first_trade_max_sl_atr: float = 3.0
    soft_daily_dd_pct: float = 0.05
    hard_daily_dd_pct: float = 0.07
    soft_dd_probability_floor: float = 0.68
    soft_dd_expected_value_floor: float = 0.35
    soft_dd_confluence_floor: float = 3.8
    current_phase: str = "PHASE_1"
    current_ai_threshold_mode: str = "conservative"
    current_base_risk_pct: float = 0.0
    current_max_risk_pct: float = 0.0
    lane_consecutive_losses: int = 0
    same_lane_loss_caution_streak: int = 3
    low_equity_safety_enabled: bool = True
    low_equity_max_equity: float = 300.0
    low_equity_risk_floor_pct: float = 0.01
    low_equity_risk_ceiling_pct: float = 0.02
    low_equity_monte_carlo_floor: float = 0.88
    candidate_monte_carlo_win_rate: float = 0.0
    spread_atr_reference_points: float = 0.0
    low_equity_spread_atr_cap: float = 1.2
    trade_quality_score: float = 0.0
    trade_quality_band: str = ""
    trade_quality_detail: str = ""
    quality_size_multiplier: float = 1.0
    regime_confidence: float = 0.0
    execution_quality_score: float = 0.70
    execution_quality_state: str = "GOOD"
    spread_quality_score: float = 0.70
    session_quality_score: float = 0.70
    recent_expectancy_r: float = 0.0
    recent_win_rate: float = 0.5
    total_open_risk_pct: float = 0.0
    correlation_multiplier: float = 1.0
    correlation_cluster: str = ""
    news_state: str = "NEWS_SAFE"
    news_confidence: float = 1.0
    degraded_mode_active: bool = False
    daily_caution_threshold_pct: float = 0.03
    daily_defensive_threshold_pct: float = 0.05
    daily_hard_stop_threshold_pct: float = 0.07
    daily_governor_timeout_hours: float = 4.0
    daily_governor_started_at: str = ""
    daily_governor_trigger_day_key: str = ""
    daily_governor_force_release: bool = False
    daily_governor_emergency_review_ready: bool = False
    daily_normal_quality_floor: float = 0.58
    daily_caution_quality_floor: float = 0.70
    daily_defensive_quality_floor: float = 0.85
    daily_normal_risk_multiplier: float = 1.0
    daily_caution_risk_multiplier: float = 0.75
    daily_defensive_risk_multiplier: float = 0.45
    daily_hard_stop_risk_multiplier: float = 0.0
    strategy_family: str = ""
    lane_name: str = ""
    proof_bucket_state: str = "neutral"
    proof_exception_allowed: bool = False
    proof_exception_reason: str = ""
    xau_grid_sub_budget_pct: float = 0.35
    xau_grid_cycle_quality: float = 0.0
    xau_grid_caution_cycle_quality: float = 0.85
    xau_grid_defensive_cycle_quality: float = 0.92
    xau_grid_caution_risk_multiplier: float = 0.90
    xau_grid_defensive_risk_multiplier: float = 0.65
    stretch_max_trades_per_day: int = 0
    hard_upper_limit: int = 0
    stretch_max_trades_per_hour: int = 0
    cluster_mode_active: bool = False
    quality_cluster_score: float = 0.0
    lane_strength_multiplier: float = 1.0
    lane_score: float = 0.0
    lane_expectancy_multiplier: float = 1.0
    lane_expectancy_score: float = 0.0
    session_density_score: float = 0.0
    winning_streak_mode_active: bool = False
    current_capacity_mode: str = "BASE"
    session_priority_profile: str = "GLOBAL"
    lane_session_priority: str = "NEUTRAL"
    session_native_pair: bool = False
    session_priority_multiplier: float = 1.0
    pair_priority_rank_in_session: int = 99
    lane_budget_share: float = 0.0
    lane_available_capacity: float = 0.0
    hot_hand_active: bool = False
    hot_hand_score: float = 0.0
    session_bankroll_bias: float = 1.0
    profit_recycle_active: bool = False
    profit_recycle_boost: float = 0.0
    close_winners_score: float = 0.5
    soft_trade_budget_enabled: bool = False
    aggression_profile: str = "BOUNDED_AGGRO"
    aggression_lane_multiplier: float = 1.0
    hot_lane_concurrency_bonus: int = 0
    microstructure_alignment_score: float = 0.0
    microstructure_confidence: float = 0.0
    microstructure_composite_score: float = 0.5
    lead_lag_alignment_score: float = 0.0
    lead_lag_confidence: float = 0.0
    lead_lag_disagreement_penalty: float = 0.0
    event_playbook: str = ""
    event_base_class: str = ""
    event_pre_position_allowed: bool = False
    execution_minute_quality_score: float = 0.70
    execution_minute_size_multiplier: float = 1.0
    execution_minute_state: str = "MIXED"
    exceptional_override_used: bool = False
    exceptional_override_reason: str = ""
    recovery_mode_active: bool = False
    funded_account_mode: bool = False
    funded_phase: str = ""
    funded_daily_loss_limit_pct: float = 0.05
    funded_overall_drawdown_limit_pct: float = 0.10
    funded_profit_target_pct: float = 0.08
    funded_remaining_target_pct: float = 0.08
    funded_guard_buffer_pct: float = 0.02


@dataclass
class RiskDecision:
    approved: bool
    kill: str | None
    volume: float
    risk_pct: float
    reason: str
    extra_confluence_required: bool = False
    adjusted_size_factor: float = 1.0
    projected_loss_usd: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyGovernorState:
    state: str
    reason: str
    risk_multiplier: float
    quality_floor: float
    allowed_by_daily_governor: bool
    blocked_reason: str = ""
    proof_exception_used: bool = False
    xau_grid_override_allowed: bool = False
    xau_grid_override_reason: str = ""
    xau_grid_sub_budget: float = 0.0
    xau_grid_risk_multiplier: float = 0.0


@dataclass
class SessionGovernorState:
    session_name: str
    state: str
    reason: str
    risk_multiplier: float
    allowed: bool
    session_realized_pnl: float = 0.0
    session_realized_pnl_pct: float = 0.0
    session_trade_count: int = 0
    blocked_reason: str = ""


@dataclass
class RiskEngine:
    volatility_pauses: dict[str, datetime] = field(default_factory=dict)

    @staticmethod
    def _quality_rank(value: str, score: float) -> tuple[int, str]:
        normalized = str(value or "").strip().upper()
        if normalized in {"A+", "A", "A-", "B+", "B", "C"}:
            ranking = {"A+": 6, "A": 5, "A-": 4, "B+": 3, "B": 2, "C": 1}
            return ranking[normalized], normalized
        coarse = str(value or "").strip().lower()
        if coarse == "elite":
            return 6, "A+"
        if coarse == "strong":
            return 3 if float(score) < 0.84 else 5, "B+" if float(score) < 0.84 else "A"
        if coarse == "acceptable":
            return 2, "B"
        if float(score) >= 0.90:
            return 6, "A+"
        if float(score) >= 0.84:
            return 5, "A"
        if float(score) >= 0.78:
            return 4, "A-"
        if float(score) >= 0.70:
            return 3, "B+"
        if float(score) >= 0.58:
            return 2, "B"
        if float(score) >= 0.48:
            return 1, "C"
        return 0, "REJECT"

    @staticmethod
    def _stats_value(payload: RiskInputs, field_name: str, default: Any = 0.0) -> Any:
        return getattr(payload.stats, field_name, default)

    @classmethod
    def _stats_text(cls, payload: RiskInputs, field_name: str, default: str = "") -> str:
        return str(cls._stats_value(payload, field_name, default) or "").strip()

    @staticmethod
    def _payload_trading_day_key(payload: RiskInputs) -> str:
        trading_day_key = RiskEngine._stats_text(payload, "trading_day_key")
        if trading_day_key:
            return trading_day_key
        current_time = payload.current_time
        if current_time.tzinfo is None:
            current_time = current_time.replace(tzinfo=SYDNEY)
        return current_time.astimezone(SYDNEY).date().isoformat()

    @classmethod
    def _daily_governor_release_flags(cls, payload: RiskInputs) -> dict[str, Any]:
        timeout_hours = max(0.0, float(payload.daily_governor_timeout_hours or 0.0))
        current_day_key = cls._payload_trading_day_key(payload)
        trigger_day_key = str(payload.daily_governor_trigger_day_key or "").strip()
        sydney_reset_release = bool(trigger_day_key and current_day_key and trigger_day_key != current_day_key)
        timeout_elapsed = False
        started_at_raw = str(payload.daily_governor_started_at or "").strip()
        if started_at_raw and timeout_hours > 0.0:
            try:
                started_at = datetime.fromisoformat(started_at_raw.replace("Z", "+00:00"))
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=SYDNEY)
                current_time = payload.current_time
                if current_time.tzinfo is None:
                    current_time = current_time.replace(tzinfo=SYDNEY)
                timeout_elapsed = (current_time.astimezone(started_at.tzinfo) - started_at).total_seconds() >= (
                    timeout_hours * 3600.0
                )
            except Exception:
                timeout_elapsed = False
        force_release = bool(payload.daily_governor_force_release) or sydney_reset_release or timeout_elapsed
        return {
            "force_release": bool(force_release),
            "timeout_elapsed": bool(timeout_elapsed),
            "sydney_reset_release": bool(sydney_reset_release),
            "current_day_key": current_day_key,
            "trigger_day_key": trigger_day_key,
        }

    @classmethod
    def _apply_daily_governor_release(
        cls,
        payload: RiskInputs,
        *,
        state: str,
        state_reason: str,
    ) -> tuple[str, str, dict[str, Any]]:
        release_flags = cls._daily_governor_release_flags(payload)
        if state == "DAILY_NORMAL" or not bool(release_flags.get("force_release")):
            return state, state_reason, release_flags
        release_suffix = (
            "sydney_reset_release"
            if bool(release_flags.get("sydney_reset_release"))
            else "governor_timeout_release"
            if bool(release_flags.get("timeout_elapsed"))
            else "governor_force_release"
        )
        if (
            state == "DAILY_HARD_STOP"
            and not bool(payload.recovery_mode_active)
            and float(payload.stats.daily_pnl_pct or 0.0) < 0.0
            and float(payload.stats.daily_dd_pct_live or 0.0) > float(payload.daily_caution_threshold_pct or 0.0)
        ):
            return "DAILY_DEFENSIVE", f"{state_reason}:{release_suffix}", release_flags
        return "DAILY_NORMAL", f"{state_reason}:{release_suffix}", release_flags

    @staticmethod
    def _session_governor_name(current_time: datetime) -> str:
        session_name = str(dominant_session_name(current_time) or "").upper()
        if session_name == "OVERLAP":
            return "NEW_YORK"
        return session_name or "NONE"

    @classmethod
    def _session_thresholds(cls, session_name: str) -> tuple[float, float]:
        session_key = str(session_name or "").upper()
        if session_key == "SYDNEY":
            return 0.02, 0.035
        if session_key == "TOKYO":
            return 0.025, 0.04
        if session_key == "LONDON":
            return 0.03, 0.05
        if session_key == "NEW_YORK":
            return 0.03, 0.05
        return 1.0, 1.0

    @classmethod
    def _resolve_session_governor(cls, payload: RiskInputs) -> SessionGovernorState:
        session_name = cls._session_governor_name(payload.current_time)
        soft_stop, hard_stop = cls._session_thresholds(session_name)
        details = list(payload.stats.today_closed_trade_details or [])
        session_rows = [
            row
            for row in details
            if str(row.get("session_name", "") or "").upper() in {session_name, "OVERLAP" if session_name == "NEW_YORK" else session_name}
        ]
        session_realized_pnl = float(sum(float(row.get("pnl_amount", 0.0) or 0.0) for row in session_rows))
        baseline = max(float(payload.stats.day_start_equity or payload.equity or 1.0), 1.0)
        session_realized_pnl_pct = abs(session_realized_pnl) / baseline if session_realized_pnl < 0 else 0.0
        session_trade_count = int(len(session_rows))
        if session_realized_pnl_pct >= hard_stop:
            return SessionGovernorState(
                session_name=session_name,
                state="SESSION_HARD_STOP",
                reason="session_realized_pnl_hard_stop",
                risk_multiplier=0.0,
                allowed=False,
                session_realized_pnl=session_realized_pnl,
                session_realized_pnl_pct=session_realized_pnl_pct,
                session_trade_count=session_trade_count,
                blocked_reason="session_hard_stop",
            )
        if session_realized_pnl_pct >= soft_stop:
            return SessionGovernorState(
                session_name=session_name,
                state="SESSION_CAUTION",
                reason="session_realized_pnl_soft_stop",
                risk_multiplier=0.82,
                allowed=True,
                session_realized_pnl=session_realized_pnl,
                session_realized_pnl_pct=session_realized_pnl_pct,
                session_trade_count=session_trade_count,
            )
        return SessionGovernorState(
            session_name=session_name,
            state="SESSION_NORMAL",
            reason="session_governor_normal",
            risk_multiplier=1.0,
            allowed=True,
            session_realized_pnl=session_realized_pnl,
            session_realized_pnl_pct=session_realized_pnl_pct,
            session_trade_count=session_trade_count,
        )

    @staticmethod
    def _bootstrap_min_lot_tolerance_cap(payload: RiskInputs, per_trade_cap: float) -> float:
        base_cap = max(0.0, float(per_trade_cap))
        if base_cap <= 0:
            return 0.0
        symbol_key = str(payload.symbol or "").upper()
        setup_key = str(payload.setup or "").upper()
        session_name = str(dominant_session_name(payload.current_time) or "").upper()
        threshold = max(1.0, float(payload.bootstrap_equity_threshold or 0.0))
        equity_ratio = max(0.0, float(payload.equity) / threshold)
        multiplier = 1.10
        if float(payload.volume_min) >= 1.0:
            multiplier = 1.15
        elif float(payload.volume_min) >= 0.1 or symbol_key.startswith(("NAS100", "USTEC", "US100")):
            multiplier = 1.45
            if equity_ratio >= 0.50:
                multiplier = 1.70
            elif equity_ratio >= 0.35:
                multiplier = 1.55
        elif equity_ratio >= 0.50:
            multiplier = 1.20
        strong_bootstrap_candidate = bool(
            float(payload.ai_probability or 0.0) >= 0.78
            and float(payload.expected_value_r or 0.0) >= 0.85
            and float(payload.confluence_score or 0.0) >= 4.0
        )
        if symbol_key.startswith(("XAUUSD", "GOLD")) and setup_key.startswith("XAUUSD_M5_GRID_SCALPER"):
            multiplier = max(multiplier, 1.26 if session_name in {"TOKYO", "SYDNEY"} else 1.32)
            if strong_bootstrap_candidate:
                multiplier = max(multiplier, 1.82 if session_name in {"LONDON", "OVERLAP", "NEW_YORK"} else 1.50)
        elif symbol_key.startswith(("USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY", "AUDNZD", "EURUSD", "GBPUSD", "XAGUSD")):
            if strong_bootstrap_candidate or bool(payload.session_native_pair):
                multiplier = max(multiplier, 1.36 if equity_ratio < 0.35 else 1.28)
        elif symbol_key.startswith(("NAS100", "USTEC", "US100", "USOIL")):
            if strong_bootstrap_candidate or bool(payload.session_native_pair):
                multiplier = max(multiplier, 1.82 if equity_ratio < 0.35 else 1.68)
        return base_cap * multiplier

    @staticmethod
    def _allow_low_equity_high_spread_override(
        payload: RiskInputs,
        *,
        low_equity_xau_override: bool,
        low_equity_attack_override: bool,
        spread_reference: float,
    ) -> bool:
        spread_points = max(0.0, float(payload.spread_points or 0.0))
        if spread_points <= max(0.0, float(payload.max_spread_points or 0.0)):
            return False
        symbol_key = str(payload.symbol or "").upper()
        if symbol_key.startswith("BTCUSD") and is_weekend_market_mode(payload.current_time):
            spread_cap = max(
                float(payload.max_spread_points or 0.0) * 800.0,
                spread_reference * 60.0,
                5000.0,
            )
            return spread_points <= spread_cap
        if low_equity_xau_override:
            return spread_points <= max(
                float(payload.max_spread_points or 0.0) * 1.35,
                spread_reference * 2.45,
            )
        if not low_equity_attack_override:
            return False
        if symbol_key.startswith(("NAS100", "USTEC", "US100")):
            spread_cap = max(float(payload.max_spread_points or 0.0) * 2.40, spread_reference * 4.90, 90.0)
        elif symbol_key.startswith(("USOIL", "XAGUSD")):
            spread_cap = max(float(payload.max_spread_points or 0.0) * 1.80, spread_reference * 2.60)
        elif symbol_key.startswith(("USDJPY", "EURJPY", "GBPJPY", "AUDJPY", "NZDJPY")):
            spread_cap = max(float(payload.max_spread_points or 0.0) * 1.45, spread_reference * 1.95)
        else:
            spread_cap = max(float(payload.max_spread_points or 0.0) * 1.35, spread_reference * 1.70)
        return spread_points <= spread_cap

    @staticmethod
    def resolve_daily_state_from_stats(
        stats: TradeStats,
        *,
        caution_threshold_pct: float = 0.03,
        defensive_threshold_pct: float = 0.05,
        hard_stop_threshold_pct: float = 0.07,
    ) -> tuple[str, str]:
        daily_loss_pressure = max(0.0, -float(getattr(stats, "daily_pnl_pct", 0.0) or 0.0))
        daily_dd_pressure = max(0.0, float(getattr(stats, "daily_dd_pct_live", 0.0) or 0.0))
        effective_pressure = max(daily_loss_pressure, daily_dd_pressure)
        if effective_pressure >= max(float(hard_stop_threshold_pct), float(defensive_threshold_pct), float(caution_threshold_pct)):
            if daily_dd_pressure >= float(hard_stop_threshold_pct):
                return "DAILY_HARD_STOP", "daily_dd_pct_live_hard_stop"
            if daily_loss_pressure >= float(hard_stop_threshold_pct):
                return "DAILY_HARD_STOP", "daily_pnl_pct_hard_stop"
        if effective_pressure >= max(float(defensive_threshold_pct), float(caution_threshold_pct)):
            if daily_dd_pressure >= float(defensive_threshold_pct):
                return "DAILY_DEFENSIVE", "daily_dd_pct_live_defensive"
            if daily_loss_pressure >= float(defensive_threshold_pct):
                return "DAILY_DEFENSIVE", "daily_pnl_pct_defensive"
        if effective_pressure >= float(caution_threshold_pct):
            if daily_dd_pressure >= float(caution_threshold_pct):
                return "DAILY_CAUTION", "daily_dd_pct_live_caution"
            if daily_loss_pressure >= float(caution_threshold_pct):
                return "DAILY_CAUTION", "daily_pnl_pct_caution"
        return "DAILY_NORMAL", "daily_governor_normal"

    @classmethod
    def _resolve_daily_governor(
        cls,
        payload: RiskInputs,
        *,
        spread_elevated: bool,
    ) -> DailyGovernorState:
        state, state_reason = cls.resolve_daily_state_from_stats(
            payload.stats,
            caution_threshold_pct=float(payload.daily_caution_threshold_pct),
            defensive_threshold_pct=float(payload.daily_defensive_threshold_pct),
            hard_stop_threshold_pct=float(payload.daily_hard_stop_threshold_pct),
        )
        state, state_reason, _release_flags = cls._apply_daily_governor_release(
            payload,
            state=state,
            state_reason=state_reason,
        )
        if (
            str(state).upper() == "DAILY_NORMAL"
            and max(0, int(payload.lane_consecutive_losses or 0)) >= max(2, int(payload.same_lane_loss_caution_streak or 3))
        ):
            state = "DAILY_CAUTION"
            state_reason = "same_lane_loss_streak_caution"
        confluence_value = float(payload.confluence_score or 0.0)
        confluence_component = clamp((confluence_value / 5.0) if confluence_value > 1.0 else confluence_value, 0.0, 1.0)
        expected_value_component = clamp(float(payload.expected_value_r or 0.0) / 1.5, 0.0, 1.0)
        explicit_quality_score = clamp(float(payload.trade_quality_score or 0.0), 0.0, 1.0)
        inferred_quality_score = max(
            float(payload.ai_probability or 0.0),
            confluence_component,
            expected_value_component * 0.8,
        )
        quality_score = clamp(explicit_quality_score or inferred_quality_score, 0.0, 1.0)
        band_rank, band_label = cls._quality_rank(
            str(payload.trade_quality_detail or payload.trade_quality_band or ""),
            quality_score,
        )
        proof_state = str(payload.proof_bucket_state or "neutral").strip().lower()
        proof_allowed = bool(payload.proof_exception_allowed) and proof_state in {"trusted", "elite-proof"}
        is_elite = band_rank >= 5 or quality_score >= 0.90
        is_strong = band_rank >= 3 or quality_score >= 0.70
        is_acceptable = band_rank >= 2 or quality_score >= float(payload.daily_normal_quality_floor)
        session_name = dominant_session_name(payload.current_time)
        is_xau_grid = cls._is_xau_grid_setup(payload)
        strategy_family = str(payload.strategy_family or "").strip().upper()
        weak_family = strategy_family in {"RANGE/REVERSION", "RANGE", "REVERSION"}
        session_native_pair = bool(payload.session_native_pair) or float(payload.session_priority_multiplier or 1.0) >= 1.08
        session_priority_boosted = float(payload.session_priority_multiplier or 1.0) >= 1.05
        non_native_pair = not session_native_pair and not is_xau_grid
        strong_lane = (
            is_xau_grid
            or float(payload.lane_strength_multiplier or 1.0) >= 1.02
            or float(payload.lane_budget_share or 0.0) >= 0.20
            or str(payload.lane_name or "").upper() in {"FX_SESSION_SCALP", "FX_DAYTRADE"}
            or is_xau_grid_lane(str(payload.lane_name or ""))
        )
        preserve_strong_session_flow = bool(
            (session_native_pair or is_xau_grid or (session_priority_boosted and strong_lane))
            and str(payload.execution_quality_state or "GOOD").upper() == "GOOD"
            and not bool(payload.degraded_mode_active)
            and float(payload.correlation_multiplier or 1.0) >= 0.82
        )
        btc_weekday_caution_block = (
            str(payload.symbol or "").upper().startswith("BTC")
            and session_name in {"LONDON", "OVERLAP", "NEW_YORK", "TOKYO", "SYDNEY"}
            and state in {"DAILY_CAUTION", "DAILY_DEFENSIVE"}
            and float(payload.news_confidence or 0.0) < 0.78
        )
        xau_grid_sub_budget = max(
            0.0,
            float(payload.equity)
            * max(float(payload.current_base_risk_pct or payload.requested_risk_pct or 0.0), 0.0)
            * max(float(payload.xau_grid_sub_budget_pct or 0.0), 0.0),
        )
        xau_grid_remaining_budget = max(0.0, xau_grid_sub_budget - max(0.0, float(payload.projected_cycle_risk_usd or 0.0)))
        xau_grid_override_allowed = False
        xau_grid_override_reason = ""
        xau_grid_risk_multiplier = 0.0
        if is_xau_grid and state in {"DAILY_CAUTION", "DAILY_DEFENSIVE"}:
            required_cycle_quality = (
                float(payload.xau_grid_caution_cycle_quality)
                if state == "DAILY_CAUTION"
                else float(payload.xau_grid_defensive_cycle_quality)
            )
            xau_grid_risk_multiplier = (
                float(payload.xau_grid_caution_risk_multiplier)
                if state == "DAILY_CAUTION"
                else float(payload.xau_grid_defensive_risk_multiplier)
            )
            xau_grid_override_allowed = (
                session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
                and float(payload.xau_grid_cycle_quality or 0.0) >= required_cycle_quality
                and str(payload.execution_quality_state or "GOOD").upper() == "GOOD"
                and float(payload.spread_quality_score or 0.0) >= 0.75
                and not spread_elevated
                and str(payload.news_state or "NEWS_SAFE").upper() not in {"NEWS_BLOCK", "NEWS_BLOCKED", "NEWS_DISTORTION"}
                and float(payload.news_confidence or 0.0) >= 0.70
                and not bool(payload.degraded_mode_active)
                and float(payload.correlation_multiplier or 1.0) >= 0.85
                and xau_grid_remaining_budget > 0.0
            )
            if xau_grid_override_allowed:
                xau_grid_override_reason = "protected_xau_grid_lane"
            else:
                xau_grid_override_reason = "xau_grid_lane_requirements_not_met"

        if state == "DAILY_HARD_STOP" and bool(payload.recovery_mode_active):
            state = "DAILY_DEFENSIVE"
            state_reason = f"{state_reason}:hard_stop_recovery_mode"

        if state == "DAILY_HARD_STOP":
            return DailyGovernorState(
                state=state,
                reason=state_reason,
                risk_multiplier=float(payload.daily_hard_stop_risk_multiplier),
                quality_floor=1.0,
                allowed_by_daily_governor=False,
                blocked_reason="daily_hard_stop",
                xau_grid_override_allowed=False,
                xau_grid_override_reason="hard_stop_manage_only",
                xau_grid_sub_budget=xau_grid_sub_budget,
                xau_grid_risk_multiplier=xau_grid_risk_multiplier,
            )

        if state == "DAILY_DEFENSIVE":
            defensive_quality_floor = float(payload.daily_defensive_quality_floor)
            defensive_general_floor = max(defensive_quality_floor - 0.17, 0.68)
            defensive_native_floor = max(defensive_quality_floor - 0.23, 0.62)
            allowed = bool(
                (band_rank >= 3 and quality_score >= defensive_general_floor)
                or (band_rank >= 2 and proof_allowed and quality_score >= 0.60)
            )
            proof_exception_used = False
            if preserve_strong_session_flow and band_rank >= 3 and quality_score >= defensive_native_floor:
                allowed = True
                state_reason = f"{state_reason}:preserve_strong_session_lane"
            if weak_family and band_rank < 4 and not proof_allowed and not xau_grid_override_allowed:
                allowed = False
                state_reason = f"{state_reason}:range_reversion_defensive_block"
            if (
                non_native_pair
                and band_rank < 4
                and quality_score < 0.72
                and not proof_allowed
                and not xau_grid_override_allowed
            ):
                allowed = False
                state_reason = f"{state_reason}:non_native_defensive_reduce"
            if btc_weekday_caution_block and not xau_grid_override_allowed:
                allowed = False
                state_reason = f"{state_reason}:btc_news_confidence_block"
            if not allowed and xau_grid_override_allowed:
                allowed = True
            if not allowed and proof_allowed and proof_state == "elite-proof" and band_rank >= 2 and quality_score >= 0.58:
                allowed = True
                proof_exception_used = True
            defensive_risk_multiplier = float(payload.daily_defensive_risk_multiplier)
            if preserve_strong_session_flow and not xau_grid_override_allowed:
                defensive_risk_multiplier = max(defensive_risk_multiplier, 0.60)
            return DailyGovernorState(
                state=state,
                reason=state_reason,
                risk_multiplier=float(defensive_risk_multiplier if not xau_grid_override_allowed else payload.xau_grid_defensive_risk_multiplier),
                quality_floor=defensive_quality_floor,
                allowed_by_daily_governor=allowed,
                blocked_reason="" if allowed else "daily_defensive_quality_block",
                proof_exception_used=proof_exception_used,
                xau_grid_override_allowed=bool(xau_grid_override_allowed),
                xau_grid_override_reason=xau_grid_override_reason,
                xau_grid_sub_budget=xau_grid_sub_budget,
                xau_grid_risk_multiplier=float(payload.xau_grid_defensive_risk_multiplier),
            )

        if state == "DAILY_CAUTION":
            caution_quality_floor = float(payload.daily_caution_quality_floor)
            caution_general_floor = max(caution_quality_floor - 0.10, 0.64)
            caution_native_floor = max(caution_quality_floor - 0.14, 0.58)
            allowed = bool(
                (band_rank >= 3 and quality_score >= caution_general_floor)
                or (band_rank >= 2 and quality_score >= 0.60 and proof_allowed)
                or (band_rank >= 2 and quality_score >= 0.66 and not weak_family)
            )
            proof_exception_used = False
            if preserve_strong_session_flow and band_rank >= 2 and quality_score >= caution_native_floor and not weak_family:
                allowed = True
                state_reason = f"{state_reason}:preserve_strong_session_lane"
            if weak_family and band_rank < 3 and not proof_allowed and not xau_grid_override_allowed:
                allowed = False
                state_reason = f"{state_reason}:range_reversion_caution_block"
            if non_native_pair and band_rank < 3 and not proof_allowed and not xau_grid_override_allowed:
                allowed = False
                state_reason = f"{state_reason}:non_native_caution_reduce"
            if btc_weekday_caution_block and not xau_grid_override_allowed:
                allowed = False
                state_reason = f"{state_reason}:btc_news_confidence_block"
            if not allowed and xau_grid_override_allowed:
                allowed = True
            if not allowed and proof_allowed and proof_state == "elite-proof" and band_rank >= 2 and quality_score >= 0.62:
                allowed = True
                proof_exception_used = True
            caution_risk_multiplier = float(payload.daily_caution_risk_multiplier)
            if preserve_strong_session_flow and not xau_grid_override_allowed:
                caution_risk_multiplier = max(caution_risk_multiplier, 0.85)
            return DailyGovernorState(
                state=state,
                reason=state_reason,
                risk_multiplier=float(caution_risk_multiplier if not xau_grid_override_allowed else payload.xau_grid_caution_risk_multiplier),
                quality_floor=caution_quality_floor,
                allowed_by_daily_governor=allowed,
                blocked_reason="" if allowed else "daily_caution_quality_block",
                proof_exception_used=proof_exception_used,
                xau_grid_override_allowed=bool(xau_grid_override_allowed),
                xau_grid_override_reason=xau_grid_override_reason,
                xau_grid_sub_budget=xau_grid_sub_budget,
                xau_grid_risk_multiplier=float(payload.xau_grid_caution_risk_multiplier),
            )

        allowed = bool(
            (band_rank >= 2 and quality_score >= max(float(payload.daily_normal_quality_floor), 0.58))
            or (band_rank == 1 and proof_allowed and quality_score >= 0.48)
        )
        proof_exception_used = False
        if not allowed and proof_allowed and proof_state in {"trusted", "elite-proof"} and quality_score >= 0.48:
            allowed = True
            proof_exception_used = True
        return DailyGovernorState(
            state=state,
            reason=state_reason,
            risk_multiplier=float(payload.daily_normal_risk_multiplier),
            quality_floor=float(payload.daily_normal_quality_floor),
            allowed_by_daily_governor=allowed,
            blocked_reason="" if allowed else "daily_normal_quality_block",
            proof_exception_used=proof_exception_used,
            xau_grid_override_allowed=False,
            xau_grid_override_reason="",
            xau_grid_sub_budget=xau_grid_sub_budget,
            xau_grid_risk_multiplier=0.0,
        )

    @classmethod
    def resolve_daily_governor(cls, payload: RiskInputs, *, spread_elevated: bool = False) -> DailyGovernorState:
        return cls._resolve_daily_governor(payload, spread_elevated=spread_elevated)

    @classmethod
    def _resolve_capacity_state(cls, payload: RiskInputs, *, daily_state: str) -> dict[str, Any]:
        base_daily_target = max(1, int(payload.max_trades_per_day or 1))
        stretch_daily_target = max(base_daily_target, int(payload.stretch_max_trades_per_day or base_daily_target))
        hard_upper_limit = max(stretch_daily_target, int(payload.hard_upper_limit or stretch_daily_target))
        base_hourly_target = max(1, int(payload.max_trades_per_hour or 1))
        stretch_hourly_target = max(base_hourly_target, int(payload.stretch_max_trades_per_hour or base_hourly_target))
        quality_cluster_score = clamp(float(payload.quality_cluster_score or 0.0), 0.0, 1.0)
        lane_strength_multiplier = clamp(float(payload.lane_strength_multiplier or 1.0), 0.80, 1.20)
        lane_score = clamp(float(payload.lane_score or 0.0), 0.0, 1.0)
        lane_expectancy_multiplier = clamp(float(payload.lane_expectancy_multiplier or 1.0), 0.80, 1.25)
        lane_expectancy_score = clamp(float(payload.lane_expectancy_score or lane_score), 0.0, 1.0)
        session_density_score = clamp(float(payload.session_density_score or 0.0), 0.0, 1.0)
        session_priority_multiplier = clamp(float(payload.session_priority_multiplier or 1.0), 0.85, 1.20)
        lane_budget_share = clamp(float(payload.lane_budget_share or 0.0), 0.0, 1.0)
        hot_hand_score = clamp(float(payload.hot_hand_score or 0.0), 0.0, 1.0)
        session_bankroll_bias = clamp(float(payload.session_bankroll_bias or 1.0), 0.85, 1.35)
        profit_recycle_boost = clamp(float(payload.profit_recycle_boost or 0.0), 0.0, 0.25)
        close_winners_score = clamp(float(payload.close_winners_score or 0.5), 0.0, 1.0)
        aggression_lane_multiplier = clamp(float(payload.aggression_lane_multiplier or 1.0), 0.75, 1.60)
        execution_minute_quality_score = clamp(float(payload.execution_minute_quality_score or 0.70), 0.0, 1.0)
        execution_minute_size_multiplier = clamp(float(payload.execution_minute_size_multiplier or 1.0), 0.70, 1.35)
        microstructure_alignment = clamp(float(payload.microstructure_alignment_score or 0.0), -1.0, 1.0)
        microstructure_confidence = clamp(float(payload.microstructure_confidence or 0.0), 0.0, 1.0)
        lead_lag_alignment = clamp(float(payload.lead_lag_alignment_score or 0.0), -1.0, 1.0)
        lead_lag_confidence = clamp(float(payload.lead_lag_confidence or 0.0), 0.0, 1.0)
        lead_lag_disagreement_penalty = clamp(float(payload.lead_lag_disagreement_penalty or 0.0), 0.0, 1.0)
        signal_alignment_score = clamp(
            (microstructure_alignment * 0.46)
            + (lead_lag_alignment * 0.34)
            + ((microstructure_confidence - 0.5) * 0.14)
            + ((lead_lag_confidence - 0.5) * 0.10)
            - (lead_lag_disagreement_penalty * 0.16),
            -1.0,
            1.0,
        )
        event_playbook = str(payload.event_playbook or "").strip().lower()
        playbook_frequency_boost = 0.0
        if event_playbook in {"breakout", "risk_on_follow", "swing_hold"}:
            playbook_frequency_boost = 0.08
        elif event_playbook == "wait_then_retest":
            playbook_frequency_boost = 0.05
        if bool(payload.event_pre_position_allowed):
            playbook_frequency_boost += 0.05
        funded_aggression_bonus = 0.08 if bool(payload.funded_account_mode) else 0.0
        trade_quality_score = clamp(float(payload.trade_quality_score or 0.0), 0.0, 1.0)
        session_priority_multiplier = clamp(
            (session_priority_multiplier * (0.96 + (0.04 * session_bankroll_bias))) + (0.05 * hot_hand_score),
            0.85,
            1.35,
        )
        session_priority_multiplier = clamp(
            session_priority_multiplier
            * (0.96 + (0.08 * execution_minute_size_multiplier))
            * (0.96 + (0.08 * aggression_lane_multiplier))
            + playbook_frequency_boost,
            0.85,
            1.50,
        )
        lane_strength_multiplier = clamp(
            lane_strength_multiplier
            * (0.94 + (0.06 * lane_expectancy_multiplier))
            * aggression_lane_multiplier
            * (0.96 + (0.08 * execution_minute_size_multiplier))
            * (1.0 + funded_aggression_bonus)
            * (0.96 + (0.08 * max(0.0, signal_alignment_score))),
            0.80,
            1.45,
        )
        lane_score = clamp(max(lane_score, lane_expectancy_score * 0.85), 0.0, 1.0)
        lane_budget_share = clamp(
            lane_budget_share + ((lane_expectancy_multiplier - 1.0) * 0.25),
            0.0,
            1.0,
        )
        lane_budget_share = clamp(
            lane_budget_share
            + ((session_bankroll_bias - 1.0) * 0.40)
            + (hot_hand_score * 0.10)
            + (profit_recycle_boost * 0.45)
            + (max(0.0, signal_alignment_score) * 0.18)
            + ((execution_minute_quality_score - 0.5) * 0.12)
            + playbook_frequency_boost,
            0.0,
            1.0,
        )
        lane_expectancy_multiplier = clamp(
            lane_expectancy_multiplier + ((close_winners_score - 0.50) * 0.10),
            0.80,
            1.25,
        )
        hot_lane_borrow_share = 0.0
        hot_lane_bonus = max(0, int(payload.hot_lane_concurrency_bonus or 0))
        if (
            trade_quality_score >= 0.72
            and execution_minute_quality_score >= 0.70
            and session_priority_multiplier >= 1.06
            and (
                hot_hand_score >= 0.60
                or hot_lane_bonus > 0
                or (lane_expectancy_multiplier >= 1.10 and lane_expectancy_score >= 0.60)
            )
        ):
            hot_lane_borrow_share = clamp(
                0.02
                + (hot_hand_score * 0.04)
                + (hot_lane_bonus * 0.02)
                + (max(0.0, lane_expectancy_multiplier - 1.0) * 0.20)
                + (max(0.0, lane_budget_share - 0.18) * 0.30),
                0.0,
                0.24 if hot_hand_score >= 0.70 or lane_expectancy_score >= 0.70 else 0.18,
            )
            lane_budget_share = clamp(lane_budget_share + hot_lane_borrow_share, 0.0, 1.0)
        execution_quality_state = str(payload.execution_quality_state or "GOOD").upper()
        execution_good = execution_quality_state == "GOOD" and not bool(payload.degraded_mode_active)
        correlation_ok = float(payload.correlation_multiplier or 1.0) >= 0.80
        open_risk_ok = float(payload.total_open_risk_pct or 0.0) <= max(0.03, float(payload.hard_risk_cap or 0.02) * 1.5)
        cluster_mode = bool(payload.cluster_mode_active) or (
            quality_cluster_score >= 0.65
            and trade_quality_score >= 0.70
            and execution_good
            and correlation_ok
            and open_risk_ok
        )
        winning_streak_mode = bool(payload.winning_streak_mode_active) or int(payload.stats.winning_streak or 0) >= 2
        stretch_mode_active = False
        current_capacity_mode = "BASE"
        allowed_daily_target = base_daily_target
        allowed_hourly_target = base_hourly_target

        normalized_daily_state = str(daily_state or "DAILY_NORMAL").upper()
        if normalized_daily_state == "DAILY_HARD_STOP":
            return {
                "base_daily_target": base_daily_target,
                "stretch_daily_target": stretch_daily_target,
                "hard_upper_limit": hard_upper_limit,
                "base_hourly_target": base_hourly_target,
                "stretch_hourly_target": stretch_hourly_target,
                "allowed_daily_target": 0,
                "allowed_hourly_target": 1,
                "stretch_mode_active": False,
                "cluster_mode_active": cluster_mode,
                "current_capacity_mode": "HARD_STOP",
                "quality_cluster_score": quality_cluster_score,
                "lane_strength_multiplier": lane_strength_multiplier,
                "lane_score": lane_score,
                "lane_expectancy_multiplier": float(lane_expectancy_multiplier),
                "lane_expectancy_score": float(lane_expectancy_score),
                "session_density_score": session_density_score,
                "session_priority_multiplier": float(session_priority_multiplier),
                "lane_budget_share": float(lane_budget_share),
                "lane_available_capacity": 0,
                "hot_lane_borrow_share": 0.0,
                "signal_alignment_score": float(signal_alignment_score),
                "playbook_frequency_boost": float(playbook_frequency_boost),
                "aggression_lane_multiplier": float(aggression_lane_multiplier),
                "execution_minute_quality_score": float(execution_minute_quality_score),
                "execution_minute_size_multiplier": float(execution_minute_size_multiplier),
                "soft_trade_budget_enabled": bool(payload.soft_trade_budget_enabled),
            }

        if normalized_daily_state == "DAILY_DEFENSIVE":
            allowed_daily_target = max(6, int(round(base_daily_target * 0.75)))
            allowed_hourly_target = max(2, int(round(base_hourly_target * 0.75)))
            current_capacity_mode = "DEFENSIVE"
            if (
                trade_quality_score >= 0.72
                and execution_good
                and lane_strength_multiplier >= 0.98
            ):
                allowed_daily_target = max(allowed_daily_target, max(8, int(round(base_daily_target * 0.85))))
                allowed_hourly_target = max(allowed_hourly_target, max(2, int(round(base_hourly_target * 0.85))))
                current_capacity_mode = "DEFENSIVE_FLOW"
        elif normalized_daily_state == "DAILY_CAUTION":
            allowed_daily_target = max(8, int(round(base_daily_target * 0.90)))
            allowed_hourly_target = max(2, int(round(base_hourly_target * 0.90)))
            current_capacity_mode = "CAUTION"
            if (
                cluster_mode
                and execution_good
                and trade_quality_score >= 0.72
                and lane_strength_multiplier >= 1.00
            ):
                allowed_daily_target = min(stretch_daily_target, allowed_daily_target + 3)
                allowed_hourly_target = min(stretch_hourly_target, allowed_hourly_target + 1)
                stretch_mode_active = True
                current_capacity_mode = "CAUTION_STRETCH"
        else:
            current_capacity_mode = "BASE"
            if (
                cluster_mode
                or (
                    execution_good
                    and trade_quality_score >= 0.72
                    and quality_cluster_score >= 0.55
                    and lane_strength_multiplier >= 1.00
                )
            ):
                allowed_daily_target = stretch_daily_target
                allowed_hourly_target = stretch_hourly_target
                stretch_mode_active = True
                current_capacity_mode = "STRETCH"
            elif (
                execution_good
                and trade_quality_score >= 0.70
                and lane_strength_multiplier >= 1.02
            ):
                allowed_daily_target = min(stretch_daily_target, base_daily_target + 2)
                allowed_hourly_target = min(stretch_hourly_target, base_hourly_target + 1)
                current_capacity_mode = "ENHANCED"
            if winning_streak_mode and execution_good and correlation_ok:
                allowed_daily_target = min(hard_upper_limit, allowed_daily_target + 1)
                allowed_hourly_target = min(stretch_hourly_target, allowed_hourly_target + 1)
                if current_capacity_mode == "BASE":
                    current_capacity_mode = "WIN_STREAK"

        expectancy_bonus = 0
        if lane_expectancy_multiplier >= 1.08 and lane_expectancy_score >= 0.58:
            expectancy_bonus = 1
        elif lane_expectancy_multiplier <= 0.94 and lane_expectancy_score <= 0.42:
            expectancy_bonus = -1
        alignment_bonus = 0
        if signal_alignment_score >= 0.25 and execution_minute_quality_score >= 0.70:
            alignment_bonus = 1
        elif signal_alignment_score <= -0.20:
            alignment_bonus = -1

        priority_bonus = 0
        if session_priority_multiplier >= 1.10 and lane_budget_share >= 0.20:
            priority_bonus += 1
        elif session_priority_multiplier <= 0.94 and lane_budget_share <= 0.10:
            priority_bonus -= 1
        hot_hand_bonus = 0
        if bool(payload.hot_hand_active) and hot_hand_score >= 0.55 and execution_good and correlation_ok:
            hot_hand_bonus = 2 if session_priority_multiplier >= 1.10 else 1
        profit_recycle_bonus = 0
        if bool(payload.profit_recycle_active) and profit_recycle_boost >= 0.05 and execution_good:
            profit_recycle_bonus = 1
        if current_capacity_mode not in {"HARD_STOP"}:
            allowed_daily_target = max(
                0,
                min(
                    hard_upper_limit,
                    allowed_daily_target + priority_bonus + expectancy_bonus + hot_hand_bonus + profit_recycle_bonus + alignment_bonus,
                ),
            )
            allowed_hourly_target = max(
                1,
                min(
                    stretch_hourly_target + max(0, int(payload.hot_lane_concurrency_bonus or 0)),
                    allowed_hourly_target
                    + (
                        1
                        if (
                            priority_bonus > 0
                            or expectancy_bonus > 0
                            or hot_hand_bonus > 0
                            or profit_recycle_bonus > 0
                            or alignment_bonus > 0
                        )
                        and execution_good
                        else 0
                    ),
                ),
            )
            if hot_lane_borrow_share > 0.0 and execution_good:
                borrowed_daily_bonus = max(1, int(round(max(base_daily_target, allowed_daily_target) * hot_lane_borrow_share * 0.45)))
                borrowed_hourly_bonus = max(1, int(round(max(base_hourly_target, allowed_hourly_target) * hot_lane_borrow_share * 0.80)))
                allowed_daily_target = min(hard_upper_limit, allowed_daily_target + borrowed_daily_bonus)
                allowed_hourly_target = min(
                    stretch_hourly_target + max(0, int(payload.hot_lane_concurrency_bonus or 0)),
                    allowed_hourly_target + borrowed_hourly_bonus,
                )
                if current_capacity_mode in {"BASE", "ENHANCED", "STRETCH", "SOFT_BUDGET", "SOFT_BUDGET_AGGRO"}:
                    current_capacity_mode = "HOT_LANE_BORROW"
        if bool(payload.soft_trade_budget_enabled) and current_capacity_mode not in {"HARD_STOP"}:
            soft_multiplier = clamp(
                aggression_lane_multiplier
                * execution_minute_size_multiplier
                * (1.0 + (max(0.0, signal_alignment_score) * 0.24) + playbook_frequency_boost + funded_aggression_bonus),
                0.75,
                1.65,
            )
            soft_daily_limit = max(hard_upper_limit, int(round(max(base_daily_target, stretch_daily_target) * max(1.0, soft_multiplier))))
            soft_hourly_limit = max(
                stretch_hourly_target + max(0, int(payload.hot_lane_concurrency_bonus or 0)),
                int(round(max(base_hourly_target, stretch_hourly_target) * max(1.0, soft_multiplier))),
            )
            allowed_daily_target = min(soft_daily_limit, max(allowed_daily_target, int(round(allowed_daily_target * soft_multiplier))))
            allowed_hourly_target = min(soft_hourly_limit, max(allowed_hourly_target, int(round(allowed_hourly_target * soft_multiplier))))
            hard_upper_limit = max(hard_upper_limit, soft_daily_limit)
            if soft_multiplier > 1.08 and current_capacity_mode == "BASE":
                current_capacity_mode = "SOFT_BUDGET"
            elif soft_multiplier > 1.18 and current_capacity_mode in {"BASE", "STRETCH", "ENHANCED"}:
                current_capacity_mode = "SOFT_BUDGET_AGGRO"

        lane_available_capacity = 0
        if allowed_daily_target > 0:
            lane_share = max(0.15, lane_budget_share if lane_budget_share > 0.0 else 0.15)
            lane_available_capacity = max(1, int(round(allowed_daily_target * lane_share)))
            if session_priority_multiplier >= 1.10:
                lane_available_capacity = min(allowed_daily_target, lane_available_capacity + 1)
            if hot_hand_bonus > 0 or profit_recycle_bonus > 0:
                lane_available_capacity = min(allowed_daily_target, lane_available_capacity + 1)
            lane_available_capacity = min(allowed_daily_target, lane_available_capacity)

        return {
            "base_daily_target": base_daily_target,
            "stretch_daily_target": stretch_daily_target,
            "hard_upper_limit": hard_upper_limit,
            "base_hourly_target": base_hourly_target,
            "stretch_hourly_target": stretch_hourly_target,
            "allowed_daily_target": max(0, int(allowed_daily_target)),
            "allowed_hourly_target": max(1, int(allowed_hourly_target)),
            "stretch_mode_active": bool(stretch_mode_active),
            "cluster_mode_active": bool(cluster_mode),
            "current_capacity_mode": str(current_capacity_mode),
            "quality_cluster_score": float(quality_cluster_score),
            "lane_strength_multiplier": float(lane_strength_multiplier),
            "lane_score": float(lane_score),
            "lane_expectancy_multiplier": float(lane_expectancy_multiplier),
            "lane_expectancy_score": float(lane_expectancy_score),
            "session_density_score": float(session_density_score),
            "session_priority_multiplier": float(session_priority_multiplier),
            "lane_budget_share": float(lane_budget_share),
            "lane_available_capacity": int(lane_available_capacity),
            "hot_lane_borrow_share": float(hot_lane_borrow_share),
            "signal_alignment_score": float(signal_alignment_score),
            "playbook_frequency_boost": float(playbook_frequency_boost),
            "aggression_lane_multiplier": float(aggression_lane_multiplier),
            "execution_minute_quality_score": float(execution_minute_quality_score),
            "execution_minute_size_multiplier": float(execution_minute_size_multiplier),
            "soft_trade_budget_enabled": bool(payload.soft_trade_budget_enabled),
            "hot_hand_active": bool(payload.hot_hand_active),
            "hot_hand_score": float(hot_hand_score),
            "session_bankroll_bias": float(session_bankroll_bias),
            "profit_recycle_active": bool(payload.profit_recycle_active),
            "profit_recycle_boost": float(profit_recycle_boost),
            "close_winners_score": float(close_winners_score),
        }

    def evaluate(self, payload: RiskInputs) -> RiskDecision:
        symbol_key = payload.symbol.upper()
        paused_until = self.volatility_pauses.get(symbol_key)
        if paused_until and paused_until > payload.current_time:
            return RiskDecision(False, None, 0.0, 0.0, f"volatility_pause_until_{paused_until.isoformat()}")

        hot_lane_bonus = 0
        if (
            int(payload.hot_lane_concurrency_bonus or 0) > 0
            and (
                bool(payload.hot_hand_active)
                or float(payload.lane_strength_multiplier or 1.0) >= 1.05
                or float(payload.microstructure_alignment_score or 0.0) >= 0.25
            )
        ):
            hot_lane_bonus = max(0, int(payload.hot_lane_concurrency_bonus or 0))
        effective_max_positions_per_symbol = max(int(payload.max_positions_per_symbol), int(payload.max_positions_per_symbol) + hot_lane_bonus)
        if payload.mode.upper() == "LIVE" and (not payload.live_enabled):
            return RiskDecision(False, None, 0.0, 0.0, "live_blocked_requires_live_enable_flag")
        if payload.open_positions >= payload.max_positions:
            return RiskDecision(False, None, 0.0, 0.0, "max_positions_total_reached")
        if payload.open_positions_symbol >= effective_max_positions_per_symbol:
            return RiskDecision(False, None, 0.0, 0.0, "max_positions_per_symbol_reached")
        if payload.same_direction_positions >= effective_max_positions_per_symbol:
            return RiskDecision(False, None, 0.0, 0.0, "same_direction_limit")
        current_daily_state, current_daily_state_reason = self.resolve_daily_state_from_stats(
            payload.stats,
            caution_threshold_pct=float(payload.daily_caution_threshold_pct),
            defensive_threshold_pct=float(payload.daily_defensive_threshold_pct),
            hard_stop_threshold_pct=float(payload.daily_hard_stop_threshold_pct),
        )
        current_daily_state, current_daily_state_reason, daily_release_flags = self._apply_daily_governor_release(
            payload,
            state=current_daily_state,
            state_reason=current_daily_state_reason,
        )
        capacity_state = self._resolve_capacity_state(payload, daily_state=current_daily_state)
        effective_max_trades_per_hour = int(capacity_state.get("allowed_hourly_target", payload.max_trades_per_hour))
        daily_trade_overflow_active = False
        base_daily_target = max(1, int(payload.max_trades_per_day or 1))
        allowed_daily_target = max(0, int(capacity_state.get("allowed_daily_target", base_daily_target)))
        if current_daily_state != "DAILY_HARD_STOP":
            if payload.stats.trades_today >= allowed_daily_target:
                if not self._overflow_trade_cap_allowed(payload, capacity_state=capacity_state):
                    return RiskDecision(
                        False,
                        None,
                        0.0,
                        0.0,
                        "max_trades_per_day_reached",
                        diagnostics={
                            "base_daily_trade_target": int(capacity_state.get("base_daily_target", payload.max_trades_per_day)),
                            "stretch_daily_trade_target": int(capacity_state.get("stretch_daily_target", payload.stretch_max_trades_per_day or payload.max_trades_per_day)),
                            "hard_upper_limit": int(capacity_state.get("hard_upper_limit", payload.hard_upper_limit or payload.max_trades_per_day)),
                            "projected_trade_capacity_today": int(capacity_state.get("allowed_daily_target", payload.max_trades_per_day)),
                            "stretch_mode_active": bool(capacity_state.get("stretch_mode_active", False)),
                            "cluster_mode_active": bool(capacity_state.get("cluster_mode_active", False)),
                            "current_capacity_mode": str(capacity_state.get("current_capacity_mode", payload.current_capacity_mode or "BASE")),
                            "quality_cluster_score": float(capacity_state.get("quality_cluster_score", payload.quality_cluster_score or 0.0)),
                            "lane_strength_multiplier": float(capacity_state.get("lane_strength_multiplier", payload.lane_strength_multiplier or 1.0)),
                            "lane_score": float(capacity_state.get("lane_score", payload.lane_score or 0.0)),
                            "session_density_score": float(capacity_state.get("session_density_score", payload.session_density_score or 0.0)),
                        },
                    )
                daily_trade_overflow_active = True
            elif payload.stats.trades_today >= base_daily_target:
                daily_trade_overflow_active = True
            effective_max_trades_per_hour = self._effective_max_trades_per_hour(
                payload,
                daily_state=current_daily_state,
                capacity_state=capacity_state,
            )
            if payload.recent_trades_last_hour >= effective_max_trades_per_hour:
                return RiskDecision(
                    False,
                    None,
                    0.0,
                    0.0,
                    "max_trades_per_hour_reached",
                    diagnostics={
                        "hourly_base_target": int(capacity_state.get("base_hourly_target", payload.max_trades_per_hour)),
                        "hourly_stretch_target": int(capacity_state.get("stretch_hourly_target", payload.stretch_max_trades_per_hour or payload.max_trades_per_hour)),
                        "allowed_hourly_target": int(capacity_state.get("allowed_hourly_target", payload.max_trades_per_hour)),
                        "effective_hourly_trade_cap": int(effective_max_trades_per_hour),
                        "stretch_mode_active": bool(capacity_state.get("stretch_mode_active", False)),
                        "cluster_mode_active": bool(capacity_state.get("cluster_mode_active", False)),
                        "current_capacity_mode": str(capacity_state.get("current_capacity_mode", payload.current_capacity_mode or "BASE")),
                    },
                )

        micro_active = payload.micro_enabled and payload.mode.upper() in {"DEMO", "PAPER", "LIVE"}
        bootstrap_active = (
            micro_active
            and bool(payload.bootstrap_enabled)
            and float(payload.equity) <= max(0.0, float(payload.bootstrap_equity_threshold))
        )
        soft_daily_dd_limit = max(0.0, float(payload.soft_daily_dd_pct))
        hard_daily_dd_limit = max(soft_daily_dd_limit, float(payload.hard_daily_dd_pct))
        effective_drawdown_kill = float(payload.max_drawdown_kill)
        if bootstrap_active and payload.bootstrap_drawdown_kill is not None:
            effective_drawdown_kill = max(effective_drawdown_kill, float(payload.bootstrap_drawdown_kill))
        if (
            float(payload.stats.daily_dd_pct_live) >= hard_daily_dd_limit
            and not bool(payload.recovery_mode_active)
            and not bool(daily_release_flags.get("force_release"))
        ):
            return RiskDecision(False, "HARD", 0.0, 0.0, "hard_daily_dd")
        if payload.stats.rolling_drawdown_pct >= effective_drawdown_kill:
            return RiskDecision(False, "SOFT", 0.0, 0.0, "rolling_drawdown_kill")
        effective_absolute_drawdown_hard_stop = float(payload.absolute_drawdown_hard_stop)
        if bootstrap_active and payload.bootstrap_drawdown_kill is not None:
            effective_absolute_drawdown_hard_stop = max(
                effective_absolute_drawdown_hard_stop,
                float(payload.bootstrap_drawdown_kill),
            )
        if payload.stats.absolute_drawdown_pct >= effective_absolute_drawdown_hard_stop and not bool(payload.recovery_mode_active):
            return RiskDecision(False, "HARD", 0.0, 0.0, "absolute_drawdown_hard_stop")
        if bool(payload.funded_account_mode):
            funded_daily_limit = max(0.0, float(payload.funded_daily_loss_limit_pct or 0.0))
            funded_overall_limit = max(0.0, float(payload.funded_overall_drawdown_limit_pct or 0.0))
            funded_target = max(0.0, float(payload.funded_profit_target_pct or 0.0))
            funded_remaining = float(payload.funded_remaining_target_pct or 0.0)
            if funded_target > 0.0 and funded_remaining <= 0.0 and is_funded_challenge_phase(payload.funded_phase):
                return RiskDecision(False, "SOFT", 0.0, 0.0, "funded_target_reached_protect_pass")
            funded_daily_buffer = max(0.0, funded_daily_limit - max(0.0, float(payload.stats.daily_dd_pct_live or 0.0)))
            funded_overall_buffer = max(0.0, funded_overall_limit - max(0.0, float(payload.stats.absolute_drawdown_pct or 0.0)))
            funded_daily_quality = clamp(
                funded_daily_buffer / max(funded_daily_limit, 1e-9),
                0.0,
                1.0,
            ) if funded_daily_limit > 0.0 else 1.0
            funded_overall_quality = clamp(
                funded_overall_buffer / max(funded_overall_limit, 1e-9),
                0.0,
                1.0,
            ) if funded_overall_limit > 0.0 else 1.0
            funded_buffer_quality = clamp(min(funded_daily_quality, funded_overall_quality), 0.0, 1.0)
            if funded_buffer_quality <= max(0.02, float(payload.funded_guard_buffer_pct or 0.0) * 0.5):
                return RiskDecision(False, "SOFT", 0.0, 0.0, "funded_buffer_exhausted")
        if not payload.weekend_trading_allowed and is_weekend_market_mode(payload.current_time):
            return RiskDecision(False, None, 0.0, 0.0, "weekend_disabled")
        low_equity_safety_active = bool(payload.low_equity_safety_enabled) and float(payload.equity or 0.0) <= float(payload.low_equity_max_equity or 0.0)
        spread_reference = max(0.0, float(payload.spread_atr_reference_points or 0.0))
        candidate_mc = clamp(float(payload.candidate_monte_carlo_win_rate or 0.0), 0.0, 1.0)
        low_equity_xau_override = False
        low_equity_attack_override = False
        if low_equity_safety_active:
            low_equity_xau_override = self._allow_low_equity_xau_grid_bootstrap(
                payload,
                candidate_mc=candidate_mc,
                spread_reference=spread_reference,
            )
            low_equity_attack_override = self._allow_low_equity_attack_bootstrap(
                payload,
                candidate_mc=candidate_mc,
                spread_reference=spread_reference,
            )
        if payload.spread_points > payload.max_spread_points and not self._allow_low_equity_high_spread_override(
            payload,
            low_equity_xau_override=low_equity_xau_override,
            low_equity_attack_override=low_equity_attack_override,
            spread_reference=spread_reference,
        ):
            return RiskDecision(False, None, 0.0, 0.0, "spread_too_wide")
        spread_elevated = payload.spread_points > (payload.max_spread_points * 0.75)
        if low_equity_safety_active:
            if spread_reference > 0.0 and float(payload.spread_points or 0.0) > (
                spread_reference * max(0.5, float(payload.low_equity_spread_atr_cap or 1.2))
            ) and not low_equity_xau_override and not low_equity_attack_override:
                return RiskDecision(False, "SOFT", 0.0, 0.0, "low_equity_spread_atr_guard")
            if (
                candidate_mc > 0.0
                and candidate_mc < clamp(float(payload.low_equity_monte_carlo_floor or 0.88), 0.5, 0.99)
                and not low_equity_xau_override
                and not low_equity_attack_override
            ):
                return RiskDecision(False, "SOFT", 0.0, 0.0, "low_equity_mc_floor")
        session_governor = self._resolve_session_governor(payload)
        if not session_governor.allowed:
            return RiskDecision(
                False,
                "SOFT",
                0.0,
                0.0,
                str(session_governor.blocked_reason or session_governor.reason),
                diagnostics={
                    "session_name": str(session_governor.session_name),
                    "session_stop_state": str(session_governor.state),
                    "session_stop_reason": str(session_governor.reason),
                    "session_entries_blocked": True,
                    "session_realized_pnl": float(session_governor.session_realized_pnl),
                    "session_realized_pnl_pct": float(session_governor.session_realized_pnl_pct),
                    "session_trade_count": int(session_governor.session_trade_count),
                    "daily_state": str(current_daily_state),
                    "daily_state_reason": str(current_daily_state_reason),
                    "trading_day_key": self._stats_text(payload, "trading_day_key"),
                    "timezone_used": self._stats_text(payload, "timezone_used", "Australia/Sydney"),
                },
            )
        daily_governor = self._resolve_daily_governor(payload, spread_elevated=spread_elevated)
        if not daily_governor.allowed_by_daily_governor:
            kill = "HARD" if daily_governor.state == "DAILY_HARD_STOP" else ("SOFT" if daily_governor.state in {"DAILY_CAUTION", "DAILY_DEFENSIVE"} else None)
            blocked_reason = str(daily_governor.blocked_reason or daily_governor.state.lower())
            return RiskDecision(
                False,
                kill,
                0.0,
                0.0,
                blocked_reason,
                diagnostics={
                    "daily_state": str(daily_governor.state),
                    "daily_state_reason": str(daily_governor.reason),
                    "trading_day_key": self._stats_text(payload, "trading_day_key"),
                    "timezone_used": self._stats_text(payload, "timezone_used", "Australia/Sydney"),
                    "day_start_equity": float(payload.stats.day_start_equity),
                    "current_equity": float(payload.equity),
                    "daily_realized_pnl": float(payload.stats.daily_realized_pnl),
                    "daily_pnl_pct": float(payload.stats.daily_pnl_pct),
                    "daily_dd_pct_live": float(payload.stats.daily_dd_pct_live),
                    "closed_trades_today": int(payload.stats.trades_today),
                    "allowed_by_daily_governor": False,
                    "risk_multiplier_applied": float(daily_governor.risk_multiplier),
                    "session_name": str(session_governor.session_name),
                    "session_stop_state": str(session_governor.state),
                    "session_stop_reason": str(session_governor.reason),
                    "session_entries_blocked": bool(not session_governor.allowed),
                    "session_realized_pnl": float(session_governor.session_realized_pnl),
                    "session_realized_pnl_pct": float(session_governor.session_realized_pnl_pct),
                    "session_trade_count": int(session_governor.session_trade_count),
                    "proof_exception_used": bool(daily_governor.proof_exception_used),
                    "xau_grid_override_allowed": bool(daily_governor.xau_grid_override_allowed),
                    "xau_grid_override_reason": str(daily_governor.xau_grid_override_reason),
                    "xau_grid_sub_budget": float(daily_governor.xau_grid_sub_budget),
                    "xau_grid_risk_multiplier": float(daily_governor.xau_grid_risk_multiplier),
                    "soft_dd_active": bool(daily_governor.state in {"DAILY_CAUTION", "DAILY_DEFENSIVE"}),
                    "hard_dd_active": bool(daily_governor.state == "DAILY_HARD_STOP"),
                    "soft_dd_elite_mode_active": False,
                    "soft_dd_trade_count": int(payload.stats.soft_dd_trade_count),
                    "last_soft_dd_rejection_reason": str(daily_governor.blocked_reason or daily_governor.reason),
                },
            )
        if payload.atr_average > 0 and payload.atr_current > (payload.atr_average * payload.atr_spike_multiple):
            self.volatility_pauses[symbol_key] = payload.current_time + timedelta(minutes=payload.volatility_pause_minutes)
            return RiskDecision(False, None, 0.0, 0.0, "atr_spike_pause")
        current_ny = payload.current_time.astimezone(NEW_YORK)
        if (
            (not payload.weekend_trading_allowed)
            and current_ny.weekday() == 4
            and current_ny.hour >= payload.friday_cutoff_hour
        ):
            return RiskDecision(False, "SOFT", 0.0, 0.0, "friday_flat_window")
        if payload.stop_price <= 0 or payload.tp_price <= 0 or payload.stop_price == payload.entry_price:
            return RiskDecision(False, None, 0.0, 0.0, "invalid_exit_levels")

        stop_distance = abs(payload.entry_price - payload.stop_price)
        if stop_distance <= 0:
            return RiskDecision(False, None, 0.0, 0.0, "per_lot_risk_zero")
        min_stop_distance = max(0.0, payload.min_stop_distance_points) * max(payload.symbol_point, 1e-9)
        if min_stop_distance > 0 and stop_distance < min_stop_distance:
            return RiskDecision(False, None, 0.0, 0.0, "stop_too_tight")
        if micro_active and payload.stats.trades_today < payload.first_trade_protection_trades:
            allowed_first_trade_sl_atr = max(payload.first_trade_max_sl_atr, payload.candidate_stop_atr * 1.05)
            if bootstrap_active:
                allowed_first_trade_sl_atr = max(
                    allowed_first_trade_sl_atr,
                    float(payload.bootstrap_first_trade_max_sl_atr),
                )
            if payload.symbol.upper().startswith(("XAUUSD", "GOLD")) and payload.setup.upper().startswith("XAUUSD_M5_GRID_SCALPER"):
                allowed_first_trade_sl_atr = max(allowed_first_trade_sl_atr, payload.candidate_stop_atr, 2.5)
            if payload.atr_current > 0:
                point_size, point_source = self._resolve_point_size(payload)
                sl_distance_points = stop_distance / point_size
                max_allowed_points = (payload.atr_current * allowed_first_trade_sl_atr) / point_size
                if bootstrap_active and float(payload.volume_min) >= 1.0:
                    max_allowed_points *= 1.10
                if sl_distance_points > max_allowed_points:
                    return RiskDecision(
                        approved=False,
                        kill=None,
                        volume=0.0,
                        risk_pct=0.0,
                        reason="first_trade_sl_too_wide",
                        diagnostics={
                            "symbol": payload.symbol.upper(),
                            "sl_distance_price": stop_distance,
                            "sl_distance_points": sl_distance_points,
                            "max_allowed_points": max_allowed_points,
                            "atr_current": payload.atr_current,
                            "first_trade_max_sl_atr": allowed_first_trade_sl_atr,
                            "bootstrap_first_trade_max_sl_atr": payload.bootstrap_first_trade_max_sl_atr,
                            "point_size": point_size,
                            "point_source": point_source,
                            "digits": payload.symbol_digits,
                            "tick_size": payload.symbol_tick_size,
                        },
                    )

        hard_cap = max(payload.hard_risk_cap, float(payload.strategy_risk_cap)) if payload.strategy_risk_cap is not None else payload.hard_risk_cap
        hard_cap = max(hard_cap, float(payload.current_base_risk_pct or 0.0))
        if bootstrap_active and payload.equity > 0:
            bootstrap_hard_cap_pct = max(0.0, float(payload.bootstrap_per_trade_hard_cap)) / max(float(payload.equity), 1e-9)
            hard_cap = max(hard_cap, bootstrap_hard_cap_pct)
        base_requested_risk_pct = min(payload.requested_risk_pct, hard_cap)
        risk_pct = max(0.0, base_requested_risk_pct)
        boost_active = False
        if payload.no_trade_boost_enabled and payload.no_trade_boost_eligible:
            if payload.no_trade_boost_elapsed_minutes >= max(0, int(payload.no_trade_boost_after_minutes)):
                interval_minutes = max(1, int(payload.no_trade_boost_interval_minutes))
                intervals = int((payload.no_trade_boost_elapsed_minutes - payload.no_trade_boost_after_minutes) // interval_minutes) + 1
                boosted_pct = base_requested_risk_pct + (max(0, intervals) * max(0.0, float(payload.no_trade_boost_step_pct)))
                risk_pct = min(boosted_pct, hard_cap, max(0.0, float(payload.no_trade_boost_max_pct)))
                boost_active = risk_pct > base_requested_risk_pct

        funded_buffer_quality = 1.0
        if bool(payload.funded_account_mode):
            funded_daily_limit = max(0.0, float(payload.funded_daily_loss_limit_pct or 0.0))
            funded_overall_limit = max(0.0, float(payload.funded_overall_drawdown_limit_pct or 0.0))
            funded_daily_buffer = max(0.0, funded_daily_limit - max(0.0, float(payload.stats.daily_dd_pct_live or 0.0)))
            funded_overall_buffer = max(0.0, funded_overall_limit - max(0.0, float(payload.stats.absolute_drawdown_pct or 0.0)))
            funded_daily_quality = clamp(
                funded_daily_buffer / max(funded_daily_limit, 1e-9),
                0.0,
                1.0,
            ) if funded_daily_limit > 0.0 else 1.0
            funded_overall_quality = clamp(
                funded_overall_buffer / max(funded_overall_limit, 1e-9),
                0.0,
                1.0,
            ) if funded_overall_limit > 0.0 else 1.0
            funded_buffer_quality = clamp(min(funded_daily_quality, funded_overall_quality), 0.0, 1.0)
            if funded_buffer_quality <= max(0.02, float(payload.funded_guard_buffer_pct or 0.0) * 0.5):
                return RiskDecision(False, "SOFT", 0.0, 0.0, "funded_buffer_exhausted")
            if payload.stats.daily_pnl_pct > 0.0 and float(payload.funded_profit_target_pct or 0.0) > 0.0:
                funded_progress = 1.0 - clamp(
                    float(payload.funded_remaining_target_pct or 0.0) / max(float(payload.funded_profit_target_pct or 0.0), 1e-9),
                    0.0,
                    1.0,
                )
                risk_pct = min(hard_cap, risk_pct * (1.0 + min(0.12, funded_progress * 0.12)))

        low_equity_risk_clamp_active = bool(
            low_equity_safety_active
            and not bool(payload.bootstrap_enabled)
            and not self._is_xau_grid_setup(payload)
        )
        if low_equity_risk_clamp_active:
            low_equity_ceiling = clamp(
                float(payload.low_equity_risk_ceiling_pct or 0.02),
                0.0,
                max(0.0, hard_cap),
            )
            low_equity_floor = clamp(
                float(payload.low_equity_risk_floor_pct or 0.01),
                0.0,
                max(0.0, low_equity_ceiling),
            )
            risk_pct = clamp(risk_pct, low_equity_floor, max(low_equity_floor, low_equity_ceiling))

        risk_pct = max(0.0, risk_pct)
        if micro_active and (not payload.skip_micro_risk_clamp):
            capped_requested = min(risk_pct, payload.micro_risk_pct_ceiling)
            if bootstrap_active:
                bootstrap_floor_pct = (
                    max(0.0, float(payload.bootstrap_min_risk_amount)) / max(float(payload.equity), 1e-9)
                )
                bootstrap_requested = risk_pct if boost_active else capped_requested
                risk_pct = clamp(max(bootstrap_requested, bootstrap_floor_pct), payload.micro_risk_pct_floor, hard_cap)
            else:
                progress = 0.0 if payload.micro_min_trades <= 0 else clamp(payload.stats.closed_trades_total / payload.micro_min_trades, 0.0, 1.0)
                ramped = payload.micro_risk_pct_floor + ((capped_requested - payload.micro_risk_pct_floor) * progress)
                risk_pct = clamp(ramped, payload.micro_risk_pct_floor, capped_requested)

        if payload.use_kelly:
            b = max(0.1, payload.stats.avg_win_r / max(payload.stats.avg_loss_r, 0.1))
            p = clamp(payload.stats.win_rate, 0.01, 0.99)
            q = 1 - p
            kelly_fraction = max(0.0, ((p * b) - q) / b)
            kelly_cap = max(0.0, kelly_fraction * payload.kelly_fraction)
            if kelly_cap > 0:
                risk_pct = min(risk_pct, kelly_cap)

        size_factor = payload.session_multiplier * payload.ai_size_multiplier * payload.portfolio_size_multiplier
        size_factor *= clamp(float(payload.quality_size_multiplier or 1.0), 0.20, 2.00)
        if spread_elevated:
            size_factor *= 0.75
        if payload.stats.rolling_drawdown_pct > 0.02:
            size_factor *= 0.5
        if payload.regime == "VOLATILE":
            size_factor *= 0.7
        if payload.stats.consecutive_losses >= 7 and payload.stats.cooldown_trades_remaining > 0:
            return RiskDecision(False, None, 0.0, 0.0, "loss_streak_cooldown")
        if payload.stats.consecutive_losses >= 5 or (
            payload.stats.consecutive_losses >= 3 and payload.stats.cooldown_trades_remaining > 0
        ):
            size_factor *= 0.65
        elif payload.stats.consecutive_losses >= 3 or payload.stats.cooldown_trades_remaining > 0:
            size_factor *= 0.80
        elif payload.stats.consecutive_losses >= 2:
            size_factor *= 0.90
        if daily_governor.state == "DAILY_CAUTION":
            size_factor *= 0.98
        elif daily_governor.state == "DAILY_DEFENSIVE":
            size_factor *= 0.90
        if payload.stats.daily_pnl_pct <= -payload.max_daily_loss:
            size_factor *= 0.80
        if payload.stats.trades_today < payload.first_trade_protection_trades:
            if bootstrap_active:
                size_factor *= max(0.85, payload.first_trade_size_factor)
            else:
                size_factor *= payload.first_trade_size_factor
        if payload.anti_martingale_enabled and (not micro_active) and payload.stats.winning_streak >= 2 and payload.stats.rolling_drawdown_pct < 0.02:
            anti = min(payload.anti_martingale_cap, 1.0 + (payload.anti_martingale_step * payload.stats.winning_streak))
            size_factor *= anti
        if bool(payload.funded_account_mode):
            size_factor *= clamp(0.75 + (funded_buffer_quality * 0.35), 0.55, 1.10)
        if bootstrap_active and boost_active:
            size_factor = max(size_factor, 0.85)

        adaptive_multiplier, risk_modifiers, overflow_band_active, allowed_risk_cap = self._adaptive_risk_profile(
            payload,
            spread_elevated=spread_elevated,
            daily_state=str(daily_governor.state),
            hard_cap=hard_cap,
            daily_trade_overflow_active=daily_trade_overflow_active,
            capacity_state=capacity_state,
        )
        effective_risk_pct = clamp(
            risk_pct * size_factor * adaptive_multiplier * max(0.0, float(daily_governor.risk_multiplier)) * max(0.0, float(session_governor.risk_multiplier)),
            0.0,
            max(0.0, allowed_risk_cap),
        )

        per_lot_risk, risk_formula = self._resolve_per_lot_risk(payload, stop_distance)
        if per_lot_risk <= 0:
            return RiskDecision(False, None, 0.0, 0.0, "per_lot_risk_zero")
        effective_budget = max(0.0, payload.equity * effective_risk_pct)
        if bootstrap_active:
            per_trade_cap = max(
                effective_budget,
                max(0.0, float(payload.bootstrap_min_risk_amount)),
                max(0.0, float(payload.bootstrap_per_trade_hard_cap)),
            )
            total_exposure_cap = max(
                per_trade_cap,
                max(0.0, float(payload.bootstrap_total_exposure_cap)),
            )
            effective_risk_model = "bootstrap"
        else:
            per_trade_cap = effective_budget
            total_exposure_cap = max(0.0, float(payload.micro_total_risk_usd)) if micro_active else 0.0
            effective_risk_model = "standard_micro" if micro_active else "standard"
        risk_mode_label = "bootstrap" if bootstrap_active else "standard"

        if self._is_xau_grid_setup(payload):
            return self._evaluate_xau_grid_dynamic_lot(
                payload=payload,
                stop_distance=stop_distance,
                effective_risk_pct=effective_risk_pct,
                boost_active=boost_active,
                base_requested_risk_pct=base_requested_risk_pct,
                bootstrap_active=bootstrap_active,
                per_trade_cap=per_trade_cap,
                total_exposure_cap=total_exposure_cap,
                soft_dd_active=bool(daily_governor.state in {"DAILY_CAUTION", "DAILY_DEFENSIVE"}),
                daily_governor=daily_governor,
                capacity_state=capacity_state,
            )

        if payload.use_fixed_lot:
            volume = clamp(payload.fixed_lot * size_factor, payload.volume_min, payload.volume_max)
            if volume <= 0:
                return RiskDecision(False, None, 0.0, 0.0, "fixed_lot_zero")
            if payload.margin_free <= 0:
                return RiskDecision(False, None, 0.0, risk_pct, "insufficient_margin")
            implied_loss = per_lot_risk * volume
            if bootstrap_active:
                if implied_loss > per_trade_cap:
                    tolerance_cap = self._bootstrap_min_lot_tolerance_cap(payload, per_trade_cap)
                    if not (volume <= (payload.volume_min + 1e-9) and implied_loss <= tolerance_cap):
                        return RiskDecision(False, None, 0.0, risk_pct, "bootstrap_trade_risk_exceeds_cap", diagnostics={
                            "estimated_loss_usd": implied_loss,
                            "budget_usd": effective_budget,
                            "effective_risk_cap": per_trade_cap,
                            "effective_risk_tolerance_cap": tolerance_cap,
                            "total_exposure_cap": total_exposure_cap,
                            "risk_formula": risk_formula,
                            "risk_mode": risk_mode_label,
                            "effective_risk_model": effective_risk_model,
                            "account_currency": payload.account_currency,
                            "economics_source": payload.economics_source,
                        })
                if (payload.projected_open_risk_usd + implied_loss) > total_exposure_cap:
                    return RiskDecision(False, None, 0.0, risk_pct, "bootstrap_total_risk_exceeds_cap", diagnostics={
                        "estimated_loss_usd": implied_loss,
                        "effective_risk_cap": per_trade_cap,
                        "effective_risk_tolerance_cap": self._bootstrap_min_lot_tolerance_cap(payload, per_trade_cap),
                        "total_exposure_cap": total_exposure_cap,
                        "risk_formula": risk_formula,
                        "risk_mode": risk_mode_label,
                        "effective_risk_model": effective_risk_model,
                        "account_currency": payload.account_currency,
                        "economics_source": payload.economics_source,
                    })
            else:
                if micro_active and implied_loss > payload.micro_max_loss_usd:
                    return RiskDecision(False, None, 0.0, risk_pct, "micro_survival_trade_risk_exceeds_usd")
                if micro_active and (payload.projected_open_risk_usd + implied_loss) > payload.micro_total_risk_usd:
                    return RiskDecision(False, None, 0.0, risk_pct, "micro_survival_total_risk_exceeds_usd")
            allowed_loss = max(0.0, per_trade_cap)
            if bootstrap_active and volume <= (payload.volume_min + 1e-9):
                allowed_loss = max(allowed_loss, self._bootstrap_min_lot_tolerance_cap(payload, per_trade_cap))
            if implied_loss > allowed_loss:
                return RiskDecision(
                    False,
                    None,
                    0.0,
                    risk_pct,
                    "fixed_lot_risk_exceeds_budget",
                    diagnostics={
                        "estimated_loss_usd": implied_loss,
                        "budget_usd": allowed_loss,
                        "base_risk_pct": base_requested_risk_pct,
                        "effective_risk_pct": effective_risk_pct,
                        "risk_boost_active": boost_active,
                        "risk_formula": risk_formula,
                        "risk_mode": risk_mode_label,
                        "effective_risk_model": effective_risk_model,
                        "account_currency": payload.account_currency,
                        "economics_source": payload.economics_source,
                    },
                )
            reason = "approved_fixed_lot_spread_elevated" if spread_elevated else "approved_fixed_lot"
            if bootstrap_active:
                reason = "approved_bootstrap_fixed_lot_spread_elevated" if spread_elevated else "approved_bootstrap_fixed_lot"
            return RiskDecision(
                True,
                None,
                round(volume, 2),
                effective_risk_pct,
                reason,
                extra_confluence_required=spread_elevated,
                adjusted_size_factor=size_factor,
                projected_loss_usd=implied_loss,
                diagnostics={
                    "estimated_loss_usd": implied_loss,
                    "budget_usd": effective_budget,
                    "base_risk_pct": base_requested_risk_pct,
                    "effective_risk_pct": effective_risk_pct,
                    "adjusted_risk_pct": effective_risk_pct,
                    "risk_boost_active": boost_active,
                    "risk_formula": risk_formula,
                    "risk_mode": risk_mode_label,
                    "effective_risk_model": effective_risk_model,
                    "adaptive_risk_state": "overflow_band" if overflow_band_active else "base_band",
                    "risk_modifiers": risk_modifiers,
                    "adaptive_multiplier": adaptive_multiplier,
                    "correlation_adjustment": float(payload.correlation_multiplier or 1.0),
                    "total_open_risk_pct": float(payload.total_open_risk_pct or 0.0),
                    "overflow_band_active": bool(overflow_band_active),
                    "overflow_cap_pct": float(allowed_risk_cap),
                    "base_daily_trade_target": int(capacity_state.get("base_daily_target", payload.max_trades_per_day)),
                    "stretch_daily_trade_target": int(capacity_state.get("stretch_daily_target", payload.stretch_max_trades_per_day or payload.max_trades_per_day)),
                    "hard_upper_limit": int(capacity_state.get("hard_upper_limit", payload.hard_upper_limit or payload.max_trades_per_day)),
                    "projected_trade_capacity_today": int(capacity_state.get("allowed_daily_target", payload.max_trades_per_day)),
                    "stretch_mode_active": bool(capacity_state.get("stretch_mode_active", False)),
                    "cluster_mode_active": bool(capacity_state.get("cluster_mode_active", False)),
                    "current_capacity_mode": str(capacity_state.get("current_capacity_mode", payload.current_capacity_mode or "BASE")),
                    "quality_cluster_score": float(capacity_state.get("quality_cluster_score", payload.quality_cluster_score or 0.0)),
                    "lane_strength_multiplier": float(capacity_state.get("lane_strength_multiplier", payload.lane_strength_multiplier or 1.0)),
                    "lane_score": float(capacity_state.get("lane_score", payload.lane_score or 0.0)),
                    "session_density_score": float(capacity_state.get("session_density_score", payload.session_density_score or 0.0)),
                    "effective_risk_cap": per_trade_cap,
                    "total_exposure_cap": total_exposure_cap,
                    "account_currency": payload.account_currency,
                    "economics_source": payload.economics_source,
                    "soft_dd_active": bool(daily_governor.state in {"DAILY_CAUTION", "DAILY_DEFENSIVE"}),
                    "daily_trade_overflow_active": bool(daily_trade_overflow_active),
                    "hard_dd_active": bool(daily_governor.state == "DAILY_HARD_STOP"),
                    "day_start_equity": float(payload.stats.day_start_equity),
                    "day_high_equity": float(payload.stats.day_high_equity),
                    "daily_realized_pnl": float(payload.stats.daily_realized_pnl),
                    "daily_pnl_pct": float(payload.stats.daily_pnl_pct),
                    "daily_dd_pct_live": float(payload.stats.daily_dd_pct_live),
                    "trading_day_key": self._stats_text(payload, "trading_day_key"),
                    "timezone_used": self._stats_text(payload, "timezone_used", "Australia/Sydney"),
                    "daily_state": str(daily_governor.state),
                    "daily_state_reason": str(daily_governor.reason),
                    "allowed_by_daily_governor": bool(daily_governor.allowed_by_daily_governor),
                    "risk_multiplier_applied": float(daily_governor.risk_multiplier),
                    "session_name": str(session_governor.session_name),
                    "session_stop_state": str(session_governor.state),
                    "session_stop_reason": str(session_governor.reason),
                    "session_entries_blocked": bool(not session_governor.allowed),
                    "session_realized_pnl": float(session_governor.session_realized_pnl),
                    "session_realized_pnl_pct": float(session_governor.session_realized_pnl_pct),
                    "session_trade_count": int(session_governor.session_trade_count),
                    "proof_exception_used": bool(daily_governor.proof_exception_used),
                    "xau_grid_override_allowed": bool(daily_governor.xau_grid_override_allowed),
                    "xau_grid_override_reason": str(daily_governor.xau_grid_override_reason),
                    "xau_grid_sub_budget": float(daily_governor.xau_grid_sub_budget),
                    "xau_grid_risk_multiplier": float(daily_governor.xau_grid_risk_multiplier),
                    "closed_trades_today": int(payload.stats.trades_today),
                    "soft_dd_elite_mode_active": False,
                    "soft_dd_trade_count": int(payload.stats.soft_dd_trade_count),
                    "last_soft_dd_trade_reason": "proof_exception" if daily_governor.proof_exception_used else "",
                    "current_phase": str(payload.current_phase),
                    "current_ai_threshold_mode": str(payload.current_ai_threshold_mode),
                },
            )

        risk_amount = effective_budget
        if risk_amount <= 0:
            return RiskDecision(False, None, 0.0, 0.0, "risk_amount_zero")

        raw_volume = risk_amount / per_lot_risk
        stepped = round(raw_volume / payload.volume_step) * payload.volume_step
        volume = min(max(0.0, stepped), payload.volume_max)
        bootstrap_min_lot_applied = False
        if volume < payload.volume_min:
            volume = payload.volume_min
            if bootstrap_active:
                min_lot_loss = per_lot_risk * payload.volume_min
                tolerance_cap = self._bootstrap_min_lot_tolerance_cap(payload, per_trade_cap)
                bootstrap_min_lot_applied = (
                    min_lot_loss <= tolerance_cap
                    and (payload.projected_open_risk_usd + min_lot_loss) <= total_exposure_cap
                )
        if volume <= 0:
            return RiskDecision(False, None, 0.0, effective_risk_pct, "volume_zero")
        if payload.margin_free <= 0:
            return RiskDecision(False, None, 0.0, effective_risk_pct, "insufficient_margin")
        implied_loss = per_lot_risk * volume
        if bootstrap_active:
            if implied_loss > per_trade_cap:
                tolerance_cap = self._bootstrap_min_lot_tolerance_cap(payload, per_trade_cap)
                if not (bootstrap_min_lot_applied and implied_loss <= tolerance_cap):
                    return RiskDecision(False, None, 0.0, effective_risk_pct, "bootstrap_trade_risk_exceeds_cap", diagnostics={
                        "estimated_loss_usd": implied_loss,
                        "budget_usd": effective_budget,
                        "effective_risk_cap": per_trade_cap,
                        "effective_risk_tolerance_cap": tolerance_cap,
                        "total_exposure_cap": total_exposure_cap,
                        "risk_formula": risk_formula,
                        "risk_mode": risk_mode_label,
                        "effective_risk_model": effective_risk_model,
                        "account_currency": payload.account_currency,
                        "economics_source": payload.economics_source,
                    })
            if (payload.projected_open_risk_usd + implied_loss) > total_exposure_cap:
                return RiskDecision(False, None, 0.0, effective_risk_pct, "bootstrap_total_risk_exceeds_cap", diagnostics={
                    "estimated_loss_usd": implied_loss,
                    "projected_open_risk_usd": payload.projected_open_risk_usd,
                    "effective_risk_cap": per_trade_cap,
                    "total_exposure_cap": total_exposure_cap,
                    "risk_formula": risk_formula,
                    "risk_mode": risk_mode_label,
                    "effective_risk_model": effective_risk_model,
                    "account_currency": payload.account_currency,
                    "economics_source": payload.economics_source,
                })
        else:
            if micro_active and implied_loss > payload.micro_max_loss_usd:
                return RiskDecision(False, None, 0.0, effective_risk_pct, "micro_survival_trade_risk_exceeds_usd")
            if micro_active and (payload.projected_open_risk_usd + implied_loss) > payload.micro_total_risk_usd:
                return RiskDecision(False, None, 0.0, effective_risk_pct, "micro_survival_total_risk_exceeds_usd")

        reason = "approved_spread_elevated" if spread_elevated else "approved"
        if bootstrap_active:
            if bootstrap_min_lot_applied:
                reason = "approved_bootstrap_min_lot"
            else:
                reason = "approved_bootstrap_spread_elevated" if spread_elevated else "approved_bootstrap"
        return RiskDecision(
            True,
            None,
            round(volume, 2),
            effective_risk_pct,
            reason,
            extra_confluence_required=spread_elevated,
            adjusted_size_factor=size_factor,
            projected_loss_usd=implied_loss,
            diagnostics={
                "estimated_loss_usd": implied_loss,
                "budget_usd": risk_amount,
                "base_risk_pct": base_requested_risk_pct,
                "effective_risk_pct": effective_risk_pct,
                "adjusted_risk_pct": effective_risk_pct,
                "risk_boost_active": boost_active,
                "risk_formula": risk_formula,
                "risk_mode": risk_mode_label,
                "effective_risk_model": effective_risk_model,
                "adaptive_risk_state": "overflow_band" if overflow_band_active else "base_band",
                "risk_modifiers": risk_modifiers,
                "adaptive_multiplier": adaptive_multiplier,
                "correlation_adjustment": float(payload.correlation_multiplier or 1.0),
                "total_open_risk_pct": float(payload.total_open_risk_pct or 0.0),
                "overflow_band_active": bool(overflow_band_active),
                "overflow_cap_pct": float(allowed_risk_cap),
                "effective_risk_cap": per_trade_cap,
                "total_exposure_cap": total_exposure_cap,
                "bootstrap_min_lot_applied": bootstrap_min_lot_applied,
                "account_currency": payload.account_currency,
                "economics_source": payload.economics_source,
                "soft_dd_active": bool(daily_governor.state in {"DAILY_CAUTION", "DAILY_DEFENSIVE"}),
                "daily_trade_overflow_active": bool(daily_trade_overflow_active),
                "hard_dd_active": bool(daily_governor.state == "DAILY_HARD_STOP"),
                "day_start_equity": float(payload.stats.day_start_equity),
                "day_high_equity": float(payload.stats.day_high_equity),
                "daily_realized_pnl": float(payload.stats.daily_realized_pnl),
                "daily_pnl_pct": float(payload.stats.daily_pnl_pct),
                "daily_dd_pct_live": float(payload.stats.daily_dd_pct_live),
                "trading_day_key": self._stats_text(payload, "trading_day_key"),
                "timezone_used": self._stats_text(payload, "timezone_used", "Australia/Sydney"),
                "daily_state": str(daily_governor.state),
                "daily_state_reason": str(daily_governor.reason),
                "allowed_by_daily_governor": bool(daily_governor.allowed_by_daily_governor),
                "risk_multiplier_applied": float(daily_governor.risk_multiplier),
                "session_name": str(session_governor.session_name),
                "session_stop_state": str(session_governor.state),
                "session_stop_reason": str(session_governor.reason),
                "session_entries_blocked": bool(not session_governor.allowed),
                "session_realized_pnl": float(session_governor.session_realized_pnl),
                "session_realized_pnl_pct": float(session_governor.session_realized_pnl_pct),
                "session_trade_count": int(session_governor.session_trade_count),
                "proof_exception_used": bool(daily_governor.proof_exception_used),
                "xau_grid_override_allowed": bool(daily_governor.xau_grid_override_allowed),
                "xau_grid_override_reason": str(daily_governor.xau_grid_override_reason),
                "xau_grid_sub_budget": float(daily_governor.xau_grid_sub_budget),
                "xau_grid_risk_multiplier": float(daily_governor.xau_grid_risk_multiplier),
                "base_daily_trade_target": int(capacity_state.get("base_daily_target", payload.max_trades_per_day)),
                "stretch_daily_trade_target": int(capacity_state.get("stretch_daily_target", payload.stretch_max_trades_per_day or payload.max_trades_per_day)),
                "hard_upper_limit": int(capacity_state.get("hard_upper_limit", payload.hard_upper_limit or payload.max_trades_per_day)),
                "projected_trade_capacity_today": int(capacity_state.get("allowed_daily_target", payload.max_trades_per_day)),
                "hourly_base_target": int(capacity_state.get("base_hourly_target", payload.max_trades_per_hour)),
                "hourly_stretch_target": int(capacity_state.get("stretch_hourly_target", payload.stretch_max_trades_per_hour or payload.max_trades_per_hour)),
                "allowed_hourly_target": int(capacity_state.get("allowed_hourly_target", payload.max_trades_per_hour)),
                "effective_hourly_trade_cap": int(effective_max_trades_per_hour),
                "stretch_mode_active": bool(capacity_state.get("stretch_mode_active", False)),
                "cluster_mode_active": bool(capacity_state.get("cluster_mode_active", False)),
                "current_capacity_mode": str(capacity_state.get("current_capacity_mode", payload.current_capacity_mode or "BASE")),
                "quality_cluster_score": float(capacity_state.get("quality_cluster_score", payload.quality_cluster_score or 0.0)),
                "lane_strength_multiplier": float(capacity_state.get("lane_strength_multiplier", payload.lane_strength_multiplier or 1.0)),
                "lane_score": float(capacity_state.get("lane_score", payload.lane_score or 0.0)),
                "session_density_score": float(capacity_state.get("session_density_score", payload.session_density_score or 0.0)),
                "session_priority_multiplier": float(capacity_state.get("session_priority_multiplier", payload.session_priority_multiplier or 1.0)),
                "lane_budget_share": float(capacity_state.get("lane_budget_share", payload.lane_budget_share or 0.0)),
                "lane_available_capacity": int(capacity_state.get("lane_available_capacity", payload.lane_available_capacity or 0.0)),
                "hot_lane_borrow_share": float(capacity_state.get("hot_lane_borrow_share", 0.0)),
                "hot_hand_active": bool(capacity_state.get("hot_hand_active", payload.hot_hand_active)),
                "hot_hand_score": float(capacity_state.get("hot_hand_score", payload.hot_hand_score or 0.0)),
                "session_bankroll_bias": float(capacity_state.get("session_bankroll_bias", payload.session_bankroll_bias or 1.0)),
                "profit_recycle_active": bool(capacity_state.get("profit_recycle_active", payload.profit_recycle_active)),
                "profit_recycle_boost": float(capacity_state.get("profit_recycle_boost", payload.profit_recycle_boost or 0.0)),
                "close_winners_score": float(capacity_state.get("close_winners_score", payload.close_winners_score or 0.5)),
                "closed_trades_today": int(payload.stats.trades_today),
                "soft_dd_elite_mode_active": False,
                "soft_dd_trade_count": int(payload.stats.soft_dd_trade_count),
                "last_soft_dd_trade_reason": "proof_exception" if daily_governor.proof_exception_used else "",
                "current_phase": str(payload.current_phase),
                "current_ai_threshold_mode": str(payload.current_ai_threshold_mode),
            },
        )

    def _evaluate_xau_grid_dynamic_lot(
        self,
        *,
        payload: RiskInputs,
        stop_distance: float,
        effective_risk_pct: float,
        boost_active: bool,
        base_requested_risk_pct: float,
        bootstrap_active: bool,
        per_trade_cap: float,
        total_exposure_cap: float,
        soft_dd_active: bool,
        daily_governor: DailyGovernorState | None = None,
        capacity_state: dict[str, Any] | None = None,
    ) -> RiskDecision:
        capacity_state = capacity_state or {}
        budget_usd = max(0.0, payload.equity * effective_risk_pct)
        remaining_budget_usd = max(0.0, budget_usd - max(0.0, payload.projected_cycle_risk_usd))
        floor = max(0.0, float(payload.max_loss_usd_floor))
        _, band_label = self._quality_rank(
            str(payload.trade_quality_detail or payload.trade_quality_band or ""),
            float(payload.trade_quality_score or payload.ai_probability or 0.0),
        )
        point_size, _ = self._resolve_point_size(payload)
        tick_size = float(payload.symbol_tick_size or 0.0)
        tick_value = float(payload.symbol_tick_value or 0.0)

        if tick_size > 0 and tick_value > 0:
            per_lot_loss_at_sl_usd = (stop_distance / tick_size) * tick_value
            risk_formula = "tick_size_tick_value"
        else:
            per_lot_loss_at_sl_usd = stop_distance * max(payload.contract_size, 1.0)
            risk_formula = "contract_size_fallback"
        per_lot_loss_at_sl_usd = max(per_lot_loss_at_sl_usd, 1e-9)
        lot_risk_cap = max(0.0, remaining_budget_usd / per_lot_loss_at_sl_usd)

        margin_per_lot = float(payload.margin_per_lot or 0.0)
        if margin_per_lot > 0:
            lot_margin_cap = max(0.0, (payload.margin_free * max(0.0, payload.margin_safety_fraction)) / margin_per_lot)
            margin_formula = "margin_per_lot"
        elif float(payload.account_leverage or 0.0) > 0 and payload.entry_price > 0 and payload.contract_size > 0:
            margin_per_lot_est = (payload.entry_price * payload.contract_size) / max(float(payload.account_leverage or 1.0), 1.0)
            lot_margin_cap = max(0.0, (payload.margin_free * max(0.0, payload.margin_safety_fraction)) / max(margin_per_lot_est, 1e-9))
            margin_formula = "estimated_from_leverage"
        else:
            approx_margin_budget = max(0.0, payload.margin_free) * max(0.0, payload.margin_safety_fraction)
            if approx_margin_budget < 1.0:
                lot_margin_cap = 0.0
                margin_formula = "free_margin_too_low"
            else:
                lot_margin_cap = max(payload.volume_min, payload.volume_max)
                margin_formula = "free_margin_approx"

        raw_lot = min(lot_risk_cap, lot_margin_cap, payload.volume_max)
        final_lot = self._floor_to_step(raw_lot, payload.volume_step)
        tolerance_cap = self._bootstrap_min_lot_tolerance_cap(payload, per_trade_cap)
        min_lot_loss = per_lot_loss_at_sl_usd * payload.volume_min
        low_equity_xau_candidate = bool(
            bool(payload.low_equity_safety_enabled)
            and float(payload.equity or 0.0) <= float(payload.low_equity_max_equity or 0.0)
            and self._is_xau_grid_setup(payload)
        )
        strong_bootstrap_min_lot_override = bool(
            (bootstrap_active or low_equity_xau_candidate)
            and self._is_xau_grid_setup(payload)
            and bool(payload.session_native_pair)
            and float(payload.session_priority_multiplier or 1.0) >= 1.15
            and float(payload.trade_quality_score or 0.0) >= 0.82
            and float(payload.execution_quality_score or 0.0) >= 0.88
            and float(payload.session_quality_score or 0.0) >= 0.80
            and float(payload.ai_probability or 0.0) >= 0.80
            and float(payload.expected_value_r or 0.0) >= 1.20
            and float(payload.confluence_score or 0.0) >= 4.5
            and float(payload.candidate_monte_carlo_win_rate or 0.0) >= max(float(payload.low_equity_monte_carlo_floor or 0.88) - 0.10, 0.78)
            and str(payload.news_state or "").upper() not in {"NEWS_BLOCK", "NEWS_UNSAFE"}
            and min_lot_loss <= tolerance_cap
            and payload.volume_min <= lot_margin_cap
        )
        if strong_bootstrap_min_lot_override:
            lot_risk_cap = max(lot_risk_cap, payload.volume_min)
            raw_lot = min(max(raw_lot, payload.volume_min), lot_margin_cap, payload.volume_max)
            final_lot = max(final_lot, payload.volume_min)
        if final_lot < payload.volume_min and payload.volume_min <= lot_risk_cap and payload.volume_min <= lot_margin_cap:
            final_lot = payload.volume_min
        elif bootstrap_active and final_lot < payload.volume_min and payload.volume_min <= lot_margin_cap and min_lot_loss <= tolerance_cap:
            final_lot = payload.volume_min
        elif strong_bootstrap_min_lot_override and final_lot < payload.volume_min:
            final_lot = payload.volume_min
        diagnostics = {
            "sl_distance_price": stop_distance,
            "sl_distance_points": stop_distance / point_size,
            "budget_usd": budget_usd,
            "remaining_budget_usd": remaining_budget_usd,
            "per_lot_loss_at_sl_usd": per_lot_loss_at_sl_usd,
            "lot_risk_cap": lot_risk_cap,
            "lot_margin_cap": lot_margin_cap,
            "base_risk_pct": base_requested_risk_pct,
            "effective_risk_pct": effective_risk_pct,
            "risk_boost_active": boost_active,
            "tick_size": tick_size,
            "tick_value": tick_value,
            "risk_formula": risk_formula,
            "margin_formula": margin_formula,
            "effective_risk_tolerance_cap": tolerance_cap,
            "min_lot_loss_usd": min_lot_loss,
            "strong_bootstrap_min_lot_override": bool(strong_bootstrap_min_lot_override),
        }
        if floor > 0 and remaining_budget_usd < floor and not strong_bootstrap_min_lot_override:
            return RiskDecision(False, None, 0.0, effective_risk_pct, "lot_below_min_or_margin_too_low", diagnostics=diagnostics)
        if final_lot < payload.volume_min:
            return RiskDecision(False, None, 0.0, effective_risk_pct, "lot_below_min_or_margin_too_low", diagnostics=diagnostics)
        implied_loss = per_lot_loss_at_sl_usd * final_lot
        if bootstrap_active:
            if implied_loss > per_trade_cap:
                if not (final_lot <= (payload.volume_min + 1e-9) and implied_loss <= tolerance_cap):
                    diagnostics["effective_risk_cap"] = per_trade_cap
                    diagnostics["total_exposure_cap"] = total_exposure_cap
                    diagnostics["risk_mode"] = "bootstrap"
                    diagnostics["account_currency"] = payload.account_currency
                    diagnostics["economics_source"] = payload.economics_source
                    return RiskDecision(False, None, 0.0, effective_risk_pct, "bootstrap_trade_risk_exceeds_cap", diagnostics=diagnostics)
            if (payload.projected_open_risk_usd + implied_loss) > total_exposure_cap:
                diagnostics["effective_risk_cap"] = per_trade_cap
                diagnostics["total_exposure_cap"] = total_exposure_cap
                diagnostics["risk_mode"] = "bootstrap"
                diagnostics["account_currency"] = payload.account_currency
                diagnostics["economics_source"] = payload.economics_source
                return RiskDecision(False, None, 0.0, effective_risk_pct, "bootstrap_total_risk_exceeds_cap", diagnostics=diagnostics)
        elif micro_active := (payload.micro_enabled and payload.mode.upper() in {"DEMO", "PAPER", "LIVE"}):
            if implied_loss > payload.micro_max_loss_usd:
                return RiskDecision(False, None, 0.0, effective_risk_pct, "micro_survival_trade_risk_exceeds_usd", diagnostics=diagnostics)
            if (payload.projected_open_risk_usd + implied_loss) > payload.micro_total_risk_usd:
                return RiskDecision(False, None, 0.0, effective_risk_pct, "micro_survival_total_risk_exceeds_usd", diagnostics=diagnostics)
        diagnostics["estimated_loss_usd"] = implied_loss
        diagnostics["effective_risk_cap"] = per_trade_cap
        diagnostics["total_exposure_cap"] = total_exposure_cap
        diagnostics["risk_mode"] = "bootstrap" if bootstrap_active else "standard"
        diagnostics["account_currency"] = payload.account_currency
        diagnostics["economics_source"] = payload.economics_source
        diagnostics["execution_quality_score"] = float(payload.execution_quality_score or 0.0)
        diagnostics["execution_quality_state"] = str(payload.execution_quality_state or "GOOD")
        diagnostics["spread_quality_score"] = float(payload.spread_quality_score or 0.0)
        diagnostics["slippage_quality_score"] = float(
            clamp(
                (0.55 * float(payload.execution_quality_score or 0.0))
                + (0.45 * float(payload.spread_quality_score or 0.0)),
                0.0,
                1.0,
            )
        )
        diagnostics["soft_dd_active"] = bool(soft_dd_active)
        diagnostics["hard_dd_active"] = bool(daily_governor.state == "DAILY_HARD_STOP") if daily_governor else False
        diagnostics["day_start_equity"] = float(payload.stats.day_start_equity)
        diagnostics["day_high_equity"] = float(payload.stats.day_high_equity)
        diagnostics["daily_realized_pnl"] = float(payload.stats.daily_realized_pnl)
        diagnostics["daily_pnl_pct"] = float(payload.stats.daily_pnl_pct)
        diagnostics["daily_dd_pct_live"] = float(payload.stats.daily_dd_pct_live)
        diagnostics["trading_day_key"] = self._stats_text(payload, "trading_day_key")
        diagnostics["timezone_used"] = self._stats_text(payload, "timezone_used", "Australia/Sydney")
        diagnostics["daily_state"] = str(daily_governor.state) if daily_governor else "DAILY_NORMAL"
        diagnostics["daily_state_reason"] = str(daily_governor.reason) if daily_governor else "daily_governor_normal"
        diagnostics["allowed_by_daily_governor"] = bool(daily_governor.allowed_by_daily_governor) if daily_governor else True
        diagnostics["risk_multiplier_applied"] = float(daily_governor.risk_multiplier) if daily_governor else 1.0
        diagnostics["proof_exception_used"] = bool(daily_governor.proof_exception_used) if daily_governor else False
        diagnostics["xau_grid_override_allowed"] = bool(daily_governor.xau_grid_override_allowed) if daily_governor else False
        diagnostics["xau_grid_override_reason"] = str(daily_governor.xau_grid_override_reason) if daily_governor else ""
        diagnostics["xau_grid_sub_budget"] = float(daily_governor.xau_grid_sub_budget) if daily_governor else 0.0
        diagnostics["xau_grid_risk_multiplier"] = float(daily_governor.xau_grid_risk_multiplier) if daily_governor else 0.0
        diagnostics["base_daily_trade_target"] = int(capacity_state.get("base_daily_target", payload.max_trades_per_day))
        diagnostics["stretch_daily_trade_target"] = int(capacity_state.get("stretch_daily_target", payload.stretch_max_trades_per_day or payload.max_trades_per_day))
        diagnostics["hard_upper_limit"] = int(capacity_state.get("hard_upper_limit", payload.hard_upper_limit or payload.max_trades_per_day))
        diagnostics["projected_trade_capacity_today"] = int(capacity_state.get("allowed_daily_target", payload.max_trades_per_day))
        diagnostics["hourly_base_target"] = int(capacity_state.get("base_hourly_target", payload.max_trades_per_hour))
        diagnostics["hourly_stretch_target"] = int(capacity_state.get("stretch_hourly_target", payload.stretch_max_trades_per_hour or payload.max_trades_per_hour))
        diagnostics["stretch_mode_active"] = bool(capacity_state.get("stretch_mode_active", False))
        diagnostics["cluster_mode_active"] = bool(capacity_state.get("cluster_mode_active", False))
        diagnostics["current_capacity_mode"] = str(capacity_state.get("current_capacity_mode", payload.current_capacity_mode or "BASE"))
        diagnostics["quality_cluster_score"] = float(capacity_state.get("quality_cluster_score", payload.quality_cluster_score or 0.0))
        diagnostics["lane_strength_multiplier"] = float(capacity_state.get("lane_strength_multiplier", payload.lane_strength_multiplier or 1.0))
        diagnostics["lane_score"] = float(capacity_state.get("lane_score", payload.lane_score or 0.0))
        diagnostics["lane_expectancy_multiplier"] = float(capacity_state.get("lane_expectancy_multiplier", payload.lane_expectancy_multiplier or 1.0))
        diagnostics["lane_expectancy_score"] = float(capacity_state.get("lane_expectancy_score", payload.lane_expectancy_score or 0.0))
        diagnostics["session_density_score"] = float(capacity_state.get("session_density_score", payload.session_density_score or 0.0))
        diagnostics["session_priority_multiplier"] = float(capacity_state.get("session_priority_multiplier", payload.session_priority_multiplier or 1.0))
        diagnostics["lane_budget_share"] = float(capacity_state.get("lane_budget_share", payload.lane_budget_share or 0.0))
        diagnostics["lane_available_capacity"] = int(capacity_state.get("lane_available_capacity", payload.lane_available_capacity or 0.0))
        diagnostics["hot_lane_borrow_share"] = float(capacity_state.get("hot_lane_borrow_share", 0.0))
        diagnostics["hot_hand_active"] = bool(capacity_state.get("hot_hand_active", payload.hot_hand_active))
        diagnostics["hot_hand_score"] = float(capacity_state.get("hot_hand_score", payload.hot_hand_score or 0.0))
        diagnostics["session_bankroll_bias"] = float(capacity_state.get("session_bankroll_bias", payload.session_bankroll_bias or 1.0))
        diagnostics["profit_recycle_active"] = bool(capacity_state.get("profit_recycle_active", payload.profit_recycle_active))
        diagnostics["profit_recycle_boost"] = float(capacity_state.get("profit_recycle_boost", payload.profit_recycle_boost or 0.0))
        diagnostics["close_winners_score"] = float(capacity_state.get("close_winners_score", payload.close_winners_score or 0.5))
        diagnostics["soft_dd_elite_mode_active"] = False
        diagnostics["selected_trade_band"] = band_label
        diagnostics["soft_dd_trade_count"] = int(payload.stats.soft_dd_trade_count)
        diagnostics["last_soft_dd_trade_reason"] = "proof_exception" if (daily_governor and daily_governor.proof_exception_used) else ""
        diagnostics["current_phase"] = str(payload.current_phase)
        diagnostics["current_ai_threshold_mode"] = str(payload.current_ai_threshold_mode)
        diagnostics["closed_trades_today"] = int(payload.stats.trades_today)
        return RiskDecision(
            True,
            None,
            round(final_lot, 2),
            effective_risk_pct,
            "approved_grid_dynamic_lot",
            projected_loss_usd=implied_loss,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _effective_max_trades_per_hour(
        payload: RiskInputs,
        *,
        daily_state: str = "DAILY_NORMAL",
        capacity_state: dict[str, Any] | None = None,
    ) -> int:
        capacity_state = capacity_state or {}
        limit = max(1, int(capacity_state.get("allowed_hourly_target", payload.max_trades_per_hour)))
        if payload.stats.consecutive_losses >= 7 and payload.stats.cooldown_trades_remaining > 0:
            limit = max(1, int(limit * 0.45))
        elif payload.stats.consecutive_losses >= 5:
            limit = max(1, int(limit * 0.78))
        elif payload.stats.consecutive_losses >= 3 or payload.stats.cooldown_trades_remaining > 0:
            limit = max(1, int(limit * 0.90))
        elif payload.stats.consecutive_losses >= 2:
            limit = max(1, int(limit * 0.96))
        if str(daily_state).upper() == "DAILY_CAUTION":
            limit = max(1, int(limit * 0.95))
        elif str(daily_state).upper() == "DAILY_DEFENSIVE":
            limit = max(1, int(limit * 0.85))
        elif str(daily_state).upper() == "DAILY_HARD_STOP":
            return 1
        if payload.stats.rolling_drawdown_pct >= 0.03:
            limit = max(1, int(limit * 0.92))
        if payload.stats.daily_pnl_pct <= -payload.max_daily_loss:
            limit = max(1, int(limit * 0.92))
        if (
            str(daily_state).upper() in {"DAILY_NORMAL", "DAILY_CAUTION"}
            and float(payload.trade_quality_score or 0.0) >= 0.72
            and str(payload.execution_quality_state or "GOOD").upper() == "GOOD"
            and float(payload.correlation_multiplier or 1.0) >= 0.85
        ):
            limit = min(limit + 1, max(limit, int(capacity_state.get("stretch_hourly_target", payload.stretch_max_trades_per_hour or payload.max_trades_per_hour)) + 1))
        if (
            str(daily_state).upper() == "DAILY_NORMAL"
            and float(payload.trade_quality_score or 0.0) >= 0.84
            and float(payload.session_quality_score or 0.0) >= 0.75
            and float(payload.spread_quality_score or 0.0) >= 0.75
        ):
            limit = min(limit + 1, max(limit, int(capacity_state.get("stretch_hourly_target", payload.stretch_max_trades_per_hour or payload.max_trades_per_hour))))
        if int(payload.stats.winning_streak or 0) >= 2:
            limit = min(limit + 1, max(limit, int(capacity_state.get("stretch_hourly_target", payload.stretch_max_trades_per_hour or payload.max_trades_per_hour))))
        if bool(payload.soft_trade_budget_enabled):
            soft_multiplier = clamp(
                float(capacity_state.get("aggression_lane_multiplier", payload.aggression_lane_multiplier or 1.0))
                * float(capacity_state.get("execution_minute_size_multiplier", payload.execution_minute_size_multiplier or 1.0)),
                0.85,
                1.60,
            )
            if float(capacity_state.get("signal_alignment_score", 0.0) or 0.0) >= 0.22:
                soft_multiplier += 0.10
            if str(payload.event_playbook or "").lower() in {"breakout", "risk_on_follow", "swing_hold"}:
                soft_multiplier += 0.06
            if bool(payload.event_pre_position_allowed):
                soft_multiplier += 0.04
            if bool(payload.funded_account_mode):
                soft_multiplier += 0.08
            limit = min(
                max(limit, int(round(limit * clamp(soft_multiplier, 0.85, 1.75))) + max(0, int(payload.hot_lane_concurrency_bonus or 0))),
                max(
                    limit,
                    int(capacity_state.get("stretch_hourly_target", payload.stretch_max_trades_per_hour or payload.max_trades_per_hour))
                    + max(0, int(payload.hot_lane_concurrency_bonus or 0))
                    + 4,
                ),
            )
        if (
            bool(payload.winning_streak_mode_active)
            or int(payload.stats.winning_streak or 0) >= 2
            or float(payload.recent_expectancy_r or 0.0) >= 0.05
            or float(capacity_state.get("lane_strength_multiplier", payload.lane_strength_multiplier or 1.0) or 1.0) >= 1.08
        ):
            stretch_cap = max(
                limit,
                int(capacity_state.get("stretch_hourly_target", payload.stretch_max_trades_per_hour or payload.max_trades_per_hour)),
            )
            hot_bonus = max(
                2,
                int(round(max(0.0, float(payload.hot_lane_concurrency_bonus or 0.0)))),
            )
            limit = max(limit, stretch_cap + hot_bonus)
        return limit

    @staticmethod
    def _performance_quality(payload: RiskInputs) -> float:
        expectancy_component = clamp((float(payload.recent_expectancy_r) + 0.25) / 0.75, 0.0, 1.0)
        win_rate_component = clamp((float(payload.recent_win_rate) - 0.35) / 0.35, 0.0, 1.0)
        closed_component = clamp(float(payload.stats.closed_trades_total) / 20.0, 0.0, 1.0)
        return clamp(
            (0.45 * expectancy_component)
            + (0.35 * win_rate_component)
            + (0.20 * closed_component),
            0.0,
            1.0,
        )

    @classmethod
    def _adaptive_risk_profile(
        cls,
        payload: RiskInputs,
        *,
        spread_elevated: bool,
        daily_state: str,
        hard_cap: float,
        daily_trade_overflow_active: bool = False,
        capacity_state: dict[str, Any] | None = None,
    ) -> tuple[float, dict[str, float], bool, float]:
        capacity_state = capacity_state or {}
        quality_score = clamp(float(payload.trade_quality_score or payload.ai_probability or 0.0), 0.0, 1.0)
        regime_confidence = clamp(float(payload.regime_confidence or 0.0), 0.0, 1.0)
        execution_quality = clamp(float(payload.execution_quality_score or 0.0), 0.0, 1.0)
        spread_quality = clamp(
            float(payload.spread_quality_score or 0.0)
            if float(payload.spread_quality_score or 0.0) > 0.0
            else (1.0 - min(float(payload.spread_points) / max(float(payload.max_spread_points), 1.0), 1.0)),
            0.0,
            1.0,
        )
        session_quality = clamp(float(payload.session_quality_score or 0.0), 0.0, 1.0)
        performance_quality = cls._performance_quality(payload)
        drawdown_pressure = max(
            clamp(float(payload.stats.daily_dd_pct_live) / max(float(payload.hard_daily_dd_pct), 1e-9), 0.0, 1.0),
            clamp(float(payload.stats.rolling_drawdown_pct) / max(float(payload.max_drawdown_kill), 1e-9), 0.0, 1.0),
        )
        drawdown_quality = clamp(1.0 - (drawdown_pressure * 0.75), 0.35, 1.0)
        exposure_quality = clamp(1.0 - (float(payload.total_open_risk_pct) / max(float(hard_cap), 0.01)), 0.40, 1.0)
        correlation_quality = clamp(float(payload.correlation_multiplier or 1.0), 0.35, 1.0)
        lane_strength_quality = clamp(float(capacity_state.get("lane_strength_multiplier", payload.lane_strength_multiplier or 1.0)), 0.35, 1.20)
        lane_score_quality = clamp(float(capacity_state.get("lane_score", payload.lane_score or quality_score)), 0.0, 1.0)
        lane_expectancy_quality = clamp(float(capacity_state.get("lane_expectancy_multiplier", payload.lane_expectancy_multiplier or 1.0)), 0.35, 1.25)
        lane_expectancy_score = clamp(float(capacity_state.get("lane_expectancy_score", payload.lane_expectancy_score or lane_score_quality)), 0.0, 1.0)
        cluster_quality = clamp(float(capacity_state.get("quality_cluster_score", payload.quality_cluster_score or 0.0)), 0.0, 1.0)
        hot_hand_score = clamp(float(payload.hot_hand_score or 0.0), 0.0, 1.0)
        session_bankroll_bias = clamp(float(payload.session_bankroll_bias or 1.0), 0.85, 1.35)
        profit_recycle_boost = clamp(float(payload.profit_recycle_boost or 0.0), 0.0, 0.25)
        close_winners_score = clamp(float(payload.close_winners_score or 0.5), 0.0, 1.0)
        microstructure_alignment = clamp(float(payload.microstructure_alignment_score or 0.0), -1.0, 1.0)
        microstructure_confidence = clamp(float(payload.microstructure_confidence or 0.0), 0.0, 1.0)
        lead_lag_alignment = clamp(float(payload.lead_lag_alignment_score or 0.0), -1.0, 1.0)
        lead_lag_confidence = clamp(float(payload.lead_lag_confidence or 0.0), 0.0, 1.0)
        lead_lag_disagreement_penalty = clamp(float(payload.lead_lag_disagreement_penalty or 0.0), 0.0, 1.0)
        execution_minute_quality = clamp(float(payload.execution_minute_quality_score or 0.70), 0.0, 1.0)
        execution_minute_size_multiplier = clamp(float(payload.execution_minute_size_multiplier or 1.0), 0.70, 1.35)
        signal_alignment = clamp(
            (microstructure_alignment * 0.44)
            + (lead_lag_alignment * 0.34)
            + ((microstructure_confidence - 0.5) * 0.12)
            + ((lead_lag_confidence - 0.5) * 0.10)
            - (lead_lag_disagreement_penalty * 0.16),
            -1.0,
            1.0,
        )
        modifiers = {
            "trade_quality": quality_score,
            "regime_confidence": regime_confidence,
            "execution_quality": execution_quality,
            "spread_quality": spread_quality,
            "session_quality": session_quality,
            "performance_quality": performance_quality,
            "drawdown_quality": drawdown_quality,
            "exposure_quality": exposure_quality,
            "correlation_quality": correlation_quality,
            "lane_strength_quality": lane_strength_quality,
            "lane_score_quality": lane_score_quality,
            "lane_expectancy_quality": lane_expectancy_quality,
            "lane_expectancy_score": lane_expectancy_score,
            "cluster_quality": cluster_quality,
            "hot_hand_score": hot_hand_score,
            "session_bankroll_bias": session_bankroll_bias,
            "profit_recycle_boost": profit_recycle_boost,
            "close_winners_score": close_winners_score,
            "signal_alignment": signal_alignment,
            "execution_minute_quality": execution_minute_quality,
        }
        weighted_quality = clamp(
            (0.20 * quality_score)
            + (0.14 * regime_confidence)
            + (0.12 * execution_quality)
            + (0.10 * spread_quality)
            + (0.10 * session_quality)
            + (0.12 * performance_quality)
            + (0.10 * drawdown_quality)
            + (0.05 * exposure_quality)
            + (0.05 * correlation_quality)
            + (0.04 * execution_minute_quality),
            0.0,
            1.0,
        )
        weighted_quality = clamp(
            weighted_quality
            + (0.05 * lane_score_quality)
            + (0.04 * lane_expectancy_score)
            + (0.04 * cluster_quality)
            + (0.03 * clamp(lane_strength_quality - 1.0, -0.5, 0.5))
            + (0.04 * max(0.0, signal_alignment)),
            0.0,
            1.0,
        )
        weighted_quality = clamp(
            weighted_quality + (0.03 * clamp(lane_expectancy_quality - 1.0, -0.5, 0.5)),
            0.0,
            1.0,
        )
        weighted_quality = clamp(
            weighted_quality
            + (0.03 * hot_hand_score)
            + (0.03 * clamp(session_bankroll_bias - 1.0, -0.5, 0.5))
            + (0.03 * clamp(close_winners_score - 0.50, -0.5, 0.5))
            + (0.02 * profit_recycle_boost)
            + (0.02 * clamp(execution_minute_size_multiplier - 1.0, -0.5, 0.5)),
            0.0,
            1.0,
        )
        explicit_adaptive_inputs = any(
            (
                float(payload.trade_quality_score or 0.0) > 0.0,
                float(payload.regime_confidence or 0.0) > 0.0,
                abs(float(payload.execution_quality_score or 0.70) - 0.70) > 1e-9,
                abs(float(payload.spread_quality_score or 0.70) - 0.70) > 1e-9,
                abs(float(payload.session_quality_score or 0.70) - 0.70) > 1e-9,
                abs(float(payload.recent_expectancy_r or 0.0)) > 1e-9,
                abs(float(payload.recent_win_rate or 0.5) - 0.5) > 1e-9,
                float(payload.total_open_risk_pct or 0.0) > 0.0,
                abs(float(payload.correlation_multiplier or 1.0) - 1.0) > 1e-9,
                bool(payload.degraded_mode_active),
                str(payload.news_state or "NEWS_SAFE").upper() != "NEWS_SAFE",
                float(payload.news_confidence or 1.0) < 0.999,
            )
        )
        multiplier = 1.0 if not explicit_adaptive_inputs else clamp(0.76 + (weighted_quality * 0.44), 0.58, 1.20)
        if bool(payload.hot_hand_active) and hot_hand_score >= 0.55:
            multiplier *= 1.0 + min(0.06, hot_hand_score * 0.06)
        if bool(payload.profit_recycle_active) and profit_recycle_boost > 0.0:
            multiplier *= 1.0 + min(0.05, profit_recycle_boost * 0.30)
        if bool(payload.soft_trade_budget_enabled):
            multiplier *= 1.0 + min(0.08, max(0.0, signal_alignment) * 0.10)
        if bool(payload.funded_account_mode):
            multiplier *= 1.02
        if spread_elevated:
            multiplier *= 0.92
        if str(daily_state).upper() == "DAILY_CAUTION":
            multiplier *= 0.96
        elif str(daily_state).upper() == "DAILY_DEFENSIVE":
            multiplier *= 0.90
        elif str(daily_state).upper() == "DAILY_HARD_STOP":
            multiplier *= 0.0
        if str(payload.execution_quality_state or "").upper() == "DEGRADED" or bool(payload.degraded_mode_active):
            multiplier *= 0.75
        elif str(payload.execution_quality_state or "").upper() == "CAUTION":
            multiplier *= 0.90
        multiplier = clamp(multiplier, 0.45, 1.15)
        news_state = str(payload.news_state or "").upper()
        overflow_eligible = (
            not daily_trade_overflow_active
            and float(payload.current_max_risk_pct or 0.0) > float(payload.current_base_risk_pct or payload.requested_risk_pct)
            and quality_score >= 0.90
            and regime_confidence >= 0.75
            and execution_quality >= 0.75
            and news_state in {"NEWS_SAFE", "NEWS_CAUTION"}
            and float(payload.news_confidence or 0.0) >= 0.72
            and spread_quality >= 0.75
            and correlation_quality >= 0.85
            and exposure_quality >= 0.70
            and str(daily_state).upper() == "DAILY_NORMAL"
            and not bool(payload.degraded_mode_active)
        )
        explicit_base_band = float(payload.current_base_risk_pct or 0.0) > 0.0
        if overflow_eligible:
            allowed_cap = float(payload.current_max_risk_pct or hard_cap)
        elif explicit_base_band:
            allowed_cap = min(
                float(hard_cap),
                max(float(payload.current_base_risk_pct or 0.0), float(payload.requested_risk_pct or 0.0)),
            )
        else:
            allowed_cap = float(hard_cap)
        if (
            bool(payload.profit_recycle_active)
            and profit_recycle_boost >= 0.05
            and str(daily_state).upper() == "DAILY_NORMAL"
            and execution_quality >= 0.70
            and spread_quality >= 0.70
        ):
            allowed_cap = min(float(hard_cap), allowed_cap + min(0.0015, profit_recycle_boost * 0.01))
        if bool(payload.soft_trade_budget_enabled) and max(0.0, signal_alignment) >= 0.18 and execution_minute_quality >= 0.68:
            allowed_cap = min(float(hard_cap), allowed_cap + min(0.0015, max(0.0, signal_alignment) * 0.005))
        return multiplier, modifiers, overflow_eligible, max(0.0, allowed_cap)

    @staticmethod
    def _soft_dd_elite_allowed(payload: RiskInputs, spread_elevated: bool) -> bool:
        if spread_elevated:
            return False
        if str(payload.regime or "").upper() == "VOLATILE":
            return False
        return (
            float(payload.ai_probability) >= float(payload.soft_dd_probability_floor)
            and float(payload.expected_value_r) >= float(payload.soft_dd_expected_value_floor)
            and float(payload.confluence_score) >= float(payload.soft_dd_confluence_floor)
            and float(payload.trade_quality_score or 0.0) >= 0.85
            and float(payload.regime_confidence or 0.0) >= 0.65
            and str(payload.execution_quality_state or "GOOD").upper() != "DEGRADED"
        )

    @staticmethod
    def _overflow_trade_cap_allowed(payload: RiskInputs, *, capacity_state: dict[str, Any] | None = None) -> bool:
        capacity_state = capacity_state or {}
        base_target = max(1, int(payload.max_trades_per_day or 1))
        overflow_cap = max(
            int(capacity_state.get("allowed_daily_target", 0) or 0),
            int(payload.overflow_max_trades_per_day or 0),
            base_target,
        )
        hard_upper_limit = max(int(capacity_state.get("hard_upper_limit", 0) or 0), overflow_cap)
        if hard_upper_limit <= base_target:
            return False
        if int(payload.stats.trades_today) >= hard_upper_limit:
            return False
        quality_score = float(payload.trade_quality_score or 0.0)
        if quality_score < 0.54:
            return False
        if float(payload.regime_confidence or 0.0) < 0.46:
            return False
        if str(payload.execution_quality_state or "GOOD").upper() == "DEGRADED":
            return False
        if str(payload.news_state or "NEWS_SAFE").upper() not in {"NEWS_SAFE", "NEWS_CAUTION"}:
            return False
        if float(payload.news_confidence or 0.0) < 0.48:
            return False
        if float(payload.spread_quality_score or 0.0) < 0.50:
            return False
        if float(payload.correlation_multiplier or 1.0) < 0.62:
            return False
        if bool(payload.degraded_mode_active):
            return False
        if str(payload.news_state or "NEWS_SAFE").upper() == "NEWS_CAUTION" and float(payload.trade_quality_score or 0.0) < 0.60:
            return False
        if float(payload.recent_win_rate or 0.0) < 0.42 and float(payload.recent_expectancy_r or 0.0) <= 0.0:
            return False
        if float(payload.recent_expectancy_r or 0.0) < -0.12:
            return False
        if float(payload.total_open_risk_pct or 0.0) > 0.065:
            return False
        if float(payload.stats.daily_dd_pct_live or 0.0) >= float(payload.hard_daily_dd_pct or 0.07):
            return False
        if bool(payload.soft_trade_budget_enabled):
            signal_alignment_score = clamp(float(capacity_state.get("signal_alignment_score", 0.0) or 0.0), -1.0, 1.0)
            aggression_lane_multiplier = clamp(
                float(capacity_state.get("aggression_lane_multiplier", payload.aggression_lane_multiplier or 1.0) or 1.0),
                0.75,
                1.60,
            )
            execution_minute_quality = clamp(
                float(capacity_state.get("execution_minute_quality_score", payload.execution_minute_quality_score or 0.70) or 0.70),
                0.0,
                1.0,
            )
            if signal_alignment_score <= -0.10:
                return False
            if execution_minute_quality < 0.58:
                return False
            if aggression_lane_multiplier >= 1.04 and quality_score >= 0.60:
                return True
            if signal_alignment_score >= 0.10 and quality_score >= 0.58:
                return True
            if (
                bool(payload.funded_account_mode)
                or bool(payload.winning_streak_mode_active)
                or int(payload.stats.winning_streak or 0) >= 2
            ) and quality_score >= 0.58 and execution_minute_quality >= 0.60:
                return True
        if str(payload.execution_quality_state or "GOOD").upper() == "GOOD" and bool(capacity_state.get("cluster_mode_active", False)):
            return True
        if bool(payload.winning_streak_mode_active or int(payload.stats.winning_streak or 0) >= 2) and quality_score >= 0.62:
            return True
        if float(capacity_state.get("lane_strength_multiplier", payload.lane_strength_multiplier or 1.0)) >= 1.02 and quality_score >= 0.62:
            return True
        if int(payload.stats.trades_today) >= overflow_cap and quality_score < 0.70:
            return False
        return True

    @staticmethod
    def _resolve_point_size(payload: RiskInputs) -> tuple[float, str]:
        point = float(payload.symbol_point or 0.0)
        if point > 0:
            return point, "symbol_point"
        tick = float(payload.symbol_tick_size or 0.0)
        if tick > 0:
            return tick, "symbol_tick_size"
        return 0.01, "fallback_default"

    @staticmethod
    def _floor_to_step(value: float, step: float) -> float:
        if step <= 0:
            return max(0.0, value)
        steps = int(value / step)
        return max(0.0, steps * step)

    @staticmethod
    def _resolve_per_lot_risk(payload: RiskInputs, stop_distance: float) -> tuple[float, str]:
        tick_size = float(payload.symbol_tick_size or 0.0)
        tick_value = float(payload.symbol_tick_value or 0.0)
        if tick_size > 0 and tick_value > 0:
            return max(0.0, (stop_distance / tick_size) * tick_value), "tick_size_tick_value"
        return max(0.0, stop_distance * max(payload.contract_size, 1.0)), "contract_size_fallback"

    @staticmethod
    def _is_xau_grid_setup(payload: RiskInputs) -> bool:
        return payload.symbol.upper().startswith(("XAUUSD", "GOLD")) and payload.setup.upper().startswith("XAUUSD_M5_GRID_SCALPER")

    @classmethod
    def _allow_low_equity_xau_grid_bootstrap(
        cls,
        payload: RiskInputs,
        *,
        candidate_mc: float,
        spread_reference: float,
    ) -> bool:
        if not cls._is_xau_grid_setup(payload):
            return False
        session_name = str(dominant_session_name(payload.current_time) or "").upper()
        prime_session = session_name in {"LONDON", "OVERLAP", "NEW_YORK"}
        asia_session = session_name in {"TOKYO", "SYDNEY"}
        if not prime_session and not asia_session:
            return False
        strong_prime_candidate = bool(
            prime_session
            and float(payload.trade_quality_score or 0.0) >= 0.80
            and float(payload.execution_quality_score or 0.0) >= 0.88
            and float(payload.session_quality_score or 0.0) >= 0.84
            and float(payload.ai_probability or 0.0) >= 0.78
            and float(payload.expected_value_r or 0.0) >= 1.05
            and float(payload.confluence_score or 0.0) >= 4.4
        )
        mc_floor = clamp(float(payload.low_equity_monte_carlo_floor or 0.88), 0.5, 0.99)
        if prime_session:
            mc_override_floor = clamp(max(0.72, mc_floor - 0.14), 0.70, mc_floor)
            trade_quality_floor = 0.42
            execution_quality_floor = 0.46
            execution_minute_floor = 0.44
            session_quality_floor = 0.42
            spread_multiplier_cap = 1.78
            adverse_lead_lag_floor = -0.30
            microstructure_floor = 0.36
            if bool(payload.session_native_pair) and float(payload.session_priority_multiplier or 1.0) >= 1.15:
                spread_multiplier_cap = 1.95
            if strong_prime_candidate:
                mc_override_floor = clamp(max(0.70, mc_floor - 0.18), 0.68, mc_floor)
                trade_quality_floor = min(trade_quality_floor, 0.40)
                execution_quality_floor = min(execution_quality_floor, 0.44)
                execution_minute_floor = min(execution_minute_floor, 0.40)
                session_quality_floor = min(session_quality_floor, 0.40)
                spread_multiplier_cap = max(spread_multiplier_cap, 2.25)
                adverse_lead_lag_floor = min(adverse_lead_lag_floor, -0.36)
                microstructure_floor = min(microstructure_floor, 0.28)
        else:
            # Tokyo/Sydney bootstrap should only bypass the low-equity MC floor
            # when the XAU grid candidate is already very clean. This keeps the
            # live Asia probe active without broadly weakening the bootstrap guard.
            mc_override_floor = clamp(max(0.72, mc_floor - 0.12), 0.70, mc_floor)
            trade_quality_floor = 0.54
            execution_quality_floor = 0.58
            execution_minute_floor = 0.56
            session_quality_floor = 0.52
            spread_multiplier_cap = 1.22
            adverse_lead_lag_floor = -0.12
            microstructure_floor = 0.44
        if float(candidate_mc) < mc_override_floor:
            return False
        if strong_prime_candidate and str(payload.news_state or "").upper() not in {"NEWS_BLOCK", "NEWS_UNSAFE"}:
            return True
        if float(payload.trade_quality_score or 0.0) < trade_quality_floor:
            return False
        if float(payload.execution_quality_score or 0.0) < execution_quality_floor:
            return False
        if float(payload.execution_minute_quality_score or 0.0) < execution_minute_floor:
            return False
        if float(payload.session_quality_score or 0.0) < session_quality_floor:
            return False
        if (
            float(payload.microstructure_composite_score or 0.5) < microstructure_floor
            and float(payload.lead_lag_alignment_score or 0.0) < adverse_lead_lag_floor
        ):
            return False
        if str(payload.news_state or "").upper() in {"NEWS_BLOCK", "NEWS_UNSAFE"}:
            return False
        if spread_reference > 0.0 and float(payload.spread_points or 0.0) > (spread_reference * spread_multiplier_cap):
            return False
        return True

    @staticmethod
    def _allow_low_equity_attack_bootstrap(
        payload: RiskInputs,
        *,
        candidate_mc: float,
        spread_reference: float,
    ) -> bool:
        symbol_key = str(payload.symbol or "").upper()
        if symbol_key.startswith("BTCUSD") and is_weekend_market_mode(payload.current_time):
            if str(payload.news_state or "").upper() in {"NEWS_BLOCK", "NEWS_UNSAFE"}:
                return False
            if float(payload.ai_probability or 0.0) < 0.54:
                return False
            if float(payload.expected_value_r or 0.0) < 0.25:
                return False
            if float(payload.confluence_score or 0.0) < 2.40:
                return False
            if float(payload.trade_quality_score or 0.0) > 0.0 and float(payload.trade_quality_score or 0.0) < 0.42:
                return False
            if float(candidate_mc or 0.0) > 0.0 and float(candidate_mc or 0.0) < 0.48:
                return False
            if float(payload.lead_lag_alignment_score or 0.0) < -0.45:
                return False
            if float(payload.microstructure_composite_score or 0.5) < 0.08:
                return False
            if spread_reference > 0.0 and float(payload.spread_points or 0.0) > (spread_reference * 60.0):
                return False
            return True
        if not symbol_key.startswith(
            (
                "USDJPY",
                "BTCUSD",
                "AUDJPY",
                "NZDJPY",
                "AUDNZD",
                "EURUSD",
                "GBPUSD",
                "EURJPY",
                "GBPJPY",
                "NAS100",
                "USTEC",
                "US100",
                "XAGUSD",
                "USOIL",
            )
        ):
            return False
        if str(payload.news_state or "").upper() in {"NEWS_BLOCK", "NEWS_UNSAFE"}:
            return False
        if not bool(payload.session_native_pair) and float(payload.session_priority_multiplier or 1.0) < 1.02:
            return False
        quality_floor = 0.80
        execution_floor = 0.72
        execution_minute_floor = 0.66
        session_quality_floor = 0.64
        expected_value_floor = 0.70
        confluence_floor = 3.8
        candidate_mc_floor = 0.76
        strong_quality_floor = 0.78
        strong_execution_floor = 0.76
        strong_session_floor = 0.68
        strong_expected_value_floor = 0.82
        strong_confluence_floor = 3.45
        strong_candidate_mc_floor = 0.68
        if symbol_key.startswith(("EURUSD", "GBPUSD")):
            quality_floor = 0.72
            execution_floor = 0.64
            execution_minute_floor = 0.60
            session_quality_floor = 0.58
            expected_value_floor = 0.52
            confluence_floor = 3.25
            candidate_mc_floor = 0.72
            strong_quality_floor = 0.74
            strong_execution_floor = 0.66
            strong_session_floor = 0.60
            strong_expected_value_floor = 0.64
            strong_confluence_floor = 3.25
            strong_candidate_mc_floor = 0.68
        elif symbol_key.startswith(("NAS100", "USTEC", "US100")):
            quality_floor = 0.76
            execution_floor = 0.68
            execution_minute_floor = 0.64
            session_quality_floor = 0.60
            expected_value_floor = 0.58
            confluence_floor = 3.20
            candidate_mc_floor = 0.72
            strong_quality_floor = 0.78
            strong_execution_floor = 0.70
            strong_session_floor = 0.66
            strong_expected_value_floor = 0.78
            strong_confluence_floor = 3.60
            strong_candidate_mc_floor = 0.70
        elif symbol_key.startswith(("XAGUSD", "USOIL")):
            quality_floor = 0.74
            execution_floor = 0.66
            execution_minute_floor = 0.62
            session_quality_floor = 0.60
            expected_value_floor = 0.54
            confluence_floor = 3.10
            candidate_mc_floor = 0.72
            strong_quality_floor = 0.76
            strong_execution_floor = 0.68
            strong_session_floor = 0.62
            strong_expected_value_floor = 0.70
            strong_confluence_floor = 3.30
            strong_candidate_mc_floor = 0.69
        elif symbol_key.startswith(("EURJPY", "GBPJPY")):
            quality_floor = 0.78
            execution_floor = 0.72
            execution_minute_floor = 0.64
            session_quality_floor = 0.62
            expected_value_floor = 0.62
            confluence_floor = 3.4
            candidate_mc_floor = 0.72
            strong_quality_floor = 0.78
            strong_execution_floor = 0.72
            strong_session_floor = 0.64
            strong_expected_value_floor = 0.72
            strong_confluence_floor = 3.40
            strong_candidate_mc_floor = 0.69
        strong_attack_candidate = bool(
            float(payload.trade_quality_score or 0.0) >= strong_quality_floor
            and float(payload.execution_quality_score or 0.0) >= strong_execution_floor
            and float(payload.session_quality_score or 0.0) >= strong_session_floor
            and float(payload.ai_probability or 0.0) >= 0.78
            and float(payload.expected_value_r or 0.0) >= max(expected_value_floor, strong_expected_value_floor)
            and float(payload.confluence_score or 0.0) >= max(confluence_floor, strong_confluence_floor)
            and float(candidate_mc or 0.0) >= max(candidate_mc_floor - 0.08, strong_candidate_mc_floor)
        )
        if strong_attack_candidate:
            spread_multiplier_cap = 1.42
            if symbol_key.startswith(("USDJPY", "EURJPY", "GBPJPY")):
                spread_multiplier_cap = 1.95
            elif symbol_key.startswith(("EURUSD", "GBPUSD")):
                spread_multiplier_cap = 1.65
            elif symbol_key.startswith(("NAS100", "USTEC", "US100")):
                spread_multiplier_cap = 4.90
            elif symbol_key.startswith(("XAGUSD", "USOIL")):
                spread_multiplier_cap = 2.60
            if spread_reference > 0.0 and float(payload.spread_points or 0.0) > (
                spread_reference
                * spread_multiplier_cap
            ):
                return False
            return True
        if float(payload.trade_quality_score or 0.0) < quality_floor:
            return False
        if float(payload.execution_quality_score or 0.0) < execution_floor:
            return False
        if float(payload.execution_minute_quality_score or 0.0) < execution_minute_floor:
            return False
        if float(payload.session_quality_score or 0.0) < session_quality_floor:
            return False
        if float(payload.expected_value_r or 0.0) < expected_value_floor:
            return False
        if float(payload.confluence_score or 0.0) < confluence_floor:
            return False
        if float(candidate_mc) < candidate_mc_floor:
            return False
        if float(payload.lead_lag_alignment_score or 0.0) < -0.18:
            return False
        if (
            float(payload.microstructure_composite_score or 0.5) < 0.40
            and symbol_key.startswith(("BTCUSD", "USDJPY"))
        ):
            return False
        spread_multiplier_cap = (
            1.48
            if symbol_key.startswith(("NAS100", "USTEC", "US100"))
            else 1.40
            if symbol_key.startswith(("XAGUSD", "USOIL"))
            else 1.42
            if symbol_key.startswith(("BTCUSD", "USDJPY"))
            else 1.35
            if symbol_key.startswith(("EURUSD", "GBPUSD"))
            else 1.30
        )
        if spread_reference > 0.0 and float(payload.spread_points or 0.0) > (spread_reference * spread_multiplier_cap):
            return False
        return True
