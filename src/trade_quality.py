from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.utils import clamp


@dataclass(frozen=True)
class ExecutionQuality:
    state: str
    score: float
    spread_anomaly: bool
    slippage_anomaly: bool
    bridge_latency_alert: bool
    stale_idea_alert: bool
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TradeQuality:
    score: float
    band: str
    legacy_band: str
    elite: bool
    acceptable: bool
    should_skip: bool
    overflow_eligible: bool
    size_multiplier: float
    lane_name: str
    session_priority_profile: str = "GLOBAL"
    session_priority_multiplier: float = 1.0
    session_native_pair: bool = False
    lane_session_priority: str = "NEUTRAL"
    pair_priority_rank_in_session: int = 99
    components: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SessionPriorityContext:
    session_priority_profile: str
    lane_session_priority: str
    session_priority_multiplier: float
    session_native_pair: bool
    pair_priority_rank_in_session: int
    quality_floor_edge: float
    lane_budget_share: float
    native_override_delta: float
    native_override_band_delta: float


@dataclass(frozen=True)
class SessionPriorityDecision:
    allowed: bool
    exceptional_override_used: bool
    exceptional_override_reason: str
    why_non_native_pair_won: str
    why_native_pair_lost_priority: str


ASIA_PRIMARY_PAIRS = {"AUDJPY", "NZDJPY", "AUDNZD"}
ASIA_SECONDARY_PAIRS = {"USDJPY", "EURJPY", "GBPJPY"}
LONDON_PRIMARY_PAIRS = {"EURUSD", "GBPUSD", "USDJPY", "EURJPY", "GBPJPY"}
NEW_YORK_PRIMARY_PAIRS = {"EURUSD", "GBPUSD", "USDJPY", "NAS100", "BTCUSD", "USOIL"}
XAU_GRID_LANES = {
    "XAU_M5_GRID",
    "XAU_LONDON_ATTACK",
    "XAU_LONDON_RECLAIM",
    "XAU_LONDON_BREAKOUT",
    "XAU_LONDON_REENTRY",
    "XAU_OVERLAP_ATTACK",
    "XAU_OVERLAP_RECLAIM",
    "XAU_OVERLAP_BREAKOUT",
    "XAU_OVERLAP_REENTRY",
    "XAU_NEW_YORK_ATTACK",
    "XAU_NEW_YORK_RECLAIM",
    "XAU_NEW_YORK_BREAKOUT",
    "XAU_NEW_YORK_REENTRY",
}


def _normalize_symbol_key(value: str) -> str:
    normalized = "".join(char for char in str(value or "").upper() if char.isalnum())
    if normalized.startswith("GOLD") or normalized.startswith("XAUUSD"):
        return "XAUUSD"
    if normalized.startswith("BTCUSD") or normalized.startswith("BTCUSDT") or normalized.startswith("XBTUSD"):
        return "BTCUSD"
    if normalized.startswith(("NAS100", "US100", "NASDAQ", "USTEC", "NAS", "NQ")):
        return "NAS100"
    if normalized.startswith(("USOIL", "XTIUSD", "OILUSD", "WTI", "CL", "OIL", "USO")):
        return "USOIL"
    for core in (
        "EURUSD",
        "GBPUSD",
        "USDJPY",
        "EURJPY",
        "GBPJPY",
        "AUDJPY",
        "NZDJPY",
        "AUDNZD",
    ):
        if normalized.startswith(core):
            return core
    return normalized


def normalize_strategy_family(value: str) -> str:
    family = str(value or "").strip().upper()
    if family in {"GRID", "TREND", "CRYPTO MOMENTUM", "RANGE/REVERSION"}:
        return family
    if family in {"RANGE", "REVERSION", "FAKEOUT"}:
        return "RANGE/REVERSION"
    if family in {"SCALP", "DAYTRADE", "BREAKOUT", "MOMENTUM"}:
        return "TREND"
    return "TREND"


def is_xau_grid_lane(lane_name: str) -> bool:
    return str(lane_name or "").strip().upper() in XAU_GRID_LANES


def xau_primary_session_for_lane(lane_name: str) -> str:
    lane_key = str(lane_name or "").strip().upper()
    if lane_key.startswith("XAU_LONDON_"):
        return "LONDON"
    if lane_key.startswith("XAU_OVERLAP_"):
        return "OVERLAP"
    if lane_key.startswith("XAU_NEW_YORK_"):
        return "NEW_YORK"
    return ""


def xau_attack_lane_category(lane_name: str) -> str:
    lane_key = str(lane_name or "").strip().upper()
    if lane_key.endswith("_RECLAIM"):
        return "RECLAIM"
    if lane_key.endswith("_BREAKOUT"):
        return "BREAKOUT"
    if lane_key.endswith("_REENTRY"):
        return "REENTRY"
    if lane_key.endswith("_ATTACK"):
        return "ATTACK"
    return "GRID"


def xau_grid_lane_for_session(session_name: str, *, category: str = "ATTACK") -> str:
    session_key = str(session_name or "").strip().upper()
    normalized_category = str(category or "ATTACK").strip().upper()
    if normalized_category not in {"ATTACK", "RECLAIM", "BREAKOUT", "REENTRY"}:
        normalized_category = "ATTACK"
    prefix = {
        "LONDON": "XAU_LONDON",
        "OVERLAP": "XAU_OVERLAP",
        "NEW_YORK": "XAU_NEW_YORK",
    }.get(session_key)
    if not prefix:
        return "XAU_M5_GRID"
    return f"{prefix}_{normalized_category}"


def quality_band(score: float) -> str:
    value = clamp(float(score), 0.0, 1.0)
    if value >= 0.85:
        return "elite"
    if value >= 0.70:
        return "strong"
    if value >= 0.60:
        return "acceptable"
    return "weak"


def quality_band_detail(score: float) -> str:
    value = clamp(float(score), 0.0, 1.0)
    if value >= 0.90:
        return "A+"
    if value >= 0.84:
        return "A"
    if value >= 0.78:
        return "A-"
    if value >= 0.70:
        return "B+"
    if value >= 0.58:
        return "B"
    if value >= 0.48:
        return "C"
    return "REJECT"


def quality_band_rank(band: str) -> int:
    normalized = str(band or "").strip().upper()
    return {
        "A+": 6,
        "A": 5,
        "A-": 4,
        "B+": 3,
        "B": 2,
        "C": 1,
    }.get(normalized, 0)


def quality_size_multiplier(band: str) -> float:
    normalized = str(band or "").strip().upper()
    return {
        "A+": 1.75,
        "A": 1.40,
        "A-": 1.25,
        "B+": 0.95,
        "B": 0.65,
        "C": 0.30,
    }.get(normalized, 0.0)


def delta_proxy_score(
    *,
    side: str,
    body_efficiency: float,
    short_return: float,
    range_position: float,
    volume_ratio: float,
    upper_wick_ratio: float = 0.0,
    lower_wick_ratio: float = 0.0,
) -> float:
    side_key = str(side or "").upper()
    side_sign = 1.0 if side_key == "BUY" else -1.0 if side_key == "SELL" else 0.0
    if side_sign == 0.0:
        return 0.0
    if float(short_return) > 0.0:
        return_sign = 1.0
    elif float(short_return) < 0.0:
        return_sign = -1.0
    else:
        return_sign = 0.0
    directional_body = clamp(return_sign * float(body_efficiency) * side_sign, -1.0, 1.0)
    wick_pressure = clamp((float(lower_wick_ratio) - float(upper_wick_ratio)) * side_sign, -1.0, 1.0)
    volume_bias = clamp((float(volume_ratio) - 1.0) / 0.70, -1.0, 1.0)
    return clamp(
        (0.45 * directional_body)
        + (0.30 * wick_pressure)
        + (0.25 * volume_bias),
        -1.0,
        1.0,
    )


def session_loosen_factor(
    *,
    session_name: str,
    symbol: str,
    weekend_mode: bool = False,
    candidate_scarcity: bool = False,
) -> float:
    session_key = str(session_name or "").upper()
    symbol_key = _normalize_symbol_key(symbol)
    factor = 1.0
    if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
        factor = 0.72
    elif symbol_key == "BTCUSD" and weekend_mode:
        factor = 0.64
    elif session_key in {"SYDNEY", "TOKYO"}:
        factor = 1.06
    if candidate_scarcity:
        factor -= 0.12
    return clamp(factor, 0.60, 1.12)


def compression_strategy_bias(strategy_key: str, compression_proxy_state: str) -> float:
    key = str(strategy_key or "").strip().upper()
    state = str(compression_proxy_state or "").strip().upper()
    if state != "COMPRESSION":
        return 0.0
    if any(token in key for token in ("RETEST", "SWEEP", "ROTATION", "REVERSION", "TRAP", "VWAP")):
        return 0.10
    if any(token in key for token in ("CONTINUATION", "TREND_SCALP", "BREAKOUT", "IMPULSE", "EXPANSION")):
        return -0.10
    return 0.0


def winner_promotion_bonus(
    *,
    symbol: str,
    strategy_key: str,
    regime_state: str,
    session_name: str = "",
    weekend_mode: bool = False,
) -> float:
    symbol_key = _normalize_symbol_key(symbol)
    key = str(strategy_key or "").strip().upper()
    regime_key = runtime_regime_state(regime_state)
    session_key = str(session_name or "").upper()
    bonus = 0.0
    if "PRICE_ACTION_CONTINUATION" in key:
        if symbol_key == "BTCUSD":
            bonus += 0.18 if weekend_mode else 0.16
        else:
            bonus += 0.12
    if "VOLATILE_RETEST" in key:
        bonus += 0.12 if symbol_key != "BTCUSD" else (0.18 if weekend_mode else 0.16)
    if symbol_key == "BTCUSD" and "RANGE_EXPANSION" in key:
        bonus += 0.10 if weekend_mode else 0.06
    if "RANGE_ROTATION" in key:
        bonus += 0.14
        if session_key in {"SYDNEY", "TOKYO"}:
            bonus += 0.04
    if symbol_key == "NAS100" and any(token in key for token in ("LIQUIDITY_SWEEP", "VWAP_TREND")):
        bonus += 0.18 if regime_key in {"TRENDING", "BREAKOUT_EXPANSION"} else 0.14
    if symbol_key == "GBPUSD" and key == "GBPUSD_LONDON_EXPANSION_BREAKOUT" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
        bonus += 0.16 if regime_key in {"TRENDING", "BREAKOUT_EXPANSION"} else 0.10
    if symbol_key == "GBPUSD" and key == "GBPUSD_LONDON_EXPANSION_BREAKOUT" and session_key == "LONDON" and regime_key == "LOW_LIQUIDITY_CHOP":
        bonus -= 0.10
    if symbol_key == "GBPUSD" and key == "GBPUSD_TREND_PULLBACK_RIDE" and session_key in {"OVERLAP", "NEW_YORK"} and regime_key == "LOW_LIQUIDITY_CHOP":
        bonus -= 0.04
    if symbol_key == "BTCUSD" and key == "BTCUSD_PRICE_ACTION_CONTINUATION" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}:
        bonus += 0.18 if weekend_mode else 0.08
    if symbol_key == "BTCUSD" and key == "BTCUSD_PRICE_ACTION_CONTINUATION" and session_key in {"TOKYO", "SYDNEY"} and regime_key in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION", "RANGING"}:
        bonus -= 0.18 if weekend_mode else 0.12
    if symbol_key == "BTCUSD" and key == "BTCUSD_PRICE_ACTION_CONTINUATION" and session_key in {"OVERLAP", "NEW_YORK"} and regime_key in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        bonus -= 0.14
    if symbol_key == "BTCUSD" and key == "BTCUSD_VOLATILE_RETEST" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}:
        bonus += 0.16 if weekend_mode else 0.06
    if symbol_key == "BTCUSD" and key == "BTCUSD_VOLATILE_RETEST" and session_key in {"TOKYO", "SYDNEY"} and regime_key in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION", "RANGING"}:
        bonus -= 0.24 if weekend_mode else 0.12
    if symbol_key == "BTCUSD" and key == "BTCUSD_VOLATILE_RETEST" and session_key == "NEW_YORK" and regime_key == "LOW_LIQUIDITY_CHOP":
        bonus -= 0.12
    if symbol_key == "BTCUSD" and key == "BTCUSD_RANGE_EXPANSION" and weekend_mode and session_key in {"SYDNEY", "TOKYO"} and regime_key in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        bonus -= 0.26
    if symbol_key == "BTCUSD" and key == "BTCUSD_RANGE_EXPANSION" and weekend_mode and session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and regime_key in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        bonus -= 0.16
    if symbol_key == "BTCUSD" and key == "BTCUSD_TREND_SCALP" and weekend_mode and session_key == "SYDNEY" and regime_key == "MEAN_REVERSION":
        bonus -= 0.24
    if symbol_key == "BTCUSD" and key == "BTCUSD_PRICE_ACTION_CONTINUATION" and weekend_mode and session_key == "LONDON" and regime_key == "MEAN_REVERSION":
        bonus -= 0.18
    if symbol_key == "EURUSD" and key == "EURUSD_VWAP_PULLBACK" and session_key == "NEW_YORK" and regime_key == "LOW_LIQUIDITY_CHOP":
        bonus += 0.12
    if symbol_key == "EURUSD" and key == "EURUSD_LONDON_BREAKOUT" and session_key in {"OVERLAP", "NEW_YORK"} and regime_key == "LOW_LIQUIDITY_CHOP":
        bonus -= 0.06
    if symbol_key == "EURJPY" and key == "EURJPY_MOMENTUM_IMPULSE" and session_key == "NEW_YORK" and regime_key in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        bonus += 0.16
    if symbol_key == "EURJPY" and key == "EURJPY_SESSION_PULLBACK_CONTINUATION" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and regime_key == "LOW_LIQUIDITY_CHOP":
        bonus -= 0.16
    if symbol_key == "GBPJPY" and key == "GBPJPY_LIQUIDITY_SWEEP_REVERSAL" and regime_key == "MEAN_REVERSION":
        bonus += 0.12
    if symbol_key == "GBPJPY" and key in {"GBPJPY_MOMENTUM_IMPULSE", "GBPJPY_SESSION_PULLBACK_CONTINUATION"} and regime_key == "LOW_LIQUIDITY_CHOP":
        bonus -= 0.18
    if symbol_key == "AUDJPY" and key == "AUDJPY_LONDON_CARRY_TREND" and session_key == "NEW_YORK" and regime_key == "LOW_LIQUIDITY_CHOP":
        bonus += 0.12
    if symbol_key == "AUDJPY" and key == "AUDJPY_LONDON_CARRY_TREND" and session_key == "NEW_YORK" and regime_key in {"MEAN_REVERSION", "RANGING"}:
        bonus -= 0.06
    if symbol_key == "AUDJPY" and key == "AUDJPY_LIQUIDITY_SWEEP_REVERSAL" and session_key == "OVERLAP" and regime_key == "MEAN_REVERSION":
        bonus += 0.16
    if symbol_key == "AUDJPY" and key == "AUDJPY_LIQUIDITY_SWEEP_REVERSAL" and session_key == "SYDNEY" and regime_key == "MEAN_REVERSION":
        bonus += 0.06
    if symbol_key == "AUDJPY" and key == "AUDJPY_ATR_COMPRESSION_BREAKOUT" and session_key == "NEW_YORK" and regime_key == "LOW_LIQUIDITY_CHOP":
        bonus += 0.14
    if symbol_key == "AUDJPY" and key == "AUDJPY_ATR_COMPRESSION_BREAKOUT" and session_key == "OVERLAP" and regime_key == "MEAN_REVERSION":
        bonus -= 0.08
    if symbol_key == "NZDJPY" and key == "NZDJPY_LIQUIDITY_TRAP_REVERSAL" and session_key == "SYDNEY" and regime_key == "MEAN_REVERSION":
        bonus += 0.14
    if symbol_key == "NZDJPY" and key == "NZDJPY_LIQUIDITY_TRAP_REVERSAL" and session_key in {"TOKYO", "OVERLAP"} and regime_key == "MEAN_REVERSION":
        bonus -= 0.08
    if symbol_key == "NZDJPY" and key == "NZDJPY_SESSION_RANGE_EXPANSION" and session_key == "LONDON" and regime_key == "LOW_LIQUIDITY_CHOP":
        bonus -= 0.06
    if symbol_key == "XAUUSD" and "GRID" in key and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
        if regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}:
            bonus += 0.24
        elif regime_key == "MEAN_REVERSION" and session_key == "LONDON":
            bonus += 0.08
        elif regime_key in {"LOW_LIQUIDITY_CHOP", "RANGING"}:
            bonus -= 0.12
    if symbol_key == "XAUUSD" and key == "XAUUSD_ADAPTIVE_M5_GRID" and session_key == "LONDON" and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}:
        bonus += 0.18
    if symbol_key == "XAUUSD" and key == "XAUUSD_ADAPTIVE_M5_GRID" and session_key == "LONDON" and regime_key == "MEAN_REVERSION":
        bonus += 0.10
    if symbol_key == "XAUUSD" and key == "XAUUSD_ADAPTIVE_M5_GRID" and session_key == "LONDON" and regime_key == "RANGING":
        bonus -= 0.12
    if symbol_key == "XAUUSD" and key == "XAUUSD_ADAPTIVE_M5_GRID" and session_key in {"OVERLAP", "NEW_YORK"} and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}:
        bonus += 0.18
    if symbol_key == "XAUUSD" and key == "XAUUSD_ADAPTIVE_M5_GRID" and session_key in {"OVERLAP", "NEW_YORK"} and regime_key in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"}:
        bonus -= 0.02
    if symbol_key == "XAUUSD" and key == "XAUUSD_ADAPTIVE_M5_GRID" and session_key in {"OVERLAP", "NEW_YORK"} and regime_key == "RANGING":
        bonus -= 0.12
    if symbol_key == "XAUUSD" and key == "XAUUSD_LONDON_LIQUIDITY_SWEEP" and session_key == "LONDON" and regime_key == "MEAN_REVERSION":
        bonus += 0.18
    if symbol_key == "XAUUSD" and key == "XAUUSD_LONDON_LIQUIDITY_SWEEP" and session_key == "NEW_YORK" and regime_key == "MEAN_REVERSION":
        bonus += 0.12
    if symbol_key == "USOIL" and key == "USOIL_LONDON_TREND_EXPANSION" and session_key == "SYDNEY" and regime_key == "MEAN_REVERSION":
        bonus += 0.10
    if symbol_key == "USOIL" and key == "USOIL_LONDON_TREND_EXPANSION" and (
        (session_key == "OVERLAP" and regime_key in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"})
        or (session_key == "NEW_YORK" and regime_key == "LOW_LIQUIDITY_CHOP")
        or (session_key == "SYDNEY" and regime_key == "TRENDING")
    ):
        bonus -= 0.14
    if symbol_key == "BTCUSD" and "TREND_SCALP" in key:
        if weekend_mode and regime_key == "BREAKOUT_EXPANSION":
            bonus -= 0.10
        elif session_key == "TOKYO":
            bonus -= 0.12
    if symbol_key == "BTCUSD" and weekend_mode and any(
        token in key for token in ("PRICE_ACTION_CONTINUATION", "VOLATILE_RETEST")
    ):
        bonus = max(bonus, 0.50)
    if symbol_key == "NAS100" and "MOMENTUM_IMPULSE" in key and regime_key == "LOW_LIQUIDITY_CHOP":
        bonus -= 0.12
    if symbol_key == "NAS100" and "MOMENTUM_IMPULSE" in key and session_key == "SYDNEY":
        bonus -= 0.18
    if symbol_key == "NAS100" and "MOMENTUM_IMPULSE" in key and regime_key == "TRENDING" and session_key == "TOKYO":
        bonus -= 0.08
    if symbol_key == "NAS100" and key == "NAS100_VWAP_TREND_STRATEGY" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and regime_key in {"TRENDING", "BREAKOUT_EXPANSION"}:
        bonus += 0.18
    if symbol_key == "NAS100" and key == "NAS100_VWAP_TREND_STRATEGY" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
        bonus += 0.12
    if symbol_key == "NAS100" and key == "NAS100_OPENING_DRIVE_BREAKOUT" and session_key in {"OVERLAP", "NEW_YORK"} and regime_key == "BREAKOUT_EXPANSION":
        bonus += 0.16
    if symbol_key == "NAS100" and key == "NAS100_LIQUIDITY_SWEEP_REVERSAL" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and regime_key in {"RANGING", "MEAN_REVERSION"}:
        bonus += 0.12
    return clamp(bonus, -0.34, 0.50)


def quality_tier_from_scores(
    *,
    structure_cleanliness: float,
    regime_fit: float,
    execution_quality_fit: float,
    high_liquidity: bool,
    throughput_recovery_active: bool = False,
) -> str:
    loosen = 0.03 if throughput_recovery_active else 0.0
    structure = clamp(float(structure_cleanliness) + loosen, 0.0, 1.0)
    regime = clamp(float(regime_fit) + (0.02 if throughput_recovery_active else 0.0), 0.0, 1.0)
    execution = clamp(float(execution_quality_fit), 0.0, 1.0)
    if high_liquidity and structure > 0.80 - loosen and regime > 0.75 - loosen and execution > 0.70:
        return "A+"
    if structure > 0.72 - loosen and regime > 0.66 - loosen and execution > 0.62:
        return "A"
    if structure > 0.56 - loosen and regime > 0.52 - loosen and execution > 0.52:
        return "B"
    return "C"


def quality_tier_size_multiplier(
    *,
    quality_tier: str,
    strategy_score: float,
    b_tier_min: float = 0.70,
    b_tier_max: float = 0.90,
) -> float:
    tier = str(quality_tier or "").upper()
    score = clamp(float(strategy_score), 0.0, 1.0)
    if tier == "A+":
        return 1.00
    if tier == "A":
        return clamp(0.90 + (0.10 * score), 0.90, 1.00)
    if tier == "B":
        return clamp(float(b_tier_min) + ((float(b_tier_max) - float(b_tier_min)) * score), float(b_tier_min), float(b_tier_max))
    return 0.58


def infer_trade_lane(*, symbol: str, setup: str = "", setup_family: str = "", session_name: str = "") -> str:
    symbol_key = _normalize_symbol_key(symbol)
    setup_key = str(setup or "").upper()
    family = normalize_strategy_family(setup_family)
    session_key = str(session_name or "").upper()
    if symbol_key == "XAUUSD" and (
        setup_key.startswith("XAUUSD_M5_GRID_SCALPER")
        or "XAUUSD_ADAPTIVE_M5_GRID" in setup_key
    ):
        return xau_grid_lane_for_session(session_key, category="ATTACK")
    if symbol_key == "XAUUSD":
        return "XAU_DIRECTIONAL"
    if symbol_key == "BTCUSD":
        return "BTC_MOMENTUM"
    if symbol_key in {"NAS100"}:
        return "INDEX_DAYTRADE"
    if symbol_key in {"USOIL"}:
        return "OIL_EVENT_MOMENTUM"
    if symbol_key == "AUDNZD":
        return "FX_DAYTRADE"
    if symbol_key in {"AUDJPY", "NZDJPY", "EURJPY", "GBPJPY"} and session_key in {"TOKYO", "SYDNEY"}:
        return "FX_SESSION_SCALP"
    if family == "TREND" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
        return "FX_DAYTRADE"
    return "FX_SESSION_SCALP"


def session_lane_budget_share(*, lane_name: str, session_name: str) -> float:
    lane_key = str(lane_name or "").upper()
    session_key = str(session_name or "").upper()
    if is_xau_grid_lane(lane_key):
        primary_session = xau_primary_session_for_lane(lane_key)
        category = xau_attack_lane_category(lane_key)
        if session_key in {"TOKYO", "SYDNEY"}:
            base = 0.14 if lane_key == "XAU_M5_GRID" else 0.12
        elif session_key == "LONDON":
            base = 0.23 + (0.10 if primary_session == "LONDON" else 0.03)
        elif session_key == "OVERLAP":
            base = 0.22 + (0.10 if primary_session == "OVERLAP" else 0.04)
        elif session_key == "NEW_YORK":
            base = 0.21 + (0.10 if primary_session == "NEW_YORK" else 0.05)
        else:
            base = 0.16
        if category in {"RECLAIM", "BREAKOUT"}:
            base += 0.02
        elif category == "REENTRY":
            base += 0.03
        elif category == "ATTACK":
            base += 0.01
        return clamp(base, 0.10, 0.38)
    mapping = {
        "TOKYO": {
            "FX_SESSION_SCALP": 0.40,
            "FX_DAYTRADE": 0.20,
            "BTC_MOMENTUM": 0.12,
            "INDEX_DAYTRADE": 0.08,
            "OIL_EVENT_MOMENTUM": 0.05,
            "XAU_DIRECTIONAL": 0.15,
        },
        "SYDNEY": {
            "FX_SESSION_SCALP": 0.40,
            "FX_DAYTRADE": 0.20,
            "BTC_MOMENTUM": 0.12,
            "INDEX_DAYTRADE": 0.08,
            "OIL_EVENT_MOMENTUM": 0.05,
            "XAU_DIRECTIONAL": 0.15,
        },
        "LONDON": {
            "XAU_DIRECTIONAL": 0.12,
            "FX_DAYTRADE": 0.33,
            "FX_SESSION_SCALP": 0.10,
            "INDEX_DAYTRADE": 0.07,
            "BTC_MOMENTUM": 0.04,
            "OIL_EVENT_MOMENTUM": 0.04,
        },
        "OVERLAP": {
            "XAU_DIRECTIONAL": 0.10,
            "FX_DAYTRADE": 0.27,
            "INDEX_DAYTRADE": 0.15,
            "BTC_MOMENTUM": 0.12,
            "OIL_EVENT_MOMENTUM": 0.07,
            "FX_SESSION_SCALP": 0.04,
        },
        "NEW_YORK": {
            "XAU_DIRECTIONAL": 0.10,
            "FX_DAYTRADE": 0.27,
            "INDEX_DAYTRADE": 0.15,
            "BTC_MOMENTUM": 0.12,
            "OIL_EVENT_MOMENTUM": 0.07,
            "FX_SESSION_SCALP": 0.04,
        },
    }
    return float(mapping.get(session_key, {}).get(lane_key, 0.10))


def session_priority_context(*, symbol: str, lane_name: str, session_name: str) -> SessionPriorityContext:
    symbol_key = _normalize_symbol_key(symbol)
    lane_key = str(lane_name or "").upper()
    session_key = str(session_name or "").upper()
    multiplier = 1.0
    native = False
    rank = 99
    profile = "GLOBAL"
    lane_priority = "NEUTRAL"
    quality_floor_edge = 0.0
    native_override_delta = 0.06
    native_override_band_delta = 0.03

    if session_key in {"TOKYO", "SYDNEY"}:
        profile = "ASIA_NATIVE"
        if symbol_key in ASIA_PRIMARY_PAIRS:
            multiplier = 1.14
            native = True
            rank = 1
            lane_priority = "PRIMARY"
            quality_floor_edge = 0.03
        elif symbol_key in ASIA_SECONDARY_PAIRS:
            multiplier = 1.05
            native = True
            rank = 2
            lane_priority = "SECONDARY"
            quality_floor_edge = 0.015
        elif is_xau_grid_lane(lane_key):
            multiplier = 0.98 if lane_key != "XAU_M5_GRID" else 0.94
            rank = 3
            lane_priority = "SECONDARY"
            quality_floor_edge = 0.01
        elif symbol_key in {"EURUSD", "GBPUSD", "XAUUSD", "NAS100", "BTCUSD", "USOIL"}:
            multiplier = 0.90
            rank = 4
            lane_priority = "OFF_SESSION"
        else:
            multiplier = 0.94
            rank = 3
            lane_priority = "NEUTRAL"
    elif session_key == "LONDON":
        profile = "LONDON_MAJOR"
        if is_xau_grid_lane(lane_key):
            primary_session = xau_primary_session_for_lane(lane_key)
            multiplier = 1.20 if primary_session == "LONDON" else 1.15
            native = True
            rank = 1
            lane_priority = "PRIMARY"
            quality_floor_edge = 0.035 if primary_session == "LONDON" else 0.028
        elif symbol_key in LONDON_PRIMARY_PAIRS:
            multiplier = 1.10
            native = True
            rank = 2
            lane_priority = "PRIMARY"
            quality_floor_edge = 0.015
        elif symbol_key in {"NAS100", "BTCUSD", "USOIL"}:
            multiplier = 1.02
            rank = 3
            lane_priority = "SECONDARY"
        elif symbol_key in ASIA_PRIMARY_PAIRS:
            multiplier = 0.94
            rank = 4
            lane_priority = "OFF_SESSION"
    elif session_key in {"OVERLAP", "NEW_YORK"}:
        profile = "NY_RISK_ON"
        if is_xau_grid_lane(lane_key):
            primary_session = xau_primary_session_for_lane(lane_key)
            multiplier = 1.20 if primary_session == session_key else 1.16
            native = True
            rank = 1
            lane_priority = "PRIMARY"
            quality_floor_edge = 0.035 if primary_session == session_key else 0.03
        elif symbol_key in NEW_YORK_PRIMARY_PAIRS:
            multiplier = 1.10
            native = True
            rank = 2
            lane_priority = "PRIMARY"
            quality_floor_edge = 0.015
        elif symbol_key in {"EURJPY", "GBPJPY"}:
            multiplier = 1.00
            rank = 3
            lane_priority = "SECONDARY"
        elif symbol_key in ASIA_PRIMARY_PAIRS:
            multiplier = 0.92
            rank = 4
            lane_priority = "OFF_SESSION"

    return SessionPriorityContext(
        session_priority_profile=profile,
        lane_session_priority=lane_priority,
        session_priority_multiplier=float(multiplier),
        session_native_pair=bool(native),
        pair_priority_rank_in_session=int(rank),
        quality_floor_edge=float(quality_floor_edge),
        lane_budget_share=float(session_lane_budget_share(lane_name=lane_key, session_name=session_key)),
        native_override_delta=float(native_override_delta),
        native_override_band_delta=float(native_override_band_delta),
    )


def session_adjusted_score(
    *,
    base_score: float,
    session_priority_multiplier: float,
    lane_strength_multiplier: float = 1.0,
    quality_floor_edge: float = 0.0,
) -> float:
    return clamp(
        (float(base_score) * clamp(float(session_priority_multiplier), 0.80, 1.25) * clamp(float(lane_strength_multiplier), 0.80, 1.25))
        + float(quality_floor_edge),
        0.0,
        1.5,
    )


def session_priority_override_decision(
    *,
    session_name: str,
    candidate_symbol: str,
    candidate_band: str,
    candidate_adjusted_score: float,
    candidate_probability: float,
    candidate_native: bool,
    candidate_lane_priority: str = "",
    candidate_override_delta: float,
    candidate_override_band_delta: float,
    best_native_symbol: str = "",
    best_native_band: str = "",
    best_native_adjusted_score: float = 0.0,
    best_native_probability: float = 0.0,
    throughput_recovery_active: bool = False,
    trajectory_catchup_pressure: float = 0.0,
) -> SessionPriorityDecision:
    session_key = str(session_name or "").upper()
    candidate_key = _normalize_symbol_key(candidate_symbol)
    native_key = _normalize_symbol_key(best_native_symbol)
    lane_priority_key = str(candidate_lane_priority or "").upper()
    if candidate_native or session_key not in {"TOKYO", "SYDNEY", "LONDON", "OVERLAP", "NEW_YORK"}:
        return SessionPriorityDecision(True, False, "", "", "")
    if not native_key:
        return SessionPriorityDecision(True, False, "", "", "")
    candidate_band_rank = quality_band_rank(candidate_band)
    native_band_rank = quality_band_rank(best_native_band)
    adjusted_delta = float(candidate_adjusted_score) - float(best_native_adjusted_score)
    confidence_delta = float(candidate_probability) - float(best_native_probability)
    stronger_band = candidate_band_rank >= (native_band_rank + 1) and adjusted_delta >= float(candidate_override_band_delta)
    higher_confidence = confidence_delta >= 0.08 and adjusted_delta >= float(candidate_override_band_delta)
    clear_superiority = adjusted_delta >= float(candidate_override_delta)
    if stronger_band:
        return SessionPriorityDecision(
            True,
            True,
            "stronger_band_override",
            f"{candidate_key} outranked native {native_key} with a stronger band",
            f"{native_key} lost priority because {candidate_key} was one full band stronger",
        )
    if higher_confidence:
        return SessionPriorityDecision(
            True,
            True,
            "higher_confidence_override",
            f"{candidate_key} outranked native {native_key} with materially higher confidence",
            f"{native_key} lost priority because {candidate_key} had materially higher confidence",
        )
    if clear_superiority:
        return SessionPriorityDecision(
            True,
            True,
            "adjusted_score_superiority_override",
            f"{candidate_key} outranked native {native_key} with clear adjusted-score superiority",
            f"{native_key} lost priority because {candidate_key} had clear adjusted-score superiority",
        )
    always_on_btc_shared_flow = (
        candidate_key == "BTCUSD"
        and session_key in {"TOKYO", "SYDNEY", "LONDON", "OVERLAP", "NEW_YORK"}
        and candidate_band_rank >= 4
        and float(candidate_probability) >= max(0.80, float(best_native_probability) - 0.02)
        and adjusted_delta >= -0.08
    )
    if always_on_btc_shared_flow:
        return SessionPriorityDecision(
            True,
            True,
            "always_on_btc_shared_flow",
            f"{candidate_key} shared flow with native {native_key} as a high-grade always-on lane",
            f"{native_key} shared priority because {candidate_key} met the always-on BTC quality threshold",
        )
    recovery_pressure = clamp(float(trajectory_catchup_pressure), 0.0, 1.0)
    recovery_near_tie = (
        (bool(throughput_recovery_active) or recovery_pressure >= 0.72)
        and candidate_band_rank >= max(4, native_band_rank - 2)
        and adjusted_delta >= (-0.03 - (0.06 * recovery_pressure))
        and confidence_delta >= (-0.03 - (0.04 * recovery_pressure))
    )
    if recovery_near_tie:
        return SessionPriorityDecision(
            True,
            True,
            "throughput_recovery_override",
            f"{candidate_key} shared flow with native {native_key} under throughput recovery",
            f"{native_key} shared priority because stalled trade flow activated throughput recovery",
        )
    prime_session_shared_flow = (
        session_key in {"LONDON", "OVERLAP", "NEW_YORK"}
        and recovery_pressure >= 0.50
        and candidate_band_rank >= max(4, native_band_rank)
        and adjusted_delta >= (-0.035 - (0.04 * recovery_pressure))
        and confidence_delta >= (-0.03 - (0.03 * recovery_pressure))
    )
    if prime_session_shared_flow:
        return SessionPriorityDecision(
            True,
            True,
            "prime_session_shared_flow",
            f"{candidate_key} shared prime-session flow with native {native_key} under density pressure",
            f"{native_key} shared priority because prime-session density pressure widened the flow basket",
        )
    return SessionPriorityDecision(
        False,
        False,
        "",
        "",
        f"{candidate_key} remained below native-session priority held by {native_key}",
    )


def runtime_regime_state(regime_state: str) -> str:
    normalized = str(regime_state or "").strip().upper()
    if normalized in {"TRENDING_UP", "TRENDING_DOWN", "TREND_EXPANSION", "TRENDING"}:
        return "TRENDING"
    if normalized in {"BREAKOUT_COMPRESSION", "BREAKOUT_EXPANSION", "VOLATILITY_SPIKE", "EXPANSION"}:
        return "BREAKOUT_EXPANSION"
    if normalized in {"MEAN_REVERSION", "LIQUIDITY_SWEEP"}:
        return "MEAN_REVERSION"
    if normalized in {"RANGING", "QUIET_ACCUMULATION"}:
        return "RANGING"
    if normalized in {"LOW_LIQUIDITY_DRIFT", "LOW_LIQUIDITY_CHOP"}:
        return "LOW_LIQUIDITY_CHOP"
    if normalized in {"NEWS_DISTORTION", "NEWS_VOLATILE"}:
        return "NEWS_VOLATILE"
    return "RANGING"


def strategy_allowed_regimes(strategy_key: str) -> tuple[str, ...]:
    key = str(strategy_key or "").strip().upper()
    explicit: dict[str, tuple[str, ...]] = {
        "AUDJPY_TOKYO_MOMENTUM_BREAKOUT": ("TRENDING", "BREAKOUT_EXPANSION"),
        "AUDJPY_TOKYO_CONTINUATION_PULLBACK": ("TRENDING",),
        "AUDJPY_SYDNEY_RANGE_BREAK": ("BREAKOUT_EXPANSION", "TRENDING"),
        "AUDJPY_LONDON_CARRY_TREND": ("TRENDING",),
        "AUDJPY_LIQUIDITY_SWEEP_REVERSAL": ("RANGING", "MEAN_REVERSION"),
        "AUDJPY_ATR_COMPRESSION_BREAKOUT": ("BREAKOUT_EXPANSION", "TRENDING"),
        "NZDJPY_TOKYO_BREAKOUT": ("BREAKOUT_EXPANSION", "TRENDING"),
        "NZDJPY_PULLBACK_CONTINUATION": ("TRENDING",),
        "NZDJPY_LIQUIDITY_TRAP_REVERSAL": ("RANGING", "MEAN_REVERSION"),
        "NZDJPY_SESSION_RANGE_EXPANSION": ("BREAKOUT_EXPANSION", "TRENDING"),
        "AUDNZD_RANGE_ROTATION": ("RANGING", "MEAN_REVERSION"),
        "AUDNZD_COMPRESSION_RELEASE": ("BREAKOUT_EXPANSION", "TRENDING"),
        "AUDNZD_VWAP_MEAN_REVERSION": ("RANGING", "MEAN_REVERSION"),
        "AUDNZD_STRUCTURE_BREAK_RETEST": ("TRENDING", "BREAKOUT_EXPANSION"),
        "USDJPY_MOMENTUM_IMPULSE": ("TRENDING", "BREAKOUT_EXPANSION"),
        "USDJPY_VWAP_TREND_CONTINUATION": ("TRENDING",),
        "USDJPY_LIQUIDITY_SWEEP_REVERSAL": ("RANGING", "MEAN_REVERSION"),
        "USDJPY_MACRO_TREND_RIDE": ("TRENDING",),
        "EURJPY_MOMENTUM_IMPULSE": ("TRENDING", "BREAKOUT_EXPANSION"),
        "EURJPY_SESSION_PULLBACK_CONTINUATION": ("TRENDING",),
        "EURJPY_LIQUIDITY_SWEEP_REVERSAL": ("RANGING", "MEAN_REVERSION"),
        "EURJPY_RANGE_FADE": ("RANGING", "MEAN_REVERSION"),
        "GBPJPY_MOMENTUM_IMPULSE": ("TRENDING", "BREAKOUT_EXPANSION"),
        "GBPJPY_SESSION_PULLBACK_CONTINUATION": ("TRENDING",),
        "GBPJPY_LIQUIDITY_SWEEP_REVERSAL": ("RANGING", "MEAN_REVERSION"),
        "GBPJPY_RANGE_FADE": ("RANGING", "MEAN_REVERSION"),
        "EURUSD_LONDON_BREAKOUT": ("BREAKOUT_EXPANSION", "TRENDING"),
        "EURUSD_VWAP_PULLBACK": ("TRENDING",),
        "EURUSD_LIQUIDITY_SWEEP": ("RANGING", "MEAN_REVERSION"),
        "EURUSD_RANGE_FADE": ("RANGING", "MEAN_REVERSION"),
        "GBPUSD_LONDON_EXPANSION_BREAKOUT": ("BREAKOUT_EXPANSION", "TRENDING"),
        "GBPUSD_TREND_PULLBACK_RIDE": ("TRENDING",),
        "GBPUSD_STOP_HUNT_REVERSAL": ("RANGING", "MEAN_REVERSION"),
        "GBPUSD_ATR_EXPANSION_SCALPER": ("BREAKOUT_EXPANSION", "TRENDING"),
        "XAUUSD_ADAPTIVE_M5_GRID": ("TRENDING", "BREAKOUT_EXPANSION", "MEAN_REVERSION"),
        "XAUUSD_LONDON_LIQUIDITY_SWEEP": ("RANGING", "MEAN_REVERSION"),
        "XAUUSD_NY_MOMENTUM_BREAKOUT": ("BREAKOUT_EXPANSION", "TRENDING"),
        "XAUUSD_VWAP_REVERSION": ("RANGING", "MEAN_REVERSION"),
        "XAUUSD_ATR_EXPANSION_SCALPER": ("BREAKOUT_EXPANSION", "TRENDING"),
        "NAS100_OPENING_DRIVE_BREAKOUT": ("BREAKOUT_EXPANSION", "TRENDING"),
        "NAS100_VWAP_TREND_STRATEGY": ("TRENDING",),
        "NAS100_LIQUIDITY_SWEEP_REVERSAL": ("RANGING", "MEAN_REVERSION"),
        "NAS100_MOMENTUM_IMPULSE": ("BREAKOUT_EXPANSION", "TRENDING"),
        "USOIL_INVENTORY_MOMENTUM": ("BREAKOUT_EXPANSION", "TRENDING"),
        "USOIL_LONDON_TREND_EXPANSION": ("BREAKOUT_EXPANSION", "TRENDING"),
        "USOIL_VWAP_REVERSION": ("RANGING", "MEAN_REVERSION"),
        "USOIL_BREAKOUT_RETEST": ("BREAKOUT_EXPANSION", "TRENDING"),
        "BTCUSD_TREND_SCALP": ("TRENDING", "BREAKOUT_EXPANSION"),
        "BTCUSD_RANGE_EXPANSION": ("BREAKOUT_EXPANSION",),
        "BTCUSD_PRICE_ACTION_CONTINUATION": ("TRENDING", "BREAKOUT_EXPANSION"),
        "BTCUSD_VOLATILE_RETEST": ("BREAKOUT_EXPANSION", "TRENDING"),
    }
    if key in explicit:
        return explicit[key]
    if "GRID" in key:
        return ("TRENDING", "BREAKOUT_EXPANSION", "MEAN_REVERSION")
    if any(token in key for token in ("VWAP", "RANGE", "REVERSION", "ROTATION", "FADE", "SWEEP")):
        return ("RANGING", "MEAN_REVERSION")
    if any(token in key for token in ("BREAKOUT", "EXPANSION", "IMPULSE", "CONTINUATION", "TREND", "RETEST")):
        return ("TRENDING", "BREAKOUT_EXPANSION")
    return ("TRENDING", "RANGING", "BREAKOUT_EXPANSION", "MEAN_REVERSION")


def strategy_management_template(strategy_key: str) -> str:
    key = str(strategy_key or "").strip().upper()
    if "GRID" in key:
        return "GRID_BASKET"
    if any(token in key for token in ("VWAP", "RANGE", "REVERSION", "ROTATION", "FADE")):
        return "MEAN_REVERSION_TARGET"
    if any(token in key for token in ("BREAKOUT", "EXPANSION", "IMPULSE")):
        return "VOLATILITY_TRAIL"
    if any(token in key for token in ("PULLBACK", "CONTINUATION", "TREND", "RETEST")):
        return "RUNNER_WITH_PARTIALS"
    return "BALANCED_ACTIVE"


def strategy_regime_fit(strategy_key: str, regime_state: str) -> float:
    strategy_key_upper = str(strategy_key or "").strip().upper()
    allowed = strategy_allowed_regimes(strategy_key)
    normalized_regime = runtime_regime_state(regime_state)
    if strategy_key_upper == "USDJPY_MOMENTUM_IMPULSE" and normalized_regime in {"RANGING", "MEAN_REVERSION"}:
        return 0.05
    if strategy_key_upper == "USDJPY_MOMENTUM_IMPULSE" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.10
    if strategy_key_upper == "USDJPY_MACRO_TREND_RIDE" and normalized_regime == "RANGING":
        return 0.05
    if strategy_key_upper == "USDJPY_MACRO_TREND_RIDE" and normalized_regime in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        return 0.10
    if strategy_key_upper == "USDJPY_LIQUIDITY_SWEEP_REVERSAL" and normalized_regime == "RANGING":
        return 0.72
    if strategy_key_upper == "USDJPY_LIQUIDITY_SWEEP_REVERSAL" and normalized_regime == "MEAN_REVERSION":
        return 0.78
    if strategy_key_upper == "GBPJPY_MOMENTUM_IMPULSE" and normalized_regime in {"RANGING", "MEAN_REVERSION"}:
        return 0.08
    if strategy_key_upper == "GBPJPY_MOMENTUM_IMPULSE" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.06
    if strategy_key_upper == "EURJPY_MOMENTUM_IMPULSE" and normalized_regime in {"RANGING", "MEAN_REVERSION"}:
        return 0.10
    if strategy_key_upper == "EURJPY_MOMENTUM_IMPULSE" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.08
    if strategy_key_upper == "GBPJPY_SESSION_PULLBACK_CONTINUATION" and normalized_regime == "RANGING":
        return 0.06
    if strategy_key_upper == "GBPJPY_SESSION_PULLBACK_CONTINUATION" and normalized_regime in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"}:
        return 0.04
    if strategy_key_upper in {"EURJPY_SESSION_PULLBACK_CONTINUATION", "GBPJPY_SESSION_PULLBACK_CONTINUATION"} and normalized_regime == "RANGING":
        return 0.04
    if strategy_key_upper in {"EURJPY_SESSION_PULLBACK_CONTINUATION", "GBPJPY_SESSION_PULLBACK_CONTINUATION"} and normalized_regime in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"}:
        return 0.03
    if strategy_key_upper == "BTCUSD_PRICE_ACTION_CONTINUATION" and normalized_regime == "RANGING":
        return 0.22
    if strategy_key_upper in {"EURJPY_LIQUIDITY_SWEEP_REVERSAL", "GBPJPY_LIQUIDITY_SWEEP_REVERSAL"} and normalized_regime == "RANGING":
        return 0.68
    if strategy_key_upper in {"EURJPY_LIQUIDITY_SWEEP_REVERSAL", "GBPJPY_LIQUIDITY_SWEEP_REVERSAL"} and normalized_regime == "MEAN_REVERSION":
        return 0.72
    if strategy_key_upper in {"EURJPY_RANGE_FADE", "GBPJPY_RANGE_FADE"} and normalized_regime == "RANGING":
        return 0.26
    if strategy_key_upper in {"EURJPY_RANGE_FADE", "GBPJPY_RANGE_FADE"} and normalized_regime == "MEAN_REVERSION":
        return 0.20
    if strategy_key_upper in {"EURJPY_RANGE_FADE", "GBPJPY_RANGE_FADE"} and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.10
    if strategy_key_upper == "EURUSD_LONDON_BREAKOUT" and normalized_regime == "RANGING":
        return 0.12
    if strategy_key_upper == "EURUSD_LONDON_BREAKOUT" and normalized_regime in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        return 0.12
    if strategy_key_upper == "EURUSD_RANGE_FADE" and normalized_regime == "RANGING":
        return 0.18
    if strategy_key_upper == "GBPUSD_TREND_PULLBACK_RIDE" and normalized_regime == "RANGING":
        return 0.06
    if strategy_key_upper == "GBPUSD_TREND_PULLBACK_RIDE" and normalized_regime in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        return 0.03
    if strategy_key_upper == "EURUSD_VWAP_PULLBACK" and normalized_regime == "RANGING":
        return 0.08
    if strategy_key_upper == "EURUSD_VWAP_PULLBACK" and normalized_regime in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        return 0.04
    if strategy_key_upper == "EURUSD_LIQUIDITY_SWEEP" and normalized_regime == "RANGING":
        return 0.08
    if strategy_key_upper == "EURUSD_LIQUIDITY_SWEEP" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.04
    if strategy_key_upper == "EURUSD_LIQUIDITY_SWEEP" and normalized_regime == "MEAN_REVERSION":
        return 0.06
    if strategy_key_upper == "GBPUSD_LONDON_EXPANSION_BREAKOUT" and normalized_regime == "RANGING":
        return 0.16
    if strategy_key_upper == "GBPUSD_LONDON_EXPANSION_BREAKOUT" and normalized_regime in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        return 0.10
    if strategy_key_upper == "AUDNZD_VWAP_MEAN_REVERSION" and normalized_regime == "RANGING":
        return 0.40
    if strategy_key_upper == "AUDNZD_VWAP_MEAN_REVERSION" and normalized_regime == "MEAN_REVERSION":
        return 0.28
    if strategy_key_upper == "AUDNZD_RANGE_ROTATION" and normalized_regime == "RANGING":
        return 0.82
    if strategy_key_upper == "AUDNZD_STRUCTURE_BREAK_RETEST" and normalized_regime in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        return 0.12
    if strategy_key_upper == "AUDJPY_TOKYO_MOMENTUM_BREAKOUT" and normalized_regime in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        return 0.14
    if strategy_key_upper == "AUDJPY_ATR_COMPRESSION_BREAKOUT" and normalized_regime in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        return 0.14
    if strategy_key_upper == "AUDJPY_TOKYO_CONTINUATION_PULLBACK" and normalized_regime == "RANGING":
        return 0.05
    if strategy_key_upper == "AUDJPY_TOKYO_CONTINUATION_PULLBACK" and normalized_regime in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        return 0.10
    if strategy_key_upper == "NZDJPY_PULLBACK_CONTINUATION" and normalized_regime in {"RANGING", "MEAN_REVERSION"}:
        return 0.05
    if strategy_key_upper == "NZDJPY_PULLBACK_CONTINUATION" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.12
    if strategy_key_upper == "NZDJPY_SESSION_RANGE_EXPANSION" and normalized_regime == "RANGING":
        return 0.04
    if strategy_key_upper == "NZDJPY_SESSION_RANGE_EXPANSION" and normalized_regime in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        return 0.06
    if strategy_key_upper == "XAUUSD_ADAPTIVE_M5_GRID" and normalized_regime == "TRENDING":
        return 0.98
    if strategy_key_upper == "XAUUSD_ADAPTIVE_M5_GRID" and normalized_regime == "BREAKOUT_EXPANSION":
        return 0.84
    if strategy_key_upper == "XAUUSD_ADAPTIVE_M5_GRID" and normalized_regime == "MEAN_REVERSION":
        return 0.52
    if strategy_key_upper == "XAUUSD_ADAPTIVE_M5_GRID" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.04
    if strategy_key_upper == "XAUUSD_LONDON_LIQUIDITY_SWEEP" and normalized_regime == "RANGING":
        return 0.58
    if strategy_key_upper == "XAUUSD_LONDON_LIQUIDITY_SWEEP" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.06
    if strategy_key_upper == "XAUUSD_LONDON_LIQUIDITY_SWEEP" and normalized_regime == "MEAN_REVERSION":
        return 0.84
    if strategy_key_upper == "XAUUSD_VWAP_REVERSION" and normalized_regime == "MEAN_REVERSION":
        return 0.86
    if strategy_key_upper == "XAUUSD_VWAP_REVERSION" and normalized_regime == "RANGING":
        return 0.72
    if strategy_key_upper == "XAUUSD_NY_MOMENTUM_BREAKOUT" and normalized_regime == "TRENDING":
        return 0.72
    if strategy_key_upper == "XAUUSD_NY_MOMENTUM_BREAKOUT" and normalized_regime in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"}:
        return 0.18
    if strategy_key_upper == "XAUUSD_ATR_EXPANSION_SCALPER" and normalized_regime == "TRENDING":
        return 0.62
    if strategy_key_upper == "XAUUSD_ATR_EXPANSION_SCALPER" and normalized_regime in {"RANGING", "MEAN_REVERSION"}:
        return 0.18
    if strategy_key_upper == "XAUUSD_ATR_EXPANSION_SCALPER" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.08
    if strategy_key_upper == "USOIL_LONDON_TREND_EXPANSION" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.18
    if strategy_key_upper == "USOIL_LONDON_TREND_EXPANSION" and normalized_regime == "MEAN_REVERSION":
        return 0.22
    if strategy_key_upper == "USOIL_INVENTORY_MOMENTUM" and normalized_regime == "RANGING":
        return 0.08
    if strategy_key_upper == "USOIL_INVENTORY_MOMENTUM" and normalized_regime in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
        return 0.06
    if strategy_key_upper == "NAS100_LIQUIDITY_SWEEP_REVERSAL" and normalized_regime == "MEAN_REVERSION":
        return 0.72
    if strategy_key_upper == "NAS100_LIQUIDITY_SWEEP_REVERSAL" and normalized_regime == "TRENDING":
        return 0.78
    if strategy_key_upper == "NAS100_LIQUIDITY_SWEEP_REVERSAL" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.62
    if strategy_key_upper == "NAS100_OPENING_DRIVE_BREAKOUT" and normalized_regime == "TRENDING":
        return 0.52
    if strategy_key_upper == "NAS100_OPENING_DRIVE_BREAKOUT" and normalized_regime in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"}:
        return 0.08
    if strategy_key_upper == "NAS100_MOMENTUM_IMPULSE" and normalized_regime == "TRENDING":
        return 0.34
    if strategy_key_upper == "NAS100_MOMENTUM_IMPULSE" and normalized_regime == "RANGING":
        return 0.16
    if strategy_key_upper == "NAS100_MOMENTUM_IMPULSE" and normalized_regime in {"MEAN_REVERSION", "LOW_LIQUIDITY_CHOP"}:
        return 0.12
    if strategy_key_upper == "NAS100_VWAP_TREND_STRATEGY" and normalized_regime == "TRENDING":
        return 0.80
    if strategy_key_upper == "NAS100_VWAP_TREND_STRATEGY" and normalized_regime == "BREAKOUT_EXPANSION":
        return 0.72
    if strategy_key_upper == "BTCUSD_TREND_SCALP" and normalized_regime in {"RANGING", "LOW_LIQUIDITY_CHOP"}:
        return 0.08
    if strategy_key_upper == "BTCUSD_TREND_SCALP" and normalized_regime == "TRENDING":
        return 0.68
    if strategy_key_upper == "BTCUSD_TREND_SCALP" and normalized_regime == "MEAN_REVERSION":
        return 0.12
    if strategy_key_upper == "BTCUSD_PRICE_ACTION_CONTINUATION" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.36
    if strategy_key_upper == "BTCUSD_PRICE_ACTION_CONTINUATION" and normalized_regime == "MEAN_REVERSION":
        return 0.26
    if strategy_key_upper == "BTCUSD_PRICE_ACTION_CONTINUATION" and normalized_regime == "TRENDING":
        return 0.86
    if strategy_key_upper == "BTCUSD_RANGE_EXPANSION" and normalized_regime == "RANGING":
        return 0.06
    if strategy_key_upper == "BTCUSD_RANGE_EXPANSION" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.02
    if strategy_key_upper == "BTCUSD_VOLATILE_RETEST" and normalized_regime == "RANGING":
        return 0.42
    if strategy_key_upper == "BTCUSD_VOLATILE_RETEST" and normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.24
    if strategy_key_upper == "BTCUSD_VOLATILE_RETEST" and normalized_regime == "MEAN_REVERSION":
        return 0.18
    if strategy_key_upper == "BTCUSD_VOLATILE_RETEST" and normalized_regime == "TRENDING":
        return 0.80
    if normalized_regime in allowed:
        return 0.94
    if normalized_regime == "NEWS_VOLATILE":
        return 0.20
    if normalized_regime == "LOW_LIQUIDITY_CHOP":
        return 0.34 if any(token in str(strategy_key or "").upper() for token in ("VWAP", "RANGE", "REVERSION")) else 0.20
    if normalized_regime == "BREAKOUT_EXPANSION" and "TRENDING" in allowed:
        return 0.70
    if normalized_regime == "TRENDING" and "BREAKOUT_EXPANSION" in allowed:
        return 0.72
    if normalized_regime == "RANGING" and "MEAN_REVERSION" in allowed:
        return 0.74
    return 0.42


def pair_behavior_fit(
    *,
    symbol: str,
    strategy_key: str,
    session_name: str,
    regime_state: str,
    weekend_mode: bool = False,
) -> float:
    symbol_key = _normalize_symbol_key(symbol)
    session_key = str(session_name or "").upper()
    regime_key = runtime_regime_state(regime_state)
    strategy_key_upper = str(strategy_key or "").upper()
    score = 0.60
    if symbol_key in {"AUDJPY", "NZDJPY"}:
        if session_key in {"SYDNEY", "TOKYO"}:
            score += 0.18
        if any(token in strategy_key_upper for token in ("BREAKOUT", "CONTINUATION", "IMPULSE", "EXPANSION")):
            score += 0.10
        if any(token in strategy_key_upper for token in ("SWEEP", "RECLAIM", "TRAP", "REVERSAL")):
            score += 0.06
    elif symbol_key == "AUDNZD":
        score += 0.04
        if session_key in {"SYDNEY", "TOKYO"}:
            score += 0.12
        if any(token in strategy_key_upper for token in ("ROTATION", "RANGE", "REVERSION", "RETEST", "COMPRESSION")):
            score += 0.10
        if "SCALP" in strategy_key_upper or "IMPULSE" in strategy_key_upper:
            score -= 0.04
    elif symbol_key in {"EURUSD", "GBPUSD"}:
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            score += 0.16
        elif session_key in {"SYDNEY", "TOKYO"}:
            score -= 0.08
    elif symbol_key == "USDJPY":
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            score += 0.14
        elif session_key in {"SYDNEY", "TOKYO"}:
            score += 0.08
        if any(token in strategy_key_upper for token in ("SWEEP", "REVERSAL")):
            score += 0.04
    elif symbol_key in {"EURJPY", "GBPJPY"}:
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            score += 0.10
        elif session_key in {"SYDNEY", "TOKYO"}:
            score += 0.02
        if any(token in strategy_key_upper for token in ("SWEEP", "REVERSAL")):
            score += 0.03
    elif symbol_key == "XAUUSD":
        if "GRID" in strategy_key_upper:
            if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
                score += 0.28
            else:
                score -= 0.25
        elif "VWAP_REVERSION" in strategy_key_upper and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            score += 0.08
        elif session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            score += 0.02
    elif symbol_key in {"NAS100", "USOIL"}:
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            score += 0.16
        else:
            score -= 0.05
        if symbol_key == "NAS100" and session_key == "TOKYO" and "MOMENTUM_IMPULSE" in strategy_key_upper:
            score -= 0.24
        if symbol_key == "NAS100" and session_key == "TOKYO" and "OPENING_DRIVE_BREAKOUT" in strategy_key_upper:
            score -= 0.22
        if symbol_key == "NAS100" and session_key == "TOKYO" and "LIQUIDITY_SWEEP_REVERSAL" in strategy_key_upper:
            score += 0.16
        if symbol_key == "NAS100" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and "VWAP_TREND_STRATEGY" in strategy_key_upper:
            score += 0.12
    elif symbol_key == "BTCUSD":
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            score += 0.08
        elif session_key in {"SYDNEY", "TOKYO"} and "PRICE_ACTION_CONTINUATION" in strategy_key_upper:
            score += 0.18 if weekend_mode else 0.12
        elif session_key in {"SYDNEY", "TOKYO"} and "VOLATILE_RETEST" in strategy_key_upper:
            score += 0.16 if weekend_mode else 0.10
        elif session_key in {"SYDNEY", "TOKYO"} and "TREND_SCALP" in strategy_key_upper:
            score -= 0.14 if not weekend_mode else 0.06
    if regime_key == "LOW_LIQUIDITY_CHOP" and any(token in strategy_key_upper for token in ("BREAKOUT", "IMPULSE", "EXPANSION")):
        score -= 0.12
    if regime_key == "LOW_LIQUIDITY_CHOP" and symbol_key in {"AUDJPY", "NZDJPY", "USDJPY"} and any(
        token in strategy_key_upper for token in ("CONTINUATION", "PULLBACK", "BREAKOUT", "IMPULSE")
    ):
        score -= 0.06
    if regime_key == "LOW_LIQUIDITY_CHOP" and symbol_key in {"EURJPY", "GBPJPY"} and any(
        token in strategy_key_upper for token in ("CONTINUATION", "PULLBACK", "BREAKOUT", "IMPULSE", "TREND")
    ):
        score -= 0.14
    if regime_key == "LOW_LIQUIDITY_CHOP" and symbol_key in {"EURUSD", "GBPUSD"} and any(
        token in strategy_key_upper for token in ("BREAKOUT", "PULLBACK", "EXPANSION", "IMPULSE")
    ):
        score -= 0.06
    if regime_key == "LOW_LIQUIDITY_CHOP" and session_key == "TOKYO" and symbol_key in {"EURUSD", "GBPUSD"} and any(
        token in strategy_key_upper for token in ("SWEEP", "RECLAIM", "REVERSAL", "LONDON_BREAKOUT", "EXPANSION")
    ):
        score -= 0.18
    if regime_key == "LOW_LIQUIDITY_CHOP" and symbol_key == "BTCUSD" and any(
        token in strategy_key_upper for token in ("TREND", "SCALP", "CONTINUATION", "IMPULSE")
    ):
        score -= 0.10
    if regime_key == "LOW_LIQUIDITY_CHOP" and symbol_key == "BTCUSD" and "PRICE_ACTION_CONTINUATION" in strategy_key_upper:
        score += 0.16
    if regime_key == "LOW_LIQUIDITY_CHOP" and symbol_key == "BTCUSD" and "VOLATILE_RETEST" in strategy_key_upper:
        score += 0.18
    if regime_key == "LOW_LIQUIDITY_CHOP" and symbol_key == "NAS100" and "LIQUIDITY_SWEEP_REVERSAL" in strategy_key_upper:
        score += 0.10
    if regime_key == "RANGING":
        if symbol_key in {"AUDJPY", "NZDJPY", "USDJPY", "EURUSD", "GBPUSD"} and any(
            token in strategy_key_upper for token in ("BREAKOUT", "IMPULSE", "CONTINUATION", "TREND", "RETEST")
        ):
            score -= 0.22
        if symbol_key in {"EURJPY", "GBPJPY"} and any(
            token in strategy_key_upper for token in ("BREAKOUT", "IMPULSE", "CONTINUATION", "TREND", "RETEST")
        ):
            score -= 0.24
        if symbol_key in {"AUDJPY", "NZDJPY", "USDJPY"} and any(
            token in strategy_key_upper for token in ("SWEEP", "RECLAIM", "REVERSAL", "TRAP")
        ):
            score += 0.10
        if symbol_key in {"EURJPY", "GBPJPY"} and any(
            token in strategy_key_upper for token in ("SWEEP", "RECLAIM", "REVERSAL")
        ):
            score += 0.02
        if symbol_key == "BTCUSD" and any(token in strategy_key_upper for token in ("TREND", "SCALP", "CONTINUATION", "IMPULSE")):
            score -= 0.18
        if symbol_key == "BTCUSD" and session_key in {"SYDNEY", "TOKYO"} and "PRICE_ACTION_CONTINUATION" in strategy_key_upper:
            score += 0.08
        if symbol_key == "BTCUSD" and session_key in {"SYDNEY", "TOKYO"} and "VOLATILE_RETEST" in strategy_key_upper:
            score += 0.10
        if symbol_key == "AUDNZD" and any(token in strategy_key_upper for token in ("ROTATION", "REVERSION", "RANGE", "VWAP")):
            score += 0.10
        if symbol_key == "AUDNZD" and any(token in strategy_key_upper for token in ("IMPULSE", "BREAKOUT")):
            score -= 0.06
        if symbol_key == "GBPUSD" and any(
            token in strategy_key_upper for token in ("VWAP", "REVERSION", "SWEEP")
        ):
            score += 0.02
        if symbol_key == "GBPUSD" and session_key == "TOKYO" and "LONDON_EXPANSION_BREAKOUT" in strategy_key_upper:
            score -= 0.28
        if symbol_key == "EURUSD" and "RANGE_FADE" in strategy_key_upper:
            score -= 0.16
        if symbol_key == "USDJPY" and any(token in strategy_key_upper for token in ("MOMENTUM", "IMPULSE", "VWAP_TREND", "CONTINUATION")):
            score -= 0.10
        if symbol_key == "USDJPY" and "LIQUIDITY_SWEEP_REVERSAL" in strategy_key_upper:
            score += 0.04
        if symbol_key == "USDJPY" and "MACRO_TREND_RIDE" in strategy_key_upper and session_key in {"OVERLAP", "NEW_YORK", "TOKYO"}:
            score -= 0.10
        if symbol_key in {"AUDJPY", "NZDJPY"} and any(token in strategy_key_upper for token in ("CONTINUATION", "BREAKOUT", "IMPULSE")):
            score -= 0.12
        if symbol_key == "NZDJPY" and "PULLBACK_CONTINUATION" in strategy_key_upper and session_key == "TOKYO":
            score -= 0.12
        if symbol_key == "USDJPY" and "MOMENTUM_IMPULSE" in strategy_key_upper and session_key == "TOKYO":
            score -= 0.12
        if symbol_key == "USDJPY" and "MOMENTUM_IMPULSE" in strategy_key_upper and session_key == "SYDNEY":
            score -= 0.08
        if symbol_key == "USDJPY" and "MOMENTUM_IMPULSE" in strategy_key_upper and session_key == "LONDON":
            score -= 0.06
        if symbol_key == "USDJPY" and "MACRO_TREND_RIDE" in strategy_key_upper and session_key == "OVERLAP":
            score -= 0.08
        if symbol_key == "USDJPY" and "LIQUIDITY_SWEEP_REVERSAL" in strategy_key_upper and session_key == "TOKYO":
            score += 0.08
        if symbol_key == "USDJPY" and "LIQUIDITY_SWEEP_REVERSAL" in strategy_key_upper and session_key == "SYDNEY":
            score += 0.04
        if symbol_key in {"EURJPY", "GBPJPY"} and session_key == "TOKYO" and any(
            token in strategy_key_upper for token in ("MOMENTUM_IMPULSE", "SESSION_PULLBACK_CONTINUATION")
        ):
            score -= 0.34
        if symbol_key == "GBPJPY" and session_key == "TOKYO" and "SESSION_PULLBACK_CONTINUATION" in strategy_key_upper:
            score -= 0.14
        if symbol_key in {"EURJPY", "GBPJPY"} and session_key == "TOKYO" and "LIQUIDITY_SWEEP_REVERSAL" in strategy_key_upper:
            score += 0.08
        if symbol_key in {"EURJPY", "GBPJPY"} and session_key == "TOKYO" and "RANGE_FADE" in strategy_key_upper:
            score -= 0.08
        if symbol_key in {"EURJPY", "GBPJPY"} and session_key == "SYDNEY" and any(
            token in strategy_key_upper for token in ("MOMENTUM_IMPULSE", "SESSION_PULLBACK_CONTINUATION")
        ):
            score -= 0.24
        if symbol_key == "EURUSD" and "RANGE_FADE" in strategy_key_upper and session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            score -= 0.18
        if symbol_key == "EURUSD" and session_key == "TOKYO" and any(
            token in strategy_key_upper for token in ("LONDON_BREAKOUT", "LIQUIDITY_SWEEP", "VWAP_PULLBACK")
        ):
            score -= 0.26
        if symbol_key == "GBPUSD" and session_key == "TOKYO" and any(
            token in strategy_key_upper for token in ("STOP_HUNT_REVERSAL", "LONDON_EXPANSION_BREAKOUT", "TREND_PULLBACK")
        ):
            score -= 0.22
        if symbol_key == "NZDJPY" and "SESSION_RANGE_EXPANSION" in strategy_key_upper and session_key == "TOKYO":
            score -= 0.24
        if symbol_key == "NZDJPY" and session_key == "TOKYO" and any(
            token in strategy_key_upper for token in ("LIQUIDITY_TRAP_REVERSAL", "SESSION_RANGE_EXPANSION")
        ):
            score += 0.06 if "LIQUIDITY_TRAP_REVERSAL" in strategy_key_upper else -0.12
        if symbol_key == "NZDJPY" and session_key == "TOKYO" and "PULLBACK_CONTINUATION" in strategy_key_upper:
            score -= 0.04
        if symbol_key == "AUDJPY" and "TOKYO_CONTINUATION_PULLBACK" in strategy_key_upper and session_key == "TOKYO":
            score -= 0.16
        if symbol_key == "AUDJPY" and session_key == "TOKYO" and any(
            token in strategy_key_upper for token in ("LIQUIDITY_SWEEP_REVERSAL", "TOKYO_MOMENTUM_BREAKOUT")
        ):
            score += 0.10 if "LIQUIDITY_SWEEP_REVERSAL" in strategy_key_upper else -0.12
        if symbol_key == "AUDJPY" and session_key == "SYDNEY" and "TOKYO_MOMENTUM_BREAKOUT" in strategy_key_upper:
            score -= 0.14
        if symbol_key == "AUDNZD" and session_key == "TOKYO" and any(
            token in strategy_key_upper for token in ("STRUCTURE_BREAK_RETEST", "VWAP_MEAN_REVERSION")
        ):
            score -= 0.20 if "VWAP_MEAN_REVERSION" in strategy_key_upper else -0.10
        if symbol_key == "AUDNZD" and "VWAP_MEAN_REVERSION" in strategy_key_upper and session_key in {"SYDNEY", "TOKYO"}:
            score -= 0.22
        if symbol_key == "AUDNZD" and "RANGE_ROTATION" in strategy_key_upper and session_key in {"SYDNEY", "TOKYO"}:
            score += 0.14
        if symbol_key == "XAUUSD" and "ATR_EXPANSION_SCALPER" in strategy_key_upper and session_key in {"SYDNEY", "TOKYO"}:
            score -= 0.08
        if symbol_key == "BTCUSD" and "TREND_SCALP" in strategy_key_upper and session_key == "TOKYO":
            score -= 0.28 if not weekend_mode else 0.12
        if symbol_key == "BTCUSD" and "PRICE_ACTION_CONTINUATION" in strategy_key_upper and session_key == "TOKYO":
            score += 0.02 if weekend_mode else -0.08
        if symbol_key == "BTCUSD" and "PRICE_ACTION_CONTINUATION" in strategy_key_upper and session_key == "TOKYO" and regime_key == "RANGING":
            score -= 0.10 if weekend_mode else 0.18
        if symbol_key == "BTCUSD" and "VOLATILE_RETEST" in strategy_key_upper and session_key == "TOKYO" and regime_key in {"RANGING", "MEAN_REVERSION"}:
            score += 0.16 if weekend_mode else 0.08
        if symbol_key == "BTCUSD" and "VOLATILE_RETEST" in strategy_key_upper and session_key == "TOKYO" and regime_key == "MEAN_REVERSION" and not weekend_mode:
            score -= 0.18
        if symbol_key == "BTCUSD" and "RANGE_EXPANSION" in strategy_key_upper and session_key in {"TOKYO", "OFF"}:
            score -= 0.32
        if symbol_key == "BTCUSD" and "RANGE_EXPANSION" in strategy_key_upper and session_key == "TOKYO" and regime_key == "LOW_LIQUIDITY_CHOP":
            score -= 0.18
        if symbol_key == "USOIL" and "INVENTORY_MOMENTUM" in strategy_key_upper and session_key == "TOKYO":
            score -= 0.32
        if symbol_key == "USOIL" and "INVENTORY_MOMENTUM" in strategy_key_upper and session_key == "TOKYO" and regime_key in {"RANGING", "LOW_LIQUIDITY_CHOP", "MEAN_REVERSION"}:
            score -= 0.22
    if regime_key == "MEAN_REVERSION" and session_key == "TOKYO":
        if symbol_key in {"AUDJPY", "NZDJPY", "USDJPY"} and any(
            token in strategy_key_upper for token in ("SWEEP", "RECLAIM", "REVERSAL", "TRAP")
        ):
            score += 0.08 if symbol_key in {"AUDJPY", "NZDJPY"} else 0.05
        if symbol_key in {"EURJPY", "GBPJPY"} and "LIQUIDITY_SWEEP_REVERSAL" in strategy_key_upper:
            score -= 0.04
        if symbol_key in {"AUDJPY", "NZDJPY", "USDJPY"} and any(
            token in strategy_key_upper for token in ("CONTINUATION", "PULLBACK", "BREAKOUT", "IMPULSE", "TREND")
        ):
            score -= 0.08
        if symbol_key in {"EURJPY", "GBPJPY"} and any(
            token in strategy_key_upper for token in ("CONTINUATION", "PULLBACK", "BREAKOUT", "IMPULSE", "TREND", "RANGE_FADE")
        ):
            score -= 0.28
        if symbol_key in {"AUDNZD", "EURUSD", "GBPUSD"} and any(
            token in strategy_key_upper for token in ("VWAP", "SWEEP", "REVERSION", "RETEST", "BREAKOUT")
        ):
            score -= 0.16
        if symbol_key == "AUDNZD" and "VWAP_MEAN_REVERSION" in strategy_key_upper:
            score -= 0.16
        if symbol_key == "AUDNZD" and any(
            token in strategy_key_upper for token in ("VWAP", "REVERSION", "RANGE", "RETEST", "BREAKOUT")
        ):
            score -= 0.10
        if symbol_key == "USOIL" and "LONDON_TREND_EXPANSION" in strategy_key_upper:
            score -= 0.12
        if symbol_key == "USOIL" and "INVENTORY_MOMENTUM" in strategy_key_upper:
            score -= 0.14
        if symbol_key == "NAS100" and "MOMENTUM_IMPULSE" in strategy_key_upper:
            score -= 0.28
        if symbol_key == "NAS100" and "LIQUIDITY_SWEEP_REVERSAL" in strategy_key_upper:
            score += 0.16
        if symbol_key == "NAS100" and "OPENING_DRIVE_BREAKOUT" in strategy_key_upper:
            score -= 0.22
    if regime_key == "MEAN_REVERSION" and session_key == "LONDON":
        if symbol_key == "EURUSD" and "LIQUIDITY_SWEEP" in strategy_key_upper:
            score -= 0.18
        if symbol_key == "XAUUSD" and "NY_MOMENTUM_BREAKOUT" in strategy_key_upper:
            score -= 0.12
    if symbol_key == "EURUSD" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and "LIQUIDITY_SWEEP" in strategy_key_upper:
        if regime_key == "RANGING":
            score -= 0.22
        if regime_key == "MEAN_REVERSION":
            score -= 0.18
    if session_key == "LONDON" and symbol_key == "EURUSD" and "LIQUIDITY_SWEEP" in strategy_key_upper and regime_key == "MEAN_REVERSION":
        score -= 0.34
    if session_key == "LONDON" and symbol_key == "EURUSD" and "VWAP_PULLBACK" in strategy_key_upper and regime_key == "LOW_LIQUIDITY_CHOP":
        score -= 0.40
    if session_key == "LONDON" and symbol_key == "EURUSD" and "VWAP_PULLBACK" in strategy_key_upper and regime_key == "MEAN_REVERSION":
        score -= 0.24
    if session_key == "LONDON" and symbol_key == "EURUSD" and "VWAP_PULLBACK" in strategy_key_upper:
        score -= 0.04
    if session_key == "LONDON" and symbol_key == "XAUUSD" and "ADAPTIVE_M5_GRID" in strategy_key_upper:
        score += 0.12
    if session_key == "LONDON" and symbol_key == "XAUUSD" and "ATR_EXPANSION_SCALPER" in strategy_key_upper and regime_key in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION", "TRENDING", "RANGING"}:
        score -= 0.36
    if session_key == "LONDON" and symbol_key == "XAUUSD" and "LONDON_LIQUIDITY_SWEEP" in strategy_key_upper and regime_key in {"LOW_LIQUIDITY_CHOP", "RANGING"}:
        score -= 0.34
    if session_key == "LONDON" and symbol_key == "XAUUSD" and "LONDON_LIQUIDITY_SWEEP" in strategy_key_upper and regime_key == "MEAN_REVERSION":
        score += 0.14
    if session_key == "NEW_YORK" and symbol_key == "XAUUSD" and "LONDON_LIQUIDITY_SWEEP" in strategy_key_upper and regime_key == "MEAN_REVERSION":
        score += 0.10
    if session_key == "OVERLAP" and symbol_key == "XAUUSD" and "LONDON_LIQUIDITY_SWEEP" in strategy_key_upper and regime_key == "MEAN_REVERSION":
        score -= 0.40
    if session_key == "LONDON" and symbol_key == "XAUUSD" and "NY_MOMENTUM_BREAKOUT" in strategy_key_upper and regime_key in {"LOW_LIQUIDITY_CHOP", "MEAN_REVERSION", "TRENDING", "RANGING"}:
        score -= 0.40
    if session_key in {"OVERLAP", "NEW_YORK"} and symbol_key == "XAUUSD" and "ATR_EXPANSION_SCALPER" in strategy_key_upper and regime_key in {"TRENDING", "RANGING", "MEAN_REVERSION"}:
        score -= 0.28
    if session_key in {"OVERLAP", "NEW_YORK"} and symbol_key == "XAUUSD" and "NY_MOMENTUM_BREAKOUT" in strategy_key_upper and regime_key == "TRENDING":
        score -= 0.24
    if session_key in {"OVERLAP", "NEW_YORK"} and symbol_key == "XAUUSD" and "NY_MOMENTUM_BREAKOUT" in strategy_key_upper and regime_key == "BREAKOUT_EXPANSION":
        score += 0.08
    if regime_key == "BREAKOUT_EXPANSION" and any(token in strategy_key_upper for token in ("BREAKOUT", "IMPULSE", "EXPANSION", "CONTINUATION")):
        score += 0.10
    if regime_key == "TRENDING" and symbol_key == "BTCUSD" and session_key == "LONDON" and "TREND_SCALP" in strategy_key_upper:
        score -= 0.16
    if regime_key == "TRENDING" and symbol_key == "BTCUSD" and session_key == "LONDON" and any(
        token in strategy_key_upper for token in ("RANGE_EXPANSION", "SESSION_EXPANSION")
    ):
        score -= 0.05
    if regime_key == "TRENDING" and symbol_key == "BTCUSD" and session_key == "TOKYO" and "PRICE_ACTION_CONTINUATION" in strategy_key_upper:
        score += 0.04 if weekend_mode else -0.18
    if regime_key == "TRENDING" and symbol_key == "BTCUSD" and session_key == "TOKYO" and "TREND_SCALP" in strategy_key_upper:
        score -= 0.12 if not weekend_mode else 0.00
    if regime_key == "TRENDING" and symbol_key == "BTCUSD" and session_key == "TOKYO" and "VOLATILE_RETEST" in strategy_key_upper:
        score -= 0.06 if not weekend_mode else 0.00
    if regime_key == "TRENDING" and symbol_key == "BTCUSD" and session_key == "SYDNEY" and "PRICE_ACTION_CONTINUATION" in strategy_key_upper:
        score += 0.10 if weekend_mode else 0.06
    if regime_key == "TRENDING" and symbol_key == "BTCUSD" and session_key == "SYDNEY" and "TREND_SCALP" in strategy_key_upper:
        score -= 0.10
    if regime_key == "TRENDING" and symbol_key == "NAS100" and session_key == "TOKYO" and "MOMENTUM_IMPULSE" in strategy_key_upper:
        score -= 0.24
    if regime_key == "TRENDING" and symbol_key == "NAS100" and session_key == "TOKYO" and "LIQUIDITY_SWEEP_REVERSAL" in strategy_key_upper:
        score += 0.12
    if regime_key == "TRENDING" and symbol_key == "NAS100" and session_key == "TOKYO" and "OPENING_DRIVE_BREAKOUT" in strategy_key_upper:
        score -= 0.16
    if regime_key == "TRENDING" and symbol_key == "NAS100" and session_key in {"LONDON", "OVERLAP", "NEW_YORK"} and "VWAP_TREND_STRATEGY" in strategy_key_upper:
        score += 0.10
    score += winner_promotion_bonus(
        symbol=symbol_key,
        strategy_key=strategy_key_upper,
        regime_state=regime_key,
        session_name=session_key,
        weekend_mode=bool(weekend_mode),
    )
    return clamp(score, 0.20, 1.00)


def structure_cleanliness_score(
    *,
    spread_points: float,
    spread_limit: float,
    structure_score: float,
    liquidity_score: float,
    volatility_state: str,
    regime_state: str,
    pressure_alignment: float = 0.6,
) -> float:
    spread_quality = clamp(1.0 - (max(0.0, float(spread_points)) / max(1.0, float(spread_limit))), 0.0, 1.0)
    structure_component = clamp(float(structure_score), 0.0, 1.0)
    liquidity_component = clamp(float(liquidity_score), 0.0, 1.0)
    pressure_component = clamp(float(pressure_alignment), 0.0, 1.0)
    volatility_key = str(volatility_state or "").upper()
    regime_key = runtime_regime_state(regime_state)
    volatility_component = 0.70
    if volatility_key in {"BALANCED", "EXPANSION_IMMINENT"}:
        volatility_component = 0.88
    elif volatility_key == "COMPRESSION":
        volatility_component = 0.76
    elif volatility_key == "SPIKE":
        volatility_component = 0.34
    if regime_key == "LOW_LIQUIDITY_CHOP":
        volatility_component *= 0.75
    return clamp(
        (0.28 * structure_component)
        + (0.22 * liquidity_component)
        + (0.18 * spread_quality)
        + (0.16 * pressure_component)
        + (0.16 * volatility_component),
        0.0,
        1.0,
    )


def entry_timing_score(
    *,
    structure_cleanliness: float,
    probability: float,
    expected_value_r: float,
    spread_points: float,
    spread_limit: float,
    volatility_state: str,
    regime_state: str,
    chase_penalty: float = 0.0,
    delta_proxy_score_value: float = 0.0,
) -> float:
    spread_quality = clamp(1.0 - (max(0.0, float(spread_points)) / max(1.0, float(spread_limit))), 0.0, 1.0)
    prob_component = clamp(float(probability), 0.0, 1.0)
    ev_component = clamp((float(expected_value_r) + 0.10) / 1.50, 0.0, 1.0)
    base = (
        (0.34 * clamp(float(structure_cleanliness), 0.0, 1.0))
        + (0.24 * prob_component)
        + (0.18 * ev_component)
        + (0.14 * spread_quality)
    )
    volatility_key = str(volatility_state or "").upper()
    regime_key = runtime_regime_state(regime_state)
    if volatility_key == "SPIKE":
        base -= 0.08
    if regime_key == "LOW_LIQUIDITY_CHOP":
        base -= 0.10
    base += 0.10 * clamp((float(delta_proxy_score_value) + 1.0) / 2.0, 0.0, 1.0)
    base -= clamp(float(chase_penalty), 0.0, 0.30)
    return clamp(base, 0.0, 1.0)


def strategy_recent_performance_score(
    *,
    win_rate: float,
    profit_factor: float,
    expectancy_r: float,
    management_quality: float,
) -> float:
    return clamp(
        (0.32 * clamp(float(win_rate), 0.0, 1.0))
        + (0.24 * clamp(float(profit_factor) / 2.0, 0.0, 1.0))
        + (0.24 * clamp((float(expectancy_r) + 0.20) / 0.80, 0.0, 1.0))
        + (0.20 * clamp(float(management_quality), 0.0, 1.0)),
        0.0,
        1.0,
    )


def strategy_health_state(
    *,
    win_rate: float,
    profit_factor: float,
    expectancy_r: float,
    management_quality: float,
    sample_size: int,
) -> str:
    if sample_size >= 6 and (
        float(expectancy_r) <= -0.10
        or float(profit_factor) < 0.75
        or (float(win_rate) < 0.38 and float(management_quality) < 0.48)
    ):
        return "QUARANTINED"
    if sample_size >= 4 and (
        float(expectancy_r) < -0.01
        or float(profit_factor) < 0.92
        or float(win_rate) < 0.44
    ):
        return "REDUCED"
    if sample_size >= 5 and (
        float(expectancy_r) >= 0.08
        and float(profit_factor) >= 1.12
        and float(win_rate) >= 0.52
        and float(management_quality) >= 0.50
    ):
        return "ATTACK"
    return "NORMAL"


def strategy_selection_score(
    *,
    ev_estimate: float,
    regime_fit: float,
    session_fit: float,
    volatility_fit: float,
    pair_behavior_fit_score: float,
    strategy_recent_performance: float,
    execution_quality_fit: float,
    entry_timing_score_value: float,
    structure_cleanliness_score_value: float,
    drawdown_penalty: float = 0.0,
    false_break_penalty: float = 0.0,
    chop_penalty: float = 0.0,
) -> float:
    score = (
        (0.20 * clamp(float(ev_estimate), 0.0, 1.0))
        + (0.16 * clamp(float(regime_fit), 0.0, 1.0))
        + (0.10 * clamp(float(session_fit), 0.0, 1.0))
        + (0.08 * clamp(float(volatility_fit), 0.0, 1.0))
        + (0.08 * clamp(float(pair_behavior_fit_score), 0.0, 1.0))
        + (0.10 * clamp(float(strategy_recent_performance), 0.0, 1.0))
        + (0.10 * clamp(float(execution_quality_fit), 0.0, 1.0))
        + (0.10 * clamp(float(entry_timing_score_value), 0.0, 1.0))
        + (0.08 * clamp(float(structure_cleanliness_score_value), 0.0, 1.0))
    )
    score -= (
        0.08 * clamp(float(drawdown_penalty), 0.0, 1.0)
        + 0.12 * clamp(float(false_break_penalty), 0.0, 1.0)
        + 0.08 * clamp(float(chop_penalty), 0.0, 1.0)
    )
    return clamp(score, 0.0, 1.0)


def evaluate_execution_quality(
    *,
    spread_points: float,
    typical_spread_points: float,
    slippage_points: float = 0.0,
    fill_delay_ms: float = 0.0,
    reject_rate: float = 0.0,
    stale_idea_rate: float = 0.0,
    bridge_latency_ms: float = 0.0,
) -> ExecutionQuality:
    typical = max(1.0, float(typical_spread_points))
    spread_ratio = clamp(float(spread_points) / typical, 0.0, 4.0)
    spread_quality = clamp(1.0 - ((spread_ratio - 1.0) * 0.45), 0.0, 1.0)
    slippage_quality = clamp(1.0 - (max(0.0, float(slippage_points)) / max(typical, 1.0)), 0.0, 1.0)
    latency_quality = clamp(1.0 - (max(0.0, float(fill_delay_ms)) / 4000.0), 0.0, 1.0)
    bridge_quality = clamp(1.0 - (max(0.0, float(bridge_latency_ms)) / 3000.0), 0.0, 1.0)
    reject_quality = clamp(1.0 - max(0.0, float(reject_rate)), 0.0, 1.0)
    stale_quality = clamp(1.0 - max(0.0, float(stale_idea_rate)), 0.0, 1.0)
    score = clamp(
        (0.28 * spread_quality)
        + (0.18 * slippage_quality)
        + (0.14 * latency_quality)
        + (0.14 * bridge_quality)
        + (0.14 * reject_quality)
        + (0.12 * stale_quality),
        0.0,
        1.0,
    )
    score = clamp(
        score
        - (0.10 if spread_ratio >= 2.5 else 0.0)
        - (0.08 if float(stale_idea_rate) >= 0.30 else 0.0)
        - (0.06 if max(float(fill_delay_ms), float(bridge_latency_ms)) >= 1500.0 else 0.0),
        0.0,
        1.0,
    )
    state = "GOOD"
    if score < 0.45 or (
        spread_ratio >= 2.5
        and float(stale_idea_rate) >= 0.30
        and max(float(fill_delay_ms), float(bridge_latency_ms)) >= 1500.0
    ):
        state = "DEGRADED"
    elif score < 0.65:
        state = "CAUTION"
    return ExecutionQuality(
        state=state,
        score=score,
        spread_anomaly=spread_ratio >= 1.50,
        slippage_anomaly=float(slippage_points) > max(3.0, typical * 0.40),
        bridge_latency_alert=max(float(fill_delay_ms), float(bridge_latency_ms)) >= 1500.0,
        stale_idea_alert=float(stale_idea_rate) >= 0.20,
        components={
            "spread_quality": spread_quality,
            "slippage_quality": slippage_quality,
            "latency_quality": latency_quality,
            "bridge_quality": bridge_quality,
            "reject_quality": reject_quality,
            "stale_quality": stale_quality,
        },
    )


def _session_alignment_score(*, symbol: str, session_name: str, setup_family: str) -> float:
    symbol_key = _normalize_symbol_key(symbol)
    session_key = str(session_name or "").upper()
    family = normalize_strategy_family(setup_family)
    if symbol_key == "XAUUSD" and family == "GRID":
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            return 1.0
        return 0.20
    if symbol_key in ASIA_PRIMARY_PAIRS:
        if session_key in {"TOKYO", "SYDNEY"}:
            return 0.96 if symbol_key != "AUDNZD" else 0.90
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            return 0.54 if symbol_key != "AUDNZD" else 0.48
        return 0.62
    if symbol_key in ASIA_SECONDARY_PAIRS:
        if session_key in {"TOKYO", "SYDNEY"}:
            return 0.88
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            return 0.84
        return 0.66
    if symbol_key == "BTCUSD":
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            return 0.88
        if session_key in {"TOKYO", "SYDNEY"}:
            return 0.74
        return 0.60
    if symbol_key in {"EURUSD", "GBPUSD"}:
        if session_key in {"TOKYO", "SYDNEY"}:
            return 0.62
        if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
            return 0.97
        return 0.66
    if session_key in {"LONDON", "OVERLAP", "NEW_YORK"}:
        return 0.95
    if session_key in {"TOKYO", "SYDNEY"}:
        return 0.72
    return 0.58


def _regime_alignment_score(*, setup_family: str, regime_state: str) -> float:
    family = normalize_strategy_family(setup_family)
    state = str(regime_state or "").upper()
    mapping = {
        "GRID": {
            "QUIET_ACCUMULATION": 0.92,
            "MEAN_REVERSION": 0.95,
            "LIQUIDITY_SWEEP": 0.88,
            "BREAKOUT_COMPRESSION": 0.72,
            "TREND_EXPANSION": 0.55,
            "VOLATILITY_SPIKE": 0.30,
            "NEWS_DISTORTION": 0.15,
            "LOW_LIQUIDITY_DRIFT": 0.45,
        },
        "TREND": {
            "TREND_EXPANSION": 0.96,
            "BREAKOUT_COMPRESSION": 0.86,
            "VOLATILITY_SPIKE": 0.72,
            "LIQUIDITY_SWEEP": 0.68,
            "QUIET_ACCUMULATION": 0.55,
            "MEAN_REVERSION": 0.32,
            "NEWS_DISTORTION": 0.22,
            "LOW_LIQUIDITY_DRIFT": 0.40,
        },
        "RANGE/REVERSION": {
            "MEAN_REVERSION": 0.95,
            "QUIET_ACCUMULATION": 0.80,
            "LIQUIDITY_SWEEP": 0.82,
            "LOW_LIQUIDITY_DRIFT": 0.70,
            "BREAKOUT_COMPRESSION": 0.58,
            "TREND_EXPANSION": 0.28,
            "VOLATILITY_SPIKE": 0.20,
            "NEWS_DISTORTION": 0.10,
        },
        "CRYPTO MOMENTUM": {
            "TREND_EXPANSION": 0.96,
            "VOLATILITY_SPIKE": 0.88,
            "BREAKOUT_COMPRESSION": 0.90,
            "LIQUIDITY_SWEEP": 0.76,
            "LOW_LIQUIDITY_DRIFT": 0.60,
            "QUIET_ACCUMULATION": 0.46,
            "MEAN_REVERSION": 0.34,
            "NEWS_DISTORTION": 0.24,
        },
    }
    selected = mapping.get(family, mapping["TREND"])
    return float(selected.get(state, 0.60))


def _volatility_quality(*, setup_family: str, volatility_state: str) -> float:
    family = normalize_strategy_family(setup_family)
    state = str(volatility_state or "").upper()
    if family == "GRID":
        if state == "COMPRESSION":
            return 0.90
        if state == "BALANCED":
            return 0.82
        if state == "EXPANSION_EXHAUSTION":
            return 0.70
        if state == "SPIKE":
            return 0.25
        return 0.52
    if family in {"TREND", "CRYPTO MOMENTUM"}:
        if state == "EXPANSION_IMMINENT":
            return 0.92
        if state == "BALANCED":
            return 0.84
        if state == "SPIKE":
            return 0.70 if family == "CRYPTO MOMENTUM" else 0.52
        if state == "EXPANSION_EXHAUSTION":
            return 0.34
        return 0.58
    if state == "BALANCED":
        return 0.85
    if state == "COMPRESSION":
        return 0.74
    if state == "EXPANSION_EXHAUSTION":
        return 0.78
    if state == "SPIKE":
        return 0.20
    return 0.60


def _news_quality(news_state: str, decision_confidence: float) -> float:
    state = str(news_state or "").upper()
    confidence = clamp(float(decision_confidence), 0.0, 1.0)
    if state == "NEWS_SAFE":
        return 0.95
    if state == "NEWS_CAUTION":
        return clamp(0.55 + (confidence * 0.25), 0.45, 0.82)
    if state == "NEWS_DISTORTION":
        return 0.20
    if state == "NEWS_BLOCK":
        return 0.0
    return clamp(0.45 + (confidence * 0.20), 0.30, 0.70)


def evaluate_trade_quality(
    *,
    symbol: str,
    session_name: str,
    setup_family: str,
    regime_state: str,
    regime_confidence: float,
    spread_points: float,
    spread_limit: float,
    volatility_state: str,
    liquidity_score: float,
    news_state: str,
    news_confidence: float,
    structure_score: float,
    execution_feasibility: float,
    expected_value_r: float,
    probability: float,
    performance_score: float = 0.55,
    execution_quality_score: float = 0.70,
    pressure_alignment: float = 0.60,
) -> TradeQuality:
    normalized_family = normalize_strategy_family(setup_family)
    inferred_lane = infer_trade_lane(
        symbol=symbol,
        setup_family=normalized_family,
        session_name=session_name,
    )
    session_priority = session_priority_context(
        symbol=symbol,
        lane_name=inferred_lane,
        session_name=session_name,
    )
    spread_limit_value = max(1.0, float(spread_limit))
    spread_quality = clamp(1.0 - (max(0.0, float(spread_points)) / spread_limit_value), 0.0, 1.0)
    volatility_quality = _volatility_quality(setup_family=normalized_family, volatility_state=volatility_state)
    session_alignment = _session_alignment_score(symbol=symbol, session_name=session_name, setup_family=normalized_family)
    regime_alignment = _regime_alignment_score(setup_family=normalized_family, regime_state=regime_state)
    regime_quality = clamp((0.60 * regime_alignment) + (0.40 * clamp(float(regime_confidence), 0.0, 1.0)), 0.0, 1.0)
    structure_quality = clamp(float(structure_score), 0.0, 1.0)
    liquidity_alignment = clamp(float(liquidity_score), 0.0, 1.0)
    execution_quality = clamp(float(execution_quality_score), 0.0, 1.0)
    execution_feasible = clamp(float(execution_feasibility), 0.0, 1.0)
    news_quality = _news_quality(news_state, news_confidence)
    performance_component = clamp(float(performance_score), 0.0, 1.0)
    probability_component = clamp(float(probability), 0.0, 1.0)
    expected_value_component = clamp((float(expected_value_r) + 0.10) / 1.40, 0.0, 1.0)
    pressure_component = clamp(float(pressure_alignment), 0.0, 1.0)
    score = clamp(
        (0.14 * regime_quality)
        + (0.10 * session_alignment)
        + (0.10 * spread_quality)
        + (0.08 * volatility_quality)
        + (0.13 * structure_quality)
        + (0.08 * liquidity_alignment)
        + (0.08 * news_quality)
        + (0.07 * performance_component)
        + (0.10 * probability_component)
        + (0.07 * expected_value_component)
        + (0.05 * execution_quality)
        + (0.05 * execution_feasible)
        + (0.05 * pressure_component),
        0.0,
        1.0,
    )
    band = quality_band_detail(score)
    legacy_band = quality_band(score)
    elite = quality_band_rank(band) >= quality_band_rank("A")
    acceptable = quality_band_rank(band) >= quality_band_rank("B")
    overflow_eligible = (
        quality_band_rank(band) >= quality_band_rank("A")
        and clamp(float(regime_confidence), 0.0, 1.0) >= 0.75
        and execution_quality >= 0.75
        and news_quality >= 0.72
        and spread_quality >= 0.70
    )
    return TradeQuality(
        score=score,
        band=band,
        legacy_band=legacy_band,
        elite=elite,
        acceptable=acceptable,
        should_skip=quality_band_rank(band) == 0,
        overflow_eligible=overflow_eligible,
        size_multiplier=quality_size_multiplier(band),
        lane_name=inferred_lane,
        session_priority_profile=session_priority.session_priority_profile,
        session_priority_multiplier=session_priority.session_priority_multiplier,
        session_native_pair=session_priority.session_native_pair,
        lane_session_priority=session_priority.lane_session_priority,
        pair_priority_rank_in_session=session_priority.pair_priority_rank_in_session,
        components={
            "regime_alignment": regime_quality,
            "session_alignment": session_alignment,
            "spread_quality": spread_quality,
            "volatility_quality": volatility_quality,
            "structure_quality": structure_quality,
            "liquidity_alignment": liquidity_alignment,
            "news_condition": news_quality,
            "recent_symbol_performance": performance_component,
            "setup_confluence": probability_component,
            "expected_value": expected_value_component,
            "execution_quality": execution_quality,
            "execution_feasibility": execution_feasible,
            "pressure_alignment": pressure_component,
            "normalized_family": normalized_family,
            "band_detail": band,
            "legacy_band": legacy_band,
            "session_priority_profile": session_priority.session_priority_profile,
            "session_priority_multiplier": session_priority.session_priority_multiplier,
            "session_native_pair": 1.0 if session_priority.session_native_pair else 0.0,
            "pair_priority_rank_in_session": float(session_priority.pair_priority_rank_in_session),
            "lane_budget_share": float(session_priority.lane_budget_share),
            "quality_floor_edge": float(session_priority.quality_floor_edge),
            "lane_session_priority": session_priority.lane_session_priority,
        },
    )
