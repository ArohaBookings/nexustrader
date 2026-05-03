from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils import clamp


@dataclass(frozen=True)
class XauSmcDecision:
    allowed: bool
    reason: str
    smc_score: float
    confidence_boost: float
    tags: tuple[str, ...]


def evaluate_xau_smc_setup(
    *,
    setup: str,
    reason: str,
    side: str,
    regime: str,
    probability: float,
    expected_value_r: float,
    confluence_score: float,
    spread_points: float,
    spread_cap_points: float,
    news_status: str,
    session_name: str,
) -> XauSmcDecision:
    setup_text = str(setup or "").upper()
    reason_text = str(reason or "").upper()
    regime_text = str(regime or "").upper()
    session_text = str(session_name or "").upper()
    news_text = str(news_status or "").lower()

    text = f"{setup_text} {reason_text}"
    has_liquidity_sweep = any(token in text for token in ("SWEEP", "LIQUIDITY", "GRAB", "FAKEOUT", "RECLAIM", "STOPHUNT", "STOP_HUNT"))
    has_order_block = any(token in text for token in ("ORDERBLOCK", "ORDER_BLOCK", " OB ", "SUPPLY", "DEMAND"))
    has_breaker_block = any(token in text for token in ("BREAKER", "FAILED_OB", "FAILED ORDERBLOCK"))
    has_fvg = any(token in text for token in ("FVG", "FAIRVALUEGAP", "IMBALANCE", "GAP"))
    has_mitigation = any(token in text for token in ("MITIGATION", "REBALANCE", "FILL_GAP", "FVG_FILL"))
    has_equal_levels = any(token in text for token in ("EQUAL_HIGH", "EQUAL_LOW", "EQH", "EQL", "DOUBLE_TOP", "DOUBLE_BOTTOM"))
    has_displacement = any(token in text for token in ("DISPLACEMENT", "IMPULSE", "EXPLOSIVE", "WIDE_RANGE", "MARUBOZU"))
    has_absorption = any(token in text for token in ("REJECTION", "ENGULF", "PINBAR", "ABSORPTION", "STALL", "DELTA_DIVERGENCE"))
    tags = tuple(
        tag
        for tag, active in (
            ("liquidity_sweep", has_liquidity_sweep),
            ("order_block", has_order_block),
            ("breaker_block", has_breaker_block),
            ("fair_value_gap", has_fvg),
            ("mitigation", has_mitigation),
            ("equal_levels", has_equal_levels),
            ("displacement", has_displacement),
            ("absorption", has_absorption),
        )
        if active
    )

    smc_required = any(token in setup_text for token in ("SMC", "FAKEOUT", "LIQUIDITY", "SWEEP", "FVG", "ORDERBLOCK"))
    base_score = clamp(float(confluence_score), 0.0, 1.0)
    score = base_score
    score += 0.10 if has_liquidity_sweep else 0.0
    score += 0.08 if has_fvg else 0.0
    score += 0.08 if (has_order_block or has_breaker_block) else 0.0
    score += 0.06 if has_displacement else 0.0
    score += 0.05 if has_absorption else 0.0
    score += 0.04 if has_mitigation else 0.0
    score += 0.03 if has_equal_levels else 0.0
    if "RANG" in regime_text and has_liquidity_sweep:
        score += 0.04
    if "TREND" in regime_text and (has_displacement or "BREAKOUT" in setup_text or "RETEST" in setup_text):
        score += 0.03
    if session_text in {"LONDON", "OVERLAP", "NEW_YORK"}:
        score += 0.02
    if float(spread_points) > float(spread_cap_points):
        return XauSmcDecision(False, "spread_too_wide", clamp(score, 0.0, 1.0), 0.0, tags)
    if float(spread_points) > (float(spread_cap_points) * 0.9):
        score -= 0.05
    if "VOLATILE" in regime_text:
        score -= 0.06
    if "blocked" in news_text and float(probability) < 0.86:
        score -= 0.06
    score = clamp(score, 0.0, 1.0)

    min_floor = 0.58
    if session_text in {"TOKYO", "SYDNEY"}:
        min_floor += 0.03
    if "VOLATILE" in regime_text:
        min_floor += 0.02

    smc_component_count = int(
        has_liquidity_sweep
        + has_order_block
        + has_breaker_block
        + has_fvg
        + has_mitigation
        + has_equal_levels
        + has_displacement
        + has_absorption
    )
    if smc_required and smc_component_count < 2:
        return XauSmcDecision(False, "smc_not_confirmed", score, max(0.0, score - base_score), tags)
    if smc_required and smc_component_count == 2 and not (float(probability) >= 0.70 or float(expected_value_r) >= 0.60):
        return XauSmcDecision(False, "smc_not_confirmed", score, max(0.0, score - base_score), tags)
    if smc_required and score < min_floor:
        return XauSmcDecision(False, "smc_not_confirmed", score, max(0.0, score - base_score), tags)
    if smc_required and expected_value_r < 0.20 and probability < 0.60:
        return XauSmcDecision(False, "smc_ev_too_low", score, max(0.0, score - base_score), tags)

    if not smc_required:
        return XauSmcDecision(True, "smc_optional", score, max(0.0, score - base_score), tags)
    return XauSmcDecision(True, "smc_confirmed", score, max(0.0, score - base_score), tags)
