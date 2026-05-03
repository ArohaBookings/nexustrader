from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.bridge_stop_validation import StopValidationInput, SymbolRule, validate_and_normalize_stops
from src.utils import clamp

MIN_DAYTRADE_RR = 1.8


@dataclass(frozen=True)
class TradePlanValidation:
    valid: bool
    reason: str
    normalized_plan: dict[str, Any]


def parse_trade_plan(payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    decision = str(payload.get("decision", "PASS")).upper()
    setup_type = str(payload.get("setup_type", "scalp")).lower()
    side = str(payload.get("side", "")).upper()
    if decision not in {"TAKE", "PASS"}:
        return None
    if setup_type not in {"scalp", "daytrade", "grid_manage"}:
        return None
    if side not in {"BUY", "SELL", ""}:
        return None
    management = payload.get("management_plan", {})
    if not isinstance(management, dict):
        management = {}
    risk_tier = str(payload.get("risk_tier", "NORMAL")).upper()
    if risk_tier not in {"LOW", "NORMAL", "HIGH"}:
        risk_tier = "NORMAL"
    parsed = {
        "decision": decision,
        "setup_type": setup_type,
        "side": side,
        "sl_points": float(payload.get("sl_points", 0.0) or 0.0),
        "tp_points": float(payload.get("tp_points", 0.0) or 0.0),
        "rr_target": float(payload.get("rr_target", 0.0) or 0.0),
        "confidence": clamp(float(payload.get("confidence", 0.0) or 0.0), 0.0, 1.0),
        "expected_value_r": float(payload.get("expected_value_r", 0.0) or 0.0),
        "risk_tier": risk_tier,
        "management_plan": {
            "move_sl_to_be_at_r": management.get("move_sl_to_be_at_r"),
            "trail_after_r": management.get("trail_after_r"),
            "trail_method": str(management.get("trail_method", "atr")).lower(),
            "trail_value": management.get("trail_value"),
            "take_partial_at_r": management.get("take_partial_at_r"),
            "time_stop_minutes": management.get("time_stop_minutes"),
            "early_exit_rules": str(management.get("early_exit_rules", "")),
        },
        "notes": str(payload.get("notes", "")),
    }
    if parsed["management_plan"]["trail_method"] not in {"atr", "structure", "fixed", "none"}:
        parsed["management_plan"]["trail_method"] = "atr"
    return parsed


def validate_trade_plan(
    *,
    plan: dict[str, Any],
    symbol: str,
    side: str,
    entry_price: float,
    spread_points: float,
    spread_cap_points: float,
    symbol_rule: SymbolRule,
    safety_buffer_points: int,
    live_bid: float = 0.0,
    live_ask: float = 0.0,
) -> TradePlanValidation:
    parsed = parse_trade_plan(plan)
    if parsed is None:
        return TradePlanValidation(False, "plan_parse_failed", {})
    if parsed["decision"] == "PASS":
        return TradePlanValidation(True, "pass", parsed)
    resolved_side = str(side or parsed.get("side") or "").upper()
    if resolved_side not in {"BUY", "SELL"}:
        return TradePlanValidation(False, "invalid_side", parsed)
    parsed["side"] = resolved_side
    if float(spread_points) > float(spread_cap_points):
        return TradePlanValidation(False, "spread_cap_exceeded", parsed)
    sl_points = max(0.0, float(parsed.get("sl_points", 0.0)))
    tp_points = max(0.0, float(parsed.get("tp_points", 0.0)))
    rr_target = max(0.0, float(parsed.get("rr_target", 0.0)))
    min_points = float(symbol_rule.min_stop_points + symbol_rule.freeze_points) + max(0.0, float(spread_points)) + max(0, int(safety_buffer_points))
    if sl_points <= 0:
        return TradePlanValidation(False, "sl_points_missing", parsed)
    if tp_points <= 0:
        if rr_target > 0:
            tp_points = max(min_points, sl_points * rr_target)
            parsed["tp_points"] = tp_points
        else:
            return TradePlanValidation(False, "tp_points_missing", parsed)
    if sl_points < min_points:
        sl_points = min_points
        parsed["sl_points"] = sl_points
    if tp_points < min_points:
        tp_points = min_points
        parsed["tp_points"] = tp_points
    rr_realized = float(tp_points) / max(float(sl_points), 1e-9)
    parsed["rr_target"] = rr_realized
    if parsed["setup_type"] == "daytrade" and rr_realized < float(MIN_DAYTRADE_RR):
        return TradePlanValidation(False, "daytrade_rr_below_minimum", parsed)

    point_size = max(float(symbol_rule.point), 1e-9)
    if resolved_side == "BUY":
        sl_price = float(entry_price) - (sl_points * point_size)
        tp_price = float(entry_price) + (tp_points * point_size)
    else:
        sl_price = float(entry_price) + (sl_points * point_size)
        tp_price = float(entry_price) - (tp_points * point_size)
    validation = validate_and_normalize_stops(
        StopValidationInput(
            symbol=symbol,
            side=resolved_side,
            entry_price=float(entry_price),
            sl=sl_price,
            tp=tp_price,
            spread_points=float(spread_points),
            live_bid=float(live_bid or 0.0),
            live_ask=float(live_ask or 0.0),
            safety_buffer_points=max(0, int(safety_buffer_points)),
            allow_tp_none=False,
            push_sl_if_too_close=True,
        ),
        symbol_rule,
    )
    if not validation.valid or validation.normalized_sl is None or validation.normalized_tp is None:
        return TradePlanValidation(False, f"stop_validation_failed:{validation.reason}", parsed)
    parsed["sl_price"] = float(validation.normalized_sl)
    parsed["tp_price"] = float(validation.normalized_tp)
    parsed["sl_points"] = float(validation.sl_distance_points)
    parsed["tp_points"] = float(validation.tp_distance_points)
    parsed["rr_target"] = float(validation.tp_distance_points) / max(float(validation.sl_distance_points), 1e-9)
    parsed["validation_reason"] = validation.reason
    return TradePlanValidation(True, "validated", parsed)
