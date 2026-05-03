from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import math

import yaml

from src.symbol_universe import normalize_symbol_key, symbol_rule_defaults


def _normalize_symbol_key(value: str) -> str:
    return normalize_symbol_key(value)


@dataclass(frozen=True)
class SymbolRule:
    symbol: str
    digits: int
    tick_size: float
    point: float
    min_stop_points: int
    freeze_points: int
    typical_spread_points: int
    max_slippage_points: int
    tick_value: float
    contract_size: float


@dataclass(frozen=True)
class StopValidationInput:
    symbol: str
    side: str
    entry_price: float
    sl: float
    tp: float
    spread_points: float
    live_bid: float = 0.0
    live_ask: float = 0.0
    safety_buffer_points: int = 5
    allow_tp_none: bool = True
    push_sl_if_too_close: bool = True


@dataclass(frozen=True)
class StopValidationResult:
    valid: bool
    symbol_key: str
    normalized_sl: float | None
    normalized_tp: float | None
    reason: str
    min_required_points: float
    sl_distance_points: float
    tp_distance_points: float
    rule: SymbolRule
    live_bid: float = 0.0
    live_ask: float = 0.0
    market_geometry_used: bool = False

    def snapshot(self, entry_price: float) -> dict[str, Any]:
        payload = {
            "valid": self.valid,
            "symbol_key": self.symbol_key,
            "entry_price": float(entry_price),
            "normalized_sl": self.normalized_sl,
            "normalized_tp": self.normalized_tp,
            "reason": self.reason,
            "min_required_points": float(self.min_required_points),
            "sl_distance_points": float(self.sl_distance_points),
            "tp_distance_points": float(self.tp_distance_points),
            "live_bid": float(self.live_bid),
            "live_ask": float(self.live_ask),
            "market_geometry_used": bool(self.market_geometry_used),
            "rule": asdict(self.rule),
        }
        return payload


def _default_symbol_rules() -> dict[str, SymbolRule]:
    def _rule(
        symbol: str,
        *,
        digits: int,
        tick_size: float,
        point: float,
        min_stop_points: int,
        freeze_points: int,
        typical_spread_points: int,
        max_slippage_points: int,
        tick_value: float,
        contract_size: float,
    ) -> SymbolRule:
        return SymbolRule(
            symbol=symbol,
            digits=int(digits),
            tick_size=float(tick_size),
            point=float(point),
            min_stop_points=int(min_stop_points),
            freeze_points=int(freeze_points),
            typical_spread_points=int(typical_spread_points),
            max_slippage_points=int(max_slippage_points),
            tick_value=float(tick_value),
            contract_size=float(contract_size),
        )

    return {
        "XAUUSD": _rule(
            "XAUUSD",
            digits=2,
            tick_size=0.01,
            point=0.01,
            min_stop_points=120,
            freeze_points=30,
            typical_spread_points=35,
            max_slippage_points=50,
            tick_value=1.0,
            contract_size=100.0,
        ),
        "EURUSD": _rule(
            "EURUSD",
            digits=5,
            tick_size=0.00001,
            point=0.00001,
            min_stop_points=40,
            freeze_points=20,
            typical_spread_points=15,
            max_slippage_points=25,
            tick_value=1.0,
            contract_size=100000.0,
        ),
        "GBPUSD": _rule(
            "GBPUSD",
            digits=5,
            tick_size=0.00001,
            point=0.00001,
            min_stop_points=45,
            freeze_points=20,
            typical_spread_points=18,
            max_slippage_points=28,
            tick_value=1.0,
            contract_size=100000.0,
        ),
        "USDJPY": _rule(
            "USDJPY",
            digits=3,
            tick_size=0.001,
            point=0.001,
            min_stop_points=45,
            freeze_points=20,
            typical_spread_points=16,
            max_slippage_points=25,
            tick_value=0.9,
            contract_size=100000.0,
        ),
        "AUDJPY": _rule(
            "AUDJPY",
            digits=3,
            tick_size=0.001,
            point=0.001,
            min_stop_points=55,
            freeze_points=20,
            typical_spread_points=95,
            max_slippage_points=120,
            tick_value=0.9,
            contract_size=100000.0,
        ),
        "NZDJPY": _rule(
            "NZDJPY",
            digits=3,
            tick_size=0.001,
            point=0.001,
            min_stop_points=55,
            freeze_points=20,
            typical_spread_points=22,
            max_slippage_points=35,
            tick_value=0.9,
            contract_size=100000.0,
        ),
        "EURJPY": _rule(
            "EURJPY",
            digits=3,
            tick_size=0.001,
            point=0.001,
            min_stop_points=50,
            freeze_points=20,
            typical_spread_points=24,
            max_slippage_points=38,
            tick_value=0.9,
            contract_size=100000.0,
        ),
        "GBPJPY": _rule(
            "GBPJPY",
            digits=3,
            tick_size=0.001,
            point=0.001,
            min_stop_points=60,
            freeze_points=20,
            typical_spread_points=28,
            max_slippage_points=45,
            tick_value=0.9,
            contract_size=100000.0,
        ),
        "AUDNZD": _rule(
            "AUDNZD",
            digits=5,
            tick_size=0.00001,
            point=0.00001,
            min_stop_points=55,
            freeze_points=20,
            typical_spread_points=140,
            max_slippage_points=180,
            tick_value=1.0,
            contract_size=100000.0,
        ),
        "EURGBP": _rule(
            "EURGBP",
            digits=5,
            tick_size=0.00001,
            point=0.00001,
            min_stop_points=45,
            freeze_points=20,
            typical_spread_points=18,
            max_slippage_points=28,
            tick_value=1.0,
            contract_size=100000.0,
        ),
        "BTCUSD": _rule(
            "BTCUSD",
            digits=2,
            tick_size=0.01,
            point=0.01,
            min_stop_points=600,
            freeze_points=100,
            typical_spread_points=120,
            max_slippage_points=120,
            tick_value=1.0,
            contract_size=1.0,
        ),
        "NAS100": _rule(
            "NAS100",
            digits=1,
            tick_size=0.1,
            point=0.1,
            min_stop_points=80,
            freeze_points=20,
            typical_spread_points=60,
            max_slippage_points=90,
            tick_value=1.0,
            contract_size=1.0,
        ),
        "USOIL": _rule(
            "USOIL",
            digits=2,
            tick_size=0.01,
            point=0.01,
            min_stop_points=150,
            freeze_points=40,
            typical_spread_points=8,
            max_slippage_points=20,
            tick_value=1.0,
            contract_size=100.0,
        ),
    }


def load_symbol_rules(path: Path | None = None) -> dict[str, SymbolRule]:
    rules = _default_symbol_rules()
    if path is None or (not path.exists()):
        return rules
    try:
        payload = yaml.safe_load(path.read_text()) or {}
    except Exception:
        return rules
    if not isinstance(payload, dict):
        return rules
    symbols = payload.get("symbols")
    if not isinstance(symbols, dict):
        return rules
    for key, spec in symbols.items():
        if not isinstance(spec, dict):
            continue
        symbol_key = _normalize_symbol_key(str(key))
        default = rules.get(symbol_key, next(iter(rules.values())))
        digits = int(spec.get("digits", default.digits))
        tick_size = float(spec.get("tick_size", default.tick_size))
        point = float(spec.get("point", default.point if default.point > 0 else tick_size))
        rules[symbol_key] = SymbolRule(
            symbol=str(spec.get("symbol", symbol_key)),
            digits=digits,
            tick_size=max(tick_size, 10 ** (-max(0, digits))),
            point=max(point, 10 ** (-max(0, digits))),
            min_stop_points=int(spec.get("min_stop_points", default.min_stop_points)),
            freeze_points=int(spec.get("freeze_points", default.freeze_points)),
            typical_spread_points=int(spec.get("typical_spread_points", default.typical_spread_points)),
            max_slippage_points=int(spec.get("max_slippage_points", default.max_slippage_points)),
            tick_value=float(spec.get("tick_value", default.tick_value)),
            contract_size=float(spec.get("contract_size", default.contract_size)),
        )
    return rules


def resolve_symbol_rule(symbol: str, rules: dict[str, SymbolRule]) -> SymbolRule:
    symbol_key = _normalize_symbol_key(symbol)
    if symbol_key in rules:
        return rules[symbol_key]
    if any(token in symbol_key for token in ("NAS", "NQ", "USTEC", "US100")) and "NAS100" in rules:
        return rules["NAS100"]
    if any(token in symbol_key for token in ("OIL", "XTI", "WTI", "CL")) and "USOIL" in rules:
        return rules["USOIL"]
    if symbol_key.endswith("JPY") and "USDJPY" in rules:
        return rules["USDJPY"]
    if "USD" in symbol_key and len(symbol_key) >= 6 and "EURUSD" in rules:
        return rules["EURUSD"]
    inferred = symbol_rule_defaults(symbol_key)
    return SymbolRule(
        symbol=str(symbol_key),
        digits=int(inferred.get("digits", 2)),
        tick_size=float(inferred.get("trade_tick_size", inferred.get("point", 0.01))),
        point=float(inferred.get("point", inferred.get("trade_tick_size", 0.01))),
        min_stop_points=int(inferred.get("min_stop_points", 45)),
        freeze_points=int(inferred.get("freeze_points", 20)),
        typical_spread_points=int(inferred.get("typical_spread_points", 18)),
        max_slippage_points=int(inferred.get("max_slippage_points", 28)),
        tick_value=float(inferred.get("trade_tick_value", 1.0)),
        contract_size=float(inferred.get("trade_contract_size", 1.0)),
    )


def _normalize_price(value: float, tick_size: float, digits: int) -> float:
    if tick_size <= 0:
        return round(float(value), max(0, int(digits)))
    ticks = round(float(value) / tick_size)
    normalized = ticks * tick_size
    return round(normalized, max(0, int(digits)))


def validate_and_normalize_stops(payload: StopValidationInput, rule: SymbolRule) -> StopValidationResult:
    symbol_key = _normalize_symbol_key(payload.symbol)
    side = str(payload.side or "").upper()
    entry = float(payload.entry_price or 0.0)
    live_bid = float(payload.live_bid or 0.0)
    live_ask = float(payload.live_ask or 0.0)
    market_geometry_used = bool(live_bid > 0.0 and live_ask > 0.0)
    if side not in {"BUY", "SELL"}:
        return StopValidationResult(
            valid=False,
            symbol_key=symbol_key,
            normalized_sl=None,
            normalized_tp=None,
            reason="invalid_side",
            min_required_points=0.0,
            sl_distance_points=0.0,
            tp_distance_points=0.0,
            rule=rule,
            live_bid=live_bid,
            live_ask=live_ask,
            market_geometry_used=market_geometry_used,
        )
    if entry <= 0:
        return StopValidationResult(
            valid=False,
            symbol_key=symbol_key,
            normalized_sl=None,
            normalized_tp=None,
            reason="invalid_entry_price",
            min_required_points=0.0,
            sl_distance_points=0.0,
            tp_distance_points=0.0,
            rule=rule,
            live_bid=live_bid,
            live_ask=live_ask,
            market_geometry_used=market_geometry_used,
        )

    point = max(rule.point, 1e-9)
    spread_points = max(float(payload.spread_points or 0.0), float(rule.typical_spread_points))
    tick = max(rule.tick_size, point)
    digits = int(rule.digits)
    safety_buffer_points = max(0.0, float(payload.safety_buffer_points))
    min_gap_points = max(float(rule.min_stop_points), float(rule.freeze_points)) + safety_buffer_points
    min_required_points = min_gap_points if market_geometry_used else (float(rule.min_stop_points + rule.freeze_points) + spread_points + safety_buffer_points)
    min_required_price = max(tick, min_required_points * point)
    min_gap_price = max(tick, min_gap_points * point)
    sl = float(payload.sl or 0.0)
    tp = float(payload.tp or 0.0)

    if market_geometry_used:
        if side == "BUY":
            sl_reference = live_bid
            tp_reference = live_ask
            desired_sl = sl_reference - min_gap_price
            desired_tp = tp_reference + min_gap_price
        else:
            sl_reference = live_ask
            tp_reference = live_bid
            desired_sl = sl_reference + min_gap_price
            desired_tp = tp_reference - min_gap_price
    else:
        sl_reference = entry
        tp_reference = entry
        if side == "BUY":
            desired_sl = entry - min_required_price
            desired_tp = entry + min_required_price
        else:
            desired_sl = entry + min_required_price
            desired_tp = entry - min_required_price

    if side == "BUY":
        if sl <= 0 or sl >= entry:
            sl = desired_sl
        if sl >= sl_reference or sl > desired_sl:
            if payload.push_sl_if_too_close:
                sl = desired_sl
            else:
                return StopValidationResult(False, symbol_key, None, None, "sl_too_close_buy", min_required_points, 0.0, 0.0, rule)
        if tp <= tp_reference:
            tp = desired_tp
        if tp < desired_tp:
            if payload.allow_tp_none:
                tp = desired_tp
            else:
                return StopValidationResult(False, symbol_key, None, None, "tp_too_close_buy", min_required_points, 0.0, 0.0, rule)
    else:
        if sl <= 0 or sl <= entry:
            sl = desired_sl
        if sl <= sl_reference or sl < desired_sl:
            if payload.push_sl_if_too_close:
                sl = desired_sl
            else:
                return StopValidationResult(False, symbol_key, None, None, "sl_too_close_sell", min_required_points, 0.0, 0.0, rule)
        if tp >= tp_reference or tp <= 0:
            tp = desired_tp
        if tp > desired_tp:
            if payload.allow_tp_none:
                tp = desired_tp
            else:
                return StopValidationResult(False, symbol_key, None, None, "tp_too_close_sell", min_required_points, 0.0, 0.0, rule)

    sl_norm = _normalize_price(sl, tick, digits)
    tp_norm: float | None = _normalize_price(tp, tick, digits) if tp > 0 else None

    if side == "BUY":
        if sl_norm >= sl_reference:
            sl_norm = _normalize_price(sl_reference - max(min_gap_price if market_geometry_used else min_required_price, tick), tick, digits)
        if tp_norm is not None and tp_norm <= tp_reference:
            tp_norm = _normalize_price(tp_reference + max(min_gap_price if market_geometry_used else min_required_price, tick), tick, digits)
    else:
        if sl_norm <= sl_reference:
            sl_norm = _normalize_price(sl_reference + max(min_gap_price if market_geometry_used else min_required_price, tick), tick, digits)
        if tp_norm is not None and tp_norm >= tp_reference:
            tp_norm = _normalize_price(tp_reference - max(min_gap_price if market_geometry_used else min_required_price, tick), tick, digits)

    sl_distance_points = abs(entry - sl_norm) / point
    tp_distance_points = abs(entry - tp_norm) / point if tp_norm is not None else 0.0
    if sl_distance_points + 1e-9 < min_required_points:
        return StopValidationResult(
            valid=False,
            symbol_key=symbol_key,
            normalized_sl=sl_norm,
            normalized_tp=tp_norm,
            reason="sl_distance_below_required",
            min_required_points=min_required_points,
            sl_distance_points=sl_distance_points,
            tp_distance_points=tp_distance_points,
            rule=rule,
            live_bid=live_bid,
            live_ask=live_ask,
            market_geometry_used=market_geometry_used,
        )
    if tp_norm is not None and tp_distance_points + 1e-9 < min_required_points:
        return StopValidationResult(
            valid=False,
            symbol_key=symbol_key,
            normalized_sl=sl_norm,
            normalized_tp=tp_norm,
            reason="tp_distance_below_required",
            min_required_points=min_required_points,
            sl_distance_points=sl_distance_points,
            tp_distance_points=tp_distance_points,
            rule=rule,
            live_bid=live_bid,
            live_ask=live_ask,
            market_geometry_used=market_geometry_used,
        )

    return StopValidationResult(
        valid=True,
        symbol_key=symbol_key,
        normalized_sl=sl_norm,
        normalized_tp=tp_norm,
        reason="validated",
        min_required_points=min_required_points,
        sl_distance_points=sl_distance_points,
        tp_distance_points=tp_distance_points,
        rule=rule,
        live_bid=live_bid,
        live_ask=live_ask,
        market_geometry_used=market_geometry_used,
    )


def estimate_loss_usd(entry_price: float, stop_price: float, lot: float, rule: SymbolRule) -> float:
    stop_distance = abs(float(entry_price) - float(stop_price))
    if stop_distance <= 0 or lot <= 0:
        return 0.0
    tick_size = max(rule.tick_size, 1e-9)
    tick_value = max(rule.tick_value, 0.0)
    if tick_value > 0:
        per_lot = (stop_distance / tick_size) * tick_value
    else:
        per_lot = stop_distance * max(rule.contract_size, 1.0)
    return max(0.0, per_lot * float(lot))
