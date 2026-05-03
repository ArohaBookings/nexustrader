from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils import clamp


@dataclass(frozen=True)
class ExecutionValidationResult:
    executable: bool
    status: str
    reason: str
    symbol: str
    lot: float
    min_lot: float
    stop_distance: float
    estimated_loss_usd: float
    min_lot_loss_usd: float
    risk_budget_usd: float
    trade_plan_risk_cap_usd: float
    micro_account_mode: bool
    risk_target_usd: float
    min_lot_tolerance_applied: bool
    required_margin: float
    margin_free: float | None
    diagnostics: dict[str, Any]


def _per_lot_loss_usd(
    *,
    stop_distance: float,
    contract_size: float,
    tick_size: float | None,
    tick_value: float | None,
) -> float:
    distance = max(0.0, float(stop_distance))
    if distance <= 0:
        return 0.0
    if tick_size is not None and tick_value is not None and float(tick_size) > 0 and float(tick_value) > 0:
        return max(0.0, (distance / float(tick_size)) * float(tick_value))
    return max(0.0, distance * max(1.0, float(contract_size)))


def validate_trade_executable(
    *,
    account_equity: float,
    symbol: str,
    lot: float,
    stop_distance: float,
    contract_size: float,
    tick_size: float | None = None,
    tick_value: float | None = None,
    min_lot: float = 0.01,
    margin_free: float | None = None,
    margin_per_lot: float | None = None,
    risk_budget_usd: float | None = None,
    trade_plan_risk_cap_usd: float | None = None,
    projected_open_risk_usd: float = 0.0,
    max_total_risk_usd: float | None = None,
    micro_account_equity_threshold: float = 500.0,
    micro_min_risk_usd: float = 0.5,
    micro_risk_pct: float = 0.005,
    micro_min_lot_risk_multiplier: float = 4.0,
) -> ExecutionValidationResult:
    normalized_symbol = str(symbol).upper()
    normalized_lot = max(0.0, float(lot))
    normalized_min_lot = max(0.0, float(min_lot))
    normalized_stop = max(0.0, float(stop_distance))
    equity = max(0.0, float(account_equity))
    per_lot_loss = _per_lot_loss_usd(
        stop_distance=normalized_stop,
        contract_size=float(contract_size),
        tick_size=tick_size,
        tick_value=tick_value,
    )
    estimated_loss = per_lot_loss * normalized_lot
    min_lot_loss = per_lot_loss * normalized_min_lot
    budget = max(0.0, float(risk_budget_usd or 0.0))
    trade_plan_cap = max(0.0, float(trade_plan_risk_cap_usd if trade_plan_risk_cap_usd is not None else budget))
    micro_mode = equity < max(0.0, float(micro_account_equity_threshold))
    risk_target = max(0.0, max(float(micro_min_risk_usd), equity * max(0.0, float(micro_risk_pct))))

    required_margin = 0.0
    if margin_per_lot is not None and float(margin_per_lot) > 0:
        required_margin = normalized_lot * float(margin_per_lot)

    tolerance_limit = max(0.0, risk_target * max(0.0, float(micro_min_lot_risk_multiplier)))
    broker_min_lot_tolerance_limit = tolerance_limit
    if micro_mode and normalized_lot <= (normalized_min_lot + 1e-9):
        broker_min_lot_tolerance_limit = max(
            broker_min_lot_tolerance_limit,
            max(budget, trade_plan_cap) * 1.15,
        )

    diagnostics = {
        "symbol": normalized_symbol,
        "lot": normalized_lot,
        "min_lot": normalized_min_lot,
        "stop_distance": normalized_stop,
        "per_lot_loss_usd": per_lot_loss,
        "estimated_loss_usd": estimated_loss,
        "min_lot_loss_usd": min_lot_loss,
        "risk_budget_usd": budget,
        "trade_plan_risk_cap_usd": trade_plan_cap,
        "micro_account_mode": micro_mode,
        "risk_target_usd": risk_target,
        "broker_min_lot_tolerance_limit": broker_min_lot_tolerance_limit,
        "required_margin": required_margin,
        "margin_free": float(margin_free) if margin_free is not None else None,
    }

    def _blocked(reason: str, *, tolerance: bool = False) -> ExecutionValidationResult:
        return ExecutionValidationResult(
            executable=False,
            status="EXECUTION_BLOCK",
            reason=reason,
            symbol=normalized_symbol,
            lot=normalized_lot,
            min_lot=normalized_min_lot,
            stop_distance=normalized_stop,
            estimated_loss_usd=estimated_loss,
            min_lot_loss_usd=min_lot_loss,
            risk_budget_usd=budget,
            trade_plan_risk_cap_usd=trade_plan_cap,
            micro_account_mode=micro_mode,
            risk_target_usd=risk_target,
            min_lot_tolerance_applied=tolerance,
            required_margin=required_margin,
            margin_free=float(margin_free) if margin_free is not None else None,
            diagnostics=diagnostics,
        )

    if normalized_lot <= 0:
        return _blocked("lot_zero")
    if normalized_stop <= 0:
        return _blocked("invalid_stop_distance")
    if normalized_lot + 1e-9 < normalized_min_lot:
        return _blocked("lot_below_min_or_margin_too_low")
    if budget <= 0 and trade_plan_cap <= 0:
        return _blocked("risk_budget_zero")
    if margin_free is not None and required_margin > 0 and required_margin > max(0.0, float(margin_free)):
        return _blocked("insufficient_margin")
    if max_total_risk_usd is not None and float(max_total_risk_usd) > 0:
        if (max(0.0, float(projected_open_risk_usd)) + estimated_loss) > float(max_total_risk_usd):
            return _blocked("risk_budget_exceeded")
    tolerance_applied = False

    def _allows_micro_min_lot_tolerance() -> bool:
        if not micro_mode:
            return False
        if normalized_lot > (normalized_min_lot + 1e-9):
            return False
        if broker_min_lot_tolerance_limit <= 0:
            return False
        return min_lot_loss <= broker_min_lot_tolerance_limit

    if trade_plan_cap > 0 and estimated_loss > trade_plan_cap:
        if _allows_micro_min_lot_tolerance():
            tolerance_applied = True
        else:
            return _blocked("trade_plan_risk_exceeded")
    if budget > 0 and estimated_loss > budget:
        if _allows_micro_min_lot_tolerance():
            tolerance_applied = True
        else:
            return _blocked("risk_budget_exceeded")

    return ExecutionValidationResult(
        executable=True,
        status="EXECUTABLE",
        reason="micro_min_lot_tolerance" if tolerance_applied else "ok",
        symbol=normalized_symbol,
        lot=normalized_lot,
        min_lot=normalized_min_lot,
        stop_distance=normalized_stop,
        estimated_loss_usd=estimated_loss,
        min_lot_loss_usd=min_lot_loss,
        risk_budget_usd=budget,
        trade_plan_risk_cap_usd=trade_plan_cap,
        micro_account_mode=micro_mode,
        risk_target_usd=risk_target,
        min_lot_tolerance_applied=tolerance_applied,
        required_margin=required_margin,
        margin_free=float(margin_free) if margin_free is not None else None,
        diagnostics=diagnostics,
    )
