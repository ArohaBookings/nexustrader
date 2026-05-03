from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import pandas as pd

from src.hyperliquid_lab.risk import RiskConfig, daily_loss_limit_hit, stop_loss_triggered
from src.hyperliquid_lab.simulator import MarketSimulator, SimulatedOrder, SimulationResult
from src.hyperliquid_lab.strategy import StrategyOrder


@dataclass
class PaperPortfolio:
    cash: float = 10_000.0
    position_qty: float = 0.0
    entry_price: float = 0.0
    day_start_equity: float = 10_000.0
    halted: bool = False

    def equity(self, mark_price: float) -> float:
        return float(self.cash) + float(self.position_qty) * float(mark_price)


@dataclass
class PaperTradingEngine:
    simulator: MarketSimulator
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    portfolio: PaperPortfolio = field(default_factory=PaperPortfolio)
    event_log: list[dict[str, Any]] = field(default_factory=list)

    def submit_order(self, order: StrategyOrder, book: pd.DataFrame, *, mark_price: float) -> SimulationResult:
        if self.portfolio.halted:
            result = self._rejected(order, "daily_loss_limit_halted")
            self._log("paper_order_rejected", result.to_dict())
            return result
        current_equity = self.portfolio.equity(mark_price)
        if daily_loss_limit_hit(self.portfolio.day_start_equity, current_equity, self.risk_config):
            self.portfolio.halted = True
            result = self._rejected(order, "daily_loss_limit_halted")
            self._log("paper_order_rejected", result.to_dict())
            return result
        if self.portfolio.position_qty > 0.0 and stop_loss_triggered(self.portfolio.entry_price, mark_price, "buy", self.risk_config):
            self.portfolio.halted = True
            result = self._rejected(order, "hard_stop_loss_triggered")
            self._log("paper_order_rejected", result.to_dict())
            return result

        sim_order = SimulatedOrder(
            order_id=f"paper::{order.symbol}::{order.timestamp_utc.isoformat()}::{order.side}",
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            quantity=order.quantity,
            timestamp_utc=order.timestamp_utc,
        )
        result = self.simulator.execute_taker(sim_order, book)
        if result.filled_qty > 0.0:
            self._apply_fill(result)
        self._log("paper_order_result", result.to_dict())
        return result

    def snapshot(self, mark_price: float) -> dict[str, Any]:
        return {
            "cash": float(self.portfolio.cash),
            "position_qty": float(self.portfolio.position_qty),
            "entry_price": float(self.portfolio.entry_price),
            "equity": float(self.portfolio.equity(mark_price)),
            "halted": bool(self.portfolio.halted),
            "events": len(self.event_log),
        }

    def _apply_fill(self, result: SimulationResult) -> None:
        notional = float(result.avg_price) * float(result.filled_qty)
        if result.side == "buy":
            self.portfolio.cash -= notional + float(result.fee)
            previous_qty = self.portfolio.position_qty
            self.portfolio.position_qty += float(result.filled_qty)
            if self.portfolio.position_qty > 0.0:
                self.portfolio.entry_price = (
                    (self.portfolio.entry_price * previous_qty) + notional
                ) / max(self.portfolio.position_qty, 1e-12)
        elif result.side == "sell":
            self.portfolio.cash += notional - float(result.fee)
            self.portfolio.position_qty = max(0.0, self.portfolio.position_qty - float(result.filled_qty))
            if self.portfolio.position_qty <= 1e-12:
                self.portfolio.entry_price = 0.0

    def _log(self, event: str, payload: Mapping[str, Any]) -> None:
        self.event_log.append({"event": event, "payload": dict(payload)})

    @staticmethod
    def _rejected(order: StrategyOrder, reason: str) -> SimulationResult:
        return SimulationResult(
            order_id=f"paper::{order.symbol}::{order.timestamp_utc.isoformat()}::{order.side}",
            symbol=order.symbol,
            side=order.side,
            order_type=order.order_type,
            requested_qty=float(order.quantity),
            filled_qty=0.0,
            unfilled_qty=float(order.quantity),
            avg_price=0.0,
            fee=0.0,
            slippage_bps=0.0,
            status="rejected",
            rejection_reason=reason,
            mid_price=0.0,
            book_timestamp_utc=order.timestamp_utc.isoformat(),
            context={"strategy_order": asdict(order)},
            fills=[],
        )
