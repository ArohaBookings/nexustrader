from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class ManagedPosition:
    signal_id: str
    ticket: int
    side: str
    symbol: str
    volume: float
    entry_price: float
    current_sl: float
    current_tp: float
    initial_sl: float
    setup: str
    regime: str
    opened_at: datetime
    probability: float
    partial_close_enabled: bool = True
    trailing_enabled: bool = True
    partial_closed: bool = False
    partial2_closed: bool = False
    be_moved: bool = False
    trail_active: bool = False

    @property
    def initial_risk(self) -> float:
        return abs(self.entry_price - self.initial_sl)


@dataclass
class PositionAction:
    action: str
    signal_id: str
    ticket: int
    volume: float | None = None
    new_sl: float | None = None
    new_tp: float | None = None
    symbol: str | None = None
    reason: str = ""


@dataclass
class PositionManager:
    partial_close_r: float
    partial_close_fraction: float
    trail_activation_r: float
    trail_atr_multiple: float
    time_stop_hours: int
    allow_partial_closes: bool = True
    partial_close_r2: float = 2.0
    partial_close_fraction2: float = 0.3
    break_even_trigger_r: float = 0.7
    break_even_buffer_r: float = 0.05
    basket_take_profit_r: float = 3.0
    positions: dict[int, ManagedPosition] = field(default_factory=dict)

    def sync(self, journal_positions: list[dict], mt5_positions: list[dict]) -> list[ManagedPosition]:
        mt5_by_ticket = {int(position["ticket"]): position for position in mt5_positions if "ticket" in position}
        kept: dict[int, ManagedPosition] = {}
        for row in journal_positions:
            ticket_value = row.get("ticket")
            if ticket_value is None:
                continue
            try:
                ticket = int(ticket_value)
            except (TypeError, ValueError):
                continue
            mt5_position = mt5_by_ticket.get(ticket)
            if mt5_position is None:
                continue
            existing = self.positions.get(ticket)
            if existing:
                existing.current_sl = float(mt5_position.get("sl", existing.current_sl))
                existing.current_tp = float(mt5_position.get("tp", existing.current_tp))
                existing.volume = float(mt5_position.get("volume", existing.volume))
                existing.partial_close_enabled = bool(self.allow_partial_closes) and bool(
                    row.get("partial_close_enabled", existing.partial_close_enabled)
                )
                existing.trailing_enabled = bool(row.get("trailing_enabled", existing.trailing_enabled))
                kept[ticket] = existing
                continue
            kept[ticket] = ManagedPosition(
                signal_id=str(row["signal_id"]),
                ticket=ticket,
                side=str(row["side"]).upper(),
                symbol=str(row["symbol"]),
                volume=float(mt5_position.get("volume", row["volume"])),
                entry_price=float(row["entry_price"]),
                current_sl=float(mt5_position.get("sl", row["sl"])),
                current_tp=float(mt5_position.get("tp", row["tp"])),
                initial_sl=float(row["sl"]),
                setup=str(row["setup"]),
                regime=str(row["regime"]),
                opened_at=datetime.fromisoformat(str(row["opened_at"])),
                probability=float(row.get("probability") or 0.0),
                partial_close_enabled=bool(self.allow_partial_closes) and bool(row.get("partial_close_enabled", False)),
                trailing_enabled=bool(row.get("trailing_enabled", True)),
            )
        self.positions = kept
        return list(self.positions.values())

    def plan_actions(
        self,
        market_snapshot: dict[str, dict[str, float]],
        current_time: datetime,
        advice_by_ticket: dict[int, dict[str, Any]] | None = None,
    ) -> list[PositionAction]:
        actions: list[PositionAction] = []
        advice_map = advice_by_ticket or {}
        for position in self.positions.values():
            if position.initial_risk <= 0:
                continue
            symbol_snapshot = market_snapshot.get(position.symbol)
            if not symbol_snapshot:
                continue
            price = float(symbol_snapshot.get("price", 0.0))
            bid = float(symbol_snapshot.get("bid", price))
            ask = float(symbol_snapshot.get("ask", price))
            atr_value = float(symbol_snapshot.get("atr", 0.0))
            if price <= 0:
                continue
            advice = advice_map.get(position.ticket, {})
            effective_be_trigger = self._tighten_lower_is_tighter(self.break_even_trigger_r, advice.get("break_even_r"), floor=0.1)
            effective_partial_r = self._tighten_lower_is_tighter(self.partial_close_r, advice.get("partial_close_r"), floor=0.2)
            effective_trail_multiple = self._effective_trail_multiple(advice)
            if bool(advice.get("close_now", False)):
                actions.append(
                    PositionAction(
                        action="CLOSE_NOW",
                        signal_id=position.signal_id,
                        ticket=position.ticket,
                        reason=str(advice.get("reason", "ai_close_now")),
                    )
                )
                continue
            mark_price = bid if position.side == "BUY" else ask
            profit_r = self._profit_r(position, mark_price)

            if (not position.be_moved) and profit_r >= effective_be_trigger:
                be_buffer = max(position.initial_risk * self.break_even_buffer_r, atr_value * 0.05)
                be_sl = position.entry_price + be_buffer if position.side == "BUY" else position.entry_price - be_buffer
                actions.append(
                    PositionAction(
                        action="MOVE_SL",
                        signal_id=position.signal_id,
                        ticket=position.ticket,
                        new_sl=be_sl,
                        reason="move_to_break_even_buffer",
                    )
                )

            if (
                self.allow_partial_closes
                and position.partial_close_enabled
                and self.partial_close_fraction > 0.0
                and (not position.partial_closed)
                and profit_r >= effective_partial_r
            ):
                actions.append(
                    PositionAction(
                        action="PARTIAL_CLOSE",
                        signal_id=position.signal_id,
                        ticket=position.ticket,
                        volume=round(position.volume * self.partial_close_fraction, 2),
                        reason="partial_close_threshold_reached",
                    )
                )
            if (
                self.allow_partial_closes
                and position.partial_close_enabled
                and self.partial_close_fraction2 > 0.0
                and (not position.partial2_closed)
                and profit_r >= self.partial_close_r2
            ):
                actions.append(
                    PositionAction(
                        action="PARTIAL_CLOSE",
                        signal_id=position.signal_id,
                        ticket=position.ticket,
                        volume=round(position.volume * self.partial_close_fraction2, 2),
                        reason="partial_close_tier2_reached",
                    )
                )

            if position.trailing_enabled and profit_r >= self.trail_activation_r:
                proposed = self._trailing_stop(position, bid, ask, atr_value, effective_trail_multiple)
                if proposed is not None:
                    actions.append(
                        PositionAction(
                            action="MOVE_SL",
                            signal_id=position.signal_id,
                            ticket=position.ticket,
                            new_sl=proposed,
                            reason="trailing_stop_update",
                        )
                    )

            if current_time - position.opened_at >= timedelta(hours=self.time_stop_hours) and abs(profit_r) < 0.25:
                actions.append(
                    PositionAction(
                        action="TIME_STOP",
                        signal_id=position.signal_id,
                        ticket=position.ticket,
                        reason="time_stop_no_progress",
                    )
                )
        return actions

    def plan_basket_actions(self, market_snapshot: dict[str, dict[str, float]]) -> list[PositionAction]:
        actions: list[PositionAction] = []
        symbols = {position.symbol for position in self.positions.values()}
        for symbol in symbols:
            basket_r = self.basket_profit_r(symbol, market_snapshot)
            if basket_r >= self.basket_take_profit_r:
                actions.append(
                    PositionAction(
                        action="CLOSE_ALL_SYMBOL",
                        signal_id=f"basket-{symbol}",
                        ticket=-1,
                        symbol=symbol,
                        reason=f"basket_take_profit_r_{basket_r:.2f}",
                    )
                )
        return actions

    def basket_profit_r(self, symbol: str, market_snapshot: dict[str, dict[str, float]]) -> float:
        snapshot = market_snapshot.get(symbol)
        if not snapshot:
            return 0.0
        price = float(snapshot.get("price", 0.0))
        bid = float(snapshot.get("bid", price))
        ask = float(snapshot.get("ask", price))
        if price <= 0:
            return 0.0
        total = 0.0
        for position in self.positions.values():
            if position.symbol != symbol:
                continue
            total += self._profit_r(position, bid if position.side == "BUY" else ask)
        return total

    def mark_partial(self, ticket: int) -> None:
        if ticket in self.positions:
            position = self.positions[ticket]
            if not position.partial_closed:
                position.partial_closed = True
            elif not position.partial2_closed:
                position.partial2_closed = True

    def update_sl(self, ticket: int, new_sl: float) -> None:
        if ticket in self.positions:
            position = self.positions[ticket]
            if position.side == "BUY":
                if new_sl > position.current_sl:
                    position.current_sl = new_sl
                    position.trail_active = True
                    if new_sl >= position.entry_price:
                        position.be_moved = True
            else:
                if new_sl < position.current_sl:
                    position.current_sl = new_sl
                    position.trail_active = True
                    if new_sl <= position.entry_price:
                        position.be_moved = True

    def _profit_r(self, position: ManagedPosition, price: float) -> float:
        move = price - position.entry_price if position.side == "BUY" else position.entry_price - price
        return move / position.initial_risk

    def _trailing_stop(self, position: ManagedPosition, bid: float, ask: float, atr_value: float, trail_multiple: float) -> float | None:
        distance = atr_value * trail_multiple
        if distance <= 0:
            return None
        if position.side == "BUY":
            candidate = bid - distance
            if candidate > position.current_sl:
                return candidate
            return None
        candidate = ask + distance
        if candidate < position.current_sl:
            return candidate
        return None

    @staticmethod
    def _tighten_lower_is_tighter(base: float, advised: Any, floor: float) -> float:
        if advised is None:
            return base
        try:
            value = float(advised)
        except (TypeError, ValueError):
            return base
        return max(floor, min(base, value))

    def _effective_trail_multiple(self, advice: dict[str, Any]) -> float:
        mode = str(advice.get("trail_mode", "ATR")).upper()
        base = float(self.trail_atr_multiple)
        if mode == "STRUCTURE":
            base *= 0.85
        advised = advice.get("trail_atr_mult")
        if advised is None:
            return max(0.25, base)
        try:
            value = float(advised)
        except (TypeError, ValueError):
            return max(0.25, base)
        # Smaller ATR multiple tightens risk. Never allow a wider trail than base.
        return max(0.25, min(base, value))
