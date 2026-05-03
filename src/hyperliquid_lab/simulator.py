from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SimulatedOrder:
    order_id: str
    symbol: str
    side: str
    order_type: str
    quantity: float
    timestamp_utc: pd.Timestamp
    limit_price: float | None = None


@dataclass(frozen=True)
class SimulatedFill:
    price: float
    quantity: float
    fee: float
    liquidity: str
    timestamp_utc: str


@dataclass
class SimulationResult:
    order_id: str
    symbol: str
    side: str
    order_type: str
    requested_qty: float
    filled_qty: float
    unfilled_qty: float
    avg_price: float
    fee: float
    slippage_bps: float
    status: str
    rejection_reason: str
    mid_price: float
    book_timestamp_utc: str
    context: dict[str, Any] = field(default_factory=dict)
    fills: list[SimulatedFill] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fills"] = [asdict(fill) for fill in self.fills]
        return payload


@dataclass
class MarketSimulator:
    maker_fee_rate: float = 0.00015
    taker_fee_rate: float = 0.00045
    max_slippage_bps: float = 10.0
    max_order_book_levels: int = 20
    min_fill_ratio: float = 0.50
    stale_book_timeout_seconds: float = 5.0

    def execute_taker(self, order: SimulatedOrder, book: pd.DataFrame) -> SimulationResult:
        self._validate_order(order, expected_types={"market"})
        clean_book = self._prepare_book(book, order)
        book_ts = pd.Timestamp(clean_book["timestamp_utc"].iloc[0]).tz_convert("UTC")
        stale_reason = self._stale_reason(order, book_ts)
        mid = self._mid_price(clean_book)
        if stale_reason:
            return self._empty_result(order, mid_price=mid, book_timestamp=book_ts, status="rejected", reason=stale_reason)

        target_side = "ask" if order.side.lower() == "buy" else "bid"
        ascending = target_side == "ask"
        levels = (
            clean_book[clean_book["side"].str.lower() == target_side]
            .sort_values("price", ascending=ascending)
            .head(int(self.max_order_book_levels))
        )
        if levels.empty:
            return self._empty_result(order, mid_price=mid, book_timestamp=book_ts, status="rejected", reason="empty_book_side")

        max_price = mid * (1.0 + float(self.max_slippage_bps) / 10000.0)
        min_price = mid * (1.0 - float(self.max_slippage_bps) / 10000.0)
        remaining = float(order.quantity)
        fills: list[SimulatedFill] = []
        for row in levels.itertuples(index=False):
            price = float(row.price)
            if order.side.lower() == "buy" and price > max_price:
                break
            if order.side.lower() == "sell" and price < min_price:
                break
            fill_qty = min(remaining, max(0.0, float(row.size)))
            if fill_qty <= 0.0:
                continue
            fills.append(self._fill(price=price, quantity=fill_qty, liquidity="taker", timestamp=book_ts, fee_rate=self.taker_fee_rate))
            remaining -= fill_qty
            if remaining <= 1e-12:
                break

        filled_qty = sum(fill.quantity for fill in fills)
        if filled_qty <= 0.0:
            return self._empty_result(order, mid_price=mid, book_timestamp=book_ts, status="rejected", reason="slippage_limit_or_no_depth")
        fill_ratio = filled_qty / max(float(order.quantity), 1e-12)
        if fill_ratio < float(self.min_fill_ratio):
            return self._empty_result(order, mid_price=mid, book_timestamp=book_ts, status="rejected", reason="min_fill_ratio_not_met")
        return self._result_from_fills(order, fills, mid_price=mid, book_timestamp=book_ts)

    def execute_maker(self, order: SimulatedOrder, trades: pd.DataFrame | None) -> SimulationResult:
        self._validate_order(order, expected_types={"limit"})
        order_ts = pd.Timestamp(order.timestamp_utc).tz_convert("UTC")
        if order.limit_price is None or float(order.limit_price) <= 0.0:
            return self._empty_result(order, mid_price=0.0, book_timestamp=order_ts, status="rejected", reason="limit_price_required")
        if trades is None or trades.empty:
            return self._empty_result(order, mid_price=0.0, book_timestamp=order_ts, status="open", reason="maker_requires_trade_prints")

        clean = trades.copy()
        required = {"timestamp_utc", "side", "price", "size"}
        missing = required - set(clean.columns)
        if missing:
            raise ValueError(f"trades frame missing columns: {sorted(missing)}")
        clean["timestamp_utc"] = pd.to_datetime(clean["timestamp_utc"], utc=True)
        clean["side"] = clean["side"].astype(str).str.lower()
        clean = clean[clean["timestamp_utc"] >= order_ts].sort_values("timestamp_utc")
        if order.side.lower() == "buy":
            crossing = clean[(clean["side"] == "sell") & (clean["price"].astype(float) <= float(order.limit_price))]
        else:
            crossing = clean[(clean["side"] == "buy") & (clean["price"].astype(float) >= float(order.limit_price))]
        if crossing.empty:
            return self._empty_result(order, mid_price=0.0, book_timestamp=order_ts, status="open", reason="maker_not_crossed_by_trades")

        remaining = float(order.quantity)
        fills: list[SimulatedFill] = []
        for row in crossing.itertuples(index=False):
            fill_qty = min(remaining, max(0.0, float(row.size)))
            if fill_qty <= 0.0:
                continue
            fills.append(
                self._fill(
                    price=float(order.limit_price),
                    quantity=fill_qty,
                    liquidity="maker",
                    timestamp=pd.Timestamp(row.timestamp_utc).tz_convert("UTC"),
                    fee_rate=self.maker_fee_rate,
                )
            )
            remaining -= fill_qty
            if remaining <= 1e-12:
                break

        filled_qty = sum(fill.quantity for fill in fills)
        if filled_qty <= 0.0:
            return self._empty_result(order, mid_price=0.0, book_timestamp=order_ts, status="open", reason="maker_not_filled")
        return self._result_from_fills(order, fills, mid_price=0.0, book_timestamp=order_ts)

    @staticmethod
    def _validate_order(order: SimulatedOrder, *, expected_types: set[str]) -> None:
        if order.side.lower() not in {"buy", "sell"}:
            raise ValueError(f"Unsupported side: {order.side}")
        if order.order_type.lower() not in expected_types:
            raise ValueError(f"Unsupported order type for this simulator path: {order.order_type}")
        if float(order.quantity) <= 0.0:
            raise ValueError("order quantity must be positive")

    @staticmethod
    def _prepare_book(book: pd.DataFrame, order: SimulatedOrder) -> pd.DataFrame:
        required = {"timestamp_utc", "side", "price", "size"}
        missing = required - set(book.columns)
        if missing:
            raise ValueError(f"book frame missing columns: {sorted(missing)}")
        clean = book.copy()
        clean["timestamp_utc"] = pd.to_datetime(clean["timestamp_utc"], utc=True)
        clean["side"] = clean["side"].astype(str).str.lower()
        clean = clean[clean["symbol"].astype(str).str.upper() == order.symbol.upper()] if "symbol" in clean.columns else clean
        if clean.empty:
            raise ValueError("book frame has no rows for order symbol")
        return clean

    def _stale_reason(self, order: SimulatedOrder, book_ts: pd.Timestamp) -> str:
        order_ts = pd.Timestamp(order.timestamp_utc).tz_convert("UTC")
        age = abs((order_ts - book_ts).total_seconds())
        if age > float(self.stale_book_timeout_seconds):
            return "stale_order_book"
        return ""

    @staticmethod
    def _mid_price(book: pd.DataFrame) -> float:
        bids = book[book["side"].str.lower() == "bid"]["price"].astype(float)
        asks = book[book["side"].str.lower() == "ask"]["price"].astype(float)
        if bids.empty or asks.empty:
            raise ValueError("book requires at least one bid and one ask")
        return (float(bids.max()) + float(asks.min())) / 2.0

    @staticmethod
    def _fill(*, price: float, quantity: float, liquidity: str, timestamp: pd.Timestamp, fee_rate: float) -> SimulatedFill:
        fee = float(price) * float(quantity) * float(fee_rate)
        return SimulatedFill(
            price=float(price),
            quantity=float(quantity),
            fee=float(fee),
            liquidity=str(liquidity),
            timestamp_utc=timestamp.isoformat(),
        )

    def _result_from_fills(
        self,
        order: SimulatedOrder,
        fills: list[SimulatedFill],
        *,
        mid_price: float,
        book_timestamp: pd.Timestamp,
    ) -> SimulationResult:
        filled_qty = sum(fill.quantity for fill in fills)
        notional = sum(fill.price * fill.quantity for fill in fills)
        avg_price = notional / max(filled_qty, 1e-12)
        fee = sum(fill.fee for fill in fills)
        unfilled = max(0.0, float(order.quantity) - filled_qty)
        status = "filled" if unfilled <= 1e-12 else "partial"
        slippage = 0.0
        if mid_price > 0.0:
            if order.side.lower() == "buy":
                slippage = (avg_price - mid_price) / mid_price * 10000.0
            else:
                slippage = (mid_price - avg_price) / mid_price * 10000.0
        return SimulationResult(
            order_id=order.order_id,
            symbol=order.symbol.upper(),
            side=order.side.lower(),
            order_type=order.order_type.lower(),
            requested_qty=float(order.quantity),
            filled_qty=float(filled_qty),
            unfilled_qty=float(unfilled),
            avg_price=float(avg_price),
            fee=float(fee),
            slippage_bps=float(slippage),
            status=status,
            rejection_reason="",
            mid_price=float(mid_price),
            book_timestamp_utc=book_timestamp.isoformat(),
            context={"order": asdict(order)},
            fills=fills,
        )

    @staticmethod
    def _empty_result(
        order: SimulatedOrder,
        *,
        mid_price: float,
        book_timestamp: pd.Timestamp,
        status: str,
        reason: str,
    ) -> SimulationResult:
        return SimulationResult(
            order_id=order.order_id,
            symbol=order.symbol.upper(),
            side=order.side.lower(),
            order_type=order.order_type.lower(),
            requested_qty=float(order.quantity),
            filled_qty=0.0,
            unfilled_qty=float(order.quantity),
            avg_price=0.0,
            fee=0.0,
            slippage_bps=0.0,
            status=str(status),
            rejection_reason=str(reason),
            mid_price=float(mid_price),
            book_timestamp_utc=book_timestamp.isoformat(),
            context={"order": asdict(order)},
            fills=[],
        )
