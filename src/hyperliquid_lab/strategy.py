from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import pandas as pd


@dataclass(frozen=True)
class StrategyOrder:
    symbol: str
    side: str
    order_type: str
    quantity: float
    reason: str
    timestamp_utc: pd.Timestamp


@dataclass(frozen=True)
class Strategy(ABC):
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return signals using only information available at each bar close."""

    @abstractmethod
    def size_position(self, signal: Mapping[str, Any], portfolio: Mapping[str, Any]) -> StrategyOrder | None:
        """Return the order implied by a signal and current portfolio state."""


@dataclass(frozen=True)
class MovingAverageCrossoverStrategy(Strategy):
    short_window: int = 20
    long_window: int = 50
    risk_fraction: float = 0.10

    def __init__(self, short_window: int = 20, long_window: int = 50, risk_fraction: float = 0.10) -> None:
        if int(short_window) <= 0 or int(long_window) <= 0:
            raise ValueError("moving-average windows must be positive")
        if int(short_window) >= int(long_window):
            raise ValueError("short_window must be less than long_window")
        if not 0.0 < float(risk_fraction) <= 1.0:
            raise ValueError("risk_fraction must be in (0, 1]")
        object.__setattr__(self, "name", "moving_average_crossover")
        object.__setattr__(
            self,
            "parameters",
            {"short_window": int(short_window), "long_window": int(long_window), "risk_fraction": float(risk_fraction)},
        )
        object.__setattr__(self, "short_window", int(short_window))
        object.__setattr__(self, "long_window", int(long_window))
        object.__setattr__(self, "risk_fraction", float(risk_fraction))

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        required = {"close_time_utc", "symbol", "close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"data frame missing columns: {sorted(missing)}")
        frame = data.copy().sort_values("close_time_utc").reset_index(drop=True)
        frame["close_time_utc"] = pd.to_datetime(frame["close_time_utc"], utc=True)
        frame["close"] = frame["close"].astype(float)
        if "source" not in frame.columns:
            frame["source"] = ""
        if "venue" not in frame.columns:
            frame["venue"] = ""
        if "timeframe" not in frame.columns:
            frame["timeframe"] = ""
        frame["short_ma"] = frame["close"].rolling(self.short_window, min_periods=self.short_window).mean()
        frame["long_ma"] = frame["close"].rolling(self.long_window, min_periods=self.long_window).mean()
        frame["target_position"] = (frame["short_ma"] > frame["long_ma"]).astype(int)
        frame.loc[frame["long_ma"].isna(), "target_position"] = 0
        frame["previous_target_position"] = frame["target_position"].shift(1).fillna(0).astype(int)
        frame["signal"] = "hold"
        frame.loc[
            (frame["target_position"] == 1) & (frame["previous_target_position"] == 0),
            "signal",
        ] = "enter_long"
        frame.loc[
            (frame["target_position"] == 0) & (frame["previous_target_position"] == 1),
            "signal",
        ] = "exit_long"
        return frame[
            [
                "source",
                "venue",
                "symbol",
                "timeframe",
                "close_time_utc",
                "close",
                "short_ma",
                "long_ma",
                "signal",
                "target_position",
            ]
        ].rename(columns={"close_time_utc": "signal_time_utc"})

    def size_position(self, signal: Mapping[str, Any], portfolio: Mapping[str, Any]) -> StrategyOrder | None:
        signal_name = str(signal.get("signal", "hold"))
        if signal_name == "hold":
            return None
        price = float(signal["close"])
        if price <= 0.0:
            raise ValueError("signal close price must be positive")
        equity = float(portfolio.get("equity", 0.0))
        current_qty = float(portfolio.get("position_qty", 0.0))
        symbol = str(signal["symbol"]).upper()
        timestamp = pd.Timestamp(signal["signal_time_utc"]).tz_convert("UTC")
        if signal_name == "enter_long" and current_qty <= 0.0:
            quantity = (equity * float(self.risk_fraction)) / price
            return StrategyOrder(symbol, "buy", "market", float(quantity), signal_name, timestamp)
        if signal_name == "exit_long" and current_qty > 0.0:
            return StrategyOrder(symbol, "sell", "market", current_qty, signal_name, timestamp)
        return None
