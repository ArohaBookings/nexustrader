from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    metrics: dict[str, float]
    fill_model: str = "bar_close_baseline"


@dataclass(frozen=True)
class BarCloseBacktester:
    initial_cash: float = 10_000.0
    fee_rate: float = 0.00045
    periods_per_year: int = 365 * 24 * 60

    def run(self, data: pd.DataFrame, signals: pd.DataFrame) -> BacktestResult:
        required_data = {"close_time_utc", "close", "symbol"}
        required_signals = {"signal_time_utc", "close", "symbol", "target_position", "signal"}
        missing_data = required_data - set(data.columns)
        missing_signals = required_signals - set(signals.columns)
        if missing_data:
            raise ValueError(f"data frame missing columns: {sorted(missing_data)}")
        if missing_signals:
            raise ValueError(f"signals frame missing columns: {sorted(missing_signals)}")

        frame = signals.copy().sort_values("signal_time_utc").reset_index(drop=True)
        frame["signal_time_utc"] = pd.to_datetime(frame["signal_time_utc"], utc=True)
        cash = float(self.initial_cash)
        qty = 0.0
        entry_time: pd.Timestamp | None = None
        entry_price = 0.0
        equity_rows: list[dict[str, object]] = []
        trade_rows: list[dict[str, object]] = []

        for row in frame.itertuples(index=False):
            price = float(row.close)
            signal = str(row.signal)
            timestamp = pd.Timestamp(row.signal_time_utc).tz_convert("UTC")
            if signal == "enter_long" and qty <= 0.0:
                gross_notional = cash
                fee = gross_notional * float(self.fee_rate)
                net_notional = max(0.0, gross_notional - fee)
                qty = net_notional / price if price > 0.0 else 0.0
                cash -= net_notional + fee
                entry_time = timestamp
                entry_price = price
            elif signal == "exit_long" and qty > 0.0:
                gross_notional = qty * price
                fee = gross_notional * float(self.fee_rate)
                cash += gross_notional - fee
                pnl = (price - entry_price) * qty - fee
                trade_rows.append(
                    {
                        "symbol": str(row.symbol).upper(),
                        "entry_time_utc": entry_time,
                        "exit_time_utc": timestamp,
                        "entry_price": entry_price,
                        "exit_price": price,
                        "quantity": qty,
                        "pnl": pnl,
                        "return_pct": (price / entry_price - 1.0) if entry_price > 0.0 else 0.0,
                        "duration_seconds": (timestamp - entry_time).total_seconds() if entry_time is not None else 0.0,
                    }
                )
                qty = 0.0
                entry_time = None
                entry_price = 0.0
            equity_rows.append(
                {
                    "timestamp_utc": timestamp,
                    "cash": cash,
                    "position_qty": qty,
                    "mark_price": price,
                    "equity": cash + qty * price,
                }
            )

        equity_curve = pd.DataFrame(equity_rows)
        trades = pd.DataFrame(trade_rows)
        return BacktestResult(equity_curve=equity_curve, trades=trades, metrics=performance_metrics(equity_curve, trades, self.periods_per_year))


def performance_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame, periods_per_year: int) -> dict[str, float]:
    if equity_curve.empty:
        return {"total_return": 0.0, "sharpe": 0.0, "max_drawdown": 0.0, "win_rate": 0.0, "avg_trade_duration_seconds": 0.0}
    equity = equity_curve["equity"].astype(float)
    returns = equity.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    volatility = float(returns.std(ddof=0))
    sharpe = float((returns.mean() / volatility) * np.sqrt(periods_per_year)) if volatility > 0.0 else 0.0
    peak = equity.cummax()
    drawdown = (equity / peak - 1.0).fillna(0.0)
    wins = 0.0
    avg_duration = 0.0
    if not trades.empty:
        wins = float((trades["pnl"].astype(float) > 0.0).mean())
        avg_duration = float(trades["duration_seconds"].astype(float).mean())
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if float(equity.iloc[0]) != 0.0 else 0.0,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": wins,
        "avg_trade_duration_seconds": avg_duration,
    }


def buy_and_hold_benchmark(data: pd.DataFrame, initial_cash: float = 10_000.0, fee_rate: float = 0.00045) -> BacktestResult:
    required = {"close_time_utc", "close", "symbol"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"data frame missing columns: {sorted(missing)}")
    frame = data.copy().sort_values("close_time_utc").reset_index(drop=True)
    frame["close_time_utc"] = pd.to_datetime(frame["close_time_utc"], utc=True)
    if frame.empty:
        return BacktestResult(pd.DataFrame(), pd.DataFrame(), performance_metrics(pd.DataFrame(), pd.DataFrame(), 365 * 24 * 60))
    first = float(frame["close"].iloc[0])
    qty = (float(initial_cash) * (1.0 - float(fee_rate))) / first
    equity_curve = pd.DataFrame(
        {
            "timestamp_utc": frame["close_time_utc"],
            "cash": [0.0] * len(frame),
            "position_qty": [qty] * len(frame),
            "mark_price": frame["close"].astype(float),
            "equity": frame["close"].astype(float) * qty,
        }
    )
    trades = pd.DataFrame()
    return BacktestResult(equity_curve, trades, performance_metrics(equity_curve, trades, 365 * 24 * 60), fill_model="buy_and_hold")
