from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.hyperliquid_lab.backtesting import BacktestResult, BarCloseBacktester
from src.hyperliquid_lab.strategy import Strategy


@dataclass(frozen=True)
class WalkForwardResult:
    windows: pd.DataFrame
    in_sample_results: list[BacktestResult]
    out_of_sample_results: list[BacktestResult]


def rolling_walk_forward(
    data: pd.DataFrame,
    strategy_factory: Callable[[], Strategy],
    *,
    train_bars: int,
    test_bars: int,
    step_bars: int | None = None,
    initial_cash: float = 10_000.0,
    fee_rate: float = 0.00045,
) -> WalkForwardResult:
    if train_bars <= 0 or test_bars <= 0:
        raise ValueError("train_bars and test_bars must be positive")
    step = int(step_bars or test_bars)
    if step <= 0:
        raise ValueError("step_bars must be positive")
    frame = data.copy().sort_values("close_time_utc").reset_index(drop=True)
    frame["close_time_utc"] = pd.to_datetime(frame["close_time_utc"], utc=True)
    backtester = BarCloseBacktester(initial_cash=initial_cash, fee_rate=fee_rate)
    rows: list[dict[str, object]] = []
    train_results: list[BacktestResult] = []
    test_results: list[BacktestResult] = []
    window_id = 0
    start = 0
    while start + train_bars + test_bars <= len(frame):
        train = frame.iloc[start : start + train_bars].reset_index(drop=True)
        test = frame.iloc[start + train_bars : start + train_bars + test_bars].reset_index(drop=True)
        strategy = strategy_factory()
        train_result = backtester.run(train, strategy.generate_signals(train))
        test_result = backtester.run(test, strategy.generate_signals(test))
        train_results.append(train_result)
        test_results.append(test_result)
        rows.append(
            {
                "window_id": window_id,
                "train_start_utc": train["close_time_utc"].iloc[0],
                "train_end_utc": train["close_time_utc"].iloc[-1],
                "test_start_utc": test["close_time_utc"].iloc[0],
                "test_end_utc": test["close_time_utc"].iloc[-1],
                "in_sample_total_return": train_result.metrics["total_return"],
                "out_of_sample_total_return": test_result.metrics["total_return"],
                "in_sample_sharpe": train_result.metrics["sharpe"],
                "out_of_sample_sharpe": test_result.metrics["sharpe"],
                "divergence_total_return": train_result.metrics["total_return"] - test_result.metrics["total_return"],
            }
        )
        window_id += 1
        start += step
    return WalkForwardResult(pd.DataFrame(rows), train_results, test_results)


def monte_carlo_trade_bootstrap(
    trades: pd.DataFrame,
    *,
    iterations: int = 1000,
    seed: int = 42,
    initial_equity: float = 10_000.0,
) -> pd.DataFrame:
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if trades.empty:
        return pd.DataFrame(
            [{"iterations": iterations, "p05_equity": initial_equity, "p50_equity": initial_equity, "p95_equity": initial_equity, "min_equity": initial_equity, "max_equity": initial_equity}]
        )
    pnl = trades["pnl"].astype(float).to_numpy()
    rng = np.random.default_rng(seed)
    outcomes = []
    for _ in range(int(iterations)):
        sample = rng.choice(pnl, size=len(pnl), replace=True)
        outcomes.append(float(initial_equity + sample.sum()))
    values = np.asarray(outcomes, dtype=float)
    return pd.DataFrame(
        [
            {
                "iterations": int(iterations),
                "p05_equity": float(np.percentile(values, 5)),
                "p50_equity": float(np.percentile(values, 50)),
                "p95_equity": float(np.percentile(values, 95)),
                "min_equity": float(values.min()),
                "max_equity": float(values.max()),
            }
        ]
    )
