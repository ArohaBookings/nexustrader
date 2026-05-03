from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import unittest

import numpy as np
import pandas as pd

from src.ai_gate import AIGateDecision
from src.backtest import Backtester, enforce_plausibility
from src.feature_engineering import FeatureEngineer
from src.strategy_engine import SignalCandidate, StrategyEngine
from scripts.router_replay_eval import RouterStrategyEngineAdapter


class _StaticRegime:
    @dataclass
    class _Label:
        label: str = "TRENDING"

    def classify(self, row: pd.Series) -> "_StaticRegime._Label":
        return self._Label()


class _AlwaysApproveGate:
    def evaluate(self, candidate: SignalCandidate, row: pd.Series, regime: str, consecutive_losses: int) -> AIGateDecision:
        return AIGateDecision(
            approved=True,
            probability=0.75,
            expected_value_r=0.25,
            size_multiplier=1.0,
            sl_multiplier=1.0,
            tp_r=candidate.tp_r,
            reason="ok",
        )


class _AlwaysLongStrategy:
    def __init__(self) -> None:
        self.counter = 0

    def generate(
        self,
        features: pd.DataFrame,
        regime: str,
        open_positions: list[dict] | None = None,
        max_positions_per_symbol: int = 1,
    ) -> list[SignalCandidate]:
        self.counter += 1
        ts = features.iloc[-1]["time"]
        return [SignalCandidate(f"long-{self.counter}-{ts}", "AUDIT_FORCE_LONG", "BUY", 0.7, "audit", 1.0, 1.0)]


class _StaticSessionProfile:
    @dataclass
    class _Session:
        session_name: str = "TOKYO"

    def classify(self, current_time):
        return self._Session()


class _StaticRouter:
    def generate(
        self,
        *,
        symbol: str,
        features: pd.DataFrame,
        regime,
        session,
        strategy_engine,
        open_positions,
        max_positions_per_symbol,
        current_time,
    ) -> list[SignalCandidate]:
        return [
            SignalCandidate(
                f"{symbol}-router-1",
                "AUDIT_ROUTER_POOL",
                "BUY",
                0.72,
                "router",
                1.0,
                1.5,
                meta={"strategy_key": f"{symbol}_TEST_STRATEGY"},
            )
        ]


def _make_ohlcv(rows: int, *, start_price: float, drift: float, amplitude: float = 0.2) -> pd.DataFrame:
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    index = pd.date_range(start=t0, periods=rows, freq="5min", tz="UTC")
    x = np.arange(rows, dtype=float)
    close = start_price + (x * drift) + (np.sin(x / 6.0) * amplitude)
    open_ = np.roll(close, 1)
    open_[0] = close[0] - drift
    high = np.maximum(open_, close) + 0.18
    low = np.minimum(open_, close) - 0.18
    volume = 100 + (x % 20) * 5
    spread = 12 + (x % 5)
    return pd.DataFrame(
        {
            "time": index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": volume,
            "spread": spread,
        }
    )


def _make_backtest_frame(rows: int, *, start_price: float, drift: float) -> pd.DataFrame:
    raw = _make_ohlcv(rows, start_price=start_price, drift=drift, amplitude=0.12)
    atr = pd.Series(0.35, index=raw.index)
    frame = pd.DataFrame(
        {
            "time": raw["time"],
            "m5_open": raw["open"],
            "m5_high": raw["high"],
            "m5_low": raw["low"],
            "m5_close": raw["close"],
            "m5_atr_14": atr,
            "m5_spread": raw["spread"],
            "m5_spread_ratio_20": 1.0,
            "m5_atr_pct_of_avg": 1.0,
        }
    )
    return frame


class BacktestAuditTests(unittest.TestCase):
    def test_xau_selection_limit_sums_distinct_grid_cycles(self) -> None:
        adapter = RouterStrategyEngineAdapter(
            symbol_key="XAUUSD",
            strategy_engine=StrategyEngine(symbol="XAUUSD", max_spread_points=100.0),
            strategy_router=_StaticRouter(),
            session_profile=_StaticSessionProfile(),
            regime_detector=_StaticRegime(),
            max_spread_points=100.0,
            top_n=1,
        )
        ranked = [
            (
                SignalCandidate(
                    "xau-grid-primary-1",
                    "XAUUSD_M5_GRID_SCALPER_START",
                    "BUY",
                    0.82,
                    "primary-1",
                    0.8,
                    2.0,
                    strategy_family="GRID",
                    meta={"grid_action": "START", "grid_cycle_id": "primary", "grid_burst_size": 4, "grid_level": 1},
                ),
                {},
            ),
            (
                SignalCandidate(
                    "xau-grid-primary-2",
                    "XAUUSD_M5_GRID_SCALPER_START",
                    "BUY",
                    0.81,
                    "primary-2",
                    0.8,
                    2.0,
                    strategy_family="GRID",
                    meta={"grid_action": "START", "grid_cycle_id": "primary", "grid_burst_size": 4, "grid_level": 2},
                ),
                {},
            ),
            (
                SignalCandidate(
                    "xau-grid-support-1",
                    "XAUUSD_M5_GRID_SCALPER_START",
                    "BUY",
                    0.79,
                    "support-1",
                    0.8,
                    2.0,
                    strategy_family="GRID",
                    meta={"grid_action": "START", "grid_cycle_id": "support", "grid_burst_size": 2, "grid_level": 1},
                ),
                {},
            ),
            (
                SignalCandidate(
                    "xau-grid-support-2",
                    "XAUUSD_M5_GRID_SCALPER_START",
                    "BUY",
                    0.78,
                    "support-2",
                    0.8,
                    2.0,
                    strategy_family="GRID",
                    meta={"grid_action": "START", "grid_cycle_id": "support", "grid_burst_size": 2, "grid_level": 2},
                ),
                {},
            ),
            (
                SignalCandidate(
                    "xau-grid-native-1",
                    "XAUUSD_M5_GRID_SCALPER_START",
                    "BUY",
                    0.77,
                    "native-1",
                    0.8,
                    2.0,
                    strategy_family="GRID",
                    meta={"grid_action": "START", "grid_cycle_id": "native", "grid_burst_size": 1, "grid_level": 1},
                ),
                {},
            ),
            (
                SignalCandidate(
                    "xau-grid-native-2",
                    "XAUUSD_M5_GRID_SCALPER_START",
                    "BUY",
                    0.76,
                    "native-2",
                    0.8,
                    2.0,
                    strategy_family="GRID",
                    meta={"grid_action": "START", "grid_cycle_id": "native-2", "grid_burst_size": 1, "grid_level": 1},
                ),
                {},
            ),
        ]

        limit = adapter._selection_limit(ranked=ranked, session_name="LONDON")  # noqa: SLF001 - targeted replay regression

        self.assertEqual(limit, 6)

    def test_infer_point_size_uses_instrument_override(self) -> None:
        frame = _make_backtest_frame(10, start_price=1.10, drift=0.001)
        frame["instrument_point_size"] = 0.00001
        self.assertEqual(Backtester._infer_point_size(frame), 0.00001)

    def test_effective_spread_prefers_row_override_when_cached_spread_missing(self) -> None:
        frame = _make_backtest_frame(10, start_price=1.10, drift=0.001)
        row = frame.iloc[-1].copy()
        row["m5_spread"] = 0.0
        row["backtest_spread_points"] = 8.0
        backtester = Backtester(
            strategy_engine=_AlwaysLongStrategy(),  # type: ignore[arg-type]
            regime_detector=_StaticRegime(),  # type: ignore[arg-type]
            ai_gate=_AlwaysApproveGate(),  # type: ignore[arg-type]
            spread_points=25.0,
            slippage_points=5.0,
            commission_per_lot=0.0,
            strict_plausibility=False,
        )
        self.assertEqual(backtester._effective_spread_points(row), 8.0)

    def test_router_replay_adapter_preserves_symbol_through_backtester(self) -> None:
        frame = _make_backtest_frame(500, start_price=100.0, drift=0.02)
        adapter = RouterStrategyEngineAdapter(
            symbol_key="AUDJPY",
            strategy_engine=StrategyEngine(symbol="AUDJPY", max_spread_points=100.0),
            strategy_router=_StaticRouter(),
            session_profile=_StaticSessionProfile(),
            regime_detector=_StaticRegime(),
            max_spread_points=100.0,
            top_n=1,
        )
        backtester = Backtester(
            strategy_engine=adapter,  # type: ignore[arg-type]
            regime_detector=_StaticRegime(),  # type: ignore[arg-type]
            ai_gate=_AlwaysApproveGate(),  # type: ignore[arg-type]
            spread_points=0.0,
            slippage_points=0.0,
            commission_per_lot=0.0,
            latency_ms=0,
            max_positions_per_symbol=1,
            max_positions_total=1,
            be_trigger_r=99.0,
            partial1_r=99.0,
            partial2_r=199.0,
            strict_plausibility=False,
        )

        metrics = backtester.run(frame)

        self.assertGreater(len(metrics.get("trades", [])), 0)
        self.assertTrue(all(str(trade.get("symbol") or "") == "AUDJPY" for trade in metrics.get("trades", [])))

    def test_no_lookahead_signal_shift_changes_materially(self) -> None:
        raw = _make_ohlcv(700, start_price=2200.0, drift=0.03, amplitude=0.35)
        engineer = FeatureEngineer()
        features = engineer.build(raw.copy(), raw.copy(), raw.copy(), raw.copy(), raw.copy())

        shifted = features.copy()
        non_time = [column for column in shifted.columns if column != "time"]
        shifted[non_time] = shifted[non_time].shift(1).fillna(0.0)

        strategy = StrategyEngine(symbol="XAUUSD", max_spread_points=100.0)
        compared = 0
        different = 0
        for idx in range(300, len(features) - 1):
            base_candidates = strategy.generate(features.iloc[: idx + 1], regime="TRENDING", open_positions=[], max_positions_per_symbol=10)
            shifted_candidates = strategy.generate(shifted.iloc[: idx + 1], regime="TRENDING", open_positions=[], max_positions_per_symbol=10)

            base_sig = tuple(sorted((item.setup, item.side) for item in base_candidates))
            shifted_sig = tuple(sorted((item.setup, item.side) for item in shifted_candidates))
            if not base_sig and not shifted_sig:
                continue
            compared += 1
            if base_sig != shifted_sig:
                different += 1

        self.assertGreater(compared, 30)
        self.assertGreaterEqual(different / compared, 0.20)

    def test_pnl_sanity_forced_losing_path_is_negative(self) -> None:
        frame = _make_backtest_frame(500, start_price=2200.0, drift=-0.08)
        backtester = Backtester(
            strategy_engine=_AlwaysLongStrategy(),  # type: ignore[arg-type]
            regime_detector=_StaticRegime(),  # type: ignore[arg-type]
            ai_gate=_AlwaysApproveGate(),  # type: ignore[arg-type]
            spread_points=0.0,
            slippage_points=0.0,
            commission_per_lot=0.0,
            latency_ms=0,
            max_positions_per_symbol=1,
            max_positions_total=1,
            be_trigger_r=99.0,
            partial1_r=99.0,
            partial2_r=199.0,
            strict_plausibility=False,
        )
        metrics = backtester.run(frame)
        self.assertLess(float(metrics["net_r"]), 0.0)
        self.assertLess(float(metrics["win_rate"]), 0.5)

    def test_costs_matter_realistic_costs_reduce_net_pnl(self) -> None:
        frame = _make_backtest_frame(700, start_price=2200.0, drift=0.02)

        frictionless = Backtester(
            strategy_engine=_AlwaysLongStrategy(),  # type: ignore[arg-type]
            regime_detector=_StaticRegime(),  # type: ignore[arg-type]
            ai_gate=_AlwaysApproveGate(),  # type: ignore[arg-type]
            spread_points=0.0,
            slippage_points=0.0,
            commission_per_lot=0.0,
            latency_ms=0,
            max_positions_per_symbol=1,
            max_positions_total=1,
            strict_plausibility=False,
        ).run(frame)

        realistic = Backtester(
            strategy_engine=_AlwaysLongStrategy(),  # type: ignore[arg-type]
            regime_detector=_StaticRegime(),  # type: ignore[arg-type]
            ai_gate=_AlwaysApproveGate(),  # type: ignore[arg-type]
            spread_points=25.0,
            slippage_points=5.0,
            commission_per_lot=7.0,
            latency_ms=250,
            max_positions_per_symbol=1,
            max_positions_total=1,
            strict_plausibility=False,
        ).run(frame)

        self.assertLess(float(realistic["net_r"]), float(frictionless["net_r"]))

    def test_win_rate_plausibility_guard(self) -> None:
        suspicious = {"trade_count": 250, "win_rate": 0.90}
        with self.assertRaises(ValueError):
            enforce_plausibility(suspicious, max_win_rate=0.85, min_trades=200, whitelisted=False)

        enforce_plausibility(suspicious, max_win_rate=0.85, min_trades=200, whitelisted=True)


if __name__ == "__main__":
    unittest.main()
