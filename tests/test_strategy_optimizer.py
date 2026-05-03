from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.strategy_optimizer import StrategyOptimizer


class StrategyOptimizerTests(unittest.TestCase):
    def test_summary_counts_registered_strategies_honestly(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
            )
            optimizer.register_strategies(["XAUUSD_M5_GRID", "BTC_TREND_SCALP"])
            summary = optimizer.summary()
            self.assertEqual(int(summary.get("strategy_count", 0)), 2)
            self.assertEqual(sorted(summary.get("strategies_registered", [])), ["BTC_TREND_SCALP", "XAUUSD_M5_GRID"])
            self.assertEqual(summary.get("strategies_with_history", []), [])

    def test_optimizer_records_trades_and_updates_adjustments(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=10,
                low_win_rate_threshold=0.45,
                high_win_rate_threshold=0.60,
                adjustment_step_pct=0.10,
            )
            for index in range(10):
                optimizer.record_trade(
                    {
                        "timestamp_utc": f"2026-03-06T10:{index:02d}:00+00:00",
                        "symbol": "XAUUSD",
                        "strategy": "XAUUSD_M5_GRID",
                        "setup": "XAUUSD_M5_GRID_SCALPER_START",
                        "side": "BUY",
                        "pnl_r": -0.5,
                        "pnl_money": -0.3,
                    }
                )

            summary = optimizer.summary()
            strategy = summary.get("strategies", {}).get("XAUUSD_M5_GRID", {})
            adjustments = strategy.get("adjustments", {})
            self.assertEqual(int(strategy.get("trade_count", 0)), 10)
            self.assertEqual(int(strategy.get("trades", 0)), 10)
            self.assertEqual(int(strategy.get("wins", 0)), 0)
            self.assertEqual(int(strategy.get("losses", 0)), 10)
            self.assertIn("current_probability_floor_mult", strategy)
            self.assertIn("current_confluence_floor_mult", strategy)
            self.assertIn("current_candidate_sensitivity_mult", strategy)
            self.assertGreater(float(adjustments.get("probability_floor_mult", 1.0)), 1.0)
            self.assertGreater(float(adjustments.get("confluence_floor_mult", 1.0)), 1.0)
            self.assertLess(float(adjustments.get("candidate_sensitivity_mult", 1.0)), 1.0)
            self.assertGreater(float(adjustments.get("xau_grid_spacing_mult", 1.0)), 1.0)
            self.assertLess(float(adjustments.get("xau_grid_probe_aggression", 1.0)), 1.0)
            self.assertIn("recommended_spacing_multiplier", strategy)

    def test_optimizer_adjustments_are_bounded(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=5,
                adjustment_step_pct=0.20,
                min_adjustment_multiplier=0.75,
                max_adjustment_multiplier=1.25,
            )
            for cycle in range(25):
                optimizer.record_trade(
                    {
                        "timestamp_utc": f"2026-03-06T11:{cycle:02d}:00+00:00",
                        "symbol": "USOIL",
                        "strategy": "OIL_INVENTORY_SCALPER",
                        "setup": "OIL_INVENTORY_SCALPER_BREAKOUT",
                        "side": "SELL",
                        "pnl_r": -0.8,
                        "pnl_money": -0.2,
                    }
                )
            adjustments = optimizer.adjustments_for("OIL_INVENTORY_SCALPER")
            self.assertGreaterEqual(float(adjustments["probability_floor_mult"]), 0.75)
            self.assertLessEqual(float(adjustments["probability_floor_mult"]), 1.25)
            self.assertGreaterEqual(float(adjustments["confluence_floor_mult"]), 0.75)
            self.assertLessEqual(float(adjustments["candidate_sensitivity_mult"]), 1.25)

    def test_optimizer_persists_session_and_regime_buckets(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=2,
            )
            optimizer.record_trade(
                {
                    "timestamp_utc": "2026-03-06T08:00:00+00:00",
                    "symbol": "XAUUSD",
                    "strategy": "XAUUSD_M5_GRID",
                    "setup": "XAUUSD_M5_GRID_SCALPER_START",
                    "side": "BUY",
                    "pnl_r": 1.2,
                    "pnl_money": 0.5,
                    "duration_minutes": 12,
                    "spread_points": 12,
                    "session_state": "LONDON",
                    "regime": "RANGING",
                }
            )
            optimizer.record_trade(
                {
                    "timestamp_utc": "2026-03-06T09:00:00+00:00",
                    "symbol": "XAUUSD",
                    "strategy": "XAUUSD_M5_GRID",
                    "setup": "XAUUSD_M5_GRID_SCALPER_START",
                    "side": "BUY",
                    "pnl_r": -0.4,
                    "pnl_money": -0.2,
                    "duration_minutes": 8,
                    "spread_points": 40,
                    "session_state": "NEW_YORK",
                    "regime": "VOLATILE",
                }
            )

            strategy = optimizer.summary().get("strategies", {}).get("XAUUSD_M5_GRID", {})
            self.assertEqual(str(strategy.get("best_session")), "LONDON")
            self.assertEqual(str(strategy.get("best_spread_bucket")), "TIGHT")
            self.assertEqual(str(strategy.get("best_regime_bucket")), "RANGING")

    def test_optimizer_tracks_positive_and_negative_adjustment_tags(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=2,
            )
            optimizer.record_trade(
                {
                    "timestamp_utc": "2026-03-06T08:00:00+00:00",
                    "symbol": "BTCUSD",
                    "strategy": "BTC_TREND_SCALP",
                    "side": "BUY",
                    "pnl_r": 1.1,
                    "pnl_money": 0.7,
                    "adjustment_tags": ["LEAD_LAG_ALIGNED", "LOW_SPREAD"],
                }
            )
            optimizer.record_trade(
                {
                    "timestamp_utc": "2026-03-06T09:00:00+00:00",
                    "symbol": "BTCUSD",
                    "strategy": "BTC_TREND_SCALP",
                    "side": "SELL",
                    "pnl_r": -0.6,
                    "pnl_money": -0.3,
                    "adjustment_tags": ["CROWDED_SENTIMENT", "LOW_SPREAD"],
                }
            )

            strategy = optimizer.summary().get("strategies", {}).get("BTC_TREND_SCALP", {})
            self.assertIn("LEAD_LAG_ALIGNED", strategy.get("top_positive_tags", []))
            self.assertIn("CROWDED_SENTIMENT", strategy.get("top_negative_tags", []))

    def test_optimizer_falls_back_to_pnl_money_when_pnl_r_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=2,
            )
            optimizer.record_trade(
                {
                    "timestamp_utc": "2026-03-06T08:00:00+00:00",
                    "symbol": "BTCUSD",
                    "strategy": "BTC_TREND_SCALP",
                    "side": "BUY",
                    "pnl_r": 0.0,
                    "pnl_money": 1.4,
                }
            )
            optimizer.record_trade(
                {
                    "timestamp_utc": "2026-03-06T09:00:00+00:00",
                    "symbol": "BTCUSD",
                    "strategy": "BTC_TREND_SCALP",
                    "side": "SELL",
                    "pnl_r": 0.0,
                    "pnl_money": -0.8,
                }
            )

            strategy = optimizer.summary().get("strategies", {}).get("BTC_TREND_SCALP", {})
            self.assertEqual(int(strategy.get("trade_count", 0)), 2)
            self.assertEqual(int(strategy.get("wins", 0)), 1)
            self.assertEqual(int(strategy.get("losses", 0)), 1)
            self.assertAlmostEqual(float(strategy.get("win_rate", 0.0)), 0.5)
            self.assertGreater(float(strategy.get("profit_factor", 0.0)), 1.0)


if __name__ == "__main__":
    unittest.main()
