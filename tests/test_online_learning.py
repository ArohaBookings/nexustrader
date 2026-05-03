from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import unittest

import numpy as np
import pandas as pd

from src.execution import current_trading_day_key
from src.online_learning import LEGACY_TRADE_HISTORY_COLUMNS_V1, TRADE_HISTORY_COLUMNS, OnlineLearningEngine


class OnlineLearningTests(unittest.TestCase):
    def test_setup_decision_logging_taken_and_rejected(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            engine = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=50,
            )
            engine.on_setup_decision(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "signal_id": "sig-reject",
                    "symbol": "XAUUSD",
                    "setup": "XAUUSD_M5_GRID_SCALPER_START",
                    "timeframe": "M5",
                    "decision_type": "entry_block",
                    "accepted": False,
                    "result": "rejected",
                    "reason": "spread_too_wide",
                    "ai_probability": 0.52,
                }
            )
            engine.on_setup_decision(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "signal_id": "sig-take",
                    "symbol": "EURUSD",
                    "setup": "FOREX_TREND_PULLBACK",
                    "timeframe": "M5",
                    "decision_type": "entry_take",
                    "accepted": True,
                    "result": "pending",
                    "reason": "entry_approved",
                    "ai_probability": 0.73,
                }
            )

            frame = pd.read_csv(root / "setups_log.csv")
            self.assertEqual(len(frame), 2)
            self.assertIn("decision_type", frame.columns)
            self.assertIn("accepted", frame.columns)
            self.assertIn("result", frame.columns)
            self.assertIn("entry_block", set(frame["decision_type"].astype(str)))
            self.assertIn("entry_take", set(frame["decision_type"].astype(str)))

    def test_maintenance_retrain_runs_out_of_session_or_nightly(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            engine = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=100,
                promotion_min_samples=20,
            )
            for index in range(30):
                pnl_r = 1.0 if (index % 2 == 0) else -1.0
                engine.on_trade_close(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "signal_id": f"sig-{index}",
                        "symbol": "XAUUSD",
                        "timeframe": "M5",
                        "side": "BUY" if index % 2 == 0 else "SELL",
                        "entry": 2200.0,
                        "sl": 2195.0,
                        "tp": 2210.0,
                        "exit": 2205.0,
                        "pnl_r": pnl_r,
                        "pnl_money": pnl_r * 0.5,
                        "news_state": "clear",
                        "session_state": "IN",
                        "ai_decision": "approved",
                        "ai_probability": 0.7,
                        "spread_points": 25.0,
                        "lot": 0.01,
                        "regime": "RANGING",
                        "setup": "XAUUSD_M5_GRID_SCALPER_START",
                    }
                )
            ran = engine.maybe_retrain_maintenance(
                now_utc=datetime(2026, 3, 6, 0, 30, tzinfo=timezone.utc),
                session_name="SYDNEY",
                active_sessions={"TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
                force=True,
            )
            self.assertTrue(ran)
            self.assertTrue((root / "online_model.pkl").exists())
            self.assertEqual(
                engine._last_maintenance_day,
                current_trading_day_key(now_ts=datetime(2026, 3, 6, 0, 30, tzinfo=timezone.utc)),
            )

    def test_trade_close_persists_strategy_runtime_fields(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            engine = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=5,
            )
            engine.on_trade_close(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "signal_id": "sig-strategy-runtime",
                    "symbol": "XAUUSD",
                    "timeframe": "M5",
                    "side": "BUY",
                    "entry": 3000.0,
                    "sl": 2990.0,
                    "tp": 3020.0,
                    "exit": 3012.0,
                    "pnl_r": 1.2,
                    "pnl_money": 1.0,
                    "news_state": "clear",
                    "session_state": "IN",
                    "session_name": "LONDON",
                    "ai_decision": "approved",
                    "ai_probability": 0.74,
                    "spread_points": 18.0,
                    "lot": 0.01,
                    "regime": "BREAKOUT_EXPANSION",
                    "regime_state": "BREAKOUT_EXPANSION",
                    "setup": "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                    "strategy_key": "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                    "lane_name": "XAU_DIRECTIONAL",
                    "management_template": "BREAKOUT_RUNNER",
                    "strategy_state": "ATTACK",
                    "regime_fit": 0.91,
                    "session_fit": 0.95,
                    "volatility_fit": 0.83,
                    "pair_behavior_fit": 0.79,
                    "execution_quality_fit": 0.88,
                    "entry_timing_score": 0.84,
                    "structure_cleanliness_score": 0.81,
                    "strategy_recent_performance": 0.67,
                    "market_data_source": "mt5+yahoo+twelve",
                    "market_data_consensus_state": "ALIGNED",
                    "multi_tf_alignment_score": 0.88,
                    "fractal_persistence_score": 0.77,
                    "compression_expansion_score": 0.69,
                    "dxy_support_score": 0.62,
                    "swing_continuation_score": 0.81,
                    "aggressive_pair_mode": 1.0,
                    "trajectory_catchup_pressure": 0.14,
                    "institutional_confluence_score": 0.86,
                    "candle_mastery_score": 0.79,
                    "live_shadow_gap_score": 0.08,
                    "execution_edge_score": 0.83,
                    "mc_win_rate": 0.87,
                    "ga_generation_id": 4,
                }
            )

            frame = pd.read_csv(root / "trades.csv")
            self.assertEqual(str(frame.loc[0, "strategy_key"]), "XAUUSD_LONDON_LIQUIDITY_SWEEP")
            self.assertEqual(str(frame.loc[0, "lane_name"]), "XAU_DIRECTIONAL")
            self.assertEqual(str(frame.loc[0, "management_template"]), "BREAKOUT_RUNNER")
            self.assertEqual(str(frame.loc[0, "strategy_state"]), "ATTACK")
            self.assertEqual(str(frame.loc[0, "market_data_consensus_state"]), "ALIGNED")
            self.assertGreater(float(frame.loc[0, "entry_timing_score"]), 0.0)
            self.assertGreater(float(frame.loc[0, "structure_cleanliness_score"]), 0.0)
            self.assertGreater(float(frame.loc[0, "multi_tf_alignment_score"]), 0.0)
            self.assertGreater(float(frame.loc[0, "mc_win_rate"]), 0.0)
            self.assertAlmostEqual(float(frame.loc[0, "institutional_confluence_score"]), 0.86)
            self.assertAlmostEqual(float(frame.loc[0, "candle_mastery_score"]), 0.79)
            self.assertAlmostEqual(float(frame.loc[0, "live_shadow_gap_score"]), 0.08)
            self.assertAlmostEqual(float(frame.loc[0, "execution_edge_score"]), 0.83)

    def test_feature_row_includes_aggressive_pair_learning_features(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            engine = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=5,
            )
            row = engine._feature_row(
                {
                    "symbol": "USDJPY",
                    "side": "BUY",
                    "entry": 150.0,
                    "sl": 149.9,
                    "tp": 150.3,
                    "lot": 0.01,
                    "ai_probability": 0.74,
                    "spread_points": 12.0,
                    "news_state": "clear",
                    "session_state": "IN",
                    "session_name": "TOKYO",
                    "regime": "TRENDING",
                    "setup": "USDJPY_MOMENTUM_IMPULSE",
                    "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                    "strategy_state": "ATTACK",
                    "regime_fit": 0.72,
                    "session_fit": 0.84,
                    "volatility_fit": 0.80,
                    "pair_behavior_fit": 0.76,
                    "execution_quality_fit": 0.82,
                    "entry_timing_score": 0.78,
                    "structure_cleanliness_score": 0.74,
                    "strategy_recent_performance": 0.66,
                    "market_data_consensus_state": "ALIGNED",
                    "multi_tf_alignment_score": 0.86,
                    "fractal_persistence_score": 0.78,
                    "compression_expansion_score": 0.68,
                    "dxy_support_score": 0.58,
                    "mc_win_rate": 0.88,
                    "ga_generation_id": 7,
                    "swing_continuation_score": 0.80,
                    "aggressive_pair_mode": 1.0,
                    "trajectory_catchup_pressure": 0.12,
                    "institutional_confluence_score": 0.82,
                    "candle_mastery_score": 0.77,
                    "live_shadow_gap_score": 0.05,
                    "execution_edge_score": 0.80,
                }
            )

            self.assertEqual(row.shape[1], len(engine._feature_cols))
            feature_values = dict(zip(engine._feature_cols, row[0]))
            self.assertGreater(float(feature_values["trajectory_catchup_pressure"]), 0.0)
            self.assertAlmostEqual(float(feature_values["institutional_confluence_score"]), 0.82)
            self.assertAlmostEqual(float(feature_values["candle_mastery_score"]), 0.77)
            self.assertAlmostEqual(float(feature_values["live_shadow_gap_score"]), 0.05)
            self.assertAlmostEqual(float(feature_values["execution_edge_score"]), 0.80)

    def test_mixed_trade_history_schema_is_repaired_before_eval(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            engine = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=5,
            )
            legacy_row = [
                "2026-03-07T04:13:25.814249+00:00",
                "BTCUSD",
                "M5",
                "",
                "BUY",
                "68000.25",
                "67998.75",
                "68002.12",
                "68000.25",
                "0.0",
                "1.25",
                "clear",
                "OUT",
                "{\"force_test_trade\": true}",
                "loss",
                "0.65",
                "30.0",
                "0.01",
                "FORCE_TEST",
                "BTCUSD_M15_FORCE_TEST",
                "FORCE_TEST::BTCUSD::M15::BUY",
            ]
            current_payload = engine._csv_row(
                {
                    "timestamp_utc": "2026-03-13T23:34:27.220884+00:00",
                    "symbol": "BTCUSD",
                    "timeframe": "M5",
                    "side": "SELL",
                    "entry": 70949.66,
                    "sl": 71019.29,
                    "tp": 70831.98,
                    "exit": 70949.66,
                    "pnl_r": 0.0,
                    "pnl_money": 0.35,
                    "news_state": "clear",
                    "session_state": "SYDNEY",
                    "session_name": "SYDNEY",
                    "ai_decision": "{\"approve\": true}",
                    "ai_probability": 0.82,
                    "spread_points": 1712.0,
                    "lot": 0.01,
                    "regime": "RANGING",
                    "regime_state": "QUIET_ACCUMULATION",
                    "setup": "BTC_MOMENTUM_CONTINUATION",
                    "strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION",
                    "strategy_state": "NORMAL",
                    "regime_fit": 0.22,
                    "session_fit": 0.8,
                    "volatility_fit": 0.9,
                    "pair_behavior_fit": 0.62,
                    "execution_quality_fit": 1.0,
                    "entry_timing_score": 0.60,
                    "structure_cleanliness_score": 0.78,
                    "strategy_recent_performance": 0.48,
                    "market_data_source": "parquet_cache",
                    "market_data_consensus_state": "EXTERNAL_MULTI_PROVIDER",
                    "signal_id": "sig-mixed-schema",
                }
            )
            with (root / "trades.csv").open("w", encoding="utf-8", newline="") as handle:
                import csv

                writer = csv.writer(handle)
                writer.writerow(list(LEGACY_TRADE_HISTORY_COLUMNS_V1))
                writer.writerow(legacy_row)
                writer.writerow([current_payload.get(column, "") for column in TRADE_HISTORY_COLUMNS])

            frame = engine._read_history_frame()
            self.assertEqual(len(frame), 2)
            self.assertIn("session_name", frame.columns)
            self.assertIn("strategy_key", frame.columns)
            self.assertEqual(str(frame.iloc[1]["session_name"]), "SYDNEY")
            repaired = pd.read_csv(root / "trades.csv")
            self.assertIn("session_name", repaired.columns)

    def test_maintenance_holds_when_training_split_collapses_to_one_class(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            engine = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=5,
                promotion_min_samples=20,
                market_history_backfill_enabled=False,
            )
            for index in range(24):
                pnl_r = 1.0 if index < 20 else -1.0
                engine.on_trade_close(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "signal_id": f"sig-collapsed-{index}",
                        "symbol": "XAUUSD",
                        "timeframe": "M5",
                        "side": "BUY",
                        "entry": 2200.0,
                        "sl": 2195.0,
                        "tp": 2210.0,
                        "exit": 2205.0,
                        "pnl_r": pnl_r,
                        "pnl_money": pnl_r * 0.5,
                        "news_state": "clear",
                        "session_state": "IN",
                        "session_name": "LONDON",
                        "ai_decision": "approved",
                        "ai_probability": 0.7,
                        "spread_points": 25.0,
                        "lot": 0.01,
                        "regime": "RANGING",
                        "setup": "XAUUSD_M5_GRID_SCALPER_START",
                    }
                )

            ran = engine.maybe_retrain_maintenance(
                now_utc=datetime(2026, 3, 6, 0, 30, tzinfo=timezone.utc),
                session_name="SYDNEY",
                active_sessions={"TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
                force=True,
            )

            self.assertTrue(ran)
            self.assertEqual(engine.status_snapshot()["last_maintenance_status"], "insufficient_class_balance")
            self.assertEqual(engine.status_snapshot()["last_maintenance_error"], "")

    def test_market_history_seed_can_rescue_class_balance(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir = root / "candles_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            times = pd.date_range("2024-01-01T00:00:00Z", periods=600, freq="15min", tz="UTC")
            price = 2200.0
            closes: list[float] = []
            for index in range(len(times)):
                direction = 1.0 if (index // 25) % 2 == 0 else -1.0
                price += 1.6 * direction
                closes.append(price + (0.15 if index % 2 == 0 else -0.10))
            close = pd.Series(closes)
            open_ = close.shift(1).fillna(close.iloc[0] - 0.2)
            high = pd.concat([open_, close], axis=1).max(axis=1) + 0.75
            low = pd.concat([open_, close], axis=1).min(axis=1) - 0.75
            seed_frame = pd.DataFrame(
                {
                    "time": times,
                    "open": open_.to_numpy(),
                    "high": high.to_numpy(),
                    "low": low.to_numpy(),
                    "close": close.to_numpy(),
                    "spread": [25.0] * len(times),
                }
            )
            seed_frame.to_parquet(cache_dir / "XAUUSD_M15.parquet", index=False)

            engine = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                history_cache_dir=cache_dir,
                min_retrain_trades=5,
                promotion_min_samples=20,
                market_history_backfill_enabled=False,
            )
            for index in range(24):
                engine.on_trade_close(
                    {
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        "signal_id": f"sig-seed-{index}",
                        "symbol": "XAUUSD",
                        "timeframe": "M5",
                        "side": "BUY",
                        "entry": 2200.0,
                        "sl": 2195.0,
                        "tp": 2210.0,
                        "exit": 2205.0,
                        "pnl_r": 1.0,
                        "pnl_money": 0.5,
                        "news_state": "clear",
                        "session_state": "IN",
                        "session_name": "LONDON",
                        "ai_decision": "approved",
                        "ai_probability": 0.7,
                        "spread_points": 25.0,
                        "lot": 0.01,
                        "regime": "TRENDING",
                        "setup": "XAUUSD_M5_GRID_SCALPER_START",
                    }
                )

            ran = engine.maybe_retrain_maintenance(
                now_utc=datetime(2026, 3, 6, 0, 30, tzinfo=timezone.utc),
                session_name="SYDNEY",
                active_sessions={"TOKYO", "LONDON", "OVERLAP", "NEW_YORK"},
                force=True,
            )
            status = engine.status_snapshot()

            self.assertTrue(ran)
            self.assertNotEqual(status["last_maintenance_status"], "insufficient_class_balance")
            self.assertGreater(int(status["last_market_history_seed_samples"]), 0)
            self.assertTrue(bool(status["initialized"]))
            self.assertTrue((root / "online_model.pkl").exists())

    def test_market_history_backfill_writes_long_history_files(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir = root / "candles_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "time": pd.date_range("2026-01-01T00:00:00Z", periods=16, freq="1h", tz="UTC"),
                    "open": np.linspace(100.0, 101.5, 16),
                    "high": np.linspace(100.2, 101.7, 16),
                    "low": np.linspace(99.8, 101.3, 16),
                    "close": np.linspace(100.1, 101.6, 16),
                    "spread": np.full(16, 10.0),
                }
            ).to_parquet(cache_dir / "XAUUSD_M15.parquet", index=False)

            class FakeYahoo:
                def __init__(self, *args, **kwargs) -> None:
                    pass

                def fetch_rates_with_range(self, symbol: str, timeframe: str, *, range_value: str, limit=None):
                    periods = 900 if timeframe == "D1" else 520
                    freq = "1D" if timeframe == "D1" else "1h"
                    times = pd.date_range("2020-01-01T00:00:00Z", periods=periods, freq=freq, tz="UTC")
                    close = pd.Series(np.linspace(100.0, 150.0, len(times)))
                    open_ = close.shift(1).fillna(close.iloc[0] - 0.1)
                    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.3
                    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.3
                    return pd.DataFrame(
                        {
                            "time": times,
                            "open": open_.to_numpy(),
                            "high": high.to_numpy(),
                            "low": low.to_numpy(),
                            "close": close.to_numpy(),
                            "tick_volume": np.ones(len(times)),
                            "spread": np.full(len(times), 10.0),
                            "real_volume": np.ones(len(times)),
                        }
                    )

            engine = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                history_cache_dir=cache_dir,
                market_history_backfill_enabled=True,
            )
            with patch("src.online_learning.YahooMarketDataFallback", FakeYahoo):
                summary = engine._backfill_market_history_cache()

            self.assertEqual(summary["status"], "ok")
            self.assertGreater(int(summary["files_written"]), 0)
            self.assertTrue((cache_dir / "XAUUSD_D1.parquet").exists())
            self.assertTrue((cache_dir / "XAUUSD_H1.parquet").exists())

    def test_market_history_backfill_uses_market_universe_symbols_without_cache_seed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            cache_dir = root / "candles_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)

            class FakeYahoo:
                calls: list[tuple[str, str, str]] = []

                def __init__(self, *args, **kwargs) -> None:
                    pass

                def fetch_rates_with_range(self, symbol: str, timeframe: str, *, range_value: str, limit=None):
                    FakeYahoo.calls.append((str(symbol), str(timeframe), str(range_value)))
                    periods = 1400 if timeframe == "M15" else 950 if timeframe == "H1" else 700
                    freq = "15min" if timeframe == "M15" else "1h" if timeframe == "H1" else "1D"
                    times = pd.date_range("2020-01-01T00:00:00Z", periods=periods, freq=freq, tz="UTC")
                    close = pd.Series(np.linspace(100.0, 180.0, len(times)))
                    open_ = close.shift(1).fillna(close.iloc[0] - 0.1)
                    high = pd.concat([open_, close], axis=1).max(axis=1) + 0.3
                    low = pd.concat([open_, close], axis=1).min(axis=1) - 0.3
                    return pd.DataFrame(
                        {
                            "time": times,
                            "open": open_.to_numpy(),
                            "high": high.to_numpy(),
                            "low": low.to_numpy(),
                            "close": close.to_numpy(),
                            "tick_volume": np.ones(len(times)),
                            "spread": np.full(len(times), 10.0),
                            "real_volume": np.ones(len(times)),
                        }
                    )

            engine = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                history_cache_dir=cache_dir,
                market_history_backfill_enabled=True,
                market_history_universe_symbols=("AAPL", "XAGUSD"),
            )
            with patch("src.online_learning.YahooMarketDataFallback", FakeYahoo):
                summary = engine._backfill_market_history_cache()

            self.assertEqual(summary["status"], "ok")
            self.assertTrue(any(symbol == "AAPL" for symbol, _timeframe, _range in FakeYahoo.calls))
            self.assertTrue(any(symbol == "XAGUSD" for symbol, _timeframe, _range in FakeYahoo.calls))
            self.assertTrue((cache_dir / "AAPL_D1.parquet").exists())
            self.assertTrue((cache_dir / "XAGUSD_M15.parquet").exists())


if __name__ == "__main__":
    unittest.main()
