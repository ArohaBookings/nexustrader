from __future__ import annotations

import unittest

import pandas as pd

from src.strategy_engine import StrategyEngine


class StrategyEngineQualityTests(unittest.TestCase):
    def test_generate_keeps_distinct_grid_burst_legs(self) -> None:
        class GridBurstEngine(StrategyEngine):
            def _fast_entry(self, row, regime, timestamp):
                return [
                    self._grid_candidate(signal_id="grid-start-1", level=1, score=0.82),
                    self._grid_candidate(signal_id="grid-start-2", level=2, score=0.81),
                ]

            def _set_forget_h1_h4(self, row, regime, timestamp):
                return []

            @staticmethod
            def _grid_candidate(*, signal_id: str, level: int, score: float):
                from src.strategy_engine import SignalCandidate

                return SignalCandidate(
                    signal_id=signal_id,
                    setup="XAUUSD_M5_GRID_SCALPER_START",
                    side="BUY",
                    score_hint=score,
                    reason="burst test",
                    stop_atr=0.8,
                    tp_r=2.2,
                    strategy_family="GRID",
                    meta={"grid_cycle": True, "grid_cycle_id": "cycle-1", "grid_level": level, "grid_burst_index": level},
                )

        engine = GridBurstEngine(symbol="XAUUSD")
        features = pd.DataFrame([{"time": pd.Timestamp("2026-03-12T08:00:00Z")}])

        candidates = engine.generate(features, regime="TRENDING")

        self.assertEqual(len(candidates), 2)
        self.assertEqual([candidate.signal_id for candidate in candidates], ["grid-start-1", "grid-start-2"])

    def test_major_fast_entry_blocks_weak_breakout(self) -> None:
        engine = StrategyEngine(symbol="EURUSD")
        features = pd.DataFrame(
            [
                {
                    "time": pd.Timestamp("2026-03-12T08:00:00Z"),
                    "m5_spread": 12.0,
                    "m1_atr_pct_of_avg": 1.2,
                    "m5_atr_pct_of_avg": 1.1,
                    "m1_close": 1.1015,
                    "m1_rolling_high_prev_20": 1.1010,
                    "m1_rolling_low_prev_20": 1.0990,
                    "m1_volume_ratio_20": 1.02,
                    "m1_body": 0.00008,
                    "m1_atr_14": 0.00010,
                    "m5_close": 1.1014,
                    "m5_ema_20": 1.1009,
                    "m5_ema_50": 1.1004,
                    "m5_macd_hist": 0.0002,
                    "m5_macd_hist_slope": 0.00005,
                    "m5_volume_ratio_20": 1.03,
                    "m5_body_efficiency": 0.57,
                    "m5_atr_14": 0.00055,
                    "m15_range_position_20": 0.95,
                }
            ]
        )

        candidates = engine.generate(features, regime="TRENDING")
        self.assertFalse(any(candidate.setup == "M1_M5_BREAKOUT" for candidate in candidates))

    def test_major_fast_entry_allows_clean_breakout(self) -> None:
        engine = StrategyEngine(symbol="EURUSD")
        features = pd.DataFrame(
            [
                {
                    "time": pd.Timestamp("2026-03-12T08:05:00Z"),
                    "m5_spread": 10.0,
                    "m1_atr_pct_of_avg": 1.3,
                    "m5_atr_pct_of_avg": 1.2,
                    "m1_close": 1.10145,
                    "m1_rolling_high_prev_20": 1.1010,
                    "m1_rolling_low_prev_20": 1.0990,
                    "m1_volume_ratio_20": 1.12,
                    "m1_body": 0.00011,
                    "m1_atr_14": 0.00013,
                    "m5_close": 1.10145,
                    "m5_ema_20": 1.1012,
                    "m5_ema_50": 1.1007,
                    "m5_macd_hist": 0.0003,
                    "m5_macd_hist_slope": 0.00006,
                    "m5_volume_ratio_20": 1.12,
                    "m5_body_efficiency": 0.66,
                    "m5_atr_14": 0.00065,
                    "m15_range_position_20": 0.74,
                }
            ]
        )

        candidates = engine.generate(features, regime="TRENDING")
        self.assertTrue(any(candidate.setup == "M1_M5_BREAKOUT" for candidate in candidates))

    def test_set_forget_blocks_off_session(self) -> None:
        engine = StrategyEngine(symbol="EURUSD")
        features = pd.DataFrame(
            [
                {
                    "time": pd.Timestamp("2026-03-12T20:00:00Z"),
                    "session_name": "TOKYO",
                    "m5_spread": 8.0,
                    "h4_ema_50": 1.1040,
                    "h4_ema_200": 1.1000,
                    "h1_ema_20": 1.1030,
                    "h1_ema_50": 1.1015,
                    "h1_rsi_14": 58.0,
                    "h1_atr_14": 0.0015,
                    "m15_close": 1.1032,
                    "m15_ema_20": 1.1027,
                    "m15_body_efficiency": 0.62,
                    "m15_volume_ratio_20": 1.10,
                    "m15_atr_14": 0.0007,
                }
            ]
        )

        candidates = engine.generate(features, regime="TRENDING")
        self.assertFalse(any(candidate.setup == "SET_FORGET_H1_H4" for candidate in candidates))

    def test_set_forget_allows_clean_london_trend(self) -> None:
        engine = StrategyEngine(symbol="EURUSD")
        features = pd.DataFrame(
            [
                {
                    "time": pd.Timestamp("2026-03-12T08:30:00Z"),
                    "session_name": "LONDON",
                    "m5_spread": 8.0,
                    "h4_ema_50": 1.1040,
                    "h4_ema_200": 1.1000,
                    "h1_ema_20": 1.1030,
                    "h1_ema_50": 1.1018,
                    "h1_rsi_14": 58.0,
                    "h1_atr_14": 0.0015,
                    "m15_close": 1.1032,
                    "m15_ema_20": 1.1029,
                    "m15_body_efficiency": 0.63,
                    "m15_volume_ratio_20": 1.12,
                    "m15_atr_14": 0.0007,
                }
            ]
        )

        candidates = engine.generate(features, regime="TRENDING")
        self.assertTrue(any(candidate.setup == "SET_FORGET_H1_H4" for candidate in candidates))

    def test_set_forget_allows_clean_new_york_equity_trend(self) -> None:
        engine = StrategyEngine(symbol="AAPL")
        features = pd.DataFrame(
            [
                {
                    "time": pd.Timestamp("2026-03-12T15:30:00Z"),
                    "session_name": "NEW_YORK",
                    "m5_spread": 4.0,
                    "h4_ema_50": 222.0,
                    "h4_ema_200": 216.0,
                    "h1_ema_20": 221.4,
                    "h1_ema_50": 219.6,
                    "h1_rsi_14": 61.0,
                    "h1_atr_14": 2.2,
                    "m15_close": 221.8,
                    "m15_ema_20": 221.1,
                    "m15_body_efficiency": 0.60,
                    "m15_volume_ratio_20": 1.04,
                    "m15_atr_14": 0.85,
                }
            ]
        )

        candidates = engine.generate(features, regime="TRENDING")
        self.assertTrue(any(candidate.setup == "SET_FORGET_H1_H4" for candidate in candidates))


if __name__ == "__main__":
    unittest.main()
