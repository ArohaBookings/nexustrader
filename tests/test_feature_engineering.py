from __future__ import annotations

from datetime import datetime, timezone
import warnings
import unittest

import numpy as np
import pandas as pd

from src.feature_engineering import FeatureEngineer


def _frame(rows: int, freq: str) -> pd.DataFrame:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    time_index = pd.date_range(start=start, periods=rows, freq=freq, tz="UTC")
    x = np.arange(rows, dtype=float)
    close = 2200.0 + (np.sin(x / 8.0) * 2.0) + (x * 0.02)
    open_ = np.roll(close, 1)
    open_[0] = close[0]
    high = np.maximum(open_, close) + 0.3
    low = np.minimum(open_, close) - 0.3
    spread = 18 + (x % 3)
    volume = 100 + (x % 25) * 3
    return pd.DataFrame(
        {
            "time": time_index,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "tick_volume": volume,
            "spread": spread,
        }
    )


class FeatureEngineeringTests(unittest.TestCase):
    def test_feature_schema_core_columns_present(self) -> None:
        engineer = FeatureEngineer()
        output = engineer.build(
            m1=_frame(1200, "1min"),
            m5=_frame(900, "5min"),
            m15=_frame(700, "15min"),
            h1=_frame(500, "1h"),
            h4=_frame(300, "4h"),
        )
        required = {
            "time",
            "m5_close",
            "m5_atr_14",
            "m5_spread",
            "m5_spread_mean_14",
            "m5_spread_atr_14",
            "m5_rsi_14",
            "m15_range_position_20",
            "h1_ema_50",
            "h1_ema_200",
            "h4_trend_score",
            "session_liquidity_score",
            "ghost_order_book_pressure",
            "predicted_liquidity_hunt_score",
            "behavior_fear_score",
            "behavior_greed_score",
            "behavior_complacency_score",
            "behavior_bias_score",
            "m5_liquidity_sweep",
            "m5_bos_bull",
            "m5_choch_bear",
            "m5_fvg_bull",
            "m5_absorption_proxy",
            "m5_order_block_bull",
            "m5_vwap_deviation_atr",
            "m5_volume_profile_pressure",
            "candle_mastery_score",
            "institutional_confluence_score",
            "execution_edge_score",
            "live_shadow_gap_score",
        }
        self.assertTrue(required.issubset(set(output.columns)))
        bounded = [
            "candle_mastery_score",
            "institutional_confluence_score",
            "execution_edge_score",
            "live_shadow_gap_score",
            "m5_volume_profile_pressure",
        ]
        for column in bounded:
            self.assertGreaterEqual(float(output[column].iloc[-1]), 0.0)
            self.assertLessEqual(float(output[column].iloc[-1]), 1.0)

    def test_build_does_not_emit_dataframe_fragmentation_warning(self) -> None:
        engineer = FeatureEngineer()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            engineer.build(
                m1=_frame(800, "1min"),
                m5=_frame(700, "5min"),
                m15=_frame(500, "15min"),
                h1=_frame(300, "1h"),
                h4=_frame(200, "4h"),
            )
        performance_warnings = [
            warning
            for warning in caught
            if issubclass(warning.category, pd.errors.PerformanceWarning)
        ]
        self.assertEqual(performance_warnings, [])

    def test_build_normalizes_mixed_timestamp_precisions(self) -> None:
        engineer = FeatureEngineer()
        m1 = _frame(120, "1min")
        m5 = _frame(120, "5min")
        m15 = _frame(120, "15min")
        h1 = _frame(120, "1h")

        m1["time"] = pd.to_datetime(m1["time"].astype("int64") // 1_000_000, unit="ms", utc=True)
        m5["time"] = pd.to_datetime(m5["time"].astype("int64") // 1_000, unit="us", utc=True)
        m15["time"] = pd.to_datetime(m15["time"].astype("int64") // 1_000_000_000, unit="s", utc=True)

        output = engineer.build(m1=m1, m5=m5, m15=m15, h1=h1)

        self.assertFalse(output.empty)
        self.assertTrue(str(output["time"].dtype).endswith("[ns, UTC]"))

    def test_build_preserves_wall_clock_timestamp_when_upcasting_to_ns(self) -> None:
        engineer = FeatureEngineer()
        m1 = _frame(120, "1min")
        m5 = _frame(120, "5min")
        m15 = _frame(120, "15min")
        h1 = _frame(120, "1h")

        output = engineer.build(m1=m1, m5=m5, m15=m15, h1=h1)

        self.assertFalse(output.empty)
        self.assertEqual(output["time"].iloc[0], pd.Timestamp("2026-01-01 00:00:00+00:00"))
        self.assertEqual(output["time"].iloc[-1], pd.Timestamp("2026-01-01 09:55:00+00:00"))

    def test_zero_volume_bars_do_not_force_volume_ratio_to_zero(self) -> None:
        engineer = FeatureEngineer()
        m1 = _frame(120, "1min")
        m5 = _frame(120, "5min")
        m15 = _frame(120, "15min")
        h1 = _frame(120, "1h")
        h4 = _frame(120, "4h")
        m5["tick_volume"] = 0.0
        m15["tick_volume"] = 0.0

        output = engineer.build(m1=m1, m5=m5, m15=m15, h1=h1, h4=h4)

        self.assertGreaterEqual(float(output["m5_volume_ratio_20"].iloc[-1]), 0.99)
        self.assertGreaterEqual(float(output["m15_volume_ratio_20"].iloc[-1]), 0.99)

    def test_build_emits_spread_baseline_and_spread_atr_features(self) -> None:
        engineer = FeatureEngineer()
        m1 = _frame(120, "1min")
        m5 = _frame(120, "5min")
        m15 = _frame(120, "15min")
        h1 = _frame(120, "1h")
        h4 = _frame(120, "4h")
        m5.loc[:, "spread"] = np.linspace(12.0, 24.0, len(m5))

        output = engineer.build(m1=m1, m5=m5, m15=m15, h1=h1, h4=h4)

        self.assertGreater(float(output["m5_spread_mean_14"].iloc[-1]), 0.0)
        self.assertGreater(float(output["m5_spread_atr_14"].iloc[-1]), 0.0)


if __name__ == "__main__":
    unittest.main()
