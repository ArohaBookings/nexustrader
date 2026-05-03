from __future__ import annotations

import unittest

import pandas as pd

from src.regime_detector import RegimeDetector


class RegimeDetectorTests(unittest.TestCase):
    def test_rich_regime_state_is_exposed(self) -> None:
        detector = RegimeDetector()
        row = pd.Series(
            {
                "m5_close": 3000.0,
                "m5_open": 2997.0,
                "m5_high": 3002.0,
                "m5_low": 2995.0,
                "m5_atr_14": 4.2,
                "m5_atr_avg_20": 2.1,
                "m5_spread": 18.0,
                "m5_range_position_20": 0.92,
                "m5_upper_wick_ratio": 0.08,
                "m5_lower_wick_ratio": 0.12,
                "m5_body_efficiency": 0.76,
                "m5_trend_persistence": 0.84,
                "m5_breakout_flag": 1.0,
                "m5_liquidity_sweep": 0.0,
                "m5_impulse_strength": 0.88,
            }
        )

        result = detector.classify(row)

        self.assertIn(result.state_label, {"TREND_EXPANSION", "VOLATILITY_SPIKE", "BREAKOUT_COMPRESSION"})
        self.assertGreaterEqual(float(result.state_confidence), 0.0)
        self.assertIn("volatility_forecast_state", result.details)
        self.assertIn("pressure_proxy_score", result.details)


if __name__ == "__main__":
    unittest.main()
