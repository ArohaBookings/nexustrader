from __future__ import annotations

import unittest

from src.omega_regime import OmegaRegimeDetector


class OmegaRegimeTests(unittest.TestCase):
    def test_news_shock_detection_without_model(self) -> None:
        detector = OmegaRegimeDetector(model_path=None, enabled=True)
        state = detector.classify(
            {
                "atr_ratio": 2.1,
                "adx": 26.0,
                "trend_gap": 0.3,
                "ema_slope": 0.12,
                "spread_ratio": 1.8,
                "momentum": 0.4,
                "news_shock_score": 1.2,
                "regime_hint": "VOLATILE",
                "side_hint": "BUY",
            }
        )
        self.assertEqual(state.label, "NEWS_SHOCK")
        self.assertGreaterEqual(state.confidence, 0.6)
        self.assertIn("NEWS_SHOCK", state.probabilities)

    def test_trending_down_detection(self) -> None:
        detector = OmegaRegimeDetector(model_path=None, enabled=True)
        state = detector.classify(
            {
                "atr_ratio": 1.1,
                "adx": 30.0,
                "trend_gap": -0.8,
                "ema_slope": -0.3,
                "spread_ratio": 0.9,
                "momentum": -0.4,
                "news_shock_score": 0.0,
                "regime_hint": "TRENDING",
                "side_hint": "SELL",
            }
        )
        self.assertEqual(state.label, "TRENDING_DOWN")
        self.assertGreater(state.probabilities["TRENDING_DOWN"], state.probabilities["RANGING"])


if __name__ == "__main__":
    unittest.main()
