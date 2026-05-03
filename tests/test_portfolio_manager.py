from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from src.portfolio_manager import PortfolioManager


class PortfolioManagerClusterTests(unittest.TestCase):
    def test_cluster_reduces_second_related_position(self) -> None:
        portfolio = PortfolioManager(
            max_positions_total=6,
            max_positions_per_symbol=2,
            max_same_direction=2,
            correlation_window_minutes=60,
        )
        now = datetime.now(tz=timezone.utc)
        open_positions = [
            {"symbol": "EURUSD", "side": "BUY", "opened_at": now - timedelta(minutes=10)},
        ]

        decision = portfolio.assess_new_position(open_positions, "GBPUSD", "BUY")

        self.assertTrue(decision.allowed)
        self.assertLess(float(decision.size_multiplier), 1.0)
        self.assertEqual(decision.exposure_cluster_detected, "USD_BEARISH_CLUSTER")

    def test_cluster_blocks_third_related_position(self) -> None:
        portfolio = PortfolioManager(
            max_positions_total=6,
            max_positions_per_symbol=2,
            max_same_direction=2,
            correlation_window_minutes=60,
        )
        now = datetime.now(tz=timezone.utc)
        open_positions = [
            {"symbol": "EURUSD", "side": "BUY", "opened_at": now - timedelta(minutes=10)},
            {"symbol": "GBPUSD", "side": "BUY", "opened_at": now - timedelta(minutes=8)},
        ]

        decision = portfolio.assess_new_position(open_positions, "USDJPY", "SELL")

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "correlation_cluster_cap")


if __name__ == "__main__":
    unittest.main()
