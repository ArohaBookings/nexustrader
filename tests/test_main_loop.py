from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from src.portfolio_manager import PortfolioManager


class MainLoopGateTests(unittest.TestCase):
    def test_per_symbol_cap_blocks_new_entry(self) -> None:
        portfolio = PortfolioManager(
            max_positions_total=4,
            max_positions_per_symbol=2,
            max_same_direction=2,
            correlation_window_minutes=15,
        )
        now = datetime.now(tz=timezone.utc)
        open_positions = [
            {"symbol": "EURUSD", "side": "BUY", "opened_at": now - timedelta(minutes=20)},
            {"symbol": "EURUSD", "side": "SELL", "opened_at": now - timedelta(minutes=10)},
        ]
        decision = portfolio.assess_new_position(open_positions, "EURUSD", "BUY")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "max_positions_per_symbol_reached")

    def test_total_cap_blocks_new_entry(self) -> None:
        portfolio = PortfolioManager(
            max_positions_total=4,
            max_positions_per_symbol=2,
            max_same_direction=2,
            correlation_window_minutes=15,
        )
        now = datetime.now(tz=timezone.utc)
        open_positions = [
            {"symbol": "XAUUSD", "side": "BUY", "opened_at": now - timedelta(minutes=20)},
            {"symbol": "EURUSD", "side": "BUY", "opened_at": now - timedelta(minutes=19)},
            {"symbol": "GBPUSD", "side": "SELL", "opened_at": now - timedelta(minutes=18)},
            {"symbol": "USDJPY", "side": "SELL", "opened_at": now - timedelta(minutes=17)},
        ]
        decision = portfolio.assess_new_position(open_positions, "EURUSD", "BUY")
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "max_positions_total_reached")


if __name__ == "__main__":
    unittest.main()
