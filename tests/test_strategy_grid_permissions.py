from __future__ import annotations

import unittest

from src.strategies.grid_xau_m5 import evaluate_grid_permission


class GridPermissionTests(unittest.TestCase):
    def test_grid_add_requires_step_move(self) -> None:
        state = {
            "open_positions_estimate": 1,
            "grid_leg_index": 2,
            "grid_side": "BUY",
            "last_entry_price": 2200.00,
        }
        blocked = evaluate_grid_permission(
            timeframe="M5",
            setup="XAUUSD_M5_GRID_SCALPER_ADD",
            side="BUY",
            entry_price=2200.20,
            spread_points=20.0,
            max_spread_points=60.0,
            requires_m5=True,
            step_points=35.0,
            point_size=0.01,
            max_legs=10,
            state=state,
        )
        allowed = evaluate_grid_permission(
            timeframe="M5",
            setup="XAUUSD_M5_GRID_SCALPER_ADD",
            side="BUY",
            entry_price=2200.50,
            spread_points=20.0,
            max_spread_points=60.0,
            requires_m5=True,
            step_points=35.0,
            point_size=0.01,
            max_legs=10,
            state=state,
        )

        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.reason, "grid_step_not_reached")
        self.assertTrue(allowed.allowed)
        self.assertEqual(allowed.reason, "grid_add_allowed")

    def test_grid_start_blocked_when_cycle_active(self) -> None:
        state = {
            "open_positions_estimate": 2,
            "grid_leg_index": 2,
            "grid_side": "SELL",
            "last_entry_price": 2200.0,
        }
        result = evaluate_grid_permission(
            timeframe="M5",
            setup="XAUUSD_M5_GRID_SCALPER_START",
            side="SELL",
            entry_price=2199.5,
            spread_points=20.0,
            max_spread_points=60.0,
            requires_m5=True,
            step_points=35.0,
            point_size=0.01,
            max_legs=10,
            state=state,
        )
        self.assertFalse(result.allowed)
        self.assertEqual(result.reason, "grid_cycle_already_active")


if __name__ == "__main__":
    unittest.main()
