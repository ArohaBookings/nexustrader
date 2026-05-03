from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from src.position_manager import PositionManager


class PositionManagerTests(unittest.TestCase):
    def test_partial_and_breakeven_actions_are_planned(self) -> None:
        manager = PositionManager(
            partial_close_r=1.0,
            partial_close_fraction=0.5,
            trail_activation_r=1.0,
            trail_atr_multiple=1.0,
            time_stop_hours=4,
        )
        opened_at = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
        journal_positions = [
            {
                "signal_id": "sig-1",
                "ticket": "101",
                "side": "BUY",
                "symbol": "XAUUSD",
                "setup": "TREND_CONTINUATION",
                "regime": "TRENDING",
                "entry_price": 2200.0,
                "sl": 2199.0,
                "tp": 2202.0,
                "volume": 0.1,
                "opened_at": opened_at.isoformat(),
                "probability": 0.66,
                "partial_close_enabled": True,
                "trailing_enabled": True,
            }
        ]
        mt5_positions = [{"ticket": 101, "sl": 2199.0, "tp": 2202.0, "volume": 0.1}]

        manager.sync(journal_positions, mt5_positions)
        actions = manager.plan_actions({"XAUUSD": {"price": 2201.2, "atr": 0.4}}, current_time=datetime.now(tz=timezone.utc))
        action_types = [action.action for action in actions]

        self.assertIn("PARTIAL_CLOSE", action_types)
        self.assertIn("MOVE_SL", action_types)

    def test_time_stop_triggers_when_trade_stalls(self) -> None:
        manager = PositionManager(
            partial_close_r=1.0,
            partial_close_fraction=0.5,
            trail_activation_r=1.0,
            trail_atr_multiple=1.0,
            time_stop_hours=4,
        )
        opened_at = datetime.now(tz=timezone.utc) - timedelta(hours=5)
        journal_positions = [
            {
                "signal_id": "sig-2",
                "ticket": "202",
                "side": "SELL",
                "symbol": "USDJPY",
                "setup": "RANGE_REVERSAL",
                "regime": "RANGING",
                "entry_price": 2200.0,
                "sl": 2201.0,
                "tp": 2198.5,
                "volume": 0.1,
                "opened_at": opened_at.isoformat(),
                "probability": 0.58,
                "partial_close_enabled": True,
                "trailing_enabled": True,
            }
        ]
        mt5_positions = [{"ticket": 202, "sl": 2201.0, "tp": 2198.5, "volume": 0.1}]

        manager.sync(journal_positions, mt5_positions)
        actions = manager.plan_actions({"USDJPY": {"price": 2200.1, "atr": 0.4}}, current_time=datetime.now(tz=timezone.utc))
        self.assertTrue(any(action.action == "TIME_STOP" for action in actions))

    def test_trailing_stop_moves_monotonically_for_long(self) -> None:
        manager = PositionManager(
            partial_close_r=1.0,
            partial_close_fraction=0.5,
            trail_activation_r=1.0,
            trail_atr_multiple=1.0,
            time_stop_hours=4,
        )
        opened_at = datetime.now(tz=timezone.utc) - timedelta(minutes=20)
        journal_positions = [
            {
                "signal_id": "sig-trail-long",
                "ticket": "303",
                "side": "BUY",
                "symbol": "XAUUSD",
                "setup": "TREND_CONTINUATION",
                "regime": "TRENDING",
                "entry_price": 2200.0,
                "sl": 2199.0,
                "tp": 2203.0,
                "volume": 0.1,
                "opened_at": opened_at.isoformat(),
                "probability": 0.7,
                "partial_close_enabled": False,
                "trailing_enabled": True,
            }
        ]
        mt5_positions = [{"ticket": 303, "sl": 2199.0, "tp": 2203.0, "volume": 0.1}]
        manager.sync(journal_positions, mt5_positions)

        actions_up = manager.plan_actions(
            {"XAUUSD": {"price": 2202.2, "bid": 2202.0, "ask": 2202.2, "atr": 0.5}},
            current_time=datetime.now(tz=timezone.utc),
        )
        move_up = [action for action in actions_up if action.action == "MOVE_SL"]
        self.assertTrue(move_up)
        first_sl = max(float(action.new_sl or 0.0) for action in move_up)
        manager.update_sl(303, first_sl)

        actions_pullback = manager.plan_actions(
            {"XAUUSD": {"price": 2202.0, "bid": 2201.8, "ask": 2202.0, "atr": 0.5}},
            current_time=datetime.now(tz=timezone.utc),
        )
        trailing_pullback = [action for action in actions_pullback if action.reason == "trailing_stop_update"]
        self.assertFalse(trailing_pullback)

    def test_short_trailing_uses_ask_price(self) -> None:
        manager = PositionManager(
            partial_close_r=1.0,
            partial_close_fraction=0.5,
            trail_activation_r=1.0,
            trail_atr_multiple=1.0,
            time_stop_hours=4,
        )
        opened_at = datetime.now(tz=timezone.utc) - timedelta(minutes=20)
        journal_positions = [
            {
                "signal_id": "sig-trail-short",
                "ticket": "404",
                "side": "SELL",
                "symbol": "USDJPY",
                "setup": "TREND_CONTINUATION",
                "regime": "TRENDING",
                "entry_price": 2200.0,
                "sl": 2201.0,
                "tp": 2197.0,
                "volume": 0.1,
                "opened_at": opened_at.isoformat(),
                "probability": 0.7,
                "partial_close_enabled": False,
                "trailing_enabled": True,
            }
        ]
        mt5_positions = [{"ticket": 404, "sl": 2201.0, "tp": 2197.0, "volume": 0.1}]
        manager.sync(journal_positions, mt5_positions)

        actions = manager.plan_actions(
            {"USDJPY": {"price": 2198.2, "bid": 2198.0, "ask": 2198.4, "atr": 0.5}},
            current_time=datetime.now(tz=timezone.utc),
        )
        trailing = [action for action in actions if action.reason == "trailing_stop_update"]
        self.assertTrue(trailing)
        self.assertAlmostEqual(float(trailing[0].new_sl or 0.0), 2198.9, places=6)

    def test_ai_advice_cannot_widen_trailing_stop(self) -> None:
        manager = PositionManager(
            partial_close_r=1.0,
            partial_close_fraction=0.5,
            trail_activation_r=1.0,
            trail_atr_multiple=1.0,
            time_stop_hours=4,
        )
        opened_at = datetime.now(tz=timezone.utc) - timedelta(minutes=20)
        journal_positions = [
            {
                "signal_id": "sig-ai-tighten-only",
                "ticket": "505",
                "side": "BUY",
                "symbol": "XAUUSD",
                "setup": "TREND_CONTINUATION",
                "regime": "TRENDING",
                "entry_price": 2200.0,
                "sl": 2199.5,
                "tp": 2204.0,
                "volume": 0.1,
                "opened_at": opened_at.isoformat(),
                "probability": 0.7,
                "partial_close_enabled": False,
                "trailing_enabled": True,
            }
        ]
        mt5_positions = [{"ticket": 505, "sl": 2199.5, "tp": 2204.0, "volume": 0.1}]
        manager.sync(journal_positions, mt5_positions)
        actions = manager.plan_actions(
            {"XAUUSD": {"price": 2204.0, "bid": 2204.0, "ask": 2204.2, "atr": 1.0}},
            current_time=datetime.now(tz=timezone.utc),
            advice_by_ticket={
                505: {
                    "trail_mode": "ATR",
                    "trail_atr_mult": 2.0,
                }
            },
        )
        trailing = [action for action in actions if action.reason == "trailing_stop_update"]
        self.assertTrue(trailing)
        # Base trail uses 1.0 ATR => 2204.0 - 1.0 = 2203.0. Wider AI advice must be ignored.
        self.assertAlmostEqual(float(trailing[0].new_sl or 0.0), 2203.0, places=6)

    def test_partial_actions_are_suppressed_when_disabled(self) -> None:
        manager = PositionManager(
            partial_close_r=1.0,
            partial_close_fraction=0.5,
            trail_activation_r=1.0,
            trail_atr_multiple=1.0,
            time_stop_hours=4,
            allow_partial_closes=False,
        )
        opened_at = datetime.now(tz=timezone.utc) - timedelta(minutes=30)
        journal_positions = [
            {
                "signal_id": "sig-no-partials",
                "ticket": "606",
                "side": "BUY",
                "symbol": "XAUUSD",
                "setup": "TREND_CONTINUATION",
                "regime": "TRENDING",
                "entry_price": 2200.0,
                "sl": 2199.0,
                "tp": 2202.0,
                "volume": 0.1,
                "opened_at": opened_at.isoformat(),
                "probability": 0.66,
                "partial_close_enabled": True,
                "trailing_enabled": True,
            }
        ]
        mt5_positions = [{"ticket": 606, "sl": 2199.0, "tp": 2202.0, "volume": 0.1}]

        manager.sync(journal_positions, mt5_positions)
        actions = manager.plan_actions(
            {"XAUUSD": {"price": 2201.2, "bid": 2201.1, "ask": 2201.2, "atr": 0.4}},
            current_time=datetime.now(tz=timezone.utc),
        )
        action_types = [action.action for action in actions]

        self.assertNotIn("PARTIAL_CLOSE", action_types)
        self.assertIn("MOVE_SL", action_types)


if __name__ == "__main__":
    unittest.main()
