from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.session_calendar import describe_market_state, dominant_session_name


class SessionCalendarTests(unittest.TestCase):
    def test_btc_is_open_24_7(self) -> None:
        state = describe_market_state("BTCUSD", datetime(2026, 3, 8, 12, 0, tzinfo=timezone.utc))
        self.assertTrue(state.market_open)
        self.assertEqual(state.market_open_status, "OPEN_24_7")

    def test_forex_shows_prep_window_before_sunday_new_york_open(self) -> None:
        state = describe_market_state("EURUSD", datetime(2026, 3, 8, 19, 30, tzinfo=timezone.utc))
        self.assertFalse(state.market_open)
        self.assertEqual(state.market_open_status, "PREP_WINDOW")
        self.assertTrue(state.pre_open_window_active)
        self.assertTrue(bool(state.next_open_time_utc))

    def test_metals_use_later_sunday_open_than_forex(self) -> None:
        state = describe_market_state("XAUUSD", datetime(2026, 3, 8, 21, 30, tzinfo=timezone.utc))
        self.assertFalse(state.market_open)
        self.assertEqual(state.market_open_status, "PRE_OPEN")

    def test_dominant_session_name_uses_dst_aware_local_windows(self) -> None:
        london = dominant_session_name(datetime(2026, 3, 9, 8, 30, tzinfo=timezone.utc))
        overlap = dominant_session_name(datetime(2026, 3, 9, 13, 30, tzinfo=timezone.utc))
        self.assertEqual(london, "LONDON")
        self.assertEqual(overlap, "OVERLAP")


if __name__ == "__main__":
    unittest.main()
