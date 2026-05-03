from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.utils import SessionWindow, parse_hhmm


class UtilsTests(unittest.TestCase):
    def test_session_contains_accepts_tz_aware_datetime(self) -> None:
        session = SessionWindow(
            name="overlap",
            start=parse_hhmm("13:00"),
            end=parse_hhmm("17:00"),
            enabled=True,
            size_multiplier=1.0,
        )
        aware_time = datetime(2026, 3, 4, 13, 30, tzinfo=timezone.utc)

        self.assertTrue(session.contains(aware_time))


if __name__ == "__main__":
    unittest.main()
