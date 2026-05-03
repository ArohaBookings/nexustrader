from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.session_profile import SessionProfile


class SessionProfileTests(unittest.TestCase):
    def test_tokyo_detection(self) -> None:
        profile = SessionProfile.from_config({})
        session = profile.classify(datetime(2026, 3, 5, 2, 0, tzinfo=timezone.utc))
        self.assertEqual(session.session_name, "TOKYO")
        self.assertTrue(session.in_session)

    def test_overlap_detection(self) -> None:
        profile = SessionProfile.from_config({})
        session = profile.classify(datetime(2026, 3, 5, 14, 30, tzinfo=timezone.utc))
        self.assertEqual(session.session_name, "OVERLAP")
        self.assertIn("TREND", session.allowed_strategies)

    def test_sydney_wraparound_detection(self) -> None:
        profile = SessionProfile.from_config({})
        session = profile.classify(datetime(2026, 3, 5, 23, 30, tzinfo=timezone.utc))
        self.assertEqual(session.session_name, "SYDNEY")
        self.assertGreater(session.ai_threshold_offset, 0.0)


if __name__ == "__main__":
    unittest.main()
