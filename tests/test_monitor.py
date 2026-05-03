from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.monitor import KillSwitch, _sydney_day_key


class KillSwitchTests(unittest.TestCase):
    def test_hard_kill_auto_clears_after_ttl_into_recovery_mode(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "kill_switch.lock"
            switch = KillSwitch(path)
            created_at = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
            session_key = _sydney_day_key(created_at)
            switch.activate(
                "HARD",
                "hard_daily_dd",
                now=created_at,
                session_key=session_key,
                last_equity=70.0,
                hard_ttl_hours=6.0,
            )

            status = switch.status(
                now=created_at.replace(hour=7),
                equity=65.0,
                current_session_key=session_key,
                hard_ttl_hours=6.0,
                sydney_reset_enabled=True,
            )

            self.assertIsNone(status.level)
            self.assertEqual(status.auto_clear_reason, "hard_kill_ttl_elapsed")
            self.assertEqual(status.recovery_mode, "RECOVERY_DEFENSIVE")

    def test_hard_kill_auto_clears_on_new_sydney_day(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "kill_switch.lock"
            switch = KillSwitch(path)
            created_at = datetime(2026, 3, 20, 8, 0, tzinfo=timezone.utc)
            session_key = _sydney_day_key(created_at)
            next_day = datetime(2026, 3, 21, 22, 0, tzinfo=timezone.utc)
            next_session_key = _sydney_day_key(next_day)
            self.assertNotEqual(session_key, next_session_key)
            switch.activate(
                "HARD",
                "absolute_drawdown_hard_stop",
                now=created_at,
                session_key=session_key,
                last_equity=55.0,
                hard_ttl_hours=6.0,
            )

            status = switch.status(
                now=next_day,
                equity=58.0,
                current_session_key=next_session_key,
                hard_ttl_hours=6.0,
                sydney_reset_enabled=True,
            )

            self.assertIsNone(status.level)
            self.assertTrue(str(status.auto_clear_reason or "").startswith("sydney_reset:"))
            self.assertEqual(status.recovery_mode, "RECOVERY_DEFENSIVE")

    def test_recovery_progress_is_persisted(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "kill_switch.lock"
            switch = KillSwitch(path)
            now = datetime(2026, 3, 20, 0, 0, tzinfo=timezone.utc)
            switch.enter_recovery(
                reason="hard_daily_dd",
                now=now,
                session_key=_sydney_day_key(now),
                last_equity=80.0,
                auto_clear_reason="hard_kill_ttl_elapsed",
                recovery_mode="RECOVERY_DEFENSIVE",
                recovery_wins_needed=3,
            )
            switch.update_recovery_progress(wins_observed=2)

            status = switch.status(now=now, equity=80.0)

            self.assertIsNone(status.level)
            self.assertEqual(status.recovery_mode, "RECOVERY_DEFENSIVE")
            self.assertEqual(status.recovery_wins_needed, 3)
            self.assertEqual(status.recovery_wins_observed, 2)


if __name__ == "__main__":
    unittest.main()
