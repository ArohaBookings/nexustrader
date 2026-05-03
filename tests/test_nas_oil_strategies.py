from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import unittest

from src.strategies.nas_session_scalper import evaluate_nas_session_scalper
from src.strategies.oil_inventory_scalper import evaluate_oil_inventory_scalper, is_eia_inventory_window

UTC = timezone.utc


class NasOilStrategyTests(unittest.TestCase):
    def test_nas_orb_allows_cash_open_with_valid_spread(self) -> None:
        now_utc = datetime(2026, 3, 5, 14, 35, tzinfo=UTC)  # 09:35 NY (EST)
        decision = evaluate_nas_session_scalper(
            now_utc=now_utc,
            session_name="OVERLAP",
            regime="TRENDING",
            setup="NAS_ORB_RETEST",
            spread_points=24.0,
            confluence_score=0.72,
            spread_caps_by_session={"CASH_OPEN": 32.0, "OVERLAP": 36.0, "DEFAULT": 30.0},
            trade_rate_targets_by_session={"CASH_OPEN": 4.0, "DEFAULT": 1.5},
            confluence_floor=0.62,
            asia_enabled=False,
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.phase, "CASH_OPEN")

    def test_nas_blocks_when_spread_too_wide(self) -> None:
        now_utc = datetime(2026, 3, 5, 14, 35, tzinfo=UTC)
        decision = evaluate_nas_session_scalper(
            now_utc=now_utc,
            session_name="OVERLAP",
            regime="TRENDING",
            setup="NAS_ORB_RETEST",
            spread_points=50.0,
            confluence_score=0.90,
            spread_caps_by_session={"CASH_OPEN": 32.0, "OVERLAP": 36.0, "DEFAULT": 30.0},
            trade_rate_targets_by_session={"CASH_OPEN": 4.0, "DEFAULT": 1.5},
            confluence_floor=0.62,
            asia_enabled=False,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "spread_too_wide_session")

    def test_oil_news_armed_window_on_eia_release(self) -> None:
        ny_tz = ZoneInfo("America/New_York")
        local_armed = datetime(2026, 3, 4, 10, 35, tzinfo=ny_tz)  # Wednesday
        local_disarmed = datetime(2026, 3, 4, 11, 10, tzinfo=ny_tz)
        self.assertTrue(is_eia_inventory_window(now_utc=local_armed.astimezone(UTC), timezone_name="America/New_York"))
        self.assertFalse(is_eia_inventory_window(now_utc=local_disarmed.astimezone(UTC), timezone_name="America/New_York"))

        armed_decision = evaluate_oil_inventory_scalper(
            now_utc=local_armed.astimezone(UTC),
            session_name="NEW_YORK",
            spread_points=20.0,
            confluence_score=0.82,
            probability=0.85,
            regime="TRENDING",
            news_status="blocked_high_oil_inventory",
            atr_ratio=1.1,
            snapshot_age_seconds=15.0,
            spread_caps_by_session={"NEW_YORK": 44.0, "DEFAULT": 40.0},
            base_confluence_floor=0.64,
            news_armed_enabled=True,
            eia_window_minutes_pre=10,
            eia_window_minutes_post=30,
            stricter_confluence_floor=0.74,
            stricter_spread_cap=32.0,
            volatility_cap=1.3,
            asia_enabled=False,
        )
        self.assertTrue(armed_decision.allowed)
        self.assertTrue(armed_decision.news_armed)

        disarmed_decision = evaluate_oil_inventory_scalper(
            now_utc=local_disarmed.astimezone(UTC),
            session_name="NEW_YORK",
            spread_points=20.0,
            confluence_score=0.72,
            probability=0.75,
            regime="TRENDING",
            news_status="clear",
            atr_ratio=1.1,
            snapshot_age_seconds=15.0,
            spread_caps_by_session={"NEW_YORK": 44.0, "DEFAULT": 40.0},
            base_confluence_floor=0.64,
            news_armed_enabled=True,
            eia_window_minutes_pre=10,
            eia_window_minutes_post=30,
            stricter_confluence_floor=0.74,
            stricter_spread_cap=32.0,
            volatility_cap=1.3,
            asia_enabled=False,
        )
        self.assertTrue(disarmed_decision.allowed)
        self.assertFalse(disarmed_decision.news_armed)


if __name__ == "__main__":
    unittest.main()
