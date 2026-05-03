from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from src.trade_idea_lifecycle import (
    IDEA_ARCHIVED,
    IDEA_COOLDOWN,
    IDEA_ACTIVE,
    TradeIdeaLifecycle,
)

UTC = timezone.utc


class TradeIdeaLifecycleTests(unittest.TestCase):
    def test_rejected_idea_enters_cooldown_then_rechecks(self) -> None:
        manager = TradeIdeaLifecycle(
            recheck_seconds_default=60,
            recheck_seconds_by_session={"LONDON": 30},
            cooldown_seconds_by_session={"LONDON": 30},
        )
        now = datetime(2026, 3, 6, 7, 0, tzinfo=UTC)
        structure = {"rsi": 50.0, "momentum": 0.0, "liquidity_sweep": False, "confluence_score": 3.0, "zone_bucket": 1}
        idea = manager.upsert(
            symbol="BTCUSD",
            setup_type="BTC_TREND",
            side="BUY",
            confidence=0.60,
            confluence_score=3.0,
            entry_price=100000.0,
            atr=120.0,
            now=now,
            structure=structure,
        )
        manager.reject(idea=idea, reason="neutral", now=now, session_name="LONDON", structure=structure)
        self.assertEqual(idea.status, IDEA_COOLDOWN)

        allowed, reason = manager.can_evaluate(idea=idea, now=now + timedelta(seconds=10), session_name="LONDON", structure=structure)
        self.assertFalse(allowed)
        self.assertEqual(reason, "cooldown_active")

        allowed, reason = manager.can_evaluate(idea=idea, now=now + timedelta(seconds=31), session_name="LONDON", structure=structure)
        self.assertTrue(allowed)
        self.assertEqual(reason, "active")

    def test_archives_after_max_rejections(self) -> None:
        manager = TradeIdeaLifecycle(
            max_rechecks_per_idea=5,
            recheck_seconds_default=1,
            cooldown_seconds_by_session={"DEFAULT": 1},
        )
        now = datetime(2026, 3, 6, 7, 0, tzinfo=UTC)
        structure = {"rsi": 50.0, "momentum": 0.0, "liquidity_sweep": False, "confluence_score": 2.5, "zone_bucket": 2}
        idea = manager.upsert(
            symbol="XAUUSD",
            setup_type="XAUUSD_M5_GRID_SCALPER_START",
            side="BUY",
            confidence=0.55,
            confluence_score=2.5,
            entry_price=2200.0,
            atr=1.2,
            now=now,
            structure=structure,
        )
        for index in range(5):
            stamp = now + timedelta(seconds=index + 1)
            manager.reject(idea=idea, reason=f"reject-{index}", now=stamp, session_name="DEFAULT", structure=structure)
        self.assertEqual(idea.status, IDEA_ARCHIVED)

    def test_symbol_cap_archives_oldest_idea_first(self) -> None:
        manager = TradeIdeaLifecycle(max_active_ideas_per_symbol=3)
        now = datetime(2026, 3, 6, 7, 0, tzinfo=UTC)
        structure = {"rsi": 48.0, "momentum": 0.01, "liquidity_sweep": False, "confluence_score": 3.1, "zone_bucket": 1}
        created_ids: list[str] = []
        for index in range(4):
            idea = manager.upsert(
                symbol="EURUSD",
                setup_type="FOREX_TREND_PULLBACK",
                side="BUY",
                confidence=0.60,
                confluence_score=3.1,
                entry_price=1.08 + (index * 0.001),
                atr=0.002,
                now=now + timedelta(seconds=index),
                structure=structure,
            )
            created_ids.append(idea.id)
        archived = manager.ideas[created_ids[0]]
        self.assertEqual(archived.status, IDEA_ARCHIVED)
        active_count = sum(
            1
            for idea in manager.ideas.values()
            if idea.symbol == "EURUSD" and idea.status != IDEA_ARCHIVED
        )
        self.assertEqual(active_count, 3)

    def test_reactivation_on_structure_change(self) -> None:
        manager = TradeIdeaLifecycle(
            recheck_seconds_default=60,
            cooldown_seconds_by_session={"TOKYO": 60},
        )
        now = datetime(2026, 3, 6, 0, 0, tzinfo=UTC)
        neutral = {"rsi": 50.0, "momentum": 0.0, "liquidity_sweep": False, "confluence_score": 3.0, "zone_bucket": 5}
        idea = manager.upsert(
            symbol="NAS100",
            setup_type="NAS_SESSION_SCALPER",
            side="BUY",
            confidence=0.57,
            confluence_score=3.0,
            entry_price=18200.0,
            atr=45.0,
            now=now,
            structure=neutral,
        )
        manager.reject(idea=idea, reason="neutral", now=now, session_name="TOKYO", structure=neutral)
        self.assertEqual(idea.status, IDEA_COOLDOWN)

        improved = {"rsi": 65.0, "momentum": 0.05, "liquidity_sweep": True, "confluence_score": 3.8, "zone_bucket": 5}
        allowed, reason = manager.can_evaluate(idea=idea, now=now + timedelta(seconds=5), session_name="TOKYO", structure=improved)
        self.assertTrue(allowed)
        self.assertEqual(reason, "reactivated")
        self.assertEqual(idea.status, IDEA_ACTIVE)
        self.assertEqual(idea.evaluation_count, 0)

    def test_delivery_pending_allows_retry_after_short_window(self) -> None:
        manager = TradeIdeaLifecycle(recheck_seconds_default=60)
        now = datetime(2026, 3, 6, 8, 0, tzinfo=UTC)
        structure = {"rsi": 54.0, "momentum": 0.02, "liquidity_sweep": True, "confluence_score": 3.4, "zone_bucket": 4}
        idea = manager.upsert(
            symbol="XAUUSD",
            setup_type="XAUUSD_M5_GRID_SCALPER_START",
            side="BUY",
            confidence=0.67,
            confluence_score=3.4,
            entry_price=2198.5,
            atr=1.2,
            now=now,
            structure=structure,
        )

        manager.mark_delivery_pending(idea=idea, now=now, retry_after_seconds=20)
        self.assertEqual(idea.status, IDEA_COOLDOWN)

        allowed, reason = manager.can_evaluate(idea=idea, now=now + timedelta(seconds=10), session_name="LONDON", structure=structure)
        self.assertFalse(allowed)
        self.assertEqual(reason, "cooldown_active")

        allowed, reason = manager.can_evaluate(idea=idea, now=now + timedelta(seconds=21), session_name="LONDON", structure=structure)
        self.assertTrue(allowed)
        self.assertEqual(reason, "active")


if __name__ == "__main__":
    unittest.main()
