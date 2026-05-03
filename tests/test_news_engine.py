from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import unittest

from src.news_engine import NewsEngine
from src.utils import SessionWindow, parse_hhmm


class NewsEngineTests(unittest.TestCase):
    def test_investing_rss_supplements_primary_provider_bias_and_headlines(self) -> None:
        now = datetime(2026, 3, 4, 14, 0, tzinfo=timezone.utc)
        calls: list[str] = []

        def fake_http_get(url: str) -> str:
            calls.append(url)
            if "finnhub.io" in url:
                return json.dumps(
                    {
                        "economicCalendar": [
                            {
                                "id": "evt-fh-1",
                                "date": "2026-03-04T13:30:00+00:00",
                                "event": "US CPI",
                                "country": "US",
                                "impact": "high",
                            }
                        ]
                    }
                )
            return """<?xml version="1.0" encoding="utf-8"?>
<rss version="2.0">
  <channel>
    <title>Investing Forex News</title>
    <item>
      <title>Dollar weakens after softer Fed tone boosts gold</title>
      <link>https://www.investing.com/example</link>
      <guid>rss-1</guid>
      <pubDate>Wed, 04 Mar 2026 13:40:00 GMT</pubDate>
      <description>Macro risk eases and bullion demand improves.</description>
    </item>
  </channel>
</rss>"""

        original = os.environ.get("FINNHUB_API_KEY")
        os.environ["FINNHUB_API_KEY"] = "test-finnhub"
        try:
            with TemporaryDirectory() as tmp_dir:
                engine = NewsEngine(
                    cache_path=Path(tmp_dir) / "news_cache.json",
                    provider="finnhub",
                    api_base_url="https://finnhub.io/api/v1/calendar/economic",
                    api_key_env="FINNHUB_API_KEY",
                    cache_ttl_seconds=300,
                    block_high_impact=False,
                    block_medium_impact=False,
                    block_window_minutes_before=30,
                    block_window_minutes_after=15,
                    fail_open=False,
                    enabled=True,
                    http_get=fake_http_get,
                    supplemental_rss_feeds=[
                        {
                            "label": "Investing Forex News",
                            "url": "https://www.investing.com/rss/news_1.rss",
                            "enabled": True,
                            "max_items": 4,
                        }
                    ],
                )
                snapshot = engine.status_snapshot("XAUUSD", now)
        finally:
            if original is None:
                os.environ.pop("FINNHUB_API_KEY", None)
            else:
                os.environ["FINNHUB_API_KEY"] = original

        self.assertTrue(any("finnhub.io" in url for url in calls))
        self.assertTrue(any("investing.com/rss/news_1.rss" in url for url in calls))
        self.assertTrue(bool(snapshot["news_secondary_source_used"]))
        self.assertGreaterEqual(int(snapshot["news_rss_headline_count"]), 1)
        category_summary = dict(snapshot["news_category_summary"])
        self.assertIn("monetary_policy", category_summary)
        self.assertIn("inflation", category_summary)
        self.assertIn("Investing Forex News", str(snapshot["session_bias_summary"]))
        self.assertTrue(any("Investing" in str(item.get("source") or "") for item in snapshot["news_headlines"]))
        self.assertIn("event_directive", snapshot)
        self.assertIn(str(snapshot.get("event_playbook") or ""), {"wait_then_retest", "fade", "risk_on_follow", "breakout", "swing_hold"})

    def test_tradingeconomics_provider_builds_structured_event_directive(self) -> None:
        now = datetime(2026, 3, 24, 12, 0, tzinfo=timezone.utc)

        def fake_http_get(url: str) -> str:
            self.assertIn("api.tradingeconomics.com", url)
            return json.dumps(
                [
                    {
                        "CalendarId": "te-1",
                        "Date": "2026-03-24T12:30:00+00:00",
                        "Country": "United States",
                        "Category": "Inflation Rate MoM",
                        "Event": "US CPI",
                        "Importance": 3,
                    }
                ]
            )

        original = os.environ.get("TRADINGECONOMICS_API_KEY")
        os.environ["TRADINGECONOMICS_API_KEY"] = "guest:guest"
        try:
            with TemporaryDirectory() as tmp_dir:
                engine = NewsEngine(
                    cache_path=Path(tmp_dir) / "news_cache.json",
                    provider="tradingeconomics",
                    api_base_url="https://api.tradingeconomics.com/calendar",
                    api_key_env="TRADINGECONOMICS_API_KEY",
                    cache_ttl_seconds=300,
                    block_high_impact=False,
                    block_medium_impact=False,
                    block_window_minutes_before=30,
                    block_window_minutes_after=15,
                    fail_open=False,
                    enabled=True,
                    http_get=fake_http_get,
                )
                snapshot = engine.status_snapshot("XAUUSD", now)
        finally:
            if original is None:
                os.environ.pop("TRADINGECONOMICS_API_KEY", None)
            else:
                os.environ["TRADINGECONOMICS_API_KEY"] = original

        directive = dict(snapshot.get("event_directive") or {})
        self.assertEqual(str(snapshot.get("event_playbook") or ""), "wait_then_retest")
        self.assertEqual(str(directive.get("base_class") or ""), "inflation")
        self.assertEqual(str(snapshot.get("news_source_used") or ""), "tradingeconomics")

    def test_blocks_high_impact_relevant_event_from_api(self) -> None:
        now = datetime(2026, 3, 4, 13, 20, tzinfo=timezone.utc)
        payload = json.dumps(
            [
                {
                    "id": "evt-1",
                    "date": "2026-03-04T13:30:00+00:00",
                    "event": "US CPI",
                    "currency": "USD",
                    "impact": "High",
                }
            ]
        )

        def fake_http_get(url: str) -> str:
            return payload

        original = os.environ.get("NEWS_API_KEY")
        os.environ["NEWS_API_KEY"] = "test-key"
        try:
            with TemporaryDirectory() as tmp_dir:
                engine = NewsEngine(
                    cache_path=Path(tmp_dir) / "news_cache.json",
                    provider="fmp",
                    api_base_url="https://example.test/calendar",
                    api_key_env="NEWS_API_KEY",
                    cache_ttl_seconds=300,
                    block_high_impact=True,
                    block_medium_impact=False,
                    block_window_minutes_before=30,
                    block_window_minutes_after=15,
                    fail_open=False,
                    enabled=True,
                    http_get=fake_http_get,
                )
                safe, reason, next_safe = engine.is_safe_to_trade("XAUUSD", now)
        finally:
            if original is None:
                os.environ.pop("NEWS_API_KEY", None)
            else:
                os.environ["NEWS_API_KEY"] = original

        self.assertFalse(safe)
        self.assertIn("blocked_high_usd", reason)
        self.assertIsNotNone(next_safe)

    def test_uses_caution_instead_of_hard_block_when_api_missing(self) -> None:
        now = datetime(2026, 3, 6, 13, 20, tzinfo=timezone.utc)
        with TemporaryDirectory() as tmp_dir:
            engine = NewsEngine(
                cache_path=Path(tmp_dir) / "news_cache.json",
                provider="fmp",
                api_base_url="https://example.test/calendar",
                api_key_env="MISSING_NEWS_API_KEY",
                cache_ttl_seconds=300,
                block_high_impact=True,
                block_medium_impact=False,
                block_window_minutes_before=30,
                block_window_minutes_after=15,
                fail_open=False,
                enabled=True,
                fallback_session_windows=[
                    SessionWindow(
                        name="main",
                        start=parse_hhmm("13:00"),
                        end=parse_hhmm("17:00"),
                        enabled=True,
                        size_multiplier=1.0,
                    )
                ],
            )
            decision = engine.evaluate("EURUSD", now)

        self.assertTrue(decision.safe)
        self.assertEqual("NEWS_CAUTION", decision.state)
        self.assertEqual("news_caution_provider_unavailable", decision.reason)

    def test_reuses_stale_cache_before_hard_unknown(self) -> None:
        now = datetime(2026, 3, 6, 13, 20, tzinfo=timezone.utc)
        with TemporaryDirectory() as tmp_dir:
            cache_path = Path(tmp_dir) / "news_cache.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "fetched_at": "2026-03-06T12:30:00+00:00",
                        "expires_at": "2026-03-06T12:35:00+00:00",
                        "source": "reuters",
                        "api_ok": True,
                        "events": [
                            {
                                "event_id": "evt-1",
                                "timestamp": "2026-03-06T14:00:00+00:00",
                                "title": "US CPI",
                                "currency": "USD",
                                "impact": "HIGH",
                                "source": "Reuters",
                                "is_major_risk": True,
                            }
                        ],
                    }
                )
            )
            engine = NewsEngine(
                cache_path=cache_path,
                provider="fmp",
                api_base_url="https://example.test/calendar",
                api_key_env="MISSING_NEWS_API_KEY",
                cache_ttl_seconds=300,
                block_high_impact=True,
                block_medium_impact=False,
                block_window_minutes_before=30,
                block_window_minutes_after=15,
                fail_open=False,
                enabled=True,
            )
            snapshot = engine.status_snapshot("EURUSD", now)

        decision = snapshot["decision"]
        self.assertTrue(decision.safe)
        self.assertEqual("NEWS_CAUTION", decision.state)
        self.assertEqual("cached_fallback", snapshot["news_data_quality"])
        self.assertTrue(snapshot["news_fallback_used"])

    def test_news_api_env_key_takes_precedence_over_config_key(self) -> None:
        requested_urls: list[str] = []

        def fake_http_get(url: str) -> str:
            requested_urls.append(url)
            return json.dumps([])

        original = os.environ.get("NEWS_API_KEY")
        os.environ["NEWS_API_KEY"] = "env-key"
        try:
            with TemporaryDirectory() as tmp_dir:
                engine = NewsEngine(
                    cache_path=Path(tmp_dir) / "news_cache.json",
                    provider="fmp",
                    api_base_url="https://example.test/calendar",
                    api_key_env="NEWS_API_KEY",
                    cache_ttl_seconds=300,
                    block_high_impact=True,
                    block_medium_impact=False,
                    block_window_minutes_before=30,
                    block_window_minutes_after=15,
                    fail_open=False,
                    enabled=True,
                    api_key="config-key",
                    http_get=fake_http_get,
                )
                engine.is_safe_to_trade("XAUUSD", datetime(2026, 3, 4, 12, 0, tzinfo=timezone.utc))
        finally:
            if original is None:
                os.environ.pop("NEWS_API_KEY", None)
            else:
                os.environ["NEWS_API_KEY"] = original

        self.assertTrue(requested_urls)
        self.assertIn("apikey=env-key", requested_urls[0])

    def test_news_decision_exposes_source_confidence_and_authenticity_risk(self) -> None:
        now = datetime(2026, 3, 4, 13, 20, tzinfo=timezone.utc)
        payload = json.dumps(
            [
                {
                    "id": "evt-1",
                    "date": "2026-03-04T13:10:00+00:00",
                    "event": "Breaking rumor of emergency Fed meeting",
                    "currency": "USD",
                    "impact": "High",
                    "source": "Twitter Macro Feed",
                }
            ]
        )

        def fake_http_get(url: str) -> str:
            return payload

        with TemporaryDirectory() as tmp_dir:
            engine = NewsEngine(
                cache_path=Path(tmp_dir) / "news_cache.json",
                provider="fmp",
                api_base_url="https://example.test/calendar",
                api_key_env="MISSING_NEWS_API_KEY",
                cache_ttl_seconds=300,
                block_high_impact=True,
                block_medium_impact=False,
                block_window_minutes_before=30,
                block_window_minutes_after=15,
                fail_open=True,
                enabled=True,
                api_key="config-key",
                http_get=fake_http_get,
            )
            decision = engine.evaluate("XAUUSD", now)

        self.assertLess(float(decision.source_confidence), 0.5)
        self.assertGreater(float(decision.authenticity_risk), 0.3)

    def test_usoil_inventory_draw_sets_bullish_high_probability_bias(self) -> None:
        now = datetime(2026, 3, 4, 14, 0, tzinfo=timezone.utc)
        payload = json.dumps(
            [
                {
                    "id": "oil-1",
                    "date": "2026-03-04T13:45:00+00:00",
                    "event": "US crude inventory draw deepens after fresh OPEC cut",
                    "currency": "USD",
                    "impact": "High",
                    "source": "Reuters",
                }
            ]
        )

        def fake_http_get(url: str) -> str:
            return payload

        with TemporaryDirectory() as tmp_dir:
            engine = NewsEngine(
                cache_path=Path(tmp_dir) / "news_cache.json",
                provider="fmp",
                api_base_url="https://example.test/calendar",
                api_key_env="MISSING_NEWS_API_KEY",
                cache_ttl_seconds=300,
                block_high_impact=False,
                block_medium_impact=False,
                block_window_minutes_before=30,
                block_window_minutes_after=15,
                fail_open=True,
                enabled=True,
                api_key="config-key",
                http_get=fake_http_get,
            )
            snapshot = engine.status_snapshot("USOIL", now)

        decision = snapshot["decision"]
        self.assertEqual(str(decision.bias_direction), "bullish")
        self.assertGreaterEqual(float(decision.bias_confidence), 0.70)
        self.assertTrue(bool(snapshot["high_probability_bias"]))

    def test_finnhub_provider_blocks_high_impact_calendar_event(self) -> None:
        now = datetime(2026, 3, 4, 13, 20, tzinfo=timezone.utc)
        payload = json.dumps(
            {
                "economicCalendar": [
                    {
                        "id": "evt-fh-1",
                        "date": "2026-03-04T13:30:00+00:00",
                        "event": "US CPI",
                        "country": "US",
                        "impact": "high",
                    }
                ]
            }
        )

        def fake_http_get(url: str) -> str:
            self.assertIn("token=test-finnhub", url)
            return payload

        original = os.environ.get("FINNHUB_API_KEY")
        os.environ["FINNHUB_API_KEY"] = "test-finnhub"
        try:
            with TemporaryDirectory() as tmp_dir:
                engine = NewsEngine(
                    cache_path=Path(tmp_dir) / "news_cache.json",
                    provider="finnhub",
                    api_base_url="https://finnhub.io/api/v1/calendar/economic",
                    api_key_env="FINNHUB_API_KEY",
                    cache_ttl_seconds=300,
                    block_high_impact=True,
                    block_medium_impact=False,
                    block_window_minutes_before=30,
                    block_window_minutes_after=15,
                    fail_open=False,
                    enabled=True,
                    http_get=fake_http_get,
                )
                decision = engine.evaluate("XAUUSD", now)
        finally:
            if original is None:
                os.environ.pop("FINNHUB_API_KEY", None)
            else:
                os.environ["FINNHUB_API_KEY"] = original

        self.assertFalse(decision.safe)
        self.assertEqual(decision.source_used, "finnhub")
        self.assertEqual(decision.state, "NEWS_BLOCKED")

    def test_provider_unavailable_degrades_open_for_btc_instead_of_hard_blocking(self) -> None:
        now = datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc)
        with TemporaryDirectory() as tmp_dir:
            engine = NewsEngine(
                cache_path=Path(tmp_dir) / "news_cache.json",
                provider="fmp",
                api_base_url="https://example.test/calendar",
                api_key_env="MISSING_NEWS_API_KEY",
                cache_ttl_seconds=300,
                block_high_impact=True,
                block_medium_impact=False,
                block_window_minutes_before=30,
                block_window_minutes_after=15,
                fail_open=False,
                enabled=True,
                fallback_session_windows=[
                    SessionWindow(
                        name="main",
                        start=parse_hhmm("08:00"),
                        end=parse_hhmm("17:00"),
                        enabled=True,
                        size_multiplier=1.0,
                    )
                ],
            )
            decision = engine.evaluate("BTCUSD", now)

        self.assertTrue(decision.safe)
        self.assertEqual("news_caution_crypto_allowed", decision.reason)
        self.assertEqual("NEWS_CAUTION", decision.state)


if __name__ == "__main__":
    unittest.main()
