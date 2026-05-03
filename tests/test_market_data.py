from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from unittest.mock import patch

from src.external_market_data import TwelveDataMarketDataFallback, YahooMarketDataFallback
from src.market_data import MarketDataService


class _FakeMT5Client:
    def __init__(self) -> None:
        self.calls = 0
        self._status = {"mode": "external_live", "source": "yahoo_finance", "ready": True, "error": ""}

    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        self.calls += 1
        if self.calls == 1:
            end_time = pd.Timestamp.utcnow().floor("5min")
            return pd.DataFrame(
                {
                    "time": pd.date_range(end=end_time, periods=count, freq="5min", tz="UTC"),
                    "open": [1.0] * count,
                    "high": [1.1] * count,
                    "low": [0.9] * count,
                    "close": [1.05] * count,
                    "tick_volume": [10] * count,
                    "spread": [0] * count,
                    "real_volume": [10] * count,
                }
            )
        raise RuntimeError("provider_down")

    def market_data_status(self) -> dict:
        return dict(self._status)

    def uses_external_market_data(self) -> bool:
        return True


class MarketDataServiceTests(unittest.TestCase):
    def test_symbol_normalization_includes_eurgbp(self) -> None:
        self.assertEqual(MarketDataService._normalize_symbol_key("EURGBP.a"), "EURGBP")

    def test_external_fallback_supports_asia_native_pairs(self) -> None:
        fallback = YahooMarketDataFallback()

        self.assertTrue(fallback.supports("AUDJPY", "M5"))
        self.assertTrue(fallback.supports("NZDJPY", "M15"))
        self.assertTrue(fallback.supports("AUDNZD", "H1"))
        self.assertTrue(fallback.supports("EURGBP", "M5"))
        self.assertTrue(fallback.supports("EURJPY", "M5"))
        self.assertTrue(fallback.supports("GBPJPY", "H4"))

    def test_external_fallback_supports_dxy_and_daily_timeframe(self) -> None:
        yahoo = YahooMarketDataFallback()
        twelve = TwelveDataMarketDataFallback(api_key="test-key")

        self.assertTrue(yahoo.supports("DXY", "D1"))
        self.assertTrue(twelve.supports("DXY", "D1"))

    def test_external_fallback_supports_generic_equity_and_crypto_symbols(self) -> None:
        yahoo = YahooMarketDataFallback()
        twelve = TwelveDataMarketDataFallback(api_key="test-key")

        self.assertTrue(yahoo.supports("MSFT", "M15"))
        self.assertTrue(yahoo.supports("PEPEUSD", "M15"))
        self.assertTrue(twelve.supports("MSFT", "M15"))
        self.assertTrue(twelve.supports("PEPEUSD", "M15"))

    def test_cache_fallback_remains_truthful_when_external_feed_fails(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            service = MarketDataService(_FakeMT5Client(), Path(tmp_dir))
            first = service.fetch("BTCUSD", "M5", 3)
            second = service.fetch("BTCUSD", "M5", 3)
            status = service.status_for_symbol("BTCUSD")

        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 3)
        self.assertIn(status["runtime_market_data_mode"], {"cache_recent", "cache_fallback"})
        self.assertTrue(status["runtime_market_data_ready"])

    def test_cache_fallback_uses_recent_cache_when_tail_age_looks_stale(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            service = MarketDataService(_FakeMT5Client(), Path(tmp_dir))
            first = service.fetch("BTCUSD", "M5", 3)
            with patch("src.market_data.pd.Timestamp.now", return_value=pd.Timestamp("2026-04-01T00:00:00Z")):
                second = service.fetch("BTCUSD", "M5", 3)
                status = service.status_for_symbol("BTCUSD")

        self.assertEqual(len(first), 3)
        self.assertEqual(len(second), 3)
        self.assertEqual(status["runtime_market_data_mode"], "cache_fallback")
        self.assertTrue(status["runtime_market_data_ready"])

    def test_stale_cached_tail_does_not_short_circuit_live_fetch(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            client = _FakeMT5Client()
            service = MarketDataService(client, Path(tmp_dir))
            stale_end = pd.Timestamp("2026-03-19T00:00:00Z")
            stale = pd.DataFrame(
                {
                    "time": pd.date_range(end=stale_end, periods=3, freq="5min", tz="UTC"),
                    "open": [1.0, 1.0, 1.0],
                    "high": [1.1, 1.1, 1.1],
                    "low": [0.9, 0.9, 0.9],
                    "close": [1.05, 1.05, 1.05],
                    "tick_volume": [10, 10, 10],
                    "spread": [0, 0, 0],
                    "real_volume": [10, 10, 10],
                }
            )
            stale.to_parquet(service._cache_path("BTCUSD", "M5"), index=False)
            service._cache_meta_path("BTCUSD", "M5").write_text(
                '{"mode":"external_live","source":"yahoo_finance"}',
                encoding="utf-8",
            )
            frame = service.fetch("BTCUSD", "M5", 3)
            status = service.status_for_symbol("BTCUSD")

        self.assertEqual(client.calls, 1)
        self.assertEqual(len(frame), 3)
        self.assertEqual(status["runtime_market_data_mode"], "external_live")

    def test_optional_timeframe_error_does_not_mark_symbol_unavailable_when_other_frames_ready(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            service = MarketDataService(_FakeMT5Client(), Path(tmp_dir))
            service._status["AUDJPY"] = {
                "timeframes": {
                    "M5": {"mode": "external_live", "source": "yahoo_finance", "ready": True, "age_seconds": 0.0, "error": "", "latency_ms": 50},
                    "M15": {"mode": "external_live", "source": "yahoo_finance", "ready": True, "age_seconds": 0.0, "error": "", "latency_ms": 60},
                    "H1": {"mode": "external_live", "source": "yahoo_finance", "ready": True, "age_seconds": 0.0, "error": "", "latency_ms": 70},
                    "M1": {
                        "mode": "unavailable",
                        "source": "none",
                        "ready": False,
                        "age_seconds": None,
                        "error": "live_market_data_unavailable:AUDJPY:M1",
                        "latency_ms": None,
                    },
                }
            }
            status = service.status_for_symbol("AUDJPY")

        self.assertTrue(status["runtime_market_data_ready"])
        self.assertEqual(status["runtime_market_data_error"], "")

    def test_cache_meta_accepts_twelvedata_as_live_compatible_source(self) -> None:
        self.assertTrue(
            MarketDataService._cache_meta_is_live_compatible(
                {"mode": "external_live", "source": "twelve_data"}
            )
        )

    def test_fetch_drops_misaligned_partial_tail_bar(self) -> None:
        class _PartialTailClient(_FakeMT5Client):
            def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
                self.calls += 1
                return pd.DataFrame(
                    {
                        "time": pd.to_datetime(
                            [
                                "2026-03-19T00:05:00Z",
                                "2026-03-19T00:10:00Z",
                                "2026-03-19T00:15:27Z",
                            ],
                            utc=True,
                        ),
                        "open": [1.0, 1.1, 1.2],
                        "high": [1.1, 1.2, 1.2],
                        "low": [0.9, 1.0, 1.2],
                        "close": [1.05, 1.15, 1.2],
                        "tick_volume": [10, 11, 1],
                        "spread": [0, 0, 0],
                        "real_volume": [10, 11, 1],
                    }
                )

        with TemporaryDirectory() as tmp_dir:
            service = MarketDataService(_PartialTailClient(), Path(tmp_dir))
            frame = service.fetch("AUDJPY", "M5", 3)

        self.assertEqual(len(frame), 2)
        self.assertEqual(pd.Timestamp(frame.iloc[-1]["time"]), pd.Timestamp("2026-03-19T00:10:00Z"))

    def test_load_cached_drops_zero_range_open_bucket_placeholder(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            service = MarketDataService(_FakeMT5Client(), Path(tmp_dir))
            frame = pd.DataFrame(
                {
                    "time": pd.to_datetime(
                        [
                            "2026-03-19T00:10:00Z",
                            "2026-03-19T00:15:00Z",
                        ],
                        utc=True,
                    ),
                    "open": [1.1, 1.2],
                    "high": [1.2, 1.2],
                    "low": [1.0, 1.2],
                    "close": [1.15, 1.2],
                    "tick_volume": [11, 1],
                    "spread": [0, 0],
                    "real_volume": [11, 1],
                }
            )
            frame.to_parquet(service._cache_path("AUDJPY", "M5"), index=False)
            with patch("src.market_data.pd.Timestamp.now", return_value=pd.Timestamp("2026-03-19T00:17:00Z")):
                sanitized = service.load_cached("AUDJPY", "M5")

        self.assertIsNotNone(sanitized)
        assert sanitized is not None
        self.assertEqual(len(sanitized), 1)
        self.assertEqual(pd.Timestamp(sanitized.iloc[-1]["time"]), pd.Timestamp("2026-03-19T00:10:00Z"))

    def test_twelvedata_provider_parses_forex_timeseries(self) -> None:
        provider = TwelveDataMarketDataFallback(api_key="test-key")
        payload = {
            "meta": {"symbol": "EUR/USD", "interval": "5min"},
            "values": [
                {"datetime": "2026-03-11 10:00:00", "open": "1.0800", "high": "1.0810", "low": "1.0790", "close": "1.0805", "volume": "100"},
                {"datetime": "2026-03-11 10:05:00", "open": "1.0805", "high": "1.0815", "low": "1.0800", "close": "1.0812", "volume": "120"},
            ],
        }

        with patch.object(provider, "_request", return_value=payload):
            frame = provider.fetch_rates("EURUSD", "M5", 2)

        self.assertEqual(len(frame), 2)
        self.assertEqual(list(frame.columns), ["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"])
        self.assertAlmostEqual(float(frame.iloc[-1]["close"]), 1.0812)

    def test_twelvedata_provider_enters_rate_limit_cooldown(self) -> None:
        provider = TwelveDataMarketDataFallback(api_key="test-key", cooldown_seconds=30.0)

        with patch.object(provider, "_request", side_effect=RuntimeError("twelvedata_error:429:rate_limit")):
            with self.assertRaises(RuntimeError):
                provider.fetch_rates("EURUSD", "M5", 2)

        # Simulate the real request-level cooldown signal.
        provider.last_rate_limited_at = "2026-03-12T00:00:00+00:00"
        provider.rate_limit_cooldown_until = 10**9
        diagnostics = provider.diagnostics()

        self.assertTrue(diagnostics["rate_limit_cooldown_active"])
        self.assertEqual(diagnostics["source"], "twelve_data")

    def test_market_data_status_includes_provider_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            client = _FakeMT5Client()
            client._status["provider_diagnostics"] = {
                "last_source": "twelve_data",
                "providers": [{"source": "twelve_data", "rate_limit_cooldown_active": False}],
            }
            service = MarketDataService(client, Path(tmp_dir))
            service._status["EURUSD"] = {
                "timeframes": {
                    "M5": {
                        "mode": "external_live",
                        "source": "twelve_data",
                        "ready": True,
                        "age_seconds": 0.0,
                        "error": "",
                        "latency_ms": 42,
                        "provider_diagnostics": client._status["provider_diagnostics"],
                    }
                }
            }
            status = service.status_for_symbol("EURUSD")

        self.assertEqual(status["runtime_market_data_source"], "twelve_data")
        self.assertEqual(
            status["runtime_market_data_provider_diagnostics"]["last_source"],
            "twelve_data",
        )


if __name__ == "__main__":
    unittest.main()
