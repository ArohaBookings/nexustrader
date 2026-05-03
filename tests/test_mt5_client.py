from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.mt5_client import MT5Client, MT5Credentials


class FakeLogger:
    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    def info(self, message: str) -> None:
        self.messages.append(("info", message))

    def warning(self, message: str) -> None:
        self.messages.append(("warning", message))


class FakeMT5Module:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def initialize(self, **kwargs):
        self.calls.append(dict(kwargs))
        return len(self.calls) >= 2

    def shutdown(self):
        return None

    def last_error(self):
        return (5001, "init failed")


class _TupleLike:
    def __init__(self, **kwargs) -> None:
        self._payload = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def _asdict(self):
        return dict(self._payload)


class VerifyMT5Module(FakeMT5Module):
    def account_info(self):
        return _TupleLike(login=123456, server="Demo", leverage=100, balance=1000.0, equity=1002.5, margin_free=900.0)

    def terminal_info(self):
        return _TupleLike(name="MetaTrader 5", connected=True)

    def version(self):
        return (5, 0, 45)

    def symbols_get(self):
        return [_TupleLike(name="XAUUSD"), _TupleLike(name="EURUSD")]

    def symbol_select(self, symbol, enabled):
        return True


class RatesFallbackMT5Module(FakeMT5Module):
    TIMEFRAME_M5 = 5

    def copy_rates_from_pos(self, symbol, timeframe, start, count):
        return None


class MT5ClientTests(unittest.TestCase):
    def test_symbol_normalization_includes_eurgbp(self) -> None:
        self.assertEqual(MT5Client._normalize_symbol_key("EURGBP.a"), "EURGBP")

    def test_connect_tries_terminal_path_after_default_attempt(self) -> None:
        fake_module = FakeMT5Module()
        logger = FakeLogger()
        with TemporaryDirectory() as tmp_dir:
            client = MT5Client(
                credentials=MT5Credentials(login=1, password="pw", server="srv", terminal_path="C:\\Program Files\\MetaTrader\\terminal64.exe"),
                journal_db=Path(tmp_dir) / "journal.sqlite",
                logger=logger,
                mt5_loader=lambda: fake_module,
            )

            connected = client.connect()

        self.assertTrue(connected)
        self.assertEqual(len(fake_module.calls), 2)
        self.assertNotIn("path", fake_module.calls[0])
        self.assertEqual(fake_module.calls[1]["path"], "C:\\Program Files\\MetaTrader\\terminal64.exe")

    def test_connect_logs_mac_warning_on_failure(self) -> None:
        logger = FakeLogger()
        with TemporaryDirectory() as tmp_dir:
            client = MT5Client(
                credentials=MT5Credentials(),
                journal_db=Path(tmp_dir) / "journal.sqlite",
                logger=logger,
                platform_name="darwin",
                mt5_loader=lambda: None,
            )
            connected = client.connect()

        self.assertFalse(connected)
        self.assertTrue(any("Windows VM or VPS" in message for _, message in logger.messages))

    def test_verify_connection_returns_account_and_symbols(self) -> None:
        fake_module = VerifyMT5Module()
        logger = FakeLogger()
        with TemporaryDirectory() as tmp_dir:
            client = MT5Client(
                credentials=MT5Credentials(login=1, password="pw", server="srv", terminal_path="C:\\Program Files\\MetaTrader 5\\terminal64.exe"),
                journal_db=Path(tmp_dir) / "journal.sqlite",
                logger=logger,
                mt5_loader=lambda: fake_module,
            )
            verification = client.verify_connection(["XAUUSD", "EURUSD"])

        self.assertTrue(verification["ok"])
        self.assertTrue(verification["connected"])
        self.assertEqual(verification["resolved_symbols"]["XAUUSD"], "XAUUSD")
        self.assertEqual(verification["account_summary"]["login"], 123456)

    def test_discover_symbol_universe_filters_by_asset_class(self) -> None:
        class DiscoveryModule(VerifyMT5Module):
            def symbols_get(self):
                return [
                    _TupleLike(name="XAGUSD", path="Metals\\Silver", description="Silver"),
                    _TupleLike(name="DOGEUSD", path="Crypto\\DOGE", description="Dogecoin"),
                    _TupleLike(name="AAPL.24H", path="Stocks\\US", description="Apple Inc."),
                    _TupleLike(name="EURUSD", path="Forex\\Majors", description="EURUSD"),
                ]

        with TemporaryDirectory() as tmp_dir:
            client = MT5Client(
                credentials=MT5Credentials(login=1, password="pw", server="srv"),
                journal_db=Path(tmp_dir) / "journal.sqlite",
                mt5_loader=lambda: DiscoveryModule(),
            )
            discovered = client.discover_symbol_universe(
                include_asset_classes={"commodity", "crypto", "equity"},
                max_per_class={"commodity": 1, "crypto": 1, "equity": 1},
                exclude_symbols={"EURUSD"},
            )

        self.assertEqual([item["symbol"] for item in discovered], ["XAGUSD", "DOGUSD", "AAPL"])

    def test_external_market_data_rates_used_when_mt5_python_unavailable(self) -> None:
        class _Provider:
            last_source = "twelve_data"
            def diagnostics(self):
                return {"last_source": "twelve_data", "providers": [{"source": "twelve_data"}]}

            def fetch_rates(self, symbol: str, timeframe: str, count: int):
                self.called = (symbol, timeframe, count)
                return pd.DataFrame(
                    {
                        "time": pd.date_range(end=pd.Timestamp.utcnow(), periods=3, freq="5min", tz="UTC"),
                        "open": [1.0, 1.1, 1.2],
                        "high": [1.1, 1.2, 1.3],
                        "low": [0.9, 1.0, 1.1],
                        "close": [1.05, 1.15, 1.25],
                        "tick_volume": [10, 12, 15],
                        "spread": [0, 0, 0],
                        "real_volume": [10, 12, 15],
                    }
                )

        with TemporaryDirectory() as tmp_dir:
            client = MT5Client(
                credentials=MT5Credentials(),
                journal_db=Path(tmp_dir) / "journal.sqlite",
                mt5_loader=lambda: None,
            )
            client._external_market_data = _Provider()  # type: ignore[assignment]
            frame = client.get_rates("BTCUSD", "M5", 3)

        self.assertEqual(len(frame), 3)
        self.assertEqual(client.market_data_status()["mode"], "external_live")
        self.assertEqual(client.market_data_status()["source"], "twelve_data")
        self.assertEqual(client.market_data_status()["provider_diagnostics"]["last_source"], "twelve_data")

    def test_external_market_data_tick_used_when_mt5_python_unavailable(self) -> None:
        class _Provider:
            last_source = "twelve_data"
            def diagnostics(self):
                return {"last_source": "twelve_data", "providers": [{"source": "twelve_data"}]}

            def fetch_tick(self, symbol: str):
                self.called = symbol
                return {"bid": 68000.0, "ask": 68001.0, "time": 1234567890, "source": "twelve_data"}

        with TemporaryDirectory() as tmp_dir:
            client = MT5Client(
                credentials=MT5Credentials(),
                journal_db=Path(tmp_dir) / "journal.sqlite",
                mt5_loader=lambda: None,
            )
            client._external_market_data = _Provider()  # type: ignore[assignment]
            tick = client.get_tick("BTCUSD")

        self.assertEqual(float(tick["bid"]), 68000.0)
        self.assertEqual(client.market_data_status()["mode"], "external_live")
        self.assertEqual(client.market_data_status()["provider_diagnostics"]["last_source"], "twelve_data")

    def test_disabled_mt5_symbol_info_uses_asset_aware_fallbacks(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            client = MT5Client(
                credentials=MT5Credentials(),
                journal_db=Path(tmp_dir) / "journal.sqlite",
                mt5_loader=lambda: None,
            )
            btc = client.get_symbol_info("BTCUSD")
            xau = client.get_symbol_info("XAUUSD")

        self.assertEqual(float(btc["trade_contract_size"]), 1.0)
        self.assertEqual(float(btc["trade_tick_size"]), 0.01)
        self.assertEqual(float(btc["trade_tick_value"]), 0.01)
        self.assertEqual(float(xau["trade_contract_size"]), 100.0)
        self.assertEqual(float(xau["trade_tick_size"]), 0.01)
        self.assertEqual(float(xau["trade_tick_value"]), 1.0)

    def test_external_market_data_rates_used_when_mt5_rates_fail(self) -> None:
        class _Provider:
            last_source = "twelve_data"
            def diagnostics(self):
                return {"last_source": "twelve_data", "providers": [{"source": "twelve_data"}]}

            def fetch_rates(self, symbol: str, timeframe: str, count: int):
                self.called = (symbol, timeframe, count)
                return pd.DataFrame(
                    {
                        "time": pd.date_range(end=pd.Timestamp.utcnow(), periods=3, freq="5min", tz="UTC"),
                        "open": [1.0, 1.1, 1.2],
                        "high": [1.1, 1.2, 1.3],
                        "low": [0.9, 1.0, 1.1],
                        "close": [1.05, 1.15, 1.25],
                        "tick_volume": [10, 12, 15],
                        "spread": [0, 0, 0],
                        "real_volume": [10, 12, 15],
                    }
                )

        with TemporaryDirectory() as tmp_dir:
            client = MT5Client(
                credentials=MT5Credentials(),
                journal_db=Path(tmp_dir) / "journal.sqlite",
                mt5_loader=lambda: RatesFallbackMT5Module(),
            )
            client.connected = True
            client._external_market_data = _Provider()  # type: ignore[assignment]
            frame = client.get_rates("AUDJPY", "M5", 3)

        self.assertEqual(len(frame), 3)
        self.assertEqual(client.market_data_status()["mode"], "external_live")
        self.assertEqual(client.market_data_status()["source"], "twelve_data")
        self.assertEqual(client.market_data_status()["provider_diagnostics"]["last_source"], "twelve_data")

    def test_external_market_data_failure_surfaces_provider_diagnostics(self) -> None:
        class _Provider:
            last_source = "twelve_data"
            def diagnostics(self):
                return {
                    "last_source": "none",
                    "last_errors": [{"source": "twelve_data", "error": "twelvedata_rate_limit_cooldown:20.0s"}],
                    "providers": [{"source": "twelve_data", "rate_limit_cooldown_active": True}],
                }

            def fetch_rates(self, symbol: str, timeframe: str, count: int):
                raise RuntimeError("external_market_data_failed:twelvedata_rate_limit_cooldown:20.0s")

        with TemporaryDirectory() as tmp_dir:
            client = MT5Client(
                credentials=MT5Credentials(),
                journal_db=Path(tmp_dir) / "journal.sqlite",
                mt5_loader=lambda: None,
            )
            client._external_market_data = _Provider()  # type: ignore[assignment]
            with self.assertRaises(RuntimeError):
                client.get_rates("EURUSD", "M5", 3)

        status = client.market_data_status()
        self.assertEqual(status["mode"], "external_unavailable")
        self.assertTrue(status["provider_diagnostics"]["providers"][0]["rate_limit_cooldown_active"])


if __name__ == "__main__":
    unittest.main()
