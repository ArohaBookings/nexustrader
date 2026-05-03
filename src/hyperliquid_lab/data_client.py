from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import json
from urllib.request import Request, urlopen

import pandas as pd

from src.hyperliquid_lab.config import HyperliquidLabConfig, load_lab_config


UTC = timezone.utc


def _to_millis(value: datetime | int | float | pd.Timestamp) -> int:
    if isinstance(value, (int, float)):
        return int(value)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    return int(timestamp.tz_convert("UTC").timestamp() * 1000)


def _utc_timestamp(value: Any) -> pd.Timestamp:
    timestamp = pd.to_datetime(value, unit="ms" if isinstance(value, (int, float)) else None, utc=True)
    return pd.Timestamp(timestamp)


@dataclass
class HyperliquidDataClient:
    config: HyperliquidLabConfig | None = None
    timeout_seconds: float = 10.0

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = load_lab_config()

    def fetch_native_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime | int | float | pd.Timestamp,
        end_time: datetime | int | float | pd.Timestamp,
    ) -> pd.DataFrame:
        payload = {
            "type": "candleSnapshot",
            "req": {
                "coin": str(symbol).upper(),
                "interval": str(timeframe),
                "startTime": _to_millis(start_time),
                "endTime": _to_millis(end_time),
            },
        }
        response = self._post_info(payload)
        if not isinstance(response, list):
            raise ValueError("Hyperliquid candleSnapshot response must be a list")
        return self.normalize_candle_snapshot(response, source="hyperliquid_native", venue="hyperliquid")

    def fetch_native_order_book(self, symbol: str, depth: int = 20) -> pd.DataFrame:
        payload = {"type": "l2Book", "coin": str(symbol).upper()}
        response = self._post_info(payload)
        if not isinstance(response, dict):
            raise ValueError("Hyperliquid l2Book response must be a mapping")
        return self.normalize_l2_book(response, source="hyperliquid_native", venue="hyperliquid", depth=depth)

    def fetch_trades(
        self,
        symbol: str,
        since: datetime | int | float | pd.Timestamp | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        exchange = self._ccxt_exchange(str(self.config.ccxt_exchange_id if self.config else "hyperliquid"))
        market_symbol = f"{str(symbol).upper()}/USDC:USDC"
        rows = exchange.fetch_trades(market_symbol, since=_to_millis(since) if since is not None else None, limit=limit)
        return self.normalize_ccxt_trades(rows, source="hyperliquid_native", venue="hyperliquid", symbol=str(symbol).upper())

    def fetch_proxy_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime | int | float | pd.Timestamp | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        if self.config is None:
            raise ValueError("config is required")
        exchange = self._ccxt_exchange(self.config.proxy_source)
        market_symbol = self.config.proxy_symbols.get(str(symbol).upper())
        if not market_symbol:
            raise KeyError(f"No proxy symbol configured for {symbol}")
        rows = exchange.fetch_ohlcv(market_symbol, timeframe=str(timeframe), since=_to_millis(since) if since is not None else None, limit=limit)
        return self.normalize_ccxt_ohlcv(
            rows,
            source=self.config.proxy_source,
            venue=self.config.proxy_source,
            symbol=str(symbol).upper(),
            timeframe=str(timeframe),
            data_quality="proxy_only",
        )

    def _post_info(self, payload: dict[str, Any]) -> Any:
        if self.config is None:
            raise ValueError("config is required")
        request = Request(
            f"{self.config.api_base_url}/info",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=float(self.timeout_seconds)) as response:
            return json.loads(response.read().decode("utf-8"))

    @staticmethod
    def _ccxt_exchange(exchange_id: str) -> Any:
        try:
            import ccxt  # type: ignore
        except ImportError as exc:
            raise RuntimeError("ccxt is required for exchange data pulls; install requirements.txt") from exc
        exchange_cls = getattr(ccxt, exchange_id)
        exchange = exchange_cls({"enableRateLimit": True})
        exchange.load_markets()
        return exchange

    @staticmethod
    def normalize_candle_snapshot(rows: list[dict[str, Any]], *, source: str, venue: str) -> pd.DataFrame:
        normalized: list[dict[str, Any]] = []
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        for item in rows:
            open_ms = int(item["t"])
            close_ms = int(item["T"])
            normalized.append(
                {
                    "source": str(source),
                    "venue": str(venue),
                    "symbol": str(item["s"]).upper(),
                    "timeframe": str(item["i"]),
                    "open_time_utc": _utc_timestamp(open_ms),
                    "close_time_utc": _utc_timestamp(close_ms),
                    "open": float(item["o"]),
                    "high": float(item["h"]),
                    "low": float(item["l"]),
                    "close": float(item["c"]),
                    "volume": float(item["v"]),
                    "is_closed": close_ms <= now_ms,
                    "data_quality": "native",
                }
            )
        return pd.DataFrame(normalized, columns=OHLCV_COLUMNS).sort_values("close_time_utc").reset_index(drop=True)

    @staticmethod
    def normalize_ccxt_ohlcv(
        rows: list[list[Any]],
        *,
        source: str,
        venue: str,
        symbol: str,
        timeframe: str,
        data_quality: str,
    ) -> pd.DataFrame:
        duration = timeframe_to_timedelta(timeframe)
        normalized = []
        for item in rows:
            open_time = _utc_timestamp(int(item[0]))
            close_time = open_time + duration
            normalized.append(
                {
                    "source": str(source),
                    "venue": str(venue),
                    "symbol": str(symbol).upper(),
                    "timeframe": str(timeframe),
                    "open_time_utc": open_time,
                    "close_time_utc": close_time,
                    "open": float(item[1]),
                    "high": float(item[2]),
                    "low": float(item[3]),
                    "close": float(item[4]),
                    "volume": float(item[5]),
                    "is_closed": True,
                    "data_quality": str(data_quality),
                }
            )
        return pd.DataFrame(normalized, columns=OHLCV_COLUMNS).sort_values("close_time_utc").reset_index(drop=True)

    @staticmethod
    def normalize_ccxt_trades(rows: list[dict[str, Any]], *, source: str, venue: str, symbol: str) -> pd.DataFrame:
        normalized = []
        for item in rows:
            normalized.append(
                {
                    "source": str(source),
                    "venue": str(venue),
                    "symbol": str(symbol).upper(),
                    "trade_id": str(item.get("id") or item.get("info", {}).get("tid") or ""),
                    "timestamp_utc": _utc_timestamp(int(item["timestamp"])),
                    "side": str(item.get("side", "")).lower(),
                    "price": float(item["price"]),
                    "size": float(item["amount"]),
                }
            )
        return pd.DataFrame(normalized, columns=TRADES_COLUMNS).sort_values("timestamp_utc").reset_index(drop=True)

    @staticmethod
    def normalize_l2_book(response: dict[str, Any], *, source: str, venue: str, depth: int = 20) -> pd.DataFrame:
        levels = response.get("levels", [[], []])
        if not isinstance(levels, list) or len(levels) != 2:
            raise ValueError("l2Book response missing bid/ask levels")
        timestamp = _utc_timestamp(int(response.get("time", pd.Timestamp.now(tz="UTC").timestamp() * 1000)))
        symbol = str(response.get("coin", "")).upper()
        rows: list[dict[str, Any]] = []
        for side, side_levels in (("bid", levels[0]), ("ask", levels[1])):
            for index, item in enumerate(list(side_levels)[: int(depth)], start=1):
                rows.append(
                    {
                        "source": str(source),
                        "venue": str(venue),
                        "symbol": symbol,
                        "timestamp_utc": timestamp,
                        "side": side,
                        "level": int(index),
                        "price": float(item["px"]),
                        "size": float(item["sz"]),
                        "order_count": int(item.get("n", 0)),
                    }
                )
        return pd.DataFrame(rows, columns=BOOK_COLUMNS)


def timeframe_to_timedelta(timeframe: str) -> pd.Timedelta:
    value = str(timeframe)
    unit = value[-1]
    amount = int(value[:-1])
    if unit == "m":
        return pd.Timedelta(minutes=amount)
    if unit == "h":
        return pd.Timedelta(hours=amount)
    if unit == "d":
        return pd.Timedelta(days=amount)
    raise ValueError(f"Unsupported timeframe: {timeframe}")


OHLCV_COLUMNS = [
    "source",
    "venue",
    "symbol",
    "timeframe",
    "open_time_utc",
    "close_time_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "is_closed",
    "data_quality",
]

TRADES_COLUMNS = ["source", "venue", "symbol", "trade_id", "timestamp_utc", "side", "price", "size"]
BOOK_COLUMNS = ["source", "venue", "symbol", "timestamp_utc", "side", "level", "price", "size", "order_count"]
