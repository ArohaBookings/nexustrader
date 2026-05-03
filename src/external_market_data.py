from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
import json
import os
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd

from src.symbol_universe import normalize_symbol_key, symbol_asset_class


UTC = "UTC"

_SYMBOL_MAP: dict[str, str] = {
    "BTCUSD": "BTC-USD",
    "XAUUSD": "GC=F",
    "XAGUSD": "SI=F",
    "DXY": "DX-Y.NYB",
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "EURGBP": "EURGBP=X",
    "USDJPY": "USDJPY=X",
    "AUDJPY": "AUDJPY=X",
    "NZDJPY": "NZDJPY=X",
    "AUDNZD": "AUDNZD=X",
    "EURJPY": "EURJPY=X",
    "GBPJPY": "GBPJPY=X",
    "NAS100": "NQ=F",
    "USOIL": "CL=F",
    "DOGUSD": "DOGE-USD",
    "TRUMPUSD": "TRUMP-USD",
    "AAPL": "AAPL",
    "NVIDIA": "NVDA",
}

_TIMEFRAME_MINUTES: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
}

_INTERVAL_MAP: dict[str, str] = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "H1": "60m",
    "H4": "60m",
    "D1": "1d",
    "W1": "1d",
}

_POINT_MAP: dict[str, float] = {
    "BTCUSD": 0.01,
    "XAUUSD": 0.01,
    "XAGUSD": 0.01,
    "DXY": 0.01,
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "EURGBP": 0.0001,
    "USDJPY": 0.01,
    "AUDJPY": 0.01,
    "NZDJPY": 0.01,
    "AUDNZD": 0.0001,
    "EURJPY": 0.01,
    "GBPJPY": 0.01,
    "NAS100": 0.1,
    "USOIL": 0.01,
    "DOGUSD": 0.0001,
    "TRUMPUSD": 0.0001,
    "AAPL": 0.01,
    "NVIDIA": 0.01,
}

_SPREAD_POINTS_MAP: dict[str, float] = {
    "BTCUSD": 25.0,
    "XAUUSD": 25.0,
    "XAGUSD": 30.0,
    "DXY": 5.0,
    "EURUSD": 10.0,
    "GBPUSD": 12.0,
    "EURGBP": 10.0,
    "USDJPY": 12.0,
    "AUDJPY": 14.0,
    "NZDJPY": 14.0,
    "AUDNZD": 12.0,
    "EURJPY": 14.0,
    "GBPJPY": 16.0,
    "NAS100": 20.0,
    "USOIL": 15.0,
    "DOGUSD": 55.0,
    "TRUMPUSD": 85.0,
    "AAPL": 8.0,
    "NVIDIA": 10.0,
}

_TWELVE_DATA_SYMBOL_MAP: dict[str, str] = {
    "BTCUSD": "BTC/USD",
    "XAUUSD": "XAU/USD",
    "XAGUSD": "XAG/USD",
    "DXY": "DXY",
    "EURUSD": "EUR/USD",
    "GBPUSD": "GBP/USD",
    "EURGBP": "EUR/GBP",
    "USDJPY": "USD/JPY",
    "AUDJPY": "AUD/JPY",
    "NZDJPY": "NZD/JPY",
    "AUDNZD": "AUD/NZD",
    "EURJPY": "EUR/JPY",
    "GBPJPY": "GBP/JPY",
    "DOGUSD": "DOGE/USD",
    "TRUMPUSD": "TRUMP/USD",
    "AAPL": "AAPL",
    "NVIDIA": "NVDA",
}

_TWELVE_INTERVAL_MAP: dict[str, str] = {
    "M1": "1min",
    "M5": "5min",
    "M15": "15min",
    "H1": "1h",
    "H4": "4h",
    "D1": "1day",
    "W1": "1week",
}


def _normalize_symbol_key(value: str) -> str:
    normalized = normalize_symbol_key(value)
    if normalized.startswith(("DXY", "USDX", "USDINDEX", "USDOLLAR")):
        return "DXY"
    return normalized


def _yahoo_symbol_for(symbol_key: str) -> str | None:
    if symbol_key in _SYMBOL_MAP:
        return _SYMBOL_MAP[symbol_key]
    asset_class = symbol_asset_class(symbol_key)
    if asset_class == "forex" and len(symbol_key) == 6:
        return f"{symbol_key}=X"
    if asset_class == "crypto" and symbol_key.endswith("USD") and len(symbol_key) > 3:
        return f"{symbol_key[:-3]}-USD"
    if asset_class == "equity":
        return "NVDA" if symbol_key == "NVIDIA" else symbol_key
    return None


def _twelve_symbol_for(symbol_key: str) -> str | None:
    if symbol_key in _TWELVE_DATA_SYMBOL_MAP:
        return _TWELVE_DATA_SYMBOL_MAP[symbol_key]
    asset_class = symbol_asset_class(symbol_key)
    if asset_class == "forex" and len(symbol_key) == 6:
        return f"{symbol_key[:3]}/{symbol_key[3:6]}"
    if asset_class == "crypto" and symbol_key.endswith("USD") and len(symbol_key) > 3:
        return f"{symbol_key[:-3]}/USD"
    if asset_class == "equity":
        return "NVDA" if symbol_key == "NVIDIA" else symbol_key
    return None


def _point_for_symbol(symbol_key: str) -> float:
    if symbol_key in _POINT_MAP:
        return float(_POINT_MAP[symbol_key])
    asset_class = symbol_asset_class(symbol_key)
    if asset_class == "forex":
        return 0.01 if symbol_key.endswith("JPY") else 0.0001
    if asset_class == "index":
        return 0.1
    return 0.01


def _spread_points_for_symbol(symbol_key: str) -> float:
    if symbol_key in _SPREAD_POINTS_MAP:
        return float(_SPREAD_POINTS_MAP[symbol_key])
    asset_class = symbol_asset_class(symbol_key)
    if asset_class == "forex":
        return 12.0
    if asset_class == "crypto":
        return 120.0
    if asset_class == "equity":
        return 10.0
    if asset_class == "index":
        return 22.0
    if asset_class == "commodity":
        return 28.0
    return 10.0


def _required_range(timeframe: str, count: int) -> str:
    minutes = _TIMEFRAME_MINUTES.get(str(timeframe).upper(), 5)
    required_days = max(1.0, (max(1, int(count)) * minutes * 1.4) / (60.0 * 24.0))
    if required_days <= 1.0:
        return "1d"
    if required_days <= 5.0:
        return "5d"
    if required_days <= 7.0:
        return "7d"
    if required_days <= 30.0:
        return "1mo"
    if required_days <= 90.0:
        return "3mo"
    if required_days <= 180.0:
        return "6mo"
    return "1y"


@dataclass
class YahooMarketDataFallback:
    timeout_seconds: float = 5.0
    user_agent: str = "Mozilla/5.0 (ApexBot)"

    @property
    def source_name(self) -> str:
        return "yahoo_finance"

    def supports(self, symbol: str, timeframe: str) -> bool:
        symbol_key = _normalize_symbol_key(symbol)
        return _yahoo_symbol_for(symbol_key) is not None and str(timeframe).upper() in _INTERVAL_MAP

    def fetch_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        symbol_key = _normalize_symbol_key(symbol)
        if not self.supports(symbol_key, timeframe):
            raise RuntimeError(f"external_market_data_unsupported:{symbol_key}:{timeframe}")
        yahoo_symbol = _yahoo_symbol_for(symbol_key)
        if not yahoo_symbol:
            raise RuntimeError(f"external_market_data_unsupported:{symbol_key}:{timeframe}")
        tf = str(timeframe).upper()
        interval = _INTERVAL_MAP[tf]
        chart = self._chart(yahoo_symbol, interval=interval, range_value=_required_range(tf, count))
        frame = self._frame_from_chart(chart)
        if tf == "H4":
            frame = self._resample_h4(frame)
        elif tf == "D1":
            frame = self._resample_d1(frame)
        elif tf == "W1":
            frame = self._resample_w1(frame)
        if frame.empty:
            raise RuntimeError(f"external_market_data_empty:{symbol_key}:{tf}")
        return frame.tail(max(1, int(count))).reset_index(drop=True)

    def fetch_rates_with_range(
        self,
        symbol: str,
        timeframe: str,
        *,
        range_value: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        symbol_key = _normalize_symbol_key(symbol)
        if not self.supports(symbol_key, timeframe):
            raise RuntimeError(f"external_market_data_unsupported:{symbol_key}:{timeframe}")
        yahoo_symbol = _yahoo_symbol_for(symbol_key)
        if not yahoo_symbol:
            raise RuntimeError(f"external_market_data_unsupported:{symbol_key}:{timeframe}")
        tf = str(timeframe).upper()
        interval = _INTERVAL_MAP[tf]
        chart = self._chart(yahoo_symbol, interval=interval, range_value=str(range_value))
        frame = self._frame_from_chart(chart)
        if tf == "H4":
            frame = self._resample_h4(frame)
        elif tf == "D1":
            frame = self._resample_d1(frame)
        elif tf == "W1":
            frame = self._resample_w1(frame)
        if frame.empty:
            raise RuntimeError(f"external_market_data_empty:{symbol_key}:{tf}")
        if limit is not None:
            return frame.tail(max(1, int(limit))).reset_index(drop=True)
        return frame.reset_index(drop=True)

    def fetch_tick(self, symbol: str) -> dict[str, Any]:
        symbol_key = _normalize_symbol_key(symbol)
        yahoo_symbol = _yahoo_symbol_for(symbol_key)
        if not yahoo_symbol:
            raise RuntimeError(f"external_tick_unsupported:{symbol_key}")
        chart = self._chart(yahoo_symbol, interval="1m", range_value="1d")
        frame = self._frame_from_chart(chart)
        if frame.empty:
            raise RuntimeError(f"external_tick_empty:{symbol_key}")
        last_row = frame.iloc[-1]
        price = float(last_row["close"])
        point = _point_for_symbol(symbol_key)
        half_spread = max(point, (_spread_points_for_symbol(symbol_key) * point) * 0.5)
        bid = max(point, price - half_spread)
        ask = price + half_spread
        return {
            "bid": float(round(bid, 6)),
            "ask": float(round(ask, 6)),
            "time": int(pd.Timestamp(last_row["time"]).timestamp()),
            "source": "yahoo_finance",
        }

    def _chart(self, yahoo_symbol: str, *, interval: str, range_value: str) -> dict[str, Any]:
        query = urlencode({"interval": interval, "range": range_value})
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}?{query}"
        request = Request(url, headers={"User-Agent": self.user_agent})
        with urlopen(request, timeout=max(1.0, float(self.timeout_seconds))) as response:  # nosec B310
            payload = json.load(response)
        result = (((payload or {}).get("chart") or {}).get("result") or [None])[0]
        if not isinstance(result, dict):
            error_payload = ((payload or {}).get("chart") or {}).get("error")
            raise RuntimeError(f"external_market_data_invalid:{error_payload or 'missing_result'}")
        return result

    @staticmethod
    def _frame_from_chart(chart: dict[str, Any]) -> pd.DataFrame:
        timestamps = list(chart.get("timestamp") or [])
        quote_payload = (((chart.get("indicators") or {}).get("quote") or [None])[0] or {})
        if not timestamps or not isinstance(quote_payload, dict):
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"])
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(timestamps, unit="s", utc=True),
                "open": quote_payload.get("open", []),
                "high": quote_payload.get("high", []),
                "low": quote_payload.get("low", []),
                "close": quote_payload.get("close", []),
                "volume": quote_payload.get("volume", []),
            }
        )
        for column in ("open", "high", "low", "close", "volume"):
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame = frame.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
        if frame.empty:
            return pd.DataFrame(columns=["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"])
        frame["tick_volume"] = frame["volume"].fillna(0.0)
        frame["spread"] = 0.0
        frame["real_volume"] = frame["volume"].fillna(0.0)
        return frame[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]]

    @staticmethod
    def _resample_h4(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        resampled = (
            frame.set_index("time")
            .resample("4h", label="right", closed="right")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "tick_volume": "sum",
                    "spread": "mean",
                    "real_volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        return resampled

    @staticmethod
    def _resample_d1(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        resampled = (
            frame.set_index("time")
            .resample("1d", label="right", closed="right")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "tick_volume": "sum",
                    "spread": "mean",
                    "real_volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        return resampled

    @staticmethod
    def _resample_w1(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty:
            return frame
        source = YahooMarketDataFallback._resample_d1(frame)
        if source.empty:
            return source
        resampled = (
            source.set_index("time")
            .resample("1w", label="right", closed="right")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "tick_volume": "sum",
                    "spread": "last",
                    "real_volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        return resampled


@dataclass
class TwelveDataMarketDataFallback:
    api_key: str
    timeout_seconds: float = 5.0
    user_agent: str = "Mozilla/5.0 (ApexBot)"
    base_url: str = "https://api.twelvedata.com"
    cooldown_seconds: float = 60.0

    def __post_init__(self) -> None:
        self.last_error = ""
        self.last_error_code = ""
        self.last_http_status: int | None = None
        self.last_endpoint = ""
        self.last_request_at = ""
        self.last_success_at = ""
        self.last_rate_limited_at = ""
        self.rate_limit_cooldown_until = 0.0
        self.rate_limit_blocks = 0

    def supports(self, symbol: str, timeframe: str) -> bool:
        symbol_key = _normalize_symbol_key(symbol)
        return bool(self.api_key) and _twelve_symbol_for(symbol_key) is not None and str(timeframe).upper() in _TWELVE_INTERVAL_MAP

    @property
    def source_name(self) -> str:
        return "twelve_data"

    def diagnostics(self) -> dict[str, Any]:
        cooldown_remaining = max(0.0, float(self.rate_limit_cooldown_until) - time.monotonic())
        return {
            "source": self.source_name,
            "available": bool(self.api_key),
            "last_error": str(self.last_error or ""),
            "last_error_code": str(self.last_error_code or ""),
            "last_http_status": int(self.last_http_status) if self.last_http_status is not None else None,
            "last_endpoint": str(self.last_endpoint or ""),
            "last_request_at": str(self.last_request_at or ""),
            "last_success_at": str(self.last_success_at or ""),
            "last_rate_limited_at": str(self.last_rate_limited_at or ""),
            "rate_limit_cooldown_active": cooldown_remaining > 0.0,
            "rate_limit_cooldown_remaining_seconds": round(cooldown_remaining, 3),
            "rate_limit_blocks": int(self.rate_limit_blocks),
        }

    def fetch_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        symbol_key = _normalize_symbol_key(symbol)
        tf = str(timeframe).upper()
        if not self.supports(symbol_key, tf):
            raise RuntimeError(f"twelvedata_unsupported:{symbol_key}:{tf}")
        resolved_symbol = _twelve_symbol_for(symbol_key)
        if not resolved_symbol:
            raise RuntimeError(f"twelvedata_unsupported:{symbol_key}:{tf}")
        payload = self._request(
            "time_series",
            {
                "symbol": resolved_symbol,
                "interval": _TWELVE_INTERVAL_MAP[tf],
                "outputsize": str(max(3, int(count))),
                "timezone": UTC,
                "order": "ASC",
            },
        )
        values = list(payload.get("values") or [])
        if not values:
            raise RuntimeError(f"twelvedata_empty:{symbol_key}:{tf}")
        frame = pd.DataFrame(values)
        for column in ("open", "high", "low", "close", "volume"):
            if column in frame.columns:
                frame[column] = pd.to_numeric(frame[column], errors="coerce")
        frame["time"] = pd.to_datetime(frame["datetime"], utc=True, errors="coerce")
        frame = frame.dropna(subset=["time", "open", "high", "low", "close"]).reset_index(drop=True)
        if frame.empty:
            raise RuntimeError(f"twelvedata_empty:{symbol_key}:{tf}")
        if "volume" not in frame.columns:
            frame["volume"] = 0.0
        frame["tick_volume"] = pd.to_numeric(frame["volume"], errors="coerce").fillna(0.0)
        frame["spread"] = 0.0
        frame["real_volume"] = frame["tick_volume"]
        return frame[["time", "open", "high", "low", "close", "tick_volume", "spread", "real_volume"]].tail(max(1, int(count))).reset_index(drop=True)

    def fetch_tick(self, symbol: str) -> dict[str, Any]:
        symbol_key = _normalize_symbol_key(symbol)
        resolved_symbol = _twelve_symbol_for(symbol_key)
        if not resolved_symbol:
            raise RuntimeError(f"twelvedata_tick_unsupported:{symbol_key}")
        quote = self._request("quote", {"symbol": resolved_symbol})
        close_value = quote.get("close") or quote.get("price")
        if close_value in {None, ""}:
            price_payload = self._request("price", {"symbol": resolved_symbol})
            close_value = price_payload.get("price")
        price = float(close_value)
        point = _point_for_symbol(symbol_key)
        half_spread = max(point, (_spread_points_for_symbol(symbol_key) * point) * 0.5)
        bid = max(point, price - half_spread)
        ask = price + half_spread
        timestamp = int(quote.get("last_quote_at") or quote.get("timestamp") or 0)
        return {
            "bid": float(round(bid, 6)),
            "ask": float(round(ask, 6)),
            "time": int(timestamp or pd.Timestamp.utcnow().timestamp()),
            "source": self.source_name,
        }

    def _request(self, endpoint: str, params: dict[str, Any]) -> dict[str, Any]:
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        self.last_endpoint = str(endpoint)
        self.last_request_at = now_iso
        if time.monotonic() < float(self.rate_limit_cooldown_until):
            self.rate_limit_blocks += 1
            remaining = max(0.0, float(self.rate_limit_cooldown_until) - time.monotonic())
            self.last_error = f"twelvedata_rate_limit_cooldown:{remaining:.1f}s"
            self.last_error_code = "rate_limit_cooldown"
            raise RuntimeError(self.last_error)
        query = urlencode({**params, "apikey": self.api_key})
        url = f"{self.base_url.rstrip('/')}/{endpoint}?{query}"
        request = Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(request, timeout=max(1.0, float(self.timeout_seconds))) as response:  # nosec B310
                self.last_http_status = int(getattr(response, "status", 200) or 200)
                payload = json.load(response)
        except HTTPError as exc:
            self.last_http_status = int(getattr(exc, "code", 0) or 0)
            self.last_error_code = f"http_{self.last_http_status}"
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="ignore")
            except Exception:
                body = ""
            if self.last_http_status == 429 or "rate limit" in body.lower() or "credits" in body.lower():
                self.last_rate_limited_at = now_iso
                self.rate_limit_cooldown_until = time.monotonic() + max(5.0, float(self.cooldown_seconds))
                self.last_error = f"twelvedata_rate_limited:{self.last_http_status or 'unknown'}"
            else:
                self.last_error = f"twelvedata_http_error:{self.last_http_status or 'unknown'}"
            raise RuntimeError(self.last_error) from exc
        except URLError as exc:
            self.last_http_status = None
            self.last_error_code = "network_error"
            self.last_error = f"twelvedata_network_error:{exc.reason}"
            raise RuntimeError(self.last_error) from exc
        if not isinstance(payload, dict):
            self.last_error = f"twelvedata_invalid:{endpoint}"
            self.last_error_code = "invalid_payload"
            raise RuntimeError(f"twelvedata_invalid:{endpoint}")
        if str(payload.get("status", "")).lower() == "error" or payload.get("code"):
            code = str(payload.get("code", "unknown") or "unknown")
            message = str(payload.get("message", "request_failed") or "request_failed")
            if "rate" in message.lower() or "credits" in message.lower():
                self.last_rate_limited_at = now_iso
                self.rate_limit_cooldown_until = time.monotonic() + max(5.0, float(self.cooldown_seconds))
            self.last_error = f"twelvedata_error:{code}:{message}"
            self.last_error_code = code
            raise RuntimeError(self.last_error)
        self.last_success_at = now_iso
        self.last_error = ""
        self.last_error_code = ""
        return payload


@dataclass
class MultiSourceMarketDataFallback:
    timeout_seconds: float = 5.0
    user_agent: str = "Mozilla/5.0 (ApexBot)"
    twelve_data_api_key: str = ""

    def __post_init__(self) -> None:
        api_key = str(self.twelve_data_api_key or os.getenv("APEX_TWELVEDATA_API_KEY") or os.getenv("TWELVEDATA_API_KEY") or "").strip()
        self._providers: list[Any] = []
        if api_key:
            self._providers.append(
                TwelveDataMarketDataFallback(
                    api_key=api_key,
                    timeout_seconds=self.timeout_seconds,
                    user_agent=self.user_agent,
                )
            )
        self._providers.append(
            YahooMarketDataFallback(
                timeout_seconds=self.timeout_seconds,
                user_agent=self.user_agent,
            )
        )
        self.last_source = "none"
        self.last_attempted_providers: list[str] = []
        self.last_errors: list[dict[str, Any]] = []

    def supports(self, symbol: str, timeframe: str) -> bool:
        return any(getattr(provider, "supports")(symbol, timeframe) for provider in self._providers)

    def diagnostics(self) -> dict[str, Any]:
        providers: list[dict[str, Any]] = []
        for provider in self._providers:
            name = str(getattr(provider, "source_name", "unknown") or "unknown")
            if hasattr(provider, "diagnostics"):
                try:
                    providers.append(dict(provider.diagnostics()))
                    continue
                except Exception as exc:
                    providers.append({"source": name, "available": False, "last_error": f"diagnostics_failed:{exc}"})
                    continue
            providers.append({"source": name, "available": True})
        return {
            "last_source": str(self.last_source or "none"),
            "last_attempted_providers": list(self.last_attempted_providers),
            "last_errors": list(self.last_errors),
            "providers": providers,
        }

    def fetch_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        errors: list[str] = []
        attempts: list[str] = []
        error_details: list[dict[str, Any]] = []
        for provider in self._providers:
            if not provider.supports(symbol, timeframe):
                continue
            provider_name = str(getattr(provider, "source_name", "unknown") or "unknown")
            attempts.append(provider_name)
            try:
                frame = provider.fetch_rates(symbol, timeframe, count)
                self.last_source = provider_name
                self.last_attempted_providers = attempts
                self.last_errors = error_details
                return frame
            except Exception as exc:
                errors.append(str(exc))
                error_details.append({"source": provider_name, "error": str(exc)})
        self.last_attempted_providers = attempts
        self.last_errors = error_details
        raise RuntimeError("external_market_data_failed:" + "|".join(errors or ["no_provider"]))

    def fetch_tick(self, symbol: str) -> dict[str, Any]:
        errors: list[str] = []
        attempts: list[str] = []
        error_details: list[dict[str, Any]] = []
        for provider in self._providers:
            supports = True
            if hasattr(provider, "supports"):
                supports = provider.supports(symbol, "M1")
            if not supports and getattr(provider, "source_name", "") != "yahoo_finance":
                continue
            provider_name = str(getattr(provider, "source_name", "unknown") or "unknown")
            attempts.append(provider_name)
            try:
                tick = provider.fetch_tick(symbol)
                self.last_source = str(getattr(provider, "source_name", tick.get("source", "unknown")))
                self.last_attempted_providers = attempts
                self.last_errors = error_details
                return tick
            except Exception as exc:
                errors.append(str(exc))
                error_details.append({"source": provider_name, "error": str(exc)})
        self.last_attempted_providers = attempts
        self.last_errors = error_details
        raise RuntimeError("external_tick_failed:" + "|".join(errors or ["no_provider"]))
