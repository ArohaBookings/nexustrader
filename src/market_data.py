from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import pandas as pd

from src.mt5_client import MT5Client
from src.utils import ensure_directory


@dataclass
class MarketDataService:
    mt5_client: MT5Client
    cache_dir: Path

    def __post_init__(self) -> None:
        ensure_directory(self.cache_dir)
        self._status: dict[str, dict[str, Any]] = {}

    def fetch(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        symbol_key = self._normalize_symbol_key(symbol)
        tf = str(timeframe).upper()
        cache_age = self._cache_age_seconds(symbol, tf)
        freshness = self._freshness_seconds(tf)
        cache_meta = self._load_cache_meta(symbol, tf)
        cached_frame = None
        if (
            cache_age is not None
            and cache_age <= freshness
            and self.mt5_client.uses_external_market_data()
            and self._cache_meta_is_live_compatible(cache_meta)
        ):
            cached_frame = self.load_cached(symbol, tf)
            cached_bar_age = self._frame_tail_age_seconds(cached_frame)
            cache_bar_fresh = cached_bar_age is not None and cached_bar_age <= self._max_live_bar_age_seconds(tf)
            if cached_frame is not None and not cached_frame.empty and cache_bar_fresh:
                self._update_status(
                    symbol=symbol_key,
                    timeframe=tf,
                    mode="cache_recent",
                    source="parquet_cache",
                    ready=True,
                    age_seconds=cached_bar_age,
                    error="",
                )
                return cached_frame
        try:
            frame = self._sanitize_frame(tf, self.mt5_client.get_rates(symbol, tf, count).copy())
            provider_status = self.mt5_client.market_data_status()
            self._cache_frame(
                symbol,
                tf,
                frame,
                mode=str(provider_status.get("mode", "unknown")),
                source=str(provider_status.get("source", "unknown")),
            )
            self._update_status(
                symbol=symbol_key,
                timeframe=tf,
                mode=str(provider_status.get("mode", "unknown")),
                source=str(provider_status.get("source", "unknown")),
                ready=bool(provider_status.get("ready", True)),
                age_seconds=0.0,
                error=str(provider_status.get("error", "")),
            )
            return frame
        except Exception as exc:
            cached_frame = cached_frame if cached_frame is not None else self.load_cached(symbol, tf)
            if cached_frame is not None and not cached_frame.empty and self._cache_meta_is_live_compatible(cache_meta):
                fallback_age = self._frame_tail_age_seconds(cached_frame)
                cache_write_age = self._cache_age_seconds(symbol, tf)
                cache_write_recent = cache_write_age is not None and cache_write_age <= self._freshness_seconds(tf)
                ready = bool(
                    (fallback_age is not None and fallback_age <= (self._max_live_bar_age_seconds(tf) * 2.0))
                    or cache_write_recent
                )
                self._update_status(
                    symbol=symbol_key,
                    timeframe=tf,
                    mode="cache_fallback",
                    source="parquet_cache",
                    ready=bool(ready),
                    age_seconds=fallback_age,
                    error=str(exc),
                )
                if ready:
                    return cached_frame
            self._update_status(
                symbol=symbol_key,
                timeframe=tf,
                mode="unavailable",
                source="none",
                ready=False,
                age_seconds=None,
                error=str(exc),
            )
            raise

    def latest_multi_timeframe(self, symbol: str, counts: dict[str, int]) -> dict[str, pd.DataFrame]:
        return {timeframe: self.fetch(symbol, timeframe, count) for timeframe, count in counts.items()}

    def load_cached(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        parquet_path = self._cache_path(symbol, timeframe)
        if not parquet_path.exists():
            return None
        return self._sanitize_frame(timeframe, pd.read_parquet(parquet_path))

    def merge_external_context(self, frame: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
        enriched = frame.copy()
        for key, value in context.items():
            enriched[key] = value
        return enriched

    def status_for_symbol(self, symbol: str) -> dict[str, Any]:
        symbol_key = self._normalize_symbol_key(symbol)
        frame_status = dict(self._status.get(symbol_key, {}))
        timeframes = frame_status.get("timeframes", {})
        overall_mode = "unavailable"
        overall_source = "none"
        ready = False
        age_seconds = None
        errors: list[str] = []
        if isinstance(timeframes, dict) and timeframes:
            preferred = ["M5", "M15", "M1", "H1", "H4", "D1", "W1"]
            ordered = [timeframes[key] for key in preferred if key in timeframes] + [value for key, value in timeframes.items() if key not in preferred]
            primary = next((item for item in ordered if isinstance(item, dict)), {})
            overall_mode = str(primary.get("mode", overall_mode))
            overall_source = str(primary.get("source", overall_source))
            ready = any(bool(item.get("ready")) for item in ordered if isinstance(item, dict))
            ages = [float(item["age_seconds"]) for item in ordered if isinstance(item, dict) and item.get("age_seconds") is not None]
            age_seconds = min(ages) if ages else None
            errors = [str(item.get("error", "")) for item in ordered if isinstance(item, dict) and str(item.get("error", "")).strip()]
            latencies = [int(item["latency_ms"]) for item in ordered if isinstance(item, dict) and item.get("latency_ms") is not None]
        else:
            latencies = []
        provider_diagnostics = self.mt5_client.market_data_status().get("provider_diagnostics", {})
        provider_entries = list(provider_diagnostics.get("providers", [])) if isinstance(provider_diagnostics, dict) else []
        available_sources = {
            str(item.get("source") or "").strip()
            for item in provider_entries
            if isinstance(item, dict) and bool(item.get("available"))
        }
        runtime_market_data_consensus_state = "UNAVAILABLE"
        if bool(ready):
            normalized_source = str(overall_source or "").strip().lower()
            if normalized_source.startswith("mt5"):
                runtime_market_data_consensus_state = "MT5_PRIMARY"
            elif len(available_sources) >= 2:
                runtime_market_data_consensus_state = "EXTERNAL_MULTI_PROVIDER"
            elif len(available_sources) == 1:
                runtime_market_data_consensus_state = "EXTERNAL_SINGLE_PROVIDER"
            else:
                runtime_market_data_consensus_state = "EXTERNAL_SINGLE_PROVIDER"
            if provider_diagnostics and len(available_sources) >= 2:
                last_source = str(provider_diagnostics.get("last_source") or "").strip().lower()
                attempted = {
                    str(item).strip().lower()
                    for item in list(provider_diagnostics.get("last_attempted_providers", []))
                    if str(item).strip()
                }
                if last_source and attempted and last_source not in attempted:
                    runtime_market_data_consensus_state = "EXTERNAL_DIVERGENT"
        return {
            "symbol": symbol_key,
            "runtime_market_data_mode": overall_mode,
            "runtime_market_data_source": overall_source,
            "runtime_market_data_consensus_state": runtime_market_data_consensus_state,
            "runtime_market_data_ready": bool(ready),
            "runtime_market_data_age_seconds": age_seconds,
            "runtime_market_data_error": errors[0] if errors and not ready else "",
            "runtime_market_data_latency_ms": min(latencies) if latencies else None,
            "runtime_market_data_provider_diagnostics": provider_diagnostics,
            "timeframes": timeframes,
        }

    def status_snapshot(self) -> dict[str, Any]:
        return {symbol_key: self.status_for_symbol(symbol_key) for symbol_key in self._status.keys()}

    def _cache_frame(self, symbol: str, timeframe: str, frame: pd.DataFrame, *, mode: str, source: str) -> None:
        frame = self._sanitize_frame(timeframe, frame)
        if frame.empty:
            return
        cache_path = self._cache_path(symbol, timeframe)
        frame.to_parquet(cache_path, index=False)
        meta_payload = {
            "mode": str(mode),
            "source": str(source),
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
            "rows": int(len(frame)),
            "last_close": float(frame["close"].iloc[-1]) if "close" in frame.columns and not frame.empty else None,
        }
        self._cache_meta_path(symbol, timeframe).write_text(json.dumps(meta_payload, sort_keys=True), encoding="utf-8")

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.cache_dir / f"{safe_symbol}_{timeframe}.parquet"

    def _cache_meta_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return self.cache_dir / f"{safe_symbol}_{timeframe}.meta.json"

    def _load_cache_meta(self, symbol: str, timeframe: str) -> dict[str, Any]:
        path = self._cache_meta_path(symbol, timeframe)
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _cache_age_seconds(self, symbol: str, timeframe: str) -> float | None:
        path = self._cache_path(symbol, timeframe)
        if not path.exists():
            return None
        try:
            modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return None
        return max(0.0, (datetime.now(tz=timezone.utc) - modified).total_seconds())

    @staticmethod
    def _freshness_seconds(timeframe: str) -> float:
        tf = str(timeframe).upper()
        mapping = {
            "M1": 30.0,
            "M5": 90.0,
            "M15": 240.0,
            "H1": 900.0,
            "H4": 3600.0,
            "D1": 21600.0,
            "W1": 86400.0,
        }
        return float(mapping.get(tf, 120.0))

    @classmethod
    def _max_live_bar_age_seconds(cls, timeframe: str) -> float:
        return cls._freshness_seconds(timeframe) * 1.5

    @staticmethod
    def _normalize_symbol_key(value: str) -> str:
        normalized = "".join(char for char in str(value).upper() if char.isalnum())
        if normalized.startswith("XAUUSD") or normalized.startswith("GOLD"):
            return "XAUUSD"
        if normalized.startswith("BTCUSD") or normalized.startswith("BTCUSDT") or normalized.startswith("XBTUSD"):
            return "BTCUSD"
        if normalized.startswith(("DXY", "USDX", "USDINDEX", "USDOLLAR")):
            return "DXY"
        if normalized.startswith(("NAS100", "US100", "NASDAQ", "USTEC", "NAS", "NQ")):
            return "NAS100"
        if normalized.startswith(("USOIL", "XTIUSD", "OILUSD", "WTI", "CL", "OIL", "USO")):
            return "USOIL"
        for core in ("EURUSD", "GBPUSD", "EURGBP", "USDJPY", "AUDJPY", "NZDJPY", "AUDNZD", "EURJPY", "GBPJPY"):
            if normalized.startswith(core):
                return core
        return normalized

    def _update_status(
        self,
        *,
        symbol: str,
        timeframe: str,
        mode: str,
        source: str,
        ready: bool,
        age_seconds: float | None,
        error: str,
    ) -> None:
        bucket = self._status.setdefault(symbol, {"timeframes": {}, "updated_at": None})
        bucket["timeframes"][str(timeframe).upper()] = {
            "mode": str(mode),
            "source": str(source),
            "ready": bool(ready),
            "age_seconds": float(age_seconds) if age_seconds is not None else None,
            "error": str(error or ""),
            "latency_ms": self.mt5_client.market_data_status().get("latency_ms"),
            "provider_diagnostics": self.mt5_client.market_data_status().get("provider_diagnostics", {}),
            "updated_at": datetime.now(tz=timezone.utc).isoformat(),
        }
        bucket["updated_at"] = datetime.now(tz=timezone.utc).isoformat()

    @staticmethod
    def _cache_meta_is_live_compatible(meta: dict[str, Any]) -> bool:
        if not isinstance(meta, dict) or not meta:
            return False
        mode = str(meta.get("mode", "")).strip().lower()
        source = str(meta.get("source", "")).strip().lower()
        if mode in {"synthetic_fallback", "unavailable", "external_unavailable"}:
            return False
        return source in {"mt5_python", "yahoo_finance", "twelve_data", "external_provider", "parquet_cache"}

    def _sanitize_frame(self, timeframe: str, frame: pd.DataFrame | None) -> pd.DataFrame:
        if frame is None or frame.empty:
            return pd.DataFrame() if frame is None else frame
        sanitized = frame.copy()
        if "time" in sanitized.columns:
            sanitized["time"] = pd.to_datetime(sanitized["time"], utc=True, errors="coerce")
            sanitized = (
                sanitized.dropna(subset=["time"])
                .sort_values("time")
                .drop_duplicates(subset=["time"], keep="last")
                .reset_index(drop=True)
            )
            sanitized = self._drop_partial_tail(timeframe, sanitized)
        return sanitized.reset_index(drop=True)

    @staticmethod
    def _frame_tail_age_seconds(frame: pd.DataFrame | None) -> float | None:
        if frame is None or frame.empty or "time" not in frame.columns:
            return None
        try:
            tail_time = pd.Timestamp(frame.iloc[-1]["time"])
        except Exception:
            return None
        if pd.isna(tail_time):
            return None
        if tail_time.tzinfo is None:
            tail_time = tail_time.tz_localize("UTC")
        else:
            tail_time = tail_time.tz_convert("UTC")
        age_seconds = (pd.Timestamp.now(tz=timezone.utc) - tail_time).total_seconds()
        if age_seconds < -1.0:
            return None
        return max(0.0, age_seconds)

    def _drop_partial_tail(self, timeframe: str, frame: pd.DataFrame) -> pd.DataFrame:
        tf = str(timeframe).upper()
        sanitized = frame.copy()
        while len(sanitized) > 1:
            tail = sanitized.iloc[-1]
            tail_time = pd.Timestamp(tail["time"])
            if tail_time.tzinfo is None:
                tail_time = tail_time.tz_localize("UTC")
            else:
                tail_time = tail_time.tz_convert("UTC")
            if not self._tail_bar_is_partial(tf, tail_time, tail):
                break
            sanitized = sanitized.iloc[:-1].reset_index(drop=True)
        return sanitized

    @staticmethod
    def _tail_bar_is_partial(timeframe: str, tail_time: pd.Timestamp, tail_row: pd.Series) -> bool:
        tf = str(timeframe).upper()
        if not MarketDataService._time_is_aligned(tf, tail_time):
            return True
        delta = MarketDataService._timeframe_delta(tf)
        if delta is None:
            return False
        if not MarketDataService._is_zero_range_bar(tail_row):
            return False
        now = pd.Timestamp.now(tz=timezone.utc)
        return tail_time + delta > now

    @staticmethod
    def _is_zero_range_bar(row: pd.Series) -> bool:
        values: list[float] = []
        for key in ("open", "high", "low", "close"):
            value = row.get(key)
            if value is None or pd.isna(value):
                return False
            values.append(float(value))
        tolerance = max(1e-9, abs(values[-1]) * 1e-8)
        return (max(values) - min(values)) <= tolerance

    @staticmethod
    def _time_is_aligned(timeframe: str, value: pd.Timestamp) -> bool:
        freq_map = {
            "M1": "1min",
            "M5": "5min",
            "M15": "15min",
            "H1": "1h",
            "H4": "4h",
            "D1": "1d",
        }
        freq = freq_map.get(str(timeframe).upper())
        if not freq:
            return True
        return value == value.floor(freq)

    @staticmethod
    def _timeframe_delta(timeframe: str) -> pd.Timedelta | None:
        mapping = {
            "M1": pd.Timedelta(minutes=1),
            "M5": pd.Timedelta(minutes=5),
            "M15": pd.Timedelta(minutes=15),
            "H1": pd.Timedelta(hours=1),
            "H4": pd.Timedelta(hours=4),
            "D1": pd.Timedelta(days=1),
            "W1": pd.Timedelta(days=7),
        }
        return mapping.get(str(timeframe).upper())
