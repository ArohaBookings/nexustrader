from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable
import os
import sqlite3
import sys
import time

try:
    import pandas as pd
except ImportError:  # pragma: no cover - optional for limited test environments
    pd = None  # type: ignore

from src.external_market_data import MultiSourceMarketDataFallback
from src.symbol_universe import normalize_symbol_key, symbol_asset_class, symbol_family_defaults
from src.utils import clamp


_MT5_MODULE: Any = None
_MT5_IMPORT_ATTEMPTED = False


def _load_mt5_module() -> Any:
    global _MT5_MODULE, _MT5_IMPORT_ATTEMPTED
    if _MT5_IMPORT_ATTEMPTED:
        return _MT5_MODULE
    _MT5_IMPORT_ATTEMPTED = True
    try:
        import MetaTrader5 as module  # type: ignore
    except ImportError:  # pragma: no cover - optional runtime dependency
        module = None
    _MT5_MODULE = module
    return _MT5_MODULE


TIMEFRAME_ATTR_MAP: dict[str, str] = {
    "M1": "TIMEFRAME_M1",
    "M5": "TIMEFRAME_M5",
    "M15": "TIMEFRAME_M15",
    "H1": "TIMEFRAME_H1",
    "H4": "TIMEFRAME_H4",
    "D1": "TIMEFRAME_D1",
    "W1": "TIMEFRAME_W1",
}

TIMEFRAME_FALLBACKS: dict[str, int] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
}


@dataclass
class MT5Credentials:
    login: int | None = None
    password: str | None = None
    server: str | None = None
    path: str | None = None
    terminal_path: str | None = None

    @classmethod
    def from_env(cls, default_terminal_path: str | None = None) -> "MT5Credentials":
        login_value = os.getenv("MT5_LOGIN")
        legacy_path = os.getenv("MT5_PATH")
        terminal_path = os.getenv("MT5_TERMINAL_PATH") or legacy_path or default_terminal_path
        return cls(
            login=int(login_value) if login_value else None,
            password=os.getenv("MT5_PASSWORD"),
            server=os.getenv("MT5_SERVER"),
            path=legacy_path,
            terminal_path=terminal_path,
        )


@dataclass
class OrderResult:
    accepted: bool
    order_id: str | None
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass
class MT5Client:
    credentials: MT5Credentials
    journal_db: Path
    max_retries: int = 5
    connected: bool = False
    resolved_symbol: str | None = None
    resolved_symbols: dict[str, str] = field(default_factory=dict)
    logger: Any | None = None
    disable_mt5: bool = False
    platform_name: str = field(default_factory=lambda: sys.platform)
    mt5_loader: Callable[[], Any] = field(default_factory=lambda: _load_mt5_module)
    last_init_error: Any = None
    symbol_mapping: dict[str, Any] = field(default_factory=dict)
    external_market_data_enabled: bool = field(
        default_factory=lambda: os.getenv("APEX_EXTERNAL_MARKET_DATA_ENABLED", "1").strip().lower()
        not in {"0", "false", "no", "off"}
    )
    allow_synthetic_fallback: bool = field(
        default_factory=lambda: os.getenv("APEX_ALLOW_SYNTHETIC_MARKET_DATA", "0").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    external_market_data_timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("APEX_EXTERNAL_MARKET_DATA_TIMEOUT_SECONDS", "5"))
    )
    _external_market_data: MultiSourceMarketDataFallback | None = field(default=None, init=False, repr=False)
    _market_data_status: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def connect(self) -> bool:
        if self.disable_mt5:
            self.connected = False
            self._emit("DRY_RUN mode active: skipping MetaTrader5 initialization", level="info")
            return False

        module = self._get_mt5()
        if module is None:
            self.connected = False
            self._emit_mt5_failure("MetaTrader5 Python package is not available in this interpreter.")
            return False

        base_kwargs = self._credential_kwargs()
        attempts = self._initialize_attempts(base_kwargs)

        wait_seconds = 1
        for index in range(self.max_retries):
            for kwargs in attempts:
                initialized = module.initialize(**kwargs)
                if initialized:
                    self.connected = True
                    self.last_init_error = None
                    return True
                self.last_init_error = self._last_error(module)
            if index < self.max_retries - 1:
                time.sleep(wait_seconds)
                wait_seconds = min(wait_seconds * 2, 16)

        self.connected = False
        self._emit_mt5_failure("MetaTrader5 initialize() failed.")
        return False

    def verify_connection(self, symbols: list[str]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "connected": False,
            "resolved_symbols": {},
            "account_summary": None,
            "terminal_info": None,
            "version": None,
            "reasons": [],
        }
        if not self.connect():
            result["reasons"].append("mt5_initialize_failed")
            return result

        result["connected"] = True
        try:
            account_info = self.get_account_info()
            terminal_info = self.get_terminal_info()
            version = self.get_version()
            result["terminal_info"] = terminal_info
            result["version"] = version
            result["account_summary"] = {
                "login": account_info.get("login"),
                "server": account_info.get("server"),
                "leverage": account_info.get("leverage"),
                "balance": account_info.get("balance"),
                "equity": account_info.get("equity"),
            }
            self._emit(f"MT5 ACCOUNT: {result['account_summary']}", level="info")

            resolved_symbols: dict[str, str] = {}
            for symbol in symbols:
                try:
                    resolved = self.resolve_symbol(symbol)
                    resolved_symbols[symbol] = resolved
                except Exception as exc:
                    result["reasons"].append(f"symbol_select_failed:{symbol}:{exc}")
            result["resolved_symbols"] = resolved_symbols
            result["ok"] = len(result["reasons"]) == 0
            if not result["ok"]:
                self._emit(f"MT5 VERIFY WARNING: {result['reasons']}", level="warning")
            return result
        except Exception as exc:
            result["reasons"].append(f"verify_exception:{exc}")
            self._emit(f"MT5 VERIFY ERROR: {exc}", level="warning")
            return result

    def shutdown(self) -> None:
        module = self._get_mt5()
        if module is not None:
            module.shutdown()
        self.connected = False

    def ensure_connection(self) -> None:
        if not self.connected and not self.connect():
            raise RuntimeError("MetaTrader5 initialize failed")

    def resolve_symbol(self, requested_symbol: str) -> str:
        requested_symbol = requested_symbol.upper()
        if requested_symbol in self.resolved_symbols:
            self.resolved_symbol = self.resolved_symbols[requested_symbol]
            return self.resolved_symbols[requested_symbol]

        candidates = self._symbol_candidates(requested_symbol)
        module = self._get_mt5()
        if self.disable_mt5 or module is None:
            resolved = candidates[0]
            self.resolved_symbol = resolved
            self.resolved_symbols[requested_symbol] = resolved
            return resolved

        self.ensure_connection()
        available = [symbol.name for symbol in module.symbols_get() or []]
        symbols = {name.upper(): name for name in available}
        for candidate in candidates:
            resolved = symbols.get(candidate.upper())
            if resolved:
                module.symbol_select(resolved, True)
                self.resolved_symbol = resolved
                self.resolved_symbols[requested_symbol] = resolved
                return resolved

        normalized_target = self._normalize_symbol_key(requested_symbol)
        for name in available:
            normalized_name = self._normalize_symbol_key(name)
            if normalized_name == normalized_target or normalized_name.startswith(normalized_target):
                module.symbol_select(name, True)
                self.resolved_symbol = name
                self.resolved_symbols[requested_symbol] = name
                return name

        raise RuntimeError(f"Unable to resolve configured symbol {requested_symbol} in MT5")

    def resolve_symbols(self, requested_symbols: list[str]) -> dict[str, str]:
        return {symbol.upper(): self.resolve_symbol(symbol) for symbol in requested_symbols}

    def list_symbols(self) -> list[dict[str, Any]]:
        module = self._get_mt5()
        if self.disable_mt5 or module is None:
            return []
        self.ensure_connection()
        try:
            raw_symbols = module.symbols_get() or []
        except Exception:
            raw_symbols = []
        payloads: list[dict[str, Any]] = []
        for row in raw_symbols:
            payload = self._structured_row_to_dict(row)
            if payload:
                payloads.append(payload)
        return payloads

    def discover_symbol_universe(
        self,
        *,
        include_asset_classes: set[str] | None = None,
        max_per_class: dict[str, int] | None = None,
        exclude_symbols: set[str] | None = None,
    ) -> list[dict[str, Any]]:
        allowed = {str(item).strip().lower() for item in (include_asset_classes or set()) if str(item).strip()}
        limits = {str(key).strip().lower(): max(0, int(value)) for key, value in dict(max_per_class or {}).items()}
        excluded = {normalize_symbol_key(item) for item in (exclude_symbols or set()) if str(item).strip()}
        counts: dict[str, int] = {}
        output: list[dict[str, Any]] = []
        seen: set[str] = set()
        for payload in self.list_symbols():
            resolved_symbol = str(payload.get("name") or "").strip()
            if not resolved_symbol:
                continue
            symbol_key = normalize_symbol_key(resolved_symbol)
            if not symbol_key or symbol_key in seen or symbol_key in excluded:
                continue
            asset_class = symbol_asset_class(symbol_key, payload)
            if allowed and asset_class not in allowed:
                continue
            limit = int(limits.get(asset_class, 0) or 0)
            if limit > 0 and int(counts.get(asset_class, 0)) >= limit:
                continue
            counts[asset_class] = int(counts.get(asset_class, 0)) + 1
            seen.add(symbol_key)
            output.append(
                {
                    "symbol": symbol_key,
                    "resolved_symbol": resolved_symbol,
                    "asset_class": asset_class,
                    "description": str(payload.get("description") or ""),
                    "path": str(payload.get("path") or ""),
                    "currency_base": str(payload.get("currency_base") or ""),
                    "currency_profit": str(payload.get("currency_profit") or ""),
                    "visible": bool(payload.get("visible", True)),
                }
            )
        return output

    def get_account_info(self) -> dict[str, Any]:
        module = self._get_mt5()
        if self.disable_mt5 or module is None:
            return {"equity": 1000.0, "balance": 1000.0, "margin_free": 1000.0}
        self.ensure_connection()
        info = module.account_info()
        if info is None:
            raise RuntimeError("Unable to read MT5 account info")
        return info._asdict()

    def get_terminal_info(self) -> dict[str, Any]:
        module = self._get_mt5()
        if self.disable_mt5 or module is None:
            return {"connected": False, "message": "MetaTrader5 unavailable or disabled"}
        self.ensure_connection()
        info = module.terminal_info()
        if info is None:
            return {"connected": False, "message": "Unable to read terminal_info"}
        return info._asdict()

    def get_version(self) -> Any:
        module = self._get_mt5()
        if self.disable_mt5 or module is None:
            return "MetaTrader5 unavailable"
        self.ensure_connection()
        if hasattr(module, "version"):
            return module.version()
        return "unknown"

    def get_symbol_info(self, symbol: str) -> dict[str, Any]:
        module = self._get_mt5()
        if self.disable_mt5 or module is None:
            self._set_market_data_status(
                mode="synthetic_fallback" if self.disable_mt5 else "external_fallback_pending",
                source="defaults",
                ready=bool(self.disable_mt5),
            )
            defaults = dict(symbol_family_defaults(symbol))
            return {
                "point": float(defaults.get("point", 0.01)),
                "trade_tick_size": float(defaults.get("trade_tick_size", defaults.get("point", 0.01))),
                "trade_tick_value": float(defaults.get("trade_tick_value", 0.01)),
                "trade_contract_size": float(defaults.get("trade_contract_size", 1.0)),
                "volume_min": float(defaults.get("volume_min", 0.01)),
                "volume_max": float(defaults.get("volume_max", 10.0)),
                "volume_step": float(defaults.get("volume_step", 0.01)),
                "trade_stops_level": 0,
            }
        self.ensure_connection()
        info = module.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"No symbol info for {symbol}")
        return info._asdict()

    def get_tick(self, symbol: str) -> dict[str, Any]:
        started = time.perf_counter()
        module = self._get_mt5()
        if self.disable_mt5:
            self._set_market_data_status(mode="synthetic_fallback", source="dry_run", ready=True, latency_ms=0)
            return {"bid": 2200.0, "ask": 2200.08, "time": int(time.time())}
        if module is None:
            external_tick = self._external_tick(symbol)
            if external_tick is not None:
                return external_tick
            if self.allow_synthetic_fallback:
                self._set_market_data_status(mode="synthetic_fallback", source="synthetic_tick", ready=False, latency_ms=0)
                return {"bid": 2200.0, "ask": 2200.08, "time": int(time.time())}
            raise RuntimeError(f"live_market_data_unavailable:{symbol}:tick")
        self.ensure_connection()
        tick = module.symbol_info_tick(symbol)
        if tick is None:
            external_tick = self._external_tick(symbol)
            if external_tick is not None:
                return external_tick
            raise RuntimeError(f"No tick for {symbol}")
        latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000.0))
        self._set_market_data_status(mode="mt5_live", source="mt5_python", ready=True, latency_ms=latency_ms)
        return tick._asdict()

    @staticmethod
    def _structured_row_to_dict(row: Any) -> dict[str, Any]:
        if isinstance(row, dict):
            return dict(row)
        if hasattr(row, "_asdict"):
            payload = row._asdict()
            return dict(payload) if isinstance(payload, dict) else {}
        dtype = getattr(row, "dtype", None)
        names = getattr(dtype, "names", None)
        if names:
            output: dict[str, Any] = {}
            for name in names:
                value = row[name]
                if hasattr(value, "item"):
                    try:
                        value = value.item()
                    except Exception:
                        pass
                output[str(name)] = value
            return output
        return {}

    def get_recent_ticks(self, symbol: str, count: int = 96, lookback_seconds: int = 900) -> list[dict[str, Any]]:
        module = self._get_mt5()
        if self.disable_mt5 or module is None:
            return []
        self.ensure_connection()
        window_seconds = max(int(lookback_seconds), max(60, int(count) * 4))
        started_at = datetime.utcnow() - timedelta(seconds=window_seconds)
        try:
            flags = getattr(module, "COPY_TICKS_ALL", 0)
            raw = module.copy_ticks_from(symbol, started_at, max(8, int(count)), flags)
        except Exception:
            raw = None
        if raw is None:
            return []
        ticks: list[dict[str, Any]] = []
        try:
            iterable = list(raw)
        except Exception:
            iterable = []
        for row in iterable[-max(1, int(count)):]:
            payload = self._structured_row_to_dict(row)
            if payload:
                ticks.append(payload)
        return ticks

    def get_market_book(self, symbol: str) -> list[dict[str, Any]]:
        module = self._get_mt5()
        if self.disable_mt5 or module is None:
            return []
        if not hasattr(module, "market_book_get"):
            return []
        self.ensure_connection()
        subscribed = False
        if hasattr(module, "market_book_add"):
            try:
                subscribed = bool(module.market_book_add(symbol))
            except Exception:
                subscribed = False
        try:
            raw_book = module.market_book_get(symbol)
        except Exception:
            raw_book = None
        finally:
            if subscribed and hasattr(module, "market_book_release"):
                try:
                    module.market_book_release(symbol)
                except Exception:
                    pass
        if raw_book is None:
            return []
        levels: list[dict[str, Any]] = []
        try:
            iterable = list(raw_book)
        except Exception:
            iterable = []
        for row in iterable:
            payload = self._structured_row_to_dict(row)
            if payload:
                levels.append(payload)
        return levels

    def get_microstructure_snapshot(self, symbol: str, tick_count: int = 96) -> dict[str, Any]:
        ticks = self.get_recent_ticks(symbol, count=max(24, int(tick_count)))
        if len(ticks) < 4:
            return {
                "ready": False,
                "direction": "neutral",
                "confidence": 0.0,
                "pressure_score": 0.5,
                "cumulative_delta_score": 0.0,
                "depth_imbalance": 0.0,
                "drift_score": 0.0,
                "spread_stability": 0.5,
                "tick_count": int(len(ticks)),
                "book_levels": 0,
                "sweep_velocity": 0.0,
                "absorption_score": 0.0,
                "iceberg_score": 0.0,
                "spread_shock_score": 0.0,
                "quote_pull_stack_score": 0.0,
                "dom_imbalance": 0.0,
            }
        mids: list[float] = []
        spreads: list[float] = []
        buy_moves = 0
        sell_moves = 0
        previous_mid: float | None = None
        for tick in ticks:
            bid = float(tick.get("bid", 0.0) or 0.0)
            ask = float(tick.get("ask", bid) or bid)
            if bid <= 0.0 and ask <= 0.0:
                continue
            if ask <= 0.0:
                ask = bid
            if bid <= 0.0:
                bid = ask
            mid = (bid + ask) / 2.0
            spread = max(0.0, ask - bid)
            mids.append(mid)
            spreads.append(spread)
            if previous_mid is not None:
                if mid > previous_mid:
                    buy_moves += 1
                elif mid < previous_mid:
                    sell_moves += 1
            previous_mid = mid
        if len(mids) < 4:
            return {
                "ready": False,
                "direction": "neutral",
                "confidence": 0.0,
                "pressure_score": 0.5,
                "cumulative_delta_score": 0.0,
                "depth_imbalance": 0.0,
                "drift_score": 0.0,
                "spread_stability": 0.5,
                "tick_count": int(len(mids)),
                "book_levels": 0,
                "sweep_velocity": 0.0,
                "absorption_score": 0.0,
                "iceberg_score": 0.0,
                "spread_shock_score": 0.0,
                "quote_pull_stack_score": 0.0,
                "dom_imbalance": 0.0,
            }
        transitions = max(1, buy_moves + sell_moves)
        cumulative_delta_score = clamp((buy_moves - sell_moves) / transitions, -1.0, 1.0)
        avg_spread = max(1e-9, sum(spreads) / max(len(spreads), 1))
        spread_variance = sum((value - avg_spread) ** 2 for value in spreads) / max(len(spreads), 1)
        spread_stability = clamp(1.0 - ((spread_variance ** 0.5) / max(avg_spread, 1e-9)), 0.0, 1.0)
        drift_score = clamp((mids[-1] - mids[0]) / max(avg_spread * 6.0, 1e-9), -1.0, 1.0)
        book = self.get_market_book(symbol)
        bid_volume = 0.0
        ask_volume = 0.0
        book_mid = mids[-1]
        near_bid_volume = 0.0
        near_ask_volume = 0.0
        for level in book:
            price = float(level.get("price", 0.0) or 0.0)
            volume = float(level.get("volume", level.get("volume_real", 0.0)) or 0.0)
            if price <= 0.0 or volume <= 0.0:
                continue
            if price <= book_mid:
                bid_volume += volume
                if abs(book_mid - price) <= (avg_spread * 2.5):
                    near_bid_volume += volume
            else:
                ask_volume += volume
                if abs(price - book_mid) <= (avg_spread * 2.5):
                    near_ask_volume += volume
        depth_total = bid_volume + ask_volume
        depth_imbalance = clamp((bid_volume - ask_volume) / max(depth_total, 1e-9), -1.0, 1.0) if depth_total > 0 else 0.0
        near_depth_total = near_bid_volume + near_ask_volume
        near_depth_imbalance = (
            clamp((near_bid_volume - near_ask_volume) / max(near_depth_total, 1e-9), -1.0, 1.0)
            if near_depth_total > 0
            else depth_imbalance
        )
        recent_window = max(4, len(mids) // 3)
        recent_move_sum = sum(abs(mids[index] - mids[index - 1]) for index in range(max(1, len(mids) - recent_window + 1), len(mids)))
        baseline_move_sum = sum(abs(mids[index] - mids[index - 1]) for index in range(1, len(mids)))
        baseline_move = baseline_move_sum / max(len(mids) - 1, 1)
        recent_move = recent_move_sum / max(recent_window - 1, 1)
        sweep_velocity = clamp(recent_move / max(baseline_move * 1.35, avg_spread * 0.5, 1e-9), 0.0, 1.0)
        absorption_score = clamp(
            (1.0 - abs(drift_score - cumulative_delta_score)) * 0.60
            + spread_stability * 0.25
            + (1.0 - min(abs(recent_move - baseline_move) / max(avg_spread, 1e-9), 1.0)) * 0.15,
            0.0,
            1.0,
        )
        iceberg_score = clamp((abs(near_depth_imbalance) * 0.45) + (absorption_score * 0.40) + (sweep_velocity * 0.15), 0.0, 1.0)
        spread_shock_score = clamp(max(spreads) / max(avg_spread * 2.5, 1e-9) - 0.4, 0.0, 1.0)
        quote_pull_stack_score = clamp((abs(near_depth_imbalance) * 0.55) + (sweep_velocity * 0.25) - (spread_shock_score * 0.15), 0.0, 1.0)
        pressure_score = clamp(
            0.5
            + (0.20 * cumulative_delta_score)
            + (0.14 * depth_imbalance)
            + (0.12 * drift_score)
            + (0.08 * (spread_stability - 0.5))
            + (0.06 * (near_depth_imbalance - 0.0))
            + (0.05 * (sweep_velocity - 0.5))
            + (0.04 * (absorption_score - 0.5))
            - (0.06 * spread_shock_score),
            0.0,
            1.0,
        )
        confidence = clamp(
            (abs(pressure_score - 0.5) * 2.0 * min(1.0, len(mids) / 24.0))
            + (sweep_velocity * 0.12)
            + (absorption_score * 0.08)
            + (iceberg_score * 0.08)
            - (spread_shock_score * 0.10),
            0.0,
            1.0,
        )
        direction = "bullish" if pressure_score >= 0.56 else "bearish" if pressure_score <= 0.44 else "neutral"
        return {
            "ready": True,
            "direction": direction,
            "confidence": float(confidence),
            "pressure_score": float(pressure_score),
            "cumulative_delta_score": float(cumulative_delta_score),
            "depth_imbalance": float(depth_imbalance),
            "dom_imbalance": float(near_depth_imbalance),
            "drift_score": float(drift_score),
            "spread_stability": float(spread_stability),
            "tick_count": int(len(mids)),
            "book_levels": int(len(book)),
            "sweep_velocity": float(sweep_velocity),
            "absorption_score": float(absorption_score),
            "iceberg_score": float(iceberg_score),
            "spread_shock_score": float(spread_shock_score),
            "quote_pull_stack_score": float(quote_pull_stack_score),
        }

    def get_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame:
        if pd is None:
            raise RuntimeError("pandas is required for market data operations")
        started = time.perf_counter()
        module = self._get_mt5()
        if self.disable_mt5:
            self._set_market_data_status(mode="synthetic_fallback", source="dry_run", ready=True, latency_ms=0)
            return self._synthetic_rates(count)
        if module is None:
            external_frame = self._external_rates(symbol, timeframe, count)
            if external_frame is not None:
                return external_frame
            if self.allow_synthetic_fallback:
                self._set_market_data_status(mode="synthetic_fallback", source="synthetic_rates", ready=False, latency_ms=0)
                return self._synthetic_rates(count)
            raise RuntimeError(f"live_market_data_unavailable:{symbol}:{timeframe}")
        self.ensure_connection()
        try:
            rates = module.copy_rates_from_pos(symbol, self._timeframe_constant(module, timeframe), 0, count)
        except Exception:
            rates = None
        if rates is None:
            external_frame = self._external_rates(symbol, timeframe, count)
            if external_frame is not None:
                return external_frame
            raise RuntimeError(f"Unable to fetch rates for {symbol} {timeframe}")
        frame = pd.DataFrame(rates)
        if frame.empty:
            external_frame = self._external_rates(symbol, timeframe, count)
            if external_frame is not None:
                return external_frame
            raise RuntimeError(f"empty_rates:{symbol}:{timeframe}")
        frame["time"] = pd.to_datetime(frame["time"], unit="s", utc=True)
        latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000.0))
        self._set_market_data_status(mode="mt5_live", source="mt5_python", ready=True, latency_ms=latency_ms)
        return frame

    def positions(self, symbol: str | None = None) -> list[dict[str, Any]]:
        module = self._get_mt5()
        if self.disable_mt5 or module is None:
            return []
        self.ensure_connection()
        positions = module.positions_get(symbol=symbol) if symbol else module.positions_get()
        return [position._asdict() for position in (positions or [])]

    def order_send(
        self,
        *,
        symbol: str,
        side: str,
        volume: float,
        price: float,
        sl: float,
        tp: float,
        slippage_points: int,
        magic: int,
        comment: str,
    ) -> OrderResult:
        if sl <= 0 or tp <= 0:
            return OrderResult(False, None, "SL and TP must both be set", {})

        module = self._get_mt5()
        if self.disable_mt5 or module is None:
            return OrderResult(True, comment, "simulated", {"retcode": "SIMULATED"})

        self.ensure_connection()
        side_upper = side.upper()
        request = {
            "action": module.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": module.ORDER_TYPE_BUY if side_upper == "BUY" else module.ORDER_TYPE_SELL,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": slippage_points,
            "magic": magic,
            "comment": comment,
            "type_time": module.ORDER_TIME_GTC,
            "type_filling": module.ORDER_FILLING_IOC,
        }
        result = module.order_send(request)
        if result is None:
            return OrderResult(False, None, "order_send returned None", {})
        payload = result._asdict()
        accepted = int(payload.get("retcode", 0)) == getattr(module, "TRADE_RETCODE_DONE", 10009)
        order_id = str(payload.get("order")) if accepted else None
        reason = "accepted" if accepted else f"retcode={payload.get('retcode')}"
        return OrderResult(accepted, order_id, reason, payload)

    def modify_position(self, ticket: int, sl: float | None = None, tp: float | None = None) -> bool:
        module = self._get_mt5()
        if self.disable_mt5 or module is None:
            return True
        self.ensure_connection()
        request = {"action": module.TRADE_ACTION_SLTP, "position": ticket}
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp
        result = module.order_send(request)
        return result is not None and int(result.retcode) == getattr(module, "TRADE_RETCODE_DONE", 10009)

    def close_position(self, position: dict[str, Any], slippage_points: int) -> bool:
        symbol = str(position["symbol"])
        tick = self.get_tick(symbol)
        module = self._get_mt5()
        current_type = getattr(module, "POSITION_TYPE_BUY", 0) if module is not None else 0
        side_type = int(position.get("type", 0))
        side = "SELL" if side_type == current_type else "BUY"
        price = float(tick["bid"] if side == "SELL" else tick["ask"])
        result = self.order_send(
            symbol=symbol,
            side=side,
            volume=float(position["volume"]),
            price=price,
            sl=float(position.get("sl", 0.0) or price),
            tp=float(position.get("tp", 0.0) or price),
            slippage_points=slippage_points,
            magic=int(position.get("magic", 0)),
            comment=f"close-{position['ticket']}",
        )
        return result.accepted

    def reduce_position(self, position: dict[str, Any], volume_to_close: float, slippage_points: int) -> bool:
        if volume_to_close <= 0:
            return False
        symbol = str(position["symbol"])
        tick = self.get_tick(symbol)
        module = self._get_mt5()
        current_type = getattr(module, "POSITION_TYPE_BUY", 0) if module is not None else 0
        side = "SELL" if int(position.get("type", 0)) == current_type else "BUY"
        price = float(tick["bid"] if side == "SELL" else tick["ask"])
        result = self.order_send(
            symbol=symbol,
            side=side,
            volume=volume_to_close,
            price=price,
            sl=float(position.get("sl", 0.0) or 0.0),
            tp=float(position.get("tp", 0.0) or 0.0),
            slippage_points=slippage_points,
            magic=int(position.get("magic", 0)),
            comment=f"reduce-{position['ticket']}",
        )
        return result.accepted

    def verify_live_allowed(self, live_enabled: bool, minimum_demo_hours: int = 48) -> bool:
        if not live_enabled:
            return True
        connection = sqlite3.connect(self.journal_db)
        try:
            row = connection.execute(
                """
                SELECT MIN(created_at), MAX(created_at)
                FROM trade_journal
                WHERE mode = 'DEMO' AND status IN ('EXECUTED', 'CLOSED')
                """
            ).fetchone()
        finally:
            connection.close()
        if not row or not row[0] or not row[1]:
            return False
        start = datetime.fromisoformat(str(row[0]))
        end = datetime.fromisoformat(str(row[1]))
        elapsed_hours = (end - start).total_seconds() / 3600
        return elapsed_hours >= minimum_demo_hours

    def normalize_volume(self, symbol: str, requested_volume: float) -> float:
        info = self.get_symbol_info(symbol)
        min_volume = float(info.get("volume_min", 0.01))
        max_volume = float(info.get("volume_max", 100.0))
        step = float(info.get("volume_step", 0.01))
        clamped = clamp(requested_volume, min_volume, max_volume)
        steps = round(clamped / step)
        return max(min_volume, round(steps * step, 2))

    def _symbol_candidates(self, symbol_hint: str) -> list[str]:
        mapped = self._mapped_candidates(symbol_hint)
        gold_aliases = ["XAUUSD", "XAUUSDM", "GOLD", "GOLDM", "GOLD.M"]
        if symbol_hint in {"XAUUSD", "GOLD"}:
            default_candidates = [symbol_hint] + [alias for alias in gold_aliases if alias != symbol_hint]
            return self._dedupe_candidates(mapped + default_candidates)
        silver_aliases = ["XAGUSD", "XAGUSDM", "SILVER", "SILVERM", "XAGUSD.M"]
        if symbol_hint in {"XAGUSD", "SILVER"}:
            default_candidates = [symbol_hint] + [alias for alias in silver_aliases if alias != symbol_hint]
            return self._dedupe_candidates(mapped + default_candidates)
        btc_aliases = ["BTCUSD", "BTCUSDM", "BTCUSD.M", "XBTUSD", "BTCUSDT"]
        if symbol_hint in {"BTCUSD", "XBTUSD"}:
            default_candidates = [symbol_hint] + [alias for alias in btc_aliases if alias != symbol_hint]
            return self._dedupe_candidates(mapped + default_candidates)
        dog_aliases = ["DOGUSD", "DOGEUSD", "DOGUSDM", "DOGEUSDM", "DOGUSD.A", "DOGEUSD.A"]
        if symbol_hint in {"DOGUSD", "DOGEUSD"}:
            default_candidates = [symbol_hint] + [alias for alias in dog_aliases if alias != symbol_hint]
            return self._dedupe_candidates(mapped + default_candidates)
        trump_aliases = ["TRUMPUSD", "TRUMPUSDM", "TRUMPUSD.A", "TRUMPUSDT"]
        if symbol_hint in {"TRUMPUSD", "TRUMPUSDT"}:
            default_candidates = [symbol_hint] + [alias for alias in trump_aliases if alias != symbol_hint]
            return self._dedupe_candidates(mapped + default_candidates)
        if symbol_hint == "AAPL":
            return self._dedupe_candidates(mapped + ["AAPL", "#AAPL", "AAPL.24H"])
        if symbol_hint in {"NVIDIA", "NVDA"}:
            return self._dedupe_candidates(mapped + ["NVIDIA", "NVDA", "#NVDA", "NVIDIA.24H", "NVDA.24H"])
        return self._dedupe_candidates(mapped + [symbol_hint])

    def _mapped_candidates(self, symbol_hint: str) -> list[str]:
        if not self.symbol_mapping:
            return []
        key = symbol_hint.upper()
        normalized = self._normalize_symbol_key(symbol_hint)
        raw = self.symbol_mapping.get(key)
        if raw is None:
            raw = self.symbol_mapping.get(normalized)
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw.upper()]
        if isinstance(raw, list):
            return [str(item).upper() for item in raw if str(item).strip()]
        return []

    @staticmethod
    def _dedupe_candidates(values: list[str]) -> list[str]:
        output: list[str] = []
        seen: set[str] = set()
        for value in values:
            normalized = value.upper()
            if normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
        return output

    @staticmethod
    def _normalize_symbol_key(value: str) -> str:
        return normalize_symbol_key(value)

    @staticmethod
    def _synthetic_rates(count: int) -> pd.DataFrame:
        if pd is None:
            raise RuntimeError("pandas is required for synthetic market data")
        index = pd.date_range(end=pd.Timestamp.utcnow(), periods=count, freq="5min", tz="UTC")
        base = pd.Series(range(count), index=index, dtype=float)
        close = 2200 + (base * 0.2).rolling(3, min_periods=1).mean()
        open_ = close.shift(1).fillna(close.iloc[0] - 0.1)
        high = pd.concat([open_, close], axis=1).max(axis=1) + 0.15
        low = pd.concat([open_, close], axis=1).min(axis=1) - 0.15
        spread = pd.Series(20, index=index, dtype=float)
        return pd.DataFrame(
            {
                "time": index,
                "open": open_.to_numpy(),
                "high": high.to_numpy(),
                "low": low.to_numpy(),
                "close": close.to_numpy(),
                "tick_volume": 100 + (base % 15).to_numpy(),
                "spread": spread.to_numpy(),
                "real_volume": 0,
            }
        )

    def _get_mt5(self) -> Any:
        if self.disable_mt5:
            return None
        return self.mt5_loader()

    def market_data_status(self) -> dict[str, Any]:
        payload = dict(self._market_data_status)
        if not payload:
            module = self._get_mt5()
            if self.disable_mt5:
                payload = {"mode": "synthetic_fallback", "source": "dry_run", "ready": True}
            elif module is not None:
                payload = {"mode": "mt5_live", "source": "mt5_python", "ready": bool(self.connected)}
            elif self.external_market_data_enabled:
                payload = {"mode": "external_live", "source": "external_provider", "ready": False}
            else:
                payload = {"mode": "unavailable", "source": "none", "ready": False}
        payload.setdefault("updated_at", datetime.utcnow().isoformat())
        payload.setdefault("error", "")
        payload.setdefault("latency_ms", None)
        payload.setdefault("provider_diagnostics", {})
        return payload

    def market_data_backend(self) -> str:
        return str(self.market_data_status().get("mode", "unknown"))

    def uses_external_market_data(self) -> bool:
        if self.disable_mt5:
            return False
        return self._get_mt5() is None and bool(self.external_market_data_enabled)

    def _credential_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.credentials.login is not None:
            kwargs["login"] = self.credentials.login
        if self.credentials.password:
            kwargs["password"] = self.credentials.password
        if self.credentials.server:
            kwargs["server"] = self.credentials.server
        return kwargs

    def _initialize_attempts(self, base_kwargs: dict[str, Any]) -> list[dict[str, Any]]:
        attempts: list[dict[str, Any]] = [base_kwargs]
        explicit_path = self.credentials.terminal_path or self.credentials.path
        seen_paths: set[str] = set()
        if explicit_path:
            attempts.append({**base_kwargs, "path": explicit_path})
            seen_paths.add(explicit_path)
        for candidate in self._autodiscover_terminal_paths():
            if candidate in seen_paths:
                continue
            attempts.append({**base_kwargs, "path": candidate})
            seen_paths.add(candidate)
        return attempts

    def _autodiscover_terminal_paths(self) -> list[str]:
        candidates = [
            r"C:\Program Files\MetaTrader 5\terminal64.exe",
            r"C:\Program Files (x86)\MetaTrader 5\terminal64.exe",
        ]
        output: list[str] = []
        for candidate in candidates:
            if self.platform_name.startswith("win") or os.path.exists(candidate):
                output.append(candidate)
        return output

    def _emit_mt5_failure(self, message: str) -> None:
        details = f"{message} last_error={self.last_init_error!r}"
        self._emit(details, level="warning")
        if self.platform_name == "darwin":
            self._emit(
                "macOS warning: MetaTrader5 Python integration expects a Windows terminal executable "
                "(for example metatrader64.exe). A Windows VM or VPS is the reliable option.",
                level="warning",
            )

    def _emit(self, message: str, level: str = "info") -> None:
        logger = self.logger
        if logger is not None and hasattr(logger, level):
            getattr(logger, level)(message)
            return
        print(message)

    @staticmethod
    def _last_error(module: Any) -> Any:
        if hasattr(module, "last_error"):
            try:
                return module.last_error()
            except Exception:
                return None
        return None

    @staticmethod
    def _timeframe_constant(module: Any, timeframe: str) -> Any:
        attr = TIMEFRAME_ATTR_MAP.get(timeframe, "")
        if attr and hasattr(module, attr):
            return getattr(module, attr)
        return TIMEFRAME_FALLBACKS.get(timeframe, 5)

    def _external_provider(self) -> MultiSourceMarketDataFallback | None:
        if not self.external_market_data_enabled:
            return None
        if self._external_market_data is None:
            self._external_market_data = MultiSourceMarketDataFallback(
                timeout_seconds=max(1.0, float(self.external_market_data_timeout_seconds))
            )
        return self._external_market_data

    def _external_tick(self, symbol: str) -> dict[str, Any] | None:
        provider = self._external_provider()
        if provider is None:
            self._set_market_data_status(mode="unavailable", source="none", ready=False, error="external_market_data_disabled", latency_ms=0)
            return None
        started = time.perf_counter()
        try:
            tick = provider.fetch_tick(symbol)
        except Exception as exc:
            self._set_market_data_status(
                mode="external_unavailable",
                source=str(getattr(provider, "last_source", "external_provider") or "external_provider"),
                ready=False,
                error=str(exc),
                latency_ms=0,
                provider_diagnostics=dict(getattr(provider, "diagnostics", lambda: {})() or {}),
            )
            return None
        latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000.0))
        self._set_market_data_status(
            mode="external_live",
            source=str(getattr(provider, "last_source", tick.get("source", "external_provider")) or "external_provider"),
            ready=True,
            latency_ms=latency_ms,
            provider_diagnostics=dict(getattr(provider, "diagnostics", lambda: {})() or {}),
        )
        return tick

    def _external_rates(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame | None:
        provider = self._external_provider()
        if provider is None:
            self._set_market_data_status(mode="unavailable", source="none", ready=False, error="external_market_data_disabled", latency_ms=0)
            return None
        started = time.perf_counter()
        try:
            frame = provider.fetch_rates(symbol, timeframe, count)
        except Exception as exc:
            self._set_market_data_status(
                mode="external_unavailable",
                source=str(getattr(provider, "last_source", "external_provider") or "external_provider"),
                ready=False,
                error=str(exc),
                latency_ms=0,
                provider_diagnostics=dict(getattr(provider, "diagnostics", lambda: {})() or {}),
            )
            return None
        latency_ms = int(max(0.0, (time.perf_counter() - started) * 1000.0))
        self._set_market_data_status(
            mode="external_live",
            source=str(getattr(provider, "last_source", "external_provider") or "external_provider"),
            ready=True,
            latency_ms=latency_ms,
            provider_diagnostics=dict(getattr(provider, "diagnostics", lambda: {})() or {}),
        )
        return frame

    def _set_market_data_status(
        self,
        *,
        mode: str,
        source: str,
        ready: bool,
        error: str = "",
        latency_ms: int | None = None,
        provider_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self._market_data_status = {
            "mode": str(mode),
            "source": str(source),
            "ready": bool(ready),
            "error": str(error or ""),
            "latency_ms": int(latency_ms) if latency_ms is not None else None,
            "updated_at": datetime.utcnow().isoformat(),
            "provider_diagnostics": dict(provider_diagnostics or {}),
        }
