from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.hyperliquid_lab.config import HyperliquidLabConfig, load_lab_config
from src.hyperliquid_lab.data_client import HyperliquidDataClient
from src.hyperliquid_lab.data_store import ParquetDataStore
from src.hyperliquid_lab.integrity import DataIntegrityReport, inspect_ohlcv


@dataclass(frozen=True)
class DatasetWriteResult:
    dataset: str
    source: str
    symbol: str
    timeframe: str | None
    rows: int
    path: Path
    integrity_report: DataIntegrityReport | None = None


@dataclass
class HyperliquidDataPipeline:
    config: HyperliquidLabConfig | None = None
    client: HyperliquidDataClient | None = None
    store: ParquetDataStore | None = None

    def __post_init__(self) -> None:
        if self.config is None:
            self.config = load_lab_config()
        if self.client is None:
            self.client = HyperliquidDataClient(self.config)
        if self.store is None:
            self.store = ParquetDataStore(self.config.storage_root)

    def pull_native_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: datetime | int | float | pd.Timestamp,
        end_time: datetime | int | float | pd.Timestamp,
    ) -> DatasetWriteResult:
        frame = self.client.fetch_native_ohlcv(symbol, timeframe, start_time, end_time)
        report = inspect_ohlcv(frame, timeframe=timeframe)
        path = self.store.write("ohlcv", frame, source="hyperliquid_native", symbol=symbol, timeframe=timeframe)
        return DatasetWriteResult("ohlcv", "hyperliquid_native", symbol.upper(), timeframe, len(frame), path, report)

    def pull_proxy_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: datetime | int | float | pd.Timestamp | None = None,
        limit: int = 1000,
    ) -> DatasetWriteResult:
        frame = self.client.fetch_proxy_ohlcv(symbol, timeframe, since=since, limit=limit)
        report = inspect_ohlcv(frame, timeframe=timeframe)
        path = self.store.write("ohlcv", frame, source=str(self.config.proxy_source), symbol=symbol, timeframe=timeframe)
        return DatasetWriteResult("ohlcv", str(self.config.proxy_source), symbol.upper(), timeframe, len(frame), path, report)

    def pull_native_trades(
        self,
        symbol: str,
        since: datetime | int | float | pd.Timestamp | None = None,
        limit: int = 1000,
    ) -> DatasetWriteResult:
        frame = self.client.fetch_trades(symbol, since=since, limit=limit)
        path = self.store.write("trades", frame, source="hyperliquid_native", symbol=symbol)
        return DatasetWriteResult("trades", "hyperliquid_native", symbol.upper(), None, len(frame), path)

    def pull_native_order_book(self, symbol: str) -> DatasetWriteResult:
        frame = self.client.fetch_native_order_book(symbol, depth=int(self.config.max_order_book_levels))
        path = self.store.write("book", frame, source="hyperliquid_native", symbol=symbol)
        return DatasetWriteResult("book", "hyperliquid_native", symbol.upper(), None, len(frame), path)

    def pull_configured_native_ohlcv(
        self,
        start_time: datetime | int | float | pd.Timestamp,
        end_time: datetime | int | float | pd.Timestamp,
    ) -> list[DatasetWriteResult]:
        results: list[DatasetWriteResult] = []
        for symbol in self.config.assets:
            for timeframe in self.config.native_intervals:
                results.append(self.pull_native_ohlcv(symbol, timeframe, start_time, end_time))
        return results
