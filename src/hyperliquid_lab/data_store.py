from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.hyperliquid_lab.data_client import BOOK_COLUMNS, OHLCV_COLUMNS, TRADES_COLUMNS
from src.utils import ensure_directory


DATASET_COLUMNS = {
    "ohlcv": OHLCV_COLUMNS,
    "trades": TRADES_COLUMNS,
    "book": BOOK_COLUMNS,
}

DEDUP_KEYS = {
    "ohlcv": ["source", "venue", "symbol", "timeframe", "close_time_utc"],
    "trades": ["source", "venue", "symbol", "trade_id", "timestamp_utc", "price", "size"],
    "book": ["source", "venue", "symbol", "timestamp_utc", "side", "level"],
}


@dataclass
class ParquetDataStore:
    root: Path

    def __post_init__(self) -> None:
        ensure_directory(self.root)

    def write(self, dataset: str, frame: pd.DataFrame, *, source: str, symbol: str, timeframe: str | None = None) -> Path:
        normalized_dataset = self._normalize_dataset(dataset)
        clean = self._coerce_schema(normalized_dataset, frame)
        path = self.path_for(normalized_dataset, source=source, symbol=symbol, timeframe=timeframe)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = pd.read_parquet(path)
            clean = pd.concat([existing, clean], ignore_index=True)
        clean = self._dedupe_and_sort(normalized_dataset, clean)
        clean.to_parquet(path, index=False)
        return path

    def read(self, dataset: str, *, source: str, symbol: str, timeframe: str | None = None) -> pd.DataFrame:
        path = self.path_for(dataset, source=source, symbol=symbol, timeframe=timeframe)
        if not path.exists():
            return pd.DataFrame(columns=DATASET_COLUMNS[self._normalize_dataset(dataset)])
        return pd.read_parquet(path)

    def path_for(self, dataset: str, *, source: str, symbol: str, timeframe: str | None = None) -> Path:
        normalized_dataset = self._normalize_dataset(dataset)
        safe_source = self._safe_part(source)
        safe_symbol = self._safe_part(symbol.upper())
        if normalized_dataset == "ohlcv":
            if not timeframe:
                raise ValueError("timeframe is required for OHLCV data")
            return self.root / "ohlcv" / safe_source / safe_symbol / self._safe_part(timeframe) / "data.parquet"
        if normalized_dataset == "trades":
            return self.root / "trades" / safe_source / safe_symbol / "data.parquet"
        return self.root / "book" / safe_source / safe_symbol / "snapshots.parquet"

    @staticmethod
    def assert_native_evidence(frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        sources = {str(item).lower() for item in frame.get("source", pd.Series(dtype=str)).dropna().unique()}
        qualities = {str(item).lower() for item in frame.get("data_quality", pd.Series(dtype=str)).dropna().unique()}
        if "proxy_only" in qualities or any(source != "hyperliquid_native" for source in sources):
            raise ValueError("proxy_data_cannot_be_reported_as_hyperliquid_native")

    @staticmethod
    def _normalize_dataset(dataset: str) -> str:
        value = str(dataset).strip().lower()
        if value not in DATASET_COLUMNS:
            raise ValueError(f"Unsupported dataset: {dataset}")
        return value

    @staticmethod
    def _safe_part(value: str) -> str:
        return str(value).replace("/", "_").replace(":", "_")

    @staticmethod
    def _coerce_schema(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
        columns = DATASET_COLUMNS[dataset]
        missing = [column for column in columns if column not in frame.columns]
        if missing:
            raise ValueError(f"{dataset} frame missing columns: {missing}")
        clean = frame.loc[:, columns].copy()
        for column in ("open_time_utc", "close_time_utc", "timestamp_utc"):
            if column in clean.columns:
                clean[column] = pd.to_datetime(clean[column], utc=True)
        return clean

    @staticmethod
    def _dedupe_and_sort(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
        keys = DEDUP_KEYS[dataset]
        clean = frame.drop_duplicates(subset=keys, keep="last")
        sort_columns = [column for column in ("close_time_utc", "timestamp_utc", "side", "level") if column in clean.columns]
        if sort_columns:
            clean = clean.sort_values(sort_columns)
        return clean.reset_index(drop=True)
