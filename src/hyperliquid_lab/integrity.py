from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from src.hyperliquid_lab.data_client import timeframe_to_timedelta


@dataclass(frozen=True)
class DataIntegrityIssue:
    code: str
    message: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataIntegrityReport:
    dataset: str
    issues: list[DataIntegrityIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def add(self, code: str, message: str, **context: Any) -> None:
        self.issues.append(DataIntegrityIssue(code=code, message=message, context=dict(context)))

    def issue_codes(self) -> set[str]:
        return {issue.code for issue in self.issues}

    def raise_if_failed(self) -> None:
        if self.ok:
            return
        details = "; ".join(f"{issue.code}:{issue.message}" for issue in self.issues)
        raise ValueError(details)


def inspect_ohlcv(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    now_utc: pd.Timestamp | None = None,
    max_stale_seconds: float | None = None,
) -> DataIntegrityReport:
    report = DataIntegrityReport(dataset="ohlcv")
    required = {
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
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        report.add("missing_columns", "OHLCV frame is missing required columns", missing=missing)
        return report
    if frame.empty:
        report.add("empty_frame", "OHLCV frame is empty")
        return report

    close_time = pd.to_datetime(frame["close_time_utc"], utc=True)
    open_time = pd.to_datetime(frame["open_time_utc"], utc=True)
    if close_time.isna().any() or open_time.isna().any():
        report.add("invalid_timestamp", "OHLCV timestamps must parse as UTC")
        return report
    if not _is_utc_dtype(close_time) or not _is_utc_dtype(open_time):
        report.add("timestamp_not_utc", "OHLCV timestamps must be timezone-aware UTC")
    if close_time.duplicated().any():
        report.add("duplicate_timestamp", "Duplicate close_time_utc values detected", count=int(close_time.duplicated().sum()))
    if not close_time.is_monotonic_increasing:
        report.add("non_monotonic_timestamp", "close_time_utc must be strictly increasing")
    if (close_time <= open_time).any():
        report.add("lookahead_timestamp_geometry", "close_time_utc must be after open_time_utc")

    expected_delta = timeframe_to_timedelta(timeframe)
    gaps = close_time.diff().dropna()
    missing_gaps = gaps[gaps > expected_delta]
    if not missing_gaps.empty:
        report.add(
            "missing_bar_gap",
            "Missing bar gap detected; do not forward-fill silently",
            count=int(len(missing_gaps)),
            expected_seconds=float(expected_delta.total_seconds()),
            max_gap_seconds=float(missing_gaps.max().total_seconds()),
        )
    overlap_gaps = gaps[gaps <= pd.Timedelta(0)]
    if not overlap_gaps.empty:
        report.add("overlapping_bar_time", "Close timestamps overlap or repeat", count=int(len(overlap_gaps)))

    if max_stale_seconds is not None:
        now = pd.Timestamp.now(tz="UTC") if now_utc is None else pd.Timestamp(now_utc).tz_convert("UTC")
        age_seconds = (now - close_time.iloc[-1]).total_seconds()
        if age_seconds > float(max_stale_seconds):
            report.add("stale_data", "Latest OHLCV bar is stale", age_seconds=float(age_seconds))
    return report


def assert_native_only(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    qualities = {str(value).lower() for value in frame.get("data_quality", pd.Series(dtype=str)).dropna().unique()}
    sources = {str(value).lower() for value in frame.get("source", pd.Series(dtype=str)).dropna().unique()}
    if "proxy_only" in qualities or any(source != "hyperliquid_native" for source in sources):
        raise ValueError("proxy_data_cannot_be_reported_as_hyperliquid_native")


def _is_utc_dtype(series: pd.Series) -> bool:
    dtype = getattr(series.dt, "tz", None)
    return str(dtype) == "UTC"
