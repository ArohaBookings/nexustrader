from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.hyperliquid_lab import HyperliquidDataPipeline, load_lab_config


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pull Hyperliquid research data into the Phase 1 parquet store.")
    parser.add_argument("--start", required=True, help="UTC start timestamp, e.g. 2026-04-01T00:00:00Z")
    parser.add_argument("--end", required=True, help="UTC end timestamp, e.g. 2026-04-02T00:00:00Z")
    parser.add_argument("--symbols", nargs="*", help="Assets to pull, default from config")
    parser.add_argument("--timeframes", nargs="*", help="Intervals to pull, default from config")
    parser.add_argument("--include-books", action="store_true", help="Also capture one L2 snapshot per symbol")
    parser.add_argument("--include-trades", action="store_true", help="Also pull recent trades per symbol")
    parser.add_argument("--trade-limit", type=int, default=1000)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = load_lab_config()
    pipeline = HyperliquidDataPipeline(config=config)
    symbols = [item.upper() for item in (args.symbols or config.assets)]
    timeframes = args.timeframes or config.native_intervals
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    for symbol in symbols:
        for timeframe in timeframes:
            result = pipeline.pull_native_ohlcv(symbol, timeframe, start, end)
            issue_codes = sorted(result.integrity_report.issue_codes()) if result.integrity_report else []
            print(f"{result.dataset} {result.source} {symbol} {timeframe} rows={result.rows} path={result.path} issues={issue_codes}")
        if args.include_books:
            result = pipeline.pull_native_order_book(symbol)
            print(f"{result.dataset} {result.source} {symbol} rows={result.rows} path={result.path}")
        if args.include_trades:
            result = pipeline.pull_native_trades(symbol, since=start, limit=args.trade_limit)
            print(f"{result.dataset} {result.source} {symbol} rows={result.rows} path={result.path}")


if __name__ == "__main__":
    main()
