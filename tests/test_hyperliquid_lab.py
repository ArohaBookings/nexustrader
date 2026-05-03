from __future__ import annotations

import math

import pandas as pd
import pytest

from src.hyperliquid_lab.data_client import HyperliquidDataClient
from src.hyperliquid_lab.data_store import ParquetDataStore
from src.hyperliquid_lab.integrity import assert_native_only, inspect_ohlcv
from src.hyperliquid_lab.backtesting import BarCloseBacktester, buy_and_hold_benchmark
from src.hyperliquid_lab.paper import PaperTradingEngine
from src.hyperliquid_lab.pipeline import HyperliquidDataPipeline
from src.hyperliquid_lab.risk import RiskConfig, daily_loss_limit_hit, evaluate_circuit_breaker, volatility_sized_quantity
from src.hyperliquid_lab.simulator import MarketSimulator, SimulatedOrder
from src.hyperliquid_lab.strategy import MovingAverageCrossoverStrategy, StrategyOrder
from src.hyperliquid_lab.telegram import TelegramCommandRouter, TelegramNotifier, format_event_message, format_overview_message
from src.hyperliquid_lab.walk_forward import monte_carlo_trade_bootstrap, rolling_walk_forward


def _ohlcv_frame(close_times: list[str], *, source: str = "hyperliquid_native", quality: str = "native") -> pd.DataFrame:
    close = pd.to_datetime(close_times, utc=True)
    return pd.DataFrame(
        {
            "source": [source] * len(close),
            "venue": ["hyperliquid" if source == "hyperliquid_native" else source] * len(close),
            "symbol": ["BTC"] * len(close),
            "timeframe": ["1m"] * len(close),
            "open_time_utc": close - pd.Timedelta(minutes=1),
            "close_time_utc": close,
            "open": [100.0] * len(close),
            "high": [101.0] * len(close),
            "low": [99.0] * len(close),
            "close": [100.5] * len(close),
            "volume": [10.0] * len(close),
            "is_closed": [True] * len(close),
            "data_quality": [quality] * len(close),
        }
    )


def _book(sizes: list[float], *, timestamp: str = "2026-04-01T00:00:00Z") -> pd.DataFrame:
    asks = [100.10 + (index * 0.10) for index in range(len(sizes))]
    bids = [99.90 - (index * 0.10) for index in range(len(sizes))]
    rows = []
    for level, (price, size) in enumerate(zip(bids, sizes), start=1):
        rows.append(
            {
                "source": "fixture",
                "venue": "hyperliquid",
                "symbol": "BTC",
                "timestamp_utc": pd.Timestamp(timestamp),
                "side": "bid",
                "level": level,
                "price": price,
                "size": size,
                "order_count": 1,
            }
        )
    for level, (price, size) in enumerate(zip(asks, sizes), start=1):
        rows.append(
            {
                "source": "fixture",
                "venue": "hyperliquid",
                "symbol": "BTC",
                "timestamp_utc": pd.Timestamp(timestamp),
                "side": "ask",
                "level": level,
                "price": price,
                "size": size,
                "order_count": 1,
            }
        )
    return pd.DataFrame(rows)


def _market_order(quantity: float, *, side: str = "buy") -> SimulatedOrder:
    return SimulatedOrder(
        order_id=f"order-{side}-{quantity}",
        symbol="BTC",
        side=side,
        order_type="market",
        quantity=quantity,
        timestamp_utc=pd.Timestamp("2026-04-01T00:00:00Z"),
    )


def test_ohlcv_integrity_accepts_utc_monotonic_close_time() -> None:
    frame = _ohlcv_frame(["2026-04-01T00:01:00Z", "2026-04-01T00:02:00Z", "2026-04-01T00:03:00Z"])
    report = inspect_ohlcv(frame, timeframe="1m")

    assert report.ok
    assert report.issue_codes() == set()


def test_gap_detector_flags_missing_bars_without_forward_filling() -> None:
    frame = _ohlcv_frame(["2026-04-01T00:01:00Z", "2026-04-01T00:03:00Z"])
    report = inspect_ohlcv(frame, timeframe="1m")

    assert not report.ok
    assert "missing_bar_gap" in report.issue_codes()


def test_proxy_data_cannot_be_reported_as_hyperliquid_native() -> None:
    frame = _ohlcv_frame(["2026-04-01T00:01:00Z"], source="kraken", quality="proxy_only")

    with pytest.raises(ValueError, match="proxy_data_cannot_be_reported_as_hyperliquid_native"):
        assert_native_only(frame)
    with pytest.raises(ValueError, match="proxy_data_cannot_be_reported_as_hyperliquid_native"):
        ParquetDataStore.assert_native_evidence(frame)


def test_parquet_store_round_trips_and_deduplicates_ohlcv(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    frame = _ohlcv_frame(["2026-04-01T00:01:00Z", "2026-04-01T00:01:00Z", "2026-04-01T00:02:00Z"])

    path = store.write("ohlcv", frame, source="hyperliquid_native", symbol="BTC", timeframe="1m")
    loaded = store.read("ohlcv", source="hyperliquid_native", symbol="BTC", timeframe="1m")

    assert path.exists()
    assert len(loaded) == 2
    assert list(loaded["close_time_utc"]) == list(pd.to_datetime(["2026-04-01T00:01:00Z", "2026-04-01T00:02:00Z"], utc=True))


def test_pipeline_pulls_native_ohlcv_into_store_from_fixed_client(tmp_path) -> None:
    class FixedClient:
        def fetch_native_ohlcv(self, symbol, timeframe, start_time, end_time):
            return _ohlcv_frame(["2026-04-01T00:01:00Z", "2026-04-01T00:02:00Z"])

    config = type(
        "Config",
        (),
        {
            "storage_root": tmp_path,
            "assets": ["BTC"],
            "native_intervals": ["1m"],
            "proxy_source": "kraken",
            "max_order_book_levels": 20,
        },
    )()
    pipeline = HyperliquidDataPipeline(config=config, client=FixedClient(), store=ParquetDataStore(tmp_path))

    result = pipeline.pull_native_ohlcv("BTC", "1m", pd.Timestamp("2026-04-01T00:00:00Z"), pd.Timestamp("2026-04-01T00:02:00Z"))

    assert result.rows == 2
    assert result.integrity_report.ok
    assert result.path.exists()


def test_taker_fee_is_applied_per_fill_exactly() -> None:
    simulator = MarketSimulator(taker_fee_rate=0.00045, min_fill_ratio=0.0, max_slippage_bps=50.0)
    result = simulator.execute_taker(_market_order(1.0), _book([2.0]))

    assert result.status == "filled"
    assert result.filled_qty == 1.0
    assert result.avg_price == 100.10
    assert math.isclose(result.fee, 100.10 * 1.0 * 0.00045, rel_tol=0.0, abs_tol=1e-12)
    assert len(result.fills) == 1


def test_taker_slippage_increases_with_order_size_vs_depth() -> None:
    simulator = MarketSimulator(taker_fee_rate=0.00045, min_fill_ratio=0.0, max_slippage_bps=100.0)
    book = _book([1.0, 1.0, 1.0])

    small = simulator.execute_taker(_market_order(1.0), book)
    large = simulator.execute_taker(_market_order(3.0), book)

    assert small.status == "filled"
    assert large.status == "filled"
    assert large.avg_price > small.avg_price
    assert large.slippage_bps > small.slippage_bps


def test_partial_fill_when_depth_is_insufficient_inside_slippage_limit() -> None:
    simulator = MarketSimulator(taker_fee_rate=0.00045, min_fill_ratio=0.25, max_slippage_bps=100.0)
    result = simulator.execute_taker(_market_order(4.0), _book([1.0, 1.0]))

    assert result.status == "partial"
    assert result.filled_qty == 2.0
    assert result.unfilled_qty == 2.0
    assert result.rejection_reason == ""


def test_maker_order_does_not_fill_from_ohlcv_only_data() -> None:
    simulator = MarketSimulator(maker_fee_rate=0.00015)
    order = SimulatedOrder(
        order_id="maker-1",
        symbol="BTC",
        side="buy",
        order_type="limit",
        quantity=1.0,
        limit_price=99.5,
        timestamp_utc=pd.Timestamp("2026-04-01T00:00:00Z"),
    )

    result = simulator.execute_maker(order, trades=None)

    assert result.status == "open"
    assert result.filled_qty == 0.0
    assert result.rejection_reason == "maker_requires_trade_prints"


def test_maker_order_requires_crossing_trade_prints_and_applies_maker_fee() -> None:
    simulator = MarketSimulator(maker_fee_rate=0.00015)
    order = SimulatedOrder(
        order_id="maker-2",
        symbol="BTC",
        side="buy",
        order_type="limit",
        quantity=2.0,
        limit_price=99.5,
        timestamp_utc=pd.Timestamp("2026-04-01T00:00:00Z"),
    )
    trades = pd.DataFrame(
        {
            "source": ["fixture", "fixture"],
            "venue": ["hyperliquid", "hyperliquid"],
            "symbol": ["BTC", "BTC"],
            "trade_id": ["a", "b"],
            "timestamp_utc": pd.to_datetime(["2026-04-01T00:00:01Z", "2026-04-01T00:00:02Z"], utc=True),
            "side": ["sell", "sell"],
            "price": [99.4, 99.3],
            "size": [0.75, 2.0],
        }
    )

    result = simulator.execute_maker(order, trades)

    assert result.status == "filled"
    assert result.avg_price == 99.5
    assert result.filled_qty == 2.0
    assert math.isclose(result.fee, 99.5 * 2.0 * 0.00015, rel_tol=0.0, abs_tol=1e-12)


def test_normalizers_are_deterministic_from_fixed_fixtures() -> None:
    candles = [
        {"t": 1775001600000, "T": 1775001659999, "s": "BTC", "i": "1m", "o": "100", "h": "101", "l": "99", "c": "100.5", "v": "12.3", "n": 4},
        {"t": 1775001660000, "T": 1775001719999, "s": "BTC", "i": "1m", "o": "100.5", "h": "102", "l": "100", "c": "101.5", "v": "9.1", "n": 3},
    ]
    first = HyperliquidDataClient.normalize_candle_snapshot(candles, source="hyperliquid_native", venue="hyperliquid")
    second = HyperliquidDataClient.normalize_candle_snapshot(candles, source="hyperliquid_native", venue="hyperliquid")

    pd.testing.assert_frame_equal(first, second)
    assert list(first.columns) == [
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


def _trend_frame(rows: int = 90) -> pd.DataFrame:
    times = pd.date_range("2026-04-01T00:01:00Z", periods=rows, freq="min")
    close = [100.0 + index * 0.2 for index in range(rows)]
    return pd.DataFrame(
        {
            "source": ["fixture"] * rows,
            "venue": ["hyperliquid"] * rows,
            "symbol": ["BTC"] * rows,
            "timeframe": ["1m"] * rows,
            "open_time_utc": times - pd.Timedelta(minutes=1),
            "close_time_utc": times,
            "open": close,
            "high": [value + 0.5 for value in close],
            "low": [value - 0.5 for value in close],
            "close": close,
            "volume": [10.0] * rows,
            "is_closed": [True] * rows,
            "data_quality": ["native"] * rows,
        }
    )


def test_strategy_signals_are_deterministic_and_prefix_stable() -> None:
    strategy = MovingAverageCrossoverStrategy(short_window=3, long_window=5, risk_fraction=0.2)
    data = _trend_frame(30)

    full = strategy.generate_signals(data)
    prefix = strategy.generate_signals(data.iloc[:20])

    pd.testing.assert_frame_equal(full.iloc[:20].reset_index(drop=True), prefix.reset_index(drop=True))
    assert "enter_long" in set(full["signal"])


def test_bar_close_backtester_reports_required_metrics() -> None:
    strategy = MovingAverageCrossoverStrategy(short_window=3, long_window=5)
    data = _trend_frame(40)
    signals = strategy.generate_signals(data)
    result = BarCloseBacktester(initial_cash=1000.0, fee_rate=0.0).run(data, signals)
    benchmark = buy_and_hold_benchmark(data, initial_cash=1000.0, fee_rate=0.0)

    assert not result.equity_curve.empty
    assert {"total_return", "sharpe", "max_drawdown", "win_rate", "avg_trade_duration_seconds"} <= set(result.metrics)
    assert benchmark.fill_model == "buy_and_hold"


def test_walk_forward_and_monte_carlo_are_deterministic() -> None:
    data = _trend_frame(80)
    result = rolling_walk_forward(
        data,
        lambda: MovingAverageCrossoverStrategy(short_window=3, long_window=5),
        train_bars=30,
        test_bars=20,
        step_bars=20,
        initial_cash=1000.0,
        fee_rate=0.0,
    )
    trades = pd.DataFrame({"pnl": [10.0, -5.0, 3.0, 4.0]})

    first = monte_carlo_trade_bootstrap(trades, iterations=100, seed=7, initial_equity=1000.0)
    second = monte_carlo_trade_bootstrap(trades, iterations=100, seed=7, initial_equity=1000.0)

    assert len(result.windows) == 2
    pd.testing.assert_frame_equal(first, second)


def test_risk_sizing_daily_loss_and_circuit_breaker() -> None:
    config = RiskConfig(max_position_pct=0.2, volatility_risk_fraction=0.01, max_daily_loss_pct=0.03, stale_data_seconds=10.0, max_api_errors=2)
    qty = volatility_sized_quantity(equity=1000.0, price=100.0, atr=2.0, config=config)
    state = evaluate_circuit_breaker(
        _trend_frame(20),
        last_data_time_utc=pd.Timestamp("2026-04-01T00:00:00Z"),
        now_utc=pd.Timestamp("2026-04-01T00:01:00Z"),
        api_errors=2,
        config=config,
    )

    assert qty <= 2.0
    assert daily_loss_limit_hit(1000.0, 960.0, config)
    assert state.halted
    assert "exchange_api_errors" in state.reasons
    assert "stale_data" in state.reasons


def test_paper_engine_logs_simulated_order_results() -> None:
    engine = PaperTradingEngine(simulator=MarketSimulator(max_slippage_bps=100.0, min_fill_ratio=0.0))
    order = StrategyOrder("BTC", "buy", "market", 1.0, "test", pd.Timestamp("2026-04-01T00:00:00Z"))

    result = engine.submit_order(order, _book([2.0]), mark_price=100.0)

    assert result.status == "filled"
    assert engine.portfolio.position_qty == 1.0
    assert len(engine.event_log) == 1


def test_telegram_formatting_and_command_router_do_not_require_network() -> None:
    sent: list[tuple[int | str, str]] = []

    class FakeClient:
        def send_message(self, chat_id, text, parse_mode="HTML"):
            sent.append((chat_id, text))
            return {"message_id": len(sent)}

    router = TelegramCommandRouter(
        lambda chat_id: TelegramNotifier(FakeClient(), chat_id),
        lambda: {"risk": {"halted": False}, "paper": {"mode": "internal_simulator_only"}},
    )
    handled = router.handle_update({"message": {"chat": {"id": 123}, "text": "/overview"}})

    assert handled
    assert sent and sent[0][0] == 123
    assert "server overview" in sent[0][1]
    assert "TEST" in format_event_message("hello", {"ok": True}, severity="TEST")
    assert "RISK" in format_overview_message({"risk": {"halted": False}})
