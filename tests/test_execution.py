from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from datetime import datetime, timezone
import sqlite3
import unittest
from unittest.mock import patch

try:
    from src.execution import ExecutionRequest, ExecutionService, TradeJournal
    from src.logger import LoggerFactory
    from src.mt5_client import OrderResult
    from src.performance_report import build_performance_report
    from src.risk_engine import RiskEngine
    from src.session_calendar import SYDNEY
    _HAS_EXEC_DEPS = True
    _SKIP_REASON = ""
except ModuleNotFoundError as exc:
    ExecutionRequest = None  # type: ignore
    ExecutionService = None  # type: ignore
    TradeJournal = None  # type: ignore
    LoggerFactory = None  # type: ignore
    OrderResult = None  # type: ignore
    build_performance_report = None  # type: ignore
    RiskEngine = None  # type: ignore
    SYDNEY = None  # type: ignore
    _HAS_EXEC_DEPS = False
    _SKIP_REASON = f"missing dependency: {exc.name}"


class FakeMT5Client:
    def __init__(self) -> None:
        self.calls = 0

    def order_send(self, **kwargs):
        self.calls += 1
        return OrderResult(True, f"ticket-{self.calls}", "accepted", {"request": kwargs})

    def modify_position(self, ticket, sl=None, tp=None):
        return True

    def reduce_position(self, position, volume_to_close, slippage_points):
        return True

    def positions(self, symbol=None):
        return []

    def close_position(self, position, slippage_points):
        return True


@unittest.skipUnless(_HAS_EXEC_DEPS, _SKIP_REASON)
class ExecutionTests(unittest.TestCase):
    def test_trade_journal_connect_falls_back_when_wal_cannot_be_enabled(self) -> None:
        class FakeConnection:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def execute(self, sql: str):
                self.calls.append(sql)
                if sql == "PRAGMA journal_mode=WAL":
                    raise sqlite3.OperationalError("database or disk is full")
                return self

        connection = FakeConnection()
        journal = TradeJournal(Path("/tmp/nonexistent-trades.sqlite"))

        with patch("src.execution.sqlite3.connect", return_value=connection):
            resolved = journal._connect()

        self.assertIs(resolved, connection)
        self.assertIn("PRAGMA busy_timeout=10000", connection.calls)
        self.assertIn("PRAGMA journal_mode=WAL", connection.calls)
        self.assertIn("PRAGMA journal_mode=DELETE", connection.calls)

    def _record_and_close_loss(
        self,
        journal: TradeJournal,
        *,
        signal_id: str,
        symbol: str,
        account: str,
        magic: int,
        equity_open: float,
        equity_close: float,
    ) -> None:
        request = ExecutionRequest(
            signal_id=signal_id,
            symbol=symbol,
            side="BUY",
            volume=0.01,
            entry_price=2200.0 if symbol == "XAUUSD" else 1.1000,
            stop_price=2198.0 if symbol == "XAUUSD" else 1.0990,
            take_profit_price=2204.0 if symbol == "XAUUSD" else 1.1020,
            mode="LIVE",
            setup=f"{symbol}_TEST",
            regime="TRENDING",
            probability=0.65,
            expected_value_r=0.25,
            slippage_points=10,
            trading_enabled=True,
            account=account,
            magic=magic,
        )
        journal.record_execution(request, OrderResult(True, signal_id, "accepted", {}), equity=equity_open)
        journal.mark_closed(signal_id, pnl_amount=-1.0, pnl_r=-1.0, equity_after_close=equity_close)

    def test_idempotent_signal_submission(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)
            logger = LoggerFactory().build()
            service = ExecutionService(FakeMT5Client(), journal, logger)

            request = ExecutionRequest(
                signal_id="sig-1",
                symbol="XAUUSD",
                side="BUY",
                volume=0.1,
                entry_price=2200.0,
                stop_price=2199.0,
                take_profit_price=2201.5,
                mode="DEMO",
                setup="TREND_CONTINUATION",
                regime="TRENDING",
                probability=0.66,
                expected_value_r=0.22,
                slippage_points=10,
                trading_enabled=True,
            )

            first = service.place(request, equity=1000.0)
            second = service.place(request, equity=1000.0)

            self.assertTrue(first.accepted)
            self.assertFalse(second.accepted)
            self.assertEqual(second.reason, "duplicate_signal")
            self.assertEqual(len(journal.get_open_positions()), 1)

    def test_strategy_key_persists_through_journal_reads(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)

            request = ExecutionRequest(
                signal_id="sig-strategy-key",
                symbol="AUDJPY",
                side="BUY",
                volume=0.01,
                entry_price=97.10,
                stop_price=96.90,
                take_profit_price=97.50,
                mode="LIVE",
                setup="GENERIC_BREAKOUT",
                regime="BREAKOUT_EXPANSION",
                probability=0.67,
                expected_value_r=0.42,
                slippage_points=10,
                trading_enabled=True,
                account="Main",
                magic=7777,
                timeframe="M15",
                strategy_key="AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
            )

            journal.record_execution(request, OrderResult(True, "ticket-1", "accepted", {}), equity=1000.0)

            trade = journal.get_trade("sig-strategy-key")
            open_positions = journal.get_open_positions(account="Main", magic=7777)

            self.assertIsNotNone(trade)
            assert trade is not None
            self.assertEqual(trade.get("strategy_key"), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")
            self.assertEqual(len(open_positions), 1)
            self.assertEqual(open_positions[0].get("strategy_key"), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")

    def test_record_execution_backfills_identity_from_entry_context(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)

            request = ExecutionRequest(
                signal_id="sig-context-identity",
                symbol="AUDJPY",
                side="BUY",
                volume=0.01,
                entry_price=97.10,
                stop_price=96.90,
                take_profit_price=97.50,
                mode="LIVE",
                setup="GENERIC_BREAKOUT",
                regime="BREAKOUT_EXPANSION",
                probability=0.67,
                expected_value_r=0.42,
                slippage_points=10,
                trading_enabled=True,
                account="Main",
                magic=7777,
                timeframe="M15",
                entry_context_json='{"session_name":"TOKYO","strategy_key":"AUDJPY_TOKYO_MOMENTUM_BREAKOUT"}',
            )

            journal.record_execution(request, OrderResult(True, "ticket-2", "accepted", {}), equity=1000.0)

            trade = journal.get_trade("sig-context-identity")

            self.assertIsNotNone(trade)
            assert trade is not None
            self.assertEqual(trade.get("session_name"), "TOKYO")
            self.assertEqual(trade.get("strategy_key"), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")

    def test_record_execution_prefers_actual_fill_price_from_broker_result(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)

            request = ExecutionRequest(
                signal_id="sig-fill-price",
                symbol="XAUUSD",
                side="BUY",
                volume=0.01,
                entry_price=2200.0,
                stop_price=2198.0,
                take_profit_price=2204.0,
                mode="LIVE",
                setup="XAUUSD_M5_GRID_SCALPER_START",
                regime="RANGING",
                probability=0.71,
                expected_value_r=0.42,
                slippage_points=20,
                trading_enabled=True,
                broker_snapshot_json='{"avg_entry":2205.25}',
            )

            journal.record_execution(
                request,
                OrderResult(True, "ticket-fill", "accepted", {"price": 2206.5}),
                equity=1000.0,
            )

            trade = journal.get_trade("sig-fill-price")

            self.assertIsNotNone(trade)
            assert trade is not None
            self.assertAlmostEqual(float(trade.get("entry_price") or 0.0), 2206.5, places=6)

    def test_closed_trades_backfills_missing_strategy_key_from_legacy_setup(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)

            with journal._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO trade_journal (
                        signal_id, ticket, symbol, side, setup, mode, status, created_at, opened_at, closed_at,
                        entry_price, sl, tp, volume, probability, expected_value_r, regime, strategy_key,
                        equity_at_open, equity_after_close, pnl_amount, pnl_r, account, magic, timeframe,
                        proof_trade, entry_reason, ai_summary_json, broker_snapshot_json, entry_context_json,
                        account_currency, entry_spread_points, session_name
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "legacy-sig-1",
                        "legacy-ticket-1",
                        "AUDJPY",
                        "BUY",
                        "AUDJPY_SESSION_PULLBACK",
                        "LIVE",
                        "CLOSED",
                        "2026-03-12T00:00:00+00:00",
                        "2026-03-12T00:01:00+00:00",
                        "2026-03-12T00:10:00+00:00",
                        97.10,
                        96.90,
                        97.50,
                        0.01,
                        0.67,
                        0.42,
                        "TRENDING",
                        "",
                        1000.0,
                        1004.2,
                        4.2,
                        1.0,
                        "Main",
                        7777,
                        "M15",
                        0,
                        "",
                        "{}",
                        "{}",
                        "{}",
                        "USD",
                        10.0,
                        "TOKYO",
                    ),
                )

            closed = journal.closed_trades(5, account="Main", magic=7777)

            self.assertEqual(len(closed), 1)
            self.assertEqual(closed[0]["strategy_key"], "AUDJPY_TOKYO_CONTINUATION_PULLBACK")

    def test_live_mode_is_tradeable_only_if_upstream_enabled_trading(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)
            logger = LoggerFactory().build()
            service = ExecutionService(FakeMT5Client(), journal, logger)

            request = ExecutionRequest(
                signal_id="sig-live",
                symbol="XAUUSD",
                side="BUY",
                volume=0.1,
                entry_price=2200.0,
                stop_price=2199.0,
                take_profit_price=2201.5,
                mode="LIVE",
                setup="TREND_CONTINUATION",
                regime="TRENDING",
                probability=0.66,
                expected_value_r=0.22,
                slippage_points=10,
                trading_enabled=True,
            )

            receipt = service.place(request, equity=1000.0)

            self.assertTrue(receipt.accepted)

    def test_cooldown_blocks_after_loss(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)
            logger = LoggerFactory().build()
            service = ExecutionService(FakeMT5Client(), journal, logger)
            request = ExecutionRequest(
                signal_id="sig-loss",
                symbol="XAUUSD",
                side="BUY",
                volume=0.01,
                entry_price=2200.0,
                stop_price=2198.0,
                take_profit_price=2202.0,
                mode="DEMO",
                setup="TREND_CONTINUATION",
                regime="TRENDING",
                probability=0.61,
                expected_value_r=0.2,
                slippage_points=10,
                trading_enabled=True,
            )
            receipt = service.place(request, equity=1000.0)
            self.assertTrue(receipt.accepted)
            journal.mark_closed("sig-loss", pnl_amount=-3.0, pnl_r=-1.0, equity_after_close=997.0)

            reason = journal.cooldown_block_reason(
                now=datetime.now(tz=timezone.utc),
                symbol="XAUUSD",
                cooldown_after_loss_minutes=20,
                cooldown_after_win_minutes=5,
            )
            self.assertIsNotNone(reason)
            self.assertIn("cooldown_after_loss", str(reason))

    def test_force_test_rows_are_excluded_from_live_stats_and_closed_trade_rollups(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)

            journal.record_execution(
                ExecutionRequest(
                    signal_id="FORCE_TEST::BTCUSD::M15::BUY::ACC::1",
                    symbol="BTCUSD",
                    side="BUY",
                    volume=0.01,
                    entry_price=68000.0,
                    stop_price=67998.0,
                    take_profit_price=68002.0,
                    mode="LIVE",
                    setup="BTCUSD_M15_FORCE_TEST",
                    regime="TRENDING",
                    probability=0.65,
                    expected_value_r=0.25,
                    slippage_points=20,
                    trading_enabled=True,
                ),
                OrderResult(True, "91001", "accepted", {}),
                equity=50.0,
            )
            journal.mark_closed("FORCE_TEST::BTCUSD::M15::BUY::ACC::1", pnl_amount=1.25, pnl_r=0.5, equity_after_close=51.25)

            journal.record_execution(
                ExecutionRequest(
                    signal_id="LIVE::BTCUSD::ENTRY::1",
                    symbol="BTCUSD",
                    side="BUY",
                    volume=0.01,
                    entry_price=68010.0,
                    stop_price=68000.0,
                    take_profit_price=68030.0,
                    mode="LIVE",
                    setup="BTCUSD_M15_WEEKEND_BREAKOUT",
                    regime="TRENDING",
                    probability=0.66,
                    expected_value_r=0.30,
                    slippage_points=20,
                    trading_enabled=True,
                ),
                OrderResult(True, "91002", "accepted", {}),
                equity=51.25,
            )
            journal.mark_closed("LIVE::BTCUSD::ENTRY::1", pnl_amount=-2.0, pnl_r=-1.0, equity_after_close=49.25)

            stats = journal.stats(current_equity=49.25)
            closed = journal.closed_trades(10)
            summary = journal.summary_last(10)
            survivability = journal.micro_survivability_summary(limit=10)
            last_closed = journal.last_closed_trade()

        self.assertEqual(stats.closed_trades_total, 1)
        self.assertEqual(len(closed), 1)
        self.assertEqual(closed[0]["signal_id"], "LIVE::BTCUSD::ENTRY::1")
        self.assertEqual(summary["trades"], 1.0)
        self.assertEqual(survivability["trades"], 1.0)
        self.assertIsNotNone(last_closed)
        assert last_closed is not None
        self.assertEqual(last_closed["signal_id"], "LIVE::BTCUSD::ENTRY::1")

    def test_summary_and_report_fallback_to_pnl_amount_when_pnl_r_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)

            journal.record_execution(
                ExecutionRequest(
                    signal_id="LIVE::BTCUSD::WIN::AMOUNT_ONLY",
                    symbol="BTCUSD",
                    side="SELL",
                    volume=0.01,
                    entry_price=68000.0,
                    stop_price=68100.0,
                    take_profit_price=67800.0,
                    mode="LIVE",
                    setup="BTC_TREND_SCALP",
                    regime="TRENDING",
                    probability=0.7,
                    expected_value_r=0.5,
                    slippage_points=20,
                    trading_enabled=True,
                ),
                OrderResult(True, "93001", "accepted", {}),
                equity=50.0,
            )
            journal.mark_closed("LIVE::BTCUSD::WIN::AMOUNT_ONLY", pnl_amount=2.0, pnl_r=0.0, equity_after_close=52.0)

            journal.record_execution(
                ExecutionRequest(
                    signal_id="LIVE::BTCUSD::LOSS::AMOUNT_ONLY",
                    symbol="BTCUSD",
                    side="SELL",
                    volume=0.01,
                    entry_price=68050.0,
                    stop_price=68150.0,
                    take_profit_price=67850.0,
                    mode="LIVE",
                    setup="BTC_TREND_SCALP",
                    regime="TRENDING",
                    probability=0.68,
                    expected_value_r=0.45,
                    slippage_points=20,
                    trading_enabled=True,
                ),
                OrderResult(True, "93002", "accepted", {}),
                equity=52.0,
            )
            journal.mark_closed("LIVE::BTCUSD::LOSS::AMOUNT_ONLY", pnl_amount=-1.0, pnl_r=0.0, equity_after_close=51.0)

            summary = journal.summary_last(10, symbol="BTCUSD")
            report = build_performance_report(journal.closed_trades(10))

        self.assertEqual(summary["trades"], 2.0)
        self.assertAlmostEqual(summary["win_rate"], 0.5)
        self.assertGreater(summary["profit_factor"], 1.0)
        self.assertEqual(report["overall"]["trades"], 2.0)
        self.assertAlmostEqual(report["overall"]["win_rate"], 0.5)
        self.assertGreater(report["overall"]["profit_factor"], 1.0)

    def test_proof_trade_loss_does_not_increment_live_consecutive_losses(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)

            journal.record_execution(
                ExecutionRequest(
                    signal_id="LIVE::BTCUSD::WIN::1",
                    symbol="BTCUSD",
                    side="BUY",
                    volume=0.01,
                    entry_price=68010.0,
                    stop_price=68000.0,
                    take_profit_price=68030.0,
                    mode="LIVE",
                    setup="BTCUSD_M15_WEEKEND_BREAKOUT",
                    regime="TRENDING",
                    probability=0.66,
                    expected_value_r=0.30,
                    slippage_points=20,
                    trading_enabled=True,
                ),
                OrderResult(True, "92001", "accepted", {}),
                equity=51.25,
            )
            journal.mark_closed("LIVE::BTCUSD::WIN::1", pnl_amount=1.0, pnl_r=0.5, equity_after_close=52.25)

            journal.record_execution(
                ExecutionRequest(
                    signal_id="FORCE_TEST::BTCUSD::M15::BUY::ACC::2",
                    symbol="BTCUSD",
                    side="BUY",
                    volume=0.01,
                    entry_price=68000.0,
                    stop_price=67998.0,
                    take_profit_price=68002.0,
                    mode="LIVE",
                    setup="BTCUSD_M15_FORCE_TEST",
                    regime="TRENDING",
                    probability=0.65,
                    expected_value_r=0.25,
                    slippage_points=20,
                    trading_enabled=True,
                    proof_trade=True,
                ),
                OrderResult(True, "92002", "accepted", {}),
                equity=52.25,
            )
            journal.mark_closed("FORCE_TEST::BTCUSD::M15::BUY::ACC::2", pnl_amount=-0.5, pnl_r=-0.2, equity_after_close=51.75)

            stats = journal.stats(current_equity=51.75)

        self.assertEqual(stats.closed_trades_total, 1)
        self.assertEqual(stats.consecutive_losses, 0)

    def test_reset_daily_guard_clears_today_loss_state_for_scope_only(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)

            request = ExecutionRequest(
                signal_id="LIVE::EURUSD::LOSS::1",
                symbol="EURUSD",
                side="SELL",
                volume=0.01,
                entry_price=1.1000,
                stop_price=1.1010,
                take_profit_price=1.0980,
                mode="LIVE",
                setup="EURUSD_PULLBACK",
                regime="TRENDING",
                probability=0.64,
                expected_value_r=0.20,
                slippage_points=10,
                trading_enabled=True,
                account="Main",
                magic=7777,
            )
            journal.record_execution(request, OrderResult(True, "1001", "accepted", {}), equity=80.0)
            journal.mark_closed("LIVE::EURUSD::LOSS::1", pnl_amount=-2.0, pnl_r=-1.0, equity_after_close=78.0)

            before = journal.stats(current_equity=78.0, account="Main", magic=7777)
            self.assertLess(before.daily_pnl_pct, 0.0)

            journal.reset_daily_guard(account="Main", magic=7777, current_equity=78.0)

            after = journal.stats(current_equity=78.0, account="Main", magic=7777)
            self.assertEqual(after.daily_pnl_pct, 0.0)
            self.assertEqual(after.daily_dd_pct_live, 0.0)
            self.assertEqual(after.soft_dd_trade_count, 0)

    def test_reset_daily_guard_clears_same_day_cooldown_block_reason(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)

            request = ExecutionRequest(
                signal_id="LIVE::EURUSD::LOSS::COOLDOWN",
                symbol="EURUSD",
                side="SELL",
                volume=0.01,
                entry_price=1.1000,
                stop_price=1.1010,
                take_profit_price=1.0980,
                mode="LIVE",
                setup="EURUSD_PULLBACK",
                regime="TRENDING",
                probability=0.64,
                expected_value_r=0.20,
                slippage_points=10,
                trading_enabled=True,
                account="Main",
                magic=7777,
            )
            open_ts = datetime(2026, 3, 10, 5, 0, tzinfo=timezone.utc)
            close_ts = datetime(2026, 3, 10, 5, 2, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", side_effect=[open_ts, open_ts, close_ts, close_ts]):
                journal.record_execution(request, OrderResult(True, "1002", "accepted", {}), equity=80.0)
                journal.mark_closed("LIVE::EURUSD::LOSS::COOLDOWN", pnl_amount=-2.0, pnl_r=-1.0, equity_after_close=78.0)

            same_day_now = datetime(2026, 3, 10, 5, 5, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=same_day_now):
                before = journal.cooldown_block_reason(
                    now=same_day_now,
                    symbol="EURUSD",
                    cooldown_after_loss_minutes=20,
                    cooldown_after_win_minutes=5,
                    account="Main",
                    magic=7777,
                )
            self.assertIsNotNone(before)

            reset_ts = datetime(2026, 3, 10, 5, 6, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=reset_ts):
                journal.reset_daily_guard(account="Main", magic=7777, current_equity=78.0)

            after_ts = datetime(2026, 3, 10, 5, 7, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=after_ts):
                after = journal.cooldown_block_reason(
                    now=after_ts,
                    symbol="EURUSD",
                    cooldown_after_loss_minutes=20,
                    cooldown_after_win_minutes=5,
                    account="Main",
                    magic=7777,
                )
            self.assertIsNone(after)

    def test_cooldown_is_symbol_scoped_not_global(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)

            request = ExecutionRequest(
                signal_id="LIVE::EURUSD::LOSS::GLOBAL_SCOPE",
                symbol="EURUSD",
                side="SELL",
                volume=0.01,
                entry_price=1.1000,
                stop_price=1.1010,
                take_profit_price=1.0980,
                mode="LIVE",
                setup="EURUSD_PULLBACK",
                regime="TRENDING",
                probability=0.64,
                expected_value_r=0.20,
                slippage_points=10,
                trading_enabled=True,
                account="Main",
                magic=7777,
            )
            open_ts = datetime(2026, 3, 10, 5, 0, tzinfo=timezone.utc)
            close_ts = datetime(2026, 3, 10, 5, 2, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", side_effect=[open_ts, open_ts, close_ts, close_ts]):
                journal.record_execution(request, OrderResult(True, "2002", "accepted", {}), equity=80.0)
                journal.mark_closed("LIVE::EURUSD::LOSS::GLOBAL_SCOPE", pnl_amount=-2.0, pnl_r=-1.0, equity_after_close=78.0)

            same_day_now = datetime(2026, 3, 10, 5, 5, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=same_day_now):
                eurusd_reason = journal.cooldown_block_reason(
                    now=same_day_now,
                    symbol="EURUSD",
                    cooldown_after_loss_minutes=20,
                    cooldown_after_win_minutes=5,
                    account="Main",
                    magic=7777,
                )
                audjpy_reason = journal.cooldown_block_reason(
                    now=same_day_now,
                    symbol="AUDJPY",
                    cooldown_after_loss_minutes=20,
                    cooldown_after_win_minutes=5,
                    account="Main",
                    magic=7777,
                )

            self.assertIsNotNone(eurusd_reason)
            self.assertIn("symbol_cooldown_after_loss", str(eurusd_reason))
            self.assertIsNone(audjpy_reason)

    def test_stats_use_sydney_trading_day_boundary(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "trades.sqlite"
            journal = TradeJournal(db_path)

            request = ExecutionRequest(
                signal_id="LIVE::USDJPY::WIN::SYDNEY",
                symbol="USDJPY",
                side="BUY",
                volume=0.01,
                entry_price=150.0,
                stop_price=149.5,
                take_profit_price=151.0,
                mode="LIVE",
                setup="USDJPY_TOKYO_SCALP",
                regime="TRENDING",
                probability=0.61,
                expected_value_r=0.2,
                slippage_points=8,
                trading_enabled=True,
                account="Main",
                magic=7777,
            )
            open_ts = datetime(2026, 3, 9, 12, 50, tzinfo=timezone.utc)
            close_ts = datetime(2026, 3, 9, 13, 10, tzinfo=timezone.utc)
            stats_ts = datetime(2026, 3, 9, 13, 15, tzinfo=timezone.utc)

            with patch("src.execution.utc_now", side_effect=[open_ts, open_ts, close_ts, close_ts]):
                journal.record_execution(request, OrderResult(True, "2001", "accepted", {}), equity=82.0)
                journal.mark_closed("LIVE::USDJPY::WIN::SYDNEY", pnl_amount=1.2, pnl_r=0.6, equity_after_close=83.2)

            with patch("src.execution.utc_now", return_value=stats_ts):
                stats = journal.stats(current_equity=83.2, account="Main", magic=7777)

        self.assertEqual(stats.trades_today, 1)
        self.assertGreater(stats.daily_pnl_pct, 0.0)

    def test_closed_trades_split_across_sydney_midnight_into_different_buckets(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            journal = TradeJournal(Path(tmp_dir) / "trades.sqlite")
            request_a = ExecutionRequest(
                signal_id="LIVE::AUDJPY::ROLLOVER::A",
                symbol="AUDJPY",
                side="BUY",
                volume=0.01,
                entry_price=96.0,
                stop_price=95.6,
                take_profit_price=96.8,
                mode="LIVE",
                setup="AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                regime="TRENDING",
                probability=0.68,
                expected_value_r=0.35,
                slippage_points=8,
                trading_enabled=True,
                account="Main",
                magic=7777,
            )
            request_b = ExecutionRequest(
                signal_id="LIVE::AUDJPY::ROLLOVER::B",
                symbol="AUDJPY",
                side="BUY",
                volume=0.01,
                entry_price=96.1,
                stop_price=95.7,
                take_profit_price=96.9,
                mode="LIVE",
                setup="AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                regime="TRENDING",
                probability=0.69,
                expected_value_r=0.36,
                slippage_points=8,
                trading_enabled=True,
                account="Main",
                magic=7777,
            )
            open_ts = datetime(2026, 3, 11, 11, 50, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", side_effect=[open_ts, open_ts, open_ts, open_ts]):
                journal.record_execution(request_a, OrderResult(True, "3001", "accepted", {}), equity=100.0)
                journal.record_execution(request_b, OrderResult(True, "3002", "accepted", {}), equity=101.0)

            close_a_sydney = datetime(2026, 3, 11, 23, 30, tzinfo=SYDNEY)
            close_b_sydney = datetime(2026, 3, 12, 0, 10, tzinfo=SYDNEY)
            journal.mark_closed(
                "LIVE::AUDJPY::ROLLOVER::A",
                pnl_amount=1.0,
                pnl_r=0.5,
                equity_after_close=101.0,
                closed_at=close_a_sydney,
            )
            journal.mark_closed(
                "LIVE::AUDJPY::ROLLOVER::B",
                pnl_amount=1.5,
                pnl_r=0.75,
                equity_after_close=102.5,
                closed_at=close_b_sydney,
            )

            now_ts = datetime(2026, 3, 11, 13, 20, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=now_ts):
                stats = journal.stats(current_equity=102.5, account="Main", magic=7777)

        self.assertEqual(stats.trading_day_key, "2026-03-12")
        self.assertEqual(stats.trades_today, 1)
        self.assertEqual(stats.today_closed_trade_count, 1)
        self.assertEqual(stats.today_closed_trade_ids, ["LIVE::AUDJPY::ROLLOVER::B"])
        self.assertEqual(
            stats.today_closed_trade_times_sydney,
            [close_b_sydney.isoformat()],
        )

    def test_prior_day_realized_loss_does_not_keep_hard_stop_after_sydney_rollover_without_new_day_closes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            journal = TradeJournal(Path(tmp_dir) / "trades.sqlite")
            request = ExecutionRequest(
                signal_id="LIVE::GBPUSD::LOSS::PRIOR_DAY",
                symbol="GBPUSD",
                side="SELL",
                volume=0.01,
                entry_price=1.2800,
                stop_price=1.2820,
                take_profit_price=1.2760,
                mode="LIVE",
                setup="LONDON_BREAKOUT_RETEST",
                regime="TRENDING",
                probability=0.66,
                expected_value_r=0.30,
                slippage_points=10,
                trading_enabled=True,
                account="Main",
                magic=7777,
            )
            open_ts = datetime(2026, 3, 11, 10, 0, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", side_effect=[open_ts, open_ts]):
                journal.record_execution(request, OrderResult(True, "4001", "accepted", {}), equity=100.0)

            close_prior_day_sydney = datetime(2026, 3, 11, 22, 45, tzinfo=SYDNEY)
            journal.mark_closed(
                "LIVE::GBPUSD::LOSS::PRIOR_DAY",
                pnl_amount=-8.0,
                pnl_r=-4.0,
                equity_after_close=92.0,
                closed_at=close_prior_day_sydney,
            )

            prior_day_now = datetime(2026, 3, 11, 12, 0, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=prior_day_now):
                prior_stats = journal.stats(current_equity=92.0, account="Main", magic=7777)

            next_day_now = datetime(2026, 3, 11, 13, 30, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=next_day_now):
                next_day_stats = journal.stats(current_equity=92.0, account="Main", magic=7777)

        prior_state, _ = RiskEngine.resolve_daily_state_from_stats(prior_stats)
        next_day_state, next_day_reason = RiskEngine.resolve_daily_state_from_stats(next_day_stats)
        self.assertEqual(prior_stats.trades_today, 1)
        self.assertEqual(prior_state, "DAILY_HARD_STOP")
        self.assertEqual(next_day_stats.trades_today, 0)
        self.assertEqual(next_day_stats.today_closed_trade_count, 0)
        self.assertEqual(next_day_stats.daily_realized_pnl, 0.0)
        self.assertEqual(next_day_stats.daily_pnl_pct, 0.0)
        self.assertEqual(next_day_stats.today_closed_trade_ids, [])
        self.assertEqual(next_day_state, "DAILY_NORMAL")
        self.assertEqual(next_day_reason, "daily_governor_normal")

    def test_flat_new_sydney_day_repairs_stale_day_basis_without_today_closes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            journal = TradeJournal(Path(tmp_dir) / "trades.sqlite")
            account = "Main"
            magic = 7777
            today_key = "2026-03-12"

            with journal._connect() as connection:
                base_scope = journal._state_scope(account=account, magic=magic)
                journal._set_state(connection, f"{base_scope}daily_equity_day", today_key)
                journal._set_state(connection, f"{base_scope}day_start_equity", "103.11")
                journal._set_state(connection, f"{base_scope}day_high_equity", "103.11")
                journal._set_state(connection, f"{base_scope}soft_dd_trade_count", "3")
                journal._set_state(connection, f"{base_scope}consecutive_losses", "2")
                journal._set_state(connection, f"{base_scope}cooldown_remaining", "1")
                journal._set_state(connection, f"{base_scope}consecutive_losses_trading_day", today_key)
                connection.commit()

            now_ts = datetime(2026, 3, 11, 13, 30, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=now_ts):
                stats = journal.stats(current_equity=95.71, account=account, magic=magic)

        daily_state, daily_reason = RiskEngine.resolve_daily_state_from_stats(stats)
        self.assertEqual(stats.trading_day_key, today_key)
        self.assertEqual(stats.today_closed_trade_count, 0)
        self.assertEqual(stats.today_closed_trade_ids, [])
        self.assertEqual(stats.daily_realized_pnl, 0.0)
        self.assertEqual(stats.daily_pnl_pct, 0.0)
        self.assertAlmostEqual(stats.day_start_equity, 95.71, places=6)
        self.assertAlmostEqual(stats.day_high_equity, 95.71, places=6)
        self.assertAlmostEqual(stats.daily_dd_pct_live, 0.0, places=6)
        self.assertEqual(stats.day_start_equity_source, "flat_book_current_equity")
        self.assertEqual(stats.day_high_equity_source, "flat_book_current_equity")
        self.assertEqual(stats.reset_reason, "flat_book_day_basis_repair")
        self.assertEqual(daily_state, "DAILY_NORMAL")
        self.assertEqual(daily_reason, "daily_governor_normal")

    def test_flat_book_day_basis_repairs_once_and_does_not_drift_on_later_stats_calls(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            journal = TradeJournal(Path(tmp_dir) / "trades.sqlite")
            account = "Main"
            magic = 7777
            today_key = "2026-03-12"

            with journal._connect() as connection:
                base_scope = journal._state_scope(account=account, magic=magic)
                journal._set_state(connection, f"{base_scope}daily_equity_day", today_key)
                journal._set_state(connection, f"{base_scope}day_start_equity", "103.11")
                journal._set_state(connection, f"{base_scope}day_high_equity", "103.11")
                connection.commit()

            now_ts = datetime(2026, 3, 11, 13, 30, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=now_ts):
                repaired_stats = journal.stats(current_equity=95.71, account=account, magic=magic)
            with patch("src.execution.utc_now", return_value=now_ts):
                later_stats = journal.stats(current_equity=95.74, account=account, magic=magic)

        self.assertAlmostEqual(repaired_stats.day_start_equity, 95.71, places=6)
        self.assertAlmostEqual(repaired_stats.day_high_equity, 95.71, places=6)
        self.assertEqual(repaired_stats.day_start_equity_source, "flat_book_current_equity")
        self.assertEqual(repaired_stats.day_high_equity_source, "flat_book_current_equity")
        self.assertEqual(repaired_stats.reset_reason, "flat_book_day_basis_repair")
        self.assertAlmostEqual(later_stats.day_start_equity, 95.71, places=6)
        self.assertAlmostEqual(later_stats.day_high_equity, 95.74, places=6)
        self.assertEqual(later_stats.day_start_equity_source, "state:day_start_equity")
        self.assertEqual(later_stats.day_high_equity_source, "live_equity_max")
        self.assertEqual(later_stats.reset_reason, "")

    def test_new_sydney_day_repairs_pending_live_equity_basis_even_with_open_positions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            journal = TradeJournal(Path(tmp_dir) / "trades.sqlite")
            request = ExecutionRequest(
                signal_id="LIVE::AUDJPY::OPEN::ROLLOVER",
                symbol="AUDJPY",
                side="BUY",
                volume=0.01,
                entry_price=96.1,
                stop_price=95.7,
                take_profit_price=96.9,
                mode="LIVE",
                setup="AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                regime="TRENDING",
                probability=0.69,
                expected_value_r=0.36,
                slippage_points=8,
                trading_enabled=True,
                account="Main",
                magic=7777,
            )
            open_ts = datetime(2026, 3, 11, 12, 50, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", side_effect=[open_ts, open_ts]):
                journal.record_execution(request, OrderResult(True, "5001", "accepted", {}), equity=100.0)

            rollover_ts = datetime(2026, 3, 11, 13, 5, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=rollover_ts):
                pending_stats = journal.stats(current_equity=None, account="Main", magic=7777)

            repaired_ts = datetime(2026, 3, 11, 13, 10, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=repaired_ts):
                repaired_stats = journal.stats(current_equity=95.0, account="Main", magic=7777)

        pending_state, _ = RiskEngine.resolve_daily_state_from_stats(pending_stats)
        repaired_state, repaired_reason = RiskEngine.resolve_daily_state_from_stats(repaired_stats)
        self.assertEqual(pending_stats.trading_day_key, "2026-03-12")
        self.assertEqual(repaired_stats.trading_day_key, "2026-03-12")
        self.assertEqual(repaired_stats.today_closed_trade_count, 0)
        self.assertEqual(repaired_stats.daily_realized_pnl, 0.0)
        self.assertAlmostEqual(repaired_stats.day_start_equity, 95.0, places=6)
        self.assertAlmostEqual(repaired_stats.day_high_equity, 95.0, places=6)
        self.assertEqual(repaired_stats.day_start_equity_source, "first_live_equity_after_rollover")
        self.assertEqual(repaired_stats.day_high_equity_source, "first_live_equity_after_rollover")
        self.assertEqual(repaired_stats.reset_reason, "flat_book_day_basis_repair")
        self.assertEqual(repaired_state, "DAILY_NORMAL")
        self.assertEqual(repaired_reason, "daily_governor_normal")
        self.assertIn(pending_state, {"DAILY_NORMAL", "DAILY_CAUTION", "DAILY_DEFENSIVE", "DAILY_HARD_STOP"})

    def test_trading_day_key_normalizes_mixed_timestamp_inputs_to_same_sydney_bucket(self) -> None:
        from src.execution import trading_day_key_for_timestamp

        self.assertEqual(
            trading_day_key_for_timestamp("2026-03-11T12:30:00+00:00"),
            trading_day_key_for_timestamp(datetime(2026, 3, 11, 23, 30, tzinfo=SYDNEY)),
        )
        self.assertEqual(
            trading_day_key_for_timestamp(1773232200.0),
            trading_day_key_for_timestamp(datetime.fromtimestamp(1773232200.0, tz=timezone.utc)),
        )

    def test_daily_pnl_pct_uses_day_start_equity_not_old_historical_baseline(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            journal = TradeJournal(Path(tmp_dir) / "trades.sqlite")
            account = "Main"
            magic = 7777

            previous_day_times = [
                datetime(2026, 3, 9, 7, 0, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 7, 0, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 7, 5, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 7, 5, tzinfo=timezone.utc),
            ]
            with patch("src.execution.utc_now", side_effect=previous_day_times):
                self._record_and_close_loss(
                    journal,
                    signal_id="OLD-DAY-LOSS",
                    symbol="EURUSD",
                    account=account,
                    magic=magic,
                    equity_open=51.0,
                    equity_close=50.0,
                )

            day_reset_ts = datetime(2026, 3, 9, 14, 0, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=day_reset_ts):
                journal.reset_daily_guard(account=account, magic=magic, current_equity=100.0)

            current_day_times = [
                datetime(2026, 3, 9, 14, 10, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 14, 10, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 14, 15, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 14, 15, tzinfo=timezone.utc),
            ]
            with patch("src.execution.utc_now", side_effect=current_day_times):
                self._record_and_close_loss(
                    journal,
                    signal_id="NEW-DAY-LOSS",
                    symbol="EURUSD",
                    account=account,
                    magic=magic,
                    equity_open=100.0,
                    equity_close=98.0,
                )

            with patch("src.execution.utc_now", return_value=datetime(2026, 3, 9, 14, 20, tzinfo=timezone.utc)):
                stats = journal.stats(current_equity=98.0, account=account, magic=magic)

        self.assertAlmostEqual(stats.day_start_equity, 100.0, places=6)
        self.assertAlmostEqual(stats.daily_pnl_pct, -0.01, places=6)

    def test_loss_streak_does_not_reset_midday(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            journal = TradeJournal(Path(tmp_dir) / "trades.sqlite")
            account = "Main"
            magic = 7777
            times = [
                datetime(2026, 3, 9, 10, 0, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 10, 0, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 10, 5, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 10, 5, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 10, 10, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 10, 10, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 10, 15, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 10, 15, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 10, 20, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 10, 20, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 10, 25, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 10, 25, tzinfo=timezone.utc),
            ]
            with patch("src.execution.utc_now", side_effect=times):
                self._record_and_close_loss(journal, signal_id="LOSS-1", symbol="EURUSD", account=account, magic=magic, equity_open=80.0, equity_close=79.0)
                self._record_and_close_loss(journal, signal_id="LOSS-2", symbol="EURUSD", account=account, magic=magic, equity_open=79.0, equity_close=78.0)
                self._record_and_close_loss(journal, signal_id="LOSS-3", symbol="EURUSD", account=account, magic=magic, equity_open=78.0, equity_close=77.0)

            with patch("src.execution.utc_now", return_value=datetime(2026, 3, 9, 10, 30, tzinfo=timezone.utc)):
                stats = journal.stats(current_equity=77.0, account=account, magic=magic, symbol="EURUSD")

        self.assertEqual(stats.consecutive_losses, 3)
        self.assertEqual(stats.cooldown_trades_remaining, 2)

    def test_loss_streak_resets_on_new_sydney_trading_day(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            journal = TradeJournal(Path(tmp_dir) / "trades.sqlite")
            account = "Main"
            magic = 7777
            times = [
                datetime(2026, 3, 9, 11, 0, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 11, 0, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 11, 5, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 11, 5, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 11, 10, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 11, 10, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 11, 15, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 11, 15, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 11, 20, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 11, 20, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 11, 25, tzinfo=timezone.utc),
                datetime(2026, 3, 9, 11, 25, tzinfo=timezone.utc),
            ]
            with patch("src.execution.utc_now", side_effect=times):
                self._record_and_close_loss(journal, signal_id="LOSS-1", symbol="EURUSD", account=account, magic=magic, equity_open=80.0, equity_close=79.0)
                self._record_and_close_loss(journal, signal_id="LOSS-2", symbol="EURUSD", account=account, magic=magic, equity_open=79.0, equity_close=78.0)
                self._record_and_close_loss(journal, signal_id="LOSS-3", symbol="EURUSD", account=account, magic=magic, equity_open=78.0, equity_close=77.0)

            next_day = datetime(2026, 3, 9, 13, 15, tzinfo=timezone.utc)
            with patch("src.execution.utc_now", return_value=next_day):
                stats = journal.stats(current_equity=77.0, account=account, magic=magic, symbol="EURUSD")

            with journal._connect() as connection:
                base_scope = journal._state_scope(account=account, magic=magic)
                symbol_scope = journal._state_scope(account=account, magic=magic, symbol="EURUSD")
                base_losses = int(journal._get_state(connection, f"{base_scope}consecutive_losses", "0"))
                base_cooldown = int(journal._get_state(connection, f"{base_scope}cooldown_remaining", "0"))
                symbol_losses = int(journal._get_state(connection, f"{symbol_scope}consecutive_losses", "0"))
                symbol_cooldown = int(journal._get_state(connection, f"{symbol_scope}cooldown_remaining", "0"))

        self.assertEqual(stats.consecutive_losses, 0)
        self.assertEqual(stats.cooldown_trades_remaining, 0)
        self.assertEqual(base_losses, 0)
        self.assertEqual(base_cooldown, 0)
        self.assertEqual(symbol_losses, 0)
        self.assertEqual(symbol_cooldown, 0)

    def test_startup_new_day_clears_persisted_stale_cooldown(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            journal = TradeJournal(Path(tmp_dir) / "trades.sqlite")
            account = "Main"
            magic = 7777
            with journal._connect() as connection:
                base_scope = journal._state_scope(account=account, magic=magic)
                journal._set_state(connection, f"{base_scope}consecutive_losses", "3")
                journal._set_state(connection, f"{base_scope}cooldown_remaining", "2")
                journal._set_state(connection, f"{base_scope}consecutive_losses_trading_day", "2026-03-09")
                connection.commit()

            with patch("src.execution.utc_now", return_value=datetime(2026, 3, 9, 13, 20, tzinfo=timezone.utc)):
                stats = journal.stats(current_equity=77.0, account=account, magic=magic)

            with journal._connect() as connection:
                base_scope = journal._state_scope(account=account, magic=magic)
                base_losses = int(journal._get_state(connection, f"{base_scope}consecutive_losses", "0"))
                base_cooldown = int(journal._get_state(connection, f"{base_scope}cooldown_remaining", "0"))

        self.assertEqual(stats.consecutive_losses, 0)
        self.assertEqual(stats.cooldown_trades_remaining, 0)
        self.assertEqual(base_losses, 0)
        self.assertEqual(base_cooldown, 0)


if __name__ == "__main__":
    unittest.main()
