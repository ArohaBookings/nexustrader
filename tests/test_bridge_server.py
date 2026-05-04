from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import time
import sqlite3
import unittest

from src.bridge_server import (
    BridgeActionQueue,
    OrchestratorPolicy,
    _bridge_account_scaling_summary,
    _counts_toward_stale_family_archive,
    _delivery_exec_budget_usd,
    _effective_bridge_drawdown_guard_pct,
    _estimated_margin_per_lot,
    _queue_row_delivery_protected,
    _recent_strategy_edge_snapshot,
    _rewrite_live_management_decision,
    _scalp_delivery_entry_guard_reason,
    _spread_spike_guard_reason,
    _spread_to_stop_guard_reason,
    _targeted_live_loss_bucket_block_reason,
    create_bridge_app,
    runtime_entry_block_state,
)
from src.bridge_stop_validation import SymbolRule, StopValidationResult
from src.execution import ExecutionRequest, TradeJournal
from src.mt5_client import OrderResult
from src.online_learning import OnlineLearningEngine
from src.risk_engine import TradeStats
from src.strategy_optimizer import StrategyOptimizer
from src.trade_management import RetracementManagementDecision

try:
    from fastapi.testclient import TestClient

    _HAS_FASTAPI = True
except Exception:
    TestClient = None  # type: ignore
    _HAS_FASTAPI = False


def _action(signal_id: str, symbol: str = "XAUUSD") -> dict:
    return {
        "signal_id": signal_id,
        "action_type": "OPEN_MARKET",
        "symbol": symbol,
        "side": "BUY",
        "lot": 0.01,
        "sl": 2190.0,
        "tp": 2210.0,
        "max_slippage_points": 20,
        "setup": "XAUUSD_M5_GRID_SCALPER_START",
        "timeframe": "M5",
        "entry_price": 2200.0,
        "reason": "test",
        "probability": 0.90,
        "expected_value_r": 1.0,
        "confluence_score": 4.0,
        "news_status": "clear",
        "expiry_utc": "2030-03-25T00:00:00+00:00",
    }


WEEKDAY_TOKYO = datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc)
WEEKDAY_LONDON = datetime(2026, 3, 23, 9, 30, tzinfo=timezone.utc)
WEEKDAY_OVERLAP = datetime(2026, 3, 23, 14, 30, tzinfo=timezone.utc)


def test_bridge_account_scaling_fallback_tunes_small_funded_balance() -> None:
    scaling = _bridge_account_scaling_summary(
        active_runtime={},
        active_snapshot={
            "balance": 92.0,
            "equity": 99.31,
            "free_margin": 99.31,
            "updated_at": "2026-05-03T22:20:05+00:00",
        },
        rollout_stats={"trade_count": 0, "overall": {"win_rate": 0.0, "expectancy_r": 0.0}},
    )

    assert scaling["current_phase"] == "PHASE_1"
    assert scaling["phase_reason"] == "equity_below_growth_threshold"
    assert scaling["current_risk_pct"] == 0.01
    assert scaling["current_max_risk_pct"] == 0.01
    assert scaling["new_risk_budget"] == 0.99
    assert scaling["equity_band"] == "bootstrap_balanced"


def test_bridge_account_scaling_fallback_requires_performance_for_phase_two() -> None:
    unproven = _bridge_account_scaling_summary(
        active_runtime={},
        active_snapshot={"balance": 120.0, "equity": 120.0, "free_margin": 120.0},
        rollout_stats={"trade_count": 3, "overall": {"win_rate": 0.25, "expectancy_r": -0.10}},
    )
    proven = _bridge_account_scaling_summary(
        active_runtime={},
        active_snapshot={"balance": 120.0, "equity": 120.0, "free_margin": 120.0},
        rollout_stats={"trade_count": 12, "overall": {"win_rate": 0.50, "expectancy_r": 0.05}},
    )

    assert unproven["current_phase"] == "PHASE_1"
    assert unproven["phase_reason"] == "equity_up_but_performance_unproven"
    assert proven["current_phase"] == "PHASE_2"
    assert proven["current_max_risk_pct"] == 0.025


def _record_live_execution(
    journal: TradeJournal,
    *,
    signal_id: str,
    account: str,
    magic: int,
    symbol: str = "BTCUSD",
    side: str = "BUY",
    setup: str = "BTC_TREND_SCALP",
    opened_at: datetime,
    equity: float = 80.0,
) -> None:
    request = ExecutionRequest(
        signal_id=signal_id,
        symbol=symbol,
        side=side,
        volume=0.01,
        entry_price=68000.0,
        stop_price=67900.0 if side == "BUY" else 68100.0,
        take_profit_price=68200.0 if side == "BUY" else 67800.0,
        mode="LIVE",
        setup=setup,
        regime="TRENDING",
        probability=0.9,
        expected_value_r=0.5,
        slippage_points=20,
        trading_enabled=True,
        account=account,
        magic=magic,
        timeframe="M15",
        proof_trade=False,
        news_status="clear",
    )
    with patch("src.execution.utc_now", return_value=opened_at):
        journal.record_execution(request, OrderResult(accepted=True, order_id=f"TKT-{signal_id}"), equity)


class BridgeQueueTests(unittest.TestCase):
    def test_scalp_delivery_entry_guard_reanchors_adverse_live_drift(self) -> None:
        reason, details = _scalp_delivery_entry_guard_reason(
            setup="BTC_MOMENTUM_CONTINUATION",
            setup_type="scalp",
            symbol_key="BTCUSD",
            side="BUY",
            reference_entry_price=66000.0,
            live_entry_price=66002.0,
            stop_price=65960.0,
            tp_price=66030.0,
            spread_points=1600.0,
            typical_spread_points=1400.0,
            point_size=0.01,
            reanchor_drift_stop_ratio=0.03,
            block_drift_stop_ratio=0.07,
            reanchor_spread_mult=0.75,
            block_spread_mult=1.45,
            spread_to_target_ratio_cap=0.30,
        )

        self.assertEqual(reason, "scalp_entry_reanchor")
        self.assertGreater(float(details["adverse_entry_drift_points"]), 0.0)

    def test_scalp_delivery_entry_guard_blocks_when_spread_eats_target(self) -> None:
        reason, details = _scalp_delivery_entry_guard_reason(
            setup="XAUUSD_M5_GRID_SCALPER_START",
            setup_type="grid_manage",
            symbol_key="XAUUSD",
            side="BUY",
            reference_entry_price=4400.0,
            live_entry_price=4400.08,
            stop_price=4398.2,
            tp_price=4400.28,
            spread_points=22.0,
            typical_spread_points=15.0,
            point_size=0.01,
            reanchor_drift_stop_ratio=0.03,
            block_drift_stop_ratio=0.07,
            reanchor_spread_mult=0.75,
            block_spread_mult=1.45,
            spread_to_target_ratio_cap=0.30,
        )

        self.assertEqual(reason, "scalp_spread_eats_target")
        self.assertGreater(float(details["spread_to_target_ratio"]), 0.30)

    def test_bridge_queue_connect_falls_back_when_wal_cannot_be_enabled(self) -> None:
        class FakeConnection:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def execute(self, sql: str):
                self.calls.append(sql)
                if sql == "PRAGMA journal_mode=WAL":
                    raise sqlite3.OperationalError("database or disk is full")
                return self

        queue = object.__new__(BridgeActionQueue)
        queue.db_path = Path("/tmp/nonexistent-bridge.sqlite")
        queue.logger = None
        connection = FakeConnection()

        with patch("src.bridge_server.sqlite3.connect", return_value=connection):
            resolved = BridgeActionQueue._connect(queue)

        self.assertIs(resolved, connection)
        self.assertIn("PRAGMA busy_timeout=30000", connection.calls)
        self.assertIn("PRAGMA journal_mode=WAL", connection.calls)
        self.assertIn("PRAGMA journal_mode=DELETE", connection.calls)

    def test_recent_strategy_edge_snapshot_marks_hot_recent_lane(self) -> None:
        snapshot = _recent_strategy_edge_snapshot(
            [
                {"pnl_amount": 1.4, "pnl_r": 0.18},
                {"pnl_amount": 0.9, "pnl_r": 0.11},
                {"pnl_amount": 1.1, "pnl_r": 0.14},
                {"pnl_amount": -0.4, "pnl_r": -0.05},
                {"pnl_amount": 1.0, "pnl_r": 0.12},
                {"pnl_amount": 0.8, "pnl_r": 0.09},
            ]
        )

        self.assertTrue(bool(snapshot.get("hot")))
        self.assertFalse(bool(snapshot.get("degraded")))

    def test_recent_strategy_edge_snapshot_degrades_small_live_loss_cluster_earlier(self) -> None:
        snapshot = _recent_strategy_edge_snapshot(
            [
                {"pnl_amount": -1.1, "pnl_r": -0.09},
                {"pnl_amount": -0.8, "pnl_r": -0.05},
                {"pnl_amount": 0.2, "pnl_r": 0.01},
                {"pnl_amount": -0.6, "pnl_r": -0.04},
            ]
        )

        self.assertTrue(bool(snapshot.get("degraded")))
        self.assertFalse(bool(snapshot.get("hot")))
        self.assertLess(float(snapshot.get("performance_score") or 0.0), 0.35)

    def test_recent_strategy_edge_snapshot_marks_degraded_lane(self) -> None:
        snapshot = _recent_strategy_edge_snapshot(
            [
                {"pnl_amount": -2.3, "pnl_r": -0.34},
                {"pnl_amount": -1.8, "pnl_r": -0.28},
                {"pnl_amount": 0.4, "pnl_r": 0.05},
                {"pnl_amount": -1.2, "pnl_r": -0.17},
                {"pnl_amount": -0.9, "pnl_r": -0.13},
                {"pnl_amount": -1.0, "pnl_r": -0.14},
            ]
        )

        self.assertTrue(bool(snapshot.get("degraded")))
        self.assertTrue(bool(snapshot.get("extremely_degraded")))
        self.assertLess(float(snapshot.get("performance_score") or 1.0), 0.35)

    def test_targeted_live_loss_bucket_blocks_degraded_xau_grid(self) -> None:
        reason, details = _targeted_live_loss_bucket_block_reason(
            symbol_key="XAUUSD",
            setup="XAUUSD_M5_GRID_SCALPER_START",
            strategy_state="QUARANTINED",
        )

        self.assertEqual(reason, "xau_grid_degraded_live_block")
        self.assertTrue(bool(details.get("targeted_live_block")))

    def test_targeted_live_loss_bucket_blocks_quarantined_jpy_breakout(self) -> None:
        reason, details = _targeted_live_loss_bucket_block_reason(
            symbol_key="AUDJPY",
            setup="AUDJPY_ASIA_MOMENTUM_BREAKOUT",
            strategy_state="QUARANTINED",
        )

        self.assertEqual(reason, "breakout_quarantined_live_block")
        self.assertEqual(str(details.get("targeted_live_block_strategy_state") or ""), "QUARANTINED")

    def test_targeted_live_loss_bucket_does_not_block_attack_lane(self) -> None:
        reason, details = _targeted_live_loss_bucket_block_reason(
            symbol_key="XAUUSD",
            setup="XAUUSD_M5_GRID_SCALPER_START",
            strategy_state="ATTACK",
        )

        self.assertIsNone(reason)
        self.assertEqual(details, {})

    def test_transient_rejects_do_not_count_toward_stale_family_archive(self) -> None:
        self.assertFalse(_counts_toward_stale_family_archive("news_caution_provider_unavailable"))
        self.assertFalse(_counts_toward_stale_family_archive("idea_cooldown_active"))
        self.assertFalse(_counts_toward_stale_family_archive("stale_idea_family_archived"))
        self.assertTrue(_counts_toward_stale_family_archive("risk_budget_exceeded"))

    def test_fresh_queued_row_is_protected_from_stale_archive(self) -> None:
        now = datetime(2026, 3, 9, 8, 0, tzinfo=timezone.utc)
        row = {
            "status": "QUEUED",
            "created_at": "2026-03-09T07:59:45+00:00",
            "updated_at": "2026-03-09T07:59:45+00:00",
            "lease_until_utc": None,
        }

        self.assertTrue(_queue_row_delivery_protected(row, now_ts=now, fair_ttl_seconds=30))

    def test_active_delivered_lease_is_protected_from_stale_archive(self) -> None:
        now = datetime(2026, 3, 9, 8, 0, tzinfo=timezone.utc)
        row = {
            "status": "DELIVERED",
            "created_at": "2026-03-09T07:59:30+00:00",
            "updated_at": "2026-03-09T07:59:40+00:00",
            "lease_until_utc": "2026-03-09T08:00:05+00:00",
        }

        self.assertTrue(_queue_row_delivery_protected(row, now_ts=now, fair_ttl_seconds=30))

    def test_delivery_budget_uses_approved_validation_cap(self) -> None:
        policy = OrchestratorPolicy(
            base_risk_pct=0.01,
            min_budget_usd=0.5,
            max_budget_usd=10.0,
        )

        budget = _delivery_exec_budget_usd(
            current_equity=82.37,
            effective_risk_pct=0.0047,
            policy=policy,
            validation_snapshot={"approved_risk_cap_usd": 4.0},
            is_canary_candidate=False,
            is_force_test_trade=False,
        )

        self.assertAlmostEqual(budget, 4.0, places=6)

    def test_delivery_budget_uses_explicit_approved_risk_cap(self) -> None:
        policy = OrchestratorPolicy(
            base_risk_pct=0.01,
            min_budget_usd=0.5,
            max_budget_usd=10.0,
        )

        budget = _delivery_exec_budget_usd(
            current_equity=82.37,
            effective_risk_pct=0.0047,
            policy=policy,
            validation_snapshot={},
            approved_risk_cap_usd=4.0,
            is_canary_candidate=False,
            is_force_test_trade=False,
        )

        self.assertAlmostEqual(budget, 4.0, places=6)

    def test_bootstrap_drawdown_guard_pct_is_more_permissive_for_small_account(self) -> None:
        policy = OrchestratorPolicy(max_daily_dd_pct=0.08, bootstrap_equity_threshold=160.0, bootstrap_drawdown_guard_pct=0.12)

        self.assertEqual(_effective_bridge_drawdown_guard_pct(policy, 60.0), 0.12)
        self.assertEqual(_effective_bridge_drawdown_guard_pct(policy, 250.0), 0.08)

    def test_estimated_margin_per_lot_uses_leverage_aware_fx_formula(self) -> None:
        fx_margin = _estimated_margin_per_lot(
            symbol_key="USDJPY",
            entry_price=148.25,
            contract_size=100000.0,
            leverage=500.0,
        )
        btc_margin = _estimated_margin_per_lot(
            symbol_key="BTCUSD",
            entry_price=68000.0,
            contract_size=1.0,
            leverage=500.0,
        )

        self.assertAlmostEqual(float(fx_margin or 0.0), 200.0, places=6)
        self.assertAlmostEqual(float(btc_margin or 0.0), 136.0, places=6)

    def test_spread_to_stop_guard_blocks_tight_fx_scalp_with_large_relative_spread(self) -> None:
        reason, details = _spread_to_stop_guard_reason(
            setup="AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
            spread_points=18.0,
            entry_price=112.269,
            stop_price=112.184,
            point_size=0.001,
            is_grid=False,
            max_ratio_grid=0.26,
            max_ratio_other=0.20,
            max_ratio_scalp=0.18,
            min_trigger_points=6.0,
        )

        self.assertEqual(reason, "spread_disorder_stop_relative")
        self.assertGreater(float(details.get("spread_to_stop_ratio") or 0.0), 0.20)

    def test_spread_to_stop_guard_allows_same_spread_when_stop_box_is_wider(self) -> None:
        reason, details = _spread_to_stop_guard_reason(
            setup="AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
            spread_points=18.0,
            entry_price=112.269,
            stop_price=111.969,
            point_size=0.001,
            is_grid=False,
            max_ratio_grid=0.26,
            max_ratio_other=0.20,
            max_ratio_scalp=0.18,
            min_trigger_points=6.0,
        )

        self.assertIsNone(reason)
        self.assertLess(float(details.get("spread_to_stop_ratio") or 0.0), 0.10)

    def test_spread_spike_guard_blocks_prime_session_spread_spike(self) -> None:
        reason, details = _spread_spike_guard_reason(
            session_name="LONDON",
            spread_points=18.0,
            spread_reference_points=10.0,
            spread_atr_points=1.0,
            is_grid=False,
            atr_guard_multiplier=1.6,
            prime_session_ratio_cap=1.25,
            asia_session_ratio_cap=2.5,
            default_session_ratio_cap=1.6,
            auto_reduce_ratio=2.0,
            auto_reduce_lot_multiplier=0.5,
            auto_skip_ratio=2.5,
        )

        self.assertEqual(reason, "spread_spike_session_ratio")
        self.assertGreater(float(details.get("spread_ratio_live") or 0.0), 1.25)

    def test_spread_spike_guard_halves_size_before_extreme_skip(self) -> None:
        reason, details = _spread_spike_guard_reason(
            session_name="TOKYO",
            spread_points=22.0,
            spread_reference_points=10.0,
            spread_atr_points=2.0,
            is_grid=False,
            atr_guard_multiplier=1.6,
            prime_session_ratio_cap=1.25,
            asia_session_ratio_cap=2.5,
            default_session_ratio_cap=1.6,
            auto_reduce_ratio=2.0,
            auto_reduce_lot_multiplier=0.5,
            auto_skip_ratio=2.5,
        )

        self.assertIsNone(reason)
        self.assertEqual(float(details.get("spread_lot_multiplier") or 0.0), 0.5)

    def test_spread_spike_guard_skips_extreme_spike(self) -> None:
        reason, details = _spread_spike_guard_reason(
            session_name="OVERLAP",
            spread_points=30.0,
            spread_reference_points=10.0,
            spread_atr_points=1.0,
            is_grid=True,
            atr_guard_multiplier=1.6,
            prime_session_ratio_cap=1.25,
            asia_session_ratio_cap=2.5,
            default_session_ratio_cap=1.6,
            auto_reduce_ratio=2.0,
            auto_reduce_lot_multiplier=0.5,
            auto_skip_ratio=2.5,
        )

        self.assertEqual(reason, "grid_spread_spike_extreme_skip")
        self.assertGreater(float(details.get("spread_ratio_live") or 0.0), 2.5)

    def test_enqueue_is_idempotent_for_active_signal(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            first = queue.enqueue(_action("sig-dup"))
            second = queue.enqueue(_action("sig-dup"))

        self.assertTrue(first)
        self.assertTrue(second)

    def test_enqueue_dedupe_key_blocks_duplicate_candidate(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            first = queue.enqueue(_action("sig-a"))
            second = queue.enqueue(_action("sig-b"))

        self.assertTrue(first)
        self.assertFalse(second)

    def test_enqueue_backfills_strategy_identity_into_context_json(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            result = queue.enqueue_with_result(
                {
                    **_action("sig-strategy-context", symbol="USDJPY"),
                    "setup": "USDJPY_VWAP_TREND_CONTINUATION",
                    "timeframe": "M15",
                    "strategy_id": "USDJPY_VWAP_TREND_CONTINUATION",
                    "strategy_state": "ATTACK",
                    "strategy_score": 0.91,
                    "strategy_recent_performance": 0.64,
                    "entry_timing_score": 0.83,
                    "structure_cleanliness_score": 0.79,
                    "execution_quality_fit": 0.88,
                    "regime_fit": 0.85,
                    "session_fit": 0.90,
                    "volatility_fit": 0.76,
                    "pair_behavior_fit": 0.82,
                    "market_data_source": "mt5+yahoo+twelve",
                    "market_data_consensus_state": "ALIGNED",
                    "session_name": "OVERLAP",
                    "regime_state": "TRENDING",
                }
            )
            action = queue.get_action("sig-strategy-context")

        self.assertTrue(result.accepted)
        self.assertIsNotNone(action)
        assert action is not None
        context_json = action.get("context_json") or {}
        self.assertEqual(str(context_json.get("strategy_key") or ""), "USDJPY_VWAP_TREND_CONTINUATION")
        self.assertEqual(str(context_json.get("strategy_state") or ""), "ATTACK")
        self.assertEqual(str(context_json.get("strategy_pool_winner") or ""), "USDJPY_VWAP_TREND_CONTINUATION")
        self.assertEqual(str(context_json.get("winning_strategy_reason") or ""), "active_runtime_strategy")
        self.assertEqual(str(context_json.get("session_name") or ""), "OVERLAP")
        self.assertEqual(str(context_json.get("regime_state") or ""), "TRENDING")
        self.assertEqual(str(context_json.get("market_data_source") or ""), "mt5+yahoo+twelve")
        self.assertEqual(str(context_json.get("market_data_consensus_state") or ""), "ALIGNED")
        self.assertGreater(float(context_json.get("strategy_score") or 0.0), 0.0)
        self.assertGreater(float(context_json.get("entry_timing_score") or 0.0), 0.0)
        self.assertGreater(float(context_json.get("structure_cleanliness_score") or 0.0), 0.0)

    def test_enqueue_overwrites_blank_strategy_placeholders(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            result = queue.enqueue_with_result(
                {
                    **_action("sig-strategy-blank", symbol="XAUUSD"),
                    "setup": "XAUUSD_M5_GRID_SCALPER_START",
                    "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                    "strategy_state": "ATTACK",
                    "strategy_score": 1.11,
                    "entry_timing_score": 0.77,
                    "structure_cleanliness_score": 0.81,
                    "execution_quality_fit": 0.92,
                    "context_json": {
                        "strategy_key": "",
                        "strategy_state": "",
                        "strategy_pool": [],
                        "strategy_pool_ranking": [],
                        "strategy_pool_winner": "",
                        "winning_strategy_reason": "",
                        "entry_timing_score": "",
                        "structure_cleanliness_score": "",
                        "execution_quality_fit": "",
                    },
                }
            )
            action = queue.get_action("sig-strategy-blank")

        self.assertTrue(result.accepted)
        self.assertIsNotNone(action)
        assert action is not None
        context_json = action.get("context_json") or {}
        self.assertEqual(str(context_json.get("strategy_key") or ""), "XAUUSD_ADAPTIVE_M5_GRID")
        self.assertEqual(str(context_json.get("strategy_state") or ""), "ATTACK")
        self.assertEqual(str(context_json.get("strategy_pool_winner") or ""), "XAUUSD_ADAPTIVE_M5_GRID")
        self.assertEqual(str(context_json.get("winning_strategy_reason") or ""), "active_runtime_strategy")
        self.assertEqual(list(context_json.get("strategy_pool") or []), ["XAUUSD_ADAPTIVE_M5_GRID"])
        self.assertEqual(len(context_json.get("strategy_pool_ranking") or []), 1)
        self.assertAlmostEqual(float(context_json.get("strategy_score") or 0.0), 1.11, places=6)
        self.assertAlmostEqual(float(context_json.get("entry_timing_score") or 0.0), 0.77, places=6)
        self.assertAlmostEqual(float(context_json.get("structure_cleanliness_score") or 0.0), 0.81, places=6)
        self.assertAlmostEqual(float(context_json.get("execution_quality_fit") or 0.0), 0.92, places=6)
        self.assertGreater(float(context_json.get("execution_quality_fit") or 0.0), 0.0)
        self.assertEqual(str(action.get("strategy_key") or ""), "XAUUSD_ADAPTIVE_M5_GRID")

    def test_grid_add_allows_new_bucket_only_after_step(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10, open_enqueue_cooldown_seconds=0)
            first = queue.enqueue(
                {
                    **_action("sig-grid-1"),
                    "setup": "XAUUSD_M5_GRID_SCALPER_ADD",
                    "entry_price": 2200.00,
                    "grid_step_points_hint": 35,
                    "grid_level": 2,
                }
            )
            blocked_same_bucket = queue.enqueue(
                {
                    **_action("sig-grid-2"),
                    "setup": "XAUUSD_M5_GRID_SCALPER_ADD",
                    "entry_price": 2200.20,
                    "grid_step_points_hint": 35,
                    "grid_level": 2,
                }
            )
            allowed_new_bucket = queue.enqueue(
                {
                    **_action("sig-grid-3"),
                    "setup": "XAUUSD_M5_GRID_SCALPER_ADD",
                    "entry_price": 2200.45,
                    "grid_step_points_hint": 35,
                    "grid_level": 3,
                }
            )

        self.assertTrue(first)
        self.assertFalse(blocked_same_bucket)
        self.assertTrue(allowed_new_bucket)

    def test_pull_expired_action_is_filtered(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=1)
            queue.enqueue(
                {
                    **_action("sig-expire"),
                    "expiry_utc": (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
                }
            )
            time.sleep(1.2)
            queue.expire_old()
            pulled_after_expiry = queue.pull(symbol="XAUUSD", account="1", magic=1)
            row = queue.get_action("sig-expire")

        self.assertEqual(pulled_after_expiry, [])
        self.assertIsNotNone(row)
        self.assertEqual(str(row["status"]).upper(), "EXPIRED")

    def test_expire_old_preserves_delivered_open_market_for_stale_handler(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=1)
            queue.enqueue(
                {
                    **_action("sig-delivered-proof"),
                    "expiry_utc": (datetime.now(timezone.utc) + timedelta(seconds=1)).isoformat(),
                }
            )
            pulled = queue.pull(symbol="XAUUSD", account="1", magic=1)
            self.assertEqual(len(pulled), 1)
            time.sleep(1.2)
            queue.expire_old()
            row = queue.get_action("sig-delivered-proof")

        self.assertIsNotNone(row)
        self.assertEqual(str(row["status"]).upper(), "DELIVERED")

    def test_cancel_stale_delivered_retries_after_temporary_lock(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            queue.enqueue(_action("sig-stale"))
            pulled = queue.pull(symbol="XAUUSD", account="Main", magic=7777)
            self.assertEqual(len(pulled), 1)
            stale_delivered_at = "2026-03-09T07:00:00+00:00"
            with queue._connect() as connection:
                connection.execute(
                    "UPDATE bridge_actions SET delivered_at = ?, updated_at = ?, expiry_utc = ? WHERE signal_id = ?",
                    (stale_delivered_at, stale_delivered_at, stale_delivered_at, "sig-stale"),
                )
                connection.commit()

            real_connect = queue._connect
            attempts = {"count": 0}

            def flaky_connect(*args, **kwargs):
                if attempts["count"] == 0:
                    attempts["count"] += 1
                    raise sqlite3.OperationalError("database is locked")
                return real_connect(*args, **kwargs)

            with patch.object(queue, "_connect", side_effect=flaky_connect):
                with patch("src.bridge_server.time.sleep", return_value=None):
                    cancelled = queue.cancel_stale_delivered(max_age_seconds=5, retry_limit=0)

            row = queue.get_action("sig-stale")

        self.assertEqual(cancelled, 1)
        self.assertIsNotNone(row)
        self.assertEqual(str(row["status"]).upper(), "CANCELLED")

    def test_cancel_stale_delivered_waits_for_expiry_window(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            queue.enqueue(_action("sig-not-expired"))
            pulled = queue.pull(symbol="XAUUSD", account="Main", magic=7777)
            self.assertEqual(len(pulled), 1)
            stale_delivered_at = "2026-03-09T07:00:00+00:00"
            future_expiry = "2099-03-09T07:00:00+00:00"
            with queue._connect() as connection:
                connection.execute(
                    "UPDATE bridge_actions SET delivered_at = ?, updated_at = ?, expiry_utc = ? WHERE signal_id = ?",
                    (stale_delivered_at, stale_delivered_at, future_expiry, "sig-not-expired"),
                )
                connection.commit()

            cancelled = queue.cancel_stale_delivered(max_age_seconds=5, retry_limit=0)
            row = queue.get_action("sig-not-expired")

        self.assertEqual(cancelled, 0)
        self.assertIsNotNone(row)
        self.assertEqual(str(row["status"]).upper(), "DELIVERED")

    def test_cancel_stale_delivered_requeues_missing_execution_report_with_retry_budget(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            queue.enqueue(_action("sig-requeue"))
            pulled = queue.pull(symbol="XAUUSD", account="Main", magic=7777)
            self.assertEqual(len(pulled), 1)
            stale_delivered_at = "2026-03-09T07:00:00+00:00"
            with queue._connect() as connection:
                connection.execute(
                    """
                    UPDATE bridge_actions
                    SET delivered_at = ?, updated_at = ?, expiry_utc = ?, pull_count = 1
                    WHERE signal_id = ?
                    """,
                    (stale_delivered_at, stale_delivered_at, stale_delivered_at, "sig-requeue"),
                )
                connection.commit()

            updated = queue.cancel_stale_delivered(
                max_age_seconds=5,
                retry_limit=3,
                requeue_grace_seconds=0,
            )
            row = queue.get_action("sig-requeue")

        self.assertEqual(updated, 1)
        self.assertIsNotNone(row)
        self.assertEqual(str(row["status"]).upper(), "QUEUED")
        self.assertEqual(str(row["last_error"]).lower(), "delivery_report_timeout_requeued")
        self.assertFalse(bool(row["delivered_at"]))

    def test_cancel_stale_delivered_cancels_stale_management_action(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            queue.enqueue(
                {
                    **_action("sig-modify-timeout"),
                    "action_type": "MODIFY_SLTP",
                    "target_ticket": "12345",
                    "ticket": "12345",
                    "sl": 2199.0,
                    "tp": 2202.0,
                }
            )
            pulled = queue.pull(symbol="XAUUSD", account="Main", magic=7777)
            self.assertEqual(len(pulled), 1)
            stale_delivered_at = "2026-03-09T07:00:00+00:00"
            with queue._connect() as connection:
                connection.execute(
                    """
                    UPDATE bridge_actions
                    SET delivered_at = ?, updated_at = ?, expiry_utc = ?
                    WHERE signal_id = ?
                    """,
                    (stale_delivered_at, stale_delivered_at, stale_delivered_at, "sig-modify-timeout"),
                )
                connection.commit()

            updated = queue.cancel_stale_delivered(max_age_seconds=5, retry_limit=0)
            row = queue.get_action("sig-modify-timeout")

        self.assertEqual(updated, 1)
        self.assertIsNotNone(row)
        self.assertEqual(str(row["status"]).upper(), "CANCELLED")
        self.assertEqual(str(row["last_error"]).lower(), "delivery_report_timeout")

    def test_pull_roundtrip_keeps_action_type(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            queue.enqueue(
                {
                    **_action("sig-close-all"),
                    "action_type": "CLOSE_ALL",
                    "target_ticket": "",
                    "side": "BUY",
                    "lot": 0.01,
                }
            )
            pulled = queue.pull(symbol="XAUUSD", account="1", magic=1)

        self.assertEqual(len(pulled), 1)
        self.assertEqual(pulled[0].get("action"), "CLOSE_ALL")
        self.assertEqual(pulled[0].get("action_type"), "CLOSE_ALL")

    def test_close_position_payload_mirrors_ticket_and_target_ticket(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            queue.enqueue(
                {
                    **_action("sig-close-ticket-only", symbol="BTCUSD"),
                    "action_type": "CLOSE_POSITION",
                    "ticket": "90001",
                    "target_ticket": "",
                    "dedupe_key": "close-ticket-only",
                    "timeframe": "M15",
                    "setup": "BTCUSD_M15_WEEKEND_BREAKOUT",
                }
            )
            queue.enqueue(
                {
                    **_action("sig-close-target-only", symbol="BTCUSD"),
                    "action_type": "CLOSE_POSITION",
                    "ticket": "",
                    "target_ticket": "90002",
                    "dedupe_key": "close-target-only",
                    "timeframe": "M15",
                    "setup": "BTCUSD_M15_WEEKEND_BREAKOUT",
                    "entry_price": 2201.0,
                }
            )
            queue.enqueue(
                {
                    **_action("sig-close-both", symbol="BTCUSD"),
                    "action_type": "CLOSE_POSITION",
                    "ticket": "90003",
                    "target_ticket": "90003",
                    "dedupe_key": "close-both",
                    "timeframe": "M15",
                    "setup": "BTCUSD_M15_WEEKEND_BREAKOUT",
                    "entry_price": 2202.0,
                }
            )
            pulled = (
                queue.pull(symbol="BTCUSD", account="TKT", magic=77)
                + queue.pull(symbol="BTCUSD", account="TKT", magic=77)
                + queue.pull(symbol="BTCUSD", account="TKT", magic=77)
            )

        by_signal = {str(item.get("signal_id")): item for item in pulled}
        self.assertEqual(str(by_signal["sig-close-ticket-only"].get("ticket")), "90001")
        self.assertEqual(str(by_signal["sig-close-ticket-only"].get("target_ticket")), "90001")
        self.assertEqual(bool(by_signal["sig-close-ticket-only"].get("payload_contract_ok")), True)
        self.assertEqual(str(by_signal["sig-close-target-only"].get("ticket")), "90002")
        self.assertEqual(str(by_signal["sig-close-target-only"].get("target_ticket")), "90002")
        self.assertEqual(bool(by_signal["sig-close-target-only"].get("payload_contract_ok")), True)
        self.assertEqual(str(by_signal["sig-close-both"].get("ticket")), "90003")
        self.assertEqual(str(by_signal["sig-close-both"].get("target_ticket")), "90003")
        self.assertEqual(bool(by_signal["sig-close-both"].get("payload_contract_ok")), True)

    def test_modify_and_partial_payloads_mirror_ticket_and_target_ticket(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            queue.enqueue(
                {
                    **_action("sig-modify-target-only", symbol="BTCUSD"),
                    "action_type": "MODIFY_SLTP",
                    "ticket": "",
                    "target_ticket": "91001",
                    "dedupe_key": "modify-target-only",
                    "timeframe": "M15",
                    "setup": "BTCUSD_M15_WEEKEND_BREAKOUT",
                }
            )
            queue.enqueue(
                {
                    **_action("sig-partial-ticket-only", symbol="BTCUSD"),
                    "action_type": "CLOSE_PARTIAL",
                    "ticket": "91002",
                    "target_ticket": "",
                    "close_lot": 0.01,
                    "lot": 0.0,
                    "dedupe_key": "partial-ticket-only",
                    "timeframe": "M15",
                    "setup": "BTCUSD_M15_WEEKEND_BREAKOUT",
                    "entry_price": 2201.0,
                }
            )
            pulled = (
                queue.pull(symbol="BTCUSD", account="MGMT", magic=78)
                + queue.pull(symbol="BTCUSD", account="MGMT", magic=78)
            )

        by_signal = {str(item.get("signal_id")): item for item in pulled}
        self.assertEqual(str(by_signal["sig-modify-target-only"].get("ticket")), "91001")
        self.assertEqual(str(by_signal["sig-modify-target-only"].get("target_ticket")), "91001")
        self.assertEqual(bool(by_signal["sig-modify-target-only"].get("payload_contract_ok")), True)
        self.assertEqual(str(by_signal["sig-partial-ticket-only"].get("ticket")), "91002")
        self.assertEqual(str(by_signal["sig-partial-ticket-only"].get("target_ticket")), "91002")
        self.assertEqual(float(by_signal["sig-partial-ticket-only"].get("close_lot") or 0.0), 0.01)
        self.assertEqual(float(by_signal["sig-partial-ticket-only"].get("volume") or 0.0), 0.01)
        self.assertEqual(bool(by_signal["sig-partial-ticket-only"].get("payload_contract_ok")), True)

    def test_legacy_management_rows_are_normalized_on_init(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "bridge.sqlite"
            queue = BridgeActionQueue(db_path=db_path, ttl_seconds=10)
            with queue._connect() as connection:
                connection.execute(
                    """
                    UPDATE bridge_actions
                    SET ticket = '',
                        target_ticket = '99001'
                    WHERE signal_id = ?
                    """,
                    ("missing",),
                )
                connection.execute(
                    """
                    INSERT INTO bridge_actions (
                        signal_id, action_type, ticket, target_ticket, symbol, symbol_key, side, lot, sl, tp,
                        max_slippage_points, status, created_at, updated_at, expiry_utc, reason, ai_summary, trailing_json,
                        breakeven_json, partials_json, mode, setup, regime, probability, expected_value_r, news_status,
                        final_decision_json, entry_price, dedupe_key, tf, validation_json, last_error, confluence_score,
                        action_score, risk_cost, action
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "legacy-close",
                        "CLOSE_POSITION",
                        "",
                        "99001",
                        "BTCUSD",
                        "BTCUSD",
                        "BUY",
                        0.01,
                        67998.0,
                        68002.0,
                        120,
                        "QUEUED",
                        datetime.now(timezone.utc).isoformat(),
                        datetime.now(timezone.utc).isoformat(),
                        datetime(2099, 1, 1, tzinfo=timezone.utc).isoformat(),
                        "legacy_row",
                        "{}",
                        "{}",
                        "{}",
                        "[]",
                        "LIVE",
                        "BTCUSD_M15_FORCE_TEST_CLOSE",
                        "FORCE_TEST",
                        0.6,
                        0.2,
                        "clear",
                        "{}",
                        68000.0,
                        "legacy-close",
                        "M15",
                        "{}",
                        "",
                        0.0,
                        0.0,
                        0.0,
                        "CLOSE_POSITION",
                    ),
                )
                connection.commit()
            normalized = BridgeActionQueue(db_path=db_path, ttl_seconds=10)
            pulled = normalized.pull(symbol="BTCUSD", account="LEGACY", magic=1)

        self.assertEqual(len(pulled), 1)
        self.assertEqual(str(pulled[0].get("ticket")), "99001")
        self.assertEqual(str(pulled[0].get("target_ticket")), "99001")
        self.assertEqual(bool(pulled[0].get("payload_contract_ok")), True)

    def test_target_account_and_magic_scope_delivery(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            queue.enqueue(
                {
                    **_action("sig-scoped", symbol="BTCUSD"),
                    "timeframe": "M15",
                    "setup": "BTCUSD_M15_WEEKEND_BREAKOUT",
                    "target_account": "ACC1",
                    "target_magic": 111,
                }
            )
            blocked = queue.pull(symbol="BTCUSD", account="ACC2", magic=222)
            allowed = queue.pull(symbol="BTCUSD", account="ACC1", magic=111)

        self.assertEqual(blocked, [])
        self.assertEqual(len(allowed), 1)
        self.assertEqual(str(allowed[0].get("signal_id")), "sig-scoped")

    def test_hourly_delivered_count_includes_expired_delivered_rows(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=1)
            queue.enqueue({**_action("sig-btc-delivered", symbol="BTCUSD"), "setup": "BTC_MULTI", "timeframe": "M15"})
            pulled = queue.pull(symbol="BTCUSD", account="ACCX", magic=77)
            self.assertEqual(len(pulled), 1)
            time.sleep(1.2)
            queue.expire_old()
            delivered_count = queue.hourly_delivered_count(account="ACCX", magic=77, symbol_key="BTCUSD", window_minutes=15)

        self.assertEqual(delivered_count, 1)

    def test_reality_sync_confidence_downgrades_when_stale(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 6, 0, 0, tzinfo=timezone.utc)):
                state_high = queue.sync_reality_from_pull(
                    account="A1",
                    magic=1,
                    symbol_key="XAUUSD",
                    timeframe="M5",
                    open_count=1,
                    net_lots=0.01,
                    avg_entry=2200.0,
                    floating_pnl=0.2,
                    stale_seconds=120,
                    low_confidence_after_seconds=300,
                    drift_open_count_threshold=1,
                )
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 6, 0, 8, tzinfo=timezone.utc)):
                state_low = queue.sync_reality_from_pull(
                    account="A1",
                    magic=1,
                    symbol_key="XAUUSD",
                    timeframe="M5",
                    open_count=None,
                    net_lots=None,
                    avg_entry=None,
                    floating_pnl=None,
                    stale_seconds=120,
                    low_confidence_after_seconds=300,
                    drift_open_count_threshold=1,
                )
        self.assertEqual(state_high.get("state_confidence"), "high")
        self.assertEqual(state_low.get("state_confidence"), "low")

    def test_xau_m15_pull_sync_clears_stale_m5_cycle_state(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=10)
            queue.update_symbol_state(
                account="A1",
                magic=1,
                symbol_key="XAUUSD",
                timeframe="M5",
                updates={
                    "open_positions_estimate": 3,
                    "grid_cycle_id": "stale-cycle",
                    "grid_side": "BUY",
                    "grid_leg_index": 3,
                    "avg_entry": 2205.0,
                    "cycle_risk_used": 4.5,
                    "cycle_mode": "START",
                    "last_entry_price": 2206.0,
                    "last_entry_time": "2026-03-01T00:00:00+00:00",
                    "last_block_reason": "grid_no_reclaim_quality",
                },
            )
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 6, 0, 0, tzinfo=timezone.utc)):
                state_m15 = queue.sync_reality_from_pull(
                    account="A1",
                    magic=1,
                    symbol_key="XAUUSD",
                    timeframe="M15",
                    open_count=0,
                    net_lots=0.0,
                    avg_entry=0.0,
                    floating_pnl=0.0,
                    stale_seconds=120,
                    low_confidence_after_seconds=300,
                    drift_open_count_threshold=1,
                )
            state_m5 = queue.get_symbol_state(account="A1", magic=1, symbol_key="XAUUSD", timeframe="M5")

        self.assertEqual(int(state_m15.get("open_positions_estimate") or 0), 0)
        self.assertEqual(int(state_m5.get("open_positions_estimate") or 0), 0)
        self.assertEqual(str(state_m5.get("grid_cycle_id") or ""), "")
        self.assertEqual(int(state_m5.get("grid_leg_index") or 0), 0)
        self.assertEqual(str(state_m5.get("cycle_mode") or ""), "IDLE")
        self.assertEqual(float(state_m5.get("last_entry_price") or 0.0), 0.0)
        self.assertEqual(str(state_m5.get("last_block_reason") or ""), "")
        self.assertEqual(str(state_m5.get("state_confidence") or ""), "high")


@unittest.skipUnless(_HAS_FASTAPI, "fastapi not installed")
class BridgeCloseReportTests(unittest.TestCase):
    def test_health_and_root_status_codes(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            health = client.get("/health")
            root_response = client.get("/")

            self.assertEqual(health.status_code, 200)
            payload = health.json()
            self.assertIn("bridge_status", payload)
            self.assertIn("broker_connectivity", payload)
            self.assertIn("current_kill_state", payload)
            self.assertIn("queue_depth", payload)
            self.assertIn("open_risk_pct", payload)
            self.assertIn("current_daily_state", payload)
            self.assertIn("current_daily_state_reason", payload)
            self.assertIn("xau_grid_override_state", payload)
            self.assertIn("execution_quality_state", payload)
            self.assertIn("watchdog_state", payload)
            self.assertIn("stale_poll_warning", payload)
            self.assertIn("current_rollout_stats", payload)
            self.assertEqual(root_response.status_code, 404)

    def test_health_marks_legacy_ea_polling_as_fresh_mt5_listener(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            pull = client.get(
                "/v1/pull",
                params={
                    "symbol": "BTCUSD",
                    "account": "26919404",
                    "magic": 20260304,
                    "timeframe": "M5",
                    "balance": 100.0,
                    "equity": 101.0,
                    "free_margin": 99.0,
                },
            )
            self.assertEqual(pull.status_code, 200)

            payload = client.get("/health").json()
            broker = payload.get("broker_connectivity", {})
            self.assertEqual(broker.get("account"), "26919404")
            self.assertTrue(broker.get("ea_polling_fresh"))
            self.assertTrue(broker.get("terminal_connected"))
            self.assertTrue(broker.get("mql_trade_allowed"))
            self.assertFalse(broker.get("explicit_permission_flags"))
            self.assertEqual(broker.get("permission_source"), "legacy_ea_poll")
            self.assertIsNone(broker.get("terminal_trade_allowed"))

    def test_health_uses_risk_config_daily_thresholds(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.update_account_snapshot(
                account="Main",
                symbol="EURUSD",
                magic=7777,
                balance=61.6,
                equity=74.16,
                free_margin=58.97,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                risk_config={
                    "daily_caution_threshold_pct": 0.025,
                    "daily_defensive_threshold_pct": 0.05,
                    "daily_hard_stop_threshold_pct": 0.07,
                },
            )
            client = TestClient(app)
            with patch.object(
                journal,
                "stats",
                return_value=TradeStats(
                    trades_today=4,
                    daily_pnl_pct=-0.021,
                    daily_dd_pct_live=0.054,
                    day_start_equity=78.4,
                    day_high_equity=78.4,
                    daily_realized_pnl=-1.65,
                    trading_day_key="2026-03-11",
                    timezone_used="Australia/Sydney",
                ),
            ):
                payload = client.get("/health").json()

            self.assertEqual(payload["current_daily_state"], "DAILY_DEFENSIVE")
            self.assertEqual(payload["current_daily_state_reason"], "daily_dd_pct_live_defensive")

    def test_dashboard_requires_login_and_serves_data_after_auth(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            client = TestClient(app)

            unauth = client.get("/dashboard/data")
            self.assertEqual(unauth.status_code, 401)

            login = client.post("/dashboard/login", data={"password": "test-pass"}, follow_redirects=False)
            self.assertEqual(login.status_code, 303)

            page = client.get("/dashboard")
            payload = client.get("/dashboard/data").json()

            self.assertEqual(page.status_code, 200)
            self.assertIn("APEX Operator Console", page.text)
            self.assertIn("summary", payload)
            self.assertIn("symbols", payload)
            self.assertIn("opportunities", payload)
            self.assertIn("xau_grid", payload)
            self.assertIn("Pause Trading", page.text)
            self.assertIn("Resume Trading", page.text)
            self.assertIn("Kill Switch", page.text)

    def test_dashboard_exposes_session_priority_fields_and_asia_symbols(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.update_account_snapshot(
                account="Main",
                symbol="AUDJPY",
                magic=7777,
                balance=100.0,
                equity=101.5,
                free_margin=99.0,
                extras={"floating_pnl": 1.5},
            )
            queue.update_symbol_state(
                account="Main",
                magic=7777,
                symbol_key="AUDJPY",
                timeframe="M15",
                updates={"last_block_reason": "session_native_priority_block"},
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            client = TestClient(app)
            login = client.post("/dashboard/login", data={"password": "test-pass"}, follow_redirects=False)
            self.assertEqual(login.status_code, 303)

            page = client.get("/dashboard")
            payload = client.get("/dashboard/data").json()

        self.assertEqual(page.status_code, 200)
        self.assertIn("Session priority", page.text)
        self.assertIn("Top ranked pairs", page.text)
        self.assertIn("Asia priority", page.text)
        self.assertIn("Lane Radar", page.text)
        self.assertIn("Session Priority Summary", page.text)
        symbol_keys = {str(item.get("symbol")) for item in payload.get("symbols", [])}
        self.assertIn("AUDJPY", symbol_keys)
        self.assertIn("NZDJPY", symbol_keys)
        self.assertIn("AUDNZD", symbol_keys)
        session_diag = dict(payload.get("session_priority_diagnostics") or {})
        self.assertIn("active_session_pair_ranking", session_diag)
        self.assertIn("native_pairs_present", session_diag)
        self.assertIn("native_pairs_selected_count", session_diag)
        self.assertIn("non_native_pairs_selected_count", session_diag)
        self.assertIn("native_priority_blocks", session_diag)

    def test_dashboard_data_and_stats_use_active_snapshot_equity_without_runtime_failure(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.update_account_snapshot(
                account="Main",
                symbol="XAUUSD",
                magic=7777,
                balance=120.0,
                equity=125.5,
                free_margin=119.0,
                extras={"floating_pnl": 5.5},
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            client = TestClient(app)

            stats = client.get("/stats")
            self.assertEqual(stats.status_code, 200)
            stats_payload = stats.json()
            self.assertEqual(float((stats_payload.get("latest_account_snapshot") or {}).get("equity") or 0.0), 125.5)

            login = client.post("/dashboard/login", data={"password": "test-pass"}, follow_redirects=False)
            self.assertEqual(login.status_code, 303)
            dashboard_data = client.get("/dashboard/data")
            self.assertEqual(dashboard_data.status_code, 200)
            payload = dashboard_data.json()

        self.assertIn("summary", payload)
        self.assertEqual(float((payload.get("summary") or {}).get("equity") or 0.0), 125.5)
        session_diag = dict(payload.get("session_priority_diagnostics") or {})
        self.assertIn("exceptional_overrides_used", session_diag)
        audjpy_card = next(item for item in payload.get("symbols", []) if str(item.get("symbol")) == "AUDJPY")
        self.assertIn("session_priority_profile", audjpy_card)
        self.assertIn("session_native_pair", audjpy_card)
        self.assertIn("session_priority_multiplier", audjpy_card)
        self.assertIn("pair_priority_rank_in_session", audjpy_card)
        self.assertIn("lane_budget_share", audjpy_card)
        self.assertIn("lane_available_capacity", audjpy_card)
        self.assertIn("pair_status", audjpy_card)
        self.assertIn("pair_status_reason", audjpy_card)

    def test_dashboard_xau_panel_uses_resolved_session_priority_state(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.update_account_snapshot(
                account="Main",
                symbol="XAUUSD",
                magic=7777,
                balance=100.0,
                equity=101.0,
                free_margin=99.0,
                extras={"floating_pnl": 1.0},
            )

            def _stub_debug_symbol(*_args, **_kwargs):
                return {
                    "runtime": {
                        "lane_name": "XAU_M5_GRID",
                        "trade_quality_score": 0.86,
                        "trade_quality_band": "A",
                        "selected_trade_band": "A",
                        "current_band_target": "A+",
                        "candidate_attempts_last_15m": 3,
                        "session_allowed": True,
                    },
                    "policy": {"last_setup_family_considered": "GRID"},
                    "news_state": {"news_state": "NEWS_SAFE"},
                    "execution_state": {"pre_exec_pass": False, "pre_exec_fail_reason": ""},
                    "candidate_pipeline": {
                        "candidate_attempts_last_15m": 3,
                        "delivered_actions_last_15m": 0,
                    },
                    "market_open_status": "OPEN",
                    "session_state": {"current_session_name": "LONDON", "session_policy_current": "XAU_WEEKDAY_PRIORITY"},
                    "latest_block_reason": "no_base_candidate",
                }

            def _stub_debug_xau_grid(*_args, **_kwargs):
                return {
                    "ok": True,
                    "symbol": "XAUUSD",
                    "session_allowed": True,
                    "market_open_status": "OPEN",
                    "session_policy_current": "XAU_WEEKDAY_PRIORITY",
                    "cycle_state": "START",
                    "current_level_count": 1,
                    "active_thresholds": {"time_stop_minutes": 15},
                    "bridge_account_snapshot": {"floating_pnl": 0.0},
                }

            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            with patch("src.bridge_server.debug_symbol", side_effect=_stub_debug_symbol), patch(
                "src.bridge_server.debug_xau_grid", side_effect=_stub_debug_xau_grid
            ):
                client = TestClient(app)
                login = client.post("/dashboard/login", data={"password": "test-pass"}, follow_redirects=False)
                self.assertEqual(login.status_code, 303)
                payload = client.get("/dashboard/data").json()

        xau_card = next(item for item in payload.get("symbols", []) if str(item.get("symbol")) == "XAUUSD")
        xau_panel = dict(payload.get("xau_grid") or {})
        self.assertEqual(float(xau_panel.get("xau_session_priority") or 0.0), float(xau_card.get("session_priority_multiplier") or 0.0))
        self.assertEqual(float(xau_panel.get("xau_lane_budget_share") or 0.0), float(xau_card.get("lane_budget_share") or 0.0))
        self.assertEqual(
            float(xau_panel.get("xau_lane_available_capacity") or 0.0),
            float(xau_card.get("lane_available_capacity") or 0.0),
        )

    def test_dashboard_and_stats_fallback_strategy_identity_from_runtime(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.update_account_snapshot(
                account="Main",
                symbol="AUDJPY",
                magic=7777,
                balance=100.0,
                equity=101.0,
                free_margin=99.0,
            )

            def _runtime_metrics():
                return {
                    "AUDJPY": {
                        "last_strategy_checked": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                        "strategy_key": "",
                        "strategy_pool": [],
                        "strategy_pool_ranking": [
                            {
                                "strategy_key": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                                "strategy_score": 0.91,
                                "rank_score": 0.91,
                            },
                            {
                                "strategy_key": "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
                                "strategy_score": 0.74,
                                "rank_score": 0.74,
                            },
                        ],
                        "strategy_pool_winner": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                        "winning_strategy_reason": "better_regime_fit",
                        "strategy_score": 0.83,
                        "strategy_state": "ATTACK",
                        "trade_quality_score": 0.81,
                        "trade_quality_band": "A",
                        "session_adjusted_score": 0.86,
                        "last_setup_family_considered": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                        "lane_name": "FX_SESSION_SCALP",
                        "session_priority_profile": "TOKYO_NATIVE",
                        "lane_session_priority": "PRIMARY",
                        "session_native_pair": True,
                        "session_priority_multiplier": 1.12,
                        "pair_priority_rank_in_session": 1,
                        "lane_budget_share": 0.40,
                        "lane_available_capacity": 4.0,
                        "candidate_attempts_last_15m": 3,
                        "session_policy_current": "TOKYO_NATIVE_PRIORITY",
                    }
                }

            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                runtime_metrics_provider=_runtime_metrics,
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            client = TestClient(app)
            login = client.post("/dashboard/login", data={"password": "test-pass"}, follow_redirects=False)
            self.assertEqual(login.status_code, 303)

            stats_payload = client.get("/stats").json()
            dashboard_payload = client.get("/dashboard/data").json()

        symbol_diag = dict((stats_payload.get("symbol_diagnostics") or {}).get("AUDJPY") or {})
        audjpy_card = next(item for item in dashboard_payload.get("symbols", []) if str(item.get("symbol")) == "AUDJPY")
        self.assertEqual(str(symbol_diag.get("strategy_key")), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")
        self.assertIn("AUDJPY_TOKYO_MOMENTUM_BREAKOUT", list(symbol_diag.get("strategy_pool") or []))
        self.assertEqual(str(symbol_diag.get("strategy_pool_winner")), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")
        self.assertEqual(str(symbol_diag.get("winning_strategy_reason")), "better_regime_fit")
        self.assertEqual(len(list(symbol_diag.get("strategy_pool_ranking") or [])), 2)
        self.assertGreater(float((symbol_diag.get("strategy_pool_ranking") or [{}])[0].get("strategy_score") or 0.0), 0.0)
        self.assertEqual(str(audjpy_card.get("strategy_key")), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")
        self.assertIn("AUDJPY_TOKYO_MOMENTUM_BREAKOUT", list(audjpy_card.get("strategy_pool") or []))
        self.assertEqual(str(audjpy_card.get("strategy_pool_winner")), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")
        self.assertEqual(str(audjpy_card.get("winning_strategy_reason")), "better_regime_fit")
        self.assertEqual(len(list(audjpy_card.get("strategy_pool_ranking") or [])), 2)
        self.assertGreater(float((audjpy_card.get("strategy_pool_ranking") or [{}])[0].get("strategy_score") or 0.0), 0.0)
        self.assertEqual(str(audjpy_card.get("strategy_state")), "ATTACK")

    def test_dashboard_synthesizes_strategy_pool_snapshot_when_runtime_pool_details_are_missing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.update_account_snapshot(
                account="Main",
                symbol="AUDNZD",
                magic=7777,
                balance=100.0,
                equity=101.0,
                free_margin=99.0,
            )

            def _runtime_metrics():
                return {
                    "AUDNZD": {
                        "strategy_key": "AUDNZD_STRUCTURE_BREAK_RETEST",
                        "strategy_pool": [
                            "AUDNZD_STRUCTURE_BREAK_RETEST",
                            "AUDNZD_RANGE_ROTATION",
                            "AUDNZD_COMPRESSION_RELEASE",
                        ],
                        "strategy_pool_ranking": [],
                        "strategy_pool_winner": "",
                        "winning_strategy_reason": "",
                        "strategy_score": 0.77,
                        "strategy_state": "NORMAL",
                        "trade_quality_score": 0.74,
                        "trade_quality_band": "A-",
                        "session_adjusted_score": 0.79,
                        "last_setup_family_considered": "AUDNZD_ASIA_ROTATION_PULLBACK",
                        "lane_name": "FX_DAYTRADE",
                        "session_priority_profile": "ASIA_NATIVE",
                        "lane_session_priority": "PRIMARY",
                        "session_native_pair": True,
                        "session_priority_multiplier": 1.14,
                        "pair_priority_rank_in_session": 1,
                        "lane_budget_share": 0.20,
                        "lane_available_capacity": 2.0,
                        "candidate_attempts_last_15m": 2,
                    }
                }

            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                runtime_metrics_provider=_runtime_metrics,
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            client = TestClient(app)
            login = client.post("/dashboard/login", data={"password": "test-pass"}, follow_redirects=False)
            self.assertEqual(login.status_code, 303)
            stats_payload = client.get("/stats").json()
            dashboard_payload = client.get("/dashboard/data").json()

        symbol_diag = dict((stats_payload.get("symbol_diagnostics") or {}).get("AUDNZD") or {})
        audnzd_card = next(item for item in dashboard_payload.get("symbols", []) if str(item.get("symbol")) == "AUDNZD")
        self.assertEqual(str(symbol_diag.get("strategy_key")), "AUDNZD_STRUCTURE_BREAK_RETEST")
        self.assertEqual(str(symbol_diag.get("strategy_pool_winner")), "AUDNZD_STRUCTURE_BREAK_RETEST")
        self.assertEqual(str(symbol_diag.get("winning_strategy_reason")), "active_runtime_strategy")
        self.assertEqual(len(list(symbol_diag.get("strategy_pool_ranking") or [])), 3)
        self.assertGreater(float((symbol_diag.get("strategy_pool_ranking") or [{}])[0].get("strategy_score") or 0.0), 0.0)
        self.assertEqual(str(audnzd_card.get("strategy_pool_winner")), "AUDNZD_STRUCTURE_BREAK_RETEST")
        self.assertEqual(str(audnzd_card.get("winning_strategy_reason")), "active_runtime_strategy")
        self.assertEqual(len(list(audnzd_card.get("strategy_pool_ranking") or [])), 3)

    def test_dashboard_expands_partial_runtime_pool_and_uses_ranked_strategy_metrics(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.update_account_snapshot(
                account="Main",
                symbol="XAUUSD",
                magic=7777,
                balance=100.0,
                equity=101.0,
                free_margin=99.0,
            )

            def _runtime_metrics():
                return {
                    "XAUUSD": {
                        "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                        "strategy_pool": ["XAUUSD_ADAPTIVE_M5_GRID"],
                        "strategy_pool_ranking": [
                            {
                                "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                                "strategy_score": 0.88,
                                "strategy_state": "ATTACK",
                                "regime_state": "BREAKOUT_EXPANSION",
                                "regime_fit": 0.84,
                                "session_fit": 0.91,
                                "volatility_fit": 0.79,
                                "pair_behavior_fit": 0.83,
                                "execution_quality_fit": 0.76,
                                "entry_timing_score": 0.67,
                                "structure_cleanliness_score": 0.74,
                                "management_template": "GRID_BASKET_ADAPTIVE",
                                "lane_name": "XAU_M5_GRID",
                                "session_priority_profile": "LONDON_RISK_ON",
                                "lane_session_priority": "PRIMARY",
                                "session_native_pair": True,
                                "session_priority_multiplier": 1.16,
                                "pair_priority_rank_in_session": 1,
                                "lane_budget_share": 0.30,
                                "lane_available_capacity": 6.0,
                                "lane_capacity_usage": 0.2,
                                "exceptional_override_used": False,
                                "exceptional_override_reason": "",
                                "why_non_native_pair_won": "",
                                "why_native_pair_lost_priority": "",
                                "market_data_consensus_adjustment": 0.015,
                            }
                        ],
                        "strategy_pool_winner": "",
                        "winning_strategy_reason": "",
                        "strategy_score": "",
                        "strategy_state": "",
                        "entry_timing_score": 0.0,
                        "structure_cleanliness_score": 0.0,
                        "trade_quality_score": 0.81,
                        "trade_quality_band": "A",
                        "session_adjusted_score": 0.86,
                        "management_template": "GRID_BASKET_ADAPTIVE",
                        "regime_state": "",
                        "regime_fit": 0.0,
                        "session_fit": 0.0,
                        "volatility_fit": 0.0,
                        "pair_behavior_fit": 0.0,
                        "execution_quality_fit": 0.0,
                        "last_setup_family_considered": "XAUUSD_M5_GRID_SCALPER_START",
                        "lane_name": "XAU_M5_GRID",
                        "session_priority_profile": "LONDON_NATIVE",
                        "lane_session_priority": "PRIMARY",
                        "session_native_pair": True,
                        "session_priority_multiplier": 1.16,
                        "pair_priority_rank_in_session": 1,
                        "lane_budget_share": 0.30,
                        "lane_available_capacity": 3.0,
                        "candidate_attempts_last_15m": 2,
                        "session_stop_state": "",
                        "xau_grid_mode": "ATTACK_GRID",
                    }
                }

            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                runtime_metrics_provider=_runtime_metrics,
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            client = TestClient(app)
            login = client.post("/dashboard/login", data={"password": "test-pass"}, follow_redirects=False)
            self.assertEqual(login.status_code, 303)
            stats_payload = client.get("/stats").json()
            dashboard_payload = client.get("/dashboard/data").json()

        symbol_diag = dict((stats_payload.get("symbol_diagnostics") or {}).get("XAUUSD") or {})
        xau_card = next(item for item in dashboard_payload.get("symbols", []) if str(item.get("symbol")) == "XAUUSD")
        xau_opportunity = next(item for item in dashboard_payload.get("opportunities", []) if str(item.get("symbol")) == "XAUUSD")
        self.assertGreaterEqual(len(list(symbol_diag.get("strategy_pool") or [])), 5)
        self.assertGreaterEqual(len(list(symbol_diag.get("strategy_pool_ranking") or [])), 5)
        self.assertEqual(str(symbol_diag.get("strategy_pool_winner")), "XAUUSD_ADAPTIVE_M5_GRID")
        self.assertEqual(str(symbol_diag.get("winning_strategy_reason")), "active_runtime_strategy")
        self.assertAlmostEqual(float(symbol_diag.get("entry_timing_score") or 0.0), 0.67, places=3)
        self.assertAlmostEqual(float(symbol_diag.get("structure_cleanliness_score") or 0.0), 0.74, places=3)
        self.assertEqual(str(xau_card.get("strategy_state")), "ATTACK")
        self.assertEqual(str(xau_card.get("management_template") or ""), "GRID_BASKET_ADAPTIVE")
        self.assertGreaterEqual(len(list(xau_card.get("strategy_pool") or [])), 5)
        self.assertGreaterEqual(len(list(xau_card.get("strategy_pool_ranking") or [])), 5)
        self.assertAlmostEqual(float(xau_card.get("entry_timing_score") or 0.0), 0.67, places=3)
        self.assertAlmostEqual(float(xau_card.get("structure_cleanliness_score") or 0.0), 0.74, places=3)
        self.assertEqual(str(xau_opportunity.get("strategy_key") or ""), "XAUUSD_ADAPTIVE_M5_GRID")
        self.assertEqual(str(xau_opportunity.get("strategy_state") or ""), "ATTACK")
        self.assertEqual(str(xau_opportunity.get("management_template") or ""), "GRID_BASKET_ADAPTIVE")
        self.assertEqual(str(xau_opportunity.get("regime") or ""), "BREAKOUT_EXPANSION")
        self.assertAlmostEqual(float(xau_opportunity.get("regime_fit") or 0.0), 0.84, places=3)
        self.assertAlmostEqual(float(xau_opportunity.get("session_fit") or 0.0), 0.91, places=3)
        self.assertAlmostEqual(float(xau_opportunity.get("volatility_fit") or 0.0), 0.79, places=3)
        self.assertAlmostEqual(float(xau_opportunity.get("pair_behavior_fit") or 0.0), 0.83, places=3)
        self.assertAlmostEqual(float(xau_opportunity.get("execution_quality_fit") or 0.0), 0.76, places=3)
        self.assertEqual(str((dashboard_payload.get("xau_grid") or {}).get("xau_grid_mode") or ""), "ATTACK_GRID")
        self.assertEqual(str(xau_opportunity.get("session_stop_state") or ""), "NORMAL")

    def test_debug_symbol_resolves_top_level_strategy_fields_from_ranked_pool(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.update_account_snapshot(
                account="Main",
                symbol="XAUUSD",
                magic=7777,
                balance=100.0,
                equity=101.0,
                free_margin=99.0,
            )

            def _runtime_metrics():
                return {
                    "XAUUSD": {
                        "strategy_key": "",
                        "strategy_pool": ["XAUUSD_ADAPTIVE_M5_GRID"],
                        "strategy_pool_ranking": [
                            {
                                "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                                "strategy_score": 0.88,
                                "strategy_state": "ATTACK",
                                "regime_state": "BREAKOUT_EXPANSION",
                                "regime_fit": 0.84,
                                "session_fit": 0.91,
                                "volatility_fit": 0.79,
                                "pair_behavior_fit": 0.83,
                                "execution_quality_fit": 0.76,
                                "entry_timing_score": 0.67,
                                "structure_cleanliness_score": 0.74,
                            }
                        ],
                        "strategy_pool_winner": "",
                        "winning_strategy_reason": "",
                        "strategy_score": 0.0,
                        "strategy_state": "NORMAL",
                        "entry_timing_score": 0.0,
                        "structure_cleanliness_score": 0.0,
                        "trade_quality_score": 0.81,
                        "trade_quality_band": "A",
                        "session_adjusted_score": 0.86,
                        "regime_state": "",
                        "regime_fit": 0.0,
                        "session_fit": 0.0,
                        "volatility_fit": 0.0,
                        "pair_behavior_fit": 0.0,
                        "execution_quality_fit": 0.0,
                        "last_setup_family_considered": "XAUUSD_M5_GRID_SCALPER_START",
                        "lane_name": "XAU_M5_GRID",
                        "session_stop_state": "",
                        "runtime_market_data_source": "mt5+yahoo+twelve",
                        "runtime_market_data_consensus_state": "EXTERNAL_MULTI_PROVIDER",
                        "market_data_consensus_adjustment": 0.015,
                    }
                }

            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                runtime_metrics_provider=_runtime_metrics,
            )
            client = TestClient(app)
            payload = client.get(
                "/debug/symbol",
                params={"symbol": "XAUUSD", "account": "Main", "magic": 7777, "limit": 5},
            ).json()

        runtime = dict(payload.get("runtime") or {})
        self.assertEqual(str(payload.get("strategy_key") or ""), "XAUUSD_ADAPTIVE_M5_GRID")
        self.assertEqual(str(payload.get("strategy_pool_winner") or ""), "XAUUSD_ADAPTIVE_M5_GRID")
        self.assertEqual(str(payload.get("winning_strategy_reason") or ""), "active_runtime_strategy")
        self.assertEqual(str(payload.get("strategy_state") or ""), "ATTACK")
        self.assertAlmostEqual(float(payload.get("strategy_score") or 0.0), 0.86, places=3)
        self.assertAlmostEqual(float(payload.get("entry_timing_score") or 0.0), 0.67, places=3)
        self.assertAlmostEqual(float(payload.get("structure_cleanliness_score") or 0.0), 0.74, places=3)
        self.assertAlmostEqual(float(payload.get("regime_fit") or 0.0), 0.84, places=3)
        self.assertAlmostEqual(float(payload.get("session_fit") or 0.0), 0.91, places=3)
        self.assertAlmostEqual(float(payload.get("volatility_fit") or 0.0), 0.79, places=3)
        self.assertAlmostEqual(float(payload.get("pair_behavior_fit") or 0.0), 0.83, places=3)
        self.assertAlmostEqual(float(payload.get("execution_quality_fit") or 0.0), 0.76, places=3)
        self.assertEqual(str(payload.get("lane_name") or ""), "XAU_M5_GRID")
        self.assertEqual(str(payload.get("session_priority_profile") or ""), "LONDON_RISK_ON")
        self.assertEqual(str(payload.get("lane_session_priority") or ""), "PRIMARY")
        self.assertEqual(bool(payload.get("session_native_pair")), True)
        self.assertAlmostEqual(float(payload.get("session_priority_multiplier") or 0.0), 1.15, places=3)
        self.assertEqual(int(payload.get("pair_priority_rank_in_session") or 0), 1)
        self.assertAlmostEqual(float(payload.get("lane_budget_share") or 0.0), 0.26, places=3)
        self.assertAlmostEqual(float(payload.get("lane_available_capacity") or 0.0), 6.0, places=3)
        self.assertAlmostEqual(float(payload.get("lane_capacity_usage") or 0.0), 0.2, places=3)
        self.assertEqual(str(payload.get("session_stop_state") or ""), "NORMAL")
        self.assertEqual(str(payload.get("runtime_market_data_source") or ""), "mt5+yahoo+twelve")
        self.assertEqual(str(payload.get("runtime_market_data_consensus_state") or ""), "EXTERNAL_MULTI_PROVIDER")
        self.assertAlmostEqual(float(payload.get("market_data_consensus_adjustment") or 0.0), 0.015, places=3)
        self.assertEqual(str(runtime.get("strategy_key") or ""), "XAUUSD_ADAPTIVE_M5_GRID")
        self.assertEqual(str(runtime.get("strategy_pool_winner") or ""), "XAUUSD_ADAPTIVE_M5_GRID")
        self.assertEqual(str(runtime.get("winning_strategy_reason") or ""), "active_runtime_strategy")
        self.assertEqual(str(runtime.get("strategy_state") or ""), "ATTACK")
        self.assertAlmostEqual(float(runtime.get("strategy_score") or 0.0), 0.86, places=3)
        self.assertAlmostEqual(float(runtime.get("entry_timing_score") or 0.0), 0.67, places=3)
        self.assertAlmostEqual(float(runtime.get("structure_cleanliness_score") or 0.0), 0.74, places=3)
        self.assertAlmostEqual(float(runtime.get("regime_fit") or 0.0), 0.84, places=3)
        self.assertAlmostEqual(float(runtime.get("session_fit") or 0.0), 0.91, places=3)
        self.assertAlmostEqual(float(runtime.get("volatility_fit") or 0.0), 0.79, places=3)
        self.assertAlmostEqual(float(runtime.get("pair_behavior_fit") or 0.0), 0.83, places=3)
        self.assertAlmostEqual(float(runtime.get("execution_quality_fit") or 0.0), 0.76, places=3)
        self.assertEqual(str(runtime.get("lane_name") or ""), "XAU_M5_GRID")
        self.assertEqual(str(runtime.get("session_priority_profile") or ""), "LONDON_RISK_ON")
        self.assertEqual(str(runtime.get("lane_session_priority") or ""), "PRIMARY")
        self.assertEqual(bool(runtime.get("session_native_pair")), True)
        self.assertAlmostEqual(float(runtime.get("session_priority_multiplier") or 0.0), 1.15, places=3)
        self.assertEqual(int(runtime.get("pair_priority_rank_in_session") or 0), 1)
        self.assertAlmostEqual(float(runtime.get("lane_budget_share") or 0.0), 0.26, places=3)
        self.assertAlmostEqual(float(runtime.get("lane_available_capacity") or 0.0), 6.0, places=3)
        self.assertAlmostEqual(float(runtime.get("lane_capacity_usage") or 0.0), 0.2, places=3)
        symbol_state = dict(payload.get("symbol_state") or {})
        self.assertEqual(str(symbol_state.get("strategy_key") or ""), "XAUUSD_ADAPTIVE_M5_GRID")
        self.assertEqual(str(symbol_state.get("strategy_pool_winner") or ""), "XAUUSD_ADAPTIVE_M5_GRID")
        self.assertEqual(str(symbol_state.get("winning_strategy_reason") or ""), "active_runtime_strategy")
        self.assertEqual(str(symbol_state.get("strategy_state") or ""), "ATTACK")
        self.assertAlmostEqual(float(symbol_state.get("strategy_score") or 0.0), 0.86, places=3)
        self.assertEqual(str(symbol_state.get("session_priority_profile") or ""), "LONDON_RISK_ON")
        self.assertEqual(str(symbol_state.get("lane_session_priority") or ""), "PRIMARY")
        self.assertEqual(bool(symbol_state.get("session_native_pair")), True)
        self.assertAlmostEqual(float(symbol_state.get("session_priority_multiplier") or 0.0), 1.15, places=3)
        self.assertEqual(int(symbol_state.get("pair_priority_rank_in_session") or 0), 1)
        self.assertAlmostEqual(float(symbol_state.get("lane_budget_share") or 0.0), 0.26, places=3)
        self.assertAlmostEqual(float(symbol_state.get("lane_available_capacity") or 0.0), 6.0, places=3)
        self.assertAlmostEqual(float(symbol_state.get("lane_capacity_usage") or 0.0), 0.2, places=3)

    def test_debug_symbol_normalizes_regime_and_session_stop_from_strategy_overlay(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.update_account_snapshot(
                account="Main",
                symbol="AUDJPY",
                magic=7777,
                balance=100.0,
                equity=101.0,
                free_margin=99.0,
            )

            def _runtime_metrics():
                return {
                    "AUDJPY": {
                        "strategy_key": "",
                        "strategy_pool": ["AUDJPY_TOKYO_MOMENTUM_BREAKOUT"],
                        "strategy_pool_ranking": [
                            {
                                "strategy_key": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                                "strategy_score": 0.72,
                                "strategy_state": "NORMAL",
                                "regime_state": "LOW_LIQUIDITY_DRIFT",
                                "regime_fit": 0.40,
                                "session_fit": 0.68,
                                "volatility_fit": 0.58,
                                "pair_behavior_fit": 0.73,
                                "execution_quality_fit": 0.70,
                                "entry_timing_score": 0.61,
                                "structure_cleanliness_score": 0.57,
                            }
                        ],
                        "strategy_pool_winner": "",
                        "winning_strategy_reason": "",
                        "strategy_score": 0.0,
                        "strategy_state": "",
                        "trade_quality_score": 0.69,
                        "trade_quality_band": "B+",
                        "session_adjusted_score": 0.71,
                        "regime_state": "",
                        "session_stop_state": "",
                    }
                }

            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                runtime_metrics_provider=_runtime_metrics,
            )
            client = TestClient(app)
            payload = client.get(
                "/debug/symbol",
                params={"symbol": "AUDJPY", "account": "Main", "magic": 7777, "limit": 5},
            ).json()

        self.assertEqual(str(payload.get("strategy_key") or ""), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")
        self.assertEqual(str(payload.get("regime_state") or ""), "LOW_LIQUIDITY_CHOP")
        self.assertEqual(str(payload.get("session_stop_state") or ""), "NORMAL")
        runtime = dict(payload.get("runtime") or {})
        self.assertEqual(str(runtime.get("regime_state") or ""), "LOW_LIQUIDITY_CHOP")
        self.assertEqual(str(runtime.get("session_stop_state") or ""), "NORMAL")

    def test_debug_symbol_uses_ranked_pool_context_when_runtime_priority_fields_are_generic(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.update_account_snapshot(
                account="Main",
                symbol="GBPUSD",
                magic=7777,
                balance=100.0,
                equity=101.0,
                free_margin=99.0,
            )

            def _runtime_metrics():
                return {
                    "GBPUSD": {
                        "strategy_key": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                        "strategy_pool": [
                            "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                            "GBPUSD_STOP_HUNT_REVERSAL",
                        ],
                        "strategy_pool_ranking": [
                            {
                                "strategy_key": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                                "strategy_score": 0.81,
                                "strategy_state": "ATTACK",
                                "regime_state": "BREAKOUT_EXPANSION",
                                "regime_fit": 0.86,
                                "session_fit": 0.92,
                                "volatility_fit": 0.83,
                                "pair_behavior_fit": 0.79,
                                "execution_quality_fit": 0.78,
                                "entry_timing_score": 0.69,
                                "structure_cleanliness_score": 0.72,
                                "lane_name": "FX_DAYTRADE",
                            }
                        ],
                        "strategy_pool_winner": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                        "winning_strategy_reason": "higher_combined_strategy_score",
                        "strategy_score": 0.81,
                        "strategy_state": "ATTACK",
                        "trade_quality_score": 0.77,
                        "trade_quality_band": "A",
                        "session_adjusted_score": 0.84,
                        "regime_state": "BREAKOUT_EXPANSION",
                        "regime_fit": 0.86,
                        "session_fit": 0.92,
                        "volatility_fit": 0.83,
                        "pair_behavior_fit": 0.79,
                        "execution_quality_fit": 0.78,
                        "entry_timing_score": 0.69,
                        "structure_cleanliness_score": 0.72,
                        "last_setup_family_considered": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                        "lane_name": "FX_DAYTRADE",
                        "session_priority_profile": "GLOBAL",
                        "lane_session_priority": "NEUTRAL",
                        "session_native_pair": False,
                        "session_priority_multiplier": 1.0,
                        "pair_priority_rank_in_session": 99,
                        "lane_budget_share": 0.0,
                        "lane_available_capacity": 0.0,
                        "lane_capacity_usage": 0.0,
                        "session_policy_current": "OVERLAP",
                    }
                }

            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                runtime_metrics_provider=_runtime_metrics,
            )
            client = TestClient(app)
            payload = client.get(
                "/debug/symbol",
                params={"symbol": "GBPUSD", "account": "Main", "magic": 7777, "limit": 5},
            ).json()

        self.assertEqual(str(payload.get("strategy_key") or ""), "GBPUSD_LONDON_EXPANSION_BREAKOUT")
        self.assertEqual(str(payload.get("session_priority_profile") or ""), "NY_RISK_ON")
        self.assertEqual(str(payload.get("lane_session_priority") or ""), "PRIMARY")
        self.assertEqual(bool(payload.get("session_native_pair")), True)
        self.assertAlmostEqual(float(payload.get("session_priority_multiplier") or 0.0), 1.10, places=3)
        self.assertEqual(int(payload.get("pair_priority_rank_in_session") or 0), 2)
        self.assertAlmostEqual(float(payload.get("lane_budget_share") or 0.0), 0.27, places=3)
        symbol_state = dict(payload.get("symbol_state") or {})
        self.assertEqual(str(symbol_state.get("session_priority_profile") or ""), "NY_RISK_ON")
        self.assertEqual(str(symbol_state.get("lane_session_priority") or ""), "PRIMARY")
        self.assertEqual(bool(symbol_state.get("session_native_pair")), True)
        self.assertAlmostEqual(float(symbol_state.get("session_priority_multiplier") or 0.0), 1.10, places=3)
        self.assertEqual(int(symbol_state.get("pair_priority_rank_in_session") or 0), 2)
        self.assertAlmostEqual(float(symbol_state.get("lane_budget_share") or 0.0), 0.27, places=3)

    def test_dashboard_cards_use_resolved_overlay_when_runtime_priority_fields_are_generic(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )

            def _runtime_metrics():
                return {
                    "GBPUSD": {
                        "strategy_key": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                        "strategy_pool": [
                            "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                            "GBPUSD_STOP_HUNT_REVERSAL",
                        ],
                        "strategy_pool_ranking": [
                            {
                                "strategy_key": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                                "strategy_score": 0.84,
                                "strategy_state": "ATTACK",
                                "regime_state": "BREAKOUT_EXPANSION",
                                "regime_fit": 0.88,
                                "session_fit": 0.93,
                                "volatility_fit": 0.84,
                                "pair_behavior_fit": 0.81,
                                "execution_quality_fit": 0.79,
                                "entry_timing_score": 0.72,
                                "structure_cleanliness_score": 0.74,
                                "lane_name": "FX_DAYTRADE",
                                "session_priority_profile": "GLOBAL",
                                "lane_session_priority": "NEUTRAL",
                                "session_native_pair": False,
                                "session_priority_multiplier": 1.0,
                                "pair_priority_rank_in_session": 99,
                                "lane_budget_share": 0.0,
                                "lane_available_capacity": 0.0,
                                "lane_capacity_usage": 0.0,
                            }
                        ],
                        "strategy_pool_winner": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                        "winning_strategy_reason": "higher_combined_strategy_score",
                        "strategy_score": 0.84,
                        "strategy_state": "ATTACK",
                        "trade_quality_score": 0.79,
                        "trade_quality_band": "A",
                        "session_adjusted_score": 0.86,
                        "regime_state": "BREAKOUT_EXPANSION",
                        "regime_fit": 0.88,
                        "session_fit": 0.93,
                        "volatility_fit": 0.84,
                        "pair_behavior_fit": 0.81,
                        "execution_quality_fit": 0.79,
                        "entry_timing_score": 0.72,
                        "structure_cleanliness_score": 0.74,
                        "lane_name": "FX_DAYTRADE",
                        "session_priority_profile": "GLOBAL",
                        "lane_session_priority": "NEUTRAL",
                        "session_native_pair": False,
                        "session_priority_multiplier": 1.0,
                        "pair_priority_rank_in_session": 99,
                        "lane_budget_share": 0.0,
                        "lane_available_capacity": 0.0,
                        "lane_capacity_usage": 0.0,
                        "session_policy_current": "OVERLAP",
                    }
                }

            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                runtime_metrics_provider=_runtime_metrics,
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            client = TestClient(app)
            login = client.post("/dashboard/login", data={"password": "test-pass"}, follow_redirects=False)
            self.assertEqual(login.status_code, 303)
            payload = client.get("/dashboard/data").json()

        gbp_card = next(item for item in payload.get("symbols", []) if str(item.get("symbol")) == "GBPUSD")
        gbp_opp = next(item for item in payload.get("opportunities", []) if str(item.get("symbol")) == "GBPUSD")
        self.assertEqual(str(gbp_card.get("session_priority_profile") or ""), "NY_RISK_ON")
        self.assertEqual(str(gbp_card.get("lane_session_priority") or ""), "PRIMARY")
        self.assertEqual(bool(gbp_card.get("session_native_pair")), True)
        self.assertAlmostEqual(float(gbp_card.get("session_priority_multiplier") or 0.0), 1.10, places=3)
        self.assertEqual(int(gbp_card.get("pair_priority_rank_in_session") or 0), 2)
        self.assertAlmostEqual(float(gbp_card.get("lane_budget_share") or 0.0), 0.27, places=3)
        self.assertEqual(str(gbp_opp.get("session_priority_profile") or ""), "NY_RISK_ON")
        self.assertEqual(str(gbp_opp.get("lane_session_priority") or ""), "PRIMARY")
        self.assertEqual(bool(gbp_opp.get("session_native_pair")), True)
        self.assertAlmostEqual(float(gbp_opp.get("session_priority_multiplier") or 0.0), 1.10, places=3)
        self.assertEqual(int(gbp_opp.get("pair_priority_rank_in_session") or 0), 2)
        self.assertAlmostEqual(float(gbp_opp.get("lane_budget_share") or 0.0), 0.27, places=3)

    def test_debug_symbol_prefers_resolved_context_when_runtime_priority_is_partially_generic(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )

            def _runtime_metrics():
                return {
                    "XAUUSD": {
                        "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                        "strategy_pool": [
                            "XAUUSD_ADAPTIVE_M5_GRID",
                            "XAUUSD_VWAP_REVERSION",
                        ],
                        "strategy_pool_ranking": [
                            {
                                "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                                "strategy_score": 0.84,
                                "strategy_state": "ATTACK",
                                "regime_state": "BREAKOUT_EXPANSION",
                                "regime_fit": 0.80,
                                "session_fit": 0.90,
                                "volatility_fit": 0.82,
                                "pair_behavior_fit": 0.81,
                                "execution_quality_fit": 0.79,
                                "entry_timing_score": 0.71,
                                "structure_cleanliness_score": 0.76,
                                "lane_name": "XAU_M5_GRID",
                                "session_priority_profile": "GLOBAL",
                                "lane_session_priority": "NEUTRAL",
                                "session_native_pair": True,
                                "session_priority_multiplier": 1.16,
                                "pair_priority_rank_in_session": 1,
                                "lane_budget_share": 0.10,
                                "lane_available_capacity": 0.0,
                                "lane_capacity_usage": 0.0,
                            }
                        ],
                        "strategy_pool_winner": "XAUUSD_ADAPTIVE_M5_GRID",
                        "winning_strategy_reason": "higher_combined_strategy_score",
                        "strategy_score": 0.84,
                        "strategy_state": "ATTACK",
                        "trade_quality_score": 0.82,
                        "trade_quality_band": "A+",
                        "session_adjusted_score": 0.88,
                        "regime_state": "BREAKOUT_EXPANSION",
                        "lane_name": "XAU_M5_GRID",
                        "session_priority_profile": "GLOBAL",
                        "lane_session_priority": "NEUTRAL",
                        "session_native_pair": True,
                        "session_priority_multiplier": 1.16,
                        "pair_priority_rank_in_session": 1,
                        "lane_budget_share": 0.10,
                        "lane_available_capacity": 0.0,
                        "lane_capacity_usage": 0.0,
                        "session_policy_current": "NEW_YORK",
                    }
                }

            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                runtime_metrics_provider=_runtime_metrics,
            )
            client = TestClient(app)
            payload = client.get(
                "/debug/symbol",
                params={"symbol": "XAUUSD", "account": "Main", "magic": 7777, "limit": 5},
            ).json()

        self.assertEqual(str(payload.get("session_priority_profile") or ""), "NY_RISK_ON")
        self.assertEqual(str(payload.get("lane_session_priority") or ""), "PRIMARY")
        self.assertTrue(bool(payload.get("session_native_pair")))
        self.assertAlmostEqual(float(payload.get("session_priority_multiplier") or 0.0), 1.16, places=3)
        self.assertEqual(int(payload.get("pair_priority_rank_in_session") or 0), 1)
        self.assertAlmostEqual(float(payload.get("lane_budget_share") or 0.0), 0.26, places=3)
        self.assertGreaterEqual(float(payload.get("lane_available_capacity") or 0.0), 0.0)

    def test_health_and_stats_surface_session_and_day_aliases(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            with patch.object(
                journal,
                "stats",
                return_value=TradeStats(
                    trades_today=3,
                    daily_pnl_pct=-0.01,
                    daily_dd_pct_live=0.015,
                    day_start_equity=100.0,
                    day_high_equity=101.5,
                    daily_realized_pnl=-1.0,
                    trading_day_key="2026-03-13",
                    timezone_used="Australia/Sydney",
                ),
            ):
                health = client.get("/health").json()
                stats = client.get("/stats").json()

        self.assertIn("current_session_name", health)
        self.assertIn("trading_day_key_now", health)
        self.assertEqual(int(health.get("closed_trades_today") or 0), 3)
        self.assertEqual(int(health.get("today_closed_trade_count") or 0), 3)
        self.assertEqual(str(health.get("trading_day_key_now") or ""), "2026-03-13")
        self.assertEqual(int(stats.get("closed_trades_today") or 0), 3)
        self.assertEqual(str(stats.get("trading_day_key_now") or ""), "2026-03-13")
        self.assertIn("current_session_name", stats)

    def test_supported_asia_symbol_is_not_blocked_by_training_mode(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                execution_config={"startup_warmup_seconds": 0},
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            with TestClient(app) as client:
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "AUDJPY",
                        "account": "A1",
                        "magic": 7,
                        "timeframe": "M15",
                        "spread_points": 12,
                        "last": 96.25,
                        "equity": 100.0,
                        "free_margin": 100.0,
                    },
                )
                self.assertEqual(response.status_code, 200)
                debug = client.get(
                    "/debug/symbol",
                    params={"symbol": "AUDJPY", "account": "A1", "magic": 7, "timeframe": "M15", "limit": 5},
                ).json()

        self.assertNotEqual(str(debug.get("last_reject_reason", "")), "symbol_training_mode")
        self.assertEqual(dict(debug.get("symbol_training_mode", {})), {})

    def test_public_dashboard_access_blocks_internal_routes_for_remote_clients(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                execution_config={"startup_warmup_seconds": 0},
                dashboard_config={
                    "enabled": True,
                    "public_enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                    "bind_host": "0.0.0.0",
                    "bind_port": 8000,
                },
            )
            remote_headers = {"x-forwarded-for": "203.0.113.9"}
            with TestClient(app) as client:
                login_page = client.get("/dashboard/login", headers=remote_headers)
                unauth_data = client.get("/dashboard/data", headers=remote_headers)
                login = client.post("/dashboard/login", headers=remote_headers, data={"password": "test-pass"}, follow_redirects=False)
                authed_data = client.get("/dashboard/data", headers=remote_headers)
                stats = client.get("/stats", headers=remote_headers)
                debug = client.get("/debug/symbol", headers=remote_headers, params={"symbol": "BTCUSD", "limit": 5})
                pull = client.get("/v1/pull", headers=remote_headers, params={"symbol": "BTCUSD", "account": "A1", "magic": 7, "timeframe": "M15"})
                health = client.get("/health", headers=remote_headers)

        self.assertEqual(login_page.status_code, 200)
        self.assertEqual(unauth_data.status_code, 401)
        self.assertEqual(login.status_code, 303)
        self.assertEqual(authed_data.status_code, 200)
        self.assertEqual(stats.status_code, 403)
        self.assertEqual(debug.status_code, 403)
        self.assertEqual(pull.status_code, 403)
        self.assertEqual(health.status_code, 403)

    def test_public_dashboard_access_can_allow_public_health_by_config(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                execution_config={"startup_warmup_seconds": 0},
                dashboard_config={
                    "enabled": True,
                    "public_enabled": True,
                    "allow_public_health": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            remote_headers = {"x-forwarded-for": "203.0.113.9"}
            with TestClient(app) as client:
                health = client.get("/health", headers=remote_headers)
        self.assertEqual(health.status_code, 200)

    def test_pull_prefers_higher_band_entry_over_lower_band_higher_score(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_OVERLAP):
                queue.enqueue(
                    {
                        **_action("sig-b-band", symbol="BTCUSD"),
                        "setup": "BTC_MULTI",
                        "timeframe": "M15",
                        "dedupe_key": "band-b",
                        "probability": 0.96,
                        "expected_value_r": 1.20,
                        "confluence_score": 4.8,
                        "context_json": {
                            "selected_trade_band": "B",
                            "trade_quality_band": "B",
                            "trade_quality_score": 0.60,
                        },
                    }
                )
                queue.enqueue(
                    {
                        **_action("sig-a-band", symbol="BTCUSD"),
                        "setup": "BTC_TREND_SCALP",
                        "timeframe": "M15",
                        "dedupe_key": "band-a",
                        "probability": 0.78,
                        "expected_value_r": 0.55,
                        "confluence_score": 3.7,
                        "context_json": {
                            "selected_trade_band": "A-",
                            "trade_quality_band": "A-",
                            "trade_quality_score": 0.80,
                        },
                    }
                )
                app = create_bridge_app(
                    queue=queue,
                    journal=journal,
                    online_learning=online,
                    auth_token="",
                    execution_config={"startup_warmup_seconds": 0, "startup_revalidate": False},
                )
                with TestClient(app) as client:
                    pull = client.get(
                        "/v1/pull",
                        params={"symbol": "BTCUSD", "account": "BANDACC", "magic": 55, "timeframe": "M15", "spread_points": 25},
                    )

        self.assertEqual(pull.status_code, 200)
        actions = pull.json().get("actions", [])
        self.assertEqual(len(actions), 1)
        self.assertEqual(str(actions[0].get("signal_id") or ""), "sig-a-band")
        self.assertEqual(str(actions[0].get("selected_trade_band") or ""), "A-")

    def test_pull_prefers_higher_strategy_score_when_band_is_equal(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_TOKYO):
                queue.enqueue(
                    {
                        **_action("sig-strat-normal", symbol="AUDJPY"),
                        "setup": "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
                        "timeframe": "M15",
                        "dedupe_key": "strat-normal",
                        "probability": 0.82,
                        "expected_value_r": 0.60,
                        "confluence_score": 3.8,
                        "context_json": {
                            "selected_trade_band": "A-",
                            "trade_quality_band": "A-",
                            "trade_quality_score": 0.78,
                            "strategy_key": "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
                            "strategy_score": 0.61,
                            "strategy_state": "NORMAL",
                        },
                    }
                )
                queue.enqueue(
                    {
                        **_action("sig-strat-attack", symbol="AUDJPY"),
                        "setup": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                        "timeframe": "M15",
                        "dedupe_key": "strat-attack",
                        "probability": 0.81,
                        "expected_value_r": 0.58,
                        "confluence_score": 3.8,
                        "context_json": {
                            "selected_trade_band": "A-",
                            "trade_quality_band": "A-",
                            "trade_quality_score": 0.78,
                            "strategy_key": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                            "strategy_score": 0.92,
                            "strategy_state": "ATTACK",
                        },
                    }
                )
                app = create_bridge_app(
                    queue=queue,
                    journal=journal,
                    online_learning=online,
                    auth_token="",
                    execution_config={"startup_warmup_seconds": 0, "startup_revalidate": False},
                )
                with TestClient(app) as client:
                    pull = client.get(
                        "/v1/pull",
                        params={"symbol": "AUDJPY", "account": "STRATACC", "magic": 66, "timeframe": "M15", "spread_points": 18},
                    )

        self.assertEqual(pull.status_code, 200)
        actions = pull.json().get("actions", [])
        self.assertEqual(len(actions), 1)
        self.assertEqual(str(actions[0].get("signal_id") or ""), "sig-strat-attack")
        self.assertEqual(str(actions[0].get("strategy_key") or ""), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")

    def test_startup_revalidate_cancels_boot_queue_and_clears_locks(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=120)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            self.assertTrue(queue.enqueue(_action("sig-boot-queued", symbol="BTCUSD")))
            self.assertTrue(queue.enqueue({**_action("sig-boot-delivered", symbol="BTCUSD"), "entry_price": 2200.8}))
            delivered = queue.pull(symbol="BTCUSD", account="Main", magic=7777)
            self.assertEqual(len(delivered), 1)
            queue.acquire_open_lock(account="Main", magic=7777, symbol_key="BTCUSD", action_id="sig-boot-delivered")

            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                execution_config={
                    "startup_revalidate": True,
                    "startup_ttl_seconds": 120,
                    "startup_warmup_seconds": 600,
                },
                dashboard_config={"enabled": True, "password": "test-pass", "session_secret": "test-secret"},
            )
            with TestClient(app) as client:
                health = client.get("/health").json()

            queued_row = queue.get_action("sig-boot-queued")
            delivered_row = queue.get_action("sig-boot-delivered")

        self.assertIsNotNone(queued_row)
        self.assertIsNotNone(delivered_row)
        self.assertEqual(str(queued_row["status"]).upper(), "CANCELLED")
        self.assertEqual(str(delivered_row["status"]).upper(), "CANCELLED")
        self.assertTrue(bool(health["startup_warmup_active"]))
        self.assertEqual(int(health["startup_queue_summary"]["found"]), 2)
        self.assertEqual(int(health["startup_queue_summary"]["blocked"]), 2)
        self.assertGreaterEqual(int(health["startup_queue_summary"]["locks_cleared"]), 1)

    def test_dashboard_control_endpoints_require_auth_and_update_state(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                execution_config={"startup_warmup_seconds": 600},
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            with TestClient(app) as client:
                unauth_pause = client.post("/dashboard/control/pause_trading")
                login = client.post("/dashboard/login", data={"password": "test-pass"}, follow_redirects=False)
                clear_warmup = client.post("/dashboard/control/clear_startup_warmup").json()
                cleared_health = client.get("/health").json()
                paused = client.post("/dashboard/control/pause_trading").json()
                paused_health = client.get("/health").json()
                resumed = client.post("/dashboard/control/resume_trading").json()
                reset_daily_guard = client.post("/dashboard/control/reset_daily_guard").json()
                refreshed = client.post("/dashboard/control/refresh_state").json()
                killed = client.post("/dashboard/control/kill_switch").json()
                killed_health = client.get("/health").json()

        self.assertEqual(unauth_pause.status_code, 401)
        self.assertEqual(login.status_code, 303)
        self.assertIn("dashboard_data", clear_warmup)
        self.assertFalse(bool(cleared_health["startup_warmup_active"]))
        self.assertTrue(bool(paused["control_state"]["pause_trading"]))
        self.assertEqual(paused_health["current_kill_state"]["state"], "MANAGE_ONLY")
        self.assertFalse(bool(resumed["control_state"]["pause_trading"]))
        self.assertEqual(reset_daily_guard["action"], "reset_daily_guard")
        self.assertIn("dashboard_data", refreshed)
        self.assertTrue(bool(killed["control_state"]["kill_switch"]))
        self.assertEqual(killed_health["current_kill_state"]["state"], "HARD")

    def test_runtime_entry_block_state_tracks_warmup_and_controls(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                execution_config={"startup_warmup_seconds": 600},
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            with TestClient(app) as client:
                warmup_state = runtime_entry_block_state()
                self.assertTrue(bool(warmup_state["blocked"]))
                self.assertEqual(str(warmup_state["reason"]), "startup_warmup")
                client.post("/dashboard/login", data={"password": "test-pass"}, follow_redirects=False)
                client.post("/dashboard/control/clear_startup_warmup")
                cleared_state = runtime_entry_block_state()
                self.assertFalse(bool(cleared_state["blocked"]))
                client.post("/dashboard/control/pause_trading")
                paused_state = runtime_entry_block_state()
                self.assertTrue(bool(paused_state["blocked"]))
                self.assertEqual(str(paused_state["reason"]), "operator_pause_trading")

    def test_dashboard_data_sanitizes_non_finite_metrics(self) -> None:
        class _StubOptimizer:
            def summary(self):
                return {
                    "strategies": {
                        "WIN_ONLY": {
                            "trade_count": 20,
                            "win_rate": 1.0,
                            "profit_factor": float("inf"),
                            "expectancy_r": 0.8,
                        }
                    }
                }

            def adjustments_for(self, *_args, **_kwargs):
                return {}

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
                strategy_optimizer=_StubOptimizer(),
            )
            client = TestClient(app)
            login = client.post("/dashboard/login", data={"password": "test-pass"}, follow_redirects=False)
            self.assertEqual(login.status_code, 303)

            payload = client.get("/dashboard/data")

        self.assertEqual(payload.status_code, 200)
        data = payload.json()
        self.assertIsNone(data["learning"]["best"][0]["profit_factor"])

    def test_ai_health_endpoint_shape(self) -> None:
        class _FakeAIGate:
            def health(self):
                return {"ok": True, "mode": "local", "last_error": None, "model": "gpt-test"}

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                ai_gate=_FakeAIGate(),
                auth_token="",
            )
            client = TestClient(app)
            response = client.get("/ai/health")

            payload = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertIn("ok", payload)
            self.assertIn("mode", payload)
            self.assertIn("last_error", payload)
            self.assertIn("model", payload)

    def test_omega_status_endpoint_shape(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            response = client.get("/omega/status")

            payload = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload.get("ok"), True)
            self.assertIn("omega_enabled", payload)
            self.assertIn("kelly", payload)
            self.assertIn("symbols", payload)

    def test_omega_status_requests_scoped_journal_stats(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            with patch.object(journal, "stats", wraps=journal.stats) as stats_spy:
                response = client.get("/omega/status")

            self.assertEqual(response.status_code, 200)
            self.assertTrue(stats_spy.called)
            _, kwargs = stats_spy.call_args
            self.assertIn("account", kwargs)
            self.assertIn("magic", kwargs)

    def test_stats_endpoint_shape(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue({**_action("sig-stats-1"), "symbol": "XAUUSD"})
            queue.enqueue({**_action("sig-stats-2"), "symbol": "EURUSD"})

            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            response = client.get("/stats")

            payload = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(payload.get("ok"), True)
            self.assertIn("last_100", payload)
            self.assertIn("last_200", payload)
            self.assertIn("current_rollout_stats", payload)
            self.assertIn("queued_actions_total", payload)
            self.assertIn("queued_actions_by_symbol", payload)
            self.assertIn("micro_survivability_last_20", payload)
            self.assertIn("blocked_counts_total", payload)
            self.assertIn("blocked_counts_last_24h", payload)
            self.assertIn("current_daily_state", payload)
            self.assertIn("current_daily_state_reason", payload)
            self.assertIn("current_phase", payload)
            self.assertIn("daily_pnl_pct", payload)
            self.assertIn("daily_dd_pct_live", payload)
            self.assertIn("session_priority_diagnostics", payload)
            self.assertIn("trading_day_debug", payload)
            self.assertIn("native_candidate_to_execution_summary", payload)
            self.assertIn("trading_day_key_now", payload["trading_day_debug"])
            self.assertIn("day_bucket_source", payload["trading_day_debug"])
            self.assertIn("today_closed_trade_count", payload["trading_day_debug"])
            self.assertIn("ranked_native_candidates", payload["native_candidate_to_execution_summary"])
            self.assertGreaterEqual(int(payload.get("queued_actions_total", 0)), 2)

    def test_stats_current_rollout_excludes_pre_start_closed_trades(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            with TestClient(app) as client:
                rollout_started_at = datetime.fromisoformat(client.get("/health").json()["current_rollout_stats"]["started_at"])
                old_closed_at = (rollout_started_at - timedelta(hours=2)).isoformat()
                new_closed_at = (rollout_started_at + timedelta(seconds=1)).isoformat()
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
                            "old-rollout-sig",
                            "t-old",
                            "XAUUSD",
                            "BUY",
                            "XAUUSD_M5_GRID_SCALPER_START",
                            "LIVE",
                            "CLOSED",
                            old_closed_at,
                            old_closed_at,
                            old_closed_at,
                            2200.0,
                            2198.0,
                            2204.0,
                            0.01,
                            0.7,
                            0.3,
                            "RANGING",
                            "XAUUSD_ADAPTIVE_M5_GRID",
                            100.0,
                            99.0,
                            -1.0,
                            -1.0,
                            "",
                            0,
                            "M5",
                            0,
                            "",
                            "{}",
                            "{}",
                            "{}",
                            "USD",
                            10.0,
                            "LONDON",
                        ),
                    )
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
                            "new-rollout-sig",
                            "t-new",
                            "BTCUSD",
                            "BUY",
                            "BTC_TREND_SCALP",
                            "LIVE",
                            "CLOSED",
                            new_closed_at,
                            new_closed_at,
                            new_closed_at,
                            68000.0,
                            67900.0,
                            68200.0,
                            0.01,
                            0.8,
                            0.4,
                            "TRENDING",
                            "BTCUSD_TREND_SCALP",
                            100.0,
                            101.0,
                            1.0,
                            1.0,
                            "",
                            0,
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
                    connection.commit()

                payload = client.get("/stats").json()

        rollout = dict(payload["current_rollout_stats"])
        self.assertEqual(int(rollout["trade_count"]), 1)
        self.assertEqual(float(rollout["overall"]["trades"]), 1.0)
        self.assertAlmostEqual(float(rollout["overall"]["win_rate"]), 1.0)
        self.assertIn("BTCUSD", rollout["by_symbol"])
        self.assertNotIn("XAUUSD", rollout["by_symbol"])

    def test_stats_and_debug_include_market_data_runtime_when_provider_present(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                market_data_status_provider=lambda: {
                    "BTCUSD": {
                        "runtime_market_data_mode": "external_live",
                        "runtime_market_data_source": "yahoo_finance",
                        "runtime_market_data_consensus_state": "EXTERNAL_MULTI_PROVIDER",
                        "runtime_market_data_ready": True,
                        "runtime_market_data_age_seconds": 0.0,
                        "runtime_market_data_error": "",
                    }
                },
            )
            client = TestClient(app)

            stats = client.get("/stats").json()
            debug = client.get("/debug/symbol", params={"symbol": "BTCUSD", "limit": 5}).json()

        self.assertEqual(stats["market_data_runtime"]["BTCUSD"]["runtime_market_data_mode"], "external_live")
        self.assertEqual(stats["market_data_runtime"]["BTCUSD"]["runtime_market_data_consensus_state"], "EXTERNAL_MULTI_PROVIDER")
        self.assertEqual(debug["market_data_runtime"]["runtime_market_data_mode"], "external_live")
        self.assertEqual(debug["market_data_runtime"]["runtime_market_data_consensus_state"], "EXTERNAL_MULTI_PROVIDER")

    def test_stats_and_debug_use_runtime_candidate_attempt_counters(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                runtime_metrics_provider=lambda: {
                    "BTCUSD": {
                        "candidate_attempts_last_15m": 7,
                        "candidate_count_last_15m": 3,
                        "ai_reviews_last_15m": 3,
                        "ai_call_count_last_15m": 0,
                        "last_setup_family_considered": "FUNDING_ARB",
                        "candidate_family_counts": {"FUNDING_ARB": 1, "TREND_SCALP": 2},
                        "session_policy_current": "BTC_WEEKEND_PRIORITY",
                        "pre_open_checks_complete": True,
                        "pre_open_news_summary": "weekend volatility focus",
                        "pre_open_risk_notes": "spread sane",
                        "pre_open_setup_windows": ["BTC_WEEKEND_BREAKOUT", "BTC_RANGE_EXPANSION"],
                        "public_proxy_availability": {"funding": False, "liquidation": False},
                        "macro_event_bias": "neutral",
                        "session_bias_summary": "weekend priority",
                        "next_open_time_utc": "",
                        "next_open_time_local": "",
                        "dst_mode_active": {"new_york": True, "london": False, "sydney": True},
                        "funding_proxy_available": True,
                        "liquidation_proxy_available": False,
                        "proxy_unavailable_fallback_used": True,
                        "btc_reason_no_candidate": "proxy_unavailable_price_action_fallback",
                        "btc_last_structure_state": "RANGE_EDGE",
                        "btc_last_volatility_state": "NORMAL",
                        "btc_last_spread_state": "SANE",
                        "news_refresh_at": "2026-03-09T00:00:00+00:00",
                        "next_macro_event": "US CPI",
                        "event_risk_window_active": False,
                        "post_news_trade_window_active": True,
                        "news_bias_direction": "neutral",
                        "news_confidence": 0.55,
                        "news_data_quality": "scheduled_only",
                        "hourly_learning_summary": {"trades": 2, "wins": 1},
                        "pair_hourly_review": {"symbol": "BTCUSD", "recent_trades": 2},
                        "recent_missed_opportunity_summary": "candidate_starvation",
                        "hourly_parameter_adjustments": {"candidate_sensitivity_mult": 1.12},
                        "setup_families_promoted": ["PRICE_ACTION_FALLBACK"],
                        "setup_families_suppressed": [],
                        "live_balance": 85.0,
                        "live_equity": 90.0,
                        "live_free_margin": 88.0,
                        "account_increase_detected": True,
                        "sizing_updated": True,
                        "equity_band": "bootstrap_balanced",
                    }
                },
            )
            client = TestClient(app)

            stats = client.get("/stats").json()
            debug = client.get("/debug/symbol", params={"symbol": "BTCUSD", "limit": 5}).json()

        self.assertEqual(stats["symbol_diagnostics"]["BTCUSD"]["candidate_attempts_last_15m"], 7)
        self.assertEqual(stats["symbol_diagnostics"]["BTCUSD"]["candidate_count_last_15m"], 3)
        self.assertEqual(stats["symbol_diagnostics"]["BTCUSD"]["ai_reviews_last_15m"], 3)
        self.assertEqual(stats["symbol_diagnostics"]["BTCUSD"]["last_setup_family_considered"], "FUNDING_ARB")
        self.assertEqual(stats["symbol_diagnostics"]["BTCUSD"]["session_policy_current"], "BTC_WEEKEND_PRIORITY")
        self.assertTrue(bool(stats["symbol_diagnostics"]["BTCUSD"]["pre_open_checks_complete"]))
        self.assertEqual(stats["symbol_diagnostics"]["BTCUSD"]["btc_reason_no_candidate"], "proxy_unavailable_price_action_fallback")
        self.assertTrue(bool(stats["symbol_diagnostics"]["BTCUSD"]["account_increase_detected"]))
        self.assertIn("current_daily_state", stats)
        self.assertEqual(debug["candidate_pipeline"]["candidate_attempts_last_15m"], 7)
        self.assertEqual(debug["candidate_pipeline"]["candidate_count_last_15m"], 3)
        self.assertEqual(debug["candidate_pipeline"]["ai_reviews_last_15m"], 3)
        self.assertEqual(debug["candidate_attempts_last_15m"], 7)
        self.assertEqual(debug["session_policy_current"], "BTC_WEEKEND_PRIORITY")
        self.assertIn("session_priority_diagnostics", stats)
        self.assertIn("active_session_pair_ranking", stats["session_priority_diagnostics"])
        self.assertIn("pair_status", debug)
        self.assertTrue(bool(debug["session_state"]["pre_open_checks_complete"]))
        self.assertEqual(debug["session_state"]["dst_mode_active"]["new_york"], True)
        self.assertEqual(debug["policy"]["last_setup_family_considered"], "FUNDING_ARB")
        self.assertEqual(debug["policy"]["candidate_family_counts"]["TREND_SCALP"], 2)
        self.assertTrue(bool(debug["policy"]["pre_open_checks_complete"]))
        self.assertTrue(bool(debug["policy"]["proxy_unavailable_fallback_used"]))
        self.assertEqual(debug["news_state"]["next_macro_event"], "US CPI")
        self.assertEqual(debug["learning"]["recent_missed_opportunity_summary"], "candidate_starvation")
        self.assertTrue(bool(debug["policy"]["funding_proxy_available"]))

    def test_debug_symbol_includes_trading_day_and_native_conversion_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                runtime_metrics_provider=lambda: {
                    "AUDJPY": {
                        "session_native_pair": True,
                        "candidate_attempts_last_15m": 2,
                        "delivered_actions_last_15m": 1,
                        "primary_block_reason": "",
                    }
                },
            )
            client = TestClient(app)
            debug = client.get("/debug/symbol", params={"symbol": "AUDJPY", "account": "Main", "magic": 7777, "limit": 5}).json()

        self.assertIn("trading_day_debug", debug)
        self.assertIn("native_candidate_to_execution_summary", debug)
        self.assertIn("trading_day_key_now", debug["trading_day_debug"])
        self.assertIn("today_closed_trade_count", debug["trading_day_debug"])
        self.assertIn("ranked_native_candidates", debug["native_candidate_to_execution_summary"])
        self.assertTrue(bool(debug["account_scaling"]["sizing_updated"]))

    def test_stats_session_priority_diagnostics_use_resolved_runtime_overlay(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                runtime_metrics_provider=lambda: {
                    "AUDJPY": {
                        "candidate_attempts_last_15m": 2,
                        "trade_quality_score": 0.74,
                        "trade_quality_band": "A-",
                        "selected_trade_band": "A-",
                        "last_setup_family_considered": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                        "strategy_key": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                        "strategy_state": "ATTACK",
                        "strategy_score": 1.14,
                        "strategy_pool_winner": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                        "winning_strategy_reason": "better_regime_fit",
                        "regime_state": "BREAKOUT_EXPANSION",
                        "regime_fit": 0.91,
                        "entry_timing_score": 0.84,
                        "structure_cleanliness_score": 0.82,
                        "execution_quality_fit": 0.88,
                        "lane_budget_share": 0.40,
                        "lane_available_capacity": 4,
                        "lane_capacity_usage": 1.0,
                        "session_stop_state": "SESSION_NORMAL",
                        "pre_exec_pass": True,
                    },
                    "EURUSD": {
                        "candidate_attempts_last_15m": 2,
                        "trade_quality_score": 0.74,
                        "trade_quality_band": "A-",
                        "selected_trade_band": "A-",
                        "last_setup_family_considered": "EURUSD_LONDON_BREAKOUT",
                        "pre_exec_pass": True,
                    },
                },
            )
            client = TestClient(app)
            with patch("src.bridge_server._session_name_utc", return_value="TOKYO"):
                stats = client.get("/stats").json()

        session_diag = dict(stats.get("session_priority_diagnostics") or {})
        native_summary = dict(stats.get("native_candidate_to_execution_summary") or {})
        ranking = list(session_diag.get("active_session_pair_ranking") or [])

        self.assertIn("AUDJPY", list(session_diag.get("native_pairs_present") or []))
        self.assertGreaterEqual(int(session_diag.get("native_pairs_selected_count") or 0), 1)
        self.assertTrue(ranking)
        self.assertEqual(str(ranking[0].get("symbol") or ""), "AUDJPY")
        self.assertTrue(bool(ranking[0].get("session_native_pair")))
        self.assertEqual(str(ranking[0].get("strategy_pool_winner") or ""), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")
        self.assertEqual(str(ranking[0].get("winning_strategy_reason") or ""), "better_regime_fit")
        self.assertEqual(str(ranking[0].get("strategy_state") or ""), "ATTACK")
        self.assertEqual(str(ranking[0].get("regime_state") or ""), "BREAKOUT_EXPANSION")
        self.assertGreater(float(ranking[0].get("strategy_score") or 0.0), 1.0)
        self.assertGreater(float(ranking[0].get("entry_timing_score") or 0.0), 0.8)
        self.assertGreaterEqual(int(native_summary.get("ranked_native_candidates") or 0), 1)
        self.assertTrue(
            any(str(item.get("symbol") or "") == "AUDJPY" for item in list(native_summary.get("native_pairs") or []))
        )

    def test_dashboard_data_exposes_pair_state_and_day_scoreboard(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                runtime_metrics_provider=lambda: {
                    "AUDJPY": {
                        "session_native_pair": True,
                        "candidate_attempts_last_15m": 2,
                        "pair_status": "ATTACK",
                        "pair_status_reason": "native_pair_proving_edge_today",
                        "why_pair_is_promoted": "native_pair_proving_edge_today",
                        "rolling_expectancy_by_pair": 0.45,
                        "rolling_pf_by_pair": 1.65,
                        "rolling_expectancy_by_session": 0.52,
                        "rolling_pf_by_session": 1.72,
                        "lane_budget_share": 0.4,
                        "lane_available_capacity": 4,
                        "lane_capacity_usage": 1.0,
                        "session_priority_multiplier": 1.14,
                        "pair_priority_rank_in_session": 1,
                        "strategy_key": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                        "strategy_state": "ATTACK",
                        "strategy_score": 1.14,
                        "strategy_pool_winner": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                        "winning_strategy_reason": "better_regime_fit",
                        "regime_state": "BREAKOUT_EXPANSION",
                        "regime_fit": 0.91,
                        "entry_timing_score": 0.84,
                        "structure_cleanliness_score": 0.82,
                        "execution_quality_fit": 0.88,
                        "session_stop_state": "SESSION_NORMAL",
                    }
                },
            )
            client = TestClient(app)
            login = client.post("/dashboard/login", data={"password": "leobons4254"}, follow_redirects=False)
            self.assertIn(login.status_code, {303, 307, 200})
            payload = client.get("/dashboard/data").json()

        summary = payload["summary"]
        cards = payload["symbols"]
        self.assertIn("top_earning_pairs_today", summary)
        self.assertIn("top_losing_pairs_today", summary)
        self.assertEqual(cards[0]["pair_status"], "ATTACK")
        self.assertEqual(cards[0]["why_pair_is_promoted"], "native_pair_proving_edge_today")
        self.assertAlmostEqual(float(cards[0]["rolling_pf_by_pair"]), 1.65, places=6)
        self.assertEqual(str(cards[0]["strategy_pool_winner"]), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")
        self.assertEqual(str(cards[0]["winning_strategy_reason"]), "better_regime_fit")
        self.assertEqual(str(cards[0]["session_stop_state"]), "SESSION_NORMAL")
        self.assertGreater(float(cards[0]["lane_capacity_usage"]), 0.0)
        session_diag = dict(payload.get("summary", {}).get("session_priority_diagnostics") or payload.get("session_priority_diagnostics") or {})
        ranking = list(session_diag.get("active_session_pair_ranking") or [])
        self.assertTrue(ranking)
        self.assertEqual(str(ranking[0].get("strategy_pool_winner") or ""), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")
        self.assertEqual(str(ranking[0].get("strategy_state") or ""), "ATTACK")

    def test_dashboard_data_exposes_management_trace_fields_for_open_trades(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                dashboard_config={
                    "enabled": True,
                    "password": "test-pass",
                    "session_secret": "test-secret",
                    "read_only": True,
                },
            )
            client = TestClient(app)
            login = client.post("/dashboard/login", data={"password": "test-pass"}, follow_redirects=False)
            self.assertEqual(login.status_code, 303)

            queue.update_account_snapshot(
                account="Main",
                symbol="XAUUSD",
                magic=7777,
                balance=100.0,
                equity=103.0,
                free_margin=99.0,
                extras={"floating_pnl": 3.0},
            )
            _record_live_execution(
                journal,
                signal_id="sig-trace",
                account="Main",
                magic=7777,
                symbol="XAUUSD",
                side="BUY",
                setup="XAUUSD_M5_GRID",
                opened_at=WEEKDAY_LONDON,
                equity=103.0,
            )
            runtime = app.state.bridge_runtime
            runtime["management_progress_state"]["sig-trace"] = {
                "last_pnl_r": 0.34,
                "last_review_at": "2026-03-23T10:00:00+00:00",
                "last_review_action": "TRAIL_STOP",
                "last_review_reason": "xau_grid_capture_trail",
                "last_hold_reason": "cadence_wait",
                "last_trail_update_at": "2026-03-23T10:00:00+00:00",
                "last_tp_extension_at": "2026-03-23T10:01:00+00:00",
                "last_partial_exit_at": "2026-03-23T10:02:00+00:00",
                "last_management_cadence_seconds": 12,
                "watchdog_trigger_count": 2,
                "last_watchdog_reason": "profit_management_cadence_stale",
                "last_watchdog_trigger_at": "2026-03-23T10:03:00+00:00",
                "next_modify_due_at": "2026-03-23T10:03:12+00:00",
            }

            payload = client.get("/dashboard/data").json()
            trade = next(item for item in payload.get("open_trades", []) if str(item.get("signal_id")) == "sig-trace")

        self.assertEqual(trade["management_state"], "TRAIL_STOP")
        self.assertEqual(trade["management_reason"], "xau_grid_capture_trail")
        self.assertEqual(trade["management_watchdog_reason"], "profit_management_cadence_stale")
        self.assertEqual(int(trade["management_watchdog_triggers"]), 2)
        self.assertEqual(str(trade["last_partial_exit_at"]), "2026-03-23T10:02:00+00:00")

    def test_pull_updates_account_snapshot_when_metrics_provided(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            response = client.get(
                "/v1/pull",
                params={
                    "symbol": "XAUUSD",
                    "account": "123456",
                    "magic": 20260304,
                    "balance": 52.75,
                    "equity": 51.20,
                    "free_margin": 49.10,
                },
            )
            snapshot = queue.latest_account_snapshot()

            self.assertEqual(response.status_code, 200)
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(snapshot["account"], "123456")
            self.assertAlmostEqual(float(snapshot["balance"]), 52.75, places=2)
            self.assertAlmostEqual(float(snapshot["equity"]), 51.20, places=2)
            self.assertAlmostEqual(float(snapshot["free_margin"]), 49.10, places=2)

    def test_stats_and_debug_expose_rich_account_snapshot_and_account_scoped_context_action(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            queue.enqueue({**_action("acct-a-open", symbol="BTCUSD"), "setup": "BTC_MULTI", "timeframe": "M15"})
            queue.enqueue({**_action("acct-b-open", symbol="BTCUSD"), "setup": "BTC_MULTI", "timeframe": "M15"})

            params_a = {
                "symbol": "BTCUSD",
                "account": "ACCTA",
                "magic": 501,
                "timeframe": "M15",
                "balance": 54.06,
                "equity": 64.06,
                "free_margin": 61.25,
                "margin": 2.81,
                "margin_level": 2280.0,
                "leverage": 500,
                "spread_points": 25,
                "bid": 68000.10,
                "ask": 68000.40,
                "last": 68000.25,
                "point": 0.01,
                "digits": 2,
                "tick_size": 0.01,
                "tick_value": 1.0,
                "lot_min": 0.01,
                "lot_max": 10.0,
                "lot_step": 0.01,
                "contract_size": 1.0,
                "stops_level": 600,
                "freeze_level": 100,
                "open_count": 0,
                "net_lots": 0.0,
                "avg_entry": 0.0,
                "floating_pnl": 0.0,
                "total_open_positions": 0,
                "gross_lots_total": 0.0,
                "net_lots_total": 0.0,
                "floating_pnl_total": 0.0,
                "symbol_selected": 1,
                "symbol_trade_mode": 1,
                "terminal_connected": 1,
                "terminal_trade_allowed": 1,
                "mql_trade_allowed": 1,
                "server_time": 1772856785,
                "gmt_time": 1772856785,
            }
            params_b = {
                **params_a,
                "account": "ACCTB",
                "magic": 502,
                "ticket_size": 0.01,
            }

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_OVERLAP):
                pull_a = client.get("/v1/pull", params=params_a)
                pull_b = client.get("/v1/pull", params=params_b)
            self.assertEqual(pull_a.status_code, 200)
            self.assertEqual(pull_b.status_code, 200)

            client.post(
                "/v1/report_execution",
                json={
                    "account": "ACCTA",
                    "magic": 501,
                    "symbol": "BTCUSD",
                    "ticket": "71001",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 68000.25,
                    "sl": 67998.75,
                    "tp": 68002.12,
                },
            )
            client.post(
                "/v1/report_execution",
                json={
                    "account": "ACCTB",
                    "magic": 502,
                    "symbol": "BTCUSD",
                    "ticket": "81001",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 68000.25,
                    "sl": 67998.75,
                    "tp": 68002.12,
                },
            )

            stats = client.get("/stats").json()
            debug_a = client.get(
                "/debug/symbol",
                params={"symbol": "BTCUSD", "account": "ACCTA", "magic": 501, "timeframe": "M15", "limit": 5},
            ).json()

            recent_snapshots = stats.get("recent_account_snapshots", [])
            self.assertTrue(any(item.get("account") == "ACCTA" for item in recent_snapshots))
            bridge_snapshot = debug_a.get("bridge_account_snapshot", {})
            self.assertEqual(bridge_snapshot.get("account"), "ACCTA")
            self.assertEqual(int(bridge_snapshot.get("leverage") or 0), 500)
            self.assertEqual(int(bridge_snapshot.get("terminal_connected") or 0), 1)
            context_action = debug_a.get("bridge_context_action", {})
            self.assertEqual(str(context_action.get("ticket") or ""), "71001")
            self.assertEqual(str(context_action.get("target_ticket") or ""), "71001")
            self.assertEqual(str(debug_a.get("delivery_contract", {}).get("last_known_ticket") or ""), "71001")

    def test_latest_account_snapshot_can_filter_by_account_magic(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)

            queue.update_account_snapshot(
                account="ACCTA",
                symbol="BTCUSD",
                magic=501,
                balance=54.06,
                equity=64.06,
                free_margin=61.25,
                extras={"leverage": 500},
            )
            queue.update_account_snapshot(
                account="ACCTB",
                symbol="XAUUSD",
                magic=7777,
                balance=40.0,
                equity=41.0,
                free_margin=39.0,
                extras={"leverage": 200},
            )

            filtered = queue.latest_account_snapshot(account="ACCTA", magic=501)

            self.assertIsNotNone(filtered)
            assert filtered is not None
            self.assertEqual(str(filtered.get("account") or ""), "ACCTA")
            self.assertEqual(int(filtered.get("magic") or 0), 501)
            self.assertEqual(str(filtered.get("symbol_key") or ""), "BTCUSD")

    def test_debug_symbol_uses_delivery_fallback_counts_for_account(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=60)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("force-btc", symbol="BTCUSD"),
                    "setup": "BTC_MULTI",
                    "timeframe": "M15",
                    "tf": "M15",
                }
            )
            pulled = queue.pull(symbol="BTCUSD", account="ACCX", magic=77)
            self.assertEqual(len(pulled), 1)

            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            debug = client.get(
                "/debug/symbol",
                params={"symbol": "BTCUSD", "account": "ACCX", "magic": 77, "timeframe": "M15", "limit": 5},
            ).json()

            self.assertGreaterEqual(debug["candidate_pipeline"]["actions_sent_last_15m"], 1)
            self.assertGreaterEqual(debug["candidate_pipeline"]["queued_for_ea_last_15m"], 1)
            self.assertGreaterEqual(debug["candidate_pipeline"]["delivered_actions_last_15m"], 1)

    def test_rewrite_live_management_decision_converts_partial_into_full_exit(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=30)
            rewritten = _rewrite_live_management_decision(
                RetracementManagementDecision(
                    management_action="PARTIAL_EXIT",
                    retracement_exit_score=0.42,
                    continuation_score=0.58,
                    reversal_risk_score=0.31,
                    protection_mode="spread_recovery_partial",
                    close_fraction=0.40,
                    trade_state="PROTECTED",
                    trade_quality="ACCEPTABLE",
                    reason="spread_recovery_partial",
                    details={"management_directives": {}},
                ),
                policy=OrchestratorPolicy(disable_live_partial_exits=True),
                queue=queue,
                signal_id="sig-partial",
                ticket="123",
                symbol_key="XAUUSD",
                pnl_r=0.42,
                age_minutes=1.0,
                capture_gap_r=0.05,
            )

            self.assertEqual(rewritten.management_action, "FULL_EXIT")
            self.assertEqual(rewritten.reason, "spread_recovery_partial_full_exit")
            self.assertEqual(float(rewritten.close_fraction), 0.0)

    def test_rewrite_live_management_decision_exits_when_modify_backlog_is_stale(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue = BridgeActionQueue(db_path=Path(tmp_dir) / "bridge.sqlite", ttl_seconds=30)
            queue.enqueue(
                {
                    **_action("sig-backlog::mgmt::modify::bucket"),
                    "action_type": "MODIFY_SLTP",
                    "action": "MODIFY_SLTP",
                    "target_ticket": "777",
                    "ticket": "777",
                    "reason": "trailing_protect",
                }
            )
            rewritten = _rewrite_live_management_decision(
                RetracementManagementDecision(
                    management_action="TRAIL_STOP",
                    retracement_exit_score=0.28,
                    continuation_score=0.61,
                    reversal_risk_score=0.20,
                    protection_mode="xau_grid_capture_trail",
                    tighten_to_price=4559.90,
                    trade_state="RUNNER",
                    trade_quality="STRONG",
                    reason="xau_grid_capture_trail",
                    details={"management_directives": {}},
                ),
                policy=OrchestratorPolicy(
                    modify_backlog_limit=1,
                    modify_backlog_force_close_profit_r=0.10,
                    modify_backlog_force_close_capture_gap_r=0.20,
                ),
                queue=queue,
                signal_id="sig-backlog",
                ticket="777",
                symbol_key="XAUUSD",
                pnl_r=0.36,
                age_minutes=0.5,
                capture_gap_r=0.44,
            )

            self.assertEqual(rewritten.management_action, "FULL_EXIT")
            self.assertEqual(rewritten.reason, "modify_backlog_full_exit")

    def test_debug_symbol_counts_delivered_management_actions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=60)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("btc-open", symbol="BTCUSD"),
                    "setup": "BTC_MULTI",
                    "timeframe": "M15",
                    "tf": "M15",
                    "target_account": "ACCY",
                    "target_magic": 78,
                }
            )
            queue.enqueue(
                {
                    **_action("btc-open::mgmt::close::1", symbol="BTCUSD"),
                    "action_type": "CLOSE_POSITION",
                    "action": "CLOSE_POSITION",
                    "setup": "BTC_MULTI_CLOSE",
                    "timeframe": "M15",
                    "tf": "M15",
                    "ticket": "90078",
                    "target_ticket": "90078",
                    "target_account": "ACCY",
                    "target_magic": 78,
                    "reason": "management_close:remote",
                }
            )
            queue.mark_delivered(signal_id="btc-open", account="ACCY", magic=78)
            queue.mark_delivered(signal_id="btc-open::mgmt::close::1", account="ACCY", magic=78)

            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            debug = client.get(
                "/debug/symbol",
                params={"symbol": "BTCUSD", "account": "ACCY", "magic": 78, "timeframe": "M15", "limit": 5},
            ).json()

            self.assertGreaterEqual(debug["candidate_pipeline"]["actions_sent_last_15m"], 2)
            self.assertGreaterEqual(debug["candidate_pipeline"]["queued_for_ea_last_15m"], 2)
            self.assertGreaterEqual(debug["candidate_pipeline"]["delivered_actions_last_15m"], 2)

    def test_pull_reality_sync_fields_are_persisted(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "XAUUSD",
                        "account": "ACC1",
                        "magic": 20260304,
                        "timeframe": "M5",
                        "open_count": 2,
                        "net_lots": 0.03,
                        "avg_entry": 2201.2,
                        "floating_pnl": 0.45,
                    },
                )
            state = queue.get_symbol_state("ACC1", 20260304, "XAUUSD", "M5")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(state.get("reality_source"), "pull_params")
            self.assertEqual(state.get("state_confidence"), "high")
            self.assertEqual(int(state.get("open_positions_estimate", 0)), 2)
            self.assertEqual(int(state.get("reality_open_count", 0)), 2)
            self.assertAlmostEqual(float(state.get("reality_net_lots") or 0.0), 0.03, places=3)

    def test_btc_delivery_accepts_fraction_safe_weekend_spread(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("btc-wide-spread", symbol="BTCUSD"),
                    "setup": "BTC_TREND_SCALP",
                    "timeframe": "M15",
                    "tf": "M15",
                    "entry_price": 67811.24,
                    "sl": 67899.85,
                    "tp": 67634.02,
                    "side": "SELL",
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "MAIN",
                        "magic": 7777,
                        "timeframe": "M15",
                        "balance": 80.0,
                        "equity": 80.0,
                        "free_margin": 80.0,
                        "margin": 0.0,
                        "margin_level": 0.0,
                        "leverage": 500,
                        "spread_points": 1696,
                        "bid": 67802.76,
                        "ask": 67819.72,
                        "last": 67811.24,
                        "point": 0.01,
                        "digits": 2,
                        "tick_size": 0.01,
                        "tick_value": 0.016926,
                        "lot_min": 0.01,
                        "lot_max": 100.0,
                        "lot_step": 0.01,
                        "contract_size": 1.0,
                        "stops_level": 0,
                        "freeze_level": 0,
                        "open_count": 0,
                        "net_lots": 0.0,
                        "avg_entry": 0.0,
                        "floating_pnl": 0.0,
                        "total_open_positions": 0,
                        "gross_lots_total": 0.0,
                        "net_lots_total": 0.0,
                        "floating_pnl_total": 0.0,
                        "symbol_selected": 1,
                        "symbol_trade_mode": 1,
                        "terminal_connected": 1,
                        "terminal_trade_allowed": 1,
                        "mql_trade_allowed": 1,
                        "server_time": 1772873480,
                        "gmt_time": 1772866280,
                    },
                )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("action_type")), "OPEN_MARKET")
            state = queue.get_symbol_state("MAIN", 7777, "BTCUSD", "M15")
            self.assertNotEqual(str(state.get("last_block_reason") or ""), "trade_plan_invalid:spread_cap_exceeded")

    def test_btc_policy_blocks_outside_session_window(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("btc-outside-session", symbol="BTCUSD"),
                    "setup": "BTC_TREND_SCALP",
                    "timeframe": "M15",
                    "tf": "M15",
                    "entry_price": 67811.24,
                    "sl": 67899.85,
                    "tp": 67634.02,
                    "side": "SELL",
                    "target_account": "MAIN",
                    "target_magic": 7777,
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc)):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "MAIN",
                        "magic": 7777,
                        "timeframe": "M15",
                        "balance": 100.0,
                        "equity": 100.0,
                        "free_margin": 100.0,
                        "spread_points": 1500,
                        "bid": 67802.76,
                        "ask": 67817.76,
                        "last": 67810.26,
                        "point": 0.01,
                        "digits": 2,
                        "tick_size": 0.01,
                        "tick_value": 0.016926,
                        "lot_min": 0.01,
                        "lot_max": 100.0,
                        "lot_step": 0.01,
                        "contract_size": 1.0,
                    },
                )
            state = queue.get_symbol_state("MAIN", 7777, "BTCUSD", "M15")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("actions"), [])
        self.assertEqual(str(state.get("last_block_reason") or ""), "btc_session_only")

    def test_btc_policy_blocks_routine_trade_below_bootstrap_equity_floor(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("btc-bootstrap-routine", symbol="BTCUSD"),
                    "setup": "BTC_TREND_SCALP",
                    "timeframe": "M15",
                    "tf": "M15",
                    "entry_price": 67811.24,
                    "sl": 67899.85,
                    "tp": 67634.02,
                    "side": "SELL",
                    "probability": 0.70,
                    "target_account": "MAIN",
                    "target_magic": 7777,
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "MAIN",
                        "magic": 7777,
                        "timeframe": "M15",
                        "balance": 60.0,
                        "equity": 60.0,
                        "free_margin": 60.0,
                        "spread_points": 1500,
                        "bid": 67802.76,
                        "ask": 67817.76,
                        "last": 67810.26,
                        "point": 0.01,
                        "digits": 2,
                        "tick_size": 0.01,
                        "tick_value": 0.016926,
                        "lot_min": 0.01,
                        "lot_max": 100.0,
                        "lot_step": 0.01,
                        "contract_size": 1.0,
                    },
                )
            state = queue.get_symbol_state("MAIN", 7777, "BTCUSD", "M15")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json().get("actions"), [])
        self.assertEqual(str(state.get("last_block_reason") or ""), "btc_bootstrap_size_gate")

    def test_btc_policy_can_allow_weekend_routine_trade_when_weekend_sessions_are_configured(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("btc-weekend-routine", symbol="BTCUSD"),
                    "setup": "BTC_TREND_SCALP",
                    "timeframe": "M15",
                    "tf": "M15",
                    "entry_price": 67811.24,
                    "sl": 67899.85,
                    "tp": 67634.02,
                    "side": "SELL",
                    "probability": 0.74,
                    "expected_value_r": 0.36,
                    "target_account": "MAIN",
                    "target_magic": 7777,
                }
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={
                    "btc_policy": {
                        "trade_sessions": ["SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"],
                        "weekend_routine_sessions": ["SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"],
                        "bootstrap_min_equity_for_routine_trading": 45.0,
                    }
                },
            )
            client = TestClient(app)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc)):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "MAIN",
                        "magic": 7777,
                        "timeframe": "M15",
                        "balance": 60.0,
                        "equity": 60.0,
                        "free_margin": 60.0,
                        "spread_points": 1500,
                        "bid": 67802.76,
                        "ask": 67817.76,
                        "last": 67810.26,
                        "point": 0.01,
                        "digits": 2,
                        "tick_size": 0.01,
                        "tick_value": 0.016926,
                        "lot_min": 0.01,
                        "lot_max": 100.0,
                        "lot_step": 0.01,
                        "contract_size": 1.0,
                    },
                )
            state = queue.get_symbol_state("MAIN", 7777, "BTCUSD", "M15")

        self.assertEqual(response.status_code, 200)
        self.assertNotEqual(str(state.get("last_block_reason") or ""), "btc_session_only")
        self.assertGreaterEqual(len(response.json().get("actions", [])), 1)

    def test_btc_policy_does_not_frequency_lock_4h(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            _record_live_execution(
                journal,
                signal_id="prior-btc-open",
                account="MAIN",
                magic=7777,
                opened_at=datetime(2026, 3, 7, 13, 30, tzinfo=timezone.utc),
                equity=120.0,
            )
            queue.enqueue(
                {
                    **_action("btc-locked", symbol="BTCUSD"),
                    "setup": "BTC_TREND_SCALP",
                    "timeframe": "M15",
                    "tf": "M15",
                    "entry_price": 67811.24,
                    "sl": 67899.85,
                    "tp": 67634.02,
                    "side": "SELL",
                    "probability": 0.82,
                    "target_account": "MAIN",
                    "target_magic": 7777,
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "MAIN",
                        "magic": 7777,
                        "timeframe": "M15",
                        "balance": 120.0,
                        "equity": 120.0,
                        "free_margin": 120.0,
                        "spread_points": 1500,
                        "bid": 67802.76,
                        "ask": 67817.76,
                        "last": 67810.26,
                        "point": 0.01,
                        "digits": 2,
                        "tick_size": 0.01,
                        "tick_value": 0.016926,
                        "lot_min": 0.01,
                        "lot_max": 100.0,
                        "lot_step": 0.01,
                        "contract_size": 1.0,
                    },
                )
            state = queue.get_symbol_state("MAIN", 7777, "BTCUSD", "M15")

        self.assertEqual(response.status_code, 200)
        actions = response.json().get("actions", [])
        self.assertEqual(len(actions), 1)
        self.assertEqual(str(actions[0].get("signal_id")), "btc-locked")
        self.assertNotEqual(str(state.get("last_block_reason") or ""), "btc_frequency_lock_4h")

    def test_btc_high_conf_override_can_bypass_daily_cap(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            previous_times = [
                datetime(2026, 3, 6, 18, 0, tzinfo=timezone.utc),
                datetime(2026, 3, 6, 22, 30, tzinfo=timezone.utc),
                datetime(2026, 3, 7, 2, 30, tzinfo=timezone.utc),
                datetime(2026, 3, 7, 6, 30, tzinfo=timezone.utc),
                datetime(2026, 3, 7, 9, 30, tzinfo=timezone.utc),
                datetime(2026, 3, 7, 9, 45, tzinfo=timezone.utc),
            ]
            for idx, opened_at in enumerate(previous_times):
                _record_live_execution(
                    journal,
                    signal_id=f"prior-btc-{idx}",
                    account="MAIN",
                    magic=7777,
                    opened_at=opened_at,
                    equity=120.0,
                )
            queue.enqueue(
                {
                    **_action("btc-high-conf-override", symbol="BTCUSD"),
                    "setup": "BTC_WEEKEND_GAP_FADE",
                    "timeframe": "M15",
                    "tf": "M15",
                    "entry_price": 67811.24,
                    "sl": 67899.85,
                    "tp": 67634.02,
                    "side": "SELL",
                    "probability": 0.95,
                    "expected_value_r": 0.55,
                    "confluence_score": 5.0,
                    "target_account": "MAIN",
                    "target_magic": 7777,
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "MAIN",
                        "magic": 7777,
                        "timeframe": "M15",
                        "balance": 120.0,
                        "equity": 120.0,
                        "free_margin": 120.0,
                        "spread_points": 1500,
                        "bid": 67802.76,
                        "ask": 67817.76,
                        "last": 67810.26,
                        "point": 0.01,
                        "digits": 2,
                        "tick_size": 0.01,
                        "tick_value": 0.016926,
                        "lot_min": 0.01,
                        "lot_max": 100.0,
                        "lot_step": 0.01,
                        "contract_size": 1.0,
                    },
                )

        self.assertEqual(response.status_code, 200)
        actions = response.json().get("actions", [])
        self.assertEqual(len(actions), 1)
        self.assertEqual(str(actions[0].get("signal_id")), "btc-high-conf-override")

    def test_btc_m15_pull_can_deliver_internal_m5_action(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("btc-m5-route", symbol="BTCUSD"),
                    "setup": "BTC_TREND_SCALP",
                    "timeframe": "M5",
                    "tf": "M5",
                    "entry_price": 67811.24,
                    "sl": 67899.85,
                    "tp": 67634.02,
                    "side": "SELL",
                    "target_account": "MAIN",
                    "target_magic": 7777,
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "MAIN",
                        "magic": 7777,
                        "timeframe": "M15",
                        "balance": 80.0,
                        "equity": 80.0,
                        "free_margin": 80.0,
                        "spread_points": 1696,
                        "bid": 67802.76,
                        "ask": 67819.72,
                        "last": 67811.24,
                        "point": 0.01,
                        "digits": 2,
                        "tick_size": 0.01,
                        "tick_value": 0.016926,
                        "lot_min": 0.01,
                        "lot_max": 100.0,
                        "lot_step": 0.01,
                        "contract_size": 1.0,
                        "stops_level": 0,
                        "freeze_level": 0,
                        "symbol_selected": 1,
                        "symbol_trade_mode": 1,
                        "terminal_connected": 1,
                        "terminal_trade_allowed": 1,
                        "mql_trade_allowed": 1,
                    },
                )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("signal_id")), "btc-m5-route")

    def test_low_confidence_blocks_new_grid_exposure(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(_action("sig-low-conf"))
            queue.update_symbol_state(
                account="ACC2",
                magic=77,
                symbol_key="XAUUSD",
                timeframe="M5",
                updates={
                    "state_confidence": "low",
                    "last_sync_at": "2000-01-01T00:00:00+00:00",
                },
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                response = client.get("/v1/pull", params={"symbol": "XAUUSD", "account": "ACC2", "magic": 77, "timeframe": "M5"})
            state = queue.get_symbol_state("ACC2", 77, "XAUUSD", "M5")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json().get("actions"), [])
            self.assertEqual(state.get("last_block_reason"), "low_confidence_reality_sync")

    def test_low_confidence_blocks_new_nas_exposure(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-low-conf-nas", symbol="NAS100"),
                    "setup": "NAS_SESSION_SCALPER_ORB",
                    "entry_price": 18320.0,
                    "sl": 18305.0,
                    "tp": 18335.0,
                }
            )
            queue.update_symbol_state(
                account="NASACC",
                magic=171,
                symbol_key="NAS100",
                timeframe="M5",
                updates={"state_confidence": "low", "last_sync_at": "2000-01-01T00:00:00+00:00"},
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={
                    "nas_strategy": {
                        "enabled": True,
                        "spread_caps_by_session": {"OVERLAP": 36.0, "DEFAULT": 30.0},
                        "trade_rate_targets_by_session": {"OVERLAP": 2.0, "DEFAULT": 1.5},
                        "confluence_floor": 0.60,
                    }
                },
            )
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_OVERLAP):
                response = client.get(
                    "/v1/pull",
                    params={"symbol": "NAS100", "account": "NASACC", "magic": 171, "timeframe": "M5", "spread_points": 20},
                )
            state = queue.get_symbol_state("NASACC", 171, "NAS100", "M5")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json().get("actions"), [])
            self.assertEqual(state.get("last_block_reason"), "low_confidence_reality_sync")

    def test_recycle_cooldown_does_not_block_flow_without_risk_reduction(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(_action("sig-recycle"))
            queue.update_symbol_state(
                account="ACC3",
                magic=88,
                symbol_key="XAUUSD",
                timeframe="M5",
                updates={
                    "cycle_mode": "COOLDOWN",
                    "recycle_until": "2999-01-01T00:00:00+00:00",
                    "state_confidence": "high",
                    "last_sync_at": "2998-01-01T00:00:00+00:00",
                },
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            params = {"symbol": "XAUUSD", "account": "ACC3", "magic": 88, "timeframe": "M5"}

            blocked = client.get("/v1/pull", params=params)
            queue.update_symbol_state(
                account="ACC3",
                magic=88,
                symbol_key="XAUUSD",
                timeframe="M5",
                updates={"cycle_mode": "COOLDOWN", "recycle_until": "2000-01-01T00:00:00+00:00"},
            )
            allowed = client.get("/v1/pull", params=params)

            self.assertEqual(blocked.status_code, 200)
            self.assertLessEqual(len(blocked.json().get("actions", [])), 1)
            self.assertEqual(allowed.status_code, 200)
            self.assertLessEqual(len(allowed.json().get("actions", [])), 1)

    def test_pair_strategy_loss_cooldown_blocks_only_exact_strategy(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            now = datetime(2026, 3, 20, 8, 0, tzinfo=timezone.utc)
            with patch("src.bridge_server.utc_now", return_value=now):
                for index in range(4):
                    signal_id = f"loss-grid-{index}"
                    _record_live_execution(
                        journal,
                        signal_id=signal_id,
                        account="PAIR1",
                        magic=21,
                        symbol="XAUUSD",
                        setup="XAUUSD_M5_GRID_SCALPER_START",
                        opened_at=now - timedelta(minutes=30 + index),
                        equity=200.0,
                    )
                    journal.mark_closed(
                        signal_id,
                        pnl_amount=-0.25,
                        pnl_r=-0.10,
                        equity_after_close=199.75 - (index * 0.25),
                        closed_at=now - timedelta(minutes=10 + index),
                        exit_reason="loss",
                    )
                queue.enqueue(_action("sig-blocked-grid"))
                queue.enqueue(
                    {
                        **_action("sig-allowed-xau"),
                        "setup": "XAUUSD_ATR_EXPANSION_SCALPER",
                        "entry_price": 2200.2,
                        "sl": 2198.2,
                        "tp": 2206.2,
                    }
                )
                app = create_bridge_app(
                    queue=queue,
                    journal=journal,
                    online_learning=online,
                    auth_token="",
                    orchestrator_config={
                        "pair_strategy_loss_streak_threshold": 4,
                        "pair_strategy_loss_cooldown_minutes": 60,
                        "xau_grid_loss_streak_threshold": 4,
                    },
                )
                client = TestClient(app)
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "XAUUSD",
                        "account": "PAIR1",
                        "magic": 21,
                        "timeframe": "M5",
                        "balance": 200.0,
                        "equity": 200.0,
                        "free_margin": 200.0,
                        "spread_points": 18,
                    },
                )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("signal_id")), "sig-allowed-xau")
            self.assertEqual(str(actions[0].get("setup")), "XAUUSD_ATR_EXPANSION_SCALPER")

    def test_pair_strategy_loss_cooldown_expires_after_one_hour(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            now = datetime(2026, 3, 20, 8, 0, tzinfo=timezone.utc)
            with patch("src.bridge_server.utc_now", return_value=now):
                for index in range(4):
                    signal_id = f"expired-loss-grid-{index}"
                    _record_live_execution(
                        journal,
                        signal_id=signal_id,
                        account="PAIR2",
                        magic=22,
                        symbol="XAUUSD",
                        setup="XAUUSD_M5_GRID_SCALPER_START",
                        opened_at=now - timedelta(hours=2, minutes=10 + index),
                        equity=200.0,
                    )
                    journal.mark_closed(
                        signal_id,
                        pnl_amount=-0.20,
                        pnl_r=-0.10,
                        equity_after_close=199.80 - (index * 0.20),
                        closed_at=now - timedelta(minutes=70 + index),
                        exit_reason="loss",
                    )
                queue.enqueue(_action("sig-grid-back-live"))
                app = create_bridge_app(
                    queue=queue,
                    journal=journal,
                    online_learning=online,
                    auth_token="",
                    orchestrator_config={
                        "pair_strategy_loss_streak_threshold": 4,
                        "pair_strategy_loss_cooldown_minutes": 60,
                        "xau_grid_loss_streak_threshold": 4,
                    },
                )
                client = TestClient(app)
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "XAUUSD",
                        "account": "PAIR2",
                        "magic": 22,
                        "timeframe": "M5",
                        "balance": 200.0,
                        "equity": 200.0,
                        "free_margin": 200.0,
                        "spread_points": 18,
                    },
                )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("signal_id")), "sig-grid-back-live")

    def test_pair_strategy_loss_cooldown_bypasses_after_recent_recovery(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            now = datetime(2026, 3, 20, 8, 0, tzinfo=timezone.utc)
            with patch("src.bridge_server.utc_now", return_value=now):
                strategy_setup = "XAUUSD_M5_GRID_SCALPER_START"
                pnl_r_values = [-0.10, -0.11, -0.09, 0.24, 0.22, 0.20]
                pnl_amounts = [-0.25, -0.28, -0.22, 0.55, 0.48, 0.46]
                for index, (pnl_r, pnl_amount) in enumerate(zip(pnl_r_values, pnl_amounts)):
                    signal_id = f"recovery-grid-{index}"
                    _record_live_execution(
                        journal,
                        signal_id=signal_id,
                        account="PAIR-RECOVER",
                        magic=24,
                        symbol="XAUUSD",
                        setup=strategy_setup,
                        opened_at=now - timedelta(minutes=40 + index),
                        equity=200.0,
                    )
                    journal.mark_closed(
                        signal_id,
                        pnl_amount=pnl_amount,
                        pnl_r=pnl_r,
                        equity_after_close=200.0 + sum(pnl_amounts[: index + 1]),
                        closed_at=now - timedelta(minutes=5 + index),
                        exit_reason="loss" if pnl_amount < 0.0 else "profit",
                    )
                queue.enqueue(_action("sig-recovered-grid"))
                app = create_bridge_app(
                    queue=queue,
                    journal=journal,
                    online_learning=online,
                    auth_token="",
                    orchestrator_config={
                        "pair_strategy_loss_streak_threshold": 3,
                        "pair_strategy_loss_cooldown_minutes": 60,
                        "xau_grid_loss_streak_threshold": 4,
                    },
                )
                client = TestClient(app)
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "XAUUSD",
                        "account": "PAIR-RECOVER",
                        "magic": 24,
                        "timeframe": "M5",
                        "balance": 200.0,
                        "equity": 200.0,
                        "free_margin": 200.0,
                        "spread_points": 18,
                    },
                )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("signal_id")), "sig-recovered-grid")

    def test_runtime_performance_cooldown_expires_and_unblocks_symbol(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            now = datetime(2026, 3, 20, 8, 0, tzinfo=timezone.utc)
            with patch("src.bridge_server.utc_now", return_value=now):
                queue.enqueue(_action("sig-unblocked-after-expiry"))
                app = create_bridge_app(
                    queue=queue,
                    journal=journal,
                    online_learning=online,
                    auth_token="",
                    orchestrator_config={
                        "pair_strategy_loss_streak_threshold": 4,
                        "pair_strategy_loss_cooldown_minutes": 60,
                        "xau_grid_loss_streak_threshold": 4,
                    },
                )
                client = TestClient(app)
                queue.update_symbol_state(
                    account="PAIR3",
                    magic=23,
                    symbol_key="XAUUSD",
                    timeframe="M5",
                    updates={
                        "performance_cooldown_active": True,
                        "performance_cooldown_until": "2000-01-01T00:00:00+00:00",
                    },
                )
                response = client.get(
                    "/v1/pull",
                    params={"symbol": "XAUUSD", "account": "PAIR3", "magic": 23, "timeframe": "M5"},
                )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("signal_id")), "sig-unblocked-after-expiry")

    def test_news_armed_mode_allows_high_conf_override(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            action = _action("sig-news-armed")
            action["news_status"] = "blocked_high_usd_event"
            action["confluence_score"] = 5.0
            action["probability"] = 0.95
            queue.enqueue(action)
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={
                    "news_mode": "ALLOW_HIGH_CONF",
                    "news_armed_mode": {"enabled": True, "min_probability": 0.84, "min_confluence": 0.80, "spread_mult": 1.0},
                },
            )
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                response = client.get(
                    "/v1/pull",
                    params={"symbol": "XAUUSD", "account": "ACC4", "magic": 99, "timeframe": "M5", "spread_points": 20},
                )
            payload = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(payload.get("actions", [])), 1)
            self.assertIn("NEWS_ARMED", str(payload["actions"][0].get("reason", "")))

    def test_oil_news_armed_enforces_stricter_thresholds(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            action = _action("sig-oil-news", symbol="USOIL")
            action.update(
                {
                    "setup": "OIL_INVENTORY_SCALPER_ENTRY",
                    "entry_price": 75.40,
                    "sl": 74.90,
                    "tp": 75.90,
                    "probability": 0.68,
                    "expected_value_r": 1.1,
                    "confluence_score": 3.0,
                    "news_status": "blocked_high_oil_inventory",
                }
            )
            queue.enqueue(action)
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={
                    "news_mode": "ALLOW_HIGH_CONF",
                    "oil_strategy": {
                        "enabled": True,
                        "spread_caps_by_session": {"NEW_YORK": 44.0, "DEFAULT": 40.0},
                        "confluence_floor": 0.62,
                        "news_armed": {
                            "enabled": True,
                            "eia_window_minutes_pre": 10,
                            "eia_window_minutes_post": 30,
                            "stricter_confluence_floor": 0.74,
                            "stricter_spread_cap": 32.0,
                            "volatility_cap": 1.25,
                        },
                    },
                },
            )
            client = TestClient(app)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 4, 15, 35, tzinfo=timezone.utc)):
                response = client.get(
                    "/v1/pull",
                    params={"symbol": "USOIL", "account": "OIL1", "magic": 19, "timeframe": "M5", "spread_points": 20},
                )
            state = queue.get_symbol_state("OIL1", 19, "USOIL", "M5")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json().get("actions"), [])
            self.assertEqual(state.get("last_block_reason"), "confluence_below_threshold")

    def test_session_spread_threshold_blocks_tokyo_allows_overlap(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10, open_enqueue_cooldown_seconds=0)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue({**_action("sig-spread-tokyo"), "entry_price": 2200.0})
            queue.enqueue({**_action("sig-spread-overlap"), "entry_price": 2200.8})
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={
                    "spread_session_thresholds": {
                        "TOKYO": {"grid": 50, "other": 40},
                        "OVERLAP": {"grid": 66, "other": 48},
                    }
                },
            )
            client = TestClient(app)
            params = {"symbol": "XAUUSD", "account": "ACC5", "magic": 55, "timeframe": "M5", "spread_points": 55}
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 6, 2, 0, tzinfo=timezone.utc)):
                tokyo = client.get("/v1/pull", params=params)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 6, 14, 0, tzinfo=timezone.utc)):
                overlap = client.get("/v1/pull", params=params)

            self.assertEqual(tokyo.status_code, 200)
            self.assertEqual(tokyo.json().get("actions"), [])
            self.assertEqual(overlap.status_code, 200)
            self.assertEqual(len(overlap.json().get("actions", [])), 1)

    def test_pull_returns_at_most_one_action(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(_action("sig-a"))
            queue.enqueue({**_action("sig-b"), "setup": "XAUUSD_ALT_SETUP", "entry_price": 2200.2})
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                response = client.get("/v1/pull", params={"symbol": "XAUUSD", "account": "A1", "magic": 7, "timeframe": "M5"})
            payload = response.json()
            status_counts = queue.status_counts()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(payload.get("actions", [])), 1)
            self.assertEqual(int(status_counts.get("DELIVERED", 0)), 1)
            self.assertEqual(int(status_counts.get("QUEUED", 0)), 1)

    def test_pull_with_account_snapshot_does_not_crash_when_context_json_is_blank(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue({**_action("sig-blank-context"), "context_json": ""})
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                execution_config={"startup_warmup_seconds": 0},
            )
            client = TestClient(app)

            response = client.get(
                "/v1/pull",
                params={
                    "symbol": "XAUUSD",
                    "account": "CTXFIX",
                    "magic": 71,
                    "timeframe": "M5",
                    "balance": 62.33,
                    "equity": 79.45,
                    "free_margin": 79.45,
                    "margin": 0.0,
                    "margin_level": 0.0,
                    "leverage": 500,
                    "spread_points": 24,
                    "bid": 2200.10,
                    "ask": 2200.34,
                    "last": 2200.22,
                    "point": 0.01,
                    "digits": 2,
                    "tick_size": 0.01,
                    "tick_value": 1.0,
                    "lot_min": 0.01,
                    "lot_max": 10.0,
                    "lot_step": 0.01,
                    "contract_size": 100.0,
                    "stops_level": 10,
                    "freeze_level": 5,
                    "symbol_selected": 1,
                    "symbol_trade_mode": 1,
                    "terminal_connected": 1,
                    "terminal_trade_allowed": 1,
                    "mql_trade_allowed": 1,
                },
            )

            self.assertEqual(response.status_code, 200)
            self.assertIn("actions", response.json())

    def test_prime_session_xau_grid_is_preferred_over_other_xau_open_action(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-xau-directional"),
                    "setup": "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                    "strategy_key": "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                    "context_json": {
                        "session_name": "LONDON",
                        "strategy_key": "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                        "strategy_pool_winner": "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                        "strategy_pool": ["XAUUSD_LONDON_LIQUIDITY_SWEEP", "XAUUSD_ADAPTIVE_M5_GRID"],
                        "strategy_pool_ranking": [
                            {"strategy_key": "XAUUSD_LONDON_LIQUIDITY_SWEEP", "strategy_score": 1.10, "strategy_state": "NORMAL"},
                            {"strategy_key": "XAUUSD_ADAPTIVE_M5_GRID", "strategy_score": 1.08, "strategy_state": "NORMAL"},
                        ],
                    },
                }
            )
            queue.enqueue(
                {
                    **_action("sig-xau-grid-prime"),
                    "setup": "XAUUSD_M5_GRID_SCALPER_START",
                    "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                    "context_json": {
                        "session_name": "LONDON",
                        "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                        "strategy_pool_winner": "XAUUSD_ADAPTIVE_M5_GRID",
                        "strategy_pool": ["XAUUSD_ADAPTIVE_M5_GRID", "XAUUSD_LONDON_LIQUIDITY_SWEEP"],
                        "strategy_pool_ranking": [
                            {"strategy_key": "XAUUSD_ADAPTIVE_M5_GRID", "strategy_score": 1.12, "strategy_state": "NORMAL"},
                            {"strategy_key": "XAUUSD_LONDON_LIQUIDITY_SWEEP", "strategy_score": 1.10, "strategy_state": "NORMAL"},
                        ],
                    },
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 18, 9, 30, tzinfo=timezone.utc)):
                response = client.get(
                    "/v1/pull",
                    params={"symbol": "XAUUSD", "account": "XAUPRIME", "magic": 701, "timeframe": "M5", "spread_points": 18},
                )

            payload = response.json()
            stats = client.get("/stats").json()
            health = client.get("/health").json()
            debug = client.get("/debug/symbol", params={"symbol": "XAUUSD", "limit": 5}).json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(payload.get("actions", [])), 1)
            self.assertEqual(str(payload["actions"][0].get("strategy_key") or ""), "XAUUSD_ADAPTIVE_M5_GRID")
            self.assertIn("bridge_singleton_status", health)
            self.assertIn("approved_last_10m", stats["xau_grid_pipeline"])
            self.assertIn("delivered_last_10m", stats["xau_grid_pipeline"])
            self.assertGreaterEqual(int(stats["xau_grid_pipeline"]["approved_last_15m"]), 1)
            self.assertGreaterEqual(int(stats["xau_grid_pipeline"]["delivered_last_15m"]), 1)
            self.assertIn("raw_candidate_count", stats["xau_grid_pipeline"])
            self.assertIn("usable_candidate_count", stats["xau_grid_pipeline"])
            self.assertIn("last_candidate_stage", stats["xau_grid_pipeline"])
            self.assertIn("quota_target_10m", stats["xau_grid_pipeline"])
            self.assertIn("quota_debt_10m", stats["xau_grid_pipeline"])
            self.assertIn("last_soft_penalty_reason", stats["xau_grid_pipeline"])
            self.assertIn("active_profile", stats["xau_grid_pipeline"])
            self.assertIn("proof_mode", stats["xau_grid_pipeline"])
            self.assertIn("hard_vs_soft_gate_mode", stats["xau_grid_pipeline"])
            self.assertIn("xau_grid_pipeline", health)
            self.assertIn("prep_status", health["weekly_prep_state"])
            self.assertIn("watchlist_preview", health["weekly_prep_state"])
            self.assertGreaterEqual(int(debug["candidate_pipeline"]["approved_last_15m"]), 1)
            self.assertGreaterEqual(int(debug["candidate_pipeline"]["delivered_last_15m"]), 1)

    def test_pull_cooldown_blocks_immediate_second_open(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(_action("sig-first"))
            queue.enqueue({**_action("sig-second"), "setup": "XAUUSD_ALT_SETUP", "entry_price": 2200.2})
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={"cooldown_seconds_after_delivery": 120},
            )
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                first = client.get("/v1/pull", params={"symbol": "XAUUSD", "account": "A1", "magic": 9, "timeframe": "M5"})
                second = client.get("/v1/pull", params={"symbol": "XAUUSD", "account": "A1", "magic": 9, "timeframe": "M5"})

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(len(first.json().get("actions", [])), 1)
            self.assertEqual(len(second.json().get("actions", [])), 0)

    def test_health_and_stats_surface_xau_profile_defaults_before_runtime_updates(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                xau_grid_config={
                    "active_profile": "checkpoint",
                    "proof_mode": "checkpoint",
                    "checkpoint_artifact": "/tmp/xau_pair_pass_directional_v1.json",
                    "density_branch_artifact": "/tmp/xau_density_first_v5_window_a.json",
                    "density_first_mode": False,
                },
            )
            client = TestClient(app)

            health = client.get("/health").json()
            stats = client.get("/stats").json()

            self.assertEqual(health["xau_grid_pipeline"]["active_profile"], "checkpoint")
            self.assertEqual(health["xau_grid_pipeline"]["proof_mode"], "checkpoint")
            self.assertEqual(health["xau_grid_pipeline"]["checkpoint_artifact"], "/tmp/xau_pair_pass_directional_v1.json")
            self.assertEqual(stats["xau_grid_pipeline"]["active_profile"], "checkpoint")
            self.assertEqual(stats["xau_grid_pipeline"]["proof_mode"], "checkpoint")
            self.assertEqual(stats["xau_grid_pipeline"]["hard_vs_soft_gate_mode"], "CHECKPOINT_QUALITY")

    def test_health_and_stats_surface_learning_brain_status(self) -> None:
        class _FakeLearningBrain:
            def status_snapshot(self) -> dict[str, object]:
                return {
                    "mode": "local_live_offline_gpt",
                    "last_cycle_status": "ok",
                    "last_gpt_review_status": "ok",
                    "last_weekly_prep_status": "ok",
                    "current_ga_generation": 11,
                    "trajectory_projection": {
                        "days_to_100k_at_current_rate": 42.0,
                        "days_to_1m_at_current_rate": 240.0,
                    },
                    "last_promotion_summary": {
                        "applied": 3,
                        "symbols": ["USDJPY", "AUDJPY"],
                    },
                }

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                learning_brain=_FakeLearningBrain(),
                auth_token="",
            )
            client = TestClient(app)

            health = client.get("/health").json()
            stats = client.get("/stats").json()

            self.assertEqual(health["learning_brain"]["mode"], "local_live_offline_gpt")
            self.assertEqual(health["learning_brain"]["last_cycle_status"], "ok")
            self.assertEqual(health["weekly_prep_state"]["prep_status"], "ok")
            self.assertEqual(stats["learning_brain"]["mode"], "local_live_offline_gpt")
            self.assertEqual(stats["learning_brain"]["current_ga_generation"], 11)
            self.assertEqual(
                stats["learning_brain"]["trajectory_projection"]["days_to_100k_at_current_rate"],
                42.0,
            )

    def test_anomaly_retrain_respects_min_spacing_with_learning_brain(self) -> None:
        class _FakePromotionBundle:
            ga_generation_id = 17
            goal_state = {"short_target": 100000, "medium_target": 1000000}
            shadow_strategy_variants: list[dict[str, object]] = []
            self_heal_actions: list[dict[str, object]] = []

        class _FakeBrainReport:
            def __init__(self, generated_at: datetime) -> None:
                self.generated_at = generated_at
                self.local_cycle_status = "ok"
                self.promotion_bundle = _FakePromotionBundle()
                self.weekly_prep_status = "idle"

        class _FakeLearningBrain:
            def __init__(self) -> None:
                self.run_cycle_calls = 0
                self.runtime_apply_calls = 0

            def run_cycle(self, **kwargs) -> _FakeBrainReport:
                self.run_cycle_calls += 1
                return _FakeBrainReport(kwargs["now_utc"])

            def apply_promoted_params_to_runtime(self, _runtime_state: dict[str, object]) -> None:
                self.runtime_apply_calls += 1

            def status_snapshot(self) -> dict[str, object]:
                return {
                    "mode": "local_live_offline_gpt",
                    "autonomy_state": {"failure_counts": {}},
                }

        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            learning_brain = _FakeLearningBrain()
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                learning_brain=learning_brain,
                auth_token="",
                orchestrator_config={"maintenance_retrain_min_spacing_seconds": 60},
            )
            client = TestClient(app)
            params = {"symbol": "XAUUSD", "account": "A1", "magic": 77, "timeframe": "M5"}
            fixed_now = datetime(2026, 3, 19, 10, 0, tzinfo=timezone.utc)

            with patch.object(online, "status_snapshot", return_value={"last_maintenance_status": "insufficient_class_balance"}):
                with patch("src.bridge_server.utc_now", return_value=fixed_now):
                    first = client.get("/v1/pull", params=params)
                    second = client.get("/v1/pull", params=params)

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(learning_brain.run_cycle_calls, 1)
            self.assertEqual(learning_brain.runtime_apply_calls, 1)

    def test_pull_cooldown_expires_and_allows_next_action(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                queue.enqueue(_action("sig-cool-1"))
                queue.enqueue({**_action("sig-cool-2"), "setup": "XAUUSD_ALT_SETUP", "entry_price": 2200.6})
                app = create_bridge_app(
                    queue=queue,
                    journal=journal,
                    online_learning=online,
                    auth_token="",
                    orchestrator_config={"cooldown_seconds_after_delivery": 1},
                )
                client = TestClient(app)
                params = {"symbol": "XAUUSD", "account": "A1", "magic": 18, "timeframe": "M5"}

                first = client.get("/v1/pull", params=params)
                blocked = client.get("/v1/pull", params=params)
                time.sleep(1.2)
                with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON + timedelta(seconds=2)):
                    allowed = client.get("/v1/pull", params=params)

            self.assertEqual(len(first.json().get("actions", [])), 1)
            self.assertEqual(len(blocked.json().get("actions", [])), 0)
            self.assertEqual(len(allowed.json().get("actions", [])), 1)

    def test_known_eurgbp_symbol_does_not_fall_into_training_mode(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-eurgbp", symbol="EURGBP"),
                    "setup": "EURGBP_BREAKOUT_RETEST",
                    "entry_price": 0.8574,
                    "sl": 0.8554,
                    "tp": 0.8614,
                    "expiry_utc": (WEEKDAY_LONDON + timedelta(minutes=5)).isoformat(),
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                pull = client.get("/v1/pull", params={"symbol": "EURGBP", "account": "A1", "magic": 41, "timeframe": "M15"})
                debug = client.get("/debug/symbol", params={"symbol": "EURGBP", "account": "A1", "magic": 41, "timeframe": "M15"})

            self.assertEqual(pull.status_code, 200)
            self.assertEqual(len(pull.json().get("actions", [])), 1)
            self.assertNotEqual(debug.json().get("last_reject_reason"), "symbol_training_mode")

    def test_xagusd_symbol_is_not_blocked_by_training_mode(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                pull = client.get("/v1/pull", params={"symbol": "XAGUSD", "account": "A1", "magic": 41, "timeframe": "M15"})
                debug = client.get("/debug/symbol", params={"symbol": "XAGUSD", "account": "A1", "magic": 41, "timeframe": "M15"})

            self.assertEqual(pull.status_code, 200)
            self.assertNotEqual(debug.json().get("last_reject_reason"), "symbol_training_mode")

    def test_aapl_24h_alias_is_not_blocked_by_training_mode(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                pull = client.get("/v1/pull", params={"symbol": "AAPL.24H", "account": "A1", "magic": 41, "timeframe": "M15"})
                debug = client.get("/debug/symbol", params={"symbol": "AAPL.24H", "account": "A1", "magic": 41, "timeframe": "M15"})

            self.assertEqual(pull.status_code, 200)
            self.assertNotEqual(debug.json().get("last_reject_reason"), "symbol_training_mode")

    def test_pull_preserves_requested_broker_symbol_alias_in_action_payload(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(_action("sig-xau-alias", symbol="XAUUSD"))
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                pull = client.get("/v1/pull", params={"symbol": "XAUUSD+", "account": "A1", "magic": 41, "timeframe": "M5"})

            self.assertEqual(pull.status_code, 200)
            actions = pull.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(actions[0].get("symbol"), "XAUUSD+")
            self.assertEqual(actions[0].get("canonical_symbol"), "XAUUSD")

    def test_pull_does_not_resend_delivered_action(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(_action("sig-debounce"))
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            params = {"symbol": "XAUUSD", "account": "A1", "magic": 1, "timeframe": "M5"}
            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                first = client.get("/v1/pull", params=params)
                second = client.get("/v1/pull", params=params)

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(len(first.json().get("actions", [])), 1)
            self.assertEqual(len(second.json().get("actions", [])), 0)
            self.assertEqual(first.json().get("actions", [])[0].get("signal_id"), "sig-debounce")

    def test_pull_redelivery_waits_for_lease_expiry(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                queue.enqueue(
                    {
                        **_action("sig-lease"),
                        "action_type": "CLOSE_ALL",
                        "target_ticket": "",
                        "setup": "XAUUSD_CLOSE_ALL",
                        "entry_price": 2200.3,
                    }
                )
                app = create_bridge_app(
                    queue=queue,
                    journal=journal,
                    online_learning=online,
                    auth_token="",
                    orchestrator_config={"allow_redelivery": True, "lease_seconds": 2, "max_actions_per_poll": 1},
                )
                client = TestClient(app)
                params = {"symbol": "XAUUSD", "account": "A1", "magic": 101, "timeframe": "M5"}

                first = client.get("/v1/pull", params=params)
                second = client.get("/v1/pull", params=params)
                time.sleep(2.1)
                with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON + timedelta(seconds=3)):
                    third = client.get("/v1/pull", params=params)

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(third.status_code, 200)
            self.assertEqual(len(first.json().get("actions", [])), 1)
            self.assertEqual(len(second.json().get("actions", [])), 0)
            self.assertEqual(len(third.json().get("actions", [])), 1)
            self.assertEqual(third.json().get("actions", [])[0].get("signal_id"), "sig-lease")

    def test_ack_endpoint_prevents_redelivery(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-ack"),
                    "action_type": "CLOSE_ALL",
                    "target_ticket": "",
                    "setup": "XAUUSD_CLOSE_ALL",
                    "entry_price": 2200.5,
                    "expiry_utc": (WEEKDAY_LONDON + timedelta(minutes=5)).isoformat(),
                }
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={"allow_redelivery": True, "lease_seconds": 1},
            )
            client = TestClient(app)
            params = {"symbol": "XAUUSD", "account": "A1", "magic": 102, "timeframe": "M5"}

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                first = client.get("/v1/pull", params=params)
            ack = client.post("/v1/ack", json={"signal_id": "sig-ack", "status": "ACKED"})
            time.sleep(1.2)
            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                second = client.get("/v1/pull", params=params)

            self.assertEqual(first.status_code, 200)
            self.assertEqual(ack.status_code, 200)
            self.assertEqual(ack.json().get("ok"), True)
            self.assertEqual(len(first.json().get("actions", [])), 1)
            self.assertEqual(len(second.json().get("actions", [])), 0)

    def test_invalid_stop_fallback_enqueues_modify_after_execution(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-fallback"),
                    "entry_price": 2200.0,
                    "expiry_utc": (WEEKDAY_LONDON + timedelta(minutes=5)).isoformat(),
                }
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={"allow_open_without_stops_fallback": True},
            )
            client = TestClient(app)
            params = {"symbol": "XAUUSD", "account": "A1", "magic": 103, "timeframe": "M5"}
            rule = SymbolRule(
                symbol="XAUUSD",
                digits=2,
                tick_size=0.01,
                point=0.01,
                min_stop_points=120,
                freeze_points=30,
                typical_spread_points=35,
                max_slippage_points=50,
                tick_value=1.0,
                contract_size=100.0,
            )
            first_invalid = StopValidationResult(
                valid=False,
                symbol_key="XAUUSD",
                normalized_sl=None,
                normalized_tp=None,
                reason="sl_distance_below_required",
                min_required_points=200.0,
                sl_distance_points=20.0,
                tp_distance_points=20.0,
                rule=rule,
            )
            fallback_valid = StopValidationResult(
                valid=True,
                symbol_key="XAUUSD",
                normalized_sl=2198.0,
                normalized_tp=2202.0,
                reason="validated",
                min_required_points=200.0,
                sl_distance_points=200.0,
                tp_distance_points=200.0,
                rule=rule,
            )
            with patch("src.bridge_server.validate_and_normalize_stops", side_effect=[first_invalid, fallback_valid]):
                with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                    pull_response = client.get("/v1/pull", params=params)

            pulled_actions = pull_response.json().get("actions", [])
            self.assertEqual(pull_response.status_code, 200)
            self.assertEqual(len(pulled_actions), 1)
            self.assertEqual(pulled_actions[0].get("action_type"), "OPEN_MARKET")
            self.assertEqual(float(pulled_actions[0].get("sl", 0.0)), 0.0)
            self.assertEqual(float(pulled_actions[0].get("tp", 0.0)), 0.0)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                report_response = client.post(
                    "/v1/report_execution",
                    json={"signal_id": "sig-fallback", "accepted": True, "ticket": "9001", "entry_price": 2200.0},
                )
            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                modify_pull = client.get("/v1/pull", params=params)
            modify_actions = modify_pull.json().get("actions", [])

            self.assertEqual(report_response.status_code, 200)
            self.assertEqual(report_response.json().get("ok"), True)
            self.assertEqual(modify_pull.status_code, 200)
            self.assertEqual(len(modify_actions), 1)
            self.assertEqual(modify_actions[0].get("action_type"), "MODIFY_SLTP")
            self.assertEqual(str(modify_actions[0].get("target_ticket")), "9001")
            self.assertGreater(float(modify_actions[0].get("sl", 0.0)), 0.0)

    def test_broker_invalid_stops_retries_open_without_stops_then_enqueues_modify(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-invalid-stops"),
                    "entry_price": 2200.0,
                    "expiry_utc": (WEEKDAY_LONDON + timedelta(minutes=5)).isoformat(),
                }
            )
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={"allow_open_without_stops_fallback": True},
            )
            client = TestClient(app)
            params = {"symbol": "XAUUSD", "account": "A1", "magic": 104, "timeframe": "M5"}

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                initial_pull = client.get("/v1/pull", params=params)
            initial_actions = initial_pull.json().get("actions", [])

            self.assertEqual(initial_pull.status_code, 200)
            self.assertEqual(len(initial_actions), 1)
            self.assertEqual(initial_actions[0].get("action_type"), "OPEN_MARKET")
            self.assertGreater(float(initial_actions[0].get("sl", 0.0)), 0.0)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                reject_response = client.post(
                    "/v1/report_execution",
                    json={
                        "signal_id": "sig-invalid-stops",
                        "accepted": False,
                        "ticket": "",
                        "entry_price": 2200.0,
                        "retcode": 10016,
                        "reason": "Invalid stops",
                        "symbol": "XAUUSD",
                        "account": "A1",
                        "magic": 104,
                    },
                )

            self.assertEqual(reject_response.status_code, 200)
            self.assertEqual(reject_response.json().get("ok"), True)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                retry_pull = client.get("/v1/pull", params=params)
            retry_actions = retry_pull.json().get("actions", [])

            self.assertEqual(retry_pull.status_code, 200)
            self.assertEqual(len(retry_actions), 1)
            self.assertEqual(retry_actions[0].get("signal_id"), "sig-invalid-stops::retry::nostops")
            self.assertEqual(retry_actions[0].get("action_type"), "OPEN_MARKET")
            self.assertEqual(float(retry_actions[0].get("sl", 0.0)), 0.0)
            self.assertEqual(float(retry_actions[0].get("tp", 0.0)), 0.0)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                accepted_response = client.post(
                    "/v1/report_execution",
                    json={
                        "signal_id": "sig-invalid-stops::retry::nostops",
                        "accepted": True,
                        "ticket": "9002",
                        "entry_price": 2200.0,
                        "bid": 2199.9,
                        "ask": 2200.1,
                        "symbol": "XAUUSD",
                        "account": "A1",
                        "magic": 104,
                    },
                )

            self.assertEqual(accepted_response.status_code, 200)
            self.assertEqual(accepted_response.json().get("ok"), True)

            with patch("src.bridge_server.utc_now", return_value=WEEKDAY_LONDON):
                modify_pull = client.get("/v1/pull", params=params)
            modify_actions = modify_pull.json().get("actions", [])

            self.assertEqual(modify_pull.status_code, 200)
            self.assertEqual(len(modify_actions), 1)
            self.assertEqual(modify_actions[0].get("action_type"), "MODIFY_SLTP")
            self.assertEqual(str(modify_actions[0].get("target_ticket")), "9002")
            self.assertGreater(float(modify_actions[0].get("sl", 0.0)), 0.0)

    def test_report_close_updates_dataset_and_model(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-close"),
                    "mode": "DEMO",
                    "setup": "M1_M5_BREAKOUT",
                    "regime": "TRENDING",
                    "probability": 0.64,
                    "expected_value_r": 0.2,
                    "news_status": "clear",
                    "final_decision_json": "{\"approve\": true}",
                    "entry_price": 2200.0,
                }
            )
            journal.record_execution(
                ExecutionRequest(
                    signal_id="sig-close",
                    symbol="XAUUSD",
                    side="BUY",
                    volume=0.01,
                    entry_price=2200.0,
                    stop_price=2190.0,
                    take_profit_price=2210.0,
                    mode="DEMO",
                    setup="M1_M5_BREAKOUT",
                    regime="TRENDING",
                    probability=0.64,
                    expected_value_r=0.2,
                    slippage_points=20,
                    trailing_enabled=True,
                    partial_close_enabled=True,
                    news_status="clear",
                    final_decision_json="{\"approve\": true}",
                    trading_enabled=True,
                ),
                OrderResult(True, "1001", "accepted", {"retcode": 10009}),
                equity=1000.0,
            )

            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            response = client.post(
                "/v1/report_close",
                json={
                    "signal_id": "sig-close",
                    "exit_price": 2206.0,
                    "pnl_money": 6.0,
                    "pnl_r": 0.6,
                    "equity_after_close": 1006.0,
                    "closed_at": "2026-03-05T00:00:00+00:00",
                    "session_state": "IN",
                    "spread_points": 25,
                },
            )

            trade = journal.get_trade("sig-close")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json().get("ok"), True)
            self.assertIsNotNone(trade)
            self.assertEqual(str(trade["status"]).upper(), "CLOSED")
            self.assertTrue((root / "trades.csv").exists())
            self.assertTrue((root / "online_model.pkl").exists())

    def test_report_close_prefers_persisted_strategy_key_for_learning(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            journal.record_execution(
                ExecutionRequest(
                    signal_id="sig-close-strategy",
                    symbol="AUDJPY",
                    side="BUY",
                    volume=0.01,
                    entry_price=97.10,
                    stop_price=96.90,
                    take_profit_price=97.50,
                    mode="LIVE",
                    setup="GENERIC_BREAKOUT",
                    regime="BREAKOUT_EXPANSION",
                    probability=0.66,
                    expected_value_r=0.3,
                    slippage_points=10,
                    trading_enabled=True,
                    account="Main",
                    magic=7777,
                    timeframe="M15",
                    session_name="TOKYO",
                    strategy_key="AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                ),
                OrderResult(True, "7331", "accepted", {"retcode": 10009}),
                equity=1000.0,
            )

            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            with patch("src.bridge_server.rule_change_manager.record_trade") as rule_mock, patch(
                "src.bridge_server.strategy_optimizer.record_trade",
                return_value={},
            ) as optimizer_mock:
                response = client.post(
                    "/v1/report_close",
                    json={
                        "signal_id": "sig-close-strategy",
                        "exit_price": 97.32,
                        "pnl_money": 2.2,
                        "pnl_r": 0.55,
                        "equity_after_close": 1002.2,
                        "closed_at": "2026-03-12T00:15:00+00:00",
                        "session_state": "TOKYO",
                        "spread_points": 12,
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json().get("ok"), True)
            rule_mock.assert_called_once()
            self.assertEqual(rule_mock.call_args.kwargs.get("strategy"), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")
            optimizer_mock.assert_called_once()
            optimizer_payload = optimizer_mock.call_args.args[0]
            self.assertEqual(
                optimizer_payload.get("strategy"),
                "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
            )
            self.assertEqual(optimizer_payload.get("session_state"), "TOKYO")

    def test_pull_snapshot_stores_extended_bridge_fields(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            response = client.get(
                "/v1/pull",
                params={
                    "symbol": "BTCUSD",
                    "account": "SNAP2",
                    "magic": 88,
                    "timeframe": "M15",
                    "balance": 54.06,
                    "equity": 64.06,
                    "free_margin": 61.25,
                    "margin": 2.81,
                    "margin_level": 2280.0,
                    "leverage": 500,
                    "spread_points": 25,
                    "bid": 68000.1,
                    "ask": 68000.4,
                    "last": 68000.25,
                    "point": 0.01,
                    "digits": 2,
                    "tick_size": 0.01,
                    "tick_value": 1.0,
                    "lot_min": 0.01,
                    "lot_max": 10.0,
                    "lot_step": 0.01,
                    "contract_size": 1.0,
                    "stops_level": 600,
                    "freeze_level": 100,
                    "symbol_selected": 1,
                    "terminal_connected": 1,
                    "terminal_trade_allowed": 1,
                    "mql_trade_allowed": 1,
                },
            )

            snapshot = queue.get_account_snapshot(account="SNAP2", symbol="BTCUSD", magic=88)

            self.assertEqual(response.status_code, 200)
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertEqual(float(snapshot.get("equity") or 0.0), 64.06)
            self.assertEqual(int(snapshot.get("leverage") or 0), 500)
            self.assertEqual(float(snapshot.get("tick_size") or 0.0), 0.01)
            self.assertEqual(float(snapshot.get("lot_min") or 0.0), 0.01)
            self.assertEqual(int(snapshot.get("terminal_connected") or 0), 1)

    def test_report_execution_without_signal_id_resolves_from_account_symbol_ticket(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue({**_action("sig-bridge-report", symbol="BTCUSD"), "setup": "BTCUSD_M15_WEEKEND_BREAKOUT", "timeframe": "M15"})
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            pull = client.get("/v1/pull", params={"symbol": "BTCUSD", "account": "REPACC", "magic": 44, "timeframe": "M15", "spread_points": 25})
            self.assertEqual(pull.status_code, 200)
            self.assertEqual(len(pull.json().get("actions", [])), 1)

            report = client.post(
                "/v1/report_execution",
                json={
                    "account": "REPACC",
                    "magic": 44,
                    "symbol": "BTCUSD",
                    "ticket": "7001",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 68000.25,
                    "sl": 67998.15,
                    "tp": 68001.88,
                },
            )
            action = queue.get_action("sig-bridge-report")

            self.assertEqual(report.status_code, 200)
            self.assertEqual(report.json().get("ok"), True)
            self.assertIsNotNone(action)
            assert action is not None
            self.assertEqual(str(action.get("ticket") or ""), "7001")
            self.assertEqual(str(action.get("status") or "").upper(), "ACKED")

    def test_report_execution_updates_slippage_quality_runtime_state(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-quality", symbol="BTCUSD"),
                    "setup": "BTCUSD_PRICE_ACTION_CONTINUATION",
                    "symbol": "BTCUSD",
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            response = client.post(
                "/v1/report_execution",
                json={
                    "signal_id": "sig-quality",
                    "accepted": True,
                    "ticket": "9901",
                    "spread_points": 12.0,
                    "slippage_points": 4.0,
                },
            )
            debug = client.get("/debug/symbol", params={"symbol": "BTCUSD", "limit": 5}).json()

            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json().get("ok"), True)
            self.assertGreater(float(debug["runtime"].get("slippage_quality_score") or 0.0), 0.0)
            self.assertEqual(str(debug["runtime"].get("execution_quality_state") or ""), "GOOD")

    def test_repeated_broker_rejects_trigger_temporary_quarantine(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(_action("sig-reject-a", symbol="EURUSD"))
            queue.enqueue({**_action("sig-reject-b", symbol="EURUSD"), "side": "SELL", "setup": "EURUSD_TOKYO_PULLBACK"})
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={"broker_reject_quarantine_threshold": 2, "broker_reject_quarantine_minutes": 10},
            )
            client = TestClient(app)
            now = datetime(2026, 3, 21, 9, 0, tzinfo=timezone.utc)

            with patch("src.bridge_server.utc_now", return_value=now):
                first = client.post("/v1/report_execution", json={"signal_id": "sig-reject-a", "accepted": False, "reason": "invalid_stops"})
                second = client.post("/v1/report_execution", json={"signal_id": "sig-reject-b", "accepted": False, "reason": "invalid_stops"})
                debug = client.get("/debug/symbol", params={"symbol": "EURUSD", "limit": 5}).json()

            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertTrue(bool(debug["runtime"].get("performance_cooldown_active")))
            self.assertEqual(str(debug["runtime"].get("temporary_pause_reason") or ""), "broker_reject_quarantine")

    def test_report_execution_persists_strategy_and_session_from_action_context(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-bridge-identity", symbol="AUDJPY"),
                    "setup": "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
                    "timeframe": "M15",
                    "context_json": {
                        "session_name": "TOKYO",
                        "strategy_key": "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
                        "strategy_pool": [
                            "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                            "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
                        ],
                        "strategy_pool_winner": "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
                    },
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            report = client.post(
                "/v1/report_execution",
                json={
                    "signal_id": "sig-bridge-identity",
                    "account": "MAIN",
                    "magic": 7777,
                    "symbol": "AUDJPY",
                    "ticket": "7337",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 97.20,
                    "sl": 96.95,
                    "tp": 97.55,
                },
            )
            trade = journal.get_trade("sig-bridge-identity")
            action = queue.get_action("sig-bridge-identity")

            self.assertEqual(report.status_code, 200)
            self.assertEqual(report.json().get("ok"), True)
            self.assertIsNotNone(trade)
            self.assertIsNotNone(action)
            assert trade is not None
            assert action is not None
            self.assertEqual(str(trade.get("session_name") or ""), "TOKYO")
            self.assertEqual(str(trade.get("strategy_key") or ""), "AUDJPY_TOKYO_CONTINUATION_PULLBACK")
            self.assertEqual(str(action.get("session_name") or ""), "TOKYO")
            self.assertEqual(str(action.get("strategy_key") or ""), "AUDJPY_TOKYO_CONTINUATION_PULLBACK")

    def test_report_execution_persists_strategy_pool_metadata_into_journal_context(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-bridge-pool", symbol="XAUUSD"),
                    "setup": "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                    "timeframe": "M5",
                    "strategy_key": "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                    "strategy_state": "ATTACK",
                    "strategy_score": 1.24,
                    "entry_timing_score": 0.86,
                    "structure_cleanliness_score": 0.81,
                    "execution_quality_fit": 0.88,
                    "regime_fit": 0.91,
                    "session_fit": 0.95,
                    "volatility_fit": 0.79,
                    "pair_behavior_fit": 0.84,
                    "market_data_source": "mt5+yahoo+twelve",
                    "market_data_consensus_state": "ALIGNED",
                    "session_stop_state": "NORMAL",
                    "session_stop_reason": "",
                    "context_json": {
                        "session_name": "LONDON",
                        "strategy_key": "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                        "strategy_pool": [
                            "XAUUSD_ADAPTIVE_M5_GRID",
                            "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                            "XAUUSD_NY_MOMENTUM_BREAKOUT",
                        ],
                        "strategy_pool_ranking": [
                            {"strategy_key": "XAUUSD_LONDON_LIQUIDITY_SWEEP", "strategy_score": 1.24, "strategy_state": "ATTACK"},
                            {"strategy_key": "XAUUSD_ADAPTIVE_M5_GRID", "strategy_score": 1.11, "strategy_state": "NORMAL"},
                        ],
                        "strategy_pool_winner": "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                        "winning_strategy_reason": "higher_strategy_score",
                    },
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            report = client.post(
                "/v1/report_execution",
                json={
                    "signal_id": "sig-bridge-pool",
                    "account": "MAIN",
                    "magic": 7777,
                    "symbol": "XAUUSD",
                    "ticket": "7338",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 3010.5,
                    "sl": 3004.0,
                    "tp": 3024.0,
                },
            )
            trade = journal.get_trade("sig-bridge-pool")

            self.assertEqual(report.status_code, 200)
            self.assertEqual(report.json().get("ok"), True)
            self.assertIsNotNone(trade)
            assert trade is not None
            entry_context = trade.get("entry_context_json") or {}
            if isinstance(entry_context, str):
                entry_context = json.loads(entry_context)
            self.assertEqual(str(trade.get("strategy_key") or ""), "XAUUSD_LONDON_LIQUIDITY_SWEEP")
            self.assertEqual(str(entry_context.get("strategy_pool_winner") or ""), "XAUUSD_LONDON_LIQUIDITY_SWEEP")
            self.assertEqual(str(entry_context.get("winning_strategy_reason") or ""), "higher_strategy_score")
            self.assertEqual(list(entry_context.get("strategy_pool") or [])[:2], [
                "XAUUSD_ADAPTIVE_M5_GRID",
                "XAUUSD_LONDON_LIQUIDITY_SWEEP",
            ])
            self.assertGreater(float(entry_context.get("strategy_score") or 0.0), 0.0)
            self.assertGreater(float(entry_context.get("entry_timing_score") or 0.0), 0.0)
            self.assertGreater(float(entry_context.get("structure_cleanliness_score") or 0.0), 0.0)

    def test_report_execution_without_signal_id_resolves_even_after_delivery_expiry(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=1)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue({**_action("sig-bridge-expired", symbol="BTCUSD"), "setup": "BTCUSD_M15_WEEKEND_BREAKOUT", "timeframe": "M15"})
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            pull = client.get("/v1/pull", params={"symbol": "BTCUSD", "account": "REPACC2", "magic": 46, "timeframe": "M15", "spread_points": 25})
            self.assertEqual(pull.status_code, 200)
            self.assertEqual(len(pull.json().get("actions", [])), 1)
            time.sleep(1.2)
            queue.expire_old()

            report = client.post(
                "/v1/report_execution",
                json={
                    "account": "REPACC2",
                    "magic": 46,
                    "symbol": "BTCUSD",
                    "ticket": "7002",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 68000.25,
                    "sl": 67998.15,
                    "tp": 68001.88,
                },
            )
            action = queue.get_action("sig-bridge-expired")

            self.assertEqual(report.status_code, 200)
            self.assertEqual(report.json().get("ok"), True)
            self.assertIsNotNone(action)
            assert action is not None
            self.assertEqual(str(action.get("ticket") or ""), "7002")
            self.assertEqual(str(action.get("status") or "").upper(), "ACKED")

    def test_report_execution_persists_broker_snapshot_and_entry_context(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue({**_action("sig-context", symbol="BTCUSD"), "setup": "BTCUSD_M15_WEEKEND_BREAKOUT", "timeframe": "M15"})
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            pull = client.get(
                "/v1/pull",
                params={
                    "symbol": "BTCUSD",
                    "account": "CTXACC",
                    "magic": 88,
                    "timeframe": "M15",
                    "balance": 54.06,
                    "equity": 64.06,
                    "free_margin": 61.25,
                    "margin": 2.81,
                    "margin_level": 2280.0,
                    "leverage": 500,
                    "spread_points": 25,
                    "bid": 68000.10,
                    "ask": 68000.40,
                    "last": 68000.25,
                    "point": 0.01,
                    "digits": 2,
                    "tick_size": 0.01,
                    "tick_value": 1.0,
                    "lot_min": 0.01,
                    "lot_max": 10.0,
                    "lot_step": 0.01,
                    "contract_size": 1.0,
                    "stops_level": 600,
                    "freeze_level": 100,
                    "symbol_selected": 1,
                    "symbol_trade_mode": 1,
                    "terminal_connected": 1,
                    "terminal_trade_allowed": 1,
                    "mql_trade_allowed": 1,
                    "server_time": 1772856785,
                    "gmt_time": 1772856785,
                },
            )
            self.assertEqual(pull.status_code, 200)

            report = client.post(
                "/v1/report_execution",
                json={
                    "account": "CTXACC",
                    "magic": 88,
                    "symbol": "BTCUSD",
                    "ticket": "7301",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 68000.25,
                    "sl": 67998.15,
                    "tp": 68001.88,
                    "spread_points": 25,
                },
            )
            trade = journal.get_trade("sig-context")

            self.assertEqual(report.status_code, 200)
            self.assertEqual(report.json().get("ok"), True)
            self.assertIsNotNone(trade)
            assert trade is not None
            self.assertEqual(str(trade.get("account") or ""), "CTXACC")
            self.assertEqual(int(trade.get("magic") or 0), 88)
            self.assertEqual(float(trade.get("entry_spread_points") or 0.0), 25.0)
            self.assertIn("\"leverage\": 500", str(trade.get("broker_snapshot_json") or ""))
            self.assertIn("\"match_mode\": \"account_magic_symbol_side_lot_recent\"", str(trade.get("entry_context_json") or ""))

    def test_delivered_open_without_execution_report_is_not_counted_open_or_managed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue({**_action("sig-unconfirmed-open", symbol="BTCUSD"), "setup": "BTCUSD_M15_WEEKEND_BREAKOUT", "timeframe": "M15"})
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            pull_params = {
                "symbol": "BTCUSD",
                "account": "UNCFM",
                "magic": 904,
                "timeframe": "M15",
                "balance": 54.06,
                "equity": 64.06,
                "free_margin": 61.25,
                "margin": 2.81,
                "margin_level": 2280.0,
                "leverage": 500,
                "spread_points": 25,
                "bid": 68000.10,
                "ask": 68000.40,
                "last": 68000.25,
                "point": 0.01,
                "digits": 2,
                "tick_size": 0.01,
                "tick_value": 1.0,
                "lot_min": 0.01,
                "lot_max": 10.0,
                "lot_step": 0.01,
                "contract_size": 1.0,
                "stops_level": 600,
                "freeze_level": 100,
                "symbol_selected": 1,
                "symbol_trade_mode": 1,
                "terminal_connected": 1,
                "terminal_trade_allowed": 1,
                "mql_trade_allowed": 1,
                "server_time": 1772856785,
                "gmt_time": 1772856785,
            }
            first_pull = client.get("/v1/pull", params=pull_params)
            self.assertEqual(first_pull.status_code, 200)
            self.assertEqual(len(first_pull.json().get("actions", [])), 1)

            second_pull = client.get(
                "/v1/pull",
                params={
                    **pull_params,
                    "open_count": 1,
                    "net_lots": 0.01,
                    "avg_entry": 68000.25,
                    "floating_pnl": 1.10,
                    "total_open_positions": 1,
                    "gross_lots_total": 0.01,
                    "net_lots_total": 0.01,
                    "floating_pnl_total": 1.10,
                },
            )
            open_positions = journal.get_open_positions(account="UNCFM", magic=904)
            debug = client.get(
                "/debug/symbol",
                params={"symbol": "BTCUSD", "account": "UNCFM", "magic": 904, "timeframe": "M15", "limit": 5},
            ).json()

            self.assertEqual(second_pull.status_code, 200)
            self.assertEqual(second_pull.json().get("actions", []), [])
            self.assertEqual(open_positions, [])
            self.assertEqual(bool(debug.get("management", {}).get("generation_allowed")), False)

    def test_report_close_persists_post_trade_review_fields(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=1,
            )
            queue.enqueue({**_action("sig-close-review", symbol="BTCUSD"), "setup": "BTCUSD_M15_WEEKEND_BREAKOUT", "timeframe": "M15"})
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                strategy_optimizer=optimizer,
                auth_token="",
            )
            client = TestClient(app)
            client.get("/v1/pull", params={"symbol": "BTCUSD", "account": "RVACC", "magic": 45, "timeframe": "M15", "spread_points": 25})
            client.post(
                "/v1/report_execution",
                json={
                    "account": "RVACC",
                    "magic": 45,
                    "symbol": "BTCUSD",
                    "ticket": "8451",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 68000.25,
                    "sl": 67998.15,
                    "tp": 68001.88,
                    "spread_points": 25,
                },
            )

            report_close = client.post(
                "/v1/report_close",
                json={
                    "account": "RVACC",
                    "magic": 45,
                    "symbol": "BTCUSD",
                    "ticket": "8451",
                    "profit": 1.25,
                    "bid": 68001.20,
                    "ask": 68001.50,
                    "spread_points": 30,
                    "reason": "manual_close",
                },
            )
            trade = journal.get_trade("sig-close-review")

            self.assertEqual(report_close.status_code, 200)
            self.assertEqual(report_close.json().get("ok"), True)
            self.assertIsNotNone(trade)
            assert trade is not None
            self.assertEqual(str(trade.get("exit_reason") or ""), "manual_close")
            self.assertIn("manual_close", str(trade.get("post_trade_review_json") or ""))
            self.assertIn("winner", str(trade.get("adjustment_tags_json") or ""))
            self.assertIn("operator_intervention_close", str(trade.get("adjustment_tags_json") or ""))
            self.assertIn("management_signal_id", str(trade.get("management_effect_json") or ""))
            self.assertFalse(online.data_path.exists())
            self.assertEqual(optimizer.summary().get("strategies_with_history"), [])

    def test_report_close_without_signal_id_resolves_from_account_symbol_ticket(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue({**_action("sig-close-bridge", symbol="BTCUSD"), "setup": "BTCUSD_M15_WEEKEND_BREAKOUT", "timeframe": "M15"})
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            pull = client.get("/v1/pull", params={"symbol": "BTCUSD", "account": "CLACC", "magic": 45, "timeframe": "M15", "spread_points": 25})
            self.assertEqual(len(pull.json().get("actions", [])), 1)
            report_exec = client.post(
                "/v1/report_execution",
                json={
                    "account": "CLACC",
                    "magic": 45,
                    "symbol": "BTCUSD",
                    "ticket": "8001",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 68000.25,
                    "sl": 67998.15,
                    "tp": 68001.88,
                },
            )
            self.assertEqual(report_exec.status_code, 200)

            report_close = client.post(
                "/v1/report_close",
                json={
                    "account": "CLACC",
                    "magic": 45,
                    "symbol": "BTCUSD",
                    "ticket": "8001",
                    "profit": 1.25,
                },
            )
            trade = journal.get_trade("sig-close-bridge")

            self.assertEqual(report_close.status_code, 200)
            self.assertEqual(report_close.json().get("ok"), True)
            self.assertIsNotNone(trade)
            assert trade is not None
            self.assertEqual(str(trade.get("status") or "").upper(), "CLOSED")

    def test_report_close_clears_account_scoped_symbol_state_after_match(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue({**_action("sig-close-state", symbol="BTCUSD"), "setup": "BTCUSD_M15_WEEKEND_BREAKOUT", "timeframe": "M15", "tf": "M15"})
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            pull = client.get("/v1/pull", params={"symbol": "BTCUSD", "account": "STACC", "magic": 47, "timeframe": "M15", "spread_points": 25})
            self.assertEqual(len(pull.json().get("actions", [])), 1)

            report_exec = client.post(
                "/v1/report_execution",
                json={
                    "account": "STACC",
                    "magic": 47,
                    "symbol": "BTCUSD",
                    "ticket": "8047",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 68000.25,
                    "sl": 67998.15,
                    "tp": 68001.88,
                },
            )
            self.assertEqual(report_exec.status_code, 200)

            report_close = client.post(
                "/v1/report_close",
                json={
                    "account": "STACC",
                    "magic": 47,
                    "symbol": "BTCUSD",
                    "ticket": "8047",
                    "profit": -1.25,
                },
            )
            state = queue.get_symbol_state("STACC", 47, "BTCUSD", "M15")

            self.assertEqual(report_close.status_code, 200)
            self.assertEqual(report_close.json().get("ok"), True)
            self.assertEqual(int(state.get("open_positions_estimate") or 0), 0)
            self.assertEqual(int(state.get("reality_open_count") or 0), 0)
            self.assertEqual(float(state.get("reality_net_lots") or 0.0), 0.0)
            self.assertEqual(str(state.get("last_action_type") or ""), "CLOSED")
            self.assertEqual(str(state.get("last_known_ticket") or ""), "8047")
            self.assertEqual(str(state.get("last_known_target_ticket") or ""), "8047")

    def test_report_close_persists_management_match_mode_when_management_action_exists(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue({**_action("sig-close-mgmt", symbol="BTCUSD"), "setup": "BTCUSD_M15_WEEKEND_BREAKOUT", "timeframe": "M15"})
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            pull = client.get("/v1/pull", params={"symbol": "BTCUSD", "account": "MGACC", "magic": 46, "timeframe": "M15", "spread_points": 25})
            self.assertEqual(len(pull.json().get("actions", [])), 1)
            report_exec = client.post(
                "/v1/report_execution",
                json={
                    "account": "MGACC",
                    "magic": 46,
                    "symbol": "BTCUSD",
                    "ticket": "8046",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 68000.25,
                    "sl": 67998.15,
                    "tp": 68001.88,
                },
            )
            self.assertEqual(report_exec.status_code, 200)

            close_action = {
                **_action("sig-close-mgmt::mgmt::close::1", symbol="BTCUSD"),
                "action_type": "CLOSE_POSITION",
                "action": "CLOSE_POSITION",
                "target_account": "MGACC",
                "target_magic": 46,
                "ticket": "8046",
                "target_ticket": "8046",
                "timeframe": "M15",
                "tf": "M15",
                "reason": "management_close:remote",
            }
            result = queue.enqueue_with_result(close_action)
            self.assertEqual(result.accepted, True)
            queue.mark_delivered(signal_id="sig-close-mgmt::mgmt::close::1", account="MGACC", magic=46)

            report_close = client.post(
                "/v1/report_close",
                json={
                    "account": "MGACC",
                    "magic": 46,
                    "symbol": "BTCUSD",
                    "ticket": "8046",
                    "profit": 1.25,
                },
            )
            trade = journal.get_trade("sig-close-mgmt")

            self.assertEqual(report_close.status_code, 200)
            self.assertEqual(report_close.json().get("ok"), True)
            self.assertIsNotNone(trade)
            assert trade is not None
            self.assertIn("sig-close-mgmt::mgmt::close::1", str(trade.get("management_effect_json") or ""))
            self.assertIn("account_magic_symbol_ticket", str(trade.get("management_effect_json") or ""))
            self.assertIn("account_magic_symbol_ticket", str(trade.get("post_trade_review_json") or ""))

    def test_force_test_full_lifecycle_enqueues_close_and_reconciles_without_signal_id(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch.dict(
                os.environ,
                {
                    "APEX_FORCE_TEST_MODE": "1",
                    "APEX_FORCE_TEST_SYMBOL": "BTCUSD",
                    "APEX_FORCE_TEST_ACCOUNT": "FTBTC",
                    "APEX_FORCE_TEST_MAGIC": "9090",
                    "APEX_FORCE_TEST_TF": "M15",
                    "APEX_FORCE_TEST_SIDE": "BUY",
                    "APEX_FORCE_TEST_ONCE": "1",
                    "APEX_FORCE_TEST_LIVE": "1",
                },
                clear=False,
            ):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                pull_params = {
                    "symbol": "BTCUSD",
                    "account": "FTBTC",
                    "magic": 9090,
                    "timeframe": "M15",
                    "balance": 54.06,
                    "equity": 64.06,
                    "free_margin": 61.25,
                    "margin": 2.81,
                    "margin_level": 2280.0,
                    "leverage": 500,
                    "spread_points": 25,
                    "bid": 68000.10,
                    "ask": 68000.40,
                    "last": 68000.25,
                    "point": 0.01,
                    "digits": 2,
                    "tick_size": 0.01,
                    "tick_value": 1.0,
                    "lot_min": 0.01,
                    "lot_max": 10.0,
                    "lot_step": 0.01,
                    "contract_size": 1.0,
                    "stops_level": 600,
                    "freeze_level": 100,
                    "symbol_selected": 1,
                    "symbol_trade_mode": 1,
                    "terminal_connected": 1,
                    "terminal_trade_allowed": 1,
                    "mql_trade_allowed": 1,
                    "server_time": 1772856785,
                    "gmt_time": 1772856785,
                }
                open_pull = client.get("/v1/pull", params=pull_params)
                open_actions = open_pull.json().get("actions", [])
                self.assertEqual(open_pull.status_code, 200)
                self.assertEqual(len(open_actions), 1)
                self.assertEqual(str(open_actions[0].get("action_type")), "OPEN_MARKET")

                report_exec = client.post(
                    "/v1/report_execution",
                    json={
                        "account": "FTBTC",
                        "magic": 9090,
                        "symbol": "BTCUSD",
                        "ticket": "90001",
                        "side": "BUY",
                        "lot": 0.01,
                        "price": 68000.25,
                        "sl": 67998.75,
                        "tp": 68002.12,
                        "bid": 68000.10,
                        "ask": 68000.40,
                        "spread_points": 25,
                        "ts_utc": 1772856790,
                    },
                )
                self.assertEqual(report_exec.status_code, 200)
                self.assertEqual(report_exec.json().get("ok"), True)

                close_pull = client.get(
                    "/v1/pull",
                    params={
                        **pull_params,
                        "open_count": 1,
                        "net_lots": 0.01,
                        "avg_entry": 68000.25,
                        "floating_pnl": 1.25,
                        "total_open_positions": 1,
                        "gross_lots_total": 0.01,
                        "net_lots_total": 0.01,
                        "floating_pnl_total": 1.25,
                    },
                )
                close_actions = close_pull.json().get("actions", [])
                self.assertEqual(close_pull.status_code, 200)
                self.assertEqual(len(close_actions), 1)
                self.assertEqual(str(close_actions[0].get("action_type")), "CLOSE_POSITION")
                self.assertEqual(str(close_actions[0].get("ticket")), "90001")
                self.assertEqual(str(close_actions[0].get("target_ticket")), "90001")
                close_signal_id = str(close_actions[0].get("signal_id") or "")

                report_close = client.post(
                    "/v1/report_close",
                    json={
                        "account": "FTBTC",
                        "magic": 9090,
                        "symbol": "BTCUSD",
                        "ticket": "90001",
                        "profit": 1.25,
                        "bid": 68001.20,
                        "ask": 68001.50,
                        "spread_points": 30,
                        "ts_utc": 1772856799,
                    },
                )
                trade = journal.get_trade("FORCE_TEST::BTCUSD::M15::BUY::FTBTC::9090")
                state = queue.get_symbol_state(account="FTBTC", magic=9090, symbol_key="BTCUSD", timeframe="M15")
                close_action = queue.get_action(close_signal_id)

                self.assertEqual(report_close.status_code, 200)
                self.assertEqual(report_close.json().get("ok"), True)
                self.assertIsNotNone(trade)
                assert trade is not None
                self.assertEqual(str(trade.get("status") or "").upper(), "CLOSED")
                self.assertIsNotNone(state)
                assert state is not None
                self.assertEqual(str(state.get("last_known_ticket") or ""), "90001")
                self.assertEqual(str(state.get("last_known_target_ticket") or ""), "90001")
                self.assertIsNotNone(close_action)
                assert close_action is not None
                self.assertEqual(str(close_action.get("status") or "").upper(), "CLOSED")

    def test_force_test_uses_mt5_snapshot_bid_ask_when_pull_has_no_price(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch.dict(
                os.environ,
                {
                    "APEX_FORCE_TEST_MODE": "1",
                    "APEX_FORCE_TEST_SYMBOL": "BTCUSD",
                    "APEX_FORCE_TEST_ACCOUNT": "SNAPBTC",
                    "APEX_FORCE_TEST_MAGIC": "9091",
                    "APEX_FORCE_TEST_TF": "M5",
                    "APEX_FORCE_TEST_SIDE": "BUY",
                    "APEX_FORCE_TEST_ONCE": "1",
                    "APEX_FORCE_TEST_LIVE": "1",
                },
                clear=False,
            ):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                saturday = datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)
                with patch("src.bridge_server.utc_now", return_value=saturday):
                    snapshot = client.post(
                        "/v1/mt5_snapshot",
                        json={
                            "symbol": "BTCUSD",
                            "account": "SNAPBTC",
                            "magic": 9091,
                            "timeframe": "M5",
                            "balance": 64.06,
                            "equity": 64.06,
                            "free_margin": 61.25,
                            "leverage": 500,
                            "spread_points": 25,
                            "bid": 68000.10,
                            "ask": 68000.40,
                            "point": 0.01,
                            "digits": 2,
                            "tick_size": 0.01,
                            "tick_value": 1.0,
                            "lot_min": 0.01,
                            "lot_max": 10.0,
                            "lot_step": 0.01,
                            "contract_size": 1.0,
                            "stops_level": 600,
                            "freeze_level": 100,
                        },
                    )
                    self.assertEqual(snapshot.status_code, 200)
                    response = client.get(
                        "/v1/pull",
                        params={
                            "symbol": "BTCUSD",
                            "account": "SNAPBTC",
                            "magic": 9091,
                            "timeframe": "M5",
                            "balance": 64.06,
                            "equity": 64.06,
                            "free_margin": 61.25,
                            "spread_points": 25,
                        },
                    )
            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("action_type")), "OPEN_MARKET")
            self.assertTrue(str(actions[0].get("signal_id") or "").startswith("FORCE_TEST::BTCUSD::M5::BUY::SNAPBTC::9091"))
            self.assertGreater(float(actions[0].get("sl") or 0.0), 0.0)
            self.assertGreater(float(actions[0].get("tp") or 0.0), 0.0)

    def test_pull_derives_last_price_from_bid_ask_for_account_snapshot(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            saturday = datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)
            with patch("src.bridge_server.utc_now", return_value=saturday):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "PRICECTX",
                        "magic": 9092,
                        "timeframe": "M5",
                        "balance": 64.06,
                        "equity": 64.06,
                        "free_margin": 61.25,
                        "spread_points": 25,
                        "bid": 68000.10,
                        "ask": 68000.40,
                        "point": 0.01,
                        "digits": 2,
                        "tick_size": 0.01,
                        "tick_value": 1.0,
                        "lot_min": 0.01,
                        "lot_max": 10.0,
                        "lot_step": 0.01,
                        "contract_size": 1.0,
                        "stops_level": 600,
                        "freeze_level": 100,
                    },
                )
            self.assertEqual(response.status_code, 200)
            snapshot = queue.get_account_snapshot(account="PRICECTX", symbol="BTCUSD", magic=9092)
            self.assertIsNotNone(snapshot)
            assert snapshot is not None
            self.assertAlmostEqual(float(snapshot.get("last_price") or 0.0), 68000.25)
            self.assertAlmostEqual(float(snapshot.get("bid") or 0.0), 68000.10)
            self.assertAlmostEqual(float(snapshot.get("ask") or 0.0), 68000.40)

    def test_duplicate_execution_report_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch.dict(
                os.environ,
                {
                    "APEX_FORCE_TEST_MODE": "1",
                    "APEX_FORCE_TEST_SYMBOL": "BTCUSD",
                    "APEX_FORCE_TEST_ACCOUNT": "DUPX",
                    "APEX_FORCE_TEST_MAGIC": "9301",
                    "APEX_FORCE_TEST_TF": "M15",
                    "APEX_FORCE_TEST_SIDE": "BUY",
                    "APEX_FORCE_TEST_ONCE": "1",
                    "APEX_FORCE_TEST_LIVE": "1",
                },
                clear=False,
            ):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                pull_params = {
                    "symbol": "BTCUSD",
                    "account": "DUPX",
                    "magic": 9301,
                    "timeframe": "M15",
                    "balance": 54.06,
                    "equity": 64.06,
                    "free_margin": 61.25,
                    "margin": 2.81,
                    "margin_level": 2280.0,
                    "leverage": 500,
                    "spread_points": 25,
                    "bid": 68000.10,
                    "ask": 68000.40,
                    "last": 68000.25,
                    "point": 0.01,
                    "digits": 2,
                    "tick_size": 0.01,
                    "tick_value": 1.0,
                    "lot_min": 0.01,
                    "lot_max": 10.0,
                    "lot_step": 0.01,
                    "contract_size": 1.0,
                    "stops_level": 600,
                    "freeze_level": 100,
                    "symbol_selected": 1,
                    "symbol_trade_mode": 1,
                    "terminal_connected": 1,
                    "terminal_trade_allowed": 1,
                    "mql_trade_allowed": 1,
                    "server_time": 1772856785,
                    "gmt_time": 1772856785,
                }
                open_pull = client.get("/v1/pull", params=pull_params)
                self.assertEqual(open_pull.status_code, 200)
                exec_payload = {
                    "account": "DUPX",
                    "magic": 9301,
                    "symbol": "BTCUSD",
                    "ticket": "99001",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 68000.25,
                    "sl": 67998.75,
                    "tp": 68002.12,
                    "bid": 68000.10,
                    "ask": 68000.40,
                    "spread_points": 25,
                    "ts_utc": 1772856790,
                }
                first_exec = client.post("/v1/report_execution", json=exec_payload)
                second_exec = client.post("/v1/report_execution", json=exec_payload)
                close_pull = client.get(
                    "/v1/pull",
                    params={
                        **pull_params,
                        "open_count": 1,
                        "net_lots": 0.01,
                        "avg_entry": 68000.25,
                        "floating_pnl": 1.25,
                        "total_open_positions": 1,
                        "gross_lots_total": 0.01,
                        "net_lots_total": 0.01,
                        "floating_pnl_total": 1.25,
                    },
                )
                close_actions = close_pull.json().get("actions", [])

                self.assertEqual(first_exec.status_code, 200)
                self.assertEqual(first_exec.json().get("ok"), True)
                self.assertEqual(second_exec.status_code, 200)
                self.assertEqual(second_exec.json().get("ok"), True)
                self.assertEqual(second_exec.json().get("duplicate"), True)
                self.assertEqual(len(close_actions), 1)
                self.assertEqual(str(close_actions[0].get("action_type")), "CLOSE_POSITION")
                with sqlite3.connect(root / "bridge.sqlite") as conn:
                    close_count = conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM bridge_actions
                        WHERE signal_id LIKE 'FORCE_TEST::BTCUSD::M15::BUY::DUPX::9301::mgmt::close::%'
                        """
                    ).fetchone()[0]
                self.assertEqual(int(close_count), 1)

    def test_negative_duplicate_execution_report_does_not_override_acked_fill(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            client = TestClient(
                create_bridge_app(
                    queue,
                    journal,
                    online,
                    dashboard_config={"enabled": False},
                    orchestrator_config={"delivery_timeout_retry_limit": 3},
                )
            )
            queue.enqueue({**_action("sig-stale-neg"), "symbol": "BTCUSD", "setup": "BTC_PRICE_ACTION_CONTINUATION"})
            pull_params = {
                "symbol": "BTCUSD",
                "account": "NEG1",
                "magic": 9201,
                "tf": "M15",
                "equity": 100.0,
                "balance": 100.0,
                "free_margin": 98.0,
                "margin": 2.0,
                "margin_level": 2500.0,
                "leverage": 500,
                "spread_points": 25,
                "bid": 68000.10,
                "ask": 68000.40,
                "last": 68000.25,
                "point": 0.01,
                "digits": 2,
                "tick_size": 0.01,
                "tick_value": 1.0,
                "lot_min": 0.01,
                "lot_max": 10.0,
                "lot_step": 0.01,
                "contract_size": 1.0,
                "stops_level": 600,
                "freeze_level": 100,
                "symbol_selected": 1,
                "symbol_trade_mode": 1,
                "terminal_connected": 1,
                "terminal_trade_allowed": 1,
                "mql_trade_allowed": 1,
                "server_time": 1772856785,
                "gmt_time": 1772856785,
            }
            open_pull = client.get("/v1/pull", params=pull_params)
            self.assertEqual(open_pull.status_code, 200)
            accepted = client.post(
                "/v1/report_execution",
                json={
                    "signal_id": "sig-stale-neg",
                    "accepted": True,
                    "ticket": "551100",
                    "symbol": "BTCUSD",
                    "side": "BUY",
                    "lot": 0.01,
                    "entry_price": 68000.25,
                },
            )
            rejected = client.post(
                "/v1/report_execution",
                json={
                    "signal_id": "sig-stale-neg",
                    "accepted": False,
                    "ticket": "551100",
                    "symbol": "BTCUSD",
                    "side": "BUY",
                    "lot": 0.01,
                    "reason": "duplicate_signal_already_executed",
                },
            )
            row = queue.get_action("sig-stale-neg")
            report_payload = json.loads(str(row["execution_report_json"] or "{}"))

        self.assertEqual(accepted.status_code, 200)
        self.assertTrue(bool(accepted.json().get("ok")))
        self.assertEqual(rejected.status_code, 200)
        self.assertTrue(bool(rejected.json().get("ignored_negative")))
        self.assertEqual(str(row["status"]).upper(), "ACKED")
        self.assertTrue(bool(report_payload.get("accepted")))
        self.assertEqual(str(row["ticket"]), "551100")

    def test_duplicate_close_report_is_idempotent(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch.dict(
                os.environ,
                {
                    "APEX_FORCE_TEST_MODE": "1",
                    "APEX_FORCE_TEST_SYMBOL": "BTCUSD",
                    "APEX_FORCE_TEST_ACCOUNT": "DUPC",
                    "APEX_FORCE_TEST_MAGIC": "9302",
                    "APEX_FORCE_TEST_TF": "M15",
                    "APEX_FORCE_TEST_SIDE": "BUY",
                    "APEX_FORCE_TEST_ONCE": "1",
                    "APEX_FORCE_TEST_LIVE": "1",
                },
                clear=False,
            ):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                pull_params = {
                    "symbol": "BTCUSD",
                    "account": "DUPC",
                    "magic": 9302,
                    "timeframe": "M15",
                    "balance": 54.06,
                    "equity": 64.06,
                    "free_margin": 61.25,
                    "margin": 2.81,
                    "margin_level": 2280.0,
                    "leverage": 500,
                    "spread_points": 25,
                    "bid": 68000.10,
                    "ask": 68000.40,
                    "last": 68000.25,
                    "point": 0.01,
                    "digits": 2,
                    "tick_size": 0.01,
                    "tick_value": 1.0,
                    "lot_min": 0.01,
                    "lot_max": 10.0,
                    "lot_step": 0.01,
                    "contract_size": 1.0,
                    "stops_level": 600,
                    "freeze_level": 100,
                    "symbol_selected": 1,
                    "symbol_trade_mode": 1,
                    "terminal_connected": 1,
                    "terminal_trade_allowed": 1,
                    "mql_trade_allowed": 1,
                    "server_time": 1772856785,
                    "gmt_time": 1772856785,
                }
                open_pull = client.get("/v1/pull", params=pull_params)
                self.assertEqual(open_pull.status_code, 200)
                exec_payload = {
                    "account": "DUPC",
                    "magic": 9302,
                    "symbol": "BTCUSD",
                    "ticket": "99002",
                    "side": "BUY",
                    "lot": 0.01,
                    "price": 68000.25,
                    "sl": 67998.75,
                    "tp": 68002.12,
                    "bid": 68000.10,
                    "ask": 68000.40,
                    "spread_points": 25,
                    "ts_utc": 1772856790,
                }
                client.post("/v1/report_execution", json=exec_payload)
                client.get(
                    "/v1/pull",
                    params={
                        **pull_params,
                        "open_count": 1,
                        "net_lots": 0.01,
                        "avg_entry": 68000.25,
                        "floating_pnl": 1.25,
                        "total_open_positions": 1,
                        "gross_lots_total": 0.01,
                        "net_lots_total": 0.01,
                        "floating_pnl_total": 1.25,
                    },
                )
                close_payload = {
                    "account": "DUPC",
                    "magic": 9302,
                    "symbol": "BTCUSD",
                    "ticket": "99002",
                    "profit": 1.25,
                    "bid": 68001.20,
                    "ask": 68001.50,
                    "spread_points": 30,
                    "ts_utc": 1772856799,
                }
                first_close = client.post("/v1/report_close", json=close_payload)
                second_close = client.post("/v1/report_close", json=close_payload)
                trade = journal.get_trade("FORCE_TEST::BTCUSD::M15::BUY::DUPC::9302")

                self.assertEqual(first_close.status_code, 200)
                self.assertEqual(first_close.json().get("ok"), True)
                self.assertEqual(second_close.status_code, 200)
                self.assertEqual(second_close.json().get("ok"), True)
                self.assertEqual(second_close.json().get("duplicate"), True)
                self.assertIsNotNone(trade)
                assert trade is not None
                self.assertEqual(str(trade.get("status") or "").upper(), "CLOSED")

    def test_already_flat_close_execution_is_reconciled(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-close-flat", symbol="XAUUSD"),
                    "action_type": "CLOSE_POSITION",
                    "action": "CLOSE_POSITION",
                    "ticket": "26919404",
                    "target_ticket": "26919404",
                    "target_account": "MAIN",
                    "target_magic": 7777,
                    "timeframe": "M5",
                    "tf": "M5",
                    "setup": "XAUUSD_ADAPTIVE_M5_GRID_CLOSE",
                    "reason": "management_close:remote",
                }
            )
            queue.mark_delivered(signal_id="sig-close-flat", account="MAIN", magic=7777)

            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            response = client.post(
                "/v1/report_execution",
                json={
                    "signal_id": "sig-close-flat",
                    "accepted": False,
                    "ticket": "26919404",
                    "symbol": "XAUUSD+",
                    "reason": "Position doesn't exist",
                    "account": "MAIN",
                    "magic": 7777,
                },
            )
            row = queue.get_action("sig-close-flat")
            report_payload = json.loads(str(row["execution_report_json"] or "{}"))

        self.assertEqual(response.status_code, 200)
        self.assertTrue(bool(response.json().get("ok")))
        self.assertEqual(str(row["status"]).upper(), "ACKED")
        self.assertEqual(str(row["last_error"]), "")
        self.assertEqual(bool(report_payload.get("accepted")), True)
        self.assertEqual(str(report_payload.get("reason") or ""), "close_position_already_flat")

    def test_force_test_management_close_does_not_leak_to_other_account_context(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch.dict(
                os.environ,
                {
                    "APEX_FORCE_TEST_MODE": "1",
                    "APEX_FORCE_TEST_SYMBOL": "BTCUSD",
                    "APEX_FORCE_TEST_ACCOUNT": "ISOACC",
                    "APEX_FORCE_TEST_MAGIC": "9101",
                    "APEX_FORCE_TEST_TF": "M15",
                    "APEX_FORCE_TEST_SIDE": "BUY",
                    "APEX_FORCE_TEST_ONCE": "1",
                    "APEX_FORCE_TEST_LIVE": "1",
                },
                clear=False,
            ):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                pull_params = {
                    "symbol": "BTCUSD",
                    "account": "ISOACC",
                    "magic": 9101,
                    "timeframe": "M15",
                    "balance": 54.06,
                    "equity": 64.06,
                    "free_margin": 61.25,
                    "margin": 2.81,
                    "margin_level": 2280.0,
                    "leverage": 500,
                    "spread_points": 25,
                    "bid": 68000.10,
                    "ask": 68000.40,
                    "last": 68000.25,
                    "point": 0.01,
                    "digits": 2,
                    "tick_size": 0.01,
                    "tick_value": 1.0,
                    "lot_min": 0.01,
                    "lot_max": 10.0,
                    "lot_step": 0.01,
                    "contract_size": 1.0,
                    "stops_level": 600,
                    "freeze_level": 100,
                    "symbol_selected": 1,
                    "symbol_trade_mode": 1,
                    "terminal_connected": 1,
                    "terminal_trade_allowed": 1,
                    "mql_trade_allowed": 1,
                    "server_time": 1772856785,
                    "gmt_time": 1772856785,
                }
                open_pull = client.get("/v1/pull", params=pull_params)
                self.assertEqual(open_pull.status_code, 200)
                self.assertEqual(len(open_pull.json().get("actions", [])), 1)

                report_exec = client.post(
                    "/v1/report_execution",
                    json={
                        "account": "ISOACC",
                        "magic": 9101,
                        "symbol": "BTCUSD",
                        "ticket": "91001",
                        "side": "BUY",
                        "lot": 0.01,
                        "price": 68000.25,
                        "sl": 67998.75,
                        "tp": 68002.12,
                        "bid": 68000.10,
                        "ask": 68000.40,
                        "spread_points": 25,
                        "ts_utc": 1772856790,
                    },
                )
                self.assertEqual(report_exec.status_code, 200)
                self.assertEqual(report_exec.json().get("ok"), True)

                foreign_pull = client.get(
                    "/v1/pull",
                    params={
                        **pull_params,
                        "account": "MAIN",
                        "magic": 7777,
                        "open_count": 0,
                        "net_lots": 0.0,
                        "avg_entry": 0.0,
                        "floating_pnl": 0.0,
                        "total_open_positions": 0,
                        "gross_lots_total": 0.0,
                        "net_lots_total": 0.0,
                        "floating_pnl_total": 0.0,
                    },
                )
                self.assertEqual(foreign_pull.status_code, 200)
                self.assertEqual(foreign_pull.json().get("actions"), [])

                owning_pull = client.get(
                    "/v1/pull",
                    params={
                        **pull_params,
                        "open_count": 1,
                        "net_lots": 0.01,
                        "avg_entry": 68000.25,
                        "floating_pnl": 1.25,
                        "total_open_positions": 1,
                        "gross_lots_total": 0.01,
                        "net_lots_total": 0.01,
                        "floating_pnl_total": 1.25,
                    },
                )
                close_actions = owning_pull.json().get("actions", [])
                self.assertEqual(owning_pull.status_code, 200)
                self.assertEqual(len(close_actions), 1)
                self.assertEqual(str(close_actions[0].get("action_type")), "CLOSE_POSITION")
                self.assertEqual(str(close_actions[0].get("ticket") or ""), "91001")
                self.assertEqual(str(close_actions[0].get("target_ticket") or ""), "91001")

    def test_force_test_uses_current_account_snapshot_for_open_position_limits(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            journal.record_execution(
                ExecutionRequest(
                    signal_id="UNRELATED::OPEN",
                    symbol="BTCUSD",
                    side="BUY",
                    volume=0.01,
                    entry_price=68000.25,
                    stop_price=67998.75,
                    take_profit_price=68002.12,
                    mode="LIVE",
                    setup="BTCUSD_M15_WEEKEND_BREAKOUT",
                    regime="TRENDING",
                    probability=0.65,
                    expected_value_r=0.25,
                    slippage_points=20,
                ),
                OrderResult(accepted=True, reason="ok", order_id="99901", raw={}),
                equity=64.06,
            )
            with patch.dict(
                os.environ,
                {
                    "APEX_FORCE_TEST_MODE": "1",
                    "APEX_FORCE_TEST_SYMBOL": "BTCUSD",
                    "APEX_FORCE_TEST_ACCOUNT": "FTBTCX",
                    "APEX_FORCE_TEST_MAGIC": "9190",
                    "APEX_FORCE_TEST_TF": "M15",
                    "APEX_FORCE_TEST_SIDE": "BUY",
                    "APEX_FORCE_TEST_ONCE": "1",
                    "APEX_FORCE_TEST_LIVE": "1",
                },
                clear=False,
            ):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "FTBTCX",
                        "magic": 9190,
                        "timeframe": "M15",
                        "balance": 54.06,
                        "equity": 64.06,
                        "free_margin": 61.25,
                        "margin": 2.81,
                        "margin_level": 2280.0,
                        "leverage": 500,
                        "spread_points": 25,
                        "bid": 68000.10,
                        "ask": 68000.40,
                        "last": 68000.25,
                        "point": 0.01,
                        "digits": 2,
                        "tick_size": 0.01,
                        "tick_value": 1.0,
                        "lot_min": 0.01,
                        "lot_max": 10.0,
                        "lot_step": 0.01,
                        "contract_size": 1.0,
                        "stops_level": 600,
                        "freeze_level": 100,
                        "open_count": 0,
                        "net_lots": 0.0,
                        "avg_entry": 0.0,
                        "floating_pnl": 0.0,
                        "total_open_positions": 0,
                        "gross_lots_total": 0.0,
                        "net_lots_total": 0.0,
                        "floating_pnl_total": 0.0,
                        "symbol_selected": 1,
                        "symbol_trade_mode": 1,
                        "terminal_connected": 1,
                        "terminal_trade_allowed": 1,
                        "mql_trade_allowed": 1,
                        "server_time": 1772856785,
                        "gmt_time": 1772856785,
                    },
                )

                self.assertEqual(response.status_code, 200)
                actions = response.json().get("actions", [])
                self.assertEqual(len(actions), 1)
                self.assertEqual(str(actions[0].get("action_type") or actions[0].get("action")), "OPEN_MARKET")

    def test_force_test_uses_configured_bootstrap_position_caps_without_hidden_one_trade_clamp(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch.dict(
                os.environ,
                {
                    "APEX_FORCE_TEST_MODE": "1",
                    "APEX_FORCE_TEST_SYMBOL": "BTCUSD",
                    "APEX_FORCE_TEST_ACCOUNT": "BOOTCAP",
                    "APEX_FORCE_TEST_MAGIC": "9401",
                    "APEX_FORCE_TEST_TF": "M15",
                    "APEX_FORCE_TEST_SIDE": "BUY",
                    "APEX_FORCE_TEST_ONCE": "1",
                    "APEX_FORCE_TEST_LIVE": "1",
                },
                clear=False,
            ):
                app = create_bridge_app(
                    queue=queue,
                    journal=journal,
                    online_learning=online,
                    auth_token="",
                    orchestrator_config={
                        "max_total_open_positions": 3,
                        "max_open_positions_per_symbol": 2,
                    },
                )
                client = TestClient(app)
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "BOOTCAP",
                        "magic": 9401,
                        "timeframe": "M15",
                        "balance": 54.06,
                        "equity": 64.06,
                        "free_margin": 61.25,
                        "margin": 2.81,
                        "margin_level": 2280.0,
                        "leverage": 500,
                        "spread_points": 25,
                        "bid": 68000.10,
                        "ask": 68000.40,
                        "last": 68000.25,
                        "point": 0.01,
                        "digits": 2,
                        "tick_size": 0.01,
                        "tick_value": 1.0,
                        "lot_min": 0.01,
                        "lot_max": 10.0,
                        "lot_step": 0.01,
                        "contract_size": 1.0,
                        "stops_level": 600,
                        "freeze_level": 100,
                        "open_count": 0,
                        "net_lots": 0.0,
                        "avg_entry": 0.0,
                        "floating_pnl": 0.0,
                        "total_open_positions": 1,
                        "gross_lots_total": 0.01,
                        "net_lots_total": 0.01,
                        "floating_pnl_total": 0.25,
                        "symbol_selected": 1,
                        "symbol_trade_mode": 1,
                        "terminal_connected": 1,
                        "terminal_trade_allowed": 1,
                        "mql_trade_allowed": 1,
                        "server_time": 1772856785,
                        "gmt_time": 1772856785,
                    },
                )

                self.assertEqual(response.status_code, 200)
                actions = response.json().get("actions", [])
                self.assertEqual(len(actions), 1)
                self.assertEqual(str(actions[0].get("action_type") or actions[0].get("action")), "OPEN_MARKET")

    def test_force_test_uses_live_broker_min_lot_for_nas100(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch.dict(
                os.environ,
                {
                    "APEX_FORCE_TEST_MODE": "1",
                    "APEX_FORCE_TEST_SYMBOL": "NAS100",
                    "APEX_FORCE_TEST_ACCOUNT": "NASLOT",
                    "APEX_FORCE_TEST_MAGIC": "9510",
                    "APEX_FORCE_TEST_TF": "M15",
                    "APEX_FORCE_TEST_SIDE": "BUY",
                    "APEX_FORCE_TEST_ONCE": "1",
                    "APEX_FORCE_TEST_LIVE": "1",
                    "APEX_FORCE_TEST_NONCE": "lot",
                },
                clear=False,
            ):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 9, 2, 30, tzinfo=timezone.utc)):
                    response = client.get(
                        "/v1/pull",
                        params={
                            "symbol": "NAS100",
                            "account": "NASLOT",
                            "magic": 9510,
                            "timeframe": "M15",
                            "balance": 71.37,
                            "equity": 83.70,
                            "free_margin": 75.70,
                            "margin": 7.94,
                            "margin_level": 1050.0,
                            "leverage": 500,
                            "spread_points": 80,
                            "bid": 24052.35,
                            "ask": 24053.15,
                            "last": 24052.75,
                            "point": 0.01,
                            "digits": 2,
                            "tick_size": 0.01,
                            "tick_value": 0.01,
                            "lot_min": 0.1,
                            "lot_max": 500.0,
                            "lot_step": 0.1,
                            "contract_size": 1.0,
                            "stops_level": 50,
                            "freeze_level": 0,
                            "symbol_selected": 1,
                            "symbol_trade_mode": 1,
                            "terminal_connected": 1,
                            "terminal_trade_allowed": 1,
                            "mql_trade_allowed": 1,
                            "server_time": 1773034200,
                            "gmt_time": 1773023400,
                        },
                    )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(float(actions[0].get("lot") or 0.0), 0.1)

    def test_force_test_usdjpy_uses_leverage_aware_margin_estimate(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch.dict(
                os.environ,
                {
                    "APEX_FORCE_TEST_MODE": "1",
                    "APEX_FORCE_TEST_SYMBOL": "USDJPY",
                    "APEX_FORCE_TEST_ACCOUNT": "UJFX",
                    "APEX_FORCE_TEST_MAGIC": "9402",
                    "APEX_FORCE_TEST_TF": "M15",
                    "APEX_FORCE_TEST_SIDE": "BUY",
                    "APEX_FORCE_TEST_ONCE": "1",
                    "APEX_FORCE_TEST_LIVE": "1",
                },
                clear=False,
            ):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 9, 1, 30, tzinfo=timezone.utc)):
                    response = client.get(
                        "/v1/pull",
                        params={
                            "symbol": "USDJPY",
                            "account": "UJFX",
                            "magic": 9402,
                            "timeframe": "M15",
                            "balance": 54.06,
                            "equity": 64.06,
                            "free_margin": 10.0,
                            "margin": 0.0,
                            "margin_level": 0.0,
                            "leverage": 500,
                            "spread_points": 12,
                            "bid": 148.200,
                            "ask": 148.212,
                            "last": 148.206,
                            "point": 0.001,
                            "digits": 3,
                            "tick_size": 0.001,
                            "tick_value": 0.67,
                            "lot_min": 0.01,
                            "lot_max": 100.0,
                            "lot_step": 0.01,
                            "contract_size": 100000.0,
                            "stops_level": 10,
                            "freeze_level": 5,
                            "open_count": 0,
                            "net_lots": 0.0,
                            "avg_entry": 0.0,
                            "floating_pnl": 0.0,
                            "total_open_positions": 0,
                            "gross_lots_total": 0.0,
                            "net_lots_total": 0.0,
                            "floating_pnl_total": 0.0,
                            "symbol_selected": 1,
                            "symbol_trade_mode": 1,
                            "terminal_connected": 1,
                            "terminal_trade_allowed": 1,
                            "mql_trade_allowed": 1,
                            "server_time": 1772991000,
                            "gmt_time": 1772991000,
                        },
                    )

                self.assertEqual(response.status_code, 200)
                actions = response.json().get("actions", [])
                self.assertEqual(len(actions), 1)
                self.assertEqual(str(actions[0].get("symbol") or ""), "USDJPY")

    def test_force_test_uses_raw_broker_symbol_for_uso_alias(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch.dict(
                os.environ,
                {
                    "APEX_FORCE_TEST_MODE": "1",
                    "APEX_FORCE_TEST_SYMBOL": "USOIL",
                    "APEX_FORCE_TEST_ACCOUNT": "OILFT",
                    "APEX_FORCE_TEST_MAGIC": "9501",
                    "APEX_FORCE_TEST_TF": "M15",
                    "APEX_FORCE_TEST_SIDE": "BUY",
                    "APEX_FORCE_TEST_ONCE": "1",
                    "APEX_FORCE_TEST_LIVE": "1",
                },
                clear=False,
            ):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 9, 2, 30, tzinfo=timezone.utc)):
                    response = client.get(
                        "/v1/pull",
                        params={
                            "symbol": "USO",
                            "account": "OILFT",
                            "magic": 9501,
                            "timeframe": "M15",
                            "balance": 71.37,
                            "equity": 83.70,
                            "free_margin": 75.70,
                            "margin": 7.94,
                            "margin_level": 1050.0,
                            "leverage": 500,
                            "spread_points": 7,
                            "bid": 108.74,
                            "ask": 108.81,
                            "last": 108.78,
                            "point": 0.01,
                            "digits": 2,
                            "tick_size": 0.01,
                            "tick_value": 0.017033,
                            "lot_min": 1.0,
                            "lot_max": 300.0,
                            "lot_step": 1.0,
                            "contract_size": 1.0,
                            "stops_level": 0,
                            "freeze_level": 0,
                            "symbol_selected": 1,
                            "symbol_trade_mode": 1,
                            "terminal_connected": 1,
                            "terminal_trade_allowed": 1,
                            "mql_trade_allowed": 1,
                            "server_time": 1773034200,
                            "gmt_time": 1773023400,
                        },
                    )

                self.assertEqual(response.status_code, 200)
                actions = response.json().get("actions", [])
                self.assertEqual(len(actions), 1)
                self.assertEqual(str(actions[0].get("symbol") or ""), "USO")

    def test_nas100_high_quality_candidate_can_bypass_volatile_pause(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-nas-volatile", symbol="NAS100"),
                    "setup": "NAS_TOKYO_SCALP",
                    "entry_price": 24052.75,
                    "sl": 23993.71,
                    "tp": 24152.75,
                    "lot": 0.1,
                    "reason": "nas_tokyo_scalp",
                    "news_status": "clear",
                    "probability": 0.82,
                    "expected_value_r": 1.13,
                    "confluence_score": 3.31,
                    "estimated_loss_usd": 3.95,
                    "regime": "VOLATILE",
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 9, 2, 30, tzinfo=timezone.utc)):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "NAS100",
                        "account": "NASFT",
                        "magic": 9502,
                        "timeframe": "M15",
                        "balance": 71.37,
                        "equity": 83.70,
                        "free_margin": 75.70,
                        "margin": 7.94,
                        "margin_level": 1050.0,
                        "leverage": 500,
                        "spread_points": 80,
                        "bid": 24052.35,
                        "ask": 24053.15,
                        "last": 24052.75,
                        "point": 0.01,
                        "digits": 2,
                        "tick_size": 0.01,
                        "tick_value": 0.01,
                        "lot_min": 0.1,
                        "lot_max": 500.0,
                        "lot_step": 0.1,
                        "contract_size": 1.0,
                        "stops_level": 50,
                        "freeze_level": 0,
                        "symbol_selected": 1,
                        "symbol_trade_mode": 1,
                        "terminal_connected": 1,
                        "terminal_trade_allowed": 1,
                        "mql_trade_allowed": 1,
                        "server_time": 1773034200,
                        "gmt_time": 1773023400,
                    },
                )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("symbol") or ""), "NAS100")

            debug = client.get("/debug/symbol", params={"symbol": "NAS100", "account": "NASFT", "magic": 9502, "timeframe": "M15"})
            self.assertEqual(debug.status_code, 200)
            runtime = debug.json().get("runtime", {})
            self.assertEqual(str(runtime.get("omega_regime") or ""), "VOLATILE_SELECTIVE_OK")

    def test_force_test_nas100_bypasses_session_spread_block_and_wins_delivery_priority(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            queue.enqueue(
                {
                    **_action("sig-nas-normal", symbol="NAS100"),
                    "setup": "NAS100_SESSION_MOMENTUM",
                    "entry_price": 24052.75,
                    "sl": 23993.71,
                    "tp": 24152.75,
                    "lot": 0.1,
                    "reason": "clear",
                    "news_status": "clear",
                    "probability": 0.82,
                    "expected_value_r": 1.13,
                    "confluence_score": 3.31,
                    "estimated_loss_usd": 3.95,
                    "regime": "VOLATILE",
                }
            )
            with patch.dict(
                os.environ,
                {
                    "APEX_FORCE_TEST_MODE": "1",
                    "APEX_FORCE_TEST_SYMBOL": "NAS100",
                    "APEX_FORCE_TEST_ACCOUNT": "NASMAIN",
                    "APEX_FORCE_TEST_MAGIC": "9507",
                    "APEX_FORCE_TEST_TF": "M15",
                    "APEX_FORCE_TEST_SIDE": "BUY",
                    "APEX_FORCE_TEST_ONCE": "1",
                    "APEX_FORCE_TEST_LIVE": "1",
                    "APEX_FORCE_TEST_NONCE": "prio",
                },
                clear=False,
            ):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 9, 2, 30, tzinfo=timezone.utc)):
                    response = client.get(
                        "/v1/pull",
                        params={
                            "symbol": "NAS100",
                            "account": "NASMAIN",
                            "magic": 9507,
                            "timeframe": "M15",
                            "balance": 71.37,
                            "equity": 83.70,
                            "free_margin": 75.70,
                            "margin": 7.94,
                            "margin_level": 1050.0,
                            "leverage": 500,
                            "spread_points": 80,
                            "bid": 24052.35,
                            "ask": 24053.15,
                            "last": 24052.75,
                            "point": 0.01,
                            "digits": 2,
                            "tick_size": 0.01,
                            "tick_value": 0.01,
                            "lot_min": 0.1,
                            "lot_max": 500.0,
                            "lot_step": 0.1,
                            "contract_size": 1.0,
                            "stops_level": 50,
                            "freeze_level": 0,
                            "symbol_selected": 1,
                            "symbol_trade_mode": 1,
                            "terminal_connected": 1,
                            "terminal_trade_allowed": 1,
                            "mql_trade_allowed": 1,
                            "server_time": 1773034200,
                            "gmt_time": 1773023400,
                        },
                    )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertTrue(str(actions[0].get("signal_id") or "").startswith("FORCE_TEST::NAS100::"))
            self.assertTrue(bool(actions[0].get("force_test_priority")))

    def test_force_test_nonce_produces_unique_signal_id(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=30)
            journal = TradeJournal(root / "trades.sqlite")
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                min_retrain_trades=1,
            )
            with patch.dict(
                os.environ,
                {
                    "APEX_FORCE_TEST_MODE": "1",
                    "APEX_FORCE_TEST_SYMBOL": "BTCUSD",
                    "APEX_FORCE_TEST_ACCOUNT": "NONCEACC",
                    "APEX_FORCE_TEST_MAGIC": "9403",
                    "APEX_FORCE_TEST_TF": "M15",
                    "APEX_FORCE_TEST_SIDE": "SELL",
                    "APEX_FORCE_TEST_ONCE": "1",
                    "APEX_FORCE_TEST_LIVE": "1",
                    "APEX_FORCE_TEST_NONCE": "run2",
                },
                clear=False,
            ):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "NONCEACC",
                        "magic": 9403,
                        "timeframe": "M15",
                        "balance": 54.06,
                        "equity": 64.06,
                        "free_margin": 61.25,
                        "margin": 2.81,
                        "margin_level": 2280.0,
                        "leverage": 500,
                        "spread_points": 25,
                        "bid": 68000.10,
                        "ask": 68000.40,
                        "last": 68000.25,
                        "point": 0.01,
                        "digits": 2,
                        "tick_size": 0.01,
                        "tick_value": 1.0,
                        "lot_min": 0.01,
                        "lot_max": 10.0,
                        "lot_step": 0.01,
                        "contract_size": 1.0,
                        "stops_level": 600,
                        "freeze_level": 100,
                        "open_count": 0,
                        "net_lots": 0.0,
                        "avg_entry": 0.0,
                        "floating_pnl": 0.0,
                        "total_open_positions": 0,
                        "gross_lots_total": 0.0,
                        "net_lots_total": 0.0,
                        "floating_pnl_total": 0.0,
                        "symbol_selected": 1,
                        "symbol_trade_mode": 1,
                        "terminal_connected": 1,
                        "terminal_trade_allowed": 1,
                        "mql_trade_allowed": 1,
                        "server_time": 1772856785,
                        "gmt_time": 1772856785,
                    },
                )

                self.assertEqual(response.status_code, 200)
                actions = response.json().get("actions", [])
                self.assertEqual(len(actions), 1)
                self.assertTrue(str(actions[0].get("signal_id") or "").endswith("::run2"))


if __name__ == "__main__":
    unittest.main()
