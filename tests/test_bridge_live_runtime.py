from __future__ import annotations

from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.bridge_server import BridgeActionQueue, create_bridge_app
from src.execution import ExecutionRequest, TradeJournal
from src.mt5_client import OrderResult
from src.online_learning import OnlineLearningEngine

try:
    from fastapi.testclient import TestClient

    _HAS_FASTAPI = True
except Exception:
    TestClient = None  # type: ignore
    _HAS_FASTAPI = False


_FIXED_WEEKDAY_OPEN_UTC = datetime(2026, 3, 5, 15, 0, tzinfo=timezone.utc)


def _action(signal_id: str, symbol: str = "XAUUSD", setup: str = "XAUUSD_M5_GRID_SCALPER_START") -> dict:
    return {
        "signal_id": signal_id,
        "action_type": "OPEN_MARKET",
        "symbol": symbol,
        "side": "BUY",
        "lot": 0.01,
        "sl": 2190.0,
        "tp": 2210.0,
        "max_slippage_points": 20,
        "setup": setup,
        "timeframe": "M5",
        "entry_price": 2200.0,
        "reason": "test",
        "probability": 0.90,
        "expected_value_r": 1.0,
        "confluence_score": 4.0,
        "news_status": "clear",
        "regime": "RANGING",
    }


def _runtime_fixture(tmp_dir: str) -> tuple[BridgeActionQueue, TradeJournal, OnlineLearningEngine]:
    root = Path(tmp_dir)
    queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10, open_enqueue_cooldown_seconds=0)
    journal = TradeJournal(root / "trades.sqlite")
    online = OnlineLearningEngine(
        data_path=root / "trades.csv",
        model_path=root / "online_model.pkl",
        min_retrain_trades=1,
    )
    return queue, journal, online


@unittest.skipUnless(_HAS_FASTAPI, "fastapi not installed")
class BridgeRuntimeTests(unittest.TestCase):
    def test_snapshot_endpoint_updates_reality_sync_for_pull(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            post = client.post(
                "/v1/mt5_snapshot",
                json={
                    "symbol": "XAUUSD",
                    "account": "SNAP_ACC",
                    "magic": 17,
                    "timeframe": "M5",
                    "open_count": 2,
                    "net_lots": 0.03,
                    "avg_entry": 2201.1,
                    "floating_pnl": 0.7,
                    "equity": 51.2,
                    "free_margin": 48.0,
                },
            )
            pull = client.get("/v1/pull", params={"symbol": "XAUUSD", "account": "SNAP_ACC", "magic": 17, "timeframe": "M5"})
            state = queue.get_symbol_state("SNAP_ACC", 17, "XAUUSD", "M5")

            self.assertEqual(post.status_code, 200)
            self.assertEqual(pull.status_code, 200)
            self.assertEqual(int(state.get("reality_open_count") or 0), 2)
            self.assertAlmostEqual(float(state.get("reality_net_lots") or 0.0), 0.03, places=3)
            self.assertEqual(str(state.get("state_confidence")), "high")

    def test_debug_symbol_returns_recent_decisions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(_action("sig-debug-1"))
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=_FIXED_WEEKDAY_OPEN_UTC):
                client.get("/v1/pull", params={"symbol": "XAUUSD", "account": "DBG", "magic": 3, "timeframe": "M5", "spread_points": 25})
                debug = client.get("/debug/symbol", params={"symbol": "XAUUSD", "limit": 20, "account": "DBG", "magic": 3, "timeframe": "M5"})
            payload = debug.json()

            self.assertEqual(debug.status_code, 200)
            self.assertEqual(payload.get("symbol"), "XAUUSD")
            self.assertGreaterEqual(int(payload.get("history_count", 0)), 1)
            self.assertIsInstance(payload.get("runtime"), dict)

    def test_management_loop_emits_modify_and_close_actions(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            request = ExecutionRequest(
                signal_id="sig-mgmt-mod",
                symbol="XAUUSD",
                side="BUY",
                volume=0.01,
                entry_price=2200.0,
                stop_price=2190.0,
                take_profit_price=2220.0,
                mode="DEMO",
                setup="XAUUSD_M5_GRID_SCALPER_START",
                regime="RANGING",
                probability=0.7,
                expected_value_r=0.4,
                slippage_points=20,
                trailing_enabled=True,
                partial_close_enabled=True,
                news_status="clear",
                final_decision_json="{}",
                trading_enabled=True,
            )
            journal.record_execution(request, OrderResult(True, "1", "accepted", {}), equity=50.0)

            with patch("src.bridge_server.utc_now", return_value=_FIXED_WEEKDAY_OPEN_UTC):
                modify_pull = client.get(
                    "/v1/pull",
                    params={"symbol": "XAUUSD", "account": "MGMT", "magic": 1, "timeframe": "M5", "last": 2215.0, "spread_points": 25},
                )
            modify_actions = modify_pull.json().get("actions", [])
            self.assertEqual(modify_pull.status_code, 200)
            self.assertEqual(len(modify_actions), 1)
            self.assertEqual(modify_actions[0].get("action_type"), "MODIFY_SLTP")
            self.assertGreater(float(modify_actions[0].get("sl") or 0.0), 2190.0)

            journal.record_execution(
                ExecutionRequest(
                    signal_id="sig-mgmt-close",
                    symbol="XAUUSD",
                    side="BUY",
                    volume=0.01,
                    entry_price=2200.0,
                    stop_price=2190.0,
                    take_profit_price=2220.0,
                    mode="DEMO",
                    setup="XAUUSD_M5_GRID_SCALPER_START",
                    regime="RANGING",
                    probability=0.7,
                    expected_value_r=0.4,
                    slippage_points=20,
                    trailing_enabled=True,
                    partial_close_enabled=True,
                    news_status="clear",
                    final_decision_json="{}",
                    trading_enabled=True,
                ),
                OrderResult(True, "2", "accepted", {}),
                equity=50.0,
            )
            with patch("src.bridge_server.utc_now", return_value=_FIXED_WEEKDAY_OPEN_UTC):
                close_pull = client.get(
                    "/v1/pull",
                    params={"symbol": "XAUUSD", "account": "MGMT", "magic": 1, "timeframe": "M5", "last": 2185.0, "spread_points": 25},
                )
            close_actions = close_pull.json().get("actions", [])
            self.assertEqual(close_pull.status_code, 200)
            self.assertEqual(len(close_actions), 1)
            self.assertEqual(close_actions[0].get("action_type"), "CLOSE_POSITION")

    def test_risk_tier_mapping_clamps_daytrade_high_when_confidence_is_not_high(self) -> None:
        class _FakeAIGate:
            def propose_trade_plan(self, context):
                return (
                    {
                        "decision": "TAKE",
                        "setup_type": "daytrade",
                        "side": "BUY",
                        "sl_points": 160,
                        "tp_points": 440,
                        "rr_target": 2.75,
                        "confidence": 0.70,
                        "expected_value_r": 0.35,
                        "risk_tier": "HIGH",
                        "management_plan": {"trail_method": "atr"},
                        "notes": "test_high_tier_should_downgrade",
                    },
                    "remote",
                )

            def propose_management_plan(self, context):
                return ({"decision": "HOLD", "management_plan": {}, "notes": "hold"}, "remote")

            def health(self):
                return {"ok": True, "mode": "remote", "model": "test", "last_error": None}

        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(_action("sig-risk-tier", setup="TREND_DAYTRADE"))
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, ai_gate=_FakeAIGate(), auth_token="")
            client = TestClient(app)
            with patch("src.bridge_server.utc_now", return_value=_FIXED_WEEKDAY_OPEN_UTC):
                response = client.get(
                    "/v1/pull",
                    params={"symbol": "XAUUSD", "account": "RISK", "magic": 42, "timeframe": "M5", "equity": 200.0, "spread_points": 25},
                )
            action = response.json().get("actions", [])[0]
            self.assertEqual(response.status_code, 200)
            self.assertLessEqual(float(action.get("effective_risk_pct") or 0.0), 0.0075 + 1e-9)
            self.assertEqual(str(action.get("risk_tier")), "HIGH")

    def test_starvation_adjustment_exposes_adaptive_floors_within_bounds(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(_action("sig-starve-1"))
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={
                    "undertrading_governor": {
                        "enabled": True,
                        "target_trades_per_hour_xau_grid": 6.0,
                        "target_trades_per_hour_other": 6.0,
                        "adjustment_interval_seconds": 60,
                        "min_interval_minutes": 1,
                        "max_changes_per_day": 10,
                        "spread_loosen_step_pct": 0.05,
                        "max_spread_loosen_pct": 0.20,
                        "min_confluence_relax_step": 0.02,
                        "max_confluence_relax": 0.08,
                    }
                },
            )
            client = TestClient(app)
            fixed_now = _FIXED_WEEKDAY_OPEN_UTC
            queue.last_delivered_at = lambda account, magic, symbol_key: fixed_now - timedelta(minutes=90)  # type: ignore[method-assign]
            queue.hourly_delivered_count = lambda account, magic, symbol_key, window_minutes=60: 0  # type: ignore[method-assign]
            journal.get_open_positions = lambda: []  # type: ignore[method-assign]
            journal.stats = lambda current_equity=0.0: SimpleNamespace(daily_pnl_pct=0.0, rolling_drawdown_pct=0.0)  # type: ignore[method-assign]

            with patch("src.bridge_server.utc_now", return_value=_FIXED_WEEKDAY_OPEN_UTC):
                response = client.get(
                    "/v1/pull",
                    params={"symbol": "XAUUSD", "account": "STARVE", "magic": 5, "timeframe": "M5", "spread_points": 20, "equity": 50.0},
                )
            actions = response.json().get("actions", [])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(actions), 1)
            adaptive = actions[0].get("adaptive_floors", {})
            self.assertGreaterEqual(float(adaptive.get("confluence", 0.0)), 0.30)
            self.assertLessEqual(float(adaptive.get("confluence", 1.0)), 0.55)

    def test_performance_loosen_and_no_trade_filter_disable_after_4h(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(_action("sig-perf-1", symbol="EURUSD", setup="FOREX_TREND_PULLBACK"))
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={
                    "min_probability": 0.68,
                    "min_expected_value_r": 0.45,
                    "min_confluence_score": 0.50,
                    "adaptive_thresholds": {"min_probability_relax_max": 0.10, "min_ev_relax_max": 0.20},
                    "undertrading_governor": {
                        "enabled": True,
                        "target_trades_per_hour_other": 4.0,
                        "adjustment_interval_seconds": 60,
                        "min_interval_minutes": 1,
                        "max_changes_per_day": 20,
                        "spread_loosen_step_pct": 0.05,
                        "max_spread_loosen_pct": 0.20,
                        "min_confluence_relax_step": 0.02,
                        "max_confluence_relax": 0.10,
                        "performance_window_trades": 20,
                        "performance_win_rate_floor": 0.45,
                        "performance_loosen_pct": 0.10,
                        "no_trade_disable_filters_hours": 4,
                    },
                },
            )
            client = TestClient(app)
            fixed_now = _FIXED_WEEKDAY_OPEN_UTC
            queue.last_delivered_at = lambda account, magic, symbol_key: fixed_now - timedelta(hours=5)  # type: ignore[method-assign]
            queue.hourly_delivered_count = lambda account, magic, symbol_key, window_minutes=60: 0  # type: ignore[method-assign]
            journal.get_open_positions = lambda: []  # type: ignore[method-assign]
            journal.stats = lambda current_equity=0.0: SimpleNamespace(daily_pnl_pct=0.0, rolling_drawdown_pct=0.0)  # type: ignore[method-assign]
            online.eval_last = lambda limit: {"trades": 20.0, "win_rate": 0.40, "expectancy_r": -0.1, "profit_factor": 0.8, "max_drawdown_r": 2.0}  # type: ignore[method-assign]

            with patch("src.bridge_server.utc_now", return_value=_FIXED_WEEKDAY_OPEN_UTC):
                response = client.get(
                    "/v1/pull",
                    params={"symbol": "EURUSD", "account": "PERF", "magic": 8, "timeframe": "M5", "spread_points": 12, "equity": 50.0},
                )
            actions = response.json().get("actions", [])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(actions), 1)
            action = actions[0]
            adaptive = action.get("adaptive_floors", {})
            self.assertTrue(bool(action.get("disable_low_weight_filters")))
            self.assertGreaterEqual(float(action.get("performance_loosen_pct", 0.0)), 0.10)
            self.assertLessEqual(float(adaptive.get("min_probability", 1.0)), 0.68)

    def test_bootstrap_no_history_relief_can_activate_without_delivery_history(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(_action("sig-bootstrap-relief", symbol="BTCUSD", setup="BTC_TREND_SCALP"))
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={
                    "min_probability": 0.62,
                    "min_expected_value_r": 0.25,
                    "min_confluence_score": 0.35,
                    "adaptive_thresholds": {"min_probability_relax_max": 0.10, "min_ev_relax_max": 0.20},
                    "undertrading_governor": {
                        "enabled": True,
                        "target_trades_per_hour_other": 2.0,
                        "adjustment_interval_seconds": 60,
                        "min_interval_minutes": 1,
                        "max_changes_per_day": 20,
                    },
                    "shadow_system": {
                        "enabled": True,
                        "no_history_candidate_threshold": 1,
                        "no_history_scan_threshold": 1,
                        "no_history_spread_loosen_pct": 0.12,
                        "no_history_confluence_relax": 0.06,
                        "no_history_probability_relax": 0.06,
                    },
                },
            )
            client = TestClient(app)
            queue.last_delivered_at = lambda account, magic, symbol_key: None  # type: ignore[method-assign]
            queue.hourly_delivered_count = lambda account, magic, symbol_key, window_minutes=60: 0  # type: ignore[method-assign]
            journal.get_open_positions = lambda account=None, magic=None: []  # type: ignore[method-assign]
            journal.stats = lambda current_equity=0.0, account=None, magic=None: SimpleNamespace(daily_pnl_pct=0.0, rolling_drawdown_pct=0.0)  # type: ignore[method-assign]

            response = client.get(
                "/v1/pull",
                params={"symbol": "BTCUSD", "account": "BOOT", "magic": 12, "timeframe": "M15", "spread_points": 25, "equity": 50.0},
            )
            action = response.json().get("actions", [])[0]

            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(response.json().get("actions", [])), 1)
            self.assertLessEqual(float(action.get("adaptive_floors", {}).get("min_probability", 1.0)), 0.59)
            self.assertGreater(float(action.get("shadow_context", {}).get("ranking_bonus", 0.0)), -0.05)

    def test_xau_grid_m5_candidate_can_deliver_on_m15_pull(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(_action("sig-xau-m5-on-m15", setup="XAUUSD_M5_GRID_SCALPER_START"))
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=_FIXED_WEEKDAY_OPEN_UTC):
                response = client.get(
                    "/v1/pull",
                    params={"symbol": "XAUUSD", "account": "XAUA", "magic": 31, "timeframe": "M15", "spread_points": 20},
                )
            payload = response.json()
            state_m5 = queue.get_symbol_state("XAUA", 31, "XAUUSD", "M5")

            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(payload.get("actions", [])), 1)
            self.assertEqual(str(payload["actions"][0].get("setup")), "XAUUSD_M5_GRID_SCALPER_START")
            self.assertEqual(str(state_m5.get("last_action_id")), "sig-xau-m5-on-m15")

    def test_candidate_scan_pending_watchdog_resolves_after_timeout(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            action = _action("sig-pending-watchdog", symbol="BTCUSD", setup="BTC_TREND_SCALP")
            action["reason"] = "candidate_scan_pending"
            queue.enqueue(action)
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            params = {"symbol": "BTCUSD", "account": "BTCA", "magic": 9, "timeframe": "M15", "spread_points": 40}

            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 6, 12, 0, tzinfo=timezone.utc)):
                first = client.get("/v1/pull", params=params)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 6, 12, 0, 31, tzinfo=timezone.utc)):
                second = client.get("/v1/pull", params=params)

            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.json().get("actions"), [])
            self.assertEqual(second.status_code, 200)
            self.assertEqual(len(second.json().get("actions", [])), 1)

    def test_neutral_ai_pass_can_be_approve_small_for_grid(self) -> None:
        class _PassAIGate:
            def propose_trade_plan(self, context):
                return (
                    {
                        "decision": "PASS",
                        "setup_type": "grid_manage",
                        "side": str(context.get("side", "BUY")),
                        "confidence": 0.56,
                        "expected_value_r": 0.22,
                        "management_plan": {"trail_method": "atr"},
                        "notes": "neutral",
                    },
                    "remote",
                )

            def propose_management_plan(self, context):
                return ({"decision": "HOLD", "management_plan": {}, "notes": "hold"}, "remote")

            def health(self):
                return {"ok": True, "mode": "remote", "model": "test", "last_error": None}

        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(_action("sig-approve-small", setup="XAUUSD_M5_GRID_SCALPER_START"))
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, ai_gate=_PassAIGate(), auth_token="")
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 6, 14, 0, tzinfo=timezone.utc)):
                response = client.get(
                    "/v1/pull",
                    params={"symbol": "XAUUSD", "account": "XAUC", "magic": 41, "timeframe": "M5", "spread_points": 18},
                )
            actions = response.json().get("actions", [])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("ai_verdict_class")), "APPROVE_SMALL")
            self.assertEqual(str(actions[0].get("risk_tier")), "LOW")

    def test_weekend_btc_allowed_while_fx_market_closed(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(_action("sig-weekend-fx", symbol="EURUSD", setup="FOREX_TREND_PULLBACK"))
            queue.enqueue(_action("sig-weekend-btc", symbol="BTCUSD", setup="BTC_TREND_SCALP"))
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            saturday = datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)
            with patch("src.bridge_server.utc_now", return_value=saturday):
                fx = client.get("/v1/pull", params={"symbol": "EURUSD", "account": "WK", "magic": 11, "timeframe": "M15", "spread_points": 12})
            with patch("src.bridge_server.utc_now", return_value=saturday):
                btc = client.get("/v1/pull", params={"symbol": "BTCUSD", "account": "WK", "magic": 11, "timeframe": "M15", "spread_points": 55, "equity": 80.0, "free_margin": 80.0})
            self.assertEqual(fx.status_code, 200)
            self.assertEqual(fx.json().get("actions"), [])
            self.assertEqual(btc.status_code, 200)
            self.assertEqual(len(btc.json().get("actions", [])), 1)

    def test_weekend_btc_live_like_spread_points_can_still_deliver(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(
                {
                    **_action("sig-weekend-btc-live", symbol="BTCUSD", setup="BTC_TREND_SCALP"),
                    "entry_price": 95000.0,
                    "sl": 94998.20,
                    "tp": 95004.80,
                    "reason": "btc_live_like",
                    "news_status": "clear",
                    "probability": 0.72,
                    "expected_value_r": 0.42,
                    "confluence_score": 4.0,
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            saturday = datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)
            with patch("src.bridge_server.utc_now", return_value=saturday):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "WKLIVE",
                        "magic": 21,
                        "timeframe": "M15",
                        "spread_points": 1701,
                        "last": 95000.0,
                        "equity": 80.0,
                        "free_margin": 80.0,
                    },
                )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("symbol")), "BTCUSD")

            debug = client.get("/debug/symbol", params={"symbol": "BTCUSD", "account": "WKLIVE", "magic": 21, "timeframe": "M15"})
            self.assertEqual(debug.status_code, 200)
            runtime = debug.json().get("runtime", {})
            self.assertEqual(str(runtime.get("omega_regime")), "VOLATILE_WEEKEND_OK")
            self.assertEqual(str(debug.json().get("latest_block_reason", "")), "")

    def test_weekday_btc_selective_volatile_delivery_can_pass_in_sydney(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(
                {
                    **_action("sig-weekday-btc-sydney", symbol="BTCUSD", setup="BTC_MOMENTUM_CONTINUATION"),
                    "entry_price": 66361.5,
                    "sl": 66559.5,
                    "tp": 66054.0,
                    "reason": "btc_weekday_selective",
                    "news_status": "clear",
                    "probability": 0.80,
                    "expected_value_r": 0.42,
                    "confluence_score": 3.8,
                    "regime": "TRENDING",
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            monday_sydney = datetime(2026, 3, 8, 23, 30, tzinfo=timezone.utc)
            with patch("src.bridge_server.utc_now", return_value=monday_sydney):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "WKDAY",
                        "magic": 22,
                        "timeframe": "M15",
                        "spread_points": 1712,
                        "last": 66361.5,
                        "equity": 57.86,
                        "free_margin": 57.86,
                    },
                )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("symbol")), "BTCUSD")

            debug = client.get("/debug/symbol", params={"symbol": "BTCUSD", "account": "WKDAY", "magic": 22, "timeframe": "M15"})
            self.assertEqual(debug.status_code, 200)
            runtime = debug.json().get("runtime", {})
            self.assertEqual(str(runtime.get("omega_regime")), "VOLATILE_SELECTIVE_OK")

    def test_weekday_btc_selective_volatile_delivery_can_pass_price_action_family_with_moderate_probability(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(
                {
                    **_action("sig-weekday-btc-pa", symbol="BTCUSD", setup="BTC_NY_LIQUIDITY"),
                    "entry_price": 66240.0,
                    "sl": 66045.0,
                    "tp": 66560.0,
                    "reason": "btc_weekday_price_action",
                    "news_status": "clear",
                    "probability": 0.73,
                    "expected_value_r": 0.18,
                    "confluence_score": 2.8,
                    "regime": "VOLATILE",
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            monday_sydney = datetime(2026, 3, 8, 23, 35, tzinfo=timezone.utc)
            with patch("src.bridge_server.utc_now", return_value=monday_sydney):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "WKDAYPA",
                        "magic": 23,
                        "timeframe": "M15",
                        "spread_points": 1720,
                        "last": 66240.0,
                        "equity": 57.86,
                        "free_margin": 57.86,
                    },
                )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("symbol")), "BTCUSD")

            debug = client.get("/debug/symbol", params={"symbol": "BTCUSD", "account": "WKDAYPA", "magic": 23, "timeframe": "M15"})
            self.assertEqual(debug.status_code, 200)
            runtime = debug.json().get("runtime", {})
            self.assertEqual(str(runtime.get("omega_regime")), "VOLATILE_SELECTIVE_OK")
            self.assertTrue(bool(runtime.get("omega_selective_family_allowed")))

    def test_weekday_btc_trend_scalp_is_allowed_under_bootstrap_gate_when_family_is_explicit(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(
                {
                    **_action("sig-weekday-btc-trend", symbol="BTCUSD", setup="BTC_TREND_SCALP"),
                    "entry_price": 66010.0,
                    "sl": 65870.0,
                    "tp": 66295.0,
                    "reason": "btc_weekday_trend_scalp",
                    "news_status": "clear",
                    "probability": 0.81,
                    "expected_value_r": 0.40,
                    "confluence_score": 3.4,
                    "regime": "TRENDING",
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            monday_sydney = datetime(2026, 3, 8, 23, 45, tzinfo=timezone.utc)
            with patch("src.bridge_server.utc_now", return_value=monday_sydney):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "WKDAYTREND",
                        "magic": 24,
                        "timeframe": "M15",
                        "spread_points": 1705,
                        "last": 66010.0,
                        "equity": 57.86,
                        "free_margin": 57.86,
                    },
                )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("symbol")), "BTCUSD")

    def test_weekend_btc_stale_entry_reference_is_reanchored_to_live_pull(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(
                {
                    **_action("sig-weekend-btc-reanchor", symbol="BTCUSD", setup="BTC_TREND_SCALP"),
                    "entry_price": 2379.6,
                    "sl": 2355.48,
                    "tp": 2403.72,
                    "reason": "btc_stale_reference",
                    "news_status": "clear",
                    "probability": 0.81,
                    "expected_value_r": 1.34,
                    "confluence_score": 0.96,
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            saturday = datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)
            with patch("src.bridge_server.utc_now", return_value=saturday):
                response = client.get(
                    "/v1/pull",
                    params={
                        "symbol": "BTCUSD",
                        "account": "WKRE",
                        "magic": 22,
                        "timeframe": "M15",
                        "spread_points": 1701,
                        "last": 95000.0,
                        "equity": 80.0,
                        "free_margin": 80.0,
                    },
                )

            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            action = actions[0]
            self.assertGreater(float(action.get("sl") or 0.0), 94970.0)
            self.assertGreater(float(action.get("tp") or 0.0), 95000.0)
            validation_snapshot = action.get("validation_snapshot", {})
            self.assertTrue(bool(validation_snapshot.get("delivery_reanchored")))

    def test_debug_xau_grid_and_min_lot_reject_diagnostics(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            queue.enqueue(
                {
                    **_action("sig-nas-minlot", symbol="NAS100", setup="NAS_SESSION_SCALPER_ORB"),
                    "lot": 0.001,
                    "entry_price": 18300.0,
                    "sl": 18292.0,
                    "tp": 18320.0,
                    "reason": "nas_test",
                }
            )
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)

            with patch("src.bridge_server.utc_now", return_value=_FIXED_WEEKDAY_OPEN_UTC):
                pull = client.get(
                    "/v1/pull",
                    params={"symbol": "NAS100", "account": "NASM", "magic": 12, "timeframe": "M15", "spread_points": 20, "equity": 500.0},
                )
                debug_symbol = client.get("/debug/symbol", params={"symbol": "NAS100", "account": "NASM", "magic": 12, "timeframe": "M15"})
                debug_xau = client.get("/debug/xau_grid", params={"symbol": "XAUUSD", "account": "NASM", "magic": 12, "timeframe": "M15"})

            self.assertEqual(pull.status_code, 200)
            self.assertEqual(pull.json().get("actions"), [])
            self.assertEqual(debug_symbol.status_code, 200)
            symbol_payload = debug_symbol.json()
            self.assertIn("candidate_pipeline", symbol_payload)
            self.assertIn("latest_block_reason", symbol_payload)
            self.assertEqual(str(symbol_payload.get("latest_block_reason")), "lot_below_min_or_margin_too_low")
            self.assertGreaterEqual(int(symbol_payload.get("candidate_pipeline", {}).get("pre_exec_rejects_last_15m", 0)), 1)
            self.assertGreaterEqual(int(symbol_payload.get("candidate_pipeline", {}).get("min_lot_over_budget_last_15m", 0)), 1)
            self.assertEqual(debug_xau.status_code, 200)
            xau_payload = debug_xau.json()
            self.assertEqual(xau_payload.get("timeframe_used"), "M5")
            self.assertIn("pre_exec_pass", xau_payload)
            self.assertIn("grid_engine_enabled", xau_payload)

    def test_canary_mode_queues_single_xau_trade_through_normal_pull(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            env = {
                "APEX_CANARY_MODE": "1",
                "APEX_CANARY_SYMBOL": "XAUUSD",
                "APEX_CANARY_TF": "M5",
                "APEX_CANARY_SIDE": "BUY",
                "APEX_CANARY_ACCOUNT": "CAN",
                "APEX_CANARY_MAGIC": "77",
                "APEX_CANARY_LIVE": "1",
                "APEX_CANARY_ONCE": "1",
            }
            with patch.dict(os.environ, env, clear=False):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 5, 9, 0, tzinfo=timezone.utc)):
                    response = client.get(
                        "/v1/pull",
                        params={
                            "symbol": "XAUUSD",
                            "account": "CAN",
                            "magic": 77,
                            "timeframe": "M5",
                            "spread_points": 15,
                            "last": 2200.0,
                            "equity": 50.0,
                            "free_margin": 50.0,
                        },
                    )

            payload = response.json()
            self.assertEqual(response.status_code, 200)
            self.assertEqual(len(payload.get("actions", [])), 1)
            action = payload["actions"][0]
            self.assertEqual(str(action.get("setup")), "XAUUSD_M5_CANARY")
            self.assertEqual(str(action.get("symbol")), "XAUUSD")

            debug = client.get("/debug/symbol", params={"symbol": "XAUUSD", "account": "CAN", "magic": 77, "timeframe": "M5"})
            self.assertEqual(debug.status_code, 200)
            pipeline = debug.json().get("candidate_pipeline", {})
            self.assertGreaterEqual(int(pipeline.get("queued_for_ea_last_15m", 0)), 1)
            self.assertGreaterEqual(int(pipeline.get("delivered_actions_last_15m", 0)), 1)

    def test_canary_mode_respects_target_account_and_magic(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            env = {
                "APEX_CANARY_MODE": "1",
                "APEX_CANARY_SYMBOL": "XAUUSD",
                "APEX_CANARY_TF": "M5",
                "APEX_CANARY_SIDE": "BUY",
                "APEX_CANARY_ACCOUNT": "CANX",
                "APEX_CANARY_MAGIC": "99",
                "APEX_CANARY_LIVE": "1",
                "APEX_CANARY_ONCE": "1",
            }
            with patch.dict(os.environ, env, clear=False):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                now = datetime(2026, 3, 5, 9, 0, tzinfo=timezone.utc)
                with patch("src.bridge_server.utc_now", return_value=now):
                    other_response = client.get(
                        "/v1/pull",
                        params={
                            "symbol": "XAUUSD",
                            "account": "MAIN",
                            "magic": 77,
                            "timeframe": "M5",
                            "spread_points": 15,
                            "last": 2200.0,
                            "equity": 50.0,
                            "free_margin": 50.0,
                        },
                    )
                with patch("src.bridge_server.utc_now", return_value=now):
                    target_response = client.get(
                        "/v1/pull",
                        params={
                            "symbol": "XAUUSD",
                            "account": "CANX",
                            "magic": 99,
                            "timeframe": "M5",
                            "spread_points": 15,
                            "last": 2200.0,
                            "equity": 50.0,
                            "free_margin": 50.0,
                        },
                    )

            self.assertEqual(other_response.status_code, 200)
            self.assertEqual(other_response.json().get("actions"), [])
            self.assertEqual(target_response.status_code, 200)
            self.assertEqual(len(target_response.json().get("actions", [])), 1)

    def test_btc_canary_mode_delivers_on_weekend(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            env = {
                "APEX_CANARY_MODE": "1",
                "APEX_CANARY_SYMBOL": "BTCUSD",
                "APEX_CANARY_ACCOUNT": "BTCC",
                "APEX_CANARY_MAGIC": "55",
                "APEX_CANARY_LIVE": "1",
                "APEX_CANARY_ONCE": "1",
            }
            with patch.dict(os.environ, env, clear=False):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                saturday = datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)
                with patch("src.bridge_server.utc_now", return_value=saturday):
                    response = client.get(
                        "/v1/pull",
                        params={
                            "symbol": "BTCUSD",
                            "account": "BTCC",
                            "magic": 55,
                            "timeframe": "M15",
                            "spread_points": 40,
                            "last": 95000.0,
                            "equity": 50.0,
                            "free_margin": 50.0,
                        },
                    )
            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("symbol")), "BTCUSD")
            self.assertIn("CANARY", str(actions[0].get("setup")))

    def test_force_test_trade_mode_queues_single_targeted_trade(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            env = {
                "APEX_FORCE_TEST_TRADE_MODE": "1",
                "APEX_FORCE_TEST_TRADE_SYMBOL": "BTCUSD",
                "APEX_FORCE_TEST_TRADE_ACCOUNT": "FT",
                "APEX_FORCE_TEST_TRADE_MAGIC": "66",
                "APEX_FORCE_TEST_TRADE_LIVE": "1",
                "APEX_FORCE_TEST_TRADE_ONCE": "1",
            }
            with patch.dict(os.environ, env, clear=False):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                saturday = datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)
                with patch("src.bridge_server.utc_now", return_value=saturday):
                    response = client.get(
                        "/v1/pull",
                        params={
                            "symbol": "BTCUSD",
                            "account": "FT",
                            "magic": 66,
                            "timeframe": "M15",
                            "spread_points": 40,
                            "last": 95000.0,
                            "equity": 50.0,
                            "free_margin": 50.0,
                        },
                    )
            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertIn("FORCE_TEST", str(actions[0].get("setup")))

    def test_force_test_mode_alias_envs_queue_single_targeted_trade(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            env = {
                "APEX_FORCE_TEST_MODE": "1",
                "APEX_FORCE_TEST_SYMBOL": "BTCUSD",
                "APEX_FORCE_TEST_ACCOUNT": "FTA",
                "APEX_FORCE_TEST_MAGIC": "67",
                "APEX_FORCE_TEST_LIVE": "1",
                "APEX_FORCE_TEST_ONCE": "1",
                "APEX_FORCE_TEST_TF": "M15",
                "APEX_FORCE_TEST_SIDE": "SELL",
            }
            with patch.dict(os.environ, env, clear=False):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                saturday = datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)
                with patch("src.bridge_server.utc_now", return_value=saturday):
                    response = client.get(
                        "/v1/pull",
                        params={
                            "symbol": "BTCUSD",
                            "account": "FTA",
                            "magic": 67,
                            "timeframe": "M15",
                            "spread_points": 40,
                            "last": 95000.0,
                            "equity": 50.0,
                            "free_margin": 50.0,
                        },
                    )
            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("side")), "SELL")
            self.assertIn("FORCE_TEST", str(actions[0].get("setup")))

    def test_force_test_xau_pull_does_not_crash_on_spread_runtime_update(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            env = {
                "APEX_FORCE_TEST_TRADE_MODE": "1",
                "APEX_FORCE_TEST_TRADE_SYMBOL": "XAUUSD",
                "APEX_FORCE_TEST_TRADE_ACCOUNT": "XFT",
                "APEX_FORCE_TEST_TRADE_MAGIC": "68",
                "APEX_FORCE_TEST_TRADE_LIVE": "1",
                "APEX_FORCE_TEST_TRADE_ONCE": "1",
                "APEX_FORCE_TEST_TRADE_TF": "M5",
            }
            with patch.dict(os.environ, env, clear=False):
                app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
                client = TestClient(app)
                london = datetime(2026, 3, 5, 9, 0, tzinfo=timezone.utc)
                with patch("src.bridge_server.utc_now", return_value=london):
                    response = client.get(
                        "/v1/pull",
                        params={
                            "symbol": "XAUUSD",
                            "account": "XFT",
                            "magic": 68,
                            "timeframe": "M5",
                            "spread_points": 18,
                            "last": 2200.0,
                            "equity": 100.0,
                            "free_margin": 100.0,
                        },
                    )
            self.assertEqual(response.status_code, 200)
            actions = response.json().get("actions", [])
            self.assertEqual(len(actions), 1)
            self.assertEqual(str(actions[0].get("symbol")), "XAUUSD")
            self.assertIn("FORCE_TEST", str(actions[0].get("setup")))

    def test_closed_market_symbols_fail_honestly_without_scan_noise(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            saturday = datetime(2026, 3, 7, 4, 0, tzinfo=timezone.utc)
            with patch("src.bridge_server.utc_now", return_value=saturday):
                response = client.get(
                    "/v1/pull",
                    params={"symbol": "EURUSD", "account": "WK", "magic": 11, "timeframe": "M15", "spread_points": 12, "last": 1.08},
                )
            debug = client.get("/debug/symbol", params={"symbol": "EURUSD", "account": "WK", "magic": 11, "timeframe": "M15"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json().get("actions"), [])
            self.assertEqual(debug.status_code, 200)
            payload = debug.json()
            self.assertEqual(str(payload.get("latest_block_reason")), "market_closed")
            self.assertEqual(int(payload.get("candidate_pipeline", {}).get("scans_last_15m", 0)), 0)
            self.assertGreaterEqual(int(payload.get("candidate_pipeline", {}).get("market_closed_rejects_last_15m", 0)), 1)

    def test_debug_symbol_exposes_timeframe_routing(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            app = create_bridge_app(queue=queue, journal=journal, online_learning=online, auth_token="")
            client = TestClient(app)
            with patch("src.bridge_server.utc_now", return_value=datetime(2026, 3, 5, 9, 0, tzinfo=timezone.utc)):
                client.get(
                    "/v1/pull",
                    params={"symbol": "XAUUSD", "account": "TR", "magic": 12, "timeframe": "M15", "spread_points": 18, "last": 2200.0, "equity": 50.0, "free_margin": 50.0},
                )
            debug = client.get("/debug/symbol", params={"symbol": "XAUUSD", "account": "TR", "magic": 12, "timeframe": "M15"})
            self.assertEqual(debug.status_code, 200)
            routing = debug.json().get("timeframe_routing", {})
            self.assertEqual(str(routing.get("requested_timeframe")), "M15")
            self.assertEqual(str(routing.get("execution_timeframe_used")), "M5")
            self.assertIn("M1", list(routing.get("internal_timeframes_used", [])))
            self.assertTrue(bool(routing.get("attachment_dependency_resolved")))

    def test_unknown_symbol_enters_training_mode_then_activates(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            queue, journal, online = _runtime_fixture(tmp_dir)
            app = create_bridge_app(
                queue=queue,
                journal=journal,
                online_learning=online,
                auth_token="",
                orchestrator_config={
                    "symbol_auto_discovery": {
                        "min_samples": 2,
                        "max_spread_points": 100,
                        "min_pass_score": 0.10,
                    }
                },
            )
            client = TestClient(app)
            params = {
                "symbol": "SOLUSD",
                "account": "DISC",
                "magic": 91,
                "timeframe": "M15",
                "spread_points": 12,
                "last": 100.0,
            }
            with patch("src.bridge_server.utc_now", return_value=_FIXED_WEEKDAY_OPEN_UTC):
                first = client.get("/v1/pull", params=params)
                debug_first = client.get("/debug/symbol", params={"symbol": "SOLUSD"})
                second = client.get("/v1/pull", params=params)
                debug_second = client.get("/debug/symbol", params={"symbol": "SOLUSD"})

            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.json().get("actions"), [])
            self.assertEqual(debug_first.status_code, 200)
            self.assertEqual(
                str(debug_first.json().get("symbol_training_mode", {}).get("status", "")),
                "TRAINING",
            )
            self.assertEqual(second.status_code, 200)
            self.assertEqual(debug_second.status_code, 200)
            self.assertEqual(
                str(debug_second.json().get("symbol_training_mode", {}).get("status", "")),
                "ACTIVE",
            )


if __name__ == "__main__":
    unittest.main()
