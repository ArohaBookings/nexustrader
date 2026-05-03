from __future__ import annotations

from pathlib import Path
import socket
from tempfile import TemporaryDirectory
import json
from datetime import datetime, timezone
from unittest.mock import patch
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import time

from src.bridge_server import BridgeActionQueue, start_bridge_background
from src.execution import TradeJournal
from src.online_learning import OnlineLearningEngine

def _free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=5) as response:
        payload = response.read().decode("utf-8")
    return json.loads(payload)


def _post_json(url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = Request(url=url, data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(request, timeout=5) as response:
        data = response.read().decode("utf-8")
    return json.loads(data)


def main() -> int:
    class _FakeAIGate:
        def propose_trade_plan(self, context):
            sl_points = float(context.get("min_stop_points", 120.0))
            tp_points = max(sl_points * 1.6, sl_points + 20.0)
            return (
                {
                    "decision": "TAKE",
                    "setup_type": "scalp",
                    "side": str(context.get("side", "BUY")).upper(),
                    "sl_points": sl_points,
                    "tp_points": tp_points,
                    "rr_target": tp_points / max(sl_points, 1e-9),
                    "confidence": 0.83,
                    "expected_value_r": 0.8,
                    "risk_tier": "NORMAL",
                    "management_plan": {
                        "move_sl_to_be_at_r": 0.8,
                        "trail_after_r": 1.0,
                        "trail_method": "atr",
                        "trail_value": 1.0,
                        "take_partial_at_r": 1.0,
                        "time_stop_minutes": 45,
                        "early_exit_rules": "stall_or_reversal",
                    },
                    "notes": "smoke_fake_remote",
                },
                "remote",
            )

        def propose_management_plan(self, context):
            pnl_r = float(context.get("pnl_r") or 0.0)
            decision = "MODIFY" if pnl_r >= 1.0 else "HOLD"
            return (
                {
                    "decision": decision,
                    "confidence": 0.8,
                    "management_plan": {
                        "move_sl_to_be_at_r": 0.8,
                        "trail_after_r": 1.0,
                        "trail_method": "atr",
                        "trail_value": 1.0,
                        "take_partial_at_r": 1.0,
                        "time_stop_minutes": 45,
                        "early_exit_rules": "stall_or_reversal",
                    },
                    "notes": "smoke_manage",
                },
                "remote",
            )

        def health(self):
            return {"ok": True, "mode": "remote", "last_error": None, "model": "smoke-fake"}

    with TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10, open_enqueue_cooldown_seconds=0)
        journal = TradeJournal(root / "trades.sqlite")
        online = OnlineLearningEngine(
            data_path=root / "trades.csv",
            model_path=root / "online_model.pkl",
            min_retrain_trades=1,
        )
        queue.enqueue(
            {
                "signal_id": "smoke-btc-001",
                "action_type": "OPEN_MARKET",
                "symbol": "BTCUSD",
                "side": "BUY",
                "lot": 0.01,
                "sl": 94999.0,
                "tp": 95002.0,
                "max_slippage_points": 40,
                "setup": "BTC_WEEKEND_SMOKE_BREAKOUT",
                "timeframe": "M15",
                "entry_price": 95000.0,
                "reason": "smoke_test",
                "probability": 0.86,
                "expected_value_r": 0.9,
                "confluence_score": 4.2,
                "news_status": "clear",
                "regime": "TRENDING",
            }
        )
        port = _free_port()
        fixed_now = datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc)
        with patch("src.bridge_server.utc_now", return_value=fixed_now):
            handle = start_bridge_background(
                host="127.0.0.1",
                port=port,
                queue=queue,
                journal=journal,
                online_learning=online,
                ai_gate=_FakeAIGate(),
                auth_token="",
                execution_config={
                    "startup_revalidate": False,
                    "startup_warmup_seconds": 0,
                },
                orchestrator_config={
                    "max_budget_usd": 20.0,
                    "nas_strategy": {
                        "enabled": True,
                        "spread_caps_by_session": {"CASH_OPEN": 32.0, "OVERLAP": 36.0, "DEFAULT": 30.0},
                        "trade_rate_targets_by_session": {"CASH_OPEN": 4.0, "DEFAULT": 1.5},
                        "confluence_floor": 0.60,
                        "sessions": {"asia_enabled": True},
                    }
                },
            )
            try:
                for _ in range(30):
                    try:
                        health = _get_json(f"http://127.0.0.1:{port}/health")
                        if bool(health.get("ok", False)):
                            break
                    except Exception:
                        time.sleep(0.1)
                params = urlencode(
                    {
                        "symbol": "BTCUSD",
                        "tf": "M15",
                        "account": "SMOKE_ACC",
                        "magic": 20260304,
                        "spread_points": 25,
                        "balance": 5000.0,
                        "equity": 5000.0,
                        "free_margin": 4900.0,
                        "last": 95000.0,
                    }
                )
                payload = _get_json(f"http://127.0.0.1:{port}/v1/pull?{params}")
                actions = payload.get("actions", [])
                if len(actions) != 1:
                    print("SMOKE FAIL: expected exactly 1 action")
                    print(json.dumps(payload, indent=2, sort_keys=True))
                    return 1
                open_action = actions[0]
                if float(open_action.get("sl", 0.0)) <= 0 or float(open_action.get("tp", 0.0)) <= 0:
                    print("SMOKE FAIL: SL/TP invalid on OPEN action")
                    print(json.dumps(open_action, indent=2, sort_keys=True))
                    return 1
                signal_id = str(open_action.get("signal_id") or "")
                _post_json(
                    f"http://127.0.0.1:{port}/v1/ack",
                    {"signal_id": signal_id, "status": "ACKED", "reason": "smoke_ack"},
                )
                _post_json(
                    f"http://127.0.0.1:{port}/v1/report_execution",
                    {
                        "signal_id": signal_id,
                        "accepted": True,
                        "ticket": "99001",
                        "entry_price": float(open_action.get("entry_price") or 95000.0),
                        "equity": 5000.0,
                    },
                )
                manage_params = urlencode(
                    {
                        "symbol": "BTCUSD",
                        "tf": "M15",
                        "account": "SMOKE_ACC",
                        "magic": 20260304,
                        "spread_points": 25,
                        "last": 95001.5,
                        "equity": 5000.0,
                        "balance": 5000.0,
                        "free_margin": 4900.0,
                    }
                )
                manage_payload = _get_json(f"http://127.0.0.1:{port}/v1/pull?{manage_params}")
            finally:
                handle.stop()
                handle.thread.join(timeout=2)
        management_actions = manage_payload.get("actions", [])
        manage_action = management_actions[0] if management_actions else None
        if manage_action is not None and str(manage_action.get("action_type") or "").upper() != "MODIFY_SLTP":
            print("SMOKE FAIL: expected MODIFY_SLTP when management action is present")
            print(json.dumps(manage_action, indent=2, sort_keys=True))
            return 1
        print("SMOKE PASS")
        print(
            json.dumps(
                {
                    "open_signal_id": signal_id,
                    "open_symbol": open_action.get("symbol"),
                    "open_side": open_action.get("side"),
                    "open_sl": open_action.get("sl"),
                    "open_tp": open_action.get("tp"),
                    "management_action": manage_action.get("action_type") if manage_action else None,
                    "management_sl": manage_action.get("sl") if manage_action else None,
                    "management_signal_id": manage_action.get("signal_id") if manage_action else None,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
