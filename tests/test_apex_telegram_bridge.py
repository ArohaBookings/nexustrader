from __future__ import annotations

import os
import inspect
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src.bridge_server import BridgeActionQueue, create_bridge_app, run_bridge_forever
from src.execution import TradeJournal
from src.online_learning import OnlineLearningEngine

try:
    from fastapi.testclient import TestClient

    _HAS_FASTAPI = True
except Exception:
    TestClient = None  # type: ignore
    _HAS_FASTAPI = False


class _FakeTelegramClient:
    def __init__(self) -> None:
        self.sent: list[dict[str, str]] = []

    def send_message(self, chat_id: str, text: str, *, parse_mode: str = "HTML", disable_web_page_preview: bool = True) -> dict:
        self.sent.append({"chat_id": str(chat_id), "text": text, "parse_mode": parse_mode})
        return {"message_id": len(self.sent)}


def _make_app(root: Path, *, telegram_config: dict | None = None):
    queue = BridgeActionQueue(db_path=root / "bridge.sqlite", ttl_seconds=10)
    journal = TradeJournal(root / "trades.sqlite")
    online = OnlineLearningEngine(
        data_path=root / "trades.csv",
        model_path=root / "online_model.pkl",
        min_retrain_trades=1,
    )
    return create_bridge_app(
        queue=queue,
        journal=journal,
        online_learning=online,
        auth_token="",
        dashboard_config={"enabled": True, "password": "test-pass", "session_secret": "test-secret"},
        telegram_config=telegram_config or {"enabled": True, "allow_controls": True, "ai_enabled": False},
        risk_config={"funded": {"enabled": True, "starting_balance": 100.0}},
    )


def test_telegram_webhook_sends_status_and_can_pause() -> None:
    if not _HAS_FASTAPI:
        return
    with TemporaryDirectory() as tmp_dir:
        fake = _FakeTelegramClient()
        env = {
            **os.environ,
            "TELEGRAM_WEBHOOK_SECRET": "secret",
            "TELEGRAM_CHAT_ID": "123",
        }
        with patch.dict(os.environ, env, clear=True), patch("src.bridge_server.telegram_client_from_env", return_value=fake):
            client = TestClient(_make_app(Path(tmp_dir)))
            status_payload = {
                "update_id": 1,
                "message": {"chat": {"id": 123}, "from": {"id": 999}, "text": "/status"},
            }
            response = client.post(
                "/telegram/webhook",
                headers={"x-telegram-bot-api-secret-token": "secret"},
                json=status_payload,
            )
            assert response.status_code == 200
            assert response.json()["ok"] is True
            assert "APEX Status" in fake.sent[-1]["text"]

            pause_payload = {
                "update_id": 2,
                "message": {"chat": {"id": 123}, "from": {"id": 999}, "text": "/pause"},
            }
            paused = client.post(
                "/telegram/webhook",
                headers={"x-telegram-bot-api-secret-token": "secret"},
                json=pause_payload,
            )
            assert paused.status_code == 200
            assert paused.json()["action"] == "pause_trading"
            health = client.get("/health").json()
            assert health["operator_control_state"]["pause_trading"] is True


def test_telegram_webhook_fails_closed_without_owner_chat_id() -> None:
    if not _HAS_FASTAPI:
        return
    with TemporaryDirectory() as tmp_dir:
        fake = _FakeTelegramClient()
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in {"TELEGRAM_CHAT_ID", "APEX_TELEGRAM_OWNER_CHAT_ID"}
        }
        env["TELEGRAM_WEBHOOK_SECRET"] = "secret"
        with patch.dict(os.environ, env, clear=True), patch(
            "src.bridge_server.telegram_client_from_env", return_value=fake
        ), patch("src.bridge_server.owner_chat_id_from_env", return_value=""):
            client = TestClient(_make_app(Path(tmp_dir)))
            response = client.post(
                "/telegram/webhook",
                headers={"x-telegram-bot-api-secret-token": "secret"},
                json={"update_id": 1, "message": {"chat": {"id": 456}, "from": {"id": 456}, "text": "/status"}},
            )

            assert response.status_code == 200
            assert response.json()["ok"] is True
            assert "locked until TELEGRAM_CHAT_ID is configured" in fake.sent[-1]["text"]
            assert "APEX Status" not in fake.sent[-1]["text"]


def test_telegram_webhook_blocks_non_owner_controls() -> None:
    if not _HAS_FASTAPI:
        return
    with TemporaryDirectory() as tmp_dir:
        fake = _FakeTelegramClient()
        env = {
            **os.environ,
            "TELEGRAM_WEBHOOK_SECRET": "secret",
            "TELEGRAM_CHAT_ID": "123",
        }
        with patch.dict(os.environ, env, clear=True), patch("src.bridge_server.telegram_client_from_env", return_value=fake):
            client = TestClient(_make_app(Path(tmp_dir)))
            response = client.post(
                "/telegram/webhook",
                headers={"x-telegram-bot-api-secret-token": "secret"},
                json={"update_id": 1, "message": {"chat": {"id": 456}, "from": {"id": 456}, "text": "/pause"}},
            )

            assert response.status_code == 200
            assert "locked to the configured owner chat" in fake.sent[-1]["text"]
            health = client.get("/health").json()
            assert health["operator_control_state"]["pause_trading"] is False


def test_telegram_confirmations_expire() -> None:
    if not _HAS_FASTAPI:
        return
    with TemporaryDirectory() as tmp_dir:
        fake = _FakeTelegramClient()
        env = {
            **os.environ,
            "TELEGRAM_WEBHOOK_SECRET": "secret",
            "TELEGRAM_CHAT_ID": "123",
        }
        now = datetime(2026, 5, 3, 4, 0, tzinfo=timezone.utc)
        with (
            patch.dict(os.environ, env, clear=True),
            patch("src.bridge_server.telegram_client_from_env", return_value=fake),
            patch("src.bridge_server.utc_now", return_value=now),
        ):
            client = TestClient(
                _make_app(
                    Path(tmp_dir),
                    telegram_config={"enabled": True, "allow_controls": True, "ai_enabled": False, "confirmation_ttl_seconds": 30},
                )
            )
            response = client.post(
                "/telegram/webhook",
                headers={"x-telegram-bot-api-secret-token": "secret"},
                json={"update_id": 1, "message": {"chat": {"id": 123}, "from": {"id": 123}, "text": "/kill"}},
            )
            assert response.status_code == 200
            match = re.search(r"/confirm\s+(tg_\d+)", fake.sent[-1]["text"])
            assert match is not None
            command_id = match.group(1)

        with (
            patch.dict(os.environ, env, clear=True),
            patch("src.bridge_server.telegram_client_from_env", return_value=fake),
            patch("src.bridge_server.utc_now", return_value=now + timedelta(seconds=31)),
        ):
            expired = client.post(
                "/telegram/webhook",
                headers={"x-telegram-bot-api-secret-token": "secret"},
                json={"update_id": 2, "message": {"chat": {"id": 123}, "from": {"id": 123}, "text": f"/confirm {command_id}"}},
            )
            assert expired.status_code == 200
            assert "No pending command found" in fake.sent[-1]["text"] or "expired" in fake.sent[-1]["text"]
            health = client.get("/health").json()
            assert health["operator_control_state"]["kill_switch"] is False


def test_run_bridge_forever_accepts_learning_brain_argument() -> None:
    signature = inspect.signature(run_bridge_forever)
    assert "learning_brain" in signature.parameters
