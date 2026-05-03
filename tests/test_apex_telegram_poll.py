from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_telegram_polling_sidecar_forwards_updates_to_bridge_webhook() -> None:
    script = (ROOT / "scripts" / "apex_telegram_poll.py").read_text(encoding="utf-8")

    assert "/telegram/webhook" in script
    assert "X-Telegram-Bot-Api-Secret-Token" in script
    assert "get_updates" in script
    assert "delete_webhook(drop_pending_updates=False)" in script
    assert "set_my_commands(BOT_COMMANDS)" in script
    assert '"command": "aggression"' in script


def test_local_telegram_polling_sidecar_can_claim_owner_chat_id() -> None:
    script = (ROOT / "scripts" / "apex_telegram_poll.py").read_text(encoding="utf-8")

    assert "--claim-owner" in script
    assert "_set_secret_key(config.owner_chat_id_env, message.chat_id)" in script
    assert "APEX owner chat linked" in script


def test_local_telegram_polling_sidecar_persists_update_offset() -> None:
    script = (ROOT / "scripts" / "apex_telegram_poll.py").read_text(encoding="utf-8")

    assert "data\" / \"telegram_poll_offset.json" in script
    assert "_write_offset(offset)" in script


def test_telegram_long_poll_transport_timeout_exceeds_poll_timeout() -> None:
    module = (ROOT / "src" / "apex_telegram.py").read_text(encoding="utf-8")

    assert "transport_timeout = max" in module
    assert "int(timeout))) + 10.0" in module
    assert "except socket.timeout" in module


def test_bridge_can_load_claimed_owner_chat_from_local_secret_file() -> None:
    module = (ROOT / "src" / "apex_telegram.py").read_text(encoding="utf-8")

    assert "_secret_file_value(config.owner_chat_id_env)" in module
    assert 'cwd / "config" / "secrets.env"' in module
    assert "os.environ[config.owner_chat_id_env] = chat_id" in module
