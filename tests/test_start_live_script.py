from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_start_live_routes_through_fail_closed_prod_launcher() -> None:
    script = (ROOT / "scripts" / "start_live.sh").read_text(encoding="utf-8")

    assert "scripts/start_bridge_prod.py" in script
    assert "python -m src.main" not in script
    assert "export APEX_MODE=LIVE" in script


def test_start_live_loads_env_without_shell_evaluating_secret_values() -> None:
    script = (ROOT / "scripts" / "start_live.sh").read_text(encoding="utf-8")

    assert "load_env_file()" in script
    assert "source " not in script
    assert "export \"${key}=${value}\"" in script
    assert "LIVE_TRADING=true is required before live mode can start." in script


def test_start_live_requires_operator_telemetry_env_before_live_start() -> None:
    script = (ROOT / "scripts" / "start_live.sh").read_text(encoding="utf-8")

    assert "required_live_env=(" in script
    assert "TELEGRAM_BOT_TOKEN" in script
    assert "TELEGRAM_CHAT_ID" in script
    assert "TELEGRAM_WEBHOOK_SECRET" in script
    assert "OPENAI_API_KEY" in script
    assert "Secret values were not printed." in script
