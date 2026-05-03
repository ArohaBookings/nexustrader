from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_local_mac_start_loads_secrets_without_shell_source() -> None:
    script = (ROOT / "scripts" / "start_local_mac.sh").read_text(encoding="utf-8")

    assert "load_env_file()" in script
    assert "source " not in script
    assert "export \"${key}=${value}\"" in script
    assert "Secret values were not printed." in script


def test_local_mac_start_runs_bridge_and_telegram_polling_sidecar() -> None:
    script = (ROOT / "scripts" / "start_local_mac.sh").read_text(encoding="utf-8")

    assert "python3 -u -m src.main --bridge-only" in script
    assert "scripts/start_bridge_prod.py" in script
    assert "scripts/apex_telegram_poll.py --claim-owner" in script
    assert "APEX_LOCAL_START_STRATEGY" in script
    assert "data/run/apex_bridge.pid" in script
    assert "data/run/apex_telegram_poll.pid" in script
    assert "wait_for_bridge" in script
    assert "screen -L -dmS" in script
    assert "telegram_poller_alive" in script
    assert "tail_log_on_failure" in script


def test_local_mac_stop_stops_bridge_and_poller_pids() -> None:
    script = (ROOT / "scripts" / "stop_local_mac.sh").read_text(encoding="utf-8")

    assert "launchctl bootout" in script
    assert "com.apexbot.bridge" in script
    assert "com.apexbot.telegram" in script
    assert "screen -S \"${session}\" -X quit" in script
    assert "data/run/apex_telegram_poll.pid" in script
    assert "data/run/apex_bridge.pid" in script
    assert "kill \"${pid}\"" in script


def test_local_mac_launch_agents_are_defined_for_persistent_services() -> None:
    install_script = (ROOT / "scripts" / "install_local_mac_launch_agents.sh").read_text(encoding="utf-8")
    bridge_runner = (ROOT / "scripts" / "run_local_bridge_service.sh").read_text(encoding="utf-8")
    telegram_runner = (ROOT / "scripts" / "run_local_telegram_service.sh").read_text(encoding="utf-8")

    assert "com.apexbot.bridge" in install_script
    assert "com.apexbot.telegram" in install_script
    assert "launchctl bootstrap" in install_script
    assert "KeepAlive" in install_script
    assert "scripts/run_local_bridge_service.sh" in install_script
    assert "scripts/run_local_telegram_service.sh" in install_script
    assert "python3 -u -m src.main --bridge-only" in bridge_runner
    assert "scripts/start_bridge_prod.py" in bridge_runner
    assert "scripts/apex_telegram_poll.py --claim-owner" in telegram_runner
