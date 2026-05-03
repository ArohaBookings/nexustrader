from __future__ import annotations

from pathlib import Path

from scripts.start_bridge_prod import (
    _allow_bridge_without_mt5,
    _ea_bridge_mode,
    _live_trading_requested,
    _should_abort_without_mt5,
    _truthy,
)


def test_truthy_accepts_common_live_flags() -> None:
    assert _truthy("true")
    assert _truthy("1")
    assert _truthy("LIVE")
    assert not _truthy("false")
    assert not _truthy("")


def test_live_bridge_start_aborts_without_verified_mt5_by_default() -> None:
    env = {"LIVE_TRADING": "true"}

    assert _live_trading_requested(env)
    assert not _allow_bridge_without_mt5(env)
    assert _should_abort_without_mt5(env)


def test_live_bridge_start_allows_explicit_observation_override() -> None:
    env = {"LIVE_TRADING": "true", "APEX_START_BRIDGE_WITHOUT_MT5": "1"}

    assert _live_trading_requested(env)
    assert _allow_bridge_without_mt5(env)
    assert not _should_abort_without_mt5(env)


def test_ea_bridge_mode_allows_local_mac_bridge_without_mt5_python() -> None:
    env = {"LIVE_TRADING": "true", "APEX_MT5_RUNTIME_MODE": "EA_BRIDGE"}

    assert _ea_bridge_mode(env)
    assert _allow_bridge_without_mt5(env)
    assert not _should_abort_without_mt5(env)


def test_non_live_bridge_can_start_for_dashboard_observation() -> None:
    env = {"LIVE_TRADING": "false", "APEX_MODE": "PAPER"}

    assert not _live_trading_requested(env)
    assert not _should_abort_without_mt5(env)


def test_bridge_launcher_loads_local_secrets_env_before_runtime_checks() -> None:
    script = Path("scripts/start_bridge_prod.py").read_text(encoding="utf-8")

    assert "from src.env_loader import load_env_files" in script
    assert "load_env_files(PROJECT_ROOT)" in script
