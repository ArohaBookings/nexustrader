from __future__ import annotations

from scripts.start_bridge_prod import (
    _allow_bridge_without_mt5,
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


def test_non_live_bridge_can_start_for_dashboard_observation() -> None:
    env = {"LIVE_TRADING": "false", "APEX_MODE": "PAPER"}

    assert not _live_trading_requested(env)
    assert not _should_abort_without_mt5(env)
