from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_mt5_ea_pull_sends_trade_permission_and_connection_flags() -> None:
    source = (ROOT / "mt5_bridge" / "ApexBridgeEA.mq5").read_text(encoding="utf-8")

    assert "terminal_connected=%d" in source
    assert "terminal_trade_allowed=%d" in source
    assert "mql_trade_allowed=%d" in source
    assert "TerminalInfoInteger(TERMINAL_CONNECTED)" in source
    assert "TerminalInfoInteger(TERMINAL_TRADE_ALLOWED)" in source
    assert "MQLInfoInteger(MQL_TRADE_ALLOWED)" in source


def test_mt5_ea_pull_sends_execution_quality_context() -> None:
    source = (ROOT / "mt5_bridge" / "ApexBridgeEA.mq5").read_text(encoding="utf-8")

    for key in (
        "last=%.10f",
        "spread_points=%.2f",
        "tick_size=%.10f",
        "tick_value=%.10f",
        "lot_min=%.4f",
        "lot_max=%.4f",
        "lot_step=%.4f",
        "contract_size=%.4f",
        "stops_level=%d",
        "freeze_level=%d",
        "symbol_trade_mode=%d",
    ):
        assert key in source


def test_mt5_ea_derives_last_price_when_symbol_last_is_missing() -> None:
    source = (ROOT / "mt5_bridge" / "ApexBridgeEA.mq5").read_text(encoding="utf-8")

    assert "double last = SymbolInfoDouble(_Symbol, SYMBOL_LAST);" in source
    assert "last = (bid + ask) * 0.5;" in source
