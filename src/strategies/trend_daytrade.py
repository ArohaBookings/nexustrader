from __future__ import annotations


def _normalize_symbol_key(value: str) -> str:
    normalized = "".join(char for char in str(value).upper() if char.isalnum())
    if normalized.startswith("XAUUSD") or normalized.startswith("GOLD"):
        return "XAUUSD"
    if normalized.startswith("BTCUSD") or normalized.startswith("BTCUSDT") or normalized.startswith("XBTUSD"):
        return "BTCUSD"
    if normalized.startswith(("NAS100", "US100", "NASDAQ", "USTEC", "NAS", "NQ")):
        return "NAS100"
    if normalized.startswith(("USOIL", "XTIUSD", "OILUSD", "WTI", "CL", "OIL", "USO")):
        return "USOIL"
    if normalized.startswith("EURUSD"):
        return "EURUSD"
    if normalized.startswith("GBPUSD"):
        return "GBPUSD"
    if normalized.startswith("USDJPY"):
        return "USDJPY"
    if normalized.startswith("AUDJPY"):
        return "AUDJPY"
    if normalized.startswith("NZDJPY"):
        return "NZDJPY"
    if normalized.startswith("AUDNZD"):
        return "AUDNZD"
    if normalized.startswith("EURJPY"):
        return "EURJPY"
    if normalized.startswith("GBPJPY"):
        return "GBPJPY"
    return normalized


def resolve_strategy_key(symbol_key: str, setup: str) -> str:
    normalized_symbol = _normalize_symbol_key(symbol_key)
    setup_upper = str(setup or "").upper()
    if normalized_symbol == "AUDJPY":
        if "SESSION_PULLBACK" in setup_upper or ("PULLBACK" in setup_upper and "LONDON_CARRY" not in setup_upper):
            return "AUDJPY_TOKYO_CONTINUATION_PULLBACK"
        if "SESSION_MOMENTUM" in setup_upper:
            return "AUDJPY_TOKYO_MOMENTUM_BREAKOUT"
        if "SYDNEY_RANGE_BREAK" in setup_upper:
            return "AUDJPY_SYDNEY_RANGE_BREAK"
        if "LONDON_CARRY" in setup_upper:
            return "AUDJPY_LONDON_CARRY_TREND"
        if "ASIA_CONTINUATION_PULLBACK" in setup_upper:
            return "AUDJPY_TOKYO_CONTINUATION_PULLBACK"
        if "ASIA_MOMENTUM_BREAKOUT" in setup_upper:
            return "AUDJPY_TOKYO_MOMENTUM_BREAKOUT"
        if "SWEEP" in setup_upper or "RECLAIM" in setup_upper:
            return "AUDJPY_LIQUIDITY_SWEEP_REVERSAL"
        if "COMPRESSION" in setup_upper or "BREAKOUT_RETEST" in setup_upper:
            return "AUDJPY_ATR_COMPRESSION_BREAKOUT"
        return "AUDJPY_TOKYO_MOMENTUM_BREAKOUT"
    if normalized_symbol == "NZDJPY":
        if "SESSION_PULLBACK" in setup_upper or "PULLBACK" in setup_upper:
            return "NZDJPY_PULLBACK_CONTINUATION"
        if "SESSION_MOMENTUM" in setup_upper:
            return "NZDJPY_TOKYO_BREAKOUT"
        if "SYDNEY_BREAKOUT_RETEST" in setup_upper:
            return "NZDJPY_SESSION_RANGE_EXPANSION"
        if "ASIA_CONTINUATION_PULLBACK" in setup_upper:
            return "NZDJPY_PULLBACK_CONTINUATION"
        if "ASIA_MOMENTUM_BREAKOUT" in setup_upper:
            return "NZDJPY_TOKYO_BREAKOUT"
        if "TRAP" in setup_upper or "SWEEP" in setup_upper or "RECLAIM" in setup_upper:
            return "NZDJPY_LIQUIDITY_TRAP_REVERSAL"
        return "NZDJPY_SESSION_RANGE_EXPANSION"
    if normalized_symbol == "AUDNZD":
        if any(token in setup_upper for token in ("FOREX_RANGE_REVERSION", "RANGE_REVERSION", "MEAN_REVERSION", "VWAP", "FADE")):
            return "AUDNZD_VWAP_MEAN_REVERSION"
        if "ROTATION_BREAKOUT" in setup_upper:
            return "AUDNZD_RANGE_ROTATION"
        if "ROTATION_PULLBACK" in setup_upper:
            return "AUDNZD_RANGE_ROTATION"
        if "RANGE_REJECTION" in setup_upper:
            return "AUDNZD_RANGE_ROTATION"
        if "COMPRESSION_RELEASE" in setup_upper:
            return "AUDNZD_COMPRESSION_RELEASE"
        return "AUDNZD_STRUCTURE_BREAK_RETEST"
    if normalized_symbol == "USDJPY":
        if "SESSION_MOMENTUM" in setup_upper or "M1_M5_BREAKOUT" in setup_upper or "M1_M5_EMA_IMPULSE" in setup_upper:
            return "USDJPY_MOMENTUM_IMPULSE"
        if "TOKYO_LUNCH_FADE" in setup_upper or "SWEEP" in setup_upper or "RECLAIM" in setup_upper:
            return "USDJPY_LIQUIDITY_SWEEP_REVERSAL"
        if "BREAKOUT" in setup_upper or "IMPULSE" in setup_upper:
            return "USDJPY_MOMENTUM_IMPULSE"
        if "PULLBACK" in setup_upper or "CONTINUATION" in setup_upper:
            return "USDJPY_VWAP_TREND_CONTINUATION"
        return "USDJPY_MACRO_TREND_RIDE"
    if normalized_symbol == "EURJPY":
        if "SWEEP" in setup_upper or "RECLAIM" in setup_upper or "LIQUIDITY" in setup_upper:
            return "EURJPY_LIQUIDITY_SWEEP_REVERSAL"
        if "SESSION_PULLBACK" in setup_upper or "PULLBACK" in setup_upper or "CONTINUATION" in setup_upper:
            return "EURJPY_SESSION_PULLBACK_CONTINUATION"
        if any(token in setup_upper for token in ("FOREX_RANGE_REVERSION", "RANGE_REVERSION", "MEAN_REVERSION", "VWAP", "FADE")):
            return "EURJPY_RANGE_FADE"
        if "BREAKOUT" in setup_upper or "IMPULSE" in setup_upper or "MOMENTUM" in setup_upper or "RETEST" in setup_upper:
            return "EURJPY_MOMENTUM_IMPULSE"
        return "EURJPY_MOMENTUM_IMPULSE"
    if normalized_symbol == "GBPJPY":
        if "SWEEP" in setup_upper or "RECLAIM" in setup_upper or "LIQUIDITY" in setup_upper:
            return "GBPJPY_LIQUIDITY_SWEEP_REVERSAL"
        if "SESSION_PULLBACK" in setup_upper or "PULLBACK" in setup_upper or "CONTINUATION" in setup_upper:
            return "GBPJPY_SESSION_PULLBACK_CONTINUATION"
        if any(token in setup_upper for token in ("FOREX_RANGE_REVERSION", "RANGE_REVERSION", "MEAN_REVERSION", "VWAP", "FADE")):
            return "GBPJPY_RANGE_FADE"
        if "BREAKOUT" in setup_upper or "IMPULSE" in setup_upper or "MOMENTUM" in setup_upper or "RETEST" in setup_upper:
            return "GBPJPY_MOMENTUM_IMPULSE"
        return "GBPJPY_MOMENTUM_IMPULSE"
    if normalized_symbol == "EURUSD":
        if "SWEEP" in setup_upper or "RECLAIM" in setup_upper:
            return "EURUSD_LIQUIDITY_SWEEP"
        if "PULLBACK" in setup_upper or "TREND_PULLBACK" in setup_upper:
            return "EURUSD_VWAP_PULLBACK"
        if "VWAP" in setup_upper:
            return "EURUSD_VWAP_PULLBACK"
        if "RANGE" in setup_upper or "FADE" in setup_upper:
            return "EURUSD_RANGE_FADE"
        return "EURUSD_LONDON_BREAKOUT"
    if normalized_symbol == "GBPUSD":
        if "STOP_HUNT" in setup_upper or "SWEEP" in setup_upper or "RECLAIM" in setup_upper:
            return "GBPUSD_STOP_HUNT_REVERSAL"
        if "PULLBACK" in setup_upper:
            return "GBPUSD_TREND_PULLBACK_RIDE"
        if "ATR" in setup_upper or "EXPANSION" in setup_upper:
            return "GBPUSD_ATR_EXPANSION_SCALPER"
        return "GBPUSD_LONDON_EXPANSION_BREAKOUT"
    if normalized_symbol == "XAUUSD":
        if "GRID" in setup_upper:
            return "XAUUSD_ADAPTIVE_M5_GRID"
        if "VWAP" in setup_upper:
            return "XAUUSD_VWAP_REVERSION"
        if any(token in setup_upper for token in ("LIQUIDITY", "SWEEP", "SMC", "ORDERBLOCK", "FVG", "FAKEOUT")):
            return "XAUUSD_LONDON_LIQUIDITY_SWEEP"
        if any(token in setup_upper for token in ("MOMENTUM", "BREAKOUT", "EXPANSION")):
            return "XAUUSD_NY_MOMENTUM_BREAKOUT"
        return "XAUUSD_ATR_EXPANSION_SCALPER"
    if normalized_symbol == "NAS100":
        if "PREMARKET" in setup_upper or "SCALPER_ORB" in setup_upper or "ORB" in setup_upper:
            return "NAS100_OPENING_DRIVE_BREAKOUT"
        if "VWAP_MR" in setup_upper or "SCALPER_VWAP_MR" in setup_upper:
            return "NAS100_LIQUIDITY_SWEEP_REVERSAL"
        if "OPENING" in setup_upper or "DRIVE" in setup_upper:
            return "NAS100_OPENING_DRIVE_BREAKOUT"
        if "VWAP" in setup_upper:
            return "NAS100_VWAP_TREND_STRATEGY"
        if "SWEEP" in setup_upper or "RECLAIM" in setup_upper:
            return "NAS100_LIQUIDITY_SWEEP_REVERSAL"
        return "NAS100_MOMENTUM_IMPULSE"
    if normalized_symbol == "USOIL":
        if "INVENTORY" in setup_upper:
            return "USOIL_INVENTORY_MOMENTUM"
        if "VWAP" in setup_upper:
            return "USOIL_VWAP_REVERSION"
        if "BREAKOUT" in setup_upper or "RETEST" in setup_upper:
            return "USOIL_BREAKOUT_RETEST"
        return "USOIL_LONDON_TREND_EXPANSION"
    if normalized_symbol == "BTCUSD":
        if any(
            token in setup_upper
            for token in (
                "WEEKEND_BREAKOUT_RECLAIM",
                "LIQUIDITY_SWEEP",
                "SWEEP_FADE",
                "LIQUIDATION",
                "GAP_FADE",
                "BREAKOUT_RECLAIM",
                "EXHAUSTION_REVERSION",
            )
        ):
            return "BTCUSD_VOLATILE_RETEST"
        if any(token in setup_upper for token in ("WEEKEND_BREAKOUT", "SESSION_EXPANSION", "WHALE_FLOW_BREAKOUT")):
            return "BTCUSD_RANGE_EXPANSION"
        if "RANGE" in setup_upper or "EXPANSION" in setup_upper:
            return "BTCUSD_RANGE_EXPANSION"
        if any(
            token in setup_upper
            for token in ("PRICE_ACTION", "CONTINUATION", "MOMENTUM_CONTINUATION", "NY_LIQUIDITY", "FUNDING_ARB", "DXY_LAG_ARB")
        ):
            return "BTCUSD_PRICE_ACTION_CONTINUATION"
        if "VOLATILE" in setup_upper or "RETEST" in setup_upper:
            return "BTCUSD_VOLATILE_RETEST"
        if "SCALP" in setup_upper:
            return "BTCUSD_TREND_SCALP"
        return "BTCUSD_PRICE_ACTION_CONTINUATION"
    return f"{normalized_symbol}_MULTI"
