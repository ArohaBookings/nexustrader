from __future__ import annotations

from typing import Any


_FOREX_CURRENCIES: set[str] = {
    "AUD",
    "CAD",
    "CHF",
    "EUR",
    "GBP",
    "JPY",
    "NZD",
    "USD",
}

_CRYPTO_BASES: set[str] = {
    "ADA",
    "ARB",
    "ATOM",
    "AVAX",
    "BNB",
    "BTC",
    "DOG",
    "DOGE",
    "DOT",
    "ETH",
    "LINK",
    "LTC",
    "PEPE",
    "SHIB",
    "SOL",
    "SUI",
    "TON",
    "TRUMP",
    "XBT",
    "XRP",
}


def _is_jpy_cross(value: Any) -> bool:
    key = normalize_symbol_key(value)
    return len(key) == 6 and key[:3] in _FOREX_CURRENCIES and key[3:6] == "JPY"


def normalize_symbol_key(value: Any) -> str:
    compact = "".join(char for char in str(value or "").upper() if char.isalnum())
    if not compact:
        return ""
    if compact.startswith(("XAUUSD", "GOLD")):
        return "XAUUSD"
    if compact.startswith(("XAGUSD", "SILVER")):
        return "XAGUSD"
    if compact.startswith(("BTCUSD", "BTCUSDT", "XBTUSD")):
        return "BTCUSD"
    if compact.startswith(("DOGUSD", "DOGEUSD", "DOGEUSDT")):
        return "DOGUSD"
    if compact.startswith(("TRUMPUSD", "TRUMPUSDT")):
        return "TRUMPUSD"
    if compact.startswith(("ETHUSD", "ETHUSDT")):
        return "ETHUSD"
    if compact.startswith(("SOLUSD", "SOLUSDT")):
        return "SOLUSD"
    if compact.startswith(("XRPUSD", "XRPUSDT")):
        return "XRPUSD"
    if compact.startswith(("ADAUSD", "ADAUSDT")):
        return "ADAUSD"
    if compact.startswith(("NAS100", "US100", "NASDAQ", "USTEC", "NAS", "NQ")):
        return "NAS100"
    if compact.startswith(("USOIL", "XTIUSD", "OILUSD", "WTI", "CL", "OIL", "USO")):
        return "USOIL"
    if compact.startswith(("NVIDIA", "NVDA")):
        return "NVIDIA"
    if compact.startswith("AAPL"):
        return "AAPL"
    if len(compact) >= 6 and compact[:3] in _FOREX_CURRENCIES and compact[3:6] in _FOREX_CURRENCIES:
        return compact[:6]
    return compact


def symbol_asset_class(value: Any, metadata: dict[str, Any] | None = None) -> str:
    key = normalize_symbol_key(value)
    meta = dict(metadata or {})
    haystack = " ".join(
        str(meta.get(field) or "")
        for field in ("path", "description", "category", "sector", "exchange", "currency_base", "currency_profit")
    ).upper()
    if len(key) == 6 and key[:3] in _FOREX_CURRENCIES and key[3:6] in _FOREX_CURRENCIES:
        return "forex"
    if key in {"XAUUSD", "XAGUSD", "USOIL"}:
        return "commodity"
    if key == "NAS100" or any(token in haystack for token in ("INDEX", "INDICES", "NASDAQ", "USTEC", "US100", "NQ")):
        return "index"
    if key in {"AAPL", "NVIDIA"}:
        return "equity"
    if key in {"BTCUSD", "DOGUSD", "TRUMPUSD", "ETHUSD", "SOLUSD", "XRPUSD", "ADAUSD"}:
        return "crypto"
    if any(token in haystack for token in ("CRYPTO", "COIN", "TOKEN", "DOGE", "BITCOIN", "ETHEREUM", "MEME")):
        return "crypto"
    if any(token in haystack for token in ("METAL", "GOLD", "SILVER", "COMMODITY", "ENERGY", "OIL", "WTI", "BRENT", "GAS")):
        return "commodity"
    if any(token in haystack for token in ("STOCK", "EQUITY", "SHARE", "24H")):
        return "equity"
    if key.endswith("USD") and key[:-3] in _CRYPTO_BASES:
        return "crypto"
    if key.isalpha() and 1 <= len(key) <= 6 and key not in _FOREX_CURRENCIES:
        return "equity"
    return "other"


def symbol_family_defaults(value: Any, metadata: dict[str, Any] | None = None) -> dict[str, float]:
    asset_class = symbol_asset_class(value, metadata)
    if asset_class == "forex":
        if _is_jpy_cross(value):
            return {
                "point": 0.001,
                "trade_tick_size": 0.001,
                "trade_tick_value": 0.9,
                "trade_contract_size": 100000.0,
                "volume_min": 0.01,
                "volume_max": 100.0,
                "volume_step": 0.01,
            }
        return {
            "point": 0.0001,
            "trade_tick_size": 0.0001,
            "trade_tick_value": 10.0,
            "trade_contract_size": 100000.0,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
        }
    if asset_class == "crypto":
        return {
            "point": 0.01,
            "trade_tick_size": 0.01,
            "trade_tick_value": 0.01,
            "trade_contract_size": 1.0,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
        }
    if asset_class == "commodity":
        return {
            "point": 0.01,
            "trade_tick_size": 0.01,
            "trade_tick_value": 1.0,
            "trade_contract_size": 100.0,
            "volume_min": 0.01,
            "volume_max": 100.0,
            "volume_step": 0.01,
        }
    if asset_class == "index":
        return {
            "point": 0.1,
            "trade_tick_size": 0.1,
            "trade_tick_value": 0.1,
            "trade_contract_size": 1.0,
            "volume_min": 0.01,
            "volume_max": 50.0,
            "volume_step": 0.01,
        }
    if asset_class == "equity":
        return {
            "point": 0.01,
            "trade_tick_size": 0.01,
            "trade_tick_value": 0.01,
            "trade_contract_size": 1.0,
            "volume_min": 0.01,
            "volume_max": 500.0,
            "volume_step": 0.01,
        }
    return {
        "point": 0.01,
        "trade_tick_size": 0.01,
        "trade_tick_value": 1.0,
        "trade_contract_size": 100.0,
        "volume_min": 0.01,
        "volume_max": 100.0,
        "volume_step": 0.01,
    }


def symbol_rule_defaults(value: Any, metadata: dict[str, Any] | None = None) -> dict[str, float]:
    family = symbol_family_defaults(value, metadata)
    asset_class = symbol_asset_class(value, metadata)
    rule = {
        **family,
        "digits": 2,
        "min_stop_points": 45,
        "freeze_points": 20,
        "typical_spread_points": 18,
        "max_slippage_points": 28,
    }
    if asset_class == "forex" and _is_jpy_cross(value):
        rule.update(
            {
                "digits": 3,
                "point": 0.001,
                "trade_tick_size": 0.001,
                "trade_tick_value": 0.9,
                "min_stop_points": 45,
                "freeze_points": 20,
                "typical_spread_points": 16 if normalize_symbol_key(value) == "USDJPY" else 24,
                "max_slippage_points": 25 if normalize_symbol_key(value) == "USDJPY" else 38,
            }
        )
        if normalize_symbol_key(value) in {"AUDJPY", "NZDJPY"}:
            rule["typical_spread_points"] = 22 if normalize_symbol_key(value) == "NZDJPY" else 95
            rule["max_slippage_points"] = 35 if normalize_symbol_key(value) == "NZDJPY" else 120
            rule["min_stop_points"] = 55
        elif normalize_symbol_key(value) == "GBPJPY":
            rule["typical_spread_points"] = 28
            rule["max_slippage_points"] = 45
            rule["min_stop_points"] = 60
        return rule
    if asset_class == "crypto":
        rule.update(
            {
                "digits": 2,
                "min_stop_points": 500,
                "freeze_points": 80,
                "typical_spread_points": 140,
                "max_slippage_points": 160,
            }
        )
    elif asset_class == "index":
        rule.update(
            {
                "digits": 1,
                "point": 0.1,
                "trade_tick_size": 0.1,
                "min_stop_points": 80,
                "freeze_points": 20,
                "typical_spread_points": 65,
                "max_slippage_points": 95,
            }
        )
    elif asset_class == "equity":
        rule.update(
            {
                "digits": 2,
                "min_stop_points": 20,
                "freeze_points": 5,
                "typical_spread_points": 10,
                "max_slippage_points": 20,
            }
        )
    elif asset_class == "commodity":
        rule.update(
            {
                "digits": 2,
                "min_stop_points": 110 if normalize_symbol_key(value) == "XAGUSD" else 140,
                "freeze_points": 30,
                "typical_spread_points": 25 if normalize_symbol_key(value) == "USOIL" else 35,
                "max_slippage_points": 40 if normalize_symbol_key(value) == "USOIL" else 55,
            }
        )
    return rule
