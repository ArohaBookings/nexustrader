from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class HyperliquidLabConfig:
    project_root: Path
    raw: dict[str, Any]

    @property
    def storage_root(self) -> Path:
        value = str(self.raw.get("storage", {}).get("root", "data/hyperliquid_lab"))
        path = Path(value).expanduser()
        if path.is_absolute():
            return path.resolve()
        return (self.project_root / path).resolve()

    @property
    def assets(self) -> list[str]:
        return [str(item).upper() for item in self.raw.get("assets", ["BTC", "ETH"]) if str(item).strip()]

    @property
    def native_intervals(self) -> list[str]:
        return [str(item) for item in self.raw.get("native_intervals", ["1m", "5m", "15m", "1h"])]

    @property
    def api_base_url(self) -> str:
        return str(self.raw.get("venue", {}).get("api_base_url", "https://api.hyperliquid.xyz")).rstrip("/")

    @property
    def ccxt_exchange_id(self) -> str:
        return str(self.raw.get("venue", {}).get("ccxt_exchange_id", "hyperliquid"))

    @property
    def proxy_source(self) -> str:
        return str(self.raw.get("proxy_history", {}).get("source", "kraken")).strip().lower()

    @property
    def proxy_symbols(self) -> dict[str, str]:
        raw_symbols = self.raw.get("proxy_history", {}).get("symbols", {})
        if not isinstance(raw_symbols, dict):
            return {}
        return {str(key).upper(): str(value) for key, value in raw_symbols.items()}

    @property
    def taker_fee_rate(self) -> float:
        return float(self.raw.get("fees", {}).get("perps", {}).get("taker_rate", 0.00045))

    @property
    def maker_fee_rate(self) -> float:
        return float(self.raw.get("fees", {}).get("perps", {}).get("maker_rate", 0.00015))

    @property
    def simulator_config(self) -> dict[str, Any]:
        value = self.raw.get("simulator", {})
        return dict(value) if isinstance(value, dict) else {}

    @property
    def max_slippage_bps(self) -> float:
        return float(self.simulator_config.get("max_slippage_bps", 10.0))

    @property
    def max_order_book_levels(self) -> int:
        return int(self.simulator_config.get("max_order_book_levels", 20))

    @property
    def min_fill_ratio(self) -> float:
        return float(self.simulator_config.get("min_fill_ratio", 0.50))

    @property
    def stale_book_timeout_seconds(self) -> float:
        return float(self.simulator_config.get("stale_book_timeout_seconds", 5.0))

    @property
    def strategy_config(self) -> dict[str, Any]:
        value = self.raw.get("strategy", {})
        return dict(value) if isinstance(value, dict) else {}

    @property
    def backtest_config(self) -> dict[str, Any]:
        value = self.raw.get("backtest", {})
        return dict(value) if isinstance(value, dict) else {}

    @property
    def walk_forward_config(self) -> dict[str, Any]:
        value = self.raw.get("walk_forward", {})
        return dict(value) if isinstance(value, dict) else {}

    @property
    def risk_config(self) -> dict[str, Any]:
        value = self.raw.get("risk", {})
        return dict(value) if isinstance(value, dict) else {}

    @property
    def paper_config(self) -> dict[str, Any]:
        value = self.raw.get("paper", {})
        return dict(value) if isinstance(value, dict) else {}


def load_lab_config(path: Path | str | None = None) -> HyperliquidLabConfig:
    project_root = Path(__file__).resolve().parents[2]
    config_path = Path(path).expanduser().resolve() if path is not None else project_root / "config" / "hyperliquid_lab.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML mapping in {config_path}")
    return HyperliquidLabConfig(project_root=project_root, raw=payload)
