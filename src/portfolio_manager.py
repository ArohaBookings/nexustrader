from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

from src.utils import utc_now


@dataclass
class PortfolioDecision:
    allowed: bool
    reason: str
    requires_ai_override: bool
    size_multiplier: float
    correlation_adjustment: float = 1.0
    portfolio_bias: str = "BALANCED"
    exposure_cluster_detected: str = ""


@dataclass
class PortfolioManager:
    max_positions_total: int
    max_positions_per_symbol: int
    max_same_direction: int
    correlation_window_minutes: int

    def assess_new_position(self, open_positions: list[dict], symbol: str, side: str) -> PortfolioDecision:
        if len(open_positions) >= self.max_positions_total:
            return PortfolioDecision(False, "max_positions_total_reached", False, 0.0)

        positions_for_symbol = [position for position in open_positions if str(position.get("symbol", "")).upper() == symbol.upper()]
        if len(positions_for_symbol) >= self.max_positions_per_symbol:
            return PortfolioDecision(False, "max_positions_per_symbol_reached", False, 0.0)

        same_direction_symbol = [position for position in positions_for_symbol if str(position.get("side", "")).upper() == side.upper()]
        if len(same_direction_symbol) >= self.max_same_direction:
            return PortfolioDecision(False, "same_direction_limit", False, 0.0)

        threshold = utc_now() - timedelta(minutes=self.correlation_window_minutes)
        recent_same_direction = []
        for position in same_direction_symbol:
            opened_at = position.get("opened_at")
            if opened_at and opened_at >= threshold:
                recent_same_direction.append(position)
        if len(recent_same_direction) >= 2:
            return PortfolioDecision(False, "correlation_window_limit", False, 0.0)

        requires_ai_override = False
        size_multiplier = 1.0
        portfolio_bias = "BALANCED"
        exposure_cluster = ""
        if self._is_usd_concentrated(open_positions, symbol, side):
            requires_ai_override = True
            size_multiplier *= 0.7

        if self._xau_with_two_usd_pairs(open_positions, symbol):
            requires_ai_override = True
            size_multiplier *= 0.6

        cluster_size = self._cluster_exposure_count(open_positions, symbol=symbol, side=side)
        proposed_cluster_size = cluster_size + 1 if self._cluster_name(symbol, side) else 0
        if proposed_cluster_size >= 3:
            return PortfolioDecision(
                False,
                "correlation_cluster_cap",
                False,
                0.0,
                correlation_adjustment=0.0,
                portfolio_bias=self._portfolio_bias_label(symbol, side),
                exposure_cluster_detected=self._cluster_name(symbol, side),
            )
        if proposed_cluster_size >= 2:
            requires_ai_override = True
            size_multiplier *= 0.65
            exposure_cluster = self._cluster_name(symbol, side)
            portfolio_bias = self._portfolio_bias_label(symbol, side)

        return PortfolioDecision(
            True,
            "ok",
            requires_ai_override,
            size_multiplier,
            correlation_adjustment=size_multiplier,
            portfolio_bias=portfolio_bias,
            exposure_cluster_detected=exposure_cluster,
        )

    @staticmethod
    def _is_usd_concentrated(open_positions: list[dict], symbol: str, side: str) -> bool:
        if not PortfolioManager._is_usd_pair(symbol):
            return False
        same_direction_usd = [
            position
            for position in open_positions
            if PortfolioManager._is_usd_pair(str(position.get("symbol", "")))
            and str(position.get("side", "")).upper() == side.upper()
        ]
        return len(same_direction_usd) >= 2

    @staticmethod
    def _xau_with_two_usd_pairs(open_positions: list[dict], symbol: str) -> bool:
        has_xau = any(PortfolioManager._is_xau(str(position.get("symbol", ""))) for position in open_positions)
        usd_pair_count = sum(1 for position in open_positions if PortfolioManager._is_usd_pair(str(position.get("symbol", ""))))
        if PortfolioManager._is_xau(symbol):
            return False
        return has_xau and usd_pair_count >= 2

    @classmethod
    def _cluster_exposure_count(cls, open_positions: list[dict], *, symbol: str, side: str) -> int:
        cluster_name = cls._cluster_name(symbol, side)
        if not cluster_name:
            return 0
        count = 0
        for position in open_positions:
            if cls._cluster_name(str(position.get("symbol", "")), str(position.get("side", ""))) == cluster_name:
                count += 1
        return count

    @classmethod
    def _portfolio_bias_label(cls, symbol: str, side: str) -> str:
        return cls._cluster_name(symbol, side) or "BALANCED"

    @classmethod
    def _cluster_name(cls, symbol: str, side: str) -> str:
        normalized = cls._normalize_cluster_symbol(symbol)
        direction = str(side or "").upper()
        if direction not in {"BUY", "SELL"}:
            return ""
        if normalized in {"EURUSD", "GBPUSD"}:
            return "USD_BEARISH_CLUSTER" if direction == "BUY" else "USD_BULLISH_CLUSTER"
        if normalized == "USDJPY":
            return "USD_BULLISH_CLUSTER" if direction == "BUY" else "USD_BEARISH_CLUSTER"
        if normalized == "EURGBP":
            return "EURO_STERLING_CLUSTER"
        if normalized in {"XAUUSD", "NAS100", "USOIL"}:
            return "RISK_ON_CLUSTER" if direction == "BUY" else "RISK_OFF_CLUSTER"
        if normalized == "BTCUSD":
            return "CRYPTO_RISK_ON_CLUSTER" if direction == "BUY" else "CRYPTO_RISK_OFF_CLUSTER"
        return ""

    @staticmethod
    def _normalize_cluster_symbol(symbol: str) -> str:
        normalized = "".join(char for char in str(symbol).upper() if char.isalnum())
        if normalized.startswith("EURUSD"):
            return "EURUSD"
        if normalized.startswith("GBPUSD"):
            return "GBPUSD"
        if normalized.startswith("USDJPY"):
            return "USDJPY"
        if normalized.startswith("EURGBP"):
            return "EURGBP"
        if normalized.startswith(("XAUUSD", "GOLD")):
            return "XAUUSD"
        if normalized.startswith(("NAS100", "US100", "USTEC", "NASDAQ", "NQ")):
            return "NAS100"
        if normalized.startswith(("USOIL", "USO", "WTI", "CL", "OIL")):
            return "USOIL"
        if normalized.startswith(("BTCUSD", "BTCUSDT", "XBTUSD")):
            return "BTCUSD"
        return normalized

    @staticmethod
    def _is_xau(symbol: str) -> bool:
        normalized = symbol.upper()
        return normalized.startswith("XAUUSD") or normalized.startswith("GOLD")

    @staticmethod
    def _is_usd_pair(symbol: str) -> bool:
        normalized = symbol.upper()
        compact = "".join(char for char in normalized if char.isalnum())
        if PortfolioManager._is_xau(compact):
            return False
        if compact.startswith("BTCUSD") or compact.startswith("XBTUSD"):
            return True
        if compact.startswith("EURUSD") or compact.startswith("GBPUSD") or compact.startswith("USDJPY"):
            return True
        return len(compact) >= 6 and (compact.startswith("USD") or compact.endswith("USD"))
