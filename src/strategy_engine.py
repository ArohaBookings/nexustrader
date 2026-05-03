from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import pandas as pd

from src.symbol_universe import normalize_symbol_key, symbol_asset_class
from src.utils import deterministic_id


@dataclass
class SignalCandidate:
    signal_id: str
    setup: str
    side: str
    score_hint: float
    reason: str
    stop_atr: float
    tp_r: float
    entry_kind: str = "BASE"
    strategy_family: str = "GENERIC"
    confluence_score: float = 0.0
    confluence_required: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyEngine:
    symbol: str
    entry_method: str = "BREAKOUT_VOLUME_ATR"
    breakout_lookback: int = 20
    min_atr_pct: float = 0.75
    min_volume_ratio: float = 1.0
    max_spread_points: float = 35.0
    allow_long: bool = True
    allow_short: bool = True
    scaling_enabled: bool = True
    scale_in_step_atr: float = 0.6
    pullback_add_enabled: bool = True
    pullback_ema_distance_atr: float = 0.35

    def generate(
        self,
        features: pd.DataFrame,
        regime: str,
        open_positions: list[dict] | None = None,
        max_positions_per_symbol: int = 1,
    ) -> list[SignalCandidate]:
        if features.empty:
            return []
        row = features.iloc[-1]
        timestamp = self._resolve_timestamp(features, row)
        candidates: list[SignalCandidate] = []

        candidates.extend(self._fast_entry(row, regime, timestamp))
        candidates.extend(self._set_forget_h1_h4(row, regime, timestamp))
        if self.scaling_enabled and open_positions:
            candidates.extend(self._scale_in_entries(row, regime, timestamp, open_positions, max_positions_per_symbol))

        unique: list[SignalCandidate] = []
        seen: set[tuple[str, ...]] = set()
        for candidate in sorted(candidates, key=lambda item: item.score_hint, reverse=True):
            key = self._candidate_unique_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        return unique

    @staticmethod
    def _candidate_unique_key(candidate: SignalCandidate) -> tuple[str, ...]:
        meta = candidate.meta if isinstance(candidate.meta, dict) else {}
        if bool(meta.get("grid_cycle")):
            cycle_id = str(meta.get("grid_cycle_id") or "")
            grid_level = str(meta.get("grid_level") or meta.get("grid_burst_index") or "")
            signal_id = str(candidate.signal_id or "")
            return ("GRID", str(candidate.setup), str(candidate.side), cycle_id, grid_level, signal_id)
        return (str(candidate.setup), str(candidate.side))

    @staticmethod
    def _resolve_timestamp(features: pd.DataFrame, row: pd.Series):
        value = row.get("time")
        if pd.notna(value):
            return value
        value = row.get("timestamp")
        if pd.notna(value):
            return value
        if len(features.index) > 0:
            idx_value = features.index[-1]
            if pd.notna(idx_value):
                return idx_value
        return pd.Timestamp.now(tz="UTC")

    def _fast_entry(self, row: pd.Series, regime: str, timestamp) -> Sequence[SignalCandidate]:
        if float(row.get("m5_spread", 0.0)) > self.max_spread_points:
            return []
        if float(row.get("m1_atr_pct_of_avg", 0.0)) < self.min_atr_pct:
            return []
        if float(row.get("m5_atr_pct_of_avg", 0.0)) < (self.min_atr_pct * 0.8):
            return []
        symbol_key = self._normalize_symbol_key(self.symbol)
        m5_atr = max(float(row.get("m5_atr_14", 0.0)), 0.0001)
        m5_close = float(row.get("m5_close", 0.0))
        m5_ema20 = float(row.get("m5_ema_20", m5_close))
        m5_ema50 = float(row.get("m5_ema_50", m5_close))
        m5_volume_ratio = float(row.get("m5_volume_ratio_20", row.get("m15_volume_ratio_20", 1.0)) or 1.0)
        m1_volume_ratio = float(row.get("m1_volume_ratio_20", 1.0) or 1.0)
        m5_body_efficiency = float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)) or 0.55)
        m1_body = float(row.get("m1_body", 0.0))
        m1_impulse = abs(m1_body) / max(float(row.get("m1_atr_14", 0.0)), 0.0001)
        range_position = float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5)
        trend_gap_ratio = abs(m5_ema20 - m5_ema50) / m5_atr
        distance_to_ema = abs(m5_close - m5_ema20)

        if self.entry_method.upper() == "EMA_IMPULSE_BREAKOUT":
            long_ok, short_ok, quality = self._method_b_conditions(row)
            setup = "M1_M5_EMA_IMPULSE"
        else:
            long_ok, short_ok, quality = self._method_a_conditions(row)
            setup = "M1_M5_BREAKOUT"

        if symbol_key in {"EURUSD", "GBPUSD", "USDJPY"}:
            min_volume = 1.06 if symbol_key == "EURUSD" else 1.10
            min_body_efficiency = 0.60 if symbol_key == "EURUSD" else 0.64
            min_trend_gap = 0.42 if symbol_key == "EURUSD" else 0.50
            max_extension = 0.52 if symbol_key == "EURUSD" else 0.46
            if (
                m5_volume_ratio < min_volume
                or m1_volume_ratio < min_volume
                or m5_body_efficiency < min_body_efficiency
                or m1_impulse < 0.72
                or m1_impulse > 1.85
                or trend_gap_ratio < min_trend_gap
                or distance_to_ema > (m5_atr * max_extension)
            ):
                return []
            if long_ok and not (0.54 <= range_position <= 0.90):
                long_ok = False
            if short_ok and not (0.10 <= range_position <= 0.46):
                short_ok = False

        output: list[SignalCandidate] = []
        if self.allow_long and long_ok:
            score = 0.58 + min(0.24, quality + abs(float(row.get("m5_macd_hist_slope", 0.0))) * 8)
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(self.symbol, "fast", "BUY", timestamp),
                    setup=setup,
                    side="BUY",
                    score_hint=score,
                    reason="M1 breakout trigger with M5 momentum confirmation",
                    stop_atr=1.1,
                    tp_r=1.4,
                    entry_kind="BASE",
                    strategy_family="TREND",
                    confluence_score=3.0,
                    confluence_required=5.0,
                )
            )
        if self.allow_short and short_ok:
            score = 0.58 + min(0.24, quality + abs(float(row.get("m5_macd_hist_slope", 0.0))) * 8)
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(self.symbol, "fast", "SELL", timestamp),
                    setup=setup,
                    side="SELL",
                    score_hint=score,
                    reason="M1 downside breakout trigger with M5 momentum confirmation",
                    stop_atr=1.1,
                    tp_r=1.4,
                    entry_kind="BASE",
                    strategy_family="TREND",
                    confluence_score=3.0,
                    confluence_required=5.0,
                )
            )
        return output

    def _set_forget_h1_h4(self, row: pd.Series, regime: str, timestamp) -> Sequence[SignalCandidate]:
        if float(row.get("m5_spread", 0.0)) > self.max_spread_points:
            return []
        session_name = self._resolve_session_name(row, timestamp)
        symbol_key = self._normalize_symbol_key(self.symbol)
        asset_class = symbol_asset_class(symbol_key)
        if asset_class == "crypto":
            allowed_sessions = {"SYDNEY", "TOKYO", "LONDON", "OVERLAP", "NEW_YORK"}
        elif asset_class == "equity":
            allowed_sessions = {"OVERLAP", "NEW_YORK"}
        else:
            allowed_sessions = {"LONDON", "OVERLAP", "NEW_YORK"}
        if session_name and session_name not in allowed_sessions:
            return []
        if symbol_key == "USDJPY" and session_name and session_name not in {"OVERLAP", "NEW_YORK"}:
            return []
        if symbol_key not in {"EURUSD", "USDJPY", "BTCUSD", "XAUUSD"} and asset_class not in {"commodity", "crypto", "index", "equity"}:
            return []

        h1_atr = max(float(row.get("h1_atr_14", row.get("m15_atr_14", 0.0)) or 0.0), 0.0001)
        m15_close = float(row.get("m15_close", 0.0))
        m15_ema20 = float(row.get("m15_ema_20", m15_close) or m15_close)
        h4_gap_ratio = abs(float(row.get("h4_ema_50", 0.0)) - float(row.get("h4_ema_200", 0.0))) / h1_atr
        h1_gap_ratio = abs(float(row.get("h1_ema_20", 0.0)) - float(row.get("h1_ema_50", 0.0))) / h1_atr
        m15_body_efficiency = float(row.get("m15_body_efficiency", row.get("m15_candle_efficiency", 0.55)) or 0.55)
        m15_volume_ratio = float(row.get("m15_volume_ratio_20", row.get("m5_volume_ratio_20", 1.0)) or 1.0)
        distance_to_ema = abs(m15_close - m15_ema20)
        h4_gap_floor = 0.95 if symbol_key == "EURUSD" else 0.75 if symbol_key == "XAUUSD" else 0.55 if asset_class == "crypto" else 0.60 if asset_class == "equity" else 0.70 if asset_class in {"commodity", "index"} else 1.10
        h1_gap_floor = 0.42 if symbol_key == "EURUSD" else 0.45 if symbol_key == "XAUUSD" else 0.30 if asset_class == "crypto" else 0.34 if asset_class == "equity" else 0.38 if asset_class in {"commodity", "index"} else 0.55
        body_floor = 0.57 if symbol_key == "EURUSD" else 0.05 if symbol_key == "XAUUSD" else 0.50 if asset_class == "crypto" else 0.52 if asset_class in {"commodity", "index"} else 0.54 if asset_class == "equity" else 0.61
        volume_floor = 1.03 if symbol_key == "EURUSD" else 0.68 if symbol_key == "XAUUSD" else 0.86 if asset_class == "crypto" else 0.88 if asset_class in {"commodity", "index"} else 0.82 if asset_class == "equity" else 1.08
        distance_cap = 1.35 if symbol_key == "XAUUSD" else 1.55 if asset_class == "crypto" else 0.72 if asset_class == "equity" else 0.95 if asset_class in {"commodity", "index"} else 0.38
        if (
            h4_gap_ratio < h4_gap_floor
            or h1_gap_ratio < h1_gap_floor
            or m15_body_efficiency < body_floor
            or m15_volume_ratio < volume_floor
            or distance_to_ema > (h1_atr * distance_cap)
        ):
            return []
        output: list[SignalCandidate] = []
        long_ok = (
            float(row.get("h4_ema_50", 0.0)) > float(row.get("h4_ema_200", 0.0))
            and float(row.get("h1_ema_20", 0.0)) > float(row.get("h1_ema_50", 0.0))
            and float(row.get("m15_close", 0.0)) >= float(row.get("m15_ema_20", 0.0))
            and 42 <= float(row.get("h1_rsi_14", 50.0)) <= 72
        )
        short_ok = (
            float(row.get("h4_ema_50", 0.0)) < float(row.get("h4_ema_200", 0.0))
            and float(row.get("h1_ema_20", 0.0)) < float(row.get("h1_ema_50", 0.0))
            and float(row.get("m15_close", 0.0)) <= float(row.get("m15_ema_20", 0.0))
            and 28 <= float(row.get("h1_rsi_14", 50.0)) <= 58
        )
        if self.allow_long and long_ok:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(self.symbol, "setforget", "BUY", timestamp),
                    setup="SET_FORGET_H1_H4",
                    side="BUY",
                    score_hint=0.62,
                    reason="H4 trend alignment with H1 structure for set-and-forget leg",
                    stop_atr=1.8,
                    tp_r=2.6,
                    entry_kind="BASE",
                    strategy_family="TREND",
                    confluence_score=4.0,
                    confluence_required=5.0,
                )
            )
        if self.allow_short and short_ok:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(self.symbol, "setforget", "SELL", timestamp),
                    setup="SET_FORGET_H1_H4",
                    side="SELL",
                    score_hint=0.62,
                    reason="H4 downside trend alignment with H1 structure for set-and-forget leg",
                    stop_atr=1.8,
                    tp_r=2.6,
                    entry_kind="BASE",
                    strategy_family="TREND",
                    confluence_score=4.0,
                    confluence_required=5.0,
                )
            )
        return output

    @staticmethod
    def _normalize_symbol_key(value: str) -> str:
        return normalize_symbol_key(value)

    @staticmethod
    def _resolve_session_name(row: pd.Series, timestamp) -> str:
        explicit = str(row.get("session_name", "") or "").upper()
        if explicit:
            return explicit
        return ""

    def _scale_in_entries(
        self,
        row: pd.Series,
        regime: str,
        timestamp,
        open_positions: list[dict],
        max_positions_per_symbol: int,
    ) -> Sequence[SignalCandidate]:
        if not open_positions or len(open_positions) >= max_positions_per_symbol:
            return []
        side = str(open_positions[-1].get("side", "")).upper()
        if side not in {"BUY", "SELL"}:
            return []
        if float(row.get("m5_spread", 0.0)) > self.max_spread_points:
            return []
        atr = max(float(row.get("m5_atr_14", 0.0)), 0.0001)
        m5_close = float(row.get("m5_close", 0.0))
        m5_ema20 = float(row.get("m5_ema_20", m5_close))
        m5_ema50 = float(row.get("m5_ema_50", m5_close))
        trend_ok = (m5_ema20 > m5_ema50) if side == "BUY" else (m5_ema20 < m5_ema50)

        last_entry = float(open_positions[-1].get("entry_price", m5_close))
        step_distance = atr * self.scale_in_step_atr
        if side == "BUY":
            step_ok = m5_close >= (last_entry + step_distance)
            pullback_ok = abs(m5_close - m5_ema20) <= (atr * self.pullback_ema_distance_atr) and m5_close >= m5_ema20
        else:
            step_ok = m5_close <= (last_entry - step_distance)
            pullback_ok = abs(m5_close - m5_ema20) <= (atr * self.pullback_ema_distance_atr) and m5_close <= m5_ema20

        output: list[SignalCandidate] = []
        if trend_ok and step_ok:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(self.symbol, "scale_step", side, timestamp, len(open_positions)),
                    setup="SCALE_IN_STEP",
                    side=side,
                    score_hint=0.60,
                    reason="Scale-in on favorable momentum extension",
                    stop_atr=1.0,
                    tp_r=1.3,
                    entry_kind="SCALE",
                    strategy_family="TREND",
                    confluence_score=3.0,
                    confluence_required=5.0,
                )
            )
        if self.pullback_add_enabled and trend_ok and pullback_ok:
            output.append(
                SignalCandidate(
                    signal_id=deterministic_id(self.symbol, "scale_pullback", side, timestamp, len(open_positions)),
                    setup="SCALE_IN_PULLBACK",
                    side=side,
                    score_hint=0.59,
                    reason="Scale-in on pullback to EMA with trend intact",
                    stop_atr=1.0,
                    tp_r=1.35,
                    entry_kind="SCALE",
                    strategy_family="TREND",
                    confluence_score=3.0,
                    confluence_required=5.0,
                )
            )
        return output

    def _method_a_conditions(self, row: pd.Series) -> tuple[bool, bool, float]:
        prev_high = float(row.get("m1_rolling_high_prev_20", row.get("m1_rolling_high_20", 0.0)))
        prev_low = float(row.get("m1_rolling_low_prev_20", row.get("m1_rolling_low_20", 0.0)))
        m1_close = float(row.get("m1_close", 0.0))
        m1_volume_ratio = float(row.get("m1_volume_ratio_20", 0.0))
        m5_trend_up = float(row.get("m5_ema_20", 0.0)) >= float(row.get("m5_ema_50", 0.0)) and float(row.get("m5_macd_hist", 0.0)) >= 0
        m5_trend_down = float(row.get("m5_ema_20", 0.0)) <= float(row.get("m5_ema_50", 0.0)) and float(row.get("m5_macd_hist", 0.0)) <= 0
        long_ok = m1_close > prev_high and m1_volume_ratio >= self.min_volume_ratio and m5_trend_up
        short_ok = m1_close < prev_low and m1_volume_ratio >= self.min_volume_ratio and m5_trend_down
        quality = min(0.18, max(0.0, (m1_volume_ratio - self.min_volume_ratio) * 0.12))
        return long_ok, short_ok, quality

    def _method_b_conditions(self, row: pd.Series) -> tuple[bool, bool, float]:
        m1_body = float(row.get("m1_body", 0.0))
        m1_atr = max(float(row.get("m1_atr_14", 0.0)), 0.0001)
        impulse = abs(m1_body) / m1_atr
        ema_slope = float(row.get("m1_momentum_3", 0.0))
        prev_high = float(row.get("m1_rolling_high_prev_20", row.get("m1_rolling_high_20", 0.0)))
        prev_low = float(row.get("m1_rolling_low_prev_20", row.get("m1_rolling_low_20", 0.0)))
        m1_close = float(row.get("m1_close", 0.0))
        m5_trend_up = float(row.get("m5_ema_20", 0.0)) >= float(row.get("m5_ema_50", 0.0))
        m5_trend_down = float(row.get("m5_ema_20", 0.0)) <= float(row.get("m5_ema_50", 0.0))
        long_ok = ema_slope > 0 and m1_body > 0 and impulse >= 0.6 and m1_close > prev_high and m5_trend_up
        short_ok = ema_slope < 0 and m1_body < 0 and impulse >= 0.6 and m1_close < prev_low and m5_trend_down
        quality = min(0.2, max(0.0, (impulse - 0.6) * 0.1))
        return long_ok, short_ok, quality
