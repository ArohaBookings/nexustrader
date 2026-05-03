from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import hashlib
import math
import random

import numpy as np
import pandas as pd

from src.ai_gate import AIGate
from src.regime_detector import RegimeDetector
from src.strategy_engine import StrategyEngine


def enforce_plausibility(
    metrics: dict[str, Any],
    *,
    max_win_rate: float = 0.85,
    min_trades: int = 200,
    whitelisted: bool = False,
) -> None:
    if whitelisted:
        return
    trade_count = int(metrics.get("trade_count", 0) or 0)
    win_rate = float(metrics.get("win_rate", 0.0) or 0.0)
    if trade_count > min_trades and win_rate > max_win_rate:
        raise ValueError(
            f"suspicious_backtest_win_rate win_rate={win_rate:.4f} "
            f"trade_count={trade_count} threshold={max_win_rate:.2f}"
        )


@dataclass
class BacktestTrade:
    signal_id: str
    symbol: str
    setup: str
    strategy_key: str
    session_name: str
    regime_state: str
    strategy_state: str
    lane_name: str
    entry_timing_score: float
    structure_cleanliness_score: float
    execution_quality_fit: float
    entry_kind: str
    side: str
    opened_at: str
    closed_at: str
    scale_level: int
    r_multiple: float
    result: str
    exit_reason: str = ""
    mae_r: float = 0.0
    mfe_r: float = 0.0
    duration_minutes: float = 0.0


@dataclass
class SimPosition:
    signal_id: str
    symbol: str
    setup: str
    strategy_key: str
    session_name: str
    regime_state: str
    strategy_state: str
    lane_name: str
    entry_timing_score: float
    structure_cleanliness_score: float
    execution_quality_fit: float
    entry_kind: str
    side: str
    opened_at: str
    entry_price: float
    stop_price: float
    take_profit_price: float
    initial_risk: float
    volume: float
    remaining_volume: float
    scale_level: int
    be_trigger_r: float
    trail_start_r: float
    trail_atr_mult: float
    partial1_r: float
    partial1_fraction: float
    partial2_r: float
    partial2_fraction: float
    basket_take_profit_r: float
    be_buffer_r: float
    min_profit_protection_r: float
    trail_backoff_r: float
    trail_requires_partial1: bool
    no_progress_bars: int
    no_progress_mfe_r: float
    early_invalidation_r: float
    time_stop_bars: int
    time_stop_max_r: float
    realized_r: float = 0.0
    be_moved: bool = False
    partial1_done: bool = False
    partial2_done: bool = False
    trail_active: bool = False
    bars_open: int = 0
    mfe_r: float = 0.0
    mae_r: float = 0.0


def label_trade_outcome(
    frame: pd.DataFrame,
    start_index: int,
    side: str,
    entry_price: float,
    stop_price: float,
    tp_price: float,
    max_bars: int = 48,
) -> tuple[float, int]:
    risk = abs(entry_price - stop_price)
    if risk <= 0:
        return 0.0, start_index
    direction = 1 if side.upper() == "BUY" else -1
    end_index = min(len(frame) - 1, start_index + max_bars)
    for index in range(start_index + 1, end_index + 1):
        row = frame.iloc[index]
        high = float(row["m5_high"])
        low = float(row["m5_low"])
        if side.upper() == "BUY":
            stop_hit = low <= stop_price
            tp_hit = high >= tp_price
        else:
            stop_hit = high >= stop_price
            tp_hit = low <= tp_price
        if stop_hit and tp_hit:
            return -1.0, index
        if stop_hit:
            return -1.0, index
        if tp_hit:
            return abs(tp_price - entry_price) / risk, index
    close = float(frame.iloc[end_index]["m5_close"])
    pnl = (close - entry_price) * direction
    return pnl / risk, end_index


@dataclass
class Backtester:
    strategy_engine: StrategyEngine
    regime_detector: RegimeDetector
    ai_gate: AIGate
    spread_points: float
    slippage_points: float
    commission_per_lot: float
    latency_ms: int = 0
    min_fill_ratio: float = 0.60
    max_positions_per_symbol: int = 10
    max_positions_total: int = 20
    be_trigger_r: float = 0.7
    be_buffer_r: float = 0.05
    trail_start_r: float = 1.0
    trail_atr_mult: float = 1.0
    partial1_r: float = 1.0
    partial1_fraction: float = 0.4
    partial2_r: float = 2.0
    partial2_fraction: float = 0.3
    basket_tp_r: float = 3.0
    close_on_signal_flip: bool = True
    use_fixed_lot: bool = False
    fixed_lot: float = 0.01
    risk_per_trade: float = 0.0025
    initial_equity: float = 1000.0
    contract_size: float = 100.0
    strict_plausibility: bool = True
    max_plausible_win_rate: float = 0.85
    min_trades_for_plausibility: int = 200
    whitelist_high_win_rate: bool = False
    scale_factors: tuple[float, ...] = (1.0, 0.7, 0.5, 0.4, 0.35, 0.3, 0.25, 0.2, 0.2, 0.2)

    def run(self, frame: pd.DataFrame) -> dict[str, Any]:
        if frame.empty:
            return {"trade_count": 0, "trades": [], "monte_carlo": {}, "events": {}}
        required_columns = {"time", "m5_open", "m5_high", "m5_low", "m5_close", "m5_atr_14"}
        if not required_columns.issubset(set(frame.columns)):
            raise ValueError(f"Backtest frame missing required columns: {sorted(required_columns - set(frame.columns))}")

        trades: list[BacktestTrade] = []
        r_results: list[float] = []
        events = {
            "entries": 0,
            "entry_rejected_fill": 0,
            "scale_in_added": 0,
            "sl_to_be": 0,
            "trail_updates": 0,
            "partials_tp1": 0,
            "partials_tp2": 0,
            "basket_closes": 0,
            "signal_flip_closes": 0,
        }

        open_positions: list[SimPosition] = []
        equity = self.initial_equity
        warmup = min(max(260, int(len(frame) * 0.1)), max(len(frame) - 2, 1))
        point_size = self._infer_point_size(frame)
        bar_ms = self._infer_bar_ms(frame)
        latency_bars = 0 if self.latency_ms <= 0 else int(math.ceil(self.latency_ms / max(bar_ms, 1)))

        for index in range(warmup, len(frame) - 1):
            row = frame.iloc[index]
            next_row = frame.iloc[index + 1]
            price_now_mid = float(row["m5_close"])
            spread_points_now = self._effective_spread_points(row)
            atr = max(float(row["m5_atr_14"]), 0.0001)

            self._manage_open_positions(
                row=row,
                next_row=next_row,
                open_positions=open_positions,
                trades=trades,
                r_results=r_results,
                events=events,
                point_size=point_size,
            )

            basket_r = self._basket_profit_r(open_positions, price_now_mid, spread_points_now, point_size)
            basket_take_profit_r = self._active_basket_take_profit_r(open_positions)
            if open_positions and basket_r >= basket_take_profit_r:
                self._close_all_positions(
                    open_positions=open_positions,
                    close_mid_price=price_now_mid,
                    spread_points=spread_points_now,
                    point_size=point_size,
                    closed_at=str(row["time"]),
                    trades=trades,
                    r_results=r_results,
                    reason="BASKET_CLOSE",
                )
                events["basket_closes"] += 1

            regime = self.regime_detector.classify(row)
            open_view = [
                {
                    "signal_id": position.signal_id,
                    "symbol": position.symbol,
                    "setup": position.setup,
                    "side": position.side,
                    "opened_at": position.opened_at,
                    "entry_price": position.entry_price,
                    "volume": position.remaining_volume,
                    "sl": position.stop_price,
                    "tp": position.take_profit_price,
                }
                for position in open_positions
            ]
            # Slice to the current closed candle only (no future bars available to signal generation).
            candidates = self.strategy_engine.generate(
                frame.iloc[: index + 1],
                regime.label,
                open_positions=open_view,
                max_positions_per_symbol=self.max_positions_per_symbol,
            )
            if not candidates:
                continue

            for candidate in candidates:
                if len(open_positions) >= self.max_positions_per_symbol or len(open_positions) >= self.max_positions_total:
                    break

                decision = self.ai_gate.evaluate(candidate, row, regime.label, consecutive_losses=self._recent_loss_streak(r_results))
                if not decision.approved:
                    continue

                opposite_open = [position for position in open_positions if position.side != candidate.side]
                if opposite_open and self.close_on_signal_flip and decision.probability >= 0.67:
                    self._close_all_positions(
                        open_positions=open_positions,
                        close_mid_price=price_now_mid,
                        spread_points=spread_points_now,
                        point_size=point_size,
                        closed_at=str(row["time"]),
                        trades=trades,
                        r_results=r_results,
                        reason="SIGNAL_FLIP",
                    )
                    events["signal_flip_closes"] += len(opposite_open)

                execute_index = min(index + 1 + latency_bars, len(frame) - 1)
                if execute_index <= index:
                    continue
                execute_row = frame.iloc[execute_index]
                entry_mid = float(execute_row["m5_open"])
                entry_spread_points = self._effective_spread_points(execute_row)
                entry_price = self._entry_fill_price(candidate.side, entry_mid, entry_spread_points, point_size)

                stop_distance = atr * max(0.2, candidate.stop_atr * decision.sl_multiplier)
                if stop_distance <= 0:
                    continue
                if candidate.side == "BUY":
                    stop_price = entry_price - stop_distance
                    tp_price = entry_price + (stop_distance * decision.tp_r)
                else:
                    stop_price = entry_price + stop_distance
                    tp_price = entry_price - (stop_distance * decision.tp_r)

                scale_level = len(open_positions)
                size_factor = self._scale_factor(scale_level)
                requested_volume = self._position_volume(equity, stop_distance, size_factor)
                if requested_volume <= 0:
                    continue

                fill_ratio = self._entry_fill_ratio(candidate.signal_id, execute_index, row)
                if fill_ratio < self.min_fill_ratio:
                    events["entry_rejected_fill"] += 1
                    continue
                volume = requested_volume * fill_ratio
                if volume <= 0:
                    events["entry_rejected_fill"] += 1
                    continue

                position_symbol = str(
                    (candidate.meta or {}).get("symbol")
                    or getattr(self.strategy_engine, "symbol", "")
                    or "SIM"
                )

                open_positions.append(
                    SimPosition(
                        signal_id=candidate.signal_id,
                        symbol=position_symbol,
                        setup=candidate.setup,
                        strategy_key=str((candidate.meta or {}).get("strategy_key") or ""),
                        session_name=str((candidate.meta or {}).get("session_name") or ""),
                        regime_state=str((candidate.meta or {}).get("regime_state") or regime.label),
                        strategy_state=str((candidate.meta or {}).get("strategy_state") or "NORMAL"),
                        lane_name=str((candidate.meta or {}).get("lane_name") or ""),
                        entry_timing_score=float((candidate.meta or {}).get("entry_timing_score", 0.0) or 0.0),
                        structure_cleanliness_score=float((candidate.meta or {}).get("structure_cleanliness_score", 0.0) or 0.0),
                        execution_quality_fit=float((candidate.meta or {}).get("execution_quality_fit", 0.0) or 0.0),
                        entry_kind=candidate.entry_kind,
                        side=candidate.side,
                        opened_at=str(execute_row["time"]),
                        entry_price=entry_price,
                        stop_price=stop_price,
                        take_profit_price=tp_price,
                        initial_risk=stop_distance,
                        volume=volume,
                        remaining_volume=volume,
                        scale_level=scale_level,
                        be_trigger_r=float((candidate.meta or {}).get("breakeven_trigger_r", self.be_trigger_r) or self.be_trigger_r),
                        trail_start_r=float((candidate.meta or {}).get("trail_activation_r", self.trail_start_r) or self.trail_start_r),
                        trail_atr_mult=float((candidate.meta or {}).get("trail_atr", self.trail_atr_mult) or self.trail_atr_mult),
                        partial1_r=float((candidate.meta or {}).get("partial1_r", self.partial1_r) or self.partial1_r),
                        partial1_fraction=float((candidate.meta or {}).get("partial1_fraction", self.partial1_fraction) or self.partial1_fraction),
                        partial2_r=float((candidate.meta or {}).get("partial2_r", self.partial2_r) or self.partial2_r),
                        partial2_fraction=float((candidate.meta or {}).get("partial2_fraction", self.partial2_fraction) or self.partial2_fraction),
                        basket_take_profit_r=float((candidate.meta or {}).get("basket_take_profit_r", self.basket_tp_r) or self.basket_tp_r),
                        be_buffer_r=float((candidate.meta or {}).get("be_buffer_r", self.be_buffer_r) or self.be_buffer_r),
                        min_profit_protection_r=float((candidate.meta or {}).get("min_profit_protection_r", 0.0) or 0.0),
                        trail_backoff_r=float((candidate.meta or {}).get("trail_backoff_r", 0.55) or 0.55),
                        trail_requires_partial1=bool((candidate.meta or {}).get("trail_requires_partial1", False)),
                        no_progress_bars=max(0, int((candidate.meta or {}).get("no_progress_bars", 0) or 0)),
                        no_progress_mfe_r=float((candidate.meta or {}).get("no_progress_mfe_r", 0.0) or 0.0),
                        early_invalidation_r=float((candidate.meta or {}).get("early_invalidation_r", -1.0) or -1.0),
                        time_stop_bars=max(0, int((candidate.meta or {}).get("time_stop_bars", 0) or 0)),
                        time_stop_max_r=float((candidate.meta or {}).get("time_stop_max_r", 0.0) or 0.0),
                    )
                )
                events["entries"] += 1
                if candidate.entry_kind in {"SCALE", "GRID_ADD"}:
                    events["scale_in_added"] += 1

            if r_results:
                equity = self.initial_equity + (sum(r_results) * self.initial_equity * self.risk_per_trade)

        if open_positions:
            close_row = frame.iloc[-1]
            self._close_all_positions(
                open_positions=open_positions,
                close_mid_price=float(close_row["m5_close"]),
                spread_points=self._effective_spread_points(close_row),
                point_size=point_size,
                closed_at=str(close_row["time"]),
                trades=trades,
                r_results=r_results,
                reason="END_OF_TEST",
            )

        metrics = self._metrics(trades, r_results, events)
        if self.strict_plausibility:
            enforce_plausibility(
                metrics,
                max_win_rate=self.max_plausible_win_rate,
                min_trades=self.min_trades_for_plausibility,
                whitelisted=self.whitelist_high_win_rate,
            )
        return metrics

    def _manage_open_positions(
        self,
        row: pd.Series,
        next_row: pd.Series,
        open_positions: list[SimPosition],
        trades: list[BacktestTrade],
        r_results: list[float],
        events: dict[str, int],
        point_size: float,
    ) -> None:
        if not open_positions:
            return
        current_mid = float(row["m5_close"])
        current_spread_points = self._effective_spread_points(row)
        next_spread_points = self._effective_spread_points(next_row)
        atr = max(float(row["m5_atr_14"]), 0.0001)
        next_high = float(next_row["m5_high"])
        next_low = float(next_row["m5_low"])
        closed_at = str(next_row["time"])

        for position in list(open_positions):
            current_exec_price = self._market_exit_price(position, current_mid, current_spread_points, point_size)
            profit_r = self._position_profit_r(position, current_exec_price)
            if position.side == "BUY":
                best_exec_price = next_high - (self._points_to_price(next_spread_points, point_size) / 2.0)
                worst_exec_price = next_low - (self._points_to_price(next_spread_points, point_size) / 2.0)
            else:
                best_exec_price = next_low + (self._points_to_price(next_spread_points, point_size) / 2.0)
                worst_exec_price = next_high + (self._points_to_price(next_spread_points, point_size) / 2.0)
            position.mfe_r = max(float(position.mfe_r), float(self._position_profit_r(position, best_exec_price)))
            position.mae_r = min(float(position.mae_r), float(self._position_profit_r(position, worst_exec_price)))
            position.bars_open += 1
            if (
                position.no_progress_bars > 0
                and position.bars_open >= position.no_progress_bars
                and position.mfe_r < position.no_progress_mfe_r
                and profit_r <= position.early_invalidation_r
            ):
                result_r = self._close_position(position, current_exec_price)
                r_results.append(result_r)
                trades.append(
                    BacktestTrade(
                        signal_id=position.signal_id,
                        symbol=position.symbol,
                        setup=position.setup,
                        strategy_key=position.strategy_key,
                        session_name=position.session_name,
                        regime_state=position.regime_state,
                        strategy_state=position.strategy_state,
                        lane_name=position.lane_name,
                        entry_timing_score=position.entry_timing_score,
                        structure_cleanliness_score=position.structure_cleanliness_score,
                        execution_quality_fit=position.execution_quality_fit,
                        entry_kind=position.entry_kind,
                        side=position.side,
                        opened_at=position.opened_at,
                        closed_at=closed_at,
                        scale_level=position.scale_level,
                        r_multiple=result_r,
                        result="WIN" if result_r > 0 else "LOSS",
                        exit_reason="NO_PROGRESS_INVALIDATION",
                        mae_r=float(position.mae_r),
                        mfe_r=float(position.mfe_r),
                        duration_minutes=float(position.bars_open * 5),
                    )
                )
                open_positions.remove(position)
                continue
            if (
                position.time_stop_bars > 0
                and position.bars_open >= position.time_stop_bars
                and profit_r <= position.time_stop_max_r
                and position.mfe_r < max(position.no_progress_mfe_r + 0.10, position.time_stop_max_r + 0.18)
            ):
                result_r = self._close_position(position, current_exec_price)
                r_results.append(result_r)
                trades.append(
                    BacktestTrade(
                        signal_id=position.signal_id,
                        symbol=position.symbol,
                        setup=position.setup,
                        strategy_key=position.strategy_key,
                        session_name=position.session_name,
                        regime_state=position.regime_state,
                        strategy_state=position.strategy_state,
                        lane_name=position.lane_name,
                        entry_timing_score=position.entry_timing_score,
                        structure_cleanliness_score=position.structure_cleanliness_score,
                        execution_quality_fit=position.execution_quality_fit,
                        entry_kind=position.entry_kind,
                        side=position.side,
                        opened_at=position.opened_at,
                        closed_at=closed_at,
                        scale_level=position.scale_level,
                        r_multiple=result_r,
                        result="WIN" if result_r > 0 else "LOSS",
                        exit_reason="TIME_STOP_EXIT",
                        mae_r=float(position.mae_r),
                        mfe_r=float(position.mfe_r),
                        duration_minutes=float(position.bars_open * 5),
                    )
                )
                open_positions.remove(position)
                continue
            if (not position.be_moved) and profit_r >= position.be_trigger_r:
                required_cushion_r = max(position.be_trigger_r, position.min_profit_protection_r + 0.08, position.be_buffer_r + 0.05)
                if profit_r >= required_cushion_r:
                    lock_r = min(position.be_buffer_r, max(0.0, profit_r - 0.08))
                    be_buffer = max(position.initial_risk * lock_r, atr * 0.05)
                    be_price = position.entry_price + be_buffer if position.side == "BUY" else position.entry_price - be_buffer
                    if (position.side == "BUY" and be_price > position.stop_price) or (position.side == "SELL" and be_price < position.stop_price):
                        position.stop_price = be_price
                        position.be_moved = True
                        events["sl_to_be"] += 1

            if profit_r >= position.trail_start_r and (not position.trail_requires_partial1 or position.partial1_done):
                atr_trail_price = current_exec_price - (atr * position.trail_atr_mult) if position.side == "BUY" else current_exec_price + (atr * position.trail_atr_mult)
                trail_lock_r = max(position.min_profit_protection_r, max(0.0, profit_r - position.trail_backoff_r))
                trail_lock_price = (
                    position.entry_price + (position.initial_risk * trail_lock_r)
                    if position.side == "BUY"
                    else position.entry_price - (position.initial_risk * trail_lock_r)
                )
                trail_price = min(atr_trail_price, trail_lock_price) if position.side == "BUY" else max(atr_trail_price, trail_lock_price)
                if (position.side == "BUY" and trail_price > position.stop_price) or (position.side == "SELL" and trail_price < position.stop_price):
                    position.stop_price = trail_price
                    position.trail_active = True
                    events["trail_updates"] += 1

            if (not position.partial1_done) and profit_r >= position.partial1_r:
                closed = self._partial_close(position, position.partial1_fraction, position.partial1_r)
                if closed:
                    position.partial1_done = True
                    events["partials_tp1"] += 1
            if (not position.partial2_done) and profit_r >= position.partial2_r:
                closed = self._partial_close(position, position.partial2_fraction, position.partial2_r)
                if closed:
                    position.partial2_done = True
                    events["partials_tp2"] += 1

            stop_hit, tp_hit = self._hit_flags(position, next_high, next_low, next_spread_points, point_size)
            if not stop_hit and not tp_hit:
                continue
            if stop_hit:
                exit_quote_price = position.stop_price
            else:
                exit_quote_price = position.take_profit_price
            exit_price = self._apply_exit_slippage(position, exit_quote_price, point_size)
            result_r = self._close_position(position, exit_price)
            r_results.append(result_r)
            trades.append(
                BacktestTrade(
                    signal_id=position.signal_id,
                    symbol=position.symbol,
                    setup=position.setup,
                    strategy_key=position.strategy_key,
                    session_name=position.session_name,
                    regime_state=position.regime_state,
                    strategy_state=position.strategy_state,
                    lane_name=position.lane_name,
                    entry_timing_score=position.entry_timing_score,
                    structure_cleanliness_score=position.structure_cleanliness_score,
                    execution_quality_fit=position.execution_quality_fit,
                    entry_kind=position.entry_kind,
                    side=position.side,
                    opened_at=position.opened_at,
                    closed_at=closed_at,
                    scale_level=position.scale_level,
                    r_multiple=result_r,
                    result="WIN" if result_r > 0 else "LOSS",
                    exit_reason="STOP_HIT" if stop_hit else "TP_HIT",
                    mae_r=float(position.mae_r),
                    mfe_r=float(position.mfe_r),
                    duration_minutes=float(position.bars_open * 5),
                )
            )
            open_positions.remove(position)

    def _close_all_positions(
        self,
        open_positions: list[SimPosition],
        close_mid_price: float,
        spread_points: float,
        point_size: float,
        closed_at: str,
        trades: list[BacktestTrade],
        r_results: list[float],
        reason: str,
    ) -> None:
        for position in list(open_positions):
            market_quote_price = self._market_exit_price(position, close_mid_price, spread_points, point_size)
            exit_price = self._apply_exit_slippage(position, market_quote_price, point_size)
            result_r = self._close_position(position, exit_price)
            r_results.append(result_r)
            trades.append(
                BacktestTrade(
                    signal_id=position.signal_id,
                    symbol=position.symbol,
                    setup=f"{position.setup}:{reason}",
                    strategy_key=position.strategy_key,
                    session_name=position.session_name,
                    regime_state=position.regime_state,
                    strategy_state=position.strategy_state,
                    lane_name=position.lane_name,
                    entry_timing_score=position.entry_timing_score,
                    structure_cleanliness_score=position.structure_cleanliness_score,
                    execution_quality_fit=position.execution_quality_fit,
                    entry_kind=position.entry_kind,
                    side=position.side,
                    opened_at=position.opened_at,
                    closed_at=closed_at,
                    scale_level=position.scale_level,
                    r_multiple=result_r,
                    result="WIN" if result_r > 0 else "LOSS",
                    exit_reason=str(reason),
                    mae_r=float(position.mae_r),
                    mfe_r=float(position.mfe_r),
                    duration_minutes=float(position.bars_open * 5),
                )
            )
            open_positions.remove(position)

    def _partial_close(self, position: SimPosition, fraction: float, target_r: float) -> bool:
        if fraction <= 0 or position.remaining_volume <= 0:
            return False
        close_volume = min(position.remaining_volume, position.volume * fraction)
        if close_volume <= 0:
            return False
        weighted_r = target_r * (close_volume / max(position.volume, 0.000001))
        commission_r = self._commission_r(position.initial_risk) * (close_volume / max(position.volume, 0.000001))
        position.realized_r += weighted_r - commission_r
        position.remaining_volume = max(0.0, position.remaining_volume - close_volume)
        return True

    def _close_position(self, position: SimPosition, exit_price: float) -> float:
        if position.initial_risk <= 0:
            return position.realized_r
        if position.remaining_volume <= 0:
            return position.realized_r
        direction = 1 if position.side == "BUY" else -1
        gross_r = ((exit_price - position.entry_price) * direction) / position.initial_risk
        weighted_r = gross_r * (position.remaining_volume / max(position.volume, 0.000001))
        commission_r = self._commission_r(position.initial_risk) * (position.remaining_volume / max(position.volume, 0.000001))
        total_r = position.realized_r + weighted_r - commission_r
        position.remaining_volume = 0.0
        position.realized_r = total_r
        return total_r

    def _hit_flags(
        self,
        position: SimPosition,
        high: float,
        low: float,
        spread_points: float,
        point_size: float,
    ) -> tuple[bool, bool]:
        half_spread_price = self._points_to_price(spread_points, point_size) / 2.0
        bid_high = high - half_spread_price
        bid_low = low - half_spread_price
        ask_high = high + half_spread_price
        ask_low = low + half_spread_price

        if position.side == "BUY":
            stop_hit = bid_low <= position.stop_price
            tp_hit = bid_high >= position.take_profit_price
        else:
            stop_hit = ask_high >= position.stop_price
            tp_hit = ask_low <= position.take_profit_price
        if stop_hit and tp_hit:
            # Conservative tie-break: assume adverse fill first.
            return True, False
        return stop_hit, tp_hit

    @staticmethod
    def _position_profit_r(position: SimPosition, price: float) -> float:
        direction = 1 if position.side == "BUY" else -1
        return ((price - position.entry_price) * direction) / max(position.initial_risk, 0.000001)

    def _basket_profit_r(
        self,
        open_positions: list[SimPosition],
        mid_price: float,
        spread_points: float,
        point_size: float,
    ) -> float:
        total = 0.0
        for position in open_positions:
            executable_price = self._market_exit_price(position, mid_price, spread_points, point_size)
            unrealized_r = self._position_profit_r(position, executable_price) * (
                position.remaining_volume / max(position.volume, 0.000001)
            )
            total += position.realized_r + unrealized_r
        return total

    def _active_basket_take_profit_r(self, open_positions: list[SimPosition]) -> float:
        if not open_positions:
            return float(self.basket_tp_r)
        active_targets = [
            max(0.10, float(position.basket_take_profit_r))
            for position in open_positions
            if float(position.basket_take_profit_r) > 0.0
        ]
        if not active_targets:
            return float(self.basket_tp_r)
        return float(min(active_targets))

    @staticmethod
    def _recent_loss_streak(r_results: list[float]) -> int:
        streak = 0
        for value in reversed(r_results):
            if value <= 0:
                streak += 1
                continue
            break
        return streak

    def _position_volume(self, equity: float, stop_distance: float, size_factor: float) -> float:
        if self.use_fixed_lot:
            return max(0.0, self.fixed_lot * size_factor)
        risk_amount = max(0.0, equity * self.risk_per_trade * size_factor)
        per_lot_risk = stop_distance * max(self.contract_size, 1.0)
        if per_lot_risk <= 0:
            return 0.0
        return max(0.0, risk_amount / per_lot_risk)

    def _scale_factor(self, level: int) -> float:
        if level < 0:
            return 1.0
        if level >= len(self.scale_factors):
            return self.scale_factors[-1]
        return self.scale_factors[level]

    def _commission_r(self, stop_distance: float) -> float:
        if stop_distance <= 0:
            return 0.0
        return self.commission_per_lot / max(stop_distance * self.contract_size, 1.0)

    def _metrics(self, trades: list[BacktestTrade], r_results: list[float], events: dict[str, int]) -> dict[str, Any]:
        if not r_results:
            return {
                "trade_count": 0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "expectancy_r": 0.0,
                "net_r": 0.0,
                "max_drawdown_r": 0.0,
                "sharpe": 0.0,
                "max_consecutive_losses": 0,
                "trades": [],
                "events": events,
                "monte_carlo": {},
            }
        wins = [result for result in r_results if result > 0]
        losses = [abs(result) for result in r_results if result <= 0]
        equity_curve = np.cumsum(r_results)
        running_peak = np.maximum.accumulate(equity_curve)
        drawdowns = running_peak - equity_curve
        max_drawdown = float(drawdowns.max()) if len(drawdowns) else 0.0
        sharpe = 0.0
        if np.std(r_results) > 0:
            sharpe = float((np.mean(r_results) / np.std(r_results)) * np.sqrt(max(len(r_results), 1)))

        consecutive_losses = 0
        max_consecutive_losses = 0
        for result in r_results:
            if result <= 0:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        gross_profit = sum(wins)
        gross_loss = sum(losses)
        trade_count = len(trades)
        win_rate = (sum(1 for result in r_results if result >= 0) / len(r_results)) if r_results else 0.0
        return {
            "trade_count": trade_count,
            "win_rate": win_rate,
            "profit_factor": gross_profit / gross_loss if gross_loss else float("inf"),
            "expectancy_r": float(np.mean(r_results)),
            "net_r": float(sum(r_results)),
            "max_drawdown_r": max_drawdown,
            "sharpe": sharpe,
            "max_consecutive_losses": max_consecutive_losses,
            "trades": [asdict(trade) for trade in trades],
            "events": events,
            "monte_carlo": self._monte_carlo(r_results),
        }

    def _monte_carlo(self, r_results: list[float], runs: int = 1000) -> dict[str, float]:
        if not r_results:
            return {}
        rng = random.Random(42)
        drawdowns: list[float] = []
        ending_r: list[float] = []
        for _ in range(runs):
            sample = r_results[:]
            rng.shuffle(sample)
            equity = np.cumsum(sample)
            peak = np.maximum.accumulate(equity)
            drawdown = float((peak - equity).max()) if len(equity) else 0.0
            drawdowns.append(drawdown)
            ending_r.append(float(equity[-1]))
        return {
            "median_end_r": float(np.median(ending_r)),
            "p05_end_r": float(np.percentile(ending_r, 5)),
            "p95_end_r": float(np.percentile(ending_r, 95)),
            "median_drawdown_r": float(np.median(drawdowns)),
            "p95_drawdown_r": float(np.percentile(drawdowns, 95)),
        }

    def _effective_spread_points(self, row: pd.Series) -> float:
        row_spread = float(row.get("m5_spread", 0.0) or 0.0)
        if math.isfinite(row_spread) and row_spread > 0:
            return max(0.0, row_spread)
        override_spread = float(row.get("backtest_spread_points", self.spread_points) or self.spread_points)
        if not math.isfinite(override_spread):
            override_spread = self.spread_points
        return max(0.0, override_spread)

    @staticmethod
    def _points_to_price(points: float, point_size: float) -> float:
        return max(0.0, points) * max(point_size, 1e-9)

    def _entry_fill_price(self, side: str, mid_open: float, spread_points: float, point_size: float) -> float:
        half_spread_price = self._points_to_price(spread_points, point_size) / 2.0
        slippage_price = self._points_to_price(self.slippage_points, point_size)
        if side == "BUY":
            return mid_open + half_spread_price + slippage_price
        return mid_open - half_spread_price - slippage_price

    def _market_exit_price(self, position: SimPosition, mid_price: float, spread_points: float, point_size: float) -> float:
        half_spread_price = self._points_to_price(spread_points, point_size) / 2.0
        if position.side == "BUY":
            return mid_price - half_spread_price  # close BUY at bid
        return mid_price + half_spread_price  # close SELL at ask

    def _apply_exit_slippage(self, position: SimPosition, quote_price: float, point_size: float) -> float:
        slippage_price = self._points_to_price(self.slippage_points, point_size)
        if position.side == "BUY":
            return quote_price - slippage_price
        return quote_price + slippage_price

    @staticmethod
    def _infer_bar_ms(frame: pd.DataFrame) -> int:
        times = pd.to_datetime(frame["time"], utc=True, errors="coerce")
        deltas = times.diff().dropna()
        if deltas.empty:
            return 300_000
        seconds = deltas.dt.total_seconds()
        median_seconds = float(seconds.median()) if not seconds.empty else 300.0
        return max(1, int(median_seconds * 1000))

    @staticmethod
    def _infer_point_size(frame: pd.DataFrame) -> float:
        if "instrument_point_size" in frame.columns:
            try:
                instrument_point_size = float(frame["instrument_point_size"].iloc[-1] or 0.0)
                if instrument_point_size > 0:
                    return instrument_point_size
            except Exception:
                pass
        close = frame["m5_close"].astype(float)
        diffs = close.diff().abs()
        positive = diffs[diffs > 0]
        if not positive.empty:
            sample = float(np.percentile(positive, 25))
            if sample > 0:
                exponent = math.floor(math.log10(sample))
                inferred = 10 ** exponent
                return float(min(max(inferred, 1e-6), 1.0))
        last_price = abs(float(close.iloc[-1])) if len(close) else 1.0
        if last_price >= 100:
            return 0.01
        if last_price >= 1:
            return 0.0001
        return 0.00001

    def _entry_fill_ratio(self, signal_id: str, index: int, row: pd.Series) -> float:
        seed = hashlib.sha256(f"{signal_id}:{index}".encode("utf-8")).hexdigest()
        jitter = ((int(seed[:6], 16) % 1000) / 1000.0 - 0.5) * 0.08
        atr_ratio = float(row.get("m5_atr_pct_of_avg", 1.0))
        spread_ratio = float(row.get("m5_spread_ratio_20", 1.0))
        liquidity_penalty = max(0.0, atr_ratio - 1.2) * 0.12 + max(0.0, spread_ratio - 1.0) * 0.10
        base = 1.0 - min(0.45, liquidity_penalty)
        return max(0.4, min(1.0, base + jitter))
