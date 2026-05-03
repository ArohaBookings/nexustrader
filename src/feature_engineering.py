from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.utils import clamp


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False, min_periods=span).mean()


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def _atr(frame: pd.DataFrame, period: int = 14) -> pd.Series:
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _spread_atr(series: pd.Series, period: int = 14) -> pd.Series:
    spread_delta = series.astype(float).diff().abs()
    return spread_delta.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def _adx(frame: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    high_diff = frame["high"].diff()
    low_diff = -frame["low"].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0.0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0.0)

    prev_close = frame["close"].shift(1)
    tr = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    plus_di = 100 * pd.Series(plus_dm, index=frame.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=frame.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)) * 100
    adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def _macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    fast = _ema(series, 12)
    slow = _ema(series, 26)
    macd_line = fast - slow
    signal = macd_line.ewm(span=9, adjust=False, min_periods=9).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def _bollinger(series: pd.Series, period: int = 20, width: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    mean = series.rolling(period, min_periods=period).mean()
    std = series.rolling(period, min_periods=period).std()
    upper = mean + (std * width)
    lower = mean - (std * width)
    return mean, upper, lower


def _rolling_directional_entropy(series: pd.Series, window: int = 32) -> pd.Series:
    direction = np.sign(series.fillna(0.0))
    up = pd.Series((direction > 0).astype(float), index=series.index).rolling(window, min_periods=window).mean()
    down = pd.Series((direction < 0).astype(float), index=series.index).rolling(window, min_periods=window).mean()
    flat = (1.0 - up - down).clip(lower=0.0, upper=1.0)
    entropy = pd.Series(0.0, index=series.index, dtype=float)
    for component in (up, down, flat):
        safe = component.where(component > 0.0, 1.0)
        entropy += -(component.where(component > 0.0, 0.0) * np.log(safe))
    return (entropy / np.log(3.0)).fillna(0.5)


def _trend_efficiency(series: pd.Series, window: int = 16) -> pd.Series:
    net_move = series.diff(window).abs()
    gross_move = series.diff().abs().rolling(window, min_periods=window).sum()
    return (net_move / gross_move.replace(0.0, np.nan)).fillna(0.0).clip(lower=0.0, upper=1.0)


def _hurst_proxy(series: pd.Series, window: int = 64) -> pd.Series:
    ret_1 = series.pct_change().replace([np.inf, -np.inf], np.nan)
    sigma_1 = ret_1.rolling(window, min_periods=window).std()
    sigma_4 = series.pct_change(4).replace([np.inf, -np.inf], np.nan).rolling(window, min_periods=window).std()
    hurst = np.log(sigma_4 / sigma_1.replace(0.0, np.nan)) / np.log(4.0)
    return hurst.fillna(0.5).clip(lower=0.0, upper=1.0)


def _rolling_zscore(series: pd.Series, window: int = 48) -> pd.Series:
    mean = series.rolling(window, min_periods=window).mean()
    std = series.rolling(window, min_periods=window).std()
    return ((series - mean) / std.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _normalize_time_utc_ns(series: pd.Series) -> pd.Series:
    normalized = pd.to_datetime(series, utc=True).dt.floor("s")
    # Force a consistent tz-aware ns dtype so merge_asof does not trip over
    # cached frames created with different datetime precisions, while preserving
    # the original wall-clock timestamp instead of reinterpreting microseconds
    # as nanoseconds from the epoch.
    return normalized.astype("datetime64[ns, UTC]")


@dataclass
class FeatureEngineer:
    feature_columns: list[str] | None = None

    def build(
        self,
        m1: pd.DataFrame | None,
        m5: pd.DataFrame,
        m15: pd.DataFrame,
        h1: pd.DataFrame,
        h4: pd.DataFrame | None = None,
        context: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        m1_source = m1.copy() if m1 is not None else m5.copy()
        h4_source = h4.copy() if h4 is not None else self._derive_h4_from_h1(h1)
        m1_features = self._timeframe_features(m1_source, "m1")
        m5_features = self._timeframe_features(m5.copy(), "m5")
        m15_features = self._timeframe_features(m15.copy(), "m15")
        h1_features = self._timeframe_features(h1.copy(), "h1")
        h4_features = self._timeframe_features(h4_source.copy(), "h4")

        merged = pd.merge_asof(
            m5_features.sort_values("time"),
            m1_features.sort_values("time"),
            on="time",
            direction="backward",
        )
        merged = pd.merge_asof(
            merged.sort_values("time"),
            m15_features.sort_values("time"),
            on="time",
            direction="backward",
        )
        merged = pd.merge_asof(
            merged.sort_values("time"),
            h1_features.sort_values("time"),
            on="time",
            direction="backward",
        )
        merged = pd.merge_asof(
            merged.sort_values("time"),
            h4_features.sort_values("time"),
            on="time",
            direction="backward",
        )

        merged["hour_utc"] = merged["time"].dt.hour
        merged["day_of_week"] = merged["time"].dt.dayofweek
        merged["is_monday"] = (merged["day_of_week"] == 0).astype(int)
        merged["is_friday"] = (merged["day_of_week"] == 4).astype(int)
        merged["is_asian"] = merged["hour_utc"].between(0, 7).astype(int)
        merged["is_london"] = merged["hour_utc"].between(8, 12).astype(int)
        merged["is_overlap"] = merged["hour_utc"].between(13, 17).astype(int)
        merged["is_new_york"] = merged["hour_utc"].between(18, 21).astype(int)
        merged["session_liquidity_score"] = (
            merged["is_overlap"] * 1.0
            + merged["is_london"] * 0.8
            + merged["is_new_york"] * 0.8
            + merged["is_asian"] * 0.4
        )
        merged["session_aggression_score"] = (
            merged["is_overlap"] * 1.0
            + merged["is_london"] * 0.7
            + merged["is_new_york"] * 0.65
            + merged["is_asian"] * 0.25
        )
        midweek_score = (
            (merged["day_of_week"].isin([1, 2, 3]).astype(float) * 1.0)
            + (merged["day_of_week"].isin([0, 4]).astype(float) * 0.65)
            + (merged["day_of_week"].isin([5, 6]).astype(float) * 0.45)
        )
        friday_late_penalty = ((merged["is_friday"] == 1) & (merged["hour_utc"] >= 18)).astype(float) * 0.18
        merged["seasonality_edge_score"] = clamp_series(
            (merged["session_aggression_score"] * 0.65) + (midweek_score * 0.35) - friday_late_penalty,
            low=0.0,
            high=1.0,
        )
        merged["ghost_order_book_pressure"] = (
            (merged["m5_upper_wick_ratio"] + merged["m5_lower_wick_ratio"])
            * merged["m5_volume_ratio_20"]
            * merged["m5_spread_ratio_20"].replace(0.0, 1.0)
        ).clip(lower=0.0, upper=4.0)
        merged["predicted_liquidity_hunt_score"] = (
            (merged["m5_pinbar_bull"] + merged["m5_pinbar_bear"] + merged["m5_fractal_high"] + merged["m5_fractal_low"])
            * merged["session_liquidity_score"].replace(0.0, 0.1)
            * merged["m5_atr_percentile_20"].replace(0.0, 0.1)
        ).clip(lower=0.0, upper=4.0)
        merged["behavior_fear_score"] = (
            (1.0 - merged["m5_range_position_20"].clip(0.0, 1.0))
            * merged["m5_atr_percentile_20"].clip(0.0, 1.0)
            * merged["m5_spread_ratio_20"].clip(lower=0.2, upper=3.0)
        ).clip(lower=0.0, upper=1.0)
        merged["behavior_greed_score"] = (
            merged["m5_range_position_20"].clip(0.0, 1.0)
            * merged["m5_atr_percentile_20"].clip(0.0, 1.0)
            * merged["m5_volume_ratio_20"].clip(lower=0.2, upper=3.0)
        ).clip(lower=0.0, upper=1.0)
        merged["behavior_complacency_score"] = (
            (1.0 - merged["m5_atr_percentile_20"].clip(0.0, 1.0))
            * (1.0 - merged["m5_spread_ratio_20"].clip(0.0, 1.0))
            * (1.0 - merged["m5_volume_ratio_20"].clip(0.0, 1.0))
        ).clip(lower=0.0, upper=1.0)
        merged["behavior_bias_score"] = clamp_series(
            merged["behavior_greed_score"] - merged["behavior_fear_score"],
            low=-1.0,
            high=1.0,
        )
        direction_votes = []
        for prefix in ("m1", "m5", "m15", "h1"):
            trend_bias = np.sign(merged.get(f"{prefix}_trend_bias", 0.0))
            ret_bias = np.sign(merged.get(f"{prefix}_ret_1", 0.0))
            direction_votes.append(((trend_bias + ret_bias) / 2.0).replace([np.inf, -np.inf], 0.0))
        alignment_matrix = pd.concat(direction_votes, axis=1).fillna(0.0)
        merged["multi_tf_alignment_score"] = (
            alignment_matrix.eq(alignment_matrix.iloc[:, [0]].values).mean(axis=1).fillna(0.5)
        ).clip(lower=0.0, upper=1.0)
        merged["fractal_persistence_score"] = clamp_series(
            (
                merged["m5_trend_efficiency_32"] * 0.35
                + merged["m15_trend_efficiency_16"] * 0.25
                + merged["m5_hurst_proxy_64"] * 0.20
                + merged["m15_hurst_proxy_64"] * 0.20
            ),
            low=0.0,
            high=1.0,
        )
        drift_components = [
            _rolling_zscore(merged["m5_atr_pct_of_avg"], 48).abs(),
            _rolling_zscore(merged["m5_spread_ratio_20"], 48).abs(),
            _rolling_zscore(merged["m5_volume_ratio_20"], 48).abs(),
            _rolling_zscore(merged["m5_body_efficiency"], 48).abs(),
        ]
        merged["feature_drift_score"] = (
            pd.concat(drift_components, axis=1).mean(axis=1).fillna(0.0).clip(lower=0.0, upper=3.0) / 3.0
        )
        merged["market_instability_score"] = clamp_series(
            (
                merged["m5_return_sign_entropy_32"] * 0.35
                + merged["feature_drift_score"] * 0.25
                + ((merged["m5_spread_ratio_20"] - 1.0).clip(lower=0.0, upper=2.0) / 2.0) * 0.20
                + ((merged["m5_atr_pct_of_avg"] - 1.0).clip(lower=0.0, upper=2.0) / 2.0) * 0.10
                + (1.0 - merged["multi_tf_alignment_score"]) * 0.10
            ),
            low=0.0,
            high=1.0,
        )
        merged["compression_expansion_score"] = clamp_series(
            (
                (1.15 - merged["m5_atr_pct_of_avg"].clip(lower=0.0, upper=2.5)).clip(lower=0.0, upper=1.0) * 0.30
                + (1.20 - merged["m5_spread_ratio_20"].clip(lower=0.0, upper=2.5)).clip(lower=0.0, upper=1.0) * 0.20
                + merged["m5_body_efficiency"].clip(0.0, 1.0) * 0.20
                + ((merged["m5_volume_ratio_20"].clip(lower=1.0, upper=3.0) - 1.0) / 2.0) * 0.15
                + merged["m5_trend_efficiency_16"].clip(0.0, 1.0) * 0.15
            ),
            low=0.0,
            high=1.0,
        )
        merged["candle_mastery_score"] = clamp_series(
            (
                merged["m5_body_efficiency"].clip(0.0, 1.0) * 0.16
                + merged["m5_absorption_proxy"].clip(0.0, 1.0) * 0.12
                + merged["m5_volume_profile_pressure"].clip(0.0, 1.0) * 0.12
                + merged["m5_liquidity_sweep"].clip(0.0, 1.0) * 0.12
                + (merged["m5_bos_bull"] + merged["m5_bos_bear"]).clip(0.0, 1.0) * 0.12
                + (merged["m5_choch_bull"] + merged["m5_choch_bear"]).clip(0.0, 1.0) * 0.10
                + (merged["m5_fvg_bull"] + merged["m5_fvg_bear"]).clip(0.0, 1.0) * 0.08
                + (merged["m5_order_block_bull"] + merged["m5_order_block_bear"]).clip(0.0, 1.0) * 0.08
                + (1.0 - (merged["m5_vwap_deviation_atr"].abs() / 3.0).clip(0.0, 1.0)) * 0.10
                + merged["m15_trend_efficiency_16"].clip(0.0, 1.0) * 0.10
            ),
            low=0.0,
            high=1.0,
        )
        merged["execution_edge_score"] = clamp_series(
            (
                (1.0 - ((merged["m5_spread_ratio_20"] - 1.0).clip(lower=0.0, upper=2.0) / 2.0)) * 0.38
                + merged["session_liquidity_score"].clip(0.0, 1.0) * 0.24
                + (1.0 - merged["market_instability_score"].clip(0.0, 1.0)) * 0.24
                + merged["m5_volume_profile_pressure"].clip(0.0, 1.0) * 0.14
            ),
            low=0.0,
            high=1.0,
        )
        merged["live_shadow_gap_risk_score"] = 0.0
        merged["live_shadow_gap_score"] = merged["live_shadow_gap_risk_score"]
        merged["institutional_confluence_score"] = clamp_series(
            (
                merged["candle_mastery_score"] * 0.30
                + merged["multi_tf_alignment_score"] * 0.22
                + merged["fractal_persistence_score"] * 0.14
                + merged["compression_expansion_score"] * 0.10
                + merged["session_aggression_score"].clip(0.0, 1.0) * 0.10
                + merged["execution_edge_score"] * 0.09
                - merged["market_instability_score"] * 0.10
                - merged["live_shadow_gap_risk_score"] * 0.12
            ),
            low=0.0,
            high=1.0,
        )

        if context:
            for key, value in context.items():
                merged[key] = value

        merged = merged.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).reset_index(drop=True)
        self.feature_columns = [column for column in merged.columns if column != "time"]
        return merged

    def feature_vector(self, row: pd.Series) -> pd.DataFrame:
        if not self.feature_columns:
            columns = [column for column in row.index if column != "time"]
        else:
            columns = self.feature_columns
        return pd.DataFrame([{column: float(row.get(column, 0.0)) for column in columns}])

    def _timeframe_features(self, frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        normalized = frame.copy()
        normalized["time"] = _normalize_time_utc_ns(normalized["time"])
        for column in ("open", "high", "low", "close", "tick_volume", "spread"):
            if column not in normalized.columns:
                normalized[column] = 0.0

        close = normalized["close"].astype(float)
        open_ = normalized["open"].astype(float)
        high = normalized["high"].astype(float)
        low = normalized["low"].astype(float)
        body = close - open_
        candle_range = (high - low).replace(0.0, np.nan)
        upper_wick = high - close.where(close >= open_, open_)
        lower_wick = open_.where(close >= open_, close) - low
        prev_open = open_.shift(1)
        prev_close = close.shift(1)
        ret_5 = close.pct_change(5).fillna(0.0)
        body_ratio = (body.abs() / candle_range).fillna(0.0)
        upper_wick_ratio = (upper_wick / candle_range).fillna(0.0)
        lower_wick_ratio = (lower_wick / candle_range).fillna(0.0)

        feature_map: dict[str, pd.Series] = {
            f"{prefix}_open": open_,
            f"{prefix}_high": high,
            f"{prefix}_low": low,
            f"{prefix}_close": close,
            f"{prefix}_body": body,
            f"{prefix}_range": candle_range.fillna(0.0),
            f"{prefix}_body_ratio": body_ratio,
            f"{prefix}_body_efficiency": body_ratio,
            f"{prefix}_candle_efficiency": body_ratio,
            f"{prefix}_upper_wick_ratio": upper_wick_ratio,
            f"{prefix}_lower_wick_ratio": lower_wick_ratio,
            f"{prefix}_wick_imbalance": (lower_wick_ratio - upper_wick_ratio).fillna(0.0),
            f"{prefix}_candle_direction": np.sign(body).fillna(0.0),
            f"{prefix}_bullish": (close > open_).astype(int),
            f"{prefix}_bearish": (close < open_).astype(int),
        }
        feature_map[f"{prefix}_pinbar_bull"] = (
            (feature_map[f"{prefix}_lower_wick_ratio"] > 0.55) & (feature_map[f"{prefix}_body_ratio"] < 0.3)
        ).astype(int)
        feature_map[f"{prefix}_pinbar_bear"] = (
            (feature_map[f"{prefix}_upper_wick_ratio"] > 0.55) & (feature_map[f"{prefix}_body_ratio"] < 0.3)
        ).astype(int)
        feature_map[f"{prefix}_engulf_bull"] = (
            (prev_close < prev_open) & (close > open_) & (close >= prev_open) & (open_ <= prev_close)
        ).astype(int)
        feature_map[f"{prefix}_engulf_bear"] = (
            (prev_close > prev_open) & (close < open_) & (open_ >= prev_close) & (close <= prev_open)
        ).astype(int)

        for lookback in (1, 3, 5, 10, 20):
            feature_map[f"{prefix}_ret_{lookback}"] = close.pct_change(lookback).fillna(0.0)
            feature_map[f"{prefix}_momentum_{lookback}"] = (close - close.shift(lookback)).fillna(0.0)
        feature_map[f"{prefix}_return_sign_entropy_32"] = _rolling_directional_entropy(
            feature_map[f"{prefix}_ret_1"], 32
        )
        feature_map[f"{prefix}_trend_efficiency_16"] = _trend_efficiency(close, 16)
        feature_map[f"{prefix}_trend_efficiency_32"] = _trend_efficiency(close, 32)
        feature_map[f"{prefix}_hurst_proxy_64"] = _hurst_proxy(close, 64)

        rolling_high_20 = high.rolling(20, min_periods=20).max()
        rolling_low_20 = low.rolling(20, min_periods=20).min()
        feature_map[f"{prefix}_swing_high_5"] = (high == high.rolling(5, min_periods=5).max()).astype(int)
        feature_map[f"{prefix}_swing_low_5"] = (low == low.rolling(5, min_periods=5).min()).astype(int)
        feature_map[f"{prefix}_fractal_high"] = feature_map[f"{prefix}_swing_high_5"]
        feature_map[f"{prefix}_fractal_low"] = feature_map[f"{prefix}_swing_low_5"]
        feature_map[f"{prefix}_rolling_high_20"] = rolling_high_20
        feature_map[f"{prefix}_rolling_low_20"] = rolling_low_20
        feature_map[f"{prefix}_rolling_high_prev_20"] = rolling_high_20.shift(1)
        feature_map[f"{prefix}_rolling_low_prev_20"] = rolling_low_20.shift(1)
        feature_map[f"{prefix}_range_position_20"] = (
            (close - rolling_low_20) / (rolling_high_20 - rolling_low_20).replace(0.0, np.nan)
        ).fillna(0.5)
        prior_high_20 = rolling_high_20.shift(1)
        prior_low_20 = rolling_low_20.shift(1)
        sweep_high = ((high > prior_high_20) & (close < prior_high_20) & (upper_wick_ratio > 0.30)).astype(int)
        sweep_low = ((low < prior_low_20) & (close > prior_low_20) & (lower_wick_ratio > 0.30)).astype(int)
        bos_bull = ((close > prior_high_20) & (body > 0) & (body_ratio > 0.45)).astype(int)
        bos_bear = ((close < prior_low_20) & (body < 0) & (body_ratio > 0.45)).astype(int)
        feature_map[f"{prefix}_fakeout_high"] = sweep_high
        feature_map[f"{prefix}_fakeout_low"] = sweep_low
        feature_map[f"{prefix}_liquidity_sweep_high"] = sweep_high
        feature_map[f"{prefix}_liquidity_sweep_low"] = sweep_low
        feature_map[f"{prefix}_liquidity_sweep"] = ((sweep_high + sweep_low) > 0).astype(int)
        feature_map[f"{prefix}_bos_bull"] = bos_bull
        feature_map[f"{prefix}_bos_bear"] = bos_bear
        prev_trend = np.sign(close.diff(5).shift(1).fillna(0.0))
        feature_map[f"{prefix}_choch_bull"] = ((prev_trend < 0) & (bos_bull == 1)).astype(int)
        feature_map[f"{prefix}_choch_bear"] = ((prev_trend > 0) & (bos_bear == 1)).astype(int)
        feature_map[f"{prefix}_fvg_bull"] = ((low > high.shift(2)) & (body > 0) & (body_ratio > 0.45)).astype(int)
        feature_map[f"{prefix}_fvg_bear"] = ((high < low.shift(2)) & (body < 0) & (body_ratio > 0.45)).astype(int)
        fvg_mid_bull = ((low + high.shift(2)) / 2.0).where(feature_map[f"{prefix}_fvg_bull"] == 1)
        fvg_mid_bear = ((high + low.shift(2)) / 2.0).where(feature_map[f"{prefix}_fvg_bear"] == 1)
        fvg_mid = fvg_mid_bull.ffill().fillna(fvg_mid_bear.ffill())
        feature_map[f"{prefix}_fvg_fill_pressure"] = (
            1.0 - ((close - fvg_mid).abs() / candle_range.rolling(10, min_periods=1).mean().replace(0.0, np.nan))
        ).clip(lower=0.0, upper=1.0).fillna(0.0)

        ema20 = _ema(close, 20)
        ema50 = _ema(close, 50)
        ema200 = _ema(close, 200)
        feature_map[f"{prefix}_ema_20"] = ema20
        feature_map[f"{prefix}_ema_50"] = ema50
        feature_map[f"{prefix}_ema_200"] = ema200
        feature_map[f"{prefix}_ema_gap_20_50"] = ema20 - ema50
        feature_map[f"{prefix}_ema_gap_50_200"] = ema50 - ema200
        feature_map[f"{prefix}_dist_ema_20"] = close - ema20
        feature_map[f"{prefix}_dist_ema_50"] = close - ema50
        feature_map[f"{prefix}_dist_ema_200"] = close - ema200
        feature_map[f"{prefix}_golden_cross"] = ((ema20 > ema50) & (ema50 > ema200)).astype(int)
        feature_map[f"{prefix}_dead_cross"] = ((ema20 < ema50) & (ema50 < ema200)).astype(int)

        rsi_14 = _rsi(close, 14)
        rsi_delta = rsi_14.diff().fillna(0.0)
        feature_map[f"{prefix}_rsi_14"] = rsi_14
        feature_map[f"{prefix}_rsi_delta"] = rsi_delta
        feature_map[f"{prefix}_rsi_divergence_proxy"] = (np.sign(ret_5) != np.sign(rsi_delta)).astype(int)

        macd_line, macd_signal, macd_hist = _macd(close)
        feature_map[f"{prefix}_macd"] = macd_line
        feature_map[f"{prefix}_macd_signal"] = macd_signal
        feature_map[f"{prefix}_macd_hist"] = macd_hist
        feature_map[f"{prefix}_macd_hist_slope"] = macd_hist.diff().fillna(0.0)
        feature_map[f"{prefix}_macd_zero_cross_velocity"] = (
            np.sign(macd_hist).diff().fillna(0.0).abs() * macd_hist.diff().abs().fillna(0.0)
        )

        atr_14 = _atr(normalized, 14)
        atr_avg_20 = atr_14.rolling(20, min_periods=20).mean()
        feature_map[f"{prefix}_atr_14"] = atr_14
        feature_map[f"{prefix}_atr_avg_20"] = atr_avg_20
        feature_map[f"{prefix}_atr_pct_of_avg"] = (atr_14 / atr_avg_20.replace(0.0, np.nan)).fillna(1.0)
        feature_map[f"{prefix}_atr_percentile_20"] = atr_14.rolling(20, min_periods=20).rank(pct=True).fillna(0.5)

        bb_mid, bb_upper, bb_lower = _bollinger(close, 20, 2.0)
        feature_map[f"{prefix}_bb_mid"] = bb_mid
        feature_map[f"{prefix}_bb_width"] = ((bb_upper - bb_lower) / bb_mid.replace(0.0, np.nan)).fillna(0.0)
        feature_map[f"{prefix}_bb_pctb"] = ((close - bb_lower) / (bb_upper - bb_lower).replace(0.0, np.nan)).fillna(0.5)

        feature_map[f"{prefix}_keltner_upper"] = ema20 + (atr_14 * 2)
        feature_map[f"{prefix}_keltner_lower"] = ema20 - (atr_14 * 2)
        feature_map[f"{prefix}_keltner_penetration"] = (
            ((close - feature_map[f"{prefix}_keltner_upper"]).clip(lower=0.0)
             + (feature_map[f"{prefix}_keltner_lower"] - close).clip(lower=0.0))
            / atr_14.replace(0.0, np.nan)
        ).fillna(0.0)

        adx, plus_di, minus_di = _adx(normalized, 14)
        feature_map[f"{prefix}_adx_14"] = adx
        feature_map[f"{prefix}_di_plus"] = plus_di
        feature_map[f"{prefix}_di_minus"] = minus_di
        feature_map[f"{prefix}_trend_bias"] = (plus_di - minus_di).fillna(0.0)

        volume = normalized["tick_volume"].astype(float)
        spread = normalized["spread"].astype(float)
        feature_map[f"{prefix}_tick_volume"] = volume
        volume_ratio_20 = (volume / volume.rolling(20, min_periods=20).mean().replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        # External chart feeds often publish zero volume on otherwise valid bars.
        # Treat that as "volume unavailable" rather than "volume collapse" so candidate
        # generation can fall back to neutral price-action logic.
        volume_ratio_20 = volume_ratio_20.where(volume > 0.0, 1.0).fillna(1.0)
        feature_map[f"{prefix}_volume_ratio_20"] = volume_ratio_20
        feature_map[f"{prefix}_absorption_proxy"] = (
            (np.maximum(upper_wick_ratio, lower_wick_ratio) * (1.0 - body_ratio).clip(lower=0.0, upper=1.0) * volume_ratio_20.clip(0.0, 3.0))
            / 3.0
        ).clip(lower=0.0, upper=1.0).fillna(0.0)
        feature_map[f"{prefix}_impulse_candle"] = ((body_ratio > 0.62) & (volume_ratio_20 > 1.05)).astype(int)
        feature_map[f"{prefix}_exhaustion_wick"] = (
            (np.maximum(upper_wick_ratio, lower_wick_ratio) > 0.48) & (volume_ratio_20 > 1.15)
        ).astype(int)
        feature_map[f"{prefix}_order_block_bull"] = (
            (close.shift(1) < open_.shift(1)) & (body > 0) & (body_ratio > 0.48) & (volume_ratio_20 > 1.0)
        ).astype(int)
        feature_map[f"{prefix}_order_block_bear"] = (
            (close.shift(1) > open_.shift(1)) & (body < 0) & (body_ratio > 0.48) & (volume_ratio_20 > 1.0)
        ).astype(int)
        typical_price = (high + low + close) / 3.0
        cumulative_volume = volume.replace(0.0, np.nan).fillna(1.0).rolling(96, min_periods=1).sum()
        feature_map[f"{prefix}_vwap_96"] = ((typical_price * volume.replace(0.0, 1.0)).rolling(96, min_periods=1).sum() / cumulative_volume).fillna(close)
        anchored_vwap = ((typical_price * volume.replace(0.0, 1.0)).expanding(min_periods=1).sum() / volume.replace(0.0, 1.0).expanding(min_periods=1).sum()).fillna(close)
        feature_map[f"{prefix}_anchored_vwap"] = anchored_vwap
        feature_map[f"{prefix}_vwap_deviation_atr"] = (
            (close - feature_map[f"{prefix}_vwap_96"]) / atr_14.replace(0.0, np.nan)
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        feature_map[f"{prefix}_volume_profile_pressure"] = (
            (feature_map[f"{prefix}_range_position_20"].clip(0.0, 1.0) * 0.45)
            + (volume_ratio_20.clip(0.0, 3.0) / 3.0 * 0.35)
            + ((1.0 - (feature_map[f"{prefix}_vwap_deviation_atr"].abs() / 3.0).clip(0.0, 1.0)) * 0.20)
        ).clip(lower=0.0, upper=1.0).fillna(0.5)
        feature_map[f"{prefix}_spread"] = spread
        feature_map[f"{prefix}_spread_mean_14"] = spread.rolling(14, min_periods=14).mean().fillna(spread)
        feature_map[f"{prefix}_spread_atr_14"] = _spread_atr(spread, 14).fillna(0.0)
        feature_map[f"{prefix}_spread_ratio_20"] = (spread / spread.rolling(20, min_periods=20).mean().replace(0.0, np.nan)).fillna(1.0)
        feature_map[f"{prefix}_trend_score"] = clamp_series((adx / 50.0).fillna(0.0) * np.sign((ema50 - ema200).fillna(0.0)))
        feature_map[f"{prefix}_volatility_score"] = clamp_series(((atr_14 / atr_avg_20.replace(0.0, np.nan)).fillna(1.0)) / 2.0)

        feature_frame = pd.DataFrame(feature_map, index=normalized.index)
        return pd.concat([normalized[["time"]], feature_frame], axis=1)

    @staticmethod
    def _derive_h4_from_h1(h1: pd.DataFrame) -> pd.DataFrame:
        frame = h1.copy()
        frame["time"] = _normalize_time_utc_ns(frame["time"])
        frame = frame.sort_values("time")
        indexed = frame.set_index("time")
        agg = indexed.resample("4h").agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "tick_volume": "sum",
                "spread": "mean",
            }
        )
        agg = agg.dropna().reset_index()
        return agg


def clamp_series(series: pd.Series, low: float = -1.0, high: float = 1.0) -> pd.Series:
    return series.apply(lambda value: clamp(float(value), low, high) if pd.notna(value) else 0.0)
