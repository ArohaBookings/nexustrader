from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Any

import joblib
import pandas as pd

from src.omega_regime import OmegaRegimeDetector
from src.utils import clamp, ensure_parent, utc_now


@dataclass
class RegimeClassification:
    label: str
    confidence: float
    source: str
    details: dict[str, Any]
    state_label: str = "BREAKOUT_COMPRESSION"

    @property
    def state_confidence(self) -> float:
        try:
            return float(self.details.get("regime_state_confidence", self.confidence))
        except Exception:
            return float(self.confidence)


@dataclass
class RegimeDetector:
    model_path: Path = Path("models/regime_model.pkl")
    history_path: Path = Path("data/regime_history.json")
    omega_model_path: Path | None = None
    persist_history: bool = True

    def __post_init__(self) -> None:
        self._model = self._load_model()
        self._omega = OmegaRegimeDetector(model_path=self._resolve_omega_path(), enabled=True)

    def classify(self, row: pd.Series) -> RegimeClassification:
        heuristic_details = self._heuristic_details(row)
        omega_payload = {
            "atr_ratio": heuristic_details["atr_ratio"],
            "adx": heuristic_details["adx"],
            "trend_gap": heuristic_details["trend_gap"],
            "ema_slope": heuristic_details["ema_slope"],
            "spread_ratio": heuristic_details["spread_ratio"],
            "momentum": float(row.get("m5_ret_1", 0.0)),
            "news_shock_score": self._news_shock_score(row, heuristic_details),
            "regime_hint": str(row.get("regime_hint", "")),
            "side_hint": str(row.get("side_hint", "")),
        }
        omega_state = self._omega.classify(omega_payload)
        heuristic_details["omega_regime"] = float(self._omega_id(omega_state.label))
        heuristic_details["omega_regime_label"] = omega_state.label
        heuristic_details["omega_confidence"] = float(omega_state.confidence)
        heuristic_details["omega_regime_id"] = float(self._omega_id(omega_state.label))
        heuristic_details["trend_direction"] = 1.0 if omega_state.label == "TRENDING_UP" else (-1.0 if omega_state.label == "TRENDING_DOWN" else 0.0)
        heuristic_details["news_shock_flag"] = 1.0 if omega_state.label == "NEWS_SHOCK" else 0.0
        heuristic_details["omega_source"] = 1.0 if omega_state.source == "onnx" else 0.0
        heuristic_details["omega_prob_trending_up"] = float(omega_state.probabilities.get("TRENDING_UP", 0.0))
        heuristic_details["omega_prob_trending_down"] = float(omega_state.probabilities.get("TRENDING_DOWN", 0.0))
        heuristic_details["omega_prob_ranging"] = float(omega_state.probabilities.get("RANGING", 0.0))
        heuristic_details["omega_prob_volatile"] = float(omega_state.probabilities.get("VOLATILE", 0.0))
        heuristic_details["omega_prob_news_shock"] = float(omega_state.probabilities.get("NEWS_SHOCK", 0.0))
        legacy_label = OmegaRegimeDetector.to_legacy_label(omega_state.label)
        confidence = float(omega_state.confidence)
        state_label, state_confidence, state_details = self._rich_regime_state(row=row, details=heuristic_details, omega_label=omega_state.label)
        heuristic_details.update(state_details)
        heuristic_details["regime_state_confidence"] = float(state_confidence)

        if self._model is not None:
            try:
                columns = list(getattr(self._model, "feature_names_in_", []))
                payload = pd.DataFrame([{column: float(row.get(column, 0.0)) for column in columns}])
                model_label = str(self._model.predict(payload)[0]).upper()
                model_confidence = 0.6
                if hasattr(self._model, "predict_proba"):
                    proba = self._model.predict_proba(payload)[0]
                    model_confidence = float(max(proba))
                heuristic_details["model_label"] = float(self._legacy_id(model_label))
                heuristic_details["model_confidence"] = float(model_confidence)
                result = RegimeClassification(
                    label=legacy_label,
                    confidence=max(confidence, model_confidence),
                    source=f"omega_{omega_state.source}+model",
                    details=heuristic_details,
                    state_label=state_label,
                )
                self._append_history(result)
                return result
            except Exception:
                pass

        result = RegimeClassification(
            label=legacy_label,
            confidence=confidence,
            source=f"omega_{omega_state.source}",
            details=heuristic_details,
            state_label=state_label,
        )
        self._append_history(result)
        return result

    @staticmethod
    def _heuristic_details(row: pd.Series) -> dict[str, float]:
        atr_ratio = float(row.get("m5_atr_pct_of_avg", 1.0))
        adx = float(row.get("h1_adx_14", row.get("m15_adx_14", 20.0)))
        trend_gap = float(row.get("h1_ema_gap_50_200", 0.0))
        spread_ratio = float(row.get("m5_spread_ratio_20", 1.0))
        ema_slope = float(row.get("m15_ema_gap_20_50", 0.0))
        market_instability = float(row.get("market_instability_score", 0.0) or 0.0)
        feature_drift = float(row.get("feature_drift_score", 0.0) or 0.0)
        multi_tf_alignment = float(row.get("multi_tf_alignment_score", 0.5) or 0.5)
        seasonality_edge = float(row.get("seasonality_edge_score", 0.5) or 0.5)
        fractal_persistence = float(row.get("fractal_persistence_score", 0.5) or 0.5)
        hurst_persistence = float(row.get("m5_hurst_proxy_64", 0.5) or 0.5)
        trend_flag = 1.0 if (adx >= 22 and abs(ema_slope) > 0.0) else 0.0
        range_flag = 1.0 if adx < 20 else 0.0
        volatility_high = 1.0 if (atr_ratio >= 1.6 or spread_ratio >= 1.4) else 0.0
        volatility_low = 1.0 if atr_ratio <= 0.85 else 0.0
        return {
            "atr_ratio": atr_ratio,
            "adx": adx,
            "trend_gap": trend_gap,
            "trend_gap_abs": abs(trend_gap),
            "ema_slope": ema_slope,
            "spread_ratio": spread_ratio,
            "trend_flag": trend_flag,
            "range_flag": range_flag,
            "volatility_high": volatility_high,
            "volatility_low": volatility_low,
            "market_instability_score": market_instability,
            "feature_drift_score": feature_drift,
            "multi_tf_alignment_score": multi_tf_alignment,
            "seasonality_edge_score": seasonality_edge,
            "fractal_persistence_score": fractal_persistence,
            "hurst_persistence_score": hurst_persistence,
        }

    @staticmethod
    def _news_shock_score(row: pd.Series, details: dict[str, float]) -> float:
        spread_ratio = float(details.get("spread_ratio", 1.0))
        atr_ratio = float(details.get("atr_ratio", 1.0))
        ret_1 = abs(float(row.get("m5_ret_1", 0.0)))
        return max(0.0, (spread_ratio - 1.4)) + max(0.0, (atr_ratio - 1.8)) + (ret_1 * 8.0)

    @staticmethod
    def _rich_regime_state(
        *,
        row: pd.Series,
        details: dict[str, float],
        omega_label: str,
    ) -> tuple[str, float, dict[str, float | str]]:
        atr_ratio = float(details.get("atr_ratio", 1.0))
        spread_ratio = float(details.get("spread_ratio", 1.0))
        adx = float(details.get("adx", 20.0))
        trend_gap = float(details.get("trend_gap", 0.0))
        ema_slope = float(details.get("ema_slope", 0.0))
        range_position = clamp(float(row.get("m15_range_position_20", row.get("m5_range_position_20", 0.5)) or 0.5), 0.0, 1.0)
        trend_flag = bool(details.get("trend_flag", 0.0))
        range_flag = bool(details.get("range_flag", 0.0))
        market_instability = clamp(float(details.get("market_instability_score", 0.0) or 0.0), 0.0, 1.0)
        feature_drift = clamp(float(details.get("feature_drift_score", 0.0) or 0.0), 0.0, 1.0)
        multi_tf_alignment = clamp(float(details.get("multi_tf_alignment_score", 0.5) or 0.5), 0.0, 1.0)
        seasonality_edge = clamp(float(details.get("seasonality_edge_score", 0.5) or 0.5), 0.0, 1.0)
        fractal_persistence = clamp(float(details.get("fractal_persistence_score", 0.5) or 0.5), 0.0, 1.0)
        hurst_persistence = clamp(float(details.get("hurst_persistence_score", 0.5) or 0.5), 0.0, 1.0)
        m5_ret_1 = float(row.get("m5_ret_1", 0.0))
        m5_ret_3 = float(row.get("m5_ret_3", 0.0))
        m1_ret_1 = float(row.get("m1_ret_1", 0.0))
        breakout_flag = bool(int(float(row.get("m5_breakout_flag", 0.0) or 0.0)) == 1)
        body_efficiency = clamp(abs(float(row.get("m5_body_efficiency", row.get("m5_candle_efficiency", 0.55)))), 0.0, 1.0)
        upper_wick = clamp(float(row.get("m5_upper_wick_ratio", 0.0)), 0.0, 1.0)
        lower_wick = clamp(float(row.get("m5_lower_wick_ratio", 0.0)), 0.0, 1.0)
        wick_rejection = max(upper_wick, lower_wick)
        sweep_flag = bool(
            int(float(row.get("m5_liquidity_sweep", row.get("m5_sweep", 0.0)) or 0.0)) == 1
            or int(float(row.get("m5_pinbar_bull", 0.0) or 0.0)) == 1
            or int(float(row.get("m5_pinbar_bear", 0.0) or 0.0)) == 1
        )
        compression_score = clamp(1.15 - atr_ratio, 0.0, 1.0)
        volatility_pressure = clamp((atr_ratio - 1.0) / 1.2, 0.0, 1.0)
        spread_compression = clamp((1.20 - spread_ratio) / 0.30, 0.0, 1.0)
        range_tightness = clamp(1.0 - abs(range_position - 0.50) * 2.0, 0.0, 1.0)
        compression_expansion_score = clamp(
            (0.45 * compression_score)
            + (0.25 * spread_compression)
            + (0.15 * range_tightness)
            + (0.15 * max(0.0, 1.0 - abs(m5_ret_1) * 900.0)),
            0.0,
            1.0,
        )
        compression_expansion_score = clamp(
            compression_expansion_score
            + (0.08 * multi_tf_alignment)
            + (0.05 * fractal_persistence)
            - (0.10 * market_instability),
            0.0,
            1.0,
        )
        compression_proxy_state = "NEUTRAL"
        if atr_ratio <= 0.92 and spread_ratio <= 1.20 and abs(m5_ret_1) <= 0.0012 and 0.35 <= range_position <= 0.65:
            compression_proxy_state = "COMPRESSION"
        elif (
            compression_expansion_score >= 0.42
            and (
                (range_position >= 0.70 and m5_ret_1 > 0.0)
                or (range_position <= 0.30 and m5_ret_1 < 0.0)
            )
        ):
            compression_proxy_state = "EXPANSION_READY"
        trend_persistence = clamp(
            max(
                float(row.get("m5_trend_persistence", 0.0) or 0.0),
                (abs(trend_gap) * 3.5) + (abs(ema_slope) * 50.0) + (0.25 if trend_flag else 0.0),
            ),
            0.0,
            1.0,
        )
        trend_persistence = clamp(
            trend_persistence
            + (0.10 * multi_tf_alignment)
            + (0.08 * fractal_persistence)
            + (0.06 * hurst_persistence)
            - (0.10 * market_instability),
            0.0,
            1.0,
        )
        impulse_strength = clamp(
            max(
                float(row.get("m5_impulse_strength", 0.0) or 0.0),
                (abs(m5_ret_1) * 220.0) + (abs(m5_ret_3) * 85.0) + (abs(m1_ret_1) * 260.0),
            ),
            0.0,
            1.0,
        )
        news_shock = float(details.get("news_shock_score", 0.0))
        if str(omega_label or "").upper() == "NEWS_SHOCK" or news_shock >= 0.80:
            label = "NEWS_DISTORTION"
            confidence = clamp(max(0.72, news_shock), 0.0, 1.0)
        elif sweep_flag and wick_rejection >= 0.35:
            label = "LIQUIDITY_SWEEP"
            confidence = clamp(0.55 + (wick_rejection * 0.35) + (0.10 if compression_score > 0.35 else 0.0), 0.0, 1.0)
        elif atr_ratio >= 1.7 and spread_ratio >= 1.15:
            label = "VOLATILITY_SPIKE"
            confidence = clamp(0.55 + (volatility_pressure * 0.40), 0.0, 1.0)
        elif (trend_flag or breakout_flag) and (adx >= 22.0 or trend_persistence >= 0.58) and impulse_strength >= 0.35:
            label = "TREND_EXPANSION"
            confidence = clamp(0.52 + (0.25 * trend_persistence) + (0.23 * impulse_strength), 0.0, 1.0)
        elif range_flag and wick_rejection >= 0.28:
            label = "MEAN_REVERSION"
            confidence = clamp(0.48 + (0.28 * wick_rejection) + (0.18 * compression_score), 0.0, 1.0)
        elif atr_ratio <= 0.82 and spread_ratio <= 1.15 and impulse_strength <= 0.28:
            label = "QUIET_ACCUMULATION"
            confidence = clamp(0.50 + (0.30 * compression_score) + (0.10 * (1.0 - wick_rejection)), 0.0, 1.0)
        elif compression_score >= 0.45 and impulse_strength >= 0.20:
            label = "BREAKOUT_COMPRESSION"
            confidence = clamp(0.48 + (0.24 * compression_score) + (0.18 * impulse_strength), 0.0, 1.0)
        else:
            label = "LOW_LIQUIDITY_DRIFT"
            confidence = clamp(0.40 + (0.20 * (1.0 - trend_persistence)) + (0.15 * compression_score), 0.0, 1.0)
        if market_instability >= 0.75 and trend_persistence <= 0.42 and impulse_strength <= 0.42:
            label = "LOW_LIQUIDITY_DRIFT"
            confidence = clamp(max(confidence, 0.55 + (0.25 * market_instability)), 0.0, 1.0)
        volatility_state = "BALANCED"
        if compression_score >= 0.50:
            volatility_state = "COMPRESSION"
        elif label == "VOLATILITY_SPIKE":
            volatility_state = "SPIKE"
        elif impulse_strength >= 0.55 and trend_persistence >= 0.45:
            volatility_state = "EXPANSION_IMMINENT"
        elif atr_ratio >= 1.45 and impulse_strength < 0.25:
            volatility_state = "EXPANSION_EXHAUSTION"
        continuation_pressure = clamp((0.45 * impulse_strength) + (0.35 * trend_persistence) + (0.20 * body_efficiency), 0.0, 1.0)
        exhaustion_signal = clamp((0.40 * wick_rejection) + (0.30 * volatility_pressure) + (0.30 * max(0.0, 0.45 - impulse_strength)), 0.0, 1.0)
        absorption_signal = clamp((0.55 * wick_rejection) + (0.20 * compression_score) + (0.25 * max(0.0, 1.0 - body_efficiency)), 0.0, 1.0)
        return (
            label,
            confidence,
            {
                "regime_state": label,
                "compression_score": compression_score,
                "compression_proxy_state": compression_proxy_state,
                "compression_proxy_score": compression_score if compression_proxy_state == "COMPRESSION" else 0.0,
                "compression_state": compression_proxy_state,
                "compression_expansion_score": compression_expansion_score,
                "compression_burst_ready": 1.0 if compression_proxy_state == "EXPANSION_READY" else 0.0,
                "volatility_pressure": volatility_pressure,
                "body_efficiency": body_efficiency,
                "wick_rejection_score": wick_rejection,
                "trend_persistence": trend_persistence,
                "impulse_strength": impulse_strength,
                "market_instability_score": market_instability,
                "feature_drift_score": feature_drift,
                "multi_tf_alignment_score": multi_tf_alignment,
                "seasonality_edge_score": seasonality_edge,
                "fractal_persistence_score": fractal_persistence,
                "hurst_persistence_score": hurst_persistence,
                "liquidity_sweep_flag": 1.0 if sweep_flag else 0.0,
                "volatility_forecast_state": volatility_state,
                "compression_detected": 1.0 if volatility_state == "COMPRESSION" else 0.0,
                "expansion_imminent": 1.0 if volatility_state == "EXPANSION_IMMINENT" else 0.0,
                "expansion_exhaustion": 1.0 if volatility_state == "EXPANSION_EXHAUSTION" else 0.0,
                "pressure_proxy_score": continuation_pressure,
                "continuation_pressure": continuation_pressure,
                "exhaustion_signal": exhaustion_signal,
                "absorption_signal": absorption_signal,
                "fake_breakout_risk": clamp((0.40 * wick_rejection) + (0.35 * max(0.0, 0.50 - trend_persistence)) + (0.25 * max(0.0, 0.45 - impulse_strength)), 0.0, 1.0),
            },
        )

    def _resolve_omega_path(self) -> Path | None:
        if self.omega_model_path is not None:
            return self.omega_model_path
        default = self.model_path.with_suffix(".onnx")
        return default if default.exists() else None

    @staticmethod
    def _omega_id(label: str) -> int:
        mapping = {
            "TRENDING_UP": 1,
            "TRENDING_DOWN": 2,
            "RANGING": 3,
            "VOLATILE": 4,
            "NEWS_SHOCK": 5,
        }
        return int(mapping.get(str(label).upper(), 0))

    @staticmethod
    def _legacy_id(label: str) -> int:
        mapping = {"TRENDING": 1, "RANGING": 2, "VOLATILE": 3}
        return int(mapping.get(str(label).upper(), 0))

    def _load_model(self):
        if not self.model_path.exists():
            return None
        try:
            return joblib.load(self.model_path)
        except Exception:
            return None

    def _append_history(self, result: RegimeClassification) -> None:
        if not self.persist_history:
            return
        ensure_parent(self.history_path)
        payload = []
        if self.history_path.exists():
            try:
                payload = json.loads(self.history_path.read_text())
            except json.JSONDecodeError:
                payload = []
        payload.append(
            {
                "timestamp": utc_now().isoformat(),
                "label": result.label,
                "state_label": result.state_label,
                "confidence": result.confidence,
                "source": result.source,
                "details": result.details,
            }
        )
        self.history_path.write_text(json.dumps(payload[-1000:], indent=2))
