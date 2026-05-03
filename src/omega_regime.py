from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.utils import clamp


OMEGA_LABELS: tuple[str, ...] = (
    "TRENDING_UP",
    "TRENDING_DOWN",
    "RANGING",
    "VOLATILE",
    "NEWS_SHOCK",
)


@dataclass(frozen=True)
class OmegaRegimeState:
    label: str
    confidence: float
    source: str
    features: dict[str, float]
    probabilities: dict[str, float]


class OmegaRegimeDetector:
    def __init__(self, model_path: Path | None = None, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.model_path = model_path
        self._session = None
        self._input_name: str | None = None
        if self.enabled and model_path is not None and model_path.exists():
            try:
                import onnxruntime as ort  # type: ignore

                self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
                inputs = self._session.get_inputs()
                if inputs:
                    self._input_name = str(inputs[0].name)
            except Exception:
                self._session = None
                self._input_name = None

    def classify(self, payload: dict[str, Any]) -> OmegaRegimeState:
        features = self._normalize_features(payload)
        if self._session is not None and self._input_name:
            try:
                tensor = np.array(
                    [
                        [
                            features["atr_ratio"],
                            features["adx"],
                            features["trend_gap"],
                            features["ema_slope"],
                            features["spread_ratio"],
                            features["momentum"],
                            features["news_shock_score"],
                            features["trend_direction_hint"],
                        ]
                    ],
                    dtype=np.float32,
                )
                raw = self._session.run(None, {self._input_name: tensor})[0]
                probabilities = self._extract_probabilities(raw)
                label, confidence = self._pick(probabilities)
                return OmegaRegimeState(
                    label=label,
                    confidence=confidence,
                    source="onnx",
                    features=features,
                    probabilities=probabilities,
                )
            except Exception:
                pass

        label = self._heuristic_label(features)
        probabilities = self._heuristic_probabilities(label, features)
        _, confidence = self._pick(probabilities)
        return OmegaRegimeState(
            label=label,
            confidence=confidence,
            source="rules",
            features=features,
            probabilities=probabilities,
        )

    @staticmethod
    def to_legacy_label(omega_label: str) -> str:
        normalized = str(omega_label or "").upper()
        if normalized in {"TRENDING_UP", "TRENDING_DOWN"}:
            return "TRENDING"
        if normalized == "RANGING":
            return "RANGING"
        return "VOLATILE"

    @staticmethod
    def _normalize_features(payload: dict[str, Any]) -> dict[str, float]:
        regime_hint = str(payload.get("regime_hint", "")).upper()
        side_hint = str(payload.get("side_hint", "")).upper()
        direction_hint = 0.0
        if "UP" in regime_hint or side_hint == "BUY":
            direction_hint = 1.0
        elif "DOWN" in regime_hint or side_hint == "SELL":
            direction_hint = -1.0

        return {
            "atr_ratio": max(0.0, float(payload.get("atr_ratio", 1.0) or 1.0)),
            "adx": max(0.0, float(payload.get("adx", 20.0) or 20.0)),
            "trend_gap": float(payload.get("trend_gap", 0.0) or 0.0),
            "ema_slope": float(payload.get("ema_slope", 0.0) or 0.0),
            "spread_ratio": max(0.0, float(payload.get("spread_ratio", 1.0) or 1.0)),
            "momentum": float(payload.get("momentum", 0.0) or 0.0),
            "news_shock_score": max(0.0, float(payload.get("news_shock_score", 0.0) or 0.0)),
            "trend_direction_hint": direction_hint,
        }

    @staticmethod
    def _heuristic_label(features: dict[str, float]) -> str:
        if features["news_shock_score"] >= 1.0:
            return "NEWS_SHOCK"
        if features["atr_ratio"] >= 1.9 or features["spread_ratio"] >= 1.6:
            return "VOLATILE"
        trend_strength = (features["adx"] / 50.0) + abs(features["trend_gap"]) + abs(features["ema_slope"])
        if trend_strength >= 0.9 and features["adx"] >= 22.0:
            if (features["trend_gap"] + features["ema_slope"] + (features["momentum"] * 0.25)) >= 0:
                return "TRENDING_UP"
            return "TRENDING_DOWN"
        if features["trend_direction_hint"] > 0.5:
            return "TRENDING_UP"
        if features["trend_direction_hint"] < -0.5:
            return "TRENDING_DOWN"
        return "RANGING"

    @staticmethod
    def _heuristic_probabilities(label: str, features: dict[str, float]) -> dict[str, float]:
        base = {key: 0.05 for key in OMEGA_LABELS}
        base[label] = 0.65
        if label in {"TRENDING_UP", "TRENDING_DOWN"}:
            base[label] = clamp(0.55 + (features["adx"] / 140.0) + (abs(features["trend_gap"]) * 0.2), 0.5, 0.9)
            base["RANGING"] = 0.15
        elif label == "RANGING":
            base["RANGING"] = clamp(0.55 + ((1.2 - min(features["atr_ratio"], 1.2)) * 0.2), 0.5, 0.9)
        elif label == "VOLATILE":
            base["VOLATILE"] = clamp(0.6 + ((max(features["atr_ratio"], features["spread_ratio"]) - 1.0) * 0.18), 0.55, 0.92)
        elif label == "NEWS_SHOCK":
            base["NEWS_SHOCK"] = clamp(0.7 + (features["news_shock_score"] * 0.1), 0.65, 0.97)

        total = sum(base.values())
        if total <= 0:
            return {key: 0.2 for key in OMEGA_LABELS}
        return {key: value / total for key, value in base.items()}

    @staticmethod
    def _extract_probabilities(raw: Any) -> dict[str, float]:
        values: list[float] = []
        if isinstance(raw, np.ndarray):
            flattened = raw.reshape(-1).tolist()
            values = [float(item) for item in flattened]
        elif isinstance(raw, (list, tuple)):
            for item in raw:
                try:
                    values.append(float(item))
                except Exception:
                    continue
        if not values:
            return {key: 1.0 / len(OMEGA_LABELS) for key in OMEGA_LABELS}
        if len(values) < len(OMEGA_LABELS):
            values = values + [0.0] * (len(OMEGA_LABELS) - len(values))
        clipped = np.clip(np.array(values[: len(OMEGA_LABELS)], dtype=np.float32), 0.0, None)
        if float(clipped.sum()) <= 0:
            clipped = np.array([1.0] * len(OMEGA_LABELS), dtype=np.float32)
        clipped = clipped / clipped.sum()
        return {label: float(clipped[idx]) for idx, label in enumerate(OMEGA_LABELS)}

    @staticmethod
    def _pick(probabilities: dict[str, float]) -> tuple[str, float]:
        label = max(probabilities, key=lambda key: float(probabilities.get(key, 0.0)))
        confidence = float(probabilities.get(label, 0.0))
        return label, confidence

