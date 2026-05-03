from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.neural_network import MLPRegressor

from src.backtest import label_trade_outcome
from src.feature_engineering import FeatureEngineer
from src.market_data import MarketDataService
from src.regime_detector import RegimeDetector
from src.strategy_engine import StrategyEngine
from src.utils import ensure_parent, utc_now


@dataclass
class Trainer:
    market_data: MarketDataService
    feature_engineer: FeatureEngineer
    strategy_engine: StrategyEngine
    regime_detector: RegimeDetector
    model_paths: dict[str, Path]
    train_ratio: float
    validation_ratio: float
    test_ratio: float

    def run(self, symbol: str, counts: dict[str, int] | None = None) -> dict[str, Any]:
        requested_counts = counts or {"M1": 120000, "M5": 50000, "M15": 20000, "H1": 5000, "H4": 2000}
        frames = self.market_data.latest_multi_timeframe(symbol, requested_counts)
        features = self.feature_engineer.build(frames.get("M1"), frames["M5"], frames["M15"], frames["H1"], frames.get("H4"))
        dataset = self._build_dataset(features)
        if dataset.empty:
            raise RuntimeError("No trainable samples were generated from the current strategy")

        feature_columns = [column for column in dataset.columns if column not in {"label_win", "label_r", "regime_label"}]
        X = dataset[feature_columns]
        y_cls = dataset["label_win"]
        y_reg = dataset["label_r"]

        train_end = int(len(dataset) * self.train_ratio)
        valid_end = train_end + int(len(dataset) * self.validation_ratio)
        X_train, X_valid, X_test = X.iloc[:train_end], X.iloc[train_end:valid_end], X.iloc[valid_end:]
        y_train_cls, y_valid_cls, y_test_cls = y_cls.iloc[:train_end], y_cls.iloc[train_end:valid_end], y_cls.iloc[valid_end:]
        y_train_reg, y_valid_reg, y_test_reg = y_reg.iloc[:train_end], y_reg.iloc[train_end:valid_end], y_reg.iloc[valid_end:]

        scorer = GradientBoostingClassifier(random_state=7)
        scorer.fit(X_train, y_train_cls)
        scorer_valid = scorer.predict(X_valid)

        value_model = GradientBoostingRegressor(random_state=7)
        value_model.fit(X_train, y_train_reg)
        value_valid = value_model.predict(X_valid)

        risk_target = np.clip(0.5 + (y_train_reg / 4.0), 0.25, 1.0)
        risk_model = MLPRegressor(hidden_layer_sizes=(16, 8), max_iter=500, random_state=7)
        risk_model.fit(X_train, risk_target)

        regime_model = RandomForestClassifier(n_estimators=100, random_state=7)
        regime_labels = dataset["regime_label"].iloc[:train_end]
        regime_model.fit(X_train, regime_labels)

        classifier_test_accuracy = None
        value_test_mae = None
        if len(y_test_cls):
            classifier_test_accuracy = accuracy_score(y_test_cls, scorer.predict(X_test))
        if len(y_test_reg):
            value_test_mae = mean_absolute_error(y_test_reg, value_model.predict(X_test))

        metrics = {
            "classifier_valid_accuracy": accuracy_score(y_valid_cls, scorer_valid) if len(y_valid_cls) else None,
            "classifier_test_accuracy": classifier_test_accuracy,
            "value_valid_mae": mean_absolute_error(y_valid_reg, value_valid) if len(y_valid_reg) else None,
            "value_test_mae": value_test_mae,
            "sample_count": int(len(dataset)),
        }

        self._save_model(self.model_paths["trade_scorer"], scorer)
        self._save_model(self.model_paths["trade_value_model"], value_model)
        self._save_model(self.model_paths["risk_modulator"], risk_model)
        self._save_model(self.model_paths["regime_classifier"], regime_model)
        self._save_schema(self.model_paths["feature_schema"], feature_columns)
        self._save_metadata(self.model_paths["metadata"], metrics)
        return metrics

    def _build_dataset(self, features: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for index in range(250, len(features) - 50):
            row = features.iloc[index]
            regime = self.regime_detector.classify(row)
            candidates = self.strategy_engine.generate(features.iloc[: index + 1], regime.label)
            if not candidates:
                continue

            candidate = candidates[0]
            entry_price = float(features.iloc[index + 1]["m5_open"])
            atr = float(row["m5_atr_14"])
            stop_distance = atr * candidate.stop_atr
            if stop_distance <= 0:
                continue
            if candidate.side == "BUY":
                stop_price = entry_price - stop_distance
                tp_price = entry_price + (stop_distance * candidate.tp_r)
            else:
                stop_price = entry_price + stop_distance
                tp_price = entry_price - (stop_distance * candidate.tp_r)
            label_r, _ = label_trade_outcome(features, index + 1, candidate.side, entry_price, stop_price, tp_price)
            label_win = 1 if label_r > 0 else 0

            payload = {column: float(row[column]) for column in self.feature_engineer.feature_columns or []}
            payload["label_win"] = label_win
            payload["label_r"] = label_r
            payload["regime_label"] = regime.label
            rows.append(payload)
        return pd.DataFrame(rows)

    def _save_model(self, path: Path, model: Any) -> None:
        ensure_parent(path)
        joblib.dump(model, path)

    def _save_schema(self, path: Path, feature_columns: list[str]) -> None:
        ensure_parent(path)
        path.write_text(json.dumps({"version": "1.0.0", "features": feature_columns}, indent=2))

    def _save_metadata(self, path: Path, metrics: dict[str, Any]) -> None:
        ensure_parent(path)
        payload = {
            "version": "1.0.0",
            "trained_at": utc_now().isoformat(),
            "metrics": metrics,
        }
        path.write_text(json.dumps(payload, indent=2))
