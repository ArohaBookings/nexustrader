from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import csv
import hashlib
import json
import threading
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

from src.external_market_data import YahooMarketDataFallback
from src.execution import current_trading_day_key
from src.logger import ApexLogger
from src.symbol_universe import normalize_symbol_key
from src.utils import clamp, ensure_parent, utc_now

warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn\.utils\.extmath")
warnings.filterwarnings("ignore", category=ConvergenceWarning, module=r"sklearn\.linear_model\._stochastic_gradient")
warnings.filterwarnings("ignore", message=".*encountered in matmul.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*Maximum number of iteration reached before convergence.*", category=ConvergenceWarning)


TRADE_HISTORY_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "symbol",
    "timeframe",
    "features_used_hash",
    "side",
    "entry",
    "sl",
    "tp",
    "exit",
    "pnl_r",
    "pnl_money",
    "news_state",
    "session_state",
    "session_name",
    "ai_decision",
    "result",
    "ai_probability",
    "spread_points",
    "lot",
    "regime",
    "regime_state",
    "setup",
    "strategy_key",
    "lane_name",
    "management_template",
    "strategy_state",
    "regime_fit",
    "session_fit",
    "volatility_fit",
    "pair_behavior_fit",
    "execution_quality_fit",
    "entry_timing_score",
    "structure_cleanliness_score",
    "strategy_recent_performance",
    "market_data_source",
    "market_data_consensus_state",
    "multi_tf_alignment_score",
    "fractal_persistence_score",
    "compression_expansion_score",
    "dxy_support_score",
    "swing_continuation_score",
    "aggressive_pair_mode",
    "trajectory_catchup_pressure",
    "institutional_confluence_score",
    "candle_mastery_score",
    "live_shadow_gap_score",
    "execution_edge_score",
    "xau_engine",
    "grid_source_role",
    "native_burst_window_id",
    "mc_win_rate",
    "ga_generation_id",
    "reentry_source_tag",
    "signal_id",
)


def _sanitize_feature_matrix(values: np.ndarray, *, clip_value: float = 8.0) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(-1, 1)
    sanitized = np.nan_to_num(array, nan=0.0, posinf=float(clip_value), neginf=-float(clip_value))
    return np.clip(sanitized, -float(clip_value), float(clip_value))

LEGACY_TRADE_HISTORY_COLUMNS_V1: tuple[str, ...] = (
    "timestamp_utc",
    "symbol",
    "timeframe",
    "features_used_hash",
    "side",
    "entry",
    "sl",
    "tp",
    "exit",
    "pnl_r",
    "pnl_money",
    "news_state",
    "session_state",
    "ai_decision",
    "result",
    "ai_probability",
    "spread_points",
    "lot",
    "regime",
    "setup",
    "signal_id",
)

SETUP_LOG_COLUMNS: tuple[str, ...] = (
    "timestamp_utc",
    "signal_id",
    "symbol",
    "setup",
    "timeframe",
    "decision",
    "decision_type",
    "accepted",
    "result",
    "pnl_r",
    "pnl_money",
    "ai_probability",
    "reason",
    "news_state",
    "session_state",
    "session_name",
    "regime",
    "strategy_key",
    "lane_name",
    "management_template",
    "strategy_state",
    "regime_fit",
    "session_fit",
    "volatility_fit",
    "pair_behavior_fit",
    "execution_quality_fit",
    "entry_timing_score",
    "structure_cleanliness_score",
    "strategy_recent_performance",
    "market_data_source",
    "market_data_consensus_state",
    "multi_tf_alignment_score",
    "fractal_persistence_score",
    "compression_expansion_score",
    "dxy_support_score",
    "swing_continuation_score",
    "aggressive_pair_mode",
    "trajectory_catchup_pressure",
    "institutional_confluence_score",
    "candle_mastery_score",
    "live_shadow_gap_score",
    "execution_edge_score",
    "xau_engine",
    "grid_source_role",
    "native_burst_window_id",
    "mc_win_rate",
    "ga_generation_id",
    "reentry_source_tag",
    "mode",
    "source",
)

LEGACY_SETUP_LOG_COLUMNS_V1: tuple[str, ...] = (
    "timestamp_utc",
    "signal_id",
    "symbol",
    "setup",
    "timeframe",
    "decision",
    "decision_type",
    "accepted",
    "result",
    "pnl_r",
    "pnl_money",
    "ai_probability",
    "reason",
    "news_state",
    "session_state",
    "regime",
    "mode",
    "source",
)


def _default_csv_value(column: str) -> Any:
    if column in {"accepted"}:
        return 0
    if column in {
        "entry",
        "sl",
        "tp",
        "exit",
        "pnl_r",
        "pnl_money",
        "ai_probability",
        "spread_points",
        "lot",
        "regime_fit",
        "session_fit",
        "volatility_fit",
        "pair_behavior_fit",
        "execution_quality_fit",
        "entry_timing_score",
        "structure_cleanliness_score",
        "strategy_recent_performance",
        "multi_tf_alignment_score",
        "fractal_persistence_score",
        "compression_expansion_score",
        "dxy_support_score",
        "swing_continuation_score",
        "aggressive_pair_mode",
        "trajectory_catchup_pressure",
        "institutional_confluence_score",
        "candle_mastery_score",
        "live_shadow_gap_score",
        "execution_edge_score",
        "mc_win_rate",
        "ga_generation_id",
    }:
        return 0.0
    return ""


def _normalize_csv_records(
    *,
    path: Path,
    current_columns: tuple[str, ...],
    legacy_columns: tuple[tuple[str, ...], ...],
) -> tuple[list[dict[str, Any]], bool]:
    if not path.exists():
        return [], False
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        return [], False
    header = tuple(str(item) for item in rows[0])
    data_rows = rows[1:]
    normalized_rows: list[dict[str, Any]] = []
    needs_repair = header != current_columns
    schema_options = [header, current_columns, *legacy_columns]
    for row in data_rows:
        if not row:
            continue
        matched_columns = next((columns for columns in schema_options if len(columns) == len(row)), None)
        if matched_columns is None:
            if len(row) < len(current_columns):
                matched_columns = current_columns[: len(row)]
            else:
                matched_columns = current_columns
            needs_repair = True
        elif matched_columns != header:
            needs_repair = True
        payload = {column: _default_csv_value(column) for column in current_columns}
        for index, column in enumerate(matched_columns[: len(row)]):
            if column in payload:
                payload[column] = row[index]
        normalized_rows.append(payload)
    return normalized_rows, needs_repair


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: Any, default: float = 0.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = float(default)
    return max(0.0, min(1.0, numeric))


def _symbol_code(symbol: str) -> float:
    digest = hashlib.sha256(symbol.upper().encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1000.0


def _normalize_symbol_key(value: Any) -> str:
    return normalize_symbol_key(value)


_SUPER_AGGRESSIVE_LEARNING_SYMBOLS: set[str] = {
    "AUDJPY",
    "NZDJPY",
    "USDJPY",
    "AUDNZD",
    "EURUSD",
    "GBPUSD",
    "EURGBP",
    "EURJPY",
    "GBPJPY",
    "NAS100",
    "USOIL",
    "XAGUSD",
    "DOGUSD",
    "TRUMPUSD",
    "AAPL",
    "NVIDIA",
}


def _default_spread_points(symbol: str) -> float:
    symbol_key = _normalize_symbol_key(symbol)
    return {
        "XAUUSD": 25.0,
        "XAGUSD": 30.0,
        "EURUSD": 12.0,
        "GBPUSD": 16.0,
        "EURGBP": 14.0,
        "USDJPY": 14.0,
        "AUDJPY": 18.0,
        "NZDJPY": 18.0,
        "AUDNZD": 24.0,
        "EURJPY": 18.0,
        "GBPJPY": 22.0,
        "BTCUSD": 180.0,
        "DOGUSD": 220.0,
        "TRUMPUSD": 280.0,
        "NAS100": 45.0,
        "USOIL": 10.0,
        "AAPL": 8.0,
        "NVIDIA": 10.0,
    }.get(symbol_key, 15.0)


def _session_name_from_timestamp(value: Any) -> str:
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return "OUT"
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    hour = int(timestamp.hour)
    if 0 <= hour < 7:
        return "TOKYO"
    if 7 <= hour < 13:
        return "LONDON"
    if 13 <= hour < 16:
        return "OVERLAP"
    if 16 <= hour < 21:
        return "NEW_YORK"
    return "SYDNEY"


def _hash_code(value: Any) -> float:
    text = str(value or "").strip().upper()
    if not text:
        return 0.0
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) % 1000) / 1000.0


def _strategy_state_score(value: Any) -> float:
    state = str(value or "").strip().upper()
    return {
        "ATTACK": 1.0,
        "NORMAL": 0.7,
        "REDUCED": 0.4,
        "QUARANTINED": 0.1,
    }.get(state, 0.7)


def _consensus_score(value: Any) -> float:
    state = str(value or "").strip().upper()
    if state in {"ALIGNED", "CONSENSUS", "STRONG_CONSENSUS"}:
        return 1.0
    if state in {"SOFT_MISMATCH", "MIXED"}:
        return 0.5
    if state in {"CONFLICT", "DISAGREE", "DIVERGENT"}:
        return 0.1
    return 0.75


@dataclass
class OnlineLearningEngine:
    data_path: Path
    model_path: Path
    setup_log_path: Path | None = None
    history_cache_dir: Path | None = None
    min_retrain_trades: int = 50
    min_score_floor: float = 0.35
    rolling_window_trades: int = 1000
    rolling_window_days: int = 90
    maintenance_interval_hours: int = 4
    promotion_min_delta: float = 0.02
    promotion_min_samples: int = 100
    market_history_seed_enabled: bool = True
    market_history_seed_max_samples: int = 3000
    market_history_seed_forward_bars: int = 8
    market_history_seed_min_rows_per_file: int = 240
    market_history_backfill_enabled: bool = True
    market_history_backfill_years: int = 10
    market_history_universe_symbols: tuple[str, ...] = ()
    logger: ApexLogger | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()
        if self.setup_log_path is None:
            self.setup_log_path = self.data_path.with_name("setups_log.csv")
        if self.history_cache_dir is None:
            self.history_cache_dir = self.data_path.with_name("candles_cache")
        self.market_history_universe_symbols = tuple(
            symbol
            for symbol in (_normalize_symbol_key(item) for item in self.market_history_universe_symbols)
            if symbol
        )
        self._feature_cols = [
            "symbol_code",
            "side_buy",
            "rr",
            "sl_dist",
            "tp_dist",
            "ai_probability",
            "spread_points",
            "news_safe",
            "session_in",
            "lot",
            "regime_code",
            "setup_code",
            "strategy_code",
            "session_code",
            "lane_code",
            "management_code",
            "strategy_state_score",
            "regime_fit",
            "session_fit",
            "volatility_fit",
            "pair_behavior_fit",
            "execution_quality_fit",
            "entry_timing_score",
            "structure_cleanliness_score",
            "strategy_recent_performance",
            "market_data_consensus_score",
            "multi_tf_alignment_score",
            "fractal_persistence_score",
            "compression_expansion_score",
            "dxy_support_score",
            "mc_win_rate",
            "ga_generation_norm",
            "swing_continuation_score",
            "aggressive_pair_mode",
            "institutional_confluence_score",
            "candle_mastery_score",
            "live_shadow_gap_score",
            "execution_edge_score",
            "trajectory_catchup_pressure",
        ]
        self._model: Any | None = None
        self._initialized = False
        self._trained_samples = 0
        self._pending_samples = 0
        self._last_maintenance_retrain_at: str | None = None
        self._last_maintenance_day: str | None = None
        self._last_maintenance_status = "idle"
        self._last_maintenance_error = ""
        self._last_history_repair_at: str | None = None
        self._last_market_history_seed_at: str | None = None
        self._last_market_history_seed_status = "idle"
        self._last_market_history_seed_samples = 0
        self._last_market_history_backfill_at: str | None = None
        self._last_market_history_backfill_status = "idle"
        self._last_market_history_backfill_files = 0
        self._ga_generation_id = 0
        self._last_promotion: dict[str, float] = {"baseline_acc": 0.0, "candidate_acc": 0.0, "holdout": 0.0}
        self._ensure_storage_schema()
        self._load_model()

    def predict_score(self, payload: dict[str, Any]) -> float:
        row = self._feature_row(payload)
        if self._model is None or not self._initialized:
            return 0.5
        with self._lock:
            try:
                score = float(self._model.predict_proba(row)[0][1])
                return max(0.0, min(1.0, score))
            except Exception:
                return 0.5

    def on_trade_close(self, payload: dict[str, Any]) -> None:
        row_payload = self._csv_row(payload)
        with self._lock:
            self._append_csv_row(row_payload)
            self._append_setup_decision(
                {
                    "timestamp_utc": row_payload.get("timestamp_utc"),
                    "signal_id": row_payload.get("signal_id", ""),
                    "symbol": row_payload.get("symbol", ""),
                    "setup": row_payload.get("setup", ""),
                    "timeframe": row_payload.get("timeframe", "M5"),
                    "decision": "outcome",
                    "decision_type": "outcome",
                    "accepted": True,
                    "result": "win" if float(row_payload.get("pnl_r", 0.0)) >= 0 else "loss",
                    "pnl_r": float(row_payload.get("pnl_r", 0.0)),
                    "pnl_money": float(row_payload.get("pnl_money", 0.0)),
                    "ai_probability": float(row_payload.get("ai_probability", 0.5)),
                    "reason": "trade_closed",
                    "news_state": row_payload.get("news_state", "unknown"),
                    "session_state": row_payload.get("session_state", "OUT"),
                    "session_name": row_payload.get("session_name", ""),
                    "regime": row_payload.get("regime", "UNKNOWN"),
                    "strategy_key": row_payload.get("strategy_key", ""),
                    "lane_name": row_payload.get("lane_name", ""),
                    "management_template": row_payload.get("management_template", ""),
                    "strategy_state": row_payload.get("strategy_state", "NORMAL"),
                    "regime_fit": float(row_payload.get("regime_fit", 0.0)),
                    "session_fit": float(row_payload.get("session_fit", 0.0)),
                    "volatility_fit": float(row_payload.get("volatility_fit", 0.0)),
                    "pair_behavior_fit": float(row_payload.get("pair_behavior_fit", 0.0)),
                    "execution_quality_fit": float(row_payload.get("execution_quality_fit", 0.0)),
                    "entry_timing_score": float(row_payload.get("entry_timing_score", 0.0)),
                    "structure_cleanliness_score": float(row_payload.get("structure_cleanliness_score", 0.0)),
                    "strategy_recent_performance": float(row_payload.get("strategy_recent_performance", 0.0)),
                    "market_data_source": row_payload.get("market_data_source", ""),
                    "market_data_consensus_state": row_payload.get("market_data_consensus_state", ""),
                    "institutional_confluence_score": float(row_payload.get("institutional_confluence_score", 0.0)),
                    "candle_mastery_score": float(row_payload.get("candle_mastery_score", 0.0)),
                    "live_shadow_gap_score": float(row_payload.get("live_shadow_gap_score", 0.0)),
                    "execution_edge_score": float(row_payload.get("execution_edge_score", 0.0)),
                    "mode": "LIVE",
                    "source": "bridge_close",
                }
            )
            try:
                self._pending_samples += 1
                if self._pending_samples >= max(1, int(self.min_retrain_trades)):
                    self._retrain_from_history()
                    self._pending_samples = 0
            except Exception as exc:
                self._log("warning", f"online_learning_update_failed:{exc}")
            finally:
                try:
                    self._save_model()
                except Exception as exc:
                    self._log("warning", f"online_learning_model_save_failed:{exc}")

    def on_setup_decision(self, payload: dict[str, Any]) -> None:
        with self._lock:
            self._append_setup_decision(payload)

    def maybe_retrain_maintenance(
        self,
        *,
        now_utc: datetime | None = None,
        session_name: str = "",
        active_sessions: set[str] | None = None,
        force: bool = False,
    ) -> bool:
        now = now_utc or utc_now()
        session = str(session_name or "").upper()
        active = active_sessions or {"LONDON", "OVERLAP", "NEW_YORK", "TOKYO"}
        in_active = session in active
        day_key = current_trading_day_key(now_ts=now)
        nightly_window = now.hour in {0, 1, 2}
        should_run = bool(force) or self._pending_samples >= max(1, int(self.min_retrain_trades))
        if not should_run:
            if self._last_maintenance_retrain_at:
                try:
                    previous = datetime.fromisoformat(str(self._last_maintenance_retrain_at).replace("Z", "+00:00"))
                    if previous.tzinfo is None:
                        previous = previous.replace(tzinfo=utc_now().tzinfo)
                    elapsed_hours = max(0.0, (now - previous).total_seconds() / 3600.0)
                    if elapsed_hours >= float(max(1, int(self.maintenance_interval_hours))):
                        should_run = True
                except Exception:
                    pass
        if not should_run:
            if nightly_window and self._last_maintenance_day != day_key:
                should_run = True
            elif (not in_active) and self._last_maintenance_day != day_key:
                should_run = True
        if not should_run:
            return False
        with self._lock:
            try:
                self._ensure_storage_schema()
                self._retrain_from_history()
                self._pending_samples = 0
                self._last_maintenance_retrain_at = now.isoformat()
                self._last_maintenance_day = day_key
                if self._last_maintenance_status not in {"insufficient_class_balance", "insufficient_history_split"}:
                    self._last_maintenance_status = "ok"
                    self._last_maintenance_error = ""
                self._save_model()
                self._log(
                    "info",
                    f"online_learning_maintenance_retrain session={session} force={force} status={self._last_maintenance_status}",
                )
                return True
            except Exception as exc:
                self._last_maintenance_status = "failed"
                self._last_maintenance_error = str(exc)
                self._log("warning", f"online_learning_maintenance_failed:{exc}")
                return False

    def eval_last(self, limit: int) -> dict[str, float]:
        if not self.data_path.exists():
            return {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0, "profit_factor": 0.0, "max_drawdown_r": 0.0}
        frame = self._read_history_frame()
        if frame.empty:
            return {"trades": 0, "win_rate": 0.0, "expectancy_r": 0.0, "profit_factor": 0.0, "max_drawdown_r": 0.0}
        tail = frame.tail(max(1, int(limit)))
        pnl_r = tail["pnl_r"].astype(float).to_numpy()
        wins = pnl_r[pnl_r > 0]
        losses = np.abs(pnl_r[pnl_r < 0])
        win_rate = float((pnl_r >= 0).mean()) if len(pnl_r) else 0.0
        expectancy = float(pnl_r.mean()) if len(pnl_r) else 0.0
        gross_profit = float(wins.sum()) if len(wins) else 0.0
        gross_loss = float(losses.sum()) if len(losses) else 0.0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        equity = np.cumsum(pnl_r)
        peak = np.maximum.accumulate(equity) if len(equity) else np.array([])
        drawdown = peak - equity if len(equity) else np.array([])
        return {
            "trades": float(len(pnl_r)),
            "win_rate": win_rate,
            "expectancy_r": expectancy,
            "profit_factor": float(profit_factor),
            "max_drawdown_r": float(drawdown.max()) if len(drawdown) else 0.0,
        }

    def _feature_row(self, payload: dict[str, Any]) -> np.ndarray:
        symbol = str(payload.get("symbol", "XAUUSD"))
        side = str(payload.get("side", "BUY")).upper()
        entry = _safe_float(payload.get("entry"), 0.0)
        sl = _safe_float(payload.get("sl"), entry)
        tp = _safe_float(payload.get("tp"), entry)
        price_scale = max(abs(entry), 1.0)
        sl_dist = max(abs(entry - sl), 1e-6)
        tp_dist = max(abs(tp - entry), 1e-6)
        rr = clamp(tp_dist / sl_dist, 0.0, 8.0)
        sl_dist_norm = clamp((sl_dist / price_scale) * 1000.0, 1e-6, 8.0)
        tp_dist_norm = clamp((tp_dist / price_scale) * 1000.0, 1e-6, 12.0)
        ai_probability = _safe_float(payload.get("ai_probability"), 0.5)
        spread_points = clamp(_safe_float(payload.get("spread_points"), 0.0) / 1000.0, 0.0, 10.0)
        lot = clamp(_safe_float(payload.get("lot"), 0.0), 0.0, 5.0)
        news_state = str(payload.get("news_state", "unknown")).lower()
        session_state = str(payload.get("session_state", "OUT")).upper()
        session_name = str(payload.get("session_name") or payload.get("session_state") or "OUT").upper()
        regime = str(payload.get("regime", "UNKNOWN")).upper()
        regime_state = str(payload.get("regime_state") or regime).upper()
        setup = str(payload.get("setup", "UNKNOWN")).upper()
        strategy_key = str(payload.get("strategy_key", "UNKNOWN")).upper()
        lane_name = str(payload.get("lane_name", "UNASSIGNED")).upper()
        management_template = str(payload.get("management_template", "UNKNOWN")).upper()
        multi_tf_alignment_score = _clamp01(payload.get("multi_tf_alignment_score"), 0.5)
        fractal_persistence_score = _clamp01(payload.get("fractal_persistence_score"), 0.5)
        compression_expansion_score = _clamp01(payload.get("compression_expansion_score"), 0.0)
        dxy_support_score = _clamp01(
            payload.get("dxy_support_score"),
            0.5,
        )
        mc_win_rate = _clamp01(payload.get("mc_win_rate"), 0.5)
        ga_generation_norm = _clamp01(_safe_float(payload.get("ga_generation_id"), 0.0) / 100.0, 0.0)
        swing_continuation_score = _clamp01(payload.get("swing_continuation_score"), 0.0)
        aggressive_pair_mode = _clamp01(
            payload.get("aggressive_pair_mode"),
            1.0 if symbol in _SUPER_AGGRESSIVE_LEARNING_SYMBOLS else 0.0,
        )
        institutional_confluence_score = _clamp01(payload.get("institutional_confluence_score"), 0.5)
        candle_mastery_score = _clamp01(payload.get("candle_mastery_score"), 0.5)
        live_shadow_gap_score = _clamp01(payload.get("live_shadow_gap_score"), 0.0)
        execution_edge_score = _clamp01(
            payload.get("execution_edge_score"),
            _safe_float(payload.get("execution_quality_fit"), 0.7),
        )
        trajectory_catchup_pressure = _clamp01(payload.get("trajectory_catchup_pressure"), 0.0)
        regime_code = (int(hashlib.sha256(regime.encode("utf-8")).hexdigest()[:4], 16) % 1000) / 1000.0
        setup_code = (int(hashlib.sha256(setup.encode("utf-8")).hexdigest()[:4], 16) % 1000) / 1000.0
        strategy_code = _hash_code(strategy_key)
        session_code = _hash_code(session_name)
        lane_code = _hash_code(lane_name)
        management_code = _hash_code(management_template)
        row = np.array(
            [
                _symbol_code(symbol),
                1.0 if side == "BUY" else 0.0,
                rr,
                sl_dist_norm,
                tp_dist_norm,
                ai_probability,
                spread_points,
                0.0 if "block" in news_state else 1.0,
                1.0 if session_state == "IN" else 0.0,
                lot,
                regime_code,
                setup_code,
                strategy_code,
                session_code,
                lane_code,
                management_code,
                _strategy_state_score(payload.get("strategy_state")),
                _safe_float(payload.get("regime_fit"), 0.0),
                _safe_float(payload.get("session_fit"), 0.0),
                _safe_float(payload.get("volatility_fit"), 0.0),
                _safe_float(payload.get("pair_behavior_fit"), 0.0),
                _safe_float(payload.get("execution_quality_fit"), 0.0),
                _safe_float(payload.get("entry_timing_score"), 0.0),
                _safe_float(payload.get("structure_cleanliness_score"), 0.0),
                _safe_float(payload.get("strategy_recent_performance"), 0.0),
                _consensus_score(payload.get("market_data_consensus_state")),
                multi_tf_alignment_score,
                fractal_persistence_score,
                compression_expansion_score,
                dxy_support_score,
                mc_win_rate,
                ga_generation_norm,
                swing_continuation_score,
                aggressive_pair_mode,
                institutional_confluence_score,
                candle_mastery_score,
                live_shadow_gap_score,
                execution_edge_score,
                trajectory_catchup_pressure,
            ],
            dtype=float,
        ).reshape(1, -1)
        return row

    def _csv_row(self, payload: dict[str, Any]) -> dict[str, Any]:
        pnl_r = _safe_float(payload.get("pnl_r"), 0.0)
        row = {
            "timestamp_utc": str(payload.get("timestamp_utc") or utc_now().isoformat()),
            "symbol": str(payload.get("symbol", "XAUUSD")).upper(),
            "timeframe": str(payload.get("timeframe", "M5")).upper(),
            "features_used_hash": str(payload.get("features_used_hash", "")),
            "side": str(payload.get("side", "BUY")).upper(),
            "entry": _safe_float(payload.get("entry"), 0.0),
            "sl": _safe_float(payload.get("sl"), 0.0),
            "tp": _safe_float(payload.get("tp"), 0.0),
            "exit": _safe_float(payload.get("exit"), 0.0),
            "pnl_r": pnl_r,
            "pnl_money": _safe_float(payload.get("pnl_money"), 0.0),
            "news_state": str(payload.get("news_state", "unknown")),
            "session_state": str(payload.get("session_state", "OUT")),
            "session_name": str(payload.get("session_name") or payload.get("session_state", "OUT")),
            "ai_decision": str(payload.get("ai_decision", "")),
            "result": "win" if pnl_r >= 0 else "loss",
            "ai_probability": _safe_float(payload.get("ai_probability"), 0.5),
            "spread_points": _safe_float(payload.get("spread_points"), 0.0),
            "lot": _safe_float(payload.get("lot"), 0.0),
            "regime": str(payload.get("regime", "UNKNOWN")),
            "regime_state": str(payload.get("regime_state") or payload.get("regime", "UNKNOWN")),
            "setup": str(payload.get("setup", "UNKNOWN")),
            "strategy_key": str(payload.get("strategy_key", "")),
            "lane_name": str(payload.get("lane_name", "")),
            "management_template": str(payload.get("management_template", "")),
            "strategy_state": str(payload.get("strategy_state", "NORMAL")),
            "regime_fit": _safe_float(payload.get("regime_fit"), 0.0),
            "session_fit": _safe_float(payload.get("session_fit"), 0.0),
            "volatility_fit": _safe_float(payload.get("volatility_fit"), 0.0),
            "pair_behavior_fit": _safe_float(payload.get("pair_behavior_fit"), 0.0),
            "execution_quality_fit": _safe_float(payload.get("execution_quality_fit"), 0.0),
            "entry_timing_score": _safe_float(payload.get("entry_timing_score"), 0.0),
            "structure_cleanliness_score": _safe_float(payload.get("structure_cleanliness_score"), 0.0),
            "strategy_recent_performance": _safe_float(payload.get("strategy_recent_performance"), 0.0),
            "market_data_source": str(payload.get("market_data_source", "")),
            "market_data_consensus_state": str(payload.get("market_data_consensus_state", "")),
            "multi_tf_alignment_score": _safe_float(payload.get("multi_tf_alignment_score"), 0.0),
            "fractal_persistence_score": _safe_float(payload.get("fractal_persistence_score"), 0.0),
            "compression_expansion_score": _safe_float(payload.get("compression_expansion_score"), 0.0),
            "dxy_support_score": _safe_float(payload.get("dxy_support_score"), 0.0),
            "swing_continuation_score": _safe_float(payload.get("swing_continuation_score"), 0.0),
            "aggressive_pair_mode": _safe_float(payload.get("aggressive_pair_mode"), 0.0),
            "trajectory_catchup_pressure": _safe_float(payload.get("trajectory_catchup_pressure"), 0.0),
            "institutional_confluence_score": _safe_float(payload.get("institutional_confluence_score"), 0.0),
            "candle_mastery_score": _safe_float(payload.get("candle_mastery_score"), 0.0),
            "live_shadow_gap_score": _safe_float(payload.get("live_shadow_gap_score"), 0.0),
            "execution_edge_score": _safe_float(
                payload.get("execution_edge_score"),
                _safe_float(payload.get("execution_quality_fit"), 0.0),
            ),
            "xau_engine": str(payload.get("xau_engine", "")),
            "grid_source_role": str(payload.get("grid_source_role", "")),
            "native_burst_window_id": str(payload.get("native_burst_window_id", "")),
            "mc_win_rate": _safe_float(payload.get("mc_win_rate"), 0.0),
            "ga_generation_id": _safe_float(payload.get("ga_generation_id", self._ga_generation_id), float(self._ga_generation_id)),
            "reentry_source_tag": str(payload.get("reentry_source_tag", "")),
            "signal_id": str(payload.get("signal_id", "")),
        }
        return row

    def _append_csv_row(self, row: dict[str, Any]) -> None:
        self._append_ordered_row(
            path=self.data_path,
            row=row,
            current_columns=TRADE_HISTORY_COLUMNS,
            legacy_columns=(LEGACY_TRADE_HISTORY_COLUMNS_V1,),
        )

    def _append_setup_decision(self, payload: dict[str, Any]) -> None:
        if self.setup_log_path is None:
            return
        row = {
            "timestamp_utc": str(payload.get("timestamp_utc") or utc_now().isoformat()),
            "signal_id": str(payload.get("signal_id") or ""),
            "symbol": str(payload.get("symbol") or "").upper(),
            "setup": str(payload.get("setup") or ""),
            "timeframe": str(payload.get("timeframe") or "M5").upper(),
            "decision": str(payload.get("decision") or payload.get("decision_type") or "unknown"),
            "decision_type": str(payload.get("decision_type") or ""),
            "accepted": 1 if bool(payload.get("accepted", False)) else 0,
            "result": str(payload.get("result") or ("rejected" if not bool(payload.get("accepted", False)) else "pending")),
            "pnl_r": _safe_float(payload.get("pnl_r"), 0.0),
            "pnl_money": _safe_float(payload.get("pnl_money"), 0.0),
            "ai_probability": _safe_float(payload.get("ai_probability"), 0.5),
            "reason": str(payload.get("reason") or ""),
            "news_state": str(payload.get("news_state") or "unknown"),
            "session_state": str(payload.get("session_state") or "OUT"),
            "session_name": str(payload.get("session_name") or payload.get("session_state") or "OUT"),
            "regime": str(payload.get("regime") or "UNKNOWN"),
            "strategy_key": str(payload.get("strategy_key") or ""),
            "lane_name": str(payload.get("lane_name") or ""),
            "management_template": str(payload.get("management_template") or ""),
            "strategy_state": str(payload.get("strategy_state") or "NORMAL"),
            "regime_fit": _safe_float(payload.get("regime_fit"), 0.0),
            "session_fit": _safe_float(payload.get("session_fit"), 0.0),
            "volatility_fit": _safe_float(payload.get("volatility_fit"), 0.0),
            "pair_behavior_fit": _safe_float(payload.get("pair_behavior_fit"), 0.0),
            "execution_quality_fit": _safe_float(payload.get("execution_quality_fit"), 0.0),
            "entry_timing_score": _safe_float(payload.get("entry_timing_score"), 0.0),
            "structure_cleanliness_score": _safe_float(payload.get("structure_cleanliness_score"), 0.0),
            "strategy_recent_performance": _safe_float(payload.get("strategy_recent_performance"), 0.0),
            "market_data_source": str(payload.get("market_data_source") or ""),
            "market_data_consensus_state": str(payload.get("market_data_consensus_state") or ""),
            "multi_tf_alignment_score": _safe_float(payload.get("multi_tf_alignment_score"), 0.0),
            "fractal_persistence_score": _safe_float(payload.get("fractal_persistence_score"), 0.0),
            "compression_expansion_score": _safe_float(payload.get("compression_expansion_score"), 0.0),
            "dxy_support_score": _safe_float(payload.get("dxy_support_score"), 0.0),
            "swing_continuation_score": _safe_float(payload.get("swing_continuation_score"), 0.0),
            "aggressive_pair_mode": _safe_float(payload.get("aggressive_pair_mode"), 0.0),
            "trajectory_catchup_pressure": _safe_float(payload.get("trajectory_catchup_pressure"), 0.0),
            "institutional_confluence_score": _safe_float(payload.get("institutional_confluence_score"), 0.0),
            "candle_mastery_score": _safe_float(payload.get("candle_mastery_score"), 0.0),
            "live_shadow_gap_score": _safe_float(payload.get("live_shadow_gap_score"), 0.0),
            "execution_edge_score": _safe_float(
                payload.get("execution_edge_score"),
                _safe_float(payload.get("execution_quality_fit"), 0.0),
            ),
            "xau_engine": str(payload.get("xau_engine") or ""),
            "grid_source_role": str(payload.get("grid_source_role") or ""),
            "native_burst_window_id": str(payload.get("native_burst_window_id") or ""),
            "mc_win_rate": _safe_float(payload.get("mc_win_rate"), 0.0),
            "ga_generation_id": _safe_float(payload.get("ga_generation_id"), float(self._ga_generation_id)),
            "reentry_source_tag": str(payload.get("reentry_source_tag") or ""),
            "mode": str(payload.get("mode") or "LIVE"),
            "source": str(payload.get("source") or "bridge"),
        }
        self._append_ordered_row(
            path=self.setup_log_path,
            row=row,
            current_columns=SETUP_LOG_COLUMNS,
            legacy_columns=(LEGACY_SETUP_LOG_COLUMNS_V1,),
        )

    def _retrain_from_history(self) -> None:
        frame = self._read_history_frame() if self.data_path.exists() else pd.DataFrame(columns=list(TRADE_HISTORY_COLUMNS))
        if "timestamp_utc" in frame.columns:
            try:
                frame["timestamp_utc"] = pd.to_datetime(frame["timestamp_utc"], utc=True, errors="coerce")
                cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=max(1, int(self.rolling_window_days)))
                frame = frame.loc[frame["timestamp_utc"].isna() | (frame["timestamp_utc"] >= cutoff)]
            except Exception:
                pass
        frame = frame.tail(max(200, int(self.rolling_window_trades))).reset_index(drop=True)
        vectors = []
        labels = []
        for _, row in frame.iterrows():
            payload = {
                "symbol": row.get("symbol"),
                "side": row.get("side"),
                "entry": row.get("entry"),
                "sl": row.get("sl"),
                "tp": row.get("tp"),
                "lot": row.get("lot"),
                "ai_probability": row.get("ai_probability"),
                "spread_points": row.get("spread_points"),
                "news_state": row.get("news_state"),
                "session_state": row.get("session_state"),
                "session_name": row.get("session_name"),
                "regime": row.get("regime"),
                "regime_state": row.get("regime_state"),
                "setup": row.get("setup"),
                "strategy_key": row.get("strategy_key"),
                "lane_name": row.get("lane_name"),
                "management_template": row.get("management_template"),
                "strategy_state": row.get("strategy_state"),
                "regime_fit": row.get("regime_fit"),
                "session_fit": row.get("session_fit"),
                "volatility_fit": row.get("volatility_fit"),
                "pair_behavior_fit": row.get("pair_behavior_fit"),
                "execution_quality_fit": row.get("execution_quality_fit"),
                "entry_timing_score": row.get("entry_timing_score"),
                "structure_cleanliness_score": row.get("structure_cleanliness_score"),
                "strategy_recent_performance": row.get("strategy_recent_performance"),
                "market_data_source": row.get("market_data_source"),
                "market_data_consensus_state": row.get("market_data_consensus_state"),
                "multi_tf_alignment_score": row.get("multi_tf_alignment_score"),
                "fractal_persistence_score": row.get("fractal_persistence_score"),
                "compression_expansion_score": row.get("compression_expansion_score"),
                "dxy_support_score": row.get("dxy_support_score"),
                "swing_continuation_score": row.get("swing_continuation_score"),
                "aggressive_pair_mode": row.get("aggressive_pair_mode"),
                "trajectory_catchup_pressure": row.get("trajectory_catchup_pressure"),
                "institutional_confluence_score": row.get("institutional_confluence_score"),
                "candle_mastery_score": row.get("candle_mastery_score"),
                "live_shadow_gap_score": row.get("live_shadow_gap_score"),
                "execution_edge_score": row.get("execution_edge_score"),
                "mc_win_rate": row.get("mc_win_rate"),
                "ga_generation_id": row.get("ga_generation_id"),
            }
            vectors.append(self._feature_row(payload)[0])
            labels.append(1 if str(row.get("result", "")).lower() == "win" else 0)
        X = np.array(vectors, dtype=float) if vectors else np.empty((0, len(self._feature_cols)), dtype=float)
        y = np.array(labels, dtype=int) if labels else np.empty((0,), dtype=int)
        unique_classes, class_counts = np.unique(y, return_counts=True) if len(y) else (np.array([], dtype=int), np.array([], dtype=int))
        min_class_share = (float(class_counts.min()) / float(len(y))) if len(y) and len(class_counts) >= 2 else 0.0
        seed_required = (
            len(y) < 20
            or len(unique_classes) < 2
            or len(y) < int(self.promotion_min_samples)
            or min_class_share < 0.35
        )
        if self.market_history_backfill_enabled:
            backfill_summary = self._backfill_market_history_cache()
            self._last_market_history_backfill_status = str(backfill_summary.get("status") or "idle")
            self._last_market_history_backfill_files = int(backfill_summary.get("files_written", 0) or 0)
            backfilled_at = str(backfill_summary.get("generated_at") or "")
            self._last_market_history_backfill_at = backfilled_at or self._last_market_history_backfill_at
        if self.market_history_seed_enabled and seed_required:
            history_vectors, history_labels, seed_summary = self._market_history_seed_examples()
            self._last_market_history_seed_status = str(seed_summary.get("status") or "unavailable")
            self._last_market_history_seed_samples = int(seed_summary.get("samples", 0) or 0)
            seeded_at = str(seed_summary.get("generated_at") or "")
            self._last_market_history_seed_at = seeded_at or self._last_market_history_seed_at
            if len(history_labels):
                if len(y):
                    X = np.vstack([X, history_vectors])
                    y = np.concatenate([y, history_labels])
                else:
                    X = history_vectors
                    y = history_labels
                rng = np.random.default_rng(7)
                permutation = rng.permutation(len(y))
                X = X[permutation]
                y = y[permutation]
                self._log(
                    "info",
                    f"online_learning_market_history_seed_applied samples={len(history_labels)} status={self._last_market_history_seed_status}",
                )
        elif not self.market_history_seed_enabled:
            self._last_market_history_seed_status = "disabled"
            self._last_market_history_seed_samples = 0
        else:
            self._last_market_history_seed_status = "not_needed"
            self._last_market_history_seed_samples = 0
        if len(y) == 0:
            return
        if len(y) < 20 or len(np.unique(y)) < 2:
            return

        X = _sanitize_feature_matrix(X)
        holdout = max(20, int(len(y) * 0.2))
        if holdout >= len(y):
            holdout = max(1, len(y) // 3)
        split_idx = len(y) - holdout
        if split_idx <= 1:
            self._last_maintenance_status = "insufficient_history_split"
            self._last_maintenance_error = ""
            self._log("info", "online_learning_hold_insufficient_history_split")
            return
        X_train = X[:split_idx]
        y_train = y[:split_idx]
        X_test = X[split_idx:]
        y_test = y[split_idx:]
        if len(X_train) < 2 or len(X_test) < 2:
            self._last_maintenance_status = "insufficient_history_split"
            self._last_maintenance_error = ""
            self._log("info", "online_learning_hold_insufficient_history_split")
            return
        if len(X_test) < max(10, int(self.promotion_min_samples * 0.5)):
            return
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            self._last_maintenance_status = "insufficient_class_balance"
            self._last_maintenance_error = ""
            self._log("info", "online_learning_hold_insufficient_class_balance")
            return

        X_train = _sanitize_feature_matrix(X_train)
        X_test = _sanitize_feature_matrix(X_test)
        feature_scale = np.nan_to_num(np.std(X_train, axis=0), nan=0.0, posinf=0.0, neginf=0.0)
        stable_mask = feature_scale > 1e-9
        if int(np.count_nonzero(stable_mask)) >= min(8, X_train.shape[1]):
            X_train = X_train[:, stable_mask]
            X_test = X_test[:, stable_mask]

        candidate = make_pipeline(
            RobustScaler(quantile_range=(10.0, 90.0)),
            SGDClassifier(
                loss="log_loss",
                random_state=7,
                alpha=0.001,
                learning_rate="adaptive",
                eta0=0.01,
                max_iter=2000,
                tol=1e-3,
            ),
        )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                warnings.simplefilter("ignore", ConvergenceWarning)
                candidate.fit(X_train, y_train)
        except ValueError as exc:
            if "greater than one class" in str(exc):
                self._last_maintenance_status = "insufficient_class_balance"
                self._last_maintenance_error = ""
                self._log("info", "online_learning_hold_insufficient_class_balance")
                return
            raise
        candidate_acc = float((candidate.predict(X_test) == y_test).mean()) if len(X_test) else 0.0

        baseline_acc = 0.0
        baseline_available = self._model is not None and self._initialized
        if baseline_available:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    baseline_acc = float((self._model.predict(X_test) == y_test).mean()) if len(X_test) else 0.0
            except Exception:
                baseline_acc = 0.0
                baseline_available = False

        promote = (not baseline_available) or (
            len(X_test) >= int(self.promotion_min_samples)
            and candidate_acc >= (baseline_acc + self.promotion_min_delta)
        )
        self._ga_generation_id += 1
        if promote:
            self._model = candidate
            self._initialized = True
            self._trained_samples = len(y)
            self._last_maintenance_status = "ok"
            self._last_maintenance_error = ""
            self._last_promotion = {
                "baseline_acc": baseline_acc,
                "candidate_acc": candidate_acc,
                "holdout": float(len(X_test)),
            }
            self._save_model()
            self._log("info", f"online_learning_promoted candidate={candidate_acc:.3f} baseline={baseline_acc:.3f} holdout={len(X_test)}")
        else:
            self._trained_samples = len(y)
            self._last_maintenance_status = "ok"
            self._last_maintenance_error = ""
            self._last_promotion = {
                "baseline_acc": baseline_acc,
                "candidate_acc": candidate_acc,
                "holdout": float(len(X_test)),
            }
            self._save_model()
            self._log("info", f"online_learning_hold baseline={baseline_acc:.3f} candidate={candidate_acc:.3f} holdout={len(X_test)}")

    def _market_history_seed_examples(self) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        cache_dir = self.history_cache_dir
        generated_at = utc_now().isoformat()
        if not self.market_history_seed_enabled:
            return (
                np.empty((0, len(self._feature_cols)), dtype=float),
                np.empty((0,), dtype=int),
                {"status": "disabled", "samples": 0, "generated_at": generated_at},
            )
        if cache_dir is None or not Path(cache_dir).exists():
            return (
                np.empty((0, len(self._feature_cols)), dtype=float),
                np.empty((0,), dtype=int),
                {"status": "cache_dir_missing", "samples": 0, "generated_at": generated_at},
            )
        timeframe_rank = {
            "D1": 0,
            "H4": 1,
            "H1": 2,
            "M15": 3,
            "M5": 4,
            "M1": 5,
            "W1": 6,
        }
        preferred_files: dict[str, list[tuple[int, str, Path]]] = {}
        for path in sorted(Path(cache_dir).glob("*.parquet")):
            stem = path.stem.upper()
            if "_" not in stem:
                continue
            symbol_raw, timeframe = stem.rsplit("_", 1)
            timeframe_key = str(timeframe).upper()
            rank = timeframe_rank.get(timeframe_key)
            if rank is None:
                continue
            symbol_key = _normalize_symbol_key(symbol_raw)
            preferred_files.setdefault(symbol_key, []).append((rank, timeframe_key, path))
        if not preferred_files:
            return (
                np.empty((0, len(self._feature_cols)), dtype=float),
                np.empty((0,), dtype=int),
                {"status": "no_history_files", "samples": 0, "generated_at": generated_at},
            )

        selected_files: list[tuple[str, str, Path]] = []
        for symbol_key, items in sorted(preferred_files.items()):
            top_items = sorted(items, key=lambda item: item[0])[:3]
            for _, timeframe_key, path in top_items:
                selected_files.append((symbol_key, timeframe_key, path))
        per_file_limit = max(30, int(max(1, self.market_history_seed_max_samples) / max(1, len(selected_files))))
        vectors: list[np.ndarray] = []
        labels: list[int] = []
        seeded_symbols: list[str] = []
        for symbol_key, timeframe_key, path in selected_files:
            try:
                frame = pd.read_parquet(path)
            except Exception:
                continue
            required_columns = {"time", "open", "high", "low", "close"}
            if not required_columns.issubset(set(frame.columns)):
                continue
            frame = frame.loc[:, [column for column in frame.columns if column in {"time", "open", "high", "low", "close", "spread"}]].copy()
            frame["time"] = pd.to_datetime(frame["time"], utc=True, errors="coerce")
            frame = frame.dropna(subset=["time", "open", "high", "low", "close"]).sort_values("time").reset_index(drop=True)
            if len(frame) < max(self.market_history_seed_min_rows_per_file, self.market_history_seed_forward_bars + 80):
                continue

            close = pd.to_numeric(frame["close"], errors="coerce")
            open_ = pd.to_numeric(frame["open"], errors="coerce")
            high = pd.to_numeric(frame["high"], errors="coerce")
            low = pd.to_numeric(frame["low"], errors="coerce")
            spread = pd.to_numeric(frame["spread"], errors="coerce") if "spread" in frame.columns else pd.Series([_default_spread_points(symbol_key)] * len(frame))
            prev_close = close.shift(1).fillna(close)
            true_range = pd.concat(
                [
                    (high - low).abs(),
                    (high - prev_close).abs(),
                    (low - prev_close).abs(),
                ],
                axis=1,
            ).max(axis=1)
            atr = true_range.rolling(14, min_periods=5).mean().bfill()
            atr_avg = atr.rolling(40, min_periods=10).mean().fillna(atr)
            ema20 = close.ewm(span=20, adjust=False).mean()
            ema50 = close.ewm(span=50, adjust=False).mean()
            body_efficiency = ((close - open_).abs() / (high - low).clip(lower=1e-6)).clip(0.0, 1.0)
            range_low = low.rolling(20, min_periods=10).min().fillna(low)
            range_high = high.rolling(20, min_periods=10).max().fillna(high)
            range_position = ((close - range_low) / (range_high - range_low).clip(lower=1e-6)).clip(0.0, 1.0)
            trend_strength = ((ema20 - ema50) / atr.clip(lower=1e-6)).clip(-3.0, 3.0)
            atr_regime = (atr / atr_avg.clip(lower=1e-6)).clip(0.0, 3.0)

            last_valid_index = len(frame) - int(self.market_history_seed_forward_bars) - 1
            if last_valid_index <= 60:
                continue
            candidate_indices = list(range(60, last_valid_index))
            stride = max(1, len(candidate_indices) // max(1, per_file_limit // 2))
            symbol_samples = 0
            for index in candidate_indices[::stride]:
                atr_value = _safe_float(atr.iloc[index], 0.0)
                close_value = _safe_float(close.iloc[index], 0.0)
                if atr_value <= 0.0 or close_value <= 0.0:
                    continue
                future_high = float(high.iloc[index + 1 : index + 1 + self.market_history_seed_forward_bars].max())
                future_low = float(low.iloc[index + 1 : index + 1 + self.market_history_seed_forward_bars].min())
                body_value = _safe_float(body_efficiency.iloc[index], 0.0)
                trend_value = float(trend_strength.iloc[index])
                alignment_score = _clamp01(0.50 + (abs(trend_value) * 0.12) + (body_value * 0.18), 0.50)
                fractal_score = _clamp01(
                    body_value * 0.45
                    + max(0.0, 0.35 - abs(float(range_position.iloc[index]) - 0.5))
                    + min(0.30, abs(float(trend_value)) * 0.08),
                    0.40,
                )
                compression_score = _clamp01(0.45 + ((float(atr_regime.iloc[index]) - 1.0) * 0.25), 0.45)
                for side in ("BUY", "SELL"):
                    stop_dist = max(atr_value * 0.90, abs(close_value) * 0.00035)
                    tp_dist = max(stop_dist * 1.45, atr_value * 1.10)
                    if side == "BUY":
                        favorable = max(0.0, future_high - close_value)
                        adverse = max(0.0, close_value - future_low)
                        directional_support = max(0.0, trend_value)
                    else:
                        favorable = max(0.0, close_value - future_low)
                        adverse = max(0.0, future_high - close_value)
                        directional_support = max(0.0, -trend_value)
                    if favorable >= (tp_dist * 0.85) and adverse <= (stop_dist * 1.05):
                        label = 1
                    elif adverse >= (stop_dist * 0.90):
                        label = 0
                    else:
                        continue
                    ai_probability = _clamp01(
                        0.52
                        + (alignment_score * 0.22)
                        + min(0.12, directional_support * 0.06),
                        0.52,
                    )
                    regime_key = "TRENDING" if abs(trend_value) >= 0.12 else "RANGING"
                    payload = {
                        "symbol": symbol_key,
                        "side": side,
                        "entry": close_value,
                        "sl": close_value - stop_dist if side == "BUY" else close_value + stop_dist,
                        "tp": close_value + tp_dist if side == "BUY" else close_value - tp_dist,
                        "lot": 0.01,
                        "ai_probability": ai_probability,
                        "spread_points": max(_default_spread_points(symbol_key), _safe_float(spread.iloc[index], _default_spread_points(symbol_key))),
                        "news_state": "clear",
                        "session_state": "IN",
                        "session_name": _session_name_from_timestamp(frame.iloc[index]["time"]),
                        "regime": regime_key,
                        "regime_state": regime_key,
                        "setup": f"{symbol_key}_{timeframe_key}_HISTORICAL_SEED",
                        "strategy_key": f"{symbol_key}_{timeframe_key}_HISTORICAL_SEED",
                        "lane_name": "HISTORICAL_SEED",
                        "management_template": "SCALP_DAY_SWING",
                        "strategy_state": "ATTACK" if symbol_key in _SUPER_AGGRESSIVE_LEARNING_SYMBOLS else "NORMAL",
                        "regime_fit": _clamp01(0.55 + min(0.25, abs(trend_value) * 0.08), 0.55),
                        "session_fit": _clamp01(0.60 + (0.10 if _session_name_from_timestamp(frame.iloc[index]["time"]) in {"LONDON", "OVERLAP", "NEW_YORK"} else 0.04), 0.60),
                        "volatility_fit": _clamp01(0.50 + min(0.30, abs(float(atr_regime.iloc[index]) - 1.0) * 0.20), 0.50),
                        "pair_behavior_fit": _clamp01(0.55 + (0.12 if symbol_key in _SUPER_AGGRESSIVE_LEARNING_SYMBOLS else 0.06), 0.55),
                        "execution_quality_fit": _clamp01(0.55 + (0.15 if _safe_float(spread.iloc[index], _default_spread_points(symbol_key)) <= (_default_spread_points(symbol_key) * 1.25) else 0.04), 0.55),
                        "entry_timing_score": alignment_score,
                        "structure_cleanliness_score": fractal_score,
                        "strategy_recent_performance": _clamp01(0.55 + (0.12 if label == 1 else 0.04), 0.55),
                        "market_data_source": "historical_candle_cache",
                        "market_data_consensus_state": "ALIGNED",
                        "multi_tf_alignment_score": alignment_score,
                        "fractal_persistence_score": fractal_score,
                        "compression_expansion_score": compression_score,
                        "dxy_support_score": _clamp01(0.50 + min(0.14, directional_support * 0.06), 0.50),
                        "swing_continuation_score": _clamp01(0.45 + min(0.25, directional_support * 0.10), 0.45),
                        "aggressive_pair_mode": 1.0 if symbol_key in _SUPER_AGGRESSIVE_LEARNING_SYMBOLS else 0.0,
                        "trajectory_catchup_pressure": 0.0,
                        "mc_win_rate": _clamp01(0.58 + (alignment_score * 0.22) + (0.08 if label == 1 else 0.0), 0.58),
                        "ga_generation_id": float(self._ga_generation_id),
                    }
                    vectors.append(self._feature_row(payload)[0])
                    labels.append(int(label))
                    symbol_samples += 1
                    if symbol_samples >= per_file_limit:
                        break
                if symbol_samples >= per_file_limit:
                    break
            if symbol_samples > 0:
                seeded_symbols.append(f"{symbol_key}:{timeframe_key}")
        if not labels:
            return (
                np.empty((0, len(self._feature_cols)), dtype=float),
                np.empty((0,), dtype=int),
                {"status": "insufficient_market_history", "samples": 0, "generated_at": generated_at},
            )
        return (
            np.array(vectors, dtype=float),
            np.array(labels, dtype=int),
            {
                "status": "ok",
                "samples": int(len(labels)),
                "generated_at": generated_at,
                "seeded_symbols": list(seeded_symbols),
            },
        )

    def _backfill_market_history_cache(self) -> dict[str, Any]:
        generated_at = utc_now().isoformat()
        if not self.market_history_backfill_enabled:
            return {"status": "disabled", "files_written": 0, "generated_at": generated_at}
        if self.history_cache_dir is None:
            return {"status": "cache_dir_missing", "files_written": 0, "generated_at": generated_at}
        cache_dir = Path(self.history_cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        provider = YahooMarketDataFallback(timeout_seconds=10.0)
        backfill_years = max(1, int(self.market_history_backfill_years))
        history_targets = {
            "D1": "10y" if backfill_years >= 10 else f"{backfill_years}y",
            "H4": "5y" if backfill_years >= 10 else f"{min(backfill_years, 5)}y",
            "H1": "3y" if backfill_years >= 10 else f"{min(backfill_years, 3)}y",
            "M15": "1y" if backfill_years >= 2 else "180d",
        }
        symbols: set[str] = set()
        for symbol in self.market_history_universe_symbols:
            if symbol:
                symbols.add(_normalize_symbol_key(symbol))
        for path in cache_dir.glob("*.parquet"):
            stem = path.stem.upper()
            if "_" not in stem:
                continue
            symbol_raw, _ = stem.rsplit("_", 1)
            symbol_key = _normalize_symbol_key(symbol_raw)
            if symbol_key:
                symbols.add(symbol_key)
        if not symbols:
            return {"status": "no_cache_seed", "files_written": 0, "generated_at": generated_at}

        files_written = 0
        files: list[str] = []
        errors: list[str] = []
        min_rows_by_tf = {"D1": 750, "H4": 600, "H1": 900, "M15": 1200}
        for symbol_key in sorted(symbols):
            for timeframe, range_value in history_targets.items():
                target_path = cache_dir / f"{symbol_key}_{timeframe}.parquet"
                if target_path.exists():
                    try:
                        existing = pd.read_parquet(target_path, columns=["time"])
                        if len(existing.index) >= int(min_rows_by_tf.get(timeframe, 250)):
                            continue
                    except Exception:
                        pass
                try:
                    frame = provider.fetch_rates_with_range(symbol_key, timeframe, range_value=range_value)
                except Exception as exc:
                    errors.append(f"{symbol_key}_{timeframe}:{exc}")
                    continue
                if frame.empty:
                    continue
                try:
                    frame.to_parquet(target_path, index=False)
                    files_written += 1
                    files.append(target_path.name)
                except Exception as exc:
                    errors.append(f"{target_path.name}:write_failed:{exc}")
        return {
            "status": "ok" if files_written > 0 or not errors else "failed",
            "files_written": int(files_written),
            "generated_at": generated_at,
            "files": files,
            "errors": errors[:10],
        }

    def _read_history_frame(self) -> pd.DataFrame:
        rows, repaired = _normalize_csv_records(
            path=self.data_path,
            current_columns=TRADE_HISTORY_COLUMNS,
            legacy_columns=(LEGACY_TRADE_HISTORY_COLUMNS_V1,),
        )
        if repaired:
            self._rewrite_csv(
                path=self.data_path,
                rows=rows,
                current_columns=TRADE_HISTORY_COLUMNS,
            )
        if not rows:
            return pd.DataFrame(columns=list(TRADE_HISTORY_COLUMNS))
        return pd.DataFrame(rows, columns=list(TRADE_HISTORY_COLUMNS))

    def _ensure_storage_schema(self) -> None:
        self._ensure_csv_schema(
            path=self.data_path,
            current_columns=TRADE_HISTORY_COLUMNS,
            legacy_columns=(LEGACY_TRADE_HISTORY_COLUMNS_V1,),
        )
        if self.setup_log_path is not None:
            self._ensure_csv_schema(
                path=self.setup_log_path,
                current_columns=SETUP_LOG_COLUMNS,
                legacy_columns=(LEGACY_SETUP_LOG_COLUMNS_V1,),
            )

    def _ensure_csv_schema(
        self,
        *,
        path: Path,
        current_columns: tuple[str, ...],
        legacy_columns: tuple[tuple[str, ...], ...],
    ) -> None:
        rows, repaired = _normalize_csv_records(
            path=path,
            current_columns=current_columns,
            legacy_columns=legacy_columns,
        )
        if repaired:
            self._rewrite_csv(path=path, rows=rows, current_columns=current_columns)

    def _append_ordered_row(
        self,
        *,
        path: Path | None,
        row: dict[str, Any],
        current_columns: tuple[str, ...],
        legacy_columns: tuple[tuple[str, ...], ...],
    ) -> None:
        if path is None:
            return
        self._ensure_csv_schema(path=path, current_columns=current_columns, legacy_columns=legacy_columns)
        ensure_parent(path)
        ordered_row = {column: row.get(column, _default_csv_value(column)) for column in current_columns}
        frame = pd.DataFrame([ordered_row], columns=list(current_columns))
        if path.exists():
            frame.to_csv(path, mode="a", header=False, index=False)
        else:
            frame.to_csv(path, index=False)

    def _rewrite_csv(
        self,
        *,
        path: Path,
        rows: list[dict[str, Any]],
        current_columns: tuple[str, ...],
    ) -> None:
        ensure_parent(path)
        frame = pd.DataFrame(rows, columns=list(current_columns))
        frame.to_csv(path, index=False)
        self._last_history_repair_at = utc_now().isoformat()
        self._last_maintenance_status = "history_repaired"
        self._last_maintenance_error = ""
        self._log("info", f"online_learning_history_repaired:{path.name}")

    def status_snapshot(self) -> dict[str, Any]:
        return {
            "trained_samples": int(self._trained_samples),
            "pending_samples": int(self._pending_samples),
            "initialized": bool(self._initialized),
            "last_maintenance_retrain_at": str(self._last_maintenance_retrain_at or ""),
            "last_maintenance_day": str(self._last_maintenance_day or ""),
            "last_maintenance_status": str(self._last_maintenance_status or ""),
            "last_maintenance_error": str(self._last_maintenance_error or ""),
            "last_history_repair_at": str(self._last_history_repair_at or ""),
            "last_market_history_seed_at": str(self._last_market_history_seed_at or ""),
            "last_market_history_seed_status": str(self._last_market_history_seed_status or ""),
            "last_market_history_seed_samples": int(self._last_market_history_seed_samples),
            "last_market_history_backfill_at": str(self._last_market_history_backfill_at or ""),
            "last_market_history_backfill_status": str(self._last_market_history_backfill_status or ""),
            "last_market_history_backfill_files": int(self._last_market_history_backfill_files),
            "ga_generation_id": int(self._ga_generation_id),
            "last_promotion": dict(self._last_promotion),
        }

    def _load_model(self) -> None:
        if not self.model_path.exists():
            return
        try:
            payload = joblib.load(self.model_path)
            if isinstance(payload, dict):
                saved_feature_columns = [str(item) for item in (payload.get("feature_columns") or [])]
                if saved_feature_columns and saved_feature_columns != list(self._feature_cols):
                    self._model = None
                    self._initialized = False
                    self._trained_samples = 0
                    self._pending_samples = 0
                    self._log(
                        "warning",
                        "online_model_feature_mismatch_reset",
                    )
                    return
                self._model = payload.get("model")
                self._initialized = bool(payload.get("initialized", False))
                self._trained_samples = int(payload.get("trained_samples", 0))
                self._pending_samples = int(payload.get("pending_samples", 0))
                if isinstance(payload.get("last_promotion"), dict):
                    self._last_promotion = {
                        "baseline_acc": _safe_float(payload["last_promotion"].get("baseline_acc"), 0.0),
                        "candidate_acc": _safe_float(payload["last_promotion"].get("candidate_acc"), 0.0),
                        "holdout": _safe_float(payload["last_promotion"].get("holdout"), 0.0),
                    }
                self._last_maintenance_retrain_at = str(payload.get("last_maintenance_retrain_at") or "") or None
                self._last_maintenance_day = str(payload.get("last_maintenance_day") or "") or None
                self._last_maintenance_status = str(payload.get("last_maintenance_status") or self._last_maintenance_status)
                self._last_maintenance_error = str(payload.get("last_maintenance_error") or self._last_maintenance_error)
                self._last_history_repair_at = str(payload.get("last_history_repair_at") or "") or None
                self._last_market_history_seed_at = str(payload.get("last_market_history_seed_at") or "") or None
                self._last_market_history_seed_status = str(payload.get("last_market_history_seed_status") or self._last_market_history_seed_status)
                self._last_market_history_seed_samples = int(payload.get("last_market_history_seed_samples", 0) or 0)
                self._last_market_history_backfill_at = str(payload.get("last_market_history_backfill_at") or "") or None
                self._last_market_history_backfill_status = str(payload.get("last_market_history_backfill_status") or self._last_market_history_backfill_status)
                self._last_market_history_backfill_files = int(payload.get("last_market_history_backfill_files", 0) or 0)
                self._ga_generation_id = max(0, int(payload.get("ga_generation_id", 0) or 0))
            elif isinstance(payload, SGDClassifier):
                self._log("warning", "online_model_legacy_payload_ignored")
                self._model = None
                self._initialized = False
        except Exception as exc:
            self._log("warning", f"online_model_load_failed:{exc}")

    def _save_model(self) -> None:
        ensure_parent(self.model_path)
        payload = {
            "model": self._model,
            "initialized": self._initialized,
            "trained_samples": self._trained_samples,
            "pending_samples": self._pending_samples,
            "saved_at": utc_now().isoformat(),
            "feature_columns": self._feature_cols,
            "last_promotion": self._last_promotion,
            "last_maintenance_retrain_at": self._last_maintenance_retrain_at,
            "last_maintenance_day": self._last_maintenance_day,
            "last_maintenance_status": self._last_maintenance_status,
            "last_maintenance_error": self._last_maintenance_error,
            "last_history_repair_at": self._last_history_repair_at,
            "last_market_history_seed_at": self._last_market_history_seed_at,
            "last_market_history_seed_status": self._last_market_history_seed_status,
            "last_market_history_seed_samples": int(self._last_market_history_seed_samples),
            "last_market_history_backfill_at": self._last_market_history_backfill_at,
            "last_market_history_backfill_status": self._last_market_history_backfill_status,
            "last_market_history_backfill_files": int(self._last_market_history_backfill_files),
            "ga_generation_id": int(self._ga_generation_id),
        }
        joblib.dump(payload, self.model_path)

    def _log(self, level: str, message: str) -> None:
        if self.logger is not None and hasattr(self.logger, level):
            getattr(self.logger, level)(message)
            return
        print(json.dumps({"level": level, "message": message}))
