from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import Any
import json

from src.utils import clamp, ensure_parent

UTC = timezone.utc


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _trade_score(payload: dict[str, Any]) -> float:
    pnl_r = _safe_float(payload.get("pnl_r"))
    if abs(pnl_r) > 1e-9:
        return pnl_r
    pnl_money = _safe_float(payload.get("pnl_money", payload.get("profit")))
    if abs(pnl_money) > 1e-9:
        return pnl_money
    return pnl_r


def _bucket_spread(points: float) -> str:
    value = max(0.0, float(points))
    if value <= 15:
        return "tight"
    if value <= 35:
        return "normal"
    if value <= 80:
        return "wide"
    return "extreme"


def _normalize_tags(value: Any) -> list[str]:
    if isinstance(value, (list, tuple, set)):
        return sorted({str(item).strip().upper() for item in value if str(item).strip()})
    if isinstance(value, str) and value.strip():
        return sorted({part.strip().upper() for part in value.split(",") if part.strip()})
    return []


def _quality_band(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {"elite", "strong", "acceptable", "weak", "a+", "a", "a-", "b+", "b", "c", "reject"}:
        return text
    try:
        score = float(value)
    except (TypeError, ValueError):
        return "unknown"
    if score >= 0.85:
        return "elite"
    if score >= 0.70:
        return "strong"
    if score >= 0.60:
        return "acceptable"
    return "weak"


@dataclass
class StrategyOptimizer:
    trade_history_path: Path
    metrics_path: Path
    enabled: bool = True
    min_trades_per_strategy: int = 50
    lookback_days: int = 90
    low_win_rate_threshold: float = 0.45
    high_win_rate_threshold: float = 0.60
    adjustment_step_pct: float = 0.10
    min_adjustment_multiplier: float = 0.75
    max_adjustment_multiplier: float = 1.25
    cooldown_minutes: int = 30
    logger: Any | None = None
    _history: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _metrics: dict[str, Any] = field(default_factory=dict, init=False, repr=False)
    _lock: RLock = field(default_factory=RLock, init=False, repr=False)

    def __post_init__(self) -> None:
        self._history = self._load_json(self.trade_history_path, {"updated_at": _now_iso(), "trades": []})
        self._metrics = self._load_json(
            self.metrics_path,
            {
                "updated_at": _now_iso(),
                "strategies": {},
                "registered_strategies": [],
                "last_optimizer_run": None,
            },
        )
        if "strategies" not in self._metrics or not isinstance(self._metrics.get("strategies"), dict):
            self._metrics["strategies"] = {}
        if "registered_strategies" not in self._metrics or not isinstance(self._metrics.get("registered_strategies"), list):
            self._metrics["registered_strategies"] = []
        if "trades" not in self._history or not isinstance(self._history.get("trades"), list):
            self._history["trades"] = []
        self._prune_history()
        self._save_all()

    def record_trade(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            return {"enabled": False}
        strategy = str(payload.get("strategy") or payload.get("strategy_key") or "UNKNOWN").upper()
        timestamp = str(payload.get("timestamp_utc") or payload.get("closed_at") or _now_iso())
        trade_score = _trade_score(payload)
        with self._lock:
            trade_row = {
                "timestamp_utc": timestamp,
                "symbol": str(payload.get("symbol") or "").upper(),
                "strategy": strategy,
                "lane": str(payload.get("lane") or payload.get("lane_name") or "UNKNOWN").upper(),
                "setup": str(payload.get("setup") or ""),
                "entry_type": str(payload.get("entry_type") or payload.get("setup_type") or "unknown"),
                "side": str(payload.get("side") or "").upper(),
                "sl": _safe_float(payload.get("sl")),
                "tp": _safe_float(payload.get("tp")),
                "outcome": "win" if trade_score > 0 else "loss",
                "profit": _safe_float(payload.get("pnl_money")),
                "pnl_r": trade_score,
                "duration_minutes": _safe_float(payload.get("duration_minutes")),
                "spread_points": _safe_float(payload.get("spread_points")),
                "session": str(payload.get("session_state") or payload.get("session") or "UNKNOWN").upper(),
                "regime": str(payload.get("regime") or "UNKNOWN").upper(),
                "spread_bucket": _bucket_spread(_safe_float(payload.get("spread_points"))),
                "quality_band": _quality_band(payload.get("quality_band", payload.get("trade_quality_score"))),
                "entry_reason": str(payload.get("entry_reason") or payload.get("reason") or ""),
                "exit_reason": str(payload.get("exit_reason") or ""),
                "narrative": str(payload.get("narrative") or ""),
                "adjustment_tags": _normalize_tags(payload.get("adjustment_tags")),
            }
            trades = self._history.setdefault("trades", [])
            trades.append(trade_row)
            self._prune_history()
            strategy_metrics = self._recompute_strategy_metrics(strategy)
            self._maybe_optimize(strategy=strategy, metrics=strategy_metrics)
            self._save_all()
            return strategy_metrics

    def adjustments_for(self, strategy: str) -> dict[str, float]:
        strategy_key = str(strategy or "UNKNOWN").upper()
        with self._lock:
            state = self._metrics.get("strategies", {}).get(strategy_key, {})
            adjustments = state.get("adjustments", {})
            if not isinstance(adjustments, dict):
                return self._default_adjustments()
            output = self._default_adjustments()
            for key in output:
                if key in adjustments:
                    output[key] = clamp(
                        _safe_float(adjustments.get(key), output[key]),
                        self.min_adjustment_multiplier,
                        self.max_adjustment_multiplier,
                    )
            return output

    def register_strategies(self, strategies: list[str] | tuple[str, ...] | set[str]) -> None:
        with self._lock:
            current = {
                str(item).upper()
                for item in self._metrics.get("registered_strategies", [])
                if str(item).strip()
            }
            current.update(str(item).upper() for item in strategies if str(item).strip())
            self._metrics["registered_strategies"] = sorted(current)
            self._metrics["updated_at"] = _now_iso()
            self._save_all()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            registered = [
                str(item).upper()
                for item in self._metrics.get("registered_strategies", [])
                if str(item).strip()
            ]
            strategy_keys = {
                str(row.get("strategy") or "").upper()
                for row in self._history.get("trades", [])
                if str(row.get("strategy") or "").strip()
            }
            for strategy in sorted(strategy_keys):
                self._recompute_strategy_metrics(strategy)
            strategies = dict(self._metrics.get("strategies", {}))
            return {
                "enabled": bool(self.enabled),
                "updated_at": str(self._metrics.get("updated_at") or _now_iso()),
                "strategy_count": len(set(registered) | set(strategies.keys())),
                "strategies_registered": registered,
                "strategies_with_history": sorted(strategies.keys()),
                "last_optimizer_run": self._metrics.get("last_optimizer_run"),
                "strategies": strategies,
                "performance_buckets": self.performance_buckets(),
            }

    def performance_buckets(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            buckets: dict[str, list[dict[str, Any]]] = {}
            for row in self._history.get("trades", []):
                key = self._bucket_key_from_row(row)
                buckets.setdefault(key, []).append(row)
            return {
                key: self._summarize_bucket(rows)
                for key, rows in buckets.items()
                if rows
            }

    def bucket_proof_state(
        self,
        *,
        symbol: str,
        strategy: str,
        lane: str = "",
        regime: str,
        session: str,
        direction: str,
        quality_band: str,
        trusted_min_samples: int = 12,
        elite_min_samples: int = 20,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._bucket_key(
                symbol=symbol,
                strategy=strategy,
                lane=lane,
                regime=regime,
                session=session,
                direction=direction,
                quality_band=quality_band,
            )
            rows = [
                row
                for row in self._history.get("trades", [])
                if self._bucket_key_from_row(row) == key
            ]
            summary = self._summarize_bucket(rows)
            trades = int(summary.get("trades", 0) or 0)
            win_rate = _safe_float(summary.get("win_rate"), 0.0)
            expectancy = _safe_float(summary.get("expectancy_r"), 0.0)
            profit_factor = _safe_float(summary.get("profit_factor"), 0.0)
            state = "neutral"
            if trades >= max(1, int(elite_min_samples)) and profit_factor >= 1.25 and expectancy >= 0.20:
                state = "elite-proof"
            elif trades >= max(1, int(trusted_min_samples)) and profit_factor >= 1.10 and expectancy > 0.0:
                state = "trusted"
            return {
                "state": state,
                "summary": summary,
                "trades": trades,
                "win_rate": win_rate,
                "expectancy_r": expectancy,
                "profit_factor": profit_factor,
            }

    def bucket_adjustment(
        self,
        *,
        symbol: str,
        strategy: str,
        lane: str = "",
        regime: str,
        session: str,
        direction: str,
        quality_band: str,
        min_samples: int = 5,
    ) -> dict[str, Any]:
        with self._lock:
            key = self._bucket_key(
                symbol=symbol,
                strategy=strategy,
                lane=lane,
                regime=regime,
                session=session,
                direction=direction,
                quality_band=quality_band,
            )
            rows = [
                row
                for row in self._history.get("trades", [])
                if self._bucket_key_from_row(row) == key
            ]
            summary = self._summarize_bucket(rows)
            if int(summary.get("trades", 0)) < max(1, int(min_samples)):
                return {"multiplier": 1.0, "state": "neutral_insufficient_sample", "summary": summary}
            win_rate = _safe_float(summary.get("win_rate"), 0.5)
            expectancy = _safe_float(summary.get("expectancy_r"), 0.0)
            multiplier = 1.0
            state = "neutral"
            if win_rate >= 0.62 and expectancy > 0.08:
                multiplier = 1.08
                state = "promoted"
            elif win_rate <= 0.42 or expectancy < -0.08:
                multiplier = 0.90
                state = "suppressed"
            return {"multiplier": clamp(multiplier, 0.85, 1.12), "state": state, "summary": summary}

    def _maybe_optimize(self, *, strategy: str, metrics: dict[str, Any]) -> None:
        trade_count = _safe_int(metrics.get("trade_count"), 0)
        if trade_count <= 0:
            return
        if trade_count < max(1, int(self.min_trades_per_strategy)):
            return
        if trade_count % max(1, int(self.min_trades_per_strategy)) != 0:
            return
        strategies = self._metrics.setdefault("strategies", {})
        strategy_state = strategies.setdefault(strategy, {})
        last_optimized = str(strategy_state.get("last_optimized_at") or "")
        if last_optimized:
            try:
                last_dt = datetime.fromisoformat(last_optimized.replace("Z", "+00:00"))
                if last_dt.tzinfo is None:
                    last_dt = last_dt.replace(tzinfo=UTC)
                if (datetime.now(tz=UTC) - last_dt).total_seconds() < max(60, int(self.cooldown_minutes) * 60):
                    return
            except Exception:
                pass
        adjustments = self.adjustments_for(strategy)
        win_rate = _safe_float(metrics.get("win_rate"), 0.5)
        expectancy_r = _safe_float(metrics.get("expectancy_r"), 0.0)
        profit_factor = _safe_float(metrics.get("profit_factor"), 0.0)
        step = max(0.01, min(0.20, float(self.adjustment_step_pct)))
        is_xau_grid = "XAU" in strategy or "GRID" in strategy

        # Keep updates bounded to soft knobs only.
        if win_rate < float(self.low_win_rate_threshold):
            adjustments["probability_floor_mult"] *= (1.0 + (step * 0.40))
            adjustments["confluence_floor_mult"] *= (1.0 + (step * 0.40))
            adjustments["candidate_sensitivity_mult"] *= (1.0 - (step * 0.35))
            if is_xau_grid:
                adjustments["xau_grid_spacing_mult"] *= (1.0 + (step * 0.5))
                adjustments["xau_grid_probe_aggression"] *= (1.0 - (step * 0.5))
        elif win_rate > float(self.high_win_rate_threshold) and expectancy_r > 0.10 and profit_factor > 1.0:
            adjustments["probability_floor_mult"] *= (1.0 - (step * 0.5))
            adjustments["confluence_floor_mult"] *= (1.0 - (step * 0.5))
            adjustments["candidate_sensitivity_mult"] *= (1.0 + (step * 0.25))
            if is_xau_grid:
                adjustments["xau_grid_spacing_mult"] *= (1.0 - (step * 0.25))
                adjustments["xau_grid_probe_aggression"] *= (1.0 + (step * 0.25))

        for key in list(adjustments.keys()):
            adjustments[key] = clamp(
                adjustments[key],
                self.min_adjustment_multiplier,
                self.max_adjustment_multiplier,
            )

        strategy_state["adjustments"] = adjustments
        strategy_state["last_optimized_at"] = _now_iso()
        strategy_state["last_optimization_reason"] = (
            "underperforming_tighten" if win_rate < float(self.low_win_rate_threshold) else "performing_scale"
        )
        self._metrics["last_optimizer_run"] = strategy_state["last_optimized_at"]

    def _recompute_strategy_metrics(self, strategy: str) -> dict[str, Any]:
        trades = [
            row
            for row in self._history.get("trades", [])
            if str(row.get("strategy") or "").upper() == strategy
        ]
        trades.sort(key=lambda item: str(item.get("timestamp_utc") or ""))
        pnl = [_trade_score(row) for row in trades]
        wins = [value for value in pnl if value > 0]
        losses = [abs(value) for value in pnl if value < 0]
        trade_count = len(pnl)
        win_rate = (sum(1 for value in pnl if value >= 0) / trade_count) if trade_count else 0.0
        expectancy_r = (sum(pnl) / trade_count) if trade_count else 0.0
        profit_factor = (sum(wins) / sum(losses)) if losses else (float("inf") if wins else 0.0)
        avg_rr = (sum(abs(value) for value in pnl) / trade_count) if trade_count else 0.0
        avg_duration = (
            sum(_safe_float(row.get("duration_minutes")) for row in trades) / trade_count
        ) if trade_count else 0.0
        max_drawdown = 0.0
        equity_curve = 0.0
        peak = 0.0
        for value in pnl:
            equity_curve += value
            peak = max(peak, equity_curve)
            max_drawdown = max(max_drawdown, peak - equity_curve)

        def _best_bucket(key: str, *, reverse: bool) -> str:
            buckets: dict[str, list[float]] = {}
            for row in trades:
                bucket = str(row.get(key) or "UNKNOWN").upper()
                buckets.setdefault(bucket, []).append(_trade_score(row))
            if not buckets:
                return ""
            scored = {
                bucket: (sum(values) / len(values)) if values else 0.0
                for bucket, values in buckets.items()
            }
            ordered = sorted(scored.items(), key=lambda item: (item[1], item[0]), reverse=reverse)
            return ordered[0][0] if ordered else ""

        def _top_tags(*, reverse: bool) -> list[str]:
            tag_scores: dict[str, list[float]] = {}
            for row in trades:
                for tag in _normalize_tags(row.get("adjustment_tags")):
                    tag_scores.setdefault(tag, []).append(_trade_score(row))
            if not tag_scores:
                return []
            scored = sorted(
                (
                    (tag, (sum(values) / len(values)) if values else 0.0)
                    for tag, values in tag_scores.items()
                ),
                key=lambda item: (item[1], item[0]),
                reverse=reverse,
            )
            return [tag for tag, _ in scored[:3]]

        strategies = self._metrics.setdefault("strategies", {})
        state = strategies.setdefault(strategy, {})
        metrics = {
            "trade_count": trade_count,
            "trades": trade_count,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 4),
            "avg_r": round(expectancy_r, 4),
            "avg_rr": round(avg_rr, 4),
            "expectancy_r": round(expectancy_r, 4),
            "profit_factor": round(float(profit_factor), 4) if profit_factor != float("inf") else float("inf"),
            "max_drawdown_r": round(max_drawdown, 4),
            "avg_duration_minutes": round(avg_duration, 2),
            "best_session": _best_bucket("session", reverse=True),
            "worst_session": _best_bucket("session", reverse=False),
            "best_spread_bucket": _best_bucket("spread_bucket", reverse=True),
            "best_regime_bucket": _best_bucket("regime", reverse=True),
            "best_quality_band": _best_bucket("quality_band", reverse=True),
            "top_positive_tags": _top_tags(reverse=True),
            "top_negative_tags": _top_tags(reverse=False),
            "last_trade_at": str(trades[-1].get("timestamp_utc") if trades else ""),
            "last_updated": _now_iso(),
        }
        state.update(metrics)
        state.setdefault("adjustments", self._default_adjustments())
        state["current_probability_floor_mult"] = round(
            _safe_float(state.get("adjustments", {}).get("probability_floor_mult"), 1.0),
            4,
        )
        state["current_confluence_floor_mult"] = round(
            _safe_float(state.get("adjustments", {}).get("confluence_floor_mult"), 1.0),
            4,
        )
        state["current_candidate_sensitivity_mult"] = round(
            _safe_float(state.get("adjustments", {}).get("candidate_sensitivity_mult"), 1.0),
            4,
        )
        state["recommended_probability_floor_multiplier"] = round(
            _safe_float(state.get("adjustments", {}).get("probability_floor_mult"), 1.0),
            4,
        )
        state["recommended_confluence_multiplier"] = round(
            _safe_float(state.get("adjustments", {}).get("confluence_floor_mult"), 1.0),
            4,
        )
        state["recommended_spacing_multiplier"] = round(
            _safe_float(state.get("adjustments", {}).get("xau_grid_spacing_mult"), 1.0),
            4,
        )
        self._metrics["updated_at"] = _now_iso()
        return state

    def _bucket_key(
        self,
        *,
        symbol: str,
        strategy: str,
        lane: str,
        regime: str,
        session: str,
        direction: str,
        quality_band: str,
    ) -> str:
        return "|".join(
            [
                str(symbol or "").upper(),
                str(strategy or "").upper(),
                str(lane or "UNKNOWN").upper(),
                str(regime or "UNKNOWN").upper(),
                str(session or "UNKNOWN").upper(),
                str(direction or "UNKNOWN").upper(),
                _quality_band(quality_band),
            ]
        )

    def _bucket_key_from_row(self, row: dict[str, Any]) -> str:
        return self._bucket_key(
            symbol=str(row.get("symbol") or ""),
            strategy=str(row.get("strategy") or ""),
            lane=str(row.get("lane") or "UNKNOWN"),
            regime=str(row.get("regime") or ""),
            session=str(row.get("session") or ""),
            direction=str(row.get("side") or ""),
            quality_band=str(row.get("quality_band") or "unknown"),
        )

    @staticmethod
    def _summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
        pnl = [_trade_score(row) for row in rows]
        wins = [value for value in pnl if value > 0]
        losses = [abs(value) for value in pnl if value < 0]
        trade_count = len(pnl)
        return {
            "trades": trade_count,
            "win_rate": (sum(1 for value in pnl if value >= 0) / trade_count) if trade_count else 0.0,
            "expectancy_r": (sum(pnl) / trade_count) if trade_count else 0.0,
            "profit_factor": (sum(wins) / max(sum(losses), 1e-9)) if losses else (float("inf") if wins else 0.0),
            "avg_win_r": (sum(wins) / len(wins)) if wins else 0.0,
            "avg_loss_r": (sum(losses) / len(losses)) if losses else 0.0,
        }

    def _prune_history(self) -> None:
        rows = self._history.get("trades", [])
        if not isinstance(rows, list):
            self._history["trades"] = []
            return
        cutoff = datetime.now(tz=UTC) - timedelta(days=max(1, int(self.lookback_days)))
        kept: list[dict[str, Any]] = []
        for row in rows:
            timestamp_raw = str((row or {}).get("timestamp_utc") or "")
            if not timestamp_raw:
                continue
            try:
                timestamp = datetime.fromisoformat(timestamp_raw.replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=UTC)
                timestamp = timestamp.astimezone(UTC)
            except Exception:
                continue
            if timestamp >= cutoff:
                kept.append(dict(row))
        self._history["trades"] = kept
        self._history["updated_at"] = _now_iso()

    def _save_all(self) -> None:
        self._save_json(self.trade_history_path, self._history)
        self._save_json(self.metrics_path, self._metrics)

    @staticmethod
    def _default_adjustments() -> dict[str, float]:
        return {
            "probability_floor_mult": 1.0,
            "confluence_floor_mult": 1.0,
            "candidate_sensitivity_mult": 1.0,
            "xau_grid_spacing_mult": 1.0,
            "xau_grid_probe_aggression": 1.0,
        }

    def _load_json(self, path: Path, fallback: dict[str, Any]) -> dict[str, Any]:
        if not path.exists():
            return dict(fallback)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return dict(fallback)
        if isinstance(data, dict):
            return data
        return dict(fallback)

    def _save_json(self, path: Path, payload: dict[str, Any]) -> None:
        ensure_parent(path)
        path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
