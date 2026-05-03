from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import json
import sqlite3

from src.utils import ensure_parent, utc_now

UTC = timezone.utc


@dataclass
class RuleChangeConfig:
    enabled: bool = False
    min_samples: int = 50
    min_grid_cycles: int = 10
    cooldown_hours: int = 24
    shadow_trades_required: int = 25
    max_delta_pct: float = 0.20
    min_profit_factor_gain: float = 0.05
    min_expectancy_gain: float = 0.02
    allowed_params: tuple[str, ...] = (
        "min_confluence_score",
        "xau_grid_step_points",
        "xau_grid_cycle_risk_pct",
        "xau_recycle_cooldown_seconds",
    )
    locked_params: tuple[str, ...] = (
        "xau_grid_max_legs",
        "max_total_open_positions",
        "max_open_positions_per_symbol",
        "max_daily_loss_pct",
        "max_daily_dd_pct",
    )


@dataclass
class RuleChangeDecision:
    approved: bool
    reason: str
    strategy: str
    version: int = 0
    applied_params: dict[str, Any] | None = None


class RuleChangeManager:
    def __init__(self, db_path: Path, config: RuleChangeConfig, logger: Any | None = None) -> None:
        self.db_path = db_path
        self.config = config
        self.logger = logger
        ensure_parent(db_path)
        self._init_db()

    def record_trade(self, strategy: str, pnl_r: float, mode: str = "LIVE") -> None:
        now = utc_now().isoformat()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rule_change_samples (strategy, pnl_r, win, mode, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(strategy),
                    float(pnl_r),
                    1 if float(pnl_r) > 0 else 0,
                    str(mode).upper(),
                    now,
                ),
            )
            connection.commit()

    def latest_decision(self, strategy: str) -> dict[str, Any] | None:
        with self._connect(row_factory=True) as connection:
            row = connection.execute(
                """
                SELECT strategy, version, params_json, reason, applied_at
                FROM rule_change_history
                WHERE strategy = ?
                ORDER BY version DESC
                LIMIT 1
                """,
                (str(strategy),),
            ).fetchone()
        if row is None:
            return None
        return {
            "strategy": str(row["strategy"]),
            "version": int(row["version"]),
            "params": json.loads(str(row["params_json"] or "{}")),
            "reason": str(row["reason"] or ""),
            "applied_at": str(row["applied_at"] or ""),
        }

    def evaluate_and_apply(
        self,
        *,
        strategy: str,
        baseline_metrics: dict[str, float],
        candidate_metrics: dict[str, float],
        current_params: dict[str, float],
        proposed_params: dict[str, float],
        shadow_trades: int,
        now_utc: datetime | None = None,
    ) -> RuleChangeDecision:
        now = (now_utc or utc_now()).astimezone(UTC)
        strategy_name = str(strategy)
        if not self.config.enabled:
            return RuleChangeDecision(False, "rule_change_disabled", strategy_name)

        sample_count = self._sample_count(strategy_name)
        if sample_count < int(self.config.min_samples):
            return RuleChangeDecision(False, "insufficient_samples", strategy_name)
        if int(shadow_trades) < int(self.config.shadow_trades_required):
            return RuleChangeDecision(False, "insufficient_shadow_trades", strategy_name)

        last = self.latest_decision(strategy_name)
        if last and last.get("applied_at"):
            try:
                last_applied = datetime.fromisoformat(str(last["applied_at"]).replace("Z", "+00:00")).astimezone(UTC)
                if now < (last_applied + timedelta(hours=int(self.config.cooldown_hours))):
                    return RuleChangeDecision(False, "cooldown_active", strategy_name)
            except Exception:
                pass

        pf_base = float(baseline_metrics.get("profit_factor", 0.0))
        pf_new = float(candidate_metrics.get("profit_factor", 0.0))
        exp_base = float(baseline_metrics.get("expectancy_r", 0.0))
        exp_new = float(candidate_metrics.get("expectancy_r", 0.0))
        wr_base = float(baseline_metrics.get("win_rate", 0.0))
        wr_new = float(candidate_metrics.get("win_rate", 0.0))
        dd_base = float(baseline_metrics.get("max_drawdown_pct", 1.0))
        dd_new = float(candidate_metrics.get("max_drawdown_pct", 1.0))

        if (pf_new - pf_base) < float(self.config.min_profit_factor_gain):
            return RuleChangeDecision(False, "pf_gain_too_small", strategy_name)
        if (exp_new - exp_base) < float(self.config.min_expectancy_gain):
            return RuleChangeDecision(False, "expectancy_gain_too_small", strategy_name)
        if wr_new + 1e-9 < wr_base:
            return RuleChangeDecision(False, "win_rate_not_improved", strategy_name)
        if dd_new > dd_base:
            return RuleChangeDecision(False, "drawdown_worse", strategy_name)

        bounded_params: dict[str, float] = {}
        for key, proposed_value in proposed_params.items():
            if self.config.allowed_params and str(key) not in set(self.config.allowed_params):
                return RuleChangeDecision(False, f"param_not_allowed:{key}", strategy_name)
            if self.config.locked_params and str(key) in set(self.config.locked_params):
                return RuleChangeDecision(False, f"param_locked:{key}", strategy_name)
            base_value = float(current_params.get(key, proposed_value))
            if base_value <= 0:
                bounded_params[key] = float(proposed_value)
                continue
            delta_pct = abs(float(proposed_value) - base_value) / base_value
            if delta_pct > float(self.config.max_delta_pct):
                return RuleChangeDecision(False, f"delta_too_large:{key}", strategy_name)
            bounded_params[key] = float(proposed_value)

        version = int(last.get("version", 0) if last else 0) + 1
        self._insert_history(strategy_name, version, bounded_params, "metrics_improved_and_bounded", now)
        return RuleChangeDecision(True, "applied", strategy_name, version=version, applied_params=bounded_params)

    def summary(self) -> dict[str, Any]:
        with self._connect(row_factory=True) as connection:
            rows = connection.execute(
                """
                SELECT strategy, MAX(version) AS version, MAX(applied_at) AS applied_at
                FROM rule_change_history
                GROUP BY strategy
                """
            ).fetchall()
        return {
            str(row["strategy"]): {
                "version": int(row["version"] or 0),
                "applied_at": str(row["applied_at"] or ""),
            }
            for row in rows
        }

    def _sample_count(self, strategy: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM rule_change_samples WHERE strategy = ?",
                (str(strategy),),
            ).fetchone()
        return int(row[0]) if row else 0

    def _insert_history(self, strategy: str, version: int, params: dict[str, float], reason: str, now: datetime) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rule_change_history (strategy, version, params_json, reason, applied_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    str(strategy),
                    int(version),
                    json.dumps(params, sort_keys=True),
                    str(reason),
                    now.isoformat(),
                ),
            )
            connection.commit()

    def _connect(self, row_factory: bool = False) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        if row_factory:
            connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rule_change_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    pnl_r REAL NOT NULL,
                    win INTEGER NOT NULL,
                    mode TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rule_change_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    params_json TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    applied_at TEXT NOT NULL
                )
                """
            )
            connection.commit()
