from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import math
import json
import logging
import sqlite3

from src.logger import ApexLogger
from src.mt5_client import MT5Client, OrderResult
from src.risk_engine import TradeStats
from src.session_calendar import SYDNEY
from src.strategies.trend_daytrade import resolve_strategy_key
from src.utils import ensure_parent, utc_now

_RUNTIME_LOGGER = logging.getLogger("apex")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(value)
    except Exception:
        return float(default)
    if math.isfinite(parsed):
        return parsed
    return float(default)


def _sqlite_low_space_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return "database or disk is full" in message or "disk i/o error" in message


def _coerce_timestamp(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        try:
            parsed = datetime.fromtimestamp(float(value), tz=timezone.utc)
        except Exception:
            return None
    else:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            try:
                parsed = datetime.fromtimestamp(float(raw), tz=timezone.utc)
            except Exception:
                return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _first_positive_float(*values: Any) -> float:
    for value in values:
        try:
            parsed = float(value)
        except Exception:
            continue
        if math.isfinite(parsed) and parsed > 0.0:
            return float(parsed)
    return 0.0


def trading_day_key_for_timestamp(timestamp: Any, tz: timezone = SYDNEY) -> str:
    parsed = _coerce_timestamp(timestamp)
    if parsed is None:
        return ""
    return parsed.astimezone(tz).date().isoformat()


def current_trading_day_key(tz: timezone = SYDNEY, now_ts: datetime | None = None) -> str:
    return trading_day_key_for_timestamp(now_ts or utc_now(), tz=tz)


def trading_day_time_for_timestamp(timestamp: Any, tz: timezone = SYDNEY) -> str:
    parsed = _coerce_timestamp(timestamp)
    if parsed is None:
        return ""
    return parsed.astimezone(tz).isoformat()


@dataclass
class ExecutionRequest:
    signal_id: str
    symbol: str
    side: str
    volume: float
    entry_price: float
    stop_price: float
    take_profit_price: float
    mode: str
    setup: str
    regime: str
    probability: float
    expected_value_r: float
    slippage_points: int
    trailing_enabled: bool = True
    partial_close_enabled: bool = False
    news_status: str = ""
    final_decision_json: str = ""
    trading_enabled: bool = False
    account: str = ""
    magic: int = 55_001
    timeframe: str = ""
    proof_trade: bool = False
    entry_reason: str = ""
    ai_summary_json: str = ""
    broker_snapshot_json: str = ""
    entry_context_json: str = ""
    account_currency: str = ""
    entry_spread_points: float = 0.0
    session_name: str = ""
    strategy_key: str = ""


@dataclass
class ExecutionReceipt:
    accepted: bool
    reason: str
    order_id: str | None = None
    raw: dict[str, Any] | None = None


@dataclass
class TradeJournal:
    db_path: Path

    def __post_init__(self) -> None:
        ensure_parent(self.db_path)
        self._init_db()

    @staticmethod
    def _trading_day_key(timestamp: datetime) -> str:
        return trading_day_key_for_timestamp(timestamp, tz=SYDNEY)

    @staticmethod
    def _trading_day_start(timestamp: datetime) -> datetime:
        parsed = _coerce_timestamp(timestamp)
        if parsed is None:
            parsed = utc_now()
        local_now = parsed.astimezone(SYDNEY)
        local_start = datetime.combine(local_now.date(), datetime.min.time(), tzinfo=SYDNEY)
        return local_start.astimezone(timezone.utc)

    @classmethod
    def neutral_stats(
        cls,
        *,
        current_equity: float | None = None,
        now_ts: datetime | None = None,
    ) -> TradeStats:
        timestamp = now_ts or utc_now()
        baseline = max(1.0, float(current_equity if current_equity is not None else 1.0))
        return TradeStats(
            win_rate=0.5,
            avg_win_r=1.0,
            avg_loss_r=1.0,
            consecutive_losses=0,
            winning_streak=0,
            cooldown_trades_remaining=0,
            closed_trades_total=0,
            trades_today=0,
            daily_pnl_pct=0.0,
            day_start_equity=baseline,
            day_high_equity=baseline,
            daily_realized_pnl=0.0,
            daily_dd_pct_live=0.0,
            rolling_drawdown_pct=0.0,
            absolute_drawdown_pct=0.0,
            soft_dd_trade_count=0,
            trading_day_key=str(cls._trading_day_key(timestamp)),
            timezone_used=str(SYDNEY),
            today_closed_trade_count=0,
            day_bucket_source="close_time_sydney",
            day_start_equity_source="neutral_baseline",
            day_high_equity_source="neutral_baseline",
        )

    @classmethod
    def _closed_row_day_key(cls, closed_at: Any) -> str:
        return trading_day_key_for_timestamp(closed_at, tz=SYDNEY)

    @staticmethod
    def _live_trade_filter(alias: str = "") -> str:
        prefix = f"{alias}." if alias else ""
        return (
            f"COALESCE({prefix}proof_trade, 0) = 0 AND "
            f"{prefix}signal_id NOT LIKE 'FORCE_TEST::%' "
            f"AND {prefix}signal_id NOT LIKE 'CANARY::%' "
            f"AND COALESCE({prefix}setup, '') NOT LIKE '%FORCE_TEST%' "
            f"AND COALESCE({prefix}setup, '') NOT LIKE '%CANARY%'"
        )

    @staticmethod
    def _scope_clause(
        *,
        account: str | None = None,
        magic: int | None = None,
        symbol: str | None = None,
        include_proof: bool = False,
        alias: str = "",
    ) -> tuple[str, list[Any]]:
        prefix = f"{alias}." if alias else ""
        clauses: list[str] = []
        params: list[Any] = []
        if not include_proof:
            clauses.append(f"COALESCE({prefix}proof_trade, 0) = 0")
        if account is not None and str(account).strip():
            clauses.append(f"{prefix}account = ?")
            params.append(str(account).strip())
        if magic is not None:
            clauses.append(f"{prefix}magic = ?")
            params.append(int(magic))
        if symbol is not None and str(symbol).strip():
            clauses.append(f"{prefix}symbol = ?")
            params.append(str(symbol).upper())
        if not clauses:
            return "", params
        return " AND " + " AND ".join(clauses), params

    def has_signal(self, signal_id: str) -> bool:
        with self._connect() as connection:
            row = connection.execute("SELECT 1 FROM trade_journal WHERE signal_id = ?", (signal_id,)).fetchone()
        return row is not None

    def record_execution(self, request: ExecutionRequest, result: OrderResult, equity: float) -> None:
        session_name = str(request.session_name or "").strip()
        strategy_key = str(request.strategy_key or "").strip()
        if (not session_name) or (not strategy_key):
            try:
                entry_context = json.loads(str(request.entry_context_json or "{}"))
            except Exception:
                entry_context = {}
            if isinstance(entry_context, dict):
                if not session_name:
                    session_name = str(entry_context.get("session_name") or "").strip()
                if not strategy_key:
                    strategy_key = str(
                        entry_context.get("strategy_key")
                        or entry_context.get("strategy_family")
                        or ""
                    ).strip()
        result_raw = dict(result.raw or {}) if isinstance(result.raw, dict) else {}
        broker_snapshot: dict[str, Any] = {}
        try:
            broker_snapshot = json.loads(str(request.broker_snapshot_json or "{}"))
        except Exception:
            broker_snapshot = {}
        if not isinstance(broker_snapshot, dict):
            broker_snapshot = {}
        resolved_entry_price = float(request.entry_price or 0.0)
        if result.accepted:
            resolved_entry_price = _first_positive_float(
                result_raw.get("entry_price"),
                result_raw.get("price"),
                result_raw.get("avg_entry"),
                broker_snapshot.get("avg_entry"),
                broker_snapshot.get("price"),
                request.entry_price,
            ) or float(request.entry_price or 0.0)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO trade_journal (
                    signal_id, ticket, symbol, side, setup, mode, status, created_at, opened_at,
                    entry_price, sl, tp, volume, probability, expected_value_r, regime, strategy_key, equity_at_open,
                    trailing_enabled, partial_close_enabled, news_status, final_decision,
                    account, magic, timeframe, proof_trade, entry_reason, ai_summary_json,
                    broker_snapshot_json, entry_context_json, account_currency, entry_spread_points, session_name
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    request.signal_id,
                    result.order_id,
                    request.symbol,
                    request.side,
                    request.setup,
                    request.mode.upper(),
                    "EXECUTED" if result.accepted else "REJECTED",
                    utc_now().isoformat(),
                    utc_now().isoformat() if result.accepted else None,
                    resolved_entry_price,
                    request.stop_price,
                    request.take_profit_price,
                    request.volume,
                    request.probability,
                    request.expected_value_r,
                    request.regime,
                    strategy_key,
                    equity,
                    1 if request.trailing_enabled else 0,
                    1 if request.partial_close_enabled else 0,
                    request.news_status,
                    request.final_decision_json,
                    str(request.account or ""),
                    int(request.magic),
                    str(request.timeframe or ""),
                    1 if request.proof_trade else 0,
                    str(request.entry_reason or ""),
                    str(request.ai_summary_json or ""),
                    str(request.broker_snapshot_json or ""),
                    str(request.entry_context_json or ""),
                    str(request.account_currency or ""),
                    float(request.entry_spread_points or 0.0),
                    session_name,
                ),
            )
            connection.execute(
                "INSERT OR REPLACE INTO journal_state (key, value) VALUES (?, ?)",
                ("last_signal_id", request.signal_id),
            )
            if result.accepted and (not bool(request.proof_trade)):
                base_scope = self._state_scope(account=request.account, magic=request.magic)
                symbol_scope = self._state_scope(account=request.account, magic=request.magic, symbol=request.symbol)
                self._consume_cooldown(connection, f"{base_scope}cooldown_remaining")
                self._consume_cooldown(connection, f"{symbol_scope}cooldown_remaining")
                try:
                    entry_context = json.loads(str(request.entry_context_json or "{}"))
                except Exception:
                    entry_context = {}
                if bool(entry_context.get("soft_dd_active")):
                    count_key = f"{base_scope}soft_dd_trade_count"
                    current_count = int(self._get_state(connection, count_key, "0"))
                    self._set_state(connection, count_key, str(current_count + 1))
            connection.commit()

    def mark_closed(
        self,
        signal_id: str,
        pnl_amount: float,
        pnl_r: float,
        equity_after_close: float,
        *,
        closed_at: datetime | str | float | int | None = None,
        exit_reason: str = "",
        exit_price: float = 0.0,
        duration_minutes: float = 0.0,
        exit_spread_points: float = 0.0,
        slippage_points: float = 0.0,
        close_context_json: str = "",
        post_trade_review_json: str = "",
        adjustment_tags_json: str = "",
        management_effect_json: str = "",
    ) -> None:
        closed_at_value = _coerce_timestamp(closed_at or utc_now()) or utc_now()
        with self._connect() as connection:
            symbol = self._get_trade_symbol(connection, signal_id)
            trade_scope = self._get_trade_scope(connection, signal_id)
            proof_trade = self._is_proof_trade(connection, signal_id)
            connection.execute(
                """
                UPDATE trade_journal
                SET status = 'CLOSED',
                    closed_at = ?,
                    pnl_amount = ?,
                    pnl_r = ?,
                    equity_after_close = ?,
                    exit_reason = ?,
                    exit_price = ?,
                    duration_minutes = ?,
                    exit_spread_points = ?,
                    slippage_points = ?,
                    close_context_json = ?,
                    post_trade_review_json = ?,
                    adjustment_tags_json = ?,
                    management_effect_json = ?
                WHERE signal_id = ?
                """,
                (
                    closed_at_value.isoformat(),
                    pnl_amount,
                    pnl_r,
                    equity_after_close,
                    str(exit_reason or ""),
                    float(exit_price or 0.0),
                    float(duration_minutes or 0.0),
                    float(exit_spread_points or 0.0),
                    float(slippage_points or 0.0),
                    str(close_context_json or ""),
                    str(post_trade_review_json or ""),
                    str(adjustment_tags_json or ""),
                    str(management_effect_json or ""),
                    signal_id,
                ),
            )
            if not proof_trade:
                base_scope = self._state_scope(account=trade_scope.get("account"), magic=trade_scope.get("magic"))
                trading_day_key = self._trading_day_key(closed_at_value)
                self._update_loss_state(
                    connection,
                    f"{base_scope}consecutive_losses",
                    f"{base_scope}cooldown_remaining",
                    pnl_amount,
                    trading_day_key=trading_day_key,
                )
                if symbol:
                    self._update_loss_state(
                        connection,
                        f"{self._state_scope(account=trade_scope.get('account'), magic=trade_scope.get('magic'), symbol=symbol)}consecutive_losses",
                        f"{self._state_scope(account=trade_scope.get('account'), magic=trade_scope.get('magic'), symbol=symbol)}cooldown_remaining",
                        pnl_amount,
                        trading_day_key=trading_day_key,
                    )
            connection.commit()

    def update_trade_identity(
        self,
        signal_id: str,
        *,
        session_name: str = "",
        strategy_key: str = "",
    ) -> None:
        session_value = str(session_name or "").strip()
        strategy_value = str(strategy_key or "").strip()
        if not session_value and not strategy_value:
            return
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE trade_journal
                SET session_name = CASE
                        WHEN TRIM(COALESCE(session_name, '')) = '' AND ? <> '' THEN ?
                        ELSE session_name
                    END,
                    strategy_key = CASE
                        WHEN TRIM(COALESCE(strategy_key, '')) = '' AND ? <> '' THEN ?
                        ELSE strategy_key
                    END
                WHERE signal_id = ?
                """,
                (
                    session_value,
                    session_value,
                    strategy_value,
                    strategy_value,
                    signal_id,
                ),
            )
            connection.commit()

    def reconcile_missing_close(
        self,
        signal_id: str,
        *,
        exit_reason: str = "broker_snapshot_reconcile",
        equity_after_close: float | None = None,
        exit_price: float = 0.0,
        exit_spread_points: float = 0.0,
        close_context_json: str = "",
        post_trade_review_json: str = "",
        adjustment_tags_json: str = "",
        management_effect_json: str = "",
    ) -> None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT status, COALESCE(equity_at_open, 0), account_currency
                FROM trade_journal
                WHERE signal_id = ?
                LIMIT 1
                """,
                (signal_id,),
            ).fetchone()
            if row is None or str(row[0] or "").upper() != "EXECUTED":
                return
            fallback_equity = float(row[1] or 0.0)
            connection.execute(
                """
                UPDATE trade_journal
                SET status = 'RECONCILED_CLOSED',
                    closed_at = ?,
                    pnl_amount = 0.0,
                    pnl_r = 0.0,
                    equity_after_close = ?,
                    exit_reason = ?,
                    exit_price = ?,
                    duration_minutes = COALESCE(duration_minutes, 0.0),
                    exit_spread_points = ?,
                    slippage_points = COALESCE(slippage_points, 0.0),
                    close_context_json = ?,
                    post_trade_review_json = ?,
                    adjustment_tags_json = ?,
                    management_effect_json = ?
                WHERE signal_id = ?
                  AND status = 'EXECUTED'
                """,
                (
                    utc_now().isoformat(),
                    float(equity_after_close if equity_after_close is not None else fallback_equity),
                    str(exit_reason or "broker_snapshot_reconcile"),
                    float(exit_price or 0.0),
                    float(exit_spread_points or 0.0),
                    str(close_context_json or ""),
                    str(post_trade_review_json or ""),
                    str(adjustment_tags_json or ""),
                    str(management_effect_json or ""),
                    signal_id,
                ),
            )
            connection.commit()

    def update_open_volume(self, signal_id: str, volume: float) -> None:
        with self._connect() as connection:
            connection.execute(
                "UPDATE trade_journal SET volume = ? WHERE signal_id = ? AND status = 'EXECUTED'",
                (volume, signal_id),
            )
            connection.commit()

    def log_event(self, signal_id: str, event_type: str, payload: dict[str, Any]) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO trade_events (signal_id, created_at, event_type, payload) VALUES (?, ?, ?, ?)",
                (signal_id, utc_now().isoformat(), event_type, json.dumps(payload, sort_keys=True, default=str)),
            )
            connection.commit()

    def events_for_signal(self, signal_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT created_at, event_type, payload
                FROM trade_events
                WHERE signal_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (signal_id, max(1, int(limit))),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            output.append(
                {
                    "created_at": str(row[0] or ""),
                    "event_type": str(row[1] or ""),
                    "payload": str(row[2] or ""),
                }
            )
        return output

    def recent_review_summary(
        self,
        limit: int = 50,
        symbol: str | None = None,
        *,
        account: str | None = None,
        magic: int | None = None,
        include_proof: bool = False,
    ) -> dict[str, Any]:
        live_filter = "1=1" if include_proof else self._live_trade_filter()
        scope_clause, scope_params = self._scope_clause(account=account, magic=magic, symbol=symbol, include_proof=include_proof)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT exit_reason, post_trade_review_json, adjustment_tags_json, management_effect_json
                FROM trade_journal
                WHERE status = 'CLOSED'
                  AND {live_filter}
                  {scope_clause}
                ORDER BY closed_at DESC
                LIMIT ?
                """,
                tuple(scope_params + [max(1, int(limit))]),
            ).fetchall()

        issue_counts: dict[str, int] = {}
        exit_reason_counts: dict[str, int] = {}
        adjustment_counts: dict[str, int] = {}
        management_counts: dict[str, int] = {}
        for row in rows:
            exit_reason = str(row[0] or "").strip() or "unknown"
            exit_reason_counts[exit_reason] = int(exit_reason_counts.get(exit_reason, 0)) + 1
            try:
                review_payload = json.loads(str(row[1] or "{}"))
            except Exception:
                review_payload = {}
            try:
                adjustment_tags = json.loads(str(row[2] or "[]"))
            except Exception:
                adjustment_tags = []
            try:
                management_effect = json.loads(str(row[3] or "{}"))
            except Exception:
                management_effect = {}

            likely_issue = str(review_payload.get("likely_issue") or "").strip()
            if likely_issue:
                issue_counts[likely_issue] = int(issue_counts.get(likely_issue, 0)) + 1
            for item in adjustment_tags if isinstance(adjustment_tags, list) else []:
                tag = str(item or "").strip()
                if not tag:
                    continue
                adjustment_counts[tag] = int(adjustment_counts.get(tag, 0)) + 1
            management_mode = str(
                management_effect.get("protection_mode")
                or management_effect.get("management_action")
                or management_effect.get("management_match_mode")
                or ""
            ).strip()
            if management_mode:
                management_counts[management_mode] = int(management_counts.get(management_mode, 0)) + 1

        return {
            "trades_reviewed": int(len(rows)),
            "issue_counts": issue_counts,
            "exit_reason_counts": exit_reason_counts,
            "adjustment_tag_counts": adjustment_counts,
            "management_counts": management_counts,
        }

    def reset_daily_guard(
        self,
        *,
        account: str | None = None,
        magic: int | None = None,
        current_equity: float | None = None,
        reset_at: datetime | None = None,
    ) -> None:
        base_scope = self._state_scope(account=account, magic=magic)
        if not base_scope:
            return
        now_ts = reset_at or utc_now()
        baseline = max(1.0, float(current_equity if current_equity is not None else 1.0))
        with self._connect() as connection:
            current_day = self._trading_day_key(now_ts)
            self._set_state(connection, f"{base_scope}daily_equity_day", current_day)
            self._set_state(connection, f"{base_scope}day_start_equity", str(baseline))
            self._set_state(connection, f"{base_scope}day_high_equity", str(baseline))
            self._set_state(connection, f"{base_scope}day_start_equity_source", "operator_reset_current_equity")
            self._set_state(connection, f"{base_scope}day_high_equity_source", "operator_reset_current_equity")
            self._set_state(connection, f"{base_scope}daily_reset_at", now_ts.isoformat())
            self._set_state(connection, f"{base_scope}soft_dd_trade_count", "0")
            self._set_state(connection, f"{base_scope}consecutive_losses", "0")
            self._set_state(connection, f"{base_scope}cooldown_remaining", "0")
            self._set_state(connection, f"{base_scope}consecutive_losses_trading_day", current_day)
            rows = connection.execute(
                """
                SELECT key
                FROM journal_state
                WHERE key LIKE ?
                  AND (key LIKE '%consecutive_losses' OR key LIKE '%cooldown_remaining' OR key LIKE '%consecutive_losses_trading_day')
                """,
                (f"{base_scope}%__",),
            ).fetchall()
            for row in rows:
                key = str(row[0] or "")
                if not key or key.startswith(base_scope) is False:
                    continue
                if key.endswith("consecutive_losses"):
                    self._set_state(connection, key, "0")
                elif key.endswith("cooldown_remaining"):
                    self._set_state(connection, key, "0")
                elif key.endswith("consecutive_losses_trading_day"):
                    self._set_state(connection, key, current_day)
            connection.commit()

    def reset_daily_counters(
        self,
        *,
        account: str | None = None,
        magic: int | None = None,
        current_equity: float | None = None,
        reset_at: datetime | None = None,
    ) -> None:
        base_scope = self._state_scope(account=account, magic=magic)
        if not base_scope:
            return
        now_ts = reset_at or utc_now()
        baseline = max(1.0, float(current_equity if current_equity is not None else 1.0))
        current_day = self._trading_day_key(now_ts)
        with self._connect() as connection:
            self._set_state(connection, f"{base_scope}daily_equity_day", current_day)
            self._set_state(connection, f"{base_scope}day_start_equity", str(baseline))
            self._set_state(connection, f"{base_scope}day_high_equity", str(baseline))
            self._set_state(connection, f"{base_scope}day_start_equity_source", "operator_reset_current_equity")
            self._set_state(connection, f"{base_scope}day_high_equity_source", "operator_reset_current_equity")
            self._set_state(connection, f"{base_scope}daily_reset_at", now_ts.isoformat())
            self._set_state(connection, f"{base_scope}soft_dd_trade_count", "0")
            self._set_state(connection, f"{base_scope}consecutive_losses", "0")
            self._set_state(connection, f"{base_scope}cooldown_remaining", "0")
            self._set_state(connection, f"{base_scope}consecutive_losses_trading_day", current_day)
            connection.commit()
        _RUNTIME_LOGGER.info(
            "daily_counters_reset",
            extra={
                "extra_fields": {
                    "account": str(account or ""),
                    "magic": int(magic or 0),
                    "trading_day_key": str(current_day),
                    "day_start_equity": float(baseline),
                    "reset_daily_realized_pnl": True,
                    "reset_closed_trades_today": True,
                    "reset_day_high_equity": True,
                    "timezone_used_for_reset": str(getattr(SYDNEY, "key", SYDNEY)),
                }
            },
        )

    def get_open_positions(
        self,
        *,
        account: str | None = None,
        magic: int | None = None,
        include_proof: bool = False,
    ) -> list[dict[str, Any]]:
        scope_clause, scope_params = self._scope_clause(account=account, magic=magic, include_proof=include_proof)
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT signal_id, ticket, side, symbol, setup, regime, entry_price, sl, tp, volume, opened_at, probability,
                       trailing_enabled, partial_close_enabled, news_status, final_decision, account, magic, timeframe,
                       proof_trade, ai_summary_json, broker_snapshot_json, entry_context_json, account_currency,
                       entry_spread_points, session_name, strategy_key
                FROM trade_journal
                WHERE status = 'EXECUTED'
                  {scope_clause}
                ORDER BY opened_at ASC
                """.format(scope_clause=scope_clause),
                tuple(scope_params),
            ).fetchall()
        output = []
        for row in rows:
            output.append(
                {
                    "signal_id": row[0],
                    "ticket": row[1],
                    "side": row[2],
                    "symbol": row[3],
                    "setup": row[4],
                    "regime": row[5],
                    "entry_price": row[6],
                    "sl": row[7],
                    "tp": row[8],
                    "volume": row[9],
                    "opened_at": row[10],
                    "probability": row[11],
                    "trailing_enabled": bool(row[12]),
                    "partial_close_enabled": bool(row[13]),
                    "news_status": row[14],
                    "final_decision": row[15],
                    "account": row[16],
                    "magic": int(row[17] or 0),
                    "timeframe": row[18],
                    "proof_trade": bool(row[19]),
                    "ai_summary_json": row[20],
                    "broker_snapshot_json": row[21],
                    "entry_context_json": row[22],
                    "account_currency": row[23],
                    "entry_spread_points": float(row[24] or 0.0),
                    "session_name": row[25],
                    "strategy_key": row[26],
                }
            )
        return output

    def get_trade(self, signal_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT signal_id, ticket, symbol, side, setup, regime, mode, status, created_at, opened_at, closed_at,
                       entry_price, sl, tp, volume, probability, expected_value_r, trailing_enabled, partial_close_enabled,
                       news_status, final_decision, pnl_amount, pnl_r, equity_at_open, equity_after_close,
                       account, magic, timeframe, proof_trade, entry_reason, ai_summary_json, broker_snapshot_json,
                       entry_context_json, close_context_json, account_currency, entry_spread_points, exit_spread_points,
                       exit_reason, exit_price, duration_minutes, slippage_points, post_trade_review_json,
                       adjustment_tags_json, management_effect_json, session_name, strategy_key
                FROM trade_journal
                WHERE signal_id = ?
                """,
                (signal_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "signal_id": row[0],
            "ticket": row[1],
            "symbol": row[2],
            "side": row[3],
            "setup": row[4],
            "regime": row[5],
            "mode": row[6],
            "status": row[7],
            "created_at": row[8],
            "opened_at": row[9],
            "closed_at": row[10],
            "entry_price": row[11],
            "sl": row[12],
            "tp": row[13],
            "volume": row[14],
            "probability": row[15],
            "expected_value_r": row[16],
            "trailing_enabled": bool(row[17]),
            "partial_close_enabled": bool(row[18]),
            "news_status": row[19],
            "final_decision": row[20],
            "pnl_amount": row[21],
            "pnl_r": row[22],
            "equity_at_open": row[23],
            "equity_after_close": row[24],
            "account": row[25],
            "magic": int(row[26] or 0),
            "timeframe": row[27],
            "proof_trade": bool(row[28]),
            "entry_reason": row[29],
            "ai_summary_json": row[30],
            "broker_snapshot_json": row[31],
            "entry_context_json": row[32],
            "close_context_json": row[33],
            "account_currency": row[34],
            "entry_spread_points": float(row[35] or 0.0),
            "exit_spread_points": float(row[36] or 0.0),
            "exit_reason": row[37],
            "exit_price": float(row[38] or 0.0),
            "duration_minutes": float(row[39] or 0.0),
            "slippage_points": float(row[40] or 0.0),
            "post_trade_review_json": row[41],
            "adjustment_tags_json": row[42],
            "management_effect_json": row[43],
            "session_name": row[44],
            "strategy_key": row[45],
        }

    def stats(
        self,
        current_equity: float | None = None,
        symbol: str | None = None,
        *,
        account: str | None = None,
        magic: int | None = None,
        include_proof: bool = False,
    ) -> TradeStats:
        now_ts = utc_now()
        today_key = current_trading_day_key(now_ts=now_ts)
        live_filter = "1=1" if include_proof else self._live_trade_filter()
        scope_clause, scope_params = self._scope_clause(account=account, magic=magic, symbol=symbol, include_proof=include_proof)
        previous_day = ""
        reset_triggered_at = ""
        reset_reason = ""
        day_start_equity_source = "state:day_start_equity"
        day_high_equity_source = "state:day_high_equity"
        with self._connect() as connection:
            base_scope = self._state_scope(account=account, magic=magic)
            reset_at_key = f"{base_scope}daily_reset_at"
            day_key_key = f"{base_scope}daily_equity_day"
            day_start_key = f"{base_scope}day_start_equity"
            day_high_key = f"{base_scope}day_high_equity"
            day_start_source_key = f"{base_scope}day_start_equity_source"
            day_high_source_key = f"{base_scope}day_high_equity_source"
            soft_dd_count_key = f"{base_scope}soft_dd_trade_count"
            current_equity_value = float(current_equity if current_equity is not None else 0.0)
            self._maybe_reset_loss_streak_state(
                connection,
                now_ts=now_ts,
                account=account,
                magic=magic,
            )
            if symbol:
                self._maybe_reset_loss_streak_state(
                    connection,
                    now_ts=now_ts,
                    account=account,
                    magic=magic,
                    symbol=symbol,
                )
            stored_day = self._get_state(connection, day_key_key, "")
            if stored_day != today_key:
                has_live_equity = current_equity_value > 0.0
                baseline_equity = float(current_equity_value) if has_live_equity else 0.0
                previous_day = str(stored_day or "")
                self._set_state(connection, day_key_key, today_key)
                self._set_state(connection, day_start_key, str(baseline_equity))
                self._set_state(connection, day_high_key, str(baseline_equity))
                self._set_state(connection, day_start_source_key, "reset:current_equity" if has_live_equity else "reset:pending_live_equity")
                self._set_state(connection, day_high_source_key, "reset:current_equity" if has_live_equity else "reset:pending_live_equity")
                self._set_state(connection, soft_dd_count_key, "0")
                self._set_state(connection, reset_at_key, "")
                connection.commit()
                reset_triggered_at = now_ts.isoformat()
                reset_reason = "sydney_day_rollover"
                day_start_equity_source = "reset:current_equity" if has_live_equity else "reset:pending_live_equity"
                day_high_equity_source = "reset:current_equity" if has_live_equity else "reset:pending_live_equity"
                _RUNTIME_LOGGER.info(
                    "sydney_day_rollover_detected",
                    extra={
                        "extra_fields": {
                            "previous_trading_day_key": previous_day,
                            "new_trading_day_key": str(today_key),
                            "reset_closed_trades_today": True,
                            "reset_daily_realized_pnl": True,
                            "reset_day_start_equity": float(baseline_equity),
                            "reset_day_high_equity": float(baseline_equity),
                            "reset_basis_pending_live_equity": bool(not has_live_equity),
                            "timezone_used": str(getattr(SYDNEY, "key", SYDNEY)),
                            "reset_triggered_at": reset_triggered_at,
                            "reset_reason": reset_reason,
                            "day_reset_complete": True,
                        }
                    },
                )
            reset_at = self._get_state(connection, reset_at_key, "").strip()
            reset_filter = ""
            reset_params: list[Any] = []
            if reset_at:
                reset_filter = " AND closed_at >= ?"
                reset_params.append(reset_at)
            closed_scope = connection.execute(
                f"""
                SELECT signal_id, pnl_amount, pnl_r, equity_at_open, equity_after_close, closed_at, symbol, side, setup, session_name
                FROM trade_journal
                WHERE status = 'CLOSED'
                  AND {live_filter}
                  {scope_clause}
                  {reset_filter}
                ORDER BY closed_at ASC
                """,
                tuple(scope_params + reset_params),
            ).fetchall()
            closed_account = connection.execute(
                f"""
                SELECT signal_id, pnl_amount, pnl_r, equity_at_open, equity_after_close, closed_at, symbol, side, setup, session_name
                FROM trade_journal
                WHERE status = 'CLOSED'
                  AND {live_filter}
                  {scope_clause}
                  {reset_filter}
                ORDER BY closed_at ASC
                """,
                tuple(scope_params + reset_params),
            ).fetchall()
            open_positions_count = int(
                connection.execute(
                    f"""
                    SELECT COUNT(*)
                    FROM trade_journal
                    WHERE status = 'EXECUTED'
                      AND {live_filter}
                      {scope_clause}
                    """,
                    tuple(scope_params),
                ).fetchone()[0]
                or 0
            )
            state_scope = self._state_scope(account=account, magic=magic, symbol=symbol)
            cooldown_key = f"{state_scope}cooldown_remaining"
            cooldown_remaining = int(self._get_state(connection, cooldown_key, "0"))
            day_start_equity = float(self._get_state(connection, day_start_key, str(max(1.0, current_equity_value or 1.0))))
            day_high_equity = float(self._get_state(connection, day_high_key, str(day_start_equity)))
            day_start_equity_source = str(self._get_state(connection, day_start_source_key, day_start_equity_source or "state:day_start_equity") or "state:day_start_equity")
            day_high_equity_source = str(self._get_state(connection, day_high_source_key, day_high_equity_source or "state:day_high_equity") or "state:day_high_equity")
            persisted_day_start_equity_source = day_start_equity_source
            persisted_day_high_equity_source = day_high_equity_source
            soft_dd_trade_count = int(self._get_state(connection, soft_dd_count_key, "0"))

        win_rs = [float(row[2]) for row in closed_scope if float(row[2]) > 0]
        loss_rs = [abs(float(row[2])) for row in closed_scope if float(row[2]) < 0]
        win_rate = 0.5
        if closed_scope:
            wins = sum(1 for row in closed_scope if float(row[1]) >= 0)
            win_rate = wins / len(closed_scope)

        equity_points: list[float] = []
        for row in closed_account:
            if row[4]:
                equity_points.append(float(row[4]))
        if current_equity is not None:
            equity_points.append(current_equity)

        rolling_drawdown = 0.0
        absolute_drawdown = 0.0
        if equity_points:
            peak = equity_points[0]
            start = equity_points[0]
            for equity in equity_points:
                peak = max(peak, equity)
                drawdown = 0.0 if peak == 0 else (peak - equity) / peak
                rolling_drawdown = max(rolling_drawdown, drawdown)
                absolute_drawdown = max(absolute_drawdown, 0.0 if start == 0 else (start - equity) / start)

        today_closed_rows = [row for row in closed_scope if self._closed_row_day_key(row[5]) == today_key]
        current_equity_live = max(float(current_equity or 0.0), 0.0)
        day_basis_repaired = False
        if today_closed_rows:
            first_today_open_equity = _safe_float(today_closed_rows[0][3], 0.0)
            if first_today_open_equity > 0:
                day_start_equity = float(first_today_open_equity)
                day_start_equity_source = "today_first_close_equity_at_open"
        else:
            needs_pending_repair = (
                day_start_equity <= 1.0
                or day_high_equity <= 1.0
                or persisted_day_start_equity_source == "reset:pending_live_equity"
                or persisted_day_high_equity_source == "reset:pending_live_equity"
            )
            needs_flat_book_repair = (
                open_positions_count == 0
                and persisted_day_start_equity_source not in {"flat_book_current_equity"}
                and persisted_day_high_equity_source not in {"flat_book_current_equity"}
                and (
                    abs(day_start_equity - current_equity_live) > 1e-6
                    or abs(day_high_equity - current_equity_live) > 1e-6
                )
            )
            needs_open_position_repair = (
                open_positions_count > 0
                and persisted_day_start_equity_source not in {"first_live_equity_after_rollover"}
                and persisted_day_high_equity_source not in {"first_live_equity_after_rollover"}
                and (
                    abs(day_start_equity - current_equity_live) > 1e-6
                    or abs(day_high_equity - current_equity_live) > 1e-6
                )
            )
            needs_basis_repair = needs_pending_repair or needs_flat_book_repair or needs_open_position_repair
        if (not today_closed_rows) and current_equity_live > 0 and needs_basis_repair:
            # If the new Sydney day has no closed trades, the live equity is the
            # only truthful basis available. This also repairs a rollover that
            # happened before a valid live equity snapshot existed.
            if (
                abs(day_start_equity - current_equity_live) > 1e-6
                or abs(day_high_equity - current_equity_live) > 1e-6
            ):
                day_basis_repaired = True
            day_start_equity = float(current_equity_live)
            day_high_equity = float(current_equity_live)
            if open_positions_count == 0:
                day_start_equity_source = "flat_book_current_equity"
                day_high_equity_source = "flat_book_current_equity"
            else:
                day_start_equity_source = "first_live_equity_after_rollover"
                day_high_equity_source = "first_live_equity_after_rollover"
        elif not today_closed_rows:
            if persisted_day_start_equity_source == "flat_book_current_equity":
                day_start_equity_source = "state:day_start_equity"
            if persisted_day_high_equity_source == "flat_book_current_equity":
                day_high_equity_source = "state:day_high_equity"

        today_closed_high_equity = max((_safe_float(row[4], 0.0) for row in today_closed_rows), default=0.0)
        if today_closed_high_equity > 0:
            day_high_equity = max(day_high_equity, day_start_equity, today_closed_high_equity)
            day_high_equity_source = "today_closed_equity_after_max"
        if current_equity_live > 0 and current_equity_live > (day_high_equity + 1e-9):
            day_high_equity = max(day_high_equity, day_start_equity, current_equity_live)
            day_high_equity_source = "live_equity_max"

        baseline = max(day_start_equity or current_equity_live or 1.0, 1.0)
        today_closed_trade_details = [
            {
                "signal_id": str(row[0] or ""),
                "closed_at_raw": str(row[5] or ""),
                "closed_at_sydney": trading_day_time_for_timestamp(row[5], tz=SYDNEY),
                "trading_day_key": trading_day_key_for_timestamp(row[5], tz=SYDNEY),
                "pnl_amount": float(row[1] or 0.0),
                "pnl_r": float(row[2] or 0.0),
                "symbol": str(row[6] or ""),
                "side": str(row[7] or ""),
                "setup": str(row[8] or ""),
                "session_name": str(row[9] or ""),
            }
            for row in today_closed_rows
        ]
        today_pnl_amount = float(sum(float(row[1] or 0.0) for row in today_closed_rows))
        daily_pnl_pct = today_pnl_amount / baseline if baseline else 0.0
        trades_today = int(len(today_closed_rows))
        daily_dd_pct_live = 0.0 if day_high_equity <= 0 else max(0.0, (day_high_equity - current_equity_live) / day_high_equity)
        winning_streak = 0
        for row in reversed(closed_scope):
            if float(row[1]) >= 0:
                winning_streak += 1
                continue
            break
        consecutive_losses_live = 0
        for row in reversed(today_closed_rows):
            if float(row[1]) < 0:
                consecutive_losses_live += 1
                continue
            break

        with self._connect() as connection:
            self._set_state(connection, day_high_key, str(day_high_equity))
            self._set_state(connection, day_start_key, str(day_start_equity))
            self._set_state(connection, day_start_source_key, str(day_start_equity_source))
            self._set_state(connection, day_high_source_key, str(day_high_equity_source))
            connection.commit()

        if day_basis_repaired:
            reset_triggered_at = now_ts.isoformat()
            reset_reason = "flat_book_day_basis_repair"
            _RUNTIME_LOGGER.info(
                "sydney_day_basis_repaired",
                extra={
                    "extra_fields": {
                        "trading_day_key": str(today_key),
                        "timezone_used": str(getattr(SYDNEY, "key", SYDNEY)),
                        "open_positions_count": int(open_positions_count),
                        "today_closed_trade_count": int(len(today_closed_rows)),
                        "repaired_day_start_equity": float(day_start_equity),
                        "repaired_day_high_equity": float(day_high_equity),
                        "repair_reason": reset_reason,
                    }
                },
            )

        return TradeStats(
            win_rate=win_rate,
            avg_win_r=sum(win_rs) / len(win_rs) if win_rs else 1.0,
            avg_loss_r=sum(loss_rs) / len(loss_rs) if loss_rs else 1.0,
            consecutive_losses=consecutive_losses_live,
            winning_streak=winning_streak,
            cooldown_trades_remaining=cooldown_remaining if consecutive_losses_live >= 2 else 0,
            closed_trades_total=len(closed_scope),
            trades_today=trades_today,
            daily_pnl_pct=daily_pnl_pct,
            day_start_equity=day_start_equity,
            day_high_equity=day_high_equity,
            daily_realized_pnl=today_pnl_amount,
            daily_dd_pct_live=daily_dd_pct_live,
            rolling_drawdown_pct=rolling_drawdown,
            absolute_drawdown_pct=absolute_drawdown,
            soft_dd_trade_count=soft_dd_trade_count,
            trading_day_key=str(today_key),
            timezone_used=str(SYDNEY),
            previous_trading_day_key=str(previous_day),
            today_closed_trade_ids=[str(item["signal_id"]) for item in today_closed_trade_details],
            today_closed_trade_count=int(trades_today),
            today_closed_trade_times_raw=[str(item["closed_at_raw"]) for item in today_closed_trade_details],
            today_closed_trade_times_sydney=[str(item["closed_at_sydney"]) for item in today_closed_trade_details],
            today_closed_trade_details=today_closed_trade_details,
            day_bucket_source="close_time_sydney",
            reset_triggered_at=str(reset_triggered_at),
            reset_reason=str(reset_reason),
            day_start_equity_source=str(day_start_equity_source),
            day_high_equity_source=str(day_high_equity_source),
        )

    def closed_trades(
        self,
        limit: int,
        symbol: str | None = None,
        *,
        account: str | None = None,
        magic: int | None = None,
        include_proof: bool = False,
        closed_after: datetime | None = None,
    ) -> list[dict[str, Any]]:
        live_filter = "1=1" if include_proof else self._live_trade_filter()
        scope_clause, scope_params = self._scope_clause(account=account, magic=magic, symbol=symbol, include_proof=include_proof)
        cutoff_clause = ""
        cutoff_params: list[Any] = []
        if closed_after is not None:
            cutoff_dt = closed_after if closed_after.tzinfo is not None else closed_after.replace(tzinfo=timezone.utc)
            cutoff_clause = " AND closed_at >= ?"
            cutoff_params.append(cutoff_dt.astimezone(timezone.utc).isoformat())
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT signal_id, symbol, side, setup, regime, mode, opened_at, closed_at, pnl_amount, pnl_r, probability, expected_value_r
                     , session_name, strategy_key, post_trade_review_json, close_context_json, management_effect_json,
                       duration_minutes, exit_reason, exit_spread_points, slippage_points
                FROM trade_journal
                WHERE status = 'CLOSED'
                  AND {live_filter}
                  {scope_clause}
                  {cutoff_clause}
                ORDER BY closed_at DESC
                LIMIT ?
                """,
                tuple(scope_params + cutoff_params + [max(1, int(limit))]),
            ).fetchall()
        output: list[dict[str, Any]] = []
        for row in rows:
            strategy_key = str(row[13] or "").strip()
            if not strategy_key:
                strategy_key = resolve_strategy_key(str(row[1] or ""), str(row[3] or ""))
            output.append(
                {
                    "signal_id": row[0],
                    "symbol": row[1],
                    "side": row[2],
                    "setup": row[3],
                    "regime": row[4],
                    "mode": row[5],
                    "opened_at": row[6],
                    "closed_at": row[7],
                    "pnl_amount": float(row[8] or 0.0),
                    "pnl_r": float(row[9] or 0.0),
                    "probability": float(row[10] or 0.0),
                    "expected_value_r": float(row[11] or 0.0),
                    "session_name": str(row[12] or ""),
                    "strategy_key": strategy_key,
                    "post_trade_review_json": str(row[14] or "{}"),
                    "close_context_json": str(row[15] or "{}"),
                    "management_effect_json": str(row[16] or "{}"),
                    "duration_minutes": float(row[17] or 0.0),
                    "exit_reason": str(row[18] or ""),
                    "exit_spread_points": float(row[19] or 0.0),
                    "slippage_points": float(row[20] or 0.0),
                }
            )
        return output

    def recent_executions(
        self,
        symbol: str,
        minutes: int = 60,
        *,
        account: str | None = None,
        magic: int | None = None,
        include_proof: bool = False,
    ) -> int:
        cutoff = (utc_now() - timedelta(minutes=minutes)).isoformat()
        live_filter = "1=1" if include_proof else self._live_trade_filter()
        scope_clause, scope_params = self._scope_clause(account=account, magic=magic, symbol=symbol, include_proof=include_proof)
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(*)
                FROM trade_journal
                WHERE {live_filter}
                  {scope_clause}
                  AND status IN ('EXECUTED', 'CLOSED')
                  AND created_at >= ?
                """,
                tuple(scope_params + [cutoff]),
            ).fetchone()
        return int(row[0]) if row else 0

    def adaptive_feedback(
        self,
        symbol: str,
        setup: str,
        limit: int = 40,
        decay: float = 0.92,
        *,
        account: str | None = None,
        magic: int | None = None,
        include_proof: bool = False,
    ) -> dict[str, float]:
        live_filter = "1=1" if include_proof else self._live_trade_filter()
        scope_clause, scope_params = self._scope_clause(account=account, magic=magic, symbol=symbol, include_proof=include_proof)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT pnl_r
                FROM trade_journal
                WHERE status = 'CLOSED'
                  AND {live_filter}
                  {scope_clause}
                  AND setup = ?
                ORDER BY closed_at DESC
                LIMIT ?
                """,
                tuple(scope_params + [setup, limit]),
            ).fetchall()

        if not rows:
            return {
                "samples": 0.0,
                "weighted_win_rate": 0.5,
                "weighted_avg_r": 0.0,
                "recent_loss_streak": 0.0,
            }

        weighted_total = 0.0
        weighted_wins = 0.0
        weighted_r_sum = 0.0
        for index, row in enumerate(rows):
            pnl_r = float(row[0])
            weight = math.pow(decay, index)
            weighted_total += weight
            weighted_r_sum += pnl_r * weight
            if pnl_r >= 0:
                weighted_wins += weight
        recent_loss_streak = 0
        for row in rows:
            pnl_r = float(row[0])
            if pnl_r < 0:
                recent_loss_streak += 1
                continue
            break
        if weighted_total <= 0:
            weighted_total = 1.0
        return {
            "samples": float(len(rows)),
            "weighted_win_rate": weighted_wins / weighted_total,
            "weighted_avg_r": weighted_r_sum / weighted_total,
            "recent_loss_streak": float(recent_loss_streak),
        }

    def last_closed_trade(
        self,
        symbol: str | None = None,
        *,
        account: str | None = None,
        magic: int | None = None,
        include_proof: bool = False,
        closed_after: datetime | None = None,
    ) -> dict[str, Any] | None:
        live_filter = "1=1" if include_proof else self._live_trade_filter()
        scope_clause, scope_params = self._scope_clause(account=account, magic=magic, symbol=symbol, include_proof=include_proof)
        cutoff_clause = ""
        cutoff_params: list[Any] = []
        if closed_after is not None:
            cutoff_dt = closed_after if closed_after.tzinfo is not None else closed_after.replace(tzinfo=timezone.utc)
            cutoff_clause = " AND closed_at >= ?"
            cutoff_params.append(cutoff_dt.astimezone(timezone.utc).isoformat())
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT signal_id, symbol, closed_at, pnl_amount, pnl_r, volume
                FROM trade_journal
                WHERE status = 'CLOSED'
                  AND {live_filter}
                  {scope_clause}
                  {cutoff_clause}
                ORDER BY closed_at DESC
                LIMIT 1
                """,
                tuple(scope_params + cutoff_params),
            ).fetchone()
        if row is None:
            return None
        return {
            "signal_id": row[0],
            "symbol": row[1],
            "closed_at": row[2],
            "pnl_amount": float(row[3] or 0.0),
            "pnl_r": float(row[4] or 0.0),
            "volume": float(row[5] or 0.0),
        }

    def _cooldown_cutoff(
        self,
        now: datetime,
        *,
        account: str | None = None,
        magic: int | None = None,
    ) -> datetime:
        cutoff = self._trading_day_start(now)
        if not (account and magic):
            return cutoff
        base_scope = self._state_scope(account=account, magic=magic)
        if not base_scope:
            return cutoff
        with self._connect() as connection:
            reset_at_raw = self._get_state(connection, f"{base_scope}daily_reset_at", "").strip()
        if not reset_at_raw:
            return cutoff
        try:
            reset_at = datetime.fromisoformat(reset_at_raw.replace("Z", "+00:00"))
        except ValueError:
            return cutoff
        if reset_at.tzinfo is None:
            reset_at = reset_at.replace(tzinfo=timezone.utc)
        return max(cutoff, reset_at.astimezone(timezone.utc))

    def cooldown_block_reason(
        self,
        now: datetime,
        symbol: str,
        cooldown_after_loss_minutes: int,
        cooldown_after_win_minutes: int,
        *,
        account: str | None = None,
        magic: int | None = None,
    ) -> str | None:
        if cooldown_after_loss_minutes <= 0 and cooldown_after_win_minutes <= 0:
            return None
        cooldown_cutoff = self._cooldown_cutoff(now, account=account, magic=magic)
        symbol_last = self.last_closed_trade(symbol=symbol, account=account, magic=magic, closed_after=cooldown_cutoff)
        return self._cooldown_reason(
            symbol_last,
            now,
            "symbol",
            cooldown_after_loss_minutes,
            cooldown_after_win_minutes,
        )

    def record_block(self, reason: str, symbol: str | None = None) -> None:
        normalized_reason = str(reason).strip() or "unknown"
        normalized_symbol = str(symbol or "").upper()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO trade_blocks (created_at, symbol, reason)
                VALUES (?, ?, ?)
                """,
                (utc_now().isoformat(), normalized_symbol, normalized_reason),
            )
            connection.commit()

    def blocked_counts(self, since_hours: int | None = None) -> dict[str, Any]:
        params: tuple[Any, ...] = ()
        clause = ""
        if since_hours is not None and since_hours > 0:
            cutoff = (utc_now() - timedelta(hours=int(since_hours))).isoformat()
            clause = "WHERE created_at >= ?"
            params = (cutoff,)
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT reason, COUNT(*)
                FROM trade_blocks
                {clause}
                GROUP BY reason
                ORDER BY COUNT(*) DESC
                """,
                params,
            ).fetchall()
        by_reason = {str(row[0]): int(row[1]) for row in rows}
        return {
            "since_hours": int(since_hours) if since_hours is not None else None,
            "total": int(sum(by_reason.values())),
            "by_reason": by_reason,
        }

    def micro_survivability_summary(self, limit: int = 20, fee_per_lot_estimate: float = 7.0) -> dict[str, float]:
        live_filter = self._live_trade_filter()
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT closed_at, pnl_amount, pnl_r, volume
                FROM trade_journal
                WHERE status = 'CLOSED'
                  AND {live_filter}
                ORDER BY closed_at DESC
                LIMIT ?
                """,
                (max(1, int(limit)),),
            ).fetchall()
        if not rows:
            return {
                "trades": 0.0,
                "max_consecutive_losses": 0.0,
                "max_intraday_dd_usd": 0.0,
                "total_fees_estimate_usd": 0.0,
            }

        ordered = list(reversed(rows))
        max_consecutive_losses = 0
        current_losses = 0
        max_intraday_dd = 0.0
        daily_state: dict[str, dict[str, float]] = {}
        total_fees = 0.0
        for row in ordered:
            closed_at = str(row[0] or "")
            pnl_amount = float(row[1] or 0.0)
            volume = abs(float(row[3] or 0.0))
            total_fees += volume * max(0.0, float(fee_per_lot_estimate))
            if pnl_amount < 0:
                current_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_losses = 0

            day = closed_at[:10] if len(closed_at) >= 10 else "unknown"
            state = daily_state.setdefault(day, {"running": 0.0, "peak": 0.0, "max_dd": 0.0})
            state["running"] += pnl_amount
            state["peak"] = max(state["peak"], state["running"])
            dd = state["peak"] - state["running"]
            state["max_dd"] = max(state["max_dd"], dd)
            max_intraday_dd = max(max_intraday_dd, state["max_dd"])

        return {
            "trades": float(len(ordered)),
            "max_consecutive_losses": float(max_consecutive_losses),
            "max_intraday_dd_usd": float(max_intraday_dd),
            "total_fees_estimate_usd": float(total_fees),
        }

    def summary_last(self, limit: int, symbol: str | None = None) -> dict[str, float]:
        return self.summary_scope(limit=limit, symbol=symbol)

    def summary_scope(
        self,
        limit: int,
        symbol: str | None = None,
        *,
        account: str | None = None,
        magic: int | None = None,
        include_proof: bool = False,
        closed_after: datetime | None = None,
    ) -> dict[str, float]:
        live_filter = "1=1" if include_proof else self._live_trade_filter()
        scope_clause, scope_params = self._scope_clause(account=account, magic=magic, symbol=symbol, include_proof=include_proof)
        cutoff_clause = ""
        cutoff_params: list[Any] = []
        if closed_after is not None:
            cutoff_dt = closed_after if closed_after.tzinfo is not None else closed_after.replace(tzinfo=timezone.utc)
            cutoff_clause = " AND closed_at >= ?"
            cutoff_params.append(cutoff_dt.astimezone(timezone.utc).isoformat())
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT pnl_r, pnl_amount
                FROM trade_journal
                WHERE status = 'CLOSED'
                  AND {live_filter}
                  {scope_clause}
                  {cutoff_clause}
                ORDER BY closed_at DESC
                LIMIT ?
                """,
                tuple(scope_params + cutoff_params + [max(1, int(limit))]),
            ).fetchall()
        if not rows:
            return {
                "trades": 0.0,
                "win_rate": 0.0,
                "expectancy_r": 0.0,
                "profit_factor": 0.0,
                "max_drawdown_r": 0.0,
            }
        ordered = list(reversed(rows))
        pnl_r = [float(row[0] or 0.0) for row in ordered]
        if any(abs(value) > 1e-9 for value in pnl_r):
            pnl = pnl_r
        else:
            pnl_amount = [float(row[1] or 0.0) for row in ordered]
            pnl = pnl_amount if any(abs(value) > 1e-9 for value in pnl_amount) else pnl_r
        wins = [value for value in pnl if value > 0]
        losses = [abs(value) for value in pnl if value < 0]
        win_rate = sum(1 for value in pnl if value >= 0) / len(pnl)
        expectancy = sum(pnl) / len(pnl)
        gross_profit = sum(wins)
        gross_loss = sum(losses)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
        equity = []
        running = 0.0
        for value in pnl:
            running += value
            equity.append(running)
        peak = equity[0]
        max_drawdown = 0.0
        for value in equity:
            peak = max(peak, value)
            max_drawdown = max(max_drawdown, peak - value)
        return {
            "trades": float(len(pnl)),
            "win_rate": float(win_rate),
            "expectancy_r": float(expectancy),
            "profit_factor": float(profit_factor),
            "max_drawdown_r": float(max_drawdown),
        }

    @staticmethod
    def _cooldown_reason(
        last_closed: dict[str, Any] | None,
        now: datetime,
        scope: str,
        cooldown_after_loss_minutes: int,
        cooldown_after_win_minutes: int,
    ) -> str | None:
        if not last_closed:
            return None
        closed_at_raw = str(last_closed.get("closed_at", ""))
        if not closed_at_raw:
            return None
        try:
            closed_at = datetime.fromisoformat(closed_at_raw.replace("Z", "+00:00"))
        except ValueError:
            return None
        if closed_at.tzinfo is None:
            closed_at = closed_at.replace(tzinfo=timezone.utc)
        now_utc = now.astimezone(timezone.utc) if now.tzinfo is not None else now.replace(tzinfo=timezone.utc)
        elapsed_minutes = max(0.0, (now_utc - closed_at.astimezone(timezone.utc)).total_seconds() / 60.0)
        pnl_amount = float(last_closed.get("pnl_amount", 0.0))
        if pnl_amount < 0 and cooldown_after_loss_minutes > 0 and elapsed_minutes < cooldown_after_loss_minutes:
            remaining = int(max(1, math.ceil(cooldown_after_loss_minutes - elapsed_minutes)))
            return f"{scope}_cooldown_after_loss_{remaining}m"
        if pnl_amount > 0 and cooldown_after_win_minutes > 0 and elapsed_minutes < cooldown_after_win_minutes:
            remaining = int(max(1, math.ceil(cooldown_after_win_minutes - elapsed_minutes)))
            return f"{scope}_cooldown_after_win_{remaining}m"
        return None

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        connection.execute("PRAGMA busy_timeout=10000")
        try:
            connection.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError as exc:
            if not _sqlite_low_space_error(exc):
                raise
            _RUNTIME_LOGGER.warning(
                "trade_journal_sqlite_wal_fallback db=%s reason=%s",
                self.db_path,
                exc,
            )
            try:
                connection.execute("PRAGMA journal_mode=DELETE")
            except sqlite3.OperationalError:
                pass
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_journal (
                    signal_id TEXT PRIMARY KEY,
                    ticket TEXT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    setup TEXT NOT NULL,
                    mode TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    opened_at TEXT,
                    closed_at TEXT,
                    entry_price REAL,
                    sl REAL,
                    tp REAL,
                    volume REAL,
                    probability REAL,
                    expected_value_r REAL,
                    regime TEXT,
                    strategy_key TEXT DEFAULT '',
                    trailing_enabled INTEGER DEFAULT 1,
                    partial_close_enabled INTEGER DEFAULT 0,
                    news_status TEXT DEFAULT '',
                    final_decision TEXT DEFAULT '',
                    pnl_amount REAL DEFAULT 0,
                    pnl_r REAL DEFAULT 0,
                    equity_at_open REAL DEFAULT 0,
                    equity_after_close REAL DEFAULT 0,
                    account TEXT DEFAULT '',
                    magic INTEGER DEFAULT 0,
                    timeframe TEXT DEFAULT '',
                    proof_trade INTEGER DEFAULT 0,
                    entry_reason TEXT DEFAULT '',
                    ai_summary_json TEXT DEFAULT '{}',
                    broker_snapshot_json TEXT DEFAULT '{}',
                    entry_context_json TEXT DEFAULT '{}',
                    close_context_json TEXT DEFAULT '{}',
                    account_currency TEXT DEFAULT '',
                    entry_spread_points REAL DEFAULT 0,
                    exit_spread_points REAL DEFAULT 0,
                    exit_reason TEXT DEFAULT '',
                    exit_price REAL DEFAULT 0,
                    duration_minutes REAL DEFAULT 0,
                    slippage_points REAL DEFAULT 0,
                    post_trade_review_json TEXT DEFAULT '{}',
                    adjustment_tags_json TEXT DEFAULT '[]',
                    management_effect_json TEXT DEFAULT '{}',
                    session_name TEXT DEFAULT ''
                )
                """
            )
            self._ensure_column(connection, "trade_journal", "trailing_enabled", "INTEGER DEFAULT 1")
            self._ensure_column(connection, "trade_journal", "partial_close_enabled", "INTEGER DEFAULT 0")
            self._ensure_column(connection, "trade_journal", "news_status", "TEXT DEFAULT ''")
            self._ensure_column(connection, "trade_journal", "final_decision", "TEXT DEFAULT ''")
            self._ensure_column(connection, "trade_journal", "strategy_key", "TEXT DEFAULT ''")
            self._ensure_column(connection, "trade_journal", "account", "TEXT DEFAULT ''")
            self._ensure_column(connection, "trade_journal", "magic", "INTEGER DEFAULT 0")
            self._ensure_column(connection, "trade_journal", "timeframe", "TEXT DEFAULT ''")
            self._ensure_column(connection, "trade_journal", "proof_trade", "INTEGER DEFAULT 0")
            self._ensure_column(connection, "trade_journal", "entry_reason", "TEXT DEFAULT ''")
            self._ensure_column(connection, "trade_journal", "ai_summary_json", "TEXT DEFAULT '{}'")
            self._ensure_column(connection, "trade_journal", "broker_snapshot_json", "TEXT DEFAULT '{}'")
            self._ensure_column(connection, "trade_journal", "entry_context_json", "TEXT DEFAULT '{}'")
            self._ensure_column(connection, "trade_journal", "close_context_json", "TEXT DEFAULT '{}'")
            self._ensure_column(connection, "trade_journal", "account_currency", "TEXT DEFAULT ''")
            self._ensure_column(connection, "trade_journal", "entry_spread_points", "REAL DEFAULT 0")
            self._ensure_column(connection, "trade_journal", "exit_spread_points", "REAL DEFAULT 0")
            self._ensure_column(connection, "trade_journal", "exit_reason", "TEXT DEFAULT ''")
            self._ensure_column(connection, "trade_journal", "exit_price", "REAL DEFAULT 0")
            self._ensure_column(connection, "trade_journal", "duration_minutes", "REAL DEFAULT 0")
            self._ensure_column(connection, "trade_journal", "slippage_points", "REAL DEFAULT 0")
            self._ensure_column(connection, "trade_journal", "post_trade_review_json", "TEXT DEFAULT '{}'")
            self._ensure_column(connection, "trade_journal", "adjustment_tags_json", "TEXT DEFAULT '[]'")
            self._ensure_column(connection, "trade_journal", "management_effect_json", "TEXT DEFAULT '{}'")
            self._ensure_column(connection, "trade_journal", "session_name", "TEXT DEFAULT ''")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS journal_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS trade_blocks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    reason TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _get_state(self, connection: sqlite3.Connection, key: str, default: str) -> str:
        row = connection.execute("SELECT value FROM journal_state WHERE key = ?", (key,)).fetchone()
        return str(row[0]) if row else default

    def _set_state(self, connection: sqlite3.Connection, key: str, value: str) -> None:
        connection.execute(
            "INSERT OR REPLACE INTO journal_state (key, value) VALUES (?, ?)",
            (key, value),
        )

    def _consume_cooldown(self, connection: sqlite3.Connection, key: str) -> None:
        remaining = int(self._get_state(connection, key, "0"))
        if remaining > 0:
            self._set_state(connection, key, str(max(0, remaining - 1)))

    def _update_loss_state(
        self,
        connection: sqlite3.Connection,
        loss_key: str,
        cooldown_key: str,
        pnl_amount: float,
        *,
        trading_day_key: str,
    ) -> None:
        day_key = f"{loss_key}_trading_day"
        self._set_state(connection, day_key, str(trading_day_key))
        if pnl_amount >= 0:
            self._set_state(connection, loss_key, "0")
            self._set_state(connection, cooldown_key, "0")
            return
        consecutive_losses = int(self._get_state(connection, loss_key, "0")) + 1
        self._set_state(connection, loss_key, str(consecutive_losses))
        if consecutive_losses >= 2:
            self._set_state(connection, cooldown_key, "2")

    def _latest_closed_day_key_for_scope(
        self,
        connection: sqlite3.Connection,
        *,
        account: str | None = None,
        magic: int | None = None,
        symbol: str | None = None,
    ) -> str:
        scope_clause, scope_params = self._scope_clause(account=account, magic=magic, symbol=symbol, include_proof=False)
        row = connection.execute(
            f"""
            SELECT closed_at
            FROM trade_journal
            WHERE status IN ('CLOSED', 'RECONCILED_CLOSED')
              AND {self._live_trade_filter()}
              {scope_clause}
            ORDER BY closed_at DESC
            LIMIT 1
            """,
            tuple(scope_params),
        ).fetchone()
        if row is None or not row[0]:
            return ""
        return self._closed_row_day_key(row[0])

    def _maybe_reset_loss_streak_state(
        self,
        connection: sqlite3.Connection,
        *,
        now_ts: datetime,
        account: str | None = None,
        magic: int | None = None,
        symbol: str | None = None,
    ) -> bool:
        scope = self._state_scope(account=account, magic=magic, symbol=symbol)
        if not scope:
            return False
        current_day = self._trading_day_key(now_ts)
        day_key = f"{scope}consecutive_losses_trading_day"
        loss_key = f"{scope}consecutive_losses"
        cooldown_key = f"{scope}cooldown_remaining"
        previous_day = self._get_state(connection, day_key, "").strip()
        if not previous_day:
            previous_day = self._latest_closed_day_key_for_scope(
                connection,
                account=account,
                magic=magic,
                symbol=symbol,
            )
        streak_before = int(self._get_state(connection, loss_key, "0"))
        cooldown_before = int(self._get_state(connection, cooldown_key, "0"))
        if previous_day == current_day:
            if not self._get_state(connection, day_key, "").strip():
                self._set_state(connection, day_key, current_day)
            return False
        if streak_before <= 0 and cooldown_before <= 0:
            if current_day and self._get_state(connection, day_key, "").strip() != current_day:
                self._set_state(connection, day_key, current_day)
            return False
        self._set_state(connection, loss_key, "0")
        self._set_state(connection, cooldown_key, "0")
        self._set_state(connection, day_key, current_day)
        _RUNTIME_LOGGER.info(
            "loss_streak_day_reset",
            extra={
                "extra_fields": {
                    "account": str(account or ""),
                    "magic": int(magic or 0),
                    "symbol": str(symbol or "").upper(),
                    "previous_trading_day": str(previous_day or ""),
                    "current_trading_day": str(current_day),
                    "streak_before_reset": int(streak_before),
                    "streak_after_reset": 0,
                    "cooldown_before_reset": int(cooldown_before),
                    "cooldown_after_reset": 0,
                    "timezone_used_for_reset": str(getattr(SYDNEY, "key", SYDNEY)),
                }
            },
        )
        return True

    def _get_trade_symbol(self, connection: sqlite3.Connection, signal_id: str) -> str | None:
        row = connection.execute("SELECT symbol FROM trade_journal WHERE signal_id = ?", (signal_id,)).fetchone()
        return str(row[0]) if row and row[0] else None

    def _get_trade_scope(self, connection: sqlite3.Connection, signal_id: str) -> dict[str, Any]:
        row = connection.execute(
            "SELECT account, magic FROM trade_journal WHERE signal_id = ?",
            (signal_id,),
        ).fetchone()
        if row is None:
            return {"account": "", "magic": 0}
        return {
            "account": str(row[0] or ""),
            "magic": int(row[1] or 0),
        }

    def _is_proof_trade(self, connection: sqlite3.Connection, signal_id: str) -> bool:
        row = connection.execute(
            "SELECT proof_trade, signal_id, setup FROM trade_journal WHERE signal_id = ?",
            (signal_id,),
        ).fetchone()
        if row is None:
            return False
        proof_flag = bool(row[0])
        signal_value = str(row[1] or "")
        setup_value = str(row[2] or "")
        return proof_flag or signal_value.startswith(("FORCE_TEST::", "CANARY::")) or "FORCE_TEST" in setup_value or "CANARY" in setup_value

    @staticmethod
    def _symbol_state_key(prefix: str, symbol: str) -> str:
        return f"{prefix}__{symbol.upper()}"

    @classmethod
    def _state_scope(cls, *, account: str | None = None, magic: int | None = None, symbol: str | None = None) -> str:
        scope_parts: list[str] = []
        if account is not None and str(account).strip():
            scope_parts.append(str(account).strip())
        if magic is not None:
            scope_parts.append(str(int(magic)))
        if symbol is not None and str(symbol).strip():
            scope_parts.append(str(symbol).upper())
        if not scope_parts:
            return ""
        return "__".join(scope_parts) + "__"

    @staticmethod
    def _ensure_column(connection: sqlite3.Connection, table: str, column: str, definition: str) -> None:
        columns = connection.execute(f"PRAGMA table_info({table})").fetchall()
        if any(str(item[1]) == column for item in columns):
            return
        connection.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


@dataclass
class ExecutionService:
    mt5_client: MT5Client
    journal: TradeJournal
    logger: ApexLogger

    def place(self, request: ExecutionRequest, equity: float) -> ExecutionReceipt:
        if self.journal.has_signal(request.signal_id):
            return ExecutionReceipt(False, "duplicate_signal")
        # LIVE is allowed here only after upstream guards have enabled trading.
        # run_bot() and RiskEngine enforce explicit LIVE unlock requirements.
        allowed_modes = {"DEMO", "PAPER", "LIVE"}
        if not request.trading_enabled:
            return ExecutionReceipt(False, "trading_disabled")
        if request.mode.upper() not in allowed_modes:
            return ExecutionReceipt(False, f"mode_not_tradeable:{request.mode.upper()}")
        if hasattr(self.mt5_client, "disable_mt5") and getattr(self.mt5_client, "disable_mt5"):
            return ExecutionReceipt(False, "mt5_disabled")
        if hasattr(self.mt5_client, "connect") and not getattr(self.mt5_client, "connected", False):
            try:
                connected = self.mt5_client.connect()
            except Exception:
                connected = False
            if not connected:
                return ExecutionReceipt(False, "mt5_not_connected")
        result = self.mt5_client.order_send(
            symbol=request.symbol,
            side=request.side,
            volume=request.volume,
            price=request.entry_price,
            sl=request.stop_price,
            tp=request.take_profit_price,
            slippage_points=request.slippage_points,
            magic=request.magic,
            comment=request.signal_id,
        )
        self.journal.record_execution(request, result, equity)
        self.logger.info(
            "order_processed",
            extra={"extra_fields": {"signal_id": request.signal_id, "symbol": request.symbol, "accepted": result.accepted, "reason": result.reason}},
        )
        return ExecutionReceipt(result.accepted, result.reason, result.order_id, result.raw)

    def move_protection(self, ticket: int, signal_id: str, sl: float | None = None, tp: float | None = None) -> bool:
        success = self.mt5_client.modify_position(ticket, sl=sl, tp=tp)
        event = "MOVE_PROTECTION_OK" if success else "MOVE_PROTECTION_FAILED"
        self.journal.log_event(signal_id, event, {"ticket": ticket, "sl": sl, "tp": tp})
        return success

    def partial_close(self, position: dict[str, Any], signal_id: str, volume_to_close: float, slippage_points: int) -> bool:
        success = self.mt5_client.reduce_position(position, volume_to_close, slippage_points)
        event = "PARTIAL_CLOSE_OK" if success else "PARTIAL_CLOSE_FAILED"
        self.journal.log_event(signal_id, event, {"ticket": position.get("ticket"), "volume": volume_to_close})
        if success:
            remaining_volume = max(0.0, float(position.get("volume", 0.0)) - volume_to_close)
            self.journal.update_open_volume(signal_id, round(remaining_volume, 2))
        return success

    def hard_flatten(self, symbol: str | None, slippage_points: int) -> int:
        count = 0
        for position in self.mt5_client.positions(symbol):
            if self.mt5_client.close_position(position, slippage_points):
                count += 1
        return count
