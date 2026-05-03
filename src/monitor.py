from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
import json
from zoneinfo import ZoneInfo

from src.logger import ApexLogger
from src.utils import ensure_parent, utc_now

SYDNEY = ZoneInfo("Australia/Sydney")


def _sydney_day_key(now_ts: datetime | None = None) -> str:
    current = now_ts or utc_now()
    if current.tzinfo is None:
        current = current.replace(tzinfo=utc_now().tzinfo)
    return current.astimezone(SYDNEY).date().isoformat()


@dataclass
class KillStatus:
    level: str | None
    reason: str | None
    created_at: str | None = None
    session_key: str | None = None
    expires_at: str | None = None
    auto_clear_reason: str | None = None
    last_equity: float | None = None
    recovery_mode: str | None = None
    recovery_started_at: str | None = None
    recovery_wins_needed: int = 0
    recovery_wins_observed: int = 0


@dataclass
class KillSwitch:
    lock_path: Path

    @staticmethod
    def _parse_iso(value: Any) -> datetime | None:
        raw = str(value or "").strip()
        if not raw:
            return None
        try:
            parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except Exception:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=utc_now().tzinfo)
        return parsed

    def _load_payload(self) -> dict[str, Any]:
        if not self.lock_path.exists():
            return {}
        try:
            payload = json.loads(self.lock_path.read_text())
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {"level": "SOFT", "reason": "invalid_lock_file"}

    def status(
        self,
        *,
        now: datetime | None = None,
        equity: float | None = None,
        current_session_key: str | None = None,
        hard_ttl_hours: float = 6.0,
        sydney_reset_enabled: bool = True,
    ) -> KillStatus:
        payload = self._load_payload()
        if not payload:
            return KillStatus(None, None, None)
        now_ts = now or utc_now()
        created_at = str(payload.get("created_at") or "") or None
        level = str(payload.get("level") or "").strip().upper() or None
        reason = str(payload.get("reason") or "").strip() or None
        session_key = str(payload.get("session_key") or _sydney_day_key(now_ts)).strip() or None
        expires_at = str(payload.get("expires_at") or "").strip() or None
        recovery_mode = str(payload.get("recovery_mode") or "").strip() or None
        recovery_started_at = str(payload.get("recovery_started_at") or "").strip() or None
        recovery_wins_needed = max(0, int(payload.get("recovery_wins_needed", 0) or 0))
        recovery_wins_observed = max(0, int(payload.get("recovery_wins_observed", 0) or 0))
        last_equity = payload.get("last_equity")
        try:
            last_equity_value = float(last_equity) if last_equity is not None else (float(equity) if equity is not None else None)
        except (TypeError, ValueError):
            last_equity_value = float(equity) if equity is not None else None
        current_key = str(current_session_key or _sydney_day_key(now_ts)).strip() or None
        auto_clear_reason: str | None = None
        if level == "HARD":
            created_ts = self._parse_iso(created_at)
            expiry_ts = self._parse_iso(expires_at)
            if expiry_ts is None and created_ts is not None:
                expiry_ts = created_ts + timedelta(hours=max(0.5, float(hard_ttl_hours)))
                expires_at = expiry_ts.isoformat()
            if sydney_reset_enabled and session_key and current_key and session_key != current_key:
                auto_clear_reason = f"sydney_reset:{session_key}->{current_key}"
                level = None
                recovery_mode = recovery_mode or "RECOVERY_DEFENSIVE"
                recovery_started_at = recovery_started_at or now_ts.isoformat()
            elif expiry_ts is not None and now_ts >= expiry_ts:
                auto_clear_reason = "hard_kill_ttl_elapsed"
                level = None
                recovery_mode = recovery_mode or "RECOVERY_DEFENSIVE"
                recovery_started_at = recovery_started_at or now_ts.isoformat()
        return KillStatus(
            level=level,
            reason=reason,
            created_at=created_at,
            session_key=session_key,
            expires_at=expires_at,
            auto_clear_reason=auto_clear_reason,
            last_equity=last_equity_value,
            recovery_mode=recovery_mode,
            recovery_started_at=recovery_started_at,
            recovery_wins_needed=recovery_wins_needed,
            recovery_wins_observed=recovery_wins_observed,
        )

    def activate(
        self,
        level: str,
        reason: str,
        *,
        now: datetime | None = None,
        session_key: str | None = None,
        last_equity: float | None = None,
        hard_ttl_hours: float = 6.0,
        recovery_mode: str = "RECOVERY_DEFENSIVE",
        recovery_wins_needed: int = 3,
    ) -> None:
        now_ts = now or utc_now()
        level_key = str(level or "").upper()
        session_value = str(session_key or _sydney_day_key(now_ts))
        payload: dict[str, Any] = {
            "level": level_key,
            "reason": str(reason or ""),
            "created_at": now_ts.isoformat(),
            "session_key": session_value,
            "expires_at": (
                (now_ts + timedelta(hours=max(0.5, float(hard_ttl_hours)))).isoformat()
                if level_key == "HARD"
                else ""
            ),
            "auto_clear_reason": "",
            "last_equity": float(last_equity) if last_equity is not None else None,
            "recovery_mode": recovery_mode if level_key == "HARD" else "",
            "recovery_started_at": "",
            "recovery_wins_needed": int(max(0, recovery_wins_needed)),
            "recovery_wins_observed": 0,
        }
        ensure_parent(self.lock_path)
        self.lock_path.write_text(json.dumps(payload))

    def enter_recovery(
        self,
        *,
        reason: str,
        now: datetime | None = None,
        session_key: str | None = None,
        last_equity: float | None = None,
        auto_clear_reason: str = "",
        recovery_mode: str = "RECOVERY_DEFENSIVE",
        recovery_wins_needed: int = 3,
    ) -> None:
        now_ts = now or utc_now()
        payload = {
            "level": "",
            "reason": str(reason or ""),
            "created_at": now_ts.isoformat(),
            "session_key": str(session_key or _sydney_day_key(now_ts)),
            "expires_at": "",
            "auto_clear_reason": str(auto_clear_reason or ""),
            "last_equity": float(last_equity) if last_equity is not None else None,
            "recovery_mode": str(recovery_mode or "RECOVERY_DEFENSIVE"),
            "recovery_started_at": now_ts.isoformat(),
            "recovery_wins_needed": int(max(1, recovery_wins_needed)),
            "recovery_wins_observed": 0,
        }
        ensure_parent(self.lock_path)
        self.lock_path.write_text(json.dumps(payload))

    def update_recovery_progress(self, *, wins_observed: int) -> None:
        payload = self._load_payload()
        if not payload:
            return
        payload["recovery_wins_observed"] = int(max(0, wins_observed))
        ensure_parent(self.lock_path)
        self.lock_path.write_text(json.dumps(payload))

    def clear(self) -> None:
        if self.lock_path.exists():
            self.lock_path.unlink()


@dataclass
class SymbolStatus:
    symbol: str
    regime: str
    news: str
    trading_allowed: bool
    reason: str
    open_positions: int
    max_positions: int
    last_signal: str
    last_score: float
    queued_actions: int = 0
    ai_reason: str = ""
    grid_cycle_state: str = ""
    grid_leg: str = ""
    grid_cycle_id: str = ""
    grid_last_entry: str = ""
    current_state: str = ""
    engine: str = ""
    market_open_status: str = ""
    eligible_session: str = ""
    session_allowed: str = ""
    requested_timeframe: str = ""
    execution_timeframe_used: str = ""
    internal_timeframes_used: list[str] = field(default_factory=list)
    attachment_dependency_resolved: bool = False
    pre_exec_status: str = ""
    pre_exec_reason: str = ""
    last_block_reason: str = ""
    last_ai_mode: str = ""
    delivered_actions: int = 0
    stale_archives: int = 0
    market_closed_blocks: int = 0
    runtime_market_data_mode: str = ""
    runtime_market_data_source: str = ""
    runtime_market_data_consensus_state: str = ""
    runtime_market_data_ready: str = ""
    runtime_market_data_error: str = ""


@dataclass
class DashboardState:
    mode: str
    current_utc: datetime
    current_local: datetime
    session_status: str
    active_session_name: str
    equity: float
    daily_pnl_pct: float
    drawdown_pct: float
    total_open_positions: int
    max_total_positions: int
    ai_active: bool
    win_rate_last_100: float
    win_rate_last_200: float
    queued_actions_total: int
    symbols: list[SymbolStatus]
    next_check_seconds: int
    rollout_win_rate: float = 0.0
    rollout_trade_count: int = 0
    account_label: str = "MT5_FEED"


@dataclass
class Monitor:
    logger: ApexLogger
    alert_log_path: Path
    print_dashboard: bool = True

    def dashboard(self, state: DashboardState) -> None:
        if not self.print_dashboard:
            return
        if str(state.session_status).upper().startswith("WEEKEND"):
            session_line = (
                f"UTC: {state.current_utc.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Local: {state.current_local.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
                f"Session: {state.session_status}"
            )
        else:
            session_line = (
                f"UTC: {state.current_utc.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Local: {state.current_local.strftime('%Y-%m-%d %H:%M:%S %Z')} | "
                f"Session: {state.session_status} ({state.active_session_name})"
            )
        lines = [
            f"[APEX | MULTI-SYMBOL | {state.mode}]",
            session_line,
            f"Equity[{state.account_label}]: ${state.equity:,.2f} | Daily: {state.daily_pnl_pct * 100:+.2f}% | DD: {state.drawdown_pct * 100:.2f}%",
            f"Open: {state.total_open_positions}/{state.max_total_positions} | Queued: {state.queued_actions_total} | AI: {'ACTIVE' if state.ai_active else 'OFF'} | Next Check: {state.next_check_seconds}s",
        ]
        if int(state.rollout_trade_count) > 0:
            lines.append(
                f"WinRate Run: {state.rollout_win_rate * 100:.1f}% ({int(state.rollout_trade_count)}) | "
                f"WinRate L100: {state.win_rate_last_100 * 100:.1f}% | WinRate L200: {state.win_rate_last_200 * 100:.1f}%"
            )
        else:
            lines.append(
                f"WinRate L100: {state.win_rate_last_100 * 100:.1f}% | WinRate L200: {state.win_rate_last_200 * 100:.1f}%"
            )
        for symbol_state in state.symbols:
            state_label = symbol_state.current_state or ("ALLOW" if symbol_state.trading_allowed else "BLOCK")
            engine_label = symbol_state.engine or symbol_state.last_signal or "NONE"
            tf_suffix = ""
            if symbol_state.execution_timeframe_used:
                requested_tf = symbol_state.requested_timeframe or symbol_state.execution_timeframe_used
                internal = "/".join(symbol_state.internal_timeframes_used) if symbol_state.internal_timeframes_used else symbol_state.execution_timeframe_used
                tf_suffix = f" | TF {requested_tf}->{symbol_state.execution_timeframe_used} | Int {internal}"
            precheck_suffix = ""
            if symbol_state.pre_exec_status:
                precheck_suffix = f" | Pre {symbol_state.pre_exec_status}"
                if symbol_state.pre_exec_reason:
                    precheck_suffix += f"({symbol_state.pre_exec_reason})"
            market_suffix = ""
            if symbol_state.market_open_status or symbol_state.eligible_session:
                market_bits: list[str] = []
                if symbol_state.market_open_status:
                    market_bits.append(f"Mkt {symbol_state.market_open_status}")
                if symbol_state.eligible_session:
                    market_bits.append(f"Sess {symbol_state.eligible_session}")
                if symbol_state.session_allowed:
                    market_bits.append(f"Allowed {symbol_state.session_allowed}")
                market_suffix = " | " + " | ".join(market_bits)
            block_suffix = ""
            if symbol_state.last_block_reason:
                block_suffix = f" | Block {symbol_state.last_block_reason}"
            data_suffix = ""
            if symbol_state.runtime_market_data_mode:
                data_suffix = f" | Data {symbol_state.runtime_market_data_mode}"
                if symbol_state.runtime_market_data_source:
                    data_suffix += f"({symbol_state.runtime_market_data_source})"
                if symbol_state.runtime_market_data_consensus_state:
                    data_suffix += f" {symbol_state.runtime_market_data_consensus_state}"
                if symbol_state.runtime_market_data_ready:
                    data_suffix += f" Ready {symbol_state.runtime_market_data_ready}"
                if symbol_state.runtime_market_data_error:
                    data_suffix += f" Err {symbol_state.runtime_market_data_error}"
            grid_suffix = ""
            if symbol_state.grid_cycle_state:
                parts = [f"GRID {symbol_state.grid_cycle_state}"]
                if symbol_state.grid_leg:
                    parts.append(f"Leg {symbol_state.grid_leg}")
                if symbol_state.grid_cycle_id:
                    parts.append(f"Cycle {symbol_state.grid_cycle_id}")
                if symbol_state.grid_last_entry:
                    parts.append(f"Entry {symbol_state.grid_last_entry}")
                grid_suffix = " | " + " | ".join(parts)
            lines.append(
                f"{symbol_state.symbol} | {state_label} | Engine {engine_label} | Regime {symbol_state.regime} | News {symbol_state.news} | "
                f"Open {symbol_state.open_positions}/{symbol_state.max_positions} | Queued {symbol_state.queued_actions} | Delivered {symbol_state.delivered_actions} | "
                f"{symbol_state.last_signal} @ {symbol_state.last_score:.2f} | {symbol_state.reason} | AI {symbol_state.ai_reason}"
                f"{market_suffix}{precheck_suffix}{block_suffix}{tf_suffix}{data_suffix}{grid_suffix}"
            )
        for line in lines:
            print(line)

    def alert(self, event_type: str, message: str, **fields) -> None:
        payload = {"timestamp": utc_now().isoformat(), "event_type": event_type, "message": message, **fields}
        ensure_parent(self.alert_log_path)
        with self.alert_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        self.logger.warning(message, extra={"extra_fields": payload})
