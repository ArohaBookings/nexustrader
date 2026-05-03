from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping
import json
import math
import os

UTC = timezone.utc


@dataclass(frozen=True)
class LiveEvidence:
    trade_count: int = 0
    win_rate: float = 0.0
    expectancy_r: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_r: float = 0.0


@dataclass(frozen=True)
class AggressionDecision:
    allowed: bool
    reason: str
    tier: str
    cap: int
    used: int
    remaining: int


@dataclass(frozen=True)
class AggressionConfig:
    enabled: bool = False
    owner_unlock_required: bool = True
    bucket_minutes: int = 120
    base_live_cap: int = 5
    proven_live_cap: int = 10
    full_live_cap_bootstrap: int = 20
    full_live_cap_growth: int = 30
    full_live_cap_growth_hot: int = 40
    bootstrap_equity_threshold: float = 160.0
    growth_equity_threshold: float = 300.0
    min_trades_for_proven: int = 10
    min_trades_for_full: int = 20
    min_win_rate: float = 0.50
    min_expectancy_r: float = 0.0
    min_full_expectancy_r: float = 0.10
    state_file: str = "data/aggression_controller_state.json"

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "AggressionConfig":
        data = dict(raw or {})
        return cls(
            enabled=bool(data.get("enabled", False)),
            owner_unlock_required=bool(data.get("owner_unlock_required", True)),
            bucket_minutes=max(15, int(_number(data.get("bucket_minutes"), 120))),
            base_live_cap=max(0, int(_number(data.get("base_live_cap"), 5))),
            proven_live_cap=max(0, int(_number(data.get("proven_live_cap"), 10))),
            full_live_cap_bootstrap=max(0, int(_number(data.get("full_live_cap_bootstrap"), 20))),
            full_live_cap_growth=max(0, int(_number(data.get("full_live_cap_growth"), 30))),
            full_live_cap_growth_hot=max(0, int(_number(data.get("full_live_cap_growth_hot"), 40))),
            bootstrap_equity_threshold=max(0.0, _number(data.get("bootstrap_equity_threshold"), 160.0)),
            growth_equity_threshold=max(0.0, _number(data.get("growth_equity_threshold"), 300.0)),
            min_trades_for_proven=max(1, int(_number(data.get("min_trades_for_proven"), 10))),
            min_trades_for_full=max(1, int(_number(data.get("min_trades_for_full"), 20))),
            min_win_rate=max(0.0, min(1.0, _number(data.get("min_win_rate"), 0.50))),
            min_expectancy_r=_number(data.get("min_expectancy_r"), 0.0),
            min_full_expectancy_r=_number(data.get("min_full_expectancy_r"), 0.10),
            state_file=str(data.get("state_file") or "data/aggression_controller_state.json"),
        )


class LiveAggressionController:
    def __init__(self, config: AggressionConfig, *, project_root: Path | None = None) -> None:
        self.config = config
        root = Path(project_root or Path.cwd())
        state_path = Path(config.state_file).expanduser()
        self.state_path = state_path if state_path.is_absolute() else (root / state_path)
        self.state: dict[str, Any] = self._load_state()

    @classmethod
    def from_mapping(
        cls,
        raw: Mapping[str, Any] | None,
        *,
        project_root: Path | None = None,
    ) -> "LiveAggressionController":
        return cls(AggressionConfig.from_mapping(raw), project_root=project_root)

    def unlock(self, *, source: str = "telegram", now: datetime | None = None) -> dict[str, Any]:
        current = _utc(now)
        self.state["owner_unlocked"] = True
        self.state["owner_unlocked_at"] = current.isoformat()
        self.state["owner_unlocked_source"] = str(source or "telegram")
        self.state["updated_at"] = current.isoformat()
        self._append_audit({"event": "owner_unlock", "source": source, "at": current.isoformat()})
        self._save_state()
        return self.snapshot(now=current)

    def lock(self, *, source: str = "operator", now: datetime | None = None) -> dict[str, Any]:
        current = _utc(now)
        self.state["owner_unlocked"] = False
        self.state["updated_at"] = current.isoformat()
        self._append_audit({"event": "owner_lock", "source": source, "at": current.isoformat()})
        self._save_state()
        return self.snapshot(now=current)

    def try_consume(
        self,
        *,
        signal_id: str,
        symbol: str,
        evidence: LiveEvidence | Mapping[str, Any] | None = None,
        equity: float = 0.0,
        hard_blockers: list[str] | None = None,
        now: datetime | None = None,
    ) -> AggressionDecision:
        current = _utc(now)
        snapshot = self.snapshot(evidence=evidence, equity=equity, hard_blockers=hard_blockers, now=current)
        if not bool(snapshot["enabled"]):
            return AggressionDecision(True, "aggression_controller_disabled", str(snapshot["tier"]), 0, 0, 0)
        if not bool(snapshot["allowed"]):
            return AggressionDecision(
                False,
                str(snapshot["primary_blocker"]),
                str(snapshot["tier"]),
                int(snapshot["cap"]),
                int(snapshot["used"]),
                int(snapshot["remaining"]),
            )
        clean_signal_id = str(signal_id or "").strip()
        if not clean_signal_id:
            return AggressionDecision(False, "missing_signal_id", str(snapshot["tier"]), int(snapshot["cap"]), int(snapshot["used"]), int(snapshot["remaining"]))
        bucket = str(snapshot["bucket_start"])
        entries = self._entries()
        existing = entries.get(clean_signal_id)
        if isinstance(existing, Mapping) and str(existing.get("bucket_start") or "") == bucket:
            return AggressionDecision(True, "already_counted", str(snapshot["tier"]), int(snapshot["cap"]), int(snapshot["used"]), int(snapshot["remaining"]))
        if int(snapshot["remaining"]) <= 0:
            return AggressionDecision(False, "aggression_bucket_cap_reached", str(snapshot["tier"]), int(snapshot["cap"]), int(snapshot["used"]), 0)
        entries[clean_signal_id] = {
            "signal_id": clean_signal_id,
            "symbol": str(symbol or "").upper(),
            "bucket_start": bucket,
            "counted_at": current.isoformat(),
            "status": "delivered",
        }
        self.state["entries"] = self._trim_entries(entries, keep_bucket=bucket)
        self.state["last_consumed_signal_id"] = clean_signal_id
        self.state["updated_at"] = current.isoformat()
        self._save_state()
        used = self._used_for_bucket(bucket)
        cap = int(snapshot["cap"])
        return AggressionDecision(True, "allowed", str(snapshot["tier"]), cap, used, max(0, cap - used))

    def mark_accepted(self, signal_id: str, *, now: datetime | None = None) -> None:
        self._set_entry_status(signal_id, "accepted", now=now)

    def release(self, signal_id: str, *, reason: str = "rejected", now: datetime | None = None) -> None:
        clean_signal_id = str(signal_id or "").strip()
        if not clean_signal_id:
            return
        entries = self._entries()
        entry = entries.pop(clean_signal_id, None)
        if entry is not None:
            self.state["entries"] = entries
            self.state["updated_at"] = _utc(now).isoformat()
            self._append_audit({"event": "release", "signal_id": clean_signal_id, "reason": reason, "at": self.state["updated_at"]})
            self._save_state()

    def snapshot(
        self,
        *,
        evidence: LiveEvidence | Mapping[str, Any] | None = None,
        equity: float = 0.0,
        hard_blockers: list[str] | None = None,
        now: datetime | None = None,
    ) -> dict[str, Any]:
        current = _utc(now)
        evidence_payload = coerce_live_evidence(evidence)
        bucket_start = _bucket_start(current, self.config.bucket_minutes)
        bucket_end = bucket_start + self.config.bucket_minutes * 60
        bucket_key = datetime.fromtimestamp(bucket_start, tz=UTC).isoformat()
        cap, tier, promotion = self._tier(evidence_payload, equity)
        if not bool(self.config.enabled):
            cap = 0
            tier = "DISABLED"
        used = self._used_for_bucket(bucket_key)
        blockers = list(hard_blockers or [])
        if bool(self.config.enabled) and bool(self.config.owner_unlock_required) and not bool(self.state.get("owner_unlocked")):
            blockers.append("telegram_aggression_unlock_required")
        if bool(self.config.enabled) and used >= cap:
            blockers.append("aggression_bucket_cap_reached")
        why_not_full = self._why_not_full(evidence_payload, equity, hard_blockers or [])
        allowed = bool(self.config.enabled) and not blockers and cap > 0
        return {
            "enabled": bool(self.config.enabled),
            "owner_unlock_required": bool(self.config.owner_unlock_required),
            "owner_unlocked": bool(self.state.get("owner_unlocked", False)),
            "owner_unlocked_at": str(self.state.get("owner_unlocked_at") or ""),
            "owner_unlocked_source": str(self.state.get("owner_unlocked_source") or ""),
            "tier": tier,
            "cap": int(cap),
            "used": int(used),
            "remaining": int(max(0, cap - used)),
            "bucket_minutes": int(self.config.bucket_minutes),
            "bucket_start": bucket_key,
            "bucket_end": datetime.fromtimestamp(bucket_end, tz=UTC).isoformat(),
            "next_reset": datetime.fromtimestamp(bucket_end, tz=UTC).isoformat(),
            "allowed": allowed,
            "primary_blocker": str(blockers[0]) if blockers else "",
            "blockers": blockers,
            "promotion": promotion,
            "live_evidence": evidence_payload.__dict__,
            "why_not_full_aggression": why_not_full,
            "state_file": str(self.state_path),
            "last_restart_at": str(self.state.get("last_restart_at") or ""),
            "last_consumed_signal_id": str(self.state.get("last_consumed_signal_id") or ""),
            "audit_tail": list(self.state.get("audit", []) or [])[-10:],
        }

    def note_restart(self, *, now: datetime | None = None) -> None:
        current = _utc(now)
        self.state.setdefault("created_at", current.isoformat())
        self.state["last_restart_at"] = current.isoformat()
        self.state["updated_at"] = current.isoformat()
        self._append_audit({"event": "restart", "at": current.isoformat()})
        self._save_state()

    def _tier(self, evidence: LiveEvidence, equity: float) -> tuple[int, str, dict[str, Any]]:
        proven = evidence.trade_count >= self.config.min_trades_for_proven and evidence.win_rate >= self.config.min_win_rate and evidence.expectancy_r > self.config.min_expectancy_r
        full = evidence.trade_count >= self.config.min_trades_for_full and evidence.win_rate >= self.config.min_win_rate and evidence.expectancy_r >= self.config.min_full_expectancy_r
        hot_growth = full and equity >= self.config.growth_equity_threshold and evidence.expectancy_r >= max(self.config.min_full_expectancy_r, 0.15)
        if full and equity < self.config.bootstrap_equity_threshold:
            tier = "FULL_BOOTSTRAP"
            cap = self.config.full_live_cap_bootstrap
        elif hot_growth:
            tier = "FULL_GROWTH_HOT"
            cap = self.config.full_live_cap_growth_hot
        elif full:
            tier = "FULL_GROWTH"
            cap = self.config.full_live_cap_growth
        elif proven:
            tier = "PROVEN"
            cap = self.config.proven_live_cap
        else:
            tier = "BASE"
            cap = self.config.base_live_cap
        return int(cap), tier, {
            "proven_ready": bool(proven),
            "full_ready": bool(full),
            "hot_growth_ready": bool(hot_growth),
            "min_trades_for_proven": int(self.config.min_trades_for_proven),
            "min_trades_for_full": int(self.config.min_trades_for_full),
            "min_win_rate": float(self.config.min_win_rate),
            "min_expectancy_r": float(self.config.min_expectancy_r),
            "min_full_expectancy_r": float(self.config.min_full_expectancy_r),
        }

    def _why_not_full(self, evidence: LiveEvidence, equity: float, hard_blockers: list[str]) -> list[str]:
        reasons = list(hard_blockers)
        if bool(self.config.owner_unlock_required) and not bool(self.state.get("owner_unlocked")):
            reasons.append("telegram_aggression_unlock_required")
        if evidence.trade_count < self.config.min_trades_for_full:
            reasons.append("insufficient_real_closed_trades_for_full")
        if evidence.win_rate < self.config.min_win_rate:
            reasons.append("win_rate_below_full_gate")
        if evidence.expectancy_r < self.config.min_full_expectancy_r:
            reasons.append("expectancy_below_full_gate")
        if equity <= 0.0:
            reasons.append("missing_live_equity")
        return reasons

    def _entries(self) -> dict[str, dict[str, Any]]:
        raw = self.state.get("entries")
        return {str(k): dict(v) for k, v in raw.items()} if isinstance(raw, Mapping) else {}

    def _used_for_bucket(self, bucket_start: str) -> int:
        return sum(1 for entry in self._entries().values() if str(entry.get("bucket_start") or "") == bucket_start)

    def _set_entry_status(self, signal_id: str, status: str, *, now: datetime | None = None) -> None:
        clean_signal_id = str(signal_id or "").strip()
        if not clean_signal_id:
            return
        entries = self._entries()
        if clean_signal_id not in entries:
            return
        entries[clean_signal_id]["status"] = str(status or "")
        entries[clean_signal_id]["status_at"] = _utc(now).isoformat()
        self.state["entries"] = entries
        self.state["updated_at"] = entries[clean_signal_id]["status_at"]
        self._save_state()

    def _append_audit(self, payload: Mapping[str, Any]) -> None:
        audit = list(self.state.get("audit", []) or [])
        audit.append(dict(payload))
        self.state["audit"] = audit[-100:]

    def _trim_entries(self, entries: Mapping[str, Mapping[str, Any]], *, keep_bucket: str) -> dict[str, dict[str, Any]]:
        current = {str(k): dict(v) for k, v in entries.items() if str(v.get("bucket_start") or "") == keep_bucket}
        overflow = {str(k): dict(v) for k, v in entries.items() if str(v.get("bucket_start") or "") != keep_bucket}
        for key in list(sorted(overflow.keys()))[-50:]:
            current[key] = overflow[key]
        return current

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"created_at": _utc(None).isoformat(), "owner_unlocked": False, "entries": {}, "audit": []}
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {"created_at": _utc(None).isoformat(), "owner_unlocked": False, "entries": {}, "audit": [{"event": "state_load_failed"}]}
        return dict(payload) if isinstance(payload, Mapping) else {"owner_unlocked": False, "entries": {}, "audit": []}

    def _save_state(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.state_path.with_suffix(self.state_path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, self.state_path)


def coerce_live_evidence(raw: LiveEvidence | Mapping[str, Any] | None) -> LiveEvidence:
    if isinstance(raw, LiveEvidence):
        return raw
    data = dict(raw or {})
    overall = data.get("overall") if isinstance(data.get("overall"), Mapping) else {}
    return LiveEvidence(
        trade_count=int(_number(data.get("trade_count", overall.get("trades")), 0.0)),
        win_rate=_number(overall.get("win_rate", data.get("win_rate")), 0.0),
        expectancy_r=_number(overall.get("expectancy_r", data.get("expectancy_r")), 0.0),
        profit_factor=_number(overall.get("profit_factor", data.get("profit_factor")), 0.0),
        max_drawdown_r=_number(overall.get("max_drawdown_r", data.get("max_drawdown_r")), 0.0),
    )


def _bucket_start(now: datetime, bucket_minutes: int) -> int:
    bucket_seconds = max(60, int(bucket_minutes) * 60)
    ts = int(_utc(now).timestamp())
    return ts - (ts % bucket_seconds)


def _utc(value: datetime | None) -> datetime:
    if value is None:
        return datetime.now(tz=UTC)
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _number(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default
