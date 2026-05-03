from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timezone
from typing import Any

from src.session_calendar import dominant_session_name
from src.utils import parse_hhmm


UTC = timezone.utc


@dataclass(frozen=True)
class SessionRule:
    name: str
    start: time
    end: time
    enabled: bool
    liquidity_tier: str
    size_multiplier: float
    confluence_delta: int
    ai_threshold_offset: float
    allow_trend: bool
    allow_range: bool
    allow_fakeout: bool

    def contains(self, current_utc: datetime) -> bool:
        current = current_utc.astimezone(UTC).timetz().replace(tzinfo=None)
        if self.start <= self.end:
            return self.start <= current <= self.end
        return current >= self.start or current <= self.end


@dataclass(frozen=True)
class SessionContext:
    session_name: str
    in_session: bool
    liquidity_tier: str
    size_multiplier: float
    confluence_delta: int
    ai_threshold_offset: float
    allow_trend: bool
    allow_range: bool
    allow_fakeout: bool

    @property
    def allowed_strategies(self) -> set[str]:
        output: set[str] = set()
        if self.allow_trend:
            output.add("TREND")
        if self.allow_range:
            output.add("RANGE")
        if self.allow_fakeout:
            output.add("FAKEOUT")
        return output


DEFAULT_SESSIONS: tuple[dict[str, Any], ...] = (
    {
        "name": "TOKYO",
        "start": "00:00",
        "end": "07:00",
        "enabled": True,
        "liquidity_tier": "LOW",
        "size_multiplier": 0.65,
        "confluence_delta": 0,
        "ai_threshold_offset": 0.01,
        "allow_trend": True,
        "allow_range": True,
        "allow_fakeout": True,
    },
    {
        "name": "LONDON",
        "start": "07:00",
        "end": "13:00",
        "enabled": True,
        "liquidity_tier": "HIGH",
        "size_multiplier": 1.0,
        "confluence_delta": 0,
        "ai_threshold_offset": 0.0,
        "allow_trend": True,
        "allow_range": True,
        "allow_fakeout": True,
    },
    {
        "name": "OVERLAP",
        "start": "13:00",
        "end": "17:00",
        "enabled": True,
        "liquidity_tier": "VERY_HIGH",
        "size_multiplier": 1.0,
        "confluence_delta": -1,
        "ai_threshold_offset": -0.01,
        "allow_trend": True,
        "allow_range": True,
        "allow_fakeout": True,
    },
    {
        "name": "NEW_YORK",
        "start": "17:00",
        "end": "21:00",
        "enabled": True,
        "liquidity_tier": "HIGH",
        "size_multiplier": 0.95,
        "confluence_delta": 0,
        "ai_threshold_offset": 0.0,
        "allow_trend": True,
        "allow_range": True,
        "allow_fakeout": True,
    },
    {
        "name": "SYDNEY",
        "start": "21:00",
        "end": "00:00",
        "enabled": True,
        "liquidity_tier": "LOW",
        "size_multiplier": 0.6,
        "confluence_delta": 1,
        "ai_threshold_offset": 0.03,
        "allow_trend": True,
        "allow_range": True,
        "allow_fakeout": True,
    },
)


class SessionProfile:
    def __init__(self, rules: list[SessionRule]) -> None:
        self._rules = rules

    @classmethod
    def from_config(cls, config: dict[str, Any] | None) -> "SessionProfile":
        configured = config or {}
        session_rows: list[dict[str, Any]] = []
        raw_sessions = configured.get("windows")
        if isinstance(raw_sessions, list):
            for item in raw_sessions:
                if isinstance(item, dict):
                    session_rows.append(item)
        if not session_rows:
            session_rows = [dict(row) for row in DEFAULT_SESSIONS]

        rules: list[SessionRule] = []
        for row in session_rows:
            rules.append(
                SessionRule(
                    name=str(row.get("name", "UNKNOWN")).upper(),
                    start=parse_hhmm(str(row.get("start", "00:00"))),
                    end=parse_hhmm(str(row.get("end", "23:59"))),
                    enabled=bool(row.get("enabled", True)),
                    liquidity_tier=str(row.get("liquidity_tier", "MEDIUM")).upper(),
                    size_multiplier=float(row.get("size_multiplier", 1.0)),
                    confluence_delta=int(row.get("confluence_delta", 0)),
                    ai_threshold_offset=float(row.get("ai_threshold_offset", 0.0)),
                    allow_trend=bool(row.get("allow_trend", True)),
                    allow_range=bool(row.get("allow_range", True)),
                    allow_fakeout=bool(row.get("allow_fakeout", True)),
                )
            )
        return cls(rules)

    def classify(self, now_utc: datetime) -> SessionContext:
        dynamic_name = dominant_session_name(now_utc)
        for rule in self._rules:
            if not rule.enabled:
                continue
            if rule.name == dynamic_name:
                return SessionContext(
                    session_name=rule.name,
                    in_session=True,
                    liquidity_tier=rule.liquidity_tier,
                    size_multiplier=rule.size_multiplier,
                    confluence_delta=rule.confluence_delta,
                    ai_threshold_offset=rule.ai_threshold_offset,
                    allow_trend=rule.allow_trend,
                    allow_range=rule.allow_range,
                    allow_fakeout=rule.allow_fakeout,
                )
        for rule in self._rules:
            if not rule.enabled:
                continue
            if rule.contains(now_utc):
                return SessionContext(
                    session_name=rule.name,
                    in_session=True,
                    liquidity_tier=rule.liquidity_tier,
                    size_multiplier=rule.size_multiplier,
                    confluence_delta=rule.confluence_delta,
                    ai_threshold_offset=rule.ai_threshold_offset,
                    allow_trend=rule.allow_trend,
                    allow_range=rule.allow_range,
                    allow_fakeout=rule.allow_fakeout,
                )
        return SessionContext(
            session_name="OFF",
            in_session=False,
            liquidity_tier="OFF",
            size_multiplier=0.0,
            confluence_delta=2,
            ai_threshold_offset=0.05,
            allow_trend=False,
            allow_range=False,
            allow_fakeout=False,
        )

    def infer_name(self, timestamp_iso: str) -> str:
        try:
            current = datetime.fromisoformat(str(timestamp_iso).replace("Z", "+00:00"))
        except ValueError:
            return "UNKNOWN"
        if current.tzinfo is None:
            current = current.replace(tzinfo=UTC)
        return self.classify(current).session_name
