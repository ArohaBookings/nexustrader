from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
import hashlib
import json
import math
import os
import uuid


UTC = timezone.utc


@dataclass
class SessionWindow:
    name: str
    start: time
    end: time
    enabled: bool
    size_multiplier: float

    def contains(self, ts: datetime) -> bool:
        current = ts.astimezone(UTC).timetz().replace(tzinfo=None)
        if self.start <= self.end:
            return self.start <= current <= self.end
        return current >= self.start or current <= self.end


def utc_now() -> datetime:
    return datetime.now(tz=UTC)


def parse_hhmm(value: str) -> time:
    hour_str, minute_str = value.split(":", 1)
    return time(hour=int(hour_str), minute=int(minute_str))


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    if denominator == 0:
        return default
    return numerator / denominator


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def rolling_mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return sum(materialized) / len(materialized)


def json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def deterministic_id(*parts: Any) -> str:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return digest[:20]


def random_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def minutes_ago(minutes: int) -> datetime:
    return utc_now() - timedelta(minutes=minutes)


def floor_non_negative(value: float) -> float:
    return max(0.0, value)


def pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return (current - previous) / previous


def sqrt_or_zero(value: float) -> float:
    if value <= 0:
        return 0.0
    return math.sqrt(value)
