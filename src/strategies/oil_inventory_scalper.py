from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc


@dataclass(frozen=True)
class OilInventoryDecision:
    allowed: bool
    reason: str
    news_armed: bool
    phase: str
    spread_cap: float
    confluence_floor: float
    volatility_cap: float


def is_eia_inventory_window(
    *,
    now_utc: datetime,
    timezone_name: str = "America/New_York",
    pre_minutes: int = 10,
    post_minutes: int = 30,
) -> bool:
    local = now_utc.astimezone(ZoneInfo(timezone_name))
    if local.weekday() != 2:  # Wednesday
        return False
    event_time = local.replace(hour=10, minute=30, second=0, microsecond=0)
    window_start = event_time - timedelta(minutes=max(0, int(pre_minutes)))
    window_end = event_time + timedelta(minutes=max(0, int(post_minutes)))
    return window_start <= local <= window_end


def evaluate_oil_inventory_scalper(
    *,
    now_utc: datetime,
    session_name: str,
    spread_points: float,
    confluence_score: float,
    probability: float,
    regime: str,
    news_status: str,
    atr_ratio: float,
    snapshot_age_seconds: float,
    spread_caps_by_session: dict[str, float],
    base_confluence_floor: float,
    news_armed_enabled: bool,
    eia_window_minutes_pre: int,
    eia_window_minutes_post: int,
    stricter_confluence_floor: float,
    stricter_spread_cap: float,
    volatility_cap: float,
    asia_enabled: bool = False,
    timezone_name: str = "America/New_York",
    spread_loosen_pct: float = 0.0,
    confluence_relax: float = 0.0,
) -> OilInventoryDecision:
    normalized_session = str(session_name or "").upper()
    if snapshot_age_seconds > 180:
        return OilInventoryDecision(
            allowed=False,
            reason="snapshot_stale",
            news_armed=False,
            phase=normalized_session,
            spread_cap=0.0,
            confluence_floor=0.0,
            volatility_cap=max(1.0, float(volatility_cap)),
        )
    if normalized_session in {"TOKYO", "SYDNEY"} and not bool(asia_enabled):
        return OilInventoryDecision(
            allowed=False,
            reason="session_filter_block",
            news_armed=False,
            phase=normalized_session,
            spread_cap=0.0,
            confluence_floor=0.0,
            volatility_cap=max(1.0, float(volatility_cap)),
        )
    event_armed = bool(news_armed_enabled) and is_eia_inventory_window(
        now_utc=now_utc,
        timezone_name=timezone_name,
        pre_minutes=eia_window_minutes_pre,
        post_minutes=eia_window_minutes_post,
    )
    status_text = str(news_status or "").lower()
    if bool(news_armed_enabled) and any(token in status_text for token in ("inventory", "eia", "high", "blocked", "event")):
        event_armed = True
    spread_cap = float(spread_caps_by_session.get(normalized_session, spread_caps_by_session.get("DEFAULT", 75.0)))
    spread_cap *= max(1.0, 1.0 + float(spread_loosen_pct))
    if event_armed:
        spread_cap = min(spread_cap, float(stricter_spread_cap))
    if float(spread_points) > spread_cap:
        return OilInventoryDecision(
            allowed=False,
            reason="spread_too_wide_session",
            news_armed=event_armed,
            phase=normalized_session,
            spread_cap=spread_cap,
            confluence_floor=0.0,
            volatility_cap=max(1.0, float(volatility_cap)),
        )
    floor = float(stricter_confluence_floor if event_armed else base_confluence_floor) - max(0.0, float(confluence_relax))
    floor = max(0.35, min(0.95, floor))
    if float(confluence_score) < floor:
        return OilInventoryDecision(
            allowed=False,
            reason="confluence_below_threshold",
            news_armed=event_armed,
            phase=normalized_session,
            spread_cap=spread_cap,
            confluence_floor=floor,
            volatility_cap=max(1.0, float(volatility_cap)),
        )
    if event_armed and float(probability) < 0.70:
        return OilInventoryDecision(
            allowed=False,
            reason="news_armed_probability_low",
            news_armed=True,
            phase=normalized_session,
            spread_cap=spread_cap,
            confluence_floor=floor,
            volatility_cap=max(1.0, float(volatility_cap)),
        )
    if float(atr_ratio) > max(1.0, float(volatility_cap)):
        return OilInventoryDecision(
            allowed=False,
            reason="volatility_too_high",
            news_armed=event_armed,
            phase=normalized_session,
            spread_cap=spread_cap,
            confluence_floor=floor,
            volatility_cap=max(1.0, float(volatility_cap)),
        )
    normalized_regime = str(regime or "").upper()
    if (not event_armed) and (normalized_session in {"NEW_YORK", "OVERLAP"}) and ("RANGING" in normalized_regime):
        return OilInventoryDecision(
            allowed=True,
            reason="oil_range_scalper_allowed",
            news_armed=False,
            phase=normalized_session,
            spread_cap=spread_cap,
            confluence_floor=floor,
            volatility_cap=max(1.0, float(volatility_cap)),
        )
    return OilInventoryDecision(
        allowed=True,
        reason="oil_inventory_scalper_allowed",
        news_armed=event_armed,
        phase=normalized_session,
        spread_cap=spread_cap,
        confluence_floor=floor,
        volatility_cap=max(1.0, float(volatility_cap)),
    )
