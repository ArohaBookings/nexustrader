from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo


UTC = timezone.utc
NEW_YORK = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")
SYDNEY = ZoneInfo("Australia/Sydney")
TOKYO = ZoneInfo("Asia/Tokyo")
AUCKLAND = ZoneInfo("Pacific/Auckland")


@dataclass(frozen=True)
class MarketScheduleState:
    symbol_key: str
    market_open: bool
    market_open_status: str
    next_open_time_utc: str
    next_open_time_local: str
    pre_open_window_active: bool
    dst_mode_active: dict[str, bool]
    dominant_session_name: str


def _normalize_symbol_key(value: str) -> str:
    compact = "".join(char for char in str(value).upper() if char.isalnum())
    if compact.startswith("XAUUSD") or compact.startswith("GOLD"):
        return "XAUUSD"
    if compact.startswith("BTCUSD") or compact.startswith("BTCUSDT") or compact.startswith("XBTUSD"):
        return "BTCUSD"
    if compact.startswith(("NAS100", "US100", "NASDAQ", "USTEC", "NAS", "NQ")):
        return "NAS100"
    if compact.startswith(("USOIL", "XTIUSD", "OILUSD", "WTI", "CL", "OIL", "USO")):
        return "USOIL"
    if compact.startswith("EURUSD"):
        return "EURUSD"
    if compact.startswith("GBPUSD"):
        return "GBPUSD"
    if compact.startswith("USDJPY"):
        return "USDJPY"
    return compact


def _as_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=UTC)
    return timestamp.astimezone(UTC)


def _in_local_window(current_utc: datetime, tz: ZoneInfo, start_hour: float, end_hour: float) -> bool:
    current_local = _as_utc(current_utc).astimezone(tz)
    hour = current_local.hour + (current_local.minute / 60.0) + (current_local.second / 3600.0)
    if start_hour <= end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def _next_weekday_time(current_local: datetime, weekday: int, target: time) -> datetime:
    days_ahead = (weekday - current_local.weekday()) % 7
    candidate = current_local.replace(
        year=current_local.year,
        month=current_local.month,
        day=current_local.day,
        hour=target.hour,
        minute=target.minute,
        second=target.second,
        microsecond=0,
    ) + timedelta(days=days_ahead)
    if candidate <= current_local:
        candidate += timedelta(days=7)
    return candidate


def dst_mode_flags(timestamp: datetime) -> dict[str, bool]:
    current = _as_utc(timestamp)
    return {
        "new_york": bool(current.astimezone(NEW_YORK).dst()),
        "london": bool(current.astimezone(LONDON).dst()),
        "sydney": bool(current.astimezone(SYDNEY).dst()),
    }


def dominant_session_name(timestamp: datetime) -> str:
    current = _as_utc(timestamp)
    london_open = _in_local_window(current, LONDON, 8.0, 16.5)
    new_york_open = _in_local_window(current, NEW_YORK, 8.0, 17.0)
    tokyo_open = _in_local_window(current, TOKYO, 9.0, 15.0)
    sydney_open = _in_local_window(current, SYDNEY, 7.0, 16.0)
    if london_open and new_york_open:
        return "OVERLAP"
    if london_open:
        return "LONDON"
    if new_york_open:
        return "NEW_YORK"
    if tokyo_open:
        return "TOKYO"
    if sydney_open:
        return "SYDNEY"
    return "OFF"


def is_weekend_market_mode(timestamp: datetime) -> bool:
    current_ny = _as_utc(timestamp).astimezone(NEW_YORK)
    weekday = current_ny.weekday()
    current_time = current_ny.timetz().replace(tzinfo=None)
    if weekday == 5:
        return True
    if weekday == 6 and current_time < time(17, 0):
        return True
    if weekday == 4 and current_time >= time(17, 0):
        return True
    return False


def _asset_group(symbol_key: str) -> str:
    normalized = _normalize_symbol_key(symbol_key)
    if normalized in {"EURUSD", "GBPUSD", "USDJPY", "AUDJPY", "NZDJPY", "AUDNZD", "EURJPY", "GBPJPY"}:
        return "FOREX"
    if normalized == "XAUUSD":
        return "METALS"
    if normalized in {"NAS100", "USOIL"}:
        return "FUTURES"
    if normalized == "BTCUSD":
        return "CRYPTO"
    return "FOREX"


def _describe_weekly_market(symbol_key: str, timestamp: datetime, *, display_tz: ZoneInfo) -> MarketScheduleState:
    normalized = _normalize_symbol_key(symbol_key)
    if normalized == "BTCUSD":
        return MarketScheduleState(
            symbol_key=normalized,
            market_open=True,
            market_open_status="OPEN_24_7",
            next_open_time_utc="",
            next_open_time_local="",
            pre_open_window_active=False,
            dst_mode_active=dst_mode_flags(timestamp),
            dominant_session_name=dominant_session_name(timestamp),
        )

    group = _asset_group(normalized)
    open_hour = 17 if group == "FOREX" else 18
    daily_reopen_hour = None if group == "FOREX" else 18
    daily_close_hour = None if group == "FOREX" else 17

    current_utc = _as_utc(timestamp)
    current_ny = current_utc.astimezone(NEW_YORK)
    weekday = current_ny.weekday()
    current_local_time = current_ny.timetz().replace(tzinfo=None)

    market_open = False
    status = "CLOSED"

    if weekday == 5:
        market_open = False
        status = "WEEKEND_CLOSED"
    elif weekday == 6:
        market_open = current_local_time >= time(open_hour, 0)
        status = "OPEN" if market_open else "WEEKEND_CLOSED"
    elif weekday == 4 and current_local_time >= time(17, 0):
        market_open = False
        status = "WEEKEND_CLOSED"
    else:
        market_open = True
        status = "OPEN"
        if daily_close_hour is not None and daily_reopen_hour is not None:
            if time(daily_close_hour, 0) <= current_local_time < time(daily_reopen_hour, 0):
                market_open = False
                status = "CLOSED"

    if market_open:
        next_open_utc = ""
        next_open_local = ""
        pre_open = False
    else:
        if status == "WEEKEND_CLOSED":
            next_open_local_dt = _next_weekday_time(current_ny, 6, time(open_hour, 0))
        elif daily_reopen_hour is not None and current_local_time < time(daily_reopen_hour, 0):
            next_open_local_dt = current_ny.replace(
                hour=daily_reopen_hour,
                minute=0,
                second=0,
                microsecond=0,
            )
        else:
            candidate = current_ny + timedelta(days=1)
            while candidate.weekday() >= 5:
                candidate += timedelta(days=1)
            next_open_local_dt = candidate.replace(
                hour=daily_reopen_hour or open_hour,
                minute=0,
                second=0,
                microsecond=0,
            )
        next_open_utc_dt = next_open_local_dt.astimezone(UTC)
        next_open_display = next_open_utc_dt.astimezone(display_tz)
        seconds_to_open = max(0.0, (next_open_utc_dt - current_utc).total_seconds())
        if 0.0 < seconds_to_open <= 3600.0:
            status = "PRE_OPEN"
        elif 3600.0 < seconds_to_open <= 3 * 3600.0:
            status = "PREP_WINDOW"
        pre_open = status in {"PRE_OPEN", "PREP_WINDOW"}
        next_open_utc = next_open_utc_dt.isoformat()
        next_open_local = next_open_display.isoformat()

    return MarketScheduleState(
        symbol_key=normalized,
        market_open=bool(market_open),
        market_open_status=status,
        next_open_time_utc=next_open_utc,
        next_open_time_local=next_open_local,
        pre_open_window_active=bool(pre_open),
        dst_mode_active=dst_mode_flags(timestamp),
        dominant_session_name=dominant_session_name(timestamp),
    )


def describe_market_state(symbol_key: str, timestamp: datetime, *, display_timezone: str = "Pacific/Auckland") -> MarketScheduleState:
    try:
        display_tz = ZoneInfo(str(display_timezone))
    except Exception:
        display_tz = AUCKLAND
    return _describe_weekly_market(symbol_key, timestamp, display_tz=display_tz)


def market_open_tuple(symbol_key: str, timestamp: datetime) -> tuple[bool, str]:
    state = describe_market_state(symbol_key, timestamp)
    return bool(state.market_open), str(state.market_open_status)
