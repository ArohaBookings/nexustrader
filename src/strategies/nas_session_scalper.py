from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

UTC = timezone.utc


@dataclass(frozen=True)
class NasSessionDecision:
    allowed: bool
    reason: str
    phase: str
    spread_cap: float
    confluence_floor: float
    trade_rate_target: float
    orb_retest_required: bool
    strategy_mode: str


def _parse_hhmm(value: str, default_hour: int, default_minute: int) -> time:
    try:
        parts = str(value).split(":", 1)
        hour = int(parts[0])
        minute = int(parts[1]) if len(parts) > 1 else 0
        return time(hour=hour, minute=minute)
    except Exception:
        return time(hour=default_hour, minute=default_minute)


def _phase_for_nas(
    *,
    now_utc: datetime,
    timezone_name: str,
    ny_cash_open: str,
    ny_cash_close: str,
    cash_open_minutes: int,
    power_hour_minutes: int,
) -> str:
    local = now_utc.astimezone(ZoneInfo(timezone_name))
    open_t = _parse_hhmm(ny_cash_open, 9, 30)
    close_t = _parse_hhmm(ny_cash_close, 16, 0)
    open_dt = local.replace(hour=open_t.hour, minute=open_t.minute, second=0, microsecond=0)
    close_dt = local.replace(hour=close_t.hour, minute=close_t.minute, second=0, microsecond=0)
    if close_dt <= open_dt:
        close_dt = close_dt + timedelta(days=1)
    if open_dt <= local <= close_dt:
        if local <= (open_dt + timedelta(minutes=max(1, int(cash_open_minutes)))):
            return "CASH_OPEN"
        if local >= (close_dt - timedelta(minutes=max(1, int(power_hour_minutes)))):
            return "POWER_HOUR"
        return "NY_CORE"
    return "OFF_CASH"


def evaluate_nas_session_scalper(
    *,
    now_utc: datetime,
    session_name: str,
    regime: str,
    setup: str,
    spread_points: float,
    confluence_score: float,
    spread_caps_by_session: dict[str, float],
    trade_rate_targets_by_session: dict[str, float],
    confluence_floor: float,
    asia_enabled: bool,
    timezone_name: str = "America/New_York",
    ny_cash_open: str = "09:30",
    ny_cash_close: str = "16:00",
    cash_open_minutes: int = 120,
    power_hour_minutes: int = 60,
    spread_loosen_pct: float = 0.0,
    confluence_relax: float = 0.0,
) -> NasSessionDecision:
    normalized_session = str(session_name or "").upper()
    phase = _phase_for_nas(
        now_utc=now_utc,
        timezone_name=timezone_name,
        ny_cash_open=ny_cash_open,
        ny_cash_close=ny_cash_close,
        cash_open_minutes=cash_open_minutes,
        power_hour_minutes=power_hour_minutes,
    )
    if phase == "OFF_CASH" and normalized_session in {"TOKYO", "SYDNEY"} and not bool(asia_enabled):
        return NasSessionDecision(
            allowed=False,
            reason="session_filter_block",
            phase=phase,
            spread_cap=0.0,
            confluence_floor=0.0,
            trade_rate_target=0.0,
            orb_retest_required=True,
            strategy_mode="ORB",
        )
    spread_cap = float(
        spread_caps_by_session.get(phase, spread_caps_by_session.get(normalized_session, spread_caps_by_session.get("DEFAULT", 55.0)))
    )
    spread_cap *= max(1.0, 1.0 + float(spread_loosen_pct))
    if float(spread_points) > spread_cap:
        return NasSessionDecision(
            allowed=False,
            reason="spread_too_wide_session",
            phase=phase,
            spread_cap=spread_cap,
            confluence_floor=0.0,
            trade_rate_target=float(
                trade_rate_targets_by_session.get(phase, trade_rate_targets_by_session.get(normalized_session, trade_rate_targets_by_session.get("DEFAULT", 2.0)))
            ),
            orb_retest_required=True,
            strategy_mode="ORB",
        )
    normalized_regime = str(regime or "").upper()
    use_orb = "TREND" in normalized_regime
    mode = "ORB" if use_orb else "VWAP_MR"
    floor = float(confluence_floor) - max(0.0, float(confluence_relax))
    if phase in {"CASH_OPEN", "POWER_HOUR"}:
        floor += 0.0
    elif phase == "NY_CORE":
        floor += 0.02
    else:
        floor += 0.05
    if use_orb and "RETEST" not in str(setup or "").upper():
        floor += 0.02
    floor = max(0.35, min(0.95, floor))
    if float(confluence_score) < floor:
        return NasSessionDecision(
            allowed=False,
            reason="confluence_below_threshold",
            phase=phase,
            spread_cap=spread_cap,
            confluence_floor=floor,
            trade_rate_target=float(
                trade_rate_targets_by_session.get(phase, trade_rate_targets_by_session.get(normalized_session, trade_rate_targets_by_session.get("DEFAULT", 2.0)))
            ),
            orb_retest_required=use_orb,
            strategy_mode=mode,
        )
    return NasSessionDecision(
        allowed=True,
        reason="nas_session_scalper_allowed",
        phase=phase,
        spread_cap=spread_cap,
        confluence_floor=floor,
        trade_rate_target=float(
            trade_rate_targets_by_session.get(phase, trade_rate_targets_by_session.get(normalized_session, trade_rate_targets_by_session.get("DEFAULT", 2.0)))
        ),
        orb_retest_required=use_orb,
        strategy_mode=mode,
    )
