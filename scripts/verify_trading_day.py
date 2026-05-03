from __future__ import annotations

from pathlib import Path
import argparse
import json

from src.bridge_server import BridgeActionQueue
from src.config_loader import load_settings
from src.execution import TradeJournal, current_trading_day_key, trading_day_time_for_timestamp
from src.risk_engine import RiskEngine
from src.session_calendar import SYDNEY
from src.utils import utc_now


def _default_journal_db(root: Path) -> Path:
    for name in ("data/trades_db.sqlite", "data/trade_journal.db", "data/trade_journal.sqlite", "data/trades.sqlite"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / "data" / "trades_db.sqlite"


def _default_bridge_db(root: Path) -> Path:
    for name in ("data/bridge_actions.sqlite", "data/bridge.sqlite", "data/bridge_actions.db"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / "data" / "bridge_actions.sqlite"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Verify Sydney trading-day ledger state.")
    parser.add_argument("--account", default="Main")
    parser.add_argument("--magic", type=int, default=7777)
    parser.add_argument("--equity", type=float, default=None, help="Current live equity snapshot for daily-state evaluation.")
    parser.add_argument("--db", default=str(_default_journal_db(root)), help="Optional path to the trade journal database.")
    parser.add_argument("--bridge-db", default=str(_default_bridge_db(root)), help="Optional path to the bridge action database for latest live equity.")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser()
    settings = load_settings()
    risk_settings = settings.section("risk")
    caution_threshold = float(risk_settings.get("daily_caution_threshold_pct", 0.02))
    defensive_threshold = float(risk_settings.get("daily_defensive_threshold_pct", 0.035))
    hard_stop_threshold = float(risk_settings.get("daily_hard_stop_threshold_pct", risk_settings.get("hard_daily_dd_pct", 0.05)))
    journal = TradeJournal(db_path)
    current_time = utc_now()
    current_equity = args.equity
    if current_equity is None:
        bridge_db = Path(args.bridge_db).expanduser()
        if bridge_db.exists():
            queue = BridgeActionQueue(db_path=bridge_db, ttl_seconds=60)
            latest_snapshot = queue.latest_account_snapshot(account=str(args.account), magic=int(args.magic))
            if isinstance(latest_snapshot, dict):
                try:
                    current_equity = float(latest_snapshot.get("equity") or 0.0)
                except Exception:
                    current_equity = None
    stats = journal.stats(current_equity=float(current_equity or 0.0), account=str(args.account), magic=int(args.magic))
    daily_state, daily_reason = RiskEngine.resolve_daily_state_from_stats(
        stats,
        caution_threshold_pct=caution_threshold,
        defensive_threshold_pct=defensive_threshold,
        hard_stop_threshold_pct=hard_stop_threshold,
    )

    print(f"Sydney now: {current_time.astimezone(SYDNEY).isoformat()}")
    print(f"UTC now: {current_time.isoformat()}")
    print(f"Current trading_day_key: {current_trading_day_key(now_ts=current_time)}")
    print(f"Stats trading_day_key: {stats.trading_day_key}")
    print(f"Timezone used: {stats.timezone_used}")
    print(f"Day bucket source: {stats.day_bucket_source}")
    print(f"Previous trading day key: {stats.previous_trading_day_key}")
    print(f"Reset triggered at: {stats.reset_triggered_at}")
    print(f"Reset reason: {stats.reset_reason}")
    print(f"Day start equity: {stats.day_start_equity} ({stats.day_start_equity_source})")
    print(f"Day high equity: {stats.day_high_equity} ({stats.day_high_equity_source})")
    print(f"Daily realized pnl: {stats.daily_realized_pnl}")
    print(f"Daily pnl pct: {stats.daily_pnl_pct}")
    print(f"Daily dd pct live: {stats.daily_dd_pct_live}")
    print(
        "Daily thresholds: "
        f"caution={caution_threshold} defensive={defensive_threshold} hard_stop={hard_stop_threshold}"
    )
    print(f"Closed trades today: {stats.today_closed_trade_count}")
    print(f"Current daily state: {daily_state}")
    print(f"Current daily state reason: {daily_reason}")
    print("Today closed trades:")
    if not stats.today_closed_trade_details:
        print("  (none)")
    for row in stats.today_closed_trade_details:
        raw_time = str(row.get("closed_at_raw") or "")
        print(
            json.dumps(
                {
                    "signal_id": row.get("signal_id"),
                    "closed_at_raw": raw_time,
                    "closed_at_sydney": trading_day_time_for_timestamp(raw_time, tz=SYDNEY),
                    "trading_day_key": row.get("trading_day_key"),
                    "pnl_amount": row.get("pnl_amount"),
                    "pnl_r": row.get("pnl_r"),
                },
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
