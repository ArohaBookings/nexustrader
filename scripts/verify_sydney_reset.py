from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.bridge_server import BridgeActionQueue
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
    parser = argparse.ArgumentParser(description="Verify Sydney trading-day ledger and current daily state.")
    parser.add_argument("--journal-db", default=str(_default_journal_db(root)))
    parser.add_argument("--bridge-db", default=str(_default_bridge_db(root)))
    parser.add_argument("--account", default="Main")
    parser.add_argument("--magic", type=int, default=7777)
    parser.add_argument("--equity", type=float, default=None)
    args = parser.parse_args()

    journal = TradeJournal(Path(args.journal_db))
    bridge_db = Path(args.bridge_db)
    current_equity = args.equity
    latest_snapshot = None
    if current_equity is None and bridge_db.exists():
        queue = BridgeActionQueue(db_path=bridge_db, ttl_seconds=60)
        latest_snapshot = queue.latest_account_snapshot(account=args.account, magic=args.magic)
        if isinstance(latest_snapshot, dict):
            try:
                current_equity = float(latest_snapshot.get("equity") or 0.0)
            except Exception:
                current_equity = None
    if current_equity is None:
        current_equity = 0.0

    stats = journal.stats(
        current_equity=float(current_equity or 0.0),
        account=str(args.account or "").strip() or None,
        magic=int(args.magic),
    )
    daily_state, daily_reason = RiskEngine.resolve_daily_state_from_stats(stats)

    print(json.dumps(
        {
            "current_utc_time": utc_now().isoformat(),
            "current_sydney_time": utc_now().astimezone(SYDNEY).isoformat(),
            "current_trading_day_key": current_trading_day_key(),
            "timezone_used": str(getattr(SYDNEY, "key", SYDNEY)),
            "account": str(args.account),
            "magic": int(args.magic),
            "current_equity": float(current_equity or 0.0),
            "daily_state": str(daily_state),
            "daily_state_reason": str(daily_reason),
            "day_start_equity": float(stats.day_start_equity),
            "day_high_equity": float(stats.day_high_equity),
            "daily_realized_pnl": float(stats.daily_realized_pnl),
            "daily_pnl_pct": float(stats.daily_pnl_pct),
            "daily_dd_pct_live": float(stats.daily_dd_pct_live),
            "today_closed_trade_count": int(stats.today_closed_trade_count),
            "today_closed_trade_ids": list(stats.today_closed_trade_ids),
            "today_closed_trade_times_raw": list(stats.today_closed_trade_times_raw),
            "today_closed_trade_times_sydney": list(stats.today_closed_trade_times_sydney),
            "today_closed_trade_rows": [
                {
                    **dict(item),
                    "closed_at_sydney": trading_day_time_for_timestamp(item.get("closed_at_raw")),
                }
                for item in list(stats.today_closed_trade_details)
            ],
            "trading_day_debug": {
                "previous_trading_day_key": str(stats.previous_trading_day_key),
                "reset_triggered_at": str(stats.reset_triggered_at),
                "reset_reason": str(stats.reset_reason),
                "day_bucket_source": str(stats.day_bucket_source),
                "day_start_equity_source": str(stats.day_start_equity_source),
                "day_high_equity_source": str(stats.day_high_equity_source),
            },
            "latest_account_snapshot": latest_snapshot or {},
        },
        indent=2,
        sort_keys=True,
        default=str,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
