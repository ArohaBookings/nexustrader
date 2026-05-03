#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.session_calendar import dominant_session_name
from src.strategies.trend_daytrade import resolve_strategy_key


def _metric_value(row: dict[str, Any]) -> float:
    pnl_r = float(row.get("pnl_r", 0.0) or 0.0)
    if abs(pnl_r) > 1e-9:
        return pnl_r
    return float(row.get("pnl_amount", 0.0) or 0.0)


def _parse_iso_utc(value: Any) -> datetime | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except Exception:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _parse_management_effect(raw: Any) -> dict[str, Any]:
    try:
        value = json.loads(str(raw or "{}"))
        return value if isinstance(value, dict) else {}
    except Exception:
        return {}


def _metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {
            "trades": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "expectancy_r": 0.0,
            "avg_r": 0.0,
            "avg_mae_r": 0.0,
            "avg_mfe_r": 0.0,
            "avg_duration_minutes": 0.0,
        }
    pnl = [_metric_value(row) for row in rows]
    wins = [value for value in pnl if value > 0]
    losses = [abs(value) for value in pnl if value < 0]
    gross_profit = sum(wins)
    gross_loss = sum(losses)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else 0.0)
    mae_values = [float(row.get("mae_r", 0.0) or 0.0) for row in rows]
    mfe_values = [float(row.get("mfe_r", 0.0) or 0.0) for row in rows]
    duration_values = [float(row.get("duration_minutes", 0.0) or 0.0) for row in rows]
    return {
        "trades": float(len(rows)),
        "win_rate": float(sum(1 for value in pnl if value >= 0) / len(pnl)),
        "profit_factor": float(profit_factor),
        "expectancy_r": float(sum(pnl) / len(pnl)),
        "avg_r": float(sum(pnl) / len(pnl)),
        "avg_mae_r": float(sum(mae_values) / len(mae_values)),
        "avg_mfe_r": float(sum(mfe_values) / len(mfe_values)),
        "avg_duration_minutes": float(sum(duration_values) / len(duration_values)),
    }


def _default_journal_db(root: Path) -> Path:
    for name in ("data/trades_db.sqlite", "data/trade_journal.db", "data/trade_journal.sqlite", "data/trades.sqlite"):
        candidate = root / name
        if candidate.exists():
            return candidate
    return root / "data" / "trades_db.sqlite"


def _fetch_rows(db_path: Path, *, account: str | None, magic: int | None, symbol: str | None) -> list[dict[str, Any]]:
    connection = sqlite3.connect(db_path)
    try:
        clauses = [
            "status = 'CLOSED'",
            "COALESCE(proof_trade, 0) = 0",
            "signal_id NOT LIKE 'FORCE_TEST::%'",
            "signal_id NOT LIKE 'CANARY::%'",
        ]
        params: list[Any] = []
        if account:
            clauses.append("account = ?")
            params.append(str(account).strip())
        if magic is not None:
            clauses.append("magic = ?")
            params.append(int(magic))
        if symbol:
            clauses.append("symbol = ?")
            params.append(str(symbol).upper())
        query = f"""
            SELECT signal_id, symbol, strategy_key, setup, session_name, regime, pnl_amount, pnl_r,
                   duration_minutes, management_effect_json, closed_at
            FROM trade_journal
            WHERE {' AND '.join(clauses)}
            ORDER BY closed_at ASC
        """
        rows = connection.execute(query, params).fetchall()
    finally:
        connection.close()
    output: list[dict[str, Any]] = []
    for row in rows:
        management = _parse_management_effect(row[9])
        symbol_key = str(row[1] or "")
        strategy_key = str(row[2] or "").strip()
        setup = str(row[3] or "")
        resolved_strategy_key = strategy_key or resolve_strategy_key(symbol_key, setup)
        closed_at = str(row[10] or "")
        session_name = str(row[4] or "").strip().upper()
        if not session_name:
            parsed_closed_at = _parse_iso_utc(closed_at)
            if parsed_closed_at is not None:
                session_name = dominant_session_name(parsed_closed_at)
        output.append(
            {
                "signal_id": str(row[0] or ""),
                "symbol": symbol_key,
                "strategy_key": str(resolved_strategy_key or ""),
                "setup": setup,
                "session_name": session_name,
                "regime": str(row[5] or ""),
                "pnl_amount": float(row[6] or 0.0),
                "pnl_r": float(row[7] or 0.0),
                "duration_minutes": float(row[8] or 0.0),
                "mfe_r": float(management.get("mfe_r", 0.0) or 0.0),
                "mae_r": float(management.get("mae_r", 0.0) or 0.0),
                "closed_at": closed_at,
            }
        )
    return output


def _group_report(rows: list[dict[str, Any]], keys: tuple[str, ...], *, min_trades: int) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(key, "") or "") for key in keys)].append(row)
    report: list[dict[str, Any]] = []
    for group_key, group_rows in grouped.items():
        if len(group_rows) < min_trades:
            continue
        metrics = _metrics(group_rows)
        entry = {key: value for key, value in zip(keys, group_key)}
        entry.update(metrics)
        report.append(entry)
    report.sort(
        key=lambda item: (
            -float(item.get("expectancy_r", 0.0) or 0.0),
            -float(item.get("profit_factor", 0.0) or 0.0),
            -float(item.get("trades", 0.0) or 0.0),
        )
    )
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Summarize pair+strategy+session+regime performance from trade_journal.")
    parser.add_argument(
        "--db",
        default=str(_default_journal_db(root)),
        help="Path to trade journal database",
    )
    parser.add_argument("--account", default=None)
    parser.add_argument("--magic", type=int, default=None)
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    db_path = Path(args.db).expanduser().resolve()
    rows = _fetch_rows(
        db_path,
        account=args.account,
        magic=args.magic,
        symbol=args.symbol,
    )
    payload = {
        "db_path": str(db_path),
        "rows": len(rows),
        "overall": _metrics(rows),
        "by_pair": _group_report(rows, ("symbol",), min_trades=args.min_trades),
        "by_pair_strategy": _group_report(rows, ("symbol", "strategy_key"), min_trades=args.min_trades),
        "by_pair_strategy_session": _group_report(rows, ("symbol", "strategy_key", "session_name"), min_trades=args.min_trades),
        "by_pair_strategy_session_regime": _group_report(
            rows,
            ("symbol", "strategy_key", "session_name", "regime"),
            min_trades=args.min_trades,
        ),
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"DB: {payload['db_path']}")
    print(f"Rows: {payload['rows']}")
    overall = payload["overall"]
    print(
        "Overall: "
        f"trades={int(overall['trades'])} "
        f"win_rate={overall['win_rate']:.3f} "
        f"pf={overall['profit_factor']:.3f} "
        f"expectancy_r={overall['expectancy_r']:.3f} "
        f"avg_mae_r={overall['avg_mae_r']:.3f} "
        f"avg_mfe_r={overall['avg_mfe_r']:.3f}"
    )
    for label in ("by_pair", "by_pair_strategy", "by_pair_strategy_session", "by_pair_strategy_session_regime"):
        print(f"\n[{label}]")
        for item in payload[label][:20]:
            scope = " | ".join(
                str(item.get(key) or "")
                for key in ("symbol", "strategy_key", "session_name", "regime")
                if key in item
            )
            print(
                f"{scope}: trades={int(item['trades'])} "
                f"win_rate={item['win_rate']:.3f} "
                f"pf={item['profit_factor']:.3f} "
                f"expectancy_r={item['expectancy_r']:.3f} "
                f"avg_mae_r={item['avg_mae_r']:.3f} "
                f"avg_mfe_r={item['avg_mfe_r']:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
