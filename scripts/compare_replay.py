#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_payload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _overall(payload: dict[str, Any]) -> dict[str, Any]:
    suite = payload.get("suite_mean")
    if isinstance(suite, dict):
        overall = suite.get("overall")
        if isinstance(overall, dict) and overall:
            return dict(overall)
    frozen = payload.get("frozen_replay_overall")
    if isinstance(frozen, dict) and frozen:
        return dict(frozen)
    overall = payload.get("overall")
    return dict(overall) if isinstance(overall, dict) else {}


def _metric(payload: dict[str, Any], key: str) -> float:
    try:
        return float(_overall(payload).get(key, 0.0) or 0.0)
    except Exception:
        return 0.0


def _pair_floor_failures(candidate: dict[str, Any], minimum_win_rate: float, minimum_trades: int) -> list[str]:
    failures: list[str] = []
    pair_rows = list(candidate.get("suite_by_pair_mean") or candidate.get("by_pair") or [])
    for entry in pair_rows:
        trades = int(float(entry.get("trades", 0.0) or 0.0))
        win_rate = float(entry.get("win_rate", 0.0) or 0.0)
        symbol = str(entry.get("symbol") or "")
        if trades >= minimum_trades and win_rate < minimum_win_rate:
            failures.append(
                f"pair floor failed: {symbol} trades={trades} win_rate={win_rate:.4f} floor={minimum_win_rate:.4f}"
            )
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare replay JSON reports and fail on frozen replay gate misses.")
    parser.add_argument("--baseline", required=True, help="Baseline replay JSON path.")
    parser.add_argument("--candidate", required=True, help="Candidate replay JSON path.")
    parser.add_argument("--gate-win-rate", type=float, default=0.55)
    parser.add_argument("--gate-trades", type=float, default=700.0)
    parser.add_argument("--gate-rr", type=float, default=0.0)
    parser.add_argument("--pair-floor-win-rate", type=float, default=0.0)
    parser.add_argument("--pair-floor-min-trades", type=int, default=20)
    parser.add_argument("--stretch-win-rate", type=float, default=0.70)
    parser.add_argument("--stretch-profit-factor", type=float, default=3.0)
    parser.add_argument("--stretch-expectancy-r", type=float, default=0.85)
    parser.add_argument("--stretch-trades", type=float, default=1000.0)
    args = parser.parse_args()

    baseline = _load_payload(Path(args.baseline))
    candidate = _load_payload(Path(args.candidate))

    candidate_overall = _overall(candidate)
    baseline_overall = _overall(baseline)
    failures: list[str] = []
    stretch_misses: list[str] = []

    gate_checks = {
        "win_rate": args.gate_win_rate,
        "trades": args.gate_trades,
        "winner_loss_ratio": args.gate_rr,
    }
    for key, minimum in gate_checks.items():
        if key == "winner_loss_ratio" and minimum <= 0.0:
            continue
        candidate_value = _metric(candidate, key)
        if candidate_value < minimum:
            failures.append(f"{key} below gate: candidate={candidate_value:.4f} gate={minimum:.4f}")

    if args.pair_floor_win_rate > 0.0:
        failures.extend(
            _pair_floor_failures(
                candidate,
                minimum_win_rate=float(args.pair_floor_win_rate),
                minimum_trades=int(args.pair_floor_min_trades),
            )
        )
    if list(candidate.get("protected_regressions") or []):
        failures.extend(
            [
                f"protected regression: {item.get('symbol')} baseline_win_rate={float(item.get('baseline_win_rate', 0.0)):.4f} current_win_rate={float(item.get('current_win_rate', 0.0)):.4f} baseline_pf={float(item.get('baseline_profit_factor', 0.0)):.4f} current_pf={float(item.get('current_profit_factor', 0.0)):.4f}"
                for item in list(candidate.get("protected_regressions") or [])
            ]
        )

    stretch_checks = {
        "win_rate": args.stretch_win_rate,
        "profit_factor": args.stretch_profit_factor,
        "expectancy_r": args.stretch_expectancy_r,
        "trades": args.stretch_trades,
    }
    for key, target in stretch_checks.items():
        candidate_value = _metric(candidate, key)
        if candidate_value < target:
            stretch_misses.append(f"{key} missed stretch: candidate={candidate_value:.4f} target={target:.4f}")

    print("Baseline overall:")
    print(json.dumps(baseline_overall, indent=2, sort_keys=True))
    print("\nCandidate frozen overall:")
    print(json.dumps(candidate_overall, indent=2, sort_keys=True))
    if isinstance(candidate.get("suite_stddev"), dict):
        print("\nCandidate suite stddev:")
        print(json.dumps(candidate.get("suite_stddev", {}), indent=2, sort_keys=True))
    protected_regressions = list(candidate.get("protected_regressions") or [])
    if protected_regressions:
        print("\nProtected regressions:")
        print(json.dumps(protected_regressions, indent=2, sort_keys=True))
    if isinstance(candidate.get("moving_replay_overall"), dict):
        print("\nCandidate moving overall:")
        print(json.dumps(candidate.get("moving_replay_overall", {}), indent=2, sort_keys=True))
    print("\nReplay gate:")
    print(
        json.dumps(
            {
                "gate_win_rate": args.gate_win_rate,
                "gate_trades": args.gate_trades,
                "gate_winner_loss_ratio": args.gate_rr,
                "pair_floor_win_rate": args.pair_floor_win_rate,
                "pair_floor_min_trades": args.pair_floor_min_trades,
                "stretch_win_rate_target": args.stretch_win_rate,
                "stretch_profit_factor_target": args.stretch_profit_factor,
                "stretch_expectancy_target": args.stretch_expectancy_r,
                "stretch_trade_count_target": args.stretch_trades,
            },
            indent=2,
            sort_keys=True,
        )
    )

    if failures:
        print("\nFrozen replay gate failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    if stretch_misses:
        print("\nStretch target misses:")
        for miss in stretch_misses:
            print(f"- {miss}")

    print("\nReplay comparison passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
