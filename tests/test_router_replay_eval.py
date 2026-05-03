from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.router_replay_eval import _apply_xau_profile, _load_frames, _prime_session_window_specs, _resolve_snapshot


UTC = timezone.utc


def _frame(rows: int, *, step_minutes: int = 5) -> pd.DataFrame:
    start = datetime(2026, 3, 1, tzinfo=UTC)
    payload: list[dict[str, object]] = []
    for idx in range(rows):
        timestamp = start + timedelta(minutes=step_minutes * idx)
        close = 2200.0 + (idx * 0.1)
        payload.append(
            {
                "time": pd.Timestamp(timestamp),
                "open": close - 0.05,
                "high": close + 0.10,
                "low": close - 0.10,
                "close": close,
            }
        )
    return pd.DataFrame(payload)


class _FakeMarketData:
    def __init__(self, cached_frames: dict[str, pd.DataFrame]) -> None:
        self.cached_frames = dict(cached_frames)
        self.fetch_calls: list[tuple[str, str, int]] = []

    def load_cached(self, symbol: str, timeframe: str) -> pd.DataFrame | None:
        return self.cached_frames.get(str(timeframe).upper())

    def fetch(self, symbol: str, timeframe: str, count: int) -> pd.DataFrame | None:
        self.fetch_calls.append((str(symbol), str(timeframe).upper(), int(count)))
        return _frame(20)


class _FakeSettings:
    def __init__(self, raw: dict[str, object] | None = None) -> None:
        self.raw = dict(raw or {})

    def resolve_path_value(self, value: str) -> str:
        return str(value)


class _FakeBaselineStats:
    trading_day_key = "2026-03-19"


class _FakeJournal:
    def stats(self, current_equity: float = 100.0) -> _FakeBaselineStats:
        return _FakeBaselineStats()

    def closed_trades(self, closed_limit: int) -> list[dict[str, object]]:
        return [
            {
                "signal_id": "seed",
                "closed_at": datetime(2026, 3, 18, tzinfo=UTC).isoformat(),
                "opened_at": datetime(2026, 3, 18, tzinfo=UTC).isoformat(),
                "symbol": "XAUUSD",
                "r_multiple": 1.0,
            }
        ]


class RouterReplayEvalTests(unittest.TestCase):
    def test_load_frames_prefers_cached_history_over_refresh(self) -> None:
        cached_frames = {
            "M1": _frame(300, step_minutes=1),
            "M5": _frame(400, step_minutes=5),
            "M15": _frame(350, step_minutes=15),
            "H1": _frame(320, step_minutes=60),
            "H4": _frame(200, step_minutes=240),
        }
        market_data = _FakeMarketData(cached_frames)
        runtime = {
            "market_data": market_data,
            "resolved_symbols": {"XAUUSD": "XAUUSD"},
        }

        frames = _load_frames(runtime, "XAUUSD")

        self.assertEqual(sorted(frames.keys()), ["H1", "H4", "M1", "M15", "M5"])
        self.assertEqual({key: len(value) for key, value in frames.items()}, {key: len(value) for key, value in cached_frames.items()})
        self.assertEqual(market_data.fetch_calls, [])

    def test_resolve_snapshot_refreshes_when_window_metadata_is_stale(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "replay_snapshot.json"
            window_path = Path(temp_dir) / "replay_snapshot_window_a.json"
            window_path.write_text(
                json.dumps(
                    {
                        "snapshot_timestamp": "2026-03-19T00:00:00+00:00",
                        "trading_day_key": "2026-03-18",
                        "window_id": "window_a",
                        "window_start": "2026-03-10T00:00:00+00:00",
                        "window_end": "2026-03-11T00:00:00+00:00",
                        "baseline_stats": {},
                        "closed_trades": [],
                    }
                ),
                encoding="utf-8",
            )
            runtime = {
                "settings": _FakeSettings(),
                "journal": _FakeJournal(),
                "bridge_orchestrator_config": {"replay_snapshot_file": str(base_path)},
            }
            window_spec = {
                "window_id": "window_a",
                "start": datetime(2026, 3, 17, 4, 36, 40, tzinfo=UTC),
                "end": datetime(2026, 3, 18, 23, 0, 0, tzinfo=UTC),
            }

            snapshot, resolved_path = _resolve_snapshot(
                runtime,
                use_snapshot_path="",
                refresh_snapshot=False,
                window_spec=window_spec,
            )

            self.assertEqual(resolved_path, window_path)
            self.assertEqual(snapshot["window_id"], "window_a")
            self.assertEqual(snapshot["window_start"], "2026-03-17T04:36:40+00:00")
            self.assertEqual(snapshot["window_end"], "2026-03-18T23:00:00+00:00")
            self.assertEqual(snapshot["trading_day_key"], "2026-03-19")
            persisted = json.loads(window_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["window_start"], snapshot["window_start"])
            self.assertEqual(persisted["window_end"], snapshot["window_end"])

    def test_resolve_snapshot_keeps_explicit_snapshot_path_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            base_path = Path(temp_dir) / "frozen_snapshot.json"
            window_path = Path(temp_dir) / "frozen_snapshot_window_a.json"
            frozen_payload = {
                "snapshot_timestamp": "2026-03-19T00:00:00+00:00",
                "trading_day_key": "2026-03-18",
                "window_id": "window_a",
                "window_start": "2026-03-10T00:00:00+00:00",
                "window_end": "2026-03-11T00:00:00+00:00",
                "baseline_stats": {},
                "closed_trades": [],
            }
            window_path.write_text(json.dumps(frozen_payload), encoding="utf-8")
            runtime = {
                "settings": _FakeSettings(),
                "journal": _FakeJournal(),
            }
            window_spec = {
                "window_id": "window_a",
                "start": datetime(2026, 3, 17, 4, 36, 40, tzinfo=UTC),
                "end": datetime(2026, 3, 18, 23, 0, 0, tzinfo=UTC),
            }

            snapshot, resolved_path = _resolve_snapshot(
                runtime,
                use_snapshot_path=str(base_path),
                refresh_snapshot=False,
                window_spec=window_spec,
            )

            self.assertEqual(resolved_path, window_path)
            self.assertEqual(snapshot["window_start"], frozen_payload["window_start"])
            self.assertEqual(snapshot["window_end"], frozen_payload["window_end"])
            persisted = json.loads(window_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["window_start"], frozen_payload["window_start"])
            self.assertEqual(persisted["window_end"], frozen_payload["window_end"])

    def test_apply_xau_profile_reinstantiates_grid_scalper_with_selected_profile(self) -> None:
        runtime = {
            "settings": _FakeSettings(
                {
                    "xau_grid_scalper": {
                        "enabled": True,
                        "active_profile": "checkpoint",
                        "proof_mode": "checkpoint",
                        "checkpoint_artifact": "/tmp/xau_checkpoint.json",
                        "density_branch_artifact": "/tmp/xau_density.json",
                        "density_first_mode": True,
                        "prime_burst_entries": 8,
                        "profiles": {
                            "checkpoint": {
                                "density_first_mode": False,
                                "prime_burst_entries": 6,
                            },
                            "density_branch": {
                                "density_first_mode": True,
                                "prime_burst_entries": 9,
                            },
                        },
                    }
                }
            ),
            "logger": None,
        }

        proof_state = _apply_xau_profile(runtime, xau_profile="density_branch", proof_mode="vps_soak")

        self.assertEqual(runtime["grid_scalper"].active_profile, "density_branch")
        self.assertEqual(runtime["grid_scalper"].proof_mode, "vps_soak")
        self.assertTrue(bool(runtime["grid_scalper"].density_first_mode))
        self.assertEqual(runtime["grid_scalper"].prime_burst_entries, 9)
        self.assertEqual(proof_state["active_profile"], "density_branch")
        self.assertEqual(proof_state["proof_mode"], "vps_soak")
        self.assertEqual(proof_state["checkpoint_artifact"], "/tmp/xau_checkpoint.json")
        self.assertEqual(proof_state["density_branch_artifact"], "/tmp/xau_density.json")

    def test_prime_session_window_specs_inherit_frozen_bounds(self) -> None:
        specs = _prime_session_window_specs(
            {
                "window_start": "2026-03-17T06:26:40+00:00",
                "window_end": "2026-03-19T00:50:00+00:00",
            }
        )

        self.assertEqual([item["window_id"] for item in specs], ["london", "overlap", "new_york"])
        self.assertEqual(str(specs[0]["start"]), "2026-03-17 06:26:40+00:00")
        self.assertEqual(str(specs[0]["end"]), "2026-03-19 00:50:00+00:00")
        self.assertEqual(specs[1]["session_names"], ["OVERLAP"])


if __name__ == "__main__":
    unittest.main()
