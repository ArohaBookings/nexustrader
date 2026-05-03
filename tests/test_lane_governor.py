from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from src.lane_governor import (
    build_loss_attribution_summary,
    build_shadow_challenger_pool,
    build_walk_forward_scorecards,
    classify_loss_attribution,
    evaluate_execution_quality_gate,
    resolve_lane_lifecycle,
)


class LaneGovernorTests(unittest.TestCase):
    def test_walk_forward_attack_state_favors_hot_recent_lane(self) -> None:
        now = datetime(2026, 3, 27, 10, 0, tzinfo=timezone.utc)
        rows = []
        for index in range(8):
            rows.append(
                {
                    "closed_at": (now - timedelta(hours=index * 4)).isoformat(),
                    "session_name": "LONDON",
                    "pnl_r": 0.8 if index < 6 else -0.1,
                    "spread_points": 11.0,
                    "slippage_points": 1.0,
                    "mfe_r": 1.2,
                    "mae_r": -0.25,
                }
            )

        scorecards = build_walk_forward_scorecards(rows, now_utc=now)
        lifecycle = resolve_lane_lifecycle(scorecards)

        self.assertGreater(float(scorecards.get("current_edge_score") or 0.0), 0.60)
        self.assertEqual(str(lifecycle.get("state") or ""), "attack")
        self.assertTrue(bool(lifecycle.get("live_allowed")))

    def test_walk_forward_recent_damage_keeps_lane_shadowed_until_fresh_recovery(self) -> None:
        now = datetime(2026, 3, 27, 10, 0, tzinfo=timezone.utc)
        rows = []
        for index in range(6):
            rows.append(
                {
                    "closed_at": (now - timedelta(hours=index * 6)).isoformat(),
                    "session_name": "TOKYO",
                    "pnl_r": -0.18 if index < 4 else 0.04,
                    "spread_points": 13.0,
                    "slippage_points": 1.4,
                    "mfe_r": 0.4,
                    "mae_r": -0.5,
                }
            )
        for index in range(8, 16):
            rows.append(
                {
                    "closed_at": (now - timedelta(days=4, hours=index)).isoformat(),
                    "session_name": "TOKYO",
                    "pnl_r": 0.10,
                    "spread_points": 12.0,
                    "slippage_points": 1.0,
                    "mfe_r": 0.9,
                    "mae_r": -0.2,
                }
            )

        scorecards = build_walk_forward_scorecards(rows, now_utc=now)
        lifecycle = resolve_lane_lifecycle(scorecards)

        self.assertEqual(str(lifecycle.get("state") or ""), "shadow_only")
        self.assertTrue(bool(lifecycle.get("recent_edge_broken")))
        self.assertFalse(bool(lifecycle.get("recovery_ready")))

    def test_loss_attribution_prefers_management_when_trail_reason_is_present(self) -> None:
        row = {
            "pnl_r": -0.35,
            "management_reason": "runner_trailing_update",
            "mfe_r": 0.75,
            "spread_points": 12.0,
        }
        self.assertEqual(classify_loss_attribution(row), "bad_management")
        summary = build_loss_attribution_summary([row])
        self.assertEqual(str(summary.get("primary_cause") or ""), "bad_management")

    def test_shadow_challenger_pool_promotes_when_live_lane_is_degraded(self) -> None:
        pool = build_shadow_challenger_pool(
            symbol_key="XAUUSD",
            shadow_strategy_variants=[
                {
                    "variant_id": "XAUUSD_SHADOW_1",
                    "symbol": "XAUUSD",
                    "session": "LONDON",
                    "promotion_score": 0.71,
                    "slippage_adjusted_score": 0.68,
                    "promoted_candidate": True,
                },
                {
                    "variant_id": "XAUUSD_SHADOW_2",
                    "symbol": "XAUUSD",
                    "session": "TOKYO",
                    "promotion_score": 0.61,
                    "slippage_adjusted_score": 0.62,
                    "promoted_candidate": False,
                },
            ],
            lifecycle_state={"state": "degrade"},
            current_session_name="LONDON",
            limit=3,
        )

        self.assertTrue(bool(pool.get("promote_now")))
        self.assertEqual(str(pool.get("top_challenger", {}).get("variant_id") or ""), "XAUUSD_SHADOW_1")

    def test_execution_quality_gate_blocks_rough_shadow_only_lane(self) -> None:
        gate = evaluate_execution_quality_gate(
            spread_points=28.0,
            typical_spread_points=10.0,
            stop_distance_points=18.0,
            slippage_quality_score=0.32,
            execution_quality_score=0.38,
            microstructure_alignment=0.34,
            adverse_entry_risk=0.82,
            lifecycle_state="shadow_only",
        )

        self.assertTrue(bool(gate.get("blocked")))
        self.assertEqual(str(gate.get("state") or ""), "ROUGH")


if __name__ == "__main__":
    unittest.main()
