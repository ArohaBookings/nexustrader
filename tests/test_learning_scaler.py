from __future__ import annotations

from src.learning_scaler import build_learning_scaler_scorecard


def test_scorecard_refuses_world_class_claim_without_real_closed_trades() -> None:
    scorecard = build_learning_scaler_scorecard(
        rollout_stats={"trade_count": 4, "overall": {"win_rate": 0.75, "expectancy_r": 0.20, "profit_factor": 2.0}},
        aggression_snapshot={
            "owner_unlocked": True,
            "tier": "BASE",
            "cap": 5,
            "used": 1,
            "remaining": 4,
            "blockers": [],
            "live_evidence": {"trade_count": 4, "win_rate": 0.75, "expectancy_r": 0.20, "profit_factor": 2.0},
        },
        account_scaling={"equity": 120.0},
    )

    assert scorecard["claim"] == "not_proven_world_class_yet"
    assert scorecard["status"] == "collecting_live_evidence"
    assert "not_enough_real_closed_trades_for_learning_proof" in scorecard["why_not_world_class"]
    assert "collect_more_real_closed_trades_inside_current_cap" == scorecard["next_safe_action"]


def test_scorecard_allows_full_scaling_only_with_real_edge_and_no_hard_rails() -> None:
    scorecard = build_learning_scaler_scorecard(
        rollout_stats={
            "trade_count": 24,
            "overall": {"win_rate": 0.58, "expectancy_r": 0.14, "profit_factor": 1.6, "max_drawdown_r": -2.2},
            "last_20": {"expectancy_r": 0.17},
        },
        aggression_snapshot={
            "owner_unlocked": True,
            "tier": "FULL_BOOTSTRAP",
            "cap": 20,
            "used": 4,
            "remaining": 16,
            "blockers": [],
            "live_evidence": {"trade_count": 24, "win_rate": 0.58, "expectancy_r": 0.14, "profit_factor": 1.6, "max_drawdown_r": -2.2},
        },
        account_scaling={"equity": 140.0},
    )

    assert scorecard["claim"] == "measured_scaling_ready"
    assert scorecard["status"] == "full_scaling_ready_inside_caps"
    assert scorecard["quick_learner_score"] >= 80.0
    assert scorecard["quick_scaler_score"] >= 80.0
    assert scorecard["why_not_world_class"] == []


def test_scorecard_hard_rails_override_good_edge() -> None:
    scorecard = build_learning_scaler_scorecard(
        rollout_stats={
            "trade_count": 30,
            "overall": {"win_rate": 0.62, "expectancy_r": 0.18, "profit_factor": 1.9, "max_drawdown_r": -1.4},
            "last_20": {"expectancy_r": 0.20},
        },
        aggression_snapshot={
            "owner_unlocked": True,
            "tier": "FULL_GROWTH",
            "cap": 30,
            "used": 2,
            "remaining": 28,
            "blockers": ["daily_drawdown_hard_stop"],
            "live_evidence": {"trade_count": 30, "win_rate": 0.62, "expectancy_r": 0.18, "profit_factor": 1.9, "max_drawdown_r": -1.4},
        },
        account_scaling={"equity": 450.0},
    )

    assert scorecard["status"] == "protected_by_hard_rails"
    assert scorecard["claim"] == "not_proven_world_class_yet"
    assert "hard_rail_active:daily_drawdown_hard_stop" in scorecard["why_not_world_class"]
    assert scorecard["next_safe_action"] == "clear_hard_rail_before_any_scaling"
