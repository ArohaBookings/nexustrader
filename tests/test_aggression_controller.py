from __future__ import annotations

from datetime import datetime, timedelta, timezone

from src.aggression_controller import LiveAggressionController


UTC = timezone.utc


def _controller(tmp_path, **overrides) -> LiveAggressionController:
    config = {
        "enabled": True,
        "owner_unlock_required": True,
        "bucket_minutes": 120,
        "base_live_cap": 5,
        "proven_live_cap": 10,
        "full_live_cap_bootstrap": 20,
        "full_live_cap_growth": 30,
        "full_live_cap_growth_hot": 40,
        "bootstrap_equity_threshold": 160,
        "growth_equity_threshold": 300,
        "min_trades_for_proven": 10,
        "min_trades_for_full": 20,
        "min_win_rate": 0.50,
        "min_expectancy_r": 0.0,
        "min_full_expectancy_r": 0.10,
        "state_file": str(tmp_path / "aggression_state.json"),
    }
    config.update(overrides)
    return LiveAggressionController.from_mapping(config)


def test_autonomous_base_tier_can_run_without_telegram_unlock(tmp_path) -> None:
    controller = _controller(tmp_path, owner_unlock_required=False, autonomous_base_enabled=True)
    now = datetime(2026, 5, 4, 0, 0, tzinfo=UTC)

    decision = controller.try_consume(signal_id="sig-auto", symbol="XAUUSD", evidence={}, equity=100.0, now=now)
    snapshot = controller.snapshot(now=now)

    assert decision.allowed is True
    assert decision.reason == "allowed"
    assert snapshot["owner_unlocked"] is False
    assert snapshot["autonomous_base_active"] is True
    assert snapshot["remaining"] == 4


def test_base_tier_requires_telegram_owner_unlock(tmp_path) -> None:
    controller = _controller(tmp_path)
    now = datetime(2026, 5, 4, 0, 0, tzinfo=UTC)

    decision = controller.try_consume(signal_id="sig-1", symbol="XAUUSD", evidence={}, equity=100.0, now=now)

    assert decision.allowed is False
    assert decision.reason == "telegram_aggression_unlock_required"
    assert controller.snapshot(now=now)["owner_unlocked"] is False


def test_base_tier_allows_five_real_live_entries_per_two_hour_bucket(tmp_path) -> None:
    controller = _controller(tmp_path)
    now = datetime(2026, 5, 4, 0, 0, tzinfo=UTC)
    controller.unlock(now=now)

    decisions = [
        controller.try_consume(signal_id=f"sig-{idx}", symbol="XAUUSD", evidence={}, equity=100.0, now=now + timedelta(minutes=idx))
        for idx in range(6)
    ]

    assert [item.allowed for item in decisions[:5]] == [True, True, True, True, True]
    assert decisions[5].allowed is False
    assert decisions[5].reason == "aggression_bucket_cap_reached"
    snapshot = controller.snapshot(now=now + timedelta(minutes=10))
    assert snapshot["tier"] == "BASE"
    assert snapshot["used"] == 5
    assert snapshot["remaining"] == 0


def test_bucket_resets_after_two_hours(tmp_path) -> None:
    controller = _controller(tmp_path)
    now = datetime(2026, 5, 4, 0, 0, tzinfo=UTC)
    controller.unlock(now=now)
    for idx in range(5):
        assert controller.try_consume(signal_id=f"sig-{idx}", symbol="BTCUSD", evidence={}, equity=100.0, now=now).allowed

    reset_decision = controller.try_consume(
        signal_id="sig-reset",
        symbol="BTCUSD",
        evidence={},
        equity=100.0,
        now=now + timedelta(hours=2, seconds=1),
    )

    assert reset_decision.allowed is True
    assert reset_decision.used == 1
    assert reset_decision.remaining == 4


def test_shadow_trades_do_not_promote_tier(tmp_path) -> None:
    controller = _controller(tmp_path)
    now = datetime(2026, 5, 4, 0, 0, tzinfo=UTC)
    controller.unlock(now=now)

    snapshot = controller.snapshot(
        evidence={"shadow_trade_count": 500, "shadow_win_rate": 0.80, "shadow_expectancy_r": 1.2},
        equity=100.0,
        now=now,
    )

    assert snapshot["tier"] == "BASE"
    assert snapshot["promotion"]["proven_ready"] is False
    assert snapshot["live_evidence"]["trade_count"] == 0


def test_proven_tier_requires_real_closed_trade_win_rate_and_expectancy(tmp_path) -> None:
    controller = _controller(tmp_path)
    now = datetime(2026, 5, 4, 0, 0, tzinfo=UTC)
    controller.unlock(now=now)

    snapshot = controller.snapshot(
        evidence={
            "trade_count": 10,
            "overall": {"win_rate": 0.50, "expectancy_r": 0.01, "profit_factor": 1.1},
            "pnl_r_values": [0.30, -0.10] * 5,
        },
        equity=120.0,
        now=now,
    )

    assert snapshot["tier"] == "PROVEN"
    assert snapshot["cap"] == 10
    assert snapshot["promotion"]["proven_ready"] is True
    assert abs(float(snapshot["live_evidence"]["payoff_ratio"]) - 3.0) < 1e-9


def test_proven_tier_requires_payoff_ratio_gate(tmp_path) -> None:
    controller = _controller(tmp_path)
    now = datetime(2026, 5, 4, 0, 0, tzinfo=UTC)
    controller.unlock(now=now)

    snapshot = controller.snapshot(
        evidence={
            "trade_count": 10,
            "overall": {"win_rate": 0.70, "expectancy_r": 0.01, "profit_factor": 1.1},
            "pnl_r_values": [0.10] * 7 + [-0.20] * 3,
        },
        equity=120.0,
        now=now,
    )

    assert snapshot["tier"] == "BASE"
    assert snapshot["promotion"]["proven_ready"] is False
    assert snapshot["live_evidence"]["payoff_ratio"] < snapshot["promotion"]["min_payoff_ratio_for_proven"]


def test_full_bootstrap_tier_requires_stronger_real_evidence_and_keeps_hard_rails(tmp_path) -> None:
    controller = _controller(tmp_path)
    now = datetime(2026, 5, 4, 0, 0, tzinfo=UTC)
    controller.unlock(now=now)
    evidence = {
        "trade_count": 20,
        "overall": {"win_rate": 0.55, "expectancy_r": 0.10, "profit_factor": 1.4},
        "pnl_r_values": [0.30] * 11 + [-0.10] * 9,
    }

    snapshot = controller.snapshot(evidence=evidence, equity=100.0, hard_blockers=["daily_guard_daily_defensive"], now=now)
    decision = controller.try_consume(signal_id="sig-hard", symbol="XAUUSD", evidence=evidence, equity=100.0, hard_blockers=["daily_guard_daily_defensive"], now=now)

    assert snapshot["tier"] == "FULL_BOOTSTRAP"
    assert snapshot["cap"] == 20
    assert decision.allowed is False
    assert decision.reason == "daily_guard_daily_defensive"


def test_restart_restores_bucket_state_and_release_frees_rejected_slot(tmp_path) -> None:
    now = datetime(2026, 5, 4, 0, 0, tzinfo=UTC)
    controller = _controller(tmp_path)
    controller.unlock(now=now)
    assert controller.try_consume(signal_id="sig-1", symbol="XAUUSD", evidence={}, equity=100.0, now=now).allowed
    assert controller.try_consume(signal_id="sig-2", symbol="XAUUSD", evidence={}, equity=100.0, now=now).allowed
    controller.release("sig-2", reason="broker_rejected", now=now + timedelta(minutes=1))

    restored = _controller(tmp_path)
    snapshot = restored.snapshot(now=now + timedelta(minutes=2))

    assert snapshot["owner_unlocked"] is True
    assert snapshot["used"] == 1
    assert snapshot["remaining"] == 4
