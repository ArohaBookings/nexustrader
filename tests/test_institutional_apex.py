from __future__ import annotations

from src.apex_telegram import ApexTelegramConfig, ApexTelegramResponder
from src.edge_gated_apex import build_edge_gated_apex_policy
from src.institutional_apex import build_institutional_apex_snapshot


def _dashboard_payload() -> dict:
    health = {
        "bridge_status": "UP",
        "broker_connectivity": {
            "account": "FTMO_EVAL_01",
            "magic": 77,
            "terminal_connected": True,
            "terminal_trade_allowed": True,
            "mql_trade_allowed": True,
        },
        "current_daily_state": "DAILY_NORMAL",
        "execution_quality_state": "GOOD",
        "open_risk_pct": 0.004,
    }
    stats = {
        "latest_account_snapshot": {
            "account": "FTMO_EVAL_01",
            "magic": 77,
            "equity": 104.0,
            "balance": 102.0,
            "free_margin": 96.0,
            "floating_pnl": 2.0,
            "symbol": "XAUUSD",
        },
        "account_scaling": {"equity": 104.0, "balance": 102.0, "free_margin": 96.0},
        "risk_state": {"day_start_equity": 100.0, "day_high_equity": 105.0, "absolute_drawdown_pct": 0.0},
        "daily_dd_pct_live": 0.01,
        "open_positions": 1,
    }
    symbols = [
        {
            "symbol": "XAUUSD",
            "quality_score": 0.82,
            "probability": 0.67,
            "confluence": 3.8,
            "entry_timing_score": 0.78,
            "structure_cleanliness_score": 0.81,
            "regime_fit": 0.73,
            "session_fit": 0.9,
            "execution_quality_fit": 0.88,
            "pair_behavior_fit": 0.77,
            "runtime_market_data_source": "mt5 polygon finnhub consensus_ok",
            "runtime_market_data_consensus_state": "READY",
            "strategy_key": "XAUUSD_SMC",
            "strategy_state": "ATTACK",
        }
    ]
    learning = {
        "best": [{"strategy": "XAUUSD_SMC", "trade_count": 220, "expectancy_r": 0.16, "profit_factor": 1.8}],
        "worst": [{"strategy": "OLD", "trade_count": 220, "expectancy_r": 0.04, "profit_factor": 1.1}],
        "optimizer_summary": {
            "validation_sample_size": 120,
            "recent_expectancy_delta": 0.04,
            "validation_expectancy_delta": 0.035,
            "recent_expectancy": 0.16,
            "validation_expectancy": 0.12,
        },
    }
    apex = build_institutional_apex_snapshot(
        health=health,
        stats=stats,
        symbols=symbols,
        learning=learning,
        risk_config={
            "funded": {
                "enabled": True,
                "group": "FTMO",
                "phase": "evaluation",
                "starting_balance": 100.0,
                "daily_loss_limit_pct": 0.05,
                "overall_drawdown_limit_pct": 0.10,
                "profit_target_pct": 0.08,
            }
        },
    )
    edge = build_edge_gated_apex_policy(
        health={
            **health,
            "learning_brain": {
                "trained_samples": 320,
                "last_market_history_seed_status": "ok",
                "last_market_history_backfill_status": "failed",
            },
        },
        stats=stats,
        symbols=symbols,
        institutional_apex=apex,
        risk_config={
            "funded": {
                "enabled": True,
                "group": "FTMO",
                "phase": "evaluation",
                "starting_balance": 100.0,
                "daily_loss_limit_pct": 0.05,
                "overall_drawdown_limit_pct": 0.10,
                "profit_target_pct": 0.08,
            }
        },
        orchestrator_config={
            "edge_promotion": {
                "recent_window": 200,
                "validation_window": 100,
                "min_expectancy_gain": 0.03,
                "max_drawdown_degradation": 0.0,
            },
            "frequency_policy": {"priority_symbols": ["XAUUSD", "BTCUSD"]},
        },
        xau_btc_trajectory_stats={
            "XAUUSD": {
                "actual_candidates_last_10m": 2,
                "actual_trades_last_10m": 1,
                "soft_target_trades_last_10m": {"low": 6, "high": 8},
                "catchup_pressure": 0.4,
            }
        },
    )
    return {
        "summary": {"equity": 104.0, "current_daily_state": "DAILY_NORMAL"},
        "institutional_apex": apex,
        "institutional_intelligence": edge,
        "training_bootstrap_status": edge["training_bootstrap_status"],
        "data_quality": edge["data_quality"],
        "self_repair": edge["self_repair"],
        "promotion_audit": edge["promotion_audit"],
        "funded_mission": edge["funded_mission"],
        "trajectory_forecast": edge["trajectory_forecast"],
        "xau_btc_opportunity_pipeline": edge["xau_btc_opportunity_pipeline"],
        "live_shadow_gap": edge["live_shadow_gap"],
        "symbols": symbols,
    }


def test_institutional_apex_uses_mt5_and_funded_buffers() -> None:
    dashboard = _dashboard_payload()
    apex = dashboard["institutional_apex"]

    assert apex["mt5_bridge"]["connected"] is True
    assert apex["funded_mission"]["enabled"] is True
    assert apex["funded_mission"]["mt5_derived"] is True
    assert apex["funded_mission"]["needed_to_pass"] == 4.0
    assert apex["anti_overfit"]["promotion_allowed"] is True
    assert apex["readiness"] in {"expand_inside_caps", "protect_funded_pass", "observe_validate"}


def test_telegram_responder_blocks_trade_placement_and_reports_status() -> None:
    dashboard = _dashboard_payload()
    responder = ApexTelegramResponder(config=ApexTelegramConfig(enabled=True, allow_controls=True, ai_enabled=False))

    blocked = responder.handle_text("buy XAUUSD now with max risk", dashboard, chat_id="123")
    status = responder.handle_text("/status", dashboard, chat_id="123")
    pause = responder.handle_text("/pause", dashboard, chat_id="123")
    kill = responder.handle_text("/kill", dashboard, chat_id="123")

    assert "Blocked" in blocked.text
    assert "APEX Status" in status.text
    assert pause.action == "pause_trading"
    assert kill.confirmation_required is True


def test_edge_policy_prioritizes_shadow_without_forcing_live_frequency() -> None:
    dashboard = _dashboard_payload()
    edge = dashboard["institutional_intelligence"]
    pipeline = edge["xau_btc_opportunity_pipeline"]
    xau = pipeline["priority_symbols"][0]

    assert edge["policy"] == "edge_gated_no_forced_live_frequency"
    assert pipeline["live_frequency_forced"] is False
    assert xau["symbol"] == "XAUUSD"
    assert xau["candidate_debt_10m"] == 4
    assert xau["forced_live_frequency"] is False
    assert xau["shadow_burst_allowed"] is True
    assert xau["live_gate"] in {"eligible_if_bridge_approves", "blocked_by_edge_or_risk_gate"}


def test_edge_policy_locks_hard_rails_and_blocks_expansion() -> None:
    dashboard = _dashboard_payload()
    apex = dict(dashboard["institutional_apex"])
    apex["self_repair"] = {
        "soft_blockers": [],
        "hard_rails": [{"symbol": "SYSTEM", "reason": "daily_hard_stop"}],
        "status": "hard_rail_holds",
    }
    edge = build_edge_gated_apex_policy(
        health={"learning_brain": {"trained_samples": 500, "last_market_history_seed_status": "ok"}},
        stats={},
        symbols=dashboard["symbols"],
        institutional_apex=apex,
        orchestrator_config={},
    )

    assert edge["self_repair"]["hard_rails_locked"] is True
    assert edge["training_bootstrap_status"]["live_risk_expansion_allowed"] is False
    assert edge["xau_btc_opportunity_pipeline"]["hard_rails_present"] is True


def test_telegram_reports_edge_surfaces_without_runtime_bypass() -> None:
    dashboard = _dashboard_payload()
    responder = ApexTelegramResponder(config=ApexTelegramConfig(enabled=True, allow_controls=True, ai_enabled=False))

    trajectory = responder.handle_text("/trajectory", dashboard, chat_id="123")
    blockers = responder.handle_text("/blockers", dashboard, chat_id="123")
    frequency = responder.handle_text("increase frequency on XAU", dashboard, chat_id="123")

    assert "Trajectory" in trajectory.text
    assert "Blockers" in blockers.text
    assert "Frequency Policy" in frequency.text
    assert "cannot force entries" in frequency.text
    assert not frequency.action
