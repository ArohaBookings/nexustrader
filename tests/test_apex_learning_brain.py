from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.apex_learning_brain import ApexLearningBrain
from src.online_learning import OnlineLearningEngine
from src.strategy_optimizer import StrategyOptimizer


class _FakeJournal:
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self._rows = list(rows)

    def closed_trades(self, limit: int, symbol: str | None = None) -> list[dict[str, object]]:
        rows = self._rows[-max(1, int(limit)) :]
        if symbol:
            rows = [row for row in rows if str(row.get("symbol") or "").upper() == str(symbol).upper()]
        return list(rows)


class ApexLearningBrainTests(unittest.TestCase):
    def test_run_cycle_persists_bundle_and_offline_review(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=5,
                promotion_min_samples=10,
            )
            now = datetime(2026, 3, 19, 9, 0, tzinfo=timezone.utc)
            for index in range(24):
                pnl_r = 1.0 if index % 3 != 0 else -0.5
                online.on_trade_close(
                    {
                        "timestamp_utc": now.isoformat(),
                        "signal_id": f"sig-{index}",
                        "symbol": "USDJPY" if index % 2 == 0 else "AUDJPY",
                        "timeframe": "M15",
                        "side": "BUY",
                        "entry": 150.0,
                        "sl": 149.9,
                        "tp": 150.3,
                        "exit": 150.2,
                        "pnl_r": pnl_r,
                        "pnl_money": pnl_r,
                        "news_state": "clear",
                        "session_state": "IN",
                        "session_name": "TOKYO",
                        "ai_decision": "approved",
                        "ai_probability": 0.72,
                        "spread_points": 10.0,
                        "lot": 0.01,
                        "regime": "TRENDING",
                        "setup": "USDJPY_MOMENTUM_IMPULSE",
                        "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                        "lane_name": "FX_SCALP",
                    }
                )
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=5,
                cooldown_minutes=1,
            )
            optimizer.record_trade(
                {
                    "timestamp_utc": now.isoformat(),
                    "symbol": "USDJPY",
                    "strategy": "USDJPY_MOMENTUM_IMPULSE",
                    "pnl_r": 1.2,
                    "pnl_money": 1.2,
                    "session": "TOKYO",
                    "regime": "TRENDING",
                }
            )
            brain = ApexLearningBrain(
                online_learning=online,
                strategy_optimizer=optimizer,
                journal=_FakeJournal(
                    [
                        {"symbol": "USDJPY", "pnl_r": 1.2},
                        {"symbol": "AUDJPY", "pnl_r": 0.8},
                        {"symbol": "GBPUSD", "pnl_r": -1.0},
                    ]
                ),
                data_dir=root / "brain",
                offline_gpt_enabled=True,
            )
            brain._offline_gpt.offline_review = lambda _context: (  # type: ignore[method-assign]
                {
                    "summary": "offline review ok",
                    "weak_patterns": ["gbpusd_london"],
                    "strategy_ideas": ["promote usdjpy trend pullback"],
                    "next_cycle_focus": ["tighten gbpusd"],
                    "reentry_watchlist": ["usdjpy tokyo reclaim"],
                    "weekly_trade_ideas": ["watch tokyo trend continuation"],
                    "hybrid_pair_ideas": [
                        {
                            "symbol": "AUDJPY",
                            "session_focus": ["TOKYO", "SYDNEY"],
                            "setup_bias": "MOMENTUM_BREAKOUT",
                            "direction_bias": "LONG",
                            "conviction": 0.78,
                            "aggression_delta": 0.14,
                            "threshold_delta": -0.03,
                            "reason": "asia momentum plus supportive macro tape",
                        }
                    ],
                },
                None,
            )

            report = brain.run_cycle(
                now_utc=now,
                session_name="TOKYO",
                account_state={"equity": 550.0},
                runtime_state={
                    "symbols": ["USDJPY", "AUDJPY", "GBPUSD"],
                    "news_coverage_summary": {
                        "AUDJPY": {
                            "news_state": "NEWS_SAFE",
                            "news_source_used": "finnhub+supplemental",
                            "news_confidence": 0.78,
                            "news_headlines": [{"title": "Asia FX rallies on softer USD"}],
                        }
                    },
                },
                weekly_prep=True,
                force_local_retrain=True,
            )

            self.assertEqual(report.offline_gpt_status, "ok")
            self.assertEqual(report.promotion_bundle.mode, "local_live_offline_gpt")
            self.assertTrue((root / "brain" / "apex_learning_promotions.json").exists())
            self.assertTrue((root / "brain" / "apex_offline_gpt_review.json").exists())
            self.assertTrue((root / "brain" / "apex_weekly_ideas.json").exists())
            self.assertTrue((root / "brain" / "apex_portable_funded_profile.json").exists())
            self.assertIn("USDJPY", report.promotion_bundle.pair_directives)
            self.assertIn("meeting_mode", report.promotion_bundle.meeting_packet)
            self.assertIn("proof_window", report.promotion_bundle.meeting_packet)
            self.assertIn("lane_execution_summary", report.promotion_bundle.meeting_packet)
            self.assertIn("live_news_coverage_summary", report.promotion_bundle.meeting_packet)
            self.assertEqual(str(report.promotion_bundle.meeting_packet.get("current_session_name") or ""), "TOKYO")
            audjpy_directive = dict(report.promotion_bundle.pair_directives.get("AUDJPY") or {})
            self.assertIn("lane_state_machine", audjpy_directive)
            self.assertIn("walk_forward_scorecards", audjpy_directive)
            self.assertIn("loss_attribution_summary", audjpy_directive)
            self.assertIn("shadow_challenger_pool", audjpy_directive)
            self.assertIn("execution_quality_directives", audjpy_directive)
            self.assertTrue(bool(audjpy_directive.get("gpt_hybrid_advisory", {}).get("enabled")))
            self.assertGreaterEqual(
                int(audjpy_directive.get("frequency_directives", {}).get("soft_burst_target_10m", 0) or 0),
                5,
            )
            status = brain.status_snapshot()
            self.assertEqual(status["mode"], "local_live_offline_gpt")
            self.assertIn("trajectory_projection", status)
            self.assertGreaterEqual(int(status["monte_carlo_min_realities"]), 500)
            live_policy = brain.live_policy_snapshot(symbol="USDJPY", setup="USDJPY_MOMENTUM_IMPULSE")
            self.assertIn("bundle", live_policy)
            self.assertIn("reentry_watchlist", live_policy)
            self.assertIn("pair_directive", live_policy)
            self.assertIn("aggression_multiplier", live_policy)
            self.assertIn("portable_funded_profile", live_policy)
            self.assertIn("lane_state_machine", live_policy)
            brain.record_trade_feedback({"symbol": "USDJPY", "management_action": "TRAIL_STOP"})
            refreshed = brain.status_snapshot()
            self.assertGreaterEqual(int(refreshed["live_feedback_count"]), 1)

    def test_proof_window_can_auto_ramp_risk_after_five_good_days(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=1,
                promotion_min_samples=5,
            )
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=3,
                cooldown_minutes=1,
            )
            brain = ApexLearningBrain(
                online_learning=online,
                strategy_optimizer=optimizer,
                journal=_FakeJournal([]),
                data_dir=root / "brain",
                offline_gpt_enabled=False,
            )
            now = datetime(2026, 3, 21, 9, 0, tzinfo=timezone.utc)
            journal_rows: list[dict[str, object]] = []
            for day in range(5):
                trade_day = now - timedelta(days=day)
                for index in range(3):
                    trade_row = {
                        "timestamp_utc": trade_day.isoformat(),
                        "closed_at": trade_day.isoformat(),
                        "signal_id": f"proof-{day}-{index}",
                        "symbol": "USDJPY",
                        "timeframe": "M15",
                        "side": "BUY",
                        "entry": 150.0,
                        "sl": 149.8,
                        "tp": 150.4,
                        "exit": 150.2,
                        "pnl_r": 0.8,
                        "pnl_money": 0.8,
                        "news_state": "clear",
                        "session_state": "IN",
                        "session_name": "TOKYO",
                        "ai_decision": "approved",
                        "ai_probability": 0.75,
                        "spread_points": 10.0,
                        "lot": 0.01,
                        "regime": "TRENDING",
                        "setup": "USDJPY_MOMENTUM_IMPULSE",
                        "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                        "lane_name": "FX_SCALP",
                    }
                    journal_rows.append(dict(trade_row))
                    online.on_trade_close(trade_row)
            brain.journal = _FakeJournal(journal_rows)
            report = brain.run_cycle(
                now_utc=now,
                session_name="TOKYO",
                account_state={"equity": 550.0},
                runtime_state={"symbols": ["USDJPY"]},
                weekly_prep=False,
                force_local_retrain=True,
            )

            self.assertTrue(bool(report.promotion_bundle.local_summary.get("proof_window", {}).get("proof_ready_5d")))
            self.assertGreaterEqual(float(report.promotion_bundle.risk_pct_target), 0.035)

    def test_repeated_autonomy_failures_trigger_risk_reduction_and_shadow_variants(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=1,
                promotion_min_samples=5,
            )
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=3,
                cooldown_minutes=1,
            )
            brain = ApexLearningBrain(
                online_learning=online,
                strategy_optimizer=optimizer,
                journal=_FakeJournal([{"symbol": "XAUUSD", "setup": "XAUUSD_ADAPTIVE_M5_GRID", "pnl_r": -0.8}]),
                data_dir=root / "brain",
                offline_gpt_enabled=False,
            )
            now = datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc)
            runtime_state = {
                "symbols": ["XAUUSD", "USDJPY"],
                "no_trade_minutes": 180.0,
                "rolling_drawdown_pct": 0.07,
                "absolute_drawdown_pct": 0.09,
                "learner_status": "insufficient_class_balance",
                "bridge_singleton_status": {
                    "singleton_enforced": False,
                    "listener_conflict": True,
                    "owner_pid": 0,
                },
                "news_unknown_symbols": ["USDJPY"],
                "spread_spike_symbols": ["XAUUSD"],
            }

            first = brain.run_cycle(
                now_utc=now,
                session_name="LONDON",
                account_state={"equity": 120.0},
                runtime_state=runtime_state,
                weekly_prep=False,
                force_local_retrain=True,
            )
            second = brain.run_cycle(
                now_utc=now,
                session_name="LONDON",
                account_state={"equity": 120.0},
                runtime_state=runtime_state,
                weekly_prep=False,
                force_local_retrain=True,
            )

            self.assertFalse(first.promotion_bundle.risk_reduction_active)
            self.assertTrue(second.promotion_bundle.risk_reduction_active)
            self.assertEqual(len(second.promotion_bundle.shadow_strategy_variants), 50)
            self.assertIn("goal_state", brain.status_snapshot())
            self.assertIn("self_heal_actions", brain.status_snapshot())

    def test_trade_stall_and_learner_hold_do_not_force_risk_halving_without_critical_failure(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=1,
                promotion_min_samples=5,
            )
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=3,
                cooldown_minutes=1,
            )
            brain = ApexLearningBrain(
                online_learning=online,
                strategy_optimizer=optimizer,
                journal=_FakeJournal([{"symbol": "XAUUSD", "setup": "XAUUSD_ADAPTIVE_M5_GRID", "pnl_r": -0.2}]),
                data_dir=root / "brain",
                offline_gpt_enabled=False,
            )
            now = datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc)
            runtime_state = {
                "symbols": ["XAUUSD", "USDJPY", "EURUSD"],
                "no_trade_minutes": 180.0,
                "learner_status": "insufficient_class_balance",
                "bridge_singleton_status": {
                    "singleton_enforced": True,
                    "listener_conflict": False,
                    "owner_pid": 123,
                },
            }

            first = brain.run_cycle(
                now_utc=now,
                session_name="LONDON",
                account_state={"equity": 120.0},
                runtime_state=runtime_state,
                weekly_prep=False,
                force_local_retrain=True,
            )
            second = brain.run_cycle(
                now_utc=now,
                session_name="LONDON",
                account_state={"equity": 120.0},
                runtime_state=runtime_state,
                weekly_prep=False,
                force_local_retrain=True,
            )

            self.assertFalse(first.promotion_bundle.risk_reduction_active)
            self.assertFalse(second.promotion_bundle.risk_reduction_active)
            self.assertEqual(
                second.promotion_bundle.weak_pair_focus,
                ["XAUUSD", "USDJPY", "EURUSD"],
            )

    def test_pair_directives_capture_execution_quality_and_shadow_flags(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=1,
                promotion_min_samples=5,
            )
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=3,
                cooldown_minutes=1,
            )
            brain = ApexLearningBrain(
                online_learning=online,
                strategy_optimizer=optimizer,
                journal=_FakeJournal([{"symbol": "USOIL", "pnl_r": 0.5}]),
                data_dir=root / "brain",
                offline_gpt_enabled=False,
            )
            now = datetime(2026, 3, 21, 8, 0, tzinfo=timezone.utc)
            report = brain.run_cycle(
                now_utc=now,
                session_name="NEW_YORK",
                account_state={"equity": 220.0},
                runtime_state={
                    "symbol_runtime": {
                        "USOIL": {
                            "news_bias_direction": "bullish",
                            "news_confidence": 0.80,
                            "slippage_quality_score": 0.82,
                            "broker_reject_streak": 0,
                            "recent_opportunity_capture_gap_r": 0.72,
                            "recent_management_quality_score": 0.42,
                            "shadow_experiment_active": True,
                        }
                    }
                },
                force_local_retrain=True,
            )

            directive = report.promotion_bundle.pair_directives["USOIL"]
            self.assertEqual(str(directive.get("slippage_regime") or ""), "clean")
            self.assertEqual(str(directive.get("broker_reject_risk") or ""), "normal")
            self.assertGreaterEqual(float(directive.get("opportunity_capture_gap_r") or 0.0), 0.70)
            self.assertTrue(bool(directive.get("shadow_experiment_active")))

    def test_hour_expectancy_matrix_feeds_pair_and_setup_directives(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=1,
                promotion_min_samples=5,
            )
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=3,
                cooldown_minutes=1,
            )
            lane_rows = [
                {
                    "symbol": "USDJPY",
                    "setup": "USDJPY_MOMENTUM_IMPULSE",
                    "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                    "closed_at": datetime(2026, 3, 20, 22, 0, tzinfo=timezone.utc).isoformat(),
                    "pnl_r": 0.9,
                    "mfe_r": 1.4,
                    "mae_r": 0.2,
                },
                {
                    "symbol": "USDJPY",
                    "setup": "USDJPY_MOMENTUM_IMPULSE",
                    "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                    "closed_at": datetime(2026, 3, 20, 22, 20, tzinfo=timezone.utc).isoformat(),
                    "pnl_r": 0.7,
                    "mfe_r": 1.2,
                    "mae_r": 0.2,
                },
                {
                    "symbol": "USDJPY",
                    "setup": "USDJPY_MOMENTUM_IMPULSE",
                    "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                    "closed_at": datetime(2026, 3, 21, 3, 0, tzinfo=timezone.utc).isoformat(),
                    "pnl_r": -0.6,
                    "mfe_r": 0.2,
                    "mae_r": 0.7,
                },
            ]
            brain = ApexLearningBrain(
                online_learning=online,
                strategy_optimizer=optimizer,
                journal=_FakeJournal(lane_rows),
                data_dir=root / "brain",
                offline_gpt_enabled=False,
            )

            report = brain.run_cycle(
                now_utc=datetime(2026, 3, 21, 9, 0, tzinfo=timezone.utc),
                session_name="TOKYO",
                account_state={"equity": 220.0},
                runtime_state={"symbols": ["USDJPY"]},
                force_local_retrain=True,
            )

            matrix = dict(report.promotion_bundle.meeting_packet.get("hour_expectancy_matrix") or {})
            self.assertIn("USDJPY:USDJPY_MOMENTUM_IMPULSE", matrix)
            directive = report.promotion_bundle.pair_directives["USDJPY"]
            self.assertGreater(float(directive.get("hour_expectancy_score") or 0.0), 0.0)
            self.assertIn("strong_hours_sydney", directive)
            self.assertIn("weak_hours_sydney", directive)
            live_policy = brain.live_policy_snapshot(symbol="USDJPY", setup="USDJPY_MOMENTUM_IMPULSE")
            self.assertIn("setup_hour_directive", live_policy)
            self.assertGreaterEqual(float(live_policy["setup_hour_directive"].get("lane_expectancy_multiplier") or 0.0), 0.8)

    def test_hot_hand_and_profit_recycle_bias_feed_pair_directives(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=1,
                promotion_min_samples=5,
            )
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=3,
                cooldown_minutes=1,
            )
            journal_rows = [
                {
                    "symbol": "XAUUSD",
                    "setup": "XAUUSD_ADAPTIVE_M5_GRID",
                    "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                    "session_name": "LONDON",
                    "closed_at": datetime(2026, 3, 20, 9, 0, tzinfo=timezone.utc).isoformat(),
                    "pnl_r": 1.1,
                    "mfe_r": 1.8,
                    "mae_r": 0.3,
                },
                {
                    "symbol": "XAUUSD",
                    "setup": "XAUUSD_ADAPTIVE_M5_GRID",
                    "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                    "session_name": "LONDON",
                    "closed_at": datetime(2026, 3, 21, 9, 15, tzinfo=timezone.utc).isoformat(),
                    "pnl_r": 0.9,
                    "mfe_r": 1.6,
                    "mae_r": 0.2,
                },
                {
                    "symbol": "XAUUSD",
                    "setup": "XAUUSD_ADAPTIVE_M5_GRID",
                    "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                    "session_name": "LONDON",
                    "closed_at": datetime(2026, 3, 22, 9, 20, tzinfo=timezone.utc).isoformat(),
                    "pnl_r": 0.8,
                    "mfe_r": 1.5,
                    "mae_r": 0.2,
                },
                {
                    "symbol": "XAUUSD",
                    "setup": "XAUUSD_ADAPTIVE_M5_GRID",
                    "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                    "session_name": "LONDON",
                    "closed_at": datetime(2026, 3, 22, 10, 5, tzinfo=timezone.utc).isoformat(),
                    "pnl_r": 0.7,
                    "mfe_r": 1.2,
                    "mae_r": 0.1,
                },
                {
                    "symbol": "XAUUSD",
                    "setup": "XAUUSD_ADAPTIVE_M5_GRID",
                    "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                    "session_name": "LONDON",
                    "closed_at": datetime(2026, 3, 22, 10, 25, tzinfo=timezone.utc).isoformat(),
                    "pnl_r": 0.6,
                    "mfe_r": 1.1,
                    "mae_r": 0.1,
                },
            ]
            brain = ApexLearningBrain(
                online_learning=online,
                strategy_optimizer=optimizer,
                journal=_FakeJournal(journal_rows),
                data_dir=root / "brain",
                offline_gpt_enabled=False,
            )

            report = brain.run_cycle(
                now_utc=datetime(2026, 3, 22, 10, 0, tzinfo=timezone.utc),
                session_name="LONDON",
                account_state={"equity": 320.0},
                runtime_state={"symbols": ["XAUUSD"]},
                force_local_retrain=True,
            )

            directive = dict(report.promotion_bundle.pair_directives.get("XAUUSD") or {})
            self.assertTrue(bool(directive.get("hot_hand_active")))
            self.assertGreater(float(directive.get("session_bankroll_bias") or 1.0), 1.0)
            self.assertTrue(bool(directive.get("profit_recycle_active")))
            self.assertGreater(float(directive.get("profit_recycle_boost") or 0.0), 0.0)

    def test_idle_native_lane_gets_undertrade_recovery_directives(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            online = OnlineLearningEngine(
                data_path=root / "trades.csv",
                model_path=root / "online_model.pkl",
                setup_log_path=root / "setups_log.csv",
                min_retrain_trades=1,
                promotion_min_samples=5,
            )
            optimizer = StrategyOptimizer(
                trade_history_path=root / "trade_history.json",
                metrics_path=root / "strategy_metrics.json",
                min_trades_per_strategy=3,
                cooldown_minutes=1,
            )
            brain = ApexLearningBrain(
                online_learning=online,
                strategy_optimizer=optimizer,
                journal=_FakeJournal([]),
                data_dir=root / "brain",
                offline_gpt_enabled=False,
            )

            report = brain.run_cycle(
                now_utc=datetime(2026, 3, 23, 1, 0, tzinfo=timezone.utc),
                session_name="TOKYO",
                account_state={"equity": 320.0},
                runtime_state={
                    "symbols": ["AUDJPY"],
                    "symbol_runtime": {
                        "AUDJPY": {
                            "today_closed_trade_count": 0,
                            "session_trade_count": 0,
                        }
                    },
                },
                force_local_retrain=True,
            )

            directive = dict(report.promotion_bundle.pair_directives.get("AUDJPY") or {})
            frequency = dict(directive.get("frequency_directives") or {})
            self.assertTrue(bool(frequency.get("idle_lane_recovery_active")))
            self.assertTrue(bool(frequency.get("undertrade_fix_mode")))
            self.assertTrue(bool(frequency.get("quota_boost_allowed")))
            self.assertGreaterEqual(int(frequency.get("soft_burst_target_10m", 0) or 0), 5)
