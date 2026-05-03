from __future__ import annotations

import os
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from src.bridge_stop_validation import load_symbol_rules
from src.strategy_engine import SignalCandidate

try:
    from src.main import (
        _apply_runtime_account_snapshot,
        _approve_small_session_allowed,
        _augment_learning_policy_for_density,
        _btc_weekend_force_candidate,
        _candidate_strategy_pool_rankings,
        _detect_account_scaling_update,
        _equity_momentum_throttle,
        _effective_cooldown_trades_remaining,
        _effective_live_entry_price,
        _effective_min_stop_distance_points,
        _expand_market_universe,
        _family_rotation_penalty,
        _filter_broker_confirmed_positions,
        _is_always_on_symbol,
        _fallback_account_snapshot,
        _is_weekend_market_mode,
        _market_open_status,
        _micro_lot_cap,
        _micro_cooldown_minutes,
        _micro_position_caps,
        _phase_state,
        _pair_session_performance_state,
        _pair_strategy_session_performance_state,
        _normalize_pre_risk_exit_geometry,
        _normalize_runtime_spread_points,
        _normalize_symbol_key,
        _preserve_approved_broker_min_lot,
        _prep_checks_complete,
        _resolve_candidate_stop_distance,
        _resolve_bridge_symbol_snapshot,
        _resolve_runtime_symbol_info,
        _resolve_timeframe_route,
        _runtime_bootstrap_tolerance_cap,
        _runtime_bootstrap_trade_allowed,
        _quality_tier_exit_profile,
        _symbol_entry_cap,
        _strategy_exit_profile,
        _streak_adjustment_mode,
        _summarize_strategy_pool_rankings,
        _velocity_decay_profile,
        _warm_market_universe_history,
        _load_symbol_frames,
        _soft_kill_recovery_note,
        _write_runtime_heartbeat,
        build_runtime,
        determine_trading_state,
        run_bridge_only,
    )
    _HAS_MAIN_DEPS = True
    _SKIP_REASON = ""
except ModuleNotFoundError as exc:
    build_runtime = None  # type: ignore
    determine_trading_state = None  # type: ignore
    run_bridge_only = None  # type: ignore
    _apply_runtime_account_snapshot = None  # type: ignore
    _approve_small_session_allowed = None  # type: ignore
    _augment_learning_policy_for_density = None  # type: ignore
    _btc_weekend_force_candidate = None  # type: ignore
    _detect_account_scaling_update = None  # type: ignore
    _equity_momentum_throttle = None  # type: ignore
    _effective_cooldown_trades_remaining = None  # type: ignore
    _effective_min_stop_distance_points = None  # type: ignore
    _effective_live_entry_price = None  # type: ignore
    _expand_market_universe = None  # type: ignore
    _family_rotation_penalty = None  # type: ignore
    _filter_broker_confirmed_positions = None  # type: ignore
    _micro_position_caps = None  # type: ignore
    _micro_lot_cap = None  # type: ignore
    _micro_cooldown_minutes = None  # type: ignore
    _normalize_pre_risk_exit_geometry = None  # type: ignore
    _normalize_runtime_spread_points = None  # type: ignore
    _normalize_symbol_key = None  # type: ignore
    _prep_checks_complete = None  # type: ignore
    _resolve_candidate_stop_distance = None  # type: ignore
    _resolve_bridge_symbol_snapshot = None  # type: ignore
    _resolve_runtime_symbol_info = None  # type: ignore
    _is_always_on_symbol = None  # type: ignore
    _fallback_account_snapshot = None  # type: ignore
    _is_weekend_market_mode = None  # type: ignore
    _market_open_status = None  # type: ignore
    _resolve_timeframe_route = None  # type: ignore
    _runtime_bootstrap_tolerance_cap = None  # type: ignore
    _runtime_bootstrap_trade_allowed = None  # type: ignore
    _quality_tier_exit_profile = None  # type: ignore
    _symbol_entry_cap = None  # type: ignore
    _strategy_exit_profile = None  # type: ignore
    _streak_adjustment_mode = None  # type: ignore
    _velocity_decay_profile = None  # type: ignore
    _warm_market_universe_history = None  # type: ignore
    _load_symbol_frames = None  # type: ignore
    _phase_state = None  # type: ignore
    _pair_strategy_session_performance_state = None  # type: ignore
    _soft_kill_recovery_note = None  # type: ignore
    _write_runtime_heartbeat = None  # type: ignore
    _preserve_approved_broker_min_lot = None  # type: ignore
    _HAS_MAIN_DEPS = False
    _SKIP_REASON = f"missing dependency: {exc.name}"


@unittest.skipUnless(_HAS_MAIN_DEPS, _SKIP_REASON)
class MainDryRunTests(unittest.TestCase):
    def test_btc_weekend_force_candidate_uses_wall_clock_emit_time_not_stale_bar_time(self) -> None:
        row = pd.Series(
            {
                "m5_close": 66475.0,
                "m5_atr_14": 42.0,
                "m5_ema_20": 66440.0,
                "m5_ema_50": 66410.0,
                "h1_ema_20": 66460.0,
                "h1_ema_50": 66380.0,
                "m5_ret_1": 0.0004,
                "m15_ret_1": 0.0003,
                "m5_body_efficiency": 0.42,
                "market_instability_score": 0.22,
                "m15_range_position_20": 0.61,
                "m5_spread_ratio_20": 1.1,
                "m5_atr_pct_of_avg": 1.02,
            }
        )
        stale_bar_time = datetime(2026, 3, 28, 6, 0, tzinfo=timezone.utc)

        first = _btc_weekend_force_candidate(
            symbol="BTCUSD",
            row=row,
            session_name="TOKYO",
            timestamp=stale_bar_time,
            emit_time=datetime(2026, 3, 28, 6, 0, tzinfo=timezone.utc),
            cadence_seconds=90,
        )
        second = _btc_weekend_force_candidate(
            symbol="BTCUSD",
            row=row,
            session_name="TOKYO",
            timestamp=stale_bar_time,
            emit_time=datetime(2026, 3, 28, 6, 2, tzinfo=timezone.utc),
            cadence_seconds=90,
        )

        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertNotEqual(str(first.signal_id), str(second.signal_id))

    def test_btc_weekend_force_candidate_can_emit_sell_on_high_rejection(self) -> None:
        row = pd.Series(
            {
                "m5_close": 66510.0,
                "m5_atr_14": 40.0,
                "m5_ema_20": 66480.0,
                "m5_ema_50": 66460.0,
                "h1_ema_20": 66410.0,
                "h1_ema_50": 66405.0,
                "m5_ret_1": -0.00010,
                "m15_ret_1": -0.00005,
                "m5_body_efficiency": 0.16,
                "market_instability_score": 0.24,
                "m15_range_position_20": 0.84,
                "m5_spread_ratio_20": 1.05,
                "m5_atr_pct_of_avg": 1.08,
                "m5_rsi_14": 62.0,
                "m5_upper_wick_ratio": 0.42,
                "m5_lower_wick_ratio": 0.06,
            }
        )

        candidate = _btc_weekend_force_candidate(
            symbol="BTCUSD",
            row=row,
            session_name="TOKYO",
            timestamp=datetime(2026, 3, 28, 7, 0, tzinfo=timezone.utc),
            emit_time=datetime(2026, 3, 28, 7, 0, tzinfo=timezone.utc),
            cadence_seconds=0,
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate.side, "SELL")

    def test_micro_cooldown_minutes_relaxes_weekend_btc(self) -> None:
        loss_minutes, win_minutes = _micro_cooldown_minutes(
            "BTCUSD",
            datetime(2026, 3, 28, 7, 0, tzinfo=timezone.utc),
            {"cooldown_minutes_after_loss": 20, "cooldown_minutes_after_win": 5},
        )

        self.assertEqual(loss_minutes, 1)
        self.assertEqual(win_minutes, 0)

    def test_expand_market_universe_skips_live_startup_discovery_by_default(self) -> None:
        settings = SimpleNamespace(
            section=lambda name: {
                "mode": "LIVE",
                "market_universe": {
                    "enabled": True,
                    "live_startup_discovery_enabled": False,
                },
            }
        )
        mt5_client = SimpleNamespace(discover_symbol_universe=lambda **_kwargs: (_ for _ in ()).throw(AssertionError("should not discover at live startup")))
        logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

        symbols, discovered = _expand_market_universe(
            settings=settings,
            mt5_client=mt5_client,
            configured_symbols=["XAUUSD", "BTCUSD"],
            dry_run=False,
            logger=logger,
        )

        self.assertEqual(symbols, ["XAUUSD", "BTCUSD"])
        self.assertEqual(discovered, [])

    def test_warm_market_universe_history_skips_live_startup_warmup_by_default(self) -> None:
        settings = SimpleNamespace(
            section=lambda name: {
                "mode": "LIVE",
                "market_universe": {
                    "history_warmup_enabled": True,
                    "live_history_warmup_enabled": False,
                    "history_warmup_timeframes": {"M5": 600},
                    "history_warmup_max_symbols": 8,
                },
            }
        )
        market_data = SimpleNamespace(
            load_cached=lambda *_args, **_kwargs: None,
            fetch=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not warm cache at live startup")),
        )
        logger = SimpleNamespace(info=lambda *_args, **_kwargs: None)

        summary = _warm_market_universe_history(
            settings=settings,
            market_data=market_data,
            resolved_symbols={"XAUUSD": "XAUUSD+"},
            dry_run=False,
            logger=logger,
        )

        self.assertFalse(summary["enabled"])
        self.assertEqual(summary["skipped_reason"], "live_startup_skip")

    def test_run_bridge_only_passes_learning_brain_into_bridge_runtime(self) -> None:
        fake_brain = object()
        fake_settings = SimpleNamespace(
            raw={"dashboard": {}},
            section=lambda name: {"enabled": False} if name == "execution" else {},
            resolve_path_value=lambda value: value,
        )
        fake_runtime = {
            "bridge_config": {"enabled": True, "host": "127.0.0.1", "port": 8000, "auth_token": "", "orchestrator": {}},
            "settings": fake_settings,
            "bridge_queue": object(),
            "journal": object(),
            "online_learning": object(),
            "learning_brain": fake_brain,
            "session_profile": SimpleNamespace(infer_name=lambda *_args, **_kwargs: "LONDON"),
            "ai_gate": object(),
            "logger": object(),
            "strategy_optimizer": object(),
            "market_data": SimpleNamespace(status_snapshot=lambda: {}),
        }

        with patch("src.main.build_runtime", return_value=fake_runtime), patch("src.bridge_server.run_bridge_forever") as run_bridge:
            result = run_bridge_only()

        self.assertEqual(result, 0)
        self.assertIs(run_bridge.call_args.kwargs["learning_brain"], fake_brain)

    def test_phase_state_fast_ramps_to_two_percent_at_five_hundred_equity(self) -> None:
        phase = _phase_state(
            500.0,
            {"overall": {"trades": 35, "win_rate": 0.56, "expectancy_r": 0.11}},
        )

        self.assertEqual(str(phase.get("current_phase") or ""), "PHASE_3")
        self.assertAlmostEqual(float(phase.get("current_risk_pct") or 0.0), 0.02, places=6)
        self.assertAlmostEqual(float(phase.get("current_max_risk_pct") or 0.0), 0.025, places=6)

    def test_phase_state_unlocks_full_aggression_after_multi_day_green_streak(self) -> None:
        phase = _phase_state(
            550.0,
            {
                "overall": {"trades": 40, "win_rate": 0.58, "expectancy_r": 0.10},
                "daily_green_streak": 3,
            },
        )

        self.assertEqual(str(phase.get("current_phase") or ""), "PHASE_3")
        self.assertAlmostEqual(float(phase.get("current_risk_pct") or 0.0), 0.035, places=6)
        self.assertAlmostEqual(float(phase.get("current_max_risk_pct") or 0.0), 0.04, places=6)
        self.assertEqual(int(phase.get("daily_green_streak") or 0), 3)

    def test_phase_state_enables_smart_scaling_after_two_profitable_days(self) -> None:
        baseline = _phase_state(
            140.0,
            {
                "overall": {"trades": 18, "win_rate": 0.52, "expectancy_r": 0.02},
                "daily_green_streak": 1,
            },
        )
        ramped = _phase_state(
            140.0,
            {
                "overall": {"trades": 18, "win_rate": 0.52, "expectancy_r": 0.02},
                "daily_green_streak": 2,
            },
        )

        self.assertFalse(bool(baseline.get("smart_scaling_ready")))
        self.assertTrue(bool(ramped.get("smart_scaling_ready")))
        self.assertEqual(str(ramped.get("smart_scaling_mode") or ""), "PROVEN_2DAY_RAMP")
        self.assertEqual(str(ramped.get("current_compounding_state") or ""), "validated_density_ramp")
        self.assertGreater(int(ramped.get("current_daily_trade_cap") or 0), int(baseline.get("current_daily_trade_cap") or 0))
        self.assertGreater(int(ramped.get("stretch_daily_trade_target") or 0), int(baseline.get("stretch_daily_trade_target") or 0))
        self.assertGreater(int(ramped.get("hourly_stretch_target") or 0), int(baseline.get("hourly_stretch_target") or 0))

    def test_phase_state_keeps_smart_scaling_off_when_trade_gate_is_not_met(self) -> None:
        phase = _phase_state(
            140.0,
            {
                "overall": {"trades": 11, "win_rate": 0.60, "expectancy_r": 0.08},
                "daily_green_streak": 2,
            },
        )

        self.assertFalse(bool(phase.get("smart_scaling_ready")))
        self.assertEqual(str(phase.get("smart_scaling_mode") or ""), "BASELINE_XAU_LEAD")
        self.assertEqual(int(phase.get("smart_scaling_soft_burst_bonus") or 0), 0)

    def test_augment_learning_policy_for_density_makes_xau_aggressive_from_start(self) -> None:
        policy = _augment_learning_policy_for_density(
            symbol_key="XAUUSD",
            session_name="LONDON",
            learning_policy={},
            current_scaling_state={"current_phase": "PHASE_1", "smart_scaling_ready": False},
        )

        pair_directive = dict(policy.get("pair_directive") or {})
        frequency_directives = dict(pair_directive.get("frequency_directives") or {})
        self.assertGreater(float(pair_directive.get("aggression_multiplier") or 0.0), 1.30)
        self.assertTrue(bool(pair_directive.get("proof_lane_ready")))
        self.assertTrue(bool(pair_directive.get("hot_hand_active")))
        self.assertEqual(str(pair_directive.get("trade_horizon_bias") or ""), "scalp")
        self.assertGreaterEqual(int(frequency_directives.get("soft_burst_target_10m") or 0), 14)

    def test_augment_learning_policy_for_density_ramps_other_pairs_after_proof(self) -> None:
        scaling_state = _phase_state(
            140.0,
            {
                "overall": {"trades": 18, "win_rate": 0.54, "expectancy_r": 0.04},
                "daily_green_streak": 2,
            },
        )

        policy = _augment_learning_policy_for_density(
            symbol_key="AUDNZD",
            session_name="TOKYO",
            learning_policy={"pair_directive": {"frequency_directives": {"soft_burst_target_10m": 2}}},
            current_scaling_state=scaling_state,
        )

        pair_directive = dict(policy.get("pair_directive") or {})
        frequency_directives = dict(pair_directive.get("frequency_directives") or {})
        self.assertGreater(int(pair_directive.get("density_entry_cap_bonus") or 0), 1)
        self.assertTrue(bool(pair_directive.get("profit_recycle_active")))
        self.assertGreater(float(pair_directive.get("aggression_multiplier") or 0.0), 1.20)
        self.assertTrue(bool(frequency_directives.get("idle_lane_recovery_active")))
        self.assertGreaterEqual(int(frequency_directives.get("soft_burst_target_10m") or 0), 5)

    def test_strategy_exit_profile_uses_lane_specific_rr_bands(self) -> None:
        xau_profile = _strategy_exit_profile(
            symbol_key="XAUUSD",
            strategy_key="XAUUSD_ADAPTIVE_M5_GRID",
            quality_tier="A+",
            exits_config={"trail_activation_r": 1.0, "trail_atr": 1.0},
        )
        btc_profile = _strategy_exit_profile(
            symbol_key="BTCUSD",
            strategy_key="BTCUSD_PRICE_ACTION_CONTINUATION",
            quality_tier="A",
            exits_config={"trail_activation_r": 1.0, "trail_atr": 1.0},
        )
        nas_profile = _strategy_exit_profile(
            symbol_key="NAS100",
            strategy_key="NAS100_LIQUIDITY_SWEEP_REVERSAL",
            quality_tier="B",
            exits_config={"trail_activation_r": 1.0, "trail_atr": 1.0},
        )

        self.assertEqual(str(xau_profile.get("approved_rr_target") or ""), "2.6-3.2")
        self.assertEqual(str(btc_profile.get("approved_rr_target") or ""), "2.0-2.4")
        self.assertEqual(str(nas_profile.get("approved_rr_target") or ""), "1.5-1.9")
        self.assertEqual(float(xau_profile.get("breakeven_trigger_r") or 0.0), 0.78)
        self.assertEqual(float(xau_profile.get("basket_take_profit_r") or 0.0), 2.10)

    def test_write_runtime_heartbeat_persists_runtime_file(self) -> None:
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as tmp_dir:
            heartbeat_path = Path(tmp_dir) / "heartbeat.json"
            _write_runtime_heartbeat(
                heartbeat_path,
                now=datetime(2026, 3, 21, 9, 0, tzinfo=timezone.utc),
                mode="LIVE",
                account_label="primary",
                account_state={"equity": 125.5, "balance": 120.0, "margin_free": 118.0},
                summary={"loops": 4, "accepted": 2, "rejected": 1, "errors": 0},
                extra={"session_name": "LONDON"},
            )

            payload = json.loads(heartbeat_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["mode"], "LIVE")
            self.assertEqual(payload["account_label"], "primary")
            self.assertEqual(payload["session_name"], "LONDON")
            self.assertAlmostEqual(float(payload["equity"]), 125.5, places=6)

    def test_effective_cooldown_trades_remaining_expires_after_six_hours(self) -> None:
        stats = SimpleNamespace(
            cooldown_trades_remaining=2,
            today_closed_trade_times_raw=["2026-03-21T00:00:00+00:00"],
        )

        remaining = _effective_cooldown_trades_remaining(
            stats,
            now=datetime(2026, 3, 21, 7, 5, tzinfo=timezone.utc),
        )

        self.assertEqual(remaining, 0)

    def test_effective_cooldown_trades_remaining_keeps_recent_block(self) -> None:
        stats = SimpleNamespace(
            cooldown_trades_remaining=2,
            today_closed_trade_times_raw=["2026-03-21T05:30:00+00:00"],
        )

        remaining = _effective_cooldown_trades_remaining(
            stats,
            now=datetime(2026, 3, 21, 7, 5, tzinfo=timezone.utc),
        )

        self.assertEqual(remaining, 2)

    def test_strategy_exit_profile_applies_streak_bias_for_xau_grid(self) -> None:
        streak_profile = _strategy_exit_profile(
            symbol_key="XAUUSD",
            strategy_key="XAUUSD_ADAPTIVE_M5_GRID",
            quality_tier="A+",
            exits_config={"trail_activation_r": 1.0, "trail_atr": 1.0},
            streak_adjust_mode="WIN_STREAK",
        )
        defensive_profile = _strategy_exit_profile(
            symbol_key="XAUUSD",
            strategy_key="XAUUSD_ADAPTIVE_M5_GRID",
            quality_tier="A+",
            exits_config={"trail_activation_r": 1.0, "trail_atr": 1.0},
            streak_adjust_mode="LOSS_STREAK",
        )

        self.assertEqual(str(streak_profile.get("approved_rr_target") or ""), "3.0-3.6")
        self.assertEqual(str(defensive_profile.get("approved_rr_target") or ""), "1.9-2.4")
        self.assertGreater(float(streak_profile.get("trail_activation_r") or 0.0), float(defensive_profile.get("trail_activation_r") or 0.0))

    def test_strategy_exit_profile_uses_weekend_btc_rr_band(self) -> None:
        btc_profile = _strategy_exit_profile(
            symbol_key="BTCUSD",
            strategy_key="BTCUSD_PRICE_ACTION_CONTINUATION",
            quality_tier="A",
            exits_config={"trail_activation_r": 1.0, "trail_atr": 1.0},
            weekend_mode=True,
        )

        self.assertEqual(str(btc_profile.get("approved_rr_target") or ""), "2.2-2.8")
        self.assertEqual(float(btc_profile.get("breakeven_trigger_r") or 0.0), 0.86)

    def test_strategy_exit_profile_uses_xau_mean_reversion_sweep_capture_profile(self) -> None:
        xau_profile = _strategy_exit_profile(
            symbol_key="XAUUSD",
            strategy_key="XAUUSD_LONDON_LIQUIDITY_SWEEP",
            quality_tier="A",
            exits_config={"trail_activation_r": 1.0, "trail_atr": 1.0},
            session_name="LONDON",
            regime_state="MEAN_REVERSION",
        )

        self.assertEqual(str(xau_profile.get("approved_rr_target") or ""), "1.7-2.2")
        self.assertEqual(float(xau_profile.get("basket_take_profit_r") or 0.0), 0.95)
        self.assertEqual(int(xau_profile.get("time_stop_bars") or 0), 8)

    def test_strategy_exit_profile_uses_gbpusd_london_chop_breakout_profile(self) -> None:
        gbp_profile = _strategy_exit_profile(
            symbol_key="GBPUSD",
            strategy_key="GBPUSD_LONDON_EXPANSION_BREAKOUT",
            quality_tier="A",
            exits_config={"trail_activation_r": 1.0, "trail_atr": 1.0},
            session_name="LONDON",
            regime_state="LOW_LIQUIDITY_CHOP",
        )

        self.assertEqual(str(gbp_profile.get("approved_rr_target") or ""), "1.6-2.0")
        self.assertEqual(float(gbp_profile.get("basket_take_profit_r") or 0.0), 1.0)
        self.assertEqual(int(gbp_profile.get("time_stop_bars") or 0), 9)

    def test_strategy_exit_profile_uses_audjpy_sweep_profile(self) -> None:
        audjpy_profile = _strategy_exit_profile(
            symbol_key="AUDJPY",
            strategy_key="AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
            quality_tier="A",
            exits_config={"trail_activation_r": 1.0, "trail_atr": 1.0},
            session_name="OVERLAP",
            regime_state="MEAN_REVERSION",
        )

        self.assertEqual(str(audjpy_profile.get("approved_rr_target") or ""), "1.5-1.9")
        self.assertEqual(float(audjpy_profile.get("basket_take_profit_r") or 0.0), 1.02)
        self.assertEqual(int(audjpy_profile.get("time_stop_bars") or 0), 8)

    def test_strategy_exit_profile_uses_super_aggro_overlay_for_usdjpy_home_session(self) -> None:
        profile = _strategy_exit_profile(
            symbol_key="USDJPY",
            strategy_key="USDJPY_MOMENTUM_IMPULSE",
            quality_tier="A",
            exits_config={"trail_activation_r": 1.0, "trail_atr": 1.0},
            session_name="TOKYO",
            regime_state="TRENDING",
        )

        self.assertEqual(str(profile.get("approved_rr_target") or ""), "2.2-2.9")
        self.assertGreaterEqual(float(profile.get("trail_backoff_r") or 0.0), 0.72)
        self.assertGreaterEqual(float(profile.get("basket_take_profit_r") or 0.0), 1.65)

    def test_normalize_symbol_key_handles_eurgbp(self) -> None:
        self.assertEqual(_normalize_symbol_key("EURGBP.a"), "EURGBP")

    def test_velocity_decay_only_applies_to_btc(self) -> None:
        closed_trades = []
        for index in range(12):
            closed_trades.append(
                {
                    "symbol": "BTCUSD",
                    "strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION",
                    "setup_family": "TREND",
                    "closed_at": f"2026-03-14T0{index % 9}:00:00+00:00",
                    "pnl_r": 0.4,
                }
            )
        btc_profile = _velocity_decay_profile(
            symbol_key="BTCUSD",
            strategy_key="BTCUSD_PRICE_ACTION_CONTINUATION",
            setup_family="TREND",
            closed_trades=closed_trades,
            row_timestamp="2026-03-14T10:00:00+00:00",
            weekend_mode=True,
        )
        fx_profile = _velocity_decay_profile(
            symbol_key="EURUSD",
            strategy_key="EURUSD_LONDON_BREAKOUT",
            setup_family="TREND",
            closed_trades=closed_trades,
            row_timestamp="2026-03-14T10:00:00+00:00",
            weekend_mode=False,
        )

        self.assertLess(float(btc_profile.get("size_multiplier") or 1.0), 1.0)
        self.assertEqual(float(fx_profile.get("size_multiplier") or 1.0), 1.0)

    def test_streak_adjustment_mode_uses_symbol_recent_results(self) -> None:
        trades = [
            {"symbol": "XAUUSD", "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID", "pnl_r": 0.6},
            {"symbol": "XAUUSD", "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID", "pnl_r": 0.7},
            {"symbol": "XAUUSD", "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID", "pnl_r": 0.5},
        ]
        streak = _streak_adjustment_mode(
            closed_trades=trades,
            symbol_key="XAUUSD",
            strategy_key="XAUUSD_ADAPTIVE_M5_GRID",
            orchestrator_config={"xau_grid_win_streak_threshold": 3, "xau_grid_loss_streak_threshold": 2},
        )

        self.assertEqual(str(streak.get("mode") or ""), "WIN_STREAK")

    def test_strategy_pool_rankings_prefer_better_combined_strategy_score(self) -> None:
        stronger = SignalCandidate(
            signal_id="audjpy-strong",
            setup="AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
            side="BUY",
            score_hint=0.70,
            reason="strong combined quality",
            stop_atr=1.0,
            tp_r=1.8,
            strategy_family="TREND",
            meta={
                "strategy_key": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                "router_rank_score": 0.70,
                "regime_state": "BREAKOUT_EXPANSION",
                "regime_fit": 0.95,
                "session_fit": 0.95,
                "volatility_fit": 0.90,
                "pair_behavior_fit": 0.88,
                "execution_quality_fit": 0.92,
                "entry_timing_score": 0.91,
                "structure_cleanliness_score": 0.93,
                "strategy_recent_performance_seed": 0.89,
            },
        )
        weaker = SignalCandidate(
            signal_id="audjpy-weak",
            setup="AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
            side="BUY",
            score_hint=0.78,
            reason="higher raw, weaker combined quality",
            stop_atr=1.0,
            tp_r=1.5,
            strategy_family="TREND",
            meta={
                "strategy_key": "AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
                "router_rank_score": 0.78,
                "regime_state": "BREAKOUT_EXPANSION",
                "regime_fit": 0.55,
                "session_fit": 0.75,
                "volatility_fit": 0.52,
                "pair_behavior_fit": 0.58,
                "execution_quality_fit": 0.60,
                "entry_timing_score": 0.49,
                "structure_cleanliness_score": 0.46,
                "strategy_recent_performance_seed": 0.44,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="AUDJPY",
            candidates=[weaker, stronger],
            session_name="TOKYO",
        )

        self.assertEqual(str(ranked[0][1]["strategy_key"]), "AUDJPY_TOKYO_MOMENTUM_BREAKOUT")
        self.assertGreater(float(ranked[0][1]["strategy_score"]), float(ranked[1][1]["strategy_score"]))
        self.assertEqual(int(ranked[0][0].meta.get("strategy_pool_rank") or 0), 1)

    def test_strategy_pool_rankings_apply_local_loser_bucket_demotions_early(self) -> None:
        punished = SignalCandidate(
            signal_id="eurusd-loser-bucket",
            setup="EURUSD_LONDON_BREAKOUT",
            side="BUY",
            score_hint=0.77,
            reason="raw score is high but local loser bucket should demote it",
            stop_atr=1.0,
            tp_r=1.6,
            strategy_family="TREND",
            meta={
                "strategy_key": "EURUSD_LONDON_BREAKOUT",
                "router_rank_score": 0.77,
                "regime_state": "LOW_LIQUIDITY_CHOP",
                "regime_fit": 0.74,
                "session_fit": 0.58,
                "volatility_fit": 0.56,
                "pair_behavior_fit": 0.60,
                "execution_quality_fit": 0.84,
                "entry_timing_score": 0.78,
                "structure_cleanliness_score": 0.76,
                "strategy_recent_performance_seed": 0.63,
            },
        )
        cleaner = SignalCandidate(
            signal_id="eurusd-cleaner",
            setup="EURUSD_RANGE_FADE",
            side="SELL",
            score_hint=0.69,
            reason="slightly lower raw score but cleaner local fit",
            stop_atr=1.0,
            tp_r=1.4,
            strategy_family="RANGE",
            meta={
                "strategy_key": "EURUSD_RANGE_FADE",
                "router_rank_score": 0.69,
                "regime_state": "MEAN_REVERSION",
                "regime_fit": 0.87,
                "session_fit": 0.72,
                "volatility_fit": 0.76,
                "pair_behavior_fit": 0.74,
                "execution_quality_fit": 0.88,
                "entry_timing_score": 0.82,
                "structure_cleanliness_score": 0.84,
                "strategy_recent_performance_seed": 0.67,
            },
        )
        closed_trades = [
            {
                "symbol": "EURUSD",
                "strategy_key": "EURUSD_LONDON_BREAKOUT",
                "session_name": "TOKYO",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_amount": -10.0,
                "pnl_r": -1.0,
                "close_context_json": "{\"failure_buckets\":[\"late_entry\",\"false_breakout\",\"poor_structure\"]}",
                "management_effect_json": "{\"management_quality\":0.2,\"gave_back_edge\":true}",
            },
            {
                "symbol": "EURUSD",
                "strategy_key": "EURUSD_LONDON_BREAKOUT",
                "session_name": "TOKYO",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_amount": -9.0,
                "pnl_r": -0.8,
                "close_context_json": "{\"failure_buckets\":[\"late_entry\",\"immediate_invalidation\",\"poor_structure\"]}",
                "management_effect_json": "{\"management_quality\":0.2,\"gave_back_edge\":true}",
            },
            {
                "symbol": "EURUSD",
                "strategy_key": "EURUSD_LONDON_BREAKOUT",
                "session_name": "TOKYO",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T02:00:00+00:00",
                "pnl_amount": -8.0,
                "pnl_r": -0.9,
                "close_context_json": "{\"failure_buckets\":[\"false_breakout\",\"poor_structure\"]}",
                "management_effect_json": "{\"management_quality\":0.2,\"gave_back_edge\":true}",
            },
        ]

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="EURUSD",
            candidates=[punished, cleaner],
            session_name="TOKYO",
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(str(ranked[0][1]["strategy_key"]), "EURUSD_RANGE_FADE")

    def test_strategy_pool_rankings_filter_severe_exact_loser_buckets_without_history(self) -> None:
        punished = SignalCandidate(
            signal_id="xau-severe-loser",
            setup="XAUUSD_NY_MOMENTUM_BREAKOUT",
            side="BUY",
            score_hint=0.86,
            reason="bad exact london non-grid loser bucket should be blocked unless elite",
            stop_atr=1.0,
            tp_r=1.9,
            strategy_family="TREND",
            meta={
                "strategy_key": "XAUUSD_NY_MOMENTUM_BREAKOUT",
                "router_rank_score": 0.86,
                "regime_state": "LOW_LIQUIDITY_CHOP",
                "regime_fit": 0.74,
                "session_fit": 0.76,
                "volatility_fit": 0.70,
                "pair_behavior_fit": 0.56,
                "execution_quality_fit": 0.74,
                "entry_timing_score": 0.75,
                "structure_cleanliness_score": 0.77,
                "strategy_recent_performance_seed": 0.55,
            },
        )
        cleaner = SignalCandidate(
            signal_id="xau-grid",
            setup="XAUUSD_ADAPTIVE_M5_GRID",
            side="BUY",
            score_hint=0.71,
            reason="cleaner lane should survive",
            stop_atr=1.0,
            tp_r=1.5,
            strategy_family="GRID",
            meta={
                "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                "router_rank_score": 0.71,
                "regime_state": "MEAN_REVERSION",
                "regime_fit": 0.88,
                "session_fit": 0.90,
                "volatility_fit": 0.82,
                "pair_behavior_fit": 0.85,
                "execution_quality_fit": 0.86,
                "entry_timing_score": 0.84,
                "structure_cleanliness_score": 0.86,
                "strategy_recent_performance_seed": 0.66,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="XAUUSD",
            candidates=[punished, cleaner],
            session_name="LONDON",
        )

        self.assertGreaterEqual(len(ranked), 1)
        self.assertEqual(str(ranked[0][1]["strategy_key"]), "XAUUSD_ADAPTIVE_M5_GRID")
        punished_entries = [
            entry for _, entry in ranked if str(entry.get("strategy_key") or "") == "XAUUSD_NY_MOMENTUM_BREAKOUT"
        ]
        if punished_entries:
            self.assertGreater(
                float(ranked[0][1].get("strategy_pool_rank_score", 0.0) or 0.0),
                float(punished_entries[0].get("strategy_pool_rank_score", 0.0) or 0.0),
            )

    def test_strategy_pool_rankings_drop_xau_mirror_rollout_candidate_and_keep_native_grid(self) -> None:
        mirror_grid = SignalCandidate(
            signal_id="xau-grid-mirror",
            setup="XAUUSD_ADAPTIVE_M5_GRID",
            side="BUY",
            score_hint=0.84,
            reason="mirror rollout should be skipped once native-only mode is active",
            stop_atr=1.0,
            tp_r=1.8,
            strategy_family="GRID",
            meta={
                "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                "xau_engine": "GRID_DIRECTIONAL_MIRROR_SUPPORT",
                "grid_source_role": "SECONDARY_SUPPORT",
                "grid_source_setup": "XAUUSD_M1_MICRO_SCALPER",
                "grid_entry_profile": "grid_directional_flow_long",
                "router_rank_score": 0.84,
                "regime_state": "MEAN_REVERSION",
                "regime_fit": 0.74,
                "session_fit": 0.90,
                "volatility_fit": 0.82,
                "pair_behavior_fit": 0.85,
                "execution_quality_fit": 0.88,
                "entry_timing_score": 0.95,
                "structure_cleanliness_score": 0.84,
                "multi_tf_alignment_score": 1.0,
                "fractal_persistence_score": 1.0,
                "seasonality_edge_score": 0.80,
                "market_instability_score": 0.03,
                "feature_drift_score": 0.06,
                "quality_tier": "A",
                "strategy_recent_performance_seed": 0.66,
            },
        )
        native_grid = SignalCandidate(
            signal_id="xau-grid-native",
            setup="XAUUSD_ADAPTIVE_M5_GRID",
            side="BUY",
            score_hint=0.86,
            reason="native grid remains the live production lane",
            stop_atr=1.0,
            tp_r=2.0,
            strategy_family="GRID",
            meta={
                "strategy_key": "XAUUSD_ADAPTIVE_M5_GRID",
                "xau_engine": "GRID_NATIVE_SCALPER",
                "grid_source_role": "NATIVE_ATTACK",
                "grid_entry_profile": "grid_directional_flow_long",
                "router_rank_score": 0.86,
                "regime_state": "MEAN_REVERSION",
                "regime_fit": 0.76,
                "session_fit": 0.92,
                "volatility_fit": 0.84,
                "pair_behavior_fit": 0.87,
                "execution_quality_fit": 0.89,
                "entry_timing_score": 0.96,
                "structure_cleanliness_score": 0.86,
                "multi_tf_alignment_score": 1.0,
                "fractal_persistence_score": 1.0,
                "seasonality_edge_score": 0.82,
                "market_instability_score": 0.02,
                "feature_drift_score": 0.04,
                "quality_tier": "A",
                "strategy_recent_performance_seed": 0.68,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="XAUUSD",
            candidates=[mirror_grid, native_grid],
            session_name="OVERLAP",
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(str(ranked[0][1]["strategy_key"]), "XAUUSD_ADAPTIVE_M5_GRID")
        self.assertEqual(str(ranked[0][1]["setup"]), "XAUUSD_ADAPTIVE_M5_GRID")
        self.assertNotIn("GRID_DIRECTIONAL_MIRROR", str(ranked[0][1].get("xau_engine") or ""))

    def test_strategy_pool_summary_prefers_rich_session_native_fields(self) -> None:
        summarized = _summarize_strategy_pool_rankings(
            symbol_key="AUDNZD",
            ranked_entries=[
                {
                    "strategy_key": "AUDNZD_STRUCTURE_BREAK_RETEST",
                    "strategy_score": 0.61,
                    "strategy_state": "NORMAL",
                    "session_priority_profile": "GLOBAL",
                    "lane_session_priority": "NEUTRAL",
                    "session_native_pair": False,
                    "session_priority_multiplier": 1.0,
                    "pair_priority_rank_in_session": 99,
                    "lane_budget_share": 0.0,
                    "lane_available_capacity": 0.0,
                },
                {
                    "strategy_key": "AUDNZD_STRUCTURE_BREAK_RETEST",
                    "strategy_score": 0.61,
                    "strategy_state": "NORMAL",
                    "session_priority_profile": "ASIA_NATIVE",
                    "lane_session_priority": "PRIMARY",
                    "session_native_pair": True,
                    "session_priority_multiplier": 1.14,
                    "pair_priority_rank_in_session": 1,
                    "lane_budget_share": 0.20,
                    "lane_available_capacity": 3.0,
                },
            ],
            preferred_strategy_key="AUDNZD_STRUCTURE_BREAK_RETEST",
            session_name="SYDNEY",
            regime_state="RANGING",
        )

        self.assertGreaterEqual(len(summarized), 1)
        winner = summarized[0]
        self.assertEqual(str(winner.get("strategy_key") or ""), "AUDNZD_STRUCTURE_BREAK_RETEST")
        self.assertEqual(str(winner.get("session_priority_profile") or ""), "ASIA_NATIVE")
        self.assertEqual(str(winner.get("lane_session_priority") or ""), "PRIMARY")
        self.assertTrue(bool(winner.get("session_native_pair", False)))
        self.assertEqual(int(winner.get("pair_priority_rank_in_session") or 99), 1)
        self.assertAlmostEqual(float(winner.get("lane_budget_share") or 0.0), 0.20)
        self.assertAlmostEqual(float(winner.get("lane_available_capacity") or 0.0), 3.0)

    def test_strategy_pool_rankings_keep_b_tier_candidate_alive_when_throughput_recovery_active(self) -> None:
        btc_candidate = SignalCandidate(
            signal_id="btc-b-tier",
            setup="BTCUSD_VOLATILE_RETEST",
            side="BUY",
            score_hint=0.63,
            reason="good-but-imperfect btc weekend retest",
            stop_atr=1.0,
            tp_r=1.7,
            strategy_family="TREND",
            meta={
                "strategy_key": "BTCUSD_VOLATILE_RETEST",
                "router_rank_score": 0.63,
                "regime_state": "MEAN_REVERSION",
                "regime_fit": 0.64,
                "session_fit": 0.74,
                "volatility_fit": 0.70,
                "pair_behavior_fit": 0.72,
                "execution_quality_fit": 0.69,
                "entry_timing_score": 0.61,
                "structure_cleanliness_score": 0.60,
                "strategy_recent_performance_seed": 0.58,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[btc_candidate],
            session_name="TOKYO",
            row={"time": "2026-03-14T01:00:00+00:00"},
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(str(ranked[0][1]["quality_tier"]), "B")
        self.assertTrue(bool(ranked[0][1]["throughput_recovery_active"]))
        self.assertGreaterEqual(float(ranked[0][1]["tier_size_multiplier"]), 0.70)
        self.assertLessEqual(float(ranked[0][1]["tier_size_multiplier"]), 1.10)
        self.assertTrue(bool(ranked[0][1]["btc_weekend_mode"]))

    def test_strategy_pool_rankings_allow_btc_weekend_london_price_action_density(self) -> None:
        btc_candidate = SignalCandidate(
            signal_id="btc-weekend-london",
            setup="BTC_PRICE_ACTION_CONTINUATION",
            side="BUY",
            score_hint=0.74,
            reason="weekend london btc should stay active",
            stop_atr=1.0,
            tp_r=2.0,
            strategy_family="TREND",
            meta={
                "strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION",
                "router_rank_score": 0.74,
                "regime_state": "MEAN_REVERSION",
                "regime_fit": 0.58,
                "session_fit": 0.78,
                "volatility_fit": 0.74,
                "pair_behavior_fit": 0.56,
                "execution_quality_fit": 0.58,
                "entry_timing_score": 0.59,
                "structure_cleanliness_score": 0.58,
                "strategy_recent_performance_seed": 0.56,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[btc_candidate],
            session_name="LONDON",
            row={"time": "2026-03-14T09:00:00+00:00"},
        )

        self.assertEqual(len(ranked), 1)
        self.assertTrue(bool(ranked[0][1]["btc_weekend_mode"]))
        self.assertTrue(bool(ranked[0][1]["btc_weekend_printer_ready"]))
        self.assertGreater(float(ranked[0][1]["btc_directional_ranking_bonus"] or 0.0), 0.0)

    def test_normalize_runtime_spread_points_converts_live_btc_points_to_price_units(self) -> None:
        normalized = _normalize_runtime_spread_points(
            "BTCUSD",
            1712.0,
            symbol_info={"point": 0.01, "trade_tick_size": 0.01, "digits": 2},
            max_spread_points=60.0,
        )

        self.assertAlmostEqual(float(normalized), 17.12, places=6)

    def test_strategy_pool_rankings_normalize_live_btc_spread_before_quality_gate(self) -> None:
        btc_candidate = SignalCandidate(
            signal_id="btc-live-spread-normalized",
            setup="BTC_WEEKEND_BREAKOUT",
            side="BUY",
            score_hint=0.72,
            reason="live weekend spread should not fake a disorder reject",
            stop_atr=1.0,
            tp_r=1.9,
            strategy_family="TREND",
            meta={
                "strategy_key": "BTCUSD_TREND_SCALP",
                "router_rank_score": 0.72,
                "regime_state": "LOW_LIQUIDITY_CHOP",
                "regime_fit": 0.70,
                "session_fit": 0.74,
                "volatility_fit": 0.72,
                "pair_behavior_fit": 0.60,
                "entry_timing_score": 0.69,
                "structure_cleanliness_score": 0.70,
                "strategy_recent_performance_seed": 0.64,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[btc_candidate],
            session_name="TOKYO",
            row={
                "time": "2026-03-28T04:58:59+00:00",
                "m5_spread": 1712.0,
                "m5_spread_avg_20": 1620.0,
            },
            regime=SimpleNamespace(label="LOW_LIQUIDITY_CHOP", details={}),
            symbol_info={"point": 0.01, "trade_tick_size": 0.01, "digits": 2},
            max_spread_points=60.0,
        )

        self.assertEqual(len(ranked), 1)
        self.assertGreater(float(ranked[0][1]["execution_quality_fit"]), 0.65)
        self.assertNotEqual(str(ranked[0][1]["quality_tier"]), "C")

    def test_strategy_pool_rankings_allow_btc_weekend_tokyo_price_action_density(self) -> None:
        btc_candidate = SignalCandidate(
            signal_id="btc-weekend-tokyo",
            setup="BTC_PRICE_ACTION_CONTINUATION",
            side="BUY",
            score_hint=0.72,
            reason="weekend tokyo btc should stay active",
            stop_atr=1.0,
            tp_r=1.9,
            strategy_family="TREND",
            meta={
                "strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION",
                "router_rank_score": 0.72,
                "regime_state": "RANGING",
                "regime_fit": 0.56,
                "session_fit": 0.74,
                "volatility_fit": 0.72,
                "pair_behavior_fit": 0.54,
                "execution_quality_fit": 0.56,
                "entry_timing_score": 0.58,
                "structure_cleanliness_score": 0.57,
                "strategy_recent_performance_seed": 0.55,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[btc_candidate],
            session_name="TOKYO",
            row={"time": "2026-03-14T01:00:00+00:00"},
        )

        self.assertEqual(len(ranked), 1)
        self.assertTrue(bool(ranked[0][1]["btc_weekend_mode"]))
        self.assertTrue(bool(ranked[0][1]["btc_weekend_printer_ready"]))
        self.assertGreater(float(ranked[0][1]["btc_directional_ranking_bonus"] or 0.0), 0.0)

    def test_strategy_pool_rankings_disable_throughput_recovery_for_xau_atr_expansion(self) -> None:
        xau_candidate = SignalCandidate(
            signal_id="xau-atr-b-tier",
            setup="XAUUSD_M15_STRUCTURED_BREAKOUT",
            side="BUY",
            score_hint=0.79,
            reason="strong off-session directional gold candidate",
            stop_atr=1.0,
            tp_r=1.7,
            strategy_family="TREND",
            meta={
                "strategy_key": "XAUUSD_ATR_EXPANSION_SCALPER",
                "router_rank_score": 0.79,
                "quality_tier": "A",
                "regime_state": "TRENDING",
                "session_fit": 0.78,
                "regime_fit": 0.78,
                "volatility_fit": 0.75,
                "pair_behavior_fit": 0.66,
                "execution_quality_fit": 0.74,
                "entry_timing_score": 0.76,
                "structure_cleanliness_score": 0.78,
                "strategy_recent_performance_seed": 0.64,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="XAUUSD",
            candidates=[xau_candidate],
            session_name="TOKYO",
            row={"time": "2026-03-19T01:00:00+00:00"},
        )

        self.assertEqual(len(ranked), 1)
        self.assertFalse(bool(ranked[0][1]["throughput_recovery_active"]))

    def test_strategy_pool_rankings_block_low_quality_xau_atr_directional_in_london(self) -> None:
        xau_candidate = SignalCandidate(
            signal_id="xau-atr-london",
            setup="XAUUSD_M15_STRUCTURED_BREAKOUT",
            side="BUY",
            score_hint=0.64,
            reason="structured but still weak directional gold candidate",
            stop_atr=1.0,
            tp_r=1.7,
            strategy_family="TREND",
            meta={
                "strategy_key": "XAUUSD_ATR_EXPANSION_SCALPER",
                "router_rank_score": 0.64,
                "regime_state": "TRENDING",
                "session_fit": 0.72,
                "regime_fit": 0.64,
                "volatility_fit": 0.72,
                "pair_behavior_fit": 0.56,
                "execution_quality_fit": 0.64,
                "entry_timing_score": 0.62,
                "structure_cleanliness_score": 0.64,
                "strategy_recent_performance_seed": 0.56,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="XAUUSD",
            candidates=[xau_candidate],
            session_name="LONDON",
            row={"time": "2026-03-19T09:00:00+00:00"},
        )

        self.assertEqual(ranked, [])

    def test_strategy_pool_rankings_block_low_quality_xau_directional_off_session(self) -> None:
        xau_candidate = SignalCandidate(
            signal_id="xau-ny-tokyo",
            setup="XAUUSD_M15_STRUCTURED_BREAKOUT",
            side="BUY",
            score_hint=0.68,
            reason="off-session directional gold candidate",
            stop_atr=1.0,
            tp_r=1.9,
            strategy_family="TREND",
            meta={
                "strategy_key": "XAUUSD_NY_MOMENTUM_BREAKOUT",
                "router_rank_score": 0.68,
                "regime_state": "TRENDING",
                "session_fit": 0.70,
                "regime_fit": 0.68,
                "volatility_fit": 0.74,
                "pair_behavior_fit": 0.58,
                "execution_quality_fit": 0.66,
                "entry_timing_score": 0.65,
                "structure_cleanliness_score": 0.66,
                "strategy_recent_performance_seed": 0.58,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="XAUUSD",
            candidates=[xau_candidate],
            session_name="TOKYO",
            row={"time": "2026-03-19T01:00:00+00:00"},
        )

        self.assertEqual(ranked, [])

    def test_strategy_pool_rankings_allow_strong_b_tier_xau_breakout_retest_in_overlap(self) -> None:
        xau_candidate = SignalCandidate(
            signal_id="xau-breakout-overlap",
            setup="XAU_BREAKOUT_RETEST",
            side="BUY",
            score_hint=0.74,
            reason="strong overlap gold breakout retest",
            stop_atr=1.0,
            tp_r=2.0,
            strategy_family="TREND",
            meta={
                "strategy_key": "XAUUSD_NY_MOMENTUM_BREAKOUT",
                "router_rank_score": 0.74,
                "quality_tier": "B",
                "regime_state": "TRENDING",
                "session_fit": 0.76,
                "regime_fit": 0.74,
                "volatility_fit": 0.76,
                "pair_behavior_fit": 0.62,
                "execution_quality_fit": 0.70,
                "entry_timing_score": 0.70,
                "structure_cleanliness_score": 0.72,
                "strategy_recent_performance_seed": 0.60,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="XAUUSD",
            candidates=[xau_candidate],
            session_name="OVERLAP",
            row={"time": "2026-03-19T14:00:00+00:00"},
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(str(ranked[0][1]["strategy_key"] or ""), "XAUUSD_NY_MOMENTUM_BREAKOUT")

    def test_strategy_pool_rankings_allow_strong_xau_breakout_retest_off_session(self) -> None:
        xau_candidate = SignalCandidate(
            signal_id="xau-breakout-tokyo",
            setup="XAU_BREAKOUT_RETEST",
            side="BUY",
            score_hint=0.82,
            reason="elite off-session gold breakout retest should still survive",
            stop_atr=1.0,
            tp_r=2.2,
            strategy_family="TREND",
            meta={
                "strategy_key": "XAUUSD_NY_MOMENTUM_BREAKOUT",
                "router_rank_score": 0.82,
                "quality_tier": "A",
                "regime_state": "TRENDING",
                "session_fit": 0.78,
                "regime_fit": 0.80,
                "volatility_fit": 0.78,
                "pair_behavior_fit": 0.62,
                "execution_quality_fit": 0.74,
                "entry_timing_score": 0.74,
                "structure_cleanliness_score": 0.76,
                "strategy_recent_performance_seed": 0.62,
                "multi_tf_alignment_score": 0.66,
                "market_instability_score": 0.18,
                "feature_drift_score": 0.12,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="XAUUSD",
            candidates=[xau_candidate],
            session_name="TOKYO",
            row={"time": "2026-03-19T01:00:00+00:00"},
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(str(ranked[0][1]["strategy_key"] or ""), "XAUUSD_NY_MOMENTUM_BREAKOUT")
        self.assertTrue(bool(ranked[0][1]["xau_directional_off_session_attack_ready"]))
        self.assertGreater(float(ranked[0][1]["xau_directional_breakout_ranking_bonus"] or 0.0), 0.0)

    def test_strategy_pool_rankings_allow_elite_xau_breakout_retest_in_london(self) -> None:
        xau_candidate = SignalCandidate(
            signal_id="xau-breakout-london",
            setup="XAU_BREAKOUT_RETEST",
            side="BUY",
            score_hint=0.89,
            reason="elite london gold breakout retest should override the severe bucket",
            stop_atr=1.0,
            tp_r=2.4,
            strategy_family="TREND",
            meta={
                "strategy_key": "XAUUSD_NY_MOMENTUM_BREAKOUT",
                "router_rank_score": 0.89,
                "quality_tier": "A",
                "regime_state": "TRENDING",
                "session_fit": 0.80,
                "regime_fit": 0.84,
                "volatility_fit": 0.80,
                "pair_behavior_fit": 0.66,
                "execution_quality_fit": 0.80,
                "entry_timing_score": 0.82,
                "structure_cleanliness_score": 0.84,
                "strategy_recent_performance_seed": 0.64,
                "multi_tf_alignment_score": 0.68,
                "market_instability_score": 0.16,
                "feature_drift_score": 0.10,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="XAUUSD",
            candidates=[xau_candidate],
            session_name="LONDON",
            row={"time": "2026-03-19T09:00:00+00:00"},
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(str(ranked[0][1]["strategy_key"] or ""), "XAUUSD_NY_MOMENTUM_BREAKOUT")
        self.assertTrue(bool(ranked[0][1]["xau_directional_prime_attack_ready"]))
        self.assertGreater(float(ranked[0][1]["xau_directional_breakout_ranking_bonus"] or 0.0), 0.0)

    def test_strategy_pool_rankings_exact_reject_btc_tokyo_trend_scalp_bad_liquidity(self) -> None:
        punished = SignalCandidate(
            signal_id="btc-tokyo-scalp",
            setup="BTC_TOKYO_DRIFT_SCALP",
            side="BUY",
            score_hint=0.76,
            reason="known loser pocket",
            stop_atr=1.0,
            tp_r=1.6,
            strategy_family="TREND",
            meta={
                "strategy_key": "BTCUSD_TREND_SCALP",
                "router_rank_score": 0.76,
                "regime_state": "MEAN_REVERSION",
                "regime_fit": 0.78,
                "session_fit": 0.70,
                "volatility_fit": 0.72,
                "pair_behavior_fit": 0.42,
                "execution_quality_fit": 0.76,
                "entry_timing_score": 0.73,
                "structure_cleanliness_score": 0.74,
                "strategy_recent_performance_seed": 0.58,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[punished],
            session_name="TOKYO",
        )

        self.assertEqual(ranked, [])

    def test_strategy_pool_rankings_exact_reject_btc_sydney_trend_scalp_weekday(self) -> None:
        punished = SignalCandidate(
            signal_id="btc-sydney-scalp",
            setup="BTC_TOKYO_DRIFT_SCALP",
            side="BUY",
            score_hint=0.79,
            reason="weak weekday sydney btc scalp",
            stop_atr=1.0,
            tp_r=1.8,
            strategy_family="TREND",
            meta={
                "strategy_key": "BTCUSD_TREND_SCALP",
                "router_rank_score": 0.79,
                "regime_state": "TRENDING",
                "regime_fit": 0.80,
                "session_fit": 0.74,
                "volatility_fit": 0.76,
                "pair_behavior_fit": 0.58,
                "execution_quality_fit": 0.78,
                "entry_timing_score": 0.75,
                "structure_cleanliness_score": 0.76,
                "strategy_recent_performance_seed": 0.60,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[punished],
            session_name="SYDNEY",
        )

        self.assertEqual(ranked, [])

    def test_strategy_pool_rankings_exact_reject_btc_london_price_action_continuation_weekday(self) -> None:
        punished = SignalCandidate(
            signal_id="btc-london-pac",
            setup="BTC_PRICE_ACTION_CONTINUATION",
            side="BUY",
            score_hint=0.82,
            reason="weak weekday london btc continuation",
            stop_atr=1.0,
            tp_r=2.0,
            strategy_family="TREND",
            meta={
                "strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION",
                "router_rank_score": 0.82,
                "regime_state": "TRENDING",
                "regime_fit": 0.82,
                "session_fit": 0.76,
                "volatility_fit": 0.78,
                "pair_behavior_fit": 0.62,
                "execution_quality_fit": 0.80,
                "entry_timing_score": 0.76,
                "structure_cleanliness_score": 0.78,
                "strategy_recent_performance_seed": 0.62,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[punished],
            session_name="LONDON",
        )

        self.assertEqual(ranked, [])

    def test_strategy_pool_rankings_exact_reject_btc_tokyo_price_action_continuation_weekday(self) -> None:
        punished = SignalCandidate(
            signal_id="btc-tokyo-pac",
            setup="BTC_PRICE_ACTION_CONTINUATION",
            side="BUY",
            score_hint=0.81,
            reason="weak weekday tokyo btc continuation",
            stop_atr=1.0,
            tp_r=2.0,
            strategy_family="TREND",
            meta={
                "strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION",
                "router_rank_score": 0.81,
                "regime_state": "TRENDING",
                "regime_fit": 0.80,
                "session_fit": 0.74,
                "volatility_fit": 0.76,
                "pair_behavior_fit": 0.60,
                "execution_quality_fit": 0.79,
                "entry_timing_score": 0.75,
                "structure_cleanliness_score": 0.77,
                "strategy_recent_performance_seed": 0.60,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[punished],
            session_name="TOKYO",
        )

        self.assertEqual(ranked, [])

    def test_strategy_pool_rankings_exact_reject_btc_tokyo_trend_scalp_trending_weekday(self) -> None:
        punished = SignalCandidate(
            signal_id="btc-tokyo-trend-scalp-trending",
            setup="BTC_TOKYO_DRIFT_SCALP",
            side="BUY",
            score_hint=0.83,
            reason="weekday tokyo btc trend scalp remains blocked even when trending",
            stop_atr=1.0,
            tp_r=1.9,
            strategy_family="TREND",
            meta={
                "strategy_key": "BTCUSD_TREND_SCALP",
                "router_rank_score": 0.83,
                "regime_state": "TRENDING",
                "regime_fit": 0.82,
                "session_fit": 0.76,
                "volatility_fit": 0.80,
                "pair_behavior_fit": 0.62,
                "execution_quality_fit": 0.82,
                "entry_timing_score": 0.78,
                "structure_cleanliness_score": 0.80,
                "strategy_recent_performance_seed": 0.64,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[punished],
            session_name="TOKYO",
        )

        self.assertEqual(ranked, [])

    def test_strategy_pool_rankings_allow_elite_btc_new_york_price_action_continuation_weekday(self) -> None:
        allowed = SignalCandidate(
            signal_id="btc-ny-pac-elite",
            setup="BTC_PRICE_ACTION_CONTINUATION",
            side="BUY",
            score_hint=0.87,
            reason="elite weekday new york continuation should survive",
            stop_atr=1.0,
            tp_r=2.2,
            strategy_family="TREND",
            meta={
                "strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION",
                "router_rank_score": 0.87,
                "quality_tier": "A",
                "regime_state": "TRENDING",
                "regime_fit": 0.76,
                "session_fit": 0.78,
                "volatility_fit": 0.78,
                "pair_behavior_fit": 0.62,
                "execution_quality_fit": 0.72,
                "entry_timing_score": 0.72,
                "structure_cleanliness_score": 0.72,
                "strategy_recent_performance_seed": 0.64,
                "multi_tf_alignment_score": 0.62,
                "market_instability_score": 0.18,
                "feature_drift_score": 0.12,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[allowed],
            session_name="NEW_YORK",
            row={"time": "2026-03-19T18:00:00+00:00"},
        )

        self.assertEqual(len(ranked), 1)
        self.assertEqual(str(ranked[0][1]["strategy_key"] or ""), "BTCUSD_PRICE_ACTION_CONTINUATION")
        self.assertTrue(bool(ranked[0][1]["btc_weekday_price_action_ready"]))
        self.assertGreater(float(ranked[0][1]["btc_directional_ranking_bonus"] or 0.0), 0.0)

    def test_strategy_pool_rankings_exact_reject_btc_overlap_price_action_continuation_trending_non_elite(self) -> None:
        punished = SignalCandidate(
            signal_id="btc-overlap-pac-trending",
            setup="BTC_PRICE_ACTION_CONTINUATION",
            side="BUY",
            score_hint=0.85,
            reason="weekday overlap trending continuation must be elite now",
            stop_atr=1.0,
            tp_r=2.3,
            strategy_family="TREND",
            meta={
                "strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION",
                "router_rank_score": 0.85,
                "quality_tier": "B",
                "regime_state": "TRENDING",
                "regime_fit": 0.64,
                "session_fit": 0.78,
                "volatility_fit": 0.78,
                "pair_behavior_fit": 0.55,
                "execution_quality_fit": 0.64,
                "entry_timing_score": 0.63,
                "structure_cleanliness_score": 0.65,
                "strategy_recent_performance_seed": 0.60,
                "multi_tf_alignment_score": 0.57,
                "market_instability_score": 0.20,
                "feature_drift_score": 0.16,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[punished],
            session_name="OVERLAP",
            row={"time": "2026-03-19T13:00:00+00:00"},
        )

        self.assertEqual(ranked, [])

    def test_strategy_pool_rankings_exact_reject_btc_overlap_price_action_continuation_trending_elite_weekday(self) -> None:
        punished = SignalCandidate(
            signal_id="btc-overlap-pac-trending-elite",
            setup="BTC_PRICE_ACTION_CONTINUATION",
            side="BUY",
            score_hint=0.89,
            reason="weekday overlap trending continuation remains fully blocked",
            stop_atr=1.0,
            tp_r=2.4,
            strategy_family="TREND",
            meta={
                "strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION",
                "router_rank_score": 0.89,
                "quality_tier": "A",
                "regime_state": "TRENDING",
                "regime_fit": 0.72,
                "session_fit": 0.80,
                "volatility_fit": 0.80,
                "pair_behavior_fit": 0.58,
                "execution_quality_fit": 0.68,
                "entry_timing_score": 0.66,
                "structure_cleanliness_score": 0.68,
                "strategy_recent_performance_seed": 0.62,
                "multi_tf_alignment_score": 0.60,
                "market_instability_score": 0.18,
                "feature_drift_score": 0.12,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[punished],
            session_name="OVERLAP",
            row={"time": "2026-03-19T13:10:00+00:00"},
        )

        self.assertEqual(ranked, [])

    def test_strategy_pool_rankings_exact_reject_btc_sydney_volatile_retest_low_liquidity_weekday(self) -> None:
        punished = SignalCandidate(
            signal_id="btc-sydney-vr-llc",
            setup="BTC_VOLATILE_RETEST",
            side="BUY",
            score_hint=0.83,
            reason="weekday sydney retest low-liquidity chop remains blocked",
            stop_atr=1.0,
            tp_r=1.9,
            strategy_family="RETEST",
            meta={
                "strategy_key": "BTCUSD_VOLATILE_RETEST",
                "router_rank_score": 0.83,
                "regime_state": "LOW_LIQUIDITY_CHOP",
                "regime_fit": 0.78,
                "session_fit": 0.74,
                "volatility_fit": 0.76,
                "pair_behavior_fit": 0.58,
                "execution_quality_fit": 0.78,
                "entry_timing_score": 0.74,
                "structure_cleanliness_score": 0.76,
                "strategy_recent_performance_seed": 0.60,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[punished],
            session_name="SYDNEY",
        )

        self.assertEqual(ranked, [])

    def test_strategy_pool_rankings_exact_reject_btc_tokyo_volatile_retest_trending_weekday(self) -> None:
        punished = SignalCandidate(
            signal_id="btc-tokyo-vr-trending",
            setup="BTC_VOLATILE_RETEST",
            side="BUY",
            score_hint=0.84,
            reason="weekday tokyo retest trending stays blocked",
            stop_atr=1.0,
            tp_r=2.0,
            strategy_family="RETEST",
            meta={
                "strategy_key": "BTCUSD_VOLATILE_RETEST",
                "router_rank_score": 0.84,
                "regime_state": "TRENDING",
                "regime_fit": 0.80,
                "session_fit": 0.76,
                "volatility_fit": 0.78,
                "pair_behavior_fit": 0.60,
                "execution_quality_fit": 0.80,
                "entry_timing_score": 0.76,
                "structure_cleanliness_score": 0.78,
                "strategy_recent_performance_seed": 0.62,
            },
        )

        ranked = _candidate_strategy_pool_rankings(
            symbol_key="BTCUSD",
            candidates=[punished],
            session_name="TOKYO",
        )

        self.assertEqual(ranked, [])


    def test_equity_momentum_throttle_switches_hot_and_cold(self) -> None:
        hot = _equity_momentum_throttle(
            [{"r_multiple": 1.3} for _ in range(50)],
            {"equity_momentum_hot_expectancy_r": 1.0, "equity_momentum_cold_expectancy_r": 0.6},
        )
        cold = _equity_momentum_throttle(
            [{"r_multiple": 0.2} for _ in range(50)],
            {"equity_momentum_hot_expectancy_r": 1.0, "equity_momentum_cold_expectancy_r": 0.6},
        )

        self.assertEqual(str(hot["mode"]), "HOT")
        self.assertEqual(str(cold["mode"]), "COLD")
        self.assertAlmostEqual(float(hot["b_tier_adjust_pct"]), 0.15, places=6)
        self.assertAlmostEqual(float(hot["a_plus_size_boost"]), 0.10, places=6)
        self.assertAlmostEqual(float(cold["b_tier_adjust_pct"]), -0.10, places=6)

    def test_family_rotation_penalty_ignores_grid_and_hits_dominant_family(self) -> None:
        trades = [{"strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION"} for _ in range(13)] + [{"strategy_key": "BTCUSD_VOLATILE_RETEST"} for _ in range(7)]
        penalty = _family_rotation_penalty(
            strategy_key="BTCUSD_PRICE_ACTION_CONTINUATION",
            setup_family="TREND",
            closed_trades=trades,
            candidate_tier_config={"family_rotation_window_trades": 20, "family_rotation_share_threshold": 0.60, "family_rotation_score_penalty": 0.20},
        )
        grid_penalty = _family_rotation_penalty(
            strategy_key="XAUUSD_ADAPTIVE_M5_GRID",
            setup_family="GRID",
            closed_trades=trades,
            candidate_tier_config={"family_rotation_window_trades": 20, "family_rotation_share_threshold": 0.60, "family_rotation_score_penalty": 0.20},
        )

        self.assertAlmostEqual(float(penalty), 0.20, places=6)
        self.assertEqual(float(grid_penalty), 0.0)

    def test_quality_tier_exit_profile_sets_rr_target_band(self) -> None:
        profile = _quality_tier_exit_profile("A+", {"trail_activation_r": 1.2, "trail_atr": 1.0})
        self.assertEqual(str(profile["approved_rr_target"]), "2.4-2.8")
        self.assertEqual(float(profile["partials"][1]["triggerR"]), 1.2)

    def test_load_symbol_frames_allows_optional_m1_gap_when_required_frames_exist(self) -> None:
        class _Service:
            def fetch(self, symbol: str, timeframe: str, count: int):
                if timeframe == "M1":
                    raise RuntimeError("live_market_data_unavailable:AUDJPY:M1")
                return {"symbol": symbol, "timeframe": timeframe, "count": count}

            def load_cached(self, symbol: str, timeframe: str):
                if timeframe == "M1":
                    return None
                return {"symbol": symbol, "timeframe": timeframe, "source": "cache"}

        frames, reason = _load_symbol_frames(
            _Service(),
            "AUDJPY",
            {"M1": 10, "M5": 10, "M15": 10, "H1": 10, "H4": 10},
            dry_run=False,
        )

        self.assertIsNotNone(frames)
        self.assertIn("M5", frames)
        self.assertIn("M15", frames)
        self.assertIn("H1", frames)
        self.assertNotIn("M1", frames)
        self.assertIsNotNone(reason)
        self.assertIn("market_data_optional_missing:AUDJPY", str(reason))

    def test_load_symbol_frames_requires_h1_even_when_optional_frames_missing(self) -> None:
        class _Service:
            def fetch(self, symbol: str, timeframe: str, count: int):
                if timeframe == "H1":
                    raise RuntimeError("live_market_data_unavailable:AUDJPY:H1")
                return {"symbol": symbol, "timeframe": timeframe, "count": count}

            def load_cached(self, symbol: str, timeframe: str):
                return None

        frames, reason = _load_symbol_frames(
            _Service(),
            "AUDJPY",
            {"M1": 10, "M5": 10, "M15": 10, "H1": 10, "H4": 10},
            dry_run=False,
        )

        self.assertIsNone(frames)
        self.assertIn("market_data_unavailable:AUDJPY:H1", str(reason))

    def test_load_symbol_frames_falls_back_to_canonical_symbol_when_alias_cache_is_stale(self) -> None:
        now = pd.Timestamp.now(tz="UTC")
        stale_alias = pd.DataFrame(
            {
                "time": [now - pd.Timedelta(minutes=45), now - pd.Timedelta(minutes=40)],
                "open": [4400.0, 4401.0],
                "high": [4401.0, 4402.0],
                "low": [4399.0, 4400.0],
                "close": [4400.5, 4401.5],
            }
        )
        fresh_canonical = pd.DataFrame(
            {
                "time": [now - pd.Timedelta(minutes=5), now],
                "open": [4460.0, 4461.0],
                "high": [4461.0, 4462.0],
                "low": [4459.0, 4460.0],
                "close": [4460.5, 4461.5],
            }
        )

        class _Service:
            def fetch(self, symbol: str, timeframe: str, count: int):
                if symbol == "XAUUSD+":
                    return stale_alias.copy()
                if symbol == "XAUUSD":
                    return fresh_canonical.copy()
                raise AssertionError(f"unexpected symbol {symbol}")

            def load_cached(self, symbol: str, timeframe: str):
                return None

        frames, reason = _load_symbol_frames(
            _Service(),
            "XAUUSD+",
            {"M1": 10, "M5": 10, "M15": 10, "H1": 10},
            dry_run=False,
        )

        self.assertIsNotNone(frames)
        self.assertIsNotNone(reason)
        self.assertIn("market_data_alias_fallback_used:XAUUSD+", str(reason))
        self.assertEqual(pd.Timestamp(frames["M5"].iloc[-1]["time"]), pd.Timestamp(fresh_canonical.iloc[-1]["time"]))

    def test_build_runtime_dry_run_does_not_touch_mt5_loader(self) -> None:
        original_mode = os.environ.get("APEX_MODE")
        os.environ["APEX_MODE"] = "DRY_RUN"
        try:
            with patch("src.mt5_client._load_mt5_module", side_effect=AssertionError("MT5 import should not be attempted in DRY_RUN")):
                runtime = build_runtime()
        finally:
            if original_mode is None:
                os.environ.pop("APEX_MODE", None)
            else:
                os.environ["APEX_MODE"] = original_mode

        self.assertTrue(runtime["dry_run"])
        self.assertEqual(runtime["resolved_symbols"][runtime["configured_symbols"][0]], runtime["configured_symbols"][0])

    def test_build_runtime_keeps_tokyo_out_of_native_only_xau_grid(self) -> None:
        original_mode = os.environ.get("APEX_MODE")
        os.environ["APEX_MODE"] = "DRY_RUN"
        try:
            runtime = build_runtime()
        finally:
            if original_mode is None:
                os.environ.pop("APEX_MODE", None)
            else:
                os.environ["APEX_MODE"] = original_mode

        strategy_router = runtime["strategy_router"]
        grid_scalper = runtime["grid_scalper"]
        self.assertIn("SYDNEY", tuple(strategy_router.xau_active_sessions))
        self.assertIn("TOKYO", tuple(strategy_router.xau_active_sessions))
        self.assertIn("TOKYO", tuple(grid_scalper.allowed_sessions))
        self.assertTrue(bool(grid_scalper.asia_probe_enabled))

    def test_pair_strategy_session_performance_state_quarantines_targeted_ranging_bucket(self) -> None:
        closed_trades = [
            {
                "symbol": "USDJPY",
                "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -0.40,
                "pnl_amount": -4.0,
                "post_trade_review_json": '{"issues":["late_entry","false_breakout","immediate_invalidation"]}',
                "management_effect_json": '{"mfe_r":0.10,"mae_r":-0.65}',
                "duration_minutes": 12.0,
            },
            {
                "symbol": "USDJPY",
                "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.22,
                "pnl_amount": -2.2,
                "post_trade_review_json": '{"issues":["late_entry","poor_structure","immediate_invalidation"]}',
                "management_effect_json": '{"mfe_r":0.06,"mae_r":-0.42}',
                "duration_minutes": 9.0,
            },
            {
                "symbol": "USDJPY",
                "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T02:00:00+00:00",
                "pnl_r": -0.18,
                "pnl_amount": -1.8,
                "post_trade_review_json": '{"issues":["false_breakout","late_entry"]}',
                "management_effect_json": '{"mfe_r":0.08,"mae_r":-0.38}',
                "duration_minutes": 7.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="USDJPY",
            strategy_key="USDJPY_MOMENTUM_IMPULSE",
            session_name="TOKYO",
            regime_state="RANGING",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertTrue(bool(state["strategy_bucket_should_block_all_bands"]))
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")

    def test_pair_strategy_session_performance_state_quarantines_tokyo_secondary_jpy_exact_bucket(self) -> None:
        closed_trades = [
            {
                "symbol": "EURJPY",
                "strategy_key": "EURJPY_MOMENTUM_IMPULSE",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -0.45,
                "pnl_amount": -4.5,
                "post_trade_review_json": '{"issues":["late_entry","false_breakout","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.08,"mae_r":-0.62}',
                "duration_minutes": 10.0,
            },
            {
                "symbol": "EURJPY",
                "strategy_key": "EURJPY_MOMENTUM_IMPULSE",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.31,
                "pnl_amount": -3.1,
                "post_trade_review_json": '{"issues":["immediate_invalidation","false_breakout"]}',
                "management_effect_json": '{"mfe_r":0.04,"mae_r":-0.51}',
                "duration_minutes": 8.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="EURJPY",
            strategy_key="EURJPY_MOMENTUM_IMPULSE",
            session_name="TOKYO",
            regime_state="RANGING",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")

    def test_pair_strategy_session_performance_state_quarantines_gbpusd_london_trend_pullback_in_chop(self) -> None:
        closed_trades = [
            {
                "symbol": "GBPUSD",
                "strategy_key": "GBPUSD_TREND_PULLBACK_RIDE",
                "session_name": "LONDON",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T08:00:00+00:00",
                "pnl_r": -0.92,
                "pnl_amount": -9.2,
                "post_trade_review_json": '{"issues":["late_entry","poor_structure","false_breakout"]}',
                "management_effect_json": '{"mfe_r":0.04,"mae_r":-0.71}',
                "duration_minutes": 16.0,
            },
            {
                "symbol": "GBPUSD",
                "strategy_key": "GBPUSD_TREND_PULLBACK_RIDE",
                "session_name": "LONDON",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T09:00:00+00:00",
                "pnl_r": -0.74,
                "pnl_amount": -7.4,
                "post_trade_review_json": '{"issues":["immediate_invalidation","false_breakout","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.02,"mae_r":-0.63}',
                "duration_minutes": 11.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="GBPUSD",
            strategy_key="GBPUSD_TREND_PULLBACK_RIDE",
            session_name="LONDON",
            regime_state="LOW_LIQUIDITY_CHOP",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertTrue(bool(state["strategy_bucket_should_block_all_bands"]))
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")

    def test_pair_strategy_session_performance_state_preserves_audnzd_rotation_bucket_near_flat(self) -> None:
        closed_trades = [
            {
                "symbol": "AUDNZD",
                "strategy_key": "AUDNZD_VWAP_MEAN_REVERSION",
                "session_name": "SYDNEY",
                "regime": "RANGING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.08,
                "pnl_amount": 0.8,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.22,"mae_r":-0.12}',
                "duration_minutes": 35.0,
            },
            {
                "symbol": "AUDNZD",
                "strategy_key": "AUDNZD_VWAP_MEAN_REVERSION",
                "session_name": "SYDNEY",
                "regime": "RANGING",
                "closed_at": "2026-03-12T02:00:00+00:00",
                "pnl_r": -0.06,
                "pnl_amount": -0.6,
                "post_trade_review_json": '{"issues":["slow_rotation"]}',
                "management_effect_json": '{"mfe_r":0.18,"mae_r":-0.14}',
                "duration_minutes": 42.0,
            },
            {
                "symbol": "AUDNZD",
                "strategy_key": "AUDNZD_VWAP_MEAN_REVERSION",
                "session_name": "SYDNEY",
                "regime": "RANGING",
                "closed_at": "2026-03-12T03:00:00+00:00",
                "pnl_r": 0.02,
                "pnl_amount": 0.2,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.15,"mae_r":-0.08}',
                "duration_minutes": 27.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="AUDNZD",
            strategy_key="AUDNZD_VWAP_MEAN_REVERSION",
            session_name="SYDNEY",
            regime_state="RANGING",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "NORMAL")
        self.assertEqual(state["strategy_bucket_reason"], "audnzd_rotation_preserved")
        self.assertFalse(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_promotes_audnzd_rotation_bucket_when_proving(self) -> None:
        closed_trades = [
            {
                "symbol": "AUDNZD",
                "strategy_key": "AUDNZD_RANGE_ROTATION",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.28,
                "pnl_amount": 2.8,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.72,"mae_r":-0.12}',
                "duration_minutes": 38.0,
            },
            {
                "symbol": "AUDNZD",
                "strategy_key": "AUDNZD_RANGE_ROTATION",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T02:00:00+00:00",
                "pnl_r": 0.10,
                "pnl_amount": 1.0,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.44,"mae_r":-0.10}',
                "duration_minutes": 31.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="AUDNZD",
            strategy_key="AUDNZD_RANGE_ROTATION",
            session_name="TOKYO",
            regime_state="RANGING",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "ATTACK")
        self.assertEqual(state["strategy_bucket_reason"], "audnzd_rotation_attack")
        self.assertFalse(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_promotes_audjpy_tokyo_breakout_when_proving(self) -> None:
        closed_trades = [
            {
                "symbol": "AUDJPY",
                "strategy_key": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.34,
                "pnl_amount": 3.4,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.92,"mae_r":-0.12}',
                "duration_minutes": 24.0,
            },
            {
                "symbol": "AUDJPY",
                "strategy_key": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": 0.14,
                "pnl_amount": 1.4,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.54,"mae_r":-0.08}',
                "duration_minutes": 21.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="AUDJPY",
            strategy_key="AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
            session_name="TOKYO",
            regime_state="TRENDING",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "ATTACK")
        self.assertEqual(state["strategy_bucket_reason"], "audjpy_tokyo_breakout_attack")

    def test_pair_strategy_session_performance_state_demotes_eurusd_london_breakout_tokyo_chop(self) -> None:
        closed_trades = [
            {
                "symbol": "EURUSD",
                "strategy_key": "EURUSD_LONDON_BREAKOUT",
                "session_name": "TOKYO",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -0.42,
                "pnl_amount": -4.2,
                "post_trade_review_json": '{"issues":["late_entry","false_breakout","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.05,"mae_r":-0.58}',
                "duration_minutes": 11.0,
            },
            {
                "symbol": "EURUSD",
                "strategy_key": "EURUSD_LONDON_BREAKOUT",
                "session_name": "TOKYO",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.31,
                "pnl_amount": -3.1,
                "post_trade_review_json": '{"issues":["late_entry","immediate_invalidation"]}',
                "management_effect_json": '{"mfe_r":0.03,"mae_r":-0.45}',
                "duration_minutes": 8.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="EURUSD",
            strategy_key="EURUSD_LONDON_BREAKOUT",
            session_name="TOKYO",
            regime_state="LOW_LIQUIDITY_CHOP",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")
        self.assertFalse(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_demotes_eurusd_london_sweep_ranging(self) -> None:
        closed_trades = [
            {
                "symbol": "EURUSD",
                "strategy_key": "EURUSD_LIQUIDITY_SWEEP",
                "session_name": "LONDON",
                "regime": "RANGING",
                "closed_at": "2026-03-12T08:00:00+00:00",
                "pnl_r": -0.38,
                "pnl_amount": -3.8,
                "post_trade_review_json": '{"issues":["late_entry","false_breakout","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.04,"mae_r":-0.56}',
                "duration_minutes": 9.0,
            },
            {
                "symbol": "EURUSD",
                "strategy_key": "EURUSD_LIQUIDITY_SWEEP",
                "session_name": "LONDON",
                "regime": "RANGING",
                "closed_at": "2026-03-12T09:00:00+00:00",
                "pnl_r": -0.27,
                "pnl_amount": -2.7,
                "post_trade_review_json": '{"issues":["late_entry","immediate_invalidation"]}',
                "management_effect_json": '{"mfe_r":0.03,"mae_r":-0.41}',
                "duration_minutes": 7.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="EURUSD",
            strategy_key="EURUSD_LIQUIDITY_SWEEP",
            session_name="LONDON",
            regime_state="RANGING",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")
        self.assertFalse(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_demotes_audnzd_structure_break_retest_tokyo_chop(self) -> None:
        closed_trades = [
            {
                "symbol": "AUDNZD",
                "strategy_key": "AUDNZD_STRUCTURE_BREAK_RETEST",
                "session_name": "TOKYO",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -0.34,
                "pnl_amount": -3.4,
                "post_trade_review_json": '{"issues":["weak_retest","poor_structure","immediate_invalidation"]}',
                "management_effect_json": '{"mfe_r":0.04,"mae_r":-0.49}',
                "duration_minutes": 13.0,
            },
            {
                "symbol": "AUDNZD",
                "strategy_key": "AUDNZD_STRUCTURE_BREAK_RETEST",
                "session_name": "TOKYO",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.22,
                "pnl_amount": -2.2,
                "post_trade_review_json": '{"issues":["weak_retest","late_entry"]}',
                "management_effect_json": '{"mfe_r":0.06,"mae_r":-0.38}',
                "duration_minutes": 10.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="AUDNZD",
            strategy_key="AUDNZD_STRUCTURE_BREAK_RETEST",
            session_name="TOKYO",
            regime_state="LOW_LIQUIDITY_CHOP",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")
        self.assertFalse(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_demotes_exact_loser_bucket(self) -> None:
        closed_trades = [
            {
                "symbol": "BTCUSD",
                "strategy_key": "BTCUSD_TREND_SCALP",
                "session_name": "LONDON",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T08:00:00+00:00",
                "pnl_r": -0.25,
                "pnl_amount": -2.5,
                "post_trade_review_json": '{"issues":["late_entry","false_breakout","immediate_invalidation"]}',
                "management_effect_json": '{"mfe_r":0.09,"mae_r":-0.42}',
                "duration_minutes": 10.0,
            },
            {
                "symbol": "BTCUSD",
                "strategy_key": "BTCUSD_TREND_SCALP",
                "session_name": "LONDON",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T09:00:00+00:00",
                "pnl_r": -0.12,
                "pnl_amount": -1.2,
                "post_trade_review_json": '{"issues":["late_entry","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.04,"mae_r":-0.30}',
                "duration_minutes": 7.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="BTCUSD",
            strategy_key="BTCUSD_TREND_SCALP",
            session_name="LONDON",
            regime_state="TRENDING",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")
        self.assertFalse(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_preserves_audjpy_tokyo_trending_pullback(self) -> None:
        closed_trades = [
            {
                "symbol": "AUDJPY",
                "strategy_key": "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.05,
                "pnl_amount": 0.5,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.30,"mae_r":-0.10}',
                "duration_minutes": 24.0,
            },
            {
                "symbol": "AUDJPY",
                "strategy_key": "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.02,
                "pnl_amount": -0.2,
                "post_trade_review_json": '{"issues":["small_giveback"]}',
                "management_effect_json": '{"mfe_r":0.18,"mae_r":-0.08}',
                "duration_minutes": 18.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="AUDJPY",
            strategy_key="AUDJPY_TOKYO_CONTINUATION_PULLBACK",
            session_name="TOKYO",
            regime_state="TRENDING",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "NORMAL")
        self.assertEqual(state["strategy_bucket_reason"], "audjpy_trending_pullback_preserved")

    def test_pair_strategy_session_performance_state_preserves_nzdjpy_tokyo_trending_pullback(self) -> None:
        closed_trades = [
            {
                "symbol": "NZDJPY",
                "strategy_key": "NZDJPY_PULLBACK_CONTINUATION",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.08,
                "pnl_amount": 0.8,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.32,"mae_r":-0.10}',
                "duration_minutes": 26.0,
            },
            {
                "symbol": "NZDJPY",
                "strategy_key": "NZDJPY_PULLBACK_CONTINUATION",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.01,
                "pnl_amount": -0.1,
                "post_trade_review_json": '{"issues":["small_giveback"]}',
                "management_effect_json": '{"mfe_r":0.22,"mae_r":-0.08}',
                "duration_minutes": 20.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="NZDJPY",
            strategy_key="NZDJPY_PULLBACK_CONTINUATION",
            session_name="TOKYO",
            regime_state="TRENDING",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "NORMAL")
        self.assertEqual(state["strategy_bucket_reason"], "nzdjpy_trending_pullback_preserved")

    def test_pair_strategy_session_performance_state_preserves_usdjpy_tokyo_sweep_reversal(self) -> None:
        closed_trades = [
            {
                "symbol": "USDJPY",
                "strategy_key": "USDJPY_LIQUIDITY_SWEEP_REVERSAL",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.12,
                "pnl_amount": 1.2,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.40,"mae_r":-0.10}',
                "duration_minutes": 19.0,
            },
            {
                "symbol": "USDJPY",
                "strategy_key": "USDJPY_LIQUIDITY_SWEEP_REVERSAL",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.01,
                "pnl_amount": -0.1,
                "post_trade_review_json": '{"issues":["small_giveback"]}',
                "management_effect_json": '{"mfe_r":0.24,"mae_r":-0.08}',
                "duration_minutes": 14.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="USDJPY",
            strategy_key="USDJPY_LIQUIDITY_SWEEP_REVERSAL",
            session_name="TOKYO",
            regime_state="RANGING",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "NORMAL")
        self.assertEqual(state["strategy_bucket_reason"], "usdjpy_tokyo_sweep_preserved")

    def test_pair_strategy_session_performance_state_promotes_nzdjpy_sydney_trap_when_proving(self) -> None:
        closed_trades = [
            {
                "symbol": "NZDJPY",
                "strategy_key": "NZDJPY_LIQUIDITY_TRAP_REVERSAL",
                "session_name": "SYDNEY",
                "regime": "RANGING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.22,
                "pnl_amount": 2.2,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.64,"mae_r":-0.10}',
                "duration_minutes": 33.0,
            },
            {
                "symbol": "NZDJPY",
                "strategy_key": "NZDJPY_LIQUIDITY_TRAP_REVERSAL",
                "session_name": "SYDNEY",
                "regime": "MEAN_REVERSION",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": 0.05,
                "pnl_amount": 0.5,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.34,"mae_r":-0.08}',
                "duration_minutes": 26.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="NZDJPY",
            strategy_key="NZDJPY_LIQUIDITY_TRAP_REVERSAL",
            session_name="SYDNEY",
            regime_state="RANGING",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "ATTACK")
        self.assertEqual(state["strategy_bucket_reason"], "nzdjpy_sydney_trap_attack")

    def test_pair_strategy_session_performance_state_promotes_usdjpy_overlap_momentum_when_proving(self) -> None:
        closed_trades = [
            {
                "symbol": "USDJPY",
                "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                "session_name": "OVERLAP",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.19,
                "pnl_amount": 1.9,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.52,"mae_r":-0.09}',
                "duration_minutes": 18.0,
            },
            {
                "symbol": "USDJPY",
                "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                "session_name": "OVERLAP",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": 0.09,
                "pnl_amount": 0.9,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.36,"mae_r":-0.07}',
                "duration_minutes": 14.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="USDJPY",
            strategy_key="USDJPY_MOMENTUM_IMPULSE",
            session_name="OVERLAP",
            regime_state="LOW_LIQUIDITY_CHOP",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "ATTACK")
        self.assertEqual(state["strategy_bucket_reason"], "usdjpy_momentum_attack")

    def test_pair_strategy_session_performance_state_promotes_usdjpy_tokyo_momentum_when_proving(self) -> None:
        closed_trades = [
            {
                "symbol": "USDJPY",
                "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.20,
                "pnl_amount": 2.0,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.62,"mae_r":-0.09}',
                "duration_minutes": 18.0,
            },
            {
                "symbol": "USDJPY",
                "strategy_key": "USDJPY_MOMENTUM_IMPULSE",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": 0.16,
                "pnl_amount": 1.6,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.58,"mae_r":-0.08}',
                "duration_minutes": 15.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="USDJPY",
            strategy_key="USDJPY_MOMENTUM_IMPULSE",
            session_name="TOKYO",
            regime_state="TRENDING",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "ATTACK")
        self.assertEqual(state["strategy_bucket_reason"], "usdjpy_asia_momentum_attack")

    def test_pair_strategy_session_performance_state_preserves_xau_mean_reversion_sweep(self) -> None:
        closed_trades = [
            {
                "symbol": "XAUUSD",
                "strategy_key": "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                "session_name": "LONDON",
                "regime": "MEAN_REVERSION",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.42,
                "pnl_amount": 4.2,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":1.00,"mae_r":-0.18}',
                "duration_minutes": 26.0,
            },
            {
                "symbol": "XAUUSD",
                "strategy_key": "XAUUSD_LONDON_LIQUIDITY_SWEEP",
                "session_name": "LONDON",
                "regime": "MEAN_REVERSION",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": 0.16,
                "pnl_amount": 1.6,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.62,"mae_r":-0.12}',
                "duration_minutes": 19.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="XAUUSD",
            strategy_key="XAUUSD_LONDON_LIQUIDITY_SWEEP",
            session_name="LONDON",
            regime_state="MEAN_REVERSION",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "ATTACK")
        self.assertEqual(state["strategy_bucket_reason"], "xau_mean_reversion_sweep_attack")

    def test_pair_strategy_session_performance_state_throttles_gbpusd_london_breakout_chop_bucket(self) -> None:
        closed_trades = [
            {
                "symbol": "GBPUSD",
                "strategy_key": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                "session_name": "LONDON",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -1.02,
                "pnl_amount": -10.2,
                "post_trade_review_json": '{"issues":["false_break"]}',
                "management_effect_json": '{"mfe_r":0.18,"mae_r":-1.05}',
                "duration_minutes": 21.0,
            },
            {
                "symbol": "GBPUSD",
                "strategy_key": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                "session_name": "LONDON",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.84,
                "pnl_amount": -8.4,
                "post_trade_review_json": '{"issues":["late_entry"]}',
                "management_effect_json": '{"mfe_r":0.10,"mae_r":-0.92}',
                "duration_minutes": 17.0,
            },
            {
                "symbol": "GBPUSD",
                "strategy_key": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                "session_name": "LONDON",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T02:00:00+00:00",
                "pnl_r": 0.12,
                "pnl_amount": 1.2,
                "post_trade_review_json": '{"issues":["small_giveback"]}',
                "management_effect_json": '{"mfe_r":0.44,"mae_r":-0.22}',
                "duration_minutes": 14.0,
            },
            {
                "symbol": "GBPUSD",
                "strategy_key": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                "session_name": "LONDON",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T03:00:00+00:00",
                "pnl_r": -0.22,
                "pnl_amount": -2.2,
                "post_trade_review_json": '{"issues":["weak_follow_through"]}',
                "management_effect_json": '{"mfe_r":0.20,"mae_r":-0.40}',
                "duration_minutes": 12.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="GBPUSD",
            strategy_key="GBPUSD_LONDON_EXPANSION_BREAKOUT",
            session_name="LONDON",
            regime_state="LOW_LIQUIDITY_CHOP",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "REDUCED")
        self.assertEqual(state["strategy_bucket_reason"], "gbpusd_london_breakout_london_chop_throttle")

    def test_pair_strategy_session_performance_state_preserves_audjpy_overlap_sweep(self) -> None:
        closed_trades = [
            {
                "symbol": "AUDJPY",
                "strategy_key": "AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
                "session_name": "OVERLAP",
                "regime": "MEAN_REVERSION",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.24,
                "pnl_amount": 2.4,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.80,"mae_r":-0.14}',
                "duration_minutes": 24.0,
            },
            {
                "symbol": "AUDJPY",
                "strategy_key": "AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
                "session_name": "OVERLAP",
                "regime": "MEAN_REVERSION",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": 0.10,
                "pnl_amount": 1.0,
                "post_trade_review_json": '{"issues":["small_giveback"]}',
                "management_effect_json": '{"mfe_r":0.60,"mae_r":-0.10}',
                "duration_minutes": 20.0,
            },
            {
                "symbol": "AUDJPY",
                "strategy_key": "AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
                "session_name": "OVERLAP",
                "regime": "MEAN_REVERSION",
                "closed_at": "2026-03-12T02:00:00+00:00",
                "pnl_r": -0.02,
                "pnl_amount": -0.2,
                "post_trade_review_json": '{"issues":["small_giveback"]}',
                "management_effect_json": '{"mfe_r":0.34,"mae_r":-0.08}',
                "duration_minutes": 18.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="AUDJPY",
            strategy_key="AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
            session_name="OVERLAP",
            regime_state="MEAN_REVERSION",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "ATTACK")
        self.assertEqual(state["strategy_bucket_reason"], "audjpy_overlap_sweep_attack")

    def test_pair_strategy_session_performance_state_throttles_audjpy_sydney_sweep_when_weak(self) -> None:
        closed_trades = [
            {
                "symbol": "AUDJPY",
                "strategy_key": "AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
                "session_name": "SYDNEY",
                "regime": "MEAN_REVERSION",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -1.01,
                "pnl_amount": -10.1,
                "post_trade_review_json": '{"issues":["fast_failure"]}',
                "management_effect_json": '{"mfe_r":0.08,"mae_r":-1.04}',
                "duration_minutes": 7.0,
            },
            {
                "symbol": "AUDJPY",
                "strategy_key": "AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
                "session_name": "SYDNEY",
                "regime": "MEAN_REVERSION",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.16,
                "pnl_amount": -1.6,
                "post_trade_review_json": '{"issues":["small_giveback"]}',
                "management_effect_json": '{"mfe_r":0.14,"mae_r":-0.20}',
                "duration_minutes": 11.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="AUDJPY",
            strategy_key="AUDJPY_LIQUIDITY_SWEEP_REVERSAL",
            session_name="SYDNEY",
            regime_state="MEAN_REVERSION",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "REDUCED")
        self.assertEqual(state["strategy_bucket_reason"], "audjpy_sydney_sweep_throttle")

    def test_verify_mode_forces_trading_disabled(self) -> None:
        enabled, reason = determine_trading_state(
            "DEMO",
            {"trading_enabled": True},
            live_allowed=True,
            verify_only=True,
        )

        self.assertFalse(enabled)
        self.assertEqual(reason, "verify_mode")

    def test_live_mode_requires_trading_and_live_flags(self) -> None:
        enabled_ok, reason_ok = determine_trading_state(
            "LIVE",
            {"trading_enabled": True, "live_trading_enabled": True},
            live_allowed=True,
            verify_only=False,
        )
        enabled_blocked, reason_blocked = determine_trading_state(
            "LIVE",
            {"trading_enabled": False, "live_trading_enabled": True},
            live_allowed=True,
            verify_only=False,
        )

        self.assertTrue(enabled_ok)
        self.assertEqual(reason_ok, "live_unlocked")
        self.assertFalse(enabled_blocked)
        self.assertEqual(reason_blocked, "live_blocked_trading_disabled")

    def test_symbol_normalization_for_crypto_alias(self) -> None:
        self.assertEqual(_normalize_symbol_key("btcusdt"), "BTCUSD")
        self.assertEqual(_normalize_symbol_key("gold.m"), "XAUUSD")

    def test_always_on_symbol_recognizes_btc_alias(self) -> None:
        always_on = {"BTCUSD"}
        self.assertTrue(_is_always_on_symbol("BTCUSD", "BTCUSDm", always_on))

    def test_symbol_entry_cap_uses_density_profile_cap_for_hot_pairs(self) -> None:
        candidates = [
            SignalCandidate(
                signal_id="usdjpy-hot",
                setup="USDJPY_MOMENTUM_IMPULSE",
                side="BUY",
                score_hint=0.72,
                reason="tokyo momentum",
                stop_atr=1.0,
                tp_r=1.8,
                strategy_family="TREND",
                meta={"density_entry_cap": 5},
            )
        ]

        self.assertEqual(
            _symbol_entry_cap("USDJPY", candidates, default_cap=2, xau_grid_cap=24),
            5,
        )

    def test_symbol_entry_cap_preserves_xau_grid_burst_override(self) -> None:
        candidates = [
            SignalCandidate(
                signal_id="xau-grid-1",
                setup="XAUUSD_M5_GRID_SCALPER_START",
                side="BUY",
                score_hint=0.82,
                reason="grid start",
                stop_atr=1.0,
                tp_r=1.8,
                strategy_family="GRID",
                meta={"grid_cycle": True, "grid_cycle_id": "c1", "grid_level": 1},
            ),
            SignalCandidate(
                signal_id="xau-grid-2",
                setup="XAUUSD_M5_GRID_SCALPER_START",
                side="BUY",
                score_hint=0.81,
                reason="grid add",
                stop_atr=1.0,
                tp_r=1.8,
                strategy_family="GRID",
                meta={"grid_cycle": True, "grid_cycle_id": "c1", "grid_level": 2},
            ),
        ]

        self.assertEqual(
            _symbol_entry_cap("XAUUSD", candidates, default_cap=4, xau_grid_cap=24),
            24,
        )

    def test_micro_position_caps_use_bootstrap_caps_under_threshold(self) -> None:
        total, per_symbol, active = _micro_position_caps(
            micro_config={
                "enabled": True,
                "bootstrap_enabled": True,
                "bootstrap_equity_threshold": 100,
                "bootstrap_max_positions_total": 3,
                "bootstrap_max_positions_per_symbol": 2,
                "one_trade_at_a_time": True,
                "one_trade_until_equity": 50,
            },
            mode="LIVE",
            equity=20.0,
            base_total=10,
            base_per_symbol=5,
        )
        self.assertTrue(active)
        self.assertEqual(total, 3)
        self.assertEqual(per_symbol, 2)

    def test_micro_position_caps_fall_back_to_one_trade_when_bootstrap_disabled(self) -> None:
        total, per_symbol, active = _micro_position_caps(
            micro_config={"enabled": True, "bootstrap_enabled": False, "one_trade_at_a_time": True, "one_trade_until_equity": 50},
            mode="LIVE",
            equity=20.0,
            base_total=10,
            base_per_symbol=5,
        )
        self.assertTrue(active)
        self.assertEqual(total, 1)
        self.assertEqual(per_symbol, 1)

    def test_micro_lot_cap_scales_with_equity_thresholds(self) -> None:
        config = {
            "enabled": True,
            "max_lot_cap_low": 0.01,
            "max_lot_cap_mid": 0.02,
            "max_lot_cap_high": 0.05,
            "lot_cap_mid_equity": 50,
            "lot_cap_high_equity": 100,
        }
        self.assertEqual(_micro_lot_cap(config, "LIVE", 20), 0.01)
        self.assertEqual(_micro_lot_cap(config, "LIVE", 70), 0.02)
        self.assertEqual(_micro_lot_cap(config, "LIVE", 120), 0.05)

    def test_fallback_account_snapshot_uses_internal_estimate(self) -> None:
        snapshot = _fallback_account_snapshot(50.0)
        self.assertEqual(float(snapshot["equity"]), 50.0)
        self.assertEqual(float(snapshot["balance"]), 50.0)
        self.assertEqual(float(snapshot["margin_free"]), 50.0)

    def test_bridge_snapshot_preserves_live_equity_when_mt5_feed_is_unavailable(self) -> None:
        merged, label, bridge_active = _apply_runtime_account_snapshot(
            _fallback_account_snapshot(50.0),
            account_from_mt5=False,
            bridge_snapshot={
                "balance": 53.96,
                "equity": 63.96,
                "free_margin": 63.96,
                "margin": 0.0,
                "margin_level": 0.0,
                "leverage": 500,
            },
            internal_equity_estimate=50.0,
        )

        self.assertTrue(bridge_active)
        self.assertEqual(label, "LIVE_BRIDGE_FEED")
        self.assertEqual(float(merged["balance"]), 53.96)
        self.assertEqual(float(merged["equity"]), 63.96)
        self.assertEqual(float(merged["margin_free"]), 63.96)
        self.assertEqual(float(merged["leverage"]), 500.0)

    def test_pair_session_state_promotes_native_pair_with_proving_edge(self) -> None:
        closed_trades = [
            {
                "symbol": "AUDJPY",
                "session_name": "SYDNEY",
                "closed_at": "2026-03-12T01:10:00+13:00",
                "pnl_amount": 1.2,
                "pnl_r": 0.7,
                "setup": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
            },
            {
                "symbol": "AUDJPY",
                "session_name": "SYDNEY",
                "closed_at": "2026-03-12T02:10:00+13:00",
                "pnl_amount": 1.5,
                "pnl_r": 0.8,
                "setup": "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
            },
            {
                "symbol": "AUDJPY",
                "session_name": "SYDNEY",
                "closed_at": "2026-03-12T03:10:00+13:00",
                "pnl_amount": 0.8,
                "pnl_r": 0.4,
                "setup": "AUDJPY_SYDNEY_RANGE_BREAK",
            },
            {
                "symbol": "AUDJPY",
                "session_name": "SYDNEY",
                "closed_at": "2026-03-12T04:10:00+13:00",
                "pnl_amount": 1.1,
                "pnl_r": 0.6,
                "setup": "AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
            },
        ]

        state = _pair_session_performance_state(
            symbol="AUDJPY",
            session_name="SYDNEY",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["pair_status"], "ATTACK")
        self.assertEqual(state["why_pair_is_promoted"], "native_pair_proving_edge_today")
        self.assertGreater(float(state["pair_state_multiplier"]), 1.0)

    def test_pair_session_state_quarantines_weak_pair_session_edge(self) -> None:
        closed_trades = [
            {
                "symbol": "GBPUSD",
                "session_name": "TOKYO",
                "closed_at": "2026-03-12T01:10:00+13:00",
                "pnl_amount": -1.1,
                "pnl_r": -0.7,
                "setup": "TOKYO_FAKE_BREAKOUT",
            },
            {
                "symbol": "GBPUSD",
                "session_name": "TOKYO",
                "closed_at": "2026-03-12T02:10:00+13:00",
                "pnl_amount": -1.3,
                "pnl_r": -0.8,
                "setup": "TOKYO_BREAKOUT",
            },
            {
                "symbol": "GBPUSD",
                "session_name": "TOKYO",
                "closed_at": "2026-03-12T03:10:00+13:00",
                "pnl_amount": -0.9,
                "pnl_r": -0.5,
                "setup": "TOKYO_BREAKOUT_RETEST",
            },
            {
                "symbol": "GBPUSD",
                "session_name": "TOKYO",
                "closed_at": "2026-03-12T04:10:00+13:00",
                "pnl_amount": -1.0,
                "pnl_r": -0.6,
                "setup": "TOKYO_MOMENTUM_BREAKOUT",
            },
            {
                "symbol": "GBPUSD",
                "session_name": "TOKYO",
                "closed_at": "2026-03-12T05:10:00+13:00",
                "pnl_amount": -0.7,
                "pnl_r": -0.4,
                "setup": "TOKYO_BREAKOUT",
            },
        ]

        state = _pair_session_performance_state(
            symbol="GBPUSD",
            session_name="TOKYO",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["pair_status"], "QUARANTINED")
        self.assertEqual(state["why_pair_is_throttled"], "weak_pair_session_edge")
        self.assertLess(float(state["pair_state_multiplier"]), 1.0)

    def test_pre_risk_stop_geometry_uses_live_broker_minimum_for_fx(self) -> None:
        result = _normalize_pre_risk_exit_geometry(
            symbol="EURUSD",
            side="BUY",
            entry_price=1.15267,
            stop_price=1.15262,
            tp_price=1.15280,
            spread_points=14.0,
            symbol_info={
                "digits": 5,
                "trade_tick_size": 0.00001,
                "point": 0.00001,
                "stops_level": 0,
                "freeze_level": 0,
                "trade_tick_value": 1.706456,
                "trade_contract_size": 100000.0,
            },
            symbol_rules=load_symbol_rules(),
            safety_buffer_points=5,
        )

        widened_points = round((1.15267 - float(result["stop_price"])) / 0.00001, 5)
        self.assertTrue(bool(result["validated"]))
        self.assertAlmostEqual(float(result["min_stop_distance_points"]), 19.0, places=6)
        self.assertAlmostEqual(float(widened_points), 19.0, places=5)
        self.assertTrue(bool(result["clamped"]))

    def test_prep_checks_complete_is_true_when_market_is_open(self) -> None:
        self.assertTrue(_prep_checks_complete({}, {}, True))
        self.assertTrue(_prep_checks_complete({"pre_open_news_summary": "clear"}, {}, False))
        self.assertFalse(_prep_checks_complete({}, {}, False))

    def test_effective_live_entry_price_prefers_bridge_snapshot_quote(self) -> None:
        entry_price, source = _effective_live_entry_price(
            side="BUY",
            tick={"bid": 5076.10, "ask": 5076.40},
            symbol_info={"bid": 5081.23, "ask": 5081.53, "economics_source": "mt5_client+bridge_snapshot"},
        )

        self.assertEqual(source, "bridge_snapshot")
        self.assertAlmostEqual(entry_price, 5081.53, places=6)

    def test_preserve_approved_broker_min_lot_keeps_floor(self) -> None:
        adjusted = _preserve_approved_broker_min_lot(
            normalized_volume=0.02,
            approved_volume=1.0,
            broker_min_lot=1.0,
            preserve_min_lot=True,
        )

        self.assertEqual(adjusted, 1.0)

    def test_runtime_bootstrap_tolerance_cap_scales_for_nas100_min_lot(self) -> None:
        tolerance = _runtime_bootstrap_tolerance_cap(
            symbol_key="NAS100",
            per_trade_cap=4.0,
            account_equity=82.0,
            bootstrap_equity_threshold=160.0,
            broker_min_lot=0.1,
        )

        self.assertAlmostEqual(tolerance, 6.8, places=6)

    def test_runtime_bootstrap_trade_allowed_accepts_nas100_broker_min_lot(self) -> None:
        allowed, tolerance = _runtime_bootstrap_trade_allowed(
            symbol_key="NAS100",
            projected_loss_usd=6.15,
            volume=0.1,
            broker_min_lot=0.1,
            projected_open_risk_usd=0.0,
            per_trade_cap=4.0,
            total_exposure_cap=10.0,
            account_equity=82.0,
            bootstrap_equity_threshold=160.0,
        )

        self.assertTrue(allowed)
        self.assertGreaterEqual(tolerance, 6.15)

    def test_effective_min_stop_distance_points_prefers_live_broker_value(self) -> None:
        self.assertEqual(_effective_min_stop_distance_points(20.0, 19.0), 19.0)
        self.assertEqual(_effective_min_stop_distance_points(20.0, 0.0), 20.0)

    def test_resolve_candidate_stop_distance_prefers_explicit_stop_points(self) -> None:
        candidate = SignalCandidate(
            signal_id="sig",
            setup="XAUUSD_M5_GRID_SCALPER_START",
            side="BUY",
            score_hint=0.6,
            reason="test",
            stop_atr=1.5,
            tp_r=1.5,
            meta={"stop_points": 240.0},
        )

        stop_distance, source = _resolve_candidate_stop_distance(
            candidate=candidate,
            atr_for_candidate=12.0,
            point_size=0.01,
            sl_multiplier=2.0,
        )

        self.assertAlmostEqual(stop_distance, 2.4, places=6)
        self.assertEqual(source, "explicit_points")

    def test_resolve_bridge_symbol_snapshot_falls_back_to_latest_account_snapshot(self) -> None:
        class StubQueue:
            def get_account_snapshot(self, *, account: str, symbol: str, magic: int):
                return None

            def latest_account_snapshot(self, *, account: str, magic: int, symbol: str):
                return {"account": account, "magic": magic, "symbol_key": symbol, "point": 0.00001}

        snapshot = _resolve_bridge_symbol_snapshot(
            StubQueue(),
            account="Main",
            magic=7777,
            symbol="EURUSD",
        )

        self.assertIsNotNone(snapshot)
        self.assertEqual(float(snapshot["point"]), 0.00001)

    def test_resolve_runtime_symbol_info_merges_bridge_snapshot_once_and_caches(self) -> None:
        class StubMt5:
            def __init__(self) -> None:
                self.calls = 0

            def get_symbol_info(self, symbol: str):
                self.calls += 1
                return {"symbol": symbol, "point": 0.01, "ask": 66352.34, "bid": 66335.34}

        class StubQueue:
            def get_account_snapshot(self, *, account: str, symbol: str, magic: int):
                return None

            def latest_account_snapshot(self, *, account: str, magic: int, symbol: str):
                return {"account": account, "magic": magic, "symbol_key": symbol, "spread_points": 1700.0}

        mt5 = StubMt5()
        cache: dict[str, dict] = {}
        info = _resolve_runtime_symbol_info(
            mt5_client=mt5,
            bridge_queue=StubQueue(),
            symbol_info_cache=cache,
            symbol="BTCUSD",
            bridge_trade_mode=True,
            bridge_context_account="Main",
            bridge_context_magic=7777,
        )

        self.assertEqual(mt5.calls, 1)
        self.assertEqual(float(info["point"]), 0.01)
        self.assertIn("mt5_client", str(info["economics_source"]))
        self.assertEqual(float(info["bridge_snapshot"]["spread_points"]), 1700.0)
        self.assertIn("BTCUSD", cache)

        cached = _resolve_runtime_symbol_info(
            mt5_client=mt5,
            bridge_queue=StubQueue(),
            symbol_info_cache=cache,
            symbol="BTCUSD",
            bridge_trade_mode=True,
            bridge_context_account="Main",
            bridge_context_magic=7777,
        )

        self.assertIs(cached, info)
        self.assertEqual(mt5.calls, 1)

    def test_filter_broker_confirmed_positions_drops_stale_rows_when_account_snapshot_is_flat(self) -> None:
        now = datetime(2026, 3, 9, 1, 0, tzinfo=timezone.utc)
        confirmed, stale = _filter_broker_confirmed_positions(
            positions=[
                {
                    "signal_id": "sig-stale",
                    "symbol": "EURUSD",
                    "ticket": "123",
                    "opened_at": "2026-03-09T00:10:00+00:00",
                }
            ],
            account_snapshot={
                "updated_at": "2026-03-09T00:59:00+00:00",
                "total_open_positions": 0,
            },
            symbol_snapshots={
                "EURUSD": {
                    "updated_at": "2026-03-09T00:59:00+00:00",
                    "open_count": 0,
                }
            },
            now_ts=now,
        )

        self.assertEqual(confirmed, [])
        self.assertEqual(len(stale), 1)
        self.assertEqual(stale[0]["reason"], "broker_account_flat")

    def test_filter_broker_confirmed_positions_keeps_recent_unconfirmed_trade_within_grace(self) -> None:
        now = datetime(2026, 3, 9, 1, 0, tzinfo=timezone.utc)
        confirmed, stale = _filter_broker_confirmed_positions(
            positions=[
                {
                    "signal_id": "sig-fresh",
                    "symbol": "BTCUSD",
                    "ticket": "456",
                    "opened_at": "2026-03-09T00:59:10+00:00",
                }
            ],
            account_snapshot={
                "updated_at": "2026-03-09T00:59:30+00:00",
                "total_open_positions": 0,
            },
            symbol_snapshots={
                "BTCUSD": {
                    "updated_at": "2026-03-09T00:59:30+00:00",
                    "open_count": 0,
                }
            },
            now_ts=now,
        )

        self.assertEqual(len(confirmed), 1)
        self.assertEqual(stale, [])

    def test_account_scaling_update_detects_material_deposit_and_band_change(self) -> None:
        scaling = _detect_account_scaling_update(
            {"balance": 50.0, "equity": 50.0, "margin_free": 50.0, "high_watermark_equity": 55.0},
            {"balance": 85.0, "equity": 90.0, "margin_free": 88.0},
            bootstrap_equity_threshold=160.0,
        )

        self.assertTrue(bool(scaling["material_change_detected"]))
        self.assertTrue(bool(scaling["account_increase_detected"]))
        self.assertTrue(bool(scaling["sizing_updated"]))
        self.assertEqual(str(scaling["equity_band"]), "bootstrap_balanced")
        self.assertGreater(float(scaling["high_watermark_equity"]), 89.0)

    def test_phase_state_promotes_only_with_equity_and_performance(self) -> None:
        phase = _phase_state(
            140.0,
            {
                "overall": {
                    "trades": 16,
                    "win_rate": 0.56,
                    "expectancy_r": 0.12,
                }
            },
        )

        self.assertEqual(phase["current_phase"], "PHASE_2")
        self.assertEqual(phase["current_daily_trade_cap"], 280)
        self.assertEqual(phase["current_overflow_daily_trade_cap"], 760)
        self.assertEqual(phase["current_ai_threshold_mode"], "moderate")
        self.assertEqual(phase["current_max_risk_pct"], 0.025)

    def test_phase_state_does_not_promote_on_equity_alone(self) -> None:
        phase = _phase_state(
            140.0,
            {
                "overall": {
                    "trades": 4,
                    "win_rate": 0.25,
                    "expectancy_r": -0.20,
                }
            },
        )

        self.assertEqual(phase["current_phase"], "PHASE_1")
        self.assertEqual(phase["phase_reason"], "equity_up_but_performance_unproven")
        self.assertEqual(phase["current_daily_trade_cap"], 240)
        self.assertEqual(phase["current_overflow_daily_trade_cap"], 720)
        self.assertEqual(phase["current_max_risk_pct"], 0.01)

    def test_internal_snapshot_clamps_when_no_mt5_or_bridge_feed_exists(self) -> None:
        merged, label, bridge_active = _apply_runtime_account_snapshot(
            {"balance": 80.0, "equity": 75.0, "margin_free": 72.0},
            account_from_mt5=False,
            bridge_snapshot=None,
            internal_equity_estimate=50.0,
        )

        self.assertFalse(bridge_active)
        self.assertEqual(label, "INTERNAL (NO MT5 EQUITY FEED)")
        self.assertEqual(float(merged["balance"]), 50.0)
        self.assertEqual(float(merged["equity"]), 50.0)
        self.assertEqual(float(merged["margin_free"]), 50.0)

    def test_weekend_market_mode_keeps_btc_open_and_xau_closed(self) -> None:
        saturday_utc = datetime(2026, 3, 7, 10, 0, tzinfo=timezone.utc)
        self.assertTrue(_is_weekend_market_mode(saturday_utc))
        btc_open, btc_status = _market_open_status("BTCUSD", saturday_utc)
        xau_open, xau_status = _market_open_status("XAUUSD", saturday_utc)
        self.assertTrue(btc_open)
        self.assertEqual(btc_status, "OPEN_24_7")
        self.assertFalse(xau_open)
        self.assertEqual(xau_status, "WEEKEND_CLOSED")

    def test_forex_week_opens_at_sunday_5pm_new_york_after_dst_shift(self) -> None:
        before_open = datetime(2026, 3, 8, 20, 30, tzinfo=timezone.utc)
        after_open = datetime(2026, 3, 8, 21, 30, tzinfo=timezone.utc)

        before_open_state = _market_open_status("EURUSD", before_open)
        after_open_state = _market_open_status("EURUSD", after_open)

        self.assertEqual(before_open_state, (False, "PRE_OPEN"))
        self.assertEqual(after_open_state, (True, "OPEN"))

    def test_xau_transitions_from_pre_open_to_open_on_sunday_evening_new_york(self) -> None:
        before_open = datetime(2026, 3, 8, 21, 30, tzinfo=timezone.utc)
        after_open = datetime(2026, 3, 8, 22, 30, tzinfo=timezone.utc)

        before_open_state = _market_open_status("XAUUSD", before_open)
        after_open_state = _market_open_status("XAUUSD", after_open)

        self.assertEqual(before_open_state, (False, "PRE_OPEN"))
        self.assertEqual(after_open_state, (True, "OPEN"))

    def test_xau_timeframe_route_resolves_fast_m3_stack_from_m15_request(self) -> None:
        route = _resolve_timeframe_route(
            "XAUUSD",
            "M15",
            {
                "attached_tf_fallback_enabled": True,
                "preferred_execution_tf_by_symbol": {"XAUUSD": "M3"},
                "internal_analysis_tfs_by_symbol": {"XAUUSD": ["M1", "M3", "M5", "M15"]},
                "symbol_timeframe_map": {
                    "XAUUSD": {
                        "accepted": ["M15", "M5", "M3", "M1"],
                        "execution": "M3",
                        "internal": ["M1", "M3", "M5", "M15"],
                    }
                },
            },
        )
        self.assertEqual(route["requested_timeframe"], "M15")
        self.assertEqual(route["execution_timeframe_used"], "M3")
        self.assertEqual(route["internal_timeframes_used"], ["M1", "M3", "M5", "M15"])
        self.assertTrue(route["attachment_dependency_resolved"])

    def test_usdjpy_timeframe_route_uses_fast_execution_profile(self) -> None:
        route = _resolve_timeframe_route("USDJPY", "M15", {"attached_tf_fallback_enabled": True})

        self.assertEqual(route["requested_timeframe"], "M15")
        self.assertEqual(route["execution_timeframe_used"], "M3")
        self.assertEqual(route["internal_timeframes_used"], ["M1", "M3", "M5", "M15"])

    def test_soft_kill_recovery_note_clears_recovered_drawdown_kill(self) -> None:
        note = _soft_kill_recovery_note(
            "rolling_drawdown_kill",
            now=datetime(2026, 3, 9, 0, 0, tzinfo=timezone.utc),
            system_config={"force_flat_friday_gmt": "20:00"},
            risk_config={"max_drawdown_kill": 0.15, "circuit_breaker_daily_loss": 0.10},
            global_stats=SimpleNamespace(rolling_drawdown_pct=0.0, daily_pnl_pct=0.0),
        )
        self.assertEqual(note, "rolling_drawdown_recovered:0.000000<0.150000")

    def test_soft_kill_recovery_note_uses_bootstrap_drawdown_threshold(self) -> None:
        note = _soft_kill_recovery_note(
            "rolling_drawdown_kill",
            now=datetime(2026, 3, 9, 0, 0, tzinfo=timezone.utc),
            system_config={"force_flat_friday_gmt": "20:00"},
            risk_config={"max_drawdown_kill": 0.05, "circuit_breaker_daily_loss": 0.10},
            micro_config={"bootstrap_enabled": True, "bootstrap_equity_threshold": 160.0, "bootstrap_drawdown_kill": 0.12},
            account_equity=63.57,
            global_stats=SimpleNamespace(rolling_drawdown_pct=0.055, daily_pnl_pct=0.0),
        )
        self.assertEqual(note, "rolling_drawdown_recovered:0.055000<0.120000")

    def test_soft_kill_recovery_note_uses_bootstrap_absolute_drawdown_threshold(self) -> None:
        note = _soft_kill_recovery_note(
            "absolute_drawdown_hard_stop",
            now=datetime(2026, 3, 9, 0, 0, tzinfo=timezone.utc),
            system_config={"force_flat_friday_gmt": "20:00"},
            risk_config={"absolute_drawdown_hard_stop": 0.08, "max_drawdown_kill": 0.05, "circuit_breaker_daily_loss": 0.10},
            micro_config={"bootstrap_enabled": True, "bootstrap_equity_threshold": 160.0, "bootstrap_drawdown_kill": 0.12},
            account_equity=63.57,
            global_stats=SimpleNamespace(absolute_drawdown_pct=0.089, rolling_drawdown_pct=0.055, daily_pnl_pct=0.0),
        )
        self.assertEqual(note, "absolute_drawdown_recovered:0.089000<0.120000")

    def test_soft_kill_recovery_note_keeps_active_daily_circuit_breaker(self) -> None:
        note = _soft_kill_recovery_note(
            "daily_circuit_breaker",
            now=datetime(2026, 3, 9, 0, 0, tzinfo=timezone.utc),
            system_config={"force_flat_friday_gmt": "20:00"},
            risk_config={"max_drawdown_kill": 0.15, "circuit_breaker_daily_loss": 0.10},
            global_stats=SimpleNamespace(rolling_drawdown_pct=0.0, daily_pnl_pct=-0.11),
        )
        self.assertIsNone(note)

    def test_soft_kill_recovery_note_clears_hard_daily_dd_on_new_sydney_day(self) -> None:
        note = _soft_kill_recovery_note(
            "hard_daily_dd",
            now=datetime(2026, 3, 9, 21, 38, tzinfo=timezone.utc),
            created_at="2026-03-09T11:00:18.174559+00:00",
            system_config={"force_flat_friday_gmt": "20:00"},
            risk_config={"hard_daily_dd_pct": 0.05},
            global_stats=SimpleNamespace(daily_dd_pct_live=0.0, daily_pnl_pct=0.0),
        )
        self.assertEqual("new_trading_day:2026-03-09->2026-03-10", note)

    def test_approve_small_session_allowed_extends_to_tokyo_sydney_for_fx_xau_and_btc(self) -> None:
        now = datetime(2026, 3, 9, 0, 30, tzinfo=timezone.utc)
        self.assertTrue(_approve_small_session_allowed("BTCUSD", "SYDNEY", now))
        self.assertTrue(_approve_small_session_allowed("EURUSD", "TOKYO", now))
        self.assertTrue(_approve_small_session_allowed("XAUUSD", "SYDNEY", now))
        self.assertFalse(_approve_small_session_allowed("NAS100", "SYDNEY", now))
        self.assertFalse(_approve_small_session_allowed("AAPL", "TOKYO", now))

    def test_resolve_timeframe_route_applies_generic_equity_defaults(self) -> None:
        route = _resolve_timeframe_route("AAPL", "M5", {"attached_tf_fallback_enabled": True})

        self.assertEqual(route["requested_timeframe"], "M5")
        self.assertEqual(route["execution_timeframe_used"], "M15")
        self.assertEqual(route["internal_timeframes_used"], ["M5", "M15", "H1"])
        self.assertTrue(route["attachment_dependency_resolved"])

    def test_pair_strategy_session_performance_state_demotes_audnzd_vwap_reversion_tokyo_ranging(self) -> None:
        closed_trades = [
            {
                "symbol": "AUDNZD",
                "strategy_key": "AUDNZD_VWAP_MEAN_REVERSION",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -0.72,
                "pnl_amount": -7.2,
                "post_trade_review_json": '{"issues":["weak_retest","poor_structure","late_entry"]}',
                "management_effect_json": '{"mfe_r":0.05,"mae_r":-0.82}',
                "duration_minutes": 12.0,
            },
            {
                "symbol": "AUDNZD",
                "strategy_key": "AUDNZD_VWAP_MEAN_REVERSION",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.64,
                "pnl_amount": -6.4,
                "post_trade_review_json": '{"issues":["weak_retest","poor_structure","immediate_invalidation"]}',
                "management_effect_json": '{"mfe_r":0.03,"mae_r":-0.76}',
                "duration_minutes": 10.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="AUDNZD",
            strategy_key="AUDNZD_VWAP_MEAN_REVERSION",
            session_name="TOKYO",
            regime_state="RANGING",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")
        self.assertTrue(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_demotes_eurusd_vwap_pullback_london_ranging(self) -> None:
        closed_trades = [
            {
                "symbol": "EURUSD",
                "strategy_key": "EURUSD_VWAP_PULLBACK",
                "session_name": "LONDON",
                "regime": "RANGING",
                "closed_at": "2026-03-12T08:00:00+00:00",
                "pnl_r": -0.81,
                "pnl_amount": -8.1,
                "post_trade_review_json": '{"issues":["late_entry","poor_structure","weak_retest"]}',
                "management_effect_json": '{"mfe_r":0.04,"mae_r":-0.85}',
                "duration_minutes": 14.0,
            },
            {
                "symbol": "EURUSD",
                "strategy_key": "EURUSD_VWAP_PULLBACK",
                "session_name": "LONDON",
                "regime": "RANGING",
                "closed_at": "2026-03-12T09:00:00+00:00",
                "pnl_r": -0.73,
                "pnl_amount": -7.3,
                "post_trade_review_json": '{"issues":["late_entry","poor_structure","immediate_invalidation"]}',
                "management_effect_json": '{"mfe_r":0.03,"mae_r":-0.78}',
                "duration_minutes": 11.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="EURUSD",
            strategy_key="EURUSD_VWAP_PULLBACK",
            session_name="LONDON",
            regime_state="RANGING",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")
        self.assertTrue(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_demotes_nas100_tokyo_trending_momentum_impulse(self) -> None:
        closed_trades = [
            {
                "symbol": "NAS100",
                "strategy_key": "NAS100_MOMENTUM_IMPULSE",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -0.66,
                "pnl_amount": -6.6,
                "post_trade_review_json": '{"issues":["late_entry","chased_extension","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.09,"mae_r":-0.71}',
                "duration_minutes": 9.0,
            },
            {
                "symbol": "NAS100",
                "strategy_key": "NAS100_MOMENTUM_IMPULSE",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.61,
                "pnl_amount": -6.1,
                "post_trade_review_json": '{"issues":["late_entry","poor_structure","fast_failure"]}',
                "management_effect_json": '{"mfe_r":0.08,"mae_r":-0.68}',
                "duration_minutes": 8.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="NAS100",
            strategy_key="NAS100_MOMENTUM_IMPULSE",
            session_name="TOKYO",
            regime_state="TRENDING",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")
        self.assertTrue(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_attacks_audnzd_compression_release_in_tokyo(self) -> None:
        closed_trades = [
            {
                "symbol": "AUDNZD",
                "strategy_key": "AUDNZD_COMPRESSION_RELEASE",
                "session_name": "TOKYO",
                "regime": "BREAKOUT_EXPANSION",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.22,
                "pnl_amount": 2.2,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.72,"mae_r":-0.10}',
                "duration_minutes": 22.0,
            },
            {
                "symbol": "AUDNZD",
                "strategy_key": "AUDNZD_COMPRESSION_RELEASE",
                "session_name": "TOKYO",
                "regime": "BREAKOUT_EXPANSION",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": 0.18,
                "pnl_amount": 1.8,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.64,"mae_r":-0.08}',
                "duration_minutes": 18.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="AUDNZD",
            strategy_key="AUDNZD_COMPRESSION_RELEASE",
            session_name="TOKYO",
            regime_state="BREAKOUT_EXPANSION",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "ATTACK")
        self.assertEqual(state["strategy_bucket_reason"], "audnzd_compression_release_attack")

    def test_pair_strategy_session_performance_state_attacks_nzdjpy_tokyo_breakout(self) -> None:
        closed_trades = [
            {
                "symbol": "NZDJPY",
                "strategy_key": "NZDJPY_TOKYO_BREAKOUT",
                "session_name": "TOKYO",
                "regime": "BREAKOUT_EXPANSION",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.24,
                "pnl_amount": 2.4,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.78,"mae_r":-0.12}',
                "duration_minutes": 20.0,
            },
            {
                "symbol": "NZDJPY",
                "strategy_key": "NZDJPY_TOKYO_BREAKOUT",
                "session_name": "TOKYO",
                "regime": "BREAKOUT_EXPANSION",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": 0.16,
                "pnl_amount": 1.6,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.60,"mae_r":-0.10}',
                "duration_minutes": 17.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="NZDJPY",
            strategy_key="NZDJPY_TOKYO_BREAKOUT",
            session_name="TOKYO",
            regime_state="BREAKOUT_EXPANSION",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "ATTACK")
        self.assertEqual(state["strategy_bucket_reason"], "nzdjpy_tokyo_breakout_attack")

    def test_pair_strategy_session_performance_state_preserves_xau_density_recovery_lane(self) -> None:
        closed_trades = [
            {
                "symbol": "XAUUSD",
                "strategy_key": "XAUUSD_ATR_EXPANSION_SCALPER",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": 0.10,
                "pnl_amount": 1.0,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.74,"mae_r":-0.08,"reason":"profit_lock"}',
                "duration_minutes": 7.0,
            },
            {
                "symbol": "XAUUSD",
                "strategy_key": "XAUUSD_ATR_EXPANSION_SCALPER",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": 0.09,
                "pnl_amount": 0.9,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.60,"mae_r":-0.06,"reason":"trail"}',
                "duration_minutes": 6.0,
            },
            {
                "symbol": "XAUUSD",
                "strategy_key": "XAUUSD_ATR_EXPANSION_SCALPER",
                "session_name": "TOKYO",
                "regime": "RANGING",
                "closed_at": "2026-03-12T02:00:00+00:00",
                "pnl_r": -0.12,
                "pnl_amount": -1.2,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.26,"mae_r":-0.28}',
                "duration_minutes": 8.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="XAUUSD",
            strategy_key="XAUUSD_ATR_EXPANSION_SCALPER",
            session_name="TOKYO",
            regime_state="RANGING",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "NORMAL")
        self.assertEqual(state["strategy_bucket_reason"], "density_recovery_lane_preserved")
        self.assertFalse(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_demotes_gbpusd_london_breakout_ranging(self) -> None:
        closed_trades = [
            {
                "symbol": "GBPUSD",
                "strategy_key": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                "session_name": "LONDON",
                "regime": "RANGING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -0.74,
                "pnl_amount": -7.4,
                "post_trade_review_json": '{"issues":["late_entry","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.05,"mae_r":-0.80}',
                "duration_minutes": 11.0,
            },
            {
                "symbol": "GBPUSD",
                "strategy_key": "GBPUSD_LONDON_EXPANSION_BREAKOUT",
                "session_name": "LONDON",
                "regime": "RANGING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.68,
                "pnl_amount": -6.8,
                "post_trade_review_json": '{"issues":["fast_failure","immediate_invalidation"]}',
                "management_effect_json": '{"mfe_r":0.04,"mae_r":-0.73}',
                "duration_minutes": 9.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="GBPUSD",
            strategy_key="GBPUSD_LONDON_EXPANSION_BREAKOUT",
            session_name="LONDON",
            regime_state="RANGING",
            session_native_pair=True,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "REDUCED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_growth_bucket_soft_protect")

    def test_pair_strategy_session_performance_state_demotes_nas100_opening_drive_london_ranging(self) -> None:
        closed_trades = [
            {
                "symbol": "NAS100",
                "strategy_key": "NAS100_OPENING_DRIVE_BREAKOUT",
                "session_name": "LONDON",
                "regime": "RANGING",
                "closed_at": "2026-03-12T08:00:00+00:00",
                "pnl_r": -0.70,
                "pnl_amount": -7.0,
                "post_trade_review_json": '{"issues":["late_entry","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.06,"mae_r":-0.76}',
                "duration_minutes": 9.0,
            },
            {
                "symbol": "NAS100",
                "strategy_key": "NAS100_OPENING_DRIVE_BREAKOUT",
                "session_name": "LONDON",
                "regime": "RANGING",
                "closed_at": "2026-03-12T09:00:00+00:00",
                "pnl_r": -0.66,
                "pnl_amount": -6.6,
                "post_trade_review_json": '{"issues":["fast_failure","spread_wide_exit"]}',
                "management_effect_json": '{"mfe_r":0.05,"mae_r":-0.72}',
                "duration_minutes": 8.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="NAS100",
            strategy_key="NAS100_OPENING_DRIVE_BREAKOUT",
            session_name="LONDON",
            regime_state="RANGING",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "REDUCED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_growth_bucket_soft_protect")

    def test_pair_strategy_session_performance_state_demotes_usoil_inventory_london_ranging(self) -> None:
        closed_trades = [
            {
                "symbol": "USOIL",
                "strategy_key": "USOIL_INVENTORY_MOMENTUM",
                "session_name": "LONDON",
                "regime": "RANGING",
                "closed_at": "2026-03-12T08:00:00+00:00",
                "pnl_r": -0.62,
                "pnl_amount": -6.2,
                "post_trade_review_json": '{"issues":["late_entry","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.08,"mae_r":-0.68}',
                "duration_minutes": 10.0,
            },
            {
                "symbol": "USOIL",
                "strategy_key": "USOIL_INVENTORY_MOMENTUM",
                "session_name": "LONDON",
                "regime": "RANGING",
                "closed_at": "2026-03-12T09:00:00+00:00",
                "pnl_r": -0.59,
                "pnl_amount": -5.9,
                "post_trade_review_json": '{"issues":["fast_failure","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.06,"mae_r":-0.64}',
                "duration_minutes": 8.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="USOIL",
            strategy_key="USOIL_INVENTORY_MOMENTUM",
            session_name="LONDON",
            regime_state="RANGING",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")

    def test_pair_strategy_session_performance_state_attacks_usoil_inventory_new_york_trending(self) -> None:
        closed_trades = [
            {
                "symbol": "USOIL",
                "strategy_key": "USOIL_INVENTORY_MOMENTUM",
                "session_name": "NEW_YORK",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T13:00:00+00:00",
                "pnl_r": 0.18,
                "pnl_amount": 1.8,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.66,"mae_r":-0.10}',
                "duration_minutes": 12.0,
            },
            {
                "symbol": "USOIL",
                "strategy_key": "USOIL_INVENTORY_MOMENTUM",
                "session_name": "NEW_YORK",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T14:00:00+00:00",
                "pnl_r": 0.14,
                "pnl_amount": 1.4,
                "post_trade_review_json": '{"issues":[]}',
                "management_effect_json": '{"mfe_r":0.58,"mae_r":-0.09}',
                "duration_minutes": 10.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="USOIL",
            strategy_key="USOIL_INVENTORY_MOMENTUM",
            session_name="NEW_YORK",
            regime_state="TRENDING",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "ATTACK")
        self.assertEqual(state["strategy_bucket_reason"], "usoil_inventory_attack")

    def test_pair_strategy_session_performance_state_demotes_btc_price_action_tokyo_trending(self) -> None:
        closed_trades = [
            {
                "symbol": "BTCUSD",
                "strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -0.92,
                "pnl_amount": -9.2,
                "post_trade_review_json": '{"issues":["late_entry","chased_extension","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.02,"mae_r":-0.94}',
                "duration_minutes": 6.0,
            },
            {
                "symbol": "BTCUSD",
                "strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.88,
                "pnl_amount": -8.8,
                "post_trade_review_json": '{"issues":["late_entry","fast_failure","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.03,"mae_r":-0.90}',
                "duration_minutes": 7.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="BTCUSD",
            strategy_key="BTCUSD_PRICE_ACTION_CONTINUATION",
            session_name="TOKYO",
            regime_state="TRENDING",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")
        self.assertTrue(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_demotes_btc_volatile_retest_sydney_mean_reversion(self) -> None:
        closed_trades = [
            {
                "symbol": "BTCUSD",
                "strategy_key": "BTCUSD_VOLATILE_RETEST",
                "session_name": "SYDNEY",
                "regime": "MEAN_REVERSION",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -0.74,
                "pnl_amount": -7.4,
                "post_trade_review_json": '{"issues":["late_entry","spread_wide_exit","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.03,"mae_r":-0.82}',
                "duration_minutes": 7.0,
            },
            {
                "symbol": "BTCUSD",
                "strategy_key": "BTCUSD_VOLATILE_RETEST",
                "session_name": "SYDNEY",
                "regime": "MEAN_REVERSION",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.68,
                "pnl_amount": -6.8,
                "post_trade_review_json": '{"issues":["fast_failure","late_entry","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.05,"mae_r":-0.76}',
                "duration_minutes": 8.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="BTCUSD",
            strategy_key="BTCUSD_VOLATILE_RETEST",
            session_name="SYDNEY",
            regime_state="MEAN_REVERSION",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")
        self.assertTrue(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_demotes_btc_volatile_retest_sydney_low_liquidity(self) -> None:
        closed_trades = [
            {
                "symbol": "BTCUSD",
                "strategy_key": "BTCUSD_VOLATILE_RETEST",
                "session_name": "SYDNEY",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -0.82,
                "pnl_amount": -8.2,
                "post_trade_review_json": '{"issues":["late_entry","poor_structure","spread_wide_exit"]}',
                "management_effect_json": '{"mfe_r":0.02,"mae_r":-0.90}',
                "duration_minutes": 6.0,
            },
            {
                "symbol": "BTCUSD",
                "strategy_key": "BTCUSD_VOLATILE_RETEST",
                "session_name": "SYDNEY",
                "regime": "LOW_LIQUIDITY_CHOP",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.64,
                "pnl_amount": -6.4,
                "post_trade_review_json": '{"issues":["fast_failure","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.04,"mae_r":-0.70}',
                "duration_minutes": 7.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="BTCUSD",
            strategy_key="BTCUSD_VOLATILE_RETEST",
            session_name="SYDNEY",
            regime_state="LOW_LIQUIDITY_CHOP",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")
        self.assertTrue(bool(state["strategy_bucket_should_block_all_bands"]))

    def test_pair_strategy_session_performance_state_demotes_btc_volatile_retest_tokyo_trending(self) -> None:
        closed_trades = [
            {
                "symbol": "BTCUSD",
                "strategy_key": "BTCUSD_VOLATILE_RETEST",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T00:00:00+00:00",
                "pnl_r": -0.73,
                "pnl_amount": -7.3,
                "post_trade_review_json": '{"issues":["late_entry","poor_structure"]}',
                "management_effect_json": '{"mfe_r":0.05,"mae_r":-0.79}',
                "duration_minutes": 8.0,
            },
            {
                "symbol": "BTCUSD",
                "strategy_key": "BTCUSD_VOLATILE_RETEST",
                "session_name": "TOKYO",
                "regime": "TRENDING",
                "closed_at": "2026-03-12T01:00:00+00:00",
                "pnl_r": -0.69,
                "pnl_amount": -6.9,
                "post_trade_review_json": '{"issues":["fast_failure","spread_wide_exit"]}',
                "management_effect_json": '{"mfe_r":0.04,"mae_r":-0.74}',
                "duration_minutes": 9.0,
            },
        ]

        state = _pair_strategy_session_performance_state(
            symbol="BTCUSD",
            strategy_key="BTCUSD_VOLATILE_RETEST",
            session_name="TOKYO",
            regime_state="TRENDING",
            session_native_pair=False,
            closed_trades=closed_trades,
            current_day_key="2026-03-12",
        )

        self.assertEqual(state["strategy_bucket_state"], "QUARANTINED")
        self.assertEqual(state["strategy_bucket_reason"], "targeted_exact_bucket_demoted")
        self.assertTrue(bool(state["strategy_bucket_should_block_all_bands"]))


if __name__ == "__main__":
    unittest.main()
