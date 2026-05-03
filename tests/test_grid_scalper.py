from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

import pandas as pd

from src.grid_scalper import XAUGridScalper


UTC = timezone.utc


def _features(
    closes: list[float],
    *,
    atr: float = 1.0,
    atr_avg: float = 1.0,
    ema20: float = 100.0,
    ema50: float = 100.0,
    spread: float = 20.0,
    bodies: list[float] | None = None,
    volume_ratio: float = 1.2,
) -> pd.DataFrame:
    if bodies is None:
        bodies = [0.6 for _ in closes]
    start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    rows: list[dict] = []
    for index, close in enumerate(closes):
        rows.append(
            {
                "time": pd.Timestamp(start + timedelta(minutes=5 * index)),
                "m5_close": float(close),
                "m5_body": float(bodies[index]),
                "m5_open": float(close - bodies[index]),
                "m5_high": float(close + max(0.1, abs(bodies[index]) * 0.5)),
                "m5_low": float(close - max(0.1, abs(bodies[index]) * 0.5)),
                "m5_atr_14": float(atr),
                "m5_atr_avg_20": float(atr_avg),
                "m5_ema_20": float(ema20),
                "m5_ema_50": float(ema50),
                "m5_spread": float(spread),
                "m5_volume_ratio_20": float(volume_ratio),
                "m5_pinbar_bull": 0,
                "m5_pinbar_bear": 0,
                "m5_engulf_bull": 0,
                "m5_engulf_bear": 0,
                "m5_trend_bias": 0.0,
            }
        )
    return pd.DataFrame(rows)


def _open_position(
    *,
    symbol: str = "XAUUSD",
    side: str = "BUY",
    entry: float = 100.0,
    volume: float = 0.01,
    opened_at: datetime | None = None,
) -> dict:
    ts = opened_at or datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    return {
        "symbol": symbol,
        "side": side,
        "entry_price": float(entry),
        "volume": float(volume),
        "opened_at": ts.isoformat(),
        "sl": float(entry - 2.0 if side == "BUY" else entry + 2.0),
    }


class GridScalperTests(unittest.TestCase):
    def test_from_config_applies_checkpoint_profile_overrides(self) -> None:
        scalper = XAUGridScalper.from_config(
            {
                "enabled": True,
                "active_profile": "checkpoint",
                "proof_mode": "checkpoint",
                "prime_burst_entries": 9,
                "density_first_mode": True,
                "profiles": {
                    "checkpoint": {
                        "prime_burst_entries": 6,
                        "density_first_mode": False,
                    },
                    "density_branch": {
                        "prime_burst_entries": 8,
                        "density_first_mode": True,
                    },
                },
                "checkpoint_artifact": "/tmp/checkpoint.json",
                "density_branch_artifact": "/tmp/density.json",
            }
        )

        self.assertEqual(scalper.active_profile, "checkpoint")
        self.assertEqual(scalper.proof_mode, "checkpoint")
        self.assertEqual(scalper.prime_burst_entries, 6)
        self.assertFalse(bool(scalper.density_first_mode))
        self.assertEqual(scalper.checkpoint_artifact, "/tmp/checkpoint.json")
        self.assertEqual(scalper.density_branch_artifact, "/tmp/density.json")

    def test_learning_policy_expands_density_branch_quota_targets(self) -> None:
        scalper = XAUGridScalper(
            enabled=True,
            active_profile="density_branch",
            proof_mode="density_branch",
            density_first_mode=True,
            quota_target_actions_per_window=6,
            quota_min_actions_per_window=5,
            quota_catchup_burst_cap=8,
            prime_burst_entries=6,
            aggressive_add_burst_entries=4,
        )

        scalper.apply_learning_policy(
            {
                "bundle": {
                    "promoted_patterns": ["XAUUSD_ADAPTIVE_M5_GRID"],
                    "quota_catchup_pressure": 0.82,
                },
                "reentry_watchlist": ["XAUUSD_ADAPTIVE_M5_GRID"],
            }
        )

        session_config = scalper._session_density_config(session_name="LONDON")

        self.assertEqual(int(session_config["quota_target_actions_per_window"]), 7)
        self.assertEqual(int(session_config["quota_min_actions_per_window"]), 6)
        self.assertEqual(int(session_config["quota_catchup_burst_cap"]), 9)
        self.assertEqual(int(session_config["prime_burst_entries"]), 7)
        self.assertEqual(int(session_config["aggressive_add_burst_entries"]), 5)

    def test_learning_policy_checkpoint_recovery_activates_for_stalled_xau(self) -> None:
        scalper = XAUGridScalper(
            enabled=True,
            active_profile="checkpoint",
            proof_mode="checkpoint",
            density_first_mode=False,
        )

        scalper.apply_learning_policy(
            {
                "bundle": {
                    "quota_catchup_pressure": 0.95,
                },
                "symbol_is_weak_focus": True,
                "reentry_watchlist": ["XAUUSD_ADAPTIVE_M5_GRID"],
            }
        )

        learning_state = scalper._learning_policy_state(session_name="NEW_YORK")

        self.assertFalse(bool(learning_state["density_scaling_active"]))
        self.assertTrue(bool(learning_state["checkpoint_recovery_active"]))
        self.assertGreater(float(learning_state["checkpoint_recovery_relax"]), 0.0)
        self.assertGreater(float(learning_state["mc_floor_relax"]), 0.0)

    def test_learning_pair_directive_increases_xau_quota_bonus_in_prime_session(self) -> None:
        scalper = XAUGridScalper(
            enabled=True,
            active_profile="density_branch",
            proof_mode="density_branch",
            density_first_mode=True,
        )
        scalper.apply_learning_policy(
            {
                "bundle": {"quota_catchup_pressure": 0.70},
                "pair_directive": {
                    "aggression_multiplier": 1.20,
                    "session_focus": ["LONDON", "OVERLAP", "NEW_YORK"],
                },
            }
        )

        learning_state = scalper._learning_policy_state(session_name="LONDON")

        self.assertGreaterEqual(int(learning_state["quota_bonus"]), 1)
        self.assertGreater(float(learning_state["aggression_multiplier"]), 1.0)

    def test_learning_soft_burst_target_pushes_prime_session_density_harder(self) -> None:
        scalper = XAUGridScalper(
            enabled=True,
            active_profile="density_branch",
            proof_mode="density_branch",
            density_first_mode=True,
            prime_burst_entries=6,
            aggressive_add_burst_entries=4,
        )
        scalper.apply_learning_policy(
            {
                "pair_directive": {
                    "hot_hand_active": True,
                    "profit_recycle_active": True,
                    "frequency_directives": {
                        "soft_burst_target_10m": 10,
                        "idle_lane_recovery_active": True,
                    },
                },
            }
        )

        session_config = scalper._session_density_config(session_name="LONDON")

        self.assertGreaterEqual(int(session_config["prime_burst_entries"]), 9)
        self.assertGreaterEqual(int(session_config["aggressive_add_burst_entries"]), 6)

    def test_grid_loss_cooldown_uses_configured_threshold(self) -> None:
        scalper = XAUGridScalper(enabled=True, loss_streak_threshold=4, cooldown_after_stop_minutes=30)
        now = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)

        for _ in range(3):
            scalper._record_cycle_result(now, -1.0, hard_loss=True)

        self.assertIsNone(scalper._cooldown_until)
        scalper._record_cycle_result(now, -1.0, hard_loss=True)
        self.assertIsNotNone(scalper._cooldown_until)

    def test_checkpoint_recovery_scaler_allows_live_like_prime_pullback_bar(self) -> None:
        scalper = XAUGridScalper(
            enabled=True,
            active_profile="checkpoint",
            proof_mode="checkpoint",
            density_first_mode=False,
        )
        scalper.apply_learning_policy(
            {
                "bundle": {
                    "quota_catchup_pressure": 1.0,
                },
                "symbol_is_weak_focus": True,
                "reentry_watchlist": ["XAUUSD_ADAPTIVE_M5_GRID"],
            }
        )
        frame = _features(
            [100.8, 101.4, 102.1, 102.9, 103.6],
            atr=9.6,
            atr_avg=9.2,
            ema20=101.9,
            ema50=100.4,
            bodies=[0.6, 0.8, 0.9, 0.7, -0.4],
            volume_ratio=0.14,
        )
        frame.loc[:, "m5_open"] = [100.1, 100.8, 101.6, 102.4, 104.0]
        frame.loc[:, "m5_high"] = [101.1, 101.7, 102.5, 103.3, 105.4]
        frame.loc[:, "m5_low"] = [99.8, 100.5, 101.2, 101.9, 100.9]
        frame.loc[:, "m5_body_efficiency"] = 0.36
        frame.loc[:, "compression_state"] = "COMPRESSION"
        frame.loc[:, "multi_tf_alignment_score"] = 0.25
        frame.loc[:, "fractal_persistence_score"] = 0.44
        frame.loc[:, "seasonality_edge_score"] = 0.77
        frame.loc[:, "market_instability_score"] = 0.36
        frame.loc[:, "feature_drift_score"] = 0.26
        frame.loc[:, "m5_trend_efficiency_16"] = 0.24
        frame.loc[:, "m5_range_position_20"] = 0.73
        frame.loc[:, "m15_range_position_20"] = 0.93
        frame.loc[:, "m15_ema_20"] = 101.4
        frame.loc[:, "m15_ema_50"] = 102.0
        frame.loc[:, "m1_momentum_3"] = -1.4
        frame.loc[:, "m15_ret_1"] = 0.0002
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 9, 10, tzinfo=UTC),
            spread_points=25.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(decision.deny_reason, "")
        self.assertEqual(
            str(decision.candidates[0].meta.get("grid_entry_profile") or ""),
            "grid_prime_session_momentum_long",
        )

    def test_new_cycle_candidate_created_for_stretch_and_deceleration(self) -> None:
        scalper = XAUGridScalper(enabled=True, ema_stretch_k=0.7, density_first_mode=False)
        frame = _features(
            [100.4, 100.0, 99.4, 99.1, 98.9],
            ema20=99.7,
            ema50=100.0,
            bodies=[0.9, 0.8, 0.6, 0.4, 0.2],
        )
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(decision.candidates[0].setup, "XAUUSD_M5_GRID_SCALPER_START")
        self.assertEqual(decision.candidates[0].side, "SELL")
        self.assertEqual(str(decision.candidates[0].meta.get("grid_mode") or ""), "ATTACK_GRID")

    def test_ai_deny_blocks_cycle_start(self) -> None:
        scalper = XAUGridScalper(enabled=True)
        frame = _features(
            [100.2, 99.8, 99.4, 99.0],
            ema20=99.6,
            ema50=100.0,
            bodies=[0.8, 0.6, 0.4, 0.2],
        )
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 8, 5, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
            approver=lambda _: {"approve": False, "reason": "committee_deny", "ai_mode": "mock"},
        )

        self.assertEqual(decision.candidates, [])
        self.assertTrue(decision.deny_reason.startswith("grid_ai_deny:committee_deny"))

    def test_new_cycle_uses_configured_entry_stop_profile(self) -> None:
        scalper = XAUGridScalper(enabled=True, stop_atr_k=2.4, entry_stop_atr_k=0.72, density_first_mode=False)
        frame = _features(
            [100.4, 100.0, 99.4, 99.1, 98.9],
            ema20=99.7,
            ema50=100.0,
            bodies=[0.9, 0.8, 0.6, 0.4, 0.2],
        )
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
            spread_points=12.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertAlmostEqual(float(decision.candidates[0].stop_atr), 0.792, places=6)

    def test_new_cycle_emits_broker_safe_stop_points(self) -> None:
        scalper = XAUGridScalper(
            enabled=True,
            entry_stop_step_multiplier=6.0,
            entry_stop_points_min=170.0,
            entry_stop_points_max=220.0,
            density_first_mode=False,
        )
        frame = _features(
            [100.4, 100.0, 99.4, 99.1, 98.9],
            ema20=99.7,
            ema50=100.0,
            bodies=[0.9, 0.8, 0.6, 0.4, 0.2],
        )
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
            spread_points=12.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertGreaterEqual(float(decision.candidates[0].meta.get("stop_points", 0.0)), 170.0)

    def test_new_cycle_spacing_respects_live_spread_floor(self) -> None:
        scalper = XAUGridScalper(
            enabled=True,
            step_atr_k=0.20,
            step_points_min=20.0,
            step_points_max=25.0,
            density_first_mode=False,
        )
        frame = _features(
            [100.4, 100.0, 99.4, 99.1, 98.9],
            ema20=99.7,
            ema50=100.0,
            bodies=[0.9, 0.8, 0.6, 0.4, 0.2],
            spread=25.0,
        )
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 8, 0, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertGreaterEqual(float(decision.candidates[0].meta.get("chosen_spacing_points", 0.0)), 80.0)

    def test_news_block_denies_cycle_without_override(self) -> None:
        scalper = XAUGridScalper(enabled=True, news_override_min_probability=0.8)
        frame = _features(
            [100.0, 99.8, 99.5, 99.1],
            ema20=99.7,
            ema50=100.0,
            bodies=[0.7, 0.6, 0.5, 0.3],
        )
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=False,
            now_utc=datetime(2026, 1, 1, 8, 10, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
            approver=lambda _: {"approve": True, "confidence": 0.65, "ai_mode": "mock"},
        )

        self.assertEqual(decision.candidates, [])
        self.assertEqual(decision.deny_reason, "grid_news_block_new_cycle")

    def test_news_high_confluence_override_allows_cycle(self) -> None:
        scalper = XAUGridScalper(enabled=True, news_override_min_probability=0.7, density_first_mode=False)
        frame = _features(
            [101.0, 100.2, 99.6, 99.0],
            ema20=99.8,
            ema50=100.2,
            bodies=[1.0, 0.8, 0.4, 0.2],
        )
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=False,
            now_utc=datetime(2026, 1, 1, 8, 15, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
            approver=lambda _: {"approve": True, "confidence": 0.9, "lot_multiplier": 1.0, "ai_mode": "mock"},
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertTrue(bool(decision.candidates[0].meta.get("news_override", False)))

    def test_prime_session_momentum_fallback_allows_cycle_when_reclaim_stack_is_borderline(self) -> None:
        scalper = XAUGridScalper(enabled=True)
        frame = _features(
            [100.10, 100.18, 100.34, 100.52, 100.74],
            ema20=100.48,
            ema50=100.32,
            bodies=[0.12, 0.14, 0.18, 0.22, 0.28],
            volume_ratio=0.78,
        )
        frame.loc[:, "compression_state"] = "COMPRESSION"
        frame.loc[:, "compression_expansion_score"] = 0.34
        frame.loc[:, "multi_tf_alignment_score"] = 0.38
        frame.loc[:, "fractal_persistence_score"] = 0.36
        frame.loc[:, "seasonality_edge_score"] = 0.27
        frame.loc[:, "market_instability_score"] = 0.24
        frame.loc[:, "feature_drift_score"] = 0.18
        frame.loc[:, "m5_body_efficiency"] = 0.30
        frame.loc[:, "m5_range_position_20"] = 0.62
        frame.loc[:, "m15_range_position_20"] = 0.58
        frame.loc[:, "m1_momentum_3"] = 0.004
        frame.loc[:, "m15_ret_1"] = 0.006
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 10, 15, tzinfo=UTC),
            spread_points=14.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertIn(
            str(decision.candidates[0].meta.get("grid_entry_profile") or ""),
            {"grid_prime_session_momentum_long", "grid_directional_flow_long"},
        )
        self.assertEqual(
            str(decision.candidates[0].meta.get("grid_cycle_id") or ""),
            str(decision.candidates[-1].meta.get("grid_cycle_id") or ""),
        )

    def test_prime_session_directional_profiles_use_wider_stop_and_higher_tp(self) -> None:
        scalper = XAUGridScalper(enabled=True)
        frame = _features(
            [100.10, 100.18, 100.34, 100.52, 100.74],
            ema20=100.48,
            ema50=100.32,
            bodies=[0.12, 0.14, 0.18, 0.22, 0.28],
            volume_ratio=0.78,
        )
        frame.loc[:, "compression_state"] = "COMPRESSION"
        frame.loc[:, "compression_expansion_score"] = 0.34
        frame.loc[:, "multi_tf_alignment_score"] = 0.38
        frame.loc[:, "fractal_persistence_score"] = 0.36
        frame.loc[:, "seasonality_edge_score"] = 0.27
        frame.loc[:, "market_instability_score"] = 0.24
        frame.loc[:, "feature_drift_score"] = 0.18
        frame.loc[:, "m5_body_efficiency"] = 0.30
        frame.loc[:, "m5_range_position_20"] = 0.62
        frame.loc[:, "m15_range_position_20"] = 0.58
        frame.loc[:, "m1_momentum_3"] = 0.004
        frame.loc[:, "m15_ret_1"] = 0.006
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 10, 15, tzinfo=UTC),
            spread_points=14.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        candidate = decision.candidates[0]
        profile = str(candidate.meta.get("grid_entry_profile") or "")
        self.assertIn(profile, {"grid_prime_session_momentum_long", "grid_directional_flow_long"})
        self.assertGreaterEqual(float(candidate.tp_r), 2.2)
        base_stop_points = scalper._entry_stop_points(
            step_points=float(candidate.meta.get("chosen_spacing_points", 0.0)),
            probe_candidate=bool(candidate.meta.get("grid_probe", False)),
        )
        self.assertGreater(float(candidate.meta.get("stop_points", 0.0)), float(base_stop_points))

    def test_density_micro_profile_helpers_widen_stop_and_keep_rr_above_scalp_floor(self) -> None:
        scalper = XAUGridScalper(enabled=True)

        base_stop_points = scalper._entry_stop_points(step_points=80.0, probe_candidate=False)
        density_stop_points = scalper._entry_stop_points_for_profile(
            step_points=80.0,
            probe_candidate=False,
            entry_profile="grid_density_micro_scaler_long",
            session_profile="AGGRESSIVE",
        )
        density_tp_r = scalper._entry_tp_r(
            entry_profile="grid_density_micro_scaler_long",
            grid_mode="ATTACK_GRID",
            session_profile="AGGRESSIVE",
        )

        self.assertGreater(float(density_stop_points), float(base_stop_points))
        self.assertGreaterEqual(float(density_tp_r), 1.9)

    def test_asia_density_relief_can_emit_directional_flow_candidate_in_sydney(self) -> None:
        scalper = XAUGridScalper(enabled=True)
        scalper.active_profile = "density_branch"
        scalper.density_first_mode = True
        scalper.apply_learning_policy(
            {
                "bundle": {"quota_catchup_pressure": 0.44},
                "pair_directive": {"frequency_directives": {"idle_lane_recovery_active": True}},
            }
        )
        frame = _features(
            [100.02, 100.08, 100.16, 100.20, 100.24],
            ema20=100.18,
            ema50=100.10,
            bodies=[0.18, 0.18, 0.18, 0.18, 0.18],
            volume_ratio=0.55,
        )
        frame.loc[:, "compression_state"] = "NEUTRAL"
        frame.loc[:, "compression_expansion_score"] = 0.14
        frame.loc[:, "m5_atr_avg_20"] = float(frame["m5_atr_14"].iloc[-1]) / 0.80
        frame.loc[:, "multi_tf_alignment_score"] = 0.30
        frame.loc[:, "fractal_persistence_score"] = 0.28
        frame.loc[:, "seasonality_edge_score"] = 0.42
        frame.loc[:, "market_instability_score"] = 0.32
        frame.loc[:, "feature_drift_score"] = 0.18
        frame.loc[:, "m5_body_efficiency"] = 0.18
        frame.loc[:, "m5_range_position_20"] = 0.48
        frame.loc[:, "m15_range_position_20"] = 0.46
        frame.loc[:, "m1_momentum_3"] = 0.005
        frame.loc[:, "m15_ret_1"] = 0.005
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="SYDNEY",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 22, 10, tzinfo=UTC),
            spread_points=15.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(str(decision.candidates[0].meta.get("grid_entry_profile") or ""), "grid_asia_probe_recovery_long")

    def test_prime_session_reclaim_softens_dxy_conflict_to_penalty(self) -> None:
        scalper = XAUGridScalper(enabled=True)
        frame = _features(
            [100.4, 100.1, 99.8, 99.55, 99.72],
            ema20=99.68,
            ema50=99.54,
            bodies=[0.32, 0.30, 0.28, 0.22, 0.26],
            volume_ratio=1.05,
        )
        frame.loc[:, "compression_state"] = "EXPANSION_READY"
        frame.loc[:, "compression_expansion_score"] = 0.52
        frame.loc[:, "multi_tf_alignment_score"] = 0.56
        frame.loc[:, "fractal_persistence_score"] = 0.50
        frame.loc[:, "seasonality_edge_score"] = 0.34
        frame.loc[:, "market_instability_score"] = 0.22
        frame.loc[:, "feature_drift_score"] = 0.16
        frame.loc[:, "m5_body_efficiency"] = 0.36
        frame.loc[:, "m5_range_position_20"] = 0.70
        frame.loc[:, "m15_range_position_20"] = 0.58
        frame.loc[:, "m1_momentum_3"] = 0.003
        frame.loc[:, "m15_ret_1"] = 0.004
        frame.loc[:, "dxy_ret_60"] = 0.0031
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="OVERLAP",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 14, 10, tzinfo=UTC),
            spread_points=16.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertIn(
            str(decision.candidates[0].meta.get("grid_entry_profile") or ""),
            {
                "grid_directional_flow_long",
                "grid_expansion_ready_scaler_long",
                "grid_prime_session_momentum_long",
                "grid_trend_reclaim_long",
            },
        )

    def test_prime_extreme_continuation_sell_allows_fresh_low_breakdown_cycle(self) -> None:
        scalper = XAUGridScalper(enabled=True)
        frame = _features(
            [100.8, 100.1, 99.4, 98.8, 98.2],
            ema20=99.4,
            ema50=100.0,
            bodies=[-0.18, -0.32, -0.42, 0.12, 0.18],
            volume_ratio=0.34,
        )
        frame.loc[:, "m5_open"] = [100.9, 100.3, 99.8, 98.6, 98.0]
        frame.loc[:, "m5_high"] = [101.0, 100.4, 99.9, 99.0, 98.5]
        frame.loc[:, "m5_low"] = [100.5, 99.8, 99.0, 98.1, 97.7]
        frame.loc[:, "compression_state"] = "NEUTRAL"
        frame.loc[:, "compression_expansion_score"] = 0.18
        frame.loc[:, "multi_tf_alignment_score"] = 0.50
        frame.loc[:, "fractal_persistence_score"] = 0.40
        frame.loc[:, "seasonality_edge_score"] = 0.80
        frame.loc[:, "market_instability_score"] = 0.30
        frame.loc[:, "feature_drift_score"] = 0.20
        frame.loc[:, "m5_body_efficiency"] = 0.35
        frame.loc[:, "m5_range_position_20"] = 0.12
        frame.loc[:, "m15_range_position_20"] = 0.05
        frame.loc[:, "m15_ema_20"] = 99.7
        frame.loc[:, "m15_ema_50"] = 100.4
        frame.loc[:, "m15_atr_14"] = 1.6
        frame.loc[:, "m15_volume_ratio_20"] = 1.18
        frame.loc[:, "m1_momentum_3"] = -4.0
        frame.loc[:, "m15_ret_1"] = -0.004
        frame.loc[:, "m15_rolling_low_prev_20"] = 98.55
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 10, 10, tzinfo=UTC),
            spread_points=14.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(str(decision.candidates[0].meta.get("grid_entry_profile") or ""), "grid_prime_session_momentum_short")

    def test_prime_extreme_continuation_sell_allows_low_alignment_with_clean_htf_stack(self) -> None:
        scalper = XAUGridScalper(enabled=True)
        frame = _features(
            [100.4, 99.8, 99.1, 98.5, 97.9],
            ema20=99.0,
            ema50=99.7,
            bodies=[-0.24, -0.30, -0.38, -0.42, -0.46],
            volume_ratio=1.02,
        )
        frame.loc[:, "m5_open"] = [100.5, 99.95, 99.35, 98.75, 98.25]
        frame.loc[:, "m5_high"] = [100.62, 100.05, 99.42, 98.88, 98.36]
        frame.loc[:, "m5_low"] = [100.16, 99.54, 98.88, 98.24, 97.62]
        frame.loc[:, "compression_state"] = "NEUTRAL"
        frame.loc[:, "compression_expansion_score"] = 0.18
        frame.loc[:, "multi_tf_alignment_score"] = 0.25
        frame.loc[:, "fractal_persistence_score"] = 0.46
        frame.loc[:, "seasonality_edge_score"] = 0.80
        frame.loc[:, "market_instability_score"] = 0.31
        frame.loc[:, "feature_drift_score"] = 0.16
        frame.loc[:, "m5_body_efficiency"] = 0.66
        frame.loc[:, "m5_range_position_20"] = 0.02
        frame.loc[:, "m15_range_position_20"] = 0.02
        frame.loc[:, "m15_ema_20"] = 99.5
        frame.loc[:, "m15_ema_50"] = 100.1
        frame.loc[:, "m15_atr_14"] = 1.7
        frame.loc[:, "m15_volume_ratio_20"] = 0.86
        frame.loc[:, "m1_momentum_3"] = -4.0
        frame.loc[:, "m15_ret_1"] = -0.004
        frame.loc[:, "m15_rolling_low_prev_20"] = 98.05
        frame.loc[:, "h1_ema_50"] = 101.8
        frame.loc[:, "h1_ema_200"] = 103.2
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="OVERLAP",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 14, 10, tzinfo=UTC),
            spread_points=14.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(str(decision.candidates[0].meta.get("grid_entry_profile") or ""), "grid_prime_session_momentum_short")

    def test_prime_stretch_reversal_buy_allows_oversold_bounce_below_emas(self) -> None:
        scalper = XAUGridScalper(enabled=True)
        frame = _features(
            [4682.0, 4660.0, 4640.0, 4626.0, 4618.2],
            ema20=4650.0,
            ema50=4686.0,
            bodies=[-8.0, -12.0, -16.0, -18.0, 9.7],
            volume_ratio=1.04,
        )
        frame.loc[:, "m5_open"] = [4688.0, 4672.0, 4654.0, 4636.0, 4608.5]
        frame.loc[:, "m5_high"] = [4689.0, 4675.0, 4658.0, 4642.0, 4620.8]
        frame.loc[:, "m5_low"] = [4675.0, 4650.0, 4628.0, 4610.0, 4602.2]
        frame.loc[:, "m15_ema_20"] = 4686.6
        frame.loc[:, "m15_ema_50"] = 4751.4
        frame.loc[:, "h1_ema_50"] = 4875.4
        frame.loc[:, "h1_ema_200"] = 5026.0
        frame.loc[:, "m15_rolling_low_prev_20"] = 4605.0
        frame.loc[:, "m15_rolling_high_prev_20"] = 4773.4
        frame.loc[:, "m15_range_position_20"] = 0.079
        frame.loc[:, "m5_range_position_20"] = 0.159
        frame.loc[:, "m5_body_efficiency"] = 0.52
        frame.loc[:, "m15_body_efficiency"] = 0.42
        frame.loc[:, "m5_volume_ratio_20"] = 1.06
        frame.loc[:, "m15_volume_ratio_20"] = 0.31
        frame.loc[:, "multi_tf_alignment_score"] = 0.75
        frame.loc[:, "fractal_persistence_score"] = 0.42
        frame.loc[:, "seasonality_edge_score"] = 0.80
        frame.loc[:, "market_instability_score"] = 0.24
        frame.loc[:, "feature_drift_score"] = 0.06
        frame.loc[:, "m1_momentum_3"] = 2.4
        frame.loc[:, "m15_ret_1"] = 0.0013
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="OVERLAP",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 14, 10, tzinfo=UTC),
            spread_points=16.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(str(decision.candidates[0].meta.get("grid_entry_profile") or ""), "grid_prime_stretch_reversion_long")

    def test_prime_exhaustion_probe_buy_recovers_extreme_overlap_flush(self) -> None:
        scalper = XAUGridScalper(enabled=True)
        scalper.rsi_period = 3
        frame = _features(
            [4688.0, 4668.0, 4638.0, 4588.0, 4550.4],
            ema20=4607.8,
            ema50=4657.2,
            bodies=[-8.0, -18.0, -30.0, -50.0, -2.4],
            volume_ratio=0.34,
        )
        frame.loc[:, "m5_open"] = [4696.0, 4686.0, 4668.0, 4638.0, 4552.8]
        frame.loc[:, "m5_high"] = [4698.0, 4688.0, 4670.0, 4602.0, 4567.2]
        frame.loc[:, "m5_low"] = [4680.0, 4656.0, 4620.0, 4560.0, 4545.1]
        frame.loc[:, "m15_ema_20"] = 4686.6
        frame.loc[:, "m15_ema_50"] = 4751.4
        frame.loc[:, "h1_ema_50"] = 4861.1
        frame.loc[:, "h1_ema_200"] = 5020.9
        frame.loc[:, "m15_rolling_low_prev_20"] = 4544.8
        frame.loc[:, "m15_rolling_high_prev_20"] = 4773.4
        frame.loc[:, "m15_range_position_20"] = 0.080
        frame.loc[:, "m5_range_position_20"] = 0.233
        frame.loc[:, "m5_body_efficiency"] = 0.109
        frame.loc[:, "m15_body_efficiency"] = 0.328
        frame.loc[:, "m5_volume_ratio_20"] = 0.334
        frame.loc[:, "m15_volume_ratio_20"] = 1.437
        frame.loc[:, "multi_tf_alignment_score"] = 0.25
        frame.loc[:, "fractal_persistence_score"] = 0.528
        frame.loc[:, "seasonality_edge_score"] = 1.00
        frame.loc[:, "market_instability_score"] = 0.400
        frame.loc[:, "feature_drift_score"] = 0.470
        frame.loc[:, "predicted_liquidity_hunt_score"] = 0.95
        frame.loc[:, "m5_rsi_14"] = 19.28
        frame.loc[:, "m1_momentum_3"] = -0.002
        frame.loc[:, "m15_ret_1"] = -0.0054
        frame.loc[:, "m15_ret_3"] = -0.0121
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="OVERLAP",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 14, 10, tzinfo=UTC),
            spread_points=24.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(str(decision.candidates[0].meta.get("grid_entry_profile") or ""), "grid_prime_stretch_reversion_long")

    def test_green_prime_extension_does_not_trip_short_reversion_block(self) -> None:
        scalper = XAUGridScalper(enabled=True)
        frame = _features(
            [100.18, 100.42, 100.70, 101.04, 101.46],
            ema20=100.92,
            ema50=100.58,
            bodies=[0.18, 0.24, 0.28, 0.34, 0.40],
            volume_ratio=1.02,
        )
        frame.loc[:, "multi_tf_alignment_score"] = 1.00
        frame.loc[:, "fractal_persistence_score"] = 1.00
        frame.loc[:, "seasonality_edge_score"] = 0.82
        frame.loc[:, "market_instability_score"] = 0.04
        frame.loc[:, "feature_drift_score"] = 0.10
        frame.loc[:, "m5_body_efficiency"] = 0.40
        frame.loc[:, "m5_range_position_20"] = 0.965
        frame.loc[:, "m15_range_position_20"] = 0.965
        frame.loc[:, "m1_momentum_3"] = 0.006
        frame.loc[:, "m15_ret_1"] = 0.008
        frame.loc[:, "m5_high"] = frame["m5_close"] + 0.12
        frame.loc[:, "m5_low"] = frame["m5_close"] - 0.28
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 9, 25, tzinfo=UTC),
            spread_points=12.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertTrue(
            all(
                str(candidate.meta.get("grid_entry_profile") or "").startswith("grid_prime_session_momentum")
                for candidate in decision.candidates
            )
        )
        self.assertNotEqual(decision.deny_reason, "grid_stretch_reversion_short_blocked")

    def test_attack_grid_prime_session_can_emit_burst_start_candidates(self) -> None:
        scalper = XAUGridScalper(enabled=True, prime_burst_entries=4, max_open_cycles=2)
        frame = _features(
            [100.10, 100.32, 100.54, 100.86, 101.24],
            ema20=100.72,
            ema50=100.38,
            bodies=[0.22, 0.24, 0.28, 0.34, 0.40],
            volume_ratio=1.28,
        )
        frame.loc[:, "compression_state"] = "EXPANSION_READY"
        frame.loc[:, "compression_expansion_score"] = 0.56
        frame.loc[:, "multi_tf_alignment_score"] = 0.68
        frame.loc[:, "fractal_persistence_score"] = 0.62
        frame.loc[:, "seasonality_edge_score"] = 0.44
        frame.loc[:, "market_instability_score"] = 0.20
        frame.loc[:, "feature_drift_score"] = 0.14
        frame.loc[:, "m5_body_efficiency"] = 0.44
        frame.loc[:, "m5_range_position_20"] = 0.74
        frame.loc[:, "m15_range_position_20"] = 0.62
        frame.loc[:, "m1_momentum_3"] = 0.005
        frame.loc[:, "m15_ret_1"] = 0.008
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 9, 10, tzinfo=UTC),
            spread_points=14.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 2)
        self.assertTrue(all(candidate.setup == "XAUUSD_M5_GRID_SCALPER_START" for candidate in decision.candidates))
        self.assertEqual(
            len({str(candidate.meta.get("grid_cycle_id") or "") for candidate in decision.candidates}),
            1,
        )
        self.assertEqual(
            [int(candidate.meta.get("grid_level", 0) or 0) for candidate in decision.candidates],
            list(range(1, len(decision.candidates) + 1)),
        )
        self.assertGreaterEqual(float(decision.candidates[0].meta.get("mc_win_rate", 0.0) or 0.0), 0.80)
        self.assertTrue(str(decision.candidates[0].meta.get("native_burst_window_id") or ""))
        self.assertIn("NATIVE", str(decision.candidates[0].meta.get("grid_source_role") or ""))

    def test_attack_grid_prime_session_can_emit_five_plus_native_burst_candidates(self) -> None:
        scalper = XAUGridScalper(enabled=True, prime_burst_entries=8, aggressive_add_burst_entries=5, max_levels=8, max_open_cycles=3)
        frame = _features(
            [100.18, 100.46, 100.78, 101.12, 101.58],
            ema20=100.96,
            ema50=100.54,
            bodies=[0.24, 0.28, 0.34, 0.40, 0.46],
            volume_ratio=1.42,
        )
        frame.loc[:, "compression_state"] = "EXPANSION_READY"
        frame.loc[:, "compression_expansion_score"] = 0.72
        frame.loc[:, "multi_tf_alignment_score"] = 0.84
        frame.loc[:, "fractal_persistence_score"] = 0.78
        frame.loc[:, "seasonality_edge_score"] = 0.60
        frame.loc[:, "market_instability_score"] = 0.08
        frame.loc[:, "feature_drift_score"] = 0.05
        frame.loc[:, "m5_body_efficiency"] = 0.56
        frame.loc[:, "m5_range_position_20"] = 0.82
        frame.loc[:, "m15_range_position_20"] = 0.74
        frame.loc[:, "m1_momentum_3"] = 0.007
        frame.loc[:, "m15_ret_1"] = 0.010
        frame.loc[:, "h1_close"] = frame["m5_close"] + 0.14
        frame.loc[:, "h1_ema_20"] = 101.08
        frame.loc[:, "h1_ema_50"] = 100.72
        frame.loc[:, "h1_range_position_20"] = 0.78
        frame.loc[:, "h4_close"] = frame["m5_close"] + 0.28
        frame.loc[:, "h4_ema_20"] = 101.02
        frame.loc[:, "h4_ema_50"] = 100.60
        frame.loc[:, "h4_range_position_20"] = 0.80
        frame.loc[:, "d1_close"] = frame["m5_close"] + 0.42
        frame.loc[:, "d1_ema_20"] = 100.94
        frame.loc[:, "d1_ema_50"] = 100.48
        frame.loc[:, "d1_range_position_20"] = 0.84
        frame.loc[:, "dxy_ret_15"] = -0.0030
        frame.loc[:, "dxy_ret_60"] = -0.0042
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 9, 20, tzinfo=UTC),
            spread_points=12.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 5)
        self.assertTrue(all(candidate.setup == "XAUUSD_M5_GRID_SCALPER_START" for candidate in decision.candidates))
        self.assertGreaterEqual(int(decision.candidates[0].meta.get("grid_burst_size", 0) or 0), 5)
        self.assertEqual(
            len({str(candidate.meta.get("grid_cycle_id") or "") for candidate in decision.candidates}),
            1,
        )

    def test_density_first_quota_state_promotes_prime_session_burst_and_tracks_quota(self) -> None:
        scalper = XAUGridScalper(
            enabled=True,
            prime_burst_entries=8,
            aggressive_add_burst_entries=5,
            max_levels=8,
            max_open_cycles=3,
            quota_target_actions_per_window=6,
            quota_min_actions_per_window=5,
        )
        frame = _features(
            [100.18, 100.46, 100.78, 101.12, 101.58],
            ema20=100.96,
            ema50=100.54,
            bodies=[0.24, 0.28, 0.34, 0.40, 0.46],
            volume_ratio=1.42,
        )
        frame.loc[:, "compression_state"] = "EXPANSION_READY"
        frame.loc[:, "compression_expansion_score"] = 0.72
        frame.loc[:, "multi_tf_alignment_score"] = 0.84
        frame.loc[:, "fractal_persistence_score"] = 0.78
        frame.loc[:, "seasonality_edge_score"] = 0.60
        frame.loc[:, "market_instability_score"] = 0.08
        frame.loc[:, "feature_drift_score"] = 0.05
        frame.loc[:, "m5_body_efficiency"] = 0.56
        frame.loc[:, "m5_range_position_20"] = 0.82
        frame.loc[:, "m15_range_position_20"] = 0.74
        frame.loc[:, "m1_momentum_3"] = 0.007
        frame.loc[:, "m15_ret_1"] = 0.010
        frame.loc[:, "h1_close"] = frame["m5_close"] + 0.14
        frame.loc[:, "h1_ema_20"] = 101.08
        frame.loc[:, "h1_ema_50"] = 100.72
        frame.loc[:, "h1_range_position_20"] = 0.78
        frame.loc[:, "h4_close"] = frame["m5_close"] + 0.28
        frame.loc[:, "h4_ema_20"] = 101.02
        frame.loc[:, "h4_ema_50"] = 100.60
        frame.loc[:, "h4_range_position_20"] = 0.80
        frame.loc[:, "d1_close"] = frame["m5_close"] + 0.42
        frame.loc[:, "d1_ema_20"] = 100.94
        frame.loc[:, "d1_ema_50"] = 100.48
        frame.loc[:, "d1_range_position_20"] = 0.84
        frame.loc[:, "dxy_ret_15"] = -0.0030
        frame.loc[:, "dxy_ret_60"] = -0.0042
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 9, 20, tzinfo=UTC),
            spread_points=12.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 5)
        self.assertTrue(bool(decision.quota_density_first_active))
        self.assertEqual(int(decision.quota_target_10m), 6)
        self.assertEqual(int(decision.quota_approved_last_10m), 0)
        self.assertEqual(str(decision.quota_state), "CATCHUP")
        follow_up = scalper._quota_state(
            now_utc=datetime(2026, 1, 1, 9, 21, tzinfo=UTC),
            session_name="LONDON",
            session_profile="AGGRESSIVE",
        )
        self.assertGreaterEqual(int(follow_up["approved"]), len(decision.candidates))
        self.assertEqual(int(follow_up["quota_debt"]), 0)

    def test_density_first_softens_prime_quality_gate_until_quota_is_met(self) -> None:
        scalper = XAUGridScalper(enabled=True, quota_target_actions_per_window=6, quota_min_actions_per_window=5)
        frame = _features(
            [100.10, 100.18, 100.34, 100.52, 100.74],
            ema20=100.48,
            ema50=100.32,
            bodies=[0.12, 0.14, 0.18, 0.22, 0.28],
            volume_ratio=0.78,
        )
        frame.loc[:, "compression_state"] = "COMPRESSION"
        frame.loc[:, "compression_expansion_score"] = 0.34
        frame.loc[:, "multi_tf_alignment_score"] = 0.38
        frame.loc[:, "fractal_persistence_score"] = 0.36
        frame.loc[:, "seasonality_edge_score"] = 0.27
        frame.loc[:, "market_instability_score"] = 0.24
        frame.loc[:, "feature_drift_score"] = 0.18
        frame.loc[:, "m5_body_efficiency"] = 0.30
        frame.loc[:, "m5_range_position_20"] = 0.62
        frame.loc[:, "m15_range_position_20"] = 0.58
        frame.loc[:, "m1_momentum_3"] = 0.004
        frame.loc[:, "m15_ret_1"] = 0.006
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="NEW_YORK",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 14, 15, tzinfo=UTC),
            spread_points=14.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(decision.deny_reason, "")
        self.assertEqual(str(decision.soft_penalty_reason), "grid_prime_session_quality_gate")
        self.assertGreater(float(decision.soft_penalty_score), 0.0)

    def test_density_first_quota_reclaim_rescue_can_open_prime_session_cycle(self) -> None:
        scalper = XAUGridScalper(enabled=True, quota_target_actions_per_window=6, quota_min_actions_per_window=5)
        frame = _features(
            [100.08, 100.18, 100.34, 100.52, 100.72],
            ema20=100.48,
            ema50=100.40,
            bodies=[0.10, 0.12, 0.14, 0.16, 0.18],
            volume_ratio=0.74,
        )
        frame.loc[:, "compression_state"] = "COMPRESSION"
        frame.loc[:, "compression_expansion_score"] = 0.20
        frame.loc[:, "multi_tf_alignment_score"] = 0.16
        frame.loc[:, "fractal_persistence_score"] = 0.14
        frame.loc[:, "seasonality_edge_score"] = 0.18
        frame.loc[:, "market_instability_score"] = 0.30
        frame.loc[:, "feature_drift_score"] = 0.28
        frame.loc[:, "m5_body_efficiency"] = 0.10
        frame.loc[:, "m5_range_position_20"] = 0.42
        frame.loc[:, "m15_range_position_20"] = 0.54
        frame.loc[:, "m1_momentum_3"] = 0.003
        frame.loc[:, "m15_ret_1"] = 0.004
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 10, 25, tzinfo=UTC),
            spread_points=14.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(decision.deny_reason, "")
        self.assertTrue(bool(decision.candidates[0].meta.get("quota_reclaim_rescue_selected")))
        self.assertEqual(
            str(decision.candidates[0].meta.get("grid_entry_profile") or ""),
            "grid_prime_session_momentum_long",
        )

    def test_quota_reclaim_rescue_relaxes_prime_scaler_for_live_like_bootstrap_bar(self) -> None:
        scalper = XAUGridScalper(enabled=True, quota_target_actions_per_window=6, quota_min_actions_per_window=5)
        frame = _features(
            [100.02, 100.10, 100.18, 100.30, 100.42],
            ema20=100.28,
            ema50=100.20,
            bodies=[0.08, 0.10, 0.10, 0.12, 0.12],
            volume_ratio=0.52,
        )
        frame.loc[:, "compression_state"] = "NEUTRAL"
        frame.loc[:, "compression_expansion_score"] = 0.12
        frame.loc[:, "multi_tf_alignment_score"] = 0.13
        frame.loc[:, "fractal_persistence_score"] = 0.11
        frame.loc[:, "seasonality_edge_score"] = 0.18
        frame.loc[:, "market_instability_score"] = 0.42
        frame.loc[:, "feature_drift_score"] = 0.32
        frame.loc[:, "m5_body_efficiency"] = 0.09
        frame.loc[:, "m5_range_position_20"] = 0.18
        frame.loc[:, "m15_range_position_20"] = 0.90
        frame.loc[:, "m1_momentum_3"] = 0.002
        frame.loc[:, "m15_ret_1"] = -0.014
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 10, 35, tzinfo=UTC),
            spread_points=14.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(decision.deny_reason, "")
        self.assertTrue(bool(decision.candidates[0].meta.get("quota_reclaim_rescue_active")))
        self.assertEqual(
            str(decision.candidates[0].meta.get("grid_entry_profile") or ""),
            "grid_prime_session_momentum_long",
        )

    def test_placeholder_cycle_state_does_not_block_prime_restart(self) -> None:
        scalper = XAUGridScalper(enabled=True)
        frame = _features(
            [100.10, 100.32, 100.54, 100.86, 101.24],
            ema20=100.72,
            ema50=100.38,
            bodies=[0.22, 0.24, 0.28, 0.34, 0.40],
            volume_ratio=1.28,
        )
        frame.loc[:, "compression_state"] = "EXPANSION_READY"
        frame.loc[:, "compression_expansion_score"] = 0.56
        frame.loc[:, "multi_tf_alignment_score"] = 0.68
        frame.loc[:, "fractal_persistence_score"] = 0.62
        frame.loc[:, "seasonality_edge_score"] = 0.44
        frame.loc[:, "market_instability_score"] = 0.20
        frame.loc[:, "feature_drift_score"] = 0.14
        frame.loc[:, "m5_body_efficiency"] = 0.44
        frame.loc[:, "m5_range_position_20"] = 0.74
        frame.loc[:, "m15_range_position_20"] = 0.62
        frame.loc[:, "m1_momentum_3"] = 0.005
        frame.loc[:, "m15_ret_1"] = 0.008
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[{"symbol": "XAUUSD"}],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 8, 35, tzinfo=UTC),
            spread_points=12.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(decision.deny_reason, "")

    def test_prime_session_momentum_relaxes_mc_floor_when_recovery_is_active(self) -> None:
        scalper = XAUGridScalper(enabled=True, quota_target_actions_per_window=6, quota_min_actions_per_window=5)
        frame = _features(
            [100.02, 100.10, 100.18, 100.30, 100.42],
            ema20=100.28,
            ema50=100.20,
            bodies=[0.08, 0.10, 0.10, 0.12, 0.12],
            volume_ratio=0.52,
        )
        frame.loc[:, "compression_state"] = "NEUTRAL"
        frame.loc[:, "compression_expansion_score"] = 0.12
        frame.loc[:, "multi_tf_alignment_score"] = 0.13
        frame.loc[:, "fractal_persistence_score"] = 0.11
        frame.loc[:, "seasonality_edge_score"] = 0.18
        frame.loc[:, "market_instability_score"] = 0.42
        frame.loc[:, "feature_drift_score"] = 0.32
        frame.loc[:, "m5_body_efficiency"] = 0.09
        frame.loc[:, "m5_range_position_20"] = 0.18
        frame.loc[:, "m15_range_position_20"] = 0.90
        frame.loc[:, "m1_momentum_3"] = 0.002
        frame.loc[:, "m15_ret_1"] = -0.014
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 10, 35, tzinfo=UTC),
            spread_points=14.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(
            str(decision.candidates[0].meta.get("grid_entry_profile") or ""),
            "grid_prime_session_momentum_long",
        )
        self.assertLessEqual(float(decision.mc_floor), 0.74)

    def test_idle_lane_recovery_can_reopen_quota_reclaim_rescue_after_quota_is_met(self) -> None:
        scalper = XAUGridScalper(enabled=True, quota_target_actions_per_window=6, quota_min_actions_per_window=5)
        scalper.apply_learning_policy(
            {
                "bundle": {
                    "quota_catchup_pressure": 0.72,
                },
                "pair_directive": {
                    "frequency_directives": {
                        "idle_lane_recovery_active": True,
                    },
                },
            }
        )
        now = datetime(2026, 1, 1, 10, 35, tzinfo=UTC)
        scalper._record_quota_actions(
            now_utc=now - timedelta(minutes=1),
            session_name="LONDON",
            session_profile="AGGRESSIVE",
            count=6,
        )
        frame = _features(
            [100.02, 100.10, 100.18, 100.30, 100.42],
            ema20=100.28,
            ema50=100.20,
            bodies=[0.08, 0.10, 0.10, 0.12, 0.12],
            volume_ratio=0.52,
        )
        frame.loc[:, "compression_state"] = "NEUTRAL"
        frame.loc[:, "compression_expansion_score"] = 0.12
        frame.loc[:, "multi_tf_alignment_score"] = 0.13
        frame.loc[:, "fractal_persistence_score"] = 0.11
        frame.loc[:, "seasonality_edge_score"] = 0.18
        frame.loc[:, "market_instability_score"] = 0.42
        frame.loc[:, "feature_drift_score"] = 0.32
        frame.loc[:, "m5_body_efficiency"] = 0.09
        frame.loc[:, "m5_range_position_20"] = 0.18
        frame.loc[:, "m15_range_position_20"] = 0.90
        frame.loc[:, "m1_momentum_3"] = 0.002
        frame.loc[:, "m15_ret_1"] = -0.014
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=now,
            spread_points=14.0,
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(decision.deny_reason, "")
        self.assertTrue(bool(decision.candidates[0].meta.get("quota_reclaim_rescue_active")))
        self.assertEqual(
            str(decision.candidates[0].meta.get("grid_entry_profile") or ""),
            "grid_prime_session_momentum_long",
        )

    def test_quota_floor_burst_count_keeps_overlap_tighter_than_london(self) -> None:
        scalper = XAUGridScalper(
            enabled=True,
            quota_target_actions_per_window=6,
            quota_min_actions_per_window=5,
            quota_catchup_burst_cap=8,
            prime_burst_entries=8,
            max_levels=8,
        )
        quota_state = {
            "density_first_active": True,
            "quota_debt": 4,
            "minimum": 5,
            "target": 6,
        }

        london = scalper._quota_floor_burst_count(
            burst_count=2,
            quota_state=quota_state,
            session_name="LONDON",
            session_profile="AGGRESSIVE",
            grid_mode="ATTACK_GRID",
            entry_profile="grid_prime_session_momentum_long",
            support_sources=1,
            grid_max_levels=8,
        )
        overlap = scalper._quota_floor_burst_count(
            burst_count=2,
            quota_state=quota_state,
            session_name="OVERLAP",
            session_profile="AGGRESSIVE",
            grid_mode="ATTACK_GRID",
            entry_profile="grid_prime_session_momentum_long",
            support_sources=1,
            grid_max_levels=8,
        )

        self.assertGreaterEqual(london, 6)
        self.assertLess(overlap, london)

    def test_density_first_monte_carlo_soft_cap_keeps_prime_burst_above_hard_two_leg_clamp(self) -> None:
        scalper = XAUGridScalper(enabled=True, quota_target_actions_per_window=6, quota_min_actions_per_window=5)
        adjusted, reason, penalty = scalper._apply_density_first_monte_carlo_burst_cap(
            burst_count=6,
            monte_carlo_win_rate=0.74,
            mc_floor=0.80,
            quota_state={"density_first_active": True, "quota_debt": 5},
            session_name="LONDON",
            session_profile="AGGRESSIVE",
        )

        self.assertGreaterEqual(adjusted, 4)
        self.assertEqual(reason, "grid_monte_carlo_soft_cap")
        self.assertGreater(penalty, 0.0)

    def test_session_density_overrides_change_quota_targets_and_penalty_caps(self) -> None:
        scalper = XAUGridScalper(
            enabled=True,
            quota_target_actions_per_window=6,
            quota_min_actions_per_window=5,
            density_soft_penalty_max=0.12,
            session_density_overrides={
                "LONDON": {
                    "quota_target_actions_per_window": 7,
                    "quota_min_actions_per_window": 6,
                    "density_soft_penalty_max": 0.08,
                },
                "OVERLAP": {
                    "quota_target_actions_per_window": 5,
                    "quota_min_actions_per_window": 4,
                    "density_soft_penalty_max": 0.16,
                },
            },
        )

        london_state = scalper._quota_state(
            now_utc=datetime(2026, 1, 1, 9, 20, tzinfo=UTC),
            session_name="LONDON",
            session_profile="AGGRESSIVE",
        )
        overlap_state = scalper._quota_state(
            now_utc=datetime(2026, 1, 1, 14, 20, tzinfo=UTC),
            session_name="OVERLAP",
            session_profile="AGGRESSIVE",
        )
        london_penalty = scalper._quota_quality_penalty(
            gate_reason="grid_prime_session_quality_gate",
            session_name="LONDON",
            quota_debt=6,
            support_sources=0,
        )
        overlap_penalty = scalper._quota_quality_penalty(
            gate_reason="grid_prime_session_quality_gate",
            session_name="OVERLAP",
            quota_debt=6,
            support_sources=0,
        )

        self.assertEqual(int(london_state["target"]), 7)
        self.assertEqual(int(london_state["minimum"]), 6)
        self.assertEqual(int(overlap_state["target"]), 5)
        self.assertEqual(int(overlap_state["minimum"]), 4)
        self.assertLessEqual(float(london_penalty), 0.08)
        self.assertLessEqual(float(overlap_penalty), 0.16)

    def test_density_first_quota_floor_add_count_promotes_prime_follow_through_adds(self) -> None:
        scalper = XAUGridScalper(enabled=True, quota_target_actions_per_window=6, quota_min_actions_per_window=5)
        adjusted = scalper._quota_floor_add_count(
            add_count=1,
            quota_state={"density_first_active": True, "quota_debt": 5},
            session_name="LONDON",
            session_profile="AGGRESSIVE",
            grid_mode="ATTACK_GRID",
            follow_through_add_ready=True,
            support_sources=1,
            remaining_levels=5,
        )

        self.assertGreaterEqual(adjusted, 3)

    def test_density_first_monte_carlo_soft_cap_keeps_prime_adds_above_hard_two_leg_clamp(self) -> None:
        scalper = XAUGridScalper(enabled=True, quota_target_actions_per_window=6, quota_min_actions_per_window=5)
        adjusted, reason, penalty = scalper._apply_density_first_monte_carlo_add_cap(
            add_count=4,
            monte_carlo_win_rate=0.73,
            mc_floor=0.80,
            quota_state={"density_first_active": True, "quota_debt": 5},
            session_name="LONDON",
            session_profile="AGGRESSIVE",
            remaining_levels=5,
        )

        self.assertGreaterEqual(adjusted, 3)
        self.assertEqual(reason, "grid_add_monte_carlo_soft_cap")
        self.assertGreater(penalty, 0.0)

    def test_existing_cycle_can_abort_hostile_environment(self) -> None:
        scalper = XAUGridScalper(enabled=True, max_levels=6)
        frame = _features(
            [100.0, 99.7, 99.4, 99.0, 98.6],
            ema20=101.2,
            ema50=102.0,
            atr=1.6,
            atr_avg=1.1,
            bodies=[0.7, 0.6, 0.5, 0.4, -0.2],
            volume_ratio=0.8,
        )
        row = frame.iloc[-1]
        open_positions = [
            _open_position(entry=100.6, volume=0.01),
            _open_position(entry=100.2, volume=0.01, opened_at=datetime(2026, 1, 1, 0, 5, tzinfo=UTC)),
        ]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=open_positions,
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 8, 25, tzinfo=UTC),
            spread_points=18.0,
            contract_size=100.0,
        )

        self.assertTrue(decision.close_cycle)
        self.assertIn(
            decision.close_reason,
            {"grid_hostile_environment_abort", "grid_direction_lost_abort", "grid_prime_rearm_no_follow_through"},
        )

    def test_aggressive_prime_cycle_can_rearm_inside_first_minute(self) -> None:
        scalper = XAUGridScalper(enabled=True, max_levels=6, micro_take_usd=0.10)
        now_utc = datetime(2026, 1, 1, 8, 25, tzinfo=UTC)
        frame = _features(
            [100.0, 100.05, 100.08, 100.10, 100.09],
            ema20=100.25,
            ema50=100.35,
            atr=1.2,
            atr_avg=1.0,
            bodies=[0.08, 0.06, 0.05, 0.03, -0.01],
            volume_ratio=0.78,
        )
        row = frame.iloc[-1]
        open_positions = [
            _open_position(entry=100.10, volume=0.01, opened_at=now_utc - timedelta(seconds=20)),
        ]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=open_positions,
            session_name="LONDON",
            news_safe=True,
            now_utc=now_utc,
            spread_points=14.0,
            contract_size=100.0,
        )

        self.assertTrue(decision.close_cycle)
        self.assertEqual(decision.close_reason, "grid_prime_rearm_no_follow_through")

    def test_burst_start_count_expands_with_strong_support_sources(self) -> None:
        scalper = XAUGridScalper(enabled=True, max_levels=6, max_open_cycles=2, prime_burst_entries=4)

        count = scalper._burst_start_count(
            session_profile="AGGRESSIVE",
            grid_mode="ATTACK_GRID",
            entry_profile="grid_breakout_reclaim_long",
            confluence=4.9,
            alignment_score=0.64,
            fractal_score=0.58,
            trend_efficiency=0.52,
            body_efficiency=0.36,
            compression_expansion_score=0.38,
            instability_score=0.34,
            prime_recovery_active=False,
            support_sources=2,
            grid_max_levels=6,
        )

        self.assertEqual(count, 6)

    def test_burst_start_count_handles_moderate_stretch_profile_without_name_error(self) -> None:
        scalper = XAUGridScalper(enabled=True, max_levels=6, max_open_cycles=2, moderate_burst_entries=3)

        count = scalper._burst_start_count(
            session_profile="MODERATE",
            grid_mode="BALANCED_GRID",
            entry_profile="grid_trend_reclaim_long",
            confluence=3.6,
            alignment_score=0.50,
            fractal_score=0.48,
            trend_efficiency=0.38,
            body_efficiency=0.26,
            compression_expansion_score=0.30,
            instability_score=0.40,
            prime_recovery_active=False,
            support_sources=0,
            grid_max_levels=5,
        )

        self.assertEqual(count, 3)

    def test_existing_cycle_respects_max_levels(self) -> None:
        scalper = XAUGridScalper(enabled=True, max_levels=2)
        frame = _features(
            [100.0, 99.6, 99.2, 99.0],
            ema20=101.0,
            ema50=103.0,
            bodies=[0.5, 0.4, 0.3, 0.2],
        )
        row = frame.iloc[-1]
        open_positions = [
            _open_position(entry=100.2),
            _open_position(entry=100.0),
        ]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=open_positions,
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 8, 20, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )

        self.assertEqual(decision.candidates, [])
        self.assertEqual(decision.deny_reason, "grid_max_levels_reached")

    def test_symbol_exposure_cap_forces_flatten(self) -> None:
        scalper = XAUGridScalper(enabled=True, max_open_positions_symbol=1)
        frame = _features([100.0, 100.1, 100.2], ema20=100.0, ema50=100.0)
        row = frame.iloc[-1]
        open_positions = [_open_position(entry=100.0), _open_position(entry=100.1)]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=open_positions,
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 8, 25, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )

        self.assertTrue(decision.close_cycle)
        self.assertEqual(decision.close_reason, "grid_symbol_position_cap_exceeded")

    def test_hard_stop_sets_close_and_cooldown(self) -> None:
        scalper = XAUGridScalper(enabled=True, stop_atr_k=2.5, cooldown_after_stop_minutes=30)
        frame = _features([100.0, 98.8, 97.0], atr=1.0, ema20=101.0, ema50=101.0)
        row = frame.iloc[-1]
        now = datetime(2026, 1, 1, 8, 30, tzinfo=UTC)
        open_positions = [_open_position(entry=100.0)]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=open_positions,
            session_name="LONDON",
            news_safe=True,
            now_utc=now,
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )
        self.assertTrue(decision.close_cycle)
        self.assertEqual(decision.close_reason, "grid_cycle_hard_stop")

        second = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=now + timedelta(minutes=5),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )
        self.assertFalse(second.deny_reason.startswith("grid_cooldown_until_"))

    def test_asia_probe_requires_sweep_reclaim(self) -> None:
        scalper = XAUGridScalper(enabled=True)
        frame = _features(
            [100.2, 100.1, 100.0, 99.9],
            atr=0.8,
            atr_avg=1.0,
            ema20=100.0,
            ema50=100.1,
            spread=12.0,
            bodies=[0.3, 0.3, 0.2, 0.1],
        )
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="SYDNEY",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 2, 0, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )

        self.assertEqual(decision.candidates, [])
        self.assertIn(
            decision.deny_reason,
            {"grid_asia_probe_no_directional_trigger", "grid_no_reclaim_quality", "grid_asia_probe_mc_floor"},
        )

    def test_asia_probe_counts_as_density_session_for_quota_tracking(self) -> None:
        scalper = XAUGridScalper(
            enabled=True,
            asia_probe_enabled=True,
            asia_probe_sessions=("TOKYO", "SYDNEY"),
            allowed_sessions=("SYDNEY", "TOKYO", "LONDON"),
        )

        quota_state = scalper._quota_state(
            now_utc=datetime(2026, 1, 1, 0, 15, tzinfo=UTC),
            session_name="TOKYO",
            session_profile="ASIA_PROBE",
        )

        self.assertEqual(str(quota_state["state"]), "CATCHUP")
        self.assertTrue(bool(quota_state["density_first_active"]))

    def test_asia_probe_effective_mc_floor_relaxes_for_structured_probe(self) -> None:
        floor = XAUGridScalper._effective_asia_probe_mc_floor(
            base_floor=0.80,
            asia_probe_mc_floor=0.80,
            entry_profile="grid_density_micro_scaler_long",
            asia_density_relief_active=True,
            pressure_edge=0.63,
            trend_efficiency=0.28,
            alignment_score=0.31,
            body_efficiency=0.22,
            volume_ratio=0.70,
        )

        self.assertEqual(floor, 0.70)

    def test_asia_probe_effective_mc_floor_stays_strict_when_structure_is_thin(self) -> None:
        floor = XAUGridScalper._effective_asia_probe_mc_floor(
            base_floor=0.80,
            asia_probe_mc_floor=0.80,
            entry_profile="grid_asia_probe_directional_long",
            asia_density_relief_active=False,
            pressure_edge=0.52,
            trend_efficiency=0.16,
            alignment_score=0.14,
            body_efficiency=0.10,
            volume_ratio=0.55,
        )

        self.assertEqual(floor, 0.79)

    def test_density_micro_scaler_creates_london_candidate_when_reclaim_stack_is_almost_ready(self) -> None:
        scalper = XAUGridScalper(enabled=True, density_micro_scaler_enabled=True)
        frame = _features(
            [100.2, 100.5, 100.8, 101.0, 101.2],
            atr=0.9,
            atr_avg=0.9,
            ema20=100.8,
            ema50=100.6,
            spread=14.0,
            bodies=[0.18, 0.20, 0.18, 0.16, 0.12],
            volume_ratio=0.62,
        )
        frame.loc[:, "compression_state"] = "COMPRESSION"
        frame.loc[:, "m5_body_efficiency"] = 0.07
        frame.loc[:, "multi_tf_alignment_score"] = 0.11
        frame.loc[:, "fractal_persistence_score"] = 0.09
        frame.loc[:, "seasonality_edge_score"] = 0.24
        frame.loc[:, "market_instability_score"] = 0.34
        frame.loc[:, "feature_drift_score"] = 0.18
        frame.loc[:, "m5_trend_efficiency_16"] = 0.09
        frame.loc[:, "m5_range_position_20"] = 0.58
        frame.loc[:, "m15_range_position_20"] = 0.61
        frame.loc[:, "m15_volume_ratio_20"] = 0.70
        frame.loc[:, "m15_ema_20"] = 100.7
        frame.loc[:, "m15_ema_50"] = 100.5
        frame.loc[:, "m1_momentum_3"] = 0.02
        frame.loc[:, "m15_ret_1"] = 0.008
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 8, 20, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(decision.deny_reason, "")
        self.assertIn(
            str(decision.candidates[0].meta.get("grid_entry_profile") or ""),
            {"grid_density_micro_scaler_long", "grid_prime_session_momentum_long"},
        )

    def test_asia_probe_recovery_creates_candidate_when_directional_trigger_is_stalled(self) -> None:
        scalper = XAUGridScalper(enabled=True, density_micro_scaler_enabled=True)
        frame = _features(
            [100.02, 100.05, 100.09, 100.14],
            atr=0.8,
            atr_avg=1.0,
            ema20=100.10,
            ema50=100.04,
            spread=14.0,
            bodies=[0.03, 0.04, 0.05, 0.05],
            volume_ratio=0.62,
        )
        frame.loc[:, "compression_state"] = "NEUTRAL"
        frame.loc[:, "compression_expansion_score"] = 0.22
        frame.loc[:, "multi_tf_alignment_score"] = 0.38
        frame.loc[:, "fractal_persistence_score"] = 0.40
        frame.loc[:, "seasonality_edge_score"] = 0.68
        frame.loc[:, "market_instability_score"] = 0.22
        frame.loc[:, "feature_drift_score"] = 0.18
        frame.loc[:, "m5_body_efficiency"] = 0.05
        frame.loc[:, "m5_trend_efficiency_16"] = 0.40
        frame.loc[:, "m5_range_position_20"] = 0.42
        frame.loc[:, "m15_range_position_20"] = 0.44
        frame.loc[:, "m15_volume_ratio_20"] = 0.66
        frame.loc[:, "m15_ema_20"] = 100.08
        frame.loc[:, "m15_ema_50"] = 100.02
        frame.loc[:, "m1_momentum_3"] = -0.002
        frame.loc[:, "m15_ret_1"] = -0.004
        frame.loc[:, "h1_close"] = 100.40
        frame.loc[:, "h1_ema_20"] = 100.18
        frame.loc[:, "h1_ema_50"] = 100.06
        frame.loc[:, "h1_range_position_20"] = 0.62
        frame.loc[:, "h4_close"] = 100.62
        frame.loc[:, "h4_ema_20"] = 100.30
        frame.loc[:, "h4_ema_50"] = 100.14
        frame.loc[:, "h4_range_position_20"] = 0.68
        frame.loc[:, "d1_close"] = 100.90
        frame.loc[:, "d1_ema_20"] = 100.42
        frame.loc[:, "d1_ema_50"] = 100.20
        frame.loc[:, "d1_range_position_20"] = 0.74
        frame.loc[:, "dxy_ret_15"] = -0.0012
        frame.loc[:, "dxy_ret_60"] = -0.0018
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="TOKYO",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 0, 15, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )

        self.assertNotEqual(decision.entry_profile, "")
        self.assertEqual(str(decision.entry_profile), "grid_asia_probe_recovery_long")
        self.assertTrue(bool(decision.density_relief_active))
        self.assertIn(decision.deny_reason, {"", "grid_asia_probe_mc_floor"})

    def test_asia_probe_density_relief_creates_candidate_for_moderate_sydney_structure(self) -> None:
        scalper = XAUGridScalper(enabled=True, density_micro_scaler_enabled=True)
        frame = _features(
            [100.02, 100.05, 100.07, 100.10, 100.12],
            atr=0.82,
            atr_avg=1.0,
            ema20=100.08,
            ema50=100.04,
            spread=16.0,
            bodies=[0.02, 0.02, 0.03, 0.03, 0.03],
            volume_ratio=0.32,
        )
        frame.loc[:, "compression_state"] = "NEUTRAL"
        frame.loc[:, "compression_expansion_score"] = 0.16
        frame.loc[:, "multi_tf_alignment_score"] = 0.03
        frame.loc[:, "fractal_persistence_score"] = 0.03
        frame.loc[:, "seasonality_edge_score"] = 0.22
        frame.loc[:, "market_instability_score"] = 0.30
        frame.loc[:, "feature_drift_score"] = 0.22
        frame.loc[:, "m5_body_efficiency"] = 0.03
        frame.loc[:, "m5_trend_efficiency_16"] = 0.05
        frame.loc[:, "m5_range_position_20"] = 0.54
        frame.loc[:, "m15_range_position_20"] = 0.52
        frame.loc[:, "m15_volume_ratio_20"] = 0.34
        frame.loc[:, "m15_ema_20"] = 100.07
        frame.loc[:, "m15_ema_50"] = 100.03
        frame.loc[:, "m1_momentum_3"] = 0.001
        frame.loc[:, "m15_ret_1"] = -0.001
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="SYDNEY",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 19, 15, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )

        self.assertEqual(str(decision.entry_profile), "grid_asia_probe_recovery_long")
        self.assertTrue(bool(decision.density_relief_active))
        self.assertIn(decision.deny_reason, {"", "grid_asia_probe_mc_floor"})

    def test_asia_probe_normalizes_large_raw_m1_momentum_for_live_xau_pullback(self) -> None:
        scalper = XAUGridScalper(enabled=True, density_micro_scaler_enabled=True)
        frame = _features(
            [4415.0, 4422.0, 4428.0, 4435.0, 4438.1],
            atr=6.33,
            atr_avg=8.21,
            ema20=4422.0,
            ema50=4408.0,
            spread=16.0,
            bodies=[0.8, 1.2, 1.6, 1.8, -2.0],
            volume_ratio=4.14,
        )
        frame.loc[:, "m5_open"] = [4414.2, 4420.8, 4426.4, 4433.2, 4440.1]
        frame.loc[:, "m5_high"] = [4415.8, 4422.8, 4429.0, 4435.9, 4440.6]
        frame.loc[:, "m5_low"] = [4414.0, 4420.6, 4426.2, 4433.0, 4438.1]
        frame.loc[:, "compression_state"] = "COMPRESSION"
        frame.loc[:, "compression_expansion_score"] = 0.32
        frame.loc[:, "multi_tf_alignment_score"] = 1.0
        frame.loc[:, "fractal_persistence_score"] = 0.34
        frame.loc[:, "seasonality_edge_score"] = 0.39
        frame.loc[:, "market_instability_score"] = 0.35
        frame.loc[:, "feature_drift_score"] = 0.52
        frame.loc[:, "m5_body_efficiency"] = 0.80
        frame.loc[:, "m5_trend_efficiency_16"] = 0.73
        frame.loc[:, "m5_range_position_20"] = 0.92
        frame.loc[:, "m15_range_position_20"] = 0.94
        frame.loc[:, "m15_volume_ratio_20"] = 0.51
        frame.loc[:, "m15_ema_20"] = 4408.7
        frame.loc[:, "m15_ema_50"] = 4412.0
        frame.loc[:, "m1_momentum_3"] = -0.70
        frame.loc[:, "m15_ret_1"] = -0.0002
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="TOKYO",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 5, 0, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )

        self.assertNotEqual(decision.deny_reason, "grid_asia_probe_no_directional_trigger")
        self.assertIn(
            decision.deny_reason,
            {"", "grid_asia_probe_mc_floor"},
        )

    def test_asia_probe_continuation_catches_extended_tokyo_drift(self) -> None:
        scalper = XAUGridScalper(enabled=True, density_micro_scaler_enabled=True)
        frame = _features(
            [100.6, 101.2, 101.9, 102.8, 103.7],
            atr=0.82,
            atr_avg=1.0,
            ema20=101.8,
            ema50=101.1,
            spread=15.0,
            bodies=[0.30, 0.34, 0.42, 0.56, 0.62],
            volume_ratio=0.20,
        )
        frame.loc[:, "compression_state"] = "NEUTRAL"
        frame.loc[:, "compression_expansion_score"] = 0.18
        frame.loc[:, "multi_tf_alignment_score"] = 0.75
        frame.loc[:, "fractal_persistence_score"] = 0.35
        frame.loc[:, "seasonality_edge_score"] = 0.39
        frame.loc[:, "market_instability_score"] = 0.28
        frame.loc[:, "feature_drift_score"] = 0.13
        frame.loc[:, "m5_body_efficiency"] = 0.59
        frame.loc[:, "m5_trend_efficiency_16"] = 0.18
        frame.loc[:, "m5_range_position_20"] = 0.98
        frame.loc[:, "m15_range_position_20"] = 1.0
        frame.loc[:, "m15_volume_ratio_20"] = 0.42
        frame.loc[:, "m15_ema_20"] = 101.9
        frame.loc[:, "m15_ema_50"] = 101.2
        frame.loc[:, "m1_momentum_3"] = 1.0
        frame.loc[:, "m15_ret_1"] = 0.0007
        frame.loc[:, "h1_close"] = 104.2
        frame.loc[:, "h1_ema_20"] = 102.6
        frame.loc[:, "h1_ema_50"] = 101.7
        frame.loc[:, "h1_range_position_20"] = 0.96
        frame.loc[:, "h4_close"] = 104.8
        frame.loc[:, "h4_ema_20"] = 102.9
        frame.loc[:, "h4_ema_50"] = 101.9
        frame.loc[:, "h4_range_position_20"] = 0.94
        frame.loc[:, "d1_close"] = 105.5
        frame.loc[:, "d1_ema_20"] = 103.2
        frame.loc[:, "d1_ema_50"] = 102.1
        frame.loc[:, "d1_range_position_20"] = 0.92
        frame.loc[:, "dxy_ret_15"] = -0.0008
        frame.loc[:, "dxy_ret_60"] = -0.0011
        row = frame.iloc[-1]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=[],
            session_name="TOKYO",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 0, 50, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )

        self.assertGreaterEqual(len(decision.candidates), 1)
        self.assertEqual(decision.deny_reason, "")
        self.assertEqual(
            str(decision.candidates[0].meta.get("grid_entry_profile") or ""),
            "grid_asia_probe_continuation_long",
        )

    def test_spread_disorder_flattens_existing_cycle(self) -> None:
        scalper = XAUGridScalper(enabled=True, flatten_spread_points=30.0)
        frame = _features([100.0, 100.1, 100.2], ema20=100.0, ema50=100.0, spread=35.0)
        row = frame.iloc[-1]
        open_positions = [_open_position(entry=100.0)]

        decision = scalper.evaluate(
            symbol="XAUUSD",
            features=frame,
            row=row,
            open_positions=open_positions,
            session_name="LONDON",
            news_safe=True,
            now_utc=datetime(2026, 1, 1, 8, 40, tzinfo=UTC),
            spread_points=float(row["m5_spread"]),
            contract_size=100.0,
        )

        self.assertTrue(decision.close_cycle)
        self.assertEqual(decision.close_reason, "grid_spread_disorder_flat")

    def test_prime_session_quality_gate_keeps_new_york_directional_strict(self) -> None:
        deny_reason = XAUGridScalper._prime_session_native_quality_gate(
            session_name="NEW_YORK",
            entry_profile="grid_directional_flow_long",
            quality_tier="B",
            monte_carlo_win_rate=0.81,
            mc_floor=0.80,
            htf_alignment_score=0.58,
            structure_cleanliness_score=0.59,
            execution_quality_fit=0.57,
            router_rank_score=0.77,
            support_sources=1,
        )

        self.assertEqual(deny_reason, "grid_prime_session_quality_gate")

    def test_prime_session_quality_gate_allows_overlap_reclaim_recovery(self) -> None:
        deny_reason = XAUGridScalper._prime_session_native_quality_gate(
            session_name="OVERLAP",
            entry_profile="grid_breakout_reclaim_long",
            quality_tier="C",
            monte_carlo_win_rate=0.80,
            mc_floor=0.80,
            htf_alignment_score=0.56,
            structure_cleanliness_score=0.56,
            execution_quality_fit=0.55,
            router_rank_score=0.74,
            support_sources=1,
        )

        self.assertEqual(deny_reason, "")

    def test_prime_session_quality_gate_allows_new_york_b_tier_near_a_override(self) -> None:
        deny_reason = XAUGridScalper._prime_session_native_quality_gate(
            session_name="NEW_YORK",
            entry_profile="grid_prime_session_momentum_long",
            quality_tier="B",
            monte_carlo_win_rate=0.85,
            mc_floor=0.80,
            htf_alignment_score=0.63,
            structure_cleanliness_score=0.63,
            execution_quality_fit=0.61,
            router_rank_score=0.80,
            support_sources=1,
        )

        self.assertEqual(deny_reason, "")

    def test_london_native_burst_floor_promotes_strong_b_tier_attack_grid(self) -> None:
        burst_count = XAUGridScalper._london_native_burst_floor(
            burst_count=1,
            session_name="LONDON",
            session_profile="AGGRESSIVE",
            grid_mode="ATTACK_GRID",
            entry_profile="grid_breakout_reclaim_long",
            quality_tier="B",
            confluence=4.05,
            monte_carlo_win_rate=0.87,
            mc_floor=0.80,
            htf_alignment_score=0.69,
            structure_cleanliness_score=0.69,
            execution_quality_fit=0.65,
            router_rank_score=0.83,
            support_sources=1,
            grid_max_levels=8,
        )

        self.assertGreaterEqual(burst_count, 3)

    def test_micro_take_profile_raises_new_york_quick_green_threshold(self) -> None:
        profile = XAUGridScalper._micro_take_profile(session_name="NEW_YORK", session_profile="AGGRESSIVE")

        self.assertEqual(profile["min_open_minutes"], 2.0)
        self.assertGreater(profile["quick_green_multiplier"], 0.40)
        self.assertGreater(profile["mean_revert_multiplier"], 0.55)

    def test_micro_take_profile_keeps_asia_probe_ultra_fast(self) -> None:
        profile = XAUGridScalper._micro_take_profile(session_name="TOKYO", session_profile="ASIA_PROBE")

        self.assertEqual(profile["min_open_minutes"], 0.0)
        self.assertLess(profile["quick_green_multiplier"], 0.40)
        self.assertLess(profile["mean_revert_multiplier"], 0.55)

    def test_micro_take_profile_keeps_default_for_london(self) -> None:
        profile = XAUGridScalper._micro_take_profile(session_name="LONDON", session_profile="AGGRESSIVE")

        self.assertEqual(profile["min_open_minutes"], 1.0)
        self.assertEqual(profile["quick_green_multiplier"], 0.40)
        self.assertEqual(profile["mean_revert_multiplier"], 0.55)

    def test_session_profile_marks_london_open_as_aggressive_with_widened_window(self) -> None:
        scalper = XAUGridScalper(enabled=True)

        profile = scalper._session_profile(  # noqa: SLF001
            session_name="LONDON",
            now_utc=datetime(2026, 1, 1, 7, 5, tzinfo=UTC),
            atr_ratio=1.02,
            spread_points=14.0,
        )

        self.assertEqual(profile, "AGGRESSIVE")


if __name__ == "__main__":
    unittest.main()
