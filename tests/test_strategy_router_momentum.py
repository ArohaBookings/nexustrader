from __future__ import annotations

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

import pandas as pd

from src.regime_detector import RegimeClassification
from src.session_profile import SessionContext
from src.strategy_engine import SignalCandidate, StrategyEngine
from src.strategy_router import StrategyRouter
from src.strategies.trend_daytrade import resolve_strategy_key


class StrategyRouterMomentumTests(unittest.TestCase):
    def _row(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 6, 10, 0, tzinfo=timezone.utc),
                    "m5_close": 1.0820,
                    "m5_spread": 12.0,
                    "m5_atr_14": 0.0007,
                    "m5_ret_1": 0.0,
                    "m5_volume_ratio_20": 1.05,
                    "m5_ema_20": 1.0818,
                    "m5_ema_50": 1.0815,
                    "m5_bullish": 0,
                    "m5_bearish": 0,
                    "m1_atr_pct_of_avg": 1.1,
                    "m5_atr_pct_of_avg": 1.1,
                    "m1_close": 1.0820,
                    "m1_rolling_high_prev_20": 1.0819,
                    "m1_rolling_low_prev_20": 1.0810,
                    "m1_volume_ratio_20": 1.1,
                    "m1_body": 0.00012,
                    "m1_atr_14": 0.0002,
                    "m1_momentum_3": 0.0001,
                    "m5_macd_hist": 0.02,
                    "m5_macd_hist_slope": 0.01,
                    "m15_close": 1.0820,
                    "m15_ema_20": 1.0817,
                    "h1_ema_20": 1.0818,
                    "h1_ema_50": 1.0812,
                    "h1_rsi_14": 56.0,
                    "h4_ema_50": 1.0819,
                    "h4_ema_200": 1.0800,
                }
            ]
        )

    def test_btc_recycle_queue_replays_once_per_family_day(self) -> None:
        router = StrategyRouter()
        candidate = SignalCandidate(
            signal_id="btc-recycle",
            setup="BTC_WEEKEND_BREAKOUT",
            side="BUY",
            score_hint=0.62,
            reason="recyclable btc idea",
            stop_atr=1.0,
            tp_r=1.5,
            strategy_family="TREND",
            meta={"quality_tier": "B", "strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION", "regime_fit": 0.60},
        )

        router._queue_recyclable_candidate(
            candidate=candidate,
            symbol="BTCUSD",
            session_name="SYDNEY",
            timestamp=datetime(2026, 3, 14, 1, 0, tzinfo=timezone.utc),
        )
        router._queue_recyclable_candidate(
            candidate=candidate,
            symbol="BTCUSD",
            session_name="SYDNEY",
            timestamp=datetime(2026, 3, 14, 1, 5, tzinfo=timezone.utc),
        )

        recycled = router._pull_recycled_candidates(
            symbol="BTCUSD",
            session_name="SYDNEY",
            timestamp=datetime(2026, 3, 14, 8, 0, tzinfo=timezone.utc),
            weekend_mode=True,
        )

        self.assertEqual(len(recycled), 1)
        self.assertTrue(bool(recycled[0].meta.get("recycle_session")))
        self.assertAlmostEqual(float(recycled[0].meta.get("recycle_boost_applied", 0.0)), 0.25)

    def test_same_bar_substitution_keeps_btc_continuation_over_tokyo_trend_scalp(self) -> None:
        router = StrategyRouter()
        continuation = SignalCandidate(
            signal_id="btc-cont",
            setup="BTCUSD_PRICE_ACTION_CONTINUATION",
            side="BUY",
            score_hint=0.74,
            reason="keep continuation",
            stop_atr=1.0,
            tp_r=2.2,
            strategy_family="TREND",
            meta={"strategy_key": "BTCUSD_PRICE_ACTION_CONTINUATION"},
        )
        trend_scalp = SignalCandidate(
            signal_id="btc-scalp",
            setup="BTCUSD_TREND_SCALP",
            side="BUY",
            score_hint=0.75,
            reason="drop scalp if continuation exists",
            stop_atr=1.0,
            tp_r=1.8,
            strategy_family="TREND",
            meta={"strategy_key": "BTCUSD_TREND_SCALP"},
        )

        kept = router._apply_same_bar_substitution(  # noqa: SLF001
            candidates=[continuation, trend_scalp],
            symbol="BTCUSD",
            session_name="TOKYO",
            regime_state="TRENDING",
        )

        kept_keys = {str(item.meta.get("strategy_key") or "") for item in kept}
        self.assertIn("BTCUSD_PRICE_ACTION_CONTINUATION", kept_keys)
        self.assertNotIn("BTCUSD_TREND_SCALP", kept_keys)

    def test_same_bar_substitution_keeps_nas_sweep_over_tokyo_impulse(self) -> None:
        router = StrategyRouter()
        sweep = SignalCandidate(
            signal_id="nas-sweep",
            setup="NAS100_LIQUIDITY_SWEEP_REVERSAL",
            side="BUY",
            score_hint=0.72,
            reason="keep sweep",
            stop_atr=1.0,
            tp_r=2.0,
            strategy_family="TREND",
            meta={"strategy_key": "NAS100_LIQUIDITY_SWEEP_REVERSAL"},
        )
        impulse = SignalCandidate(
            signal_id="nas-impulse",
            setup="NAS100_MOMENTUM_IMPULSE",
            side="BUY",
            score_hint=0.73,
            reason="drop impulse if sweep exists",
            stop_atr=1.0,
            tp_r=1.8,
            strategy_family="TREND",
            meta={"strategy_key": "NAS100_MOMENTUM_IMPULSE"},
        )

        kept = router._apply_same_bar_substitution(  # noqa: SLF001
            candidates=[sweep, impulse],
            symbol="NAS100",
            session_name="TOKYO",
            regime_state="TRENDING",
        )

        kept_keys = {str(item.meta.get("strategy_key") or "") for item in kept}
        self.assertIn("NAS100_LIQUIDITY_SWEEP_REVERSAL", kept_keys)
        self.assertNotIn("NAS100_MOMENTUM_IMPULSE", kept_keys)

    def test_unique_keeps_distinct_grid_burst_legs(self) -> None:
        router = StrategyRouter()
        first = SignalCandidate(
            signal_id="grid-start-1",
            setup="XAUUSD_M5_GRID_SCALPER_START",
            side="BUY",
            score_hint=0.82,
            reason="burst leg 1",
            stop_atr=0.8,
            tp_r=2.2,
            strategy_family="GRID",
            meta={"grid_cycle": True, "grid_cycle_id": "cycle-1", "grid_level": 1, "grid_burst_index": 1},
        )
        second = SignalCandidate(
            signal_id="grid-start-2",
            setup="XAUUSD_M5_GRID_SCALPER_START",
            side="BUY",
            score_hint=0.81,
            reason="burst leg 2",
            stop_atr=0.8,
            tp_r=2.2,
            strategy_family="GRID",
            meta={"grid_cycle": True, "grid_cycle_id": "cycle-1", "grid_level": 2, "grid_burst_index": 2},
        )

        kept = router._unique([first, second])  # noqa: SLF001

        self.assertEqual(len(kept), 2)
        self.assertEqual([item.signal_id for item in kept], ["grid-start-1", "grid-start-2"])

    def test_learning_policy_boosts_promoted_pattern_rank(self) -> None:
        frame = self._row()
        row = frame.iloc[-1]
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.82,
            source="test",
            details={
                "trend_flag": 1.0,
                "pressure_proxy_score": 0.74,
                "compression_proxy_state": "EXPANSION_READY",
                "compression_expansion_score": 0.58,
            },
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        candidate = SignalCandidate(
            signal_id="usdjpy-brain",
            setup="USDJPY_MOMENTUM_IMPULSE",
            side="BUY",
            score_hint=0.67,
            reason="brain-promoted impulse",
            stop_atr=1.0,
            tp_r=1.8,
            strategy_family="TREND",
            meta={"strategy_key": "USDJPY_MOMENTUM_IMPULSE"},
        )

        baseline_router = StrategyRouter(max_spread_points=60.0)
        baseline = baseline_router._enrich_candidate(
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            candidate=SignalCandidate(**candidate.__dict__),
        )

        router = StrategyRouter(max_spread_points=60.0)
        router.apply_learning_policy(
            {
                "symbol": "USDJPY",
                "bundle": {
                    "promoted_patterns": ["USDJPY_MOMENTUM_IMPULSE"],
                    "weak_pair_focus": ["USDJPY"],
                    "shadow_pair_focus": [],
                    "quota_catchup_pressure": 0.82,
                },
                "reentry_watchlist": ["USDJPY_MOMENTUM_IMPULSE"],
                "weekly_trade_ideas": ["USDJPY TOKYO MOMENTUM"],
            }
        )
        enriched = router._enrich_candidate(
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            candidate=SignalCandidate(**candidate.__dict__),
        )

        self.assertTrue(bool(enriched.meta.get("learning_brain_promoted_pattern")))

    def test_learning_policy_marks_native_session_throughput_recovery_on_high_catchup_pressure(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        router.apply_learning_policy(
            {
                "symbol": "AUDNZD",
                "bundle": {
                    "promoted_patterns": [],
                    "weak_pair_focus": [],
                    "shadow_pair_focus": [],
                    "quota_catchup_pressure": 0.9,
                },
            }
        )

        state = router._learning_pattern_state(
            symbol="AUDNZD",
            strategy_key="AUDNZD_STRUCTURE_BREAK_RETEST",
            session_name="TOKYO",
        )

        self.assertTrue(bool(state["throughput_recovery_active"]))
        self.assertGreater(router._symbol_sensitivity("AUDNZD", 1.0), 1.0)

    def test_learning_pair_directive_adds_aggression_bonus_for_native_session(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        router.apply_learning_policy(
            {
                "symbol": "USDJPY",
                "bundle": {
                    "quota_catchup_pressure": 0.70,
                },
                "pair_directive": {
                    "session_focus": ["TOKYO", "SYDNEY"],
                    "aggression_multiplier": 1.25,
                    "trade_horizon_bias": "scalp",
                    "min_confluence_override": 3.2,
                    "reentry_priority": 0.75,
                },
            }
        )

        state = router._learning_pattern_state(
            symbol="USDJPY",
            strategy_key="USDJPY_MOMENTUM_IMPULSE",
            session_name="TOKYO",
        )

        self.assertGreater(float(state["aggression_multiplier"]), 1.0)
        self.assertGreater(float(state["size_bonus"]), 0.0)
        self.assertEqual(str(state["trade_horizon_bias"]), "scalp")

    def test_xau_prime_compression_burst_uses_target_window(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.82,
            source="test",
            details={"compression_proxy_state": "COMPRESSION"},
        )

        router._prime_compression_burst(  # noqa: SLF001
            symbol="XAUUSD",
            session_name="LONDON",
            regime=regime,
            weekend_mode=False,
        )

        state = router.compression_burst_state["XAUUSD|LONDON"]
        self.assertGreaterEqual(int(state["remaining"]), int(router.xau_m5_burst_target))

    def test_density_profile_boosts_usdjpy_tokyo_rank_and_entry_cap(self) -> None:
        frame = self._row()
        row = frame.iloc[-1]
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.82,
            source="test",
            details={
                "trend_flag": 1.0,
                "pressure_proxy_score": 0.72,
                "compression_proxy_state": "EXPANSION_READY",
                "compression_expansion_score": 0.50,
            },
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        candidate = SignalCandidate(
            signal_id="usdjpy-density",
            setup="USDJPY_MOMENTUM_IMPULSE",
            side="BUY",
            score_hint=0.67,
            reason="tokyo impulse",
            stop_atr=1.0,
            tp_r=1.8,
            strategy_family="TREND",
            meta={"strategy_key": "USDJPY_MOMENTUM_IMPULSE"},
        )

        baseline = StrategyRouter(max_spread_points=60.0)._enrich_candidate(
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            candidate=SignalCandidate(**candidate.__dict__),
        )
        profiled = StrategyRouter(
            max_spread_points=60.0,
            density_profiles={
                "USDJPY": {
                    "sessions": ["TOKYO"],
                    "rank_bonus": 0.04,
                    "size_bonus": 0.05,
                    "activation_score": 0.50,
                    "entry_cap": 5,
                    "compression_candidates": 5,
                    "compression_multiplier": 1.18,
                }
            },
        )._enrich_candidate(
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            candidate=SignalCandidate(**candidate.__dict__),
        )

        self.assertGreater(
            float(profiled.meta.get("router_rank_score", 0.0)),
            float(baseline.meta.get("router_rank_score", 0.0)),
        )
        self.assertTrue(bool(profiled.meta.get("density_profile_active")))
        self.assertEqual(int(profiled.meta.get("density_entry_cap", 0)), 7)

    def test_learning_pair_directive_density_bonuses_expand_entry_cap_and_compression(self) -> None:
        frame = self._row()
        row = frame.iloc[-1]
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.82,
            source="test",
            details={
                "trend_flag": 1.0,
                "pressure_proxy_score": 0.72,
                "compression_proxy_state": "EXPANSION_READY",
                "compression_expansion_score": 0.50,
            },
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        candidate = SignalCandidate(
            signal_id="usdjpy-learning-density",
            setup="USDJPY_MOMENTUM_IMPULSE",
            side="BUY",
            score_hint=0.67,
            reason="learning density bonus",
            stop_atr=1.0,
            tp_r=1.8,
            strategy_family="TREND",
            meta={"strategy_key": "USDJPY_MOMENTUM_IMPULSE"},
        )

        density_profiles = {
            "USDJPY": {
                "sessions": ["TOKYO"],
                "rank_bonus": 0.03,
                "size_bonus": 0.04,
                "activation_score": 0.50,
                "entry_cap": 4,
                "compression_candidates": 4,
                "compression_multiplier": 1.12,
            }
        }
        baseline_router = StrategyRouter(max_spread_points=60.0, density_profiles=density_profiles)
        baseline = baseline_router._enrich_candidate(
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            candidate=SignalCandidate(**candidate.__dict__),
        )

        router = StrategyRouter(max_spread_points=60.0, density_profiles=density_profiles)
        router.apply_learning_policy(
            {
                "symbol": "USDJPY",
                "pair_directive": {
                    "density_entry_cap_bonus": 2,
                    "density_compression_candidate_bonus": 3,
                    "density_compression_multiplier": 1.24,
                    "density_rank_bonus": 0.01,
                    "density_size_bonus": 0.01,
                    "density_activation_relax": 0.02,
                    "frequency_directives": {
                        "soft_burst_target_10m": 7,
                    },
                },
            }
        )
        enriched = router._enrich_candidate(
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            candidate=SignalCandidate(**candidate.__dict__),
        )

        self.assertEqual(int(baseline.meta.get("density_entry_cap", 0)), 6)
        self.assertEqual(int(enriched.meta.get("density_entry_cap", 0)), 8)
        self.assertEqual(int(baseline.meta.get("density_profile_compression_candidates", 0)), 4)
        self.assertEqual(int(enriched.meta.get("density_profile_compression_candidates", 0)), 7)
        self.assertGreater(float(enriched.meta.get("router_rank_score", 0.0)), float(baseline.meta.get("router_rank_score", 0.0)))

    def test_density_profile_can_expand_compression_burst_for_hot_symbol(self) -> None:
        router = StrategyRouter(
            max_spread_points=60.0,
            density_profiles={
                "NAS100": {
                    "sessions": ["LONDON"],
                    "compression_candidates": 6,
                    "compression_multiplier": 1.24,
                }
            },
        )
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.82,
            source="test",
            details={"compression_proxy_state": "EXPANSION_READY"},
        )

        router._prime_compression_burst(  # noqa: SLF001
            symbol="NAS100",
            session_name="LONDON",
            regime=regime,
            weekend_mode=False,
        )

        state = router.compression_burst_state["NAS100|LONDON"]
        self.assertGreaterEqual(int(state["remaining"]), 6)
        self.assertGreaterEqual(float(state["multiplier"]), 1.24)

    def test_enrich_candidate_can_extend_tp_for_quality_xau_prime_grid(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.84,
            source="test",
            details={
                "pressure_proxy_score": 0.80,
                "compression_proxy_state": "EXPANSION_READY",
                "compression_expansion_score": 0.44,
                "volatility_forecast_state": "BALANCED",
            },
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "time": datetime(2026, 3, 9, 9, 30, tzinfo=timezone.utc),
                "m5_spread": 16.0,
                "m5_structure_score": 0.82,
                "m5_atr_pct_of_avg": 1.08,
                "m5_body_efficiency": 0.74,
                "m15_range_position_20": 0.72,
                "m15_close": 2922.0,
                "m15_ema_20": 2919.0,
                "m15_atr_14": 5.2,
                "m5_ret_1": 0.0014,
                "m5_volume_ratio_20": 1.28,
            }
        )
        candidate = SignalCandidate(
            signal_id="xau-prime-grid",
            setup="XAUUSD_ADAPTIVE_M5_GRID",
            side="BUY",
            score_hint=0.71,
            reason="prime compression release",
            stop_atr=1.0,
            tp_r=1.6,
            strategy_family="GRID",
            confluence_score=4.4,
        )

        enriched = router._enrich_candidate(  # noqa: SLF001
            symbol="XAUUSD",
            row=row,
            regime=regime,
            session=session,
            candidate=candidate,
        )

        self.assertFalse(bool(enriched.meta.get("router_reject")))
        self.assertTrue(bool(enriched.meta.get("tp_extension_active")))
        self.assertGreaterEqual(float(enriched.tp_r), 1.8)

    def test_xau_prime_grid_breakout_uses_session_attack_lane_and_fast_profile(self) -> None:
        router = StrategyRouter(
            max_spread_points=60.0,
            xau_prime_session_mult=2.6,
            density_profiles={
                "XAUUSD": {
                    "sessions": ["LONDON", "OVERLAP", "NEW_YORK"],
                    "rank_bonus": 0.05,
                    "size_bonus": 0.08,
                    "activation_score": 0.50,
                    "entry_cap": 12,
                    "compression_candidates": 8,
                    "compression_multiplier": 1.30,
                }
            },
        )
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.84,
            source="test",
            details={
                "pressure_proxy_score": 0.80,
                "compression_proxy_state": "EXPANSION_READY",
                "compression_expansion_score": 0.56,
                "volatility_forecast_state": "BALANCED",
            },
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "time": datetime(2026, 3, 9, 9, 32, tzinfo=timezone.utc),
                "m5_spread": 15.0,
                "m5_structure_score": 0.84,
                "m5_atr_pct_of_avg": 1.10,
                "m5_body_efficiency": 0.78,
                "m15_range_position_20": 0.76,
                "m15_close": 2924.0,
                "m15_ema_20": 2920.0,
                "m15_atr_14": 5.0,
                "m5_ret_1": 0.0016,
                "m5_volume_ratio_20": 1.34,
            }
        )
        candidate = SignalCandidate(
            signal_id="xau-breakout-lane",
            setup="XAUUSD_ADAPTIVE_M5_GRID",
            side="BUY",
            score_hint=0.72,
            reason="prime breakout reclaim",
            stop_atr=1.0,
            tp_r=1.7,
            strategy_family="GRID",
            confluence_score=4.6,
            meta={"grid_entry_profile": "grid_breakout_reclaim_long"},
        )

        enriched = router._enrich_candidate(  # noqa: SLF001
            symbol="XAUUSD",
            row=row,
            regime=regime,
            session=session,
            candidate=candidate,
        )

        self.assertEqual(str(enriched.meta.get("lane_name") or ""), "XAU_LONDON_BREAKOUT")
        self.assertEqual(str(enriched.meta.get("xau_attack_category") or ""), "BREAKOUT")
        self.assertEqual(str(enriched.meta.get("fast_execution_profile") or ""), "M3_ATTACK")
        self.assertGreaterEqual(int(enriched.meta.get("density_entry_cap", 0) or 0), 18)

    def test_fx_session_momentum_candidate_is_generated(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.8, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="EURUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 9, 0, tzinfo=timezone.utc),
                    "m5_close": 1.1025,
                    "m5_spread": 12.0,
                    "m5_atr_14": 0.0012,
                    "m5_ret_1": 0.00025,
                    "m5_volume_ratio_20": 1.3,
                    "m5_body_efficiency": 0.68,
                    "m5_ema_20": 1.1018,
                    "m5_ema_50": 1.1012,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m5_macd_hist_slope": 0.0004,
                    "m15_close": 1.1025,
                    "m15_atr_14": 0.0012,
                    "m15_rolling_high_20": 1.1045,
                    "m15_rolling_low_20": 1.0995,
                    "h1_ema_50": 1.1010,
                    "h1_ema_200": 1.0980,
                }
            ]
        )
        row = frame.iloc[-1]
        candidates = router._session_momentum_boost(  # noqa: SLF001 - targeted routing test
            symbol="EURUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )
        self.assertTrue(any(str(item.setup).upper() == "EURUSD_SESSION_MOMENTUM" for item in candidates))

    def test_usdjpy_session_pullback_candidate_is_generated_with_clean_asia_retest(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.75, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="EURUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc),
                    "m5_close": 151.14,
                    "m5_spread": 12.0,
                    "m5_atr_14": 0.11,
                    "m5_body_efficiency": 0.68,
                    "m5_ema_20": 151.16,
                    "m5_ema_50": 151.10,
                    "m5_bullish": 0,
                    "m5_bearish": 0,
                    "m15_close": 151.14,
                    "m15_atr_14": 0.16,
                    "m15_volume_ratio_20": 1.32,
                    "m15_range_position_20": 0.58,
                    "m15_ret_1": 0.010,
                    "m15_ret_3": 0.018,
                    "m5_ret_1": 0.010,
                    "m5_lower_wick_ratio": 0.24,
                    "h1_ema_50": 151.05,
                    "h1_ema_200": 150.70,
                }
            ]
        )

        row = frame.iloc[-1]
        candidates = router._session_pullback_continuation(  # noqa: SLF001 - targeted routing test
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 9, 0, 30, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "USDJPY_SESSION_PULLBACK" for item in candidates))

    def test_eurusd_session_pullback_candidate_is_not_generated_in_tokyo(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.75, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 9, 0, 30, tzinfo=timezone.utc),
                    "m5_close": 1.0820,
                    "m5_spread": 12.0,
                    "m5_atr_14": 0.0007,
                    "m5_body_efficiency": 0.63,
                    "m5_ema_20": 1.0818,
                    "m5_ema_50": 1.0813,
                    "m15_close": 1.0820,
                    "m15_atr_14": 0.0012,
                    "m15_volume_ratio_20": 1.40,
                    "m15_range_position_20": 0.58,
                    "m15_ret_1": 0.00008,
                    "m15_ret_3": 0.00011,
                    "h1_ema_50": 1.0814,
                    "h1_ema_200": 1.0806,
                }
            ]
        )
        candidates = router._session_pullback_continuation(  # noqa: SLF001
            symbol="EURUSD",
            row=frame.iloc[-1],
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 9, 0, 30, tzinfo=timezone.utc),
        )
        self.assertFalse(candidates)

    def test_fx_session_pullback_candidate_is_not_generated_in_ranging_regime(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.78, source="test", details={"trend_flag": 0.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 9, 0, 30, tzinfo=timezone.utc),
                    "m5_close": 1.0820,
                    "m5_spread": 12.0,
                    "m5_atr_14": 0.0007,
                    "m5_body_efficiency": 0.62,
                    "m5_ema_20": 1.0818,
                    "m5_ema_50": 1.0813,
                    "m5_ret_1": 0.00002,
                    "m15_close": 1.0820,
                    "m15_atr_14": 0.0012,
                    "m15_volume_ratio_20": 1.30,
                    "m15_range_position_20": 0.62,
                    "m15_ret_1": 0.00003,
                    "m15_ret_3": 0.00004,
                    "h1_ema_50": 1.0814,
                    "h1_ema_200": 1.0806,
                }
            ]
        )

        row = frame.iloc[-1]
        candidates = router._session_pullback_continuation(  # noqa: SLF001
            symbol="EURUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 9, 0, 30, tzinfo=timezone.utc),
        )

        self.assertFalse(any(str(item.setup).upper() == "EURUSD_SESSION_PULLBACK" for item in candidates))

    def test_forex_breakout_retest_is_not_generated_for_eurusd_in_tokyo_chop(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="LOW_LIQUIDITY_CHOP",
            confidence=0.82,
            source="test",
            details={"trend_flag": 0.0, "wick_density": 0.82},
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 13, 0, 35, tzinfo=timezone.utc),
                    "m15_close": 1.0815,
                    "m15_spread": 12.0,
                    "m15_atr_14": 0.0010,
                    "m15_volume_ratio_20": 1.22,
                    "m15_range_position_20": 0.58,
                    "m15_rolling_high_prev_20": 1.0812,
                    "m15_rolling_low_prev_20": 1.0790,
                    "m15_body_efficiency": 0.68,
                    "m15_bullish": 1,
                    "m15_bearish": 0,
                    "h1_ema_50": 1.0808,
                    "h1_ema_200": 1.0798,
                }
            ]
        )

        row = frame.iloc[-1]
        candidates = router._forex_breakout_retest(  # noqa: SLF001
            symbol="EURUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )

        self.assertFalse(candidates)

    def test_usdjpy_forex_breakout_retest_is_not_generated_in_sydney_ranging(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.82,
            source="test",
            details={"trend_flag": 0.0, "range_flag": 1.0},
        )
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=1,
            ai_threshold_offset=0.02,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 13, 19, 15, tzinfo=timezone.utc),
                    "m15_close": 151.42,
                    "m5_close": 151.42,
                    "m5_spread": 10.0,
                    "m15_atr_14": 0.18,
                    "m15_volume_ratio_20": 1.18,
                    "m15_range_position_20": 0.60,
                    "m15_rolling_high_prev_20": 151.30,
                    "m15_rolling_low_prev_20": 151.00,
                    "m5_body_efficiency": 0.70,
                    "m5_bullish": 1,
                    "h1_ema_50": 151.20,
                    "h1_ema_200": 151.00,
                }
            ]
        )

        candidates = router._forex_breakout_retest(  # noqa: SLF001
            symbol="USDJPY",
            row=frame.iloc[-1],
            regime=regime,
            session=session,
            timestamp=frame.iloc[-1]["time"],
        )

        self.assertFalse(candidates)

    def test_audnzd_forex_breakout_retest_is_not_generated_in_tokyo_mean_reversion(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="MEAN_REVERSION",
            confidence=0.80,
            source="test",
            details={"trend_flag": 0.0, "range_flag": 1.0},
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 13, 0, 40, tzinfo=timezone.utc),
                    "m15_close": 1.0742,
                    "m15_spread": 10.0,
                    "m15_atr_14": 0.0008,
                    "m15_volume_ratio_20": 1.30,
                    "m15_range_position_20": 0.62,
                    "m15_rolling_high_prev_20": 1.0740,
                    "m15_rolling_low_prev_20": 1.0725,
                    "m15_body_efficiency": 0.72,
                    "m15_bullish": 1,
                    "m15_bearish": 0,
                    "h1_ema_50": 1.0738,
                    "h1_ema_200": 1.0731,
                }
            ]
        )

        row = frame.iloc[-1]
        candidates = router._forex_breakout_retest(  # noqa: SLF001
            symbol="AUDNZD",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )

        self.assertFalse(candidates)

    def test_forex_trend_is_not_generated_for_eurusd_in_tokyo(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.80, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 1.1018,
                "m5_spread": 12.0,
                "m5_atr_14": 0.0010,
                "m5_ema_20": 1.1015,
                "m5_ema_50": 1.1010,
                "m5_bullish": 1,
                "m5_macd_hist_slope": 0.0004,
                "m15_rolling_high_20": 1.1035,
                "m15_rolling_low_20": 1.0990,
                "h1_ema_50": 1.1010,
                "h1_ema_200": 1.0990,
            }
        )
        candidates = router._forex_trend(  # noqa: SLF001
            symbol="EURUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 12, 0, 10, tzinfo=timezone.utc),
        )
        self.assertFalse(candidates)

    def test_usdjpy_forex_trend_is_not_generated_in_sydney_ranging(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.80,
            source="test",
            details={"trend_flag": 0.0},
        )
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 9, 21, 30, tzinfo=timezone.utc),
                    "m5_close": 151.14,
                    "m5_spread": 10.0,
                    "m5_atr_14": 0.10,
                    "m5_ema_20": 151.12,
                    "m5_ema_50": 151.06,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m5_macd_hist_slope": 0.01,
                    "m5_body_efficiency": 0.66,
                    "m5_volume_ratio_20": 1.20,
                    "m15_close": 151.14,
                    "m15_range_position_20": 0.55,
                    "h1_ema_50": 151.10,
                    "h1_ema_200": 150.90,
                }
            ]
        )

        candidates = router._forex_trend(  # noqa: SLF001
            symbol="USDJPY",
            row=frame.iloc[-1],
            regime=regime,
            session=session,
            timestamp=frame.iloc[-1]["time"],
        )
        self.assertFalse(candidates)

    def test_xau_m15_structured_is_not_generated_in_tokyo(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.8, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_spread": 18.0,
                "m15_atr_14": 4.5,
                "m15_close": 3010.0,
                "m15_high": 3012.0,
                "m15_low": 3005.0,
                "m15_rolling_high_prev_20": 3006.0,
                "m15_rolling_low_prev_20": 2994.0,
                "m15_bullish": 1,
                "m15_volume_ratio_20": 1.1,
            }
        )
        candidates = router._xau_m15_structured(  # noqa: SLF001
            symbol="XAUUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 12, 0, 10, tzinfo=timezone.utc),
        )
        self.assertFalse(candidates)

    def test_oil_inventory_scalper_can_generate_tokyo_breakout_when_clean(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.82, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_spread": 18.0,
                "m5_atr_pct_of_avg": 1.2,
                "m15_atr_14": 0.55,
                "m15_close": 72.10,
                "m15_rolling_high_prev_20": 71.80,
                "m15_rolling_low_prev_20": 70.90,
                "m5_volume_ratio_20": 1.16,
                "m5_body_efficiency": 0.67,
                "m15_range_position_20": 0.72,
                "m5_ret_1": 0.09,
                "m5_bullish": 1,
            }
        )
        candidates = router._oil_inventory_scalper(  # noqa: SLF001
            symbol="USOIL",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 12, 0, 10, tzinfo=timezone.utc),
        )
        self.assertTrue(candidates)

    def test_strategy_key_mapping_preserves_audnzd_rotation_and_eurusd_pullback_identity(self) -> None:
        self.assertEqual(resolve_strategy_key("AUDNZD", "AUDNZD_ASIA_ROTATION_PULLBACK"), "AUDNZD_RANGE_ROTATION")
        self.assertEqual(resolve_strategy_key("EURUSD", "FOREX_TREND_PULLBACK"), "EURUSD_VWAP_PULLBACK")

    def test_usdjpy_tokyo_pullback_rejects_weak_wick_and_overextended_reclaim(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.82, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=0.75,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 0, 45, tzinfo=timezone.utc),
                    "m5_close": 150.22,
                    "m5_spread": 9.0,
                    "m5_ema_20": 150.10,
                    "m5_ema_50": 149.96,
                    "m5_body_efficiency": 0.66,
                    "m5_lower_wick_ratio": 0.05,
                    "m15_close": 150.22,
                    "m15_atr_14": 0.85,
                    "m15_volume_ratio_20": 1.30,
                    "m15_range_position_20": 0.66,
                    "m15_ret_1": 0.06,
                    "m15_ret_3": 0.25,
                    "h1_ema_50": 150.04,
                    "h1_ema_200": 149.70,
                }
            ]
        )

        row = frame.iloc[-1]
        candidates = router._session_pullback_continuation(  # noqa: SLF001 - targeted routing test
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 10, 0, 45, tzinfo=timezone.utc),
        )

        self.assertFalse(any(str(item.setup).upper() == "USDJPY_SESSION_PULLBACK" for item in candidates))

    def test_usdjpy_liquidity_sweep_reclaim_is_generated_in_sydney_ranging(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.80,
            source="test",
            details={"range_flag": 1.0, "trend_flag": 0.0},
        )
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=1,
            ai_threshold_offset=0.02,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m15_close": 151.12,
                "m5_close": 151.12,
                "m15_high": 151.28,
                "m15_low": 150.95,
                "m5_spread": 10.0,
                "m15_atr_14": 0.18,
                "m5_atr_14": 0.12,
                "m15_volume_ratio_20": 1.04,
                "m5_body_efficiency": 0.63,
                "m5_lower_wick_ratio": 0.24,
                "m5_upper_wick_ratio": 0.10,
                "m15_range_position_20": 0.22,
                "m15_rolling_high_prev_20": 151.34,
                "m15_rolling_low_prev_20": 151.00,
                "m5_pinbar_bull": 1,
                "m15_engulf_bull": 0,
            }
        )

        candidates = router._liquidity_sweep_reclaim(  # noqa: SLF001
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 13, 19, 20, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "USDJPY_SWEEP_RECLAIM" for item in candidates))

    def test_usoil_session_paths_are_blocked_in_tokyo_low_liquidity_chop(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="LOW_LIQUIDITY_CHOP",
            confidence=0.84,
            source="test",
            details={"trend_flag": 0.0, "wick_density": 0.80},
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 13, 1, 10, tzinfo=timezone.utc),
                    "m5_close": 69.22,
                    "m5_spread": 18.0,
                    "m5_atr_14": 0.21,
                    "m5_body_efficiency": 0.61,
                    "m5_ema_20": 69.15,
                    "m5_ema_50": 69.10,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m5_ret_1": 0.03,
                    "m15_close": 69.22,
                    "m15_atr_14": 0.38,
                    "m15_volume_ratio_20": 1.28,
                    "m15_range_position_20": 0.64,
                    "m15_ret_1": 0.09,
                    "m15_ret_3": 0.15,
                    "h1_ema_50": 69.12,
                    "h1_ema_200": 68.80,
                }
            ]
        )

        row = frame.iloc[-1]
        pullback_candidates = router._session_pullback_continuation(  # noqa: SLF001
            symbol="USOIL",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )
        momentum_candidates = router._session_momentum_boost(  # noqa: SLF001
            symbol="USOIL",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )

        self.assertFalse(pullback_candidates)
        self.assertFalse(momentum_candidates)

    def test_forex_range_candidate_requires_rejection_signal_for_eurusd_in_asia(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.8, source="test", details={"range_flag": 1.0})
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.6,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="EURUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 9, 23, 15, tzinfo=timezone.utc),
                    "m5_close": 1.16210,
                    "m5_spread": 12.0,
                    "m5_rsi_14": 62.0,
                    "m5_volume_ratio_20": 1.0,
                    "m5_pinbar_bull": 0,
                    "m5_pinbar_bear": 0,
                    "m5_engulf_bull": 0,
                    "m5_engulf_bear": 0,
                    "m15_close": 1.16210,
                    "m15_atr_14": 0.00120,
                    "m15_range_position_20": 0.86,
                    "m15_rolling_high_prev_20": 1.16235,
                    "m15_rolling_low_prev_20": 1.15990,
                }
            ]
        )

        row = frame.iloc[-1]
        candidates = router._forex_range(  # noqa: SLF001 - targeted routing test
            symbol="EURUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 9, 23, 15, tzinfo=timezone.utc),
        )

        self.assertFalse(any(str(item.setup).upper() == "FOREX_RANGE_REVERSION" for item in candidates))

    def test_forex_range_candidate_rejects_eurusd_asia_when_atr_is_too_hot(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.82, source="test", details={"range_flag": 1.0})
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="EURUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 9, 5, tzinfo=timezone.utc),
                    "m5_close": 1.0832,
                    "m5_spread": 9.0,
                    "m5_rsi_14": 33.0,
                    "m5_volume_ratio_20": 1.18,
                    "m5_pinbar_bull": 1,
                    "m5_pinbar_bear": 0,
                    "m5_engulf_bull": 0,
                    "m5_engulf_bear": 0,
                    "m5_body_efficiency": 0.49,
                    "m5_lower_wick_ratio": 0.29,
                    "m15_close": 1.0832,
                    "m15_atr_14": 0.0011,
                    "m15_atr_pct_of_avg": 0.98,
                    "m15_rolling_high_prev_20": 1.0860,
                    "m15_rolling_low_prev_20": 1.0830,
                    "m15_range_position_20": 0.08,
                }
            ]
        )

        candidates = router.generate(
            symbol="EURUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 9, 5, tzinfo=timezone.utc),
        )

        self.assertFalse(any(str(item.setup).upper() == "FOREX_RANGE_REVERSION" for item in candidates))

    def test_forex_range_candidate_is_not_generated_for_audnzd(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.8, source="test", details={"range_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.7,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDNZD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 1, 15, tzinfo=timezone.utc),
                    "m5_close": 1.0801,
                    "m5_spread": 8.0,
                    "m5_rsi_14": 61.0,
                    "m5_volume_ratio_20": 1.0,
                    "m15_close": 1.0801,
                    "m15_atr_14": 0.0012,
                    "m15_range_position_20": 0.87,
                    "m15_rolling_high_prev_20": 1.0803,
                    "m15_rolling_low_prev_20": 1.0782,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDNZD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 1, 15, tzinfo=timezone.utc),
        )

        self.assertFalse(any(str(item.setup).upper() == "FOREX_RANGE_REVERSION" for item in candidates))

    def test_forex_range_candidate_is_not_generated_for_eurjpy_tokyo_mean_reversion(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="MEAN_REVERSION", confidence=0.82, source="test", details={"range_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.7,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="EURJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 1, 15, tzinfo=timezone.utc),
                    "m5_close": 162.40,
                    "m5_spread": 12.0,
                    "m5_rsi_14": 60.0,
                    "m5_volume_ratio_20": 1.02,
                    "m15_close": 162.40,
                    "m15_atr_14": 0.18,
                    "m15_range_position_20": 0.88,
                    "m15_rolling_high_prev_20": 162.48,
                    "m15_rolling_low_prev_20": 161.90,
                }
            ]
        )

        candidates = router.generate(
            symbol="EURJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 1, 15, tzinfo=timezone.utc),
        )

        self.assertFalse(any(str(item.setup).upper() == "FOREX_RANGE_REVERSION" for item in candidates))

    def test_liquidity_sweep_reclaim_is_not_generated_for_eurusd_tokyo_chop(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="LOW_LIQUIDITY_CHOP",
            confidence=0.84,
            source="test",
            details={"chop_flag": 1.0},
            state_label="LOW_LIQUIDITY_CHOP",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 1.0816,
                "m5_spread": 11.0,
                "m5_rsi_14": 47.0,
                "m5_body_efficiency": 0.58,
                "m5_bullish": 1,
                "m5_bearish": 0,
                "m15_close": 1.0816,
                "m15_high": 1.0820,
                "m15_low": 1.0808,
                "m15_atr_14": 0.0009,
                "m15_atr_pct_of_avg": 0.86,
                "m15_rolling_high_prev_20": 1.0825,
                "m15_rolling_low_prev_20": 1.0805,
                "m15_range_position_20": 0.26,
                "m15_volume_ratio_20": 0.91,
            }
        )

        candidates = router._liquidity_sweep_reclaim(  # noqa: SLF001
            symbol="EURUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 10, 0, 15, tzinfo=timezone.utc),
        )

        self.assertEqual(candidates, [])

    def test_liquidity_sweep_reclaim_is_not_generated_for_gbpjpy_tokyo_chop(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="LOW_LIQUIDITY_CHOP",
            confidence=0.86,
            source="test",
            details={"chop_flag": 1.0},
            state_label="LOW_LIQUIDITY_CHOP",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 191.26,
                "m5_spread": 18.0,
                "m5_rsi_14": 49.0,
                "m5_body_efficiency": 0.54,
                "m5_bullish": 1,
                "m5_bearish": 0,
                "m15_close": 191.26,
                "m15_high": 191.42,
                "m15_low": 191.02,
                "m15_atr_14": 0.24,
                "m15_atr_pct_of_avg": 0.82,
                "m15_rolling_high_prev_20": 191.50,
                "m15_rolling_low_prev_20": 190.90,
                "m15_range_position_20": 0.30,
                "m15_volume_ratio_20": 0.88,
            }
        )

        candidates = router._liquidity_sweep_reclaim(  # noqa: SLF001
            symbol="GBPJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 10, 0, 15, tzinfo=timezone.utc),
        )

        self.assertEqual(candidates, [])

    def test_router_uses_current_time_override_for_windowed_setup_generation(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.8, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="EURUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(1970, 1, 1, 0, 0, tzinfo=timezone.utc),
                    "m5_close": 1.0820,
                    "m5_spread": 12.0,
                    "m5_atr_14": 0.0007,
                    "m5_volume_ratio_20": 1.05,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_close": 1.0820,
                    "m15_atr_14": 0.0010,
                    "dxy_ret_5": -0.0015,
                }
            ]
        )

        without_override = router.generate(
            symbol="EURUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
        )
        with_override = router.generate(
            symbol="EURUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 6, 10, 50, tzinfo=timezone.utc),
        )

        self.assertFalse(any(str(item.setup).upper() == "EURUSD_FIX_FLOW" for item in without_override))
        self.assertTrue(any(str(item.setup).upper() == "EURUSD_FIX_FLOW" for item in with_override))

    def test_btc_candidate_is_generated_in_allowed_ny_window(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.7, source="test", details={"trend_flag": 0.0})
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="VERY_HIGH",
            size_multiplier=1.0,
            confluence_delta=-1,
            ai_threshold_offset=-0.01,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc),
                    "m5_close": 68000.0,
                    "m5_spread": 1700.0,
                    "m5_spread_ratio_20": 1.2,
                    "m5_atr_14": 420.0,
                    "m5_atr_pct_of_avg": 1.15,
                    "m5_ret_1": 0.004,
                    "m5_volume_ratio_20": 1.18,
                    "m5_body_efficiency": 0.62,
                    "m5_ema_20": 67910.0,
                    "m5_ema_50": 67780.0,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_rolling_high_prev_20": 67950.0,
                    "m15_rolling_low_prev_20": 67100.0,
                    "m15_atr_14": 620.0,
                    "h1_ema_20": 67820.0,
                    "h1_ema_50": 67610.0,
                    "h4_ema_50": 67550.0,
                    "h4_ema_200": 66000.0,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
        )

        self.assertTrue(
            any(
                str(item.setup).upper() == "BTC_TREND_SCALP"
                or str(item.meta.get("strategy_key", "")).upper() == "BTCUSD_TREND_SCALP"
                for item in candidates
            )
        )

    def test_btc_weekend_heartbeat_fallback_emits_when_router_is_otherwise_empty(self) -> None:
        router = StrategyRouter(max_spread_points=60.0, btc_heartbeat_cadence_seconds=120)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.72,
            source="test",
            details={"trend_flag": 1.0},
            state_label="LOW_LIQUIDITY_DRIFT",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=0,
            ai_threshold_offset=0.01,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 28, 5, 33, tzinfo=timezone.utc),
                    "m5_close": 66314.90625,
                    "m5_spread": 1702.0,
                    "m5_spread_ratio_20": 1.0,
                    "m5_atr_14": 39.40374914842037,
                    "m5_atr_pct_of_avg": 0.9142031453524021,
                    "m5_ret_1": 0.000345889332148408,
                    "m15_ret_1": 0.000345889332148408,
                    "m5_volume_ratio_20": 1.0,
                    "m1_volume_ratio_20": 1.0,
                    "m5_body_efficiency": 0.0,
                    "m5_ema_20": 66224.27219511225,
                    "m5_ema_50": 66180.61324550923,
                    "h1_ema_20": 66474.34701849354,
                    "h1_ema_50": 67566.69830777419,
                    "m15_range_position_20": 0.8600812638038878,
                    "seasonality_edge_score": 0.32,
                    "market_instability_score": 0.31548457642400657,
                    "m5_rsi_14": 64.14698421265012,
                    "m5_bullish": 0,
                    "m5_bearish": 0,
                    "m1_atr_pct_of_avg": 0.9142031453524021,
                    "m1_body": 0.0,
                    "m1_atr_14": 10.0,
                    "m5_breakout_flag": 0,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 28, 5, 33, tzinfo=timezone.utc),
        )
        self.assertTrue(candidates)
        self.assertTrue(bool(candidates[0].meta.get("proxyless_weekend_heartbeat_mode", False)))
        self.assertIn(str(candidates[0].setup).upper(), {"BTC_TOKYO_DRIFT_SCALP", "BTC_LONDON_IMPULSE_SCALP", "BTC_NY_LIQUIDITY"})

        repeated = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 28, 5, 34, tzinfo=timezone.utc),
        )
        self.assertFalse(repeated)

    def test_btc_weekend_heartbeat_fallback_survives_elevated_weekend_spread_and_volatility(self) -> None:
        router = StrategyRouter(max_spread_points=60.0, btc_heartbeat_cadence_seconds=120)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.64,
            source="test",
            details={"trend_flag": 0.0},
            state_label="NEWS_DISTORTION",
        )
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=0,
            ai_threshold_offset=0.01,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 28, 6, 0, tzinfo=timezone.utc),
                    "m5_close": 66475.0,
                    "m5_spread": 1707.0,
                    "m5_spread_ratio_20": 4.4,
                    "m5_atr_14": 39.5,
                    "m5_atr_pct_of_avg": 3.2,
                    "m5_ret_1": 0.00018,
                    "m15_ret_1": 0.00022,
                    "m5_volume_ratio_20": 0.92,
                    "m1_volume_ratio_20": 0.90,
                    "m5_body_efficiency": 0.12,
                    "m5_ema_20": 66420.0,
                    "m5_ema_50": 66380.0,
                    "h1_ema_20": 66480.0,
                    "h1_ema_50": 66360.0,
                    "m15_range_position_20": 0.58,
                    "seasonality_edge_score": 0.41,
                    "market_instability_score": 0.44,
                    "m5_rsi_14": 57.0,
                    "m5_bullish": 0,
                    "m5_bearish": 0,
                    "m1_atr_pct_of_avg": 3.2,
                    "m1_body": 0.0,
                    "m1_atr_14": 10.0,
                    "m5_breakout_flag": 0,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 28, 6, 0, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertTrue(any(bool(item.meta.get("proxyless_weekend_heartbeat_mode", False)) for item in candidates))

    def test_btc_weekend_heartbeat_fallback_can_emit_sell_on_high_rejection(self) -> None:
        router = StrategyRouter(max_spread_points=60.0, btc_heartbeat_cadence_seconds=120)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.66,
            source="test",
            details={"trend_flag": 0.0},
            state_label="LOW_LIQUIDITY_DRIFT",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=0,
            ai_threshold_offset=0.01,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 28, 7, 10, tzinfo=timezone.utc),
                    "m5_close": 66510.0,
                    "m5_spread": 1702.0,
                    "m5_spread_ratio_20": 1.0,
                    "m5_atr_14": 40.0,
                    "m5_atr_pct_of_avg": 1.05,
                    "m5_ret_1": -0.00012,
                    "m15_ret_1": -0.00006,
                    "m5_volume_ratio_20": 1.0,
                    "m1_volume_ratio_20": 1.0,
                    "m5_body_efficiency": 0.12,
                    "m5_ema_20": 66480.0,
                    "m5_ema_50": 66460.0,
                    "h1_ema_20": 66410.0,
                    "h1_ema_50": 66405.0,
                    "m15_range_position_20": 0.86,
                    "seasonality_edge_score": 0.34,
                    "market_instability_score": 0.28,
                    "m5_rsi_14": 63.0,
                    "m5_upper_wick_ratio": 0.44,
                    "m5_lower_wick_ratio": 0.04,
                    "m5_bullish": 0,
                    "m5_bearish": 0,
                    "m1_atr_pct_of_avg": 1.05,
                    "m1_body": 0.0,
                    "m1_atr_14": 10.0,
                    "m5_breakout_flag": 0,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 28, 7, 10, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        heartbeat = next(item for item in candidates if bool(item.meta.get("proxyless_weekend_heartbeat_mode", False)))
        self.assertEqual(heartbeat.side, "SELL")

    def test_audjpy_tokyo_momentum_breakout_candidate_is_generated(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.82,
            source="test",
            details={"trend_flag": 1.0},
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=0.75,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 0, 30, tzinfo=timezone.utc),
                    "m5_close": 95.48,
                    "m5_spread": 10.0,
                    "m5_body_efficiency": 0.74,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_close": 95.48,
                    "m15_atr_14": 0.80,
                    "m15_ret_1": 0.0035,
                    "m15_range_position_20": 0.82,
                    "m15_rolling_high_prev_20": 95.10,
                    "m15_rolling_low_prev_20": 94.30,
                    "m15_volume_ratio_20": 1.12,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 0, 30, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "AUDJPY_TOKYO_MOMENTUM_BREAKOUT" for item in candidates))

    def test_nzdjpy_tokyo_pullback_candidate_is_generated(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.78, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="NZDJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 21, 15, tzinfo=timezone.utc),
                    "m5_close": 88.25,
                    "m5_spread": 9.0,
                    "m5_body_efficiency": 0.70,
                    "m15_close": 88.25,
                    "m15_atr_14": 0.42,
                    "m15_ema_20": 88.18,
                    "m15_ema_50": 88.10,
                    "m15_volume_ratio_20": 1.12,
                    "m15_range_position_20": 0.60,
                }
            ]
        )

        candidates = router.generate(
            symbol="NZDJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 21, 15, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "NZDJPY_TOKYO_CONTINUATION_PULLBACK" for item in candidates))

    def test_audjpy_does_not_emit_generic_session_pullback_when_native_pool_exists(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.80, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=0.75,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 0, 45, tzinfo=timezone.utc),
                    "m5_close": 95.34,
                    "m5_spread": 10.0,
                    "m5_atr_14": 0.48,
                    "m5_ema_20": 95.20,
                    "m5_ema_50": 95.05,
                    "m5_body_efficiency": 0.65,
                    "m5_bullish": 1,
                    "m15_close": 95.34,
                    "m15_atr_14": 0.52,
                    "m15_volume_ratio_20": 1.08,
                    "m15_range_position_20": 0.62,
                    "h1_ema_50": 95.10,
                    "h1_ema_200": 94.80,
                }
            ]
        )
        candidates = router.generate(
            symbol="AUDJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 0, 45, tzinfo=timezone.utc),
        )
        self.assertFalse(any(str(item.setup).upper() == "AUDJPY_SESSION_PULLBACK" for item in candidates))
        self.assertTrue(any(str(item.setup).upper() == "AUDJPY_TOKYO_CONTINUATION_PULLBACK" for item in candidates))

    def test_audnzd_asia_rotation_is_structured_daytrade_not_scalp(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.76, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDNZD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 1, 45, tzinfo=timezone.utc),
                    "m5_close": 1.0814,
                    "m5_spread": 8.0,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_close": 1.0814,
                    "m15_atr_14": 0.0010,
                    "m15_rolling_high_prev_20": 1.0810,
                    "m15_rolling_low_prev_20": 1.0795,
                    "m15_volume_ratio_20": 1.08,
                    "m15_range_position_20": 0.66,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDNZD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 1, 45, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "AUDNZD_ASIA_ROTATION_BREAKOUT" for item in candidates))
        audnzd = next(item for item in candidates if str(item.setup).upper() == "AUDNZD_ASIA_ROTATION_BREAKOUT")
        self.assertEqual(str(audnzd.entry_kind).upper(), "DAYTRADE")

    def test_audjpy_sydney_range_break_candidate_is_generated(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="COMPRESSION", confidence=0.78, source="test", details={"trend_flag": 0.2})
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 21, 30, tzinfo=timezone.utc),
                    "m5_close": 95.48,
                    "m5_spread": 10.0,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_close": 95.48,
                    "m15_atr_14": 0.70,
                    "m15_atr_pct_of_avg": 0.88,
                    "m15_rolling_high_prev_20": 95.35,
                    "m15_rolling_low_prev_20": 94.70,
                    "m15_volume_ratio_20": 1.00,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 21, 30, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "AUDJPY_SYDNEY_RANGE_BREAK" for item in candidates))

    def test_audjpy_asia_rotation_reclaim_candidate_is_generated(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.77,
            source="test",
            details={"range_flag": 1.0},
            state_label="RANGING",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.72,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 23, 30, tzinfo=timezone.utc),
                    "m5_close": 95.18,
                    "m5_spread": 8.0,
                    "m5_rsi_14": 44.0,
                    "m5_body_efficiency": 0.66,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_close": 95.18,
                    "m15_high": 95.22,
                    "m15_low": 94.74,
                    "m15_atr_14": 0.40,
                    "m15_atr_pct_of_avg": 0.96,
                    "m15_rolling_high_prev_20": 95.36,
                    "m15_rolling_low_prev_20": 94.78,
                    "m15_range_position_20": 0.18,
                    "m15_volume_ratio_20": 0.96,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 23, 30, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "AUDJPY_SWEEP_RECLAIM" for item in candidates))

    def test_audjpy_asia_rotation_reclaim_is_generated_in_tokyo_mean_reversion_when_reclaim_is_clean(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="MEAN_REVERSION",
            confidence=0.80,
            source="test",
            details={"reversion_flag": 1.0},
            state_label="MEAN_REVERSION",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.72,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 23, 30, tzinfo=timezone.utc),
                    "m5_close": 95.18,
                    "m5_spread": 8.0,
                    "m5_rsi_14": 44.0,
                    "m5_body_efficiency": 0.66,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_close": 95.18,
                    "m15_high": 95.22,
                    "m15_low": 94.74,
                    "m15_atr_14": 0.40,
                    "m15_atr_pct_of_avg": 0.96,
                    "m15_rolling_high_prev_20": 95.36,
                    "m15_rolling_low_prev_20": 94.78,
                    "m15_range_position_20": 0.18,
                    "m15_volume_ratio_20": 0.96,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 23, 30, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "AUDJPY_SWEEP_RECLAIM" for item in candidates))

    def test_usdjpy_asia_session_momentum_is_disabled(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.79, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=0.75,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 149.40,
                "m5_spread": 9.0,
                "m5_atr_14": 0.22,
                "m5_ret_1": 0.04,
                "m5_volume_ratio_20": 1.18,
                "m5_body_efficiency": 0.70,
                "m5_ema_20": 149.32,
                "m5_ema_50": 149.10,
                "m5_bullish": 1,
                "m5_bearish": 0,
            }
        )
        candidates = router._session_momentum_boost(  # noqa: SLF001
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 10, 1, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(candidates, [])

    def test_generate_recovers_audjpy_live_candidate_when_primary_generators_are_empty(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.81,
            source="test",
            details={"trend_flag": 1.0},
            state_label="LOW_LIQUIDITY_DRIFT",
        )
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
                    "m5_close": 111.91,
                    "m5_spread": 0.0,
                    "m5_atr_14": 0.080,
                    "m5_ema_20": 112.03,
                    "m5_ema_50": 112.11,
                    "m5_ret_1": 0.00028,
                    "m5_volume_ratio_20": 1.0,
                    "m15_close": 111.885,
                    "m15_atr_14": 0.143,
                    "m15_ema_20": 112.11,
                    "m15_ema_50": 112.24,
                    "h1_ema_50": 112.51,
                    "h1_ema_200": 112.28,
                    "m15_ret_1": -0.00115,
                    "m15_ret_3": -0.00081,
                    "m15_volume_ratio_20": 1.0,
                    "m5_body_efficiency": 0.42,
                    "m15_body_efficiency": 0.95,
                    "m15_range_position_20": 0.187,
                    "m5_range_position_20": 0.18,
                    "multi_tf_alignment_score": 0.75,
                    "fractal_persistence_score": 0.29,
                    "seasonality_edge_score": 0.805,
                    "market_instability_score": 0.24,
                    "feature_drift_score": 0.04,
                    "m5_tick_volume": 0.0,
                    "m15_tick_volume": 0.0,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 19, 12, 0, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertTrue(any(bool(item.meta.get("fallback_live_recovery")) for item in candidates))
        self.assertIn(str(candidates[0].setup), {"AUDJPY_ASIA_MOMENTUM_BREAKOUT", "AUDJPY_ASIA_CONTINUATION_PULLBACK"})

    def test_live_recovery_emits_eurusd_range_candidate_on_sparse_tick_volume(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.77,
            source="test",
            details={"range_flag": 1.0},
            state_label="LOW_LIQUIDITY_DRIFT",
        )
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 1.14784,
                "m15_close": 1.14810,
                "m5_spread": 0.0,
                "m5_atr_14": 0.00044,
                "m15_atr_14": 0.00085,
                "m5_ema_20": 1.14764,
                "m5_ema_50": 1.14735,
                "m15_ema_20": 1.14732,
                "m15_ema_50": 1.14751,
                "h1_ema_50": 1.14960,
                "h1_ema_200": 1.15387,
                "m15_ret_1": -0.00011,
                "m15_ret_3": 0.00057,
                "m5_volume_ratio_20": 1.0,
                "m15_volume_ratio_20": 1.0,
                "m5_body_efficiency": 0.75,
                "m15_body_efficiency": 0.25,
                "m15_range_position_20": 0.81,
                "multi_tf_alignment_score": 0.25,
                "fractal_persistence_score": 0.26,
                "seasonality_edge_score": 0.805,
                "market_instability_score": 0.41,
                "feature_drift_score": 0.07,
                "m5_tick_volume": 0.0,
                "m15_tick_volume": 0.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="EURUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 19, 12, 5, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "EURUSD_RANGE_REVERSION")
        self.assertTrue(bool(candidates[0].meta.get("fallback_live_recovery")))

    def test_live_recovery_emits_nas100_vwap_candidate_when_router_is_empty(self) -> None:
        router = StrategyRouter(max_spread_points=95.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.80,
            source="test",
            details={"trend_flag": 1.0},
            state_label="LOW_LIQUIDITY_DRIFT",
        )
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 24537.75,
                "m15_close": 24535.0,
                "m5_spread": 0.0,
                "m5_atr_14": 24.46,
                "m15_atr_14": 46.75,
                "m5_ema_20": 24552.94,
                "m5_ema_50": 24570.05,
                "m15_ema_20": 24571.67,
                "m15_ema_50": 24620.24,
                "h1_ema_50": 24776.74,
                "h1_ema_200": 24804.30,
                "m15_ret_1": 0.00087,
                "m15_ret_3": -0.00197,
                "m5_volume_ratio_20": 0.28,
                "m15_volume_ratio_20": 0.72,
                "m5_body_efficiency": 0.34,
                "m15_body_efficiency": 0.62,
                "m15_range_position_20": 0.23,
                "multi_tf_alignment_score": 0.75,
                "fractal_persistence_score": 0.23,
                "seasonality_edge_score": 0.805,
                "market_instability_score": 0.34,
                "feature_drift_score": 0.23,
                "m5_tick_volume": 245.0,
                "m15_tick_volume": 2692.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="NAS100",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 19, 12, 10, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "NAS100_VWAP_PULLBACK")
        self.assertEqual(str(candidates[0].meta.get("strategy_key") or ""), "NAS100_VWAP_TREND_STRATEGY")

    def test_live_recovery_emits_usdjpy_candidate_for_sparse_tick_pullback(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.78,
            source="test",
            details={"trend_flag": 1.0},
            state_label="LIQUIDITY_SWEEP",
        )
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 159.09,
                "m15_close": 159.12,
                "m5_spread": 0.0,
                "m5_atr_14": 0.08,
                "m15_atr_14": 0.14,
                "m5_ema_20": 159.12,
                "m5_ema_50": 159.18,
                "m15_ema_20": 159.21,
                "m15_ema_50": 159.36,
                "h1_ema_50": 159.31,
                "h1_ema_200": 158.80,
                "m15_ret_1": -0.00032,
                "m15_ret_3": 0.00076,
                "m5_volume_ratio_20": 1.0,
                "m15_volume_ratio_20": 1.0,
                "m5_body_efficiency": 0.134,
                "m15_body_efficiency": 0.068,
                "m15_range_position_20": 0.413,
                "multi_tf_alignment_score": 0.75,
                "fractal_persistence_score": 0.223,
                "seasonality_edge_score": 0.805,
                "market_instability_score": 0.34,
                "feature_drift_score": 0.25,
                "m5_tick_volume": 0.0,
                "m15_tick_volume": 0.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 19, 12, 10, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertTrue(any(bool(item.meta.get("fallback_live_recovery")) for item in candidates))
        self.assertIn(str(candidates[0].setup), {"USDJPY_SESSION_PULLBACK", "USDJPY_SESSION_MOMENTUM"})

    def test_live_recovery_emits_usdjpy_candidate_on_edge_row_with_low_alignment(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.76,
            source="test",
            details={"trend_flag": 1.0},
            state_label="LOW_LIQUIDITY_DRIFT",
        )
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 159.065,
                "m15_close": 159.065,
                "m5_spread": 0.0,
                "m5_atr_14": 0.09,
                "m15_atr_14": 0.16,
                "m5_ema_20": 159.114,
                "m5_ema_50": 159.173,
                "m15_ema_20": 159.195,
                "m15_ema_50": 159.352,
                "h1_ema_50": 159.307,
                "h1_ema_200": 158.800,
                "m15_ret_1": -0.00102,
                "m15_ret_3": 0.00006,
                "m5_volume_ratio_20": 1.0,
                "m15_volume_ratio_20": 1.0,
                "m5_body_efficiency": 0.286,
                "m15_body_efficiency": 0.355,
                "m15_range_position_20": 0.218,
                "multi_tf_alignment_score": 0.25,
                "fractal_persistence_score": 0.242,
                "seasonality_edge_score": 0.805,
                "market_instability_score": 0.38,
                "feature_drift_score": 0.24,
                "m5_tick_volume": 0.0,
                "m15_tick_volume": 0.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 19, 12, 14, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertTrue(bool(candidates[0].meta.get("fallback_live_recovery")))

    def test_live_recovery_emits_eurusd_candidate_on_high_range_drift(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.74,
            source="test",
            details={"range_flag": 1.0},
            state_label="LOW_LIQUIDITY_DRIFT",
        )
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 1.14784,
                "m15_close": 1.14810,
                "m5_spread": 0.0,
                "m5_atr_14": 0.00044,
                "m15_atr_14": 0.00085,
                "m5_ema_20": 1.14764,
                "m5_ema_50": 1.14741,
                "m15_ema_20": 1.14730,
                "m15_ema_50": 1.14748,
                "h1_ema_50": 1.14950,
                "h1_ema_200": 1.15380,
                "m15_ret_1": 0.00080,
                "m15_ret_3": -0.00034,
                "m5_volume_ratio_20": 1.0,
                "m15_volume_ratio_20": 1.0,
                "m5_body_efficiency": 1.0,
                "m15_body_efficiency": 1.0,
                "m15_range_position_20": 0.749,
                "multi_tf_alignment_score": 0.25,
                "fractal_persistence_score": 0.254,
                "seasonality_edge_score": 0.805,
                "market_instability_score": 0.44,
                "feature_drift_score": 0.18,
                "m5_tick_volume": 0.0,
                "m15_tick_volume": 0.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="EURUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 19, 12, 11, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "EURUSD_RANGE_REVERSION")

    def test_live_recovery_emits_audnzd_candidate_with_strong_alignment(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.80,
            source="test",
            details={"trend_flag": 1.0},
            state_label="LOW_LIQUIDITY_DRIFT",
        )
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 1.21060,
                "m15_close": 1.21058,
                "m5_spread": 0.0,
                "m5_atr_14": 0.00042,
                "m15_atr_14": 0.00064,
                "m5_ema_20": 1.21081,
                "m5_ema_50": 1.21102,
                "m15_ema_20": 1.21082,
                "m15_ema_50": 1.21093,
                "h1_ema_50": 1.21133,
                "h1_ema_200": 1.20625,
                "m15_ret_1": -0.00021,
                "m15_ret_3": -0.00161,
                "m5_volume_ratio_20": 1.0,
                "m15_volume_ratio_20": 1.0,
                "m5_body_efficiency": 0.309,
                "m15_body_efficiency": 0.309,
                "m15_range_position_20": 0.083,
                "multi_tf_alignment_score": 1.0,
                "fractal_persistence_score": 0.413,
                "seasonality_edge_score": 0.805,
                "market_instability_score": 0.29,
                "feature_drift_score": 0.13,
                "m5_tick_volume": 0.0,
                "m15_tick_volume": 0.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="AUDNZD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 19, 12, 12, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertIn(str(candidates[0].setup), {"AUDNZD_COMPRESSION_RELEASE", "AUDNZD_ROTATION_PULLBACK"})

    def test_live_recovery_emits_usoil_trend_retest_candidate(self) -> None:
        router = StrategyRouter(max_spread_points=95.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.73,
            source="test",
            details={"range_flag": 1.0},
            state_label="LOW_LIQUIDITY_DRIFT",
        )
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 96.58,
                "m15_close": 96.52,
                "m5_spread": 0.0,
                "m5_atr_14": 0.74,
                "m15_atr_14": 1.18,
                "m5_ema_20": 96.57,
                "m5_ema_50": 96.49,
                "m15_ema_20": 96.48,
                "m15_ema_50": 96.60,
                "h1_ema_50": 96.08,
                "h1_ema_200": 92.13,
                "m15_ret_1": 0.0,
                "m15_ret_3": -0.0122,
                "m5_volume_ratio_20": 0.523,
                "m15_volume_ratio_20": 0.351,
                "m5_body_efficiency": 0.833,
                "m15_body_efficiency": 0.0,
                "m15_range_position_20": 0.424,
                "multi_tf_alignment_score": 0.50,
                "fractal_persistence_score": 0.218,
                "seasonality_edge_score": 0.805,
                "market_instability_score": 0.38,
                "feature_drift_score": 0.22,
                "m5_tick_volume": 959.0,
                "m15_tick_volume": 2560.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="USOIL",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 19, 12, 13, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "USOIL_BREAKOUT_RETEST")
        self.assertEqual(str(candidates[0].meta.get("live_recovery_mode")), "trend_retest")

    def test_live_recovery_emits_usoil_low_range_reversion_when_h1_stays_bullish(self) -> None:
        router = StrategyRouter(max_spread_points=95.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.71,
            source="test",
            details={"range_flag": 1.0},
            state_label="LOW_LIQUIDITY_DRIFT",
        )
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 95.47,
                "m15_close": 95.47,
                "m5_spread": 0.0,
                "m5_atr_14": 0.82,
                "m15_atr_14": 1.22,
                "m5_ema_20": 96.37,
                "m5_ema_50": 96.41,
                "m15_ema_20": 96.37,
                "m15_ema_50": 96.55,
                "h1_ema_50": 96.04,
                "h1_ema_200": 92.12,
                "m15_ret_1": -0.0074,
                "m15_ret_3": -0.0121,
                "m5_volume_ratio_20": 1.0,
                "m15_volume_ratio_20": 0.737,
                "m5_body_efficiency": 0.80,
                "m15_body_efficiency": 0.655,
                "m15_range_position_20": 0.233,
                "multi_tf_alignment_score": 0.75,
                "fractal_persistence_score": 0.242,
                "seasonality_edge_score": 0.805,
                "market_instability_score": 0.345,
                "feature_drift_score": 0.142,
                "m5_tick_volume": 0.0,
                "m15_tick_volume": 5494.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="USOIL",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 19, 12, 15, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertIn(str(candidates[0].setup), {"USOIL_VWAP_REVERSION", "USOIL_BREAKOUT_RETEST"})

    def test_live_recovery_emits_gbpjpy_cross_pullback_candidate(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.70,
            source="test",
            details={"range_flag": 1.0},
            state_label="LIQUIDITY_SWEEP",
        )
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 211.323,
                "m15_close": 211.323,
                "m5_spread": 0.0,
                "m5_atr_14": 0.22,
                "m15_atr_14": 0.38,
                "m5_ema_20": 211.393,
                "m5_ema_50": 211.384,
                "m15_ema_20": 211.391,
                "m15_ema_50": 211.588,
                "h1_ema_50": 211.927,
                "h1_ema_200": 211.791,
                "m15_ret_1": -0.000624,
                "m15_ret_3": -0.000794,
                "m5_volume_ratio_20": 1.0,
                "m15_volume_ratio_20": 1.0,
                "m5_body_efficiency": 0.123,
                "m15_body_efficiency": 0.546,
                "m15_range_position_20": 0.615,
                "multi_tf_alignment_score": 0.25,
                "fractal_persistence_score": 0.289,
                "seasonality_edge_score": 0.805,
                "market_instability_score": 0.345,
                "feature_drift_score": 0.186,
                "m5_tick_volume": 0.0,
                "m15_tick_volume": 0.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="GBPJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 19, 12, 16, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "GBPJPY_SESSION_PULLBACK")
        self.assertEqual(str(candidates[0].meta.get("live_recovery_mode")), "cross_pullback")

    def test_xau_micro_scalper_requires_allowed_session_window(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.82, source="test", details={"trend_flag": 1.0, "state_label": "TRENDING"})
        session = SessionContext(
            session_name="PRE_OPEN",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=0.75,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_spread": 18.0,
                "m1_atr_14": 0.8,
                "m5_atr_14": 2.2,
                "m1_body": 0.8,
                "m1_volume_ratio_20": 1.25,
                "m1_body_efficiency": 0.72,
                "m15_volume_ratio_20": 1.12,
                "m1_momentum_3": 0.5,
                "m5_macd_hist_slope": 0.02,
                "m1_bullish": 1,
                "m1_bearish": 0,
            }
        )
        candidates = router._xau_m1_micro_scalper(  # noqa: SLF001
            symbol="XAUUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 10, 1, 0, tzinfo=timezone.utc),
        )
        self.assertEqual(candidates, [])

    def test_nzdjpy_sydney_breakout_retest_rejects_late_break(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.81, source="test", details={"trend_flag": 1.0, "state_label": "TRENDING"})
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_spread": 8.0,
                "m15_close": 88.62,
                "m15_atr_14": 0.40,
                "m15_rolling_high_prev_20": 88.40,
                "m15_rolling_low_prev_20": 87.90,
                "m15_volume_ratio_20": 1.10,
                "m5_body_efficiency": 0.60,
                "m15_low": 88.54,
                "m15_high": 88.68,
            }
        )
        candidates = router._nzdjpy_sydney_breakout_retest(  # noqa: SLF001
            symbol="NZDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 10, 21, 10, tzinfo=timezone.utc),
        )
        self.assertEqual(candidates, [])

    def test_nzdjpy_sydney_breakout_retest_candidate_is_generated(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.80, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="NZDJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 22, 15, tzinfo=timezone.utc),
                    "m5_close": 88.424,
                    "m5_spread": 8.0,
                    "m5_ret_1": 0.0003,
                    "m5_body_efficiency": 0.66,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_close": 88.424,
                    "m15_atr_14": 0.40,
                    "m15_atr_pct_of_avg": 1.06,
                    "m15_rolling_high_prev_20": 88.41,
                    "m15_rolling_low_prev_20": 87.90,
                    "m15_volume_ratio_20": 1.18,
                    "m15_range_position_20": 0.76,
                    "m15_low": 88.421,
                    "m15_high": 88.425,
                }
            ]
        )

        candidates = router.generate(
            symbol="NZDJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 22, 15, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "NZDJPY_SYDNEY_BREAKOUT_RETEST" for item in candidates))

    def test_audjpy_asia_pullback_rejects_ranging_regime(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.78, source="test", details={"range_flag": 1.0, "state_label": "RANGING"})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_spread": 8.0,
                "m15_close": 94.30,
                "m15_atr_14": 0.25,
                "m15_ema_20": 94.22,
                "m15_ema_50": 94.08,
                "m15_volume_ratio_20": 1.20,
                "m5_body_efficiency": 0.70,
                "m15_range_position_20": 0.60,
                "m15_ret_1": 0.02,
            }
        )

        candidates = router._asia_continuation_pullback(  # noqa: SLF001
            symbol="AUDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 10, 0, 45, tzinfo=timezone.utc),
        )

        self.assertEqual(candidates, [])

    def test_nzdjpy_asia_rotation_reclaim_candidate_is_generated(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.79,
            source="test",
            details={"range_flag": 1.0},
            state_label="RANGING",
        )
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="NZDJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 22, 45, tzinfo=timezone.utc),
                    "m5_close": 88.11,
                    "m5_spread": 7.0,
                    "m5_rsi_14": 43.0,
                    "m5_body_efficiency": 0.68,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_close": 88.11,
                    "m15_high": 88.15,
                    "m15_low": 87.71,
                    "m15_atr_14": 0.34,
                    "m15_atr_pct_of_avg": 0.98,
                    "m15_rolling_high_prev_20": 88.28,
                    "m15_rolling_low_prev_20": 87.76,
                    "m15_range_position_20": 0.16,
                    "m15_volume_ratio_20": 1.05,
                }
            ]
        )

        candidates = router.generate(
            symbol="NZDJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 22, 45, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "NZDJPY_SWEEP_RECLAIM" for item in candidates))

    def test_nzdjpy_asia_rotation_reclaim_is_generated_in_tokyo_mean_reversion_when_reclaim_is_clean(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="MEAN_REVERSION",
            confidence=0.80,
            source="test",
            details={"reversion_flag": 1.0},
            state_label="MEAN_REVERSION",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="NZDJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 22, 45, tzinfo=timezone.utc),
                    "m5_close": 88.11,
                    "m5_spread": 7.0,
                    "m5_rsi_14": 43.0,
                    "m5_body_efficiency": 0.68,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_close": 88.11,
                    "m15_high": 88.15,
                    "m15_low": 87.71,
                    "m15_atr_14": 0.34,
                    "m15_atr_pct_of_avg": 0.98,
                    "m15_rolling_high_prev_20": 88.28,
                    "m15_rolling_low_prev_20": 87.76,
                    "m15_range_position_20": 0.16,
                    "m15_volume_ratio_20": 1.05,
                }
            ]
        )

        candidates = router.generate(
            symbol="NZDJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 22, 45, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "NZDJPY_SWEEP_RECLAIM" for item in candidates))

    def test_audjpy_asia_rotation_reclaim_tolerates_live_like_spread_profile(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.77,
            source="test",
            details={"range_flag": 1.0},
            state_label="RANGING",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.72,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 23, 30, tzinfo=timezone.utc),
                    "m5_close": 95.18,
                    "m5_spread": 30.0,
                    "m5_rsi_14": 44.0,
                    "m5_body_efficiency": 0.66,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_close": 95.18,
                    "m15_high": 95.22,
                    "m15_low": 94.74,
                    "m15_atr_14": 0.40,
                    "m15_atr_pct_of_avg": 0.96,
                    "m15_rolling_high_prev_20": 95.36,
                    "m15_rolling_low_prev_20": 94.78,
                    "m15_range_position_20": 0.18,
                    "m15_volume_ratio_20": 0.96,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 23, 30, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "AUDJPY_SWEEP_RECLAIM" for item in candidates))

    def test_audnzd_rotation_breakout_tolerates_live_like_spread_profile(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.76, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDNZD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 1, 45, tzinfo=timezone.utc),
                    "m5_close": 1.0814,
                    "m5_spread": 22.0,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_close": 1.0814,
                    "m15_atr_14": 0.0010,
                    "m15_rolling_high_prev_20": 1.0810,
                    "m15_rolling_low_prev_20": 1.0795,
                    "m15_volume_ratio_20": 1.08,
                    "m15_range_position_20": 0.66,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDNZD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 1, 45, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "AUDNZD_ASIA_ROTATION_BREAKOUT" for item in candidates))

    def test_usdjpy_asia_session_momentum_rejects_late_chase_extension(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.82, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="USDJPY")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 0, 45, tzinfo=timezone.utc),
                    "m5_close": 149.82,
                    "m5_spread": 9.0,
                    "m5_atr_14": 0.18,
                    "m5_ema_20": 149.40,
                    "m5_ema_50": 149.10,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m5_ret_1": 0.0011,
                    "m5_body_efficiency": 0.66,
                    "m15_close": 149.82,
                    "m15_atr_14": 0.22,
                    "m15_ema_20": 149.38,
                    "m15_ema_50": 149.04,
                    "m15_volume_ratio_20": 1.10,
                    "m15_range_position_20": 0.96,
                }
            ]
        )

        candidates = router.generate(
            symbol="USDJPY",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 0, 45, tzinfo=timezone.utc),
        )

        self.assertFalse(any(str(item.setup).upper() == "USDJPY_SESSION_MOMENTUM" for item in candidates))

    def test_usdjpy_tokyo_pullback_requires_stronger_reclaim_and_structure(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.82, source="test", details={"trend_flag": 1.0, "state_label": "TRENDING"})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_spread": 8.0,
                "m15_close": 149.42,
                "m5_close": 149.42,
                "m15_atr_14": 0.22,
                "m5_atr_14": 0.18,
                "m5_ema_20": 149.31,
                "m5_ema_50": 149.08,
                "h1_ema_50": 149.10,
                "h1_ema_200": 148.95,
                "m15_volume_ratio_20": 1.12,
                "m5_body_efficiency": 0.59,
                "m15_range_position_20": 0.80,
                "m15_ret_1": 0.009,
                "m15_ret_3": 0.014,
            }
        )

        candidates = router._session_pullback_continuation(  # noqa: SLF001
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 10, 0, 45, tzinfo=timezone.utc),
        )

        self.assertEqual(candidates, [])

    def test_audnzd_range_rejection_is_selective_daytrade_setup(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.82,
            source="test",
            details={"range_flag": 1.0},
            state_label="RANGING",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDNZD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 1, 5, tzinfo=timezone.utc),
                    "m5_close": 1.0800,
                    "m5_spread": 7.0,
                    "m5_pinbar_bull": 1,
                    "m5_pinbar_bear": 0,
                    "m5_engulf_bull": 0,
                    "m5_engulf_bear": 0,
                    "m5_lower_wick_ratio": 0.28,
                    "m5_body_efficiency": 0.67,
                    "m15_close": 1.0800,
                    "m15_atr_14": 0.0010,
                    "m15_range_position_20": 0.15,
                    "m15_volume_ratio_20": 1.04,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDNZD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 1, 5, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "AUDNZD_RANGE_REJECTION" for item in candidates))
        audnzd = next(item for item in candidates if str(item.setup).upper() == "AUDNZD_RANGE_REJECTION")
        self.assertEqual(str(audnzd.entry_kind).upper(), "DAYTRADE")

    def test_audnzd_range_rejection_is_generated_in_mean_reversion_when_structure_is_clean(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="MEAN_REVERSION",
            confidence=0.79,
            source="test",
            details={"reversion_flag": 1.0},
            state_label="MEAN_REVERSION",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDNZD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 1, 5, tzinfo=timezone.utc),
                    "m5_close": 1.0800,
                    "m5_spread": 7.0,
                    "m5_pinbar_bull": 1,
                    "m5_pinbar_bear": 0,
                    "m5_engulf_bull": 0,
                    "m5_engulf_bear": 0,
                    "m5_lower_wick_ratio": 0.28,
                    "m5_body_efficiency": 0.67,
                    "m15_close": 1.0800,
                    "m15_atr_14": 0.0010,
                    "m15_range_position_20": 0.15,
                    "m15_volume_ratio_20": 1.04,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDNZD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 1, 5, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "AUDNZD_RANGE_REJECTION" for item in candidates))

    def test_audnzd_range_rejection_rejects_hot_volume_and_atr_spike(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.81,
            source="test",
            details={"range_flag": 1.0},
            state_label="RANGING",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.70,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDNZD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 1, 8, tzinfo=timezone.utc),
                    "m5_close": 1.0800,
                    "m5_spread": 7.0,
                    "m5_pinbar_bull": 1,
                    "m5_lower_wick_ratio": 0.29,
                    "m5_body_efficiency": 0.68,
                    "m15_close": 1.0800,
                    "m15_atr_14": 0.0010,
                    "m15_atr_pct_of_avg": 1.32,
                    "m15_range_position_20": 0.15,
                    "m15_volume_ratio_20": 1.38,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDNZD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 1, 8, tzinfo=timezone.utc),
        )

        self.assertFalse(any(str(item.setup).upper() == "AUDNZD_RANGE_REJECTION" for item in candidates))

    def test_btc_router_generates_price_action_candidate_in_tokyo_without_proxies(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.8, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 10, 0, tzinfo=timezone.utc),
                    "m5_close": 68000.0,
                    "m5_spread": 1700.0,
                    "m5_spread_ratio_20": 1.2,
                    "m5_atr_14": 420.0,
                    "m5_atr_pct_of_avg": 1.15,
                    "m5_ret_1": 0.004,
                    "m5_ret_5": 0.009,
                    "m5_volume_ratio_20": 0.0,
                    "m5_ema_20": 67910.0,
                    "m5_ema_50": 67780.0,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_rolling_high_prev_20": 67950.0,
                    "m15_rolling_low_prev_20": 67100.0,
                    "m15_atr_14": 620.0,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
        )

        self.assertTrue(any(str(item.setup).upper() in {"BTC_TOKYO_DRIFT_SCALP", "BTC_MOMENTUM_CONTINUATION", "BTC_RANGE_EXPANSION"} for item in candidates))

    def test_btc_router_generates_momentum_continuation_in_sydney_without_proxies(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.8, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.6,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 9, 23, 20, tzinfo=timezone.utc),
                    "m5_close": 66302.5,
                    "m5_spread": 1700.0,
                    "m5_spread_ratio_20": 1.15,
                    "m5_atr_14": 220.0,
                    "m5_atr_pct_of_avg": 1.02,
                    "m5_ret_1": 0.0011,
                    "m5_volume_ratio_20": 1.0,
                    "m5_ema_20": 66343.7,
                    "m5_ema_50": 66643.1,
                    "m5_rsi_14": 44.7,
                    "m5_bullish": 0,
                    "m5_bearish": 0,
                    "m15_rolling_high_prev_20": 67480.8,
                    "m15_rolling_low_prev_20": 65660.5,
                    "h1_ema_20": 67625.9,
                    "h1_ema_50": 68328.3,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 9, 23, 20, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "BTC_MOMENTUM_CONTINUATION" for item in candidates))

    def test_btc_router_generates_price_action_fallback_from_midrange_pullback(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.75, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.6,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 9, 23, 35, tzinfo=timezone.utc),
                    "m5_close": 66500.65,
                    "m5_spread": 1700.0,
                    "m5_spread_ratio_20": 1.10,
                    "m5_atr_14": 143.43,
                    "m5_atr_pct_of_avg": 0.90,
                    "m5_ret_1": -0.0003,
                    "m5_volume_ratio_20": 1.0,
                    "m5_ema_20": 66393.28,
                    "m5_ema_50": 66631.35,
                    "m5_rsi_14": 51.9,
                    "m5_bullish": 0,
                    "m5_bearish": 1,
                    "m15_range_position_20": 0.47,
                    "m15_rolling_high_prev_20": 67480.8,
                    "m15_rolling_low_prev_20": 65660.5,
                    "h1_ema_20": 66994.63,
                    "h1_ema_50": 67622.46,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 9, 23, 35, tzinfo=timezone.utc),
        )

        self.assertTrue(
            any(str(item.setup).upper() in {"BTC_TOKYO_DRIFT_SCALP", "BTC_MOMENTUM_CONTINUATION"} for item in candidates)
        )

    def test_btc_router_generates_proxyless_price_action_candidate_when_other_families_fail(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.74, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.6,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 14, 10, 10, tzinfo=timezone.utc),
                    "m5_close": 68210.0,
                    "m5_spread": 1680.0,
                    "m5_spread_ratio_20": 1.28,
                    "m5_atr_14": 380.0,
                    "m5_atr_pct_of_avg": 1.10,
                    "m5_ret_1": 0.0002,
                    "m5_ret_5": 0.0045,
                    "m5_volume_ratio_20": 0.52,
                    "m5_body_efficiency": 0.44,
                    "m5_ema_20": 68140.0,
                    "m5_ema_50": 68080.0,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m5_rsi_14": 56.0,
                    "m15_range_position_20": 0.61,
                    "m15_rolling_high_prev_20": 68490.0,
                    "m15_rolling_low_prev_20": 67620.0,
                    "m15_atr_14": 520.0,
                    "h1_ema_20": 68190.0,
                    "h1_ema_50": 68040.0,
                    "multi_tf_alignment_score": 0.63,
                    "seasonality_edge_score": 0.46,
                    "feature_drift_score": 0.12,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 14, 10, 10, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "BTC_PRICE_ACTION_CONTINUATION" for item in candidates))

    def test_btc_router_generates_weekend_proxyless_spray_candidate_for_live_like_tokyo_midrange(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.72, source="test", details={"trend_flag": 0.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.6,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 28, 2, 0, tzinfo=timezone.utc),
                    "m5_close": 66144.9609375,
                    "m5_spread": 1697.0,
                    "m5_spread_ratio_20": 1.62,
                    "m5_atr_14": 83.53,
                    "m5_atr_pct_of_avg": 1.02,
                    "m5_ret_1": 0.001237,
                    "m5_ret_5": 0.001299,
                    "m5_volume_ratio_20": 0.90,
                    "m5_body_efficiency": 0.44,
                    "m5_ema_20": 66178.55,
                    "m5_ema_50": 66139.42,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m5_rsi_14": 56.0,
                    "m15_range_position_20": 0.60,
                    "m15_rolling_high_prev_20": 66468.92,
                    "m15_rolling_low_prev_20": 65668.41,
                    "m15_atr_14": 135.11,
                    "h1_ema_20": 66693.60,
                    "h1_ema_50": 67938.07,
                    "multi_tf_alignment_score": 0.52,
                    "seasonality_edge_score": 0.34,
                    "feature_drift_score": 0.18,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 28, 2, 0, tzinfo=timezone.utc),
        )

        self.assertTrue(
            any(
                str(item.setup).upper() in {"BTC_TOKYO_DRIFT_SCALP", "BTC_MOMENTUM_CONTINUATION"}
                for item in candidates
            )
        )

    def test_btc_router_normalizes_live_spread_points_for_weekend_crypto_quotes(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.72, source="test", details={"trend_flag": 0.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.6,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 28, 4, 20, tzinfo=timezone.utc),
                    "m5_close": 66235.02,
                    "m5_spread": 1712.0,
                    "m5_spread_ratio_20": 43.47,
                    "m5_atr_14": 39.38,
                    "m5_atr_pct_of_avg": 0.64,
                    "m5_ret_1": 0.00021,
                    "m5_ret_5": 0.00101,
                    "m5_volume_ratio_20": 1.0,
                    "m5_body_efficiency": 0.83,
                    "m5_ema_20": 66140.74,
                    "m5_ema_50": 66130.96,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m5_rsi_14": 50.0,
                    "m15_range_position_20": 0.58,
                    "m15_rolling_high_prev_20": 66400.0,
                    "m15_rolling_low_prev_20": 65880.0,
                    "m15_atr_14": 58.0,
                    "h1_ema_20": 66516.34,
                    "h1_ema_50": 67673.65,
                    "multi_tf_alignment_score": 0.60,
                    "seasonality_edge_score": 0.40,
                    "feature_drift_score": 0.12,
                    "point": 0.01,
                    "trade_tick_size": 0.01,
                    "digits": 2,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 28, 4, 20, tzinfo=timezone.utc),
        )

        self.assertTrue(
            any(
                str(item.setup).upper() in {"BTC_TOKYO_DRIFT_SCALP", "BTC_MOMENTUM_CONTINUATION", "BTC_RANGE_EXPANSION", "BTC_WEEKEND_BREAKOUT"}
                for item in candidates
            )
        )

    def test_btc_router_generates_range_edge_fallback_without_proxy_feeds(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.7, source="test", details={"trend_flag": 0.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.6,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 9, 23, 50, tzinfo=timezone.utc),
                    "m5_close": 66480.0,
                    "m5_spread": 1650.0,
                    "m5_spread_ratio_20": 1.18,
                    "m5_atr_14": 165.0,
                    "m5_atr_pct_of_avg": 1.05,
                    "m5_ret_1": 0.0,
                    "m5_volume_ratio_20": 0.92,
                    "m5_ema_20": 66420.0,
                    "m5_ema_50": 66390.0,
                    "m5_bullish": 0,
                    "m5_bearish": 0,
                    "m5_rsi_14": 52.0,
                    "m15_range_position_20": 0.22,
                    "m15_rolling_high_prev_20": 66890.0,
                    "m15_rolling_low_prev_20": 66220.0,
                    "h1_ema_20": 66450.0,
                    "h1_ema_50": 66410.0,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 9, 23, 50, tzinfo=timezone.utc),
        )

        self.assertTrue(
            any(
                str(item.setup).upper() in {"BTC_TOKYO_DRIFT_SCALP", "BTC_TREND_SCALP", "BTC_PRICE_ACTION_CONTINUATION"}
                for item in candidates
            )
        )

    def test_btc_london_trend_scalp_rejects_weak_london_impulse_conditions(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.79, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 8, 15, tzinfo=timezone.utc),
                    "m5_close": 68000.0,
                    "m5_spread": 1800.0,
                    "m5_spread_ratio_20": 1.30,
                    "m5_atr_14": 420.0,
                    "m5_atr_pct_of_avg": 1.10,
                    "m5_ret_1": 0.0030,
                    "m5_ret_5": 0.0085,
                    "m5_volume_ratio_20": 1.05,
                    "m5_ema_20": 67920.0,
                    "m5_ema_50": 67780.0,
                    "m5_body_efficiency": 0.60,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_rolling_high_prev_20": 67970.0,
                    "m15_rolling_low_prev_20": 67100.0,
                    "m15_atr_14": 620.0,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 8, 15, tzinfo=timezone.utc),
        )

        self.assertFalse(any(str(item.setup).upper() == "BTC_TREND_SCALP" for item in candidates))

    def test_btc_tokyo_trend_scalp_requires_breakout_expansion_regime(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.79,
            source="test",
            details={"trend_flag": 1.0},
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 10, 0, 15, tzinfo=timezone.utc),
                    "m5_close": 66680.0,
                    "m5_spread": 1650.0,
                    "m5_spread_ratio_20": 1.15,
                    "m5_atr_14": 230.0,
                    "m5_atr_pct_of_avg": 1.18,
                    "m5_ret_1": 0.0005,
                    "m5_ret_5": 0.0045,
                    "m5_volume_ratio_20": 1.20,
                    "m5_body_efficiency": 0.70,
                    "m5_ema_20": 66600.0,
                    "m5_ema_50": 66480.0,
                    "m5_bullish": 1,
                    "m15_rolling_high_prev_20": 66620.0,
                    "m15_rolling_low_prev_20": 66150.0,
                    "m15_atr_14": 240.0,
                    "m15_range_position_20": 0.66,
                    "h1_ema_20": 66540.0,
                    "h1_ema_50": 66440.0,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 10, 0, 15, tzinfo=timezone.utc),
        )

        self.assertFalse(any(str(item.setup).upper() == "BTC_TREND_SCALP" for item in candidates))

    def test_btc_router_generates_soft_range_edge_fallback_when_alignment_is_neutral(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.72, source="test", details={"trend_flag": 0.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.6,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 9, 23, 55, tzinfo=timezone.utc),
                    "m5_close": 66352.0,
                    "m5_spread": 1690.0,
                    "m5_spread_ratio_20": 1.24,
                    "m5_atr_14": 175.0,
                    "m5_atr_pct_of_avg": 1.08,
                    "m5_ret_1": -0.00012,
                    "m5_volume_ratio_20": 0.88,
                    "m5_ema_20": 66380.0,
                    "m5_ema_50": 66392.0,
                    "m5_bullish": 0,
                    "m5_bearish": 0,
                    "m5_rsi_14": 43.0,
                    "m15_range_position_20": 0.31,
                    "m15_rolling_high_prev_20": 66890.0,
                    "m15_rolling_low_prev_20": 66210.0,
                    "h1_ema_20": 66410.0,
                    "h1_ema_50": 66405.0,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 9, 23, 55, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() in {"BTC_TOKYO_DRIFT_SCALP", "BTC_TREND_SCALP"} for item in candidates))

    def test_btc_router_uses_monday_weekday_mode_after_fx_open(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.7, source="test", details={"trend_flag": 0.0})
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.6,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 66200.0,
                "m5_spread": 1500.0,
                "m5_spread_ratio_20": 1.1,
                "m5_atr_pct_of_avg": 1.0,
                "m15_range_position_20": 0.5,
            }
        )
        timestamp = datetime(2026, 3, 9, 0, 30, tzinfo=timezone.utc)

        diagnostics = router.diagnostics("BTCUSD", row, regime, session, timestamp)

        self.assertEqual(diagnostics["weekend_vs_weekday_btc_mode"], "WEEKDAY")
        self.assertEqual(diagnostics["session_policy_current"], "BTC_WEEKDAY_SELECTIVE")

    def test_xau_diagnostics_expose_tokyo_active_window(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.75, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series({"m5_spread": 28.0})

        diagnostics = router.diagnostics("XAUUSD", row, regime, session, datetime(2026, 3, 9, 0, 30, tzinfo=timezone.utc))

        self.assertEqual(diagnostics["session_policy_current"], "XAU_WEEKDAY_PRIORITY")
        self.assertIn("XAU_TOKYO_SESSION", diagnostics["active_setup_windows"])

    def test_btc_funding_arb_can_generate_outside_primary_session_when_proxy_is_present(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.7, source="test", details={"trend_flag": 0.0})
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=0.8,
            confluence_delta=1,
            ai_threshold_offset=0.02,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 7, 23, 50, tzinfo=timezone.utc),
                    "m5_close": 68000.0,
                    "m5_spread": 1700.0,
                    "m5_spread_ratio_20": 1.1,
                    "m5_atr_14": 420.0,
                    "m5_atr_pct_of_avg": 1.0,
                    "m5_ret_1": -0.001,
                    "m5_volume_ratio_20": 1.15,
                    "m5_ema_20": 67910.0,
                    "m5_ema_50": 67890.0,
                    "m5_bullish": 0,
                    "m5_bearish": 1,
                    "m15_rolling_high_prev_20": 68400.0,
                    "m15_rolling_low_prev_20": 67600.0,
                    "m15_atr_14": 620.0,
                    "btc_funding_rate_8h": 0.0015,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
        )

        self.assertTrue(any(str(item.setup).upper() == "BTC_FUNDING_ARB" for item in candidates))

    def test_btc_weekend_gap_candidate_is_generated_when_gap_context_present(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.7, source="test", details={"trend_flag": 0.0})
        session = SessionContext(
            session_name="NEW_YORK",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=0.95,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 8, 20, 0, tzinfo=timezone.utc),
                    "m5_close": 70000.0,
                    "m5_spread": 1800.0,
                    "m5_spread_ratio_20": 1.1,
                    "m5_atr_14": 380.0,
                    "m5_atr_pct_of_avg": 1.2,
                    "m5_ret_1": -0.002,
                    "m5_ret_5": 0.006,
                    "m5_volume_ratio_20": 1.05,
                    "m5_ema_20": 70100.0,
                    "m5_ema_50": 69980.0,
                    "m5_bullish": 0,
                    "m5_bearish": 1,
                    "m15_rolling_high_prev_20": 70400.0,
                    "m15_rolling_low_prev_20": 69500.0,
                    "m15_atr_14": 560.0,
                    "btc_weekend_gap_pct": 0.024,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
        )

        self.assertTrue(any(str(item.setup).upper() == "BTC_WEEKEND_GAP_FADE" for item in candidates))

    def test_btc_diagnostics_report_weekend_priority_from_current_timestamp(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.7, source="test", details={"trend_flag": 0.0})
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_spread": 1700.0,
                "btc_weekend_gap_pct": 0.03,
            }
        )

        diagnostics = router.diagnostics(
            symbol="BTCUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 8, 10, 0, tzinfo=timezone.utc),
        )

        self.assertEqual(str(diagnostics.get("session_policy_current")), "BTC_WEEKEND_PRIORITY")
        self.assertEqual(str(diagnostics.get("weekend_vs_weekday_btc_mode")), "WEEKEND")
        self.assertTrue(bool(diagnostics.get("weekend_gap_proxy_available")))

    def test_xau_fix_flow_candidate_is_generated_when_dxy_and_yields_align(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.8, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="VERY_HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=-0.01,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="XAUUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 9, 14, 50, tzinfo=timezone.utc),
                    "m5_close": 2910.0,
                    "m5_high": 2911.2,
                    "m5_low": 2908.8,
                    "m5_spread": 18.0,
                    "m5_atr_14": 4.2,
                    "m5_volume_ratio_20": 1.12,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m5_ret_1": 0.0012,
                    "m15_close": 2910.0,
                    "m15_high": 2912.0,
                    "m15_low": 2907.5,
                    "m15_atr_14": 6.0,
                    "m15_rolling_high_prev_20": 2909.0,
                    "m15_rolling_low_prev_20": 2898.0,
                    "dxy_ret_5": -0.0035,
                    "us10y_ret_5": -0.0020,
                }
            ]
        )

        candidates = router.generate(
            symbol="XAUUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
        )

        self.assertTrue(any(str(item.setup).upper() == "XAUUSD_M15_FIX_FLOW" for item in candidates))

    def test_btc_router_remains_bounded_under_extreme_spread_conditions(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.8, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="VERY_HIGH",
            size_multiplier=1.0,
            confluence_delta=-1,
            ai_threshold_offset=-0.01,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc),
                    "m5_close": 68000.0,
                    "m5_spread": 5000.0,
                    "m5_spread_ratio_20": 3.4,
                    "m5_atr_14": 420.0,
                    "m5_atr_pct_of_avg": 3.4,
                    "m5_ret_1": 0.004,
                    "m5_volume_ratio_20": 1.18,
                    "m5_body_efficiency": 0.62,
                    "point": 1.0,
                    "trade_tick_size": 1.0,
                    "digits": 0,
                    "m5_ema_20": 67910.0,
                    "m5_ema_50": 67780.0,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_rolling_high_prev_20": 67950.0,
                    "m15_rolling_low_prev_20": 67100.0,
                    "m15_atr_14": 620.0,
                    "h1_ema_20": 67820.0,
                    "h1_ema_50": 67610.0,
                    "h4_ema_50": 67550.0,
                    "h4_ema_200": 66000.0,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
        )

        self.assertEqual(candidates, [])

    def test_btc_router_pauses_when_30m_move_is_too_large(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.8, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="OVERLAP",
            in_session=True,
            liquidity_tier="VERY_HIGH",
            size_multiplier=1.0,
            confluence_delta=-1,
            ai_threshold_offset=-0.01,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="BTCUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 7, 15, 0, tzinfo=timezone.utc),
                    "m5_close": 68000.0,
                    "m5_spread": 1800.0,
                    "m5_spread_ratio_20": 1.2,
                    "m5_atr_14": 420.0,
                    "m5_atr_pct_of_avg": 1.15,
                    "m5_ret_1": 0.004,
                    "m5_ret_5": 0.02,
                    "m5_volume_ratio_20": 1.18,
                    "m5_ema_20": 67910.0,
                    "m5_ema_50": 67780.0,
                    "m5_bullish": 1,
                    "m5_bearish": 0,
                    "m15_rolling_high_prev_20": 67950.0,
                    "m15_rolling_low_prev_20": 67100.0,
                    "m15_atr_14": 620.0,
                }
            ]
        )

        candidates = router.generate(
            symbol="BTCUSD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
        )

        self.assertEqual(candidates, [])

    def test_audnzd_tokyo_range_reversion_requires_real_rejection(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="RANGING", confidence=0.8, source="test", details={"range_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.7,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="AUDNZD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 12, 1, 0, tzinfo=timezone.utc),
                    "m5_close": 1.0820,
                    "m5_spread": 7.0,
                    "m5_rsi_14": 38.0,
                    "m5_body_efficiency": 0.58,
                    "m5_volume_ratio_20": 1.04,
                    "m5_pinbar_bull": 0,
                    "m5_engulf_bull": 0,
                    "m5_pinbar_bear": 0,
                    "m5_engulf_bear": 0,
                    "m5_lower_wick_ratio": 0.12,
                    "m5_upper_wick_ratio": 0.10,
                    "m15_close": 1.0820,
                    "m15_atr_14": 0.0020,
                    "m15_rolling_high_prev_20": 1.0840,
                    "m15_rolling_low_prev_20": 1.0814,
                    "m15_range_position_20": 0.17,
                }
            ]
        )

        candidates = router.generate(
            symbol="AUDNZD",
            features=frame,
            regime=regime,
            session=session,
            strategy_engine=engine,
            open_positions=[],
            max_positions_per_symbol=1,
            current_time=datetime(2026, 3, 12, 1, 0, tzinfo=timezone.utc),
        )

        self.assertFalse(any(str(item.setup).upper() == "FOREX_RANGE_REVERSION" for item in candidates))

    def test_audjpy_asia_momentum_breakout_rejects_overextended_break(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.83, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_spread": 8.0,
                "m15_close": 95.03,
                "m15_atr_14": 0.20,
                "m15_rolling_high_prev_20": 94.90,
                "m15_rolling_low_prev_20": 94.40,
                "m15_volume_ratio_20": 1.12,
                "m15_ret_1": 0.05,
                "m5_bullish": 1,
                "m5_bearish": 0,
                "m5_body_efficiency": 0.72,
                "m5_upper_wick_ratio": 0.08,
                "m5_lower_wick_ratio": 0.06,
                "m15_range_position_20": 0.70,
                "m15_atr_pct_of_avg": 1.08,
            }
        )

        candidates = router._asia_momentum_breakout(  # noqa: SLF001
            symbol="AUDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 12, 1, 30, tzinfo=timezone.utc),
        )

        self.assertEqual(candidates, [])

    def test_audjpy_asia_momentum_breakout_uses_throughput_recovery_relief(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        router.apply_learning_policy(
            {
                "symbol": "AUDJPY",
                "bundle": {
                    "quota_catchup_pressure": 0.88,
                },
                "pair_directive": {
                    "session_focus": ["TOKYO", "SYDNEY"],
                },
            }
        )
        regime = RegimeClassification(label="TRENDING", confidence=0.82, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_spread": 8.0,
                "m15_close": 94.98,
                "m15_atr_14": 0.20,
                "m15_rolling_high_prev_20": 94.90,
                "m15_rolling_low_prev_20": 94.35,
                "m15_volume_ratio_20": 0.93,
                "m15_ret_1": 0.03,
                "m15_ret_3": 0.04,
                "m5_bullish": 1,
                "m5_bearish": 0,
                "m5_body_efficiency": 0.53,
                "m5_upper_wick_ratio": 0.25,
                "m5_lower_wick_ratio": 0.08,
                "m15_range_position_20": 0.64,
                "m15_atr_pct_of_avg": 0.81,
            }
        )

        candidates = router._asia_momentum_breakout(  # noqa: SLF001
            symbol="AUDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 12, 1, 30, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "AUDJPY_TOKYO_MOMENTUM_BREAKOUT" for item in candidates))

    def test_nzdjpy_pullback_rejects_chased_reclaim_bar(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(label="TRENDING", confidence=0.82, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_spread": 8.0,
                "m15_close": 89.20,
                "m15_atr_14": 0.25,
                "m15_ema_20": 89.12,
                "m15_ema_50": 88.98,
                "m15_volume_ratio_20": 1.16,
                "m5_body_efficiency": 0.74,
                "m15_range_position_20": 0.58,
                "m5_lower_wick_ratio": 0.24,
                "m5_upper_wick_ratio": 0.08,
                "m15_ret_1": 0.05,
                "m15_ret_3": 0.07,
                "m15_atr_pct_of_avg": 1.04,
            }
        )

        candidates = router._asia_continuation_pullback(  # noqa: SLF001
            symbol="NZDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 12, 1, 45, tzinfo=timezone.utc),
        )

        self.assertEqual(candidates, [])

    def test_nzdjpy_pullback_uses_throughput_recovery_relief(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        router.apply_learning_policy(
            {
                "symbol": "NZDJPY",
                "bundle": {
                    "quota_catchup_pressure": 0.90,
                },
                "pair_directive": {
                    "session_focus": ["TOKYO", "SYDNEY"],
                },
            }
        )
        regime = RegimeClassification(label="TRENDING", confidence=0.82, source="test", details={"trend_flag": 1.0})
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="MEDIUM",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_spread": 8.0,
                "m15_close": 89.16,
                "m15_atr_14": 0.25,
                "m15_ema_20": 89.10,
                "m15_ema_50": 88.96,
                "h1_ema_20": 89.24,
                "h1_ema_50": 89.02,
                "m15_volume_ratio_20": 0.91,
                "m5_body_efficiency": 0.54,
                "m15_range_position_20": 0.56,
                "m5_lower_wick_ratio": 0.22,
                "m5_upper_wick_ratio": 0.08,
                "m15_ret_1": 0.045,
                "m15_ret_3": 0.064,
                "m15_atr_pct_of_avg": 0.83,
                "m5_bullish": 1,
                "m5_bearish": 0,
            }
        )

        candidates = router._asia_continuation_pullback(  # noqa: SLF001
            symbol="NZDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 12, 1, 45, tzinfo=timezone.utc),
        )

        self.assertTrue(any(str(item.setup).upper() == "NZDJPY_TOKYO_CONTINUATION_PULLBACK" for item in candidates))

    def test_gbpusd_london_trend_pullback_candidate_is_router_rejected_in_bad_regime(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="LOW_LIQUIDITY_CHOP",
            confidence=0.84,
            source="test",
            details={"trend_flag": 0.0, "wick_density": 0.84},
        )
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 1.2760,
                "m5_spread": 14.0,
                "m5_atr_14": 0.0011,
                "m5_body_efficiency": 0.58,
                "m5_ema_20": 1.2756,
                "m5_ema_50": 1.2750,
                "m5_range_position_20": 0.88,
                "m15_close": 1.2760,
                "m15_atr_14": 0.0014,
                "m15_range_position_20": 0.86,
                "m15_ema_20": 1.2754,
                "m15_volume_ratio_20": 1.10,
                "h1_ema_50": 1.2748,
                "h1_ema_200": 1.2724,
            }
        )
        candidate = SignalCandidate(
            signal_id="test-gbpusd-pullback",
            setup="GBPUSD_SESSION_PULLBACK",
            side="BUY",
            score_hint=0.72,
            reason="test",
            stop_atr=1.0,
            tp_r=1.8,
            strategy_family="TREND",
        )

        enriched = router._enrich_candidate(  # noqa: SLF001 - targeted regression test
            symbol="GBPUSD",
            row=row,
            regime=regime,
            session=session,
            candidate=candidate,
        )

        self.assertEqual(resolve_strategy_key("GBPUSD", enriched.setup), "GBPUSD_TREND_PULLBACK_RIDE")
        self.assertTrue(bool((enriched.meta or {}).get("router_reject")))
        self.assertEqual(
            str((enriched.meta or {}).get("router_reject_reason") or ""),
            "gbpusd_london_trend_pullback_bad_regime",
        )

    def test_enrich_candidate_rejects_btc_tokyo_trend_scalp_in_bad_liquidity(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="MEAN_REVERSION",
            confidence=0.74,
            source="test",
            details={"volatility_forecast_state": "BALANCED", "compression_proxy_state": "COMPRESSION"},
            state_label="MEAN_REVERSION",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.65,
            confluence_delta=1,
            ai_threshold_offset=0.03,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "time": datetime(2026, 3, 10, 1, 0, tzinfo=timezone.utc),
                "m5_spread": 25.0,
                "m5_atr_pct_of_avg": 0.88,
                "m5_body_efficiency": 0.42,
                "m15_range_position_20": 0.86,
                "m15_close": 84200.0,
                "m15_ema_20": 83900.0,
                "m15_atr_14": 420.0,
                "m5_ret_1": 0.0004,
                "m5_volume_ratio_20": 0.94,
            }
        )
        candidate = SignalCandidate(
            signal_id="btc-tokyo-scalp",
            setup="BTC_TOKYO_DRIFT_SCALP",
            side="BUY",
            score_hint=0.74,
            reason="known loser bucket",
            stop_atr=1.0,
            tp_r=1.6,
            strategy_family="TREND",
        )

        enriched = router._enrich_candidate(  # noqa: SLF001 - targeted regression test
            symbol="BTCUSD",
            row=row,
            regime=regime,
            session=session,
            candidate=candidate,
        )

        self.assertTrue(bool(enriched.meta.get("router_reject")))
        self.assertEqual(str(enriched.meta.get("router_reject_reason") or ""), "btc_tokyo_trend_scalp_bad_liquidity")

    def test_enrich_candidate_rejects_nas_tokyo_momentum_impulse_in_bad_regimes(self) -> None:
        router = StrategyRouter(max_spread_points=60.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.81,
            source="test",
            details={"trend_flag": 1.0, "volatility_forecast_state": "BALANCED"},
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "time": datetime(2026, 3, 10, 1, 15, tzinfo=timezone.utc),
                "m5_spread": 18.0,
                "m5_atr_pct_of_avg": 1.04,
                "m5_body_efficiency": 0.57,
                "m15_range_position_20": 0.84,
                "m15_close": 20340.0,
                "m15_ema_20": 20300.0,
                "m15_atr_14": 42.0,
                "m5_ret_1": 0.0008,
                "m5_volume_ratio_20": 1.05,
            }
        )
        candidate = SignalCandidate(
            signal_id="nas-tokyo-orb",
            setup="NAS_SESSION_SCALPER_ORB",
            side="BUY",
            score_hint=0.70,
            reason="known loser bucket",
            stop_atr=1.0,
            tp_r=1.8,
            strategy_family="TREND",
        )

        enriched = router._enrich_candidate(  # noqa: SLF001
            symbol="NAS100",
            row=row,
            regime=regime,
            session=session,
            candidate=candidate,
        )

        self.assertTrue(bool(enriched.meta.get("router_reject")))
        self.assertEqual(str(enriched.meta.get("router_reject_reason") or ""), "nas_opening_drive_tokyo_non_expansion")

    def test_audnzd_asia_drift_continuation_uses_extreme_range_fast_track(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.74,
            source="test",
            details={"trend_flag": 0.0},
            state_label="LOW_LIQUIDITY_DRIFT",
        )
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "time": datetime(2026, 3, 19, 22, 15, tzinfo=timezone.utc),
                "m5_spread": 16.0,
                "m5_close": 1.21234,
                "m15_close": 1.21234,
                "m5_atr_14": 0.0008,
                "m15_atr_14": 0.0008,
                "m5_atr_pct_of_avg": 1.18,
                "m15_atr_pct_of_avg": 1.26,
                "m5_volume_ratio_20": 1.0,
                "m15_volume_ratio_20": 1.0,
                "m5_body_efficiency": 0.44,
                "m15_body_efficiency": 0.33,
                "multi_tf_alignment_score": 0.75,
                "seasonality_edge_score": 0.35,
                "fractal_persistence_score": 0.28,
                "market_instability_score": 0.28,
                "feature_drift_score": 0.05,
                "m15_ema_20": 1.21228,
                "m15_ema_50": 1.21230,
                "h1_ema_50": 1.21190,
                "h1_ema_200": 1.21040,
                "m15_range_position_20": 0.97,
                "m15_ret_1": 0.00012,
                "m15_ret_3": 0.00036,
            }
        )

        self.assertTrue(
            router._drift_continuation_ready(  # noqa: SLF001
                symbol="AUDNZD",
                row=row,
                session_name="SYDNEY",
                raw_regime_label="LOW_LIQUIDITY_DRIFT",
            )
        )

        candidates = router._asia_drift_continuation(  # noqa: SLF001
            symbol="AUDNZD",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "AUDNZD_ROTATION_PULLBACK")
        self.assertEqual(str(candidates[0].side), "BUY")

    def test_audnzd_compression_release_accepts_slightly_wider_atr_and_lighter_volume(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.76,
            source="test",
            details={"trend_flag": 1.0},
            state_label="BREAKOUT_EXPANSION",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "time": datetime(2026, 3, 19, 23, 45, tzinfo=timezone.utc),
                "m5_spread": 18.0,
                "m5_close": 1.08135,
                "m15_close": 1.08135,
                "m5_atr_14": 0.0008,
                "m15_atr_14": 0.0008,
                "m5_atr_pct_of_avg": 1.24,
                "m15_atr_pct_of_avg": 1.27,
                "m5_volume_ratio_20": 0.80,
                "m15_volume_ratio_20": 0.80,
                "m15_rolling_high_prev_20": 1.08127,
                "m15_rolling_low_prev_20": 1.08045,
            }
        )

        candidates = router._audnzd_compression_release(  # noqa: SLF001
            symbol="AUDNZD",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "AUDNZD_COMPRESSION_RELEASE")

    def test_nzdjpy_asia_drift_continuation_accepts_m15_body_for_sydney_extreme(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.77,
            source="test",
            details={"trend_flag": 0.0},
            state_label="LOW_LIQUIDITY_DRIFT",
        )
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "time": datetime(2026, 3, 19, 22, 15, tzinfo=timezone.utc),
                "m5_spread": 15.0,
                "m5_close": 92.74,
                "m15_close": 92.74,
                "m5_atr_14": 0.12,
                "m15_atr_14": 0.12,
                "m5_atr_pct_of_avg": 1.12,
                "m15_atr_pct_of_avg": 1.24,
                "m5_volume_ratio_20": 1.0,
                "m15_volume_ratio_20": 1.0,
                "m5_body_efficiency": 0.08,
                "m15_body_efficiency": 0.41,
                "multi_tf_alignment_score": 1.0,
                "seasonality_edge_score": 0.35,
                "fractal_persistence_score": 0.45,
                "market_instability_score": 0.26,
                "feature_drift_score": 0.10,
                "m15_ema_20": 92.73,
                "m15_ema_50": 92.89,
                "h1_ema_50": 92.98,
                "h1_ema_200": 93.12,
                "m15_range_position_20": 0.03,
                "m15_ret_1": -0.00028,
                "m15_ret_3": -0.0029,
            }
        )

        self.assertTrue(
            router._drift_continuation_ready(  # noqa: SLF001
                symbol="NZDJPY",
                row=row,
                session_name="SYDNEY",
                raw_regime_label="LOW_LIQUIDITY_DRIFT",
            )
        )

        candidates = router._asia_drift_continuation(  # noqa: SLF001
            symbol="NZDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].side), "SELL")
        self.assertIn(
            str(candidates[0].setup),
            {"NZDJPY_ASIA_CONTINUATION_PULLBACK", "NZDJPY_ASIA_MOMENTUM_BREAKOUT", "NZDJPY_TOKYO_CONTINUATION_PULLBACK"},
        )

    def test_audjpy_asia_rotation_reclaim_allows_extreme_sydney_sweep_with_m15_body(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.71,
            source="test",
            details={"trend_flag": 1.0},
            state_label="LIQUIDITY_SWEEP",
        )
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "time": datetime(2026, 3, 19, 22, 20, tzinfo=timezone.utc),
                "m5_spread": 14.0,
                "m15_close": 112.24,
                "m15_high": 112.28,
                "m15_low": 112.23,
                "m15_atr_14": 0.06,
                "m15_atr_pct_of_avg": 1.08,
                "m15_rolling_high_prev_20": 112.94,
                "m15_rolling_low_prev_20": 112.14,
                "m15_range_position_20": 0.12,
                "m15_volume_ratio_20": 1.0,
                "m15_body_efficiency": 0.47,
                "m5_body_efficiency": 0.15,
                "m5_pinbar_bull": 1,
                "m5_bullish": 1,
                "m5_rsi_14": 42.0,
            }
        )

        candidates = router._asia_rotation_reclaim(  # noqa: SLF001
            symbol="AUDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].side), "BUY")
        self.assertEqual(str(candidates[0].setup), "AUDJPY_SWEEP_RECLAIM")

    def test_audnzd_range_rejection_allows_synthetic_upper_wick_sweep(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.74,
            source="test",
            details={"trend_flag": 0.0},
            state_label="LIQUIDITY_SWEEP",
        )
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "time": datetime(2026, 3, 19, 22, 20, tzinfo=timezone.utc),
                "m5_spread": 16.0,
                "m15_atr_14": 0.0011,
                "m15_atr_pct_of_avg": 1.48,
                "m15_range_position_20": 0.95,
                "m15_volume_ratio_20": 1.0,
                "m15_body_efficiency": 0.12,
                "m5_body_efficiency": 0.04,
                "m5_upper_wick_ratio": 0.33,
                "m5_lower_wick_ratio": 0.63,
                "m5_bearish": 1,
            }
        )

        candidates = router._audnzd_range_rejection(  # noqa: SLF001
            symbol="AUDNZD",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].side), "SELL")
        self.assertEqual(str(candidates[0].setup), "AUDNZD_RANGE_REJECTION")

    def test_forex_breakout_retest_allows_audjpy_tokyo_borderline_breakout(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.8,
            source="test",
            details={"trend_flag": 1.0},
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "time": datetime(2026, 3, 20, 0, 5, tzinfo=timezone.utc),
                "m5_spread": 16.0,
                "m15_close": 112.3174,
                "m15_high": 112.3220,
                "m15_low": 112.3150,
                "m5_close": 112.3174,
                "m5_atr_14": 0.06,
                "m15_atr_14": 0.06,
                "m15_volume_ratio_20": 0.97,
                "m5_volume_ratio_20": 0.97,
                "m15_rolling_high_prev_20": 112.30,
                "m15_rolling_low_prev_20": 112.12,
                "m15_range_position_20": 0.57,
                "m15_ret_3": 0.0180,
                "m5_body_efficiency": 0.55,
                "m5_upper_wick_ratio": 0.19,
                "m5_lower_wick_ratio": 0.08,
                "m5_bullish": 1,
                "m5_bearish": 0,
            }
        )

        candidates = router._forex_breakout_retest(  # noqa: SLF001
            symbol="AUDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "FOREX_BREAKOUT_RETEST")

    def test_session_pullback_continuation_allows_usdjpy_tokyo_borderline_reclaim(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.81,
            source="test",
            details={"trend_flag": 1.0},
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.75,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "time": datetime(2026, 3, 20, 0, 25, tzinfo=timezone.utc),
                "m5_spread": 18.0,
                "m5_close": 150.278,
                "m15_close": 150.278,
                "m5_atr_14": 0.10,
                "m15_atr_14": 0.10,
                "m5_ema_20": 150.252,
                "m5_ema_50": 150.220,
                "h1_ema_50": 150.210,
                "h1_ema_200": 150.120,
                "m15_volume_ratio_20": 1.06,
                "m5_volume_ratio_20": 1.06,
                "m15_range_position_20": 0.76,
                "m15_ret_1": 0.0045,
                "m15_ret_3": 0.0250,
                "m5_lower_wick_ratio": 0.17,
                "m5_upper_wick_ratio": 0.05,
                "m5_body_efficiency": 0.55,
            }
        )

        candidates = router._session_pullback_continuation(  # noqa: SLF001
            symbol="USDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "USDJPY_SESSION_PULLBACK")

    def test_session_momentum_boost_blocks_nas100_without_breakout_expansion(self) -> None:
        router = StrategyRouter(max_spread_points=120.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.82,
            source="test",
            details={"trend_flag": 1.0},
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "time": datetime(2026, 3, 20, 8, 5, tzinfo=timezone.utc),
                "m5_spread": 40.0,
                "m5_close": 21350.0,
                "m5_atr_14": 55.0,
                "m5_ret_1": 12.0,
                "m5_volume_ratio_20": 1.20,
                "m5_body_efficiency": 0.62,
                "m5_bullish": 1,
                "m5_bearish": 0,
                "m5_ema_20": 21310.0,
                "m5_ema_50": 21280.0,
                "m15_range_position_20": 0.74,
            }
        )

        candidates = router._session_momentum_boost(  # noqa: SLF001
            symbol="NAS100",
            row=row,
            regime=regime,
            session=session,
            timestamp=row["time"],
        )

        self.assertFalse(candidates)

    def test_live_recovery_emits_xau_candidate_in_london(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.84,
            source="test",
            details={"trend_flag": 1.0},
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 3052.0,
                "m15_close": 3052.0,
                "m5_spread": 12.0,
                "m5_atr_14": 4.2,
                "m15_atr_14": 7.0,
                "m5_ema_20": 3049.0,
                "m5_ema_50": 3046.5,
                "m15_ema_20": 3048.0,
                "m15_ema_50": 3044.5,
                "h1_ema_50": 3047.0,
                "h1_ema_200": 3038.0,
                "m15_ret_1": 0.0012,
                "m15_ret_3": 0.0036,
                "m5_volume_ratio_20": 0.62,
                "m15_volume_ratio_20": 0.74,
                "m5_body_efficiency": 0.26,
                "m15_body_efficiency": 0.30,
                "m15_range_position_20": 0.82,
                "multi_tf_alignment_score": 0.36,
                "fractal_persistence_score": 0.29,
                "seasonality_edge_score": 0.79,
                "market_instability_score": 0.24,
                "feature_drift_score": 0.11,
                "m5_tick_volume": 820.0,
                "m15_tick_volume": 1680.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="XAUUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 19, 8, 5, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "XAUUSD_M1_MICRO_SCALPER")
        self.assertTrue(bool(candidates[0].meta.get("fallback_live_recovery")))
        self.assertEqual(str(candidates[0].meta.get("setup_family") or ""), "LIVE_RECOVERY_XAU")

    def test_live_recovery_emits_btc_candidate_in_london(self) -> None:
        router = StrategyRouter(max_spread_points=4000.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.83,
            source="test",
            details={"trend_flag": 1.0},
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 87450.0,
                "m15_close": 87450.0,
                "m5_spread": 850.0,
                "m5_atr_14": 680.0,
                "m15_atr_14": 980.0,
                "m5_ema_20": 87140.0,
                "m5_ema_50": 86880.0,
                "m15_ema_20": 87020.0,
                "m15_ema_50": 86620.0,
                "h1_ema_50": 86980.0,
                "h1_ema_200": 86100.0,
                "m15_ret_1": 0.0018,
                "m15_ret_3": 0.0051,
                "m5_volume_ratio_20": 0.58,
                "m15_volume_ratio_20": 0.64,
                "m5_body_efficiency": 0.24,
                "m15_body_efficiency": 0.28,
                "m15_range_position_20": 0.79,
                "multi_tf_alignment_score": 0.34,
                "fractal_persistence_score": 0.24,
                "seasonality_edge_score": 0.77,
                "market_instability_score": 0.22,
                "feature_drift_score": 0.12,
                "m5_tick_volume": 1520.0,
                "m15_tick_volume": 4480.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="BTCUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 19, 8, 10, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "BTCUSD_PRICE_ACTION_CONTINUATION")
        self.assertTrue(bool(candidates[0].meta.get("fallback_live_recovery")))
        self.assertEqual(str(candidates[0].meta.get("setup_family") or ""), "LIVE_RECOVERY_BTC")

    def test_live_recovery_emits_xag_candidate_in_london(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.82,
            source="test",
            details={"trend_flag": 1.0},
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 34.52,
                "m15_close": 34.52,
                "m5_spread": 12.0,
                "m5_atr_14": 0.22,
                "m15_atr_14": 0.34,
                "m5_ema_20": 34.46,
                "m5_ema_50": 34.40,
                "m15_ema_20": 34.44,
                "m15_ema_50": 34.36,
                "h1_ema_50": 34.42,
                "h1_ema_200": 34.10,
                "m15_ret_1": 0.0012,
                "m15_ret_3": 0.0038,
                "m5_volume_ratio_20": 0.58,
                "m15_volume_ratio_20": 0.68,
                "m5_body_efficiency": 0.25,
                "m15_body_efficiency": 0.30,
                "m15_range_position_20": 0.78,
                "multi_tf_alignment_score": 0.32,
                "fractal_persistence_score": 0.24,
                "seasonality_edge_score": 0.74,
                "market_instability_score": 0.20,
                "feature_drift_score": 0.12,
                "m5_tick_volume": 610.0,
                "m15_tick_volume": 1520.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="XAGUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 27, 8, 10, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "XAGUSD_SESSION_MOMENTUM")
        self.assertEqual(str(candidates[0].meta.get("setup_family") or ""), "LIVE_RECOVERY_TREND")

    def test_live_recovery_emits_nas100_candidate_in_london(self) -> None:
        router = StrategyRouter(max_spread_points=120.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.83,
            source="test",
            details={"trend_flag": 1.0},
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 21480.0,
                "m15_close": 21480.0,
                "m5_spread": 24.0,
                "m5_atr_14": 58.0,
                "m15_atr_14": 84.0,
                "m5_ema_20": 21420.0,
                "m5_ema_50": 21370.0,
                "m15_ema_20": 21410.0,
                "m15_ema_50": 21320.0,
                "h1_ema_50": 21380.0,
                "h1_ema_200": 21220.0,
                "m15_ret_1": 0.0014,
                "m15_ret_3": 0.0044,
                "m5_volume_ratio_20": 0.60,
                "m15_volume_ratio_20": 0.72,
                "m5_body_efficiency": 0.26,
                "m15_body_efficiency": 0.30,
                "m15_range_position_20": 0.76,
                "multi_tf_alignment_score": 0.34,
                "fractal_persistence_score": 0.25,
                "seasonality_edge_score": 0.76,
                "market_instability_score": 0.18,
                "feature_drift_score": 0.12,
                "m5_tick_volume": 980.0,
                "m15_tick_volume": 2420.0,
            }
        )

        candidates = router._live_candidate_recovery(  # noqa: SLF001
            symbol="NAS100",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 27, 8, 12, tzinfo=timezone.utc),
        )

        self.assertTrue(candidates)
        self.assertEqual(str(candidates[0].setup), "NAS100_OPENING_DRIVE_BREAKOUT")
        self.assertEqual(str(candidates[0].meta.get("setup_family") or ""), "LIVE_RECOVERY_INDEX")

    def test_live_recovery_prefers_asia_pullback_candidates_for_thin_fx_pairs(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="TRENDING",
            confidence=0.80,
            source="test",
            details={"trend_flag": 1.0},
            state_label="TRENDING",
        )
        session = SessionContext(
            session_name="TOKYO",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 98.84,
                "m15_close": 98.84,
                "m5_spread": 14.0,
                "m5_atr_14": 0.18,
                "m15_atr_14": 0.24,
                "m5_ema_20": 98.80,
                "m5_ema_50": 98.72,
                "m15_ema_20": 98.79,
                "m15_ema_50": 98.70,
                "h1_ema_50": 98.76,
                "h1_ema_200": 98.60,
                "m15_ret_1": 0.0010,
                "m15_ret_3": 0.0055,
                "m5_volume_ratio_20": 0.86,
                "m15_volume_ratio_20": 0.92,
                "m5_body_efficiency": 0.30,
                "m15_body_efficiency": 0.34,
                "m15_range_position_20": 0.61,
                "multi_tf_alignment_score": 0.30,
                "fractal_persistence_score": 0.26,
                "seasonality_edge_score": 0.74,
                "market_instability_score": 0.20,
                "feature_drift_score": 0.12,
                "m5_tick_volume": 0.0,
                "m15_tick_volume": 0.0,
            }
        )

        audjpy = router._live_candidate_recovery(  # noqa: SLF001
            symbol="AUDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 27, 7, 55, tzinfo=timezone.utc),
        )
        audnzd = router._live_candidate_recovery(  # noqa: SLF001
            symbol="AUDNZD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 27, 7, 56, tzinfo=timezone.utc),
        )

        self.assertTrue(audjpy)
        self.assertEqual(str(audjpy[0].setup), "AUDJPY_TOKYO_CONTINUATION_PULLBACK")
        self.assertEqual(str(audjpy[0].meta.get("setup_family") or ""), "LIVE_RECOVERY_ASIA_ROTATION")
        self.assertTrue(audnzd)
        self.assertEqual(str(audnzd[0].setup), "AUDNZD_ASIA_ROTATION_PULLBACK")
        self.assertEqual(str(audnzd[0].meta.get("setup_family") or ""), "LIVE_RECOVERY_ASIA_ROTATION")

    def test_live_recovery_can_emit_range_rotation_candidates_for_london_thin_fx_pairs(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.72,
            source="test",
            details={"range_flag": 1.0},
            state_label="RANGING",
        )
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.95,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 98.82,
                "m15_close": 98.82,
                "m5_spread": 14.0,
                "m5_atr_14": 0.18,
                "m15_atr_14": 0.24,
                "m5_ema_20": 98.80,
                "m5_ema_50": 98.78,
                "m15_ema_20": 98.79,
                "m15_ema_50": 98.76,
                "h1_ema_50": 98.78,
                "h1_ema_200": 98.72,
                "m15_ret_1": 0.0003,
                "m15_ret_3": 0.0011,
                "m5_volume_ratio_20": 0.72,
                "m15_volume_ratio_20": 0.76,
                "m5_body_efficiency": 0.16,
                "m15_body_efficiency": 0.18,
                "m15_range_position_20": 0.58,
                "multi_tf_alignment_score": 0.20,
                "fractal_persistence_score": 0.20,
                "seasonality_edge_score": 0.76,
                "market_instability_score": 0.24,
                "feature_drift_score": 0.16,
                "m5_tick_volume": 0.0,
                "m15_tick_volume": 0.0,
            }
        )

        audjpy = router._live_candidate_recovery(  # noqa: SLF001
            symbol="AUDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 27, 8, 2, tzinfo=timezone.utc),
        )
        nzdjpy = router._live_candidate_recovery(  # noqa: SLF001
            symbol="NZDJPY",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 27, 8, 3, tzinfo=timezone.utc),
        )
        audnzd = router._live_candidate_recovery(  # noqa: SLF001
            symbol="AUDNZD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 27, 8, 4, tzinfo=timezone.utc),
        )

        self.assertTrue(audjpy)
        self.assertEqual(str(audjpy[0].setup), "AUDJPY_TOKYO_CONTINUATION_PULLBACK")
        self.assertEqual(str(audjpy[0].meta.get("live_recovery_mode") or ""), "range_rotation")
        self.assertTrue(nzdjpy)
        self.assertIn(str(nzdjpy[0].setup), {"NZDJPY_TOKYO_CONTINUATION_PULLBACK", "NZDJPY_SWEEP_RECLAIM"})
        self.assertEqual(str(nzdjpy[0].meta.get("live_recovery_mode") or ""), "range_rotation")
        self.assertTrue(audnzd)
        self.assertEqual(str(audnzd[0].setup), "AUDNZD_ASIA_ROTATION_PULLBACK")
        self.assertEqual(str(audnzd[0].meta.get("live_recovery_mode") or ""), "range_rotation")

    def test_live_recovery_can_emit_london_major_candidates_when_base_flow_is_thin(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.70,
            source="test",
            details={"range_flag": 1.0},
            state_label="RANGING",
        )
        session = SessionContext(
            session_name="LONDON",
            in_session=True,
            liquidity_tier="HIGH",
            size_multiplier=1.0,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        row = pd.Series(
            {
                "m5_close": 1.0834,
                "m15_close": 1.0834,
                "m5_spread": 11.0,
                "m5_atr_14": 0.0009,
                "m15_atr_14": 0.0012,
                "m5_ema_20": 1.0832,
                "m5_ema_50": 1.0830,
                "m15_ema_20": 1.0831,
                "m15_ema_50": 1.0829,
                "h1_ema_50": 1.0830,
                "h1_ema_200": 1.0825,
                "m15_ret_1": 0.0002,
                "m15_ret_3": 0.0008,
                "m5_volume_ratio_20": 0.70,
                "m15_volume_ratio_20": 0.74,
                "m5_body_efficiency": 0.24,
                "m15_body_efficiency": 0.26,
                "m15_range_position_20": 0.56,
                "multi_tf_alignment_score": 0.26,
                "fractal_persistence_score": 0.18,
                "seasonality_edge_score": 0.64,
                "market_instability_score": 0.26,
                "feature_drift_score": 0.18,
                "m5_tick_volume": 0.0,
                "m15_tick_volume": 0.0,
            }
        )

        eurusd = router._live_candidate_recovery(  # noqa: SLF001
            symbol="EURUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 27, 8, 6, tzinfo=timezone.utc),
        )
        gbpusd = router._live_candidate_recovery(  # noqa: SLF001
            symbol="GBPUSD",
            row=row,
            regime=regime,
            session=session,
            timestamp=datetime(2026, 3, 27, 8, 7, tzinfo=timezone.utc),
        )

        self.assertTrue(eurusd)
        self.assertEqual(str(eurusd[0].setup), "EURUSD_LONDON_BREAKOUT")
        self.assertIn(str(eurusd[0].meta.get("live_recovery_mode") or ""), {"range_rotation", "london_major_rotation"})
        self.assertTrue(gbpusd)
        self.assertEqual(str(gbpusd[0].setup), "GBPUSD_LONDON_EXPANSION_BREAKOUT")
        self.assertIn(str(gbpusd[0].meta.get("live_recovery_mode") or ""), {"range_rotation", "london_major_rotation"})

    def test_generate_uses_live_recovery_when_base_candidates_are_all_router_rejected(self) -> None:
        router = StrategyRouter(max_spread_points=80.0)
        regime = RegimeClassification(
            label="RANGING",
            confidence=0.78,
            source="test",
            details={"range_flag": 1.0},
            state_label="RANGING",
        )
        session = SessionContext(
            session_name="SYDNEY",
            in_session=True,
            liquidity_tier="LOW",
            size_multiplier=0.8,
            confluence_delta=0,
            ai_threshold_offset=0.0,
            allow_trend=True,
            allow_range=True,
            allow_fakeout=True,
        )
        engine = StrategyEngine(symbol="XAUUSD")
        frame = pd.DataFrame(
            [
                {
                    "time": datetime(2026, 3, 27, 7, 45, tzinfo=timezone.utc),
                    "m5_close": 3052.0,
                    "m15_close": 3052.0,
                    "m5_spread": 12.0,
                    "m5_atr_14": 4.2,
                    "m15_atr_14": 7.0,
                    "m5_ema_20": 3049.0,
                    "m5_ema_50": 3046.5,
                    "m15_ema_20": 3048.0,
                    "m15_ema_50": 3044.5,
                    "h1_ema_50": 3047.0,
                    "h1_ema_200": 3038.0,
                    "m15_ret_1": 0.0012,
                    "m15_ret_3": 0.0036,
                    "m5_volume_ratio_20": 0.62,
                    "m15_volume_ratio_20": 0.74,
                    "m5_body_efficiency": 0.26,
                    "m15_body_efficiency": 0.30,
                    "m15_range_position_20": 0.82,
                    "multi_tf_alignment_score": 0.36,
                    "fractal_persistence_score": 0.29,
                    "seasonality_edge_score": 0.79,
                    "market_instability_score": 0.24,
                    "feature_drift_score": 0.11,
                }
            ]
        )
        base_candidate = SignalCandidate(
            signal_id="base-xau",
            setup="XAUUSD_ATR_EXPANSION_SCALPER",
            side="BUY",
            score_hint=0.62,
            reason="base candidate",
            stop_atr=1.0,
            tp_r=1.7,
            strategy_family="TREND",
            meta={},
        )
        recovery_candidate = SignalCandidate(
            signal_id="recovery-xau",
            setup="XAU_BREAKOUT_RETEST",
            side="BUY",
            score_hint=0.68,
            reason="recovery candidate",
            stop_atr=0.88,
            tp_r=1.78,
            strategy_family="TREND",
            meta={"fallback_live_recovery": True, "setup_family": "LIVE_RECOVERY_XAU"},
        )

        def _enrich_candidate(**kwargs):
            candidate = kwargs["candidate"]
            if candidate.signal_id == "base-xau":
                candidate.meta["router_reject"] = True
                candidate.meta["router_reject_reason"] = "xau_grid_off_session"
            else:
                candidate.meta["router_rank_score"] = 0.74
            return candidate

        with patch.object(router, "_xau_fix_flow", return_value=[base_candidate]), \
            patch.object(router, "_xau_fakeout", return_value=[]), \
            patch.object(router, "_xau_m1_micro_scalper", return_value=[]), \
            patch.object(router, "_xau_m15_structured", return_value=[]), \
            patch.object(router, "_live_candidate_recovery", return_value=[recovery_candidate]), \
            patch.object(router, "_enrich_candidate", side_effect=_enrich_candidate):
            candidates = router.generate(
                symbol="XAUUSD",
                features=frame,
                regime=regime,
                session=session,
                strategy_engine=engine,
                open_positions=[],
                max_positions_per_symbol=1,
                current_time=datetime(2026, 3, 27, 7, 45, tzinfo=timezone.utc),
            )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(str(candidates[0].signal_id), "recovery-xau")
        self.assertTrue(bool(candidates[0].meta.get("fallback_live_recovery")))


if __name__ == "__main__":
    unittest.main()
