from __future__ import annotations

import unittest

from src.trade_quality import (
    delta_proxy_score,
    evaluate_execution_quality,
    evaluate_trade_quality,
    infer_trade_lane,
    is_xau_grid_lane,
    pair_behavior_fit,
    quality_tier_from_scores,
    quality_tier_size_multiplier,
    session_priority_context,
    session_priority_override_decision,
    strategy_allowed_regimes,
    strategy_regime_fit,
    winner_promotion_bonus,
)
from src.strategies.trend_daytrade import resolve_strategy_key


class TradeQualityTests(unittest.TestCase):
    def test_execution_quality_degrades_on_bad_spread_and_stale_rate(self) -> None:
        quality = evaluate_execution_quality(
            spread_points=45.0,
            typical_spread_points=10.0,
            stale_idea_rate=0.4,
            bridge_latency_ms=1800.0,
        )

        self.assertEqual(quality.state, "DEGRADED")
        self.assertTrue(quality.spread_anomaly)
        self.assertTrue(quality.bridge_latency_alert)

    def test_delta_proxy_score_favors_side_aligned_body_and_wick_pressure(self) -> None:
        bullish_buy = delta_proxy_score(
            side="BUY",
            body_efficiency=0.82,
            short_return=0.0012,
            range_position=0.74,
            volume_ratio=1.35,
            upper_wick_ratio=0.10,
            lower_wick_ratio=0.28,
        )
        bearish_buy = delta_proxy_score(
            side="BUY",
            body_efficiency=0.82,
            short_return=-0.0012,
            range_position=0.26,
            volume_ratio=1.35,
            upper_wick_ratio=0.28,
            lower_wick_ratio=0.10,
        )

        self.assertGreater(bullish_buy, 0.0)
        self.assertLess(bearish_buy, 0.0)

    def test_weekend_btc_winner_bonus_hits_printer_lanes(self) -> None:
        continuation = winner_promotion_bonus(
            symbol="BTCUSD",
            strategy_key="BTCUSD_PRICE_ACTION_CONTINUATION",
            regime_state="TRENDING",
            session_name="TOKYO",
            weekend_mode=True,
        )
        retest = winner_promotion_bonus(
            symbol="BTCUSD",
            strategy_key="BTCUSD_VOLATILE_RETEST",
            regime_state="RANGING",
            session_name="TOKYO",
            weekend_mode=True,
        )
        self.assertEqual(continuation, 0.5)
        self.assertEqual(retest, 0.5)

    def test_trade_quality_returns_elite_band_for_aligned_setup(self) -> None:
        quality = evaluate_trade_quality(
            symbol="XAUUSD",
            session_name="LONDON",
            setup_family="GRID",
            regime_state="MEAN_REVERSION",
            regime_confidence=0.86,
            spread_points=18.0,
            spread_limit=60.0,
            volatility_state="COMPRESSION",
            liquidity_score=0.92,
            news_state="NEWS_SAFE",
            news_confidence=0.95,
            structure_score=0.90,
            execution_feasibility=1.0,
            expected_value_r=0.82,
            probability=0.88,
            performance_score=0.76,
            execution_quality_score=0.90,
            pressure_alignment=0.84,
        )

        self.assertEqual(quality.band, "A+")
        self.assertTrue(quality.overflow_eligible)

    def test_trade_quality_uses_range_reversion_alias(self) -> None:
        quality = evaluate_trade_quality(
            symbol="EURUSD",
            session_name="TOKYO",
            setup_family="RANGE",
            regime_state="MEAN_REVERSION",
            regime_confidence=0.72,
            spread_points=10.0,
            spread_limit=35.0,
            volatility_state="BALANCED",
            liquidity_score=0.70,
            news_state="NEWS_CAUTION",
            news_confidence=0.78,
            structure_score=0.68,
            execution_feasibility=0.85,
            expected_value_r=0.22,
            probability=0.64,
            performance_score=0.55,
            execution_quality_score=0.82,
            pressure_alignment=0.58,
        )

        self.assertIn(quality.band, {"A-", "B+", "B"})
        self.assertFalse(quality.should_skip)

    def test_session_priority_context_prefers_asia_native_pairs_in_tokyo(self) -> None:
        audjpy = session_priority_context(symbol="AUDJPY", lane_name="FX_SESSION_SCALP", session_name="TOKYO")
        eurusd = session_priority_context(symbol="EURUSD", lane_name="FX_DAYTRADE", session_name="TOKYO")

        self.assertTrue(audjpy.session_native_pair)
        self.assertGreater(audjpy.session_priority_multiplier, eurusd.session_priority_multiplier)
        self.assertLess(audjpy.pair_priority_rank_in_session, eurusd.pair_priority_rank_in_session)

    def test_session_priority_context_prefers_xau_grid_in_london(self) -> None:
        xau_grid = session_priority_context(
            symbol="XAUUSD",
            lane_name=infer_trade_lane(symbol="XAUUSD", setup="XAUUSD_M5_GRID_SCALPER_START", setup_family="GRID", session_name="LONDON"),
            session_name="LONDON",
        )
        audjpy = session_priority_context(symbol="AUDJPY", lane_name="FX_SESSION_SCALP", session_name="LONDON")

        self.assertTrue(xau_grid.session_native_pair)
        self.assertGreater(xau_grid.session_priority_multiplier, audjpy.session_priority_multiplier)

    def test_infer_trade_lane_splits_xau_attack_by_session(self) -> None:
        london_lane = infer_trade_lane(symbol="XAUUSD", setup="XAUUSD_ADAPTIVE_M5_GRID", setup_family="GRID", session_name="LONDON")
        overlap_lane = infer_trade_lane(symbol="XAUUSD", setup="XAUUSD_M5_GRID_SCALPER_START", setup_family="GRID", session_name="OVERLAP")
        new_york_lane = infer_trade_lane(symbol="XAUUSD", setup="XAUUSD_ADAPTIVE_M5_GRID", setup_family="GRID", session_name="NEW_YORK")

        self.assertEqual(london_lane, "XAU_LONDON_ATTACK")
        self.assertEqual(overlap_lane, "XAU_OVERLAP_ATTACK")
        self.assertEqual(new_york_lane, "XAU_NEW_YORK_ATTACK")
        self.assertTrue(is_xau_grid_lane(london_lane))

    def test_non_native_candidate_is_blocked_when_native_score_is_close(self) -> None:
        decision = session_priority_override_decision(
            session_name="TOKYO",
            candidate_symbol="EURUSD",
            candidate_band="B+",
            candidate_adjusted_score=0.78,
            candidate_probability=0.70,
            candidate_native=False,
            candidate_lane_priority="SECONDARY",
            candidate_override_delta=0.06,
            candidate_override_band_delta=0.03,
            best_native_symbol="AUDJPY",
            best_native_band="B+",
            best_native_adjusted_score=0.76,
            best_native_probability=0.69,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("AUDJPY", decision.why_native_pair_lost_priority)

    def test_non_native_candidate_can_override_with_stronger_band(self) -> None:
        decision = session_priority_override_decision(
            session_name="TOKYO",
            candidate_symbol="EURUSD",
            candidate_band="A",
            candidate_adjusted_score=0.83,
            candidate_probability=0.74,
            candidate_native=False,
            candidate_lane_priority="SECONDARY",
            candidate_override_delta=0.06,
            candidate_override_band_delta=0.03,
            best_native_symbol="AUDJPY",
            best_native_band="B+",
            best_native_adjusted_score=0.79,
            best_native_probability=0.71,
        )

        self.assertTrue(decision.allowed)
        self.assertTrue(decision.exceptional_override_used)
        self.assertEqual(decision.exceptional_override_reason, "stronger_band_override")

    def test_btc_can_share_flow_as_high_grade_always_on_lane(self) -> None:
        decision = session_priority_override_decision(
            session_name="TOKYO",
            candidate_symbol="BTCUSD",
            candidate_band="A",
            candidate_adjusted_score=0.80,
            candidate_probability=0.82,
            candidate_native=False,
            candidate_lane_priority="OFF_SESSION",
            candidate_override_delta=0.06,
            candidate_override_band_delta=0.03,
            best_native_symbol="AUDJPY",
            best_native_band="A",
            best_native_adjusted_score=0.84,
            best_native_probability=0.83,
        )

        self.assertTrue(decision.allowed)
        self.assertTrue(decision.exceptional_override_used)
        self.assertEqual(decision.exceptional_override_reason, "always_on_btc_shared_flow")

    def test_london_secondary_lane_needs_clear_superiority(self) -> None:
        decision = session_priority_override_decision(
            session_name="LONDON",
            candidate_symbol="NAS100",
            candidate_band="B+",
            candidate_adjusted_score=0.76,
            candidate_probability=0.69,
            candidate_native=False,
            candidate_lane_priority="SECONDARY",
            candidate_override_delta=0.06,
            candidate_override_band_delta=0.03,
            best_native_symbol="EURUSD",
            best_native_band="B+",
            best_native_adjusted_score=0.79,
            best_native_probability=0.71,
        )

        self.assertFalse(decision.allowed)
        self.assertIn("EURUSD", decision.why_native_pair_lost_priority)

    def test_london_secondary_lane_can_override_with_clear_score_superiority(self) -> None:
        decision = session_priority_override_decision(
            session_name="LONDON",
            candidate_symbol="NAS100",
            candidate_band="A",
            candidate_adjusted_score=0.88,
            candidate_probability=0.78,
            candidate_native=False,
            candidate_lane_priority="SECONDARY",
            candidate_override_delta=0.06,
            candidate_override_band_delta=0.03,
            best_native_symbol="EURUSD",
            best_native_band="B+",
            best_native_adjusted_score=0.80,
            best_native_probability=0.72,
        )

        self.assertTrue(decision.allowed)
        self.assertTrue(decision.exceptional_override_used)
        self.assertIn(decision.exceptional_override_reason, {"score_superiority_override", "stronger_band_override"})

    def test_throughput_recovery_can_share_flow_with_near_tied_native_candidate(self) -> None:
        decision = session_priority_override_decision(
            session_name="OVERLAP",
            candidate_symbol="AUDJPY",
            candidate_band="A-",
            candidate_adjusted_score=0.81,
            candidate_probability=0.73,
            candidate_native=False,
            candidate_lane_priority="SECONDARY",
            candidate_override_delta=0.06,
            candidate_override_band_delta=0.03,
            best_native_symbol="BTCUSD",
            best_native_band="A-",
            best_native_adjusted_score=0.82,
            best_native_probability=0.74,
            throughput_recovery_active=True,
            trajectory_catchup_pressure=0.86,
        )

        self.assertTrue(decision.allowed)
        self.assertTrue(decision.exceptional_override_used)
        self.assertEqual(decision.exceptional_override_reason, "throughput_recovery_override")

    def test_throughput_recovery_can_share_flow_with_strong_a_minus_candidate_below_a_plus_leader(self) -> None:
        decision = session_priority_override_decision(
            session_name="OVERLAP",
            candidate_symbol="NZDJPY",
            candidate_band="A-",
            candidate_adjusted_score=0.79,
            candidate_probability=0.72,
            candidate_native=False,
            candidate_lane_priority="SECONDARY",
            candidate_override_delta=0.06,
            candidate_override_band_delta=0.03,
            best_native_symbol="NAS100",
            best_native_band="A+",
            best_native_adjusted_score=0.84,
            best_native_probability=0.74,
            throughput_recovery_active=True,
            trajectory_catchup_pressure=0.9,
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.exceptional_override_reason, "throughput_recovery_override")

    def test_prime_session_density_pressure_can_share_flow_without_native_lockout(self) -> None:
        decision = session_priority_override_decision(
            session_name="LONDON",
            candidate_symbol="GBPUSD",
            candidate_band="A-",
            candidate_adjusted_score=0.80,
            candidate_probability=0.73,
            candidate_native=False,
            candidate_lane_priority="SECONDARY",
            candidate_override_delta=0.06,
            candidate_override_band_delta=0.03,
            best_native_symbol="XAUUSD",
            best_native_band="A-",
            best_native_adjusted_score=0.82,
            best_native_probability=0.74,
            throughput_recovery_active=False,
            trajectory_catchup_pressure=0.68,
        )

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.exceptional_override_reason, "prime_session_shared_flow")

    def test_gbpusd_trend_pullback_is_hard_demoted_in_london_chop(self) -> None:
        self.assertLess(strategy_regime_fit("GBPUSD_TREND_PULLBACK_RIDE", "RANGING"), 0.10)
        self.assertLess(strategy_regime_fit("GBPUSD_TREND_PULLBACK_RIDE", "LOW_LIQUIDITY_CHOP"), 0.05)
        self.assertLess(strategy_regime_fit("GBPUSD_TREND_PULLBACK_RIDE", "MEAN_REVERSION"), 0.05)

    def test_strategy_allowed_regimes_uses_specific_bucket_overrides(self) -> None:
        self.assertEqual(
            strategy_allowed_regimes("USDJPY_MACRO_TREND_RIDE"),
            ("TRENDING",),
        )
        self.assertEqual(
            strategy_allowed_regimes("EURJPY_RANGE_FADE"),
            ("RANGING", "MEAN_REVERSION"),
        )
        self.assertEqual(
            strategy_allowed_regimes("GBPJPY_SESSION_PULLBACK_CONTINUATION"),
            ("TRENDING",),
        )
        self.assertEqual(
            strategy_allowed_regimes("AUDNZD_RANGE_ROTATION"),
            ("RANGING", "MEAN_REVERSION"),
        )
        self.assertEqual(
            strategy_allowed_regimes("BTCUSD_RANGE_EXPANSION"),
            ("BREAKOUT_EXPANSION",),
        )

    def test_resolve_strategy_key_maps_legacy_session_names_for_jpy_pairs(self) -> None:
        self.assertEqual(
            resolve_strategy_key("AUDJPY", "AUDJPY_SESSION_PULLBACK"),
            "AUDJPY_TOKYO_CONTINUATION_PULLBACK",
        )
        self.assertEqual(
            resolve_strategy_key("NZDJPY", "NZDJPY_SESSION_PULLBACK"),
            "NZDJPY_PULLBACK_CONTINUATION",
        )
        self.assertEqual(
            resolve_strategy_key("USDJPY", "USDJPY_SESSION_MOMENTUM"),
            "USDJPY_MOMENTUM_IMPULSE",
        )
        self.assertEqual(
            resolve_strategy_key("EURJPY", "EURJPY_SESSION_PULLBACK"),
            "EURJPY_SESSION_PULLBACK_CONTINUATION",
        )
        self.assertEqual(
            resolve_strategy_key("GBPJPY", "GBPJPY_SWEEP_RECLAIM"),
            "GBPJPY_LIQUIDITY_SWEEP_REVERSAL",
        )
        self.assertEqual(
            resolve_strategy_key("XAUUSD", "XAU_FAKEOUT_FADE"),
            "XAUUSD_LONDON_LIQUIDITY_SWEEP",
        )

    def test_resolve_strategy_key_maps_audnzd_legacy_range_reversion_to_current_buckets(self) -> None:
        self.assertEqual(
            resolve_strategy_key("AUDNZD", "FOREX_RANGE_REVERSION"),
            "AUDNZD_VWAP_MEAN_REVERSION",
        )
        self.assertEqual(
            resolve_strategy_key("AUDNZD", "AUDNZD_FADE"),
            "AUDNZD_VWAP_MEAN_REVERSION",
        )
        self.assertEqual(
            resolve_strategy_key("AUDNZD", "AUDNZD_RANGE_REJECTION"),
            "AUDNZD_RANGE_ROTATION",
        )

    def test_resolve_strategy_key_maps_nas_and_btc_router_setups_to_active_buckets(self) -> None:
        self.assertEqual(
            resolve_strategy_key("NAS100", "NAS_SESSION_SCALPER_ORB"),
            "NAS100_OPENING_DRIVE_BREAKOUT",
        )
        self.assertEqual(
            resolve_strategy_key("NAS100", "NAS_SESSION_SCALPER_VWAP_MR"),
            "NAS100_LIQUIDITY_SWEEP_REVERSAL",
        )
        self.assertEqual(
            resolve_strategy_key("NAS100", "NAS_PREMARKET_VOLUME_CONFIRM"),
            "NAS100_OPENING_DRIVE_BREAKOUT",
        )
        self.assertEqual(
            resolve_strategy_key("BTCUSD", "BTC_MOMENTUM_CONTINUATION"),
            "BTCUSD_PRICE_ACTION_CONTINUATION",
        )
        self.assertEqual(
            resolve_strategy_key("BTCUSD", "BTC_NY_LIQUIDITY"),
            "BTCUSD_PRICE_ACTION_CONTINUATION",
        )
        self.assertEqual(
            resolve_strategy_key("BTCUSD", "BTC_LIQUIDATION_FADE"),
            "BTCUSD_VOLATILE_RETEST",
        )
        self.assertEqual(
            resolve_strategy_key("BTCUSD", "BTC_WHALE_FLOW_BREAKOUT"),
            "BTCUSD_RANGE_EXPANSION",
        )

    def test_pair_behavior_fit_demotes_known_ranging_loser_buckets(self) -> None:
        eurusd_range = pair_behavior_fit(
            symbol="EURUSD",
            strategy_key="EURUSD_RANGE_FADE",
            session_name="TOKYO",
            regime_state="RANGING",
        )
        usdjpy_tokyo = pair_behavior_fit(
            symbol="USDJPY",
            strategy_key="USDJPY_MOMENTUM_IMPULSE",
            session_name="TOKYO",
            regime_state="RANGING",
        )
        audnzd_rotation = pair_behavior_fit(
            symbol="AUDNZD",
            strategy_key="AUDNZD_RANGE_ROTATION",
            session_name="TOKYO",
            regime_state="RANGING",
        )
        usdjpy_sweep = pair_behavior_fit(
            symbol="USDJPY",
            strategy_key="USDJPY_LIQUIDITY_SWEEP_REVERSAL",
            session_name="TOKYO",
            regime_state="RANGING",
        )

        self.assertLess(eurusd_range, 0.41)
        self.assertLess(usdjpy_tokyo, 0.40)
        self.assertGreater(audnzd_rotation, eurusd_range)
        self.assertGreater(usdjpy_sweep, usdjpy_tokyo)

    def test_pair_behavior_fit_penalizes_tokyo_mean_reversion_loser_clusters(self) -> None:
        eurusd_tokyo = pair_behavior_fit(
            symbol="EURUSD",
            strategy_key="EURUSD_LIQUIDITY_SWEEP",
            session_name="TOKYO",
            regime_state="MEAN_REVERSION",
        )
        audnzd_tokyo = pair_behavior_fit(
            symbol="AUDNZD",
            strategy_key="AUDNZD_VWAP_MEAN_REVERSION",
            session_name="TOKYO",
            regime_state="MEAN_REVERSION",
        )
        usoil_tokyo = pair_behavior_fit(
            symbol="USOIL",
            strategy_key="USOIL_LONDON_TREND_EXPANSION",
            session_name="TOKYO",
            regime_state="LOW_LIQUIDITY_CHOP",
        )

        self.assertLess(eurusd_tokyo, 0.38)
        self.assertLess(audnzd_tokyo, 0.44)
        self.assertLess(usoil_tokyo, 0.52)

    def test_pair_behavior_fit_penalizes_new_ranging_loser_clusters(self) -> None:
        audjpy_sydney = pair_behavior_fit(
            symbol="AUDJPY",
            strategy_key="AUDJPY_TOKYO_MOMENTUM_BREAKOUT",
            session_name="SYDNEY",
            regime_state="RANGING",
        )
        nzdjpy_tokyo = pair_behavior_fit(
            symbol="NZDJPY",
            strategy_key="NZDJPY_PULLBACK_CONTINUATION",
            session_name="TOKYO",
            regime_state="RANGING",
        )
        audnzd_reversion = pair_behavior_fit(
            symbol="AUDNZD",
            strategy_key="AUDNZD_VWAP_MEAN_REVERSION",
            session_name="TOKYO",
            regime_state="RANGING",
        )
        audnzd_break = pair_behavior_fit(
            symbol="AUDNZD",
            strategy_key="AUDNZD_STRUCTURE_BREAK_RETEST",
            session_name="TOKYO",
            regime_state="RANGING",
        )

        self.assertLess(audjpy_sydney, 0.42)
        self.assertLess(nzdjpy_tokyo, 0.42)
        self.assertLess(audnzd_reversion, audnzd_break)

    def test_pair_behavior_fit_penalizes_tokyo_secondary_jpy_range_leaks(self) -> None:
        eurjpy_momentum = pair_behavior_fit(
            symbol="EURJPY",
            strategy_key="EURJPY_MOMENTUM_IMPULSE",
            session_name="TOKYO",
            regime_state="RANGING",
        )
        eurjpy_sweep = pair_behavior_fit(
            symbol="EURJPY",
            strategy_key="EURJPY_LIQUIDITY_SWEEP_REVERSAL",
            session_name="TOKYO",
            regime_state="RANGING",
        )
        gbpjpy_pullback = pair_behavior_fit(
            symbol="GBPJPY",
            strategy_key="GBPJPY_SESSION_PULLBACK_CONTINUATION",
            session_name="TOKYO",
            regime_state="MEAN_REVERSION",
        )

        self.assertLess(eurjpy_momentum, 0.36)
        self.assertGreater(eurjpy_sweep, eurjpy_momentum)
        self.assertLess(gbpjpy_pullback, 0.36)

    def test_pair_behavior_fit_penalizes_btc_london_trend_scalp(self) -> None:
        btc_london = pair_behavior_fit(
            symbol="BTCUSD",
            strategy_key="BTCUSD_TREND_SCALP",
            session_name="LONDON",
            regime_state="TRENDING",
        )
        btc_overlap = pair_behavior_fit(
            symbol="BTCUSD",
            strategy_key="BTCUSD_RANGE_EXPANSION",
            session_name="OVERLAP",
            regime_state="BREAKOUT_EXPANSION",
        )

        self.assertLess(btc_london, btc_overlap)

    def test_pair_behavior_fit_promotes_weekend_btc_retests_over_weekday(self) -> None:
        weekday = pair_behavior_fit(
            symbol="BTCUSD",
            strategy_key="BTCUSD_VOLATILE_RETEST",
            session_name="TOKYO",
            regime_state="MEAN_REVERSION",
            weekend_mode=False,
        )
        weekend = pair_behavior_fit(
            symbol="BTCUSD",
            strategy_key="BTCUSD_VOLATILE_RETEST",
            session_name="TOKYO",
            regime_state="MEAN_REVERSION",
            weekend_mode=True,
        )

        self.assertGreater(weekend, weekday)

    def test_delta_proxy_and_quality_tier_helpers_bias_toward_clean_expansion(self) -> None:
        delta = delta_proxy_score(
            side="BUY",
            body_efficiency=0.82,
            short_return=0.0018,
            range_position=0.84,
            volume_ratio=1.35,
            upper_wick_ratio=0.08,
            lower_wick_ratio=0.26,
        )
        tier = quality_tier_from_scores(
            structure_cleanliness=0.83,
            regime_fit=0.80,
            execution_quality_fit=0.74,
            high_liquidity=True,
            throughput_recovery_active=False,
        )
        size_mult = quality_tier_size_multiplier(
            quality_tier=tier,
            strategy_score=0.88,
        )

        self.assertGreater(delta, 0.0)
        self.assertEqual(tier, "A+")
        self.assertAlmostEqual(size_mult, 1.0, places=6)

    def test_delta_proxy_penalizes_wrong_side_wick_pressure(self) -> None:
        supportive = delta_proxy_score(
            side="BUY",
            body_efficiency=0.74,
            short_return=0.0011,
            range_position=0.72,
            volume_ratio=1.20,
            upper_wick_ratio=0.06,
            lower_wick_ratio=0.24,
        )
        hostile = delta_proxy_score(
            side="BUY",
            body_efficiency=0.74,
            short_return=0.0011,
            range_position=0.72,
            volume_ratio=1.20,
            upper_wick_ratio=0.24,
            lower_wick_ratio=0.06,
        )

        self.assertGreater(supportive, hostile)

    def test_strategy_regime_fit_penalizes_exact_loser_clusters(self) -> None:
        self.assertEqual(strategy_regime_fit("USDJPY_MOMENTUM_IMPULSE", "RANGING"), 0.05)
        self.assertEqual(strategy_regime_fit("USDJPY_MACRO_TREND_RIDE", "RANGING"), 0.05)
        self.assertEqual(strategy_regime_fit("USDJPY_LIQUIDITY_SWEEP_REVERSAL", "RANGING"), 0.72)
        self.assertEqual(strategy_regime_fit("EURJPY_MOMENTUM_IMPULSE", "RANGING"), 0.10)
        self.assertEqual(strategy_regime_fit("GBPJPY_SESSION_PULLBACK_CONTINUATION", "MEAN_REVERSION"), 0.04)
        self.assertEqual(strategy_regime_fit("EURJPY_LIQUIDITY_SWEEP_REVERSAL", "RANGING"), 0.68)
        self.assertEqual(strategy_regime_fit("EURUSD_LONDON_BREAKOUT", "RANGING"), 0.12)
        self.assertEqual(strategy_regime_fit("EURUSD_LONDON_BREAKOUT", "LOW_LIQUIDITY_CHOP"), 0.12)
        self.assertEqual(strategy_regime_fit("EURUSD_RANGE_FADE", "RANGING"), 0.18)
        self.assertEqual(strategy_regime_fit("AUDNZD_STRUCTURE_BREAK_RETEST", "MEAN_REVERSION"), 0.12)
        self.assertEqual(strategy_regime_fit("AUDNZD_RANGE_ROTATION", "RANGING"), 0.82)
        self.assertEqual(strategy_regime_fit("AUDNZD_VWAP_MEAN_REVERSION", "MEAN_REVERSION"), 0.28)
        self.assertEqual(strategy_regime_fit("AUDJPY_TOKYO_MOMENTUM_BREAKOUT", "RANGING"), 0.14)
        self.assertEqual(strategy_regime_fit("AUDJPY_TOKYO_CONTINUATION_PULLBACK", "RANGING"), 0.05)
        self.assertEqual(strategy_regime_fit("NZDJPY_PULLBACK_CONTINUATION", "RANGING"), 0.05)
        self.assertEqual(strategy_regime_fit("AUDJPY_ATR_COMPRESSION_BREAKOUT", "LOW_LIQUIDITY_CHOP"), 0.14)
        self.assertEqual(strategy_regime_fit("NZDJPY_SESSION_RANGE_EXPANSION", "RANGING"), 0.04)
        self.assertEqual(strategy_regime_fit("XAUUSD_ATR_EXPANSION_SCALPER", "RANGING"), 0.18)
        self.assertEqual(strategy_regime_fit("XAUUSD_ADAPTIVE_M5_GRID", "MEAN_REVERSION"), 0.52)
        self.assertEqual(strategy_regime_fit("XAUUSD_LONDON_LIQUIDITY_SWEEP", "RANGING"), 0.58)
        self.assertEqual(strategy_regime_fit("XAUUSD_VWAP_REVERSION", "MEAN_REVERSION"), 0.86)
        self.assertEqual(strategy_regime_fit("USOIL_LONDON_TREND_EXPANSION", "LOW_LIQUIDITY_CHOP"), 0.18)
        self.assertEqual(strategy_regime_fit("USOIL_INVENTORY_MOMENTUM", "RANGING"), 0.08)
        self.assertEqual(strategy_regime_fit("BTCUSD_TREND_SCALP", "RANGING"), 0.08)
        self.assertEqual(strategy_regime_fit("BTCUSD_RANGE_EXPANSION", "RANGING"), 0.06)
        self.assertEqual(strategy_regime_fit("EURUSD_VWAP_PULLBACK", "LOW_LIQUIDITY_CHOP"), 0.04)
        self.assertEqual(strategy_regime_fit("EURUSD_VWAP_PULLBACK", "RANGING"), 0.08)
        self.assertEqual(strategy_regime_fit("XAUUSD_LONDON_LIQUIDITY_SWEEP", "MEAN_REVERSION"), 0.84)
        self.assertEqual(strategy_regime_fit("NAS100_LIQUIDITY_SWEEP_REVERSAL", "MEAN_REVERSION"), 0.72)
        self.assertEqual(strategy_regime_fit("NAS100_MOMENTUM_IMPULSE", "TRENDING"), 0.34)
        self.assertEqual(strategy_regime_fit("EURUSD_LIQUIDITY_SWEEP", "MEAN_REVERSION"), 0.06)
        self.assertEqual(strategy_regime_fit("BTCUSD_RANGE_EXPANSION", "LOW_LIQUIDITY_CHOP"), 0.02)

    def test_pair_behavior_fit_penalizes_london_local_loser_clusters(self) -> None:
        eurusd_london_sweep = pair_behavior_fit(
            symbol="EURUSD",
            strategy_key="EURUSD_LIQUIDITY_SWEEP",
            session_name="LONDON",
            regime_state="MEAN_REVERSION",
        )
        eurusd_london_vwap = pair_behavior_fit(
            symbol="EURUSD",
            strategy_key="EURUSD_VWAP_PULLBACK",
            session_name="LONDON",
            regime_state="LOW_LIQUIDITY_CHOP",
        )
        xau_london_breakout = pair_behavior_fit(
            symbol="XAUUSD",
            strategy_key="XAUUSD_NY_MOMENTUM_BREAKOUT",
            session_name="LONDON",
            regime_state="TRENDING",
        )
        xau_grid_london = pair_behavior_fit(
            symbol="XAUUSD",
            strategy_key="XAUUSD_ADAPTIVE_M5_GRID",
            session_name="LONDON",
            regime_state="MEAN_REVERSION",
        )
        xau_london_sweep_ranging = pair_behavior_fit(
            symbol="XAUUSD",
            strategy_key="XAUUSD_LONDON_LIQUIDITY_SWEEP",
            session_name="LONDON",
            regime_state="RANGING",
        )

        self.assertLess(eurusd_london_sweep, 0.30)
        self.assertLess(eurusd_london_vwap, 0.28)
        self.assertLess(xau_london_breakout, 0.40)
        self.assertLess(xau_london_sweep_ranging, 0.32)
        self.assertGreater(xau_grid_london, xau_london_breakout)
        self.assertGreater(xau_grid_london, xau_london_sweep_ranging)


if __name__ == "__main__":
    unittest.main()
