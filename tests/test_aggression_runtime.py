from __future__ import annotations

from datetime import datetime, timezone
import unittest

from src.aggression_runtime import (
    build_event_directive,
    build_execution_minute_profile,
    build_lead_lag_snapshot,
    build_microstructure_score,
    score_shadow_variant,
)


class AggressionRuntimeTests(unittest.TestCase):
    def test_build_microstructure_score_enriches_snapshot(self) -> None:
        score = build_microstructure_score(
            {
                "ready": True,
                "pressure_score": 0.82,
                "cumulative_delta_score": 0.55,
                "depth_imbalance": 0.48,
                "dom_imbalance": 0.52,
                "drift_score": 0.44,
                "spread_stability": 0.84,
                "tick_count": 96,
                "book_levels": 5,
            },
            symbol="XAUUSD",
            side="BUY",
        )

        self.assertTrue(score.ready)
        self.assertEqual(score.direction, "bullish")
        self.assertGreater(score.confidence, 0.5)
        self.assertGreater(score.alignment_score, 0.0)
        self.assertFalse(score.stale)

    def test_build_microstructure_score_marks_stale_when_data_is_thin(self) -> None:
        score = build_microstructure_score(
            {
                "ready": True,
                "pressure_score": 0.55,
                "cumulative_delta_score": 0.10,
                "depth_imbalance": 0.05,
                "tick_count": 4,
                "book_levels": 0,
            },
            symbol="BTCUSD",
            side="SELL",
        )

        self.assertTrue(score.stale)
        self.assertLessEqual(score.confidence, 0.5)

    def test_build_lead_lag_snapshot_uses_phase_one_matrix(self) -> None:
        snapshot = build_lead_lag_snapshot(
            symbol="BTCUSD",
            side="BUY",
            context={
                "nas100_ret_5": 0.45,
                "dxy_ret_5": -0.30,
                "usd_liquidity_score": 0.35,
                "weekend_volatility_score": 0.10,
            },
        )

        self.assertEqual(snapshot.symbol, "BTCUSD")
        self.assertEqual(snapshot.direction, "bullish")
        self.assertGreater(snapshot.alignment_score, 0.0)
        self.assertGreater(snapshot.confidence, 0.3)

    def test_build_lead_lag_snapshot_supports_equity_and_jpy_defaults(self) -> None:
        equity_snapshot = build_lead_lag_snapshot(
            symbol="AAPL",
            side="BUY",
            context={
                "nas100_ret_5": 0.42,
                "dxy_ret_5": -0.12,
                "us10y_ret_5": -0.18,
                "usd_liquidity_score": 0.24,
                "risk_sentiment_score": 0.38,
            },
        )
        jpy_snapshot = build_lead_lag_snapshot(
            symbol="AUDJPY",
            side="BUY",
            context={
                "nas100_ret_5": 0.30,
                "us10y_ret_5": 0.22,
                "risk_sentiment_score": 0.36,
                "dxy_ret_5": -0.08,
            },
        )

        self.assertEqual(equity_snapshot.direction, "bullish")
        self.assertGreater(equity_snapshot.alignment_score, 0.0)
        self.assertIn("nas100", equity_snapshot.weights_used)
        self.assertEqual(jpy_snapshot.direction, "bullish")
        self.assertGreater(jpy_snapshot.alignment_score, 0.0)
        self.assertIn("fx_spread", jpy_snapshot.weights_used)

    def test_build_event_directive_prefers_wait_then_retest_for_top_tier_macro(self) -> None:
        lead_lag = build_lead_lag_snapshot(
            symbol="XAUUSD",
            side="BUY",
            context={"dxy_ret_5": -0.30, "us10y_ret_5": -0.25, "nas100_ret_5": -0.10, "usoil_ret_5": 0.05},
        )
        micro = build_microstructure_score(
            {
                "ready": True,
                "pressure_score": 0.74,
                "cumulative_delta_score": 0.32,
                "depth_imbalance": 0.28,
                "dom_imbalance": 0.30,
                "drift_score": 0.25,
                "spread_stability": 0.82,
                "tick_count": 64,
                "book_levels": 4,
            },
            symbol="XAUUSD",
            side="BUY",
        )
        directive = build_event_directive(
            symbol="XAUUSD",
            news_snapshot={
                "news_primary_category": "inflation",
                "news_confidence": 0.82,
                "news_bias_direction": "bullish",
                "next_macro_event": {"title": "US CPI"},
                "event_risk_window_active": True,
                "news_source_used": "finnhub",
            },
            lead_lag=lead_lag,
            microstructure=micro,
        )

        self.assertEqual(directive.playbook, "wait_then_retest")
        self.assertTrue(directive.pre_position_allowed)
        self.assertEqual(directive.wait_minutes_after, 3)

    def test_build_event_directive_uses_breakout_for_equity_earnings(self) -> None:
        directive = build_event_directive(
            symbol="NVIDIA",
            news_snapshot={
                "news_primary_category": "equity_earnings",
                "news_confidence": 0.80,
                "news_bias_direction": "bullish",
                "next_macro_event": {"title": "NVIDIA earnings and AI chip guidance"},
                "event_risk_window_active": True,
                "news_source_used": "rss",
            },
        )

        self.assertEqual(directive.playbook, "breakout")
        self.assertIn("NVIDIA", directive.affected_symbols)
        self.assertIn("NAS100", directive.affected_symbols)

    def test_build_execution_minute_profile_prefers_clean_prime_window(self) -> None:
        profile = build_execution_minute_profile(
            now_utc=datetime(2026, 3, 24, 13, 5, tzinfo=timezone.utc),
            runtime={
                "session_name": "LONDON",
                "spread_quality_score": 0.86,
                "slippage_quality_score": 0.82,
            },
            management_feedback={"active_management_ratio": 0.78},
        )

        self.assertEqual(profile.state, "CLEAN")
        self.assertGreater(profile.size_multiplier, 1.0)

    def test_score_shadow_variant_promotes_high_quality_variant(self) -> None:
        variant_score = score_shadow_variant(
            {"variant_id": "XAUUSD_SHADOW_1", "score_hint": 0.84},
            performance={"expectancy_r": 0.48, "profit_factor": 1.42, "slippage_quality_score": 0.88},
            promotion_threshold=0.64,
        )

        self.assertTrue(variant_score.promoted)
        self.assertGreaterEqual(variant_score.promotion_score, 0.64)


if __name__ == "__main__":
    unittest.main()
