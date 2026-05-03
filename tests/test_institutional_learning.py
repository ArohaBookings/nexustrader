from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.institutional_learning import (
    SelfEvolutionEngine,
    adaptive_aggression_multiplier,
    build_candle_mastery_from_row,
    build_institutional_confluence,
)


class InstitutionalLearningTests(unittest.TestCase):
    def test_candle_mastery_detects_liquidity_and_structure_context(self) -> None:
        row = {
            "m5_body_efficiency": 0.68,
            "m5_upper_wick_ratio": 0.18,
            "m5_lower_wick_ratio": 0.42,
            "m5_trend_efficiency_16": 0.74,
            "m5_volume_ratio_20": 1.9,
            "m5_candle_direction": 1.0,
            "m5_liquidity_sweep": 1.0,
            "m5_fvg_bull": 1.0,
            "m5_bos_bull": 1.0,
            "m5_choch_bull": 1.0,
            "m5_absorption_proxy": 0.60,
            "m5_order_block_bull": 1.0,
            "m5_volume_profile_pressure": 0.80,
            "m5_vwap_deviation_atr": 0.35,
        }

        candle = build_candle_mastery_from_row(row, side="BUY", symbol="BTCUSD")

        self.assertEqual(candle.symbol, "BTCUSD")
        self.assertEqual(candle.direction, "bullish")
        self.assertGreater(candle.mastery_score, 0.50)
        self.assertEqual(candle.liquidity_sweep_score, 1.0)
        self.assertEqual(candle.bos_score, 1.0)
        self.assertGreater(candle.alignment_score, 0.0)

    def test_confluence_throttles_when_live_shadow_gap_is_high(self) -> None:
        row = {
            "m5_body_efficiency": 0.80,
            "m5_trend_efficiency_16": 0.75,
            "m5_volume_ratio_20": 1.5,
            "m5_candle_direction": 1.0,
            "m5_bos_bull": 1.0,
            "m5_fvg_bull": 1.0,
            "m5_volume_profile_pressure": 0.8,
            "multi_tf_alignment_score": 0.88,
            "fractal_persistence_score": 0.72,
            "compression_expansion_score": 0.62,
            "session_aggression_score": 0.90,
            "market_instability_score": 0.15,
            "execution_edge_score": 0.82,
        }
        clean = build_institutional_confluence(row, symbol="BTCUSD", side="BUY")
        throttled = build_institutional_confluence(
            {**row, "live_shadow_gap_risk_score": 0.75},
            symbol="BTCUSD",
            side="BUY",
        )

        self.assertGreater(clean.score, throttled.score)
        self.assertGreater(clean.aggression_multiplier, throttled.aggression_multiplier)
        self.assertIn("live_shadow_gap_throttle", throttled.reasons)
        self.assertLess(throttled.risk_throttle, 1.0)

    def test_adaptive_aggression_is_capped_and_drawdown_sensitive(self) -> None:
        strong = adaptive_aggression_multiplier(
            confluence_score=0.95,
            execution_edge=0.90,
            live_shadow_gap=0.0,
            market_instability=0.0,
            equity_curve_heat=0.0,
        )
        heated = adaptive_aggression_multiplier(
            confluence_score=0.95,
            execution_edge=0.90,
            live_shadow_gap=0.60,
            market_instability=0.60,
            equity_curve_heat=0.80,
        )

        self.assertLessEqual(strong, 1.35)
        self.assertGreater(strong, heated)
        self.assertGreaterEqual(heated, 0.55)

    def test_self_evolution_review_records_guardrailed_proposals_and_memory(self) -> None:
        with TemporaryDirectory() as tmp_dir:
            engine = SelfEvolutionEngine(Path(tmp_dir) / "evolution.sqlite3")
            review = engine.review(
                live_trades=[
                    {
                        "symbol": "BTCUSD",
                        "strategy_key": "BTC_SWEEP",
                        "session_name": "LONDON",
                        "regime_state": "BREAKOUT",
                        "pnl_r": -1.0,
                        "institutional_confluence_score": 0.40,
                        "candle_mastery_score": 0.35,
                        "execution_quality_fit": 0.44,
                        "live_shadow_gap_score": 0.45,
                    },
                    {
                        "symbol": "BTCUSD",
                        "strategy_key": "BTC_SWEEP",
                        "session_name": "LONDON",
                        "regime_state": "BREAKOUT",
                        "pnl_r": -0.5,
                        "institutional_confluence_score": 0.45,
                        "candle_mastery_score": 0.40,
                        "execution_quality_fit": 0.50,
                    },
                    {
                        "symbol": "ETHUSD",
                        "strategy_key": "ETH_CONTINUATION",
                        "session_name": "NEW_YORK",
                        "regime_state": "TREND",
                        "pnl_r": 1.0,
                        "institutional_confluence_score": 0.78,
                        "candle_mastery_score": 0.74,
                    },
                ],
                shadow_trades=[
                    {"symbol": "BTCUSD", "strategy_key": "BTC_SWEEP", "pnl_r": 1.0},
                    {"symbol": "BTCUSD", "strategy_key": "BTC_SWEEP", "pnl_r": 0.8},
                ],
                horizon="4h",
                now_utc=datetime(2026, 5, 3, 12, 0, tzinfo=timezone.utc),
            )

            self.assertEqual(review.live_trades, 3)
            self.assertGreaterEqual(review.loss_root_causes["poor_candle_structure"], 2)
            self.assertTrue(review.live_shadow_gaps)
            actions = {proposal.action for proposal in review.proposals}
            self.assertIn("tighten_candle_mastery_filter", actions)
            self.assertIn("shadow_gap_investigation", actions)
            self.assertTrue(all(proposal.requires_walk_forward for proposal in review.proposals))
            memory = engine.compressed_memory(limit=1)
            self.assertEqual(len(memory), 1)
            self.assertEqual(memory[0]["horizon"], "4h")


if __name__ == "__main__":
    unittest.main()
